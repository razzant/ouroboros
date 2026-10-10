"""One logging bootstrap per Ouroboros process, plus the uncaught-exception hooks.

A process configures its logging at its real start, never as a side effect of
which module happened to be ``__main__``: the server from ``server.main()``
whatever the entry point (``python server.py``, ``ouroboros server``, the
``ouroboros-web`` script, Colab), each pool worker at the top of ``worker_main``
after its fork preamble (so a forkserver parent stays single-threaded), and the
launcher keeps its own ``launcher.log`` setup because it ships frozen inside the
app bundle, sharing ``install_exception_hooks`` and ``copy_capped_output``. The
server is the single writer of ``logs/server.log``; a worker logs to its stderr,
which the launcher copies into ``logs/agent_stdout.log`` (desktop) and Docker
keeps as the container log, so two processes never rotate one file. Every local
handler carries the shared ``SecretRedactingLogFilter``; the module adds handlers
beside any a process already has and never replaces them. Importing the module
loads only the standard library: the launcher's copier must keep draining and
recording a server whose own imports are broken, which is the crash output
``agent_stdout.log`` exists for.

The hooks route an unhandled exception of a thread, or of the main thread, into
``logging`` instead of the interpreter's raw stderr print, so it is formatted like
every other record (its message passes the redaction filter; the traceback itself is
not redacted) and reaches whatever handler an operator or an in-process extension
attached. A hook somebody installed earlier (a test runner,
an error-tracking SDK) still runs after ours. An exception that escapes a
multiprocessing child's target never reaches ``sys.excepthook`` (multiprocessing
catches it first), which is why ``worker_main`` records its own crashes.
"""

from __future__ import annotations

import errno
import logging
import os
import pathlib
import sys
import threading
from logging.handlers import RotatingFileHandler
from typing import Any

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
SERVER_LOG_MAX_BYTES = 2 * 1024 * 1024
SERVER_LOG_BACKUP_COUNT = 3
OUTPUT_CHUNK_BYTES = 64 * 1024
UNCAUGHT_LOGGER_NAME = "ouroboros.uncaught"

_configured = False


def configure_process_logging(*, drive_logs: pathlib.Path | None) -> None:
    """Attach this process's local handlers once and install the exception hooks.

    ``drive_logs`` is the data root's ``logs/`` directory and is given only by
    the server process, the single writer of ``server.log``; any other caller
    gets a stderr stream handler alone. A second call in the same process
    changes nothing.
    """
    global _configured
    if _configured:
        return
    from ouroboros.observability import SecretRedactingLogFilter

    _configured = True
    handlers: list[logging.Handler] = []
    if drive_logs is not None:
        drive_logs = pathlib.Path(drive_logs)
        drive_logs.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(
            drive_logs / "server.log", maxBytes=SERVER_LOG_MAX_BYTES,
            backupCount=SERVER_LOG_BACKUP_COUNT, encoding="utf-8",
        ))
    handlers.append(logging.StreamHandler())
    formatter = logging.Formatter(LOG_FORMAT)
    root = logging.getLogger()
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(SecretRedactingLogFilter())
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    # httpx logs each request URL at INFO; polling transports put credentials in
    # the URL path, so even redacted lines are noise at this level.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    install_exception_hooks()


def install_exception_hooks() -> None:
    """Route uncaught thread and main-thread exceptions into ``logging``; idempotent."""
    if getattr(threading.excepthook, "_ouroboros_hook", False):
        return
    previous_thread_hook = threading.excepthook
    previous_sys_hook = sys.excepthook
    # The interpreter defaults only print the traceback raw; our record replaces
    # that print. Anything else installed earlier keeps running after ours.
    chain_thread = previous_thread_hook is not threading.__excepthook__
    chain_sys = previous_sys_hook is not sys.__excepthook__

    def thread_hook(args: Any) -> None:
        name = getattr(args.thread, "name", None) or "unknown"
        exc_info = (args.exc_type, args.exc_value, args.exc_traceback)
        if args.exc_type is SystemExit or not _log_uncaught(f"Uncaught exception in thread {name}", exc_info):
            previous_thread_hook(args)
        elif chain_thread:
            previous_thread_hook(args)

    def sys_hook(exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        exc_info = (exc_type, exc_value, exc_traceback)
        if issubclass(exc_type, KeyboardInterrupt) or not _log_uncaught("Uncaught exception", exc_info):
            previous_sys_hook(exc_type, exc_value, exc_traceback)
        elif chain_sys:
            previous_sys_hook(exc_type, exc_value, exc_traceback)

    thread_hook._ouroboros_hook = True  # type: ignore[attr-defined]
    sys_hook._ouroboros_hook = True  # type: ignore[attr-defined]
    threading.excepthook = thread_hook
    sys.excepthook = sys_hook


def _log_uncaught(message: str, exc_info: Any) -> bool:
    """Record one uncaught exception; False leaves it to the previous hook.

    With no root handler (logging never configured, or torn down at exit) the
    record would reach nobody, so the interpreter's own print stays in charge.
    """
    if not logging.getLogger().handlers:
        return False
    try:
        logging.getLogger(UNCAUGHT_LOGGER_NAME).error(message, exc_info=exc_info)
    except Exception:
        return False
    return True


def copy_capped_output(stream: Any, path: pathlib.Path, *, report: logging.Logger,
                       max_bytes: int = SERVER_LOG_MAX_BYTES,
                       backups: int = SERVER_LOG_BACKUP_COUNT) -> None:
    """Drain a child's output pipe into ``path``, continuing after storage errors.

    A child whose pipe fills blocks in its next write (for the server: its
    stderr log handler, so every thread that logs), so a storage error must
    not end the read loop. Filesystem calls remain synchronous: a hung device
    can still delay draining. Each read is at most 64 KiB of raw bytes, one line when it fits:
    newline-free output holds bounded memory, and the cap counts bytes on disk
    with no decoding or newline translation. The file rotates like
    ``server.log`` BEFORE a write would pass ``max_bytes``. A chunk that cannot
    be stored within the cap (open, rotation or write failure) is dropped and
    counted while reading continues, and storage is retried with the next
    chunk; a failed rotation never truncates the live file or appends past the
    cap. A file this copier did not write (an inherited oversized one) rotates
    whole into ``.1`` and ages out with the backups.

    Gaps are disclosed, never invented: a chunk refused before any byte reached
    the file is ``not written``; the bytes of a failed write call are of
    ``uncertain`` durability. When storage resumes, a bounded marker line,
    counted toward the cap, states both before the next output. ``report`` (the
    launcher's own log, never this file) receives one warning per failure kind
    per file generation and its recovery, a gap still open at end of input, and
    a failed read; a failing report never stops the copy. End of input, or a
    failed read of a closed or broken pipe, ends it: nothing retries the input.
    """
    sink = _CappedFile(pathlib.Path(path), max_bytes, backups, report)
    limit = min(OUTPUT_CHUNK_BYTES, max_bytes)
    try:
        while True:
            try:
                chunk = stream.readline(limit)
            except Exception as exc:
                _tell(report.warning, "Output pipe read failed (%s); the copy into %s stopped",
                      _failure_kind(exc), sink.path.name)
                return
            if not chunk:
                return
            sink.store(chunk)
    finally:
        sink.close()


def _tell(emit: Any, message: str, *args: Any) -> None:
    try:
        emit(message, *args)
    except Exception:
        pass  # A failing report never stops the pipe from draining.


def _failure_kind(exc: BaseException) -> str:
    """Class plus errno name: stable across chunks, so one kind is one report."""
    code = getattr(exc, "errno", None)
    name = errno.errorcode.get(code, code) if isinstance(code, int) else None
    return f"{type(exc).__name__}:{name}" if name else type(exc).__name__


class _CappedFile:
    """One live file and its numbered backups, written through an unbuffered handle.

    Unbuffered writes leave nothing in a buffer that a later close could flush
    past the cap, and each returned count is what reached the file; the size is
    re-read from the file whenever it is (re)opened.
    """

    def __init__(self, path: pathlib.Path, max_bytes: int, backups: int, report: logging.Logger) -> None:
        self.path, self.max_bytes, self.backups, self.report = path, max_bytes, backups, report
        self.handle: Any = None
        self.size = 0
        self.failure = ""  # The kind that opened the current gap; empty while storage works.
        self.not_written = self.uncertain = 0
        self.gap_reported = False
        self.warned: set[str] = set()  # Kinds already reported in this file generation.

    def store(self, chunk: bytes) -> None:
        if self.failure:
            marker = (f"[output gap: {self.not_written} bytes not written, {self.uncertain} bytes "
                      f"of uncertain durability ({self.failure})]\n").encode("ascii")
            if not self._put(marker[:self.max_bytes], marker=True):
                self.not_written += len(chunk)
                return
            if self.gap_reported:
                _tell(self.report.info, "%s storage recovered: %d bytes not written, %d bytes of uncertain durability",
                      self.path.name, self.not_written, self.uncertain)
            self.failure, self.not_written, self.uncertain = "", 0, 0
        self._put(chunk)

    def close(self) -> None:
        if self.failure:
            _tell(self.report.warning, "%s output ended with storage failing (%s): %d bytes not written, "
                  "%d bytes of uncertain durability", self.path.name, self.failure, self.not_written, self.uncertain)
        self._close()

    def _put(self, data: bytes, *, marker: bool = False) -> bool:
        stage, pending = "open", memoryview(data)
        try:
            if self.handle is None:
                self._open()
            if self.size and self.size + len(data) > self.max_bytes:
                stage = "rotate"
                self._close()
                self._rotate()
                self._open()
                if self.size and self.size + len(data) > self.max_bytes:
                    raise FileExistsError(errno.EEXIST, "the live file is still full after rotation")
            stage = "write"
            while pending:
                written = self.handle.write(pending)
                if not written:
                    raise OSError(errno.EIO, "the write made no progress")
                self.size += written
                pending = pending[written:]
        except Exception as exc:
            self._close()
            if marker:
                pass  # Host-authored bytes are no output; the gap keeps its counts.
            elif stage == "write":
                self.uncertain += len(pending)
            else:
                self.not_written += len(pending)
            if not self.failure:
                self.failure = f"{stage}:{_failure_kind(exc)}"
                self.gap_reported = self.failure not in self.warned
                if self.gap_reported:
                    self.warned.add(self.failure)
                    _tell(self.report.warning, "%s storage failed (%s); output keeps draining and is dropped "
                          "until a write succeeds", self.path.name, self.failure)
            return False
        return True

    def _open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "ab", buffering=0)
        self.size = os.fstat(self.handle.fileno()).st_size

    def _close(self) -> None:
        handle, self.handle = self.handle, None
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass  # Unbuffered: nothing is left to flush, so a failed close loses no output.

    def _rotate(self) -> None:
        names = [self.path, *(self.path.with_name(f"{self.path.name}.{index}")
                              for index in range(1, self.backups + 1))]
        # Shift only the occupied run before the first free slot (the oldest is
        # overwritten when none is free): a retry after a failed live rename finds
        # ``.1`` free and moves nothing else, so repeated failures discard no backup.
        free = next((index for index in range(1, len(names)) if not names[index].exists()), len(names) - 1)
        for index in range(free, 1, -1):
            os.replace(names[index - 1], names[index])
        if self.path.exists():
            os.replace(self.path, names[1])
        self.warned.clear()
