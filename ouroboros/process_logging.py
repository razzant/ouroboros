"""One logging bootstrap per Ouroboros process, plus the uncaught-exception hooks.

A process configures its logging at its real start, never as a side effect of
which module happened to be ``__main__``: the server before its main imports and from ``server.main()``
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

Under the launcher the server and each pool worker hand their Python text output
(``sys.stdout``, ``sys.stderr`` and so the stream handler) to ``_OutputRelay``, a
bounded queue whose one thread alone writes into the pipe, so a stopped or vanished
launcher no longer blocks a log call, a ``Process.start`` flush or a worker's write;
nothing sends that output back to the pipe directly (a failed relay drops it). Dropped
bytes are marked in the stream once the reader returns, an ordinary exit lets the thread
hand the queue over until the close's drain bound ends it, and a hard
death loses it; ``server.log`` keeps its own path. A diagnostic child can hand its output to
the same relay (``relayed_child_output``). Output from before the relay is installed, and
whatever is written to descriptors 1 and 2 themselves (native code, ``sys.__stderr__`` or
``.buffer``, children that inherit them), still goes straight to the launcher's pipe and can
wait for it, as before; so no write that holds the GIL ever waits for the relay's thread.

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

import atexit
import collections
import contextlib
import errno
import io
import logging
import os
import pathlib
import select
import stat
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from typing import Any, Iterator

# Text records keep the JSON logs' clock (``utc_now_iso``): UTC with an explicit ``Z``, so one
# moment reads the same in ``server.log`` and ``events.jsonl`` on every install (a local ``asctime``
# without a zone put the text logs three hours off the JSON ones on a UTC+3 Mac).
LOG_FORMAT = "%(asctime)s.%(msecs)03dZ [%(levelname)s] %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"
SERVER_LOG_MAX_BYTES = 2 * 1024 * 1024
SERVER_LOG_BACKUP_COUNT = 3
OUTPUT_CHUNK_BYTES = 64 * 1024
# Output the relay may hold for a reader that is not reading; beyond it, bytes are dropped and counted.
OUTPUT_RELAY_MAX_BYTES = 1024 * 1024
UNCAUGHT_LOGGER_NAME = "ouroboros.uncaught"

_configured = False
_relay: "_OutputRelay | None" = None


def utc_log_formatter() -> logging.Formatter:
    """The one text-log formatter: UTC with its zone written, the JSON logs' clock."""
    formatter = logging.Formatter(LOG_FORMAT, LOG_DATEFMT)
    formatter.converter = time.gmtime
    return formatter


def configure_process_logging(*, drive_logs: pathlib.Path | None) -> None:
    """Attach this process's local handlers once and install the exception hooks.

    ``drive_logs`` is the data root's ``logs/`` directory and is given only by
    the server process, the single writer of ``server.log``; any other caller
    gets a stderr stream handler alone. A second call in the same process
    changes nothing. The output relay comes first, so the stream handler
    writes into it (``_install_output_relay``).
    """
    global _configured
    if _configured:
        return
    from ouroboros.observability import SecretRedactingLogFilter

    _configured = True
    _install_output_relay()
    handlers: list[logging.Handler] = []
    if drive_logs is not None:
        drive_logs = pathlib.Path(drive_logs)
        drive_logs.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(
            drive_logs / "server.log", maxBytes=SERVER_LOG_MAX_BYTES,
            backupCount=SERVER_LOG_BACKUP_COUNT, encoding="utf-8",
        ))
    handlers.append(logging.StreamHandler())
    formatter = utc_log_formatter()
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
    chain_sys = previous_sys_hook not in (sys.__excepthook__, _report_uncaught)

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


def _report_uncaught(exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
    """``sys.excepthook`` of a relayed process until logging's: the interpreter's print, then, on the main thread,
    the handover. There an uncaught report precedes the process's end, which can be ``os._exit`` with no exit hook
    (a forkserver child whose preload failed: ``multiprocessing.forkserver.main``)."""
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
    if threading.current_thread() is threading.main_thread():
        finish_output_relay()


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

    A child whose pipe fills blocks in its next write (a launcher-managed
    server's Python output drops and counts instead, ``_OutputRelay``), so a
    storage error must not end the read loop. Filesystem calls remain synchronous: a hung device
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


def _install_output_relay(max_bytes: int = OUTPUT_RELAY_MAX_BYTES, *, start_thread: bool = True) -> bool:
    """Route this process's ``sys.stdout``/``sys.stderr`` through an ``_OutputRelay``.

    Only for a launcher-managed server and its pool workers (``OUROBOROS_MANAGED_BY_LAUNCHER``,
    inherited), while descriptors 1 and 2 are one pipe (the launcher's
    ``stdout=PIPE, stderr=STDOUT``): both FIFOs of one identity. Windows' ``fstat`` gives a pipe no
    identity (dev and inode 0), so there only the launcher, the one writer of that mark, vouches that
    the two pipes are one. Returns whether it installed. The descriptors themselves
    stay untouched; the relay writes through a private non-inheritable duplicate of 2, which
    no exec'd child receives. ``sys.__stdout__``/``sys.__stderr__`` keep the plain streams, and
    text they still buffer from before is flushed by the relay's thread ahead of anything relayed,
    so installing never writes to the pipe itself.

    ``start_thread=False`` keeps the forkserver's preload thread-free (``preloading_forkserver``):
    output queues until a forked child's logging starts its own relay, or until the forkserver's
    exit starts the writer for the handover (``_OutputRelay.finish``), since nothing forks after it.
    """
    global _relay
    if _relay is not None:
        if start_thread and _relay.owner == os.getpid():
            with _relay.cond:
                _relay.allow_thread = True
                _relay.start()
        return False
    if sys.stdout is None or sys.stderr is None:
        return False
    if os.environ.get("OUROBOROS_MANAGED_BY_LAUNCHER", "").strip() != "1":
        return False
    try:  # read now, while the process's imports work: an exit (a failed import's too) imports nothing
        from ouroboros.runtime_limits import SERVER_GRACEFUL_SHUTDOWN_TIMEOUT_SEC as handover_sec
    except Exception:
        return False
    try:
        out, err = os.fstat(1), os.fstat(2)
        if (not (stat.S_ISFIFO(out.st_mode) and stat.S_ISFIFO(err.st_mode))
                or (out.st_dev, out.st_ino) != (err.st_dev, err.st_ino)):
            return False
    except OSError:
        return False
    try:
        output = os.dup(2)
    except OSError:
        output = -1  # Known managed pipe: failure must not restore blocking Python writes.
    plain = (sys.stdout, sys.stderr)
    _relay = _OutputRelay(output, max_bytes, handover_sec, plain=plain,
                         output_open=output >= 0, failed=output < 0)
    _relay.allow_thread = start_thread
    sys.stdout, sys.stderr = _RelayStream(plain[0]), _RelayStream(plain[1])
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=_relay_for_forked_child)
    # An interpreter exit runs both hooks: the first before non-daemon threads are joined, the second after.
    # A pool child's end runs the first only (multiprocessing's ``_bootstrap`` calls ``threading._shutdown``
    # after printing an uncaught traceback, then fork and forkserver children leave by ``os._exit``).
    if sys.excepthook is sys.__excepthook__:
        sys.excepthook = _report_uncaught
    atexit.register(finish_output_relay)
    with contextlib.suppress(AttributeError, RuntimeError):
        threading._register_atexit(finish_output_relay)  # type: ignore[attr-defined]
    with _relay.cond:
        _relay.start()
    return True


def preloading_forkserver() -> bool:
    """Whether the caller runs in the forkserver's own preload (``multiprocessing.forkserver.main``): that
    process forks every pool child, so a thread there would be copied mid-state. A spawned child imports the
    same main module as ``__mp_main__`` from ``spawn._main``, and a child the forkserver forked (``_serve_one``)
    may import it too (Python before 3.13 never passes the forkserver its main path): both may start threads."""
    forkserver = sys.modules.get("multiprocessing.forkserver")
    main, child = (getattr(getattr(forkserver, name, None), "__code__", None) for name in ("main", "_serve_one"))
    frame = sys._getframe(1) if main is not None else None
    while frame is not None and frame.f_code is not main and frame.f_code is not child:
        frame = frame.f_back
    return frame is not None and frame.f_code is main


def finish_output_relay(stop_deadline: float | None = None) -> None:
    """Let the relay hand what it holds now to a reader that keeps taking it; this call never writes.

    It ends at an absolute deadline taken
    from the ordinary close's budget: ``SERVER_GRACEFUL_SHUTDOWN_TIMEOUT_SEC`` from now, the drain
    the server grants its open HTTP/WS work, and never past ``stop_deadline`` (monotonic), the
    ordinary stop's own end inside the launcher's grace (``owned_shutdown.stop_deadline``), so the
    handover never lengthens a stop. What the reader has not taken by then is the exit's loss.
    ``exit_after_shutdown`` calls it before ``os._exit``, an interpreter exit and a pool child's end
    through Python's shutdown hooks; Panic, the stop backstops, a kill or an abort skip it, losing the
    queue uncounted.
    """
    relay = _relay
    if relay is not None:
        try:
            deadline = time.monotonic() + relay.handover_sec
            relay.finish(deadline if stop_deadline is None else min(deadline, stop_deadline))
        except Exception:
            pass  # Never stands between an exit and the process's end.


def _close_quietly(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass


@contextlib.contextmanager
def relayed_child_output() -> Iterator[int | None]:
    """A descriptor for a diagnostic child's stdout and stderr: its bytes join this process's relay.

    For an external tool whose output is diagnostics only (an installer's progress), which would
    otherwise inherit the launcher's pipe and wait for a stopped reader: a private pipe that a
    thread of this process reads to its end, handing each chunk to the bounded relay, so the child
    never waits for the launcher and what the relay cannot hold is dropped and counted like the
    rest. Yields None where no relay runs (the child inherits, as before). Never for a child whose
    output is a protocol or a result: the relay may drop.
    """
    relay, pipe = _relay, None
    if relay is not None and relay.owner == os.getpid():
        try:
            pipe = os.pipe()  # neither end inheritable: subprocess places the write end as the child's 1 and 2
            threading.Thread(target=_drain_child_output, args=(pipe[0],), name="child-output-relay",
                             daemon=True).start()
        except Exception as exc:
            for fd in pipe or ():
                _close_quietly(fd)
            # Inheriting here would put the installer back on the very pipe whose
            # reader may be stopped. Refuse before starting any child instead.
            raise RuntimeError("Cannot prepare diagnostic child output relay; child not started") from exc
    if pipe is None:
        yield None
        return
    try:
        yield pipe[1]
    finally:
        _close_quietly(pipe[1])  # the child keeps its own copy; the drain ends when the last copy closes


def check_output_with_relayed_stderr(command: list[str]) -> bytes:
    """Capture a small query's exact stdout; only inherited stderr is diagnostic."""
    import subprocess

    with relayed_child_output() as output:
        return subprocess.check_output(command, **({"stderr": output} if output is not None else {}))


def _drain_child_output(read_end: int) -> None:
    try:
        while chunk := os.read(read_end, OUTPUT_CHUNK_BYTES):
            relay = _relay
            if relay is not None:
                relay.put(chunk)
    except OSError:
        pass
    finally:
        _close_quietly(read_end)


def _relay_for_forked_child() -> None:
    """A forked child gets an empty relay of its own on the same private descriptor (its thread starts
    with its first output); after a gone reader or a failed relay in the parent, the child's drops."""
    global _relay
    if _relay is not None:
        _relay = _relay.for_child()


class _RelayStream(io.TextIOBase):
    """``sys.stdout`` or ``sys.stderr`` of a relayed process: text joins the relay's queue or is dropped.

    It never writes to or flushes the plain stream it replaced, whose descriptor is the launcher's pipe:
    a write there can wait for a stopped reader, and that wait is what the relay removes. ``fileno`` and
    ``buffer`` are the plain stream's, so native and byte writers stay direct and can wait, as before.
    """

    def __init__(self, original: Any) -> None:
        super().__init__()
        self.original = original

    @property
    def encoding(self) -> str:
        return getattr(self.original, "encoding", None) or "utf-8"

    @property
    def errors(self) -> str:
        return getattr(self.original, "errors", None) or "strict"

    @property
    def buffer(self) -> Any:
        return self.original.buffer

    def fileno(self) -> int:
        return self.original.fileno()

    def isatty(self) -> bool:
        return self.original.isatty()

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError(f"write() argument must be str, not {type(text).__name__}")
        relay = _relay
        if relay is not None:
            relay.put(text.encode(self.encoding, self.errors))
        return len(text)

    def flush(self) -> None:
        """Nothing to wait for: only the relay's thread writes this stream's text."""


class _OutputRelay:
    """A bounded queue of one process's Python output and the one thread that writes it to the pipe.

    Only that thread writes, and it holds no lock while it does: a producer holds ``cond`` only to
    append or to count a drop, so ``put`` never waits for the reader, and ``finish`` only watches.
    Output beyond ``max_bytes`` is dropped and counted; the count reaches the stream as one
    ``[output gap: ...]`` line at that position once the reader takes it. The thread writes at most
    ``select.PIPE_BUF`` bytes at a time (512 on Windows), ending on a line when one fits, so a line that short
    reaches the reader whole beside the pipe's other writers, as a line-buffered stream's did.
    EPIPE (the launcher gone) closes the output and later output is dropped; EAGAIN (another holder made
    the shared pipe description nonblocking) waits for room as a blocking write would. The thread never
    logs, since a record would come back through its own queue. Should it end any other way,
    it writes one line saying so and this process's later output to these streams is dropped:
    the plain streams write to the same pipe, so falling back to them would bring the wait back.
    A refused thread start leaves output queued, bounded, for the next ``put`` to try again.
    """

    def __init__(self, output: int, max_bytes: int, handover_sec: float, *, plain: tuple = (),
                 output_open: bool = True, failed: bool = False) -> None:
        self.output, self.max_bytes, self.handover_sec = output, max_bytes, handover_sec
        self.plain = plain  # streams whose buffered text predates the relay: its thread flushes them first
        self.owner = os.getpid()
        self.cond = threading.Condition(threading.RLock())  # a signal handler's write may re-enter put
        self.queue: collections.deque = collections.deque()  # chunks (bytearray) and gaps (int: bytes dropped there)
        self.gaps = 0  # gap entries in the queue
        self.queued = 0  # output bytes held, the piece in flight included
        self.done = 0  # output bytes written plus one per whole gap line: what ``finish`` counts toward
        self.pending = memoryview(b"")
        self.pending_is_output = False
        self.output_open = output_open
        self.failed = failed
        self.at_line_start = True
        self.thread: threading.Thread | None = None
        self.allow_thread = True
        self.finish_deadline: float | None = None
        self.full_pause = 0.0  # the writer's pause after a refused write, grown while the output stays full

    def for_child(self) -> "_OutputRelay":
        return _OutputRelay(self.output, self.max_bytes, self.handover_sec, output_open=self.output_open,
                            failed=self.failed)

    def start(self) -> None:
        """Start the writing thread unless it runs or cannot write (``cond`` held)."""
        if self.allow_thread and self.thread is None and self.output_open and not self.failed:
            try:
                thread = threading.Thread(target=self._run, name="output-relay", daemon=True)
                thread.start()
            except Exception:  # no new thread now (an exhausted process, an interpreter shutting down)
                return
            self.thread = thread

    def put(self, data: bytes) -> None:
        """Queue output, or drop it; never wait for the reader."""
        with self.cond:
            if self.failed or not self.output_open or self.owner != os.getpid():
                return  # nobody will write it: a failed relay, a gone reader, a fork without Python's hooks
            self.start()
            if self.queued + len(data) > self.max_bytes:
                if self.queue and isinstance(self.queue[-1], int):
                    self.queue[-1] += len(data)
                else:
                    self.queue.append(len(data))
                    self.gaps += 1
            elif data:
                if self.queue and isinstance(self.queue[-1], bytearray) and len(self.queue[-1]) < OUTPUT_CHUNK_BYTES:
                    self.queue[-1] += data
                else:
                    self.queue.append(bytearray(data))
                self.queued += len(data)
            self.cond.notify_all()  # an idle writer waits for any entry, a gap included

    def finish(self, deadline: float) -> None:
        """Hand over this tail within one absolute grace, shared by repeated exit hooks.

        No-progress intervals cannot distinguish a stopped reader from one awaiting
        CPU. Only the close's deadline ends a healthy reader's handover early. A writer
        withheld until now (a forkserver's preload) starts here: no fork follows an exit.
        """
        with self.cond:
            if self.owner != os.getpid():
                return
            self.allow_thread = True
            self.start()
            if self.thread is None:  # refused (an exhausted process): the queue is the exit's loss
                return
            self.finish_deadline = min(deadline, self.finish_deadline or deadline)
            in_flight_gap = 1 if self.pending and not self.pending_is_output else 0
            goal = self.done + self.queued + self.gaps + in_flight_gap
            while (self.plain or self.done < goal) and self.output_open and not self.failed and self.thread.is_alive():
                left = self.finish_deadline - time.monotonic()
                if left <= 0:
                    return
                self.cond.wait(left)

    def _run(self) -> None:
        try:
            for stream in self.plain:  # the plain streams' text from before the relay goes out first
                try:
                    stream.flush()  # this thread may wait for the reader; it holds no lock
                except Exception:
                    pass  # a gone reader or a closed stream: the relay's own write finds out
            with self.cond:
                self.plain = ()
                self.cond.notify_all()
            while True:
                with self.cond:
                    while not (self.pending or self.queue):
                        self.cond.wait()
                    if not self.pending:
                        self._take()
                    piece = self.pending[:getattr(select, "PIPE_BUF", 512)]
                    if len(self.pending) > len(piece):  # more follows: end on a whole line when one fits
                        piece = piece[:bytes(piece).rfind(b"\n") + 1] or piece
                try:
                    written = os.write(self.output, piece)  # may wait for the reader; no lock held
                    self.full_pause = 0.0
                except BlockingIOError:  # EAGAIN: a nonblocking description is full, never a gone reader
                    written = 0
                    self._await_room()
                except OSError:
                    written = -1
                with self.cond:
                    if written < 0:
                        self._close_output()  # EPIPE: the reader is gone
                    else:
                        self._advance(written)
                    self.cond.notify_all()
                    if not self.output_open:
                        return
        except BaseException as exc:  # noqa: BLE001 - whatever ends the thread, output must not wait on it
            with self.cond:
                self.failed = True
                line = (b"" if self.at_line_start else b"\n") + (
                    f"[output relay stopped ({_failure_kind(exc)}); {self.queued} queued bytes and this process's "
                    "later output to this stream are dropped]\n").encode("ascii", "replace")
                self.queue.clear()
                self.gaps, self.queued, self.pending = 0, 0, memoryview(b"")
                self.cond.notify_all()
                still_open = self.output_open
            if still_open:
                try:
                    os.write(self.output, line[:getattr(select, "PIPE_BUF", 512)])  # only this thread waits; no lock held
                except OSError:
                    pass

    def _await_room(self) -> None:
        """Wait, holding no lock, for the room a blocking write would have waited for: ``poll`` for POLLOUT, then a
        pause grown on each refusal in a row up to one second, since POLLOUT can show less room than a whole piece
        needs (and Windows polls no pipe). A gone reader wakes it, and its next write meets EPIPE."""
        try:
            poller = select.poll()
            poller.register(self.output, select.POLLOUT)
            poller.poll()
        except (AttributeError, OSError, ValueError):
            pass
        self.full_pause = min(1.0, max(0.01, self.full_pause * 2))
        time.sleep(self.full_pause)

    def _take(self) -> None:
        item = self.queue.popleft()
        if isinstance(item, int):
            self.gaps -= 1
            marker = f"[output gap: {item} bytes dropped by the output relay while this output was not read]\n"
            self.pending = memoryview(("" if self.at_line_start else "\n").encode("ascii") + marker.encode("ascii"))
            self.pending_is_output = False
        else:  # out of the queue, so no put appends to it any more: written in place
            self.pending, self.pending_is_output = memoryview(item), True

    def _advance(self, written: int) -> None:
        if written > 0:
            self.at_line_start = self.pending[written - 1] == 0x0A
            if self.pending_is_output:
                self.queued -= written
                self.done += written
            self.pending = self.pending[written:]
            if not self.pending:
                self.done += 0 if self.pending_is_output else 1  # a whole gap line
                self.pending = memoryview(b"")

    def _close_output(self) -> None:
        self.output_open = False
        self.queue.clear()
        self.gaps, self.queued, self.pending = 0, 0, memoryview(b"")
        _close_quietly(self.output)
