"""What an arm's own launcher left behind, read from that arm's own files and never echoed.

``launcher_diagnostics`` pairs one transition's replacement request with its OSLog lookup. ``Opening`` records how one
launcher launch got toward its first ready server, even when none was ever ready: which stages it observably
reached, how the process ended, and which known fatal signature, if any, its private stderr file carries. Every public
value is a label or a number from the closed tables below. A line, path, exception message or address is matched, never
copied. A missing marker is an absent observation: it does not show that the stage did not happen, and no stage or
class says what caused an exit.

An opening covers only its launch up to the boundary where the driver freezes it: the success or failure of that
launch's first normal open of the Main room, or an earlier failure while waiting for its ready server. ``freeze`` keeps
that snapshot; later log lines, server generations, markers or launches never change it. Its ``exit`` is the process
state at the boundary, and None means the launcher was still running then, not that it never ended.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

LOG_LIMIT = 2 * 1024 * 1024  # the launcher's rotating-file bound; no fallback to other logs
STDERR_TAIL = 64 * 1024  # a fatal signature is the last thing a process writes
OSLOG_RESULTS = ("unavailable", "no_matching_entry", "entry_limit", "matched")
_WEBKIT_INFO = re.compile(rb"\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d[.,]\d{3}Z? \[INFO\] launcher\.webkit: WebKit ")
_REPLACEMENT = re.compile(rb"window_replacement_requested requested_at=(\d+\.\d{3})")
_LOOKUP = re.compile(rb"process_log_lookup scope=current_process since=\d+\.\d{3} until=(\d+\.\d{3}) result=(\w+)")


def launcher_diagnostics(arm_root: Path, began: float, ended: float) -> dict:
    """Read only this synthetic arm's launcher.log. Pair the request and lookup by their shared timestamp."""
    def result(outcome, requested=False):
        return {"window_replacement_requested": requested, "process_log_lookup": outcome}

    try:
        path = arm_root.resolve() / "data/logs/launcher.log"
        if path.resolve() != path or not path.is_file():
            return result("launcher_log_unavailable")
        with path.open("rb") as stream:
            data = stream.read(LOG_LIMIT + 1)
    except (OSError, RuntimeError):
        return result("launcher_log_unavailable")
    if len(data) > LOG_LIMIT:
        return result("launcher_log_limit")
    requests, lookups = [], {}
    for line in data.splitlines():
        prefix = _WEBKIT_INFO.match(line)
        if not prefix:
            continue
        message = line[prefix.end():]
        if (match := _REPLACEMENT.fullmatch(message)) and round(began, 3) <= float(match[1]) <= round(ended, 3):
            requests.append(match[1])
        elif match := _LOOKUP.fullmatch(message):
            lookups.setdefault(match[1], []).append(match[2].decode("ascii"))
    if len(requests) != 1:
        return result("ambiguous" if requests else "request_unobserved", bool(requests))
    outcomes = lookups.get(requests[0], [])
    return result(outcomes[0] if len(outcomes) == 1 and outcomes[0] in OSLOG_RESULTS else "lookup_unobserved", True)


# --- one launch's first opening ----------------------------------------------------------------------------------------
# Observer stages are files the fixture's own observer touches inside the launcher process (``native_window.mark``).
OBSERVER_STAGES = ("observer_loaded", "webview_imported", "window_created", "window_before_show")
_LOG_PREFIX = rb"\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d[.,]\d{3}Z? \[(?:INFO|WARNING|ERROR|CRITICAL)\] [\w.]+: "
_LOG_STAGES = (
    ("server_spawn_logged", re.compile(_LOG_PREFIX + rb"Starting agent: .*")),
    ("server_exit_before_ready_logged", re.compile(
        _LOG_PREFIX + rb"Agent server exited (?:\(or the launcher stopped\) before it published an endpoint|"
                      rb"before it answered /api/health on port \d+)")),
    ("server_exit_logged", re.compile(_LOG_PREFIX + rb"Agent exited with code -?\d+")),
    ("lock_refused_logged", re.compile(_LOG_PREFIX + rb"Another instance already running\.")),
    ("uncaught_exception_logged", re.compile(_LOG_PREFIX + rb"Uncaught exception(?: in thread .*)?")),
)
STAGES = ("spawned", "observer_loaded", "webview_imported", "launcher_log_opened", "pid_lock_held", "lock_refused_logged",
          "server_spawn_logged", "server_process_bound", "server_exit_before_ready_logged", "server_exit_logged",
          "uncaught_exception_logged", "server_ready", "window_created", "window_before_show")
# Process-fatal signatures only: the launcher's stderr also carries ordinary logged tracebacks, which prove nothing.
_NS_EXCEPTIONS = ("NSInvalidArgumentException", "NSInternalInconsistencyException", "NSGenericException",
                  "NSRangeException", "NSMallocException")
_OBJC = re.compile(rb"Terminating app due to uncaught exception '(\w+)'")
_SIGNATURES = (("python_fatal_error", re.compile(rb"Fatal Python error: ")),
               ("assertion_failed", re.compile(rb"(?m)^Assertion failed: ")),
               ("abort_called", re.compile(rb"abort\(\) called")))
ERROR_CLASSES = ("objc_exception", *(f"objc_exception:{name}" for name in _NS_EXCEPTIONS),
                 *(label for label, _ in _SIGNATURES))


def _tail(path: Path, offset: int, limit: int):
    """(bytes, evidence label): a regular file of this arm only, from ``offset`` on, bounded; never a link or a pipe."""
    try:
        if path.is_symlink() or not path.is_file():
            return b"", "absent"
        size = path.stat().st_size
        if size < offset:
            return b"", "unattributed"  # shorter than at spawn: rotated or replaced, so not this launch's
        with path.open("rb") as stream:
            stream.seek(max(offset, size - limit))
            return stream.read(limit), "present"
    except OSError:
        return b"", "unreadable"


class Opening:
    """One launcher launch's record, bound to this launch's own stderr file, stage markers and log offset."""

    def __init__(self, arm_root: Path, stage_dir: Path, ordinal: int, servers_before: int):
        self.root, self.stage_dir, self.ordinal, self.servers_before = arm_root, stage_dir, ordinal, servers_before
        self.launcher, self.ready = None, False  # the Popen of this launch, and whether it served a ready generation
        self.log = arm_root / "data/logs/launcher.log"
        self.stderr = arm_root / f"launcher-{ordinal}.stderr"
        try:
            self.offset = self.log.stat().st_size  # earlier launches of this arm wrote above this offset
        except OSError:
            self.offset = 0
        self.frozen = None

    def open_stderr(self):
        """A new private regular file in this arm's own directory: it can fill, but never stall the launcher the way a
        full pipe would. Exclusive creation refuses an existing file or link, so it is never an earlier launch's."""
        return os.fdopen(os.open(self.stderr, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")

    def record(self, servers_seen: int) -> dict:
        """Project what is observable now; ``servers_seen`` counts this arm's server generations, this launch's being
        those past ``servers_before``. Once frozen, always that snapshot, whatever the files or the process become."""
        if self.frozen is not None:
            return self.frozen
        launcher = self.launcher
        code = launcher.poll()
        log, log_state = _tail(self.log, self.offset, LOG_LIMIT)
        err, err_state = _tail(self.stderr, 0, STDERR_TAIL)
        seen = {"spawned"}
        seen |= {name for name in OBSERVER_STAGES if (self.stage_dir / name).is_file()}
        if log:  # this launch wrote: a log only older launches wrote to says nothing of it
            seen.add("launcher_log_opened")
        try:
            seen |= {"pid_lock_held"} if (self.root / "app/ouroboros.pid").read_text().strip() == str(launcher.pid) else set()
        except (OSError, ValueError):
            pass
        for line in log.splitlines():
            seen |= {name for name, pattern in _LOG_STAGES if pattern.fullmatch(line)}
        seen |= {"server_process_bound"} if servers_seen > self.servers_before else set()
        seen |= {"server_ready"} if self.ready else set()
        classes = set()
        for label, pattern in _SIGNATURES:
            if pattern.search(err):
                classes.add(label)
        for match in _OBJC.finditer(err):
            name = match[1].decode("ascii")
            classes.add(f"objc_exception:{name}" if name in _NS_EXCEPTIONS else "objc_exception")
        return {"ordinal": self.ordinal, "exit": None if code is None else {"signal": -code} if code < 0 else {"code": code},
                "stages": [name for name in STAGES if name in seen],
                "error_classes": [name for name in ERROR_CLASSES if name in classes],
                "evidence": {"launcher_log": log_state, "stderr": err_state},
                "note": "an absent stage or class is an unobserved one, not a proof, and none states a cause"}

    def freeze(self, servers_seen: int) -> dict:
        """Take the snapshot at the opening's boundary; a second call keeps the first."""
        if self.frozen is None:
            self.frozen = self.record(servers_seen)
        return self.frozen
