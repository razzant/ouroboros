"""What an arm's own launcher left behind, read from that arm's own files and never echoed.

``launcher_diagnostics`` pairs one transition's replacement request with its OSLog lookup. ``Opening`` records how one
launcher launch got toward its first ready server and its window's first document, even when neither came: which
stages it observably reached, how the process ended, which known fatal signature its private stderr file carries and
which listed Python exception class crossed a product operation the observer witnesses. Every public value is a label
or a number from the closed tables below. A line, path, exception message or address is matched, never copied. A
missing marker is an absent observation: it does not show that the stage did not happen, and no stage or class says
what caused an exit. ``inferred`` only names where the observed window stages end.

``witness_setup`` is the writing half of that contract. The observer calls it inside the launcher process as the main
window is made: pywebview's own ``shown`` and ``before_load`` events and four methods of the product's ``MainRenderer``
leave empty marker files, and every witnessed call keeps its arguments, result and exception.

An opening covers only its launch up to the boundary where the driver freezes it: the success or failure of that
launch's first normal open of the Main room, or an earlier failure while waiting for its ready server. ``freeze`` keeps
that snapshot; later log lines, server generations, markers or launches never change it, while ``project`` reads the
same launch again later. Its ``exit`` is the process state at the boundary, and None means the launcher was still
running then, not that it never ended.
"""
from __future__ import annotations

import functools
import os
import re
import sys
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
# What that observer then witnesses of the product's own window (``witness_setup``). product_session_registered and
# first_navigation_failed are outcomes beside the window's order, not steps of it.
WINDOW_STAGES = ("observer_hook_completed", "product_setup_entered", "product_setup_completed", "product_session_registered",
                 "window_shown", "first_navigation_entered", "first_navigation_requested", "first_document_committed",
                 "first_navigation_failed", "first_document_finished")
WINDOW_ORDER = ("window_created", "window_before_show", "observer_hook_completed", "product_setup_entered",
                "product_setup_completed", "window_shown", "first_navigation_entered", "first_navigation_requested",
                "first_document_committed", "first_document_finished")
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
# The product's own fixed lines (ouroboros/launcher_webkit.py, launcher_webkit_diagnostics.py): the registration
# failure names a Python class, the navigation error an NSError domain and code as WebKit reported them.
_REGISTRATION = re.compile(_LOG_PREFIX + rb"WebKit room session registration failed \((\w{1,80})\); keeping this document")
_NAVIGATION_ERROR = re.compile(_LOG_PREFIX + rb"WebKit navigation_error stage=(provisional|committed) domain=(\w{1,64}) "
                                             rb"code=(-?\d{1,12}|unavailable) observed_at=\d+\.\d{3}")
NAVIGATION_DOMAINS = ("NSURLErrorDomain", "WKErrorDomain", "WebKitErrorDomain", "NSCocoaErrorDomain", "NSPOSIXErrorDomain",
                      "NSOSStatusErrorDomain", "kCFErrorDomainCFNetwork", "other", "unavailable")
NAVIGATION_ERROR_LIMIT = 4
STAGES = ("spawned", "observer_loaded", "webview_imported", "launcher_log_opened", "pid_lock_held", "lock_refused_logged",
          "server_spawn_logged", "server_process_bound", "server_exit_before_ready_logged", "server_exit_logged",
          "uncaught_exception_logged", "server_ready", "window_created", "window_before_show", "observer_hook_completed",
          "product_setup_entered", "session_registration_failed_logged", "product_setup_completed",
          "product_session_registered", "window_shown", "first_navigation_entered", "first_navigation_requested",
          "first_document_committed", "first_navigation_failed", "first_document_finished")
# A Python exception class is named only from this list; any other stays "other" (a witness) or unnamed (a reason).
PYTHON_CLASSES = ("AttributeError", "TypeError", "ValueError", "KeyError", "IndexError", "LookupError", "RuntimeError",
                  "NameError", "UnboundLocalError", "ImportError", "ModuleNotFoundError", "AssertionError",
                  "NotImplementedError", "RecursionError", "ZeroDivisionError", "OSError", "FileNotFoundError",
                  "PermissionError", "TimeoutError", "StopIteration", "objc.error", "objc.internal_error")
# The product's MainRenderer methods a witness wraps, by the operation label of an exception that crosses one.
OPERATIONS = {"_before_show": "window_setup", "_first_document": "document_request", "document_left": "document_commit",
              "navigation_failed": "navigation_failure"}
_RAISED = {f"raised.{operation}.{label}": (operation, label)
           for operation in OPERATIONS.values() for label in (*PYTHON_CLASSES, "other")}
MARKER_LIMIT = 512
# Process-fatal signatures only: the launcher's stderr also carries ordinary logged tracebacks, which prove nothing.
# OC_PythonException is PyObjC's name for a Python exception that crossed an Objective-C callback; its reason opens
# with that Python class, which is named only when listed.
_EXCEPTION_NAMES = ("NSInvalidArgumentException", "NSInternalInconsistencyException", "NSGenericException",
                    "NSRangeException", "NSMallocException", "OC_PythonException")
_OBJC = re.compile(rb"Terminating app due to uncaught exception '(\w+)'")
_BRIDGED = re.compile(rb"uncaught exception 'OC_PythonException', reason: '<class '([A-Za-z_][\w.]{0,80})'>")
_SIGNATURES = (("python_fatal_error", re.compile(rb"Fatal Python error: ")),
               ("assertion_failed", re.compile(rb"(?m)^Assertion failed: ")),
               ("abort_called", re.compile(rb"abort\(\) called")))
ERROR_CLASSES = ("objc_exception", *(f"objc_exception:{name}" for name in _EXCEPTION_NAMES),
                 *(f"python_exception:{name}" for name in PYTHON_CLASSES), *(label for label, _ in _SIGNATURES))


def exception_label(exc: BaseException) -> str:
    """A listed class name, else "other": a class outside builtins is named with its module, never as a builtin."""
    kind = type(exc)
    name = kind.__qualname__ if kind.__module__ == "builtins" else f"{kind.__module__}.{kind.__qualname__}"
    return name if name in PYTHON_CLASSES else "other"


def _quietly(step, *args):
    """A witness's own step: its failure is swallowed, so the call it watches never sees it."""
    try:
        return step(*args)
    except Exception:  # noqa: BLE001 - a witness never changes what the product's call does
        return None


def _witnessed(original, operation, mark, before, after):
    @functools.wraps(original)  # pywebview dispatches a handler by its signature, which this keeps
    def witnessed(self, *args, **kwargs):
        seen = None if before is None else _quietly(before, self, *args)
        try:
            result = original(self, *args, **kwargs)
        except BaseException as exc:  # recorded and raised again unchanged; nothing is caught for good
            _quietly(mark, f"raised.{operation}.{exception_label(exc)}")
            raise
        if after is not None:
            _quietly(after, self, seen)
        return result
    return witnessed


def witness_setup(window, mark) -> None:
    """Inside the launcher, as its main window is made: witness the product's window setup and first document.
    ``mark`` leaves one empty file and never raises. Neither does this, and no witnessed call changes."""
    events = getattr(window, "events", None)
    # pywebview fires before_load in its didFinishNavigation, just before it injects its bridge.
    for event, stage in (("shown", "window_shown"), ("before_load", "first_document_finished")):
        handlers = getattr(events, event, None)
        if handlers is not None:
            _quietly(handlers.__iadd__, functools.partial(mark, stage))
    renderer = getattr(sys.modules.get("ouroboros.launcher_webkit"), "MainRenderer", None)
    if not isinstance(renderer, type) or renderer.__dict__.get("_window_witnessed"):
        return  # an older arm without it, or a successor image that already wrapped it
    renderer._window_witnessed = True

    def generation(self, *_):
        return getattr(getattr(self, "navigation", None), "generation", None)

    def setup_returned(self, _):
        mark("product_setup_completed")
        if getattr(self, "installed", None) is True and getattr(self, "navigation", None) is not None:
            mark("product_session_registered")

    def entered(self, *_):  # the product's own request for the window's first document begins, none bound yet
        before = generation(self)
        if getattr(self, "first", None) is True and before is None:
            mark("first_navigation_entered")
        return before

    def requested(self, before):  # the native loadRequest returned a WebKit navigation for that request
        if (after := generation(self)) is not None and after is not before:
            mark("first_navigation_requested")

    def committed(self, *_):  # the view on screen committed while its window had no document yet
        if getattr(self, "first", None) is True:
            mark("first_document_committed")

    def failed(self, view=None, *_):  # WebKit's word that the window's own first load failed
        if getattr(self, "first", None) is True and view is getattr(self, "view", None):
            mark("first_navigation_failed")

    steps = {"_before_show": (lambda self, *_: mark("product_setup_entered"), setup_returned),
             "_first_document": (entered, requested), "document_left": (committed, None),
             "navigation_failed": (failed, None)}
    for name, operation in OPERATIONS.items():
        original = renderer.__dict__.get(name)
        if callable(original):
            setattr(renderer, name, _witnessed(original, operation, mark, *steps[name]))


def window_progress(stages) -> dict:
    """An inference from marker order, never a cause: the furthest window stage observed and the one after it, which
    may still have happened unobserved."""
    reached = [index for index, name in enumerate(WINDOW_ORDER) if name in stages]
    after = reached[-1] + 1 if reached else 0
    return {"last_window_stage": WINDOW_ORDER[reached[-1]] if reached else None,
            "next_window_stage": WINDOW_ORDER[after] if after < len(WINDOW_ORDER) else None}


def _markers(stage_dir: Path):
    """(names, evidence label): the empty files this launch's own observer left there; never a link, and bounded."""
    try:
        with os.scandir(stage_dir) as entries:
            names = {entry.name for _, entry in zip(range(MARKER_LIMIT), entries) if entry.is_file(follow_symlinks=False)}
    except FileNotFoundError:
        return set(), "absent"
    except OSError:
        return set(), "unreadable"
    return names, "present" if names else "absent"


def _listed(raw: bytes) -> str:
    name = raw.decode("ascii", "replace")
    return name if name in PYTHON_CLASSES else "other"


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
        """Once frozen, always that snapshot, whatever the files or the process become; before that, ``project``."""
        return self.frozen if self.frozen is not None else self.project(servers_seen)

    def project(self, servers_seen: int) -> dict:
        """What is observable of this launch now; ``servers_seen`` counts this arm's server generations, this launch's
        being those past ``servers_before``. Reading it later never changes the frozen record."""
        launcher = self.launcher
        code = launcher.poll()
        log, log_state = _tail(self.log, self.offset, LOG_LIMIT)
        err, err_state = _tail(self.stderr, 0, STDERR_TAIL)
        markers, marker_state = _markers(self.stage_dir)
        seen = {"spawned"} | {name for name in (*OBSERVER_STAGES, *WINDOW_STAGES) if name in markers}
        raised = {(operation, label, "observer"): None for name, (operation, label) in _RAISED.items() if name in markers}
        navigation = []
        if log:  # this launch wrote: a log only older launches wrote to says nothing of it
            seen.add("launcher_log_opened")
        try:
            seen |= {"pid_lock_held"} if (self.root / "app/ouroboros.pid").read_text().strip() == str(launcher.pid) else set()
        except (OSError, ValueError):
            pass
        for line in log.splitlines():
            seen |= {name for name, pattern in _LOG_STAGES if pattern.fullmatch(line)}
            if match := _REGISTRATION.fullmatch(line):
                seen.add("session_registration_failed_logged")
                raised[("session_registration", _listed(match[1]), "launcher_log")] = None
            elif len(navigation) < NAVIGATION_ERROR_LIMIT and (match := _NAVIGATION_ERROR.fullmatch(line)):
                domain, number = match[2].decode("ascii"), match[3].decode("ascii")
                navigation.append({"stage": match[1].decode("ascii"),
                                   "domain": domain if domain in NAVIGATION_DOMAINS else "other",
                                   "code": number if number == "unavailable" else int(number)})
        seen |= {"server_process_bound"} if servers_seen > self.servers_before else set()
        seen |= {"server_ready"} if self.ready else set()
        classes = {label for label, pattern in _SIGNATURES if pattern.search(err)}
        for match in _OBJC.finditer(err):
            name = match[1].decode("ascii")
            classes.add(f"objc_exception:{name}" if name in _EXCEPTION_NAMES else "objc_exception")
        classes |= {f"python_exception:{name}" for match in _BRIDGED.finditer(err)
                    if (name := match[1].decode("ascii")) in PYTHON_CLASSES}
        stages = [name for name in STAGES if name in seen]
        return {"ordinal": self.ordinal, "exit": None if code is None else {"signal": -code} if code < 0 else {"code": code},
                "stages": stages, "error_classes": [name for name in ERROR_CLASSES if name in classes],
                "raised": [{"operation": operation, "class": label, "seen_by": source} for operation, label, source in raised],
                "navigation_errors": navigation,
                "evidence": {"launcher_log": log_state, "stderr": err_state, "markers": marker_state},
                "inferred": window_progress(stages),
                "note": "an absent stage or class is an unobserved one, not a proof, and none states a cause; inferred only "
                        "says where the observed window stages end"}

    def freeze(self, servers_seen: int) -> dict:
        """Take the snapshot at the opening's boundary; a second call keeps the first."""
        if self.frozen is None:
            self.frozen = self.record(servers_seen)
        return self.frozen
