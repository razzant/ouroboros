"""Facts one server process shares with every server leaf.

The drive root and source baseline it was launched against, the ``server`` logger every server
module writes to, and the restart-request signals plus the setter that raises
them, and the two uvicorn shapes that decide who owns the process signals
(``_SignalStopServer`` for the main server, ``_embedded_uvicorn_server`` for a server
hosted inside its loop). These live below the composition root so a leaf can read
them without importing ``server`` back.

At server start ``raise_open_file_limit`` raises the open-file limit (platform rules and failures: its docstring).
"""

from __future__ import annotations

import functools
import ipaddress
import logging
import os
import pathlib
import re
import sys
import threading
import time
import urllib.parse
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional

import uvicorn

from ouroboros.launch_roots import data_dir as _configured_data_dir
from ouroboros.utils import read_json_dict, update_json_locked, utc_now_iso


DATA_DIR = _configured_data_dir()  # config's root; a direct entry has already settled and exported it (launch_roots)


log = logging.getLogger("server")
PROCESS_STARTED_MONOTONIC = time.monotonic()  # the boot clock's zero: this module loads first in the server


_restart_requested = threading.Event()
# Set FIRST in the lifespan teardown: the supervisor loop reads it in its
# ``while`` and in its crash handler, so the bus/Manager being torn down by
# the shutdown itself never counts as a loop failure (no false fault row on a
# graceful window close / SIGTERM).
_supervisor_stop = threading.Event()
# Set by the main uvicorn server's signal handler (server._SignalStopServer) and by
# the fault latch below: the PROCESS is exiting. Unlike ``_supervisor_stop`` it is
# never cleared — a settings save that lands mid-teardown must not revive a
# supervisor generation (#1142), nor one on a dead bus.
_exit_signalled = threading.Event()
_stop_source = "external_signal"

# The two causes that end a READY generation at once (a server generation is whole or it is
# over): its task core is lost (the event-bus manager process ended: ``workers.EventBusLost``,
# a ``waitpid`` fact, no retries) or its tick failed three times in a row on a live bus (the
# reviewed rule). Both leave through the lifespan teardown, as a SIGTERM does, then
# ``fault_exit``; neither is a restart request (never exit 42) nor Panic (never 99).
FAULT_BUS_LOST = "bus_lost"
FAULT_LOOP_FAILED = "loop_failed"
FAULT_CAUSES = frozenset({FAULT_BUS_LOST, FAULT_LOOP_FAILED})


# Set only when the OWNER asked for the restart (the chat Restart button, and the
# control endpoints that restart on the owner's behalf). The single fact the
# re-exec needs to decide whether the runtime-mode ratchet pin rides along.
_owner_restart_requested = threading.Event()

# Confirmed inputs of the components this process started. Settings saves may
# replace os.environ, so it is not an applied-state baseline. Never persisted.
_applied_restart_settings: dict = {}
_applied_server_host_source = "unknown"
_applied_settings_lock = threading.Lock()
_source_baseline: str | None = None


def capture_server_source_baseline(repo_dir: pathlib.Path) -> str:
    """Capture this server generation's source before startup can change the checkout.

    This is the startup baseline, not a claim about every lazy/hot-loaded module.
    An unreadable initial identity stays unknown; later disk HEAD cannot replace it.
    ``server.py`` runs this at import, so the server and every worker importing it as
    ``__mp_main__`` also pin their executing bytes here (``preflight_receipts``).
    """
    from ouroboros.gateway.state import _git_checkout_identity
    from ouroboros.preflight_receipts import capture_executing_source_baseline

    global _source_baseline
    capture_executing_source_baseline()
    with _applied_settings_lock:
        if _source_baseline is None:
            _source_baseline = _git_checkout_identity(repo_dir)[1]
        return _source_baseline


def server_source_baseline() -> str:
    """Read the captured source, never infer adoption from mutable checkout state."""
    with _applied_settings_lock:
        return _source_baseline or ""


def restart_gate_line(repo_dir: Any, data_dir: Any) -> str:
    """A restart notice's own line: the commits since this body started that Git recorded over a review FAIL or no
    verdict, deployment unconfirmed (``review_overrides.landed_line``); "" when there is none."""
    from ouroboros.review_overrides import landed_line

    line = landed_line(repo_dir, data_dir, base=server_source_baseline(), target="HEAD", running=False)
    return f"\n{line}" if line else ""


def record_applied_restart_settings(values: dict, *, server_host_source: str | None = None) -> None:
    """Publish only known startup inputs, after their component starts."""
    from ouroboros.settings_scales import RESTART_REQUIRED_SETTINGS

    global _applied_server_host_source
    with _applied_settings_lock:
        _applied_restart_settings.update({key: value for key, value in values.items()
                                         if key in RESTART_REQUIRED_SETTINGS})
        if "OUROBOROS_SERVER_HOST" in values:
            _applied_server_host_source = server_host_source or "unknown"


def raise_open_file_limit() -> None:
    """At startup the soft limit of open files (``RLIMIT_NOFILE``) rises to the hard one (POSIX; macOS takes
    ``min(OPEN_MAX, hard)``, 10,240, as its setrlimit(2) asks): the body follows each live run on its own
    transport, and a GUI-launched macOS process starts at 256. An infinite soft limit stays; Windows is
    unchanged. A failure and the resulting limit are logged without blocking startup; there is no setting."""
    try:
        import resource
    except ImportError:
        return  # Windows: no POSIX file limit
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(hard, 10240) if sys.platform == "darwin" else hard
    if soft != resource.RLIM_INFINITY and (target == resource.RLIM_INFINITY or target > soft):
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            soft = target
        except (ValueError, OSError) as exc:
            log.info("Open-file limit not raised to %s: %s", target, exc)
    log.info("Open-file limit %s (hard %s)", soft, hard)


def applied_restart_settings() -> dict:
    """Return process facts without deriving them from mutable saved intent."""
    with _applied_settings_lock:
        return dict(_applied_restart_settings)


def applied_server_host_source(drive_root: pathlib.Path) -> str:
    """Resolve launcher provenance on read: its PID record may arrive after bind."""
    with _applied_settings_lock:
        source = _applied_server_host_source
    if source != "launcher":
        return source
    record = read_json_dict(pathlib.Path(drive_root) / "state" / "server_process.json") or {}
    server_path = pathlib.Path(__file__).resolve().parents[1] / "server.py"
    if record.get("pid") != os.getpid() or record.get("server_path") != str(server_path):
        return "unknown"
    source = record.get("server_host_source")
    return source if source in ("environment", "settings") else "unknown"


def _request_restart_exit(owner: bool = False) -> None:
    """Signal server shutdown with restart exit code.

    ``owner`` is the ONE fact the re-exec needs: an owner-initiated restart
    re-reads the runtime mode from settings, an agent- or supervisor-initiated
    one keeps inheriting the boot pin (see server_control.restart_current_process).
    """
    if owner:
        _owner_restart_requested.set()
    _restart_requested.set()


_BOUND_SERVICES = frozenset({"main", "host_service", "local_model"})
# Identity answer for a recorded or configured Ouroboros endpoint whose owning
# process cannot be verified. Truthy like a proven kind, so a consumer that
# asks "treat this as Ouroboros?" refuses; never a service name.
SERVICE_IDENTITY_UNKNOWN = "unknown"
# Bind-host spelling for an expected endpoint whose interface is not recorded:
# the integer port file names a port, not the interface the server chose.
_ANY_LOCAL_HOST = "::"
_LOCAL_MODEL_PORT_ARGV_RE = re.compile(r"(?:^|\s)--port[ =](\d+)(?=\s|$)")


def _service_bindings_path(drive_root: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(drive_root) / "state" / "server_port.bindings.json"


def _windows_service_creation_time(pid: int) -> str:
    """Read informational Windows identity without changing custody/kill helpers."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        raise OSError(ctypes.get_last_error(), "runtime service process identity is unavailable")
    try:
        created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel32.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                        ctypes.byref(kernel), ctypes.byref(user)):
            raise OSError(ctypes.get_last_error(), "runtime service creation time is unavailable")
        value = (created.dwHighDateTime << 32) | created.dwLowDateTime
        if not value:
            raise ValueError("runtime service creation time is unavailable")
        return str(value)
    finally:
        kernel32.CloseHandle(handle)


def record_service_binding(drive_root: pathlib.Path, service: str, host: str, port: int,
                           *, pid: int) -> dict:
    """Publish a service owner's observed endpoint; grants and process custody are untouched."""
    from ouroboros.platform_layer import process_start_time
    from ouroboros.process_custody import _live_cmd_sha256

    if service not in _BOUND_SERVICES or not host or not 0 < int(port) < 65536 or pid <= 0:
        raise ValueError("invalid runtime service binding")
    if os.name == "nt":
        fingerprint = {"source": "windows_creation_time", "creation_time": _windows_service_creation_time(pid)}
    else:
        started = process_start_time(pid)
        if not started:
            raise RuntimeError("runtime service process identity is unavailable")
        fingerprint = {"source": "process_start_time", "start_time": started,
                       "cmd_sha256": _live_cmd_sha256(pid)}
    binding = {"pid": int(pid), "host": str(host), "port": int(port), "fingerprint": fingerprint}

    def publish(current: dict) -> dict:
        current[service] = binding
        return current

    path = _service_bindings_path(drive_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    update_json_locked(path, publish, strict_existing_dict=True)
    return binding


def clear_service_binding(drive_root: pathlib.Path, service: str, binding: dict) -> None:
    """Retire only this exact binding; a late close cannot erase its replacement."""
    def retire(current: dict) -> dict | None:
        if current.get(service) != binding:
            return None
        current.pop(service, None)
        return current

    update_json_locked(_service_bindings_path(drive_root), retire, strict_existing_dict=True)


def read_service_bindings(drive_root: pathlib.Path) -> dict:
    """Read current endpoint facts without caching, launching or enrolling a process."""
    path = _service_bindings_path(drive_root)
    if not path.exists():
        return {}
    bindings = read_json_dict(path)
    if bindings is None or any(key not in _BOUND_SERVICES for key in bindings):
        raise ValueError("runtime service bindings are unreadable")
    return bindings


def service_binding_is_live(binding: dict) -> bool:
    """Reuse process identity, only after a caller matched the endpoint's port."""
    from ouroboros.process_custody import _fingerprint_matches
    from ouroboros.platform_layer import pid_is_alive

    if not isinstance(binding, dict) or not isinstance(binding.get("fingerprint"), dict):
        return False
    fingerprint = binding["fingerprint"]
    if fingerprint.get("source") == "windows_creation_time":
        if os.name != "nt" or not pid_is_alive(int(binding.get("pid") or 0)):
            return False
        # Permission/API failure remains explicit unknown, never a false foreign
        # binding or a PID-only match. Other service ports do not pay this probe.
        return _windows_service_creation_time(binding["pid"]) == fingerprint.get("creation_time")
    return os.name != "nt" and _fingerprint_matches(binding, require_measured=True)


def _port_file_value(drive_root: pathlib.Path) -> int | None:
    """The integer port the last main server of ``drive_root`` wrote, or None when absent/unreadable."""
    from ouroboros.config import resolve_port_file

    try:
        text = resolve_port_file(drive_root, hosting_data_dir=DATA_DIR).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def _launcher_recorded_main(drive_root: pathlib.Path, port: int) -> str:
    """``"main"`` when the optional launcher process record proves a live server on ``port``."""
    from ouroboros.platform_layer import pid_is_alive, process_command

    record = read_json_dict(pathlib.Path(drive_root) / "state" / "server_process.json") or {}
    try:
        pid = int(record.get("pid") or 0)
        recorded_port = int(record.get("port") or 0)
    except (TypeError, ValueError):
        return ""
    server_path = str(record.get("server_path") or "")
    if pid <= 0 or recorded_port != port or not server_path or not pid_is_alive(pid):
        return ""
    return "main" if server_path in process_command(pid) else ""


def _custodied_local_model(drive_root: pathlib.Path, port: int,
                           host_matches: Callable[[str], bool]) -> str:
    """``"local_model"`` when this installation's own custody row owns a live server on ``port``.

    Local models bind loopback, so a nonmatching host needs no legacy ledger read.
    A matching host still replays the ledger; live argv selects the port and the
    strict custody fingerprint decides ownership.
    """
    from ouroboros.platform_layer import pid_is_alive, process_command
    from ouroboros.process_custody import _fingerprint_matches, _read_ledger

    try:
        if not host_matches("127.0.0.1"):
            return ""
    except OSError:
        pass  # An unresolved host still needs the original port-specific check.
    for entry in _read_ledger(pathlib.Path(drive_root)):
        if str(entry.get("purpose") or "") != "local_model_server":
            continue
        pid = int(entry.get("pid") or 0)
        if pid <= 0 or not pid_is_alive(pid):
            continue
        match = _LOCAL_MODEL_PORT_ARGV_RE.search(process_command(pid))
        if match and int(match.group(1)) == port and host_matches("127.0.0.1"):
            return "local_model" if _fingerprint_matches(entry, require_measured=True) else SERVICE_IDENTITY_UNKNOWN
    return ""


def runtime_service_identity(drive_root: pathlib.Path, port: int,
                             host_matches: Callable[[str], bool]) -> str:
    """Classify one endpoint against the services this installation runs.

    Returns the service kind when the endpoint is proven ours,
    ``SERVICE_IDENTITY_UNKNOWN`` when a recorded or configured Ouroboros endpoint
    matches but its owning process cannot be verified, and ``""`` when no expected
    endpoint matches — an unrelated application that reuses a port or pathname.
    ``host_matches(bind_host)`` is the caller's view of whether the target reaches
    that bound interface (``_ANY_LOCAL_HOST`` means any local interface); it is
    asked after a published/expected port matched, or before the legacy local-model
    lookup to avoid reading that ledger for a non-loopback target.

    A live binding is authoritative for its service. An unverified binding still
    names an expected endpoint until its owner retires or replaces the record.
    An installation may predate or fail to publish one service's snapshot. Its
    existing facts still decide what is expected: the integer main port file,
    Host Service configuration beside an expected main, and local-model custody
    with live argv. Only that same service's live binding supersedes its legacy
    endpoint expectation; the optional launcher record can prove main. Unknown
    identity never becomes permission.
    """
    port = int(port)
    try:
        bindings = read_service_bindings(drive_root)
    except ValueError:
        log.warning("runtime service bindings are unreadable; using the recorded legacy facts", exc_info=True)
        bindings = {}
    for kind, binding in bindings.items():
        if not isinstance(binding, dict) or binding.get("port") != port:
            continue
        if not host_matches(str(binding.get("host") or "")):
            continue
        try:
            live = service_binding_is_live(binding)
        except OSError:
            return SERVICE_IDENTITY_UNKNOWN
        return kind if live else SERVICE_IDENTITY_UNKNOWN
    def snapshot_live(kind: str) -> bool:
        binding = bindings.get(kind)
        try:
            return isinstance(binding, dict) and service_binding_is_live(binding)
        except OSError:
            return False

    main_port = _port_file_value(drive_root)
    if main_port == port and host_matches(_ANY_LOCAL_HOST):
        if snapshot_live("main"):
            return ""
        return _launcher_recorded_main(drive_root, port) or SERVICE_IDENTITY_UNKNOWN
    from ouroboros.gateway.host_service import DEFAULT_HOST_SERVICE_HOST, host_service_port

    if main_port is not None and port == host_service_port() and host_matches(DEFAULT_HOST_SERVICE_HOST):
        return "" if snapshot_live("host_service") else SERVICE_IDENTITY_UNKNOWN
    if snapshot_live("local_model"):
        return ""  # a verified manager owns another endpoint; its custody row is the same process
    return _custodied_local_model(drive_root, port, host_matches)


def serves_data_root(url: str, data_dir: pathlib.Path) -> bool:
    """Whether DATA_DIR's own main server holds URL's port: its live service binding, or the
    launcher's server record, proves it. A CLI asks this before it uses an endpoint nobody named,
    since a shared or fallback port can belong to another installation."""
    parsed = urllib.parse.urlparse(url)

    def reaches(bind_host: str) -> bool:
        try:
            address = ipaddress.ip_address(bind_host)
            target = ipaddress.ip_address(parsed.hostname or "127.0.0.1")
        except ValueError:
            return False
        # A different loopback address can host a different installation on the same port.
        # Wildcard includes the legacy any-local-interface sentinel of runtime_service_identity.
        return address == target or (address.is_unspecified and target.is_loopback)

    try:  # a URL without a port names the main server's default, 8765
        return runtime_service_identity(data_dir, parsed.port or 8765, reaches) == "main"
    except (OSError, ValueError):
        return False


def server_stop_source() -> str:
    """Diagnostic stop ingress; no effect on custody or restart policy."""
    return _stop_source


def server_stop_cause(restart_requested: bool | None = None) -> str:
    """The cause a shutdown records: ``restart_requested``, else the stop ingress; diagnostic only.
    Doors read the live restart flag; the late row passes the one its custody write used."""
    restart = _restart_requested.is_set() if restart_requested is None else restart_requested
    return "restart_requested" if restart else _stop_source


class _SignalStopServer(uvicorn.Server):
    """uvicorn.Server whose SIGTERM/SIGINT handler stops the supervisor loop AT THE SIGNAL.

    The launcher's stop may SIGTERM the whole server process group, so the multiprocessing
    Manager and pooled workers can be gone before uvicorn's drain reaches the lifespan
    teardown (#1142). Setting the stop event here, not only in the lifespan ``finally``,
    makes the loop leave its tick and keeps a torn-Manager BrokenPipe from counting as a
    supervisor crash — the false "Supervisor loop died" owner alarm. A restart request
    (``should_exit`` without a signal) is unchanged: ``_restart_requested`` already ends
    the loop, and the lifespan ``finally`` still sets the event on every path.
    """

    def request_shutdown(self, source: str = "launcher_quit") -> None:
        """The same irreversible stop for OS signals and the launcher's owned pipe."""
        global _stop_source
        _stop_source = source
        _exit_signalled.set()
        _supervisor_stop.set()
        self.should_exit = True

    async def on_tick(self, counter: int) -> bool:
        """Observe a stop latched before Uvicorn installed its signal handler."""
        return _exit_signalled.is_set() or await super().on_tick(counter)

    def handle_exit(self, sig: int, frame) -> None:
        # Let uvicorn see the previous should_exit value (repeated SIGINT forces exit).
        global _stop_source
        _stop_source = "external_signal"
        _exit_signalled.set()
        _supervisor_stop.set()
        super().handle_exit(sig, frame)

    def watch_launcher_stop(self) -> None:
        """Consume the opt-in private stdin of a launcher-managed server, on every platform.

        ``quit\n`` is the launcher's ordinary stop; EOF means the launcher is gone, so the
        same stop follows (``launcher_gone``) instead of an orphan serving on. Either one
        latches the stop at once — before the port wait and the lifespan when it arrives
        early — and arms an independent backstop exit after the launcher's own grace.
        Opt-in only: a direct server (a container PID 1 included) keeps its stdin, and
        EOF there is no stop. Retain a private raw descriptor, then give ordinary children
        null stdin: inheriting the control pipe can strand Windows Git's output readers at
        boot. Raw I/O also avoids a daemon thread holding sys.stdin's buffered shutdown lock.
        """
        if os.environ.pop("OUROBOROS_LAUNCHER_STOP_STDIN", "") != "1":
            return
        stream = getattr(sys.stdin, "buffer", None)
        if stream is None:
            return
        stdin_fd = stream.fileno()
        private_fd = os.dup(stdin_fd)
        try:
            os.set_inheritable(private_fd, False)
            with open(os.devnull, "rb", buffering=0) as empty:
                os.dup2(empty.fileno(), stdin_fd)
            if sys.platform == "win32":
                import ctypes
                import msvcrt
                # Default child stdin uses the Win32 table, not only the CRT fd table.
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.SetStdHandle.argtypes = (ctypes.c_uint32, ctypes.c_void_p)
                kernel.SetStdHandle.restype = ctypes.c_int
                if not kernel.SetStdHandle(-10, msvcrt.get_osfhandle(stdin_fd)):
                    raise ctypes.WinError(ctypes.get_last_error())
            private = os.fdopen(private_fd, "rb", buffering=0)
        except BaseException:
            os.close(private_fd)
            raise

        def receive() -> None:
            source = "launcher_gone"
            with private:
                try:
                    while line := private.readline(64):
                        if line == b"quit\n":
                            source = "launcher_quit"
                            break
                except (OSError, ValueError):
                    log.warning("Launcher stop pipe failed; treating it as the launcher's end", exc_info=True)
            self.request_shutdown(source)
            _arm_stop_backstop(source)

        threading.Thread(target=receive, name="launcher-stop", daemon=True).start()


def _launcher_stop_backstop(source: str) -> None:
    """The ordinary stop outlived the launcher's own grace: nobody else will end this process."""
    log.critical("Ordinary stop (%s) did not finish within the launcher grace; exiting", source)
    os._exit(1)


def _arm_stop_backstop(source: str, action: Optional[Callable[[str], None]] = None) -> None:
    """One independent timer after the launcher's own grace, for a stop nobody else would end; ``action``
    is what it does then: the launcher stop's exit 1, or the fault door's own exit decision."""
    from ouroboros.runtime_limits import LAUNCHER_STOP_GRACE_SEC

    backstop = threading.Timer(LAUNCHER_STOP_GRACE_SEC + 5, action or _launcher_stop_backstop, args=(source,))
    backstop.daemon = True
    backstop.start()


def _fault_backstop(source: str) -> None:
    """The fault teardown outlived the launcher grace: take the SAME door ``main()`` would, never a bare exit
    that leaves a direct install down. Launcher-managed: exit 1 and the launcher restarts; direct: the
    pre-exec census and the in-place successor (the exec never returns), a failed successor exits 1. An
    accepted Panic owns the exit: the backstop then does nothing (the Emergency Stop is never a fault)."""
    from ouroboros.server_control import panic_termination_requested

    if panic_termination_requested():
        log.info("Fault backstop (%s) defers to the accepted Panic, which owns this process's exit", source)
        return
    log.critical("Fault stop (%s) did not finish within the launcher grace; taking the fault exit now", source)
    os._exit(fault_exit())


def latch_fault_stop(cause: str) -> None:
    """The fault's irreversible stop, exactly as ``request_shutdown`` latches one, without a server handle:
    ``on_tick`` ends uvicorn, a settings save revives nothing, the backstop takes the fault exit for a
    teardown that outlives the launcher grace; the lifespan teardown that follows is the SIGTERM path."""
    global _stop_source
    _stop_source = cause
    _exit_signalled.set()
    _supervisor_stop.set()
    _arm_stop_backstop(cause, _fault_backstop)


def _saved_running_count() -> int:
    """How many RUNNING rows the teardown holds with their saved work (``restart_retention.working_saved``, the
    predicate ``kill_workers(retain_saved_work=True)`` applies; read here, marked there)."""
    from supervisor import restart_retention as retention, workers
    from supervisor.queue import _queue_lock

    with _queue_lock:
        rows = [(str(task_id), dict(meta)) for task_id, meta in workers.RUNNING.items() if isinstance(meta, dict)]
    saved = 0
    for task_id, meta in rows:
        task = meta.get("task") if isinstance(meta.get("task"), dict) else {}
        try:
            saved += bool(task and retention.working_saved(
                pathlib.Path(task.get("budget_drive_root") or workers.DRIVE_ROOT), task_id,
                int(meta.get("attempt") or task.get("_attempt") or 1), task=task))
        except Exception:
            log.debug("Saved-work read of %s failed; not counted", task_id, exc_info=True)
    return saved


def fault_row_text(cause: str) -> str:
    """The one Main row before a fault exit: ONE fact in the main line, what happened and what happens next (the
    house rule: one fact in the main view, the rest in details). The exit code, the saved count, the raw error
    and the daemon note are details: the ``supervisor_generation_fault`` journal row and the server log; the
    held count reaches the owner in the next start's notice."""
    if cause == FAULT_BUS_LOST:
        return "🛑 The event-bus process of this server exited; the server is restarting."
    return "🛑 The task loop failed three times in a row; the server is restarting."


def end_generation_on_fault(cause: str, *, exitcode: Any = None, error: Optional[BaseException] = None) -> bool:
    """A ready generation lost its task core or its tick failed three times in a row: one journal row with
    the raw facts, one Main row in owner words, then the latch. The caller clears readiness and leaves its
    loop; the lifespan teardown holds saved work (``retain_saved_work=True``), records the cause in
    ``shutdown_entry.json`` and ``fault_exit`` is the door. An accepted Panic owns the exit already: then
    nothing here happens (no row, no cause, no latch) and False returns."""
    from ouroboros.server_control import panic_termination_requested
    from supervisor.message_bus import send_with_budget
    from supervisor.state import append_jsonl, load_state

    if panic_termination_requested():
        log.info("Panic owns this process's exit; the lost bus (%s) is Panic's own kill, not a fault", cause)
        return False
    saved = _saved_running_count()
    log.critical("Server generation ended: %s (exit code %s): %s", cause, exitcode, error, exc_info=error is not None)
    try:
        append_jsonl(DATA_DIR / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(), "type": "supervisor_generation_fault", "cause": cause, "exitcode": exitcode,
            "error": f"{type(error).__name__}: {error}" if error is not None else "", "saved_running": saved})
    except Exception:
        log.debug("Failed to record the generation fault", exc_info=True)
    try:
        st = load_state()
        if st.get("owner_chat_id"):
            send_with_budget(int(st["owner_chat_id"]), fault_row_text(cause), role="system", system_type="supervisor_failure")
    except Exception:
        log.debug("Failed to notify the owner about the generation's end", exc_info=True)
    latch_fault_stop(cause)
    return True


_FAULT_CANCEL_REASONS = {
    FAULT_BUS_LOST: ("The server's event-bus process exited and the server restarted before this task finished; "
                     "the task was interrupted by that fault, not a worker crash."),
    FAULT_LOOP_FAILED: ("The server's task loop failed three times in a row and the server restarted before this "
                        "task finished; the task was interrupted by that fault, not a worker crash."),
}


def fault_cancel_reason() -> str:
    """The cancel reason of an unsaved row the fault teardown cancels; '' for every other stop (its door's text
    stands). The documented stop-cause/Continue mapping: ``stop_source`` stays ``server_shutdown`` (a technical
    interruption, Continue-eligible: ``owner_continue.TECHNICAL_CANCEL_SOURCES``), the text names the cause, no
    new source; the process cause itself is ``shutdown_entry.json``/``server_shutdown.cause``."""
    return _FAULT_CANCEL_REASONS.get(_stop_source, "")


def end_task_runs(root: Any, task_ids: Any, *, gateway_factory: Optional[Callable[[], Any]] = None) -> None:
    """End Ouroboros's OWN open delegated runs of tasks a stop keeps for the next boot (the owner Restart's returns
    and saved pauses, the Fault door's holds; a sleep included): the tasks continue from their saved state, their
    runs end (owner 3A, 1A). The verdict
    is the abrupt loss a signal-death retry passes (``reconcile_task_runs(..., deliberate_terminal=
    STATUS_INTERRUPTED)``), over an attach-only transport (default ``attach_only_gateway``): a review panel's run
    stays its panel's (only a cancellation speaks for it), and other programs' runs on the kept engine are not
    touched. One pass on its own thread, joined for at most half the generation's remaining stop grace, so the
    owned-process stop keeps the rest and a slow engine never holds the door. A run the pass has not reached by
    then is left to the existing custody: the pass goes on while the process lives; after that the run stays live
    (the next boot's sweep has no verdict for it) until it ends, or the owner stops its task."""
    ids = [str(task_id) for task_id in dict.fromkeys(task_ids or ()) if task_id]
    if not ids:
        return
    from ouroboros.delegate_custody_reconcile import attach_only_gateway, reconcile_task_runs
    from ouroboros.owned_shutdown import stop_deadline
    from ouroboros.task_results import STATUS_INTERRUPTED

    def run() -> None:
        for task_id in ids:
            try:
                reconcile_task_runs(root, task_id, gateway_factory=gateway_factory or attach_only_gateway,
                                    deliberate_terminal=STATUS_INTERRUPTED)
            except Exception:
                log.warning("Delegated runs of %s were not ended; custody keeps them", task_id, exc_info=True)

    worker = threading.Thread(target=run, name="kept-task-runs-end", daemon=True)
    worker.start()
    worker.join(max(0.0, ((stop_deadline() or time.monotonic()) - time.monotonic()) / 2))
    if worker.is_alive():
        log.warning("Ending the delegated runs of %d kept task(s) outlived its bound; the rest stay in custody", len(ids))


def end_fault_held_runs() -> None:
    """The Fault door after its teardown's terminal custody: the rows it holds with their saved work
    (``RETAINED_FOR_BOOT``) and the saved pauses it parked under an app-stop hold (a sleep: ``held_saved_work_ids``
    since this generation's entry, the list the next boot names) continue at the next boot, and their own delegated
    runs end now (``end_task_runs``). Any other stop, a restart and an accepted Panic: nothing here."""
    from ouroboros.owned_shutdown import _GENERATION_STOP
    from ouroboros.server_control import panic_termination_requested

    if _stop_source not in FAULT_CAUSES or _restart_requested.is_set() or panic_termination_requested():
        return
    try:
        from supervisor import workers
        from supervisor.queue import _queue_lock
        from supervisor.restart_retention import RETAINED_FOR_BOOT

        since = str(_GENERATION_STOP.entry.get("entered_at") or "")  # no entry: an earlier stop's holds are not ours
        with _queue_lock:
            held = [str(task_id) for task_id, meta in workers.RUNNING.items()
                    if isinstance(meta, dict) and meta.get(RETAINED_FOR_BOOT)]
            held += held_saved_work_ids(workers.PENDING, since) if since else []
        end_task_runs(workers.DRIVE_ROOT, held)
    except Exception:
        log.warning("The fault's held tasks' delegated runs were not ended; custody keeps them", exc_info=True)


def previous_fault_entry(root: Any = None) -> Dict[str, Any]:
    """The previous generation's shutdown entry when it ended through the Fault door, else ``{}``: the record
    the boot claim replaced (``owned_shutdown.claim_shutdown_entry``), with its ``cause`` and ``entered_at``.
    A Panic pressed while the fault's teardown ran keeps the entry's first cause, yet the stop was the owner's
    Emergency Stop: while its input is unconsumed (``installation_panic``) this is ``{}`` too."""
    from ouroboros.installation_stop import installation_panic
    from ouroboros.owned_shutdown import _GENERATION_STOP

    previous = dict(_GENERATION_STOP.previous)
    if str(previous.get("cause") or "") not in FAULT_CAUSES or installation_panic(root or DATA_DIR):
        return {}
    return previous


def _fault_cause_words(cause: str) -> str:
    return "its event-bus process exited" if cause == FAULT_BUS_LOST else "its task loop failed three times in a row"


def previous_fault_sentence(held: int) -> str:
    """The startup notice's clause after a fault exit: the cause and the held count, in owner words."""
    previous = previous_fault_entry()
    if not previous:
        return ""
    kept = (" No task holds saved work." if held <= 0 else " 1 task keeps its saved work." if held == 1
            else f" {held} tasks keep their saved work.")
    return f"The server restarted itself after {_fault_cause_words(str(previous['cause']))}.{kept}"


def restored_queue_sentence(restored: int, pending: Any, drive_root: Any) -> str:
    """The startup notice's restored-queue clause: the count taken at restore, then what the queue's rows need now
    (``work_stop.restored_rows_text``: empty when a row cannot be read, and not bound to sum to that count)."""
    from ouroboros.work_stop import restored_rows_text
    from supervisor.queue import BUDGET_ROOT_FENCES as fences

    return f"Restored pending queue: {restored} tasks. {restored_rows_text(drive_root, pending, fences)}".rstrip()


def held_saved_work_ids(pending: Any, since: str) -> List[str]:
    """The restored queue rows an application stop holds with their saved work (released by Resume), by id: the
    typed stop policy decides (``work_stop.record_policy``), never a hold's name. Only holds placed at or after
    ``since`` (the faulted generation's ``entered_at``): an earlier Quit's or Restart's hold nobody resumed is
    not this fault's."""
    from ouroboros.work_stop import hold_fact, record_policy

    return [str(task.get("id") or "") for task in list(pending or ()) if isinstance(task, dict)
            and (hold := hold_fact(task)) is not None and record_policy(hold)["cause"] == "app_stop"
            and (not since or str(hold.get("held_at") or "") >= since)]


def held_saved_work_count(pending: Any) -> int:
    """This fault's holds (``held_saved_work_ids`` since the previous Fault entry); 0 after any other stop."""
    previous = previous_fault_entry()
    return len(held_saved_work_ids(pending, str(previous.get("entered_at") or ""))) if previous else 0


RECOVERY_TRIGGER = "core_lost"  # the ``auto_resume_triggered`` row's trigger for the Fault boot's recovery turn
RECOVERY_RECORD = "core_loss_recovery.json"  # state/: the admitted recovery turn and the generation that admitted it


def cancelled_unsaved_roots(root: pathlib.Path, since: str) -> List[str]:
    """Root tasks the previous generation's teardown cancelled without a saved point: ``cancel_origin.source``
    ``server_shutdown`` (Continue-eligible), written at or after that generation entered its shutdown. The
    store's status prefilter bounds the read; children are their roots' business."""
    from ouroboros.task_results import list_task_results

    try:
        rows = list_task_results(root, statuses=["cancelled"])
    except Exception:
        log.warning("Cancelled rows are unreadable; the recovery turn names none", exc_info=True)
        return []
    roots = []
    for row in rows:
        origin = row.get("cancel_origin") if isinstance(row.get("cancel_origin"), dict) else {}
        task_id = str(row.get("task_id") or "")
        if (task_id and str(origin.get("source") or "") == "server_shutdown"
                and str(row.get("root_task_id") or task_id) == task_id
                and (not since or str(row.get("updated_at") or "") >= since)):
            roots.append(task_id)
    return sorted(roots)


def _recovery_record_path(root: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(root) / "state" / RECOVERY_RECORD


def record_recovery_turn(root: pathlib.Path, task_id: str, cause: str) -> None:
    """The durable bound of owner decision 2A: the admitted recovery turn's task id and the generation it was
    admitted in (the shutdown entry carries the same generation id), one record, atomic replace, best effort.
    The journal row beside it is diagnostic; rotation never changes the decision read from this record."""
    from ouroboros.delegate_recovery import _PROCESS_GENERATION
    from ouroboros.utils import atomic_write_json

    try:
        atomic_write_json(_recovery_record_path(root), {"schema_version": 1, "task_id": str(task_id),
                                                        "generation": _PROCESS_GENERATION, "cause": str(cause),
                                                        "admitted_at": utc_now_iso()})
    except Exception:
        log.warning("The recovery turn record was not written; the next fault boot admits its own turn", exc_info=True)


def open_recovery_turn(root: pathlib.Path, died_generation: str) -> str:
    """The task id of a recovery turn admitted IN the generation that just died, while that turn's result is not
    terminal: the fault loop the bound exists for. Any other fault boot gets its own turn (an earlier recovery
    turn held with saved work is then among the held ids). ``""`` when nothing binds; an unreadable result holds;
    a result row that is gone binds nothing. No counter, no timer; journal rotation is irrelevant."""
    from ouroboros.task_results import (STATUS_CANCELLED, STATUS_COMPLETED, STATUS_FAILED, STATUS_REJECTED_DUPLICATE,
                                        load_task_result)

    record = read_json_dict(_recovery_record_path(root)) or {}
    task_id, admitted_in = str(record.get("task_id") or ""), str(record.get("generation") or "")
    if not task_id or not died_generation or admitted_in != died_generation:
        return ""
    try:
        result = load_task_result(root, task_id)
    except Exception:
        return task_id  # not proven terminal: the bound holds
    if not isinstance(result, dict):
        return ""  # the turn's row is gone: nothing to hold on
    terminal = {STATUS_CANCELLED, STATUS_COMPLETED, STATUS_FAILED, STATUS_REJECTED_DUPLICATE}
    return "" if str(result.get("status") or "") in terminal else task_id


def recovery_turn_text(cause: str, ended_at: str, held: List[str], cancelled: List[str], root: pathlib.Path) -> str:
    """The recovery turn's message: facts, and the owner's one decision (2A: the mind decides and tells once)."""
    from ouroboros.places import paused_places_sentence

    return "\n".join(filter(None, (
        f"[recovery after a server fault] The previous server generation ended at {ended_at or 'an unknown time'}: "
        f"{_fault_cause_words(cause)}. This generation restarted itself with the same code.",
        "Tasks held with their saved state (Resume continues each from that state): "
        f"{', '.join(held) if held else 'none'}.",
        "Interrupted roots with no saved point, cancelled as server_shutdown (Continue-eligible): "
        f"{', '.join(cancelled) if cancelled else 'none'}.",
        "You decide what to resume or continue, with your Resume and Continue tools, and tell the owner once.",
        paused_places_sentence(root),
    )))


def recovery_refused_row(held: List[str]) -> str:
    """Main's one row when no recovery turn was admitted: owner words, no ids, nothing written for the mind. Only a
    later fault boot admits a turn, so the row says what did not happen, never that a look is still to come."""
    row = "♻️ Ouroboros did not start its recovery turn after the fault"
    return f"{row}; tasks with saved work wait for Resume." if held else f"{row}."


def core_loss_recovery_turn(root: Any, chat_id: int, pending: Any, *, admit: Callable[..., Any],
                            live: Callable[[], bool], gate: Callable[[], bool]) -> bool:
    """One recovery turn for the mind after a Fault boot (owner 2A), through the auto-resume seam.

    False when the previous generation did not end through the Fault door (Quit, Panic, owner Restart and
    planned restarts start nothing here). Otherwise the boot is handled here and True returns: one turn with
    the facts (``recovery_turn_text``) is admitted unless a conversation turn is already live, or the recovery
    turn admitted in the generation that just died is not terminal yet (``open_recovery_turn``: a fault loop,
    bound by that fact, never a counter or timer). An admitted turn is recorded durably (``record_recovery_turn``)
    and journalled (``auto_resume_triggered``, ``trigger: core_lost``). The repository-writer gate (``gate``, the
    owner-conversation predicate, quiet) is asked first: closed, ``admit`` is not called, so no owner-message
    refusal is posted for a message the owner never sent. A closed gate or a refused admission is not a consumed
    turn: no record is written, so a later fault boot admits its own turn; Main gets one short row in owner words
    (``recovery_refused_row``) and the facts go to the ``auto_resume_skipped`` journal row."""
    from supervisor.message_bus import send_with_budget
    from supervisor.state import append_jsonl

    previous = previous_fault_entry(root)
    if not previous:
        return False
    root = pathlib.Path(root)
    cause, ended_at = str(previous["cause"]), str(previous.get("entered_at") or "")
    journal = root / "logs" / "supervisor.jsonl"
    open_turn = open_recovery_turn(root, str(previous.get("generation") or ""))
    if open_turn or live():
        append_jsonl(journal, {"ts": utc_now_iso(), "type": "auto_resume_skipped", "trigger": RECOVERY_TRIGGER,
                               "reason": "previous_recovery_turn_open" if open_turn else "conversation_turn_live",
                               "task_id": open_turn, "cause": cause})
        return True
    held, cancelled = held_saved_work_ids(pending, ended_at), cancelled_unsaved_roots(root, ended_at)
    gate_open = bool(gate())
    meta = admit(int(chat_id), recovery_turn_text(cause, ended_at, held, cancelled, root), None) if gate_open else None
    task_id = str(((meta or {}).get("task") or {}).get("id") or "") if isinstance(meta, dict) else ""
    if not task_id:
        try:
            send_with_budget(int(chat_id), recovery_refused_row(held),
                             role="system", system_type="startup_notice")
        except Exception:
            log.warning("The refused recovery turn's row could not be posted in Main", exc_info=True)
        append_jsonl(journal, {"ts": utc_now_iso(), "type": "auto_resume_skipped", "trigger": RECOVERY_TRIGGER,
                               "reason": "admission_refused" if gate_open else "repo_writer_gate_closed",
                               "cause": cause, "ended_at": ended_at, "held": held, "cancelled": cancelled})
        return True
    record_recovery_turn(root, task_id, cause)
    append_jsonl(journal, {"ts": utc_now_iso(), "type": "auto_resume_triggered", "trigger": RECOVERY_TRIGGER,
                           "task_id": task_id, "cause": cause, "ended_at": ended_at, "held": held,
                           "cancelled": cancelled})
    return True


_FAULT_EXIT: Optional[Callable[[], int]] = None  # main() binds the one decision once its port is known
_FAULT_CENSUS_BOUND_SEC: Optional[float] = None  # tests shorten it; None reads LAUNCHER_STOP_GRACE_SEC


def _bounded_census(cleanup: Callable[[], Any]) -> None:
    """The pre-exec census, bounded by the launcher grace: a teardown stuck while holding the queue lock must
    not hang the fault door behind it (the backstop runs this concurrently with that teardown). An unfinished
    census is disclosed and the successor starts over it; the next generation's custody sweep settles the rest."""
    from ouroboros.runtime_limits import LAUNCHER_STOP_GRACE_SEC

    done = threading.Event()

    def run() -> None:
        try:
            cleanup()
        finally:
            done.set()

    threading.Thread(target=run, name="fault-census", daemon=True).start()
    bound = LAUNCHER_STOP_GRACE_SEC if _FAULT_CENSUS_BOUND_SEC is None else _FAULT_CENSUS_BOUND_SEC
    if not done.wait(bound):
        log.critical("The pre-exec census did not finish within %ss; the successor starts over it and the next "
                     "generation's startup sweep settles the remainder", bound)
_FAULT_EXIT_LOCK = threading.Lock()
_FAULT_EXIT_CODE: Optional[int] = None  # the decision taken, so main's return and the backstop never take two


def bind_fault_exit(host: str, port: int, *, launcher_managed: bool, repo_dir: pathlib.Path,
                    cleanup: Callable[[], Any], restart: Callable[..., Any]) -> None:
    """``main()`` binds the fault door's one exit decision before serving: its own return and the backstop
    timer take the same one (``fault_exit``), so neither can leave a direct install down."""
    global _FAULT_EXIT
    _FAULT_EXIT = functools.partial(_decide_fault_exit, host, port, launcher_managed=launcher_managed,
                                    repo_dir=repo_dir, cleanup=cleanup, restart=restart)


def fault_exit() -> int:
    """``main()``'s return after a fault latched the stop, and the backstop's exit code: the bound decision,
    taken once (the first caller, main or the backstop timer, decides and the other reads the same answer,
    so the census and the successor happen once). 0 without a fault; 1 for a fault latched before any
    binding (the launcher reading)."""
    from ouroboros.server_control import panic_termination_requested

    global _FAULT_EXIT_CODE
    if _stop_source not in FAULT_CAUSES or _restart_requested.is_set():
        return 0
    if panic_termination_requested():
        return 1  # Panic owns the exit (``exit_after_shutdown`` parks main's return); no census, no successor
    with _FAULT_EXIT_LOCK:
        if _FAULT_EXIT_CODE is None:
            _FAULT_EXIT_CODE = _FAULT_EXIT() if _FAULT_EXIT is not None else 1
        return _FAULT_EXIT_CODE


def _decide_fault_exit(host: str, port: int, *, launcher_managed: bool, repo_dir: pathlib.Path,
                       cleanup: Callable[[], Any], restart: Callable[..., Any]) -> int:
    """The Fault door's exit decision (ARCHITECTURE §9); ``fault_exit`` takes it once.

    Launcher-managed installs (desktop, Linux browser fallback, managed Docker, systemd, Android) return 1:
    the launcher's crash path restarts the same code (sleep 2 + 3 s, fuse 5 in 120 s) and spares the shared
    daemon; never 42 (no sync, dependency install, handoff, store switch or adoption) and never 99. Direct
    installs (plain Docker, Colab, ``ouroboros server``, bare POSIX) run the pre-exec census first
    (``_emergency_process_cleanup`` with ``_restart_requested`` clear: no adoption, no pin stop, surviving
    children force-killed), drop any planned-restart transaction and hand over through
    ``restart_current_process(owner_initiated=False)``: a same-PID exec on POSIX; on Windows a supervised
    successor whose parent exit 0 is the handoff. A failed successor returns 1. Recovery is the behaviour on
    every install (owner question A, option A); an install preference to stop and wait would be one more
    gate before the restart here, nothing else.
    """
    if _stop_source not in FAULT_CAUSES or _restart_requested.is_set():
        return 0
    if launcher_managed:
        log.critical("Server generation ended (%s); exiting 1 for the launcher's restart", _stop_source)
        return 1
    try:
        _bounded_census(cleanup)
        from ouroboros.delegate_recovery import PLANNED_RESTART_TRANSACTION_ENV

        os.environ.pop(PLANNED_RESTART_TRANSACTION_ENV, None)
        log.critical("Server generation ended (%s); restarting this direct server in place", _stop_source)
        restart(host, port, repo_dir=repo_dir, log=log, owner_initiated=False)
    except Exception:
        log.exception("Fault restart failed; this server exits without a successor")
        return 1
    return 0


def _embedded_uvicorn_server(config: "uvicorn.Config") -> "uvicorn.Server":
    """A uvicorn.Server hosted INSIDE the main server's event loop (the Host Service).

    ``Server.serve()`` installs the process signal handlers whenever it runs on the main
    thread and forwards a captured signal to the previous handler only after it has
    finished — so an embedded server would take SIGTERM away from ``_SignalStopServer``
    and hold it behind its own unbounded drain (#1142). The instance-level
    ``nullcontext`` keeps the main server the one signal owner; the lifespan teardown
    still stops this server through ``should_exit``.
    """
    server = uvicorn.Server(config)
    server.capture_signals = nullcontext  # type: ignore[method-assign]
    return server
