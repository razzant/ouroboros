"""Facts one server process shares with every server leaf.

The drive root it was launched against, the ``server`` logger every server
module writes to, and the restart-request signals plus the setter that raises
them. These live below the composition root so a leaf can read them without
importing ``server`` back.
"""

from __future__ import annotations

import logging
import os
import pathlib
import re
import threading
from typing import Callable

from ouroboros.utils import read_json_dict, update_json_locked


DATA_DIR = pathlib.Path(os.environ.get("OUROBOROS_DATA_DIR",
    pathlib.Path.home() / "Ouroboros" / "data"))


log = logging.getLogger("server")


_restart_requested = threading.Event()
# Set FIRST in the lifespan teardown: the supervisor loop reads it in its
# ``while`` and in its crash handler, so the bus/Manager being torn down by
# the shutdown itself never counts as a loop crash (no false "died after 3
# consecutive crashes" alarm on a graceful window close / SIGTERM).
_supervisor_stop = threading.Event()


# Set only when the OWNER asked for the restart (the chat Restart button, and the
# control endpoints that restart on the owner's behalf). The single fact the
# re-exec needs to decide whether the runtime-mode ratchet pin rides along.
_owner_restart_requested = threading.Event()


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
    """The integer port the last main server wrote, or None when absent/unreadable."""
    try:
        text = (pathlib.Path(drive_root) / "state" / "server_port").read_text(encoding="utf-8").strip()
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
