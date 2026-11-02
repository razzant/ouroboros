"""CLI, data-root admission and port-binding helpers extracted from server.py."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import errno
import logging
import os
import pathlib
import socket
import threading
import time

log = logging.getLogger(__name__)

SERVER_LOCK_NAME = "server.lock"
_admission: dict = {}  # {"root": realpath, "fd": descriptor} of the one data root this process serves
_admission_lock = threading.Lock()


class ServerAdmissionRefused(RuntimeError):
    """This process may not serve the data root: another server holds it, or holding is unprovable."""


def admit_data_root(data_dir: pathlib.Path, *, wait_sec: float | None = None) -> None:
    """Hold this process's exclusive admission to ``data_dir`` until the process exits.

    One kernel lock on the fixed ``<data>/state/server.lock`` (``flock`` / ``LockFileEx``)
    is the authority: whatever repository, PID file, port or process record a second
    server uses, it cannot hold the same data root at once. The file's existence, PID
    stamp and age prove nothing, so the inode is never unlinked or replaced and no stale
    fallback exists. The descriptor is not inherited across exec/spawn, a forked child
    closes its copy without unlocking the shared description, and listener or lifespan
    teardown keeps it: the restart cleanup that follows still owns the root. Repeating
    the call for the same root is a no-op. A spawned direct-Restart successor waits up
    to ``SERVER_ADMISSION_WAIT_SEC`` for its exiting parent; contention past that, or a
    filesystem without kernel locks, refuses with the reason.
    """
    from ouroboros.platform_layer import file_lock_exclusive_nb
    from ouroboros.runtime_limits import SERVER_ADMISSION_WAIT_SEC

    root = os.path.realpath(data_dir)
    lock_path = pathlib.Path(root) / "state" / SERVER_LOCK_NAME
    with _admission_lock:
        if _admission.get("root") == root:
            return
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
        deadline = time.monotonic() + (SERVER_ADMISSION_WAIT_SEC if wait_sec is None else wait_sec)
        while True:
            try:
                file_lock_exclusive_nb(fd)
                break
            except OSError as exc:
                held = exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK)
                if held and time.monotonic() < deadline:
                    time.sleep(0.1)
                    continue
                try:
                    holder = os.read(fd, 64).decode("ascii", "replace").strip() or "unknown holder"
                except OSError:
                    holder = "unknown holder"
                finally:
                    os.close(fd)
                if held:
                    raise ServerAdmissionRefused(
                        f"another Ouroboros server serves this data directory ({holder}, {lock_path}); "
                        "one data directory admits one server — stop it first") from exc
                raise ServerAdmissionRefused(
                    f"kernel file locks are unavailable at {lock_path} (errno {exc.errno}): admission "
                    "cannot be proven, so this server does not start on that data directory") from exc
        try:  # informational only; the kernel hold is the admission
            os.ftruncate(fd, 0)
            os.write(fd, f"pid={os.getpid()}\n".encode("ascii"))
        except OSError:
            log.debug("Could not stamp %s", lock_path, exc_info=True)
        previous = _admission.get("fd")
        _admission.update(root=root, fd=fd)
    if previous is not None:  # a second root in one process exists only in tests
        os.close(previous)
    pin_work_root_basis(data_dir)


def pin_work_root_basis(data_dir: pathlib.Path) -> str:
    """Held now, so no other server writes this root's history: settle its work-root basis
    before any boot step creates a file or a folder, for this process and every child it
    starts (``launch_roots``). An unreadable witness, state or store answers ``unknown``,
    never a guess; the store's chat rows are counted by their own reader."""
    from ouroboros.launch_roots import pin_basis
    from supervisor.state_initialization import supervisor_evidence

    return pin_basis(data_dir, evidence=supervisor_evidence)


def admission_held(data_dir: pathlib.Path) -> bool:
    """Whether THIS process holds ``data_dir``'s admission (the same actual root).

    While it does, no other process holds or serves that root, and one that held it
    earlier has exited: a server keeps its admission until it exits.
    """
    with _admission_lock:
        return _admission.get("fd") is not None and _admission.get("root") == os.path.realpath(data_dir)


def release_admission_for_tests() -> None:
    """Test isolation only: end this process's admission between tests."""
    with _admission_lock:
        fd = _admission.pop("fd", None)
        _admission.clear()
    if fd is not None:
        os.close(fd)


def _close_admission_in_forked_child() -> None:
    # Never LOCK_UN: the parent shares this open file description and keeps its hold.
    global _admission_lock
    _admission_lock = threading.Lock()  # another thread may have held it at the fork
    fd = _admission.pop("fd", None)
    _admission.clear()
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_close_admission_in_forked_child)


def _can_bind_port(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def reserved_service_ports() -> set:
    """This installation's other service endpoints, never a main-port fallback."""
    from ouroboros.config import SETTINGS_DEFAULTS, load_settings

    settings, ports = load_settings(), set()
    for key in ("OUROBOROS_HOST_SERVICE_PORT", "LOCAL_MODEL_PORT"):
        try:
            ports.add(int(settings.get(key) or SETTINGS_DEFAULTS[key]))
        except (TypeError, ValueError):
            ports.add(int(SETTINGS_DEFAULTS[key]))
    return ports


def find_free_port(host: str, start: int = 8765, max_tries: int = 10,
                   wait_retries: int = 20, wait_interval: float = 0.5, reserved=()) -> int:
    """Prefer the old port during restart before scanning nearby fallbacks.

    ``reserved`` names this installation's other service endpoints; a fallback
    never lands on one (the requested port itself is the operator's choice).
    """
    for attempt in range(wait_retries):
        if _can_bind_port(host, start):
            return start
        if attempt < wait_retries - 1:
            time.sleep(wait_interval)

    # Fallback ports may also be winding down; retry the whole range.
    fallback_ports = [port for port in range(start + 1, start + max_tries) if port not in set(reserved)]
    for attempt in range(wait_retries):
        for port in fallback_ports:
            if _can_bind_port(host, port):
                return port
        if attempt < wait_retries - 1:
            time.sleep(wait_interval)

    raise OSError(f"No free port available in range {start}-{start + max_tries - 1}")


def parse_server_args(default_host: str, default_port: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Ouroboros web server.")
    parser.add_argument(
        "--host",
        default=None,
        help=f"Host interface to bind (default: {default_host} or OUROBOROS_SERVER_HOST).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help="Port to bind (default: %(default)s or OUROBOROS_SERVER_PORT).",
    )
    args = parser.parse_args()
    args.host_explicit = args.host is not None
    if not args.host_explicit:
        args.host = default_host
    return args


def write_port_file(port_file: pathlib.Path, port: int) -> None:
    port_file.parent.mkdir(parents=True, exist_ok=True)
    port_file.write_text(str(port), encoding="utf-8")


@contextmanager
def bound_service_socket(drive_root: pathlib.Path, service: str, host: str, port: int,
                         *, server_host_source: str | None = None):
    """Keep the actual bound socket and its informational identity in one lifetime.

    The existing port selector chooses the port. Uvicorn accepts this socket on
    Linux, macOS and Windows; no second probe/rebind race or process authority.
    """
    from ouroboros.server_process import clear_service_binding, record_service_binding, record_applied_restart_settings

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    binding = None
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        address = sock.getsockname()
        if service in {"main", "host_service"}:
            record_applied_restart_settings({
                "OUROBOROS_SERVER_HOST" if service == "main" else "OUROBOROS_HOST_SERVICE_PORT":
                    host if service == "main" else address[1],
            }, server_host_source=server_host_source if service == "main" else None)
        try:
            binding = record_service_binding(drive_root, service, address[0], address[1], pid=os.getpid())
        except Exception:
            # Endpoint discovery is informational. Main still writes its actual
            # port, which keeps restricted browsers conservative on this address.
            log.warning("Could not record %s service binding; continuing with the bound socket", service, exc_info=True)
        yield sock
    finally:
        sock.close()
        if binding is not None:
            try:
                clear_service_binding(drive_root, service, binding)
            except Exception:
                log.warning("Could not retire %s service binding", service, exc_info=True)
