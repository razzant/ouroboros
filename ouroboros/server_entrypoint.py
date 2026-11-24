"""CLI, data-root admission, port-binding and interpreter-tuning helpers extracted from server.py."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
import errno
import gc
from ipaddress import ip_address
import json
import logging
import os
import pathlib
import socket
import sys
import threading
import time
from typing import NamedTuple

log = logging.getLogger(__name__)

SERVER_LOCK_NAME = "server.lock"
_admission: dict = {}  # {"root": realpath, "fd": descriptor} of the one data root this process serves
_admission_lock = threading.Lock()


class ServerAdmissionRefused(RuntimeError):
    """This process may not serve the data root: another server holds it, or holding is unprovable."""


class CliResetReplay(NamedTuple):
    ordinary: tuple[str, ...]
    reset: tuple[str, ...]
    host: str


_cli_reset = ContextVar[CliResetReplay | None]("cli_reset_replay", default=None)


@contextmanager
def cli_reset_scope(replay: CliResetReplay | None):
    token = _cli_reset.set(replay)
    try:
        yield
    finally:
        _cli_reset.reset(token)


class ResetLaunch:
    """In-process launch evidence; never a public credential or a launcher login."""

    def __init__(self, repo_dir: pathlib.Path, managed: bool):
        from ouroboros.server_control import restart_arguments
        from ouroboros.settings_integrity import (
            LAUNCHER_RESET_PROTECTION_ENV, PROJECTED_SETTINGS_ENV,
        )

        self.repo_dir, self.managed = repo_dir, managed
        self.cli = None if managed else _cli_reset.get()
        self.argv = () if managed else restart_arguments(os.environ)
        mark = os.environ.pop(LAUNCHER_RESET_PROTECTION_ENV, "")
        if not managed or os.environ.get("OUROBOROS_LAUNCHER_STOP_STDIN") != "1":
            mark = ""
        self.launcher_mode, _, self.launcher_host = mark.partition(" ")
        self.boot_host = ""
        if not managed and self._owns_direct_replay(os.environ):
            os.environ.setdefault(PROJECTED_SETTINGS_ENV, "{}")

    def _owns_direct_replay(self, environ) -> bool:
        from ouroboros.launch_roots import same_location
        from ouroboros.server_control import restart_arguments

        try:
            if not self.argv or restart_arguments(environ) != self.argv:
                return False
            if self.cli is not None:
                script = pathlib.Path(self.cli.reset[0])
                selected = self.argv == self.cli.ordinary and script.is_absolute()
            else:
                script = pathlib.Path(self.argv[0])
                # The spawn fallback uses repo_dir; POSIX exec keeps cwd.
                selected = self.argv == tuple(sys.argv) and (
                    script.is_absolute() or same_location(pathlib.Path.cwd(), self.repo_dir))
            return selected and script.is_file() and same_location(script, self.repo_dir / "server.py")
        except (OSError, ValueError):
            return False

    def select_reset_reexec(self, outcome) -> None:
        """Only the proved Reset handoff uses the CLI producer's absolute script recipe."""
        if self.cli is not None and isinstance(outcome, dict) and outcome.get("status") == "reset_requested":
            os.environ["OUROBOROS_SERVER_REEXEC_ARGV_JSON"] = json.dumps(self.cli.reset)

    def _plan(self, data_dir, environ, *, boot: bool = False):
        from ouroboros import config
        from ouroboros.launch_roots import same_location
        from ouroboros.settings_integrity import (
            drop_projection, own_reset_settings, reset_bind_protection, reset_operator_password,
        )

        if (not same_location(data_dir, config.DATA_DIR)
                or not own_reset_settings(config.DATA_DIR, config.SETTINGS_PATH, environ)):
            return None
        password = reset_operator_password(environ)
        if password is None:
            return None
        if self.managed:
            if not self.launcher_mode:
                return None
            if not boot:
                return self.launcher_mode, self.launcher_host, password
        elif not self._owns_direct_replay(environ):
            return None
        post = dict(environ)
        drop_projection(post)
        key = "OUROBOROS_SERVER_HOST"
        default = (post.get(key) or "").strip() or config.SETTINGS_DEFAULTS[key]
        host = ((self.cli.host or default) if self.cli is not None
                else parse_server_args(default, 0, argv=list(sys.argv[1:])).host)
        mode = reset_bind_protection(host, password)
        if not mode or (self.managed and (self.launcher_mode != mode or self.launcher_host != host)):
            return None
        return mode, host, password

    @staticmethod
    def _passwords(environ) -> tuple[str, str]:
        from ouroboros import config
        from ouroboros.server_auth import resolve_network_password
        from ouroboros.settings_integrity import (
            NETWORK_PASSWORD_KEY, read_settings_json_strict, settings_read_environ,
        )

        # No settings writer lock here: Reset can already hold the profile state lock.
        raw = read_settings_json_strict(config.SETTINGS_PATH)
        with settings_read_environ(environ):
            settings = config.settings_of_document(raw, present=raw is not None)
        active = resolve_network_password(environ.get(NETWORK_PASSWORD_KEY), lambda: settings)
        projected = resolve_network_password(None, lambda: settings)
        return active, projected

    def approve(self, bound_host: str, data_dir: pathlib.Path) -> str:
        from ouroboros.settings_integrity import SETTINGS_ENV_LOCK, reset_bind_protection

        try:
            with SETTINGS_ENV_LOCK:
                env = os.environ.copy()
                plan = self._plan(data_dir, env)
                if plan is None:
                    return ""
                mode, host, password = plan
                active, projected = self._passwords(env)
                return mode if (active == projected == password and ip_address(host) == ip_address(bound_host)
                                and reset_bind_protection(bound_host, active) == mode) else ""
        except (OSError, RuntimeError, ValueError, SystemExit):
            return ""

    def at_boot(self, data_dir: pathlib.Path) -> str:
        try:
            env = os.environ.copy()
            plan = self._plan(data_dir, env, boot=True)
            if plan is None:
                return ""
            mode, host, password = plan
            # A partial cleanup may still leave settings.json for lifespan to project.
            if self._passwords(env) != (password, password):
                return ""
            self.boot_host = host
            return mode
        except (OSError, RuntimeError, ValueError, SystemExit):
            return ""


def unisolated_test_root(root: pathlib.Path) -> bool:
    """Keep both server entry points off the default profile in a bare pytest run or its subprocess."""
    from ouroboros.launch_roots import same_location

    return (("pytest" in sys.modules or bool(os.environ.get("PYTEST_CURRENT_TEST")))
            and not os.environ.get("OUROBOROS_DATA_DIR")
            and same_location(root, pathlib.Path.home() / "Ouroboros" / "data"))


def admit_data_root(data_dir: pathlib.Path, *, wait_sec: float | None = None, reset_launch: ResetLaunch | None = None) -> None:
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
        # No outgoing server can write again. Consume its one-shot Reset before
        # another caller can observe admission or initialize the new profile.
        from supervisor.state_initialization import accept_runtime_reset
        try:
            path = pathlib.Path(root)
            accept_runtime_reset(path, boot_protection=None if reset_launch is None else lambda: reset_launch.at_boot(path))
        except Exception as exc:
            os.close(fd)
            raise ServerAdmissionRefused(f"Reset at data-root admission failed: {exc}") from exc
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


def parse_server_args(default_host: str, default_port: int, *, argv: list[str] | None = None) -> argparse.Namespace:
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
    args = parser.parse_args(argv)
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


def shorten_switch_interval() -> None:
    """A computing thread of the server is asked to hand the interpreter over after 1 ms instead of CPython's 5 ms when
    another thread waits for it (owner, 14.11.2026, #3399). A hundred threads share one interpreter, and a read takes
    the lock back after each SQLite step or file call only when its holder yields: beside one busy thread the
    extensions index took 4.7 s at 5 ms and 1.9 s at 1 ms (0.1 s alone). It is a requested interval, not a bound: a
    long C call that holds the lock (a large JSON decode, a fork), other waiters and the OS scheduler can stretch a
    wait; Windows waits in whole milliseconds, so its timer resolution can limit the gain (1 ms is its shortest wait). A
    thread nobody waits for is never asked, so an uncontended interpreter pays nothing; a contended computing thread
    hands over more often, at a cost that grows with the waiters (beside one thread making file calls it ran about
    14 % slower while those calls ran 2.8 times faster; beside twelve, a quarter to a half slower). Children started by
    spawn, forkserver or exec keep CPython's default; a forked one inherits this. No setting."""
    from ouroboros.runtime_limits import SERVER_SWITCH_INTERVAL_SEC

    sys.setswitchinterval(SERVER_SWITCH_INTERVAL_SEC)
    log.info("Interpreter switch interval %.1f ms", sys.getswitchinterval() * 1000)


def raise_gc_threshold() -> None:
    """The server collects its young generation after 20,000 net new container objects instead of CPython's 700 (owner,
    15.11.2026, #3399). Decoding a task body creates thousands to tens of thousands of dicts and lists, and every 700 of
    them a collection ran while holding the interpreter: on the owner's live server collections held it 7-19 % of the
    time, about 80 % of them started inside JSON decoding. On a stand with nine million long-lived objects, 300 decodes
    of a 1.5 MB body (about 7,000 containers) ran 2,751 young and 250 middle collections, 15 % of the decoding time;
    with 20,000, one. A body above 20,000 containers still collects, in fewer and longer pauses: with eight 10 MB
    bodies alive, middle collections took about 120 ms each instead of under 1 ms. Objects still die by reference
    counting at once; cyclic garbage, and whatever only it holds, waits for a later collection: a count of objects, not
    a bound on bytes (some thousands of small cycles' payloads can wait), and an idle interpreter collects nothing. The
    older generations keep CPython's thresholds; middle collections, counted in young ones, come about 28 times rarer,
    and a full collection still waits for CPython's long-lived ratio as well. Children started by spawn, forkserver or
    exec keep CPython's default; a forked one inherits this. No setting."""
    from ouroboros.runtime_limits import SERVER_GC_THRESHOLD0

    gc.set_threshold(SERVER_GC_THRESHOLD0)  # the older two keep whatever they are
    log.info("Garbage collector thresholds %s", gc.get_threshold())
