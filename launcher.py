"""Immutable desktop launcher: bootstrap repo, manage server.py, and host UI."""

from __future__ import annotations

import base64
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from logging.handlers import RotatingFileHandler
from typing import Optional

# WA6: set sys.dont_write_bytecode BEFORE importing any project module. A signed
# macOS .app must never write __pycache__/*.pyc into its own bundle at runtime
# (that breaks the codesign seal and triggers AppTranslocation). os.environ alone
# is INSUFFICIENT for THIS process: PYTHONDONTWRITEBYTECODE is only read at
# interpreter startup, so mutating os.environ later does not stop the current
# process's own subsequent imports — only sys.dont_write_bytecode does. The env
# vars set further below propagate the same policy to child processes.
sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from ouroboros.config import (
    AGENT_SERVER_PORT,
    DATA_DIR,
    PANIC_EXIT_CODE,
    PORT_FILE,
    REPO_DIR,
    RESTART_EXIT_CODE,
    SETTINGS_PATH,
    SETTINGS_DEFAULTS,
    acquire_pid_lock,
    apply_settings_to_env as _apply_settings_to_env,
    load_settings,
    get_runtime_mode,
    normalize_runtime_mode,
    read_version,
    release_pid_lock,
    save_settings,
)
from ouroboros.launcher_bootstrap import (
    BootstrapContext,
    bootstrap_repo as _bootstrap_repo,
    check_git as _check_git,
    install_deps as _install_deps_impl,
    embedded_python_env,
    sync_existing_repo_from_bundle as _sync_existing_repo_from_bundle_impl,
)
from ouroboros.launcher_onboarding import (
    prepare_first_run_settings as _prepare_first_run_settings,
    present_first_run_onboarding as _present_first_run_onboarding,
)
from ouroboros.launcher_server_reaper import (
    reap_same_install_strays as _reap_same_install_strays_impl,
)
from ouroboros.launcher_windows_runtime import (  # noqa: F401  (re-exported: same objects, prior launcher surface)
    _prepare_windows_webview_runtime,
    _show_windows_message,
)
from ouroboros.platform_layer import (
    BUNDLE_DIR_ENV,
    IS_LINUX,
    IS_MACOS,
    IS_WINDOWS,
    assign_pid_to_job,
    close_job,
    create_kill_on_close_job,
    current_process_group_id,
    embedded_python_candidates,
    force_kill_pid,
    git_install_hint,
    install_shutdown_signal_handlers,
    kill_pid_tree,
    kill_process_group_id,
    kill_process_on_port,
    kill_process_tree,
    merge_hidden_kwargs,
    open_path_external,
    pid_is_alive,
    process_command,
    process_group_id,
    resume_process,
    subprocess_new_group_kwargs,
    terminate_job,
    terminate_process_group_id,
    terminate_process_tree,
)
from ouroboros.utils import atomic_write_json, utc_now_iso

MAX_CRASH_RESTARTS = 5
CRASH_WINDOW_SEC = 120
# One bounded, visible retry when a restart's dependency install fails (XG-7B.3):
# long enough to ride out a transient index/network hiccup, short enough not to
# stall an offline restart whose requirements are already satisfied.
_DEPS_RETRY_DELAY_SEC = 5
_CREATE_SUSPENDED = getattr(subprocess, "CREATE_SUSPENDED", 0x4) if IS_WINDOWS else 0
_CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if IS_WINDOWS else 0

# WA6: globally suppress bytecode writes for the launcher process itself and any
# naive os.environ.copy() child it spawns. A signed+notarized macOS .app must not
# write __pycache__/*.pyc into its own bundle at runtime — that breaks the codesign
# seal and triggers AppTranslocation. Uses the same data_dir/state/pycache
# convention as launcher_bootstrap.embedded_python_env so caches land outside the
# bundle. setdefault keeps any explicit caller override.
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
_pycache_dir = DATA_DIR / "state" / "pycache"
try:
    _pycache_dir.mkdir(parents=True, exist_ok=True)
except OSError:
    pass
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(_pycache_dir))

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_log_dir = DATA_DIR / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)

_file_handler = RotatingFileHandler(
    _log_dir / "launcher.log",
    maxBytes=2 * 1024 * 1024,
    backupCount=2,
    encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
_handlers: list[logging.Handler] = [_file_handler]
if not getattr(sys, "frozen", False):
    _handlers.append(logging.StreamHandler())
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, handlers=_handlers)
# Secret-redaction filter (shared SSOT in ouroboros.observability) + quiet
# third-party HTTP request-URL lines (they can carry URL credentials).
try:
    from ouroboros.observability import SecretRedactingLogFilter as _RedactFilter

    for _handler in _handlers:
        _handler.addFilter(_RedactFilter())
except Exception:
    pass  # defensive: a broken observability import must not kill the launcher
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("launcher")


APP_VERSION = read_version()


def _server_process_record_path() -> pathlib.Path:
    return pathlib.Path(DATA_DIR) / "state" / "server_process.json"


def _hidden_run(command, **kwargs):
    """subprocess.run() with platform-appropriate hidden-window flags."""
    return subprocess.run(command, **merge_hidden_kwargs(kwargs))


def _hidden_popen(command, **kwargs):
    """subprocess.Popen() with platform-appropriate hidden-window flags."""
    return subprocess.Popen(command, **merge_hidden_kwargs(kwargs))

def _find_embedded_python() -> str:
    """Locate the embedded python-build-standalone interpreter."""
    if getattr(sys, "frozen", False):
        base = pathlib.Path(sys._MEIPASS)
    else:
        base = pathlib.Path(__file__).parent
    for path in embedded_python_candidates(base):
        if path.exists():
            return str(path)
    return sys.executable


EMBEDDED_PYTHON = _find_embedded_python()


def _bundle_dir() -> pathlib.Path:
    if getattr(sys, "frozen", False):
        return pathlib.Path(sys._MEIPASS)
    return pathlib.Path(__file__).parent


def _bootstrap_context() -> BootstrapContext:
    return BootstrapContext(
        bundle_dir=_bundle_dir(),
        repo_dir=REPO_DIR,
        data_dir=DATA_DIR,
        settings_path=SETTINGS_PATH,
        embedded_python=EMBEDDED_PYTHON,
        app_version=APP_VERSION,
        hidden_run=_hidden_run,
        # Launcher is owner-process boundary; first-launch migration may set runtime mode.
        save_settings=lambda settings: save_settings(settings, allow_elevation=True),
        log=log,
    )


def check_git() -> bool:
    return _check_git(IS_WINDOWS)


def bootstrap_repo() -> bool:
    return _bootstrap_repo(_bootstrap_context())


def _sync_existing_repo_from_bundle() -> None:
    _sync_existing_repo_from_bundle_impl(_bootstrap_context())


def _install_deps() -> bool:
    return _install_deps_impl(_bootstrap_context())

_agent_proc: Optional[subprocess.Popen] = None
_agent_job: Optional[object] = None
_agent_lock = threading.Lock()
_shutdown_event = threading.Event()
# Set by _request_agent_restart(): the next agent exit is a REQUESTED recycle
# (first-run configuration adopted), not a crash — same shape as the exit-code-42
# branch, for the case where the launcher, not the agent, decided the restart.
_agent_restart_requested = threading.Event()
_webview_window = None
# Linux-only browser-fallback flag (#56): set by _detect_headless() when no
# pywebview GUI backend can initialize. Stays False on macOS/Windows — the
# probe never runs there, so every `if _headless:` branch is dead code on
# those platforms and their behavior is unchanged.
_headless = False


def _server_process_identity_matches(record: dict) -> bool:
    try:
        pid = int(record.get("pid") or 0)
    except (TypeError, ValueError):
        return False
    if pid <= 0 or pid == os.getpid() or not pid_is_alive(pid):
        return False
    expected_server = str((REPO_DIR / "server.py").resolve())
    expected_repo = str(REPO_DIR.resolve())
    record_server = str(record.get("server_path") or "")
    record_repo = str(record.get("repo_dir") or "")
    if record_server and record_server != expected_server:
        return False
    if record_repo and record_repo != expected_repo:
        return False
    live_pgid = process_group_id(pid)
    try:
        recorded_pgid = int(record.get("pgid") or 0)
    except (TypeError, ValueError):
        recorded_pgid = 0
    if not IS_WINDOWS and (recorded_pgid <= 0 or live_pgid <= 0 or recorded_pgid != live_pgid):
        return False
    command = process_command(pid)
    if not command:
        return False
    return expected_server in command or ("server.py" in command and expected_repo in command)


def _write_server_process_record(proc: subprocess.Popen, *, port: int, server_py: pathlib.Path) -> None:
    try:
        record = {
            "pid": int(proc.pid),
            "pgid": process_group_id(proc.pid),
            "server_path": str(server_py.resolve()),
            "repo_dir": str(REPO_DIR.resolve()),
            "requested_port": int(port),
            "port": int(port),
            "argv": [str(EMBEDDED_PYTHON), str(server_py.resolve())],
            "created_at": utc_now_iso(),
        }
        atomic_write_json(_server_process_record_path(), record, trailing_newline=True)
    except Exception:
        log.warning("Failed to write server process record", exc_info=True)


def _update_server_process_record_port(pid: int, actual_port: int) -> None:
    try:
        record_path = _server_process_record_path()
        if not record_path.exists():
            return
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or int(record.get("pid") or 0) != int(pid):
            return
        record["port"] = int(actual_port)
        if "requested_port" not in record:
            record["requested_port"] = int(actual_port)
        record["port_updated_at"] = utc_now_iso()
        atomic_write_json(record_path, record, trailing_newline=True)
    except Exception:
        log.debug("Failed to update server process record port", exc_info=True)


def _cleanup_recorded_server_process(reason: str = "preflight") -> None:
    try:
        record_path = _server_process_record_path()
        if not record_path.exists():
            return
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            record_path.unlink(missing_ok=True)
            return
        if not _server_process_identity_matches(record):
            log.info("Ignoring stale server process record with non-matching identity (%s)", reason)
            record_path.unlink(missing_ok=True)
            return
        pid = int(record.get("pid") or 0)
        pgid = int(record.get("pgid") or 0)
        log.info("Cleaning recorded server process pid=%d pgid=%d (%s)", pid, pgid, reason)
        if not IS_WINDOWS and pgid > 0 and pgid != current_process_group_id():
            terminate_process_group_id(pgid)
            time.sleep(0.5)
            kill_process_group_id(pgid)
        if pid_is_alive(pid):
            kill_pid_tree(pid)
        record_path.unlink(missing_ok=True)
    except Exception:
        log.warning("Failed to clean recorded server process (%s)", reason, exc_info=True)


def _cleanup_recorded_server_group_for_pid(pid: int, reason: str = "agent_exit") -> None:
    try:
        record_path = _server_process_record_path()
        if not record_path.exists():
            return
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or int(record.get("pid") or 0) != int(pid):
            return
        pgid = int(record.get("pgid") or 0)
        live_pgid = process_group_id(int(pid)) if pid_is_alive(int(pid)) else 0
        if not IS_WINDOWS and live_pgid > 0 and pgid > 0 and pgid != live_pgid:
            log.info(
                "Ignoring mismatched recorded server pgid=%d for live pid=%d pgid=%d (%s)",
                pgid,
                pid,
                live_pgid,
                reason,
            )
            pgid = 0
        if not IS_WINDOWS and pgid > 0 and pgid != current_process_group_id():
            log.info("Cleaning server process group pgid=%d after pid=%d exit (%s)", pgid, pid, reason)
            terminate_process_group_id(pgid)
            time.sleep(0.2)
            kill_process_group_id(pgid)
        if pid_is_alive(int(pid)):
            kill_pid_tree(int(pid))
        record_path.unlink(missing_ok=True)
    except Exception:
        log.warning("Failed to clean recorded server process group (%s)", reason, exc_info=True)


def start_agent(port: int = AGENT_SERVER_PORT) -> subprocess.Popen:
    """Start server.py as the managed agent subprocess."""
    global _agent_proc, _agent_job

    settings = _load_settings()
    _apply_settings_to_env(settings)
    env = embedded_python_env(DATA_DIR, os.environ.copy())
    env["PYTHONPATH"] = str(REPO_DIR)
    # ENV IS THE AUTHORITY for the bind host (config.SETTINGS_KEYS_NOT_EXPORTED_TO_ENV):
    # `settings` here is the MERGED view, so it usually carries the shipped
    # 127.0.0.1 default that no owner ever authored. Stamping it over an
    # operator-provided environment host reintroduced the exact regression the
    # export exclusion closed — setdefault lets the settings value stand in
    # only when the environment says nothing.
    saved_host = str(settings.get("OUROBOROS_SERVER_HOST") or "").strip()
    if saved_host:
        env.setdefault("OUROBOROS_SERVER_HOST", saved_host)
    env["OUROBOROS_SERVER_PORT"] = str(port)
    env["OUROBOROS_DATA_DIR"] = str(DATA_DIR)
    env["OUROBOROS_REPO_DIR"] = str(REPO_DIR)
    env["OUROBOROS_APP_VERSION"] = str(APP_VERSION)
    env["OUROBOROS_MANAGED_BY_LAUNCHER"] = "1"
    # Owner Surface Fact: the launcher is the only actor that knows HOW this
    # server will be presented. `_headless` is decided in main() before the
    # lifecycle loop ever calls start_agent(), and every managed restart funnels
    # back through here, so the export is re-stamped fresh each time. Absence of
    # the var (source mode, Docker, Colab, CLI server) truthfully means "web".
    # Env-only by design — never a SETTINGS_DEFAULTS key (pop-on-absent would
    # erase an injected value). Known bounded lie: a SIGKILLed launcher can
    # orphan the server with a stale "desktop_window" until the next launcher
    # start reaps it — the same envelope OUROBOROS_MANAGED_BY_LAUNCHER accepts.
    env["OUROBOROS_PRESENTATION"] = "browser_fallback" if _headless else "desktop_window"
    # The server runs out of the managed repo, not the bundle: without this the
    # bundled payloads (node, ripgrep) are invisible to it (platform_layer.
    # bundled_resource_bases).
    env[BUNDLE_DIR_ENV] = str(_bundle_dir())

    server_py = REPO_DIR / "server.py"
    log.info("Starting agent: %s %s (port=%d)", EMBEDDED_PYTHON, server_py, port)

    popen_kwargs: dict = {
        "cwd": str(REPO_DIR),
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
    }
    if IS_WINDOWS:
        popen_kwargs["creationflags"] = (
            popen_kwargs.get("creationflags", 0)
            | _CREATE_NEW_PROCESS_GROUP
            | _CREATE_SUSPENDED
        )
    else:
        popen_kwargs.update(subprocess_new_group_kwargs())

    proc = _hidden_popen([EMBEDDED_PYTHON, str(server_py)], **popen_kwargs)
    _agent_proc = proc

    if IS_WINDOWS:
        job = create_kill_on_close_job()
        if job is None:
            log.error(
                "Failed to create Windows Job Object; refusing to run without process-tree ownership."
            )
            proc.kill()
            return proc
        if not assign_pid_to_job(job, proc.pid):
            log.error(
                "Failed to assign agent pid %d to Windows Job Object; refusing to run without process-tree ownership.",
                proc.pid,
            )
            close_job(job)
            proc.kill()
            return proc
        _agent_job = job
        if not resume_process(proc.pid):
            log.error("Failed to resume agent process %d — killing", proc.pid)
            with _agent_lock:
                if _agent_job is job:
                    _agent_job = None
            terminate_job(job)
            close_job(job)
            return proc
        log.info("Agent pid %d assigned to Windows Job Object", proc.pid)

    _write_server_process_record(proc, port=port, server_py=server_py)

    def _stream_output() -> None:
        # Size-capped copy (CPL4-C5): same bound as the server.log stdlib
        # handler (2 MB live + numbered backups). Rotation failure must never
        # kill the copy thread — worst case the live file keeps growing, which
        # is exactly the pre-cap behavior.
        max_bytes = 2 * 1024 * 1024
        backups = 3

        def _rotate(log_path: pathlib.Path) -> None:
            try:
                for index in range(backups - 1, 0, -1):
                    older = log_path.with_name(f"{log_path.name}.{index}")
                    if older.exists():
                        os.replace(older, log_path.with_name(f"{log_path.name}.{index + 1}"))
                if log_path.exists():
                    os.replace(log_path, log_path.with_name(f"{log_path.name}.1"))
            except OSError:
                pass

        log_path = DATA_DIR / "logs" / "agent_stdout.log"
        try:
            written = log_path.stat().st_size if log_path.exists() else 0
        except OSError:
            written = 0
        handle = None
        try:
            handle = open(log_path, "a", encoding="utf-8")
            for line in iter(proc.stdout.readline, b""):
                decoded = line.decode("utf-8", errors="replace")
                if written + len(decoded) > max_bytes:
                    handle.close()
                    handle = None
                    _rotate(log_path)
                    handle = open(log_path, "a", encoding="utf-8")
                    written = 0
                handle.write(decoded)
                handle.flush()
                written += len(decoded)
        except Exception:
            pass
        finally:
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass

    threading.Thread(target=_stream_output, daemon=True).start()
    return proc


def stop_agent() -> None:
    """Gracefully stop the agent process."""
    global _agent_proc, _agent_job
    with _agent_lock:
        if _agent_proc is None:
            return
        proc = _agent_proc
        job = _agent_job
        _agent_proc = None
        _agent_job = None

    log.info("Stopping agent (pid=%s)...", proc.pid)
    try:
        if IS_WINDOWS:
            proc.terminate()
        else:
            terminate_process_tree(proc)
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if IS_WINDOWS and job is not None:
            terminate_job(job)
        else:
            kill_process_tree(proc)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log.warning("Agent process did not exit after forced stop (pid=%s)", proc.pid)
    except Exception:
        pass

    if IS_WINDOWS and job is not None:
        close_job(job)
    _cleanup_recorded_server_group_for_pid(proc.pid, "stop_agent")


def _read_port_file() -> int:
    """Read the active server port from PORT_FILE."""
    try:
        if PORT_FILE.exists():
            return int(PORT_FILE.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        pass
    return AGENT_SERVER_PORT


def _kill_stale_on_port(port: int) -> None:
    """Kill any process listening on a runtime port."""
    if IS_WINDOWS:
        kill_process_on_port(port)
        return
    try:
        # -sTCP:LISTEN keeps this a listener sweep: a bare tcp:PORT selector
        # also matches ESTABLISHED client sockets, and in browser mode the
        # owner's own browser holds one — sweeping it violates the invariant
        # documented on _open_browser_detached (the browser is the owner's
        # application, outside custody). -nP avoids resolver stalls.
        result = subprocess.run(
            ["lsof", "-nP", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        pids = result.stdout.strip().split()
        for pid_str in pids:
            try:
                pid = int(pid_str)
                if pid != os.getpid():
                    force_kill_pid(pid)
            except (TypeError, ValueError, ProcessLookupError, PermissionError, OSError):
                pass
    except Exception:
        kill_process_on_port(port)


def _host_service_port() -> int:
    default_port = int(SETTINGS_DEFAULTS.get("OUROBOROS_HOST_SERVICE_PORT", 8767))
    try:
        raw_port = os.environ.get("OUROBOROS_HOST_SERVICE_PORT")
        if not str(raw_port or "").strip():
            raw_port = _load_settings().get("OUROBOROS_HOST_SERVICE_PORT", default_port)
        return int(raw_port)
    except (TypeError, ValueError, OSError):
        return default_port


def _kill_stale_runtime_ports(port: int) -> None:
    """Clear core runtime listener ports before start/restart."""
    _kill_stale_on_port(port)
    _kill_stale_on_port(_host_service_port())


def _reap_same_install_strays(reason: str) -> list[int]:
    """Kill leftover generations of THIS install's server; return proven survivors.

    Only ever called while this process holds the single-instance pid lock, which is what makes the
    identity rule sound: with the lock held, another process running this install's server.py under
    this launcher's stamped environment cannot be a live peer's generation. Never called from a
    panic or window-close path — Emergency Stop tears down what it owns and adds no new killing.
    """
    try:
        return _reap_same_install_strays_impl(REPO_DIR, DATA_DIR, reason)
    except Exception:
        # A sweep that cannot run must not stop the launcher booting.
        log.warning("Same-install stray sweep failed (%s)", reason, exc_info=True)
        return []


def _pre_generation_cleanup(port: int) -> list[int]:
    """Clear the previous generation before starting a new one; returns proven stray survivors.

    Per GENERATION, not once per launcher: exit-42 and crash restarts are where the observed
    double-boot collisions began. Ordered recorded-cleanup -> stray sweep -> port sweep: the
    recorded pid keeps its record-driven path with the record unlinked first, the tree kill runs
    while PPID links are still live, and the port sweep stays the residual net."""
    _cleanup_recorded_server_process("startup")
    survivors = _reap_same_install_strays("startup")
    _kill_stale_runtime_ports(port)
    return survivors


def _wait_for_server(port: int, timeout: float = 30.0, abort_event=None) -> bool:
    """Wait for the agent HTTP server to respond.

    ``abort_event`` (headless mode) lets a shutdown signal cut the wait short:
    the handlers only set the event, so without this check a SIGTERM during
    startup would still sit out the full readiness timeout."""
    import urllib.request

    url = f"http://127.0.0.1:{port}/api/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if abort_event is not None and abort_event.is_set():
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _poll_port_file(timeout: float = 30.0) -> int:
    """Poll until the port file is freshly written."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if PORT_FILE.exists():
                age = time.time() - PORT_FILE.stat().st_mtime
                if age < 10:
                    return int(PORT_FILE.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            pass
        time.sleep(0.5)
    return _read_port_file()


def _kill_orphaned_children(port: int, reason: str = "window_close") -> None:
    """Final safety net: kill processes still on runtime ports.

    Module-level so both the window-close handler (_on_closing, main thread) and
    the panic-stop branch (agent_lifecycle_loop, supervisor thread) tear down the
    exact same way. ``reason`` names the actual trigger, journalled HERE because
    stop_agent() usually consumed the recorded-process row already.
    """
    log.info("Tearing down runtime orphans (%s)", reason)
    _cleanup_recorded_server_process(reason)
    _kill_stale_runtime_ports(port)
    _kill_stale_on_port(8766)
    try:
        companions = json.loads((DATA_DIR / "state" / "extension_companions.json").read_text(encoding="utf-8"))
        if isinstance(companions, dict):
            for item in companions.values():
                if not isinstance(item, dict):
                    continue
                try:
                    force_kill_pid(int(item.get("pid") or 0))
                except (TypeError, ValueError, ProcessLookupError, PermissionError, OSError):
                    pass
                for companion_port in item.get("ports") or []:
                    try:
                        kill_process_on_port(int(companion_port))
                    except (TypeError, ValueError, OSError):
                        pass
    except Exception:
        pass
    for child in __import__("multiprocessing").active_children():
        try:
            force_kill_pid(child.pid)
            log.info("Killed orphaned child pid=%d", child.pid)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def agent_lifecycle_loop(port: int = AGENT_SERVER_PORT) -> None:
    """Start/monitor agent; restart on code 42 or bounded crashes."""
    global _agent_proc, _agent_job
    crash_times: list[float] = []

    while not _shutdown_event.is_set():
        survivors = _pre_generation_cleanup(port)
        if survivors:
            # Starting now would put a second generation on the same data directory — the exact
            # collision this sweep exists to prevent. The next iteration re-sweeps.
            log.error(
                "Not starting the agent: same-install server process(es) %s are proven "
                "launcher-managed strays but survived every kill pass. Retrying in 3s.",
                survivors,
            )
            time.sleep(3)
            continue

        try:
            PORT_FILE.unlink(missing_ok=True)
        except OSError:
            pass

        proc = start_agent(port)

        actual_port = _poll_port_file(timeout=30)
        _update_server_process_record_port(proc.pid, actual_port)
        if not _wait_for_server(actual_port, timeout=45):
            log.warning("Agent server did not become responsive within 45s (port %d)", actual_port)

        proc.wait()
        exit_code = proc.returncode
        log.info("Agent exited with code %d", exit_code)
        if exit_code == RESTART_EXIT_CODE:
            try:
                from ouroboros.delegate_recovery import acknowledge_observed_restart_exit

                if not acknowledge_observed_restart_exit(
                    DATA_DIR, supervisor_pid=proc.pid, exit_code=exit_code,
                ):
                    log.info("No prepared delegated-restart transaction required acknowledgement.")
            except Exception:
                log.warning("Could not acknowledge delegated restart transaction", exc_info=True)
        _cleanup_recorded_server_group_for_pid(proc.pid, "agent_exit")

        with _agent_lock:
            _agent_proc = None
            if IS_WINDOWS and _agent_job is not None:
                close_job(_agent_job)
                _agent_job = None

        if _shutdown_event.is_set():
            break

        if exit_code == PANIC_EXIT_CODE:
            log.info("Panic stop (exit code %d) — shutting down completely.", PANIC_EXIT_CODE)
            _shutdown_event.set()
            # The agent (server child) already exited; tear down any orphans and
            # force-exit the whole process. _webview_window.destroy() from this
            # supervisor thread cannot end the main-thread Cocoa webview loop on
            # macOS (it leaves a black frozen window), so exit with parity to the
            # window-close path: kill orphans, release the pid lock, os._exit(0).
            _kill_orphaned_children(port, reason="panic_stop")
            release_pid_lock()
            os._exit(0)

        time.sleep(2)

        if _agent_restart_requested.is_set():
            # Launcher-requested recycle (first-run configuration): deliberate,
            # so it skips crash accounting exactly like the agent's own code-42
            # restart. No dependency sync — nothing about the checkout changed.
            _agent_restart_requested.clear()
            log.info("Restarting the agent to adopt the saved first-run configuration.")
            # No port sweep here: _pre_generation_cleanup at the top of the
            # next iteration runs it as its third phase.
            continue

        if exit_code == RESTART_EXIT_CODE:
            log.info("Agent requested restart (exit code 42). Restarting...")
            _sync_existing_repo_from_bundle()
            if not _install_deps():
                # An evolved checkout may have added requirements its reviewed
                # commit depends on. Pause visibly and retry once — pip failures
                # are often transient (index/network) — instead of restarting as
                # if nothing happened.
                log.error(
                    "Dependency install failed after the restart request; "
                    "retrying once in %ds.", _DEPS_RETRY_DELAY_SEC,
                )
                time.sleep(_DEPS_RETRY_DELAY_SEC)
                if not _install_deps():
                    log.error(
                        "Dependency install failed twice; starting the agent anyway "
                        "under the crash fuse (%d crashes in %ds stops the launcher). "
                        "If the new commit added packages, the server will fail to "
                        "import them — see the pip output above for the cause.",
                        MAX_CRASH_RESTARTS, CRASH_WINDOW_SEC,
                    )
            # No port sweep here: _pre_generation_cleanup owns it next iteration.
            continue

        now = time.time()
        crash_times.append(now)
        crash_times[:] = [stamp for stamp in crash_times if (now - stamp) < CRASH_WINDOW_SEC]
        if len(crash_times) >= MAX_CRASH_RESTARTS:
            log.error("Agent crashed %d times in %ds. Stopping.", MAX_CRASH_RESTARTS, CRASH_WINDOW_SEC)
            break

        log.info("Agent crashed. Restarting in 3s...")
        # No port sweep here: _pre_generation_cleanup owns it next iteration.
        time.sleep(3)

def _request_agent_restart() -> None:
    """Recycle the managed server so it adopts freshly saved settings.

    The lifecycle loop owns the restart; this flags intent and stops the child.
    """
    with _agent_lock:
        agent_alive = _agent_proc is not None
    if not agent_alive:
        # Leaving the flag set would make the NEXT ordinary agent exit look like
        # a requested restart and skip the crash fuse.
        log.info("No managed agent to recycle; the next start reads the saved settings.")
        return
    _agent_restart_requested.set()
    stop_agent()


def _await_server_ready(port: int, abort_event=None) -> tuple[bool, int]:
    """Wait for managed-server health; resolve the AUTHORITATIVE bound port.

    The server may rebind on conflict and publish the real port in
    ``data/state/server_port``; every later consumer (UI URL, onboarding window,
    teardown sweep) must use that value, not the requested one.
    """
    ready = _wait_for_server(port, timeout=15, abort_event=abort_event)
    actual_port = _read_port_file()
    if actual_port != port:
        ready = _wait_for_server(actual_port, timeout=45, abort_event=abort_event)
    else:
        ready = ready or _wait_for_server(port, timeout=45, abort_event=abort_event)
    return ready, actual_port


def _load_settings() -> dict:
    return load_settings()


def _save_settings(settings: dict) -> None:
    # Owner-process boundary: first-run/env/provider saves may elevate runtime mode.
    save_settings(settings, allow_elevation=True)


def _request_runtime_mode_change(mode: str, confirm_fn) -> dict:
    new_mode = normalize_runtime_mode(mode)
    settings = _load_settings()
    pending_mode = normalize_runtime_mode(settings.get("OUROBOROS_RUNTIME_MODE"))
    active_mode = get_runtime_mode()
    restart_required = new_mode != active_mode
    if new_mode == pending_mode:
        return {"ok": True, "runtime_mode": new_mode, "restart_required": restart_required}
    message = (
        f"Change Ouroboros runtime mode from {pending_mode} to {new_mode}?\n\n"
        f"Current boot is still running in {active_mode} mode. "
        "This is an owner-only operation. The new mode is saved by the "
        "desktop launcher and takes effect after restart."
    )
    if not confirm_fn("Confirm Runtime Mode Change", message):
        return {"ok": False, "error": "Runtime mode change cancelled."}
    settings["OUROBOROS_RUNTIME_MODE"] = new_mode
    _save_settings(settings)
    return {"ok": True, "runtime_mode": new_mode, "restart_required": restart_required}


def _request_auto_grant_reviewed_skills_change(enabled: bool, confirm_fn) -> dict:
    settings = _load_settings()
    old_enabled = str(settings.get("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS") or "false").strip().lower() in {"1", "true", "yes", "on"}
    new_enabled = bool(enabled)
    if new_enabled == old_enabled:
        return {"ok": True, "enabled": new_enabled}
    if new_enabled:
        message = (
            "Enable auto-grant for reviewed skills?\n\n"
            "After this, any fresh executable skill review will grant the "
            "skill's manifest-declared settings keys and host permissions for "
            "that exact content hash. Only enable this for trusted closed-loop "
            "skill development."
        )
    else:
        message = "Disable auto-grant for reviewed skills?"
    if not confirm_fn("Confirm Reviewed Skill Auto-Grant", message):
        return {"ok": False, "error": "Auto-grant setting change cancelled."}
    settings["OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS"] = "true" if new_enabled else "false"
    _save_settings(settings)
    return {"ok": True, "enabled": new_enabled}


def _request_skill_key_grant(skill: str, keys: list, confirm_fn) -> dict:
    from ouroboros.skill_loader import (
        find_skill,
        requested_core_setting_keys,
        requested_skill_permissions,
        review_status_allows_execution,
        save_skill_grants,
    )

    skill_name = str(skill or "").strip()
    requested_raw = [str(k or "").strip() for k in (keys or []) if str(k or "").strip()]
    loaded = find_skill(
        DATA_DIR,
        skill_name,
        repo_path=str(_load_settings().get("OUROBOROS_SKILLS_REPO_PATH") or ""),
    )
    if loaded is None:
        return {"ok": False, "error": f"Skill {skill_name!r} not found"}
    if not (loaded.manifest.is_script() or loaded.manifest.is_extension()):
        return {"ok": False, "error": "Key and permission grants are supported for script and extension skills."}
    if not review_status_allows_execution(loaded.review.status) or loaded.review.is_stale_for(loaded.content_hash):
        return {"ok": False, "error": "Key and permission grants require a fresh executable review."}
    allowed = requested_core_setting_keys(list(loaded.manifest.env_from_settings or []))
    allowed_permissions = requested_skill_permissions(
        list(getattr(loaded.manifest, "permissions", []) or []),
        list(getattr(loaded.manifest, "subscribe_events", []) or []),
    )
    allowed_permission_map = {permission.lower(): permission for permission in allowed_permissions}
    requested_keys = [item.upper() for item in requested_raw if item.upper() in allowed]
    requested_permissions = [
        allowed_permission_map[item.lower()]
        for item in requested_raw
        if item.lower() in allowed_permission_map
    ]
    if not requested_raw or len(requested_keys) + len(requested_permissions) != len(requested_raw):
        return {
            "ok": False,
            "error": f"Grant items must be requested by the current manifest: keys={allowed}, permissions={allowed_permissions}",
        }
    message = (
        f"Grant skill {loaded.name!r} access to these settings keys / host permissions?\n\n"
        + "\n".join([*requested_keys, *requested_permissions])
        + "\n\nOnly grant keys and permissions to reviewed skills you trust."
    )
    if not confirm_fn("Confirm Skill Grant", message):
        return {"ok": False, "error": "Skill grant cancelled."}
    save_skill_grants(
        DATA_DIR,
        loaded.name,
        requested_keys,
        content_hash=loaded.content_hash,
        requested_keys=allowed,
        granted_permissions=requested_permissions,
        requested_permissions=allowed_permissions,
    )
    # Extension grants must reconcile inside server.py, not the immutable launcher process.
    extension_action = None
    extension_reason = None
    extension_load_error = None
    if loaded.manifest.is_extension():
        import json as _json
        import urllib.parse as _urlparse
        import urllib.request as _urlreq

        try:
            actual_port = _read_port_file() or AGENT_SERVER_PORT
            req = _urlreq.Request(
                f"http://127.0.0.1:{actual_port}/api/skills/"
                f"{_urlparse.quote(loaded.name)}/reconcile",
                method="POST",
                data=b"{}",
                headers={"Content-Type": "application/json"},
            )
            with _urlreq.urlopen(req, timeout=10) as resp:
                payload = _json.loads(resp.read().decode("utf-8") or "{}")
            extension_action = payload.get("extension_action")
            extension_reason = payload.get("extension_reason")
            extension_load_error = payload.get("load_error")
        except Exception as exc:
            log.warning(
                "Skill grant saved but server-side reconcile failed for %s: %s",
                loaded.name, exc, exc_info=True,
            )
            extension_reason = "reconcile_call_failed"
    return {
        "ok": True,
        "skill": loaded.name,
        "granted_keys": requested_keys,
        "granted_permissions": requested_permissions,
        "extension_action": extension_action,
        "extension_reason": extension_reason,
        "load_error": extension_load_error,
    }


def _display_ready(backend) -> tuple[bool, str]:
    """Linux: does a usable DISPLAY actually exist for the chosen backend?

    `initialize()` only proves the backend MODULE imports (the first real
    display access is `BrowserView.__init__` → `Gdk.Display.get_default()`
    inside webview.start()), so a box WITH gi bindings but WITHOUT a display
    passed the import probe and crashed. The session-env check runs BEFORE
    initialize() in _webview_ready; here the chosen backend is refined: GTK's
    display handle answers None on a dead display, never a raise. DISCLOSED
    RESIDUAL: Qt is judged on the env alone (probing Qt constructs a
    QGuiApplication, which ABORTS). Full rationale: ARCHITECTURE.
    """
    if str(getattr(backend, "__name__", "")).endswith(".gtk"):
        try:
            from gi.repository import Gdk
        except Exception as exc:  # gi vanished between import and probe
            return False, f"GTK backend without gi.repository.Gdk: {type(exc).__name__}: {exc}"
        if Gdk.Display.get_default() is None:
            return False, "GTK backend has no default display (DISPLAY/WAYLAND_DISPLAY is unusable)"
    return True, ""


def _webview_ready() -> tuple[bool, str]:
    """Probe whether pywebview can actually run a GUI window here.

    pywebview 5.x picks its backend LAZILY inside webview.start(), so
    `import webview` succeeds without GTK/QT and the process would only die
    later, at the first window site (#56). initialize() reuses pywebview's
    own backend selection as the SSOT (PYWEBVIEW_GUI, session detection);
    its second run inside webview.start() is cheap. On Linux the import
    answer is then refined by `_display_ready`; macOS/Windows keep the
    import-only answer — `_detect_headless` never calls this there.
    """
    if IS_LINUX and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        # BEFORE initialize(): even SELECTING a Qt backend constructs a
        # QGuiApplication, which can ABORT on a display-less box.
        return False, "no DISPLAY or WAYLAND_DISPLAY in the environment"
    try:
        import webview  # noqa: F401  (can itself fail on bare source checkouts)
        from webview.guilib import initialize

        backend = initialize()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if not IS_LINUX:
        return True, ""
    return _display_ready(backend)


def _detect_headless() -> None:
    """Set the module _headless flag on Linux when no GUI backend exists.

    Linux-only by design (#56): macOS/Windows return before the probe and keep
    their existing behavior untouched. Called once at the top of main().
    """
    global _headless
    if not IS_LINUX:
        return
    ok, reason = _webview_ready()
    if not ok:
        _headless = True
        log.warning(
            "No usable pywebview GUI backend (%s); continuing in browser mode.",
            reason,
        )


def _headless_signal_handler(signum, frame) -> None:
    """SIGINT/SIGTERM in headless mode: ONLY set the shutdown event —
    teardown runs on the main thread (signal-frame work risks re-entrancy)."""
    _shutdown_event.set()


def _open_browser_detached(url: str, outcome: Optional[list] = None) -> threading.Thread:
    """Open the default browser without ever blocking the caller.

    `webbrowser.open` waits for the child on a stdlib-resolved console
    browser (w3m/lynx or an unrecognized $BROWSER), which would stall the
    keep-alive loop for that browser's lifetime; the URL is already printed,
    so the open is best-effort and rides a daemon thread. Returns the thread
    so a short-lived caller (the already-running notice) can bound-join it
    before process exit would kill the daemon thread under the opener.

    An ``outcome`` list, when given, receives exactly one entry — True/False
    from ``webbrowser.open`` or the raised exception — so a bounded-join
    caller (the desktop bridge) can report failure honestly.

    DELIBERATE (owner-approved): the opened browser is the USER'S own
    application, intentionally outside process custody and launcher teardown —
    the Emergency-Stop invariant governs the AGENT'S tree, and killing the
    owner's browser would be hostile. A stdlib-resolved console browser
    (w3m/lynx) may outlive the launcher; the printed URL is primary.
    """
    def _open() -> None:
        try:
            result: object = bool(webbrowser.open(url))
        except Exception as exc:
            result = exc
            log.info("Could not open the default browser for %s", url, exc_info=True)
        if outcome is not None:
            outcome.append(result)

    thread = threading.Thread(target=_open, name="ouroboros-open-browser", daemon=True)
    thread.start()
    return thread


def _run_headless_main(url: str, port: int, lifecycle_thread: threading.Thread) -> None:
    """Browser-mode replacement for the main webview window (Linux headless).

    Prints the URL, best-effort opens the browser, keeps the process alive
    until shutdown. CRITICAL: the keep-alive loop also watches lifecycle-
    thread liveness — the crash fuse exits that thread WITHOUT setting
    _shutdown_event, so waiting on the event alone would leave a zombie
    launcher. Panic needs nothing here (it os._exit()s from the lifecycle
    thread). sys.exit (not os._exit) is correct without a webview.
    Never returns.
    """
    if not _shutdown_event.is_set():
        # A signal can land between main()'s startup check and this entry:
        # no URL announcement / browser launch mid-shutdown — straight to
        # teardown (the keep-alive loop returns immediately).
        log.info("Headless browser mode: serving the UI at %s", url)
        print(
            f"Ouroboros is running at {url}. No GUI backend (GTK/QT) — "
            "opening in your default browser. Press Ctrl-C to stop.",
            flush=True,
        )
        _open_browser_detached(url)
    # Handlers were installed in main() BEFORE the lifecycle thread started;
    # the browser open above rides a daemon thread because stdlib can resolve
    # a console browser (GenericBrowser) that blocks for its whole lifetime.
    while not _shutdown_event.wait(1.0):
        if not lifecycle_thread.is_alive():
            log.error(
                "Agent lifecycle thread exited without a shutdown request "
                "(crash fuse); shutting down."
            )
            break
    requested_shutdown = _shutdown_event.is_set()
    stop_agent()
    _kill_orphaned_children(port, reason="headless_shutdown" if requested_shutdown else "crash_fuse")
    # NO explicit release_pid_lock(): sys.exit runs the atexit-registered
    # release, and a second release would unconditionally unlink a lock a
    # NEWER launcher may own (explicit calls stay only on os._exit paths).
    sys.exit(0 if requested_shutdown else 1)


def main():
    if IS_WINDOWS:
        ok, reason = _prepare_windows_webview_runtime()
        if not ok:
            log.error("Windows UI runtime initialization failed: %s", reason)
            _show_windows_message(
                "Ouroboros — Startup Failed",
                "Windows UI runtime initialization failed.\n\n"
                f"{reason}\n\n"
                "Check launcher.log for details.",
            )
            return

    _detect_headless()

    if not _headless:
        import webview

    if not acquire_pid_lock():
        log.error("Another instance already running.")
        if _headless:
            # The lock loss usually races the FIRST launcher's bootstrap
            # (repeated Open clicks): the port file may be absent (unlinked
            # pre-start) or stale from an older run on another port. Poll
            # briefly for a healthy server, re-reading the file between
            # probes, so Open during bootstrap lands on the live UI; on
            # timeout fall back to the last-read port (best-effort notice;
            # the bound is soft — a probe straddling the deadline may run
            # its own urlopen timeout, a couple of seconds of overshoot).
            port = _read_port_file()
            deadline = time.time() + 10.0
            while time.time() < deadline and not _wait_for_server(port, timeout=1.0):
                time.sleep(0.5)
                port = _read_port_file()
            existing_url = f"http://127.0.0.1:{port}"
            print(
                f"Ouroboros is already running at {existing_url}",
                file=sys.stderr,
            )
            # Desktop-icon launches have no visible stderr, so the notice
            # alone reads as "Open does nothing". Surface the running
            # instance the same way a fresh headless boot would — open the
            # default browser at it. Bounded join: the open rides a daemon
            # thread (see _open_browser_detached), and returning immediately
            # would end the process under the opener before it fires.
            _open_browser_detached(existing_url).join(timeout=5.0)
            return
        webview.create_window(
            "Ouroboros",
            html="<html><body style='background:#1a1a2e;color:white;font-family:system-ui;display:flex;align-items:center;justify-content:center;height:100vh;margin:0'>"
            "<div style='text-align:center'><h2>Ouroboros is already running</h2><p>Only one instance can run at a time.</p></div></body></html>",
            width=420,
            height=200,
        )
        webview.start()
        return

    import atexit

    atexit.register(release_pid_lock)

    if not check_git():
        log.warning("Git not found.")
        if _headless:
            # No headless sudo flows: print the platform hint and stop.
            log.error("Git is required; cannot run the GUI install helper in browser mode.")
            print("Git is required to run Ouroboros.", file=sys.stderr)
            print(git_install_hint(), file=sys.stderr)
            sys.exit(1)
        _hint = git_install_hint()
        _install_status = (
            "Installing... A system dialog may appear."
            if IS_MACOS
            else "Installing... Please wait."
        )

        def _git_page(window):
            window.evaluate_js(
                """
                document.getElementById('install-btn').onclick = function() {
                    document.getElementById('status').textContent = '__INSTALL_STATUS__';
                    window.pywebview.api.install_git();
                };
                """.replace("__INSTALL_STATUS__", _install_status)
            )

        class GitApi:
            def install_git(self):
                if IS_MACOS:
                    subprocess.Popen(["xcode-select", "--install"])
                elif IS_WINDOWS:
                    _hidden_popen(
                        ["winget", "install", "Git.Git", "--source", "winget", "--accept-source-agreements"]
                    )
                else:
                    for cmd in (
                        ["sudo", "apt", "install", "-y", "git"],
                        ["sudo", "dnf", "install", "-y", "git"],
                    ):
                        try:
                            _hidden_popen(cmd)
                            break
                        except FileNotFoundError:
                            continue
                for _ in range(300):
                    time.sleep(3)
                    if shutil.which("git"):
                        return "installed"
                return "timeout"

        git_window = webview.create_window(
            "Ouroboros — Setup Required",
            html=(
                """<html><body style="background:#1a1a2e;color:white;font-family:system-ui;display:flex;align-items:center;justify-content:center;height:100vh;margin:0">
            <div style="text-align:center">
                <h2>Git is required</h2>
                <p>Ouroboros needs Git to manage its local repository.</p>
                <button id="install-btn" style="padding:10px 24px;border-radius:8px;border:none;background:#0ea5e9;color:white;cursor:pointer;font-size:14px">
                    Install Git (Xcode CLI Tools)
                </button>
                <p id="status" style="color:#fbbf24;margin-top:12px"></p>
            </div></body></html>"""
                if IS_MACOS
                else f"""<html><body style="background:#1a1a2e;color:white;font-family:system-ui;display:flex;align-items:center;justify-content:center;height:100vh;margin:0">
            <div style="text-align:center">
                <h2>Git is required</h2>
                <p>Ouroboros needs Git to manage its local repository.</p>
                <p style="color:#94a3b8;font-size:13px;margin-top:8px">{_hint}</p>
                <button id="install-btn" style="padding:10px 24px;border-radius:8px;border:none;background:#0ea5e9;color:white;cursor:pointer;font-size:14px;margin-top:12px">
                    Install Git
                </button>
                <p id="status" style="color:#fbbf24;margin-top:12px"></p>
            </div></body></html>"""
            ),
            js_api=GitApi(),
            width=520,
            height=300,
        )
        webview.start(func=_git_page, args=[git_window])
        if not check_git():
            sys.exit(1)

    if not bootstrap_repo():
        # Disclosed, not fatal: an already-provisioned install with satisfied
        # requirements still runs, and a genuinely broken checkout is stopped
        # by the startup health check / crash fuse with this line naming why.
        log.error(
            "Continuing startup after a failed dependency install; the agent may "
            "be missing packages (see the pip output above)."
        )

    # D-8: decide here, PRESENT after the gateway is healthy. Everything above
    # (single-instance lock, Git, managed-repo bootstrap/seed validation) is a
    # precondition of the server itself and deliberately still precedes it.
    onboarding_settings, onboarding_required = _prepare_first_run_settings()

    global _webview_window
    port = AGENT_SERVER_PORT

    # Clear any stale server process or ports before starting the new agent
    _cleanup_recorded_server_process("preflight")
    _reap_same_install_strays("preflight")
    _kill_stale_runtime_ports(port)
    try:
        PORT_FILE.unlink(missing_ok=True)
    except OSError:
        pass

    # Headless: handlers BEFORE the lifecycle thread spawns server.py — a
    # SIGTERM in the ~60s readiness window must reach our teardown.
    _abort = None
    if _headless:
        install_shutdown_signal_handlers(_headless_signal_handler)
        _abort = _shutdown_event

    lifecycle_thread = threading.Thread(target=agent_lifecycle_loop, args=(port,), daemon=True)
    lifecycle_thread.start()

    server_ready, actual_port = _await_server_ready(port, _abort)

    if server_ready and onboarding_required and not _shutdown_event.is_set():
        # The gateway is live and, with no provider configured, runs WITHOUT a
        # supervisor — the supported state that lets the wizard reach /api/*.
        onboarding = _present_first_run_onboarding(
            onboarding_settings, actual_port, headless=_headless
        )
        if not onboarding["saved"]:
            log.info(
                "Setup was closed without saving. Launching anyway "
                "(the blocking onboarding overlay and Settings remain available)."
            )
        if onboarding["restart_required"] and not _shutdown_event.is_set():
            # The completion changed something this process pinned at boot (its
            # runtime-mode baseline). The launcher owns the process, so it
            # recycles it instead of leaving the owner with a restart nag.
            _request_agent_restart()
            server_ready, actual_port = _await_server_ready(port, _abort)

    if _headless and _shutdown_event.is_set():
        # Shutdown during startup: same teardown the keep-alive loop performs
        # (an aborted wait is a requested shutdown, not a startup failure).
        log.info("Shutdown requested during headless startup; tearing down.")
        stop_agent()
        _kill_orphaned_children(actual_port, reason="startup_abort")
        # atexit owns release_pid_lock on sys.exit (a second release would
        # unlink a newer launcher's lock).
        sys.exit(0)

    if not server_ready:
        log.error("Agent failed to become healthy on port %d; aborting UI startup.", actual_port)
        _shutdown_event.set()
        stop_agent()
        lifecycle_thread.join(timeout=5)
        if _headless:
            _kill_orphaned_children(actual_port, reason="startup_failure")
            print(
                "Ouroboros failed to start: the local agent server did not "
                "become ready.\n"
                f"See {_log_dir / 'launcher.log'} and "
                f"{_log_dir / 'agent_stdout.log'} for details.",
                file=sys.stderr,
            )
            sys.exit(1)
        webview.create_window(
            "Ouroboros — Startup Failed",
            html=(
                "<html><body style='background:#1a1a2e;color:white;font-family:system-ui;"
                "display:flex;align-items:center;justify-content:center;height:100vh;margin:0'>"
                "<div style='text-align:center;max-width:460px;padding:24px'>"
                "<h2>Ouroboros failed to start</h2>"
                "<p>The local agent server did not become ready.</p>"
                "<p style='color:#94a3b8;font-size:13px;margin-top:10px'>"
                "Check launcher.log and agent_stdout.log in the Ouroboros data directory "
                "for details.</p>"
                "</div></body></html>"
            ),
            width=520,
            height=260,
        )
        webview.start()
        return

    def _resolve_bridge_file_url(raw_url: str) -> str:
        """Validate a loopback file-bridge URL, returning the resolved full URL.

        Shared SSOT for both the download-to-Downloads and open-in-default-app
        bridge methods so the loopback guard cannot drift between them.
        """
        full_url = urllib.parse.urljoin(f"http://127.0.0.1:{actual_port}", str(raw_url or ""))
        parsed = urllib.parse.urlparse(full_url)
        if parsed.scheme != "http":
            raise ValueError("file URL must be http://")
        if parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("desktop file access is limited to the local Ouroboros server")
        if parsed.port != actual_port:
            raise ValueError("file URL port must match the local Ouroboros server")
        if parsed.path != "/api/files/download" and not parsed.path.startswith(("/api/extensions/", "/api/tasks/")):
            raise ValueError("file URL path must be /api/files/download, /api/extensions/<skill>/... or /api/tasks/...")
        return full_url

    def _unique_bridge_target(directory: pathlib.Path, filename: str) -> pathlib.Path:
        safe_name = pathlib.Path(str(filename or "download")).name or "download"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / safe_name
        stem, suffix = target.stem, target.suffix
        counter = 1
        while target.exists():
            target = directory / f"{stem}-{counter}{suffix}"
            counter += 1
        return target

    def _fetch_bridge_url_to(full_url: str, target: pathlib.Path) -> None:
        with urllib.request.urlopen(full_url, timeout=60) as resp, target.open("wb") as fh:  # noqa: S310 - localhost validated above
            shutil.copyfileobj(resp, fh)

    class MainApi:
        @staticmethod
        def _native_confirm(title: str, message: str) -> bool:
            return bool(_webview_window and _webview_window.create_confirmation_dialog(title, message))

        def request_runtime_mode_change(self, mode: str) -> dict:
            try:
                return _request_runtime_mode_change(mode, self._native_confirm)
            except Exception as exc:
                log.warning("Runtime mode native confirmation failed: %s", exc, exc_info=True)
                return {"ok": False, "error": f"Native confirmation failed: {exc}"}

        def request_auto_grant_reviewed_skills_change(self, enabled: bool) -> dict:
            try:
                return _request_auto_grant_reviewed_skills_change(bool(enabled), self._native_confirm)
            except Exception as exc:
                log.warning("Reviewed-skill auto-grant confirmation failed: %s", exc, exc_info=True)
                return {"ok": False, "error": f"Native confirmation failed: {exc}"}

        def request_skill_key_grant(self, skill: str, keys: list) -> dict:
            try:
                return _request_skill_key_grant(skill, keys, self._native_confirm)
            except Exception as exc:
                log.warning("Skill grant native confirmation failed: %s", exc, exc_info=True)
                return {"ok": False, "error": f"Native confirmation failed: {exc}"}

        def download_file_to_downloads(self, url: str, filename: str, open_external: bool = False) -> dict:
            try:
                full_url = _resolve_bridge_file_url(url)
                target = _unique_bridge_target(pathlib.Path.home() / "Downloads", filename)
                _fetch_bridge_url_to(full_url, target)
                if open_external:
                    open_path_external(target)
                return {"ok": True, "path": str(target)}
            except Exception as exc:
                log.warning("Desktop file download failed: %s", exc, exc_info=True)
                return {"ok": False, "error": str(exc)}

        def open_external_url(self, url: str) -> dict:
            try:
                raw = str(url or "")
                if not raw.lower().startswith(("http://", "https://", "mailto:")):
                    return {"ok": False, "error": "Only absolute http://, https:// or mailto: links can be opened."}
                outcome: list = []
                # Bounded join: settled failure reported honestly; still-running stays detached.
                _open_browser_detached(raw, outcome).join(timeout=3.0)
                if outcome and outcome[0] is not True:
                    return {"ok": False, "error": f"The default browser could not be opened: {outcome[0] or 'no handler found'}"}
                return {"ok": True}
            except Exception as exc:
                log.warning("Desktop external-URL open failed: %s", exc, exc_info=True)
                return {"ok": False, "error": str(exc)}

        def save_bytes_to_downloads(self, filename: str, b64: str) -> dict:
            try:
                target = _unique_bridge_target(pathlib.Path.home() / "Downloads", filename)
                target.write_bytes(base64.b64decode(str(b64 or ""), validate=True))
                return {"ok": True, "path": str(target)}
            except Exception as exc:
                log.warning("Desktop save-to-Downloads failed: %s", exc, exc_info=True)
                return {"ok": False, "error": str(exc)}

        def open_file_with_default_app(self, url: str, filename: str) -> dict:
            """Open a delivered file in the OS default app (external window).

            Fetches the loopback file into a private temp dir (NOT ~/Downloads)
            and hands it to the platform default handler. This never navigates
            the in-app WKWebView, which was the original fullscreen-lockup bug.
            """
            try:
                full_url = _resolve_bridge_file_url(url)
                # Per-open private dir: mkdtemp atomically creates a fresh 0700
                # directory, so a pre-placed symlink/dir at a shared temp path
                # cannot redirect the write (hardens over a fixed shared root).
                open_root = pathlib.Path(tempfile.mkdtemp(prefix="ouroboros-open-"))
                target = _unique_bridge_target(open_root, filename)
                _fetch_bridge_url_to(full_url, target)
                open_path_external(target)
                return {"ok": True, "path": str(target)}
            except Exception as exc:
                log.warning("Desktop open-in-default-app failed: %s", exc, exc_info=True)
                return {"ok": False, "error": str(exc)}

    # Prune stale externally-opened temp copies from previous sessions (privacy + disk).
    for _stale_open in pathlib.Path(tempfile.gettempdir()).glob("ouroboros-open-*"):
        shutil.rmtree(_stale_open, ignore_errors=True)

    url = f"http://127.0.0.1:{actual_port}"

    if _headless:
        # Never returns: keep-alive loop + teardown + sys.exit inside.
        _run_headless_main(url, actual_port, lifecycle_thread)

    window = webview.create_window(
        f"Ouroboros v{APP_VERSION}",
        url=url,
        js_api=MainApi(),
        width=1100,
        height=750,
        min_size=(800, 500),
        background_color="#0d0b0f",
        text_select=True,
    )

    def _on_closing() -> None:
        log.info("Window closing — graceful shutdown.")
        _shutdown_event.set()
        stop_agent()
        _kill_orphaned_children(port)
        release_pid_lock()
        os._exit(0)

    window.events.closing += _on_closing
    _webview_window = window

    webview.start(debug=False)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()

    if sys.platform == "darwin":
        try:
            shell_path = subprocess.check_output(
                ["/bin/bash", "-l", "-c", "echo $PATH"],
                text=True,
                timeout=5,
            ).strip()
            if shell_path:
                os.environ["PATH"] = shell_path
        except Exception:
            pass

    main()
