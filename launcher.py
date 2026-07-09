"""Immutable desktop launcher: bootstrap repo, manage server.py, and host UI."""

from __future__ import annotations

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
    python_bytecode_env,
    sync_existing_repo_from_bundle as _sync_existing_repo_from_bundle_impl,
    verify_claude_runtime as _verify_claude_runtime,
)
from ouroboros.onboarding_wizard import build_onboarding_html, prepare_onboarding_settings
from ouroboros.platform_layer import (
    IS_MACOS,
    IS_WINDOWS,
    assign_pid_to_job,
    close_job,
    create_kill_on_close_job,
    current_process_group_id,
    embedded_python_candidates,
    force_kill_pid,
    git_install_hint,
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
from ouroboros.server_runtime import apply_runtime_provider_defaults, has_startup_ready_provider

MAX_CRASH_RESTARTS = 5
CRASH_WINDOW_SEC = 120
_CREATE_SUSPENDED = getattr(subprocess, "CREATE_SUSPENDED", 0x4) if IS_WINDOWS else 0
_CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if IS_WINDOWS else 0

# WA6: globally suppress bytecode writes for the launcher process itself and any
# naive os.environ.copy() child it spawns. A signed+notarized macOS .app must not
# write __pycache__/*.pyc into its own bundle at runtime — that breaks the codesign
# seal and triggers AppTranslocation. Uses the same data_dir/state/pycache
# convention as launcher_bootstrap.python_bytecode_env so caches land outside the
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

_windows_dll_dir_handles: list = []


def _show_windows_message(title: str, message: str) -> None:
    if not IS_WINDOWS:
        return
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)
    except Exception:
        pass


def _prepare_windows_webview_runtime() -> tuple[bool, str]:
    """Prepare pythonnet/pywebview runtime before importing webview on Windows."""
    if not IS_WINDOWS:
        return True, ""

    base_dir = pathlib.Path(getattr(sys, "_MEIPASS", pathlib.Path(sys.executable).parent))
    exe_dir = pathlib.Path(sys.executable).parent
    runtime_dir = base_dir / "pythonnet" / "runtime"
    webview_lib_dir = base_dir / "webview" / "lib"
    py_dll_name = f"python{sys.version_info[0]}{sys.version_info[1]}.dll"

    def _unblock_file(path: pathlib.Path) -> None:
        try:
            os.remove(f"{path}:Zone.Identifier")
        except OSError:
            pass

    def _unblock_tree(root: pathlib.Path) -> None:
        if not root.is_dir():
            return
        for child in root.rglob("*"):
            if child.is_file() and child.suffix.lower() in {".dll", ".exe", ".pyd"}:
                _unblock_file(child)

    py_dll_candidates = [
        base_dir / py_dll_name,
        exe_dir / py_dll_name,
    ]
    for root, _dirs, files in os.walk(base_dir):
        if py_dll_name in files:
            py_dll_candidates.append(pathlib.Path(root) / py_dll_name)
            if len(py_dll_candidates) >= 6:
                break

    py_dll_path = next((path for path in py_dll_candidates if path.is_file()), None)
    runtime_dll_path = runtime_dir / "Python.Runtime.dll"
    if not runtime_dll_path.is_file():
        for root, _dirs, files in os.walk(base_dir):
            if "Python.Runtime.dll" in files:
                runtime_dll_path = pathlib.Path(root) / "Python.Runtime.dll"
                break

    if py_dll_path is None:
        return False, f"Bundled {py_dll_name} was not found."
    if not runtime_dll_path.is_file():
        return False, "Bundled Python.Runtime.dll was not found."

    _unblock_file(py_dll_path)
    _unblock_file(runtime_dll_path)
    _unblock_tree(runtime_dll_path.parent)
    _unblock_tree(webview_lib_dir)

    os.environ["PYTHONNET_RUNTIME"] = "netfx"
    os.environ["PYTHONNET_PYDLL"] = str(py_dll_path)

    search_dirs = []
    for candidate in (
        base_dir,
        exe_dir,
        runtime_dir,
        runtime_dll_path.parent,
        py_dll_path.parent,
        webview_lib_dir,
    ):
        candidate_str = str(candidate)
        if candidate.is_dir() and candidate_str not in search_dirs:
            search_dirs.append(candidate_str)

    current_path_parts = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
    os.environ["PATH"] = os.pathsep.join(search_dirs + [part for part in current_path_parts if part and part not in search_dirs])

    if hasattr(os, "add_dll_directory"):
        global _windows_dll_dir_handles
        for candidate in search_dirs:
            try:
                _windows_dll_dir_handles.append(os.add_dll_directory(candidate))
            except (FileNotFoundError, OSError):
                pass

    try:
        from clr_loader import get_netfx
        from pythonnet import set_runtime

        set_runtime(get_netfx())
    except Exception as exc:
        return False, f"Windows .NET runtime init failed: {exc}"

    return True, ""

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


def bootstrap_repo() -> None:
    _bootstrap_repo(_bootstrap_context())


def _sync_existing_repo_from_bundle() -> None:
    _sync_existing_repo_from_bundle_impl(_bootstrap_context())


def _install_deps() -> None:
    _install_deps_impl(_bootstrap_context())

_agent_proc: Optional[subprocess.Popen] = None
_agent_job: Optional[object] = None
_agent_lock = threading.Lock()
_shutdown_event = threading.Event()
_webview_window = None


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
    env = python_bytecode_env(DATA_DIR, os.environ.copy())
    env["PYTHONPATH"] = str(REPO_DIR)
    saved_host = str(settings.get("OUROBOROS_SERVER_HOST") or "").strip()
    if saved_host:
        env["OUROBOROS_SERVER_HOST"] = saved_host
    env["OUROBOROS_SERVER_PORT"] = str(port)
    env["OUROBOROS_DATA_DIR"] = str(DATA_DIR)
    env["OUROBOROS_REPO_DIR"] = str(REPO_DIR)
    env["OUROBOROS_APP_VERSION"] = str(APP_VERSION)
    env["OUROBOROS_MANAGED_BY_LAUNCHER"] = "1"

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
        log_path = DATA_DIR / "logs" / "agent_stdout.log"
        try:
            with open(log_path, "a", encoding="utf-8") as handle:
                for line in iter(proc.stdout.readline, b""):
                    decoded = line.decode("utf-8", errors="replace")
                    handle.write(decoded)
                    handle.flush()
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
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"],
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


def _wait_for_server(port: int, timeout: float = 30.0) -> bool:
    """Wait for the agent HTTP server to respond."""
    import urllib.request

    url = f"http://127.0.0.1:{port}/api/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
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


def _kill_orphaned_children(port: int) -> None:
    """Final safety net: kill processes still on runtime ports.

    Module-level so both the window-close handler (_on_closing, main thread) and
    the panic-stop branch (agent_lifecycle_loop, supervisor thread) tear down the
    exact same way.
    """
    _cleanup_recorded_server_process("window_close")
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

    _cleanup_recorded_server_process("startup")
    _kill_stale_runtime_ports(port)

    while not _shutdown_event.is_set():
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
            _kill_orphaned_children(port)
            release_pid_lock()
            os._exit(0)

        time.sleep(2)

        if exit_code == RESTART_EXIT_CODE:
            log.info("Agent requested restart (exit code 42). Restarting...")
            _sync_existing_repo_from_bundle()
            _install_deps()
            _kill_stale_runtime_ports(port)
            continue

        now = time.time()
        crash_times.append(now)
        crash_times[:] = [stamp for stamp in crash_times if (now - stamp) < CRASH_WINDOW_SEC]
        if len(crash_times) >= MAX_CRASH_RESTARTS:
            log.error("Agent crashed %d times in %ds. Stopping.", MAX_CRASH_RESTARTS, CRASH_WINDOW_SEC)
            break

        log.info("Agent crashed. Restarting in 3s...")
        _kill_stale_runtime_ports(port)
        time.sleep(3)

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


def _claude_code_status_payload(settings: dict | None = None) -> dict:
    current_settings = settings or _load_settings()
    _apply_settings_to_env(current_settings)

    from ouroboros.platform_layer import resolve_claude_runtime

    rt = resolve_claude_runtime()
    stderr_tail = ""
    try:
        from ouroboros.gateways.claude_code import get_last_stderr as _get_last_stderr

        stderr_tail = _get_last_stderr(max_chars=2000)
    except Exception:
        pass

    message_map = {
        "ready": f"Claude runtime ready (SDK {rt.sdk_version}, CLI {rt.cli_version})",
        "no_api_key": (
            f"Claude runtime available (SDK {rt.sdk_version}) but ANTHROPIC_API_KEY is not set. Add it in Settings."
        ),
        "error": f"Claude runtime error: {rt.error}",
        "degraded": (
            f"Claude runtime degraded (SDK {rt.sdk_version}, CLI {'found' if rt.cli_path else 'missing'}). Try Repair."
        ),
        "missing": "Claude runtime not available. Use Repair in Settings or reinstall the app.",
    }

    return {
        "status": rt.status_label(),
        "installed": bool(rt.sdk_version),
        "ready": rt.ready,
        "busy": False,
        "version": rt.sdk_version,
        "cli_version": rt.cli_version,
        "cli_path": rt.cli_path,
        "interpreter_path": rt.interpreter_path,
        "app_managed": rt.app_managed,
        "legacy_detected": rt.legacy_detected,
        "legacy_sdk_version": rt.legacy_sdk_version,
        "api_key_set": rt.api_key_set,
        "message": message_map.get(rt.status_label(), f"Claude runtime: {rt.status_label()}"),
        "error": rt.error,
        "stderr_tail": stderr_tail,
    }


def _run_first_run_wizard() -> bool:
    """Show setup wizard if no runtime provider or local model is configured."""
    settings, provider_defaults_changed, _provider_default_keys = apply_runtime_provider_defaults(_load_settings())
    if provider_defaults_changed:
        _save_settings(settings)
    _apply_settings_to_env(settings)
    if has_startup_ready_provider(settings):
        return True

    import webview

    _wizard_done = {"ok": False}

    class WizardApi:
        def save_wizard(self, data: dict) -> str:
            prepared_settings, error = prepare_onboarding_settings(data, settings)
            if error:
                return error
            settings.update(prepared_settings)
            settings.update(apply_runtime_provider_defaults(settings)[0])
            try:
                _save_settings(settings)
                _apply_settings_to_env(settings)
                _wizard_done["ok"] = True
                for window in webview.windows:
                    window.destroy()
                return "ok"
            except Exception as exc:
                return f"Failed to save: {exc}"

        def fetch_compatible_models(self, data: dict) -> dict:
            import urllib.request
            import urllib.error
            base_url = str(data.get("baseUrl", "") or "").rstrip("/")
            api_key = str(data.get("apiKey", "") or "").strip()
            if not base_url:
                return {"error": "baseUrl is required"}
            try:
                req = urllib.request.Request(
                    f"{base_url}/models",
                    headers=({"Authorization": f"Bearer {api_key}"} if api_key else {}),
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    raw = json.loads(resp.read())
                raw_models = raw.get("data") or []
                models = sorted({
                    str(m.get("id", "") or "").strip()
                    for m in raw_models if isinstance(m, dict) and m.get("id")
                })
                return {"models": models}
            except Exception as exc:
                return {"error": str(exc)}

        def claude_code_status(self) -> dict:
            return _claude_code_status_payload(settings)

        def install_claude_code(self) -> dict:
            _apply_settings_to_env(settings)
            repaired = _verify_claude_runtime(_bootstrap_context())
            payload = _claude_code_status_payload(settings)
            payload["repaired"] = repaired
            if not repaired:
                payload["status"] = "error"
                payload["ready"] = False
                payload["busy"] = False
                payload["message"] = "Claude runtime repair failed."
                if not payload.get("error"):
                    payload["error"] = "Failed to install/update claude-agent-sdk in the embedded runtime."
            return payload

    webview.create_window(
        "Ouroboros — Setup",
        html=build_onboarding_html(settings, host_mode="desktop"),
        js_api=WizardApi(),
        width=980,
        height=780,
        min_size=(840, 640),
    )
    webview.start()
    return _wizard_done["ok"]

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

    import webview

    if not acquire_pid_lock():
        log.error("Another instance already running.")
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

    bootstrap_repo()

    if not _run_first_run_wizard():
        log.info("Wizard was closed without saving. Launching anyway (Settings page available).")

    global _webview_window
    port = AGENT_SERVER_PORT

    # Clear any stale server process or ports before starting the new agent
    _cleanup_recorded_server_process("preflight")
    _kill_stale_runtime_ports(port)
    try:
        PORT_FILE.unlink(missing_ok=True)
    except OSError:
        pass

    lifecycle_thread = threading.Thread(target=agent_lifecycle_loop, args=(port,), daemon=True)
    lifecycle_thread.start()

    server_ready = _wait_for_server(port, timeout=15)
    actual_port = _read_port_file()
    if actual_port != port:
        server_ready = _wait_for_server(actual_port, timeout=45)
    else:
        server_ready = server_ready or _wait_for_server(port, timeout=45)

    if not server_ready:
        log.error("Agent failed to become healthy on port %d; aborting UI startup.", actual_port)
        _shutdown_event.set()
        stop_agent()
        lifecycle_thread.join(timeout=5)
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
        import urllib.parse

        full_url = urllib.parse.urljoin(f"http://127.0.0.1:{actual_port}", str(raw_url or ""))
        parsed = urllib.parse.urlparse(full_url)
        if parsed.scheme != "http":
            raise ValueError("file URL must be http://")
        if parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("desktop file access is limited to the local Ouroboros server")
        if parsed.port != actual_port:
            raise ValueError("file URL port must match the local Ouroboros server")
        if parsed.path != "/api/files/download" and not parsed.path.startswith("/api/extensions/"):
            raise ValueError("file URL path must be /api/files/download or /api/extensions/<skill>/...")
        return full_url

    def _unique_bridge_target(directory: pathlib.Path, filename: str) -> pathlib.Path:
        safe_name = pathlib.Path(str(filename or "download")).name or "download"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / safe_name
        stem = target.stem
        suffix = target.suffix
        counter = 1
        while target.exists():
            target = directory / f"{stem}-{counter}{suffix}"
            counter += 1
        return target

    def _fetch_bridge_url_to(full_url: str, target: pathlib.Path) -> None:
        import urllib.request

        with urllib.request.urlopen(full_url, timeout=60) as resp:  # noqa: S310 - localhost validated above
            with target.open("wb") as fh:
                shutil.copyfileobj(resp, fh)

    class MainApi:
        def request_runtime_mode_change(self, mode: str) -> dict:
            try:
                return _request_runtime_mode_change(
                    mode,
                    lambda title, message: bool(
                        _webview_window and _webview_window.create_confirmation_dialog(title, message)
                    ),
                )
            except Exception as exc:
                log.warning("Runtime mode native confirmation failed: %s", exc, exc_info=True)
                return {"ok": False, "error": f"Native confirmation failed: {exc}"}

        def request_auto_grant_reviewed_skills_change(self, enabled: bool) -> dict:
            try:
                return _request_auto_grant_reviewed_skills_change(
                    bool(enabled),
                    lambda title, message: bool(
                        _webview_window and _webview_window.create_confirmation_dialog(title, message)
                    ),
                )
            except Exception as exc:
                log.warning("Reviewed-skill auto-grant confirmation failed: %s", exc, exc_info=True)
                return {"ok": False, "error": f"Native confirmation failed: {exc}"}

        def request_skill_key_grant(self, skill: str, keys: list) -> dict:
            try:
                return _request_skill_key_grant(
                    skill,
                    keys,
                    lambda title, message: bool(
                        _webview_window and _webview_window.create_confirmation_dialog(title, message)
                    ),
                )
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
