"""Bootstrap the managed source, supervise server.py, and optionally host its UI."""

from __future__ import annotations

import base64
import html
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
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from contextlib import suppress
from logging.handlers import RotatingFileHandler
from typing import Optional


def _rehome_legacy_handoff() -> None:
    """Before package adoption can wait: repair a recognized legacy GUI handoff once."""
    marked, args = os.environ.pop("OUROBOROS_LAUNCHER_HANDOFF", ""), sys.argv[1:]
    if sys.platform == "win32" and not getattr(sys, "frozen", False) and not marked and "--no-ui" not in args \
            and any(arg.split("=", 1)[0] == "--seed-bundle" for arg in args):
        # This existing Windows leaf is stdlib-only; importing its package would run adoption first.
        import runpy

        runpy.run_path(str(pathlib.Path(__file__).parent / "ouroboros" / "launcher_windows_runtime.py"))[
            "rehome_legacy_handoff"](args)


if __name__ == "__main__":
    _rehome_legacy_handoff()

# Seal-safe imports: the current interpreter needs the flag before project imports;
# PYTHONDONTWRITEBYTECODE alone only affects later child interpreters. Bundle writes
# would invalidate a macOS signature and trigger AppTranslocation.
sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from ouroboros.config import (
    AGENT_SERVER_PORT,
    DATA_DIR,
    LAUNCHER_STOP_GRACE_SEC,
    PANIC_EXIT_CODE,
    PID_FILE, PORT_FILE,
    REPO_DIR,
    RESTART_EXIT_CODE,
    SETTINGS_PATH,
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
    check_git as _check_git, launcher_sources_changed, remember_loaded_checkout,
    install_deps as _install_deps_impl,
    embedded_python_env, update_external_host,
    exec_source_launcher, find_embedded_python, macos_source_entry, parse_launch_options, shell_runtime_facts,
    automatic_launch_allowed,
    sync_existing_repo_from_bundle as _sync_existing_repo_from_bundle_impl,
)
from ouroboros.desktop_media import attach_media_permissions, display_ready as _display_ready, host_environment, route_gtk_browser
from ouroboros.runtime_venv import PREPARATION_DEPS_TIMEOUT_SEC, RuntimeVenvRefused, mark_ready, prepare_runtime_venv
from ouroboros.launcher_onboarding import (
    prepare_first_run_settings as _prepare_first_run_settings,
    present_first_run_onboarding as _present_first_run_onboarding,
    setup_recovery_menu,
)
from ouroboros.launcher_background import (Background, DesktopApi, activate_running_instance, background_env, other_copy_line,
                                           request_tray_cleanup, restore_tray_after_handoff, stop_tray_before_exit)
from ouroboros.launcher_server_reaper import (
    reap_same_install_strays as _reap_same_install_strays_impl,
)
from ouroboros.launcher_readiness import wait_for_server as _wait_for_server
from ouroboros.launcher_window_chrome import paint_room
from ouroboros.standalone_page import themed_page
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
    force_kill_pid,
    git_install_hint,
    install_shutdown_signal_handlers,
    kill_pid_tree,
    kill_process_group_id,
    kill_process_tree,
    merge_hidden_kwargs,
    open_path_external,
    pid_provably_gone,
    process_command,
    process_group_id,
    resume_process,
    subprocess_new_group_kwargs,
    terminate_job,
    terminate_process_group_id,
)
from ouroboros.unindexed import bytecode_prefix
from ouroboros.utils import atomic_write_json, utc_now_iso

MAX_CRASH_RESTARTS = 5
CRASH_WINDOW_SEC = 120
# Entry protocol 1: --seed-bundle selects plain Python for the body and its
# installs; --runtime-venv is retained separately while the shell keeps app identity.
MACOS_SOURCE_ENTRY_PROTOCOL = 1
# One bounded visible retry when dependency installation fails.
_DEPS_RETRY_DELAY_SEC = 5

# WA6: globally suppress bytecode writes for the launcher process itself and any
# naive os.environ.copy() child it spawns. A signed+notarized macOS .app must not
# write __pycache__/*.pyc into its own bundle at runtime — that breaks the codesign
# seal and triggers AppTranslocation. Caches land in the data dir's unindexed prefix
# (as in launcher_bootstrap.embedded_python_env), renamed from an older launcher's
# before any child starts; any other explicit caller prefix is kept.
os.environ["PYTHONPYCACHEPREFIX"] = bytecode_prefix(DATA_DIR, os.environ.get("PYTHONPYCACHEPREFIX", ""))

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
# Shared SSOTs: redaction filter, uncaught exceptions into these handlers; quiet request-URL lines (credentials).
try:
    from ouroboros.observability import SecretRedactingLogFilter as _RedactFilter
    for _handler in _handlers:
        _handler.addFilter(_RedactFilter())
    from ouroboros.process_logging import install_exception_hooks
    install_exception_hooks()
except Exception:
    pass  # defensive: a broken observability import must not kill the launcher
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("launcher")


APP_VERSION = read_version()


def _server_process_record_path() -> pathlib.Path:
    return pathlib.Path(DATA_DIR) / "state" / "server_process.json"


def _hidden_run(command, **kwargs):
    """Hidden bootstrap commands keep their own Job/lifeline until cleanup."""
    from ouroboros.launcher_bootstrap import run_bootstrap_command

    if IS_LINUX:
        kwargs.setdefault("env", host_environment())
    return run_bootstrap_command(command, _shutdown_event, **merge_hidden_kwargs(kwargs))


def _hidden_popen(command, *, bootstrap=False, **kwargs):
    """subprocess.Popen() with platform-appropriate hidden-window flags."""
    if IS_LINUX:
        kwargs.setdefault("env", host_environment())
    if bootstrap:
        from ouroboros.launcher_bootstrap import spawn_bootstrap_command

        return spawn_bootstrap_command(command, _shutdown_event, interactive=True, **merge_hidden_kwargs(kwargs))["proc"]
    return subprocess.Popen(command, **merge_hidden_kwargs(kwargs))


def _bundle_dir() -> pathlib.Path:
    if getattr(sys, "frozen", False):
        return pathlib.Path(sys._MEIPASS)
    return _external_seed_bundle or pathlib.Path(__file__).parent


# The layout of a window shown before the server, on the interface's tokens (standalone_page).
_WINDOW_CSS = """
body { margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
  background: var(--bg-primary, Canvas); color: var(--text-primary, CanvasText); font-family: var(--font-sans, system-ui); }
.window { box-sizing: border-box; max-width: 460px; padding: 24px; text-align: center; }
.window h2 { margin: 0 0 12px; font-size: var(--type-page, 24px); font-weight: 600; line-height: var(--line-title); }
.window p { margin: 0; font-size: var(--type-body); line-height: var(--line-body); overflow-wrap: anywhere; }
.window .muted { margin-top: 10px; font-size: var(--type-meta); line-height: var(--line-meta); }
.window .btn { margin: 16px auto 0; }
.window .ui-status { margin-top: 12px; }
"""


def _window_page(title: str, body: str) -> str:
    """A window shown before the server: no saved choice can reach it, so it follows the device (macOS),
    in the web/ shipped with this launcher's own code (the app's when frozen, the checkout's otherwise)."""
    code_dir = pathlib.Path(sys._MEIPASS) if getattr(sys, "frozen", False) else pathlib.Path(__file__).parent
    return themed_page(code_dir / "web", title, f'<div class="window">{body}</div>', _WINDOW_CSS)


def _already_running_window(line: str) -> None:
    """A second launch's pre-server notice (``webview`` is imported by then)."""
    import webview
    paint_room(webview.create_window("Ouroboros", html=_window_page(
        "Ouroboros", f"<h2>Ouroboros is already running</h2><p>{html.escape(line)}</p>"), width=440, height=200))
    webview.start(private_mode=False)


def _exec_source_launcher() -> None:
    exec_source_launcher(_bootstrap_context(), _launch_argv, release_lock=release_pid_lock,
                         acquire_lock=acquire_pid_lock, before_release=lambda: stop_tray_before_exit(lambda: None),
                         on_failure=restore_tray_after_handoff, desktop=not _external_ui, macos=IS_MACOS,
                         cancelled=_shutdown_event.is_set)


def _shell_facts() -> dict:
    seed = _bundle_dir() if getattr(sys, "frozen", False) or _external_seed_bundle else None
    return shell_runtime_facts(REPO_DIR, seed, APP_VERSION, macos=IS_MACOS)


def _bootstrap_context() -> BootstrapContext:
    return BootstrapContext(
        bundle_dir=_bundle_dir(),
        repo_dir=REPO_DIR,
        data_dir=DATA_DIR,
        settings_path=SETTINGS_PATH,
        embedded_python=EMBEDDED_PYTHON,
        app_version=(_external_seed_bundle / "VERSION").read_text().strip() if _external_seed_bundle else APP_VERSION,
        hidden_run=_hidden_run,
        # Launcher is owner-process boundary; first-launch migration may set runtime mode.
        save_settings=lambda settings: save_settings(settings, allow_elevation=True),
        log=log,
        **({"deps_timeout_sec": PREPARATION_DEPS_TIMEOUT_SEC} if _runtime_venv and not _runtime_venv.ready else {}),
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
_agent_lock = threading.Condition()  # also tells a waiting stop that a start was published
_SPAWNING = object()
_agent_start: object = None  # _SPAWNING inside Popen; then, on Windows, the published child no resume was committed for
_agent_stopping: object = None  # the published child a stop already asked to end; a later stop only waits for it
_server_record_retire_lock = threading.Lock()  # the two retirers of state/server_process.json
_shutdown_event = threading.Event()
# Set by _request_agent_restart(): the next agent exit is a REQUESTED recycle
# (first-run configuration adopted), not a crash — same shape as the exit-code-42
# branch, for the case where the launcher, not the agent, decided the restart.
_agent_restart_requested = threading.Event()
_agent_exit_before_ready = threading.Event()  # the child exited before it served: the UI wait ends now
_webview_window = None
# Linux-only browser-fallback flag (#56): set by _detect_headless() when no
# pywebview GUI backend can initialize. Stays False on macOS/Windows — the
# probe never runs there, so every `if _headless:` branch is dead code on
# those platforms and their behavior is unchanged.
_headless = False
_external_ui = False
_external_host_update: Optional[pathlib.Path] = None
_external_host_result: dict = {}
_external_seed_bundle: Optional[pathlib.Path] = None
EMBEDDED_PYTHON = find_embedded_python(_bundle_dir(), macos=IS_MACOS)
_launch_argv: list[str] = []
_launch_intent = "owner"
_runtime_venv = None  # runtime_venv.RuntimeVenv selected by --runtime-venv
_generation_port: Optional[int] = None  # the endpoint the current generation proved healthy


def _recorded_server_identity(record: dict) -> str:
    """``"ours"``, ``"foreign"`` (gone, reused, another process) or ``"unknown"`` (unmeasurable, so no
    signal and no retirement). A measured fingerprint decides alone; older records keep their path/group/command proof."""
    try:
        pid = int(record.get("pid") or 0)
    except (TypeError, ValueError):
        return "foreign"
    if pid <= 0 or pid == os.getpid() or pid_provably_gone(pid):
        return "foreign"
    if isinstance(record.get("fingerprint"), dict):
        from ouroboros.process_custody import process_fingerprint_matches

        if process_fingerprint_matches(record, require_measured=True):
            return "ours"
        return "unknown" if process_fingerprint_matches(record) else "foreign"
    expected_server = str((REPO_DIR / "server.py").resolve())
    expected_repo = str(REPO_DIR.resolve())
    if record.get("server_path") not in (None, "", expected_server) or record.get("repo_dir") not in (None, "", expected_repo):
        return "foreign"
    try:
        recorded_pgid = int(record.get("pgid") or 0)
    except (TypeError, ValueError):
        recorded_pgid = 0
    live_pgid = process_group_id(pid)
    if not IS_WINDOWS and (recorded_pgid <= 0 or live_pgid != recorded_pgid):
        return "unknown" if recorded_pgid > 0 and live_pgid <= 0 else "foreign"
    command = process_command(pid)
    if not command:
        return "unknown"
    return "ours" if expected_server in command or ("server.py" in command and expected_repo in command) else "foreign"


def _write_server_process_record(proc: subprocess.Popen, *, port: int, server_py: pathlib.Path,
                                 server_host_source: str) -> None:
    try:
        record = {
            "pid": int(proc.pid),
            "pgid": process_group_id(proc.pid),
            "server_path": str(server_py.resolve()),
            "server_host_source": server_host_source,
            "repo_dir": str(REPO_DIR.resolve()),
            "requested_port": int(port),
            "port": int(port),
            "argv": [str(EMBEDDED_PYTHON), str(server_py.resolve())],
            "created_at": utc_now_iso(),
        }
        from ouroboros.process_custody import process_fingerprint

        record["fingerprint"] = process_fingerprint(proc.pid, record["argv"])  # the identity cleanup proves
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
        record.setdefault("requested_port", record["port"])
        record["port_updated_at"] = utc_now_iso()
        atomic_write_json(record_path, record, trailing_newline=True)
    except Exception:
        log.debug("Failed to update server process record port", exc_info=True)


def _retained_shared_daemon_pids() -> set[int]:
    """Read this installation's shared daemon custody; never start or adopt it."""
    from ouroboros.claudexor_daemon import CUSTODY_PURPOSE
    from ouroboros.process_custody import live_daemon_root_pids

    return live_daemon_root_pids(
        DATA_DIR, purposes={CUSTODY_PURPOSE}, retained_purposes={CUSTODY_PURPOSE}, strict=True,
    )


def _confirm_gone(pid: int, timeout: float = 5.0) -> bool:
    from ouroboros.process_containment import pid_is_zombie

    deadline = time.monotonic() + timeout
    while not (pid_provably_gone(pid) or pid_is_zombie(pid)):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)
    return True


def _cleanup_recorded_server_process(reason: str = "preflight") -> list[int]:
    """Stop the generation recorded in server_process.json; return the PIDs still unresolved.

    Only a positively identified server is signalled, and its record retires only after
    confirmed death or a refuted identity. An unmeasurable identity or an unconfirmed
    stop keeps the record and is returned, so no next generation starts beside it."""
    record_path = _server_process_record_path()
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError):
        record = None
    if not isinstance(record, dict):
        record_path.unlink(missing_ok=True)  # no identity to act on or to keep
        return []
    identity, pid = _recorded_server_identity(record), int(record.get("pid") or 0)
    if identity == "foreign":
        log.info("Retiring the server process record pid=%d: gone or another process (%s)", pid, reason)
        record_path.unlink(missing_ok=True)
        return []
    if identity == "unknown":
        log.warning("Recorded server pid=%d cannot be verified (%s): no signal, record retained", pid, reason)
        return [pid]
    pgid = int(record.get("pgid") or 0)
    log.info("Stopping recorded server process pid=%d pgid=%d (%s)", pid, pgid, reason)
    try:
        retained = _retained_shared_daemon_pids()
        if not IS_WINDOWS and pgid > 0 and pgid != current_process_group_id():
            terminate_process_group_id(pgid, exclude_pids=retained)
            time.sleep(0.5)
            kill_process_group_id(pgid, exclude_pids=retained)
        kill_pid_tree(pid, exclude_pids=retained)
    except Exception:
        log.warning("Shared daemon custody unavailable; stopping only the recorded server (%s)", reason, exc_info=True)
        force_kill_pid(pid)
    if not _confirm_gone(pid):
        log.error("Recorded server pid=%d survived its stop (%s); record retained", pid, reason)
        return [pid]
    record_path.unlink(missing_ok=True)
    return []


def _cleanup_recorded_server_group_for_pid(pid: int, reason: str = "agent_exit") -> bool:
    """Retire the record of a captured child this launcher has already reaped (both callers wait first).

    Nothing is signalled: once the root is reaped, its PID and group id are names a later process may
    hold, so a group that still has members is reported, never killed. Workers end through their parent
    lifeline, ledgered children through the next generation's custody sweep."""
    with _server_record_retire_lock:  # stop_agent and the lifecycle thread: Windows refuses a delete beside a read
        try:
            record_path = _server_process_record_path()
            if not record_path.exists():
                return True  # the other caller already retired it
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or int(record.get("pid") or 0) != int(pid):
                return True
            if not IS_WINDOWS and (pgid := int(record.get("pgid") or 0)) > 0:
                from ouroboros.process_containment import process_group_has_live_members
                if process_group_has_live_members(pgid):
                    log.warning("Server pid=%d exited while its former process group %d still has members; "
                                "not signalled after the reap (%s)", pid, pgid, reason)
            record_path.unlink(missing_ok=True)
        except Exception:
            log.warning("Failed to retire the recorded server process (%s)", reason, exc_info=True)
            return False
    return True


def start_agent(port: int = AGENT_SERVER_PORT) -> Optional[subprocess.Popen]:
    """Start server.py as the managed agent subprocess; None once the shutdown latch refused the start."""
    global _agent_proc, _agent_job, _agent_start
    # Stdlib-only; import before spawning so broken server imports cannot leave its pipe undrained.
    from ouroboros.process_logging import copy_capped_output

    settings = _load_settings()
    _apply_settings_to_env(settings)
    env = embedded_python_env(DATA_DIR, host_environment() if IS_LINUX else os.environ.copy())
    env["PYTHONPATH"] = str(REPO_DIR)
    # Environment owns the bind host (config.SETTINGS_KEYS_NOT_EXPORTED_TO_ENV).
    # Merged settings include the unauthored 127.0.0.1 default: setdefault supplies
    # it only when absent, preserving the operator override the export exclusion protects.
    saved_host = str(settings.get("OUROBOROS_SERVER_HOST") or "").strip()
    host_source = "environment" if str(env.get("OUROBOROS_SERVER_HOST") or "").strip() else "settings"
    if saved_host:
        env.setdefault("OUROBOROS_SERVER_HOST", saved_host)
    env["OUROBOROS_SERVER_PORT"] = str(port)
    env["OUROBOROS_DATA_DIR"] = str(DATA_DIR)
    env["OUROBOROS_REPO_DIR"] = str(REPO_DIR)
    env["OUROBOROS_APP_VERSION"] = _shell_facts()["bundle_version"] or str(APP_VERSION)
    env["OUROBOROS_MANAGED_BY_LAUNCHER"] = "1"
    env["OUROBOROS_MANAGED_REPO_DIR"] = str(REPO_DIR.resolve())
    env["OUROBOROS_LAUNCH_INTENT"] = _launch_intent  # this launcher's parsed intent, never an inherited one
    # Re-stamp presentation each generation (other hosts: web); a killed launcher's value stays until reap.
    # Env-only: SETTINGS_DEFAULTS would pop an injected value absent on disk.
    env["OUROBOROS_PRESENTATION"] = (
        str(os.environ.get("OUROBOROS_PRESENTATION") or "web")
        if _external_ui else "browser_fallback" if _headless else "desktop_window"
    )
    env.update(background_env(env["OUROBOROS_PRESENTATION"]))  # can this launcher keep running with its window hidden
    if _external_host_update is not None:
        env["OUROBOROS_EXTERNAL_HOST_UPDATE"] = str(_external_host_update)
        env["OUROBOROS_EXTERNAL_HOST_RESULT"] = json.dumps(_external_host_result)
    else:
        env.pop("OUROBOROS_EXTERNAL_HOST_UPDATE", None)
        env.pop("OUROBOROS_EXTERNAL_HOST_RESULT", None)
    # The managed server still resolves payloads from this bundle.
    env[BUNDLE_DIR_ENV] = str(_bundle_dir())

    server_py = REPO_DIR / "server.py"
    log.info("Starting agent: %s %s (port=%d)", EMBEDDED_PYTHON, server_py, port)

    env["OUROBOROS_LAUNCHER_STOP_STDIN"] = "1"  # every platform: quit stops, EOF (launcher gone) too
    popen_kwargs: dict = {
        "cwd": str(REPO_DIR),
        "env": env,
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
    }
    popen_kwargs.update(subprocess_new_group_kwargs(suspended=IS_WINDOWS))  # Windows resumes it once its Job owns it
    with _agent_lock:  # begin; a stop that finds this start inside Popen waits for it within its own grace
        if _shutdown_event.is_set():
            return None
        _agent_start, proc = _SPAWNING, None
    try:  # hidden-window flags: the server starts without a console window (CREATE_NO_WINDOW on Windows)
        proc = _hidden_popen([EMBEDDED_PYTHON, str(server_py)], **popen_kwargs)
    finally:
        with _agent_lock:  # publish it (on Windows not yet resumed), or settle the failed spawn
            _agent_proc, _agent_start = proc, proc if IS_WINDOWS else None  # none is published while one starts
            _agent_lock.notify_all()

    if IS_WINDOWS:
        job = create_kill_on_close_job(allow_breakaway=True)
        owned = job is not None and assign_pid_to_job(job, proc.pid)
        with _agent_lock:  # Job before resume, decided against a stop: one that came first keeps it suspended
            resume = owned and _agent_start is proc and not _shutdown_event.is_set()
            if resume:
                _agent_job, _agent_start = job, None
        if not (resume and resume_process(proc.pid)):  # one end for a child that must never run
            first = owned and not resume
            log.log(logging.INFO if first else logging.ERROR, "Killing agent pid %d before it runs: %s", proc.pid,
                    "its stop came first" if first else "resume failed" if owned else "it has no Windows Job Object")
            terminate_job(job)
            if not resume:  # unpublished: this starter owns it; a published Job closes when its generation retires
                close_job(job)
            proc.kill()
            return proc
        log.info("Agent pid %d assigned to Windows Job Object", proc.pid)

    _write_server_process_record(proc, port=port, server_py=server_py, server_host_source=host_source)
    # The server blocks once its output pipe fills, so the copy drains until the pipe ends.
    threading.Thread(target=copy_capped_output, args=(proc.stdout, DATA_DIR / "logs" / "agent_stdout.log"),
                     kwargs={"report": log}, name="agent-output-copy", daemon=True).start()
    return proc


def stop_agent() -> bool:
    """Ordinarily stop the captured server; True once its exit is confirmed. An unconfirmed stop keeps
    the Popen, Job and record, so no next generation starts beside a server that may still be live. A
    child another stop already asked to end (its pipe closed) is only awaited, never signalled again."""
    global _agent_start, _agent_stopping
    grace, deadline = LAUNCHER_STOP_GRACE_SEC, time.monotonic() + LAUNCHER_STOP_GRACE_SEC
    with _agent_lock:
        if _agent_start is _SPAWNING:  # its publication and this stop share one grace, never a second
            _agent_lock.wait_for(lambda: _agent_start is not _SPAWNING, grace)
            grace = max(0.0, deadline - time.monotonic())
        proc, job, unpublished = _agent_proc, _agent_job, _agent_start is _SPAWNING
        if unresumed := proc is not None and _agent_start is proc:
            _agent_start = None  # this stop decided first: its starter never resumes it
        asked, _agent_stopping = proc is not None and _agent_stopping is proc, proc  # one request per child
    if proc is None:
        if unpublished:
            log.error("A server start was still inside Popen at the stop's grace; its child is unconfirmed")
        return not unpublished

    log.info("Stopping agent (pid=%s)...", proc.pid)
    confirmed = False
    try:
        if unresumed:  # nothing to drain: straight to the forced stop and its confirmed exit
            raise subprocess.TimeoutExpired("server", 0)
        stdin = None if asked else getattr(proc, "stdin", None)
        if stdin is not None:  # the owned pipe asks the server to drain first; Windows terminate is forced
            try:
                stdin.write(b"quit\n")
                stdin.flush()
            except (OSError, ValueError):
                log.warning("Launcher stop pipe unavailable; retaining the exit wait and fallback")
                if not IS_WINDOWS:
                    proc.terminate()
            finally:
                with suppress(OSError, ValueError):  # a failed pipe must still reach the wait/forced cleanup
                    stdin.close()
        elif not asked:
            proc.terminate()
        proc.wait(timeout=grace)
        confirmed = True
    except subprocess.TimeoutExpired:
        if IS_WINDOWS and job is not None:
            terminate_job(job)
        else:
            try:
                kill_process_tree(proc, exclude_pids=_retained_shared_daemon_pids())
            except Exception:
                log.warning("Shared daemon custody unavailable; stopping only the captured server", exc_info=True)
                proc.kill()
        with suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)
            confirmed = True
    except Exception:
        log.warning("Ordinary agent stop failed", exc_info=True)
    if not (confirmed or (callable(getattr(proc, "poll", None)) and proc.poll() is not None)):
        log.error("Agent pid=%s did not confirm its exit; its handle and record stay for the next stop", proc.pid)
        return False
    return _retire_agent(proc, "stop_agent")


def _retire_agent(proc, reason: str) -> bool:
    """A reaped generation: retire its publication, the Job it owned and its record; a later one's stay."""
    global _agent_proc, _agent_job, _agent_start, _agent_stopping
    with _agent_lock:
        job = _agent_job if _agent_proc is proc else None
        if _agent_proc is proc:  # no start begins before the loop retired its generation
            _agent_proc = _agent_job = _agent_start = _agent_stopping = None
    if IS_WINDOWS and job is not None:
        close_job(job)
    return _cleanup_recorded_server_group_for_pid(proc.pid, reason)


def _read_port_file(default: Optional[int] = AGENT_SERVER_PORT) -> Optional[int]:
    """Read the active server port from PORT_FILE."""
    try:
        if PORT_FILE.exists():
            return int(PORT_FILE.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        pass
    return default


def _requested_server_port() -> int:
    """``OUROBOROS_SERVER_PORT`` as requested, else the historical default; the server keeps its fallback."""
    raw = str(os.environ.get("OUROBOROS_SERVER_PORT") or "").strip()
    return int(raw) if raw else AGENT_SERVER_PORT


def _reap_same_install_strays(reason: str) -> list[int]:
    """Kill leftover generations of THIS install's server; return proven survivors.

    Only ever called while this process holds the single-instance pid lock, which is what makes the
    identity rule sound: with the lock held, another process running this install's server.py under
    this launcher's stamped environment cannot be a live peer's generation. Never called from a
    panic or window-close path — Emergency Stop tears down what it owns and adds no new killing.
    """
    try:
        return _reap_same_install_strays_impl(
            REPO_DIR, DATA_DIR, reason, retained_descendant_roots=_retained_shared_daemon_pids(),
        )
    except Exception:
        # A sweep that cannot run must not stop the launcher booting.
        log.warning("Same-install stray sweep failed (%s)", reason, exc_info=True)
        return []


def _pre_generation_cleanup() -> list[int]:
    """Resolve the previous generation before starting a new one; returns PIDs still unresolved.

    Per GENERATION, not once per launcher: exit-42 and crash restarts are where the observed
    double-boot collisions began. Recorded cleanup first, then the stray sweep while PPID links
    are still live. No port sweep: a listener proves no ownership, and the data-root admission
    lock refuses a second updated server whatever port it would take."""
    unresolved = _cleanup_recorded_server_process("startup")
    return unresolved + [pid for pid in _reap_same_install_strays("startup") if pid not in unresolved]


def _clear_port_file() -> bool:
    """Remove the previous generation's port publication; True only once it is confirmed gone."""
    try:
        PORT_FILE.unlink(missing_ok=True)
    except OSError:
        log.warning("Could not remove %s; only this generation's binding can prove readiness", PORT_FILE)
    return not PORT_FILE.exists()


def _await_generation_endpoint(proc, *, port_file_cleared: bool, abort_event=None) -> Optional[int]:
    """The port THIS child published, or None when it exited or the launcher stopped. The wait lasts
    as long as the child lives: no clock decides that a live server failed (its boot is long on a
    cold disk; the server answers ``/api/health`` as soon as its socket serves). The main binding
    counts only with the child's PID and measured birth identity (a stale row of an earlier server
    that held the same PID fails); the plain port file only after its pre-start removal was confirmed."""
    from ouroboros.server_process import read_service_bindings, service_binding_is_live

    while not (abort_event is not None and abort_event.is_set()):
        if proc.poll() is not None:
            return None
        try:
            binding = read_service_bindings(DATA_DIR).get("main")
            if isinstance(binding, dict) and binding.get("pid") == proc.pid and service_binding_is_live(binding):
                return int(binding["port"])
        except (OSError, ValueError, TypeError):
            pass  # an informational snapshot; the confirmed-fresh port file remains
        if port_file_cleared and (published := _read_port_file(default=None)):
            return published
        time.sleep(0.25)
    return None


def _kill_orphaned_children(reason: str = "window_close") -> None:
    """Shared close/Panic final cleanup (not the startup stray sweep)."""
    from ouroboros.launcher_server_reaper import kill_runtime_orphans

    kill_runtime_orphans(DATA_DIR, _cleanup_recorded_server_process, reason=reason, log=log)


def agent_lifecycle_loop(port: int = AGENT_SERVER_PORT) -> None:
    """Start/monitor agent; restart on code 42 or bounded crashes."""
    global _external_host_result, _generation_port
    crash_times: list[float] = []
    while not _shutdown_event.is_set():
        unresolved = _pre_generation_cleanup()
        if unresolved:
            # Starting now could put a second generation on the same data directory: a known
            # predecessor is unconfirmed or unverifiable. The next iteration tries again.
            log.error("Not starting the agent: previous server process(es) %s are not confirmed "
                      "stopped. Retrying in 3s.", unresolved)
            time.sleep(3)
            continue
        if not automatic_launch_allowed(_launch_intent, DATA_DIR, log):
            # Panic landed after main()'s check (a long bootstrap, a restart): stay stopped.
            _shutdown_event.set()
            break
        port_file_cleared = _clear_port_file()
        _agent_exit_before_ready.clear()  # this generation proves itself

        _external_host_result = update_external_host(_external_host_update, EMBEDDED_PYTHON, log, _shutdown_event)
        if _shutdown_event.is_set():
            break  # Native preparation has reaped its owned processes before returning.
        proc = start_agent(port)  # None when the latch came during its preparation: begin refuses then
        if _shutdown_event.is_set():
            stop_agent()
            break

        actual_port = _await_generation_endpoint(proc, port_file_cleared=port_file_cleared, abort_event=_shutdown_event)
        if actual_port is None:
            log.warning("Agent server exited (or the launcher stopped) before it published an endpoint")
        else:
            _update_server_process_record_port(proc.pid, actual_port)
            if _wait_for_server(actual_port, abort_event=_shutdown_event, proc=proc):
                _generation_port = actual_port
            else:
                log.warning("Agent server exited before it answered /api/health on port %d", actual_port)

        proc.wait()
        proved = _generation_port is not None
        _generation_port = None
        exit_code = proc.returncode
        log.info("Agent exited with code %d", exit_code)
        if not proved and exit_code not in (RESTART_EXIT_CODE, PANIC_EXIT_CODE) and not _agent_restart_requested.is_set():
            _agent_exit_before_ready.set()  # a child that died before serving: the UI wait fails now, not at a deadline
        if exit_code == RESTART_EXIT_CODE:
            try:
                from ouroboros.delegate_recovery import acknowledge_observed_restart_exit

                if not acknowledge_observed_restart_exit(
                    DATA_DIR, supervisor_pid=proc.pid, exit_code=exit_code,
                ):
                    log.info("No prepared delegated-restart transaction required acknowledgement.")
            except Exception:
                log.warning("Could not acknowledge delegated restart transaction", exc_info=True)
        _retire_agent(proc, "agent_exit")

        if _shutdown_event.is_set() and exit_code != PANIC_EXIT_CODE:
            break

        if exit_code == PANIC_EXIT_CODE:
            log.info("Panic stop (exit code %d) — shutting down completely.", PANIC_EXIT_CODE)
            _shutdown_event.set()
            request_tray_cleanup()  # Panic removes the indicator without waiting for it
            # The agent (server child) already exited; tear down any orphans and
            # force-exit the whole process. _webview_window.destroy() from this
            # supervisor thread cannot end the main-thread Cocoa webview loop on
            # macOS (it leaves a black frozen window), so exit with parity to the
            # window-close path: kill orphans, release the pid lock, os._exit(0).
            try:  # the server's census may not have finished before its hard exit: finish it by identity, bounded
                from ouroboros.process_scope import complete_pending_panic_sweep

                completion = threading.Thread(
                    target=complete_pending_panic_sweep, args=(DATA_DIR,),
                    kwargs={"exclude": {os.getpid(), os.getppid(), 1}, "installation": True}, name="panic-scope-completion", daemon=True)
                completion.start()
                completion.join(timeout=3.0)  # later starts retry saved identities only; the exit never waits longer
            except Exception:
                log.warning("Pending Panic scope sweep failed; later starts retry saved identities only", exc_info=True)
            _kill_orphaned_children(reason="panic_stop")
            stop_tray_before_exit(release_pid_lock, wait=0)
            os._exit(0)

        time.sleep(2)

        if _shutdown_event.is_set():
            break

        if _agent_restart_requested.is_set():
            # Launcher-requested recycle (first-run configuration): deliberate,
            # so it skips crash accounting exactly like the agent's own code-42
            # restart. No dependency sync — nothing about the checkout changed.
            _agent_restart_requested.clear()
            log.info("Restarting the agent to adopt the saved first-run configuration.")
            continue

        if exit_code == RESTART_EXIT_CODE:
            log.info("Agent requested restart (exit code 42). Restarting...")
            _sync_existing_repo_from_bundle()
            if not _install_deps():
                # Retry dependency sync once before letting imports fail under the crash fuse.
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
            if launcher_sources_changed(REPO_DIR, bundle_dir=_bundle_dir(), source_root=pathlib.Path(__file__).parent):
                _exec_source_launcher()  # a refused successor: this launcher starts the next generation itself
            continue

        now = time.time()
        crash_times.append(now)
        crash_times[:] = [stamp for stamp in crash_times if (now - stamp) < CRASH_WINDOW_SEC]
        if len(crash_times) >= MAX_CRASH_RESTARTS:
            log.error("Agent crashed %d times in %ds. Stopping.", MAX_CRASH_RESTARTS, CRASH_WINDOW_SEC)
            break

        log.info("Agent crashed. Restarting in 3s...")
        time.sleep(3)

def _request_agent_restart() -> None:
    """Recycle the managed server so it adopts freshly saved settings.

    The lifecycle loop owns the restart; this flags intent and stops the child.
    """
    global _generation_port
    if _agent_proc is None:  # one published reference: read without the lock, as Panic reads it
        # Leaving the flag set would make the NEXT ordinary agent exit look like
        # a requested restart and skip the crash fuse.
        log.info("No managed agent to recycle; the next start reads the saved settings.")
        return
    _agent_restart_requested.set()
    _generation_port = None  # the next generation proves its own endpoint
    stop_agent()


def _await_server_ready(port: int, abort_event=None, lifecycle_thread=None) -> tuple[bool, int]:
    """Wait until the lifecycle thread proves the current generation serving; its port.

    No clock: the wait lasts while the generation lives and the lifecycle thread runs. It ends
    False at once when the child exits before it served (``_agent_exit_before_ready``), when the
    lifecycle thread ended (the crash fuse) or on abort. The server may rebind on conflict; every
    later consumer (UI URL, onboarding window) uses the endpoint that generation published, never
    the requested port or a stale file.
    """
    if _external_host_update is not None and lifecycle_thread is not None:
        # Native preparation has its own bounded operation before server.py exists.
        # Do not spend the HTTP readiness window waiting for a compiler. On shutdown
        # the hook observes the event and reaps before this thread can finish.
        while lifecycle_thread.is_alive() and _agent_proc is None:  # one published reference, read lock-free
            lifecycle_thread.join(timeout=0.1)
        if _agent_proc is None:
            return False, port
    while not (abort_event is not None and abort_event.is_set()):
        if _generation_port:
            return True, _generation_port
        if lifecycle_thread is not None and not lifecycle_thread.is_alive():
            break
        if _agent_exit_before_ready.is_set():
            break
        time.sleep(0.1)
    return False, _generation_port or port


def _load_settings() -> dict:
    return load_settings()


def _save_settings(changes: dict) -> None:
    # The desktop launcher is the owner-controlled writer.  The generic
    # config.save_settings ratchet deliberately makes allow_elevation inert
    # after boot, which is correct for agent-reachable callers but also made a
    # confirmed Cyber Pro selection fall back to Advanced.  Reuse the existing
    # owner writer so the confirmation is honored while its document lock,
    # persistence normalization and other owner-only checks remain in force.
    # Only ``changes`` land, on the document as it is under that lock: a merged
    # read's defaults, environment values and proposals are no choice.
    from ouroboros.gateway.owner_settings import _owner_update_settings, fields_on_saved_document

    _owner_update_settings(fields_on_saved_document(changes))


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
    _save_settings({"OUROBOROS_RUNTIME_MODE": new_mode})
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
    _save_settings({"OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS": "true" if new_enabled else "false"})
    return {"ok": True, "enabled": new_enabled}


def _request_skill_key_grant(skill: str, keys: list, confirm_fn, *, enable_after_grant: bool = False,
                             allow_pending: bool = False) -> dict:
    from ouroboros.config import runtime_setting, SETTINGS_DEFAULTS
    from ouroboros.skill_loader import find_skill
    from ouroboros.skill_lifecycle_actions import prepare_skill_grant

    skill_name = str(skill or "").strip()
    items = [str(key or "").strip() for key in (keys or []) if str(key or "").strip()]
    repo_path = str(_load_settings().get("OUROBOROS_SKILLS_REPO_PATH") or "")
    loaded = find_skill(DATA_DIR, skill_name, repo_path=repo_path)
    if loaded is None:
        return {"ok": False, "error": f"Skill {skill_name!r} not found"}
    preview = prepare_skill_grant(loaded, DATA_DIR, items)
    if preview.get("error"):
        return preview
    message = (
        f"Grant skill {loaded.name!r} access to these settings keys / host permissions?\n\n"
        + "\n".join([*preview["keys"], *preview["permissions"]])
        + ("\n\nThen turn this skill on." if enable_after_grant else "")
        + "\n\nOnly grant keys and permissions to reviewed skills you trust."
    )
    if not confirm_fn("Confirm Skill Grant", message):
        return {"ok": False, "error": "Skill grant cancelled."}
    # The immutable launcher never imports, registers or installs packages for a skill: after its
    # native confirm the server's grant owner revalidates the revision confirmed above, saves the
    # grant, turns the skill on when ready and readies its packages (the same POST the browser sends).
    actual_port = _read_port_file() or AGENT_SERVER_PORT
    req = urllib.request.Request(
        f"http://127.0.0.1:{actual_port}/api/skills/{urllib.parse.quote(loaded.name)}/grants", method="POST",
        data=json.dumps({"items": items, "expected_content_hash": preview["content_hash"],
                         **({"enable_after_grant": True} if enable_after_grant else {})}).encode("utf-8"),
        headers={"Content-Type": "application/json", **({"Prefer": "respond-async"} if allow_pending else {})},
    )
    try:
        timeout = 20 if allow_pending else float(runtime_setting(
            "OUROBOROS_SKILL_LIFECYCLE_TIMEOUT_SEC", SETTINGS_DEFAULTS["OUROBOROS_SKILL_LIFECYCLE_TIMEOUT_SEC"]))
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:  # a typed refusal is the server's answer, not a transport failure
        try:
            return json.loads(exc.read().decode("utf-8") or "{}") or {"ok": False, "error": f"HTTP {exc.code}"}
        except ValueError:
            return {"ok": False, "error": f"HTTP {exc.code}"}
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": f"The server did not answer the grant: {exc}"}


def _webview_ready() -> tuple[bool, str]:
    """Probe whether pywebview can actually run a GUI window here.

    Lazy backend selection lets import succeed without GTK/QT (#56). initialize()
    uses PYWEBVIEW_GUI/session selection and is safe to repeat at start(). Linux
    also checks `_display_ready`; macOS/Windows stay import-only and `_detect_headless` skips them.
    """
    if IS_LINUX and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        # BEFORE initialize(): even SELECTING a Qt backend constructs a
        # QGuiApplication, which can ABORT on a display-less box.
        return False, "no DISPLAY or WAYLAND_DISPLAY in the environment"
    try:
        if IS_LINUX:
            from ouroboros.desktop_media import prepare_linux_bundle

            prepare_linux_bundle(_bundle_dir())
        import webview  # noqa: F401  (can itself fail on bare source checkouts)
        from webview.guilib import initialize

        backend = initialize()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if not IS_LINUX:
        return True, ""
    route_gtk_browser(backend, lambda url, new=0, autoraise=True: bool(
        _open_browser_detached(url, new=new, autoraise=autoraise)))
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


def _open_browser_detached(url: str, outcome: Optional[list] = None, *, new=0, autoraise=True) -> threading.Thread:
    """Best-effort open on a daemon thread: console browsers can wait indefinitely.
    The printed URL is primary. Return the thread for short-lived callers to
    bound-join; ``outcome`` receives one bool or exception for honest bridge errors.
    Owner-approved: the user's browser, including a surviving console browser, is
    outside launcher custody/teardown. Emergency Stop governs only the agent tree.
    """
    def _open() -> None:
        try:
            if IS_LINUX and os.environ.get("OUROBOROS_DESKTOP_ENV_SAVED") == "1":
                # webbrowser has no env argument: resolve BROWSER/defaults in a clean child, never swap GUI env.
                opened = subprocess.run([
                    EMBEDDED_PYTHON, "-I", "-S", "-c",
                    "import sys, webbrowser; sys.exit(0 if webbrowser.open(sys.argv[1], int(sys.argv[2]), sys.argv[3]=='1') else 1)",
                    url, str(new), '1' if autoraise else '0',
                ], **merge_hidden_kwargs({"env": host_environment()}))
                result: object = opened.returncode == 0
            else:
                result = bool(webbrowser.open(url) if new == 0 and autoraise else webbrowser.open(url, new, autoraise))
        except Exception as exc:
            result = exc
            log.info("Could not open the default browser for %s", url, exc_info=True)
        if outcome is not None:
            outcome.append(result)

    thread = threading.Thread(target=_open, name="ouroboros-open-browser", daemon=True)
    thread.start()
    return thread


def _open_external_url(url: str) -> dict:
    """Shared external-link handoff for both desktop window bridges."""
    try:
        raw = str(url or "")
        if not raw.lower().startswith(("http://", "https://", "mailto:")):
            return {"ok": False, "error": "Only absolute http://, https:// or mailto: links can be opened."}
        outcome: list = []
        # Settled failure is reported; a slow browser stays detached.
        _open_browser_detached(raw, outcome).join(timeout=3.0)
        if outcome and outcome[0] is not True:
            return {"ok": False, "error": f"The default browser could not be opened: {outcome[0] or 'no handler found'}"}
        return {"ok": True}
    except Exception as exc:
        log.warning("Desktop external-URL open failed: %s", exc, exc_info=True)
        return {"ok": False, "error": str(exc)}


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
        log.info("Headless mode: serving the UI at %s", url)
        if _external_ui:
            print(f"Ouroboros is running at {url}. UI is owned by the host.", flush=True)
        else:
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
    if _external_host_update is not None:
        lifecycle_thread.join()  # The hook observes shutdown and owns its bounded reap.
    _kill_orphaned_children(reason="headless_shutdown" if requested_shutdown else "crash_fuse")
    # NO explicit release_pid_lock(): sys.exit runs the atexit-registered
    # release, and a second release would unconditionally unlink a lock a
    # NEWER launcher may own (explicit calls stay only on os._exit paths).
    sys.exit(0 if requested_shutdown else 1)


def main(argv=()):
    global _headless, _external_ui, _external_host_update, _external_seed_bundle, _launch_argv, _launch_intent, \
        _runtime_venv, EMBEDDED_PYTHON
    from ouroboros.delegate_recovery import observe_restart_parent
    if IS_WINDOWS and observe_restart_parent().get("exit_code") == PANIC_EXIT_CODE:
        return  # source handoff may be running before its parent's Popen returns
    options = parse_launch_options(argv)
    _launch_argv, _launch_intent = list(argv), options.launch_intent
    _external_ui = options.no_ui
    _external_host_update = options.host_update.resolve() if options.host_update is not None else None
    _external_seed_bundle = options.seed_bundle.resolve() if options.seed_bundle is not None else None
    if IS_MACOS and _external_seed_bundle is not None \
            and pathlib.Path(sys.executable) == macos_source_entry(_external_seed_bundle):
        EMBEDDED_PYTHON = find_embedded_python(_external_seed_bundle, macos=IS_MACOS)
    if _external_ui:
        _headless = True
    if IS_WINDOWS and not _external_ui:
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

    if not _external_ui:
        _detect_headless()

    if not _headless:
        import webview

    if not acquire_pid_lock():
        log.error("Another instance already running.")
        if other_copy := ("" if options.launch_intent == "automatic" or _headless else other_copy_line(PID_FILE, PORT_FILE, _bundle_dir())):
            _already_running_window(other_copy)  # then the running window comes forward, as before
        if options.launch_intent == "automatic" or (not _headless and activate_running_instance(PID_FILE)) or other_copy:
            return  # a sign-in start stays quiet; a manual launch showed the running window
        if _headless:
            # A second Open may race bootstrap's absent/stale port. Re-read
            # between health probes; a final probe can overshoot this soft
            # deadline by its own urlopen timeout. The last port is best-effort.
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
            if not _external_ui:
                _open_browser_detached(existing_url).join(timeout=5.0)
            return
        _already_running_window("Only one instance can run at a time.")
        return

    import atexit
    finalizer = lambda: stop_tray_before_exit(release_pid_lock)
    atexit.register(finalizer)

    if not automatic_launch_allowed(options.launch_intent, DATA_DIR, log):
        return
    # Record the loaded generation before bootstrap can advance the managed checkout.
    remember_loaded_checkout(REPO_DIR, bundle_dir=_bundle_dir() if getattr(sys, "frozen", False) or pathlib.Path(
        __file__).resolve() != (REPO_DIR / "launcher.py").resolve() else None)

    def _on_closing() -> None:  # Background's close body: its exit_launcher latched shutdown on the UI thread
        log.info("Window closing — graceful shutdown.")
        from ouroboros.launcher_bootstrap import stop_bootstrap_commands
        stop_bootstrap_commands(include_interactive=False)
        if background.panic.is_set():
            return  # Panic owns persistence and hard exit, never ordinary teardown over it
        if _webview_window is None:
            if _agent_proc is not None or _agent_start is not None:  # published, or still inside its Popen
                stop_agent()
                _cleanup_recorded_server_process("startup_quit")
            for window in list(webview.windows):
                window.destroy()  # release Git/setup's webview.start before main returns
            return  # startup owns its return; no stale companion rows or hard exit
        stop_agent()
        if background.panic.is_set():
            return  # accepted during the stop: no ordinary teardown or exit over it
        _kill_orphaned_children()
        background.leave(release_pid_lock)

    # Listen before the long boot: a manual second launch meanwhile is kept and opens the window once it exists.
    background = None if _headless else Background(_on_closing, _read_port_file, _shutdown_event,
        panic_child=lambda: (_agent_proc, _agent_job))
    if background is not None:
        background.listen(PID_FILE)
        atexit.unregister(finalizer)  # after CLR's unload registration: dispose the icon while CLR still lives
        atexit.register(finalizer)
    if _shutdown_event.is_set():
        return  # Early Panic owns persistence and exit; bootstrap must not start beside it.

    git_found = check_git()
    if _shutdown_event.is_set():
        return
    if not git_found:
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
                    _hidden_popen(["xcode-select", "--install"], bootstrap=True)
                elif IS_WINDOWS:
                    _hidden_popen(
                        ["winget", "install", "Git.Git", "--source", "winget", "--accept-source-agreements"], bootstrap=True
                    )
                else:
                    for cmd in (
                        ["sudo", "apt", "install", "-y", "git"],
                        ["sudo", "dnf", "install", "-y", "git"],
                    ):
                        try:
                            _hidden_popen(cmd, bootstrap=True)
                            break
                        except FileNotFoundError:
                            continue
                for _ in range(300):
                    if _shutdown_event.wait(3):
                        return "stopped"
                    if shutil.which("git"):
                        return "installed"
                return "timeout"

        git_window = webview.create_window(
            "Ouroboros — Setup Required",
            html=_window_page("Ouroboros — Setup Required", (
                "<h2>Git is required</h2><p>Ouroboros needs Git to manage its local repository.</p>"
                + ("" if IS_MACOS else f'<p class="muted">{_hint}</p>')
                + '<button id="install-btn" class="btn btn-primary" type="button">'
                + ("Install Git (Xcode CLI Tools)" if IS_MACOS else "Install Git")
                + '</button><p id="status" class="ui-status" role="status"></p>')),
            js_api=GitApi(),
            width=520,
            height=300,
        )
        paint_room(git_window)
        webview.start(func=_git_page, args=[git_window], private_mode=False)
        if _shutdown_event.is_set():
            return
        if not check_git():
            sys.exit(1)

    if _shutdown_event.is_set():
        return
    if options.runtime_venv is not None:  # under the PID lock, before bootstrap and world generation
        try:
            _runtime_venv = prepare_runtime_venv(options.runtime_venv, run=_hidden_run, log=log,
                base_python=(EMBEDDED_PYTHON if getattr(sys, "frozen", False)
                             else getattr(sys, "_base_executable", sys.executable)))
        except RuntimeVenvRefused as refusal:
            log.error("%s", refusal)
            print(f"Ouroboros did not start: {refusal}", file=sys.stderr, flush=True)
            sys.exit(1)
        EMBEDDED_PYTHON = _runtime_venv.python  # the body and its installs; the macOS shell keeps app identity
        os.environ["PATH"] = os.pathsep.join([str(_runtime_venv.bin_dir), os.environ.get("PATH", "")])
    if not bootstrap_repo():
        # Disclosed, not fatal: an already-provisioned install with satisfied
        # requirements still runs, and a genuinely broken checkout is stopped
        # by the startup health check / crash fuse with this line naming why.
        log.error(
            "Continuing startup after a failed dependency install; the agent may "
            "be missing packages (see the pip output above)."
        )
    elif _runtime_venv is not None and not _runtime_venv.ready and _install_deps():
        try:
            _runtime_venv = mark_ready(_runtime_venv)  # a failed install stays preparing and retries next start
        except RuntimeVenvRefused as refusal:
            log.error("%s", refusal)
            print(f"Ouroboros did not start: {refusal}", file=sys.stderr, flush=True)
            sys.exit(1)

    if _shutdown_event.is_set():
        return  # Panic during bootstrap must not hand off or start an owner-intent server.

    # Cold handoff includes helpers imported later from the seed. An old macOS app keeps its identity;
    # a current one uses the signed source entry. Preserve an Open received during boot on every desktop.
    if IS_MACOS and getattr(sys, "frozen", False) and macos_source_entry(_bundle_dir()) is None:
        log.warning("This app lacks the macOS source entry; source restarts need a new app installation "
                    "to preserve the app profile and notification identity.")
    elif launcher_sources_changed(REPO_DIR, bundle_dir=_bundle_dir(), packaged=True,
                                  source_root=pathlib.Path(__file__).parent):
        if background is not None and background._open_pending:
            _launch_argv += ["--launch-intent", "owner"]
        _exec_source_launcher()

    if _shutdown_event.is_set():
        return

    # Decide setup here; present only after the gateway serves. Lock/Git/bootstrap precede the server.
    onboarding_settings, onboarding_required = _prepare_first_run_settings()
    if _shutdown_event.is_set():
        return

    global _webview_window
    port = _requested_server_port()  # the lifecycle loop clears the previous generation first

    # Headless: handlers BEFORE the lifecycle thread spawns server.py — a
    # SIGTERM while the server still boots must reach our teardown.
    _abort = _shutdown_event
    if _headless:
        install_shutdown_signal_handlers(_headless_signal_handler)
        _abort = _shutdown_event

    lifecycle_thread = threading.Thread(target=agent_lifecycle_loop, args=(port,), daemon=True)
    lifecycle_thread.start()

    server_ready, actual_port = _await_server_ready(port, _abort, lifecycle_thread)

    if server_ready and onboarding_required and not _shutdown_event.is_set():
        # The gateway is live and, with no provider configured, runs WITHOUT a
        # supervisor — the supported state that lets the wizard reach /api/*.
        onboarding = _present_first_run_onboarding(
            onboarding_settings, actual_port, headless=_headless,
            open_external_url=_open_external_url, read_port=lambda: _generation_port,
        )
        if not onboarding["saved"]:
            log.info(
                "Setup was closed without saving. Launching anyway "
                "(the blocking onboarding overlay and Settings remain available)."
            )
        if onboarding["restart_required"] and not _shutdown_event.is_set():
            # Recycle the owned server to adopt boot-pinned settings instead of leaving a restart nag.
            _request_agent_restart()
            server_ready, actual_port = _await_server_ready(port, _abort, lifecycle_thread)

    if _shutdown_event.is_set():
        if background is not None and background.panic.is_set():
            return  # the non-daemon Panic owner, not a startup-failure window, owns exit
        # Startup shutdown is no failure; stop owned/proven servers, never stale companion rows.
        log.info("Shutdown requested during startup; tearing down.")
        stop_agent()
        _cleanup_recorded_server_process("startup_abort")
        # atexit releases on sys.exit; a second release could unlink a newer launcher's lock.
        sys.exit(0)

    if not server_ready:
        log.error("Agent failed to become healthy on port %d; aborting UI startup.", actual_port)
        _shutdown_event.set()
        stop_agent()
        lifecycle_thread.join(timeout=5)
        if _headless:
            _cleanup_recorded_server_process("startup_failure")
            print(
                "Ouroboros failed to start: the local agent server did not become ready.\n"
                f"See {_log_dir / 'launcher.log'} and {_log_dir / 'agent_stdout.log'} for details.",
                file=sys.stderr,
            )
            sys.exit(1)
        paint_room(webview.create_window(
            "Ouroboros — Startup Failed",
            html=_window_page("Ouroboros — Startup Failed", (
                "<h2>Ouroboros failed to start</h2>"
                "<p>The local agent server did not become ready.</p>"
                '<p class="muted">Check launcher.log and agent_stdout.log in the Ouroboros data directory '
                "for details.</p>")),
            width=520,
            height=260,
        ))
        webview.start(private_mode=False)
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

    class MainApi(DesktopApi):  # alerts, shell facts and system notifications: launcher_background.DesktopApi
        _background = background
        _shell_facts = _shell_facts()
        @staticmethod
        def _native_confirm(title: str, message: str) -> bool:
            return bool(_webview_window and _webview_window.create_confirmation_dialog(title, message))

        def request_runtime_mode_change(self, mode: str) -> dict:
            try:
                return _request_runtime_mode_change(mode, self._native_confirm)
            except Exception as exc:
                log.warning("Runtime mode native confirmation failed: %s", exc, exc_info=True)
                return {"ok": False, "error": f"Native confirmation failed: {exc}"}

        def confirm_runtime_mode_change(self, mode: str) -> dict:
            """Confirm a mode change without writing it.

            The SPA persists the selected mode through the owner HTTP endpoint.
            Keeping this bridge side-effect free lets older shells fall back to
            the same in-app confirmation instead of normalizing newer modes
            such as Cyber Pro through their stale local enum.
            """
            try:
                mode_text = str(mode or "").strip().lower()
                if mode_text not in {"light", "advanced", "pro", "cyber_pro"}:
                    return {"confirmed": False, "error": "Unknown runtime mode."}
                settings = _load_settings()
                current = normalize_runtime_mode(settings.get("OUROBOROS_RUNTIME_MODE"))
                message = (
                    f"Change Ouroboros runtime mode from {current} to {mode_text}?\n\n"
                    "The new mode is saved through the owner endpoint and takes effect after restart."
                )
                return {"confirmed": bool(self._native_confirm("Confirm Runtime Mode Change", message))}
            except Exception as exc:
                log.warning("Runtime mode native confirmation failed: %s", exc, exc_info=True)
                return {"confirmed": False, "error": f"Native confirmation failed: {exc}"}

        def request_auto_grant_reviewed_skills_change(self, enabled: bool) -> dict:
            try:
                return _request_auto_grant_reviewed_skills_change(bool(enabled), self._native_confirm)
            except Exception as exc:
                log.warning("Reviewed-skill auto-grant confirmation failed: %s", exc, exc_info=True)
                return {"ok": False, "error": f"Native confirmation failed: {exc}"}

        def request_skill_key_grant_with_job(self, skill: str, keys: list, enable_after_grant: bool = False) -> dict:
            return self.request_skill_key_grant(skill, keys, enable_after_grant, True)

        def request_skill_key_grant(self, skill: str, keys: list, enable_after_grant: bool = False,
                                   allow_pending: bool = False) -> dict:
            try:
                return _request_skill_key_grant(skill, keys, self._native_confirm,
                                               enable_after_grant=enable_after_grant, allow_pending=allow_pending)
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
            return _open_external_url(url)

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
        background_color="#0f0f11",
        text_select=True, hidden=background.start_hidden(options.launch_intent),
    )
    _webview_window = attach_media_permissions(background.attach(window), url)  # Persist cookies and website data (ouroboros.theme); rebuild/limits: ARCHITECTURE §3.
    webview.start(func=background.run, debug=False, private_mode=False,
                  menu=setup_recovery_menu(window, lambda: _generation_port))


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()

    if sys.platform == "darwin":
        try:
            shell_path = subprocess.check_output(
                ["/bin/bash", "-l", "-c", "echo $PATH"], **merge_hidden_kwargs({"text": True, "timeout": 5}),
            ).strip()
            if shell_path:
                os.environ["PATH"] = shell_path
        except Exception:
            pass

    main(sys.argv[1:])
