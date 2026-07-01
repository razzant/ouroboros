"""Process-control helpers for the self-editable server entrypoint."""

from __future__ import annotations

import os
import json
import pathlib
import sys
from typing import Any


def restart_current_process(host: str, port: int, *, repo_dir: pathlib.Path, log: Any) -> None:
    env = os.environ.copy()
    desired_host = str(host)
    try:
        from ouroboros.config import load_settings
        desired_host = (
            str(os.environ.get("OUROBOROS_SERVER_HOST") or "").strip()
            or str(load_settings().get("OUROBOROS_SERVER_HOST") or "").strip()
            or desired_host
        )
    except Exception:
        desired_host = str(host)
    env["OUROBOROS_SERVER_HOST"] = desired_host
    env["OUROBOROS_SERVER_PORT"] = str(port)
    env.pop("OUROBOROS_MANAGED_BY_LAUNCHER", None)
    raw_argv = sys.argv
    try:
        saved = json.loads(os.environ.get("OUROBOROS_SERVER_REEXEC_ARGV_JSON", "") or "[]")
        if isinstance(saved, list) and saved and all(isinstance(item, str) and item for item in saved):
            raw_argv = saved
    except Exception:
        raw_argv = sys.argv
    argv = [sys.executable, *raw_argv]
    log.info("Re-executing direct server mode on %s:%d", desired_host, port)
    try:
        os.execvpe(sys.executable, argv, env)
    except Exception:
        log.exception("Direct re-exec failed; attempting spawned restart fallback.")
        try:
            from ouroboros.config import DATA_DIR
            from ouroboros.process_custody import spawn_supervised

            spawn_supervised(
                argv,
                drive_root=pathlib.Path(DATA_DIR),
                # daemon, NOT session: the replacement IS the next server
                # generation. A session-scoped entry carries this dying
                # generation's session id, so the new server's startup reap
                # would see it as a foreign-session process and SIGKILL itself.
                # daemon scope is always a reaper survivor (launcher-managed
                # lifecycle), which is correct for a long-lived top-level server.
                purpose="server_restart_fallback",
                scope="daemon",
                cwd=str(repo_dir),
                env=env,
            )
            log.info("Spawned replacement server process after exec failure.")
        except Exception:
            log.exception("Spawned restart fallback failed; exiting with restart code only.")


def execute_panic_stop(
    consciousness: Any,
    kill_workers_fn,
    *,
    data_dir: pathlib.Path,
    panic_exit_code: int,
    log: Any,
) -> None:
    """Full emergency stop: kill everything, write panic flag, hard-exit."""
    log.critical("PANIC STOP initiated.")
    try:
        consciousness.stop()
    except Exception:
        pass

    try:
        from supervisor.state import load_state, save_state

        st = load_state()
        st["evolution_mode_enabled"] = False
        st["bg_consciousness_enabled"] = False
        # Panic is an owner stop: make it authoritative against the post-task pipeline too,
        # so evolution cannot autonomously re-arm on the next boot (mirrors /evolve off).
        st["evolution_owner_stopped"] = True
        st["post_task_autostop"] = False
        save_state(st)
    except Exception:
        pass

    # Terminal-close the campaign + drop any queued promotion. Each in its own guard so a
    # missing/locked file never blocks the panic hard-exit (the flag above is the durable gate).
    # cleanup_worktree=False: the Emergency Stop Invariant (BIBLE) forbids delaying panic, so
    # panic must NOT run the mid-cycle git stash/reset cleanup — the panic flag + boot reconcile
    # own that recovery. (Graceful /evolve off + toggle do run the cleanup, after cancelling.)
    try:
        from supervisor.evolution_lifecycle import complete_evolution_campaign

        complete_evolution_campaign("panic stop", status="stopped", cleanup_worktree=False)
    except Exception:
        pass
    try:
        from ouroboros.post_task_evolution import drop_pending_request

        drop_pending_request(data_dir)
    except Exception:
        pass

    try:
        panic_flag = data_dir / "state" / "panic_stop.flag"
        panic_flag.parent.mkdir(parents=True, exist_ok=True)
        panic_flag.write_text("panic", encoding="utf-8")
    except Exception:
        pass

    try:
        from ouroboros.local_model import get_manager

        get_manager().stop_server()
    except Exception:
        pass

    try:
        from ouroboros.tools.shell import kill_all_tracked_subprocesses

        kill_all_tracked_subprocesses()
    except Exception:
        pass

    try:
        from ouroboros.workspace_executor import kill_all_foreground

        kill_all_foreground(data_dir, wait=False)
    except Exception:
        pass

    try:
        from ouroboros.tools.services import kill_all_services

        kill_all_services(data_dir, wait=False)
    except Exception:
        pass

    try:
        from ouroboros.extension_companion import panic_kill_all

        panic_kill_all()
    except Exception:
        pass

    try:
        kill_workers_fn(force=True, archive_service_logs=False)
    except Exception:
        pass

    try:
        import multiprocessing
        from ouroboros.gateway.host_service import host_service_port
        from ouroboros.platform_layer import force_kill_pid, kill_process_on_port

        for child in multiprocessing.active_children():
            try:
                force_kill_pid(child.pid)
            except (ProcessLookupError, PermissionError):
                pass
        # Sweep the actually bound main port (not hardcoded 8765/8766 — a
        # custom-port install would panic-kill an unrelated listener).
        try:
            import server as _server_mod

            kill_process_on_port(_server_mod._actual_bound_port())
        except Exception:
            kill_process_on_port(8765)
        kill_process_on_port(host_service_port())
    except Exception:
        pass

    log.critical("PANIC STOP complete — hard exit with code %d.", panic_exit_code)
    os._exit(panic_exit_code)
