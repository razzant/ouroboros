"""Process-control helpers for the self-editable server entrypoint."""

from __future__ import annotations

import os
import json
import pathlib
import sys
from typing import Any


def restart_current_process(
    host: str,
    port: int,
    *,
    repo_dir: pathlib.Path,
    log: Any,
    owner_initiated: bool = False,
) -> None:
    """Re-exec this server process.

    ``owner_initiated`` marks the restart the OWNER asked for (the chat Restart
    button, and the control endpoints that restart on the owner's behalf). Only
    that restart drops the inherited runtime-mode ratchet pin, so the child
    re-pins from ``load_settings()``; an agent- or supervisor-initiated restart
    keeps inheriting it exactly as before.
    """
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
    if owner_initiated:
        # The ratchet pin is exported so a CHILD inherits the parent's baseline
        # and cannot widen its own scope. Carried across an owner restart it also
        # pinned the mode the owner just raised in Settings and pressed Restart to
        # apply: the replacement re-pinned the OLD baseline from this env and the
        # new mode never took effect, on this restart or any later one. Dropping
        # the key here makes the child re-pin from load_settings() — the file only
        # the owner can author. Agent/supervisor restarts keep inheriting it.
        from ouroboros.config import BOOT_RUNTIME_MODE_ENV_KEY

        env.pop(BOOT_RUNTIME_MODE_ENV_KEY, None)
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
    bound_port: int | None = None,
) -> None:
    """Full emergency stop: kill everything, write panic flag, hard-exit.

    ``bound_port`` is the main port the server actually bound. The caller owns
    that fact and passes it in; this leaf does not reach back into the server
    module for it. Omitted (or falsy), the sweep falls back to the default
    install port — see the sweep below.

    Known limit (disclosed residual): an ATTACHED Claudexor daemon — one this
    process did not spawn — is left alive, because ``get_owned_daemon().stop()``
    only ever kills a self-started daemon's process group (delegated harness
    runs die with that group); cross-generation cleanup of a stale owned daemon
    belongs to the process-custody reaper at the next manual start.
    """
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

    # Owned Claudexor daemon: panic is instant and hard, so no network run-cancel
    # calls — stop() kills the self-spawned daemon's whole process group, taking
    # the delegated harness runs (its children) down with it. Attached daemons
    # are never killed (see docstring residual).
    try:
        from ouroboros.claudexor_daemon import get_owned_daemon

        get_owned_daemon().stop()
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
        kill_workers_fn(
            force=True, archive_service_logs=False, reconcile_delegate_custody=False,
        )
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
        # custom-port install would panic-kill an unrelated listener). The
        # default install port stays the last resort, for a caller that has no
        # bound port to give and for a sweep that fails.
        try:
            kill_process_on_port(bound_port or 8765)
        except Exception:
            kill_process_on_port(8765)
        kill_process_on_port(host_service_port())
    except Exception:
        pass

    log.critical("PANIC STOP complete — hard exit with code %d.", panic_exit_code)
    os._exit(panic_exit_code)
