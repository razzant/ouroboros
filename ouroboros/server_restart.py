"""The restart transaction, from request to exit signal.

A restart with live tasks is recorded and drained across supervisor loop ticks
rather than slept on; the tail re-checks the evolution restart receipt,
serializes the checkout against a managed update, tears the workers down with an
honest terminal reason, and raises the exit signal. The shutdown teardown
arguments live here because the same honest reason serves the lifespan path.
"""

import pathlib
import subprocess
import time
import uuid
from typing import Any, Dict

from ouroboros.server_process import log, _request_restart_exit
from ouroboros.utils import read_json_dict


# Deferred restart-drain state (multi-project, v6.32.0). The drain MUST NOT
# sleep on the supervisor loop thread (it is the only thread that processes
# heartbeats / task_done and shrinks RUNNING). Instead a restart with live
# tasks is recorded here and re-checked every loop tick, so events keep
# flowing and the drain actually observes tasks finishing.
_pending_restart: Dict[str, Any] = {}


def _live_running_task_ids(ctx: Any) -> list:
    """RUNNING task ids with a fresh heartbeat — structured facts only.

    Heartbeat staleness belongs to the generic supervisor queue, not to the
    planning-scout wait policy.  The latter intentionally waits until terminal
    state or its shared cutoff even when a scout heartbeat is stale.
    """
    from supervisor.queue import HEARTBEAT_STALE_SEC

    now = time.time()
    live = []
    for tid, meta in dict(ctx.RUNNING or {}).items():
        if not isinstance(meta, dict):
            continue
        try:
            hb = float(meta.get("last_heartbeat_at") or 0.0)
        except (TypeError, ValueError):
            hb = 0.0
        if hb and (now - hb) < HEARTBEAT_STALE_SEC:
            live.append(str(tid))
    return live


def _handle_restart_in_supervisor(evt: Dict[str, Any], ctx: Any) -> None:
    """Handle agent restart request: drain live tasks across loop ticks, then
    graceful shutdown + exit(42). Never sleeps on the dispatch thread."""
    st = ctx.load_state()
    if st.get("owner_chat_id"):
        ctx.send_with_budget(
            int(st["owner_chat_id"]),
            f"♻️ Restart requested by agent: {evt.get('reason')}",
        )
    from ouroboros.config import get_restart_drain_max_sec

    max_wait = get_restart_drain_max_sec()
    live = _live_running_task_ids(ctx) if max_wait > 0 else []
    if live:
        # Defer: re-checked each tick by _check_pending_restart_drain so the
        # loop keeps draining events (heartbeats advance, RUNNING shrinks).
        _pending_restart.clear()
        _pending_restart.update({
            "reason": str(evt.get("reason") or "agent_restart_request"),
            "deadline": time.time() + min(max_wait, 1800),
            "evolution_restart": bool(evt.get("evolution_restart")),
        })
        if st.get("owner_chat_id"):
            ctx.send_with_budget(
                int(st["owner_chat_id"]),
                f"⏳ Restart drain: waiting up to {max_wait}s for running task(s) "
                f"{', '.join(sorted(live))} to finish.",
            )
        return
    _perform_supervisor_restart(
        ctx, restart_reason=str(evt.get("reason") or "agent_restart_request"),
        evolution_restart=bool(evt.get("evolution_restart")),
    )


def _check_pending_restart_drain(ctx: Any) -> bool:
    """Loop-tick hook: complete a deferred restart once tasks drain or the
    deadline passes (proceeds fail-closed). Returns True while STILL draining, so
    the loop can skip starting new work that the restart would immediately chop."""
    if not _pending_restart:
        return False
    live = _live_running_task_ids(ctx)
    if live and time.time() < float(_pending_restart.get("deadline") or 0.0):
        return True  # keep draining — events still flow each tick
    pending = dict(_pending_restart)
    _pending_restart.clear()
    _perform_supervisor_restart(
        ctx, restart_reason=str(pending.get("reason") or "agent_restart_request"),
        evolution_restart=bool(pending.get("evolution_restart")),
    )
    # Still "quiescing" this tick: _perform_supervisor_restart sets up the exit
    # (or fail-closed pauses) and returns to the loop — the process exits on the
    # next `while not _restart_requested` check. Returning True keeps the caller
    # from starting new enqueue/assign work on this final pre-exit tick.
    return True


def _perform_supervisor_restart(
    ctx: Any, *, restart_reason: str = "agent_restart_request",
    evolution_restart: bool = False,
) -> None:
    """Graceful shutdown + exit(42) (the post-drain tail; never sleeps)."""
    st = ctx.load_state()
    marker = read_json_dict(
        pathlib.Path(ctx.DRIVE_ROOT) / "state" / "pending_restart_verify.json"
    ) or {}
    claim = (
        marker.get("evolution_claim")
        if evolution_restart and marker.get("reason") == restart_reason
        else {}
    )
    claim = claim if isinstance(claim, dict) else {}
    if evolution_restart and not claim:
        if st.get("owner_chat_id"):
            ctx.send_with_budget(
                int(st["owner_chat_id"]),
                "🧬 Restart cancelled: the exact evolution restart receipt is missing.",
            )
        return
    if claim:
        from supervisor.evolution_lifecycle import check_evolution_authority

        authority = check_evolution_authority(
            str(claim.get("campaign_id") or ""),
            str(claim.get("transaction_id") or ""),
            str(claim.get("task_id") or ""),
            commit_sha=str(claim.get("commit_sha") or ""),
        )
        if not authority.get("ok"):
            if st.get("owner_chat_id"):
                ctx.send_with_budget(
                    int(st["owner_chat_id"]),
                    "🧬 Restart cancelled: evolution authority changed "
                    f"({authority.get('reason') or 'unknown'}).",
                )
            return
        expected_sha = str(claim.get("commit_sha") or "")
        try:
            head_proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(ctx.REPO_DIR),
                check=False,
                capture_output=True,
                text=True,
            )
            status_proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(ctx.REPO_DIR),
                check=False,
                capture_output=True,
                text=True,
            )
            head = head_proc.stdout.strip() if head_proc.returncode == 0 else ""
            clean = status_proc.returncode == 0 and not status_proc.stdout.strip()
        except Exception:
            head = ""
            clean = False
        if not expected_sha or head != expected_sha or not clean:
            if st.get("owner_chat_id"):
                ctx.send_with_budget(
                    int(st["owner_chat_id"]),
                    "🧬 Restart cancelled: the live checkout no longer matches "
                    "the exact reviewed evolution commit.",
                )
            return
    ok, msg = _safe_restart_serialized(
        ctx.safe_restart,
        reason="agent_restart_request",
        unsynced_policy="rescue_and_block",
    )
    if not ok:
        try:
            from supervisor.evolution_lifecycle import pause_evolution_campaign

            st["evolution_mode_enabled"] = False
            ctx.save_state(st)
            pause_evolution_campaign(f"agent restart blocked to protect local changes: {msg}")
        except Exception:
            log.debug("Failed to pause evolution after blocked agent restart", exc_info=True)
        if st.get("owner_chat_id"):
            ctx.send_with_budget(int(st["owner_chat_id"]), f"⚠️ Restart skipped: {msg}")
        return
    cleanup_status, cleanup_reason = _shutdown_task_cleanup_args(restart_requested=True)
    ctx.kill_workers(
        force=True,
        terminal_status=cleanup_status,
        result_reason=cleanup_reason,
        **_managed_update_pending_kwargs(),
    )
    st2 = ctx.load_state()
    st2["session_id"] = uuid.uuid4().hex
    ctx.save_state(st2)
    ctx.persist_queue_snapshot(reason="pre_restart_exit")
    _request_restart_exit()


def _managed_update_pending_kwargs() -> dict:
    """Preserve queued work while a durable tx or its pre-tx quiesce owns restart."""
    try:
        from supervisor.update_merge import active_update_tx

        if active_update_tx():
            return {"preserve_pending": True}
        from supervisor.workers import repo_writer_admission_closed, worker_pool_admission_state

        gate = repo_writer_admission_closed()
        disabled = str(worker_pool_admission_state().get("disabled_reason") or "")
        if gate.startswith("managed_update:") or disabled == "managed_update":
            return {"preserve_pending": True}
        return {}
    except Exception:
        return {"preserve_pending": True}


def _safe_restart_serialized(safe_restart_fn, *, reason: str, unsynced_policy: str):
    """Serialize checkout/reset with update apply; only a landed update may restart."""
    from supervisor import git_ops
    from supervisor.update_merge import (
        acquire_update_lock,
        read_update_tx_strict,
        release_update_lock,
    )

    try:
        lock_fh = acquire_update_lock()
    except RuntimeError:
        return False, "Managed update is changing the checkout; restart was deferred."
    try:
        status, tx = read_update_tx_strict()
        if status == "corrupt":
            return False, "Managed update state is unreadable; restart was deferred."
        if status == "absent" and not git_ops._clear_update_intent():
            return False, (
                "An update intent marker with no update transaction could not be removed; "
                "restart was deferred rather than applying an orphaned update."
            )
        allowed_phases = {"pending_boot_smoke", "applying_replace"}
        if status == "valid" and str(tx.get("phase") or "") not in allowed_phases:
            return False, "Managed update merge is still being resolved; restart was deferred."
        return safe_restart_fn(reason=reason, unsynced_policy=unsynced_policy)
    finally:
        release_update_lock(lock_fh)


def _shutdown_task_cleanup_args(restart_requested: bool) -> tuple[str, str]:
    """Return ``(terminal_status, result_reason)`` for tasks torn down by a
    graceful server shutdown.

    A graceful shutdown — a requested restart (exit 42) or an external
    stop/restart signal (SIGTERM/SIGINT) — is not a worker crash storm, so a
    still-running task is finalized as ``cancelled`` with an honest reason
    instead of the default crash-storm text the supervisor uses for real
    worker deaths.
    """
    if restart_requested:
        reason = (
            "Server restarted before this task finished; the task was "
            "interrupted by the restart, not a worker crash."
        )
    else:
        reason = (
            "Server shut down (external stop/restart signal) before this task "
            "finished; the task was interrupted, not a worker crash."
        )
    return "cancelled", reason


def _shutdown_supervisor_event_bus() -> None:
    try:
        from supervisor.workers import shutdown_event_q

        shutdown_event_q()
    except Exception:
        pass
