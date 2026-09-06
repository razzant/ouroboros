"""Restart-adjacent helpers the composition root calls at shutdown time.

The live-task census the restart drain consults, the teardown arguments that
finalize interrupted tasks with an honest reason, the managed-update guard on
preserving queued work, the checkout/update serialization gate, and the event
bus shutdown. The restart transaction itself — the deferred drain record and
the performer that raises the exit signal — stays in ``server.py`` for now:
the upstream delegation train coupled it to the composition root through the
planned-handoff transaction id (see docs/v7next/LEDGER_CORRECTIONS.md, D11).
"""

from __future__ import annotations

import time
from typing import Any

from ouroboros.server_process import DATA_DIR, _owner_restart_requested, _restart_requested


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


def _managed_update_pending_kwargs() -> dict:
    """Preserve queued work while a durable tx or its pre-tx quiesce owns restart."""
    try:
        from ouroboros.delegate_recovery import has_planned_restart_handoffs

        if (
            has_planned_restart_handoffs(DATA_DIR)
            and _restart_requested.is_set()
            and not _owner_restart_requested.is_set()
        ):
            return {"preserve_pending": True}
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
        if status == "future":
            return False, "Managed update state was recorded by a newer version; restart was deferred."
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
