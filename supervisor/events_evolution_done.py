"""Terminal handling of an evolution task and the campaign it belonged to.

Projects the reviewed cycle's outcome onto the campaign, closes a campaign the
owner stopped mid-flight, and notifies the owner of the cycle result.
"""

from __future__ import annotations

import logging
from typing import Any, Dict
from ouroboros.utils import utc_now_iso
from ouroboros.outcomes import normalize_outcome_axes

from supervisor.queue_transitions import (  # noqa: F401 -- re-exported for the events facade
    _close_campaign_after_owner_stop,
)

log = logging.getLogger(__name__)


def _handle_evolution_task_done(
    ctx: Any,
    *,
    evt: Dict[str, Any],
    task_id: Any,
    task: Dict[str, Any],
    task_done_event: Dict[str, Any],
    outcome_axes: Dict[str, Any],
    cost: Any,
    rounds: Any,
) -> None:
    """Project one evolution terminal through the existing campaign authority."""

    try:
        from supervisor.evolution_lifecycle import (
            _read_evolution_campaign,
            update_evolution_campaign_after_task,
        )

        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        if not metadata and isinstance(evt.get("metadata"), dict):
            metadata = evt.get("metadata") or {}
        transaction = (
            metadata.get("evolution_transaction")
            if isinstance(metadata.get("evolution_transaction"), dict)
            else {}
        )
        lifecycle_result = update_evolution_campaign_after_task(
            str(task_id or ""),
            cost_usd=cost,
            cost_accounting_status=str(
                task_done_event.get("cost_accounting_status") or "available"
            ),
            outcome_axes=outcome_axes,
            rounds=rounds,
            transaction=transaction,
        )
        if not isinstance(lifecycle_result, dict):
            log.warning("Evolution terminal rejected: invalid lifecycle result for %s", task_id)
            return
        if not lifecycle_result.get("accepted") or not lifecycle_result.get("persisted"):
            log.warning(
                "Evolution terminal rejected for %s: %s",
                task_id, lifecycle_result.get("reason") or "not_persisted",
            )
            return
        if lifecycle_result.get("replay"):
            return
        recorded_transaction = lifecycle_result.get("transaction")
        recorded_transaction = recorded_transaction if isinstance(recorded_transaction, dict) else {}
        try:
            from ouroboros.evolution_checkpoints import append_evolution_checkpoint

            append_evolution_checkpoint(
                ctx.DRIVE_ROOT,
                ctx.REPO_DIR,
                task_id=str(task_id or ""),
                campaign=_read_evolution_campaign(),
                outcome_axes=outcome_axes,
                cost_usd=cost,
                cost_accounting_status=str(
                    task_done_event.get("cost_accounting_status") or "available"
                ),
                rounds=rounds,
                transaction=recorded_transaction or transaction,
            )
        except Exception:
            log.debug("Failed to append evolution checkpoint", exc_info=True)
    except Exception:
        log.debug("Failed to update evolution campaign state", exc_info=True)
        return
    finally:
        # GR3-3: runs on EVERY evolution terminal — including rejected/replay
        # early returns above — so an owner stop that had to leave the campaign
        # open (still-live task) gets its deferred terminal close here.
        # GR4-6: the settling task is excluded from the liveness gate — its
        # RUNNING row is popped only later by _finish_task_done_dispatch.
        _close_campaign_after_owner_stop(exclude_task_id=str(task_id or ""))

    axes = normalize_outcome_axes({
        "status": task_done_event.get("status"),
        "outcome_axes": outcome_axes,
    })
    execution_status = str((axes.get("execution") or {}).get("status") or "").lower()
    objective_status = str((axes.get("objective") or {}).get("status") or "").lower()
    artifact_status = str((axes.get("artifacts") or {}).get("status") or "").lower()
    lifecycle_status = str(
        (axes.get("lifecycle") or {}).get("status")
        or task_done_event.get("status")
        or ""
    ).lower()
    failed_by_axes = (
        lifecycle_status in {"failed", "cancelled", "interrupted"}
        or execution_status in {"failed", "infra_failed", "degraded"}
        or objective_status in {"fail", "degraded"}
        or artifact_status in {"failed", "missing"}
    )
    if not failed_by_axes and (rounds or 0) >= 1:
        from supervisor.state import update_state

        update_state(lambda live: live.update(evolution_consecutive_failures=0))
    else:
        from supervisor.state import update_state

        failures_box: Dict[str, int] = {}

        def _bump_failures(live: Dict[str, Any]) -> None:
            failures_box["n"] = int(live.get("evolution_consecutive_failures") or 0) + 1
            live["evolution_consecutive_failures"] = failures_box["n"]

        update_state(_bump_failures)
        ctx.append_jsonl(
            ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "evolution_task_failure_tracked",
                "task_id": task_id,
                "consecutive_failures": failures_box.get("n", 0),
                "cost_usd": cost,
                "rounds": rounds,
            },
        )
    try:
        from supervisor.state import update_state

        def _consume_autostop(live: Dict[str, Any]) -> None:
            if live.get("post_task_autostop"):
                live["evolution_mode_enabled"] = False
                live["post_task_autostop"] = False

        update_state(_consume_autostop)
    except Exception:
        log.debug("Post-task evolution autostop failed", exc_info=True)
