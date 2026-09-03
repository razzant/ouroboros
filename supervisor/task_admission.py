"""Atomic admission transitions for the managed task queue."""

from __future__ import annotations

import logging
import pathlib
import uuid
from typing import Any, Dict, Optional

from ouroboros.depth_evidence import parse_task_depth
from ouroboros.task_results import (
    STATUS_FAILED,
    STATUS_REQUESTED,
    STATUS_SCHEDULED,
    load_task_result,
    write_task_result,
)
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)


def coerce_queue_order(value: Any, default: int = 0) -> int:
    """Coerce persisted queue ordering metadata without exposing parse failures."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def record_scheduled_admission(
    task: Dict[str, Any], admitted: Any, record: Dict[str, Any],
) -> None:
    """Project a cron dispatch refusal into terminal task/schedule state."""
    from supervisor import queue as q

    block = (
        str(admitted.get("_admission_blocked") or "")
        if isinstance(admitted, dict)
        else ""
    )
    if not block:
        record["failure_count"] = int(record.get("failure_count") or 0)
        record["last_error"] = ""
        return
    detail = f"Scheduled task was not queued: {block}."
    try:
        from ouroboros.task_results import STATUS_FAILED, write_task_result

        write_task_result(
            q.DRIVE_ROOT,
            str(task["id"]),
            STATUS_FAILED,
            result=detail,
            reason_code=block,
            cost_usd=0.0,
        )
    except Exception:
        q.log.warning(
            "Failed to terminalize admission-blocked scheduled task %s",
            task.get("id"),
            exc_info=True,
        )
    record["failure_count"] = int(record.get("failure_count") or 0) + 1
    record["last_error"] = detail


def prefer_terminalization_retry_rows(tasks: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Drop ordinary duplicate rows when a snapshot carries shutdown custody."""
    marker_ids = {
        str(task.get("id") or "").strip()
        for task in tasks
        if isinstance(task.get("_terminalization_retry"), dict)
        and str(task.get("id") or "").strip()
    }
    if not marker_ids:
        return tasks
    seen_markers: set[str] = set()
    preferred: list[Dict[str, Any]] = []
    for task in tasks:
        task_id = str(task.get("id") or "").strip()
        marker = isinstance(task.get("_terminalization_retry"), dict)
        if task_id in marker_ids and not marker:
            continue
        if marker and task_id in seen_markers:
            continue
        if marker:
            seen_markers.add(task_id)
        preferred.append(task)
    return preferred


def restore_terminalization_retry(
    task: Dict[str, Any], *, pending: list[Dict[str, Any]],
    running: Dict[str, Any], queue_seq_counter_ref: Dict[str, Any],
    sort_pending: Any,
) -> Optional[Dict[str, Any]]:
    """Restore a shutdown-custody row before ordinary admission gates."""
    if not isinstance(task.get("_terminalization_retry"), dict):
        return None
    restored = dict(task)
    task_id = str(restored.get("id") or "").strip()
    if task_id and (
        task_id in running
        or any(isinstance(row, dict) and str(row.get("id") or "") == task_id for row in pending)
    ):
        restored["_admission_blocked"] = "duplicate_task_id"
        return restored
    try:
        queue_seq_counter_ref["value"] = int(queue_seq_counter_ref.get("value", 0) or 0) + 1
    except (TypeError, ValueError, OverflowError):
        queue_seq_counter_ref["value"] = 1
    restored["priority"] = coerce_queue_order(restored.get("priority"))
    try:
        restored["_attempt"] = max(1, int(restored.get("_attempt") or 1))
    except (TypeError, ValueError, OverflowError):
        restored["_attempt"] = 1
    restored["_queue_seq"] = queue_seq_counter_ref["value"]
    restored.setdefault("queued_at", utc_now_iso())
    pending.append(restored)
    sort_pending()
    return restored


def restore_terminalization_retry_rows(
    tasks: list[Dict[str, Any]], *, pending: list[Dict[str, Any]],
    running: Dict[str, Any], queue_seq_counter_ref: Dict[str, Any], sort_pending: Any,
) -> tuple[list[Dict[str, Any]], dict[str, Dict[str, Any]], int]:
    """Restore marker rows and return ordinary rows plus their lineage map."""
    from supervisor import queue

    preferred = prefer_terminalization_retry_rows(tasks)
    pending_by_id = {
        str(task.get("id") or ""): task for task in preferred if str(task.get("id") or "")
    }
    ordinary: list[Dict[str, Any]] = []
    restored = 0
    with queue._queue_lock:
        for task in preferred:
            if not isinstance(task.get("_terminalization_retry"), dict):
                ordinary.append(task)
                continue
            row = restore_terminalization_retry(
                task, pending=pending, running=running,
                queue_seq_counter_ref=queue_seq_counter_ref, sort_pending=sort_pending,
            )
            if row is not None and not row.get("_admission_blocked"):
                restored += 1
    return ordinary, pending_by_id, restored


def parse_schedule_task_depth(
    ctx: Any,
    evt: Dict[str, Any],
    *,
    tid: str,
    chat_id: int,
    delegation_role: str,
    parent_id: Any,
    root_task_id: str,
    role: str,
    desc: str,
    expected_output: str,
    constraints: str,
    task_context: str,
) -> tuple[int, bool]:
    """Parse depth after an atomic final freshness check.

    The initial replay probe runs before expensive admission work, but a
    concurrent reservation or queue assignment can arrive before parsing.  The
    final check below shares the queue lock with those writers and keeps the
    malformed-result write under that lock, so an invalid replay cannot steal
    custody between the probe and rejection.
    """
    from supervisor import queue

    with queue._queue_lock:
        if tid and subagent_schedule_preflight(
            ctx, evt, chat_id, delegation_role=delegation_role,
        ):
            return 0, True
        try:
            return parse_task_depth(evt.get("depth", 0), default=0), False
        except (TypeError, ValueError) as exc:
            from supervisor.events import _reject_schedule_task

            _reject_schedule_task(
                ctx,
                tid=tid,
                chat_id=chat_id,
                delegation_role=delegation_role,
                parent_id=parent_id,
                root_task_id=root_task_id,
                role=role,
                result_fields={
                    "parent_task_id": parent_id,
                    "root_task_id": root_task_id,
                    "session_id": str(evt.get("session_id") or ""),
                    "actor_id": str(evt.get("actor_id") or "ouroboros"),
                    "delegation_role": delegation_role,
                    "role": role,
                    "description": desc,
                    "objective": desc,
                    "expected_output": expected_output,
                    "constraints": constraints,
                    "context": task_context,
                    "chat_id": chat_id,
                    "depth": 0,
                    "raw_task_depth": evt.get("depth"),
                    "invalid_task_depth": True,
                },
                detail=f"{'Subagent' if delegation_role == 'subagent' else 'Task'} rejected: invalid task depth: {exc}",
                reason_code="invalid_task_depth",
                fallback_message="⚠️ Task rejected: depth must be a non-negative integer.",
            )
            return 0, True


def reject_invalid_task_depth(
    task: Dict[str, Any], *, reservations: Dict[str, str], admission_token: str,
) -> bool:
    """Normalize an admitted task depth, or mark and release an invalid request."""
    try:
        task["depth"] = parse_task_depth(task.get("depth"), default=0)
    except (TypeError, ValueError) as exc:
        task_id = str(task.get("id") or "").strip()
        if reservations.get(task_id) == admission_token:
            reservations.pop(task_id, None)
        task["_admission_blocked"] = "invalid_task_depth"
        task["_admission_detail"] = f"Task was not queued: {exc}."
        return True
    return False


def terminalize_invalid_depth_restore(
    task: Dict[str, Any], detail: str, *, drive_root: pathlib.Path,
) -> bool:
    """Give a malformed snapshot row terminal custody outside the queue module."""
    task_id = str(task.get("id") or "").strip()
    if not task_id:
        return False
    raw_depth = task.get("depth")
    if raw_depth is not None and not isinstance(raw_depth, (str, int, float, bool)):
        raw_depth = repr(raw_depth)[:200]
    try:
        stored = write_task_result(
            pathlib.Path(task.get("budget_drive_root") or drive_root),
            task_id,
            STATUS_FAILED,
            strict_existing_dict=True,
            reason_code="invalid_task_depth",
            result=detail,
            depth=0,
            raw_task_depth=raw_depth,
            invalid_task_depth=True,
            parent_task_id=task.get("parent_task_id"),
            root_task_id=task.get("root_task_id"),
            delegation_role=task.get("delegation_role"),
            metadata=task.get("metadata") if isinstance(task.get("metadata"), dict) else {},
        )
    except Exception:
        log.warning("Failed to terminalize invalid-depth snapshot task %s", task_id, exc_info=True)
        return False
    return str((stored or {}).get("status") or "") == STATUS_FAILED


def restore_invalid_depth_admission(
    task: Dict[str, Any], admitted: Dict[str, Any], *, drive_root: pathlib.Path,
    pending: list[Dict[str, Any]], blocked: list[str], terminalized: list[str],
    queue_seq_counter_ref: Optional[Dict[str, Any]] = None,
) -> None:
    """Handle one blocked snapshot admission and retain invalid rows on failure.

    The queue caller must hold its queue lock while passing the live ``pending``
    list.  A failed terminal write remains retryable in memory and in the
    unchanged snapshot, but must not race queue assignment while it is restored.
    """
    task_id = str(task.get("id") or "")
    blocked.append(task_id)
    if str(admitted.get("_admission_blocked") or "") != "invalid_task_depth":
        return
    detail = str(
        admitted.get("_admission_detail")
        or "Task was not restored: depth must be a non-negative integer."
    )
    if terminalize_invalid_depth_restore(task, detail, drive_root=drive_root):
        terminalized.append(task_id)
        return
    if task_id and not any(
        isinstance(row, dict) and str(row.get("id") or "") == task_id
        for row in pending
    ):
        # Keep the row in live custody; the unchanged snapshot remains a retry
        # point if this process exits before the next assignment pass.  The
        # queue owns the lock and ordering around this mutation.  Normalize only
        # queue-order fields on the retry copy so malformed snapshot metadata
        # cannot poison a later enqueue; the rejected depth evidence is intact.
        pending_task = dict(task)
        for field in ("priority", "_queue_seq"):
            raw_value = pending_task.get(field)
            if raw_value is None:
                continue
            try:
                pending_task[field] = int(raw_value)
            except (TypeError, ValueError, OverflowError):
                pending_task.pop(field, None)
        if queue_seq_counter_ref is not None:
            # Snapshot sequence lives on the outer row, while this helper receives
            # only the nested task. Allocate a fresh sequence in encounter order
            # so a retained malformed row cannot sort ahead of restored rows.
            try:
                current = int(queue_seq_counter_ref.get("value", 0) or 0)
            except (TypeError, ValueError, OverflowError):
                current = 0
            highest_seen = current
            for row in pending:
                try:
                    highest_seen = max(highest_seen, abs(int(row.get("_queue_seq"))))
                except (AttributeError, TypeError, ValueError, OverflowError):
                    continue
            sequence = highest_seen + 1
            queue_seq_counter_ref["value"] = sequence
            pending_task["_queue_seq"] = sequence
        pending.append(pending_task)


def scheduled_admission_rejection(
    admitted: Dict[str, Any], *, project_id: str, root_task_id: str,
) -> Dict[str, Any]:
    """Map a queue admission fence to the canonical durable rejection shape."""
    reason = str(admitted.get("_admission_blocked") or "admission_fence")
    if reason == "task_id_lookup_failed":
        detail = (
            "Task not scheduled: the exact task-result authority became unreadable "
            "during admission and was preserved."
        )
        extra = {}
    elif reason == "duplicate_task_id":
        detail = (
            "Task not scheduled: this exact task id already has queue or durable "
            "lifecycle custody; the existing authority was preserved."
        )
        extra = {}
    elif reason.startswith("project_routing_fence"):
        lifecycle = str(admitted.get("_project_lifecycle") or "unavailable")
        detail = (
            "Subagent not scheduled: the target Project has closed its "
            f"routing/admission fence ({lifecycle}) and cannot accept new work."
        )
        extra = {
            "project_id": str(admitted.get("_project_id") or project_id),
            "project_lifecycle": lifecycle,
        }
    elif reason == "root_cancelled":
        detail = (
            "Subagent not scheduled: its root's subtree cancellation has begun, "
            "so the tree accepts no new work."
        )
        extra = {"root_task_id": str(root_task_id or "")}
    elif reason == "root_budget_fence":
        detail = (
            "Subagent not scheduled: the root budget is paused and requires an "
            "explicit replay-safe resume, cancellation, or a new run."
        )
        extra = {
            "root_task_id": str(admitted.get("_budget_root_task_id") or root_task_id),
            "budget_fence_id": str(admitted.get("_budget_fence_id") or ""),
        }
    elif reason == "invalid_task_depth":
        detail = str(
            admitted.get("_admission_detail")
            or "Subagent not scheduled: task depth must be a non-negative integer."
        )
        extra = {}
    else:
        lifecycle = str(admitted.get("_acceptance_fence_status") or "active")
        detail = (
            "Subagent not scheduled: the root task is in its atomic task-acceptance "
            f"phase ({lifecycle}); admission is closed until an explicit revision round."
        )
        reason = "task_acceptance_fence"
        extra = {
            "acceptance_fence_token": str(admitted.get("_acceptance_fence_token") or ""),
            "acceptance_fence_status": lifecycle,
        }
    return {
        "detail": detail,
        "reason_code": reason,
        "extra_fields": extra,
        "persist_result": reason not in {"task_id_lookup_failed", "duplicate_task_id"},
    }


def subagent_schedule_owned(
    ctx: Any, task_id: str, *, pending_ref: Any = None,
) -> bool:
    """Return whether an exact child id already has queue/lifecycle custody."""
    from supervisor import queue

    tid = str(task_id or "")
    with queue._queue_lock:
        pending = pending_ref if isinstance(pending_ref, list) else getattr(
            ctx, "PENDING", queue.PENDING,
        )
        running = getattr(ctx, "RUNNING", queue.RUNNING)
        if queue.ADMISSION_RESERVATIONS.get(tid):
            return True
        status = str((load_task_result(
            ctx.DRIVE_ROOT, tid, strict=True,
        ) or {}).get("status") or "")
        return (
            tid in running
            or any(
                isinstance(row, dict) and str(row.get("id") or "") == tid
                for row in pending
            )
            or status not in {"", STATUS_REQUESTED}
        )


def subagent_schedule_preflight(
    ctx: Any,
    evt: Dict[str, Any],
    chat_id: int,
    *,
    delegation_role: str = "subagent",
) -> bool:
    """Stop an owned or unreadable exact task id before parsing or side effects.

    The historical name is kept for compatibility with subagent callers, but
    the exact-id replay fence applies to every schedule role.  A task without
    an explicit id is not idempotent at this boundary and is left to the
    normal fresh-id/queue path.
    """
    tid = str(evt.get("task_id") or "").strip()
    if not tid:
        return False
    try:
        return subagent_schedule_owned(ctx, tid)
    except (OSError, ValueError):
        from supervisor.events import _reject_schedule_task

        label = "Subagent" if delegation_role == "subagent" else "Task"
        _reject_schedule_task(
            ctx, tid=tid, chat_id=chat_id, delegation_role=delegation_role,
            parent_id=evt.get("parent_task_id"),
            root_task_id=str(evt.get("root_task_id") or evt.get("parent_task_id") or tid),
            role=str(evt.get("role") or "researcher"), result_fields={},
            detail=(
                f"{label} not scheduled: the existing durable result for this task id "
                "is unreadable, so its identity authority was preserved."
            ),
            reason_code="scheduled_result_authority_unknown", persist_result=False,
        )
        return True


def enqueue_subagent_with_scheduled_result(
    ctx: Any,
    task: Dict[str, Any],
    *,
    result_fields: Dict[str, Any],
    admitted_task_contract: Dict[str, Any],
    admitted_depth_provenance: Dict[str, Any],
    direct_child_count: Any,
    pending_ref: list[Any],
) -> tuple[Any, str, str, bool]:
    """Enqueue a child only together with its first durable authority row.

    Assignment takes the same queue RLock.  A pre-commit result failure can
    therefore remove this exact still-pending object before a worker observes
    it.  A late observer exception after atomic file replacement keeps the
    already-authoritative admission instead of compensating a committed row.
    """
    from supervisor import queue

    tid = str(task.get("id") or "")
    transition_id = uuid.uuid4().hex

    def _committed(record: Any) -> bool:
        admission = record.get("delegation_admission") if isinstance(record, dict) else None
        return bool(
            isinstance(record, dict)
            and str(record.get("status") or "") == STATUS_SCHEDULED
            and isinstance(admission, dict)
            and str(admission.get("status") or "") == "accepted"
            and str(admission.get("transition_id") or "") == transition_id
        )

    with queue._queue_lock:
        try:
            previous = load_task_result(ctx.DRIVE_ROOT, tid, strict=True) or {}
            already_owned = subagent_schedule_owned(
                ctx, tid, pending_ref=pending_ref,
            )
        except (OSError, ValueError):
            log.warning(
                "Subagent schedule authority is unreadable for %s", tid,
                exc_info=True,
            )
            return (
                task,
                "scheduled_result_authority_unknown",
                "Subagent not scheduled: the existing durable result for this "
                "task id is unreadable, so the host cannot prove that the id is "
                "fresh. The existing result was preserved.",
                False,
            )
        if already_owned:
            log.info("Ignoring replayed schedule event for task %s", tid)
            return (
                task,
                "scheduled_event_replay",
                "Subagent schedule replay ignored: this task id is already owned by "
                "an existing queue or durable lifecycle row.",
                False,
            )
        admitted = ctx.enqueue_task(task)
        if isinstance(admitted, dict) and admitted.get("_admission_blocked"):
            if admitted.get("_admission_blocked") == "task_id_lookup_failed":
                return (
                    task, "scheduled_result_authority_unknown",
                    "Subagent not scheduled: the exact task-result authority became "
                    "unreadable during admission and was preserved.", False,
                )
            return admitted, "", "", False
        result_fields["task_contract"] = admitted_task_contract
        result_fields["depth_provenance"] = admitted_depth_provenance
        result_fields["delegation_admission"] = {
            "status": "accepted",
            "direct_child_count": direct_child_count,
            "transition_id": transition_id,
        }
        try:
            stored = write_task_result(
                ctx.DRIVE_ROOT,
                tid,
                STATUS_SCHEDULED,
                **result_fields,
                result="Subagent accepted and scheduled.",
            )
        except Exception:
            committed = load_task_result(ctx.DRIVE_ROOT, tid) or {}
            if _committed(committed):
                log.warning(
                    "Scheduled subagent write for %s raised after its accepted "
                    "receipt committed; keeping admission",
                    tid,
                    exc_info=True,
                )
                return admitted, "", "", False
            log.warning(
                "Failed to persist scheduled subagent status for %s; rolled back "
                "its exact queue admission",
                tid,
                exc_info=True,
            )
        else:
            if _committed(stored):
                return admitted, "", "", False
            log.warning(
                "Scheduled subagent status for %s did not commit this admission; "
                "rolling back its exact queue row",
                tid,
            )
        for index, row in enumerate(pending_ref):
            if row is admitted:
                pending_ref.pop(index)
                break
        current = load_task_result(ctx.DRIVE_ROOT, tid) or {}
        prior_status = str(previous.get("status") or "")
        current_status = str(current.get("status") or "")
        if any(
            status not in {"", STATUS_REQUESTED}
            for status in (prior_status, current_status)
        ):
            return (
                admitted,
                "scheduled_result_conflict",
                "Subagent not scheduled: another durable result already owns this "
                "task id, so the new queue admission was rolled back without "
                "overwriting that result.",
                False,
            )
        return (
            admitted,
            "scheduled_result_persist_failed",
            "Subagent not scheduled: its durable scheduled-result receipt could not "
            "be persisted, so queue admission was rolled back.",
            True,
        )


def reserve_task_admission(
    task_id: str,
    admission_token: str,
    *,
    require_worker_pool: bool = False,
    drive_root: Any = None,
    worker_pool: Any = None,
) -> Dict[str, Any]:
    """Atomically reserve one fresh user-ingress id before side effects."""
    from supervisor import queue

    tid = str(task_id or "").strip()
    token = str(admission_token or "").strip()
    if not tid or not token:
        return {"status": "blocked", "reason": "invalid_admission_reservation"}
    with queue._queue_lock:
        reserved = queue.ADMISSION_RESERVATIONS.get(tid)
        if reserved:
            if reserved == token:
                return {"status": "already_reserved", "reason": ""}
            return {"status": "blocked", "reason": "duplicate_task_id"}
        if tid in queue.RUNNING or any(
            isinstance(row, dict) and str(row.get("id") or "") == tid
            for row in queue.PENDING
        ):
            return {"status": "blocked", "reason": "duplicate_task_id"}
        try:
            from ouroboros.task_results import load_task_result

            existing = load_task_result(
                pathlib.Path(drive_root or queue.DRIVE_ROOT), tid, strict=True,
            ) or {}
        except Exception:
            return {"status": "blocked", "reason": "task_id_lookup_failed"}
        if existing:
            admission = existing.get("promotion_admission")
            if (
                isinstance(admission, dict)
                and str(admission.get("routing_token") or "") == token
            ):
                return {
                    "status": "existing_same_token",
                    "reason": "",
                    "task_status": str(existing.get("status") or ""),
                    "promotion_admission": dict(admission),
                }
            return {"status": "blocked", "reason": "duplicate_task_id"}
        if require_worker_pool:
            try:
                from supervisor import workers

                disabled_reason = str(workers._WORKER_POOL_DISABLED_REASON or "")
                pool = workers.WORKERS if worker_pool is None else worker_pool
                worker_count = len(pool)
            except Exception:
                return {"status": "blocked", "reason": "worker_pool_state_unavailable"}
            if disabled_reason or worker_count <= 0:
                return {
                    "status": "blocked",
                    "reason": "worker_pool_unavailable",
                    "worker_pool_disabled_reason": disabled_reason or "no_workers",
                }
        queue.ADMISSION_RESERVATIONS[tid] = token
        return {"status": "reserved", "reason": ""}


def release_task_admission(task_id: str, admission_token: str) -> bool:
    """Release only the reservation owned by the supplied token."""
    from supervisor import queue

    tid = str(task_id or "").strip()
    token = str(admission_token or "").strip()
    with queue._queue_lock:
        if queue.ADMISSION_RESERVATIONS.get(tid) != token:
            return False
        queue.ADMISSION_RESERVATIONS.pop(tid, None)
        return True


__all__ = [
    "enqueue_subagent_with_scheduled_result",
    "record_scheduled_admission",
    "release_task_admission",
    "reserve_task_admission",
    "subagent_schedule_owned",
    "subagent_schedule_preflight",
]
