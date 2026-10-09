"""Atomic admission transitions for the managed task queue."""

from __future__ import annotations

import logging
import pathlib
import uuid
from typing import Any, Callable, Dict, Optional

from ouroboros.depth_evidence import parse_task_depth
from ouroboros.task_results import (
    STATUS_CANCELLED,
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
    reason = str(admitted.get("_admission_blocked") or "")
    if reason == "project_routing_fence_lookup_failed":
        # Same-ID custody: assignment revalidates only the original evidence.
        # Preserve the exact invalid carrier and any pre-existing independent hold.
        if task_id and not any(row.get("id") == task_id for row in pending):
            # Keep the queue-order facts this refused admission already assigned.
            order = {key: admitted[key] for key in ("priority", "_queue_seq", "queued_at") if key in admitted}
            pending.append({**task, **order, "_project_admission_restore_hold": {
                "reason": reason, "detail": str(admitted.get("_admission_detail") or ""),
            }})
        return
    if reason in {"project_routing_fence", "project_routing_fence_changed"}:
        # An accepted snapshot row never disappears merely because a sibling Main
        # row restored. A semantic refusal owns a durable terminal or the existing
        # non-dispatchable terminalization retry. Unknown authority stays held.
        detail = str(admitted.get("_admission_detail") or "The Project no longer accepts this task.")
        try:
            stored = write_task_result(
                pathlib.Path(task.get("budget_drive_root") or drive_root), task_id,
                STATUS_FAILED, strict_existing_dict=True, reason_code=reason, result=detail,
                **{key: task.get(key) for key in ("project_id", "chat_id", "root_task_id",
                                                 "parent_task_id", "delegation_role", "metadata")})
            if isinstance(stored, dict) and stored.get("status") == STATUS_FAILED:
                return
        except Exception:
            log.warning("Project restore terminalization deferred for %s", task_id, exc_info=True)
        retained = {**task, "_terminalization_retry": {
            "status": STATUS_FAILED, "reason": detail, "trigger": reason,
            "reconcile_delegate_custody": False,
        }}
        from supervisor import queue

        restore_terminalization_retry(
            retained, pending=pending, running=queue.RUNNING,
            queue_seq_counter_ref=queue_seq_counter_ref or queue.QUEUE_SEQ_COUNTER_REF,
            sort_pending=queue.sort_pending)
        return
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


def restored_handoff_unproven(task: dict, existing: Optional[dict], *, ancestor_unknown: bool = False) -> bool:
    """Whether a restored scope-verifiable row's result cannot rule out a handoff.

    A Project row also needs its positive scheduled receipt. A host-verified
    unscoped row is vetoed only by positive handoff evidence, so a producer
    without such a receipt keeps its ordinary restore; other rows are unchanged.
    An unreadable ancestor is neither interruption nor permission: both wait.
    """
    existing = existing or {}
    possible = (bool(existing.get("started_at")) or existing.get("admitted_dispatch", "none") != "none"
                or ancestor_unknown)
    if task.get("project_id"):
        return possible or existing.get("status") != STATUS_SCHEDULED
    from ouroboros.project_admission import host_unscoped

    return host_unscoped(task) and possible


def enqueue_with_admission_receipt(task: Dict[str, Any], *, receipt_required: bool = True,
                                   prepare_admission: Optional[Callable[[dict], bool]] = None, **enqueue_kwargs: Any) -> Any:
    """Admit a host producer's row together with its positive scheduled receipt.

    Optional host preparation runs after queue validation and before the receipt;
    a refusal removes the still-unobserved row. It must never dispatch work.
    Append and receipt are ONE Q transaction, so no worker observes the row
    first; hold release later reads this receipt, never an absent result. A
    receipt proved missing rolls the unobserved row back as a typed refusal, or
    keeps it receipt-less when not ``receipt_required``. An unreadable readback
    retains custody with the receipt unconfirmed: that return proves no receipt,
    so hold release still reads it and assignment still persists dispatch first.
    A consciousness root's allowance is read before Q, as ``enqueue_task`` does.
    """
    from supervisor import queue

    if enqueue_kwargs.get("consciousness_window") is None and not enqueue_kwargs.get("restoring_snapshot"):
        window = queue.consciousness_admission_window(task)
        if window is not None:
            enqueue_kwargs["consciousness_window"] = window
    with queue.prepared_root_billing(task), queue._queue_lock:  # the ledger read happens before the lock
        admitted = queue.enqueue_task(task, **enqueue_kwargs)
        if not isinstance(admitted, dict) or admitted.get("_admission_blocked"):
            return admitted
        if prepare_admission is not None:
            try:
                prepared = prepare_admission(admitted)
            except Exception:
                log.warning("Host admission preparation failed; unobserved row removed", exc_info=True)
                prepared = False
            if not prepared:
                queue.PENDING[:] = [row for row in queue.PENDING if row is not admitted]
                return {**admitted, "_admission_blocked": "admission_preparation_failed",
                        "_admission_never_admitted": True}
        tid, token = str(admitted.get("id") or ""), uuid.uuid4().hex
        try:
            write_task_result(queue.DRIVE_ROOT, tid, STATUS_SCHEDULED, strict_existing_dict=True, create_only=True,
                              chat_id=admitted.get("chat_id"), root_task_id=admitted.get("root_task_id") or tid,
                              metadata=admitted.get("metadata") if isinstance(admitted.get("metadata"), dict) else {},
                              host_admission={"token": token, "status": "accepted"},
                              result="Task accepted and scheduled.")
        except Exception:
            log.warning("Host admission receipt write raised for %s; reading it back", tid, exc_info=True)
        try:
            stored = load_task_result(queue.DRIVE_ROOT, tid, strict=True) or {}
        except Exception:
            log.warning("Host admission receipt for %s is unreadable; custody retained", tid, exc_info=True)
            return admitted
        if (stored.get("status") == STATUS_SCHEDULED and (stored.get("host_admission") or {}).get("token") == token
                or not receipt_required):
            return admitted
        queue.PENDING[:] = [row for row in queue.PENDING if row is not admitted]
        return {**admitted, "_admission_blocked": "scheduled_result_persist_failed", "_admission_never_admitted": True,
                "_admission_detail": "its durable scheduled receipt could not be persisted"}


def record_project_dispatch_possible(task: dict) -> bool:
    """Persist possible handoff for scope-verifiable work beside admitted_dispatch.

    Assignment reads back 'possible' before handoff, so even an older PENDING
    snapshot cannot overrule it if the best-effort RUNNING mirror fails. This
    result fact never becomes 'none'; fresh admission's positive 'none' lives
    on the queue row. No early result may preempt the admission receipt owner.
    """
    if not task.get("project_id") and task.get("_project_scope_none") is not True:
        return True
    from supervisor import queue
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, STATUS_CANCEL_REQUESTED

    def project(current, _incoming):
        if (current.get("status") in _TRULY_TERMINAL_STATUSES | {STATUS_CANCEL_REQUESTED}
                or current.get("_owner_hold")):
            if current.get("_owner_hold"):
                task["_owner_hold"] = current["_owner_hold"]
            return None
        return {"admitted_dispatch": "possible", "status": current.get("status") or STATUS_REQUESTED}

    try:
        written = write_task_result(queue.DRIVE_ROOT, task["id"], STATUS_REQUESTED,
                                    _field_projector=project, strict_existing_dict=True)
        if written is False:
            return False
        stored = load_task_result(queue.DRIVE_ROOT, task["id"], strict=True) or {}
        return (stored.get("admitted_dispatch") == "possible"
                and stored.get("status") not in _TRULY_TERMINAL_STATUSES | {STATUS_CANCEL_REQUESTED}
                and not stored.get("_owner_hold"))
    except Exception:
        log.warning("Project dispatch evidence unavailable for %s", task.get("id"), exc_info=True)
        return False


def _exact_resume_granted(task: dict, drive_root: Any) -> bool:
    """Whether the owner's single-use exact Resume grant is this row's handoff authority.

    A mid-run pause was dispatched, so it never has first-dispatch 'none'. Its
    durable pause row must still name the SAME unconsumed, unrevoked grant the
    row carries; the grant continues that run and never authorizes a replay.
    An unreadable pause row raises, keeping the hold.
    """
    from ouroboros.budget_pause import STATE_RESUME_GRANTED
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, STATUS_CANCEL_REQUESTED

    handoff = task.get("_budget_pause_resume")
    if not isinstance(handoff, dict) or not str(handoff.get("grant_id") or ""):
        return False
    root = pathlib.Path(task.get("budget_drive_root") or drive_root)
    stored = load_task_result(root, str(task.get("id") or ""), strict=True) or {}
    pause = stored.get("budget_pause") if isinstance(stored.get("budget_pause"), dict) else {}
    grant = pause.get("grant") if isinstance(pause.get("grant"), dict) else {}
    return (pause.get("state") == STATE_RESUME_GRANTED
            and str(pause.get("pause_id") or "") == str(handoff.get("pause_id") or "")
            and grant.get("grant_id") == handoff["grant_id"]
            and not grant.get("revoked_at") and not grant.get("consumed_at")
            and stored.get("status") not in _TRULY_TERMINAL_STATUSES | {STATUS_CANCEL_REQUESTED})


def revalidate_project_holds() -> None:
    """Revalidate accepted work under Q; a hold never grants replay authority.

    Original identity/resource and positive no-dispatch receipts must agree,
    unless exact budget Resume or owner-wait/update authority continues the run.
    Unknown facts retain custody; interrupted parents and semantic refusals use
    existing settlement. Deadline, cancellation and independent holds still bind.
    """
    import os
    import time
    import copy
    from collections import Counter
    from contextlib import nullcontext

    from ouroboros.project_admission import (
        ProjectAdmissionError, _strict_admission_snapshot, host_unscoped,
        project_admission_guard, task_project_membership, validate_project_admission,
    )
    from supervisor import queue
    from supervisor.queue_snapshot import _descends_from, _interrupted_ancestors, _fence_snapshot_running_rows, _exact_pause_row
    from supervisor.schedule_occurrence import restore_allowed
    from supervisor.update_merge import assisted_resume_authorizes
    from ouroboros.projects_registry import _load_bindings, project_binding_for_task
    from ouroboros.owner_wait import restore_owner_wait_allowed
    from ouroboros.delegate_recovery import planned_handoff_resume_allowed

    held = [task for task in queue.PENDING if task.get("_project_admission_restore_hold")]
    if not held:
        return
    effects = ("_project_admission_restore_hold", "_terminalization_retry", "_owner_hold")
    before = [copy.deepcopy({key: task[key] for key in effects if key in task}) for task in held]
    counts = Counter(row.get("id") for row in queue.PENDING)
    # Restore's parent rule, re-read at release: exact continuations are saved work.
    children = [task for task in held if str(task.get("parent_task_id") or "")
                and not task.get("_terminalization_retry") and not task.get("_budget_pause_resume")
                and not _exact_pause_row(task)]
    pending_by_id = {str(row.get("id") or ""): row for row in queue.PENDING if row.get("id")}
    lineage_unknown: set = set()
    interrupted = _interrupted_ancestors([], children, unknown=lineage_unknown) if children else set()
    snapshot, read_error = None, None
    try:
        data, present = _strict_admission_snapshot(queue.DRIVE_ROOT, allow_missing=True, identity_only=True)
        snapshot = ({row["id"]: row for row in data["projects"]}, present)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        read_error = exc
    released = []
    bindings, bindings_error = None, None
    for task in held:
        if task.get("_terminalization_retry"):
            continue
        tid = str(task.get("id") or "")
        hold = task["_project_admission_restore_hold"]
        try:
            if tid in queue.RUNNING or tid in queue.ADMISSION_RESERVATIONS or counts[tid] != 1:
                raise ValueError("Another queue owner holds this task id; recovery remains closed.")
            deadline = queue._task_deadline_ts(task)
            if deadline and time.time() >= deadline:
                raise ProjectAdmissionError("deadline", "The task deadline elapsed while waiting for its Project.")
            owner_resume = restore_owner_wait_allowed(queue.DRIVE_ROOT, task, strict=True)
            if task.get("_owner_wait_resume") and not owner_resume:
                _fence_snapshot_running_rows([{"id": tid}], restored_ids=set())
                continue  # the same shutdown custody as an unresumable restore
            if any(task is child for child in children):
                if _descends_from(task, interrupted, pending_by_id):
                    task["_terminalization_retry"] = {
                        "status": STATUS_CANCELLED, "trigger": "pending_parent_interrupted",
                        "reason": "Parent task was interrupted before this child started.",
                        "reconcile_delegate_custody": True,
                    }
                    continue
                if _descends_from(task, lineage_unknown, pending_by_id):
                    raise ValueError("The parent task's state is unreadable; automatic recovery is not authorized.")
            # Status and missing started_at alone cannot prove no handoff: the
            # old RUNNING mirror was best-effort. Fresh queue admission supplies
            # positive 'none'; canonical possible handoff always vetoes that row.
            stored = load_task_result(queue.DRIVE_ROOT, tid, strict=True) or {}
            granted = (owner_resume or _exact_resume_granted(task, queue.DRIVE_ROOT)
                       or assisted_resume_authorizes(task, stored)
                       or planned_handoff_resume_allowed(queue.DRIVE_ROOT, task))
            if (stored.get("admission_outcome") == "never_admitted"
                    or not granted and (
                        stored.get("status") != STATUS_SCHEDULED or stored.get("started_at")
                        or task.get("admitted_dispatch") != "none"
                        or "admitted_dispatch" in stored and stored["admitted_dispatch"] != "none")):
                raise ProjectAdmissionError("project_dispatch_unconfirmed",
                    "The original task's no-dispatch evidence is unconfirmed; automatic recovery is not authorized.")
            owner_hold = task.get("_owner_hold")
            # The schedule receipt proves no first dispatch; a grant continues one.
            schedule_allowed = granted or restore_allowed(task)
            if owner_hold:
                task["_owner_hold"] = owner_hold
            if not schedule_allowed:
                raise ValueError("The schedule's original no-dispatch receipt is unconfirmed.")
            if stored.get("_owner_hold"):
                task["_owner_hold"] = stored["_owner_hold"]
            fence = queue.ACCEPTANCE_FENCES.get(str(task.get("root_task_id") or ""))
            if isinstance(fence, dict) and fence.get("status") in {"active", "sealed"}:
                raise ValueError("The task's root is in acceptance review; dispatch remains closed.")
            basis = task.get("_project_admission")
            unscoped = host_unscoped(task)
            if not unscoped and (not isinstance(basis, dict) or basis.get("legacy_basis")
                    or isinstance(basis.get("project"), dict) and not {
                        "id", "chat_id", "lifecycle", "routing_generation", "working_dir",
                        "routing_incarnation", "created_at"} <= basis["project"].keys()):
                raise ValueError("The original Project identity is missing or invalid; automatic recovery is not authorized.")
            if not unscoped:
                validate_project_admission(basis)
                if basis["project_id"] != str(task.get("project_id") or ""):
                    raise ProjectAdmissionError("project_routing_fence_changed", "The original Project scope changed.")
            if (unscoped and "_project_admission" in stored
                    or unscoped and str(stored.get("workspace_root") or "") != str(task.get("workspace_root") or "")
                    or stored.get("workspace_root") and stored["workspace_root"] != task.get("workspace_root")):
                raise ProjectAdmissionError("project_routing_fence_changed", "The saved task assignment changed.")
            if not unscoped and read_error is not None:
                raise ValueError("Project information is unreadable; recovery will be checked automatically when it is readable.") from read_error
            if bindings is None and bindings_error is None:
                try:
                    bindings = _load_bindings(queue.DRIVE_ROOT, strict=True)["bindings"]
                except (OSError, ValueError, TypeError, RuntimeError) as exc:
                    bindings_error = exc
            if bindings_error is not None:
                raise ValueError("Task scope information is unreadable; recovery will be checked automatically when it is readable.") from bindings_error
            binding = project_binding_for_task(queue.DRIVE_ROOT, tid, strict=True, bindings_snapshot=bindings)
            # Conversion owns the durable binding; its older result is a projection.
            if (stored.get("project_id") and stored["project_id"] != task.get("project_id")
                    and not (binding and binding["project_id"] == task.get("project_id"))):
                raise ProjectAdmissionError("project_routing_fence_changed", "The saved task assignment changed.")
            if unscoped:
                pid, known = task_project_membership(queue.DRIVE_ROOT, task, bindings_snapshot=bindings)
                if pid or known:
                    raise ProjectAdmissionError("project_routing_fence_changed", "The task's original unscoped assignment changed.")
                guard = nullcontext()
            else:
                if binding and basis["project"] is None:
                    raise ValueError("The original registered Project identity is missing; automatic recovery is not authorized.")
                if binding and binding["project_id"] != basis["project_id"]:
                    raise ProjectAdmissionError("project_routing_fence_changed", "The task's original Project binding changed.")
                guard = project_admission_guard(queue.DRIVE_ROOT, basis, snapshot=snapshot)
            with guard:
                workspace = str(task.get("workspace_root") or "")
                if workspace:
                    path = pathlib.Path(workspace)
                    if not path.is_dir() or os.path.normcase(str(path.resolve(strict=True))) != os.path.normcase(workspace):
                        raise ValueError("The original prepared folder is unavailable or resolves elsewhere.")
                for key in ("drive_root", "child_drive_root"):
                    if task.get(key) and not pathlib.Path(task[key]).is_dir():
                        raise ValueError("The original prepared task drive is unavailable.")
                # Preserve independent Pause/budget/restart holds in the row.
                task.pop("_project_admission_restore_hold")
                released.append((task, hold))
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            reason = getattr(exc, "reason", "project_routing_fence_lookup_failed")
            if reason in {"project_routing_fence", "project_routing_fence_changed", "deadline"}:
                task["_terminalization_retry"] = {
                    "status": STATUS_FAILED, "trigger": reason, "reason": str(exc),
                    "reconcile_delegate_custody": reason == "deadline",
                }
            task["_project_admission_restore_hold"] = {"reason": reason, "detail": str(exc)}
    # No worker sees a released row before its restart-visible removal is saved.
    # Unknown persistence puts the exact original hold back; later passes recheck.
    changed = any(prior != {key: task[key] for key in effects if key in task}
                  for prior, task in zip(before, held))
    persisted = not changed or queue.persist_queue_snapshot(reason="project_hold_revalidated") is True
    for task, hold in released:
        if persisted:
            task.pop("_managed_update_resume", None)  # a spent resolver resume never reaches a worker
        else:
            task["_project_admission_restore_hold"] = hold


def persist_never_admitted_refusal(
    drive_root: Any, task_id: str, *, admission_token: str = "", **fields: Any,
) -> Dict[str, Any]:
    """Publish a proved refusal before cleanup, without taking another id owner.

    Queue refusal can release its reservation before the producer resumes. Check
    custody again under Q and establish a durable id owner before freeing drives.
    A strict exact receipt is required even if a writer raises after replacement.
    """
    from supervisor import queue
    from ouroboros.routing_wait import is_own_admission_stub

    receipt = uuid.uuid4().hex
    with queue._queue_lock:
        reservation = queue.ADMISSION_RESERVATIONS.get(task_id)
        if ((reservation and reservation != admission_token) or task_id in queue.RUNNING
                or any(row.get("id") == task_id for row in queue.PENDING if isinstance(row, dict))):
            raise RuntimeError("Refusal settlement lost task-id ownership; resources retained")
        current = load_task_result(pathlib.Path(drive_root), task_id, strict=True) or {}
        own_stub = bool(admission_token and is_own_admission_stub(current, admission_token))
        if current and not own_stub and not (not admission_token and current.get("status") == STATUS_REQUESTED):
            raise RuntimeError("Refusal settlement found an existing result owner; resources retained")
        try:
            write_task_result(pathlib.Path(drive_root), task_id, STATUS_FAILED,
                              **{**fields, "admission_outcome": "never_admitted",
                                 "_admission_refusal_token": receipt, "strict_existing_dict": True})
        except Exception:
            log.warning("Refusal write raised for %s; checking its exact receipt", task_id, exc_info=True)
        stored = load_task_result(pathlib.Path(drive_root), task_id, strict=True) or {}
        if stored.get("_admission_refusal_token") != receipt or stored.get("status") != STATUS_FAILED:
            raise RuntimeError("Refusal persistence unconfirmed; resources retained")
        queue.release_task_admission(task_id, admission_token)
        return stored


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
    elif reason in {"admission_reservation_owned", "admission_reservation_lost"}:
        detail = "Task not scheduled: another admission owns this task id; its resources were preserved."
        extra = {}
    elif reason.startswith("project_routing_fence"):
        lifecycle = str(admitted.get("_project_lifecycle") or "unavailable")
        from ouroboros.project_dialogue import routing_refusal_cause

        detail = routing_refusal_cause("promote_chat_to_task", "failed", reason) + "."
        if admitted.get("_admission_detail"):
            detail += " " + str(admitted["_admission_detail"])
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
    elif reason in {"root_budget_fence", "root_owner_paused"}:
        detail = (
            "Subagent not scheduled: the owner paused this task tree; it accepts no new "
            "work until the owner resumes it." if reason == "root_owner_paused" else
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
        "extra_fields": {**extra, **({"admission_outcome": "never_admitted"}
                                    if admitted.get("_admission_never_admitted") else {})},
        "persist_result": reason not in {"task_id_lookup_failed", "duplicate_task_id",
                                         "admission_reservation_owned", "admission_reservation_lost"},
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
            role=str(evt.get("role") or "").strip(), result_fields={},
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
        if "_project_admission" in admitted:
            result_fields["_project_admission"] = admitted["_project_admission"]
        result_fields["task_contract"] = admitted_task_contract
        result_fields["depth_provenance"] = admitted_depth_provenance
        result_fields["delegation_admission"] = {
            "status": "accepted",
            "direct_child_count": direct_child_count,
            "transition_id": transition_id,
        }
        try:
            write_task_result(ctx.DRIVE_ROOT, tid, STATUS_SCHEDULED, **result_fields,
                              result="Subagent accepted and scheduled.")
        except Exception:
            log.warning("Scheduled subagent receipt write raised for %s; reading its token", tid, exc_info=True)
        try:
            current = load_task_result(ctx.DRIVE_ROOT, tid, strict=True) or {}
        except (OSError, ValueError):
            # A committed receipt may exist. Preserve queue/resource custody and
            # never quarantine unreadable bytes or infer permission to resend.
            admitted["_admission_uncertain"] = "Subagent admission receipt could not be checked; resources retained."
            return admitted, "", "", False
        if _committed(current):
            return admitted, "", "", False
        # Q has remained held since append: no worker observed this refused
        # scheduled-result transition, and strict readback established absence.
        for index, row in enumerate(pending_ref):
            if row is admitted:
                pending_ref.pop(index)
                break
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
        # A confirmed admission remains replayable while its task is still live.
        # Only the durable token proves this is that same admission.
        try:
            from ouroboros.task_results import load_task_result

            existing = load_task_result(
                pathlib.Path(drive_root or queue.DRIVE_ROOT), tid, strict=True,
            ) or {}
        except Exception:
            return {"status": "blocked", "reason": "task_id_lookup_failed"}
        from ouroboros.routing_wait import is_own_admission_stub

        if existing and not is_own_admission_stub(existing, token):
            # The emitted stub of THIS admission is its own pre-receipt (#1160), not
            # another task owning the id: the request it belongs to still reserves.
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
        if tid in queue.RUNNING or any(
            isinstance(row, dict) and str(row.get("id") or "") == tid
            for row in queue.PENDING
        ):
            return {"status": "blocked", "reason": "duplicate_task_id"}
        if require_worker_pool:
            try:
                from supervisor import workers

                pool_state = workers._worker_pool_execution_state(worker_pool)
            except Exception:
                return {"status": "blocked", "reason": "worker_pool_state_unavailable"}
            if not pool_state["available"]:
                return {
                    "status": "blocked",
                    "reason": "worker_pool_unavailable",
                    "worker_pool_disabled_reason": pool_state["disabled_reason"],
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
    "release_task_admission",
    "reserve_task_admission",
    "subagent_schedule_owned",
    "subagent_schedule_preflight",
]
