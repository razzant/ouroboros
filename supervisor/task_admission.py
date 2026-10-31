"""Atomic admission transitions for the managed task queue.

Ingress ids are guarded by token-owned reservations: ``reserve_task_admission``
runs before the caller creates drives, attachments or workspace artifacts, a
different token on a live id is refused as ``duplicate_task_id``, and only the
owning token can release it. A retry cannot therefore create a second task.
"""

from __future__ import annotations

import logging
import pathlib
import uuid
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

from ouroboros.depth_evidence import parse_task_depth
from ouroboros.owner_wait import governing_continuation  # re-export for admission callers
from ouroboros.task_results import (
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_REQUESTED,
    STATUS_SCHEDULED,
    load_task_result,
    write_task_result,
)
from ouroboros.store import StoreError, acquisition_budget

log = logging.getLogger(__name__)

# The requested row's preparation claim (``supervisor/workspace_preparation.py``):
# its token, owner, phase and owned resource handle. A row carrying another
# token's claim is owned by that preparation, never by a new arrival.
WORKSPACE_PREPARATION_FIELD = "workspace_preparation"


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
    restored.setdefault("queued_at", None)  # the saved queue time, or unknown: never this restore (#1805)
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
    preparation: Any = None,
    defer_contention: bool = False,
) -> tuple[int, bool]:
    """Parse depth after a final freshness check.

    The initial replay probe runs before expensive admission work, but a
    concurrent reservation or queue assignment can arrive before parsing. The
    row is read outside Q; a preparation claim's reservation is its custody
    between the probe and a malformed-result rejection, while the historical
    inline path (no claim) keeps that rejection under Q so an invalid replay
    cannot steal custody in between. A claim reads its own reservation and
    marker as its own, never as a replay.
    """
    from contextlib import nullcontext
    from supervisor import queue

    with (nullcontext() if preparation is not None else queue._queue_lock), acquisition_budget(0):
        if tid and subagent_schedule_preflight(
            ctx, evt, chat_id, delegation_role=delegation_role,
            admission_token=getattr(preparation, "token", ""),
            defer_contention=defer_contention,
        ):
            return 0, True
        try:
            return parse_task_depth(evt.get("depth", 0), default=0), False
        except (TypeError, ValueError) as exc:
            from supervisor.events import _reject_schedule_task

            _reject_schedule_task(
                ctx,
                preparation=preparation,
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
    if "_working_recovery" in task:
        from supervisor import queue

        return ancestor_unknown or not _working_resume_granted(task, queue.DRIVE_ROOT, existing=existing)
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
    """Persist possible handoff for scoped work and every saved-work successor (alone)."""
    projection = project_dispatch_projection(task)
    return projection is None or commit_dispatch_marks(task, [projection]) is True


def project_dispatch_projection(task: dict) -> Optional[SimpleNamespace]:
    """The Project half of a launch's dispatch mark, for scoped work and every saved-work
    successor; None when it does not apply (``commit_dispatch_marks`` writes it).

    Assignment reads back 'possible' before handoff, so even an older PENDING
    snapshot cannot overrule it if the best-effort RUNNING mirror fails. Its
    attempt mark never decreases; fresh admission's positive 'none' lives
    on the queue row. No early result may preempt the admission receipt owner.
    Only the governing continuation's own check applies (``governing_continuation``).
    """
    if not task.get("project_id") and task.get("_project_scope_none") is not True and "_working_recovery" not in task:
        return None
    from supervisor import queue
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, STATUS_CANCEL_REQUESTED

    attempt = task.get("_attempt") or 1
    if type(attempt) is not int or attempt < 1:
        return SimpleNamespace(project=lambda current, incoming: None, verify=lambda stored: False)
    governing = governing_continuation(task)
    source = _working_source(task, queue.DRIVE_ROOT) if governing == "working" else None
    mark = uuid.uuid4().hex[:12]  # this commit's own: an earlier mark of the same attempt never verifies it

    def project(current, _incoming):
        if (current.get("status") in _TRULY_TERMINAL_STATUSES | {STATUS_CANCEL_REQUESTED}
                or current.get("_owner_hold")):
            if current.get("_owner_hold"):
                task["_owner_hold"] = current["_owner_hold"]
            return None
        marked = current.get("admitted_dispatch_attempt")
        if (marked is not None and (type(marked) is not int or marked < 1 or marked > attempt)
                or "admitted_dispatch_attempt" in current and marked is None
                or governing == "owner_wait" and not _owner_wait_current(task, current)
                or governing == "exact_pause" and not _exact_resume_granted(task, queue.DRIVE_ROOT, existing=current)
                or source is not None and not _working_attempt_unstarted(task, current, source)):
            return None
        return {"admitted_dispatch": "possible", "admitted_dispatch_attempt": attempt,
                "admitted_dispatch_mark": mark, "status": current.get("status") or STATUS_REQUESTED}

    def verify(stored):
        return (stored.get("admitted_dispatch") == "possible"
                and type(stored.get("admitted_dispatch_attempt")) is int
                and stored["admitted_dispatch_attempt"] == attempt and stored.get("admitted_dispatch_mark") == mark
                and stored.get("status") not in _TRULY_TERMINAL_STATUSES | {STATUS_CANCEL_REQUESTED}
                and not stored.get("_owner_hold"))

    return SimpleNamespace(project=project, verify=verify)


def commit_dispatch_marks(task: dict, projections: list) -> Optional[bool]:
    """ONE task-result write carrying every applicable owner's dispatch mark (Project, schedule).

    Each projection keeps its owner's predicate and any refusal writes nothing; a
    later projection wins a shared field (the schedule receipt's ``running``), so the
    document is the one the owners' separate writes produced. True: committed and
    read back. False: positively not written (a refusal, or a transaction whose
    rollback was confirmed for another cause). None: a commit may have landed (an
    exception without confirmed rollback, an unreadable or disagreeing readback) —
    never replayable. Confirmed-rollback contention re-raises: nothing was written,
    and the caller's contention classifier (``schedule_occurrence.contention``)
    defers the same first dispatch, as demand rather than a refusal.
    """
    from supervisor import queue

    refused = [False]

    def compose(current, incoming):
        fields: Dict[str, Any] = {}
        for projection in projections:
            projected = projection.project(current, incoming)
            if projected is None:
                refused[0] = True
                return None
            fields.update(projected)
        refused[0] = False
        return fields

    try:
        write_task_result(queue.DRIVE_ROOT, task["id"], STATUS_REQUESTED, _field_projector=compose,
                          strict_existing_dict=True)
    except Exception as exc:
        if getattr(exc, "rollback_confirmed", None) is True:
            from supervisor.schedule_occurrence import contention

            if contention(exc):
                raise
            log.info("Dispatch evidence for %s was not written (%s)", task.get("id"), type(exc).__name__)
            return False
        log.warning("Dispatch evidence for %s may have committed", task.get("id"), exc_info=True)
        return None
    if refused[0]:
        return False
    try:
        stored = load_task_result(queue.DRIVE_ROOT, task["id"], strict=True) or {}
    except Exception:
        log.warning("Committed dispatch evidence for %s could not be read back", task.get("id"), exc_info=True)
        return None
    return True if all(projection.verify(stored) for projection in projections) else None


def _owner_wait_current(task: dict, stored: dict) -> bool:
    """The handoff names the result's still-waiting wait; restart authority stays with restore."""
    handoff, wait = task.get("_owner_wait_resume"), stored.get("owner_wait")
    return (isinstance(handoff, dict) and isinstance(wait, dict) and wait.get("state") == "waiting"
            and bool(wait.get("wait_id")) and wait.get("wait_id") == handoff.get("wait_id"))


def _working_source(task: dict, drive_root: Any) -> dict:
    from ouroboros.working_checkpoint import recovery_source_for_task

    try:
        return recovery_source_for_task(pathlib.Path(task.get("budget_drive_root") or drive_root), task)
    except (OSError, ValueError, TypeError, KeyError):
        log.warning("Saved work source could not authorize dispatch for %s", task.get("id"), exc_info=True)
        return {}


def _working_attempt_unstarted(task: dict, stored: dict, source: dict) -> bool:
    """An exact source permits its next attempt only before that attempt's handoff.

    New launchers stamp the attempt BEFORE put, including when the RUNNING
    mirror later fails. An old source with no addressed mark proves no such
    protocol; retain it rather than guess. Earlier attempt marks never reset.
    A direct first launch stores 0 (agent), while its checkpoint names 1
    (owner_wait). Only that unmarked, owner-bound launch uses the source's 1.
    """
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, STATUS_CANCEL_REQUESTED

    if (not source or stored.get("task_id") != task.get("id")
            or not stored.get("status") or stored.get("admission_outcome") == "never_admitted"
            or stored.get("status") in _TRULY_TERMINAL_STATUSES | {STATUS_CANCEL_REQUESTED}):
        return False
    target = task["_attempt"]  # the exact-source reader validated this integer
    marked = stored.get("admitted_dispatch_attempt")
    if "admitted_dispatch_attempt" in stored:
        if type(marked) is not int or not 0 < marked < target:
            return False
    elif (type(source.get("retained_work_dispatch_protocol")) is not int
          or source["retained_work_dispatch_protocol"] != 1):
        return False
    observed = stored.get("task_attempt")
    owner = stored.get("execution_owner")
    if (type(observed) is int and observed == 0 and target == 2 and source.get("task_attempt") == 1
            and "admitted_dispatch_attempt" not in stored and isinstance(owner, dict)
            and owner.get("kind") == "direct"
            and type(owner.get("task_attempt")) is int and owner["task_attempt"] == 0):
        observed = 1
    return ("task_attempt" not in stored
            or type(observed) is int and 0 < observed < target)


def _working_resume_granted(task: dict, drive_root: Any, *, existing: Optional[dict] = None) -> bool:
    source = _working_source(task, drive_root)
    if not source:
        return False
    stored = existing if existing is not None else load_task_result(
        pathlib.Path(task.get("budget_drive_root") or drive_root), str(task.get("id") or ""), strict=True) or {}
    return _working_attempt_unstarted(task, stored, source)


def _exact_resume_granted(task: dict, drive_root: Any, *, existing: Optional[dict] = None) -> bool:
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
    stored = existing if existing is not None else load_task_result(
        pathlib.Path(task.get("budget_drive_root") or drive_root), str(task.get("id") or ""), strict=True) or {}
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
    from supervisor import queue, workers
    from supervisor.queue_snapshot import _descends_from, _interrupted_ancestors, _fence_snapshot_running_rows, _exact_pause_row
    from supervisor.schedule_occurrence import first_dispatch_unsent, restore_allowed
    from supervisor.update_merge import assisted_resume_authorizes
    from ouroboros.projects_registry import _load_bindings, project_binding_for_task
    from ouroboros.project_placement import pending_placement, refresh_pending_placement
    from ouroboros.owner_wait import restore_owner_wait_allowed
    from ouroboros.delegate_recovery import planned_handoff_resume_allowed
    from ouroboros.workspace_admission import admitted_absent_workspace
    from supervisor.events_budget import BUDGET_HOLD_KEY, native_continuation_repair

    from supervisor.budget_resume import revalidate_consumed_resume_holds

    revalidate_consumed_resume_holds()  # spent-Resume holds re-decided every tick; memory-only without them
    held = [task for task in queue.PENDING if task.get("_project_admission_restore_hold")]
    if not held:
        return
    effects = ("_project_admission_restore_hold", "_terminalization_retry", "_owner_hold",
               "_owner_wait_resume", "_budget_pause", "_budget_pause_resume", "_resume_selection", BUDGET_HOLD_KEY)
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
            expired = bool(deadline and time.time() >= deadline)
            # An owner-wait continuation is judged on its exact authority apart from that
            # deadline (#939): unreadable keeps custody (raises), refused takes the shutdown custody.
            owner_resume = restore_owner_wait_allowed(queue.DRIVE_ROOT, task, strict=True, expiring=expired)
            if task.get("_owner_wait_resume") and (
                    task["_owner_wait_resume"].get("reason") == "owner_pause" or not owner_resume):
                from supervisor.restart_retention import (
                    park_saved_pause, saved_sleep_hold_reason, pause_ids, unseen_pause, _latch_unseen_pause)

                before_pause = pause_ids(task)
                parked = park_saved_pause(task, int(task.get("_attempt") or 1), queue.DRIVE_ROOT,
                    pause_source="project_hold_owner_wait", sleep_hold_reason=saved_sleep_hold_reason(queue.DRIVE_ROOT))
                if parked is not None:
                    task.update(parked)
                    if "_owner_wait_resume" not in parked:
                        task.pop("_owner_wait_resume", None)
                    if unseen_pause(task, before_pause):
                        _latch_unseen_pause(task)
                    continue  # retained pause, or its failed conversion; never a Stop
                if native_continuation_repair(task):
                    from ouroboros.cancel_intents import has_active_intent

                    wait = (load_task_result(queue.DRIVE_ROOT, tid, strict=True) or {}).get("owner_wait") or {}
                    if (wait.get("state") == "waiting" and wait.get("wait_id") == task["_owner_wait_resume"].get("wait_id")
                            and not has_active_intent(queue.DRIVE_ROOT, tid, strict=True)):
                        continue  # only the unavailable live source keeps this retention hold
                _fence_snapshot_running_rows([{"id": tid}], restored_ids=set())
                continue  # the same shutdown custody as an unresumable restore
            if expired:
                from supervisor.queue_timeouts import pending_deadline_expirable

                if not pending_deadline_expirable(task):
                    continue  # Pause and unconfirmed dispatch keep their own custody.
                raise ProjectAdmissionError("deadline", "The task deadline elapsed while waiting for its Project.")
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
            stored = load_task_result(queue.DRIVE_ROOT, tid, strict=True) or {}
            governing = governing_continuation(task)  # the final dispatch's selection, no fallback
            granted = (owner_resume if governing == "owner_wait"
                       else _exact_resume_granted(task, queue.DRIVE_ROOT) if governing == "exact_pause"
                       else governing == "working" and _working_resume_granted(task, queue.DRIVE_ROOT, existing=stored)
                       ) or assisted_resume_authorizes(task, stored)
            # Planned authority never replaces a governing loop carrier, valid or invalid.
            granted = granted or (not governing and planned_handoff_resume_allowed(queue.DRIVE_ROOT, task))
            # A first-attempt schedule receipt that still proves no handoff outranks an
            # earlier launcher's Project mark of that attempt (``first_dispatch_unsent``).
            unsent = not governing and first_dispatch_unsent(task)
            if (stored.get("admission_outcome") == "never_admitted"
                    or not granted and not unsent and (
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
            if not host_unscoped(task) and read_error is not None:
                raise ValueError("Project information is unreadable; recovery will be checked automatically when it is readable.") from read_error
            if bindings is None and bindings_error is None:
                try:
                    bindings = _load_bindings(queue.DRIVE_ROOT, strict=True)["bindings"]
                except (OSError, ValueError, TypeError, RuntimeError) as exc:
                    bindings_error = exc
            if bindings_error is not None:
                raise ValueError("Task scope information is unreadable; recovery will be checked automatically when it is readable.") from bindings_error
            binding = project_binding_for_task(queue.DRIVE_ROOT, tid, strict=True, bindings_snapshot=bindings)
            placement = pending_placement(queue.DRIVE_ROOT, task, bindings_snapshot=bindings)
            if placement and not refresh_pending_placement(queue.DRIVE_ROOT, task, binding=placement):
                raise ValueError("The committed placement cannot be readmitted yet.")
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
                # A folder admitted as merely absent (#1799) keeps that admission while its address stays absent.
                if workspace and not admitted_absent_workspace(task, system_repo_dir=workers.REPO_DIR,
                                                               drive_root=queue.DRIVE_ROOT):
                    path = pathlib.Path(workspace)
                    if not path.is_dir() or os.path.normcase(str(path.resolve(strict=True))) != os.path.normcase(workspace):
                        raise ValueError("The original prepared folder is unavailable or resolves elsewhere.")
                for key in ("drive_root", "child_drive_root"):
                    if task.get(key) and not pathlib.Path(task[key]).is_dir():
                        raise ValueError("The original prepared task drive is unavailable.")
                # Preserve independent Pause/budget/restart holds in the row.
                task.pop("_project_admission_restore_hold")
                released.append((task, {"_project_admission_restore_hold": hold,
                                        **({"admitted_dispatch": task.get("admitted_dispatch")} if unsent else {})}))
                if unsent:  # restore's own correction, read now that the receipt is readable
                    task["admitted_dispatch"] = "none"
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
    for task, prior in released:
        if persisted:
            task.pop("_managed_update_resume", None)  # a spent resolver resume never reaches a worker
        else:
            task.update(prior)


def persist_never_admitted_refusal(
    drive_root: Any, task_id: str, /, *, admission_token: str = "", **fields: Any,
) -> Dict[str, Any]:
    """Publish a proved refusal before cleanup, without taking another id owner.

    The root and id are positional-only: a subagent's result fields carry their
    own ``drive_root`` (its child drive), which must reach the row, not collide.

    Queue refusal can release its reservation before the producer resumes. Check
    custody under Q (memory: the reservation, PENDING, RUNNING), then establish
    the durable id owner outside Q, the token being the custody meanwhile, and a
    projector that writes only onto the token's own row; a strict exact receipt
    is required even if a writer raises after replacement. A token's own row is
    its emitted promote stub, the requested child its claim marks (or, before
    that marker, its reservation holds), or its own SCHEDULED receipt that was
    never put (``own_unput_receipt``); any other row keeps its owner.
    """
    from supervisor import queue
    from ouroboros.routing_wait import is_own_admission_stub

    # The transition id of this admission's own SCHEDULED receipt whose put was refused (inline path).
    put_refused = str(fields.pop("_admission_receipt", "") or "")
    receipt = str(fields.get("_admission_refusal_token") or "") if admission_token else ""
    receipt = receipt or uuid.uuid4().hex
    with queue._queue_lock:
        reservation = queue.ADMISSION_RESERVATIONS.get(task_id)
        if ((reservation and reservation != admission_token) or task_id in queue.RUNNING
                or any(row.get("id") == task_id for row in queue.PENDING if isinstance(row, dict))):
            raise RuntimeError("Refusal settlement lost task-id ownership; resources retained")

    def own(current: Dict[str, Any]) -> bool:
        claim = fields.get(WORKSPACE_PREPARATION_FIELD)
        own_claim = bool(admission_token and isinstance(claim, dict) and claim.get("token") == admission_token
                         and (own_requested_child(current, admission_token, reservation)
                              or own_unput_receipt(current, admission_token)))
        unput = (current.get("delegation_admission") or {}).get("transition_id") if put_refused else None
        return (not current or own_claim or bool(admission_token and is_own_admission_stub(current, admission_token))
                or not admission_token and current.get("status") == STATUS_REQUESTED
                or bool(put_refused and current.get("status") == STATUS_SCHEDULED and unput == put_refused))

    def project(current: Dict[str, Any], incoming: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not own(current):
            return None
        admission = current.get("delegation_admission")
        if isinstance(admission, dict) and admission.get("status") == "accepted":
            # Its receipt said accepted before the put; a child never admitted is no direct child.
            incoming = {**incoming, "delegation_admission": {**admission, "status": "never_admitted"}}
        return incoming

    with acquisition_budget(0):
        current = load_task_result(pathlib.Path(drive_root), task_id, strict=True) or {}
        recorded_claim = current.get(WORKSPACE_PREPARATION_FIELD)
        if (admission_token and current.get("status") == STATUS_FAILED
                and current.get("_admission_refusal_token") == receipt
                and isinstance(recorded_claim, dict) and recorded_claim.get("token") == admission_token):
            queue.release_task_admission(task_id, admission_token)
            return current  # this same refusal committed before its contended readback
        if not own(current):
            raise RuntimeError("Refusal settlement found an existing result owner; resources retained")
        try:
            write_task_result(pathlib.Path(drive_root), task_id, STATUS_FAILED, _field_projector=project,
                              **{**fields, "admission_outcome": "never_admitted",
                                 "_admission_refusal_token": receipt, "strict_existing_dict": True})
        except Exception as exc:
            from supervisor.workspace_preparation import store_contention
            if store_contention(exc):
                raise
            log.warning("Refusal write raised for %s; checking its exact receipt", task_id, exc_info=True)
        stored = load_task_result(pathlib.Path(drive_root), task_id, strict=True) or {}
        if stored.get("_admission_refusal_token") != receipt or stored.get("status") != STATUS_FAILED:
            raise RuntimeError("Refusal persistence unconfirmed; resources retained")
    queue.release_task_admission(task_id, admission_token)
    return stored


def own_requested_child(current: Dict[str, Any], token: str, reservation: Any = None) -> bool:
    """Whether ``current`` is still the requested child this preparation token claimed."""
    if not token or str(current.get("status") or "") != STATUS_REQUESTED:
        return False
    claim = current.get(WORKSPACE_PREPARATION_FIELD)
    if isinstance(claim, dict):
        return str(claim.get("token") or "") == token
    return reservation == token


def own_unput_receipt(current: Dict[str, Any], token: str) -> bool:
    """Whether ``current`` is this token's SCHEDULED receipt whose queue put was never entered: its
    marker still says ``admitting`` (the put retires it to ``admitted``)."""
    claim = current.get(WORKSPACE_PREPARATION_FIELD)
    return bool(token and str(current.get("status") or "") == STATUS_SCHEDULED and isinstance(claim, dict)
                and str(claim.get("token") or "") == token and str(claim.get("phase") or "") == "admitting")


def claim_requested_child(ctx: Any, task_id: str, token: str) -> bool:
    """Reserve the producer's REQUESTED child for one preparation token, before any effect.

    Memory only, under Q: the existing reservation map is the claim; another
    token or queued/running work is not claimable. The row's own freshness
    (still REQUESTED, no foreign marker) is read by the claim's lane before its
    first write, never here on the event loop.
    """
    from supervisor import queue

    tid = str(task_id or "")
    with queue._queue_lock:
        reserved = queue.ADMISSION_RESERVATIONS.get(tid)
        if not tid or not token or reserved and reserved != token:
            return False
        pending = getattr(ctx, "PENDING", queue.PENDING)
        if tid in getattr(ctx, "RUNNING", queue.RUNNING) or any(
                isinstance(row, dict) and str(row.get("id") or "") == tid for row in pending):
            return False
        queue.ADMISSION_RESERVATIONS[tid] = token
        return True


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
        # A never-admitted refusal names its own unput receipt, which only that refusal writer consumes.
        "extra_fields": {**extra, **({"admission_outcome": "never_admitted",
                                      "_admission_receipt": str(admitted.get("_admission_receipt") or "")}
                                     if admitted.get("_admission_never_admitted") else {})},
        "persist_result": reason not in {"task_id_lookup_failed", "duplicate_task_id",
                                         "admission_reservation_owned", "admission_reservation_lost"},
    }


def subagent_schedule_owned(
    ctx: Any, task_id: str, *, pending_ref: Any = None, admission_token: str = "",
) -> bool:
    """Return whether an exact child id already has queue/lifecycle custody.

    ``admission_token`` is a preparation claim's own token: its reservation and
    marker are not another owner, while its lost marker or reservation is. The
    row is read outside Q (zero wait: a gap is the caller's owner's to retry);
    under Q only memory and the token are compared.
    """
    from supervisor import queue

    tid = str(task_id or "")
    with acquisition_budget(0):
        current = load_task_result(ctx.DRIVE_ROOT, tid, strict=True) or {}
    with queue._queue_lock:
        return queue_owned_elsewhere(ctx, tid, current, pending_ref=pending_ref, admission_token=admission_token)


def queue_owned_elsewhere(ctx: Any, tid: str, current: Optional[Dict[str, Any]], *, pending_ref: Any = None,
                          admission_token: str = "") -> bool:
    """Under Q, memory only: another reservation, a queued/running row, or (given the row read
    outside Q) a progressed status or a foreign claim means another owner holds the id."""
    from supervisor import queue

    pending = pending_ref if isinstance(pending_ref, list) else getattr(ctx, "PENDING", queue.PENDING)
    running = getattr(ctx, "RUNNING", queue.RUNNING)
    reserved = queue.ADMISSION_RESERVATIONS.get(tid)
    if reserved and reserved != admission_token:
        return True
    if tid in running or any(isinstance(row, dict) and str(row.get("id") or "") == tid for row in pending):
        return True
    if current is None:
        return False
    claim = current.get(WORKSPACE_PREPARATION_FIELD)
    foreign_claim = (str((claim or {}).get("token") or "") != admission_token
                     if isinstance(claim, dict) else bool(admission_token))
    return str(current.get("status") or "") not in {"", STATUS_REQUESTED} or foreign_claim


def subagent_schedule_preflight(
    ctx: Any,
    evt: Dict[str, Any],
    chat_id: int,
    *,
    delegation_role: str = "subagent",
    admission_token: str = "",
    defer_contention: bool = False,
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
        # A claim names its token; the historical call shape stays for every other caller.
        return subagent_schedule_owned(
            ctx, tid, **({"admission_token": admission_token} if admission_token else {}))
    except (OSError, ValueError, StoreError) as exc:
        from supervisor.workspace_preparation import store_contention
        if (admission_token or defer_contention) and store_contention(exc):
            raise  # the existing preparation owner waits only after Q unwinds
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
    preparation: Any = None,
    recheck_child_admission: Optional[Callable[[Any], tuple[str, str]]] = None,
    durable_child_count: Optional[Callable[[], Any]] = None,
    transition_id: str = "",
) -> tuple[Any, str, str, bool]:
    """Enqueue a child only after its first durable authority row landed, never under Q.

    Receipt, then put. The SCHEDULED receipt (``delegation_admission`` accepted
    under one transition id) is written outside Q, on the claim's lane with its
    acquisition gate; the queue append and the snapshot follow under Q with a
    memory-only revalidation (the token, PENDING/RUNNING, the open generation,
    the count-dependent rights and caps); the claim's ``admitting`` marker is
    retired last. No worker can observe a row before its receipt, and a receipt
    without a put is recoverable (``workspace_preparation._never_put``).
    ``durable_child_count`` reads the parent's direct-child count on the lane
    (contention propagates to the gate); ``recheck_child_admission(count)``
    compares it with memory under Q. Admission steps are FIFO on one lane, so no
    sibling's receipt lands between that read and this append. A refusal after
    the receipt settles it typed through the same never-admitted owner (the
    caller's rejection). An uncertain receipt is inspected by re-reading under
    the token, never re-sent; a hard unreadable readback keeps queue custody as
    an uncertain admission. Without a claim (the historical inline path) a
    storage gap is the caller's.
    """
    from supervisor import queue
    from supervisor.workspace_preparation import store_contention

    tid = str(task.get("id") or "")
    transition_id = str(transition_id or uuid.uuid4().hex)
    token = ""
    if preparation is not None:
        crossed = preparation.enter_admission()
        if crossed is False:
            return (task, "scheduled_event_replay",
                    "Subagent schedule ignored: its prepared request no longer belongs to this preparation.", False)
        if crossed is None:  # this token never entered the queue below
            return (task, "preparation_marker_unwritable",
                    "Subagent not scheduled: its admission boundary could not be recorded durably, "
                    "so it was never admitted.", True)
        token = preparation.token
        task = {**task, "_admission_token": token}

    def gate(operation: Callable[[], Any], **kwargs: Any) -> Any:
        return operation() if preparation is None else preparation.store_gate(operation, **kwargs)

    def _committed(record: Any) -> bool:
        admission = record.get("delegation_admission") if isinstance(record, dict) else None
        return bool(
            isinstance(record, dict)
            and str(record.get("status") or "") == STATUS_SCHEDULED
            and isinstance(admission, dict)
            and str(admission.get("status") or "") == "accepted"
            and str(admission.get("transition_id") or "") == transition_id
        )

    def read_row() -> Dict[str, Any]:
        return load_task_result(ctx.DRIVE_ROOT, tid, strict=True) or {}

    def refused_under_q(current: Optional[Dict[str, Any]], count: Any) -> Optional[tuple]:
        """Memory only, under Q; ``current`` is None once this token's own receipt is on the row."""
        if queue_owned_elsewhere(ctx, tid, current, pending_ref=pending_ref, admission_token=token):
            log.info("Ignoring replayed schedule event for task %s", tid)
            return (task, "scheduled_event_replay",
                    "Subagent schedule replay ignored: this task id is already owned by "
                    "an existing queue or durable lifecycle row.", False)
        if preparation is not None and preparation.generation.closed():
            # A closed generation's callback never enqueues into its successor.
            return (task, "preparation_interrupted_by_generation_exit",
                    "Subagent not scheduled: the supervisor generation that prepared its workspace "
                    "ended before admission; it was never admitted.", True)
        if recheck_child_admission is not None:
            reason, detail = recheck_child_admission(count)
            if reason:
                return task, reason, detail, True
        return None

    try:
        previous = gate(read_row, busy=_store_busy)
    except (OSError, ValueError, StoreError) as exc:
        if _store_busy(exc):
            raise  # a gap off any lane is the caller's; on a lane the gate already waited
        log.warning("Subagent schedule authority is unreadable for %s", tid, exc_info=True)
        return (
            task, "scheduled_result_authority_unknown",
            "Subagent not scheduled: the existing durable result for this "
            "task id is unreadable, so the host cannot prove that the id is "
            "fresh. The existing result was preserved.", False,
        )
    count = gate(durable_child_count) if durable_child_count is not None else direct_child_count
    with queue._queue_lock:
        refused = refused_under_q(previous, count)
    if refused is not None:
        return refused
    receipt_fields = {
        **result_fields, "task_contract": admitted_task_contract, "depth_provenance": admitted_depth_provenance,
        "delegation_admission": {"status": "accepted", "direct_child_count": count, "transition_id": transition_id},
    }
    if isinstance(task.get("_project_admission"), dict):  # the frozen basis the put admits on, as the queue keeps it
        receipt_fields["_project_admission"] = {
            **{key: value for key, value in task["_project_admission"].items() if key != "workspace_claims"},
            "frozen": True}
    receipt_error: list = []

    def write_receipt() -> None:
        receipt_error.clear()
        try:
            write_task_result(ctx.DRIVE_ROOT, tid, STATUS_SCHEDULED, **receipt_fields,
                              result="Subagent accepted and scheduled.")
        except Exception as exc:
            if store_contention(exc):
                raise  # rolled back: the gate retries this same write
            receipt_error.append(exc)
            log.warning("Scheduled subagent receipt write raised for %s; reading its token", tid, exc_info=True)

    gate(write_receipt)
    uncertain = ""
    try:
        current = gate(read_row, busy=_store_busy)
    except (OSError, ValueError, StoreError) as exc:
        if _store_busy(exc):
            raise
        # A committed receipt may exist (hard unreadable, not busy). Preserve queue/resource
        # custody and never quarantine unreadable bytes or infer permission to resend.
        current, uncertain = None, "Subagent admission receipt could not be checked; resources retained."
    if current is not None and not _committed(current):
        prior_status, current_status = str(previous.get("status") or ""), str(current.get("status") or "")
        if any(status not in {"", STATUS_REQUESTED} for status in (prior_status, current_status)):
            return (
                task, "scheduled_result_conflict",
                "Subagent not scheduled: another durable result already owns this "
                "task id, so the new queue admission was rolled back without "
                "overwriting that result.", False,
            )
        return (
            task, "scheduled_result_persist_failed",
            "Subagent not scheduled: its durable scheduled-result receipt could not "
            "be persisted, so queue admission was rolled back.", True,
        )

    def put() -> tuple[str, Any]:
        with queue._queue_lock:
            refused = refused_under_q(None, count)
            if refused is not None:
                return "refused", refused
            admitted = ctx.enqueue_task(task)
            if isinstance(admitted, dict) and admitted.get("_admission_blocked"):
                if admitted.get("_admission_never_admitted") and admitted.get("_admission_blocked") == "contention":
                    # A context-manager exit can fail AFTER append, despite that refusal
                    # flag: inspect actual custody before retrying only this still-unsent token.
                    if (tid not in queue.RUNNING and not any(
                            isinstance(row, dict) and str(row.get("id") or "") == tid for row in pending_ref)
                            and queue.ADMISSION_RESERVATIONS.get(tid) in (None, token)):
                        from ouroboros.store import StoreBusy
                        if preparation is not None:
                            queue.ADMISSION_RESERVATIONS[tid] = token
                        raise StoreBusy("prepared child admission is contended", reason="contention")
                    admitted.pop("_admission_blocked", None)
                    admitted.pop("_admission_never_admitted", None)
                    admitted["_admission_uncertain"] = "Subagent admission changed before its contended outcome; custody retained."
                    ctx.persist_queue_snapshot(reason="schedule_subagent_admitted")
                    return "admitted", admitted
                if admitted.get("_admission_blocked") == "task_id_lookup_failed":
                    return "refused", (
                        task, "scheduled_result_authority_unknown",
                        "Subagent not scheduled: the exact task-result authority became "
                        "unreadable during admission and was preserved.", False,
                    )
                return "blocked", {**admitted, "_admission_receipt": transition_id}
            if uncertain:
                admitted["_admission_uncertain"] = uncertain
            ctx.persist_queue_snapshot(reason="schedule_subagent_admitted")
            return "admitted", admitted

    outcome, admitted = gate(put)
    if outcome == "refused":
        return admitted
    if outcome == "admitted" and preparation is not None and not uncertain:
        preparation.close_admitted()
    return admitted, "", "", False


def _store_busy(error: BaseException) -> bool:
    """A read's gap: any busy store, inspected again after a slice (reads commit nothing)."""
    from ouroboros.store import StoreBusy

    return isinstance(error, StoreBusy)


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
