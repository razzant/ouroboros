"""Activity observations and explicit time bounds of running work.

A task is judged on its own progress AND on its subtree's - a coordinator whose
children are working is not idle - then against the idle window, its explicit
deadline and the execution ceiling (the narrower of the global ceiling and the task's
``execution_timeout_sec``). Unknown silence requests attention, never teardown.
Only explicit hard bounds enter the off-loop reaper under the queue lock.

An explicit calendar deadline also expires a never-started PENDING row (owner choice,
#939): the row is marked under Q and settled by the same reaper without a model start;
ordinary resource holds do not erase it. Stop, saved terminals, owner Pause and
unconfirmed continuations keep their custody. Queue age and the execution duration never create one.
"""

from __future__ import annotations

import datetime

import logging
import pathlib
import time
from typing import Any, Dict

from supervisor.cognitive_operations import _active_operation_progressing
from ouroboros.model_wait import execution_elapsed_seconds
from ouroboros.runtime_limits import resolve_task_execution_timeout_sec
from supervisor.task_model_wait import model_waiting
from supervisor.task_reaper import (
    resolve_grace_episode_for_spared_task as _resolve_grace_episode_for_spared_task,
)


def _queue():
    """The parent module, read at call time.

    The queue owns PENDING/RUNNING, the drive root, the liveness settings and
    the lock that guards them, and ``init``/``init_queue_refs`` REBIND those
    names. Reading them through the module is what keeps one binding: a
    from-import here would freeze the value this module saw at import time
    (the owner-approved D18/D33 mechanical exception).
    """
    from supervisor import queue

    return queue


log = logging.getLogger(__name__)

# Reaper reservations for expired pending rows, keyed by row identity. The row's
# ``_terminalization_retry`` marker stays the durable custody; a lost reservation
# is re-made by the next timeout tick or assignment pass.
_pending_deadline_jobs: Dict[int, Dict[str, Any]] = {}

# The supervisor's timeout rails in priority order: the typed ``terminal_reason``
# the reaper stamps as the task_done ``reason_code`` and the ``task_incident`` key.
# ``project_dialogue.TASK_CAUSE_PHRASES`` carries one owner sentence per member;
# the code itself never reaches a chat.
REASON_ABSOLUTE_CEILING, REASON_DEADLINE, REASON_IDLE_TIMEOUT = TIMEOUT_TERMINAL_REASONS = (
    "absolute_ceiling", "deadline", "idle_timeout",
)


def _task_deadline_ts(task: Dict[str, Any]) -> float:
    raw = str(task.get("deadline_at") or "").strip()
    if not raw:
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        raw = str(metadata.get("deadline_at") or "").strip()
    if not raw:
        contract = task.get("task_contract") if isinstance(task.get("task_contract"), dict) else {}
        raw = str(contract.get("deadline_at") or "").strip()
    if not raw:
        return 0.0
    try:
        parsed = datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=datetime.timezone.utc)
        return float(parsed.timestamp())
    except Exception:
        return 0.0


def _task_drive_for_task(task: Dict[str, Any], task_id: str) -> pathlib.Path:
    """Active drive of a running task (child drive for forked/workspace tasks,
    canonical otherwise) — where its mailbox and observability actually live.
    Resolution mirrors forward_to_worker: task fields, then the result's typed
    drive columns (no body is decoded for one field)."""
    task = task if isinstance(task, dict) else {}
    child = str(task.get("child_drive_root") or task.get("drive_root") or "").strip()
    if not child:
        try:
            from ouroboros.task_store import task_drive_status
            child = task_drive_status(pathlib.Path(_queue().DRIVE_ROOT), [str(task_id)]).get(str(task_id), ("", ""))[1]
        except Exception:
            child = ""
    return pathlib.Path(child) if child else pathlib.Path(_queue().DRIVE_ROOT)


def enforce_task_timeouts(*, pending_events: set[str] | None = frozenset()) -> None:
    """Expire explicit pending calendar deadlines, then enforce running-task rails.

    Holds the queue lock for the whole pass: RUNNING pops and worker respawn
    decisions raced with HTTP cancel handlers (double respawn → orphaned
    worker; wrong-task dequeue). The RLock keeps nested respawn/assign calls
    re-entrant. The server supplies its pending event snapshot (None means
    unreadable) from outside Q; synchronous callers with already-current
    observations retain the empty default. Pending inputs defer only idle.
    """
    # Avoid circular dependency during module load.
    from supervisor import workers

    if not _queue().RUNNING and not _queue().PENDING:
        return
    now = time.time()
    st = _queue().load_state()
    owner_chat_id = int(st.get("owner_chat_id") or 0)

    with _queue()._queue_lock:
        _queue()._expire_pending_deadlines_locked(workers, now)
        if _queue().RUNNING:
            _queue()._enforce_task_timeouts_locked(workers, now, owner_chat_id, st,
                                                 pending_events=pending_events)


def _pending_continuation(task: Dict[str, Any]) -> str:
    """The carrier the loop resumes (``task_admission.governing_continuation``), else an assisted update.

    An exact pause can retain an older working carrier after that recovery started."""
    from supervisor.task_admission import governing_continuation

    governing = governing_continuation(task)
    if governing:
        return {"owner_wait": "_owner_wait_resume", "exact_pause": "_budget_pause_resume",
                "working": "_working_recovery"}[governing]
    return "_managed_update_resume" if task.get("_managed_update_resume") is not None else ""


def _deadline_launch_claim(stored: dict) -> dict:
    return {key: stored[key] for key in ("task_id", "status", "task_attempt", "execution_owner",
                                        "admitted_dispatch", "admitted_dispatch_attempt") if key in stored}


def _retain_unsubmitted_deadline(task: dict, before: dict) -> None:
    """Q + launch_admission caller ONLY, after its claim and before any put.

    Preserve this working successor's known non-handoff on the existing retry
    marker. Its monotonic possible claim stays intact; another owner or source
    cannot inherit the fact. Unknown remote effects/cost are still unknown.
    """
    import copy
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import execution_owner_record
    from supervisor import workers

    q, task_id = _queue(), str(task.get("id") or "")
    root = pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)
    stored = load_task_result(root, task_id, strict=True) or {}
    attempt = before["task_attempt"]
    owner = execution_owner_record(root, task, "pooled")
    claimed = {**before["claim"], "admitted_dispatch": "possible", "admitted_dispatch_attempt": attempt}
    mirrored = {**claimed, "status": "running", "task_attempt": attempt, "execution_owner": owner}
    claim = _deadline_launch_claim(stored)
    if (task.get("_terminalization_retry") or before["task_id"] != task_id
            or task.get("_attempt") != attempt or _pending_continuation(task) != "_working_recovery"
            or task.get("_working_recovery") != before["source"]
            or claim not in (claimed, mirrored)):
        return
    retry = workers._make_terminalization_retry_task(
        task, task_id, status="failed", trigger=REASON_DEADLINE,
        reason="The explicit task deadline elapsed before worker dispatch.")
    retry["_terminalization_retry"]["unsubmitted_launch"] = copy.deepcopy({
        "task_id": task_id, "task_attempt": attempt, "data_root": str(root.resolve()),
        "source": before["source"], "claim": claim,
    })
    if pending_deadline_expirable(retry):
        task.update(retry)
        q.persist_queue_snapshot(reason="pending_deadline_unsubmitted")
        _queue_pending_deadline(task)


def pending_deadline_expirable(task: Dict[str, Any]) -> bool:
    """Expire ordinary waits and positively authorized, unstarted continuations.

    The selected continuation must retain its authority; historical carriers
    neither veto it nor grant a fallback. A caller-owned non-put marker binds
    only its own working claim. Pause and unknown facts keep custody; the
    reaper reads source, claim and controls again before settlement.
    """
    from types import SimpleNamespace
    from ouroboros.owner_pause import member_fence
    from ouroboros.task_results import load_task_result, _TRULY_TERMINAL_STATUSES, STATUS_CANCEL_REQUESTED
    from supervisor.events_budget import budget_hold_fact, budget_resume_dispatch_allowed
    from ouroboros.work_stop import continuation_refusal, current_stop

    if any(task.get(key) for key in (
        "_owner_hold", "_continuation_prepared", "_cancel_intent_authority_hold",
    )) or task.get("_owner_wait_resume") and not isinstance(task["_owner_wait_resume"], dict):
        return False
    pause, hold = task.get("_budget_pause"), budget_hold_fact(task)
    stop = current_stop(task, {"budget": _queue().BUDGET_ROOT_FENCES})
    if continuation_refusal(task) or stop.get("release") == "owner" or stop.get("cause") == "pause":
        return False
    project = task.get("_project_admission_restore_hold")
    if task.get("_budget_pause_hold") and not isinstance(task["_budget_pause_hold"], dict):
        return False
    if pause and (not isinstance(pause, dict) or pause.get("exact_continuation")
                  or pause.get("status") != "paused_before_dispatch"
                  or pause.get("replay_safe") is not True
                  or type(pause.get("physical_calls")) is not int or pause["physical_calls"] != 0):
        return False
    member_stop = current_stop(task)
    if hold is not None and (member_stop.get("cause") not in {"app_stop", "awaiting_selection"}
                             or member_stop.get("door_replaceable") or member_stop.get("release") != "resume"):
        return False
    if project and (not isinstance(project, dict)
                    or project.get("reason") not in ("project_routing_fence_lookup_failed", "deadline")):
        return False
    q, task_id = _queue(), str(task.get("id") or "")
    root_id = str(task.get("root_task_id") or task_id)
    fence = q.BUDGET_ROOT_FENCES.get(root_id) or {}
    if not isinstance(fence, dict) or fence.get("cause") == "owner_pause":
        return False
    result_root = pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)
    try:
        # The durable Pause precedes its queue latch; an unavailable root is not open.
        if member_fence(SimpleNamespace(task_id=task_id, root_task_id=root_id, budget_drive_root=result_root)):
            return False
        selected = _pending_continuation(task)
        marker = task.get("_terminalization_retry") or {}
        unsubmitted = marker.get("unsubmitted_launch")
        if selected or pause or hold is not None or project or fence or unsubmitted is not None:
            stored = load_task_result(result_root, task_id, strict=True) or {}
            if (stored.get("_owner_hold") or stored.get("admission_outcome") == "never_admitted"
                    or stored.get("status") in _TRULY_TERMINAL_STATUSES | {STATUS_CANCEL_REQUESTED}):
                return False
            if unsubmitted is not None and not (
                    isinstance(unsubmitted, dict) and marker.get("trigger") == REASON_DEADLINE
                    and selected == "_working_recovery" and unsubmitted.get("task_id") == task_id
                    and stored.get("task_id") == task_id and type(task.get("_attempt")) is int
                    and type(unsubmitted.get("task_attempt")) is int
                    and task["_attempt"] == unsubmitted.get("task_attempt")
                    and type(stored.get("admitted_dispatch_attempt")) is int
                    and stored["admitted_dispatch_attempt"] == task["_attempt"]
                    and unsubmitted.get("data_root") == str(result_root.resolve())
                    and unsubmitted.get("source") == task[selected]
                    and unsubmitted.get("claim") == _deadline_launch_claim(stored)):
                return False
            if selected:
                from ouroboros.budget_pause import load_budget_pause
                from ouroboros.owner_wait import restore_owner_wait_allowed
                from supervisor.task_admission import _exact_resume_granted, _working_resume_granted, _working_source
                from supervisor.update_merge import assisted_resume_authorizes

                if hold is not None or not isinstance(task[selected], dict) or not task[selected]:
                    return False
                if selected == "_owner_wait_resume":
                    return restore_owner_wait_allowed(result_root, task, strict=True, expiring=True)
                if selected == "_budget_pause_resume":
                    if not (_exact_resume_granted(task, result_root) and budget_resume_dispatch_allowed(q, task)):
                        return False
                    source = load_budget_pause(SimpleNamespace(task_id=task_id, task_attempt=task.get("_attempt") or 1,
                        budget_drive_root=result_root, drive_root=result_root), task[selected])
                    return source["_pause_row"].get("source_ref") == task[selected].get("source_ref")
                if selected == "_working_recovery":
                    return (bool(_working_source(task, result_root)) if unsubmitted is not None
                            else _working_resume_granted(task, result_root, existing=stored))
                return assisted_resume_authorizes(task, stored)
            if (task.get("admitted_dispatch") == "none" and stored.get("status") == "scheduled"
                    and not stored.get("started_at") and stored.get("admitted_dispatch", "none") == "none"):
                return True
            from supervisor.schedule_occurrence import first_dispatch_unsent

            if task.get("admitted_dispatch") == "none" and first_dispatch_unsent(task):
                return True  # an earlier launcher's Project mark; its receipt proves no handoff
            # With no carrier selected, Project release's exact planned grant (#1500) is the
            # other positive authority, only while its successor attempt was never claimed.
            from ouroboros.delegate_recovery import planned_handoff_resume_allowed

            attempt, marked = task.get("_attempt"), stored.get("admitted_dispatch_attempt")
            observed = stored.get("task_attempt")
            return (type(attempt) is int and type(marked) is int and 0 < marked < attempt
                    and ("task_attempt" not in stored or type(observed) is int and 0 < observed < attempt)
                    and planned_handoff_resume_allowed(q.DRIVE_ROOT, task))
        return True
    except (OSError, ValueError, TypeError, KeyError, RuntimeError):  # a store that cannot answer (StoreError) is not open
        return False


def _expire_pending_deadlines_locked(workers: Any, now: float) -> None:
    """Hand rows whose explicit calendar deadline passed to non-dispatch custody (#939, owner A).

    Called under Q. Only an explicit ``deadline_at`` (or legacy submit-relative
    ``timeout_sec``) expires a queued row: queue age and ``execution_timeout_sec``
    never create one, and a row without a deadline waits as before. The existing
    ``_terminalization_retry`` marker carries the outcome; an earlier marker keeps
    its own outcome and owner. Settlement and publication run on the reaper.
    """
    changed, expired = False, []
    for task in _queue().PENDING:
        if not isinstance(task, dict):
            continue
        marker = task.get("_terminalization_retry")
        if marker:
            if isinstance(marker, dict) and marker.get("trigger") == REASON_DEADLINE:
                expired.append(task)
            continue
        deadline = _task_deadline_ts(task)
        if not deadline or now < deadline or not pending_deadline_expirable(task):
            continue
        task.update(workers._make_terminalization_retry_task(
            task, str(task.get("id") or ""), status="failed", trigger=REASON_DEADLINE,
            reason="The explicit task deadline elapsed before worker dispatch.",
        ))
        expired.append(task)
        changed = True
    if changed:
        _queue().persist_queue_snapshot(reason="pending_deadline_expired")
    for task in expired:
        _queue_pending_deadline(task, listed=True)


def _queue_pending_deadline(task: Dict[str, Any], *, listed: bool = False) -> None:
    """Reserve one exact pending row on the existing reaper, once.

    ``listed``: this locked pass just read the row from PENDING, so a batch is not
    rescanned per row; the reaper still re-checks membership before settling."""
    q = _queue()
    with q._queue_lock:
        if id(task) in _pending_deadline_jobs or not (listed or any(row is task for row in q.PENDING)):
            return
        job = {"kind": "pending_deadline", "task_id": str(task.get("id") or ""), "task": task,
               "marker": task.get("_terminalization_retry"), "drive_root": q.DRIVE_ROOT}
        try:
            q._ensure_reaper_started()
            _pending_deadline_jobs[id(task)] = job
            q._reap_queue.put(job)
        except Exception:
            _pending_deadline_jobs.pop(id(task), None)
            log.warning("Pending deadline reaper submission deferred", exc_info=True)


def _reap_pending_deadline(job: Dict[str, Any]) -> None:
    """Reaper thread: settle off Q; remove only the captured row after durable publication."""
    q, task = _queue(), job["task"]
    try:
        with q._queue_lock:
            if (q.DRIVE_ROOT != job["drive_root"] or task.get("_terminalization_retry") is not job["marker"]
                    or not any(row is task for row in q.PENDING)):
                return
        settled = _settle_pending_deadline(task, event_already_published=bool(job["marker"].get("event_published")))
        with q._queue_lock:
            if settled and task.get("_terminalization_retry") is job["marker"]:
                q.PENDING[:] = [row for row in q.PENDING if row is not task]
                q.persist_queue_snapshot(reason="pending_deadline_settled")
    finally:
        with q._queue_lock:
            _pending_deadline_jobs.pop(id(task), None)


def _settle_pending_deadline(task: Dict[str, Any], *, event_already_published: bool = False) -> bool:
    """Write and deliver the deadline outcome without any model dispatch.

    A live owner, an active Stop (task or retry root) and an already saved
    terminal result each win; the row stays in custody on any failure.
    """
    from ouroboros.cancel_intents import (
        _validated_retry_root_cancel_key, active_intents, cancellation_projection_lock,
    )
    from ouroboros.outcomes import EXECUTION_INFRA_FAILED, terminal_outcome_axes
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result, write_task_result
    from supervisor import workers
    from supervisor.worker_assignment import _direct_actor_still_registered

    q, task_id, marker = _queue(), str(task.get("id") or ""), task.get("_terminalization_retry")
    with q._queue_lock:
        if not task_id or task_id in q.RUNNING or any(
                worker.busy_task_id == task_id for worker in workers.WORKERS.values()):
            return False
    if task.get("_is_direct_chat") and _direct_actor_still_registered(task_id):
        return False
    try:
        # Do not recreate a held row's lost receipt as fresh scheduled authority.
        if not pending_deadline_expirable(task):
            existing = load_task_result(q.DRIVE_ROOT, task_id, strict=True) or {}
            if existing.get("status") not in _TRULY_TERMINAL_STATUSES:
                return False
        # The create-only seeding seam takes Q itself: do it before cancel authority (Q -> cancel order).
        q.ensure_control_task_result(task_id)
        workers._audit_delegate_terminal_custody(task_id, REASON_DEADLINE)
        try:
            cost_fields = q.reconstruct_task_cost(task_id, fields=True)
        except Exception:
            cost_fields = {"cost_accounting_status": "unavailable", "cost_final": False}
        # The durable outbox owes delivery before the terminal frame; a root reaches its
        # own lineage, a child its parent through the ordinary handoff.
        from supervisor.terminal_delivery import (
            build_unreviewed_salvage_event, deliver_miss_lane_outcome, deliver_unreviewed_salvage,
            register_pending_delivery, salvage_cancelled_output,
        )

        drive = _task_drive_for_task(task, task_id)
        salvage_root = str(task.get("delegation_role") or "") != "subagent"

        def owe_salvage(current: Dict[str, Any]) -> "Dict[str, Any] | None":
            """Register the deadline salvage answer as OWED; None when the registration failed."""
            text, path = salvage
            delivery = dict(outcome="", reason_code=REASON_DEADLINE, salvaged_text=text, preserved_path=path,
                            settled_status="failed",
                            unreconciled_runs=list(current.get("delegated_runs_unreconciled") or []))
            event = build_unreviewed_salvage_event(q.DRIVE_ROOT, {**task, **current}, task_id, **delivery)
            if event is not None and not register_pending_delivery(q.DRIVE_ROOT, event):
                return None
            return delivery

        delivery, salvage = None, ("", "")
        if salvage_root:  # the response-history read and copy run before Q; only the owed registration rides it
            salvage = tuple(salvage_cancelled_output(q.DRIVE_ROOT, drive, task_id)[1:])
        # Only the terminal write is serialized against Stop; custody and delivery
        # never make a Q-held cancel wait on this projection.
        with q._queue_lock, cancellation_projection_lock(q.DRIVE_ROOT):
            if task.get("_terminalization_retry") is not marker or not any(row is task for row in q.PENDING):
                return False
            # Custody ran off Q: a live owner may have appeared while it waited.
            if (task_id in q.RUNNING or any(worker.busy_task_id == task_id for worker in workers.WORKERS.values())
                    or task.get("_is_direct_chat") and _direct_actor_still_registered(task_id)):
                return False
            retry_root = _validated_retry_root_cancel_key(q.DRIVE_ROOT, task_id, task_hint=task)
            intents = active_intents(q.DRIVE_ROOT, strict=True)
            if task_id in intents or retry_root and retry_root in intents:
                return False
            stored = load_task_result(q.DRIVE_ROOT, task_id, strict=True) or {}
            if stored.get("status") not in _TRULY_TERMINAL_STATUSES:
                if not pending_deadline_expirable(task):  # Pause/authority can change while custody runs off Q.
                    return False
                # Owe the salvage answer BEFORE the terminal row exists (owner Fork 2): a
                # settle landing between the write and the registration sees the debt and
                # never publishes a card without the answer. A refused registration keeps
                # the row in custody, exactly as the post-write refusal used to.
                if salvage_root:
                    delivery = owe_salvage(stored)
                    if delivery is None:
                        return False
                stored = write_task_result(
                    q.DRIVE_ROOT, task_id, "failed", strict_existing_dict=True, reason_code=REASON_DEADLINE,
                    outcome_axes=terminal_outcome_axes(
                        lifecycle="failed", execution=EXECUTION_INFRA_FAILED,
                        reason_code=REASON_DEADLINE, review_trigger="supervisor_terminal"),
                    result=str((marker or {}).get("reason") or "The explicit task deadline elapsed before worker dispatch."),
                    **cost_fields,
                )
        status = str(stored.get("status") or "")
        if status not in _TRULY_TERMINAL_STATUSES:
            return False
        if event_already_published:
            return True
        row = {**task, **stored}
        unreconciled = list(stored.get("delegated_runs_unreconciled") or [])
        if status == "failed" and stored.get("reason_code") == REASON_DEADLINE:
            if salvage_root:
                if delivery is None:  # the row was already terminal when custody caught up: owe its answer now
                    delivery = owe_salvage(stored)
                    if delivery is None:
                        return False
                deliver_unreviewed_salvage(q.DRIVE_ROOT, row, task_id, **delivery)
        elif not deliver_miss_lane_outcome(q.DRIVE_ROOT, drive, row, task_id, status,
                                           unreconciled_runs=unreconciled):
            return False
        return workers._emit_task_done_terminal(task, task_id, status, reason_code=str(stored.get("reason_code") or ""),
                                                cost_fields=cost_fields)
    except Exception:
        log.warning("Pending deadline custody deferred for %s", task_id, exc_info=True)
        return False


def _is_descendant_of(task: Dict[str, Any], ancestor_id: str) -> bool:
    """True if `task` is in the subtree rooted at ancestor_id. Cheap in-memory (no I/O):
    root_task_id == ancestor_id (covers the common root-orchestrator case even when an
    INTERMEDIATE parent has already left RUNNING — a grandchild whose parent finished is
    still a descendant of the root), OR the parent_task_id chain (via RUNNING metas)
    reaches ancestor_id (covers a mid-tree ancestor while the chain is intact).
    """
    if not isinstance(task, dict) or not ancestor_id:
        return False
    if str(task.get("root_task_id") or "") == ancestor_id:
        return True
    cur = task
    hops = 0
    while isinstance(cur, dict) and hops < 25:
        pid = str(cur.get("parent_task_id") or "")
        if not pid:
            return False
        if pid == ancestor_id:
            return True
        nxt = _queue().RUNNING.get(pid)
        cur = nxt.get("task") if isinstance(nxt, dict) and isinstance(nxt.get("task"), dict) else None
        hops += 1
    return False


def _subtree_progressing(task_id: str, now: float, idle_timeout: float) -> bool:
    """True if any RUNNING descendant of task_id made real progress within idle_timeout.

    In-memory walk over RUNNING only (NO I/O — this runs under the queue lock): keeps a
    productively-waiting orchestrator alive while its children work, instead of a flat
    wall-clock kill. Descendant freshness uses last_progress_at (real progress), not the
    bare liveness heartbeat.
    """
    if not task_id:
        return False
    from ouroboros.work_diagnostics import diagnostic_ref

    for tid, m in list(_queue().RUNNING.items()):
        if tid == task_id or not isinstance(m, dict):
            continue
        if diagnostic_ref(m.get("task") or {}):
            continue  # attention to a source is not progress by that source
        if not _queue()._is_descendant_of(m.get("task") if isinstance(m.get("task"), dict) else {}, task_id):
            continue
        # Real progress only (NOT the bare 30s liveness heartbeat): a child that merely
        # pings but makes no progress must not keep its ancestor alive.
        lp = float(m.get("last_progress_at") or m.get("started_at") or 0.0)
        if lp and (now - lp) < idle_timeout:
            return True
    return False


def _has_pending_descendant(task_id: str) -> bool:
    """True if any PENDING (queued, not yet assigned) task is a descendant of task_id. A
    parent whose children are merely WAITING for worker capacity (saturation / project lease)
    is not idle/stuck — spare it (bounded by Stop, deadline, budget and any absolute ceiling) so it can integrate
    them once they run; a queued diagnostic is attention to it, not its subtree."""
    if not task_id:
        return False
    from ouroboros.work_diagnostics import diagnostic_ref

    for t in list(_queue().PENDING):
        if not isinstance(t, dict) or str(t.get("id") or "") == task_id:
            continue
        if diagnostic_ref(t):
            continue
        if _queue()._is_descendant_of(t, task_id):
            return True
    return False


def _idle_observation_pending(task_id: str, pending_events: set[str] | None) -> bool:
    from ouroboros.work_diagnostics import diagnostic_ref

    if pending_events is None or task_id in pending_events:
        return True
    # A descendant's progress is already an input to its parent's idle policy.
    # Its producer need not stamp lineage: the current RUNNING owner has it.
    return any(tid in pending_events and isinstance(meta, dict)
               and not diagnostic_ref(meta.get("task") or {})
               and _queue()._is_descendant_of(meta.get("task") or {}, task_id)
               for tid, meta in _queue().RUNNING.items())


def _enforce_task_timeouts_locked(
    workers: Any, now: float, owner_chat_id: int, st: Dict[str, Any], *,
    pending_events: set[str] | None = frozenset(),
) -> None:
    # ONE typed owner-stop predicate before the generic idle/grace consumers (S3
    # §12.2 item 8): an OPEN owner-requested finalization intent suppresses the
    # ordinary spare-withdraw/second-grace path and silence attention only while
    # the task remains inside its independent explicit deadline and absolute safety ceiling. The
    # intent stays the one owner will and cancellation custody stays the killer;
    # a later graceful request can never extend either hard axis.
    from supervisor.owner_stop import running_owner_stop_tasks
    from supervisor.restart_retention import RETAINED_FOR_BOOT
    from supervisor.worker_owner_wait import waiting_on_owner

    owner_stop_held = running_owner_stop_tasks(
        _queue().DRIVE_ROOT, grace_sec=_queue().FINALIZATION_GRACE_SEC,
    )
    for task_id, meta in list(_queue().RUNNING.items()):
        if not isinstance(meta, dict) or meta.get(RETAINED_FOR_BOOT):
            continue  # a stop already ended this attempt; its saved work belongs to the next boot
        task = meta.get("task") if isinstance(meta.get("task"), dict) else {}
        started_at = float(meta.get("started_at") or 0.0)
        if started_at <= 0:
            continue
        last_hb = float(meta.get("last_heartbeat_at") or started_at)
        # Execution time = wall clock minus quota waits minus the SEPARATE
        # budget-paused interval (#1196); started_at itself is never moved.
        runtime_sec = execution_elapsed_seconds(meta, now)
        hb_lag_sec = max(0.0, now - last_hb)
        hb_stale = hb_lag_sec >= _queue().HEARTBEAT_STALE_SEC
        _wid = meta.get("worker_id")
        worker_id = int(_wid) if _wid is not None else -1
        task_type = str(task.get("type") or "")
        _att = meta.get("attempt")
        if _att is None:
            _att = task.get("_attempt")
        attempt = int(_att) if _att is not None else 1

        deadline_ts = _queue()._task_deadline_ts(task)
        deadline_reached = bool(deadline_ts and now >= deadline_ts)

        idle_timeout = max(
            float(_queue().get_task_idle_timeout_sec()),
            float(_queue().get_per_call_timeout_ceiling_sec()) + 120.0,
        )
        # Deep review can spend a long interval reasoning between tool calls.
        # Preserve its one-hour idle tolerance; episode deadlines and absolute
        # ceilings still apply independently.
        if task_type == "deep_self_review":
            idle_timeout = max(idle_timeout, 3600.0)
        abs_ceiling = resolve_task_execution_timeout_sec(  # None = no execution bound (#939)
            task, global_ceiling=_queue().get_task_abs_ceiling_sec())
        last_progress_at = float(meta.get("last_progress_at") or started_at)
        idle_sec = max(0.0, now - last_progress_at)
        subtree_progressing = _queue()._subtree_progressing(task_id, now, idle_timeout)
        own_progress = idle_sec < idle_timeout
        # B3 external-wait lease: a held delegate_wait window over a live delegated run
        # is legitimate silence (hard-bounded by events._handle_external_wait_lease);
        # it spares ONLY this idle rail — ceiling/deadline/budget/cancel never consult it.
        lease_ts = meta.get("external_wait_lease_until")
        active_llm_call = meta.get("active_llm_call")
        llm_call_in_flight = isinstance(active_llm_call, dict) and active_llm_call.get("task_attempt") == attempt
        progressing = (own_progress or subtree_progressing or _queue()._has_pending_descendant(task_id)
                       or (isinstance(lease_ts, (int, float)) and float(lease_ts) > now)
                       or llm_call_in_flight
                       or model_waiting(meta) or waiting_on_owner(meta, attempt)
                       or _active_operation_progressing(meta, now))
        ceiling_reached = abs_ceiling is not None and runtime_sec >= float(abs_ceiling)
        if (ceiling_reached and not task.get("parent_task_id")
                and task_id == str(task.get("root_task_id") or task_id)):
            # A settled answer may still own post-task memory work in this
            # RUNNING worker. The solve ceiling cannot turn that work into a
            # failed answer; per-call, deadline and cancellation remain, and
            # silence only brings attention.
            from ouroboros.task_results import load_task_result
            from ouroboros.task_status import SETTLED_STATUSES

            try:
                # Solve settlement lives on the actor's own drive (a split root
                # settles there first; canonical copyback lags); the post-work
                # phase lives on the canonical checkpoint authority.
                settled = load_task_result(
                    _queue()._task_drive_for_task(task, str(task_id)), str(task_id)) or {}
                stored = load_task_result(_queue().DRIVE_ROOT, str(task_id)) or {}
            except Exception:
                settled, stored = {}, {}  # unreadable terminal proof never widens the ceiling
            checkpoint = stored.get("root_phase_checkpoint") or {}
            if (settled.get("status") in SETTLED_STATUSES and isinstance(checkpoint, dict)
                    and checkpoint.get("post_task_synthesis") == "running"):
                ceiling_reached = False

        if (
            str(task_id) in owner_stop_held
            and not deadline_reached
            and not ceiling_reached
        ):
            # The owner-stop episode replaces only the generic idle/grace
            # machinery.  Explicit task deadline and the absolute safety
            # ceiling remain independent hard axes and may never be extended
            # by a later graceful-stop request.
            continue

        if not ceiling_reached and not deadline_reached and _idle_observation_pending(str(task_id), pending_events):
            # Missing observation is neither progress nor permission to stop.
            # Do not date silence or withdraw a grace control on a stale mirror.
            continue

        # Hard axes (deadline_at, abs ceiling) stop the task regardless of activity; the
        # idle/subtree gate only spares a still-progressing task with NO explicit deadline —
        # an explicit/caller deadline is honored promptly, while no blanket wall-clock kills
        # a productively-waiting orchestrator.
        if not ceiling_reached and not deadline_reached and progressing:
            from supervisor.silence_attention import recovered

            recovered(_queue(), str(task_id), meta, now)
            # An outstanding episode outlives this reprieve or is withdrawn by it; the rule
            # (own progress answers the request, sparing only suspends its clock) lives with
            # the rest of the episode mechanics in task_reaper. The latch is checked here so
            # the drive resolution (which may read the result record) stays off the no-episode path.
            if meta.get("finalization_requested_at") and _resolve_grace_episode_for_spared_task(
                _queue()._task_drive_for_task(task, str(task_id)), str(task_id), meta,
                chat_id=_queue().coerce_chat_identity(task.get("chat_id"), int(owner_chat_id or 0)),
                own_progress=own_progress, now=now,
            ):
                _queue().RUNNING[task_id] = meta
            continue

        if not ceiling_reached and not deadline_reached:
            from supervisor.silence_attention import observe

            if meta.get("finalization_reason") in (REASON_ABSOLUTE_CEILING, REASON_DEADLINE):
                # The hard bound that opened this grace episode stopped holding (a lifted
                # lifetime, the root post-work exemption). Quiet is no stop condition, so its
                # finalize_now must not stay readable: the episode's owner withdraws it whole
                # or keeps the latch for the next tick. An owner-stop latch keeps its own
                # reason and owner; a spared task keeps the suspended clock above.
                from supervisor.task_reaper import withdraw_finalization_grace

                if withdraw_finalization_grace(
                    _queue()._task_drive_for_task(task, str(task_id)), str(task_id), meta,
                    chat_id=_queue().coerce_chat_identity(task.get("chat_id"), int(owner_chat_id or 0)),
                ):
                    _queue().RUNNING[task_id] = meta
            observe(_queue(), str(task_id), meta, now, idle_timeout)
            continue

        terminal_reason = REASON_ABSOLUTE_CEILING if ceiling_reached else REASON_DEADLINE
        finalization_requested_at = float(meta.get("finalization_requested_at") or 0.0)
        if finalization_requested_at <= 0 and _queue().FINALIZATION_GRACE_SEC > 0:
            meta["finalization_requested_at"] = now
            meta["finalization_reason"] = terminal_reason
            # The control's msg_id IS the episode's identity: it is what the
            # symmetric withdraw revokes, so the latch and the mailbox control
            # can never name different episodes.
            meta["finalization_control_msg_id"] = _queue()._request_finalization_grace(
                _queue()._task_drive_for_task(task, str(task_id)), str(task_id), terminal_reason,
                chat_id=_queue().coerce_chat_identity(task.get("chat_id"), int(owner_chat_id or 0)),
                stamp=int(now),
            )
            _queue().RUNNING[task_id] = meta
            continue
        if finalization_requested_at > 0 and now - finalization_requested_at < _queue().FINALIZATION_GRACE_SEC:
            continue

        # NOTE: "worker self-finalized at the timeout boundary" is handled by the reaper's
        # POST-KILL terminal re-check (kill+join FIRST, then honor an on-disk terminal
        # result, idempotent task_done). No short-circuit here: freeing the slot inline
        # would let assign_tasks reuse it mid-flight and could drop the terminal event.

        # Variant A: hand the ENTIRE teardown to the background reaper so the loop tick
        # stays fast and the terminal write happens only AFTER kill/join. Live-RUNNING
        # facts freeze HERE; neither hard bound requests a retry.
        # Linearize the timeout decision against cancellation before withdrawing
        # RUNNING.  When an intent already owns either this physical attempt or
        # its proven logical retry root, yield the whole rail: cancellation
        # custody remains the sole killer and preserves the owner's outcome and
        # reason instead of racing a generic timeout FAILED write.
        try:
            from ouroboros.cancel_intents import (
                _validated_retry_root_cancel_key,
                active_intents,
                cancellation_projection_lock,
            )

            with cancellation_projection_lock(_queue().DRIVE_ROOT):
                retry_root = _validated_retry_root_cancel_key(
                    _queue().DRIVE_ROOT, str(task_id), task_hint=task,
                )
                intents = active_intents(_queue().DRIVE_ROOT, strict=True)
                cancel_target = next(
                    (
                        candidate
                        for candidate in dict.fromkeys((str(task_id), retry_root))
                        if candidate and candidate in intents
                    ),
                    "",
                )
                if cancel_target:
                    continue
        except Exception:
            # Preserve the prior absolute/deadline timeout behavior when
            # cancellation authority itself is damaged. Neither hard bound retries.
            log.error(
                "Task timeout could not prove cancel-intent authority for %s",
                task_id,
                exc_info=True,
            )
        if task_type == "evolution":
            from supervisor.evolution_lifecycle import update_evolution_transaction
            if not update_evolution_transaction(task_id, dispatch_status="reaping"):
                log.warning("Evolution timeout teardown deferred: reaping state was not durable for %s", task_id)
                continue
        current_worker = workers.WORKERS.get(worker_id)
        if current_worker is not None and current_worker.busy_task_id != task_id:
            continue  # A stale queue row cannot hand a newer worker to teardown.
        _queue().RUNNING.pop(task_id, None)
        proc_handle = None
        captured_worker = None
        if worker_id in workers.WORKERS:
            w = workers.WORKERS[worker_id]
            captured_worker = w
            if w.busy_task_id == task_id:
                w.busy_task_id = None
            # Mark reaping under the lock so assign_tasks and the crash detector both skip
            # this slot until the reaper installs a fresh worker.
            w.reaping = True
            proc_handle = w.proc

        # Only independent hard bounds reach this memory-only reaper queue.
        # Confirmed-dead worker recovery retains its separate crash retry policy.
        _queue()._ensure_reaper_started()
        _queue()._reap_queue.put({
            "worker_id": worker_id,
            "worker": captured_worker,
            "drive_root": str(_queue().DRIVE_ROOT),
            "meta": meta,
            "proc": proc_handle,
            "task_id": str(task_id),
            "task": dict(task),
            "task_type": task_type,
            "terminal_reason": terminal_reason,
            "attempt": attempt,
            "owner_chat_id": owner_chat_id,
            "runtime_sec": runtime_sec,
            "hb_lag_sec": hb_lag_sec,
            "hb_stale": hb_stale,
            "deadline_reached": deadline_reached,
            "ceiling_reached": ceiling_reached,
            "incident_toast_once": f"{task_id}:{terminal_reason}:{int(finalization_requested_at or now)}",
        })
        _queue().persist_queue_snapshot(reason="task_timeout_reap_queued")
