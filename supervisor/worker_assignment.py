"""Handing a pending task to a free worker, and refusing the ones that must not run.

Admission order is the queue's; what this adds is the per-task gates: a cancelled
pending row is settled rather than dispatched, an evolution task without live
campaign authority is cancelled rather than started, and a repo-writing task waits
while the writer gate is closed.
"""

from __future__ import annotations

import functools
import logging
import pathlib
import time
import uuid
from typing import Any, Dict, Optional

from ouroboros.model_wait import budget_paused_seconds, native_sleep_started_at, settle_execution_carry
from supervisor.events_budget import budget_fence_selected, budget_hold_fact
from ouroboros.work_stop import continuation_refusal
from supervisor.queue import _queue_lock


def _pool():
    """The parent module, read at call time.

    The pool owns the repo/drive roots, its size, the worker table, the shared
    PENDING/RUNNING refs and the crash clock, and ``init`` REBINDS them.
    Reading them through the module is what keeps one binding: a from-import
    here would freeze the value this module saw at import time (the
    owner-approved D18/D33 mechanical exception).
    """
    from supervisor import workers

    return workers


log = logging.getLogger(__name__)


def _direct_actor_still_registered(task_id: str) -> bool:
    """Whether the in-process direct actor that paused ``task_id`` still holds its
    registry entry. A parked direct turn releases the entry as its last act; a
    grant dispatched before that would put a pooled worker beside a live actor
    under the SAME id (#1196). Unreadable registry state fences (never admits)."""
    try:
        from supervisor.active_activity import get_direct_activity_registry

        return get_direct_activity_registry().get(str(task_id or "")) is not None
    except Exception:
        log.debug("Direct activity registry unreadable during assignment", exc_info=True)
        return True


def _tick_parked_work(queue: Any) -> None:
    """Before this tick assigns: grant cold model sleeps whose own selected source
    is ready (``supervisor/sleep_wake.py``; each grant persists its own snapshot,
    a failure keeps them asleep), and re-check owner Pauses still waiting on sent
    work (``owner_pause_control.settle_requested_owner_pauses``, observe-only)."""
    # Repair a lost consumption notification from durable authority. Until
    # consumed, the live exclusion continues through capacity/persistence waits.
    from supervisor.events_budget import _handle_budget_pause

    with queue._queue_lock:
        consuming = [(task_id, meta.get("attempt"), dict(meta["task"]["_budget_pause_resume"]))
                     for task_id, meta in queue.RUNNING.items()
                     if meta.get("sleep_parked_at")
                     and ((meta.get("task") or {}).get("_budget_pause_resume") or {}).get("sleep_exclusion_since")]
    for task_id, attempt, resume in consuming:
        _handle_budget_pause({"phase": "consumed", "task_id": task_id,
            "task_attempt": attempt, "pause_id": resume.get("pause_id"),
            "grant_id": resume.get("grant_id")}, _pool())
    from supervisor.worker_owner_wait import retry_pending_owner_waits

    retry_pending_owner_waits(_pool())
    try:
        from supervisor.sleep_wake import wake_ready_sleepers

        wake_ready_sleepers(queue)
    except Exception:
        log.warning("Sleep-wake pass failed; sleepers keep sleeping", exc_info=True)
    # External stop/readback belongs to the generation's off-loop maintenance
    # owner. Assignment only advances source-backed sleep readiness here.


def _evolution_assignment_error(task: Dict[str, Any]) -> str:
    """Return the exact authority error for an evolution task about to run."""
    if str(task.get("type") or "") != "evolution":
        return ""
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    tx = metadata.get("evolution_transaction")
    tx = tx if isinstance(tx, dict) else {}
    task_id = str(task.get("id") or "")
    if str(tx.get("task_id") or "") != task_id:
        return "task_mismatch"
    from supervisor.evolution_lifecycle import check_evolution_authority

    try:
        authority = check_evolution_authority(
            campaign_id=str(tx.get("campaign_id") or ""),
            transaction_id=str(tx.get("transaction_id") or ""),
            task_id=task_id,
            require_uncommitted=True,
        )
    except Exception:
        log.warning("Evolution assignment authority check failed", exc_info=True)
        return "authority_check_failed"
    return "" if authority.get("ok") else str(authority.get("reason") or "unknown")


def _cancel_unauthorized_evolution(task: Dict[str, Any], reason: str) -> bool:
    """Terminally cancel a stale restored/retried evolution task."""
    task_id = str(task.get("id") or "")
    from ouroboros.task_results import STATUS_CANCELLED, write_task_result

    try:
        write_task_result(
            _pool().DRIVE_ROOT,
            task_id,
            STATUS_CANCELLED,
            reason_code="evolution_authority_missing",
            authority_reason=str(reason or "unknown"),
            metadata=task.get("metadata") if isinstance(task.get("metadata"), dict) else {},
            result=f"Evolution authority is no longer active ({reason or 'unknown'}).",
        )
    except Exception:
        log.debug("Failed to cancel unauthorized evolution task %s", task_id, exc_info=True)
        return False
    _pool()._emit_task_done_terminal(
        task, task_id, "cancelled", reason_code="evolution_authority_missing",
    )
    _pool().append_jsonl(
        _pool().DRIVE_ROOT / "logs" / "events.jsonl",
        {
            "ts": _pool().utc_now_iso(), "type": "evolution_assignment_rejected",
            "task_id": task_id, "reason": str(reason or "unknown"),
        },
    )
    return True


def _mirror_assigned_running_status(task: Dict[str, Any]) -> None:
    """Write the durable RUNNING mirror one step before the worker dispatches.

    Assignment is the first host-visible execution fact, and it is one for a ROOT
    too: both orphan healers key on the STORED status, so a root that exists only
    in memory and the snapshot is a ghost a stale or absent snapshot leaves
    forever. Never raises: a task that runs without its mirror is better than an
    assignment tick that stops."""
    try:
        from ouroboros.task_results import STATUS_RUNNING, write_task_result
        from ouroboros.task_status import execution_owner_record

        task["_execution_owner"] = execution_owner_record(
            task.get("budget_drive_root") or _pool().DRIVE_ROOT, task, "pooled")
        if not str(task.get("drive_root") or ""):
            return  # no fork mirror: the actual native start publishes RUNNING
        _is_subagent = str(task.get("delegation_role") or "") == "subagent"
        _mirror = {
            "execution_owner": task["_execution_owner"], "task_attempt": int(task.get("_attempt") or 0),
            "root_task_id": task.get("root_task_id"),
            "session_id": task.get("session_id"),
            "actor_id": task.get("actor_id"),
            "project_id": task.get("project_id"),
            "role": task.get("role"),
            "description": task.get("description"),
            "objective": task.get("objective") or task.get("description"),
        }
        if _is_subagent:
            from ouroboros.tools.control_delegation import stamp_task_assignment_depth
            from ouroboros.config import get_max_subagent_depth

            # Stamp the worker payload and canonical result from one
            # projection; the achieved depth is a delegation fact.
            _depth_fields = stamp_task_assignment_depth(
                task, max_depth=get_max_subagent_depth(),
            )
            _mirror.update(
                parent_task_id=task.get("parent_task_id"),
                delegation_role=task.get("delegation_role"),
                expected_output=task.get("expected_output"),
                constraints=task.get("constraints"),
                context=task.get("context"),
                memory_mode=task.get("memory_mode"),
                drive_root=task.get("drive_root"),
                child_drive_root=task.get("child_drive_root") or task.get("drive_root"),
                budget_drive_root=task.get("budget_drive_root"),
                task_constraint=task.get("task_constraint"),
                **_depth_fields,
                # INTENT ONLY. This mirror is written at ASSIGNMENT, one
                # step before the worker dispatches and resolves the
                # child; naming `effective_model_lane`/`model` here wrote
                # whatever the record happened to hold, which on a retry
                # is the PREVIOUS attempt's resolution and on a fresh
                # child is nothing at all.
                model_lane=task.get("model_lane"),
                requested_model_lane=task.get("requested_model_lane"),
                parent_model_lane=task.get("parent_model_lane"),
                requested_executor=task.get("requested_executor"),
                task_group_id=task.get("task_group_id"),
                task_group=task.get("task_group"),
                subagent_envelope=task.get("subagent_envelope"),
                configured_subagent=task.get("configured_subagent"),
                parent_cognitive_route=task.get("parent_cognitive_route"),
                metadata=task.get("metadata") if isinstance(task.get("metadata"), dict) else {},
            )
        else:
            # A root carries neither the delegation identity nor a child
            # drive. Writing None for what it lacks would ERASE what
            # admission recorded, because this write MERGES.
            _mirror["chat_id"] = task.get("chat_id")
        _mirror = {key: value for key, value in _mirror.items() if value is not None}
        def before_dispatch(current, fields):
            attempt = int(current.get("task_attempt") or 0)
            owner = current.get("execution_owner")
            same_start = (current.get("status") == STATUS_RUNNING
                          and (not owner or owner == _mirror["execution_owner"]))
            if attempt > _mirror["task_attempt"] or same_start:
                return None  # a worker/newer attempt already published; another owner is rebound
            return fields
        write_task_result(
            task.get("budget_drive_root") or _pool().DRIVE_ROOT,
            str(task.get("id") or ""),
            STATUS_RUNNING,
            _field_projector=before_dispatch, strict_existing_dict=True,
            **_mirror,
            result=("Subagent assigned to a worker." if _is_subagent
                    else "Assigned to a worker."),
        )
    except Exception:
        log.debug("Failed to mirror the running assigned status", exc_info=True)


def _settled_running_task(task: Dict[str, Any], resume: Dict[str, Any]) -> Dict[str, Any]:
    """#939: choose the clock carry on a copy, before the physical handoff."""
    running = dict(task)
    settle_execution_carry(running, origin_preserved=bool(resume.get("started_at")))
    return running


def _deadline_refuses_launch(queue, candidate, *, unsubmitted=None) -> bool:
    """#939: an explicit calendar deadline that passed before the physical handoff buys
    no worker start; an ordinary row moves to the reaper's non-dispatch custody instead.
    Rechecked at the handoff because the pass may wait on admission or persistence."""
    deadline = queue._task_deadline_ts(candidate)
    if not deadline or time.time() < deadline:
        return False
    from supervisor.queue_timeouts import pending_deadline_expirable, _retain_unsubmitted_deadline

    if unsubmitted is not None:
        _retain_unsubmitted_deadline(candidate, unsubmitted)
    if pending_deadline_expirable(candidate):
        queue._expire_pending_deadlines_locked(_pool(), time.time())
    return True  # unknown or protected custody never authorizes an expired handoff


# Memory only: launches whose dispatch marks this process wrote, or may have written, while
# the put was never entered (a later step failed or the commit's readback was unknown). The
# same queue row, attempt and marks re-read the result and may finish and put once in this
# generation, only after this commit's own marks read back; entering the put retires it, and
# so does its row leaving the queue. Until then budget and deadline stops read it as never
# dispatched (the deadline with its pre-commit claim). An unresolved one stays on its lane job and a
# process-ending stop waits for it (``workspace_preparation.pause_admission``); a crash, Panic, a stop
# that outwaits its bound, a refusal after the marks or an entered put leaves the durable ``possible``
# that restore and the occurrence reconcile never replay.
_COMMITTED_UNSENT: Dict[str, Dict[str, Any]] = {}


def _unsent(candidate) -> Optional[Dict[str, Any]]:
    """This exact queued row's witness (same row object and attempt), else None."""
    witness = _COMMITTED_UNSENT.get(str(candidate.get("id") or ""))
    return (witness if witness is not None and witness["row"] is candidate
            and witness["attempt"] == candidate.get("_attempt") else None)


def _forget_departed_launches(pending) -> None:
    """Under Q: a witness lives only while its own row is still queued."""
    live = {id(row) for row in pending}
    for task_id, witness in list(_COMMITTED_UNSENT.items()):
        if id(witness["row"]) not in live:
            _COMMITTED_UNSENT.pop(task_id, None)


def _commit_launch(queue, candidate, unsubmitted=None):
    """The durable steps before the put: the committed marks' projections; None when the launch is
    unresolved (an unknown commit or an unpersisted snapshot: this process retries it), False when refused.

    A locator snapshot first keeps the row restorable with its own dispatch fact.
    Then ONE task-result write carries the Project and schedule marks
    (``task_admission.commit_dispatch_marks``); a refusal or rolled-back write
    changes nothing, so every consumer still sees the same first dispatch. Only
    after the commit (or an unknown one) do memory and the snapshot say
    ``possible``; a later pass of the same process re-reads, rewrites under the
    same owners' predicates if the marks are absent, and continues. Legacy rows
    without a canonical mark keep the snapshot as their witness, and their prior
    value on failure. The occurrence row is marked last, just before the put
    (``_mark_launch_row``): until then the scheduler reads the pending launch as
    live and neither retires nor reclaims it.
    """
    from supervisor import schedule_occurrence as occurrences
    from supervisor.task_admission import commit_dispatch_marks, project_dispatch_projection

    task_id = str(candidate.get("id") or "")
    witness = _unsent(candidate)
    if witness is None:
        _COMMITTED_UNSENT.pop(task_id, None)  # another row or attempt of this id never inherits it
        row = occurrences.occurrence_row(candidate) if occurrences._occurrence_of(candidate) else ""
        projections = [p for p in (project_dispatch_projection(candidate),
                                   occurrences.dispatch_projection(candidate, row=row)) if p is not None]
        prior = candidate.get("admitted_dispatch")
        if not projections:
            candidate["admitted_dispatch"] = "possible"
            if queue.persist_queue_snapshot(reason="worker_launch_claimed") is not True:
                candidate["admitted_dispatch"] = prior
                return None
            return []
        if queue.persist_queue_snapshot(reason="worker_launch_locator") is not True:
            return None
        committed = commit_dispatch_marks(candidate, projections)
        if committed is False:
            return False
        candidate["admitted_dispatch"] = "possible"
        witness = _COMMITTED_UNSENT[task_id] = {"row": candidate, "attempt": candidate.get("_attempt"),
                                                "projections": projections, "unsubmitted": unsubmitted}
        if committed is None:  # it may have landed: never replayable, retried here before any put
            queue.persist_queue_snapshot(reason="worker_launch_unknown")
            return None
    else:
        from ouroboros.task_results import load_task_result

        stored = load_task_result(queue.DRIVE_ROOT, task_id, strict=True) or {}
        if not all(projection.verify(stored) for projection in witness["projections"]):
            recommitted = commit_dispatch_marks(candidate, witness["projections"])
            if recommitted is not True:  # refused now (terminal, Stop, owner hold) or unresolved: no put;
                return recommitted       # the row stays in custody
    if queue.persist_queue_snapshot(reason="worker_launch_claimed") is not True:
        return None
    return witness["projections"]


def _mark_launch_row(candidate, projections: list) -> bool:
    """The occurrence row's monotonic copy of the committed receipt mark, the last step before the put."""
    from supervisor import schedule_occurrence as occurrences

    schedule = next((p for p in projections if hasattr(p, "redispatch")), None)
    return schedule is None or occurrences.mark_occurrence_row(candidate, redispatch=schedule.redispatch)


# Memory only: the queue row's launch claim while its persistence runs on the admission lane. The
# claimed row is not chosen again and its worker is busy with it (``busy_task_id``) until the put or
# the release; nothing of it reaches the snapshot.
_LAUNCH_CLAIM = "_launch_claim"


def _claim_worker_launch(queue, candidate, worker, *, deferred: Optional[set] = None):
    """Reserve the launch under Q, persist it on the admission lane, put after revalidation.

    Under Q (the caller holds it) only memory decides: the deadline, the placement lease, the
    row's launch claim and the worker's slot (``busy_task_id``). The durable steps (the locator
    snapshot, ONE marks write, ``drive_started``, the RUNNING mirror: ``_launch_attempt``) run on
    the preparation generation's admission lane with its acquisition gate, never behind a copy;
    the lane then re-takes Q, revalidates memory only (the claim, the worker, fences, holds,
    deadline; a closed generation refused before the marks) and puts. None: reserved, the lane finishes it; False: refused, released; True: a
    host without a generation (the historical inline path) launched synchronously under Q with
    zero-wait acquisitions, contention joining ``deferred`` as residual demand. A refusal after
    the marks committed keeps ``_COMMITTED_UNSENT``'s witness: the same row, attempt and marks
    finish and put once later; the put, not the marks, is the handoff (invariant 25).
    """
    from ouroboros.owner_pause import OwnerPauseRefused
    from ouroboros.project_placement import readmit_placement
    from supervisor import workspace_preparation

    witness = _unsent(candidate)
    if (witness is None or witness["unsubmitted"] is None) and _deadline_refuses_launch(queue, candidate):
        return False
    generation = workspace_preparation.current_generation()
    try:
        if not readmit_placement(queue.DRIVE_ROOT, candidate, repo_dir=_pool().REPO_DIR,
                                running=_pool().RUNNING, pending=_pool().PENDING):
            return False
        if generation is None:
            outcome = _launch_attempt(queue, candidate, worker, "", lane=False)
            if outcome is None and deferred is not None:  # unresolved: retried here, still demand
                deferred.add(str(candidate.get("id") or ""))
            return bool(outcome)
    except OwnerPauseRefused:
        return False
    except Exception as exc:
        from supervisor.schedule_occurrence import contention

        if contention(exc):
            if deferred is not None:
                deferred.add(str(candidate.get("id") or ""))
            log.info("Worker start of %s deferred: a schedule, control or store lock is busy", candidate.get("id"))
        else:
            log.exception("Scheduled worker start refused; authority or persistence unavailable")
        return False
    if generation.launches_closed():
        return False  # a stopping generation starts nothing more
    token = uuid.uuid4().hex
    candidate[_LAUNCH_CLAIM] = {"token": token, "wid": worker.wid}
    worker.busy_task_id = str(candidate.get("id") or "")
    if not generation.submit(functools.partial(_launch_job, queue, candidate, worker, token, generation),
                             workspace_preparation.ADMISSION_LANE):
        _release_launch(queue, candidate, worker, token)
        return False
    return None


def _launch_job(queue, candidate, worker, token: str, generation) -> None:
    """The admission-lane job of the reserving GENERATION: one attempt per acquisition slice until it
    put or was refused. A closing generation withdraws a launch that wrote no marks; one whose marks
    committed stays here, unresolved attempts included, until it is put or refused: a stop waits for
    this lane (``workspace_preparation.pause_admission``), and a launch released to a later assignment
    pass would be outside that wait. A refusal or an error after the marks still ends the job."""
    from ouroboros._usage_wait import ACQUISITION_SLICE_SEC
    from supervisor import workspace_preparation
    from supervisor.schedule_occurrence import contention

    def attempt():
        return generation.wait_store(
            functools.partial(_launch_attempt, queue, candidate, worker, token, lane=True, generation=generation),
            controls=lambda: ({"detail": "generation closed"}
                              if generation.launches_closed() and _unsent(candidate) is None else None),
            busy=contention)

    outcome = False
    try:
        outcome = attempt()
        if outcome is None and _unsent(candidate) is not None:
            log.info("Worker start of %s committed but unresolved; retrying it on the admission lane",
                     candidate.get("id"))
        while outcome is None and _unsent(candidate) is not None:
            time.sleep(ACQUISITION_SLICE_SEC)
            outcome = attempt()
        if outcome is None:
            log.info("Worker start of %s unresolved; the next assignment pass retries it", candidate.get("id"))
    except workspace_preparation.PreparationStopped:
        log.info("Worker start of %s withdrawn: the supervisor generation is closing", candidate.get("id"))
    except Exception:
        log.exception("Scheduled worker start refused; authority or persistence unavailable")
    finally:
        if outcome is not True:
            _release_launch(queue, candidate, worker, token)


def _started_projection() -> list:
    """Under Q: RUNNING plus launches reserved on the admission lane; caps and leases count both,
    as when the inline launch bound RUNNING before the next choice."""
    return list(_pool().RUNNING.values()) + [{"task": row} for row in _pool().PENDING if row.get(_LAUNCH_CLAIM)]


def _release_launch(queue, candidate, worker, token: str) -> None:
    """Under Q: the row returns to residual demand and the slot frees (a reaping slot keeps its id
    for the reaper's identity check; its replacement starts free). Only this claim's own holder
    releases: once the put retired the claim, the worker holds the task even if the job raised."""
    with queue._queue_lock:
        if (candidate.get(_LAUNCH_CLAIM) or {}).get("token") != token:
            return
        candidate.pop(_LAUNCH_CLAIM, None)
        if worker.busy_task_id == str(candidate.get("id") or "") and not getattr(worker, "reaping", False):
            worker.busy_task_id = None


def _launch_claim_held(queue, candidate, worker, token: str) -> bool:
    """Under Q: the row is still queued with this claim and the slot is still this live, ready worker."""
    from supervisor.worker_pool_lifecycle import worker_ready

    return ((candidate.get(_LAUNCH_CLAIM) or {}).get("token") == token
            and any(row is candidate for row in _pool().PENDING)
            and _pool().WORKERS.get(worker.wid) is worker and worker.busy_task_id == str(candidate.get("id") or "")
            and worker_ready(worker) and getattr(worker, "active_capacity", True))


def _launch_attempt(queue, candidate, worker, token: str, *, lane: bool, generation=None) -> Optional[bool]:
    """One launch attempt: the schedule control, the durable steps under the root's launch lock,
    then the put under Q and the schedule's table hold.

    No put before every durable step lands (``_commit_launch``, ``_mark_launch_row``). The start
    fence tries its table, control and store acquisitions once (``automatic_acquisitions``):
    contention, an unresolved commit or an unwritten row mark (None: still demand, this process
    retries it) ends the attempt with the same unsent row (the lane retries after a slice; the
    inline path defers to the next pass); False is a refusal. On the lane the put follows a
    memory-only revalidation under Q. A closed reserving generation refuses before the marks; marks
    it committed while open finish their put. Once the put is entered the worker
    owns the task, even if the put raised: it is bound RUNNING/busy, so Stop, completion and
    worker health settle it before reuse.
    """
    import copy
    from types import SimpleNamespace
    from ouroboros.owner_pause import launch_admission
    from ouroboros.task_results import load_task_result
    from supervisor.followup_policy import scheduled_start
    from supervisor.queue_schedules import automatic_acquisitions
    from supervisor.queue_timeouts import _pending_continuation, _deadline_launch_claim, pending_deadline_expirable
    from ouroboros.work_diagnostics import control_reason

    witness = _unsent(candidate)
    with automatic_acquisitions():
        diagnostic_stop = control_reason(candidate, _pool().DRIVE_ROOT, before_handoff=True)
        if diagnostic_stop:
            with queue._queue_lock:
                held = (_launch_claim_held(queue, candidate, worker, token) if lane
                        else any(row is candidate for row in _pool().PENDING))
            if held and diagnostic_stop == "cancelled" and (candidate.get("admitted_dispatch") == "none" or witness is not None):
                # This launch owner proves it never entered the put. A recovered
                # source may have published while we held its diagnostic claim.
                from ouroboros.cancel_intents import request_cancel

                request_cancel(_pool().DRIVE_ROOT, candidate["id"], reason="Diagnostic source authority ended before dispatch",
                               source="silence_resolved", requested_by="supervisor", scope="single")
            return False
        # The schedule owner's control (origin Stop, hold, binding) is read before any durable
        # step, under its own short table hold; the put re-reads it inside that hold below.
        with scheduled_start(queue.DRIVE_ROOT, candidate) as allowed:
            if not allowed:
                return False
        with queue._queue_lock:
            if lane and not _launch_claim_held(queue, candidate, worker, token):
                return False
            root_id = str(candidate.get("root_task_id") or candidate.get("id"))
            latch = queue.BUDGET_ROOT_FENCES.get(root_id) or {}
            resume = candidate.get("_budget_pause_resume")
            # A monetary latch hands over only an explicit selection against it (an
            # owner-resumed cold sleeper must start), never sleep readiness.
            selected_child = bool(candidate.get("id") != root_id and isinstance(resume, dict)
                                  and (latch.get("cause") == "owner_pause" or budget_fence_selected(candidate, latch))
                                  and resume.get("root_fence_id") == latch.get("fence_id"))
        # The root's launch lock and its control reads (owner Pause, sleep grants) wait for nothing
        # and run OUTSIDE Q; Q-holders take this lock non-blocking, so the order never blocks.
        with launch_admission(SimpleNamespace(
                task_id=candidate.get("id"), root_task_id=candidate.get("root_task_id"),
                budget_drive_root=candidate.get("budget_drive_root") or queue.DRIVE_ROOT),
                root_resume=resume if not latch or selected_child else None, wait=False):
            unsubmitted = None if witness is None else witness["unsubmitted"]
            with queue._queue_lock:
                if generation is not None and generation.launches_closed() and witness is None:
                    return False  # a stopping generation writes no marks: the row stays never-started
                if unsubmitted is not None and _deadline_refuses_launch(queue, candidate, unsubmitted=unsubmitted):
                    return False  # a committed, never-put launch expires on its own pre-commit claim
                claim_source = (witness is None and queue._task_deadline_ts(candidate)
                                and _pending_continuation(candidate) == "_working_recovery"
                                and pending_deadline_expirable(candidate))
            if claim_source:
                # Before this invocation claims: retain its exact authorized
                # source and prior owner, not an inference from persisted possible.
                stored = load_task_result(candidate.get("budget_drive_root") or queue.DRIVE_ROOT,
                                          candidate["id"], strict=True) or {}
                unsubmitted = copy.deepcopy({"task_id": candidate["id"], "task_attempt": candidate["_attempt"],
                    "source": candidate["_working_recovery"], "claim": _deadline_launch_claim(stored)})
            from ouroboros.server_restart import agent_restart_boundary, agent_restart_holds

            if not agent_restart_boundary.acquire(blocking=False):
                return None  # the inline caller may own Q: never wait behind an off-lane commit
            try:
                with queue._queue_lock:
                    if witness is None and agent_restart_holds(candidate):
                        return False
                projections = _commit_launch(queue, candidate, unsubmitted)
            finally:
                agent_restart_boundary.release()
            if projections is None or projections is False:
                return projections  # unresolved (None) or refused (False): the same row stays queued
            from ouroboros.obligations import drive_started
            drive_started(_pool().DRIVE_ROOT, candidate)
            _mirror_assigned_running_status(candidate)
            with scheduled_start(queue.DRIVE_ROOT, candidate) as allowed, queue._queue_lock:
                if not allowed:
                    return False
                if lane and not (_launch_claim_held(queue, candidate, worker, token)
                                 and not _candidate_ineligible(queue, candidate, float("inf"),
                                                               list(_pool().RUNNING.values()), own_claim=token)):
                    return False
                # Marks written while the generation was open complete their handoff, as at base: a
                # refusal here would leave only a memory witness, and restore would hold the row.
                if _deadline_refuses_launch(queue, candidate, unsubmitted=unsubmitted):  # positively before any put
                    return False
                if not _mark_launch_row(candidate, projections):
                    return None  # committed, row mark unwritten: the same process retries it
                continuation = candidate.get("_owner_wait_resume") or candidate.get("_budget_pause_resume") or candidate.get("_working_recovery") or {}
                launched = _settled_running_task(candidate, continuation)
                # The handoff boundary: from here this worker may hold the task.
                _COMMITTED_UNSENT.pop(str(candidate.get("id") or ""), None)
                worker.quiescence_token = ""
                worker.quiescence_confirmed = False
                try:
                    worker.in_q.put(launched)
                except Exception:
                    log.exception("Handoff of %s raised after entering the worker queue; it stays bound to "
                                  "worker %s until that worker reports or is reaped", candidate.get("id"),
                                  getattr(worker, "wid", "?"))
                candidate.pop("_execution_handoff_sec", None)
                candidate.update(launched)
                if lane:
                    candidate.pop(_LAUNCH_CLAIM, None)
                    _bind_running(queue, candidate, worker)
                return True


def _bind_running(queue, task, worker) -> None:
    """Under Q, after the put: the row leaves PENDING, the worker and RUNNING own it, the snapshot says so."""
    _pool().PENDING[:] = [row for row in _pool().PENDING if row is not task]
    worker.busy_task_id = task["id"]
    now_ts = time.time()
    resume = task.get("_owner_wait_resume") or task.get("_budget_pause_resume") or task.get("_working_recovery") or {}
    sleeping = resume.get("sleep_exclusion_since") or native_sleep_started_at(task.get("_owner_wait_resume"))
    _pool().RUNNING[task["id"]] = {
        "task": _settled_running_task(task, resume), "worker_id": worker.wid,
        "started_at": float(resume.get("started_at") or now_ts), "last_heartbeat_at": now_ts,
        "last_progress_at": now_ts,
        **({"model_wait_quota_clock": dict(resume["model_wait_quota_clock"])}
           if resume.get("model_wait_quota_clock") else {}),
        # Separate paused-interval carrier (#1196): the original started_at is untouched;
        # lifetime rails subtract this. ONE reader for either handoff: a budget grant names it
        # ``paused_duration_sec``, an owner-wait restart ``budget_paused_sec``.
        "budget_paused_sec": budget_paused_seconds(resume),
        **({"sleep_parked_at": float(sleeping)} if sleeping is not None else {}),
        "soft_sent": False, "attempt": int(task.get("_attempt") or 1),
    }
    # A started task publishes no row: its own card says so (the agent's `task_started` live event).
    queue.persist_queue_snapshot(reason="assign_task")


def _foreign_roots_refused(candidate) -> bool:
    """A restored row whose stored roots name another install never launches here (an explicit move
    makes it ours); its own task row names the other install once, written on the admission lane."""
    from ouroboros.install_identity import disclose_foreign_task, foreign_task_refusal

    refusal = foreign_task_refusal(candidate)
    if refusal is None:
        candidate.pop("_foreign_root_disclosed", None)
        return False
    if candidate.get("_foreign_root_disclosed") != str(refusal):
        candidate["_foreign_root_disclosed"] = str(refusal)
        from supervisor import workspace_preparation

        generation = workspace_preparation.current_generation()
        if generation is not None and not generation.closed():
            generation.submit(functools.partial(disclose_foreign_task, _pool().DRIVE_ROOT,
                                                str(candidate.get("id") or ""), refusal),
                              workspace_preparation.ADMISSION_LANE)
    return True


def _candidate_ineligible(queue, candidate, remaining, running, *, selected=(), refused=(), cold_holders=None,
                          leased=None, unresolved_invalid_id_set=(), own_claim: str = "") -> bool:
    """The chooser's gates over one queued row; also the launch's revalidation under Q (``own_claim``:
    the revalidating launch's own token, the one claim that does not exclude the row). Side-effect
    free except that a foreign-root refusal is named once, by a job on the admission lane."""
    from ouroboros.project_lease import candidate_is_leasable, cold_project_holders, running_project_ids
    from ouroboros.project_placement import refresh_pending_placement
    from ouroboros.config import get_max_active_subagents_per_root
    from supervisor.events_budget import budget_resume_dispatch_allowed
    from supervisor.state import EVOLUTION_BUDGET_RESERVE
    from supervisor.task_lifecycle import root_cancellation_fenced
    from supervisor.consciousness_carriers import autonomous_work_enabled

    task_id = str(candidate.get("id") or "")
    root_task_id = str(candidate.get("root_task_id") or "").strip()
    claim = candidate.get(_LAUNCH_CLAIM) or {}
    if (task_id in set(refused) | set(selected) or candidate.get("_owner_hold")
            or candidate.get("_project_admission_restore_hold") or claim and claim.get("token") != own_claim):
        return True
    if _foreign_roots_refused(candidate):
        return True
    if not autonomous_work_enabled(candidate, _pool().DRIVE_ROOT):
        return True
    if remaining <= 0 and not candidate.get("_owner_wait_resume"):
        return True
    if _pool()._invalid_depth_deferred(candidate, set(unresolved_invalid_id_set)):
        return True
    if not _pool().repo_writer_task_allowed(candidate):
        return True
    if (isinstance(candidate.get("_budget_pause"), dict) or budget_hold_fact(candidate) is not None
            or continuation_refusal(candidate)):
        # Durable #1196 hold: sibling unselected after root release, unrestorable
        # continuation or failed grant revocation. Selection must be recorded before assignment.
        return True
    if _direct_actor_still_registered(task_id):
        # The direct actor that parked this id has not released its registry entry yet:
        # no second live actor for one id.
        return True
    if root_cancellation_fenced(candidate, root_task_id or task_id):
        return True
    if budget_resume_dispatch_allowed(queue, candidate) is not True:
        return True  # unknown holds; a stale selection is the chooser's to repair
    if (root_task_id in queue.BUDGET_ROOT_FENCES
            and not candidate.get("_owner_wait_resume")
            and not candidate.get("_budget_pause_resume")
            # One member explicitly selected against THIS fence is admitted; the latch stays up for the rest (Q9).
            and not budget_fence_selected(candidate, queue.BUDGET_ROOT_FENCES.get(root_task_id))):
        return True
    if str(candidate.get("type") or "") == "evolution" and remaining < EVOLUTION_BUDGET_RESERVE:
        return True
    if not refresh_pending_placement(_pool().DRIVE_ROOT, candidate):
        return True
    cold = cold_holders if cold_holders is not None else cold_project_holders(_pool().PENDING)
    if not candidate_is_leasable(candidate, leased if leased is not None else running_project_ids(running),
                                 cold_holders=cold):
        return True
    if str(candidate.get("delegation_role") or "") == "subagent":
        if (_pool()._running_subagent_count(root_task_id, running) >= get_max_active_subagents_per_root()
                and not _pool()._assignment_depth_reservation_admits(candidate, running)):
            return True
    return False


def _choose_candidate(queue, remaining, unresolved_invalid_id_set, refused_this_pass, running, selected):
    """One side-effecting chooser, including stale grant repair with no idle worker.

    `running` is either current custody or a local projection for a bounded batch.
    Selection never claims dispatch, consumes a grant, or creates RUNNING custody.
    """
    from ouroboros.project_lease import cold_project_holders, running_project_ids
    from supervisor.state import EVOLUTION_BUDGET_RESERVE

    leased = running_project_ids(running)
    while True:
        # Accepted cold external writers keep their lane; PENDING can shrink below.
        cold_holders = cold_project_holders(_pool().PENDING)
        chosen_idx = None
        for i, candidate in enumerate(_pool().PENDING):
            from ouroboros.server_restart import agent_restart_holds
            from supervisor.events_budget import budget_resume_dispatch_allowed

            if _unsent(candidate) is None and agent_restart_holds(candidate):
                continue

            resume_allowed = budget_resume_dispatch_allowed(queue, candidate)
            # A row its own exact pause or hold keeps queued is not repaired: the selection a paused
            # row still carries from its earlier dispatch is no stale zero-dispatch choice, and
            # re-holding it would replace the Pause's projected reason with that old hold's.
            if (resume_allowed is False and not candidate.get("_owner_hold")
                    and not candidate.get("_project_admission_restore_hold") and not candidate.get(_LAUNCH_CLAIM)
                    and not isinstance(candidate.get("_budget_pause"), dict) and budget_hold_fact(candidate) is None):
                if isinstance(candidate.get("_budget_pause_resume"), dict):
                    from supervisor.budget_resume import revoke_exact_budget_resume

                    revoke_exact_budget_resume(candidate, "root_resume_generation_stale")
                else:
                    from supervisor.events_budget import HOLD_ROOT_FENCE_LIFTED, hold_budget_row
                    from ouroboros.task_results import resolve_task_lineage

                    candidate.pop("_resume_selection", None)
                    lineage = resolve_task_lineage(candidate.get("id"), **{key: candidate.get(key) for key in (
                        "metadata", "root_task_id", "parent_task_id", "delegation_role",
                        "original_task_id", "timeout_retry_from")})
                    if not lineage["is_root_task"]:
                        root_id = lineage["root_task_id"]
                        latch = queue.BUDGET_ROOT_FENCES.get(root_id) or {}
                        hold_budget_row(candidate, reason=HOLD_ROOT_FENCE_LIFTED,
                            detail="root_resume_generation_stale",
                            extra={"root_task_id": root_id, "fence_id": str(latch.get("fence_id") or "")},
                            result_root=pathlib.Path(candidate.get("budget_drive_root") or _pool().DRIVE_ROOT))
                queue.persist_queue_snapshot(reason="stale_child_resume_held")
                continue
            if _candidate_ineligible(queue, candidate, remaining, running, selected=selected,
                                     refused=refused_this_pass, cold_holders=cold_holders, leased=leased,
                                     unresolved_invalid_id_set=unresolved_invalid_id_set):
                continue
            chosen_idx = i
            break
        if chosen_idx is None:
            # Project-leased rows wait; evolution rows below the reserve are cleaned.
            # At zero the pause block already parked or settled each row: keep it.
            if 0 < remaining < EVOLUTION_BUDGET_RESERVE:
                dropped_ids = _pool()._drop_assignable_evolution_tasks(unresolved_invalid_id_set)
                if dropped_ids:
                    queue.persist_queue_snapshot(reason="evolution_dropped_budget")
            return None, False
        task = _pool().PENDING[chosen_idx]
        depth_error = _pool()._normalize_pending_task_depth(task)
        if depth_error:
            if _pool()._terminalize_invalid_pending_depth(task, depth_error):
                _pool().PENDING.pop(chosen_idx)
                queue.persist_queue_snapshot(reason="invalid_task_depth")
                continue
            # Keep failed terminalization in custody; stop this pass.
            log.error(
                "Assignment blocked: invalid task depth could not be terminalized for %s",
                task.get("id"),
            )
            return None, True
        evolution_error = _pool()._evolution_assignment_error(task)
        if evolution_error:
            refused_this_pass.add(str(task.get("id") or ""))
            if _pool()._cancel_unauthorized_evolution(task, evolution_error):
                _pool().PENDING.pop(chosen_idx)
                queue.persist_queue_snapshot(reason="evolution_authority_rejected")
            continue
        return chosen_idx, False


def assign_tasks() -> None:
    from supervisor import queue
    from supervisor.state import budget_remaining, EVOLUTION_BUDGET_RESERVE
    from supervisor.worker_owner_wait import maintain_owner_wait_capacity

    from supervisor.worker_pool_lifecycle import _WORKER_LIFECYCLE_LOCK, reconcile_worker_capacity, worker_ready

    # The loop never waits behind an off-loop respawn or generation stop holding the lifecycle
    # serializer (teardowns kill and join outside it): the whole turn defers to the next tick, so
    # fresh work cannot overtake a Resume, and growth (below) is recomputed next tick the same way.
    if not _WORKER_LIFECYCLE_LOCK.acquire(blocking=False):
        return
    try:
        maintain_owner_wait_capacity()
    finally:
        _WORKER_LIFECYCLE_LOCK.release()
    _tick_parked_work(queue)  # custody observation may wait on a remote operation
    with _queue_lock:
        st = _pool().load_state()
        # Custody wins; quarantine malformed depth before budget/lease filters. This queue
        # hygiene runs under a fenced pool too: only assignment and growth wait for Restart.
        if not _pool()._drop_cancelled_pending():
            log.error("Task assignment blocked: cancellation authority or custody state is indeterminate")
            queue.persist_queue_snapshot(reason="cancellation_authority_indeterminate")
            return
        _pool()._retry_terminalization_pending_for_assignment(queue)
        from supervisor.task_admission import revalidate_project_holds
        revalidate_project_holds()
        _forget_departed_launches(_pool().PENDING)
        invalid_ids, unresolved_invalid_ids = _pool()._quarantine_invalid_pending_depths()
        unresolved_invalid_id_set = set(unresolved_invalid_ids)

        if invalid_ids:
            queue.persist_queue_snapshot(reason="invalid_task_depth")
        if unresolved_invalid_ids:
            log.error(
                "Invalid-depth rows deferred until terminal custody is available; continuing assignment for other tasks: %s",
                ", ".join(unresolved_invalid_ids),
            )
        if _pool()._WORKER_POOL_DISABLED_REASON:
            return
        try:  # every loop tick: ride the snapshot; a refusal re-reads exactly, reserve_attempt is the gate
            evolution_queued = any(str(row.get("type") or "") == "evolution" for row in _pool().PENDING)
            remaining = budget_remaining(  # this tick refuses at zero AND, for evolution, at its reserve
                st, strict=True, allow_stale=True,
                refuse_below=EVOLUTION_BUDGET_RESERVE if evolution_queued else 0.0)
        except Exception:
            log.error("Task assignment blocked: monetary authority unavailable")
            return
        if remaining <= 0:
            planned = []
            for task in _pool().PENDING:
                if _pool()._invalid_depth_deferred(task, unresolved_invalid_id_set):
                    continue
                if isinstance(task.get("_budget_pause"), dict):
                    continue
                if budget_hold_fact(task) is not None or continuation_refusal(task):
                    continue  # Held, not paused: no second pause row over the hold.
                if task.get("_owner_wait_resume"):
                    continue  # Restore the checkpoint; the loop still owns its budget stop.
                if isinstance(task.get("_budget_pause_resume"), dict):
                    # Money vanished between the grant and this dispatch: the
                    # single-use grant returns to its exact pause, never to a
                    # pre-dispatch replay or a terminal.
                    from supervisor.queue_transitions import revoke_exact_budget_resume

                    revoke_exact_budget_resume(task, "budget_exhausted_before_dispatch")
                    queue.persist_queue_snapshot(reason="budget_exact_resume_revoked")
                    continue
                task_id = str(task.get("id") or "")
                cost_fields = _pool().reconstruct_task_cost(
                    task_id, fields=True,
                    drive_root=pathlib.Path(task.get("budget_drive_root") or _pool().DRIVE_ROOT),
                )
                if cost_fields.get("cost_accounting_status") != "available":
                    log.error("Budget pause blocked: task attempt history unavailable for %s", task_id)
                    return
                retry_lineage = bool(
                    int(task.get("_attempt") or 1) > 1
                    or task.get("original_task_id") or task.get("timeout_retry_from")
                )
                replay_safe = (
                    (task.get("admitted_dispatch") == "none" or _unsent(task) is not None)  # never put here
                    and int(cost_fields.get("total_rounds") or 0) == 0
                    and not bool(cost_fields.get("ledger_integrity_degraded"))
                    and not retry_lineage
                )
                pause = {
                    "status": "paused_before_dispatch" if replay_safe else "resource_limited",
                    "scope": "global",
                    "physical_calls": int(cost_fields.get("total_rounds") or 0),
                    "replay_safe": replay_safe,
                    "auto_resume": False,
                    "resume_policy": "manual_same_generation" if replay_safe else "cancel_or_new_run",
                    "paused_at": _pool().utc_now_iso(),
                }
                planned.append((task, pause, cost_fields))
            newly_paused, terminal_ids = [], []
            for task, pause, cost_fields in planned:
                task_id = str(task.get("id") or "")
                result_root = pathlib.Path(task.get("budget_drive_root") or _pool().DRIVE_ROOT)
                try:
                    from ouroboros.task_results import STATUS_FAILED, STATUS_SCHEDULED, write_task_result

                    if pause["replay_safe"]:
                        task["_budget_pause"] = pause
                        newly_paused.append(task_id)
                        write_task_result(
                            result_root, task_id, STATUS_SCHEDULED,
                            reason_code="budget_exhausted", resource_limit=pause,
                        )
                    else:
                        write_task_result(
                            result_root, task_id, STATUS_FAILED,
                            reason_code="budget_exhausted", resource_limit=pause,
                            result="Budget exhausted after prior dispatch; cancel or start a new run.",
                            **cost_fields,
                        )
                        _pool()._emit_task_done_terminal(
                            task, task_id, "failed", reason_code="budget_exhausted",
                            cost_fields=cost_fields,
                        )
                        terminal_ids.append(task_id)
                except Exception:
                    log.error("Failed to project budget stop for %s", task_id, exc_info=True)
            if terminal_ids:
                terminal = set(terminal_ids)
                _pool().PENDING[:] = [task for task in _pool().PENDING if str(task.get("id") or "") not in terminal]
            if newly_paused or terminal_ids:
                _pool().append_jsonl(
                    _pool().DRIVE_ROOT / "logs" / "events.jsonl",
                    {
                        "ts": _pool().utc_now_iso(),
                        "type": "budget_tasks_paused",
                        "scope": "global",
                        "task_ids": newly_paused,
                        "resource_limited_task_ids": terminal_ids,
                        "auto_resume": False,
                    },
                )
                if st.get("owner_chat_id"):
                    _pool().send_with_budget(
                        int(st["owner_chat_id"]),
                        "🚫 Model budget reached. Queued tasks are paused before dispatch; "
                        "raising the limit does not resume them automatically.",
                        role="system", system_type="budget_notice")
                queue.persist_queue_snapshot(reason="budget_paused_before_dispatch")
            # Selection still runs (and repairs stale grants); quiescent capacity
            # maintenance also remains possible after money admission closes.

        # Evolution is hard-blocked in light runtime mode at the assignment
        # chokepoint too: a task restored from a snapshot or created before the
        # mode switch must never actually run. Cancel them terminally.
        from supervisor.evolution_lifecycle import evolution_block_reason
        evo_block = evolution_block_reason()
        blocked_ids = _pool()._drop_assignable_evolution_tasks(unresolved_invalid_id_set) if evo_block else []
        if blocked_ids:
            from ouroboros.task_results import STATUS_CANCELLED, write_task_result
            for tid in blocked_ids:
                try:
                    write_task_result(
                        _pool().DRIVE_ROOT, tid, STATUS_CANCELLED,
                        result="Evolution is disabled in light runtime mode.",
                    )
                except Exception:
                    log.debug("Failed to cancel light-mode evolution task %s", tid, exc_info=True)
            if st.get("owner_chat_id"):
                _pool().send_with_budget(int(st["owner_chat_id"]), evo_block, role="system", system_type="evolution_notice")
            queue.persist_queue_snapshot(reason="evolution_blocked_light")

        refused_this_pass = set()  # failed physical claims are not residual demand,
        deferred_this_pass = set()  # unless contention deferred them: the same work retries
        for w in _pool().WORKERS.values():
            while (w.busy_task_id is None and worker_ready(w) and not getattr(w, "resume_hold", "")
                    and getattr(w, "active_capacity", True) and _pool().PENDING):
                chosen_idx, blocked = _choose_candidate(
                    queue, remaining, unresolved_invalid_id_set, refused_this_pass, _started_projection(), set())
                if blocked:
                    return
                if chosen_idx is None:
                    break
                task = _pool().PENDING[chosen_idx]
                # Keep PENDING custody through preparation and the final handoff.
                outcome = _claim_worker_launch(queue, task, w, deferred=deferred_this_pass)
                if outcome is False:
                    # Keep custody and identity; try another candidate on THIS
                    # available worker without rechecking the refusal this pass.
                    refused_this_pass.add(str(task.get("id") or ""))
                    continue
                if outcome is True:  # the inline path put it; bind it here under the same Q hold
                    _bind_running(queue, task, w)
                # None: reserved, the admission lane persists and puts it; the slot is taken either way.

    # Recompute under lifecycle -> Q before spawning, without holding Q over start.
    def residual_demand():
        running, selected = _started_projection(), set()
        for _ in range(max(1, _pool().MAX_WORKERS)):
            idx, blocked = _choose_candidate(queue, remaining, unresolved_invalid_id_set,
                                             refused_this_pass - deferred_this_pass, running, selected)
            if blocked:
                return None
            if idx is None:
                break
            candidate = _pool().PENDING[idx]
            selected.add(str(candidate.get("id") or ""))
            running.append({"task": candidate})
        return len(selected)

    if _WORKER_LIFECYCLE_LOCK.acquire(blocking=False):
        try:
            reconcile_worker_capacity(residual_demand)
        finally:
            _WORKER_LIFECYCLE_LOCK.release()
