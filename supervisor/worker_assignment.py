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
from ouroboros.task_carrier import EXECUTOR_WORKER, LANE_CONVERSATION, LANE_PROJECT
from supervisor.events_budget import budget_fence_selected, budget_hold_fact
from ouroboros.work_stop import continuation_refusal
from supervisor.queue import _queue_lock


def _pool():
    """The parent module, read at call time.

    The pool owns the repo/drive roots, its size, the worker table and the shared
    PENDING/RUNNING refs, and ``init`` REBINDS them.
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
    assignment tick that stops. A returned owner turn, which only the conversation
    lane's carrier takes (``_assign_returned_turns``), keeps its direct origin,
    stamped as an admission stamps it."""
    try:
        from ouroboros.task_results import STATUS_RUNNING, write_task_result
        from ouroboros.task_status import execution_owner_record
        from supervisor.conversation_lane import resumes_on_lane

        root = task.get("budget_drive_root") or _pool().DRIVE_ROOT
        task["_execution_owner"] = ({**execution_owner_record(root, task, "direct"), "lane": LANE_CONVERSATION,
                                     "executor": EXECUTOR_WORKER} if resumes_on_lane(task)
                                    else execution_owner_record(root, task, "pooled"))
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
            if queue.publish_queue_snapshot(reason="worker_launch_claimed") is not True:  # durable before the put
                candidate["admitted_dispatch"] = prior
                return None
            return []
        if queue.publish_queue_snapshot(reason="worker_launch_locator") is not True:
            return None
        committed = commit_dispatch_marks(candidate, projections)
        if committed is False:
            return False
        candidate["admitted_dispatch"] = "possible"
        witness = _COMMITTED_UNSENT[task_id] = {"row": candidate, "attempt": candidate.get("_attempt"),
                                                "projections": projections, "unsubmitted": unsubmitted}
        if committed is None:  # it may have landed: never replayable, retried here before any put
            queue.publish_queue_snapshot(reason="worker_launch_unknown")
            return None
    else:
        from ouroboros.task_results import load_task_result

        stored = load_task_result(queue.DRIVE_ROOT, task_id, strict=True) or {}
        if not all(projection.verify(stored) for projection in witness["projections"]):
            recommitted = commit_dispatch_marks(candidate, witness["projections"])
            if recommitted is not True:  # refused now (terminal, Stop, owner hold) or unresolved: no put;
                return recommitted       # the row stays in custody
    if queue.publish_queue_snapshot(reason="worker_launch_claimed") is not True:  # durable before the put
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
    as when the inline launch bound RUNNING before the next choice (a returned turn's, as its lane's row)."""
    from supervisor.conversation_lane import resumes_on_lane

    return list(_pool().RUNNING.values()) + [{"task": row, **({"lane": LANE_CONVERSATION} if resumes_on_lane(row)
                                                               else {})}
                                             for row in _pool().PENDING if row.get(_LAUNCH_CLAIM)]


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
    then the put under Q and the schedule's table hold. The chooser read the same gates with the same functions
    (``dispatch_choice``); these reads are the ones that authorize, each under its own owner's hold.

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
    from supervisor.dispatch_choice import _root_resume, resume_admits, resume_read

    witness = _unsent(candidate)
    with automatic_acquisitions():
        # The schedule owner's control (origin Stop, hold, binding) is read before any durable
        # step, under its own short table hold; the put re-reads it inside that hold below.
        with scheduled_start(queue.DRIVE_ROOT, candidate) as allowed:
            if not allowed:
                return False
        with queue._queue_lock:
            if lane and not _launch_claim_held(queue, candidate, worker, token):
                return False
            resume = _root_resume(queue, candidate)
        # The root's launch lock and its control reads (owner Pause, sleep grants) wait for nothing
        # and run OUTSIDE Q; Q-holders take this lock non-blocking, so the order never blocks.
        with launch_admission(SimpleNamespace(
                task_id=candidate.get("id"), root_task_id=candidate.get("root_task_id"),
                budget_drive_root=candidate.get("budget_drive_root") or queue.DRIVE_ROOT),
                root_resume=resume, wait=False):
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
                if witness is None and agent_restart_holds(candidate):  # its store read inside the boundary, off Q
                    return False
                projections = _commit_launch(queue, candidate, unsubmitted)
            finally:
                agent_restart_boundary.release()
            if projections is None or projections is False:
                return projections  # unresolved (None) or refused (False): the same row stays queued
            from ouroboros.obligations import drive_started
            drive_started(_pool().DRIVE_ROOT, candidate)
            _mirror_assigned_running_status(candidate)
            selection = resume_read(queue, candidate) if lane else None  # C2's reads for the put, before Q
            with scheduled_start(queue.DRIVE_ROOT, candidate) as allowed, queue._queue_lock:
                if not allowed:
                    return False
                if lane and not (_launch_claim_held(queue, candidate, worker, token)
                                 and not _candidate_ineligible(queue, candidate, float("inf"),
                                                               list(_pool().RUNNING.values()), own_claim=token)
                                 and resume_admits(queue, candidate, selection)):
                    return False
                # Marks written while the generation was open complete their handoff, as at base: a
                # refusal here would leave only a memory witness, and restore would hold the row.
                if _deadline_refuses_launch(queue, candidate, unsubmitted=unsubmitted):  # positively before any put
                    return False
                if not _mark_launch_row(candidate, projections):
                    return None  # committed, row mark unwritten: the same process retries it
                continuation = candidate.get("_owner_wait_resume") or candidate.get("_budget_pause_resume") or candidate.get("_working_recovery") or {}
                launched = _settled_running_task(candidate, continuation)
                # Each physical handoff gets a fresh phase-generation token.  Attempt and
                # worker identity can both be reused by Resume, so they cannot fence a
                # buffered phase frame from the previous execution.
                launched["_phase_generation"] = uuid.uuid4().hex
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
    """Under Q, after the put: the row leaves PENDING, the worker and RUNNING own it, the snapshot says so.
    A conversation carrier's row takes the lane's facts (``conversation_lane.returned_row_facts``)."""
    from supervisor.conversation_lane import returned_row_facts

    _pool().PENDING[:] = [row for row in _pool().PENDING if row is not task]
    worker.busy_task_id = task["id"]
    now_ts = time.time()
    resume = task.get("_owner_wait_resume") or task.get("_budget_pause_resume") or task.get("_working_recovery") or {}
    sleeping = resume.get("sleep_exclusion_since") or native_sleep_started_at(task.get("_owner_wait_resume"))
    lane = returned_row_facts(task) if getattr(worker, "lane", LANE_PROJECT) == LANE_CONVERSATION else {}
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
        "soft_sent": False, "attempt": int(task.get("_attempt") or 1), **lane,
        "phase_generation": str(task.get("_phase_generation") or ""),
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


def needs_paid_admission(task: Dict[str, Any]) -> bool:
    """Owner 2A at the queue: whether a row's next attempt needs PAID admission, which alone a reached
    global limit refuses (``usage_accounting.paid_admission`` before any price is known). The worker's own
    opening rules name the route (``route_choice``) over what its dispatch resolves: a delegated child's frozen
    row selects its route (``subagent_runtime.configured_actor_route``), never Main. A continuation saved on
    another route, or a subscription route whose receipts carried cash, is admitted here and refused by its
    first reservation, which is the authority. An unresolvable route keeps the money stop."""
    from types import SimpleNamespace
    from ouroboros import route_choice
    from ouroboros.provider_models import provider_for_model
    from ouroboros.usage_accounting import paid_admission

    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    try:
        actor = task.get("configured_subagent")
        if str(task.get("delegation_role") or "").lower() == "subagent" and isinstance(actor, dict):
            from ouroboros.subagent_runtime import configured_actor_route, validate_subagent_snapshot
            from ouroboros.supervisor_route import validate_cognitive_route

            model, local = configured_actor_route(validate_subagent_snapshot(actor),
                                                  validate_cognitive_route(task.get("parent_cognitive_route")))
            metadata = {**metadata, "model": model, "use_local_model": local}
        ctx = SimpleNamespace(task_metadata=metadata, primary_route=None)
        route_choice.bind_task_route(ctx, metadata, presence=bool(task.get("_presence_turn")))
        model, use_local, _choice = route_choice.open_route(ctx, None)
    except Exception:
        log.warning("Route of queued %s unresolved; the money stop holds it", task.get("id"), exc_info=True)
        return True
    return paid_admission("local" if use_local else provider_for_model(model), None)


def _candidate_ineligible(queue, candidate, remaining, running, *, selected=(), refused=(), holders=None,
                          leased=None, unresolved_invalid_id_set=(), own_claim: str = "", stat: bool = True) -> bool:
    """The chooser's memory gates over one queued row, cheapest first, then the few stat reads a row they admitted
    pays (``stat``; the store gates are ``dispatch_choice``'s, read outside Q); also the launch's revalidation under Q
    (``own_claim``: the revalidating launch's own token, the one claim that does not exclude the row). Side-effect free
    except that a moved room's binding refreshes the row's lease identity and a foreign root is named once, off Q."""
    from ouroboros.project_lease import candidate_is_leasable, lane_holders, running_project_ids
    from ouroboros.project_placement import refresh_pending_placement
    from ouroboros.config import get_max_active_subagents_per_root
    from supervisor.conversation_lane import resumes_on_lane
    from supervisor.pool_admission import acting_children
    from supervisor.task_lifecycle import root_cancellation_fenced
    from supervisor.consciousness_carriers import autonomous_work_enabled

    task_id = str(candidate.get("id") or "")
    root_task_id = str(candidate.get("root_task_id") or "").strip()
    claim = candidate.get(_LAUNCH_CLAIM) or {}
    if (task_id in refused or task_id in selected or candidate.get("_owner_hold")
            or candidate.get("_project_admission_restore_hold") or claim and claim.get("token") != own_claim):
        return True
    if _pool()._invalid_depth_deferred(candidate, unresolved_invalid_id_set):
        return True
    from ouroboros.deployment_extensions import queue_held

    if queue_held():  # a deployment's hold: queued work waits for its required extensions; the carrier rechecks
        return True
    if (isinstance(candidate.get("_budget_pause"), dict) or budget_hold_fact(candidate) is not None
            or continuation_refusal(candidate)):
        # Durable #1196 hold: sibling unselected after root release, unrestorable
        # continuation or failed grant revocation. Selection must be recorded before assignment.
        return True
    if (root_task_id in queue.BUDGET_ROOT_FENCES
            and not candidate.get("_owner_wait_resume")
            and not candidate.get("_budget_pause_resume")
            # One member explicitly selected against THIS fence is admitted; the latch stays up for the rest (Q9).
            and not budget_fence_selected(candidate, queue.BUDGET_ROOT_FENCES.get(root_task_id))):
        return True
    if root_cancellation_fenced(candidate, root_task_id or task_id) or _direct_actor_still_registered(task_id):
        # A cancelled ancestor, or the direct actor that parked this id has not released its registry entry yet:
        # no second live actor for one id.
        return True
    if candidate.get("_presence_turn") and not claim and any(
            w.busy_task_id == task_id for w in _pool().WORKERS.values()):
        # The carrier that parked this Presence turn still holds its slot until its own settlement (#2566).
        return True
    if str(candidate.get("delegation_role") or "") == "subagent":  # the root's limit counts its acting children
        if (acting_children(root_task_id, running) >= get_max_active_subagents_per_root()
                and not _pool()._assignment_depth_reservation_admits(candidate, running)):
            return True
    if stat and not refresh_pending_placement(_pool().DRIVE_ROOT, candidate):  # a moved room's binding, before its lane
        return True
    if not resumes_on_lane(candidate):  # my human's returned turn holds no project lease (its lane's carrier takes it)
        holders = holders if holders is not None else lane_holders(running, _pool().PENDING)
        if not candidate_is_leasable(candidate, leased if leased is not None else running_project_ids(running),
                                     holders=holders):
            return True
    if not stat:
        return False
    if (_foreign_roots_refused(candidate) or not autonomous_work_enabled(candidate, _pool().DRIVE_ROOT)
            or not _pool().repo_writer_task_allowed(candidate)):
        return True
    # Owner 2A: $0 work keeps running at the limit; its every send still meets reserve_attempt.
    return remaining <= 0 and not candidate.get("_owner_wait_resume") and needs_paid_admission(candidate)


def _assign_returned_turns(queue, remaining, unresolved_invalid_id_set) -> bool:
    """My human's returned turns take the conversation lane's idle carriers, never a project slot (#2870).

    The one predicate of queued work (``dispatch_choice.choose`` over this lane's rows: holds, money, Stop,
    placement, the launch's store gates, C2 and its repair), the launch and its revalidation, without what binds the
    project pool alone: its fence, the owner's cap and the project lease. A dispatchable turn left without an idle
    carrier is the lane's to start one for. False: the chooser blocked. The caller holds no lock."""
    from supervisor.conversation_lane import note_returns_waiting, ready_idle_carrier, resumes_on_lane
    from supervisor.dispatch_choice import choose

    waiting, refused, deferred = set(), set(), set()
    while True:
        with _queue_lock:
            if not any(resumes_on_lane(task) for task in _pool().PENDING):
                break  # none queued: queued work keeps its one pass
        chosen = choose(queue, remaining, unresolved_invalid_id_set, excluded=waiting | refused, lane=LANE_CONVERSATION)
        if chosen is None:
            return False
        if not chosen:
            break
        [task] = chosen
        with _queue_lock:
            if not any(row is task for row in _pool().PENDING):
                continue  # settled since the choice: the next look decides
            carrier = ready_idle_carrier(for_owner=True)
            if carrier is None:
                waiting.add(str(task.get("id") or ""))
                continue
            outcome = _claim_worker_launch(queue, task, carrier, deferred=deferred)
            if outcome is False:
                refused.add(str(task.get("id") or ""))
            elif outcome is True:  # the inline path put it; bind it here under the same Q hold
                _bind_running(queue, task, carrier)
    with _queue_lock:
        note_returns_waiting(len(waiting))
    return True


def assign_tasks() -> None:
    """One tick's assignment: warm returns, queue hygiene under Q, my human's returned turns onto the conversation
    lane's carriers (``_assign_returned_turns``, under a fenced project pool too), then ``dispatch_choice.run_pass``
    over free capacity (nothing is chosen or read when no worker is idle and no process may start; the machine is
    read only when chosen work needs a new process), then growth to the demand that pass found, admitted by that
    pass's reading."""
    from ouroboros.store_hot_facts import statuses_ahead
    from supervisor import queue
    from supervisor.dispatch_choice import run_pass
    from supervisor.state import budget_remaining
    from supervisor.worker_owner_wait import grant_warm_returns
    from supervisor.conversation_lane import reconcile_under_project_fence
    from supervisor.worker_pool_lifecycle import _WORKER_LIFECYCLE_LOCK, reconcile_worker_capacity

    # The loop never waits behind an off-loop respawn or generation stop holding the lifecycle
    # serializer (teardowns kill and join outside it): the whole turn defers to the next tick, so
    # fresh work cannot overtake a Resume, and growth (below) is recomputed next tick the same way.
    if not _WORKER_LIFECYCLE_LOCK.acquire(blocking=False):
        return
    try:
        grant_warm_returns()
        if _pool()._WORKER_POOL_DISABLED_REASON:
            reconcile_under_project_fence()
    finally:
        _WORKER_LIFECYCLE_LOCK.release()
    _tick_parked_work(queue)  # custody observation may wait on a remote operation
    st = _pool().load_state()  # the state file, and below the money projection, are read outside Q
    # The cancel hygiene's one statuses() SELECT is read here, before Q; rows queued since are read under it.
    with statuses_ahead(_pool().DRIVE_ROOT, [row.get("id") for row in list(_pool().PENDING)]), _queue_lock:
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
        from supervisor.conversation_lane import note_returns_waiting, resumes_on_lane

        fenced = bool(_pool()._WORKER_POOL_DISABLED_REASON)  # the project pool's fence, never the lane's
        if fenced and not any(resumes_on_lane(task) for task in _pool().PENDING):
            note_returns_waiting(0)
            return
    try:  # every loop tick: ride the snapshot; a refusal re-reads exactly, reserve_attempt is the gate
        remaining = budget_remaining(st, strict=True, allow_stale=True)  # this tick refuses at zero
    except Exception:
        log.error("Task assignment blocked: monetary authority unavailable")
        return
    with _queue_lock:
        if remaining <= 0:
            planned = []
            for task in _pool().PENDING:
                if fenced and not resumes_on_lane(task) or _pool()._invalid_depth_deferred(task, unresolved_invalid_id_set):
                    continue  # (a fenced pool's own rows wait for Restart)
                if isinstance(task.get("_budget_pause"), dict):
                    continue
                if budget_hold_fact(task) is not None or continuation_refusal(task):
                    continue  # Held, not paused: no second pause row over the hold.
                if task.get("_owner_wait_resume"):
                    continue  # Restore the checkpoint; the loop still owns its budget stop.
                if not needs_paid_admission(task):
                    continue  # owner 2A: $0 work is neither paused nor stripped of its grant at the limit
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
                from supervisor.restart_retention import returned_before_body

                seed = bool((returned_before_body(task) or {}).get("seed_return"))  # D5: its given-back attempt never put
                retry_lineage = bool(
                    int(task.get("_attempt") or 1) > 1 and not seed
                    or task.get("original_task_id") or task.get("timeout_retry_from")
                )
                replay_safe = (
                    (task.get("admitted_dispatch") == "none" or _unsent(task) is not None or seed)  # never put here
                    and int(cost_fields.get("total_rounds") or 0) == 0
                    and not bool(cost_fields.get("ledger_integrity_degraded"))
                    and not retry_lineage
                )
                # Owner 4A: a row that already dispatched is HELD, nonterminal, when it carries saved work to
                # continue (its Resume and launch revalidate that source). Holding grants no replay: a row with
                # nothing saved keeps the honest terminal below.
                held = not replay_safe and "_working_recovery" in task
                pause = {
                    "status": "paused_before_dispatch" if replay_safe or held else "resource_limited",
                    "scope": "global",
                    "physical_calls": int(cost_fields.get("total_rounds") or 0),
                    "replay_safe": replay_safe,
                    **({"continuation": "working"} if held else {}),
                    "auto_resume": False,
                    "resume_policy": "manual_same_generation" if replay_safe or held else "cancel_or_new_run",
                    "paused_at": _pool().utc_now_iso(),
                }
                planned.append((task, pause, cost_fields))
            newly_paused, held_ids, terminal_ids = [], [], []
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
                    elif pause.get("continuation"):
                        task["_budget_pause"] = pause
                        held_ids.append(task_id)
                        write_task_result(
                            result_root, task_id, STATUS_SCHEDULED,
                            reason_code="budget_exhausted", resource_limit=pause,
                            result=("Budget limit reached after prior dispatch; the saved work is held and an "
                                    "explicit Resume continues the same task."),
                            **cost_fields,
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
            if newly_paused or held_ids or terminal_ids:
                _pool().append_jsonl(
                    _pool().DRIVE_ROOT / "logs" / "events.jsonl",
                    {
                        "ts": _pool().utc_now_iso(),
                        "type": "budget_tasks_paused",
                        "scope": "global",
                        "task_ids": newly_paused + held_ids,
                        "held_saved_work_task_ids": held_ids,
                        "resource_limited_task_ids": terminal_ids,
                        "auto_resume": False,
                    },
                )
                if st.get("owner_chat_id"):
                    _pool().send_with_budget(
                        int(st["owner_chat_id"]),
                        "🚫 Model budget reached. Queued tasks that need paid model calls are paused before "
                        "dispatch; subscription and local work continues. Raising the limit does not resume "
                        "them automatically: Resume continues each one.",
                        role="system", system_type="budget_notice")
                queue.persist_queue_snapshot(reason="budget_paused_before_dispatch")
            # Selection still runs (and repairs stale grants); quiescent capacity
            # maintenance also remains possible after money admission closes.

        # Evolution is hard-blocked in light runtime mode at the assignment
        # chokepoint too: a task restored from a snapshot or created before the
        # mode switch must never actually run. Cancel them terminally.
        from supervisor.evolution_lifecycle import evolution_block_reason
        evo_block = "" if fenced else evolution_block_reason()
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

    if not _assign_returned_turns(queue, remaining, unresolved_invalid_id_set) or fenced:
        return  # a fenced project pool assigns only my human's returned turns, and grows nothing
    passed = run_pass(queue, remaining, unresolved_invalid_id_set)
    if passed is None:
        return  # a malformed depth whose terminal could not be written stops this pass, growth included
    if _WORKER_LIFECYCLE_LOCK.acquire(blocking=False):
        try:
            reconcile_worker_capacity(*passed)  # the demand, and the pass's reading of the machine
        finally:
            _WORKER_LIFECYCLE_LOCK.release()
