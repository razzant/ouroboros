"""Lend dispatch capacity while the original worker retains an owner wait.

RUNNING and WORKERS keep the task's physical ownership. Only active capacity is
transferred: a parked worker cannot dispatch model work until an idle active
slot is retired and its original input queue receives the correlated grant.
The ordinary supervisor assignment tick performs this maintenance; there is no
second scheduler, queue, or durable wait store.
"""

from __future__ import annotations

import functools
import logging
import math
import uuid
from typing import Any

from ouroboros.owner_wait import set_owner_wait
from ouroboros.store import acquisition_budget
from supervisor.queue import _queue_lock
from supervisor.worker_pool_lifecycle import (
    _serialized_worker_lifecycle, request_worker_quiescence, retire_or_replace_worker,
    retire_quiescent_worker, retire_worker, worker_ready,
)

log = logging.getLogger(__name__)
_PENDING_PARK = "owner_wait_park_pending"


def _pool():
    from supervisor import workers
    return workers


def has_owner_wait_checkpoint(meta: dict, task_attempt: int) -> bool:
    """Whether this attempt saved work that an automatic retry would replay."""
    task = meta.get("task")
    wait = ((meta.get(_PENDING_PARK) or {}).get("event") or {}).get("checkpoint") or meta.get("owner_wait") or (
        task.get("_owner_wait_resume") if isinstance(task, dict) else None)
    return (isinstance(wait, dict) and wait.get("task_attempt") == task_attempt
            and bool(wait.get("source_ref")))


def waiting_on_owner(meta: dict, task_attempt: int) -> bool:
    """A current park request spares idle time before ACK too; refusal retains only custody."""
    pending = meta.get(_PENDING_PARK) or {}
    wait = (pending.get("event") or {}).get("checkpoint") if pending else meta.get("owner_wait")
    return (not pending.get("refused") and isinstance(wait, dict)
            and wait.get("task_attempt") == task_attempt
            and (bool(wait.get("source_ref")) if pending else wait.get("state") == "waiting"))


def _park_clock(meta: dict, wait: dict, pending: dict) -> tuple[bool, Any, Any]:
    """Validate this worker's closed total and actual open interval; missing is not zero."""
    cold = (meta["task"].get("_budget_pause_resume") or {}).get("sleep_exclusion_since")
    sleeping = isinstance(wait.get("sleep"), dict) or wait.get("reason") == "owner_pause"
    paused, started = wait.get("budget_paused_sec"), wait.get("sleep_started_at") if sleeping else None
    if sleeping and started is None and wait.get("reason") == "owner_pause":
        from ouroboros.deadline_utils import parse_deadline_ts

        recorded = parse_deadline_ts((wait.get("owner_pause") or {}).get("parked_at"))
        started = recorded.timestamp() if recorded else None
    def number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    if (paused is not None or cold or sleeping or pending.get("clock_error")) and (
            not number(paused) or paused < float(meta.get("budget_paused_sec") or 0)):
        return False, None, None
    if (sleeping and (not number(started) or started <= 0)
            or cold and not pending.get("clock_projected") and meta.get("sleep_parked_at") != cold):
        return False, None, None
    return True, paused, started


def _command(worker: Any, task_id: str, wait: dict, phase: str, **extra: Any) -> None:
    worker.in_q.put({"type": "owner_wait", "task_id": task_id,
                     "task_attempt": wait["task_attempt"], "wait_id": wait["wait_id"],
                     "phase": phase, **extra})


def retry_pending_owner_waits(ctx: Any) -> None:
    """The existing assignment tick retries each still-owned park once, without a store wait."""
    with _queue_lock:
        pending = [(task_id, meta, meta[_PENDING_PARK]) for task_id, meta in ctx.RUNNING.items()
                   if meta.get(_PENDING_PARK) and not meta[_PENDING_PARK].get("refused")]
    for task_id, meta, request in pending:
        with _queue_lock:
            if ctx.RUNNING.get(task_id) is not meta or meta.get(_PENDING_PARK) is not request:
                continue
        handle_owner_wait(request["event"], ctx)


def handle_owner_wait(event: dict, ctx: Any) -> None:
    """Acknowledge only the current physical attempt's durable park or wake."""
    from ouroboros.store import acquisition_budget, StoreBusy
    from supervisor import queue

    task_id, wait_id = str(event.get("task_id") or ""), str(event.get("wait_id") or "")
    try:
        attempt, wid, pid = (int(event[key]) for key in ("task_attempt", "worker_id", "pid"))
    except (KeyError, TypeError, ValueError):
        return
    with _queue_lock:
        meta = ctx.RUNNING.get(task_id)
        worker = ctx.WORKERS.get(wid)
        if (not wait_id or not isinstance(meta, dict) or worker is None
                or meta.get("worker_id") != wid or meta.get("attempt") != attempt
                or worker.busy_task_id != task_id or worker.proc.pid != pid
                or getattr(worker, "reaping", False)):
            return
        current = meta.get("owner_wait") or {}
        if event.get("phase") == "resume":
            if current.get("wait_id") == wait_id and current.get("state") == "waiting":
                meta["owner_wait_resume_requested"] = True
                reason = str(event.get("resume_reason") or "")
                if reason:
                    # Carried into the row the grant writes, so the projection
                    # can say a bound ended the wait. The notice itself is the
                    # worker's; this is the readable record beside it.
                    meta["owner_wait"] = {**current, "resume_reason": reason}
            return
        if event.get("phase") != "park":
            return
        if current.get("wait_id") == wait_id and current.get("state") == "resumed":
            return
        wait = event.get("checkpoint")
        if (not isinstance(wait, dict) or wait.get("wait_id") != wait_id
                or wait.get("task_attempt") != attempt or not wait.get("source_ref")):
            return
        if current.get("wait_id") == wait_id and not getattr(worker, "active_capacity", True):
            _command(worker, task_id, current, "parked")
            return
        pending = meta.get(_PENDING_PARK)
        if pending and pending["event"].get("wait_id") != wait_id:
            return  # One physical stack can owe only its current park acknowledgement.
        if pending and pending.get("refused"):
            return  # The rejected checkpoint prevents replay; it is not another park attempt.
        if pending is None:
            pending = meta[_PENDING_PARK] = {"event": {**event, "checkpoint": dict(wait)}}
        wait = {**wait, "started_at": meta["started_at"], "state": "waiting"}
        resume = meta["task"].get("_budget_pause_resume") or {}
        cold_interval = resume.get("sleep_exclusion_since")
        valid_clock, paused, parked_at = _park_clock(meta, wait, pending)
        if not valid_clock:
            if not pending.get("clock_error"):
                log.warning("Owner wait clock unavailable or contradictory for %s; park retained", task_id)
                pending["clock_error"] = True
            return
        if paused is not None:
            # Current worker/source facts govern its LIVE clock even before the
            # park commits. Neither this projection nor its pending request lends capacity.
            meta["budget_paused_sec"] = float(paused)
            if parked_at is not None:
                meta["sleep_parked_at"] = parked_at
            elif cold_interval:
                meta.pop("sleep_parked_at", None)
            pending["clock_projected"] = True
        park_written = False
        try:
            # A contended write rolls back before the next tick. Never hold Q or
            # the event loop waiting for SQLite; the physical request stays owned.
            with acquisition_budget(0):
                wait = set_owner_wait(ctx.DRIVE_ROOT, task_id, wait,
                                      expected_wait_id=current.get("wait_id"),
                                      expected_state=current.get("state"))
            park_written = True
            if cold_interval:
                from supervisor.events_budget import finish_resume_clock

                # The committed park retires only the old clock marker. Explicit
                # Resume selection and the truthful new open interval stay intact.
                finish_resume_clock(meta, float(paused))
                if parked_at is not None:
                    meta["sleep_parked_at"] = parked_at
            meta["owner_wait"] = wait
            # A spent exact-budget carrier still on this row is retired by the
            # revocation seam's ``_owner_wait_resume`` branch when the restart
            # reads the durable grant as consumed (#1196, F3); it decides nothing
            # while the task stays RUNNING here.
            worker.active_capacity = False
            if not queue.persist_queue_snapshot(reason="owner_wait_parked"):
                raise RuntimeError("owner wait queue snapshot was not persisted")
        except Exception as exc:
            if (not park_written and isinstance(exc, StoreBusy) and exc.reason == "contention"
                    and getattr(exc, "rollback_confirmed", None) is not False):
                return
            pending["refused"] = str(exc) or type(exc).__name__
            worker.active_capacity = True
            meta.pop("owner_wait", None)
            # The worker is still in this wait until it receives the refusal.
            # Keep its known clock/checkpoint facts, without a parked ACK or retry.
            _command(worker, task_id, wait, "refused", reason=str(exc))
            return
        meta.pop(_PENDING_PARK, None)
        _command(worker, task_id, wait, "parked")


def _resume_memory_allowed(task_id: str, meta: dict, worker: Any) -> bool:
    """Under Q, memory only: the same physical attempt, on a live worker, still allowed to write."""
    return (_pool().RUNNING.get(task_id) is meta and not worker.reaping
            and worker.busy_task_id == task_id and worker.proc.is_alive()
            and _pool().repo_writer_task_allowed(meta["task"]))


def _resume_authority_allowed(task_id: str, meta: dict) -> bool:
    """Outside Q: the durable authorities (cancellation intent, the owner Pause fence); unreadable refuses."""
    from ouroboros.cancel_intents import active_intents

    try:
        intent = active_intents(_pool().DRIVE_ROOT, strict=True).get(task_id) or {}
    except Exception:
        log.warning("Owner wait cannot read cancellation authority for %s", task_id, exc_info=True)
        return False
    if intent and intent.get("stop_policy") != "finalize_then_cancel":
        return False
    return _owner_pause_wake_still_authorized(task_id, meta)


def _owner_pause_wake_still_authorized(task_id: str, meta: dict) -> bool:
    """A warm owner-Pause park woken by the Resume needs that Resume still standing
    at the grant: a newer Pause of the root keeps the stack parked. A control
    wake (Stop, Panic, a deadline) ends the park whatever the fence says."""
    wait = meta.get("owner_wait") or {}
    if wait.get("reason") != "owner_pause" or wait.get("resume_reason") != "control:owner_resume":
        return True
    from types import SimpleNamespace
    from ouroboros.owner_pause import member_fence

    task = meta.get("task") or {}
    try:
        return not member_fence(SimpleNamespace(
            task_id=task_id, root_task_id=str(task.get("root_task_id") or task_id),
            budget_drive_root=task.get("budget_drive_root") or _pool().DRIVE_ROOT))
    except Exception:
        log.warning("Owner pause authority unreadable for %s; its warm park stays parked", task_id, exc_info=True)
        return False


# Memory only, on the RUNNING meta: a warm Resume whose durable grant runs on the admission lane.
_RESUME_CLAIM = "_resume_claim"


def _grant_resume(
    task_id: str, meta: dict, worker: Any, *, exhausted_replacement: Any = None,
) -> bool:
    """Reserve the warm Resume under Q and persist it on the admission lane.

    Under Q only memory decides (the attempt, the worker, the replacement slot) and the claim is
    taken; the lane reads the durable authorities, writes the wait ``resumed`` (a CAS on its id),
    re-takes Q, revalidates memory only and hands the capacity back with the grant command
    (``_resume_attempt``). A refusal or failure after the write rewrites ``waiting`` outside Q and
    restores the clock facts. True only on the historical inline path (no generation) once granted;
    on the lane this returns False and the caller's hold stands until the grant lands or lapses.
    """
    from supervisor import workspace_preparation

    with _queue_lock:
        if meta.get(_RESUME_CLAIM) or not _resume_memory_allowed(task_id, meta, worker):
            return False
        replacement = exhausted_replacement
        if replacement is not None and not _replacement_current(replacement):
            return False
        generation = workspace_preparation.current_generation()
        token = uuid.uuid4().hex
        meta[_RESUME_CLAIM] = token
    if generation is None:
        try:
            return _resume_attempt(task_id, meta, worker, token, replacement)
        finally:
            _release_resume(task_id, meta, token)
    if generation.launches_closed() or not generation.submit(
            functools.partial(_resume_job, task_id, meta, worker, token, replacement, generation),
            workspace_preparation.ADMISSION_LANE):
        _release_resume(task_id, meta, token)
    return False


def _resume_job(task_id: str, meta: dict, worker: Any, token: str, replacement: Any, generation: Any) -> None:
    """The reserving GENERATION's lane job: write the grant, then put it, unless that generation closed."""
    from supervisor import workspace_preparation

    try:
        generation.wait_store(functools.partial(_resume_attempt, task_id, meta, worker, token, replacement, generation),
                              controls=lambda: ({"detail": "generation closed"}
                                                if generation.launches_closed() else None))
    except workspace_preparation.PreparationStopped:
        log.info("Owner wait resume of %s withdrawn: the supervisor generation is closing", task_id)
    except Exception:
        log.warning("Owner wait resume remains pending for %s", task_id, exc_info=True)
    finally:
        _release_resume(task_id, meta, token)


def _release_resume(task_id: str, meta: dict, token: str) -> None:
    with _queue_lock:
        if meta.get(_RESUME_CLAIM) == token:
            meta.pop(_RESUME_CLAIM, None)


def _replacement_current(replacement: Any) -> bool:
    return (_pool().WORKERS.get(replacement.wid) is replacement
            and getattr(replacement, "readiness_exhausted", False)
            and getattr(replacement, "active_capacity", True)
            and not replacement.proc.is_alive())


def _resume_claim_held(task_id: str, meta: dict, worker: Any, token: str) -> bool:
    return (meta.get(_RESUME_CLAIM) == token and _resume_memory_allowed(task_id, meta, worker)
            and _pool().WORKERS.get(worker.wid) is worker)


def _active_slots() -> int:
    return sum(1 for slot in _pool().WORKERS.values() if getattr(slot, "active_capacity", True))


def _resume_attempt(task_id: str, meta: dict, worker: Any, token: str, replacement: Any,
                    generation: Any = None) -> bool:
    """One grant attempt: authorities and the durable ``resumed`` wait outside Q, then the put under Q.

    On the lane a closed generation refuses before the durable write; a grant written while it was
    open finishes its put, which without a replacement needs a free active slot: the reservation lent
    nothing, so a slot taken meanwhile leaves the park standing.
    """
    from ouroboros.owner_wait import OwnerWaitSuperseded
    from supervisor import queue

    with _queue_lock:
        if not _resume_claim_held(task_id, meta, worker, token):
            return False
        wait = meta["owner_wait"]
        cold_handoff = meta["task"].get("_owner_wait_resume")
        parked_at = meta.get("sleep_parked_at")
        prior_paused = float(meta.get("budget_paused_sec") or 0.0)
    if not _resume_authority_allowed(task_id, meta):
        return False
    if generation is not None and generation.launches_closed():
        return False  # a stopping generation writes no grant: the park stands as it is
    paused = prior_paused + (max(0.0, _pool().time.time() - float(parked_at))
                             if isinstance(parked_at, (int, float)) else 0.0)
    try:
        resumed = set_owner_wait(_pool().DRIVE_ROOT, task_id,
                                 {**wait, "state": "resumed", "budget_paused_sec": paused},
                                 expected_wait_id=wait["wait_id"])
    except OwnerWaitSuperseded:
        log.warning("Owner wait of %s changed before its Resume was granted; the park stands", task_id)
        return False
    granted = False
    try:
        with _queue_lock:
            if not _resume_claim_held(task_id, meta, worker, token) or (
                    replacement is not None and not _replacement_current(replacement)):
                log.info("Owner wait resume of %s refused after its durable grant: the stack, its worker "
                         "or the replacement slot changed; the park stands", task_id)
                return False
            if generation is not None and replacement is None and _active_slots() >= _pool().MAX_WORKERS:
                log.info("Owner wait resume of %s waits: no active slot is free; the park stands", task_id)
                return False
            if replacement is not None:
                replacement.active_capacity = False
            worker.active_capacity = True
            meta["owner_wait"] = resumed
            meta["task"].pop("_owner_wait_resume", None)
            meta["budget_paused_sec"] = paused
            meta.pop("sleep_parked_at", None)
            try:
                if not queue.persist_queue_snapshot(reason="owner_wait_resumed"):
                    raise RuntimeError("owner wait resume snapshot was not persisted")
                _command(worker, task_id, resumed, "resume_granted")
            except Exception:
                worker.active_capacity = False
                if replacement is not None:
                    replacement.active_capacity = True
                if cold_handoff is not None:
                    meta["task"]["_owner_wait_resume"] = cold_handoff
                meta["owner_wait"] = wait
                meta["budget_paused_sec"] = prior_paused
                if parked_at is not None:
                    meta["sleep_parked_at"] = parked_at
                raise
            granted = True
            meta.pop("owner_wait_resume_requested", None)
            quiz_id, chat_id = str(resumed.get("quiz_id") or ""), int((meta.get("task") or {}).get("chat_id") or 0)
            control_wake = str(resumed.get("resume_reason") or "").startswith("control:")
            # A mailbox wake is the start of useful model work, not a new attempt.
            meta["last_progress_at"] = _pool().time.time()
    finally:
        if not granted:
            # The durable grant stands without its put: rewrite ``waiting`` outside Q, best effort.
            try:
                set_owner_wait(_pool().DRIVE_ROOT, task_id, wait, expected_wait_id=wait["wait_id"])
                queue.persist_queue_snapshot(reason="owner_wait_grant_failed")
            except Exception:
                log.warning("Owner-wait rollback remains unpersisted for %s", task_id, exc_info=True)
    if quiz_id and not control_wake:
        # The bound closed and the pooled task resumed: one seam with the direct lane.
        from ouroboros.owner_wait import announce_wait_ended

        announce_wait_ended(_pool().DRIVE_ROOT, task_id, quiz_id, chat_id)
    return True


@_serialized_worker_lifecycle
def maintain_owner_wait_capacity() -> None:
    """Resume original stacks first; assignment owns demand-sized replacement.

    At full capacity an authorized warm Resume keeps ONE idle slot from fresh work
    until it acknowledges and retires; other probes stay optional, and a slot
    already retiring pays for one Resume. The parked author's slot carries the same
    mark, and growth leaves capacity only for marked authors. Holds are recomputed
    every tick, so they lapse with the authorization (Stop, Panic, a newer Pause,
    the grant itself).
    """
    holds: dict = {}
    try:
        _resume_original_stacks(holds)
    finally:
        with _queue_lock:
            for slot in _pool().WORKERS.values():
                slot.resume_hold = holds.get(slot.wid, "")


def _resume_original_stacks(holds: dict) -> None:
    pool = _pool()
    if pool.disable_exhausted_worker_pool():
        return
    with _queue_lock:
        if (not pool.WORKERS or pool._WORKER_POOL_DISABLED_REASON
                or all(getattr(w, "active_capacity", True) for w in pool.WORKERS.values())):
            return
        retired = [w for w in pool.WORKERS.values()
                   if not getattr(w, "active_capacity", True) and w.busy_task_id is None]
    for worker in retired:
        if not worker.proc.is_alive():
            try:
                retire_or_replace_worker(worker.wid, worker)
            except Exception:
                log.warning("Dead lent worker %s stays for the next tick", worker.wid, exc_info=True)
        else:
            retire_quiescent_worker(worker)
    with _queue_lock:
        waiting = [(tid, meta) for tid, meta in pool.RUNNING.items()
                   if meta.get("owner_wait_resume_requested")]
        freeing = sum(1 for w in pool.WORKERS.values()
                      if getattr(w, "active_capacity", True) and getattr(w, "retiring", False))
    for task_id, meta in waiting:
        with _queue_lock:
            worker = pool.WORKERS.get(meta.get("worker_id"))
            if worker is None or getattr(worker, "active_capacity", True):
                continue
            if not _resume_memory_allowed(task_id, meta, worker):
                continue
        # The durable authorities are read off Q without waiting; a busy read authorizes nothing
        # this tick (the hold is recomputed every tick).
        with acquisition_budget(0):
            if not _resume_authority_allowed(task_id, meta):
                continue
        with _queue_lock:
            if not _resume_memory_allowed(task_id, meta, worker):
                continue
            holds[worker.wid] = task_id  # authorized this tick: owed its lent capacity back
            active = [w for w in pool.WORKERS.values() if getattr(w, "active_capacity", True)]
            # A grant still on the lane counts as the slot it will take (its worker is not active yet).
            granting = sum(1 for other in pool.RUNNING.values() if other is not meta and other.get(_RESUME_CLAIM))
            exhausted = next((w for w in active
                              if getattr(w, "readiness_exhausted", False)
                              and not w.proc.is_alive()), None)
            idle = [w for w in active if w.busy_task_id is None and worker_ready(w)]
            if len(active) + granting >= pool.MAX_WORKERS and exhausted is None:
                for slot in idle:
                    request_worker_quiescence(slot)
                quiet = next((slot for slot in idle if slot.quiescence_confirmed), None)
                held = next((slot for slot in idle if slot.wid not in holds), None)
                if freeing:
                    freeing -= 1  # an earlier retirement already frees this Resume's slot
                elif quiet is not None:
                    retire_quiescent_worker(quiet)
                elif held is not None:
                    holds[held.wid] = task_id
                continue  # grant after the reaper confirms death, on a later tick
        if len(active) + granting < pool.MAX_WORKERS:
            exhausted = None
        try:
            if _grant_resume(task_id, meta, worker, exhausted_replacement=exhausted):
                holds.pop(worker.wid, None)
                if exhausted is not None:
                    retire_worker(exhausted.wid, exhausted)
        except Exception:
            log.warning("Owner wait resume remains pending for %s", task_id, exc_info=True)
