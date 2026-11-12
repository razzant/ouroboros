"""An owner wait or warm sleep keeps its process; its return continues there (the one return rule).

RUNNING and WORKERS keep the task's physical ownership while it waits: the park marks
the slot (``active_capacity`` False) and the waiting row does not act, so the owner's
cap and the child limit do not count it (``pool_admission.waits_on_its_process``). A
wake that asks to resume is granted on the same worker without a capacity condition
(pool-capacity D8): nothing is held idle, silenced or retired for it, and the overshoot
over the owner's cap shows in the pool's counts. The authority checks stay. The
ordinary supervisor assignment tick performs this; there is no second scheduler,
queue, or durable wait store.

Q spans memory only here: a park and a grant are decided under Q, written to the
task result outside it (``set_owner_wait``: the store is the restore authority
every reader takes the wait from, ``restart_retention``, ``queue_snapshot``,
``working_checkpoint.live_park``), then applied to memory under Q again after
revalidating the stack, and MARKED on the queue (``persist_queue_snapshot``),
never published under Q: the tick publishes the row. A crash between the store
commit and that publication restores from the store (kill-after-each-step:
``tests/test_queue_publication_revision.py``).

``grant_warm_returns`` persists on the admission lane under its claim, rolling
back if the stack changed before put. Contention retries the same row under
Stop/deadline/ceiling; pre-ACK sleep and the native-restart interval fold once.
Resume keeps the attempt and never replays it.
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
    _serialized_worker_lifecycle, is_worker_slot, retire_or_replace_worker, retire_quiescent_worker,
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


def _park_request_held(ctx: Any, task_id: str, meta: dict, worker: Any, attempt: int, pending: Any) -> bool:
    """Under Q: the park request the handler left Q with is still the one this stack owes."""
    return (ctx.RUNNING.get(task_id) is meta and ctx.WORKERS.get(worker.wid) is worker
            and meta.get(_PENDING_PARK) is pending and meta.get("worker_id") == worker.wid
            and meta.get("attempt") == attempt and worker.busy_task_id == task_id
            and not getattr(worker, "reaping", False))


def handle_owner_wait(event: dict, ctx: Any) -> None:
    """Acknowledge only the current physical attempt's durable park or wake.

    The park is decided under Q, written to the store outside it and applied under Q again
    (``_park_request_held`` revalidates the stack); the queue is marked, never published here.
    """
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
                or worker.busy_task_id != task_id
                or (worker.proc.pid != pid and is_worker_slot(worker))  # an inline carrier has no pid of its own
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
        expected_wait_id, expected_state = current.get("wait_id"), current.get("state")
        root_id = str(meta["task"].get("root_task_id") or task_id)
    try:
        # Outside Q: the durable park. A contended write rolls back before the next tick. Never hold
        # Q or the event loop waiting for SQLite; the physical request stays owned (``_PENDING_PARK``).
        with acquisition_budget(0):
            written = set_owner_wait(ctx.DRIVE_ROOT, task_id, wait,
                                     expected_wait_id=expected_wait_id, expected_state=expected_state)
    except Exception as exc:
        if (isinstance(exc, StoreBusy) and exc.reason == "contention"
                and getattr(exc, "rollback_confirmed", None) is not False):
            return
        with _queue_lock:
            if not _park_request_held(ctx, task_id, meta, worker, attempt, pending):
                return
            pending["refused"] = str(exc) or type(exc).__name__
            worker.active_capacity = True
            meta.pop("owner_wait", None)
            # The worker is still in this wait until it receives the refusal.
            # Keep its known clock/checkpoint facts, without a parked ACK or retry.
            _command(worker, task_id, wait, "refused", reason=str(exc))
        return
    with _queue_lock:
        if not _park_request_held(ctx, task_id, meta, worker, attempt, pending):
            # The park stands in the store (saved work a recovery reads); the stack it was for moved on.
            log.info("Owner wait park of %s written but its stack changed meanwhile; no acknowledgement", task_id)
            return
        if cold_interval:
            from supervisor.events_budget import finish_resume_clock

            # The committed park retires only the old clock marker. Explicit
            # Resume selection and the truthful new open interval stay intact.
            finish_resume_clock(meta, float(paused))
            if parked_at is not None:
                meta["sleep_parked_at"] = parked_at
        meta["owner_wait"] = written
        # A spent exact-budget carrier still on this row is retired by the
        # revocation seam's ``_owner_wait_resume`` branch when the restart
        # reads the durable grant as consumed (#1196, F3); it decides nothing
        # while the task stays RUNNING here.
        worker.active_capacity = False
        # A mark, not a barrier: the store's ``waiting`` row is the park's restore authority; the
        # tick publishes the row (a crash before it restores from the store all the same).
        queue.persist_queue_snapshot(reason="owner_wait_parked")
        if written.get("reason") == "owner_pause":  # it may be the tree's last member; no census follows a warm park
            from supervisor.owner_pause_control import mark_tree_changed

            mark_tree_changed(root_id)
        meta.pop(_PENDING_PARK, None)
        _command(worker, task_id, written, "parked")


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


def _grant_resume(task_id: str, meta: dict, worker: Any) -> bool:
    """Reserve the warm Resume under Q and persist it on the admission lane.

    Under Q only memory decides (the attempt, the worker) and the claim is taken; the lane reads
    the durable authorities, writes the wait ``resumed`` (a CAS on its id), re-takes Q, revalidates
    memory only, marks the queue and hands the worker its grant (``_resume_attempt``). A refusal or
    failure after the write rewrites ``waiting`` outside Q and restores the clock facts. True only on
    the historical inline path (no generation) once granted; on the lane this returns False and the
    claim keeps the next tick from granting twice until the grant lands or lapses.
    """
    from supervisor import workspace_preparation

    with _queue_lock:
        if meta.get(_RESUME_CLAIM) or not _resume_memory_allowed(task_id, meta, worker):
            return False
        generation = workspace_preparation.current_generation()
        token = uuid.uuid4().hex
        meta[_RESUME_CLAIM] = token
    if generation is None:
        try:
            return _resume_attempt(task_id, meta, worker, token)
        finally:
            _release_resume(task_id, meta, token)
    if generation.launches_closed() or not generation.submit(
            functools.partial(_resume_job, task_id, meta, worker, token, generation),
            workspace_preparation.ADMISSION_LANE):
        _release_resume(task_id, meta, token)
    return False


def _resume_job(task_id: str, meta: dict, worker: Any, token: str, generation: Any) -> None:
    """The reserving GENERATION's lane job: write the grant, then put it, unless that generation closed."""
    from supervisor import workspace_preparation

    try:
        generation.wait_store(functools.partial(_resume_attempt, task_id, meta, worker, token, generation),
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


def _resume_claim_held(task_id: str, meta: dict, worker: Any, token: str) -> bool:
    return (meta.get(_RESUME_CLAIM) == token and _resume_memory_allowed(task_id, meta, worker)
            and _pool().WORKERS.get(worker.wid) is worker)


def _resume_attempt(task_id: str, meta: dict, worker: Any, token: str, generation: Any = None) -> bool:
    """One grant attempt: authorities and the durable ``resumed`` wait outside Q, then the put under Q.

    On the lane a closed generation refuses before the durable write; a grant written while it was
    open finishes its put on the same worker, whatever the pool holds (the one return rule). The
    second hold is memory only: the queue is marked (the store's ``resumed`` row is the grant's
    restore authority), never published here.
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
            if not _resume_claim_held(task_id, meta, worker, token):
                log.info("Owner wait resume of %s refused after its durable grant: the stack or its worker "
                         "changed; the park stands", task_id)
                return False
            worker.active_capacity = True
            meta["owner_wait"] = resumed
            meta["task"].pop("_owner_wait_resume", None)
            meta["budget_paused_sec"] = paused
            meta.pop("sleep_parked_at", None)
            try:
                queue.persist_queue_snapshot(reason="owner_wait_resumed")  # a mark: the tick publishes the row
                _command(worker, task_id, resumed, "resume_granted")
            except Exception:
                worker.active_capacity = False
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
def grant_warm_returns() -> None:
    """Every authorized warm return continues on its own process (pool-capacity D8: the one return rule).

    A parked stack whose wake asked to resume is granted on its worker with no capacity condition:
    no idle process is held, silenced or retired for it, and the owner's cap does not stop it. The
    authorities stay: memory under Q (the same attempt on a live worker still allowed to write), the
    durable cancellation intent and owner Pause fence read off Q without waiting (a busy read grants
    nothing this tick), and ``_RESUME_CLAIM`` against a double return. A sleeper's process left
    without its task by a stop is retired through the quiescence probe, or forgotten when dead.
    """
    pool = _pool()
    with _queue_lock:
        if (not pool.WORKERS or pool._WORKER_POOL_DISABLED_REASON
                or all(getattr(w, "active_capacity", True) for w in pool.WORKERS.values())):
            return  # no slot is parked: nothing waits to return
        left = [w for w in pool.WORKERS.values()
                if not getattr(w, "active_capacity", True) and w.busy_task_id is None]
    for worker in left:
        if not worker.proc.is_alive():
            try:
                retire_or_replace_worker(worker.wid, worker)
            except Exception:
                log.warning("Dead parked worker %s stays for the next tick", worker.wid, exc_info=True)
        else:
            retire_quiescent_worker(worker)
    with _queue_lock:
        waiting = [(tid, meta) for tid, meta in pool.RUNNING.items()
                   if meta.get("owner_wait_resume_requested")]
    for task_id, meta in waiting:
        with _queue_lock:
            worker = pool.WORKERS.get(meta.get("worker_id"))
            if (worker is None or getattr(worker, "active_capacity", True)
                    or not _resume_memory_allowed(task_id, meta, worker)):
                continue
        with acquisition_budget(0):
            if not _resume_authority_allowed(task_id, meta):
                continue
        try:
            _grant_resume(task_id, meta, worker)
        except Exception:
            log.warning("Owner wait resume remains pending for %s", task_id, exc_info=True)
