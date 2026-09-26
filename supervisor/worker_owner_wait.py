"""Lend dispatch capacity while the original worker retains an owner wait.

RUNNING and WORKERS keep the task's physical ownership. Only active capacity is
transferred: a parked worker cannot dispatch model work until an idle active
slot is retired and its original input queue receives the correlated grant.
The ordinary supervisor assignment tick performs this maintenance; there is no
second scheduler, queue, or durable wait store.
"""

from __future__ import annotations

import logging
from typing import Any

from ouroboros.owner_wait import set_owner_wait
from supervisor.queue import _queue_lock
from supervisor.worker_pool_lifecycle import (
    _serialized_worker_lifecycle, _spawn_worker_slot, kill_worker_tree, retire_worker,
)

log = logging.getLogger(__name__)


def _pool():
    from supervisor import workers
    return workers


def has_owner_wait_checkpoint(meta: dict, task_attempt: int) -> bool:
    """Whether this attempt saved work that an automatic retry would replay."""
    task = meta.get("task")
    wait = meta.get("owner_wait") or (
        task.get("_owner_wait_resume") if isinstance(task, dict) else None)
    return (isinstance(wait, dict) and wait.get("task_attempt") == task_attempt
            and bool(wait.get("source_ref")))


def _command(worker: Any, task_id: str, wait: dict, phase: str, **extra: Any) -> None:
    worker.in_q.put({"type": "owner_wait", "task_id": task_id,
                     "task_attempt": wait["task_attempt"], "wait_id": wait["wait_id"],
                     "phase": phase, **extra})


def handle_owner_wait(event: dict, ctx: Any) -> None:
    """Acknowledge only the current physical attempt's durable park or wake."""
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
        wait = {**wait, "started_at": meta["started_at"], "state": "waiting"}
        try:
            wait = set_owner_wait(ctx.DRIVE_ROOT, task_id, wait)
            meta["owner_wait"] = wait
            # A spent exact-budget carrier still on this row is retired by the
            # revocation seam's ``_owner_wait_resume`` branch when the restart
            # reads the durable grant as consumed (#1196, F3); it decides nothing
            # while the task stays RUNNING here.
            worker.active_capacity = False
            if not queue.persist_queue_snapshot(reason="owner_wait_parked"):
                raise RuntimeError("owner wait queue snapshot was not persisted")
        except Exception as exc:
            worker.active_capacity = True
            meta.pop("owner_wait", None)
            _command(worker, task_id, wait, "refused", reason=str(exc))
            return
        _command(worker, task_id, wait, "parked")


def _retire_idle(worker: Any) -> bool:
    """Retire an idle replacement while assignment cannot reuse its slot."""
    with _queue_lock:
        if _pool().WORKERS.get(worker.wid) is not worker or worker.busy_task_id is not None:
            return False
        worker.reaping = True
        worker.active_capacity = False
    if worker.proc.is_alive():
        kill_worker_tree(worker.proc.pid, keep_services=True)
        worker.proc.join(timeout=2)
    return retire_worker(worker.wid, worker)


def _resume_allowed(task_id: str, meta: dict, worker: Any) -> bool:
    from ouroboros.cancel_intents import active_intents

    if (_pool().RUNNING.get(task_id) is not meta or worker.reaping
            or worker.busy_task_id != task_id or not worker.proc.is_alive()):
        return False
    try:
        intent = active_intents(_pool().DRIVE_ROOT, strict=True).get(task_id) or {}
    except Exception:
        log.warning("Owner wait cannot read cancellation authority for %s", task_id, exc_info=True)
        return False
    return (not intent or intent.get("stop_policy") == "finalize_then_cancel") and _pool().repo_writer_task_allowed(meta["task"])


def _grant_resume(
    task_id: str, meta: dict, worker: Any, *, exhausted_replacement: Any = None,
) -> bool:
    from supervisor import queue

    with _queue_lock:
        if not _resume_allowed(task_id, meta, worker):
            return False
        replacement = exhausted_replacement
        if replacement is not None and (
            _pool().WORKERS.get(replacement.wid) is not replacement
            or not getattr(replacement, "readiness_exhausted", False)
            or not getattr(replacement, "active_capacity", True)
            or replacement.proc.is_alive()
        ):
            return False
        wait = meta["owner_wait"]
        cold_handoff = meta["task"].get("_owner_wait_resume")
        try:
            resumed = set_owner_wait(_pool().DRIVE_ROOT, task_id, {**wait, "state": "resumed"},
                                     expected_wait_id=wait["wait_id"])
            if replacement is not None:
                replacement.active_capacity = False
            worker.active_capacity = True
            meta["owner_wait"] = resumed
            meta["task"].pop("_owner_wait_resume", None)
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
            try:
                set_owner_wait(_pool().DRIVE_ROOT, task_id, wait,
                               expected_wait_id=wait["wait_id"])
                queue.persist_queue_snapshot(reason="owner_wait_grant_failed")
            except Exception:
                log.warning("Owner-wait rollback remains unpersisted for %s", task_id, exc_info=True)
            raise
        meta.pop("owner_wait_resume_requested", None)
        if (str(resumed.get("quiz_id") or "")
                and not str(resumed.get("resume_reason") or "").startswith("control:")):
            # The bound closed and the pooled task resumed: one seam with the direct lane.
            from ouroboros.owner_wait import announce_wait_ended

            announce_wait_ended(_pool().DRIVE_ROOT, task_id, str(resumed["quiz_id"]),
                                int((meta.get("task") or {}).get("chat_id") or 0))
        # A mailbox wake is the start of useful model work, not a new attempt.
        meta["last_progress_at"] = _pool().time.time()
        return True


@_serialized_worker_lifecycle
def maintain_owner_wait_capacity() -> None:
    """Resume original stacks before assigning fresh work, then fill lent slots."""
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
        _retire_idle(worker)
    with _queue_lock:
        waiting = [(tid, meta) for tid, meta in pool.RUNNING.items()
                   if meta.get("owner_wait_resume_requested")]
    for task_id, meta in waiting:
        with _queue_lock:
            worker = pool.WORKERS.get(meta.get("worker_id"))
            if worker is None or getattr(worker, "active_capacity", True):
                continue
            if not _resume_allowed(task_id, meta, worker):
                continue
            active = [w for w in pool.WORKERS.values() if getattr(w, "active_capacity", True)]
            exhausted = next((w for w in active
                              if getattr(w, "readiness_exhausted", False)
                              and not w.proc.is_alive()), None)
            idle = next((w for w in active if w.busy_task_id is None and not w.reaping), None)
            if len(active) >= pool.MAX_WORKERS and idle is None and exhausted is None:
                continue
        if len(active) < pool.MAX_WORKERS:
            exhausted = None
        elif exhausted is None and not _retire_idle(idle):
            continue
        try:
            if _grant_resume(task_id, meta, worker, exhausted_replacement=exhausted):
                if exhausted is not None:
                    retire_worker(exhausted.wid, exhausted)
        except Exception:
            log.warning("Owner wait resume remains pending for %s", task_id, exc_info=True)
    with _queue_lock:
        missing = pool.MAX_WORKERS - sum(1 for w in pool.WORKERS.values()
                                         if getattr(w, "active_capacity", True))
        next_id = max(pool.WORKERS, default=-1) + 1
    for wid in range(next_id, next_id + missing):
        try:
            _spawn_worker_slot(wid)
        except Exception:
            log.warning("Failed to replenish owner-wait capacity", exc_info=True)
            break
