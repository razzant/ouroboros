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
        park_interval_added = False
        try:
            # A warm wait may arrive before the cold consumption notification
            # (or its repair tick). Fold that exact interval before installing
            # the new one; a stale consumed event cannot own the warm interval.
            resume = meta["task"].get("_budget_pause_resume") or {}
            if resume.get("sleep_exclusion_since"):
                from supervisor.events_budget import _handle_budget_pause

                _handle_budget_pause({"phase": "consumed", "task_id": task_id,
                    "task_attempt": attempt, "pause_id": resume.get("pause_id"),
                    "grant_id": resume.get("grant_id")}, ctx)
                if (meta["task"].get("_budget_pause_resume") or {}).get("sleep_exclusion_since"):
                    raise RuntimeError("cold sleep consumption is not yet confirmed")
            wait = set_owner_wait(ctx.DRIVE_ROOT, task_id, wait)
            meta["owner_wait"] = wait
            park_interval_added = ("sleep_parked_at" not in meta
                                   and (isinstance(wait.get("sleep"), dict) or wait.get("reason") == "owner_pause"))
            if isinstance(wait.get("sleep"), dict):
                # A model sleep is not execution: excluded live until the task runs again.
                meta.setdefault("sleep_parked_at", _pool().time.time())
            elif wait.get("reason") == "owner_pause":
                from ouroboros.deadline_utils import parse_deadline_ts

                # Include time spent awaiting this acknowledgement, without restarting
                # the interval if the same park event is delivered again.
                parked = parse_deadline_ts((wait.get("owner_pause") or {}).get("parked_at"))
                meta.setdefault("sleep_parked_at", parked.timestamp() if parked else _pool().time.time())
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
            if park_interval_added:
                meta.pop("sleep_parked_at", None)
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
    if not ((not intent or intent.get("stop_policy") == "finalize_then_cancel")
            and _pool().repo_writer_task_allowed(meta["task"])):
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
        parked_at = meta.get("sleep_parked_at")
        prior_paused = float(meta.get("budget_paused_sec") or 0.0)
        paused = prior_paused + (max(0.0, _pool().time.time() - float(parked_at))
                                 if isinstance(parked_at, (int, float)) else 0.0)
        try:
            resumed = set_owner_wait(_pool().DRIVE_ROOT, task_id,
                                     {**wait, "state": "resumed", "budget_paused_sec": paused},
                                     expected_wait_id=wait["wait_id"])
            if replacement is not None:
                replacement.active_capacity = False
            worker.active_capacity = True
            meta["owner_wait"] = resumed
            meta["task"].pop("_owner_wait_resume", None)
            meta["budget_paused_sec"] = paused
            meta.pop("sleep_parked_at", None)
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
