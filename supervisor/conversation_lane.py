"""The conversation lane: pool slots that carry the mind's foci, and their reserved spare.

Owner Main/room turns, consciousness wakes, Presence bodies and the auto-resume
turn execute in worker processes like any task, but they are admitted by routing
(never through PENDING), hold no project lease and sit outside ``MAX_WORKERS``,
the crash-storm capture and the readiness-exhaustion fence. Their slots live in
the same ``WORKERS`` table as project workers with ``lane="conversation"``, so
health, custody, Panic and quiescence retirement iterate every slot alike; the
three project fences filter by lane.

Capacity (owner decision 4A): the pool's one quiescent spare is the lane's. This
module keeps exactly one idle READY carrier warm, replenishes it when a turn
claims it, retires acknowledged surplus through the existing quiescence probe and
never spawns while the repo-writer gate is closed (a fresh carrier would import a
conflicted tree; admission then takes the disclosed degraded executor). No number
bounds concurrent carriers: the bound is spawn success, and the vitals show the
carrier count. Called from ``reconcile_worker_capacity`` under the lifecycle
serializer, before the project pool's own growth.
"""

from __future__ import annotations

import logging
from typing import Any

from ouroboros.task_carrier import LANE_CONVERSATION
from supervisor.queue import _queue_lock

log = logging.getLogger(__name__)

# The lane keeps its spare only for a STARTED pool generation: ``spawn_workers`` arms it for the
# table it filled and a whole-table stop (Panic, Restart, a managed update) disarms it, so a
# stopped pool never grows a carrier behind its stop and a table built by hand starts no child.
_ARMED_TABLE: Any = None


def arm(table: Any) -> None:
    """Arm replenishment for ``table`` (the live ``WORKERS`` dict); ``None`` disarms."""
    global _ARMED_TABLE
    _ARMED_TABLE = id(table) if table is not None else None


def armed() -> bool:
    return _ARMED_TABLE is not None and _ARMED_TABLE == id(_pool().WORKERS)


def _pool():
    """The parent module, read at call time (the owner-approved D18/D33 mechanical exception)."""
    from supervisor import workers

    return workers


def lane_carriers(pool: Any = None) -> list:
    """Every conversation carrier that is a custodied worker process."""
    from supervisor.worker_pool_lifecycle import is_worker_slot, lane_slots

    return [slot for slot in lane_slots(pool) if is_worker_slot(slot)]


def ready_idle_carrier() -> Any:
    """Under Q: the reserved spare, a READY idle lane carrier nothing else holds, or None."""
    from supervisor.worker_pool_lifecycle import worker_ready

    for slot in lane_carriers():
        if (slot.busy_task_id is None and worker_ready(slot) and not slot.retiring
                and getattr(slot, "active_capacity", True) and not getattr(slot, "resume_hold", "")):
            return slot
    return None


def reconcile_conversation_lane() -> None:
    """One ready idle carrier at all times; surplus retires; nothing spawns in a marker window.

    Lifecycle serializer held by the caller; Q is taken only for the census. A
    carrier that exhausted its readiness attempts is parked until Restart like a
    project slot, and no second carrier starts beside it: the admission discloses
    the degraded executor instead (a physical fact, never "busy").
    """
    from supervisor.worker_pool_lifecycle import (
        _spawn_worker_slot, request_worker_quiescence, retire_quiescent_worker, worker_ready,
    )

    pool = _pool()
    with _queue_lock:
        if not armed() or not pool.WORKERS or pool.repo_writer_admission_closed():
            return
        carriers = lane_carriers()
        if any(getattr(slot, "readiness_exhausted", False) for slot in carriers):
            return
        idle = [w for w in carriers if w.busy_task_id is None and worker_ready(w) and not w.retiring]
        booting = [w for w in carriers if w.busy_task_id is None and w.booting and not w.retiring]
        spare_missing = not idle and not booting
        if len(idle) > 1:
            # Ask every idle carrier: a resident with late duties is not the spare. Only
            # acknowledged surplus beyond the one spare retires, through the reaper.
            for slot in idle:
                request_worker_quiescence(slot)
            quiet = [w for w in idle if w.quiescence_confirmed]
            for slot in quiet[1:]:
                retire_quiescent_worker(slot)
        next_id = max(pool.WORKERS, default=-1) + 1
    if spare_missing:
        try:
            _spawn_worker_slot(next_id, lane=LANE_CONVERSATION)
        except Exception:
            log.warning("The conversation lane's spare carrier could not be started", exc_info=True)
