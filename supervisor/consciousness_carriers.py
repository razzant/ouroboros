"""Own-clock custody and the wider autonomous census across turn carriers."""

from ouroboros.consciousness_authority import (
    WORK_KIND_CONSCIOUSNESS, is_consciousness_origin, is_own_clock_wake, work_kind,
)


def queued_wakes(*, held_only: bool = False) -> list[str]:
    """Own-clock wakes the pool carries; ``held_only``: those in PENDING (paused, held, not yet dispatched)."""
    from supervisor import queue as q

    with q._queue_lock:
        tasks = [*q.PENDING, *([] if held_only else (meta.get("task") or {} for meta in q.RUNNING.values()))]
        return list(dict.fromkeys(str(task["id"]) for task in tasks if task.get("id")
                                 and is_own_clock_wake(task.get("metadata"))))


def is_autonomous_work(metadata) -> bool:
    """Work consciousness itself started, its tree included: the only work its switch stops.

    Unmarked work is checked at the consciousness Safety level, but it is not
    autonomous: Off must never stop system, owner or historical work.
    """
    return is_consciousness_origin(metadata) and work_kind(metadata) == WORK_KIND_CONSCIOUSNESS


def autonomous_tasks() -> list[str]:
    """All live autonomous pool members, including children and retained returns."""
    from supervisor import queue as q

    with q._queue_lock:
        tasks = [*q.PENDING, *(meta.get("task") or {} for meta in q.RUNNING.values())]
        return list(dict.fromkeys(str(task["id"]) for task in tasks if task.get("id")
                                 and is_autonomous_work(task.get("metadata"))))


def autonomous_work_enabled(task: dict, drive_root) -> bool:
    """Recheck Off/Panic at selection and launch, including restored autonomous work."""
    if not is_autonomous_work(task.get("metadata") or {}):
        return True
    from ouroboros.consciousness import panic_blocks_wake
    from supervisor.state import control_is, load_state

    return not panic_blocks_wake(drive_root) and control_is(load_state(), "bg_consciousness_enabled", True)


def stop_pooled_wake(task_id: str) -> None:
    from ouroboros.cancel_intents import request_cancel, STOP_POLICY_FINALIZE
    from supervisor import queue as q

    # The ordinary cancellation sweep owns the actual stop, including a row
    # still queued and the transition from inline park to a pooled attempt.
    request_cancel(q.DRIVE_ROOT, task_id, reason="background_consciousness_stopped",
                   source="owner", requested_by="owner", scope="cascade",
                   requested_stop_policy=STOP_POLICY_FINALIZE)


def finish_pooled_wake(ctx, task_id: str, result: dict) -> None:
    """The detached attempt's end starts the ordinary next-wake interval."""
    from ouroboros.task_carrier import execution_kind

    clock = getattr(ctx, "consciousness", None)
    if (clock is not None and execution_kind(result) == "pooled"
            and is_own_clock_wake(result.get("metadata"))):
        clock._wake_finished(task_id, result.get("status") == "completed", metadata=result["metadata"], pooled=True)
