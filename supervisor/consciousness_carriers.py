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


def live_task_rows(task_ids) -> dict[str, dict]:
    """The pool's own rows of these live tasks, from memory: each one's ``title`` and ``metadata`` (what names them
    when the switch ends them); nothing is read from a store."""
    from supervisor import queue as q

    wanted = {str(task_id) for task_id in task_ids}
    with q._queue_lock:
        tasks = [*q.PENDING, *(meta.get("task") or {} for meta in q.RUNNING.values())]
        return {str(task["id"]): {"title": str(task.get("title") or task.get("suggested_name") or ""),
                                  "metadata": dict(task["metadata"]) if isinstance(task.get("metadata"), dict) else {}}
                for task in tasks if str(task.get("id") or "") in wanted}


def paused_tree_members(root_task_id: str) -> list[tuple[str, str]]:
    """``(task_id, budget_drive_root)`` of the members of this root's tree the pool holds parked at the graceful
    ceiling (their ``_budget_pause`` marker), from memory: the members whose pause rows can say a child paused first."""
    from ouroboros.budget_pause import RAIL_GRACEFUL_CEILING
    from supervisor import queue as q

    root = str(root_task_id or "")
    with q._queue_lock:
        return [(str(task["id"]), str(task.get("budget_drive_root") or "")) for task in q.PENDING
                if task.get("id") and str(task["id"]) != root and isinstance(task.get("_budget_pause"), dict)
                and str(task["_budget_pause"].get("root_task_id") or task.get("root_task_id") or "") == root
                and task["_budget_pause"].get("rail") == RAIL_GRACEFUL_CEILING]


def _cited_quote(corpus) -> str:
    """The span a run cited from my human's place words (``cited``; older rows hold only the span as ``content``)."""
    return next((str(row.get("cited") or row.get("content") or "") for row in corpus or []
                 if isinstance(row, dict) and row.get("source") == "place_owner" and row.get("basis") == "cited"), "")


def cited_owner_roots() -> list[tuple[str, str]]:
    """Live roots the switch does not stop because they run on my human's cited words: ``(task_id, quote)`` for every
    pending or running owner-kind task whose corpus holds a ``place_owner`` row with ``basis: cited``."""
    from supervisor import queue as q

    with q._queue_lock:
        tasks = [*q.PENDING, *(meta.get("task") or {} for meta in q.RUNNING.values())]
    found: dict[str, str] = {}
    for task in tasks:
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        if not task.get("id") or work_kind(metadata) != "owner":
            continue
        quote = _cited_quote(metadata.get("owner_corpus"))
        if quote:
            found.setdefault(str(task["id"]), quote)
    return list(found.items())


def cited_owner_schedules(drive_root) -> list[tuple[str, str]]:
    """Enabled schedules and follow-ups the switch leaves firing because they run on my human's cited words:
    ``(schedule_id, quote)`` for every owner-kind record whose next occurrence carries such a row."""
    from ouroboros.consciousness_authority import schedule_owner_corpus
    from supervisor.queue import list_scheduled_tasks

    found = []
    for record in list_scheduled_tasks(drive_root).get("tasks") or []:
        if not isinstance(record, dict) or not record.get("id") or record.get("enabled") is False or record.get("writer_kind") != "owner":
            continue
        quote = _cited_quote(schedule_owner_corpus(record))
        if quote:
            found.append((str(record["id"]), quote))
    return found


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


def pooled_terminal_ok(result: dict):
    """A carried wake's recorded end as the clock reads it; None while nothing settled it (``interrupted``
    stays with its restore owner, a legacy in-process row that predates the carrier facts with the stack that
    ran it). Every lane carrier reports a wake's end only through this terminal, a worker and the disclosed
    degraded executor alike (one execution model): none calls the clock back.
    An early ``completed`` keeps its sticky lifecycle beside the fault's ``infra_failed`` axis."""
    from ouroboros.outcomes import EXECUTION_INFRA_FAILED, normalize_outcome_axes
    from ouroboros.task_carrier import carrier_executor, inline_execution
    from ouroboros.task_status import SETTLED_STATUSES

    status = result.get("status")
    if status not in SETTLED_STATUSES or (inline_execution(result) and not carrier_executor(result)):
        return None
    return status == "completed" and normalize_outcome_axes(result)["execution"]["status"] != EXECUTION_INFRA_FAILED


def finish_pooled_wake(ctx, task_id: str, result: dict, known: dict | None = None) -> None:
    """The detached attempt's end starts the ordinary next-wake interval. ``known``: the identity
    its RUNNING row or terminal event still carries when the durable result lost it."""
    known = known or {}
    clock = getattr(ctx, "consciousness", None)
    metadata = result.get("metadata") or known.get("metadata")
    ok = pooled_terminal_ok({**known, **result, "status": result.get("status")})
    if clock is not None and ok is not None and is_own_clock_wake(metadata):
        clock._wake_finished(task_id, ok, metadata=metadata, pooled=True)
