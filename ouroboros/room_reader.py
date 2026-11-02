"""Structural reader facts shared by room intake and addressed question answers.

Queued work remains addressable explicitly but cannot consume ordinary room
input. A live actor waiting for its own answer is still reading; on the question
path so is an unheld cold sleep, which room intake never treats as a reader.
No delivery here releases an owner, budget, restored-process or project
admission hold.
"""
from __future__ import annotations


def is_reading(root, task_id: str, task: dict, *, running: bool, fences=None, drain_ended=None) -> bool:
    """``drain_ended``: the caller's own read of the actor's settlement, else read here."""
    if not running or task.get("_owner_hold") or task.get("_project_admission_hold"):
        return False
    from ouroboros.owner_mailbox import mailbox_drain_ended
    from ouroboros.owner_pause import fence_closed, read_fence
    from supervisor.queue_transitions import budget_pause_fact
    if budget_pause_fact(task, fences) is not None:
        return False
    if fence_closed(read_fence(root, str(task.get("root_task_id") or task_id))):
        return False
    return not (mailbox_drain_ended(root, task_id) if drain_ended is None else drain_ended)


def unheld_cold_sleep(root, task_id: str, task: dict, *, fences=None) -> bool:
    """A cold sleep, or its own grant awaiting a worker, under no hold, owner Pause or money pause.

    Its question mail wakes it through sleep readiness, whose vetoes still apply;
    a sleep pause without selectors waits for Resume like any interrupted pause.
    """
    pause = task.get("_budget_pause")
    if not isinstance(pause, dict):
        grant = task.get("_budget_pause_resume")
        pause = grant.get("pause") if isinstance(grant, dict) else None
    if not isinstance(pause, dict) or pause.get("reason") != "sleep" or not isinstance(pause.get("sleep"), dict):
        return False
    if task.get("_owner_hold") or task.get("_project_admission_hold"):
        return False
    from ouroboros.owner_mailbox import mailbox_drain_ended
    from ouroboros.owner_pause import fence_closed, read_fence
    from supervisor.queue_transitions import budget_pause_fact
    if budget_pause_fact({k: v for k, v in task.items() if k != "_budget_pause"}, fences) is not None:
        return False  # a hold, or a money fence over its tree
    if fence_closed(read_fence(root, str(task.get("root_task_id") or task_id))):
        return False
    return not mailbox_drain_ended(root, task_id)
