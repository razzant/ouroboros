"""Resume a queued root through its existing owner-Pause fence and queue latch."""

from __future__ import annotations

import pathlib
from typing import Any, Dict

from ouroboros.control_actor import control_actor
from ouroboros.utils import utc_now_iso


def _resume_unstarted_owner_paused_root(q: Any, task: Dict[str, Any], fence: Dict[str, Any],
                                         observation: Dict[str, Any]) -> Dict[str, Any]:
    """The owner's Resume of a root the Pause caught while it was only queued.

    A never-started root has no members. An admitted retry needs a fresh tree
    census without unsettled effects (owner_pause_blockers); its attempt and
    dispatch evidence stay, and children still need explicit selection (Q9).
    Reopen both durable fence and queue latch or refuse: reopening only the
    latch would fail every launch. The latch names the LOGICAL root; task may
    be its current retry (#1730). Queue lock held.
    """
    from ouroboros.owner_pause import FENCE_RELEASED, launch_lock, read_fence, set_fence_state
    import copy
    from supervisor.events_budget import BUDGET_HOLD_KEY, hold_root_resume_descendants

    task_id, root_id = str(task.get("id") or ""), str(fence.get("root_task_id") or task.get("id") or "")
    admitted = "owner_pause_blockers" in observation
    safe, error = observation["safe"], observation["unsafe_error"]
    if not safe and not admitted:
        return {"ok": False, "error": error, "action": "cancel_or_new_run"}
    from ouroboros.work_stop import clear_refusal, current_stop, release_refusal
    if refusal := release_refusal(current_stop(task, {
            "owner": observation["authority"]["owner_fence"], "budget": q.BUDGET_ROOT_FENCES}), control_actor()):
        return refusal
    if observation.get("blockers"):
        from supervisor.continuation_admission import writers_unsettled_refusal

        return writers_unsettled_refusal(observation["blockers"])
    if observation.get("owner_pause_blockers"):
        return {"ok": False, "error": "owner_pause_effects_unsettled", "action": "wait_for_effect_settlement",
                "blockers": observation["owner_pause_blockers"]}
    root = pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)
    members = [row for row in q.PENDING if admitted and row is not task
               and str(row.get("root_task_id") or "") == root_id]
    prior = [(row, copy.deepcopy(row)) for row in [task, *members]]

    def rollback() -> None:
        for row, before in prior:
            row.clear()
            row.update(before)
        q.BUDGET_ROOT_FENCES[root_id] = fence

    with launch_lock(root, root_id):
        if read_fence(root, root_id) != observation["authority"]["owner_fence"]:
            return {"ok": False, "error": "selection_authority_changed"}
        # Publish queue eligibility first, while durable launch authority is still
        # closed. A crash or failed second write cannot start the unrun root.
        q.BUDGET_ROOT_FENCES.pop(root_id, None)
        task.pop("_budget_pause", None)
        task.pop(BUDGET_HOLD_KEY, None)
        clear_refusal(task)
        if members:
            # No root grant exists to bind: only the owner selects these members.
            hold_root_resume_descendants(q, root_id, fence, {"grant_id": "", "generation": 0}, actor_id=task_id)
        if not q.persist_queue_snapshot(reason="owner_pause_root_resumed"):
            rollback()
            return {"ok": False, "error": "snapshot_not_persisted"}
        try:
            set_fence_state(root, root_id, fence_id=str(fence.get("fence_id") or ""),
                            state=FENCE_RELEASED, release_reason="owner_resume_unstarted_root",
                            resume_actor=control_actor())
        except Exception as exc:
            rollback()
            q.persist_queue_snapshot(reason="owner_pause_root_resume_held")
            return {"ok": False, "error": "owner_pause_fence_unwritable", "detail": str(exc)[:200]}
    q.append_jsonl(q.DRIVE_ROOT / "logs" / "events.jsonl",
                   {"ts": utc_now_iso(), "type": "owner_pause_resumed", "task_id": task_id,
                    "root_task_id": root_id, "fence_id": str(fence.get("fence_id") or ""),
                    "same_generation": True, "never_started": not admitted})
    return {"ok": True, "task_id": task_id, "root_task_id": root_id, "same_generation": True,
            "owner_pause_released": True, "never_started": not admitted}
