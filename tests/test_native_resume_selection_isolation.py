"""A repaired native read gap cannot rebind or invalidate sibling selections."""
import copy

import pytest

from ouroboros import task_results
from supervisor import events_budget, queue, workers
from tests.test_native_wait_readgap_resume import _held
from tests.test_owner_wait_restart import restart_case as restart_case

pytestmark = pytest.mark.serial


def test_native_read_gap_resume_leaves_sibling_selections_untouched(restart_case, monkeypatch):
    case = restart_case
    case.task.update(root_task_id=case.task_id, delegation_role="root")
    env = _held(case, monkeypatch)
    assert not queue.BUDGET_ROOT_FENCES
    carriers = [
        {"_resume_selection": {"selected_by": "owner"}},
        {"_budget_pause_hold": {"reason": events_budget.HOLD_ROOT_FENCE_LIFTED,
                                 "root_grant_id": "older-root"}},
        {"_budget_pause_resume": {"authority": "explicit_resume", "selected_by": "owner",
                                   "grant_id": "kept-child-grant"}},
    ]
    siblings = []
    for index, carrier in enumerate(carriers):
        task_id = f"sibling-{index}"
        task_results.write_task_result(case.root, task_id, "scheduled",
                                       root_task_id=case.task_id, parent_task_id=case.task_id)
        row = {"id": task_id, "type": "task", "chat_id": 1, "depth": 1,
               "root_task_id": case.task_id, "parent_task_id": case.task_id,
               "delegation_role": "subagent", "text": "Retained sibling", **carrier}
        workers.PENDING.append(row)
        siblings.append(row)
    before = copy.deepcopy(siblings)
    resumed = queue.resume_budget_paused_task(case.task_id)
    assert resumed["ok"], resumed
    assert siblings == before
    assert "_budget_pause_hold" not in env.task
    assert not env.task.get("_budget_pause_resume")
    assert env.task["_owner_wait_resume"] == env.carrier
    assert events_budget.live_root_resume_grant(queue, case.task_id, case.root) == {}
    assert not env.sent and not queue.BUDGET_ROOT_FENCES
