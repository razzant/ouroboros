"""Owner controls remain typed at the browser submit and queued selection seams."""

import copy

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_budget_pause_holds import _fenced_member, _idle_worker



@pytest.mark.parametrize("marker", [False, True])
def test_repeated_resume_of_selected_fenced_root_is_typed_and_preserves_one_dispatch(tmp_path, monkeypatch, marker):
    from supervisor.events_budget import _set_root_budget_pause_locked
    from supervisor.queue_transitions import budget_pause_fact

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 5.0)
    root = _fenced_member(workers, "root", "root")
    root.pop("parent_task_id")
    sibling = _fenced_member(workers, "sibling", "root")
    fence = _set_root_budget_pause_locked("root", {})
    if marker:
        root["_budget_pause"] = {**fence, "physical_calls": 0, "replay_safe": True}
    first = queue.resume_budget_paused_task("root")
    assert first["ok"] and first["selection"] == "budget_hold_released"
    assert budget_pause_fact(root) is None
    selected = copy.deepcopy(root)
    second = queue.resume_budget_paused_task("root")
    assert second["ok"] is False and second["error"] == "task_not_budget_paused"
    assert root == selected and queue.BUDGET_ROOT_FENCES["root"] == fence
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["root"]
    assert sibling in workers.PENDING and budget_pause_fact(sibling) is not None
