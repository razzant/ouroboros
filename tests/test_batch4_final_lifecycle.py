"""Final handoff mirrors and narrow sleep diagnostic/cause regressions."""
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_budget_pause_holds import _idle_worker
from tests._store_shared import unreadable_result



@pytest.mark.parametrize("refusal", ["pause", "snapshot", "none", "newer", "same", "newer_scheduled"])
def test_real_assignment_mirror_obeys_final_handoff(tmp_path, monkeypatch, refusal):
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.owner_pause_control import request_owner_pause

    q, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 5)
    write_task_result(tmp_path, "root", "scheduled", root_task_id="root")
    task = {"id": "root", "root_task_id": "root", "type": "task", "chat_id": 0,
            "drive_root": str(tmp_path), "admitted_dispatch": "none", "_attempt": 1}
    workers.PENDING.append(task)
    observed = []
    class Queue:
        def append(self, candidate):
            observed.append(load_task_result(tmp_path, "root"))
    _idle_worker(workers, Queue())
    def prepare(_candidate):
        if refusal == "pause":
            assert request_owner_pause("root", request_id="at-final-gate")["ok"]
        if refusal in {"newer", "same", "newer_scheduled"}:
            write_task_result(tmp_path, "root", "scheduled" if refusal == "newer_scheduled" else "running",
                              task_attempt=1 if refusal == "same" else 2,
                              result="newer worker facts", description="newer description")
        return ""
    monkeypatch.setattr(workers, "_evolution_assignment_error", prepare)
    persist = q.persist_queue_snapshot
    monkeypatch.setattr(q, "persist_queue_snapshot", lambda **kw: False if
        refusal == "snapshot" and kw.get("reason") == "worker_launch_claimed" else persist(**kw))
    workers.assign_tasks()
    stored = load_task_result(tmp_path, "root")
    if refusal in {"pause", "snapshot"}:
        assert not observed and workers.PENDING == [task]
        assert stored["status"] == "scheduled"
    elif refusal in {"newer", "same", "newer_scheduled"}:
        assert observed[0]["task_attempt"] == (1 if refusal == "same" else 2)
        assert stored["result"] == "newer worker facts"
        assert stored["description"] == "newer description"
    else:
        assert len(observed) == 1 and observed[0]["status"] == "running"


def test_malformed_member_sleep_authority_is_typed_and_not_owner_intent(tmp_path, monkeypatch):
    from ouroboros import owner_pause
    from ouroboros.task_results import write_task_result

    _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    write_task_result(tmp_path, "child", "running", root_task_id="root")
    unreadable_result(monkeypatch, "child")
    source = SimpleNamespace(task_id="child", root_task_id="root", budget_drive_root=tmp_path)
    with pytest.raises(owner_pause.OwnerPauseRefused, match="model_sleep_authority_unreadable"):
        with owner_pause.launch_admission(source):
            pytest.fail("unreadable authority must not dispatch")
    assert not owner_pause.read_fence(tmp_path, "root")
    from ouroboros.tools.registry import ToolRegistry
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id, registry._ctx.root_task_id = "child", "root"
    result = registry.execute_result("knowledge_list", {})
    assert result.meta["control_reason"] == "model_sleep_authority_unreadable"
    # Pause and sleep share one unreadable-authority refusal: the typed reason names the
    # sleep authority, and the text never attributes the refusal to an owner Pause.
    assert "authority could not be read" in result.text
    assert "(model_sleep_authority_unreadable)" in result.text and "the owner paused" not in result.text
    assert not owner_pause.read_fence(tmp_path, "root")
