"""Owner controls remain typed at the browser submit and queued selection seams."""

import copy
import json
import threading

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_budget_pause_holds import _fenced_member, _idle_worker

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("paused", [False, True])
def test_browser_submit_before_handoff_returns_pause_control_without_running_handler(tmp_path, monkeypatch, paused):
    from ouroboros import loop_tool_execution as execution, owner_pause
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tool_call_log import CALL_SETTLED
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    registry._ctx.task_lifecycle_bound = True
    entered = []
    registry.override_handler("browser_action", lambda _ctx, **_kw: entered.append(threading.get_ident()) or "browser done")
    if paused:
        owner_pause.install_fence(tmp_path, "root", request_id="before-browser-submit")
    executor = execution.StatefulToolExecutor()
    call = {"id": "browser-call", "type": "function", "function": {
        "name": "browser_action", "arguments": json.dumps({"action": "snapshot"})}}
    try:
        result = execution._execute_with_timeout(registry, call, tmp_path / "logs", 10, "root", executor)
    finally:
        executor.shutdown()
    assert result["tool_call_id"] == "browser-call"
    assert result["result"] == result["tool_result"].text
    rows = [json.loads(line) for line in (tmp_path / "logs/tools.jsonl").read_text(encoding="utf-8").splitlines()]
    settlements = [row for row in rows if row.get("type") == CALL_SETTLED]
    assert len(settlements) == 1
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")
    if paused:
        assert entered == []
        assert result["tool_result"].code == "OWNER_PAUSE_NOT_STARTED"
        assert result["tool_result"].meta["owner_pause_not_started"] is True
        assert result["tool_result"].meta["control_reason"] == "owner_pause"
        assert settlements[0]["status"] != "host_error"
        assert owner_pause.read_fence(tmp_path, "root")["state"] == "requested"
    else:
        assert len(entered) == 1 and entered[0] != threading.get_ident()
        assert result["result"] == "browser done" and not result["is_error"]


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
