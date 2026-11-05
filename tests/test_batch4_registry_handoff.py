"""Operation outcomes through the actual registry and loop, not a handoff helper."""
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests._store_shared import unreadable_result



@pytest.mark.parametrize("outcome", ["error", "timeout", "exception", "remote_start", "exit_error", "git_error", "legacy_warning"])
def test_business_outcomes_do_not_outlive_the_joined_registry_handler(tmp_path, monkeypatch, outcome):
    from ouroboros.loop_tool_execution import _execute_single_tool
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.tool_result import ToolResult
    from supervisor.continuation_admission import conflicting_writers

    queue, _, _ = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    calls = []

    def body(*_args, **_kwargs):
        calls.append("effect may have started")
        if outcome == "exception":
            raise TimeoutError("request sent; receipt lost")
        code = {"error": "TOOL_ERROR", "timeout": "TOOL_TIMEOUT", "remote_start": "OK",
                "exit_error": "SHELL_EXIT_ERROR", "git_error": "GIT_ERROR",
                "legacy_warning": "LEGACY_WARNING"}[outcome]
        return ToolResult(status="ok" if outcome in {"remote_start", "git_error", "legacy_warning"} else
                          "timeout" if outcome == "timeout" else "error", code=code,
                          text="remote job accepted" if outcome == "remote_start" else "outcome unavailable",
                          meta={"dynamic_provider": True} if outcome == "remote_start" else
                               {"exit_code": 1} if outcome == "exit_error" else {})

    registry.override_handler("knowledge_read", body)
    call = {"id": "call-1", "function": {"name": "knowledge_read", "arguments": '{"topic":"x"}'}}
    _execute_single_tool(registry, call, tmp_path / "logs", "root")
    before = load_task_result(tmp_path, "root").get("launch_handoffs", {})
    assert before == {} and calls == ["effect may have started"]
    write_task_result(tmp_path, "root", "failed", reason_code="task_exception")
    assert load_task_result(tmp_path, "root").get("launch_handoffs", {}) == before
    blockers = conflicting_writers(queue, "root")
    assert not any(row["kind"] == "tool_handoff" for row in blockers), blockers
    assert calls == ["effect may have started"], "classification never resends"


@pytest.mark.parametrize("bound", [False, True])
def test_real_local_return_and_registry_predispatch_refusals_settle(tmp_path, monkeypatch, bound):
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools.registry import ToolRegistry

    write_task_result(tmp_path, "root", "running", root_task_id="root")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    from ouroboros.tools import tool_discovery
    monkeypatch.setattr(tool_discovery, "_registry", None)
    if bound:
        tool_discovery.bind_resident_schemas(registry, registry.schemas())
    for name, args, code in (("list_available_tools", {}, "OK" if bound else "LEGACY_UNAVAILABLE"),
                             ("knowledge_read", {"not_an_argument": 1}, "TOOL_ARG_ERROR"),
                             ("there_is_no_such_tool", {}, "UNKNOWN_TOOL")):
        result = registry.execute_result(name, args)
        assert result.code == code
        assert load_task_result(tmp_path, "root").get("launch_handoffs", {}) == {}


def test_registry_and_predispatch_wait_disclose_unreadable_pause_authority(tmp_path, monkeypatch):
    from ouroboros.model_wait import TaskModelWait
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.owner_pause import install_fence

    write_task_result(tmp_path, "root", "running", root_task_id="root")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    ran = []
    registry.override_handler("knowledge_read", lambda *_a, **_kw: ran.append(True) or "ok")
    wait = TaskModelWait(task={"id": "root"}, drive_root=tmp_path, event_queue=None, worker_slot_held=True)
    install_fence(tmp_path, "root", request_id="p")
    known = registry.execute_result("knowledge_read", {"topic": "x"})
    assert known.meta["control_reason"] == wait.pre_dispatch_pause() == "owner_pause"
    assert "the owner paused" in known.text
    unreadable_result(monkeypatch, "root")
    unknown = registry.execute_result("knowledge_read", {"topic": "x"})
    assert unknown.meta["control_reason"] == wait.pre_dispatch_pause() == "owner_pause_authority_unreadable"
    assert "could not be read" in unknown.text and "the owner paused" not in unknown.text
    assert wait.control_reason() is None, "a sent operation still waits for its result"
    assert ran == []

    from ouroboros import budget_pause
    from ouroboros.loop_round_limits import _handle_model_wait_control
    from ouroboros.model_wait import ModelWaitInterrupted

    checkpoints = []
    monkeypatch.setattr(budget_pause, "request_pause", lambda *_a, **kw: checkpoints.append(kw.get("rail")))
    limit_ctx = SimpleNamespace(tools=registry, accumulated_usage={})
    with pytest.raises(ModelWaitInterrupted) as boundary:
        budget_pause.enter_owner_pause(limit_ctx)
    assert boundary.value.control_reason == "owner_pause_authority_unreadable"
    with pytest.raises(ModelWaitInterrupted) as routed:
        _handle_model_wait_control(limit_ctx, boundary.value)
    assert routed.value is boundary.value
    # An unreadable fence is neither Stop nor money: only the technical capture is attempted at the
    # boundary; when none is saved the same typed cause ends the work, and no fence id is invented.
    assert checkpoints == [budget_pause.RAIL_CONTROL_AUTHORITY] and not hasattr(registry._ctx, "_owner_pause_fence_id")
