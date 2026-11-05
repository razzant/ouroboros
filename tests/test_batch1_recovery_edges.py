"""Public consumer regressions for the repaired recovery candidate."""
import pathlib

import pytest

from tests.test_schedule_occurrence import _row, _rows, q  # noqa: F401
from tests.test_tool_call_log import _call, _Registry
from tests.test_tool_call_log import _rows as tool_rows


def test_readonly_input_scan_failure_and_symlink_refuse_before_engine(tmp_path, monkeypatch):
    from ouroboros.delegate_readonly_inputs import prepare_readonly_inputs
    from tests.test_delegated_skill_payload import _payload_ctx

    ctx = _payload_ctx(tmp_path, monkeypatch)
    ctx.task_metadata["resource_intent"] = {"kind": "explicit_none"}
    scratch = ctx.active_repo_dir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (scratch / "delegated_readonly_inputs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        prepare_readonly_inputs(ctx, "invocation")
    assert list(outside.iterdir()) == []


def test_full_dispatch_source_is_canonical_and_available_before_execution(tmp_path):
    from ouroboros.observability import read_call_payload
    from ouroboros.tools.tool_result import ToolResult

    arguments = {"command": "x" * 12000, "nested": {"late": list(range(300))}}

    def handler(_tool, _args):
        [start] = tool_rows(tmp_path / "canonical/logs/tools.jsonl")
        assert start["args_source_status"] == "ready"
        _, payload, _ = read_call_payload(tmp_path / "canonical", task_id="task-1",
                                         call_id=f"tool_dispatch_{start['invocation_id']}")
        assert payload["arguments"] == arguments
        return ToolResult(status="ok", code="OK", text="done")

    _call(_Registry(tmp_path, handler), tmp_path, args=arguments)


@pytest.mark.parametrize("bound", [False, True], ids=["unbound-logs-only", "bound-authority"])
def test_dispatch_source_and_append_failures_disclose_loss_without_veto(tmp_path, monkeypatch, bound):
    from ouroboros import budget_pause
    from ouroboros.tools.tool_result import ToolResult

    called = []
    registry = _Registry(tmp_path, lambda *_: called.append(True) or ToolResult(status="ok", code="OK", text="done"))
    resolve = pathlib.Path.resolve
    canonical = tmp_path / "canonical"
    def unavailable(path, *args, **kwargs):
        if path == canonical:
            raise OSError(5, "canonical unavailable")
        return resolve(path, *args, **kwargs)
    monkeypatch.setattr(pathlib.Path, "resolve", unavailable)
    if bound:
        registry._ctx.task_id = "task-1"
        with pytest.raises(OSError, match="canonical unavailable"):
            _call(registry, tmp_path)
        assert not called, "unreadable required task identity must still prevent submission"
        return
    result, logs = _call(registry, tmp_path)
    assert called and result["result"] == "done"
    [start, settled] = tool_rows(logs / "tools.jsonl")
    assert start["args_source_status"] == "unavailable:OSError"
    assert settled["start_log"]["canonical"] is False
    assert budget_pause.tool_scope_key(registry._ctx) == ""
    assert "" not in budget_pause._TOOL_FUTURES
    budget_pause.hold_tool_settlement(registry._ctx, start["invocation_id"])()
    budget_pause.forget_tool_scope(registry._ctx)
    assert budget_pause.drain_local_tool_futures(registry._ctx, timeout_sec=0)["drained"]
