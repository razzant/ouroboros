"""Returned local reads and the running-loop MCP settlement interleaving."""
import asyncio
import threading
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_batch4_repair_compositions import _running

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("name,args", [
    ("read_file", {"path": "absent", "root": "active_workspace"}),
    ("list_files", {"path": "absent", "root": "active_workspace"}),
    ("search_code", {"query": "needle", "path": "absent"}),
    ("query_code", {"op": "invalid"}),
    ("view_image", {"path": "absent.png"}),
    ("list_skills", {}),
    ("compare_subagent_patches", {"task_ids": []}),
    ("knowledge_list", {"scope": "invalid"}),
    ("journal_read", {}),
    ("workpad_read", {}),
    ("get_task_result", {"task_id": "absent"}),
    ("peek_task", {"task_id": "absent", "view": "invalid"}),
])
def test_completed_local_read_warning_releases_tree_custody(tmp_path, monkeypatch, name, args):
    from ouroboros.owner_pause import read_fence
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools.registry import ToolRegistry
    from supervisor.continuation_admission import conflicting_writers
    from supervisor.owner_pause_control import refresh_owner_pause_tree, request_owner_pause

    q, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    result = registry.execute_result(name, args)
    assert result.code != "UNKNOWN_TOOL", result
    assert not load_task_result(tmp_path, "root").get("launch_handoffs"), result
    assert result.meta.get("operation_outcome") == "completed_no_effect", result
    assert request_owner_pause("root", request_id="completed-read")["ok"]
    workers.RUNNING.clear()
    refresh_owner_pause_tree("root")
    assert read_fence(tmp_path, "root")["state"] == "paused"
    write_task_result(tmp_path, "root", "failed", reason_code="task_exception")
    assert not conflicting_writers(q, "root")


@pytest.mark.parametrize("outcome", ["warning", "exception", "timeout", "dynamic"])
def test_result_metadata_cannot_override_actual_handler_unwind(tmp_path, monkeypatch, outcome):
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.tool_result import ToolResult

    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    registry.execute_result("read_file", {"path": "absent"})
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")
    entered = []
    def opaque(*_args, _resolved_binding=None, **_kwargs):
        entered.append(True)
        if outcome == "exception":
            raise OSError("unknown after dispatch")
        if outcome == "warning":
            return "⚠️ NOT_FOUND: opaque remote result unknown"
        return ToolResult(status="timeout" if outcome == "timeout" else "ok",
                          code="MCP_TIMEOUT" if outcome == "timeout" else "LEGACY_WARNING",
                          text="opaque", meta={"dynamic_provider": True,
                              "operation_outcome": "completed_no_effect"})
    registry.override_handler("read_file", opaque)
    registry.execute_result("read_file", {"path": "absent", "root": "active_workspace"})
    assert entered == [True]
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")


def test_real_reader_exception_keeps_business_error_without_ghost_invocation(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.registry import ToolRegistry

    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    # The actual query reader raises while parsing its numeric options.
    result = registry.execute_result("query_code", {"op": "symbols", "limit": "invalid"})
    assert result.status == "error" and not result.meta.get("operation_outcome")
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")


def test_mcp_start_between_timeout_result_and_handoff_close_keeps_custody(tmp_path, monkeypatch):
    from ouroboros import mcp_client, owner_pause
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.tool_result import ToolResult

    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    registry._ctx.task_metadata = {"work_kind": "owner"}  # an owner task's minted kind
    monkeypatch.setattr(registry, "_mcp_name_miss", lambda _name: None)
    initialize, entered, finish = threading.Event(), threading.Event(), threading.Event()
    threads, sent = [], []
    @asynccontextmanager
    async def transport(_cfg):
        yield (None, None)
    class Session:
        def __init__(self, *_a):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_a):
            pass
        async def initialize(self):
            threads.append(threading.current_thread())
            assert initialize.wait(5)
        async def call_tool(self, *_a):
            sent.append("effect")
            entered.set()
            assert finish.wait(5)
            return SimpleNamespace(content=[], isError=False)
    monkeypatch.setattr(mcp_client, "_MCP_SDK_AVAILABLE", True)
    monkeypatch.setattr(mcp_client, "_transport_factory", transport)
    monkeypatch.setattr(mcp_client, "ClientSession", Session)
    def timed_call(*_a):
        try:
            return mcp_client._run_async(
                lambda: mcp_client._call_tool_async(None, "effect", {}, timeout_sec=10), join_timeout=0)
        except TimeoutError:
            return ToolResult(status="timeout", code="MCP_TIMEOUT", text="wait elapsed",
                              meta={"dynamic_provider": True})
    monkeypatch.setattr(mcp_client, "_call_mcp_tool_result", timed_call)
    real_handoff = owner_pause.tool_handoff
    @contextmanager
    def at_close(*args, **kwargs):
        with real_handoff(*args, **kwargs) as handoff:
            yield handoff
            # The registry has returned its timeout and set its settlement fact.
            # The real runner now starts under the real launch lock, before close.
            initialize.set()
            assert entered.wait(5)
    monkeypatch.setattr(owner_pause, "tool_handoff", at_close)
    async def execute():
        return registry.execute_result("mcp_demo__effect", {})
    try:
        assert asyncio.run(execute()).status == "timeout"
        assert sent == ["effect"]
        assert load_task_result(tmp_path, "root").get("launch_handoffs")
    finally:
        initialize.set()
        finish.set()
        for thread in threads:
            thread.join(5)
            assert not thread.is_alive()
