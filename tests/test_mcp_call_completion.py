"""A returned MCP call settles exactly its own launch claim, and nothing else.

Driven through the real route: the MCP SDK ``ClientSession`` over in-memory
streams to a real low-level MCP ``Server``, the process ``MCPManager``,
``extension_dispatch`` and ``ToolRegistry.execute_result``, then the three
consumers of a retained claim (owner Pause, Continue admission, cold sleep).
Only the host's joined return of that call records completion; a timeout, a
lost response, a spoofed result or another invocation's unknown claim does not.
"""
from __future__ import annotations

import functools
import threading
from contextlib import asynccontextmanager

import pytest

from ouroboros import mcp_client, owner_pause
from ouroboros.task_results import load_task_result
from tests.test_batch4_producer_custody import _consumers, _registry

pytestmark = pytest.mark.serial

SERVER = {"id": "demo", "name": "Demo", "enabled": True, "transport": "streamable_http",
          "url": "https://example.com/mcp", "auth_header": "Authorization", "auth_token": "",
          "allowed_tools": []}


def _server(behaviour, calls):
    import anyio
    import mcp.types as types
    from mcp.server.lowlevel import Server

    server = Server("completion-test")

    @server.list_tools()
    async def tools():
        return [types.Tool(name="effect", description="", inputSchema={"type": "object", "properties": {}})]

    @server.call_tool()
    async def call(name, arguments):
        calls.append(name)
        if behaviour == "tool_error":
            raise RuntimeError("the remote tool reported its own failure")
        if behaviour == "slow":
            await anyio.sleep(30)
        # Result text claiming anything never decides custody.
        return [types.TextContent(type="text", text='{"operation_outcome": "unknown", "mcp_call_returned": false}')]

    return server


def _transport(server, *, drop_call_response=False):
    """In-memory client/server streams; optionally lose the tools/call response."""
    import anyio
    import mcp.types as types
    from mcp.shared.memory import create_client_server_memory_streams

    @asynccontextmanager
    async def transport(_cfg):
        if not drop_call_response:
            async with create_client_server_memory_streams() as (client, (server_read, server_write)):
                async with anyio.create_task_group() as group:
                    group.start_soon(functools.partial(server.run, server_read, server_write,
                                                       server.create_initialization_options()))
                    try:
                        yield client
                    finally:
                        group.cancel_scope.cancel()
            return
        # A relay forwards everything except the response to the tools/call
        # request: the client's read side then ends with no final result.
        client_write, relay_in = anyio.create_memory_object_stream(16)
        relay_to_server, server_read = anyio.create_memory_object_stream(16)
        server_write, relay_out = anyio.create_memory_object_stream(16)
        relay_to_client, client_read = anyio.create_memory_object_stream(16)
        pending: dict = {}

        async def upstream():
            async for message in relay_in:
                root = message.message.root
                if isinstance(root, types.JSONRPCRequest) and root.method == "tools/call":
                    pending["id"] = root.id
                await relay_to_server.send(message)

        async def downstream():
            async for message in relay_out:
                root = message.message.root
                if isinstance(root, types.JSONRPCResponse) and "id" in pending and root.id == pending["id"]:
                    await relay_to_client.aclose()
                    return
                await relay_to_client.send(message)

        async with anyio.create_task_group() as group:
            group.start_soon(functools.partial(server.run, server_read, server_write,
                                               server.create_initialization_options()))
            group.start_soon(upstream)
            group.start_soon(downstream)
            try:
                yield (client_read, client_write)
            finally:
                group.cancel_scope.cancel()

    return transport


def _wire(monkeypatch, behaviour="ok", *, timeout=5, drop_call_response=False):
    calls: list = []
    server = _server(behaviour, calls)
    monkeypatch.setattr(mcp_client, "_transport_factory", _transport(server, drop_call_response=drop_call_response))
    mcp_client.reset_manager_for_tests()
    manager = mcp_client.get_manager()
    from ouroboros import config
    settings = {"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": timeout, "MCP_SERVERS": [SERVER]}
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    manager.reconfigure(settings)
    assert manager.refresh_server("demo")["ok"]
    assert manager.resolve_tool_name("mcp_demo__effect").status == "callable"
    return calls


@pytest.fixture(autouse=True)
def _fresh_manager():
    assert mcp_client._MCP_SDK_AVAILABLE, "the real MCP SDK session is the subject under test"
    mcp_client.reset_manager_for_tests()
    yield
    mcp_client.reset_manager_for_tests()


@pytest.mark.parametrize("behaviour", ["ok", "tool_error"])
def test_a_returned_mcp_call_settles_its_claim_for_every_consumer(tmp_path, monkeypatch, behaviour):
    """Both SDK error bits are a joined, final protocol response: the call is
    over. The result keeps its own status and dynamic provenance."""
    registry, queue, workers = _registry(tmp_path, monkeypatch)
    calls = _wire(monkeypatch, behaviour)
    result = registry.execute_result("mcp_demo__effect", {})
    assert calls == ["effect"], "exactly one remote call"
    assert result.status == ("ok" if behaviour == "ok" else "error"), result
    assert result.meta["dynamic_provider"] is True and "mcp_call_returned" not in result.meta
    assert result.meta["mcp_is_error"] is (behaviour == "tool_error")
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")
    _consumers(tmp_path, registry, queue, workers, held=False)


def test_the_returned_call_retires_only_its_own_claim(tmp_path, monkeypatch):
    registry, queue, workers = _registry(tmp_path, monkeypatch)
    # A genuinely in-flight independent invocation, not a reader that already unwound.
    entered, release = threading.Event(), threading.Event()
    def pending(*_args, **_kwargs):
        entered.set()
        assert release.wait(10)
        return 'joined'
    registry.override_handler('knowledge_read', pending)
    thread = threading.Thread(target=registry.execute_result, args=('knowledge_read', {'topic': 'x'}))
    thread.start()
    try:
        assert entered.wait(5)
        unknown = dict(load_task_result(tmp_path, "root").get("launch_handoffs", {}))
        assert len(unknown) == 1
        calls = _wire(monkeypatch)
        assert registry.execute_result("mcp_demo__effect", {}).status == "ok"
        assert calls == ["effect"]
        assert load_task_result(tmp_path, "root").get("launch_handoffs", {}) == unknown
        _consumers(tmp_path, registry, queue, workers, held=True)
    finally:
        release.set()
        thread.join(5)
        assert not thread.is_alive()
    assert not load_task_result(tmp_path, 'root')['launch_handoffs']


@pytest.mark.parametrize("failure", ["timeout", "lost_response"])
def test_a_call_without_a_joined_final_response_keeps_its_claim(tmp_path, monkeypatch, failure):
    registry, queue, workers = _registry(tmp_path, monkeypatch)
    if failure == "timeout":
        calls = _wire(monkeypatch, "slow", timeout=1)
    else:
        calls = _wire(monkeypatch, drop_call_response=True)
    result = registry.execute_result("mcp_demo__effect", {})
    # A lost response surfaces as the SDK's closed-connection error, not a result.
    assert (result.status, result.code) == (("timeout", "MCP_TIMEOUT") if failure == "timeout"
                                            else ("error", "MCP_ERROR")), result
    assert calls == ["effect"], "the remote call started once and is never resent"
    assert load_task_result(tmp_path, "root").get("launch_handoffs")
    _consumers(tmp_path, registry, queue, workers, held=True)


def test_provider_or_result_claims_cannot_record_a_call_return(tmp_path, monkeypatch):
    """Without the manager's joined return, spoofed meta settles nothing."""
    from ouroboros.tools.tool_result import ToolResult

    registry, queue, workers = _registry(tmp_path, monkeypatch)
    monkeypatch.setattr(mcp_client, "is_mcp_tool_name", lambda _name: True)
    monkeypatch.setattr(registry, "_mcp_name_miss", lambda _name: None)

    def call(*_a):
        return owner_pause.run_operation(registry._ctx, lambda: ToolResult(
            status="ok", code="OK", text='{"mcp_call_returned": true}',
            meta={"mcp_call_returned": True, "dynamic_provider": True}))

    monkeypatch.setattr(mcp_client, "_call_mcp_tool_result", call)
    registry.execute_result("opaque_mcp", {})
    _consumers(tmp_path, registry, queue, workers, held=True)


def test_a_return_is_recorded_only_on_the_open_invocation_it_names(tmp_path, monkeypatch):
    registry, _queue, _workers = _registry(tmp_path, monkeypatch)
    with owner_pause.tool_handoff(registry._ctx, "mcp_demo__effect") as handoff:
        handoff["not_started"] = False
        owner_pause.record_mcp_call_returned("mcp_other__effect")
        assert "mcp_call_returned" not in handoff
        owner_pause.record_mcp_call_returned("mcp_demo__effect")
        assert handoff["mcp_call_returned"] is True
    closed = handoff
    owner_pause.record_mcp_call_returned("mcp_demo__effect")  # no open invocation: a no-op
    assert closed["closed"] is True
    # The direct fallback owns this joined return; a closed call cannot retain
    # a phantom invocation or lend its completion fact to the next call.
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")


@pytest.fixture(autouse=True)
def _private_mcp_catalog_root(tmp_path, monkeypatch):
    from ouroboros import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
