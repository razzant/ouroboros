"""Only an unsent call and clean transport join can discharge MCP preparation."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from ouroboros import mcp_client, owner_pause, store
from tests.test_batch4_producer_custody import _registry
from tests.test_invocation_ownership import claims, owner, run
from tests.test_mcp_client import _settings



@pytest.mark.parametrize(("phase", "bound"), [
    ("open", False),
    ("sdk_error", True),
    ("pause", False),
    ("cleanup_unknown", True),
    ("after_send", False),
    ("init_error", True),
    ("init_timeout", False),
    ("init_cleanup_error", True),
    ("store_preparation", False),
    ("store_admission", True),
    ("store_cleanup_error", False),
    ("store_after_send", True),
    ("end_before_preparation", False),
    ("end_after_init", True),
    ("stop_after_init", False),
    ("end_after_send", True),
    ("end_cleanup_unknown", False),
])
def test_real_manager_keeps_only_unresolved_mcp_duty(tmp_path, monkeypatch, bound, phase):
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    registry, _, _ = _registry(tmp_path, monkeypatch)
    observed = []
    ended = ['owner_end' if phase == 'end_before_preparation' else None]

    @asynccontextmanager
    async def transport(_cfg):
        observed.append("transport_open")
        try:
            yield None, None
        finally:
            observed.append("transport_close")
            if phase in {"cleanup_unknown", "init_cleanup_error", "store_cleanup_error", "end_cleanup_unknown"}:
                raise RuntimeError("transport cleanup is unconfirmed")

    class Session:
        def __init__(self, *_args):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_args):
            observed.append("session_close")
        async def initialize(self):
            observed.append("initialized")
            if phase in {"init_error", "init_cleanup_error"}:
                raise RuntimeError("initialization failed before tools/call")
            if phase == "init_timeout":
                await asyncio.Event().wait()
            if phase in {"pause", "cleanup_unknown"}:
                owner_pause.install_fence(tmp_path, "root", request_id="mcp-before-call")
            if phase in {"end_after_init", "end_cleanup_unknown", "stop_after_init"}:
                ended[0] = 'finalize_requested' if phase == 'stop_after_init' else 'owner_end'
        async def call_tool(self, *_args):
            observed.append("call")
            if phase == 'end_after_send':
                ended[0] = 'owner_end'  # the started call still returns and closes its own custody
            if phase == "after_send":
                raise owner_pause.OwnerPauseRefused("a typed error after real send is not unsent proof")
            if phase == "store_after_send":
                raise store.StoreBusy("after method started", reason="uninspectable")
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="remote answer")],
                                   isError=phase == "sdk_error")

    monkeypatch.setattr(mcp_client, "_MCP_SDK_AVAILABLE", True)
    monkeypatch.setattr(mcp_client, "_transport_factory", transport)
    monkeypatch.setattr(mcp_client, "ClientSession", Session)
    manager = mcp_client.MCPManager()
    async def catalog(*_args):
        return [{"name": "effect", "description": "fixture", "input_schema": {"type": "object"}}]
    manager._async_list_tools = catalog
    manager.reconfigure(_settings({"id": "demo", "enabled": True, "transport": "streamable_http",
                                   "url": "https://unused.invalid/mcp"}))
    assert manager.refresh_server("demo")["ok"]
    if phase in {"store_preparation", "store_admission", "store_cleanup_error"}:
        def refuse(*_args, **_kwargs):
            raise store.StoreBusy("admission inspection failed", reason="uninspectable")
        monkeypatch.setattr(owner_pause, "submit_async_preparation" if phase == "store_preparation"
                            else "submit_async_operation", refuse)
    if phase == "init_timeout":
        manager._tool_timeout_sec = 0.02
    monkeypatch.setattr(registry, "_mcp_name_miss", lambda _name: None)
    monkeypatch.setattr(mcp_client, "_call_mcp_tool_result", manager._call_tool_result)
    def invoke():
        with owner_pause.call_end_reader(registry._ctx, lambda: ended[0]):  # as a live call's batch binds it
            if bound:
                with owner(registry):
                    return run(registry, name="mcp_demo__effect", args={}, timeout=5)["tool_result"]
            async def in_loop():
                return registry.execute_result("mcp_demo__effect", {})
            return asyncio.run(in_loop())  # real copied-context MCP runner thread

    if phase in {'end_before_preparation', 'end_after_init', 'stop_after_init'}:
        with pytest.raises(PhysicalDispatchInterrupted) as stopped:
            invoke()
        assert stopped.value.control_reason == ended[0]
        assert stopped.value.invocation_not_started is (phase == 'end_before_preparation')
        assert observed == ([] if phase == 'end_before_preparation' else
                            ['transport_open', 'initialized', 'session_close', 'transport_close'])
        assert not claims(tmp_path), 'a proven unsent method joined its preparation before propagating control'
        return
    result = invoke()
    assert observed.count("call") == (0 if phase in {
        "pause", "cleanup_unknown", "init_error", "init_timeout", "init_cleanup_error",
        "store_preparation", "store_admission", "store_cleanup_error", "end_cleanup_unknown"} else 1)
    assert observed == [] if phase == "store_preparation" else observed[-2:] == ["session_close", "transport_close"]
    remaining = claims(tmp_path)
    if phase in {"cleanup_unknown", "after_send", "init_cleanup_error", "store_cleanup_error", "store_after_send", "end_cleanup_unknown"}:
        assert result.status == "error" and len(remaining) == 1
        assert next(iter(remaining.values()))["external_pending"]
        assert result.meta.get("mcp_call_not_sent") is not True
    else:
        assert not remaining
        assert result.status == {"open": "ok", "end_after_send": "ok", "pause": "blocked", "init_timeout": "timeout",
                                 "store_preparation": "unavailable", "store_admission": "unavailable"}.get(phase, "error")
        if phase in {"store_preparation", "store_admission"}:
            assert result.code == "TOOL_ADMISSION_UNAVAILABLE"
            assert result.meta["mcp_call_not_sent"] is True
            assert result.meta["owner_pause_not_started"] is (phase == "store_preparation")
            assert result.meta["launch_reason"] == "uninspectable"
        if phase in {"init_error", "init_timeout"}:
            assert result.meta["mcp_call_not_sent"] is True
        if phase == "pause":
            assert result.meta["mcp_call_not_sent"] is True
            assert result.meta["owner_pause_not_started"] is False  # preparation already ran
        if phase in {"open", "sdk_error"}:
            assert result.meta["mcp_is_error"] is (phase == "sdk_error")
            assert "remote answer" in result.text


@pytest.fixture(autouse=True)
def _private_mcp_catalog_root(tmp_path, monkeypatch):
    from ouroboros import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
