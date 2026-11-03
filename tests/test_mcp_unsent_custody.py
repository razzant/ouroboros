"""Only an unsent call and clean transport join can discharge MCP preparation."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from ouroboros import mcp_client, owner_pause, store
from tests.test_batch4_producer_custody import _registry
from tests.test_invocation_ownership import claims, owner, run
from tests.test_mcp_client import _settings

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.parametrize("phase", ["open", "sdk_error", "pause", "cleanup_unknown", "after_send",
                                  "init_error", "init_timeout", "init_cleanup_error",
                                  "store_preparation", "store_admission", "store_cleanup_error", "store_after_send"])
def test_real_manager_keeps_only_unresolved_mcp_duty(tmp_path, monkeypatch, bound, phase):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    observed = []

    @asynccontextmanager
    async def transport(_cfg):
        observed.append("transport_open")
        try:
            yield None, None
        finally:
            observed.append("transport_close")
            if phase in {"cleanup_unknown", "init_cleanup_error", "store_cleanup_error"}:
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
        async def call_tool(self, *_args):
            observed.append("call")
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
    if bound:
        with owner(registry):
            result = run(registry, name="mcp_demo__effect", args={}, timeout=5)["tool_result"]
    else:
        async def in_loop():
            return registry.execute_result("mcp_demo__effect", {})
        result = asyncio.run(in_loop())  # real copied-context MCP runner thread
    assert observed.count("call") == (0 if phase in {
        "pause", "cleanup_unknown", "init_error", "init_timeout", "init_cleanup_error",
        "store_preparation", "store_admission", "store_cleanup_error"} else 1)
    assert observed == [] if phase == "store_preparation" else observed[-2:] == ["session_close", "transport_close"]
    remaining = claims(tmp_path)
    if phase in {"cleanup_unknown", "after_send", "init_cleanup_error", "store_cleanup_error", "store_after_send"}:
        assert result.status == "error" and len(remaining) == 1
        assert next(iter(remaining.values()))["external_pending"]
        assert result.meta.get("mcp_call_not_sent") is not True
    else:
        assert not remaining
        assert result.status == {"open": "ok", "pause": "blocked", "init_timeout": "timeout",
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
