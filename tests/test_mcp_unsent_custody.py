"""Only an unsent call and clean transport join can discharge MCP preparation."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from ouroboros import mcp_client, owner_pause, store
from tests.test_batch4_producer_custody import _registry
from tests.test_invocation_ownership import claims, owner, run
from tests.test_mcp_client import _settings
from tests.test_mcp_task_browser_bridge import manager as _manager
from tests.test_mcp_task_browser_bridge import policy, safety  # noqa: F401 - bridge fixtures
from tests.test_mcp_task_browser_bridge import posix_bridge

manager = _manager  # Register the shared pytest fixture without an unused import.

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


@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.parametrize("outcome", ["refused", "safety", "returned", "sdk_error", "timeout",
                                    "broken", "broken_unsent", "unknown", "independent_duty", "pre_read_timeout"])
def test_registered_bridge_discharges_only_its_positive_unsent_receipt(tmp_path, monkeypatch, bound, outcome):
    from ouroboros import mcp_chrome_protocol, mcp_task_sessions, tool_custody
    from ouroboros.model_sleep import cold_blockers
    from tests.test_mcp_chrome_session import _session

    registry, _, _ = _registry(tmp_path, monkeypatch)
    ctx = registry._ctx
    ctx.task_lifecycle_bound = True
    manager = mcp_client.MCPManager()
    manager.reconfigure(_settings({"id": "browser", "enabled": True, "transport": "stdio",
                                   "command": "unused", "browser_bridge": True,
                                   "browser_protocol": "ouroboros_chrome"}))
    cfg = manager._servers["browser"].config
    prefixed = "mcp_browser__browser_action"
    manager._servers["browser"].tools = [mcp_client.MCPTool(
        "browser", "browser_action", prefixed, "fixture", {"type": "object"})]
    monkeypatch.setattr(mcp_client, "get_manager", lambda: manager)
    session = _session()
    session.cfg, session.ctx, session.key = cfg, ctx, (ctx.task_id, cfg.id)
    session.data_root = tmp_path
    session._chrome_state["owner"]["task_id"] = ctx.task_id
    session._chrome_invocation = None
    session._read_stream = None
    if outcome != "pre_read_timeout":
        session._observe_before = session._observe = lambda *_a, **_k: session._chrome_state
    monkeypatch.setitem(mcp_task_sessions._sessions, session.key, session)
    monkeypatch.setattr(mcp_chrome_protocol, "target_reason", lambda *_a: "")
    monkeypatch.setattr("ouroboros.safety.check_safety",
                        lambda *_a, **_k: (outcome != "safety", "host safety refusal" if outcome == "safety" else ""))
    returned, dispatched, independent, pending_before, state_reads = [], [], [], [], []
    record_returned = owner_pause.record_mcp_call_returned

    def returned_call(name):
        returned.append(name)
        record_returned(name)

    monkeypatch.setattr(owner_pause, "record_mcp_call_returned", returned_call)

    def binding(*_args):
        pending_before.append(dict(next(iter(claims(tmp_path).values()))["external_pending"]))
        if outcome == "independent_duty":
            independent.append(tool_custody.external_handoff("independent_fixture"))
            return "BROWSER_SKILL_BINDING_REVOKED: no action submitted"
        return ""

    monkeypatch.setattr(mcp_task_sessions, "_chrome_binding_reason", binding)
    # A binding refusal stops the session; that close receipt is deliberately not
    # evidence for either the action token or an independent external duty.
    session.stop = lambda: "unconfirmed: fixture retains session custody"

    def request(name, *_args, **_kwargs):
        if outcome == "pre_read_timeout" and name == "browser_state":
            state_reads.append(name)
            raise mcp_task_sessions.BrowserBridgeTimeout("pre-action state read timed out")
        with owner_pause.operation_start():
            dispatched.append(True)
        if outcome == "timeout":
            raise mcp_task_sessions.BrowserBridgeTimeout("submitted action timed out")
        if outcome in {"broken", "broken_unsent"}:
            raise mcp_task_sessions.BrowserBridgeBroken(
                "no final response", submitted=outcome == "broken", closure="confirmed")
        if outcome == "unknown":
            raise RuntimeError("the action was not dispatched")  # text is not a host receipt
        return SimpleNamespace(isError=outcome == "sdk_error", content=[
            SimpleNamespace(type="text", text="bridge response")])

    session._request = request
    args = {"operation": "unknown" if outcome == "refused" else "snapshot", "args": {}}
    if bound:
        with owner(registry):
            result = run(registry, name=prefixed, args=args, timeout=5)["tool_result"]
    else:
        result = registry.execute_result(prefixed, args)
    assert pending_before and all(set(row.values()) == {"mcp_transport"} for row in pending_before)
    refused = outcome in {"refused", "safety", "independent_duty", "pre_read_timeout"}
    answered = outcome in {"returned", "sdk_error"}
    assert len(dispatched) == (0 if refused else 1)
    assert returned == ([prefixed] if answered else [])
    if outcome == "pre_read_timeout":
        assert state_reads == ["browser_state"] and "not dispatched" in result.text
    assert result.status == ("blocked" if refused else "timeout" if outcome == "timeout"
                             else "ok" if outcome == "returned" else "error"), result
    if outcome == "timeout":
        assert result.text == "⚠️ MCP_TOOL_TIMEOUT: submitted action timed out"
    if outcome in {"broken", "broken_unsent"}:
        assert result.meta["bridge_closure"] == "confirmed"
        assert result.meta["effect"] == ("unknown" if outcome == "broken" else "none")
    remaining = claims(tmp_path)
    blockers = cold_blockers(ctx)
    if (refused and outcome != "independent_duty") or answered:
        assert not remaining, (remaining, blockers)
        assert not blockers
    else:
        assert len(remaining) == 1
        pending = next(iter(remaining.values()))["external_pending"]
        assert set(pending.values()) == ({"independent_fixture"} if independent else {"mcp_transport"})
        if not independent:
            assert pending == pending_before[0]  # the exact original token stays unknown
        assert any(item["kind"] == "member_custody" for item in blockers)
        if independent:
            independent[0]()
            assert not claims(tmp_path) and not cold_blockers(ctx)


@posix_bridge
def test_registered_unsent_refusal_keeps_only_independent_browser_session(manager, tmp_path, monkeypatch):
    from ouroboros import mcp_task_sessions
    from ouroboros.model_sleep import cold_blockers
    from tests.test_mcp_task_browser_bridge import _live, _session

    registry, _, _ = _registry(tmp_path, monkeypatch)
    ctx = registry._ctx
    ctx.task_lifecycle_bound = True
    # The real discovered stdio session has its own process custody, independent
    # of the invocation's pre-dispatch transport token.
    assert manager.instance.refresh_server("browser", authority=ctx)["ok"]
    session = _session(ctx)
    marker = session.token
    try:
        result = registry.execute_result("mcp_browser__browser_tabs", {"action": "select", "index": 0})
        assert (result.status, result.code) == ("blocked", "LEGACY_BLOCKED")
        remaining = claims(tmp_path)
        blockers = cold_blockers(ctx)
        assert session.is_current() and _live(marker)
        assert any(item["kind"] == "owned_process" for item in blockers), blockers
        assert not remaining, (remaining, blockers)
        assert not any(item["kind"] == "member_custody" for item in blockers), blockers
    finally:
        mcp_task_sessions.stop_task(ctx)
    assert not _live(marker)
    assert not cold_blockers(ctx)
