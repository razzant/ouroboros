"""A nested MCP runner thread holds its invocation until its own loop has shut down.

When the synchronous MCP call is made from a thread that already runs an event loop,
``mcp_client._run_async`` starts a copied-context runner thread. The caller's finite
join only words its answer; the runner, holding the invocation through
``asyncio.run`` cleanup, records the local end of the call and, only on a final
response or a proven no-send, discharges the exact remote duty. Driven through the
real MCP SDK session over in-memory streams, the process ``MCPManager`` and
``ToolRegistry`` on both owners of an invocation (the direct registry fallback and
the loop's tool future), then the existing consumers of a retained claim.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from ouroboros import custody_owed, mcp_client, store, task_store, tool_custody
from tests.test_batch4_producer_custody import _consumers, _registry
from tests.test_invocation_ownership import claims, owner, run
from tests.test_mcp_call_completion import SERVER, _transport

pytestmark = pytest.mark.serial

ROUTES = ["direct", "future"]
ACTUAL_JOIN = threading.Thread.join


@pytest.fixture(autouse=True)
def _isolated_manager(tmp_path, monkeypatch):
    from ouroboros import config

    assert mcp_client._MCP_SDK_AVAILABLE, "the real MCP SDK session is the subject under test"
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    mcp_client.reset_manager_for_tests()
    yield
    mcp_client.reset_manager_for_tests()


class _Calls(list):
    """The server's received tools/call names; ``sent`` is set by the first one."""

    def __init__(self):
        super().__init__()
        self.sent = threading.Event()

    def append(self, item):
        super().append(item)
        self.sent.set()


def _server(calls, gate, held=None):
    """A real low-level MCP server whose tools/call answers once ``gate`` opens.

    ``held`` leaves one default-executor job behind the answer: ``asyncio.run``
    must join it while shutting its loop down, after the coroutine has returned.
    """
    import anyio
    import mcp.types as types
    from mcp.server.lowlevel import Server

    server = Server("runner-lifetime")

    @server.list_tools()
    async def tools():
        return [types.Tool(name="effect", description="", inputSchema={"type": "object", "properties": {}})]

    @server.call_tool()
    async def call(name, _arguments):
        calls.append(name)  # the request reached the server: everything now follows a send
        while not gate.is_set():
            await anyio.sleep(0.01)
        if held is not None:
            asyncio.get_running_loop().run_in_executor(None, held.wait)
        return [types.TextContent(type="text", text="remote answer")]

    return server


def _wire(monkeypatch, server, *, drop_call_response=False):
    from ouroboros import config

    monkeypatch.setattr(mcp_client, "_transport_factory", _transport(server, drop_call_response=drop_call_response))
    mcp_client.reset_manager_for_tests()
    manager = mcp_client.get_manager()
    settings = {"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": 30, "MCP_SERVERS": [SERVER]}
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    manager.reconfigure(settings)
    assert manager.refresh_server("demo")["ok"]
    # The synchronous MCP call runs on a thread that already runs an event loop, so
    # the real ``_run_async`` starts its nested copied-context runner thread.
    real = mcp_client._call_mcp_tool_result

    def from_running_loop(name, arguments):
        async def inside():
            return real(name, arguments)
        return asyncio.run(inside())

    monkeypatch.setattr(mcp_client, "_call_mcp_tool_result", from_running_loop)
    return manager


def _caller_stops_waiting(monkeypatch, ready):
    """The caller's finite join ends once ``ready`` is set; the real runner keeps running."""
    runners = []

    def join(thread, timeout=None):
        if thread.name != "mcp-sync-runner":
            return ACTUAL_JOIN(thread, timeout)
        runners.append(thread)
        assert ready.wait(10)
        return ACTUAL_JOIN(thread, 0)

    monkeypatch.setattr(threading.Thread, "join", join)
    return runners


def _call(registry, route):
    if route == "direct":
        return registry.execute_result("mcp_demo__effect", {})
    with owner(registry):
        return run(registry, name="mcp_demo__effect", args={}, timeout=30)["tool_result"]


def _finish(runners):
    for thread in runners:
        ACTUAL_JOIN(thread, 10)
        assert not thread.is_alive()


def _only_claim(root):
    rows = claims(root)
    assert len(rows) == 1, rows
    return next(iter(rows.values()))


def _assert_unknown_timeout(result):
    assert (result.status, result.code) == ("timeout", "MCP_TIMEOUT"), result
    assert result.meta["mcp_call_not_sent"] is False
    assert "The remote outcome is unknown" in result.text and "reconcile the operation" in result.text
    assert "server-supplied" not in result.text, "the host's own wait is not server output"


def _assert_runner_holds(tmp_path, registry, route):
    """The runner thread lives: its claim is a live local call for every reader."""
    from ouroboros.model_sleep import cold_blockers

    claim = _only_claim(tmp_path)
    assert claim["state"] == "claimed" and list(claim["external_pending"].values()) == ["mcp_transport"]
    assert tool_custody.tool_handoff_reason(claim)["reason"] == "local_call_live"
    assert cold_blockers(registry._ctx)
    if route == "future":
        assert not tool_custody.local_producers_settled(), "a live runner is a live producer"


@pytest.mark.parametrize("route", ROUTES)
def test_late_final_response_settles_after_the_callers_timeout(tmp_path, monkeypatch, route):
    registry, queue, workers = _registry(tmp_path, monkeypatch)
    calls, gate = _Calls(), threading.Event()
    _wire(monkeypatch, _server(calls, gate))
    runners = _caller_stops_waiting(monkeypatch, calls.sent)
    try:
        _assert_unknown_timeout(_call(registry, route))
        _assert_runner_holds(tmp_path, registry, route)
    finally:
        gate.set()
        _finish(runners)
    assert calls == ["effect"], "one remote call, never resent"
    assert not claims(tmp_path), "the final response ended both the local call and its remote duty"
    assert tool_custody.local_producers_settled()
    _consumers(tmp_path, registry, queue, workers, held=False)


@pytest.mark.parametrize("route", ROUTES)
def test_late_error_after_send_ends_the_local_call_and_keeps_the_remote_duty(tmp_path, monkeypatch, route):
    registry, queue, workers = _registry(tmp_path, monkeypatch)
    calls, gate = _Calls(), threading.Event()
    _wire(monkeypatch, _server(calls, gate), drop_call_response=True)  # the answer never arrives
    runners = _caller_stops_waiting(monkeypatch, calls.sent)
    try:
        _assert_unknown_timeout(_call(registry, route))
        _assert_runner_holds(tmp_path, registry, route)
    finally:
        gate.set()
        _finish(runners)
    assert calls == ["effect"], "one remote call, never resent"
    claim = _only_claim(tmp_path)
    assert claim["state"] == "returned" and list(claim["external_pending"].values()) == ["mcp_transport"]
    assert tool_custody.tool_handoff_reason(claim)["reason"] == "returned_effect_unconfirmed"
    assert tool_custody.local_producers_settled()
    _consumers(tmp_path, registry, queue, workers, held=True)


@pytest.mark.parametrize("route", ROUTES)
def test_late_runner_refused_before_send_discharges_its_exact_duty(tmp_path, monkeypatch, route):
    """Scheduling delay before the coroutine: the closed invocation refuses preparation."""
    registry, queue, workers = _registry(tmp_path, monkeypatch)
    calls, gate = _Calls(), threading.Event()
    _wire(monkeypatch, _server(calls, gate))
    entered, resume = threading.Event(), threading.Event()
    actual_run = asyncio.run

    def delayed(coro, *args, **kwargs):
        if threading.current_thread().name == "mcp-sync-runner":
            entered.set()
            assert resume.wait(10)
        return actual_run(coro, *args, **kwargs)

    monkeypatch.setattr(asyncio, "run", delayed)
    runners = _caller_stops_waiting(monkeypatch, entered)
    try:
        _assert_unknown_timeout(_call(registry, route))
        _assert_runner_holds(tmp_path, registry, route)
    finally:
        gate.set()
        resume.set()
        _finish(runners)
    assert calls == [], "the closed invocation refused the method before any send"
    assert not claims(tmp_path)
    assert tool_custody.local_producers_settled()
    _consumers(tmp_path, registry, queue, workers, held=False)


@pytest.mark.parametrize("route", ROUTES)
def test_held_loop_shutdown_keeps_the_runner_live_after_its_final_response(tmp_path, monkeypatch, route):
    """The coroutine has its final response, but ``asyncio.run`` still joins an executor job."""
    registry, queue, workers = _registry(tmp_path, monkeypatch)
    calls, gate, held, responded = _Calls(), threading.Event(), threading.Event(), threading.Event()
    gate.set()
    manager = _wire(monkeypatch, _server(calls, gate, held))
    transport_call = manager._async_call_tool

    async def observed(*args):
        result = await transport_call(*args)
        responded.set()
        return result

    manager._async_call_tool = observed
    runners = _caller_stops_waiting(monkeypatch, responded)
    try:
        _assert_unknown_timeout(_call(registry, route))
        assert all(thread.is_alive() for thread in runners), "loop shutdown still owns the runner"
        _assert_runner_holds(tmp_path, registry, route)
    finally:
        held.set()
        _finish(runners)
    assert calls == ["effect"]
    assert not claims(tmp_path)
    assert tool_custody.local_producers_settled()
    _consumers(tmp_path, registry, queue, workers, held=False)


@pytest.mark.parametrize("route", ROUTES)
def test_runner_that_never_started_is_a_proven_no_send(tmp_path, monkeypatch, route):
    registry, queue, workers = _registry(tmp_path, monkeypatch)
    calls, gate, opened = _Calls(), threading.Event(), []
    gate.set()
    _wire(monkeypatch, _server(calls, gate))
    real_transport = mcp_client._transport_factory
    monkeypatch.setattr(mcp_client, "_transport_factory", lambda cfg: opened.append(cfg) or real_transport(cfg))
    actual_start = threading.Thread.start

    def start(thread):
        if thread.name == "mcp-sync-runner":
            raise RuntimeError("can't start new thread")
        return actual_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    result = _call(registry, route)
    monkeypatch.setattr(threading.Thread, "start", actual_start)
    assert (result.status, result.code) == ("error", "MCP_ERROR"), result
    assert result.meta["mcp_call_not_sent"] is True
    assert calls == [] and opened == []
    assert not claims(tmp_path), "no start: the duty is discharged and the hold released"
    assert tool_custody.local_producers_settled()
    _consumers(tmp_path, registry, queue, workers, held=False)


@pytest.mark.parametrize("route", ROUTES)
def test_late_settlement_on_a_busy_store_is_owed_never_replayed(tmp_path, monkeypatch, route):
    registry, _queue, _workers = _registry(tmp_path, monkeypatch)
    calls, gate = _Calls(), threading.Event()
    _wire(monkeypatch, _server(calls, gate))
    runners = _caller_stops_waiting(monkeypatch, calls.sent)
    monkeypatch.setattr(store, "STORE_WAIT_SEC", 0.01)  # the structural retry bound, shortened only here
    patch = task_store.patch_handoff
    refused = []

    def busy_for_the_runner(*args, **kwargs):
        if threading.current_thread().name == "mcp-sync-runner":
            refused.append(args[3])
            raise store.StoreBusy("late custody write contended", reason="contention")
        return patch(*args, **kwargs)

    try:
        _assert_unknown_timeout(_call(registry, route))
        monkeypatch.setattr(task_store, "patch_handoff", busy_for_the_runner)
    finally:
        gate.set()
        _finish(runners)
    assert refused, "the runner's own writes met the busy store"
    assert custody_owed.pending(), "the late facts are owed, not forgotten"
    assert not tool_custody.local_producers_settled()
    assert _only_claim(tmp_path)["state"] == "claimed"
    monkeypatch.setattr(task_store, "patch_handoff", patch)
    assert custody_owed.drain() == 0
    assert not claims(tmp_path) and calls == ["effect"], "the owed writes landed; nothing was resent"
    assert tool_custody.local_producers_settled()


@pytest.mark.parametrize("route", ROUTES)
def test_late_completion_during_the_registry_fold_still_settles(tmp_path, monkeypatch, route):
    """The runner records its end between the registry's fold read and the closure."""
    registry, queue, workers = _registry(tmp_path, monkeypatch)
    calls, gate = _Calls(), threading.Event()
    _wire(monkeypatch, _server(calls, gate))
    runners = _caller_stops_waiting(monkeypatch, calls.sent)
    fired = []
    actual_outcome = tool_custody.invocation_outcome

    class Interleaving(dict):
        def get(self, key, default=None):
            value = dict.get(self, key, default)  # read before the runner's write
            # After the caller stopped waiting, the first reader outside the runner is the
            # registry's fold, on whichever thread executes the invocation.
            if (key == "mcp_call_returned" and not fired and runners
                    and threading.current_thread().name != "mcp-sync-runner"):
                fired.append(value)
                gate.set()
                _finish(runners)  # the runner records, confirms and releases in between
            return value

    def outcome(*args, **kwargs):
        return Interleaving(actual_outcome(*args, **kwargs))

    monkeypatch.setattr(tool_custody, "invocation_outcome", outcome)
    try:
        _assert_unknown_timeout(_call(registry, route))
    finally:
        gate.set()
        _finish(runners)
    assert fired == [None], "the fold read the flag before the runner wrote it"
    assert calls == ["effect"]
    assert not claims(tmp_path)
    assert tool_custody.local_producers_settled()
    _consumers(tmp_path, registry, queue, workers, held=False)


def test_missing_sdk_before_any_transport_is_a_proven_no_send(tmp_path, monkeypatch):
    registry, queue, workers = _registry(tmp_path, monkeypatch)
    calls, gate, opened = _Calls(), threading.Event(), []
    _wire(monkeypatch, _server(calls, gate))
    monkeypatch.setattr(mcp_client, "_transport_factory", lambda cfg: opened.append(cfg))
    monkeypatch.setattr(mcp_client, "_MCP_SDK_AVAILABLE", False)
    result = registry.execute_result("mcp_demo__effect", {})
    assert (result.code, result.meta["mcp_call_not_sent"]) == ("MCP_ERROR", True), result
    assert "MCP client SDK not installed" in result.text
    assert calls == [] and opened == []
    assert not claims(tmp_path)
    _consumers(tmp_path, registry, queue, workers, held=False)
