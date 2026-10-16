"""Task authority on the actual prefixed registry and Chrome callback paths."""

import asyncio
import ipaddress
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ouroboros import browser_policy, mcp_client, mcp_task_sessions
from ouroboros.consciousness_wake import wake_task_metadata
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.contracts.task_contract import attach_task_contract
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests.test_mcp_chrome_session import _session, _state

pytestmark = pytest.mark.serial


def _context(root, actor):
    metadata = wake_task_metadata(actor, "heartbeat") if actor in {"observe", "act", "full"} else {}
    if actor in {"capped", "acting_capped"}:
        metadata["runtime_mode_cap"] = "light"
    if actor == "observe_uncapped":
        metadata = {**wake_task_metadata("observe", "heartbeat"), "runtime_mode_cap": ""}
    task = attach_task_contract({"id": "task-one", "type": "task", "metadata": metadata})
    profile = ("local_readonly_subagent" if actor == "readonly" else
               "acting_subagent" if actor.startswith("acting") else None)
    ctx = ToolContext(repo_dir=root, drive_root=root, task_id="task-one",
                      task_lifecycle_bound=True, task_metadata=task["metadata"],
                      task_contract=task["task_contract"], messages=[],
                      task_constraint=TaskConstraint(mode=profile, surface="external_workspace", write_root=str(root)) if profile else None)
    ctx.task_attempt = 1
    return ctx


@pytest.fixture
def consumer(tmp_path, monkeypatch):
    mcp_client.reset_manager_for_tests()
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "cyber_pro")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr(browser_policy, "_resolved_addresses", lambda host: {
        ipaddress.ip_address("169.254.169.254" if host == "169.254.169.254" else "8.8.8.8")})
    monkeypatch.setattr(browser_policy, "runtime_service_kind", lambda *_a: "")

    @contextmanager
    def handoff(*_args):
        yield {}

    monkeypatch.setattr("ouroboros.owner_pause.tool_handoff", handoff)
    session = _session()
    session.data_root = tmp_path
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    write_task_result(tmp_path, "task-one", STATUS_RUNNING)
    session._observe_before = lambda *_a, **_k: session._chrome_state
    session._observe = session._observe_before
    sent = []

    def request(name, args, *_a, **_k):
        sent.append((name, args))
        return SimpleNamespace(isError=False, content=[SimpleNamespace(type="text", text="fixture response")])

    session._request = request

    def stop():
        session.revoked = True
        return "confirmed"

    session.stop = stop

    async def listing(cfg, ctx, timeout, session_out):
        session.cfg, session.ctx = cfg, ctx
        mcp_task_sessions._sessions[session.key] = session
        session_out.append(session)
        return [{"name": name, "description": "fixture", "input_schema": {"type": "object"}}
                for name in ("browser_state", "browser_action", "browser_execute", "browser_connect")]

    monkeypatch.setattr(mcp_task_sessions, "listing", listing)
    mcp_client.reconfigure_from_settings({"MCP_ENABLED": True, "MCP_SERVERS": [{
        "id": "browser", "enabled": True, "transport": "stdio", "command": "node",
        "browser_bridge": True, "browser_protocol": "ouroboros_chrome"}]})
    monkeypatch.setitem(mcp_task_sessions._sessions, session.key, session)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    yield registry, session, sent, tmp_path
    mcp_task_sessions._sessions.pop(session.key, None)
    mcp_client.reset_manager_for_tests()


ACTORS = ("observe", "act", "capped", "acting_capped", "readonly", "observe_uncapped", "owner", "acting", "full")
UNRESTRICTED = {"owner", "acting", "full"}


@pytest.mark.parametrize("actor", ACTORS)
@pytest.mark.parametrize(("name", "args"), (
        ("browser_action", {"operation": "click", "args": {}}),
        ("browser_action", {"operation": "navigate", "args": {"url": "https://example.test/next"}}),
        ("browser_action", {"operation": "sequence", "args": {"steps": [
            {"operation": "snapshot", "args": {}}, {"operation": "fill", "args": {}}]}}),
        ("browser_execute", {"code": "await browser.click('#go')"}),
        ("browser_connect", {"url": "https://example.test/new"}),
))
def test_prefixed_registry_mutation_authority(consumer, actor, name, args):
    registry, session, sent, root = consumer
    registry.set_context(_context(root, actor))
    assert "mcp_browser__browser_action" in {item["function"]["name"] for item in registry.schemas()}
    if name == "browser_connect":
        session._chrome_state.update(connected=False, lease=None)
        session._chrome_lease_generation_floor = -1  # a cold executor has never held a lease
    result = registry.execute_result("mcp_browser__" + name, args)
    assert result.status == ("ok" if actor in UNRESTRICTED else "blocked"), result.text
    assert bool(sent) == (actor in UNRESTRICTED)


@pytest.mark.parametrize("actor", ACTORS)
def test_prefixed_registry_keeps_snapshot_and_cached_reads(consumer, actor):
    registry, _session, sent, root = consumer
    registry.set_context(_context(root, actor))
    registry.schemas()
    for operation, args in (("snapshot", {}), ("cached_text", {"snapshot_id": "saved"}),
                            ("sequence", {"steps": [{"operation": "cached_text", "args": {}}]})):
        result = registry.execute_result("mcp_browser__browser_action", {"operation": operation, "args": args})
        assert result.status == "ok", result.text
    assert len(sent) == 3


async def _ask(session, facts):
    """Exercise the actual socket callback, not just a policy helper."""
    server = await asyncio.start_server(session._answer_guard, "127.0.0.1", 0)
    try:
        reader, writer = await asyncio.open_connection(*server.sockets[0].getsockname())
        writer.write(json.dumps(facts).encode() + b"\n")
        await writer.drain()
        answer = json.loads(await asyncio.wait_for(reader.readline(), 2))
        writer.close()
        await writer.wait_closed()
        return answer["block"]
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.parametrize("actor", ACTORS)
@pytest.mark.parametrize(("name", "args", "operation"), (
        ("browser_action", {"operation": "click", "args": {}}, "click"),
        ("browser_execute", {"code": "await browser.snapshot()"}, "snapshot"),
        ("browser_connect", {"url": "https://example.test/new"}, "connect"),
        ("browser_action", {"operation": "cached_text", "args": {}}, "cached_text"),
))
def test_callbacks_recheck_actor_after_outer_admission(consumer, actor, name, args, operation):
    _registry, session, _sent, root = consumer
    session.ctx = _context(root, actor)
    session._chrome_invocation = ("admitted-before-cap", name, args)
    facts = {"kind": "primitive", "invocation_id": "admitted-before-cap", "step_id": 1,
             "operation": operation, "args": args if operation == "connect" else {}, "state": _state()}
    reason = asyncio.run(_ask(session, facts))
    assert bool(reason) == (actor not in UNRESTRICTED and operation != "cached_text"), reason


def test_task_cap_changed_during_safety_refuses_before_dispatch(consumer, monkeypatch):
    registry, _session, sent, root = consumer
    ctx = _context(root, "owner")
    registry.set_context(ctx)
    registry.schemas()

    def safety(*_a, **_k):
        ctx.task_metadata["runtime_mode_cap"] = "light"
        return True, ""

    monkeypatch.setattr("ouroboros.safety.check_safety", safety)
    result = registry.execute_result("mcp_browser__browser_action", {"operation": "click", "args": {}})
    assert result.status == "blocked" and "GUARD_UNAVAILABLE" in result.text
    assert not sent


def test_actor_policy_never_walks_or_copies_task_metadata(consumer):
    from ouroboros import mcp_chrome_protocol as chrome

    class Metadata(dict):
        def __iter__(self):
            raise AssertionError("policy must use constant-time metadata fields")

        keys = items = values = copy = __iter__

    _registry, _session, _sent, root = consumer
    ctx = _context(root, "observe")
    ctx.task_metadata = Metadata(ctx.task_metadata)
    ctx.task_metadata["history"] = object()
    assert chrome.effect_reason(ctx)
    assert chrome.action_reason(ctx, "browser_execute", {"code": "42"}, _state())
    assert chrome.target_reason(ctx, "https://example.test/") == ""
    assert chrome.request_reason(ctx, {"request_id": "r", "url": "https://example.test/",
                                       "method": "GET", "post_data": None}) == ""


@pytest.mark.parametrize("actor", ("observe", "capped", "acting_capped", "owner", "acting", "full"))
@pytest.mark.parametrize("surface", ("target", "request"))
def test_effective_target_and_request_caps(consumer, monkeypatch, actor, surface):
    registry, session, sent, root = consumer
    ctx = _context(root, actor)
    registry.set_context(ctx)
    registry.schemas()
    if surface == "target":
        session._chrome_state["lease"]["url"] = "http://169.254.169.254/latest"
        result = registry.execute_result("mcp_browser__browser_action", {"operation": "snapshot", "args": {}})
        assert result.status == ("ok" if actor in UNRESTRICTED else "blocked"), result.text
        assert bool(sent) == (actor in UNRESTRICTED)
        return
    session.ctx = ctx
    session._chrome_invocation = ("req-inv", "browser_action", {"operation": "snapshot", "args": {}})
    monkeypatch.setattr(browser_policy, "runtime_service_kind", lambda *_a: "main")
    facts = {"kind": "request", "invocation_id": "req-inv", "step_id": 1, "state": _state(),
             "request_id": "req-1", "url": "http://localhost:8765/api/owner/context-mode", "method": "POST",
             "post_data": "{}", "complete": True}
    reason = asyncio.run(_ask(session, facts))
    assert bool(reason) == (actor not in UNRESTRICTED), reason
