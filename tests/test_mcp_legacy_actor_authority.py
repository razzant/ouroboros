"""Legacy actor authority through registry, manager, session and a real guard socket.

Only the browser transport is synthetic: it records dispatches and asks the host
callback, without starting a browser or sending any of the described requests.
"""

import asyncio
import hashlib
import ipaddress
import json
import sys
import tempfile
from contextlib import nullcontext
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

from ouroboros import artifacts, browser_policy, mcp_client, mcp_task_sessions
from ouroboros.consciousness_wake import wake_task_metadata
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.contracts.task_contract import attach_task_contract
from ouroboros.tools.registry import ToolContext, ToolRegistry


pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Legacy session is POSIX-only")
ACTORS = ("observe", "observe_uncapped", "act", "capped", "acting_capped", "readonly", "metadata_readonly",
          "owner", "acting", "full")
UNRESTRICTED = {"owner", "acting", "full"}
METADATA = "http://169.254.169.254/latest/meta-data/"
CONTROL = "http://localhost:8765/api/owner/context-mode"
PUBLIC = "https://8.8.8.8/"
TOOLS = ("browser_tabs", "browser_snapshot", "browser_evaluate", "browser_run_code_unsafe",
         "browser_route", "browser_navigate", "browser_page_request")


def _context(root, actor):
    metadata = wake_task_metadata(actor, "heartbeat") if actor in {"observe", "act", "full"} else {}
    if actor in {"capped", "acting_capped"}:
        metadata["runtime_mode_cap"] = "light"
    if actor == "observe_uncapped":
        metadata = {**wake_task_metadata("observe", "heartbeat"), "runtime_mode_cap": ""}
    if actor == "metadata_readonly":
        metadata["delegation_role"] = "subagent"
    task = attach_task_contract({"id": "legacy-actor", "type": "task", "metadata": metadata})
    profile = ("local_readonly_subagent" if actor == "readonly" else
               "acting_subagent" if actor.startswith("acting") else None)
    ctx = ToolContext(repo_dir=root, drive_root=root, task_id=task["id"],
                      task_lifecycle_bound=True, task_metadata=task["metadata"],
                      task_contract=task["task_contract"], messages=[],
                      task_constraint=TaskConstraint(mode=profile, surface="external_workspace",
                                                     write_root=str(root)) if profile else None)
    ctx.task_attempt = 1
    return ctx


@pytest.fixture
def consumer(tmp_path, monkeypatch):
    from mcp.types import CallToolResult, TextContent
    from ouroboros import owner_pause

    mcp_client.reset_manager_for_tests()
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "cyber_pro")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr(browser_policy, "_resolved_addresses", lambda host: {
        ipaddress.ip_address("127.0.0.1" if host == "localhost" else host)})
    monkeypatch.setattr(browser_policy, "runtime_service_kind", lambda url, *_a:
                        "main" if urlparse(url).port == 8765 else "")
    monkeypatch.setattr(owner_pause, "tool_handoff", lambda *_a, **_k: nullcontext({}))
    monkeypatch.setattr(owner_pause, "operation_start", lambda: nullcontext())
    monkeypatch.setattr(owner_pause, "submit_async_preparation", lambda factory, **_k: factory())
    # No Unix transport is launched; keep the constructor's scratch in this test.
    monkeypatch.setattr(mcp_task_sessions, "_private_socket_dir",
                        lambda: tempfile.TemporaryDirectory(dir=tmp_path))
    returned, sent, callbacks, sessions = [], [], [], []
    monkeypatch.setattr(owner_pause, "record_mcp_call_returned", returned.append)
    wire = SimpleNamespace(url=PUBLIC, before_request=None, after_return=None, last_response=None)

    async def serve(session):
        sessions.append(session)
        server = await asyncio.start_server(session._answer_guard, "127.0.0.1", 0)

        async def ask(facts):
            reader, writer = await asyncio.open_connection(*server.sockets[0].getsockname())
            try:
                writer.write(json.dumps(facts).encode() + b"\n")
                await writer.drain()
                return json.loads(await asyncio.wait_for(reader.readline(), 2))
            finally:
                writer.close()
                await writer.wait_closed()

        class Transport:
            async def call_tool(self, name, args):
                if name == "browser_tabs":
                    assert await ask({"armed": True}) == {"armed": True}
                    text = f"### Result\n- 0: (current) [Page]({wire.url})"
                else:
                    sent.append((name, dict(args)))
                    if wire.before_request:
                        wire.before_request(session.ctx)
                    text = "exact returned response\nλ"
                    if name == "browser_page_request":
                        answer = await ask(args)
                        callbacks.append(answer)
                        text = "aborted" if answer["block"] else "sent"
                    if wire.after_return:
                        wire.after_return(session.ctx)
                raw = CallToolResult(content=[TextContent(type="text", text=text)],
                                     structuredContent={"literal": [1, "λ"]}, isError=False)
                if name != "browser_tabs":
                    wire.last_response = raw
                return raw

        async with server:
            session._session = Transport()
            session.ready.set_result([{"name": name, "description": "harmless fixture",
                                       "input_schema": {"type": "object"}} for name in TOOLS])
            await session._closing.wait()

    monkeypatch.setattr(mcp_task_sessions.TaskBrowserSession, "_serve", serve)
    mcp_client.reconfigure_from_settings({"MCP_ENABLED": True, "MCP_SERVERS": [{
        "id": "legacy", "enabled": True, "transport": "stdio", "command": "unused",
        "browser_bridge": True, "browser_protocol": "playwright"}]})
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)

    def bind(actor):
        ctx = _context(tmp_path, actor)
        registry.set_context(ctx)
        names = {item["function"]["name"] for item in registry.schemas()}
        assert "mcp_legacy__browser_run_code_unsafe" in names
        assert len(sessions) == 1
        return ctx

    def call(name, args=None):
        return registry.execute_result("mcp_legacy__" + name, args or {})

    yield SimpleNamespace(bind=bind, call=call, registry=registry, wire=wire, sent=sent,
                          callbacks=callbacks, returned=returned, sessions=sessions, root=tmp_path)
    for session in sessions:
        assert mcp_task_sessions.stop_task(session.ctx, end_attempt=False)[0]["closure"] == "confirmed"
        assert not session.thread.is_alive()
    mcp_client.reset_manager_for_tests()


@pytest.mark.parametrize("actor", ACTORS)
@pytest.mark.parametrize(("name", "args"), (
    ("browser_run_code_unsafe", {"code": "return 42"}),
    ("browser_route", {"pattern": "**/*"}),
))
def test_legacy_registry_refuses_guard_bypass_for_effective_actor(consumer, actor, name, args):
    consumer.bind(actor)
    result = consumer.call(name, args)
    allowed = actor in UNRESTRICTED
    assert result.status == ("ok" if allowed else "blocked"), result.text
    assert consumer.sent == ([(name, args)] if allowed else [])
    assert consumer.returned == (["mcp_legacy__" + name] if allowed else [])


@pytest.mark.parametrize("actor", ACTORS)
def test_legacy_cached_listing_keeps_legitimate_reads(consumer, actor):
    consumer.bind(actor)
    before = consumer.registry.schemas()
    assert consumer.registry.schemas() == before
    assert len(consumer.sessions) == 1
    for _ in range(2):
        result = consumer.call("browser_snapshot")
        assert result.status == "ok", result.text
        assert "exact returned response\nλ" in result.text
    result = consumer.call("browser_tabs", {"action": "list"})
    assert result.status == "ok", result.text
    assert consumer.sent == [("browser_snapshot", {})] * 2
    assert consumer.returned == ["mcp_legacy__browser_snapshot"] * 2 + ["mcp_legacy__browser_tabs"]


@pytest.mark.parametrize("actor", ACTORS)
def test_legacy_guarded_reads_and_fulfilled_routes_keep_existing_authority(consumer, actor):
    consumer.bind(actor)
    result = consumer.call("browser_evaluate", {"function": "() => document.title"})
    readonly = actor in {"readonly", "metadata_readonly"}
    assert result.status == ("blocked" if readonly else "ok"), result.text
    assert len(consumer.sent) == len(consumer.returned) == int(not readonly)
    result = consumer.call("browser_route", {"pattern": "**/*", "body": "local response"})
    assert result.status == "ok", result.text
    assert len(consumer.sent) == len(consumer.returned) == 1 + int(not readonly)


@pytest.mark.parametrize("actor", ACTORS)
@pytest.mark.parametrize("name", ("browser_snapshot", "browser_navigate"))
def test_legacy_target_metadata_authority(consumer, actor, name):
    consumer.bind(actor)
    args = {"url": METADATA} if name == "browser_navigate" else {}
    if name == "browser_snapshot":
        consumer.wire.url = METADATA
    result = consumer.call(name, args)
    allowed = actor in UNRESTRICTED
    assert result.status == ("ok" if allowed else "blocked"), result.text
    assert consumer.sent == ([(name, args)] if allowed else [])
    assert len(consumer.returned) == int(allowed)


@pytest.mark.parametrize("actor", ACTORS)
@pytest.mark.parametrize(("url", "method", "body"), (
    (METADATA, "GET", None), (CONTROL, "POST", '{"context_mode":"max"}'),
    (PUBLIC, "GET", None),
))
def test_legacy_callback_uses_effective_actor(consumer, actor, url, method, body):
    consumer.bind(actor)
    facts = {"url": url, "method": method, "post_data": body}
    result = consumer.call("browser_page_request", facts)
    blocked = actor not in UNRESTRICTED and url != PUBLIC
    assert result.status == "ok", result.text  # The MCP server returned, even on a denied request.
    assert bool(consumer.callbacks[-1]["block"]) == blocked, result.text
    assert ("BROWSER_REQUEST_BLOCKED" in result.text) == blocked
    assert consumer.wire.last_response.content[0].text == ("aborted" if blocked else "sent")
    assert consumer.sent == [("browser_page_request", facts)]
    assert consumer.returned == ["mcp_legacy__browser_page_request"]


@pytest.mark.parametrize("actor", ("observe_uncapped", "acting_capped", "readonly"))
def test_legacy_private_origin_grant_still_binds_and_admits(consumer, actor):
    ctx = consumer.bind(actor)
    facts = {"url": "http://10.2.3.4/path", "method": "GET", "post_data": None}
    consumer.call("browser_page_request", facts)
    assert "origin_not_granted" in consumer.callbacks[-1]["block"]
    ctx.task_contract["resource_policy"] = {"allowed_origins": ["http://10.2.3.4"]}
    result = consumer.call("browser_page_request", facts)
    assert result.status == "ok", result.text
    assert consumer.callbacks[-1]["block"] == ""


@pytest.mark.parametrize("actor", ("observe", "observe_uncapped", "acting_capped"))
def test_legacy_callback_rechecks_authority_after_dispatch(consumer, actor):
    consumer.bind("owner")
    narrowed = _context(consumer.root, actor)

    def narrow(ctx):
        ctx.task_metadata = narrowed.task_metadata
        ctx.task_constraint = narrowed.task_constraint

    consumer.wire.before_request = narrow
    result = consumer.call("browser_page_request", {"url": METADATA, "method": "GET", "post_data": None})
    assert result.status == "ok", result.text
    assert consumer.callbacks[-1]["block"]
    assert "BROWSER_REQUEST_BLOCKED" in result.text
    assert len(consumer.sent) == len(consumer.returned) == 1


def test_legacy_rechecks_cap_after_safety_before_unsafe_dispatch(consumer, monkeypatch):
    ctx = consumer.bind("acting")

    def safety(*_a, **_k):
        ctx.task_metadata["runtime_mode_cap"] = "light"
        return True, ""

    monkeypatch.setattr("ouroboros.safety.check_safety", safety)
    result = consumer.call("browser_run_code_unsafe", {"code": "return 42"})
    assert result.status == "blocked", result.text
    assert "BROWSER_REQUEST_GUARD_BYPASS" in result.text
    assert consumer.sent == consumer.returned == []


def test_legacy_post_return_cap_retains_exact_source_once_without_replay(consumer):
    ctx = consumer.bind("owner")
    consumer.wire.url = METADATA
    consumer.wire.after_return = lambda ctx: ctx.task_metadata.update(runtime_mode_cap="light")
    result = consumer.call("browser_snapshot")
    assert (result.status, result.code) == ("error", "MCP_ERROR"), result.text
    assert "BROWSER_POLICY_POST_DISPATCH" in result.text
    source = result.meta["mcp_response_source"]
    data = artifacts.read_actor_source_bytes(consumer.sessions[0].data_root, ctx.task_id, source)
    assert data == mcp_task_sessions._source_bytes_from_call_result(consumer.wire.last_response)
    assert len(data) == source["size"]
    assert hashlib.sha256(data).hexdigest() == source["sha256"]
    assert consumer.sent == [("browser_snapshot", {})]
    assert consumer.returned == ["mcp_legacy__browser_snapshot"]
