"""Chrome guard and Stop run without a browser, on the task-owned session path."""

import asyncio
import collections
import json
import sys
import threading
from types import SimpleNamespace

import pytest

from ouroboros import mcp_chrome_protocol as chrome
from ouroboros import mcp_task_sessions
from ouroboros.tools.tool_result import ToolResult

pytestmark = pytest.mark.serial


@pytest.fixture(autouse=True)
def cyber_consumer(monkeypatch, tmp_path):
    # These primitive fixtures represent the qualified unrestricted caller.
    # Ordinary/readonly refusal is exercised separately in protocol unit tests.
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "cyber_pro")
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from tests.test_mcp_chrome_authority import _context

    factory = _session

    def admitted_session():
        session = factory()
        session.ctx, session.data_root = _context(tmp_path, "owner"), tmp_path
        write_task_result(tmp_path, "task-one", STATUS_RUNNING)
        monkeypatch.setitem(mcp_task_sessions._sessions, session.key, session)
        return session

    monkeypatch.setattr(sys.modules[__name__], "_session", admitted_session)


def test_installed_skill_binding_checks_current_payload_readiness_and_executor(monkeypatch, tmp_path):
    from ouroboros import skill_loader, skill_peer_inventory, skill_readiness
    from ouroboros.tools import skill_exec

    skill_dir = tmp_path / "chrome_browser"
    script = skill_dir / "scripts" / "server.js"
    script.parent.mkdir(parents=True)
    script.write_text("console.log('one')", encoding="utf-8")
    digest = skill_loader.compute_content_hash(skill_dir, manifest_scripts=[{"name": "server.js"}])
    manifest = SimpleNamespace(is_script=lambda: True, scripts=[{"name": "server.js"}], runtime="node")
    skill = SimpleNamespace(name="chrome_browser", skill_dir=skill_dir, manifest=manifest,
                            content_hash=digest, load_error="", identity_collision=False)
    from ouroboros.mcp_client import normalize_server_config
    cfg = normalize_server_config({"id": "chrome", "transport": "stdio", "command": "node",
                                  "args": ["ignored-old-path.js"], "browser_bridge": True,
                                  "browser_protocol": "ouroboros_chrome",
                                  "browser_skill_binding": {"skill_name": "chrome_browser",
                                                            "script_path": "scripts/server.js"}})
    monkeypatch.setattr("ouroboros.tool_access_paths.canonical_data_root", lambda _ctx: tmp_path)
    monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda: "")
    def discover(*_a, **_k):
        # Like the real discovery owner, return freshly hashed payload facts.
        skill.content_hash = skill_loader.compute_content_hash(skill_dir, manifest_scripts=manifest.scripts)
        return [skill]

    monkeypatch.setattr(skill_loader, "discover_skill_identity", discover)
    monkeypatch.setattr(skill_peer_inventory, "discover_skill_peers", lambda *_a, **_k: [])
    monkeypatch.setattr(skill_exec, "_resolve_runtime_binary", lambda _runtime: (sys.executable, ""))
    readiness = SimpleNamespace(ready=True, blockers=[])
    monkeypatch.setattr(skill_readiness, "skill_readiness_for_execution", lambda *_a, **_k: readiness)

    resolved, initial_hash = chrome.resolve_binding(cfg, object())
    assert resolved.args == [str(script)] and resolved.command == sys.executable
    assert initial_hash == digest
    assert mcp_task_sessions._chrome_binding_reason(cfg, object(), initial_hash) == ""
    readiness.ready, readiness.blockers = False, ["skill_disabled"]
    assert "skill_disabled" in mcp_task_sessions._chrome_binding_reason(cfg, object())
    readiness.ready, readiness.blockers = True, []
    script.write_text("console.log('two')", encoding="utf-8")
    assert "payload changed" in mcp_task_sessions._chrome_binding_reason(cfg, object(), initial_hash)
    # A legitimately reviewed update re-admits a new session without a Settings hash edit.
    skill.content_hash = skill_loader.compute_content_hash(skill_dir, manifest_scripts=manifest.scripts)
    updated, updated_hash = chrome.resolve_binding(cfg, object())
    assert updated.args == [str(script)] and updated_hash != initial_hash
    assert chrome.binding_reason(cfg, object()) == ""
    assert "payload changed" in chrome.binding_reason(cfg, object(), initial_hash)
    manifest.scripts = [{"name": "other.js"}]
    assert "not declared" in chrome.binding_reason(cfg, object())


def test_callback_revokes_when_installed_skill_drifts(monkeypatch):
    session = _session()
    session._binding_revoked = False
    session._closing = asyncio.Event()
    monkeypatch.setattr(chrome, "binding_reason", lambda *_a: "BROWSER_SKILL_BINDING_REVOKED: skill disabled")
    facts = {"kind": "primitive", "invocation_id": "inv-1", "step_id": 1,
             "operation": "click", "args": {}, "state": _state()}
    assert "BROWSER_SKILL_BINDING_REVOKED" in asyncio.run(session._chrome_guard_reason(facts))
    assert session.revoked and session._binding_revoked and session._closing.is_set()


def test_skill_binding_blocks_discovery_before_session_or_process(monkeypatch):
    cfg = SimpleNamespace(id="chrome", browser_protocol="ouroboros_chrome")
    ctx = SimpleNamespace(task_id="bound-task", task_attempt=1, task_lifecycle_bound=True)
    monkeypatch.setattr(mcp_task_sessions, "_chrome_binding_reason",
                        lambda *_a: "BROWSER_SKILL_BINDING_REVOKED: review_stale")
    try:
        mcp_task_sessions.discover(cfg, ctx, 1)
    except RuntimeError as exc:
        assert "review_stale" in str(exc)
    else:
        assert False, "discovery must refuse before opening the MCP session"
    assert ("bound-task", "chrome") not in mcp_task_sessions._sessions


def _state():
    return {"protocol": "ouroboros_chrome", "version": 1,
            "owner": {"task_id": "task-one", "attempt": "1"},
            "executor_generation": 1, "connected": True,
            "lease": {"lease_id": "lease-1", "generation": 1, "tab_id": "tab-1",
                      "document_id": "doc-1", "url": "https://example.test/", "guard_armed": True},
            "closure": {"revoked": False, "detached": False}}


def _session():
    session = object.__new__(mcp_task_sessions.TaskBrowserSession)
    session.cfg = SimpleNamespace(browser_protocol="ouroboros_chrome")
    session.key = ("task-one", "browser")
    session.attempt = "1"
    session.ctx = None
    session.revoked = False
    session._call_lock = threading.Lock()
    session._chrome_state = _state()
    session._chrome_invocation = ("inv-1", "browser_action", {"operation": "click", "args": {}})
    session._chrome_last_step = 0
    session._chrome_navigation_pending = False
    session._chrome_guard_lock = asyncio.Lock()
    session._chrome_lease_generation_floor = 1
    session._chrome_requests = set()
    session._chrome_documents = {}
    session._blocked = collections.deque(maxlen=20)
    session._blocked_omitted = 0
    return session


def test_guard_answers_during_outer_call_lock_and_fences_steps(monkeypatch):
    monkeypatch.setattr(chrome, "request_reason", lambda *_a: "")
    session = _session()
    facts = {"kind": "request", "invocation_id": "inv-1", "step_id": 1,
             "request_id": "request-1", "url": "https://example.test/a", "method": "GET",
             "post_data": None, "complete": True, "state": _state()}

    async def exercise():
        server = await asyncio.start_server(session._answer_guard, "127.0.0.1", 0)
        address = server.sockets[0].getsockname()

        async def ask(payload):
            reader, writer = await asyncio.open_connection(*address)
            writer.write(json.dumps(payload).encode() + b"\n")
            await writer.drain()
            answer = json.loads(await asyncio.wait_for(reader.readline(), 1))
            writer.close()
            await writer.wait_closed()
            return answer

        try:
            with session._call_lock:
                assert await ask(facts) == {"block": ""}
                assert "step must increase" in (await ask(facts))["block"]
                session.revoked = True
                assert "REVOKED" in (await ask({**facts, "step_id": 2}))["block"]
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(exercise())


def test_stop_sends_disconnect_while_outer_call_lock_is_held():
    session = _session()
    seen = []

    class Client:
        async def call_tool(self, name, args, *, meta):
            seen.append((name, args, meta))
            return SimpleNamespace(isError=False)

    loop = asyncio.new_event_loop()
    worker = threading.Thread(target=loop.run_forever, daemon=True)
    worker.start()
    session._loop = loop
    session._session = Client()
    session._closing = asyncio.Event()
    session._started = False
    session.close_outcome = "confirmed"
    session.chrome_disconnect_ack = "not_sent"
    session._reap = lambda: "confirmed"
    try:
        with session._call_lock:
            assert session.stop() == "confirmed"
        assert session.chrome_disconnect_ack == "acknowledged"
        assert seen[0][0] == "browser_disconnect"
        assert seen[0][1] == {}
        assert seen[0][2]["ouroboros_invocation_id"]
    finally:
        loop.call_soon_threadsafe(loop.stop)
        worker.join(timeout=1)
        loop.close()


def test_late_callback_cannot_rebind_connect_or_navigation_lease(monkeypatch):
    monkeypatch.setattr(chrome, "target_reason", lambda *_a: "")

    async def exercise():
        session = _session()
        session._chrome_state = {**_state(), "connected": False, "lease": None}
        session._chrome_invocation = ("connect-1", "browser_connect", {"tab_id": "tab-1"})
        session._chrome_lease_generation_floor = -1
        connected = _state()
        callback = {"kind": "primitive", "invocation_id": "other", "step_id": 1,
                    "operation": "connect", "args": {"tab_id": "tab-1"}, "state": connected}
        assert "invocation" in await session._chrome_guard_reason(callback)
        assert session._chrome_state["lease"] is None
        callback["invocation_id"] = "connect-1"
        assert await session._chrome_guard_reason(callback) == ""
        assert session._chrome_state["lease"]["generation"] == 1
        assert session._chrome_lease_generation_floor == 1

        session._chrome_invocation = ("action-1", "browser_action",
                                      {"operation": "navigate", "args": {"url": "https://example.test/next"}})
        session._chrome_last_step = 0
        primitive = {"kind": "primitive", "invocation_id": "action-1", "step_id": 1,
                     "operation": "navigate", "args": session._chrome_invocation[2]["args"],
                     "state": connected}
        assert await session._chrome_guard_reason(primitive) == ""
        next_state = _state()
        next_state["lease"] = {**next_state["lease"], "lease_id": "lease-2",
                               "generation": 2, "document_id": "doc-2",
                               "url": "https://example.test/next"}
        request = {"kind": "request", "invocation_id": "old", "step_id": 2,
                   "request_id": "request-1", "url": "https://example.test/next", "method": "GET",
                   "post_data": None, "complete": True, "state": connected,
                   "resource_type": "Document", "is_main_frame": True, "target_tab_id": "tab-1"}
        assert "invocation" in await session._chrome_guard_reason(request)
        assert session._chrome_state["lease"]["generation"] == 1
        assert session._chrome_navigation_pending
        request["invocation_id"] = "action-1"
        monkeypatch.setattr(chrome, "request_reason", lambda *_a: "")
        assert await session._chrome_guard_reason(request) == ""
        committed = {"kind": "document", "invocation_id": "action-1", "step_id": 3,
                     "request_id": "request-1", "state": next_state}
        assert await session._chrome_guard_reason(committed) == ""
        assert session._chrome_state["lease"]["generation"] == 2
        assert not session._chrome_navigation_pending
        request.update(step_id=4, state=next_state)
        assert "duplicate request" in await session._chrome_guard_reason(request)

    asyncio.run(exercise())


def test_observation_refuses_unwitnessed_lease_or_executor_change():
    session = _session()
    changed = _state()
    changed["lease"] = {**changed["lease"], "lease_id": "another", "document_id": "doc-2"}
    session._request = lambda *_a, **_k: SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(changed))], isError=False)
    try:
        session._observe(1)
    except RuntimeError as exc:
        assert "without a guard callback" in str(exc)
    else:
        raise AssertionError("same generation changed lease")
    changed["executor_generation"] = 2
    try:
        session._observe(1)
    except RuntimeError as exc:
        assert "executor generation changed" in str(exc)
    else:
        raise AssertionError("executor changed within one task session")


def test_connect_control_call_uses_same_mcp_session_and_guard_handshake(monkeypatch):
    session = _session()
    disconnected = {**_state(), "connected": False, "lease": None}
    session._chrome_state = disconnected
    session._chrome_lease_generation_floor = -1
    session.ctx = SimpleNamespace(task_id="task-one", task_attempt=1, task_lifecycle_bound=True,
                                  messages=[])
    session._observe_before = lambda *_a, **_k: disconnected
    session._observe = lambda *_a, **_k: session._chrome_state
    monkeypatch.setattr(chrome, "target_reason", lambda *_a: "")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr("ouroboros.mcp_client._tool_result_from_call_result",
                        lambda *_a: ToolResult(status="ok", code="OK", text="connected"))
    calls = []

    def request(name, args, timeout, *, dispatch=False, invocation_id=""):
        calls.append((name, args, dispatch, invocation_id))
        assert session._chrome_invocation == (invocation_id, "browser_connect", args)
        callback = {"kind": "primitive", "invocation_id": invocation_id, "step_id": 1,
                    "operation": "connect", "args": args, "state": _state()}
        assert asyncio.run(session._chrome_guard_reason(callback)) == ""
        return object()

    session._request = request
    result, advice, note = session._call_chrome(
        "mcp_browser__browser_connect", "browser_connect", {"tab_id": "tab-1"}, session.ctx, 1)
    assert result.status == "ok" and not advice and not note
    assert calls == [("browser_connect", {"tab_id": "tab-1"}, True, calls[0][3])]
    assert calls[0][3]


def test_panic_request_only_signals_marked_process_without_browser_ack(monkeypatch, tmp_path):
    from ouroboros import platform_layer
    from ouroboros.process_containment import MARKER_MEMBER

    signalled = []
    marker = "OURO_PROC_CONTAINER_test"
    monkeypatch.setattr(mcp_task_sessions.ProcessContainer, "for_scope",
                        lambda *_a: SimpleNamespace(containment_env=lambda: {marker: "1"}))
    monkeypatch.setattr(mcp_task_sessions, "pids_with_env_marker", lambda *_a: [12345])
    monkeypatch.setattr(mcp_task_sessions, "pid_marker_state", lambda *_a: MARKER_MEMBER)
    monkeypatch.setattr(platform_layer, "force_kill_pid", signalled.append)
    outcome = mcp_task_sessions.stop_scope(tmp_path, "task-one", request_only=True)
    assert signalled == [12345]
    assert outcome == {"requested_pids": [12345], "closure": "unconfirmed: signal request only"}


def test_click_and_submit_document_changes_require_admitted_main_request(monkeypatch):
    monkeypatch.setattr(chrome, "target_reason", lambda *_a: "")
    monkeypatch.setattr(chrome, "request_reason", lambda *_a: "")

    async def exercise(operation):
        session = _session()
        session._chrome_invocation = ("inv-1", "browser_action", {"operation": operation, "args": {}})
        primitive = {"kind": "primitive", "invocation_id": "inv-1", "step_id": 1,
                     "operation": operation, "args": {}, "state": _state()}
        assert await session._chrome_guard_reason(primitive) == ""
        after = _state()
        after["lease"] = {**after["lease"], "lease_id": "lease-2", "generation": 2,
                          "document_id": "doc-2", "url": "https://example.test/next"}
        document = {"kind": "document", "invocation_id": "inv-1", "step_id": 2,
                    "request_id": "main-request", "state": after}
        assert "unexpected document" in await session._chrome_guard_reason(document)
        request = {"kind": "request", "invocation_id": "inv-1", "step_id": 2,
                   "request_id": "subframe", "url": "https://example.test/next", "method": "GET",
                   "post_data": None, "complete": True, "state": _state(),
                   "resource_type": "Document", "is_main_frame": False, "target_tab_id": "tab-1"}
        assert await session._chrome_guard_reason(request) == ""
        document.update(step_id=3, request_id="subframe")
        assert "unexpected document" in await session._chrome_guard_reason(document)
        request.update(step_id=3, request_id="main-request", is_main_frame=True)
        assert await session._chrome_guard_reason(request) == ""
        document.update(step_id=4, request_id="main-request")
        disconnected = {**after, "connected": False, "lease": None,
                        "closure": {"revoked": False, "detached": False}}
        assert "unexpected document" in await session._chrome_guard_reason(
            {**document, "state": disconnected})
        assert session._chrome_last_step == 3
        wrong = {**after, "lease": {**after["lease"], "tab_id": "another"}}
        assert "document lease" in await session._chrome_guard_reason({**document, "state": wrong})
        assert session._chrome_state["lease"]["generation"] == 1
        assert await session._chrome_guard_reason(document) == ""
        assert session._chrome_state["lease"]["generation"] == 2
        assert session._chrome_last_step == 4

    for operation in ("click", "submit", "press"):
        asyncio.run(exercise(operation))


def test_typed_sequence_fences_selected_order_with_network_callbacks_between_steps(monkeypatch):
    monkeypatch.setattr(chrome, "target_reason", lambda *_a: "")
    monkeypatch.setattr(chrome, "request_reason", lambda *_a: "")
    steps = [{"operation": "click", "args": {"selector": "#go"}},
             {"operation": "wait", "args": {"selector": "#result"}},
             {"operation": "get_text", "args": {"selector": "#result"}},
             {"operation": "cached_text", "args": {"contains": "ready"}}]

    async def exercise():
        session = _session()
        session._chrome_invocation = ("inv-1", "browser_action", {"operation": "sequence", "args": {"steps": steps}})
        facts = {"kind": "primitive", "invocation_id": "inv-1", "step_id": 1, "state": _state(), **steps[1]}
        assert "differs" in await session._chrome_guard_reason(facts)
        assert getattr(session, "_chrome_sequence_index", 0) == 0
        facts.update(steps[0])
        assert await session._chrome_guard_reason(facts) == ""
        request = {"kind": "request", "invocation_id": "inv-1", "step_id": 2,
                   "request_id": "request-1", "url": "https://example.test/api", "method": "GET",
                   "post_data": None, "complete": True, "state": _state()}
        assert await session._chrome_guard_reason(request) == ""
        assert session._chrome_sequence_index == 1
        for index, step in enumerate(steps[1:], 3):
            facts.update(step, step_id=index)
            assert await session._chrome_guard_reason(facts) == ""
        facts["step_id"] = 6
        assert "outside" in await session._chrome_guard_reason(facts)
        session._chrome_invocation = None
        assert "invocation" in await session._chrome_guard_reason({**facts, "step_id": 7})

    asyncio.run(exercise())


def test_callback_assessment_cannot_land_after_invocation_return(monkeypatch):
    session = _session()
    closed = threading.Event()
    release = threading.Event()

    def assess(*_args):
        closed.set()
        assert release.wait(2)
        return ""

    monkeypatch.setattr(chrome, "request_reason", assess)
    facts = {"kind": "request", "invocation_id": "inv-1", "step_id": 1,
             "request_id": "late", "url": "https://example.test/a", "method": "GET",
             "post_data": None, "complete": True, "state": _state()}

    async def exercise():
        pending = asyncio.create_task(session._chrome_guard_reason(facts))
        assert await asyncio.to_thread(closed.wait, 2)
        session._chrome_invocation = None  # normal outer return or timeout
        release.set()
        assert "closed during assessment" in await pending
        assert not session._chrome_requests
        assert session._chrome_last_step == 0

    asyncio.run(exercise())
