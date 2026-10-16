"""Returned browser MCP evidence survives a later state failure through the manager."""

import asyncio
import collections
import json
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from ouroboros import artifacts, mcp_chrome_protocol, mcp_client, mcp_task_sessions
from ouroboros.tools.tool_result import ToolResult


def _response():
    from mcp.types import CallToolResult, ImageContent, TextContent

    return CallToolResult(
        isError=False,
        content=[TextContent(type="text", text="exact returned text\nλ"),
                 ImageContent(type="image", mimeType="image/png", data="aGVsbG8=")],
        structuredContent={"answer": [1, "literal"]},
    )


def _state(*, detached=False):
    return {
        "protocol": "ouroboros_chrome", "version": 1,
        "owner": {"task_id": "response-task", "attempt": "1"},
        "executor_generation": 1, "connected": True,
        "lease": {"lease_id": "lease-1", "generation": 1, "tab_id": "tab-1",
                  "document_id": "doc-1", "url": "https://example.test/", "guard_armed": True},
        "closure": {"revoked": False, "detached": detached},
    }


def _text_result(text):
    return SimpleNamespace(isError=False, content=[SimpleNamespace(type="text", text=text)])


def _registered_call(tmp_path, monkeypatch, protocol, outcome, *, store_fails=False, transport=True,
                     failure=None, failure_phase="post", observed_state=None, tool_name=None,
                     warm_up=False, target_policy=None, state_during_assessment=None,
                     safety_advice="", guard_note=""):
    cfg = mcp_client.normalize_server_config({
        "id": "browser", "enabled": True, "transport": "stdio", "command": "node",
        "args": ["unused.js"], "browser_bridge": True, "browser_protocol": protocol,
    })
    name = tool_name or ("browser_action" if protocol == "ouroboros_chrome" else "browser_snapshot")
    prefixed = mcp_client.make_tool_name("browser", name)
    manager = mcp_client.MCPManager()
    manager._enabled = True
    manager._servers["browser"] = mcp_client.MCPServerRuntime(
        config=cfg, tools=[mcp_client.MCPTool("browser", name, prefixed, "", {})])
    ctx = SimpleNamespace(task_id="response-task", task_attempt=1, task_lifecycle_bound=True,
                          messages=[], drive_root=tmp_path)
    session = object.__new__(mcp_task_sessions.TaskBrowserSession)
    session.cfg, session.ctx, session.key, session.attempt = cfg, ctx, (ctx.task_id, cfg.id), "1"
    session.data_root = tmp_path
    session.revoked = False
    session._call_lock = threading.Lock()
    session._blocked = collections.deque(maxlen=20)
    session._blocked_omitted = 0
    session.guard_armed = True
    session._chrome_state = _state()
    session._chrome_lease_generation_floor = 1
    session._chrome_invocation = None
    session._chrome_requests = set()
    session._chrome_documents = {}
    session._chrome_navigation_pending = False
    session._read_stream = None
    monkeypatch.setitem(mcp_task_sessions._sessions, session.key, session)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, safety_advice))
    monkeypatch.setattr(mcp_task_sessions, "_block_reason", lambda *_a: "")
    monkeypatch.setattr(mcp_chrome_protocol, "target_reason", target_policy or (lambda *_a: ""))
    monkeypatch.setattr(mcp_chrome_protocol, "binding_reason", lambda *_a: "")
    if store_fails:
        def fail_store(*_a, **_k):
            raise OSError("unavailable")
        monkeypatch.setattr(artifacts, "store_actor_source_bytes", fail_store)

    calls = []
    current_state = _state()

    def request(tool, _args, _timeout, *, dispatch=False, invocation_id=""):
        calls.append((tool, dispatch))
        if failure is not None and (
                (failure_phase == "pre" and len(calls) == 1)
                or (failure_phase == "action" and dispatch)
                or (failure_phase == "post" and len(calls) == 4)):
            raise failure
        if dispatch:
            if outcome == "timeout":
                raise mcp_task_sessions.BrowserBridgeTimeout("action response timed out")
            if guard_note:
                session._blocked.append(guard_note)
            return _response()
        if outcome == "post_timeout" and len(calls) == 4:
            raise mcp_task_sessions.BrowserBridgeTimeout("post-state read timed out")
        if protocol == "ouroboros_chrome":
            if outcome == "detached" and len(calls) == 4:
                return _text_result(json.dumps(_state(detached=True)))
            if state_during_assessment is not None and len(calls) == 2:
                return _text_result(json.dumps(state_during_assessment))
            return _text_result(json.dumps(current_state))
        if outcome == "broken_read" and len(calls) == 4:
            return _text_result("unreadable post-state")
        return _text_result("### Result\n- 0: (current) [Page](https://example.test/)")

    if not transport:
        session._request = request
    else:
        from ouroboros import owner_pause

        monkeypatch.setattr(owner_pause, "operation_start", lambda: nullcontext())
        loop = asyncio.new_event_loop()
        worker = threading.Thread(target=loop.run_forever, daemon=True)
        worker.start()

        class Transport:
            async def call_tool(self, tool, args, **_kw):
                dispatch = bool(_kw.get("meta")) if protocol == "ouroboros_chrome" else tool == name
                return request(tool, args, 1, dispatch=dispatch)

        session._loop, session._session = loop, Transport()
    args = {"operation": "snapshot", "args": {}} if name == "browser_action" else {}
    try:
        if warm_up:
            assert manager._call_tool_result(prefixed, args, ctx=ctx).status == "ok"
        if observed_state is not None:
            current_state = observed_state
        return manager._call_tool_result(prefixed, args, ctx=ctx), calls
    finally:
        if transport:
            loop.call_soon_threadsafe(loop.stop)
            worker.join(timeout=2)
            assert not worker.is_alive()
            loop.close()


@pytest.mark.parametrize("protocol", ["playwright", "ouroboros_chrome"])
@pytest.mark.parametrize("annotation", ["none", "safety", "guard", "both"])
def test_real_manager_composes_returned_browser_annotations(tmp_path, monkeypatch, protocol, annotation):
    advice = "Safety advice from the host" if annotation in {"safety", "both"} else ""
    note = "GET https://example.test/blocked: host refused" if annotation in {"guard", "both"} else ""
    result, calls = _registered_call(tmp_path, monkeypatch, protocol, "ok",
                                     safety_advice=advice, guard_note=note)
    assert result.status == "ok" and result.code == ("SAFETY_WARNING" if advice else "OK")
    assert "exact returned text\nλ" in result.text
    assert result.meta["mcp_is_error"] is False and result.meta["dynamic_provider"] is True
    assert bool(result.meta.get("safety_warning")) is bool(advice)
    if advice:
        assert advice in result.text
    if note:
        assert "BROWSER_REQUEST_BLOCKED" in result.text and note in result.text
    assert sum(dispatched for _name, dispatched in calls) == 1


@pytest.mark.parametrize("protocol,outcome", [("ouroboros_chrome", "detached"),
                                         ("playwright", "broken_read")])
def test_returned_response_retained_before_failed_post_state(tmp_path, monkeypatch, protocol, outcome):
    result, calls = _registered_call(tmp_path, monkeypatch, protocol, outcome, transport=True)
    assert calls == [("browser_state" if protocol == "ouroboros_chrome" else "browser_tabs", False)] * 2 + [
        ("browser_action" if protocol == "ouroboros_chrome" else "browser_snapshot", True),
        ("browser_state" if protocol == "ouroboros_chrome" else "browser_tabs", False)]
    assert (result.status, result.code) == ("error", "MCP_ERROR")
    assert "BROWSER_POLICY_POST_DISPATCH" in result.text
    source = result.meta["mcp_response_source"]
    assert result.meta["host_verdict"] and result.meta["dynamic_provider"]
    assert f"read_file(root='artifact_store', path='{source['path']}')" in result.text
    assert f"size={source['size']} bytes; sha256={source['sha256']}" in result.text
    data = artifacts.read_actor_source_bytes(tmp_path, "response-task", source)
    assert len(data) == source["size"]
    decoded = json.loads(data)
    assert decoded["content"][0]["text"] == "exact returned text\nλ"
    assert decoded["content"][1]["mimeType"] == "image/png"
    assert decoded["content"][1]["data"] == "aGVsbG8="
    assert decoded["structuredContent"] == {"answer": [1, "literal"]}
    assert "exact returned text" not in result.text  # withheld, but source remains readable


@pytest.mark.parametrize("protocol", ["ouroboros_chrome", "playwright"])
def test_normal_result_is_still_normal(tmp_path, monkeypatch, protocol):
    returned = []
    monkeypatch.setattr("ouroboros.owner_pause.record_mcp_call_returned", returned.append)
    result, _ = _registered_call(tmp_path, monkeypatch, protocol, "success")
    assert (result.status, result.code) == ("ok", "OK")
    assert "exact returned text\nλ" in result.text
    assert "mcp_response_source" not in result.meta
    assert len(returned) == 1


def test_retention_failure_is_explicit_and_timeout_has_no_false_source(tmp_path, monkeypatch):
    failed, _ = _registered_call(tmp_path, monkeypatch, "ouroboros_chrome", "detached", store_fails=True)
    assert failed.code == "MCP_ERROR"
    assert failed.meta["mcp_response_retention_gap"] == "OSError"
    assert "retention failed (OSError)" in failed.text
    timed, _ = _registered_call(tmp_path, monkeypatch, "playwright", "timeout", transport=False)
    assert timed.code == "MCP_TIMEOUT"
    assert "mcp_response_source" not in timed.meta
    assert "retained" not in timed.text


@pytest.mark.parametrize("protocol", ["ouroboros_chrome", "playwright"])
def test_post_state_timeout_keeps_received_response(tmp_path, monkeypatch, protocol):
    returned = []
    monkeypatch.setattr("ouroboros.owner_pause.record_mcp_call_returned", returned.append)
    result, _ = _registered_call(tmp_path, monkeypatch, protocol, "post_timeout", transport=False)
    assert (result.status, result.code) == ("error", "MCP_ERROR")
    assert len(returned) == 1
    assert "post-state read timed out" in result.text
    source = result.meta["mcp_response_source"]
    decoded = json.loads(artifacts.read_actor_source_bytes(tmp_path, "response-task", source))
    assert decoded["content"][0]["text"] == "exact returned text\nλ"


def _post_error(kind):
    from mcp.shared.exceptions import McpError
    from mcp.types import ErrorData

    return {"sdk": lambda: McpError(ErrorData(code=-32001, message="post-state SDK failure")),
            "io": lambda: OSError("post-state IO failure"),
            "value": lambda: ValueError("post-state invalid value"),
            "type": lambda: TypeError("post-state invalid type"),
            "empty": lambda: Exception()}[kind]()


@pytest.mark.parametrize("protocol", ["ouroboros_chrome", "playwright"])
@pytest.mark.parametrize("kind", ["sdk", "io", "value", "type", "empty"])
def test_post_response_exception_keeps_source_and_one_return(tmp_path, monkeypatch, protocol, kind):
    returned = []
    monkeypatch.setattr("ouroboros.owner_pause.record_mcp_call_returned", returned.append)
    failure = _post_error(kind)
    result, calls = _registered_call(tmp_path, monkeypatch, protocol, "success", failure=failure)
    action = "browser_action" if protocol == "ouroboros_chrome" else "browser_snapshot"
    assert returned == [mcp_client.make_tool_name("browser", action)]
    assert len(calls) == 4 and sum(dispatch for _, dispatch in calls) == 1
    assert (result.status, result.code) == ("error", "MCP_ERROR")
    assert "BROWSER_POLICY_POST_DISPATCH" in result.text and type(failure).__name__ in result.text
    assert "Reconcile before any retry" in result.text
    assert result.meta["host_verdict"] and result.meta["dynamic_provider"]
    source = result.meta["mcp_response_source"]
    assert f"read_file(root='artifact_store', path='{source['path']}')" in result.text
    decoded = json.loads(artifacts.read_actor_source_bytes(tmp_path, "response-task", source))
    assert decoded == _response().model_dump(mode="json", by_alias=True)
    assert "exact returned text" not in result.text


@pytest.mark.parametrize("protocol", ["ouroboros_chrome", "playwright"])
@pytest.mark.parametrize("phase", ["pre", "action"])
def test_sdk_error_without_response_has_no_source_or_return(tmp_path, monkeypatch, protocol, phase):
    returned = []
    monkeypatch.setattr("ouroboros.owner_pause.record_mcp_call_returned", returned.append)
    result, calls = _registered_call(tmp_path, monkeypatch, protocol, "success",
                                     failure=_post_error("sdk"), failure_phase=phase)
    assert (result.status, result.code) == ("error", "MCP_ERROR")
    assert not returned and "mcp_response_source" not in result.meta
    assert "BROWSER_POLICY_POST_DISPATCH" not in result.text
    assert sum(dispatch for _, dispatch in calls) == (1 if phase == "action" else 0)
    assert not list(tmp_path.rglob("mcp-browser-*.json"))


@pytest.mark.parametrize("protocol", ["ouroboros_chrome", "playwright"])
def test_post_sdk_error_with_retention_gap_still_records_return(tmp_path, monkeypatch, protocol):
    returned = []
    monkeypatch.setattr("ouroboros.owner_pause.record_mcp_call_returned", returned.append)
    result, calls = _registered_call(tmp_path, monkeypatch, protocol, "success",
                                     failure=_post_error("sdk"), store_fails=True)
    assert result.status == "error" and len(returned) == 1
    assert sum(dispatch for _, dispatch in calls) == 1
    assert result.meta["mcp_response_retention_gap"] == "OSError"
    assert "mcp_response_source" not in result.meta
    assert "McpError" in result.text and "retention failed" in result.text


@pytest.mark.parametrize("protocol", ["ouroboros_chrome", "playwright"])
def test_saved_response_projection_error_keeps_source(tmp_path, monkeypatch, protocol):
    def broken_projection(_raw):
        raise TypeError("cannot project returned response")

    returned = []
    monkeypatch.setattr("ouroboros.owner_pause.record_mcp_call_returned", returned.append)
    monkeypatch.setattr(mcp_client, "_tool_result_from_call_result", broken_projection)
    result, calls = _registered_call(tmp_path, monkeypatch, protocol, "success")
    assert result.status == "error" and len(returned) == 1
    assert len(calls) == 3 and sum(dispatch for _, dispatch in calls) == 1
    assert "TypeError: cannot project returned response" in result.text
    source = result.meta["mcp_response_source"]
    assert json.loads(artifacts.read_actor_source_bytes(tmp_path, "response-task", source)) == (
        _response().model_dump(mode="json", by_alias=True))


@pytest.mark.parametrize("protocol", ["ouroboros_chrome", "playwright"])
def test_post_response_task_control_still_reaches_caller(tmp_path, monkeypatch, protocol):
    from ouroboros.owner_pause import OwnerPauseRefused

    with pytest.raises(OwnerPauseRefused, match="owner paused"):
        _registered_call(tmp_path, monkeypatch, protocol, "success",
                         failure=OwnerPauseRefused("owner paused"))


@pytest.mark.parametrize("tool_name", ["browser_state", "browser_action", "browser_handoff", "browser_disconnect"])
@pytest.mark.parametrize("url", ["https://example.test/#section", "https://example.test/history?q=2"])
def test_same_document_url_between_registered_calls(tmp_path, monkeypatch, tool_name, url):
    returned = []
    monkeypatch.setattr("ouroboros.owner_pause.record_mcp_call_returned", returned.append)
    changed = _state()
    changed["lease"]["url"] = url
    result, calls = _registered_call(tmp_path, monkeypatch, "ouroboros_chrome", "success",
                                     observed_state=changed, tool_name=tool_name, warm_up=True)
    assert (result.status, result.code) == ("ok", "OK")
    assert sum(dispatch for _, dispatch in calls) == 2
    assert returned == [mcp_client.make_tool_name("browser", tool_name)] * 2
    session = mcp_task_sessions._sessions[("response-task", "browser")]
    assert session._chrome_state["lease"]["url"] == url
    assert session._chrome_lease_generation_floor == 1


@pytest.mark.parametrize("tool_name", ["browser_state", "browser_action", "browser_handoff", "browser_disconnect"])
def test_same_document_forbidden_url_refuses_before_dispatch(tmp_path, monkeypatch, tool_name):
    changed = _state()
    changed["lease"]["url"] = "https://example.test/forbidden"
    checked = []

    def policy(_ctx, url):
        checked.append(url)
        return "BROWSER_URL_BLOCKED: forbidden path" if url == changed["lease"]["url"] else ""

    result, calls = _registered_call(tmp_path, monkeypatch, "ouroboros_chrome", "success",
                                     observed_state=changed, tool_name=tool_name, target_policy=policy)
    assert result.status == "blocked" and "BROWSER_URL_BLOCKED" in result.text
    assert checked == [changed["lease"]["url"]]
    assert not any(dispatch for _, dispatch in calls)
    assert mcp_task_sessions._sessions[("response-task", "browser")]._chrome_state == _state()


def test_same_document_url_during_assessment_still_refuses_dispatch(tmp_path, monkeypatch):
    changed = _state()
    changed["lease"]["url"] = "https://example.test/#after-safety"
    result, calls = _registered_call(tmp_path, monkeypatch, "ouroboros_chrome", "success",
                                     state_during_assessment=changed)
    assert result.status == "blocked" and "BROWSER_STATE_CHANGED" in result.text
    assert len(calls) == 2 and not any(dispatch for _, dispatch in calls)


@pytest.mark.parametrize("field,value", [("document_id", "doc-2"), ("lease_id", "lease-2"),
                                        ("generation", 0), ("generation", 2),
                                        ("tab_id", "tab-2"), ("guard_armed", False),
                                        ("executor_generation", 2), ("revoked", True),
                                        ("detached", True), ("task_id", "other-task"),
                                        ("attempt", "2")])
def test_url_drift_cannot_hide_authority_change(tmp_path, monkeypatch, field, value):
    changed = _state()
    changed["lease"]["url"] = "https://example.test/#section"
    target = (changed if field == "executor_generation" else changed["closure"]
              if field in {"revoked", "detached"} else changed["owner"]
              if field in {"task_id", "attempt"} else changed["lease"])
    target[field] = value
    returned = []
    monkeypatch.setattr("ouroboros.owner_pause.record_mcp_call_returned", returned.append)
    result, calls = _registered_call(tmp_path, monkeypatch, "ouroboros_chrome", "success",
                                     observed_state=changed)
    assert result.status == "blocked"
    assert "not dispatched" in result.text
    assert not returned and not any(dispatch for _, dispatch in calls)
    assert mcp_task_sessions._sessions[("response-task", "browser")]._chrome_state == _state()


def test_manager_preserves_existing_producer_text_on_normal_route(tmp_path, monkeypatch):
    cfg = mcp_client.normalize_server_config({"id": "browser", "enabled": True, "transport": "stdio",
                                              "command": "node", "args": ["unused.js"], "browser_bridge": True})
    name = mcp_client.make_tool_name("browser", "browser_snapshot")
    manager = mcp_client.MCPManager()
    manager._enabled = True
    manager._servers["browser"] = mcp_client.MCPServerRuntime(
        cfg, tools=[mcp_client.MCPTool("browser", "browser_snapshot", name, "", {})])
    original = ToolResult(status="ok", code="OK", text="producer\n\n⚠️ host note",
                          producer_text="producer", host_annotations=("⚠️ host note",))
    monkeypatch.setattr(mcp_task_sessions, "call", lambda *_a: (original, "", ""))
    result = manager._call_tool_result(name, {}, ctx=SimpleNamespace())
    assert result.producer_text == "producer"
    assert result.host_annotations == ("⚠️ host note",)


def test_same_document_url_refresh_checks_observation_and_live_callback(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from tests.test_mcp_chrome_authority import _context

    monkeypatch.setattr(mcp_chrome_protocol, "binding_reason", lambda *_a: "")
    monkeypatch.setattr(mcp_chrome_protocol, "target_reason",
                        lambda _ctx, url: "BROWSER_URL_BLOCKED" if "forbidden" in url else "")
    session = object.__new__(mcp_task_sessions.TaskBrowserSession)
    session.cfg = SimpleNamespace(browser_protocol="ouroboros_chrome")
    session.ctx, session.data_root = _context(tmp_path, "owner"), tmp_path
    session.ctx.task_id = "response-task"
    session.key, session.attempt, session.revoked = ("response-task", "browser"), "1", False
    write_task_result(tmp_path, "response-task", STATUS_RUNNING)
    monkeypatch.setitem(mcp_task_sessions._sessions, session.key, session)
    session._chrome_state = _state()
    session._chrome_invocation = ("live", "browser_action", {"operation": "snapshot", "args": {}})
    session._chrome_last_step = 0
    session._chrome_sequence_index = 0
    session._chrome_navigation_pending = True
    session._chrome_lease_generation_floor = 1
    session._chrome_requests = set()
    session._chrome_documents = {}

    def changed(url, **lease_changes):
        state = _state()
        state["lease"] = {**state["lease"], "url": url, **lease_changes}
        return state

    fragment = changed("https://example.test/#section")
    session._request = lambda *_a, **_k: _text_result(json.dumps(fragment))
    assert session._observe(1)["lease"]["url"] == fragment["lease"]["url"]
    session._chrome_state = _state()

    def callback(state, *, step=1, invocation="live", kind="primitive"):
        return {"kind": kind, "invocation_id": invocation, "step_id": step,
                "operation": "snapshot", "args": {}, "state": state}

    assert "invocation" in asyncio.run(session._chrome_guard_reason(
        callback(fragment, invocation="stale")))
    assert session._chrome_state["lease"]["url"] == "https://example.test/"
    assert asyncio.run(session._chrome_guard_reason(callback(fragment))) == ""
    assert session._chrome_state["lease"]["url"] == fragment["lease"]["url"]
    assert session._chrome_lease_generation_floor == 1
    assert session._chrome_navigation_pending  # a later real document transition remains possible
    assert session._observe(1)["lease"]["url"] == fragment["lease"]["url"]

    history = changed("https://example.test/history")
    assert asyncio.run(session._chrome_guard_reason(callback(history, step=2))) == ""
    session._request = lambda *_a, **_k: _text_result(json.dumps(history))
    assert session._observe(1)["lease"]["url"] == history["lease"]["url"]
    forbidden = changed("https://forbidden.test/")
    assert asyncio.run(session._chrome_guard_reason(callback(forbidden, step=3))) == "BROWSER_URL_BLOCKED"
    assert session._chrome_state["lease"]["url"] == history["lease"]["url"]
    stale_generation = changed("https://example.test/later", generation=0)
    assert "LEASE_CHANGED" in asyncio.run(session._chrome_guard_reason(callback(stale_generation, step=3)))
    different_document = changed("https://example.test/later", document_id="doc-2")
    assert "LEASE_CHANGED" in asyncio.run(session._chrome_guard_reason(callback(different_document, step=3)))
    assert "LEASE_CHANGED" in asyncio.run(session._chrome_guard_reason(
        callback(changed("https://example.test/another"), step=3, kind="document")))
    session._chrome_invocation = None
    assert "invocation" in asyncio.run(session._chrome_guard_reason(callback(changed("https://example.test/late"), step=3)))
