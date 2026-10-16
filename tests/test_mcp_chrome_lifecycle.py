"""Closed-control observations through call_chrome and the real state/guard checks."""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from ouroboros import mcp_task_sessions, owner_pause
from tests.test_mcp_chrome_idle import idle as _idle
from tests.test_mcp_chrome_session import _state

idle = _idle  # Register the shared pytest fixture without an unused import.


@pytest.fixture
def controls(idle, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    wire = SimpleNamespace(state=_state(), calls=[], error=False, disconnect_failure=None,
                           next_generation=2, next_executor=1, next_owner="task-one",
                           detached=True, revoked=False, post_failure=False)

    def request(name, args, _timeout, *, dispatch=False, invocation_id=""):
        wire.calls.append((name, dispatch))
        if name == "browser_state":
            if wire.post_failure and not wire.state["connected"]:
                return SimpleNamespace(isError=False, content=[SimpleNamespace(type="text", text="unknown")])
            payload = copy.deepcopy(wire.state)
        else:
            if name in {"browser_disconnect", "browser_handoff"}:
                if wire.disconnect_failure:
                    raise wire.disconnect_failure
                wire.state.update(connected=False, lease=None,
                                  closure={"revoked": wire.revoked, "detached": wire.detached})
            elif name in {"browser_connect", "browser_resume"}:
                candidate = _state()
                candidate["owner"]["task_id"] = wire.next_owner
                candidate["executor_generation"] = wire.next_executor
                candidate["lease"].update(generation=wire.next_generation, lease_id="lease-2",
                                          document_id="doc-2", tab_id="tab-2",
                                          url=args.get("url", "https://example.test/"))
                reason = asyncio.run(idle._chrome_guard_reason({
                    "kind": "primitive", "invocation_id": invocation_id, "step_id": 1,
                    "operation": name.removeprefix("browser_"), "args": args, "state": candidate}))
                if reason:
                    return SimpleNamespace(isError=True, content=[SimpleNamespace(type="text", text=reason)])
                wire.state = candidate
            payload = {"ok": not wire.error}
        return SimpleNamespace(isError=wire.error if dispatch else False,
                               content=[SimpleNamespace(type="text", text=json.dumps(payload))])

    idle._request = request

    def call(name, args=None):
        return idle.call("mcp_browser__" + name, name, args or {}, idle.ctx, 2)[0]

    return idle, wire, call


def test_clean_explicit_disconnect_allows_state_then_new_url_connect(controls):
    session, wire, call = controls
    assert call("browser_disconnect").status == "ok"
    assert call("browser_state").status == "ok"
    assert call("browser_connect", {"url": "https://example.test/new"}).status == "ok"
    assert session._chrome_state["lease"]["tab_id"] == "tab-2"
    assert session._chrome_lease_generation_floor == 2
    assert [name for name, dispatched in wire.calls if dispatched] == [
        "browser_disconnect", "browser_state", "browser_connect"]


@pytest.mark.parametrize("control", ["browser_disconnect", "browser_handoff"])
def test_state_is_readable_after_clean_closed_control(controls, control):
    _session, _wire, call = controls
    assert call(control).status == "ok"
    assert call("browser_state").status == "ok"


def test_cold_connect_and_explicit_resume_still_work(controls):
    session, wire, call = controls
    wire.state.update(connected=False, lease=None)
    session._chrome_state = copy.deepcopy(wire.state)
    session._chrome_lease_generation_floor = -1
    assert call("browser_connect", {"url": "https://example.test/new"}).status == "ok"
    assert call("browser_handoff").status == "ok"
    wire.next_generation = 3
    assert call("browser_resume").status == "ok"
    assert session._chrome_lease_generation_floor == 3


def test_state_can_observe_remote_revocation_without_reopening(controls):
    _session, wire, call = controls
    wire.revoked = True
    call("browser_disconnect")
    assert call("browser_state").status == "ok"
    with pytest.raises(mcp_task_sessions.BrowserBridgeRefusal):
        call("browser_connect", {"url": "https://example.test/new"})


@pytest.mark.parametrize("closure", ["handoff", "error", "not_detached", "revoked", "post_unknown", "unknown_ack"])
def test_connect_cannot_reopen_without_a_clean_disconnect_receipt(controls, closure):
    _session, wire, call = controls
    wire.error = closure == "error"
    wire.detached = closure != "not_detached"
    wire.revoked = closure == "revoked"
    wire.post_failure = closure == "post_unknown"
    if closure == "unknown_ack":
        wire.disconnect_failure = RuntimeError("lost disconnect response")
        with pytest.raises(RuntimeError, match="lost disconnect"):
            call("browser_disconnect")
        wire.state.update(connected=False, lease=None, closure={"revoked": False, "detached": True})
    else:
        call("browser_handoff" if closure == "handoff" else "browser_disconnect")
    wire.error, wire.post_failure, wire.disconnect_failure = False, False, None
    sent = len([entry for entry in wire.calls if entry[1]])
    with pytest.raises(mcp_task_sessions.BrowserBridgeRefusal):
        call("browser_connect", {"url": "https://example.test/new"})
    assert len([entry for entry in wire.calls if entry[1]]) == sent


@pytest.mark.parametrize("change", ["generation", "executor", "owner"])
def test_reconnect_still_requires_new_generation_same_owner_and_executor(controls, change):
    session, wire, call = controls
    assert call("browser_disconnect").status == "ok"
    setattr(wire, {"generation": "next_generation", "executor": "next_executor", "owner": "next_owner"}[change],
            {"generation": 1, "executor": 2, "owner": "foreign"}[change])
    answer = call("browser_connect", {"url": "https://example.test/new"})
    assert answer.status != "ok"
    assert not session._chrome_state["connected"]
    assert session._chrome_lease_generation_floor == 1


def test_stop_revocation_cannot_reuse_clean_disconnect(controls):
    session, wire, call = controls
    assert call("browser_disconnect").status == "ok"
    session.revoked = True  # Stop/Cancel closes this same session before teardown
    sent = list(wire.calls)
    with pytest.raises(mcp_task_sessions.BrowserBridgeRefusal, match="REVOKED"):
        call("browser_connect", {"url": "https://example.test/new"})
    assert wire.calls == sent


@pytest.mark.parametrize("hold", ["pause", "cancel", "network"])
def test_reconnect_checks_current_task_hold_before_dispatch(controls, hold):
    session, wire, call = controls
    assert call("browser_disconnect").status == "ok"
    if hold == "pause":
        owner_pause.install_fence(session.data_root, "task-one", request_id="pause-3")
    elif hold == "cancel":
        from ouroboros.cancel_intents import request_cancel
        request_cancel(session.data_root, "task-one", reason="stop", source="owner", requested_by="owner")
    else:
        session.ctx.task_contract["allowed_resources"] = {"network": False}
    sent = list(wire.calls)
    with pytest.raises(mcp_task_sessions.BrowserBridgeRefusal):
        call("browser_connect", {"url": "https://example.test/new"})
    assert wire.calls == sent


def test_unknown_state_read_invalidates_clean_disconnect_receipt(controls):
    _session, wire, call = controls
    assert call("browser_disconnect").status == "ok"
    wire.post_failure = True
    with pytest.raises(mcp_task_sessions.BrowserBridgeRefusal):
        call("browser_state")
    wire.post_failure = False
    with pytest.raises(mcp_task_sessions.BrowserBridgeRefusal):
        call("browser_connect", {"url": "https://example.test/new"})


def test_background_refusal_diagnostics_are_disclosed_once_on_next_response(controls):
    session, _wire, _call = controls
    session._blocked.append("BACKGROUND_REFUSAL_WITNESS")
    session._blocked_omitted = 2
    _result, _advice, first = session.call("mcp_browser__browser_state", "browser_state", {}, session.ctx, 2)
    assert "BACKGROUND_REFUSAL_WITNESS" in first
    assert "omitted: 2" in first
    assert "during this call" not in first  # the request predated the invocation
    _result, _advice, second = session.call("mcp_browser__browser_state", "browser_state", {}, session.ctx, 2)
    assert not second
