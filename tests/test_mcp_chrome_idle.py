"""Lease-bound idle reads through the real session callback, without a browser."""

import asyncio
import copy
import ipaddress
import threading

import pytest

from ouroboros import browser_policy, mcp_chrome_protocol as chrome, mcp_task_sessions, owner_pause
from ouroboros.task_results import STATUS_RUNNING, write_task_result
from tests.test_mcp_chrome_authority import ACTORS, UNRESTRICTED, _ask, _context
from tests.test_mcp_chrome_session import _session, _state

pytestmark = pytest.mark.serial


@pytest.fixture
def idle(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "cyber_pro")
    monkeypatch.setattr(browser_policy, "_resolved_addresses", lambda host: {
        ipaddress.ip_address(host if host in {"169.254.169.254", "192.168.1.2"} else "8.8.8.8")})
    monkeypatch.setattr(browser_policy, "runtime_service_kind", lambda *_a: "")
    session = _session()
    session._chrome_invocation = None
    session._chrome_last_step = 7  # the previous invocation's counter is not idle authority
    session.ctx = _context(tmp_path, "owner")
    session.data_root = tmp_path
    write_task_result(tmp_path, "task-one", STATUS_RUNNING)
    monkeypatch.setitem(mcp_task_sessions._sessions, session.key, session)
    return session


def _read(**changes):
    return {"kind": "request", "invocation_id": None, "request_id": "idle-1",
            "url": "https://example.test/echo?idle-read", "method": "GET", "post_data": None,
            "complete": True, "has_post_data": False, "resource_type": "Fetch", "is_main_frame": False,
            "target_tab_id": "tab-1", "state": _state(), **changes}


@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize("body", [None, ""])
def test_idle_read_after_call_return_is_allowed_without_invocation_or_navigation_authority(idle, method, body):
    before = copy.deepcopy(idle._chrome_state)
    with idle._call_lock:  # callbacks must not wait for a tool call
        assert asyncio.run(_ask(idle, _read(method=method, post_data=body))) == ""
    assert idle._chrome_invocation is None and idle._chrome_last_step == 7
    assert idle._chrome_state == before
    assert not idle._chrome_requests and not idle._chrome_documents
    assert not idle._chrome_navigation_pending


@pytest.mark.parametrize("resource_type", ["Fetch", "XHR", "Document"])
@pytest.mark.parametrize("is_main_frame", [True, False])
def test_idle_frame_origin_does_not_authorize_navigation(idle, resource_type, is_main_frame):
    before = copy.deepcopy(idle._chrome_state)
    reason = asyncio.run(_ask(idle, _read(resource_type=resource_type, is_main_frame=is_main_frame)))
    if resource_type == "Document":
        assert "BROWSER_POLICY_UNAVAILABLE" in reason
    else:
        assert reason == ""
    assert idle._chrome_state == before and idle._chrome_invocation is None
    assert not idle._chrome_requests and not idle._chrome_documents
    assert not idle._chrome_navigation_pending and idle._chrome_last_step == 7


@pytest.mark.parametrize("frame_facts", [
    pytest.param({}, id="missing"),
    pytest.param({"is_main_frame": None}, id="null"),
    pytest.param({"is_main_frame": 0}, id="integer-zero"),
    pytest.param({"is_main_frame": 1}, id="integer-one"),
    pytest.param({"is_main_frame": 0.0}, id="float-zero"),
    pytest.param({"is_main_frame": "false"}, id="string-false"),
    pytest.param({"is_main_frame": "true"}, id="string-true"),
    pytest.param({"is_main_frame": []}, id="list"),
    pytest.param({"is_main_frame": {}}, id="object"),
])
def test_idle_requires_boolean_frame_fact(idle, frame_facts):
    facts = _read()
    del facts["is_main_frame"]
    facts.update(frame_facts)
    assert "BROWSER_POLICY_UNAVAILABLE" in asyncio.run(_ask(idle, facts))


@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize("body", [None, ""])
@pytest.mark.parametrize("body_facts", [
    pytest.param({}, id="missing"),
    pytest.param({"has_post_data": True}, id="true"),
    pytest.param({"has_post_data": None}, id="null"),
    pytest.param({"has_post_data": 0}, id="integer-zero"),
    pytest.param({"has_post_data": 1}, id="integer-one"),
    pytest.param({"has_post_data": 0.0}, id="float-zero"),
    pytest.param({"has_post_data": "false"}, id="string-false"),
    pytest.param({"has_post_data": "unknown"}, id="unknown"),
    pytest.param({"has_post_data": []}, id="list"),
    pytest.param({"has_post_data": {}}, id="object"),
])
def test_idle_requires_explicit_false_body_presence(idle, method, body, body_facts):
    facts = _read(method=method, post_data=body)
    del facts["has_post_data"]
    facts.update(body_facts)
    reason = asyncio.run(_ask(idle, facts))
    assert "BROWSER_POLICY_UNAVAILABLE" in reason
    assert not idle._chrome_requests and not idle._chrome_documents
    assert idle._chrome_last_step == 7


@pytest.mark.parametrize("changes", [
    {"method": "POST"}, {"method": "PUT"}, {"method": "DELETE"}, {"method": "OPTIONS"},
    {"method": "UNKNOWN"}, {"method": "get"}, {"method": ""}, {"method": None},
    {"post_data": "payload"}, {"post_data": False}, {"post_data": []},
    {"complete": False}, {"complete": 1}, {"resource_type": "Document"},
    {"resource_type": "WebSocket"}, {"resource_type": "unknown"},
    {"target_tab_id": "other"},
    {"kind": "primitive", "operation": "click", "args": {}}, {"kind": "document"},
    {"invocation_id": "returned-call"}, {"invocation_id": ""},
])
def test_idle_cannot_authorize_writes_navigation_or_unknown_facts(idle, changes):
    assert asyncio.run(_ask(idle, _read(**changes)))
    assert not idle._chrome_requests and not idle._chrome_documents
    assert idle._chrome_last_step == 7


@pytest.mark.parametrize("field", ["invocation_id", "request_id", "url", "method", "post_data",
                                   "complete", "has_post_data", "resource_type", "is_main_frame", "target_tab_id", "state"])
def test_idle_requires_every_fact(idle, field):
    facts = _read()
    del facts[field]
    assert asyncio.run(_ask(idle, facts))


@pytest.mark.parametrize("section,key,value", [
    ("owner", "task_id", "other"), ("owner", "attempt", "2"),
    (None, "executor_generation", 2), (None, "connected", False),
    ("lease", "lease_id", "other"), ("lease", "generation", 2),
    ("lease", "document_id", "other"), ("lease", "tab_id", "other"),
    ("lease", "guard_armed", False), ("closure", "revoked", True), ("closure", "detached", True),
])
def test_idle_never_adopts_changed_identity_or_closed_state(idle, section, key, value):
    before = copy.deepcopy(idle._chrome_state)
    facts = _read()
    (facts["state"][section] if section else facts["state"])[key] = value
    assert asyncio.run(_ask(idle, facts))
    assert idle._chrome_state == before


@pytest.mark.parametrize("actor", ACTORS)
@pytest.mark.parametrize("target", ["request", "page"])
def test_idle_rechecks_actor_caps_and_target_policy(idle, actor, target):
    idle.ctx = _context(idle.data_root, actor)
    assert asyncio.run(_ask(idle, _read())) == ""
    facts = _read()
    if target == "page":
        idle._chrome_state["lease"]["url"] = facts["state"]["lease"]["url"] = "http://169.254.169.254/latest"
    else:
        facts["url"] = "http://169.254.169.254/latest"
    assert bool(asyncio.run(_ask(idle, facts))) == (actor not in UNRESTRICTED)


def test_idle_checks_origin_grants_and_current_network_permission(idle):
    idle.ctx = _context(idle.data_root, "readonly")
    facts = _read(url="http://192.168.1.2/echo")
    assert "origin_not_granted" in asyncio.run(_ask(idle, facts))
    idle.ctx.task_contract["resource_policy"] = {"allowed_origins": ["http://192.168.1.2"]}
    assert asyncio.run(_ask(idle, facts)) == ""
    idle.ctx.task_contract["allowed_resources"] = {"network": False}
    assert "network" in asyncio.run(_ask(idle, facts))


@pytest.mark.parametrize("change", ["pause", "sleep", "cancel", "revoked", "replaced", "owner", "active"])
def test_idle_refuses_current_task_hold_or_lost_session_authority(idle, monkeypatch, change):
    if change == "pause":
        owner_pause.install_fence(idle.data_root, "task-one", request_id="pause-1")
    elif change == "sleep":
        write_task_result(idle.data_root, "task-one", STATUS_RUNNING,
                          budget_pause={"state": "paused", "reason": "sleep"})
    elif change == "cancel":
        from ouroboros.cancel_intents import request_cancel
        request_cancel(idle.data_root, "task-one", reason="owner stop", source="owner", requested_by="owner")
    elif change == "revoked":
        idle.revoked = True
    elif change == "replaced":
        monkeypatch.setitem(mcp_task_sessions._sessions, idle.key, _session())
    elif change == "owner":
        idle.ctx.task_attempt = 2
    else:
        idle._chrome_invocation = ("new-call", "browser_action", {"operation": "snapshot", "args": {}})
    assert asyncio.run(_ask(idle, _read()))


def test_idle_refuses_unreadable_task_control_authority(idle, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise ValueError("unreadable cancel authority")

    monkeypatch.setattr("ouroboros.cancel_intents.cancel_pending", unavailable)
    assert "UNAVAILABLE" in asyncio.run(_ask(idle, _read()))


def test_idle_same_document_url_still_requires_current_target_policy(idle):
    idle.ctx = _context(idle.data_root, "readonly")
    facts = _read()
    facts["state"]["lease"]["url"] = "https://example.test/history"
    assert asyncio.run(_ask(idle, facts)) == ""
    assert idle._chrome_state["lease"]["url"] == "https://example.test/"
    facts["state"]["lease"]["url"] = "http://169.254.169.254/latest"
    assert asyncio.run(_ask(idle, facts))


def test_idle_rechecks_binding_and_revokes_drift(idle, monkeypatch):
    idle._binding_revoked, idle._closing = False, asyncio.Event()
    monkeypatch.setattr(chrome, "binding_reason", lambda *_a: "BROWSER_SKILL_BINDING_REVOKED: changed")
    assert "BINDING_REVOKED" in asyncio.run(_ask(idle, _read()))
    assert idle.revoked and idle._binding_revoked


def test_idle_binding_drift_during_request_assessment_refuses_release(idle, monkeypatch):
    idle._binding_revoked, idle._closing = False, asyncio.Event()
    current = [True]
    monkeypatch.setattr(chrome, "binding_reason", lambda *_a: "" if current[0] else "BROWSER_SKILL_BINDING_REVOKED: changed during policy")

    def assess(*_args):
        current[0] = False
        return ""

    monkeypatch.setattr(chrome, "request_reason", assess)
    reason = asyncio.run(_ask(idle, _read()))
    assert "BINDING_REVOKED" in reason
    assert idle.revoked and not idle._chrome_requests and not idle._chrome_documents


@pytest.mark.parametrize("change", ["call", "lease", "revoke", "pause", "cancel", "network"])
def test_idle_permission_cannot_land_after_authority_changes_during_policy(idle, monkeypatch, change):
    entered, release = threading.Event(), threading.Event()

    def assess(*_args):
        entered.set()
        assert release.wait(5)
        return ""

    monkeypatch.setattr(chrome, "request_reason", assess)

    async def exercise():
        pending = asyncio.create_task(idle._chrome_guard_reason(_read()))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            if change == "call":
                idle._chrome_invocation = ("new", "browser_action", {"operation": "snapshot", "args": {}})
            elif change == "lease":
                idle._chrome_state = _state()
            elif change == "revoke":
                idle.revoked = True
            elif change == "pause":
                owner_pause.install_fence(idle.data_root, "task-one", request_id="pause-2")
            elif change == "cancel":
                from ouroboros.cancel_intents import request_cancel
                request_cancel(idle.data_root, "task-one", reason="stop", source="owner", requested_by="owner")
            else:
                idle.ctx.task_contract["allowed_resources"] = {"network": False}
        finally:
            release.set()
        assert await pending
        assert not idle._chrome_requests and not idle._chrome_documents

    asyncio.run(exercise())


@pytest.mark.parametrize("change", ["unchanged", "pause", "sleep", "cancel", "root_cancel",
                                   "network", "cap", "call", "lease", "context", "replaced", "revoke"])
def test_idle_socket_rechecks_authority_after_final_binding_await(idle, monkeypatch, change):
    entered, release = threading.Event(), threading.Event()
    binding = chrome.binding_reason
    checks = []
    before = copy.deepcopy(idle._chrome_state)
    idle.ctx.task_metadata["root_task_id"] = "parent"
    write_task_result(idle.data_root, "parent", STATUS_RUNNING)

    def final_binding(*args):
        checks.append(True)
        if len(checks) == 2:
            entered.set()
            assert release.wait(5)
        return binding(*args)

    monkeypatch.setattr(chrome, "binding_reason", final_binding)

    async def exercise():
        pending = asyncio.create_task(_ask(idle, _read()))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            if change == "pause":
                owner_pause.install_fence(idle.data_root, "parent", request_id="during-final-binding")
            elif change == "sleep":
                write_task_result(idle.data_root, "task-one", STATUS_RUNNING,
                                  budget_pause={"state": "paused", "reason": "sleep"})
            elif change in {"cancel", "root_cancel"}:
                from ouroboros.cancel_intents import request_cancel
                request_cancel(idle.data_root, "parent" if change == "root_cancel" else "task-one",
                               reason="stop during binding", source="owner", requested_by="owner")
            elif change == "network":
                idle.ctx.task_contract["allowed_resources"] = {"network": False}
            elif change == "cap":
                idle.ctx.task_metadata["runtime_mode_cap"] = "light"
            elif change == "call":
                idle._chrome_invocation = ("new", "browser_action", {"operation": "snapshot", "args": {}})
            elif change == "lease":
                idle._chrome_state = _state()
            elif change == "context":
                idle.ctx = _context(idle.data_root, "owner")
            elif change == "replaced":
                monkeypatch.setitem(mcp_task_sessions._sessions, idle.key, _session())
            elif change == "revoke":
                idle.revoked = True
        finally:
            release.set()
        reason = await pending
        assert bool(reason) is (change != "unchanged"), (change, reason)
        assert len(checks) == 2
        assert idle._chrome_state == before
        assert idle._chrome_last_step == 7 and not idle._chrome_navigation_pending
        assert not idle._chrome_requests and not idle._chrome_documents

    asyncio.run(exercise())
