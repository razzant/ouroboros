"""Active and idle callbacks share a final, effect-time authority boundary."""

import asyncio
import copy
import ipaddress
import threading

import pytest

from ouroboros import browser_policy, mcp_chrome_protocol as chrome, mcp_task_sessions, owner_pause, skill_loader
from ouroboros.task_results import STATUS_RUNNING, write_task_result
from tests import test_mcp_chrome_binding as binding_cases
from tests.test_mcp_chrome_authority import _ask, _context
from tests.test_mcp_chrome_session import _session, _state

pytestmark = pytest.mark.serial

installed = binding_cases.installed


@pytest.mark.parametrize("kind", ["primitive", "request", "document"])
@pytest.mark.parametrize("change", ["unchanged", "payload", "disable", "cap", "network", "pause", "cancel"])
def test_active_socket_rechecks_authority_after_policy_await(installed, monkeypatch, kind, change):
    root, _directory, script, cfg, _ctx = installed
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "cyber_pro")
    monkeypatch.setattr(browser_policy, "_resolved_addresses", lambda _host: {ipaddress.ip_address("8.8.8.8")})
    monkeypatch.setattr(browser_policy, "runtime_service_kind", lambda *_a: "")
    session = _session()
    session.cfg, session.ctx, session.data_root = cfg, _context(root, "owner"), root
    session._binding_revoked, session._closing = False, asyncio.Event()
    _resolved, session._chrome_skill_hash = chrome.resolve_binding(cfg, session.ctx)
    write_task_result(root, "task-one", STATUS_RUNNING)
    monkeypatch.setitem(mcp_task_sessions._sessions, session.key, session)
    before = copy.deepcopy(session._chrome_state)
    facts = {"kind": kind, "invocation_id": "inv-1", "step_id": 1, "state": _state(),
             "operation": "click", "args": {}}
    policy_name = "guard_reason"
    if kind == "request":
        facts.update(request_id="request-1", url="https://example.test/resource", method="GET",
                     post_data=None, complete=True)
    elif kind == "document":
        candidate = facts["state"]
        candidate["lease"].update(lease_id="next-lease", document_id="next-document", generation=2,
                                  url="https://example.test/next")
        session._chrome_documents["navigation-1"] = candidate["lease"]["url"]
        facts["request_id"] = "navigation-1"
        policy_name = "navigation_transition"
    documents = dict(session._chrome_documents)
    entered, release = threading.Event(), threading.Event()
    policy = getattr(chrome, policy_name)
    assessments = []

    def suspended_policy(*args):
        result = policy(*args)
        assessments.append(result)
        entered.set()
        release.wait(5)
        return result

    monkeypatch.setattr(chrome, policy_name, suspended_policy)

    async def exercise():
        pending = asyncio.create_task(_ask(session, facts))
        try:
            assert await asyncio.to_thread(entered.wait, 5), "policy was not reached"
            assert assessments == [""], assessments
            if change == "payload":
                script.write_text("console.log('changed')", encoding="utf-8")
            elif change == "disable":
                skill_loader.save_enabled(root, "chrome_browser", False)
            elif change == "cap":
                session.ctx.task_metadata["runtime_mode_cap"] = "light"
            elif change == "network":
                session.ctx.task_contract["allowed_resources"] = {"network": False}
            elif change == "pause":
                owner_pause.install_fence(root, "task-one", request_id="during-active-policy")
            elif change == "cancel":
                from ouroboros.cancel_intents import request_cancel
                request_cancel(root, "task-one", reason="during active assessment", source="owner", requested_by="owner")
        finally:
            release.set()
        reason = await pending
        assert bool(reason) is (change != "unchanged"), (kind, change, reason)
        if change == "unchanged":
            assert session._chrome_last_step == 1
        else:
            assert session._chrome_state == before
            assert session._chrome_last_step == 0
            assert not session._chrome_requests
            assert session._chrome_documents == documents
            assert not session._chrome_navigation_pending
            if change in {"payload", "disable"}:
                assert "BROWSER_SKILL_BINDING_REVOKED" in reason
                assert session.revoked and session._binding_revoked

    asyncio.run(exercise())
