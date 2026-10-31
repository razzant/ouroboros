"""Actual wait controls against canonical decision and current-attempt projections."""

import copy
import json

import pytest

from ouroboros import model_wait, owner_mailbox
from ouroboros.task_results import load_task_result
from tests.test_llm_claudexor import setup as subscription_transport
from tests.test_model_wait import live_wait as wait_fixture
from tests.test_model_wait_browser import API_LANE, TASK, waiting_ui as waiting_fixture
from tests.test_subscription_setup_browser import subscription_ui as ui_fixture, capture
from tests._work_projection_shared import state_fields

setup = subscription_transport
live_wait = wait_fixture
subscription_ui = ui_fixture
waiting_ui = waiting_fixture
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def test_real_mailbox_refusal_retries_exact_request_and_applies_once(subscription_ui, live_wait, monkeypatch):
    root, _transport, _client, owner, events, decide = live_wait
    row = {"wait_id": "write-refusal", "revision": 1, "task_attempt": 1,
           "state": "waiting", "role": "light", "reason": "quota",
           "model": "claudexor::codex=gpt-test", "source": "codex",
           "credential_profile_id": "", "credential_harness": "codex",
           "worker_slot_held": True, "auto_continue": True}
    model_wait.mutate_wait(root, "task-one", row["wait_id"], lambda _: dict(row))
    owner.waits[row["wait_id"]] = dict(row)
    owner.revision = 1
    page = subscription_ui["page"]
    replies, sockets = [], []
    page.route_web_socket("**/ws", lambda socket: sockets.append(socket))

    def reply(route, value):
        route.fulfill(content_type="application/json", body=json.dumps(value))

    page.route("**/api/state", lambda route: reply(route, {
        "sha": "fixture", "supervisor_ready": True, "projects": [],
        "active_chat_activities": [{"activity_id": "task-one", "kind": "managed_task",
            "chat_id": 1, "phase": "working", "task_attempt": 1, "model_waits": {row["wait_id"]: row}}]}))
    page.route("**/api/chat/history*", lambda route: reply(route, {"messages": [{
        "role": "system", "text": "", "system_type": "task_model_wait", "task_id": "task-one",
        "ts": "2026-09-07T00:00:00Z", "model_waits": {row["wait_id"]: row}}]}))
    page.route("**/api/tasks/task-one", lambda route: reply(route, load_task_result(root, "task-one")))
    original_write = owner_mailbox.write_owner_message
    monkeypatch.setattr(owner_mailbox, "write_owner_message", lambda *args, **kwargs: False)

    def decision(route):
        body = route.request.post_data_json
        result = decide(body)
        replies.append((body, result.status_code, json.loads(result.body)))
        route.fulfill(status=result.status_code, content_type="application/json", body=result.body)

    page.route("**/api/decisions", decision)
    page.goto(subscription_ui["url"] + "/")
    waiter = page.locator('[data-wait-id="write-refusal"]')
    waiter.wait_for()
    waiter.locator('[data-wait-auto]').uncheck()
    waiter.locator('[data-wait-notice]').filter(has_text="mailbox_write_failed").wait_for()
    assert len(replies) == 1 and replies[0][1] == 503
    request, _, failure = replies[0]
    assert failure["wait"]["pending_action"]["request_id"] == request["request_id"]
    assert "applied_request_id" not in failure["wait"]
    assert waiter.locator('[data-wait-retry]').is_disabled()
    assert waiter.locator('[data-wait-change]').is_disabled()
    assert waiter.locator('[data-wait-repeat]').is_enabled()
    assert waiter.locator('[data-wait-repeat]').is_visible()
    capture(page, "mailbox-refusal-exact-retry")
    monkeypatch.setattr(owner_mailbox, "write_owner_message", original_write)
    waiter.locator('[data-wait-repeat]').click()
    waiter.locator('[data-wait-notice]').filter(has_text="Request accepted").wait_for()
    assert len(replies) == 2 and replies[1][0] == request
    assert replies[1][1] == 200 and replies[1][2]["duplicate"] is True
    owner._drain_controls()
    event = events.get_nowait()
    assert event["type"] == "task_model_wait"
    assert event["applied_request_id"] == request["request_id"]
    assert event["auto_continue"] is False and "pending_action" not in event
    sockets[-1].send(json.dumps({"type": "log", "chat_id": 1, "data": event}))
    page.wait_for_selector('[data-wait-id="write-refusal"] [data-wait-change]:not([disabled])')
    assert not waiter.locator('[data-wait-auto]').is_checked()
    assert waiter.locator('[data-wait-repeat]').is_hidden()
    owner._drain_controls()
    assert events.empty(), "An exact delivery retry must apply only once"
    capture(page, "mailbox-retry-applied-once")


@pytest.mark.parametrize("shared_local", [False, True])
def test_fallback_local_change_discloses_task_only_and_keeps_other_roles(waiting_ui, shared_local):
    ui, page = waiting_ui, waiting_ui["page"]
    ui["settings"]["USE_LOCAL_FALLBACK"] = shared_local
    row = ui["rows"]["light-wait"]
    row.update(role="fallback:1", revision=2)
    ui["emit"](row)
    waiter = page.locator('[data-wait-id="light-wait"]')
    waiter.locator('[data-wait-role]').filter(has_text="Fallback 2").wait_for()
    waiter.locator('[data-wait-change]').click()
    waiter.locator('[data-model-role-source]').select_option(API_LANE)
    waiter.locator('[data-model-role-model]').fill("replacement")
    local = waiter.locator('[data-model-local]')
    persist = waiter.locator('[data-wait-persist]')
    local.set_checked(shared_local)
    assert persist.is_enabled()
    persist.check()
    local.set_checked(not shared_local)
    assert not persist.is_checked() and persist.is_disabled()
    assert "task-only" in waiter.locator('[data-wait-scope]').inner_text()
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    waiter.locator('[data-wait-scope]').evaluate("el => el.scrollIntoView({block: 'center'})")
    capture(page, f"fallback-local-task-only-{shared_local}")
    waiter.locator('[data-wait-apply]').click()
    waiter.locator('[data-wait-notice]').filter(has_text="Request accepted").wait_for()
    assert ui["controls"][-1]["persist_role"] is False
    assert ui["controls"][-1]["use_local"] is not shared_local
    assert ui["controls"][-1]["decision_id"].endswith(":light-wait")
    assert ui["settings"]["USE_LOCAL_FALLBACK"] is shared_local
    ui["apply"]("light-wait")
    waiter.wait_for(state="detached")
    assert page.locator('[data-wait-id="main-wait"]').count() == 1


def test_current_attempt_snapshot_never_reopens_old_wait_after_reload(waiting_ui, monkeypatch, tmp_path):
    from ouroboros.gateway import state
    from supervisor import queue

    ui, page = waiting_ui, waiting_ui["page"]
    task = {"id": TASK, "root_task_id": TASK, "delegation_role": "root", "chat_id": 1,
            "_attempt": 1, "model_waits": copy.deepcopy(ui["rows"])}
    meta = {"task": task, "attempt": 2, "started_at": 1}
    monkeypatch.setattr(queue, "RUNNING", {TASK: meta})
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {})
    monkeypatch.setattr(state, "_direct_turns_snapshot_safe", lambda **_kwargs: [])

    def current_state(route):
        # The real census and the server's own work projection of it.
        fields = state_fields(tmp_path)
        assert fields["active_chat_activities"][0]["task_attempt"] == 2
        route.fulfill(content_type="application/json", body=json.dumps({
            "sha": "browser-fixture", "supervisor_ready": True, "projects": [], **fields}))

    page.route("**/api/state", current_state)
    ui["errors"].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload()
    # Activities own the header, not card creation. A retained old wait alone
    # must not mint a card for this attempt before its first real progress.
    page.locator('#chat-status').filter(has_text="Working").wait_for()
    page.wait_for_function("() => document.querySelectorAll('.model-wait-row').length === 0")
    assert len(task["model_waits"]) == 2, "Old rows remain retained as history"
    capture(page, "retried-task-without-old-wait")
    current = {**ui["rows"]["light-wait"], "wait_id": "current-wait", "task_attempt": 2}
    task["model_waits"][current["wait_id"]] = current
    ui["emit"](current)
    page.wait_for_selector('[data-wait-id="current-wait"]')
    assert page.locator('.model-wait-row').count() == 1
    ui["emit"](ui["rows"]["light-wait"])
    assert page.locator('.model-wait-row').count() == 1
    capture(page, "retried-task-current-wait-only")
