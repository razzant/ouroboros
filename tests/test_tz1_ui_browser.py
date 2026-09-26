"""Exercise TZ-1 owner history/readiness presentation in the real SPA."""
from __future__ import annotations

import json

import pytest

from tests.test_subscription_setup_browser import subscription_ui as subscription_ui, capture

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def test_saved_input_and_starting_survive_reload(subscription_ui):
    ui = subscription_ui
    page = ui["page"]
    rows = [{
        "role": "user", "text": "TZ-1 saved message", "ts": "2026-09-25T03:00:00Z",
        "client_message_id": "tz1-saved-message", "ingress_accepted": True,
    }]
    page.route("**/api/state*", lambda route: route.fulfill(
        content_type="application/json", body=json.dumps({
            "supervisor_ready": False, "supervisor_error": "still starting",
            "active_chat_activities": [], "active_chat_activities_complete": True,
            "projects": [],
        }),
    ))
    page.route("**/api/chat/history*", lambda route: route.fulfill(
        content_type="application/json", body=json.dumps({"messages": rows, "progress": []}),
    ))
    page.goto(ui["url"])
    saved = page.locator('.chat-bubble.user[data-client-message-id="tz1-saved-message"] [data-ingress-saved]')
    saved.wait_for()
    assert saved.inner_text() == "Input saved"
    assert page.get_by_text("Starting…", exact=True).count() >= 1
    capture(page, "tz1-saved-and-starting")
    page.reload()
    saved.wait_for()
    assert saved.count() == 1
    assert page.get_by_text("Starting…", exact=True).count() >= 1
    assert not ui["errors"], ui["errors"]


def test_failed_supervisor_init_never_paints_online(subscription_ui):
    """The host's failure rail publishes `supervisor_ready: false` beside the
    error; the real SPA header says Starting… and never Online over it."""
    ui = subscription_ui
    page = ui["page"]
    page.route("**/api/state*", lambda route: route.fulfill(
        content_type="application/json", body=json.dumps({
            "supervisor_ready": False, "supervisor_error": "Supervisor init failed: boot dependency refused",
            "active_chat_activities": [], "active_chat_activities_complete": True, "projects": [],
        }),
    ))
    page.route("**/api/chat/history*", lambda route: route.fulfill(
        content_type="application/json", body=json.dumps({"messages": [], "progress": []}),
    ))
    page.goto(ui["url"])
    badge = page.locator("#chat-status")
    badge.wait_for()
    page.wait_for_function("() => document.querySelector('#chat-status')?.textContent === 'Starting…'")
    assert "online" not in (badge.get_attribute("class") or "").split()
    assert page.get_by_text("Online", exact=True).count() == 0
    capture(page, "tz1-failed-init-starting")
    assert not ui["errors"], ui["errors"]
