"""Browser regressions for generated subagent preview identity preservation."""
from __future__ import annotations

import copy
import json

import pytest

from tests.test_subscription_setup_browser import subscription_ui as subscription_ui
from tests.test_subscription_setup_browser import capture

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def _watch_previews(page):
    page.evaluate("performance.clearResourceTimings()")


def _wait_for_two_previews(page):
    page.wait_for_function("""
        () => performance.getEntriesByType('resource')
            .filter(entry => entry.name.includes('/api/onboarding/subagents/preview')).length >= 2
    """)


def test_settings_same_preview_keeps_live_row_handler_and_payload(subscription_ui):
    ui, page = subscription_ui, subscription_ui["page"]
    roster = copy.deepcopy(ui["fixture"]["preview"]["available_subagents"])
    ui["settings"]["OUROBOROS_SUBAGENTS"] = ""
    ui["settings"]["_meta"]["available_subagents"] = {"source": "undecided", "candidate": roster}
    saves = []

    def settings_route(route):
        if route.request.method == "POST":
            saves.append(route.request.post_data_json)
            route.fulfill(content_type="application/json", body=json.dumps({
                "status": "saved", "saved": True, "restart_required": False,
            }))
        else:
            route.fulfill(content_type="application/json", body=json.dumps(ui["settings"]))

    page.route("**/api/settings", settings_route)
    _watch_previews(page)
    page.goto(ui["url"] + "/#settings")
    with page.expect_response("**/api/onboarding/subagents/preview"):
        page.locator('[data-settings-tab="agents"]').click()
    page.wait_for_selector("[data-subagent-row]")
    _wait_for_two_previews(page)
    page.evaluate("window.__rowBefore = document.querySelector('[data-subagent-row]')")
    row = page.locator("[data-subagent-row]").first
    field = row.locator('[data-subagent-field="recommended_use"]')
    field.fill("SETTINGS EDIT 123")
    assert page.evaluate("() => document.activeElement === document.querySelector('[data-subagent-field=recommended_use]')")
    assert page.evaluate("() => window.__rowBefore === document.querySelector('[data-subagent-row]')")
    with page.expect_response("**/api/settings"):
        page.locator("#btn-save-settings").click()
    assert saves
    payload = saves[-1]["OUROBOROS_SUBAGENTS"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    assert payload["items"][0]["recommended_use"] == "SETTINGS EDIT 123"


def test_wizard_same_preview_keeps_live_row_handler_and_finish_payload(subscription_ui):
    ui, page = subscription_ui, subscription_ui["page"]
    _watch_previews(page)
    page.goto(ui["url"] + "/onboarding")
    page.wait_for_selector("#quick-start-btn:not([hidden])")
    page.click("#next-btn")
    page.wait_for_selector('[data-model-role="main"]')
    page.locator("details:has(#onboarding-available-subagents) > summary").click()
    page.wait_for_selector("#onboarding-available-subagents [data-subagent-row]")
    _wait_for_two_previews(page)
    page.evaluate("window.__rowBefore = document.querySelector('#onboarding-available-subagents [data-subagent-row]')")
    field = page.locator("#onboarding-available-subagents [data-subagent-field='recommended_use']").first
    field.fill("WIZARD EDIT 456")
    assert page.evaluate("() => window.__rowBefore === document.querySelector('#onboarding-available-subagents [data-subagent-row]')")
    page.click("#next-btn")
    page.wait_for_selector("[data-reviewers-note]", state="attached")
    page.click("#next-btn")
    page.wait_for_selector('[data-collapse="api-budget"]')
    page.click("#next-btn")
    page.wait_for_selector(".summary-card")
    with page.expect_response("**/api/onboarding/complete"):
        page.click("#next-btn")
    writes = [body for path, body in ui["posts"] if path == "/api/onboarding/complete"]
    assert len(writes) == 1
    payload = writes[0]["OUROBOROS_SUBAGENTS"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    assert payload["items"][0]["recommended_use"] == "WIZARD EDIT 456"


@pytest.mark.parametrize("width", [1360, 430])
@pytest.mark.parametrize("surface", ["settings", "onboarding"])
def test_session_access_is_visible_and_preserved_by_the_real_editor(subscription_ui, width, surface):
    ui, page = subscription_ui, subscription_ui["page"]
    page.set_viewport_size({"width": width, "height": 1000})

    def settings_route(route):
        if route.request.method == "POST":
            ui["settings"].update(route.request.post_data_json)
            ui["posts"].append(("/api/settings", route.request.post_data_json))
            route.fulfill(content_type="application/json", body=json.dumps({
                "status": "saved", "saved": True, "restart_required": False}))
        else:
            route.fulfill(content_type="application/json", body=json.dumps(ui["settings"]))

    if surface == "settings":
        page.route("**/api/settings", settings_route)
        page.goto(ui["url"] + "/#settings")
        page.locator('[data-settings-tab="agents"]').click()
    else:
        page.goto(ui["url"] + "/onboarding")
        page.wait_for_selector("#quick-start-btn:not([hidden])")
        page.click("#next-btn")
        page.locator("details:has(#onboarding-available-subagents) > summary").click()
    access = page.locator('[data-subagent-field="access"]').first
    assert access.input_value() == "full"
    assert "Full system access" in access.inner_text()
    access.select_option("workspace_write")
    assert access.input_value() == "workspace_write"
    access.scroll_into_view_if_needed()
    capture(page, f"access-{surface}-{width}")
    if surface == "settings":
        with page.expect_response("**/api/settings"):
            page.locator("#btn-save-settings").click()
        page.reload()
        page.locator('[data-settings-tab="agents"]').click()
        assert page.locator('[data-subagent-field="access"]').first.input_value() == "workspace_write"
        path = "/api/settings"
    else:
        for selector in ("[data-reviewers-note]", '[data-collapse="api-budget"]', ".summary-card"):
            page.click("#next-btn")
            page.wait_for_selector(selector, state="attached")
        with page.expect_response("**/api/onboarding/complete"):
            page.click("#next-btn")
        path = "/api/onboarding/complete"
    payload = next(body for endpoint, body in reversed(ui["posts"]) if endpoint == path)["OUROBOROS_SUBAGENTS"]
    payload = json.loads(payload) if isinstance(payload, str) else payload
    sessions = [row for row in payload["items"] if row["route"]["kind"] == "agent_session"]
    assert sessions[0]["access"] == "workspace_write"
