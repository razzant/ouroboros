"""Browser regressions for generated subagent preview identity preservation."""
from __future__ import annotations

import copy
import json

import pytest

from tests.settings_browser_helpers import open_entry, open_settings, press_editor_save
from tests.test_subscription_setup_browser import subscription_ui as subscription_ui
from tests.test_subscription_setup_browser import capture

pytestmark = [pytest.mark.ui_browser]


def _watch_previews(page):
    # The SPA records about 250 resources before its first preview (191 modules at 7.8.1), and a full Resource
    # Timing buffer (250 entries by default) drops every later entry, the previews included.
    page.add_init_script("performance.setResourceTimingBufferSize(5000)")
    page.evaluate("performance.clearResourceTimings()")


def _wait_for_two_previews(page):
    page.wait_for_function("""
        () => performance.getEntriesByType('resource')
            .filter(entry => entry.name.includes('/api/onboarding/subagents/preview')).length >= 2
    """)


def test_settings_first_press_on_a_generated_list_writes_the_shown_list_whole(subscription_ui):
    """A roster that was never saved shows the generated candidate, enriched by the preview; the first switch
    press authors it: the whole shown list goes through the owner transaction with that one entry changed."""
    ui, page = subscription_ui, subscription_ui["page"]
    roster = copy.deepcopy(ui["fixture"]["preview"]["available_subagents"])
    ui["settings"]["OUROBOROS_SUBAGENTS"] = ""
    ui["settings"]["_meta"]["available_subagents"] = {"source": "undecided", "candidate": roster}

    def checker(route):
        body = route.request.post_data_json
        ui["posts"].append(("/api/owner/safety-model", body))
        ui["settings"]["OUROBOROS_SUBAGENTS"] = json.dumps(body["settings"]["OUROBOROS_SUBAGENTS"])
        ui["settings"]["_meta"]["available_subagents"] = {"source": "configured", "candidate": None}
        route.fulfill(content_type="application/json", body=json.dumps({"ok": True}))

    page.route("**/api/owner/safety-model", checker)
    _watch_previews(page)
    page.goto(ui["url"] + "/#settings")
    with page.expect_response("**/api/onboarding/subagents/preview"):
        level = open_settings(page, "agents")
    head = level.locator("[data-entry-id]").first
    head.wait_for(state="visible")
    agent = head.get_attribute("data-entry-id")
    assert not [body for path, body in ui["posts"] if path in {"/api/settings", "/api/owner/safety-model"}], (
        "showing a generated list writes nothing")
    head.locator(".ui-switch").click()
    page.wait_for_function("""id => { const input = document.querySelector(`#page-settings [data-entry-id="${id}"] [data-entry-switch]`);
        return input && !input.checked && input.getAttribute('aria-busy') !== 'true'; }""", arg=agent)
    writes = [body for path, body in ui["posts"] if path == "/api/owner/safety-model"]
    assert len(writes) == 1 and not [body for path, body in ui["posts"] if path == "/api/settings"]
    payload = writes[0]["settings"]["OUROBOROS_SUBAGENTS"]
    assert [row["subagent_id"] for row in payload["items"]] == [row["subagent_id"] for row in roster["items"]]
    assert [row.get("enabled", True) for row in payload["items"]] == [False] + [True] * (len(roster["items"]) - 1)


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

    if surface == "settings":
        page.goto(ui["url"] + "/#settings")
        open_settings(page, "agents")
        open_entry(page, "codex")
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
        assert press_editor_save(page).startswith("Saved")
        ui["errors"].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
        page.reload()
        open_settings(page, "agents")
        open_entry(page, "codex")
        assert page.locator('[data-subagent-field="access"]').first.input_value() == "workspace_write"
        path = "/api/owner/safety-model"
    else:
        for selector in ("[data-reviewers-note]", '[data-collapse="api-budget"]', ".summary-card"):
            page.click("#next-btn")
            page.wait_for_selector(selector, state="attached")
        with page.expect_response("**/api/onboarding/complete"):
            page.click("#next-btn")
        path = "/api/onboarding/complete"
    written = next(body for endpoint, body in reversed(ui["posts"]) if endpoint == path)
    payload = written.get("settings", written)["OUROBOROS_SUBAGENTS"]
    payload = json.loads(payload) if isinstance(payload, str) else payload
    sessions = [row for row in payload["items"] if row["route"]["kind"] == "agent_session"]
    assert sessions[0]["access"] == "workspace_write"
