"""Wizard navigation and local-action feedback through the production UI.

Reuse the existing static-server/API-boundary fixture: no installed runtime,
account, provider, or settings write is involved in these interaction checks.
"""
from __future__ import annotations

import json

import pytest

from tests.test_subscription_setup_browser import (
    capture,
    subscription_ui as subscription_ui,
)

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def _open_wizard(ui):
    page = ui["page"]
    page.goto(ui["url"] + "/onboarding")
    page.wait_for_selector("#quick-start-btn:not([hidden])")
    assert page.locator(".wizard-step").count() == 5
    return page


@pytest.mark.parametrize("viewport", [(390, 600), (1360, 900)])
def test_wizard_navigation_restores_draft_and_back_position(subscription_ui, viewport):
    ui, page = subscription_ui, subscription_ui["page"]
    page.set_viewport_size(dict(zip(("width", "height"), viewport)))
    _open_wizard(ui)
    page.click("#next-btn")
    page.wait_for_selector('[data-model-role="main"]')
    assert page.evaluate("document.activeElement.matches('.step-title')")
    assert page.evaluate("window.scrollY") == 0
    capture(page, f"wizard-models-arrival-{viewport[0]}")

    main = page.locator('[data-model-role="main"] [data-model-role-model]')
    main.fill("owner-kept-model")
    page.locator('[data-collapse="subagents"] > summary').click()
    page.wait_for_selector("#onboarding-available-subagents .available-subagent-row")
    page.locator("#next-btn").scroll_into_view_if_needed()
    previous_position = page.evaluate("window.scrollY")
    page.click("#next-btn")
    page.wait_for_selector("[data-reviewers-note]", state="attached")
    assert page.evaluate("document.activeElement.matches('.step-title')")
    assert page.evaluate("window.scrollY") == 0

    page.evaluate("window.heldReviewers = document.querySelector('[data-reviewers-note]')")
    # Test keyboard focus retention explicitly: WebKit's native pointer click
    # intentionally does not focus a button as Chromium's does.
    page.locator('[data-review-mode="blocking"]').focus()
    page.locator('[data-review-mode="blocking"]').press("Enter")
    assert page.evaluate("heldReviewers === document.querySelector('[data-reviewers-note]')")
    assert page.locator('[data-review-mode="blocking"]').get_attribute("aria-pressed") == "true"
    assert page.locator('[data-review-mode="advisory"]').get_attribute("aria-pressed") == "false"
    assert page.evaluate("document.activeElement.matches('[data-review-mode=blocking]')")

    page.click("#back-btn")
    page.wait_for_selector('[data-model-role="main"]')
    assert main.input_value() == "owner-kept-model"
    assert page.locator('[data-collapse="subagents"]').evaluate("el => el.open")
    assert abs(page.evaluate("window.scrollY") - previous_position) <= 1
    assert page.evaluate("document.activeElement.matches('.step-title')")
    capture(page, f"wizard-models-back-{viewport[0]}")
    assert not any(path == "/api/onboarding/complete" for path, _ in ui["posts"])


def test_wizard_summary_refresh_keeps_focused_action_and_position(subscription_ui):
    ui = subscription_ui
    page = _open_wizard(ui)
    page.set_viewport_size({"width": 390, "height": 600})
    page.click("#quick-start-btn")
    page.wait_for_selector(".summary-card")
    assert page.evaluate("document.activeElement.matches('.step-title') && window.scrollY === 0")
    page.locator("#next-btn").focus()
    page.evaluate("window.heldFinish = document.activeElement; window.heldPosition = window.scrollY")

    # A changed account/model discovery causes a real preview and catalog
    # refresh. The summary changes, while its focused Start action stays put.
    ui["fixture"]["status"]["harnesses"][0]["models"].append({"id": "new-discovery"})
    ui["fixture"]["preview"]["model_settings"]["OUROBOROS_MODEL"] = "claudexor::codex=updated-preview"
    page.evaluate("""async () => {
        const {claudexorStatus} = await import('/static/modules/claudexor_status_store.js');
        await claudexorStatus.refresh({includeModels: true});
    }""")
    page.wait_for_function("document.querySelector('.summary-card').textContent.includes('updated-preview')")
    assert page.evaluate("heldFinish === document.querySelector('#next-btn') && document.activeElement === heldFinish")
    assert page.evaluate("Math.abs(window.scrollY - heldPosition) <= 1")


def test_wizard_unchanged_preview_preserves_pristine_model_focus(subscription_ui):
    ui = subscription_ui
    page = _open_wizard(ui)
    page.click("#next-btn")
    page.wait_for_selector('[data-model-role="main"] [data-model-role-model]')
    page.locator('[data-model-role="main"] [data-model-role-model]').focus()
    page.evaluate("""() => {
        window.heldModel = document.activeElement;
        heldModel.setSelectionRange(1, 4);
        window.heldPosition = window.scrollY;
    }""")
    ui["fixture"]["status"]["harnesses"][0]["models"].append({"id": "another-discovered-model"})
    with page.expect_response("**/api/model-catalog") as catalog:
        page.evaluate("""async () => {
            const {claudexorStatus} = await import('/static/modules/claudexor_status_store.js');
            await claudexorStatus.refresh({includeModels: true});
        }""")
    catalog.value.finished()
    page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
    assert page.evaluate("document.activeElement === heldModel && heldModel.isConnected")
    assert page.evaluate("heldModel.selectionStart === 1 && heldModel.selectionEnd === 4")
    assert page.evaluate("Math.abs(window.scrollY - heldPosition) <= 1")


def test_wizard_local_stop_failure_preserves_edit_and_retry_feedback(subscription_ui):
    ui, page = subscription_ui, subscription_ui["page"]
    runtime = {"status": "ready", "context_length": 16384}
    stops = []

    def local_controls(route):
        response = route.fetch()
        body = response.text().replace('"supportsLocalRuntimeControls": false',
                                       '"supportsLocalRuntimeControls": true')
        route.fulfill(response=response, body=body)

    def stop(route):
        stops.append(route.request.method)
        if len(stops) == 1:
            route.fulfill(status=500, content_type="application/json",
                          body=json.dumps({"error": "Runtime could not be stopped"}))
        else:
            runtime["status"] = "offline"
            route.fulfill(content_type="application/json", body='{"status":"stopped"}')

    page.route("**/onboarding", local_controls)
    page.route("**/api/local-model/status", lambda route: route.fulfill(
        content_type="application/json", body=json.dumps(runtime)))
    page.route("**/api/local-model/stop", stop)
    _open_wizard(ui)
    page.locator('[data-collapse="api-access"] > summary').click()
    page.locator('[data-collapse="local-model"] > summary').click()
    page.click("#wizard-local-start")
    assert "Enter a local model source" in page.locator(".wizard-error").inner_text()
    source = page.get_by_label("Model Source", exact=True)
    source.fill("/tmp/owner-model.gguf")
    assert page.locator(".wizard-error").inner_text() == ""
    page.evaluate("window.heldLocalSource = document.querySelector('#local-source')")
    page.click("#wizard-local-stop")
    page.wait_for_function("document.querySelector('#wizard-local-test-result').textContent.startsWith('Stop failed:')")
    assert source.input_value() == "/tmp/owner-model.gguf"
    assert page.evaluate("heldLocalSource === document.querySelector('#local-source')")
    assert page.locator("#wizard-local-status").inner_text() == "Status: Ready (ctx: 16384)"
    assert page.locator("#wizard-local-stop").is_enabled()
    assert page.locator("#wizard-local-test-result").is_visible()
    assert page.locator("#wizard-local-test-result").get_attribute("role") == "status"
    capture(page, "wizard-local-stop-failed")

    page.click("#wizard-local-stop")
    page.wait_for_function("document.querySelector('#wizard-local-status').textContent === 'Status: Offline'")
    assert page.locator("#wizard-local-test-result").inner_text() == ""
    assert page.locator("#wizard-local-test-result").is_hidden()
    assert stops == ["POST", "POST"]
    assert source.input_value() == "/tmp/owner-model.gguf"
    capture(page, "wizard-local-stop-retried")


def test_wizard_fields_have_explicit_names_help_and_shared_family(subscription_ui):
    page = _open_wizard(subscription_ui)
    page.locator('[data-collapse="api-access"] > summary').click()
    page.locator('[data-collapse="local-model"] > summary').click()
    page.locator('[data-collapse="more-providers"] > summary').click()
    for field in page.locator(".wizard-content input, .wizard-content select").all():
        assert field.evaluate("el => el.labels.length > 0 || Boolean(el.getAttribute('aria-label'))")
        assert field.evaluate("el => el.classList.contains('ui-control')")
        hint = field.get_attribute("aria-describedby")
        if hint:
            assert page.locator(f"#{hint}").inner_text().strip()
    assert page.get_by_role("button", name="Clear OpenAI API Key", exact=True).count() == 1
    routing = page.get_by_role("group", name="Local routing")
    assert routing.locator('[aria-pressed="true"]').count() == 1
    routing.locator('[data-local-mode="fallback"]').click()
    assert routing.locator('[data-local-mode="fallback"]').get_attribute("aria-pressed") == "true"
    for _ in range(3):
        page.click("#next-btn")
    page.locator('[data-collapse="api-budget"] > summary').click()
    for name in ("Total Budget (USD)", "Per-task Cost Cap (USD)"):
        field = page.get_by_label(name, exact=True)
        assert field.is_visible()
        assert field.get_attribute("type") == "number"
        assert field.evaluate("el => el.classList.contains('ui-control')")
        assert page.locator(f"#{field.get_attribute('aria-describedby')}").inner_text().strip()
