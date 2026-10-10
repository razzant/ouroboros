"""Existing Agents save/reload through controlled APIs; no runtime or login.

Reviewers are catalog rows marked Reviewer, so every reviewer is edited where
its Available-subagents row is."""
from __future__ import annotations

import json

import pytest
from tests import test_subscription_setup_browser as setup_browser

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
subscription_ui = setup_browser.subscription_ui
capture = setup_browser.capture
# Every route picker offers an API lane per provider whose credential is
# stored, so a fixture that selects one advertises that provider's key first.
API_KEYS = {"openrouter": "OPENROUTER_API_KEY", "openai": "OPENAI_API_KEY"}


def api_lane(ui, provider="openrouter"):
    """Store `provider`'s key in the served settings and return its route choice."""
    ui["settings"][API_KEYS[provider]] = "***set***"
    return f"api:{provider}"


@pytest.fixture
def role_ui(subscription_ui):
    ui = subscription_ui

    def settings_route(route):
        if route.request.method == "POST":
            payload = route.request.post_data_json
            ui["posts"].append(("/api/settings", payload))
            ui["settings"].update(payload)
            result = {"status": "saved", "saved": True, "restart_required": False}
        else:
            result = ui["settings"]
        route.fulfill(content_type="application/json", body=json.dumps(result))

    ui["page"].route("**/api/settings", settings_route)
    ui["page"].route("**/api/owner/runtime-mode", lambda route: route.fulfill(
        content_type="application/json", body=json.dumps({"ok": True, "saved": True,
            "runtime_mode": "advanced", "restart_required": False})))
    return ui


def configure_mixed(ui):
    target = "claudexor::opaque-source=gpt-test"
    actors = {"enabled": True, "items": [
        {"subagent_id": "native", "recommended_use": "Inspect the project.", "review_eligible": True,
         "route": {"kind": "api_model", "target_id": target, "credential_profile_id": "personal"}},
        {"subagent_id": "direct", "recommended_use": "Direct API model.", "review_eligible": True,
         "delivery": "packet", "route": {"kind": "api_model", "target_id": "openai::gpt-api"}},
        {"subagent_id": "agent", "recommended_use": "Existing agent session.",
         "route": {"kind": "agent_session", "target_id": "codex=gpt-test", "credential_profile_id": "personal"}},
    ]}
    ui["settings"].update(OUROBOROS_SUBAGENTS=actors)
    ui["fixture"]["catalog"]["model_sources"] = [
        {"id": "opaque-source", "label": "Codex", "credentialHarness": "codex"},
    ]
    ui["fixture"]["catalog"]["items"] = [
        {"value": target, "id": "gpt-test", "source_id": "opaque-source"},
        {"value": "openai::gpt-api", "id": "gpt-api"},
    ]


def open_agents(ui):
    page = ui["page"]
    page.goto(ui["url"] + "/#settings")
    page.locator('[data-settings-tab="agents"]').click()
    page.wait_for_selector('[data-subagent-field="account"]')
    page.wait_for_function("""() => document.querySelector('[data-subagent-field="route"]')
        ?.querySelector('option[value="subscription:opaque-source"]')""")
    return page


def saved_catalog(ui):
    writes = [body for path, body in ui["posts"] if path == "/api/settings"]
    actors = writes[-1]["OUROBOROS_SUBAGENTS"]
    return json.loads(actors) if isinstance(actors, str) else actors


@pytest.mark.parametrize("width", [1360, 390])
def test_subscription_accounts_roundtrip_existing_editors(role_ui, width):
    ui = role_ui
    configure_mixed(ui)
    api_lane(ui, "openai")
    ui["page"].set_viewport_size({"width": width, "height": 900})
    page = open_agents(ui)
    actors = page.locator("[data-subagent-row]")
    assert actors.count() == 3
    raw, direct, agent = actors.nth(0), actors.nth(1), actors.nth(2)
    assert raw.locator('[data-subagent-field="route"]').input_value() == "subscription:opaque-source"
    assert raw.locator('[data-subagent-field="model"]').input_value() == "gpt-test"
    assert raw.locator('[data-subagent-field="account"]').input_value() == "personal"
    assert direct.locator('[data-subagent-field="account"]').count() == 0
    assert agent.locator('[data-subagent-field="account"]').input_value() == "personal"
    assert [actors.nth(i).locator('[data-subagent-field="review_eligible"]').is_checked() for i in range(3)] == [
        True, True, False]
    assert page.locator("[data-review-pool-count]").inner_text() == "Reviewers: 2"
    raw.locator('[data-subagent-field="model"]').fill("gpt-next")
    raw.locator('[data-subagent-field="account"]').select_option("work")
    agent.locator('[data-subagent-field="review_eligible"]').check()
    assert page.locator("[data-review-pool-count]").inner_text() == "Reviewers: 3"
    direct.scroll_into_view_if_needed()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    controls = direct.locator(".available-subagent-actions, [data-subagent-delivery-field]")
    boxes = controls.locator("select, input, label").evaluate_all(
        "els => els.filter(e => !e.hidden).map(e => {const b=e.getBoundingClientRect();return {x:b.x,y:b.y,w:b.width,h:b.height}})")
    assert boxes and all(b["x"] >= 0 and b["x"] + b["w"] <= width for b in boxes)
    capture(page, f"role-reviewers-{width}")
    raw.scroll_into_view_if_needed()
    capture(page, f"role-actors-{width}")
    page.locator("#btn-save-settings").click()
    page.wait_for_function("() => !document.querySelector('#btn-save-settings').disabled")
    writes = [body for path, body in ui["posts"] if path == "/api/settings"]
    assert len(writes) == 1
    assert "OUROBOROS_REVIEWER_SLOTS" not in writes[0], "review lanes are never authored"
    items = saved_catalog(ui)["items"]
    assert items[0]["route"] == {
        "kind": "api_model", "target_id": "claudexor::opaque-source=gpt-next", "credential_profile_id": "work"}
    assert items[1]["route"] == {"kind": "api_model", "target_id": "openai::gpt-api"}
    assert items[1]["delivery"] == "packet"
    assert items[2]["route"]["kind"] == "agent_session"
    assert [row.get("review_eligible") for row in items] == [True, True, True]
    open_agents(ui)
    assert page.locator('[data-subagent-row]').nth(0).locator('[data-subagent-field="account"]').input_value() == "work"
    assert page.locator('[data-subagent-row]').nth(2).locator('[data-subagent-field="review_eligible"]').is_checked()
    assert not any("login" in path for path, _ in ui["posts"])


def test_source_switch_and_catalog_refresh_keep_focus_and_draft(role_ui):
    ui = role_ui
    configure_mixed(ui)
    lane = api_lane(ui, 'openai')
    page = open_agents(ui)
    row = page.locator('[data-subagent-row]').first
    row.locator('[data-subagent-field="route"]').select_option(lane)
    assert row.locator('[data-subagent-field="account"]').count() == 0
    # A source change clears only source-bound fields: the Reviewer mark stays.
    assert row.locator('[data-subagent-field="review_eligible"]').is_checked()
    row.locator('[data-subagent-field="route"]').select_option("subscription:opaque-source")
    assert row.locator('[data-subagent-field="account"]').count() == 1
    field = row.locator('[data-subagent-field="model"]')
    field.fill("gpt-manual")
    row.scroll_into_view_if_needed()
    field.focus()
    field.evaluate("e => e.setSelectionRange(3, 3)")
    before = page.locator("#content").evaluate("e => e.scrollTop")
    page.evaluate("""detail => document.dispatchEvent(new CustomEvent(
        'settings-model-catalog:updated', {detail}))""", ui["fixture"]["catalog"])
    assert page.evaluate("document.activeElement.getAttribute('data-subagent-field') === 'model'")
    assert field.input_value() == "gpt-manual"
    assert field.evaluate("e => e.selectionStart") == 3
    assert page.locator("#content").evaluate("e => e.scrollTop") == before
    capture(page, "role-source-refresh-focus")


def test_a_reviewer_row_names_its_delivery_cost_and_last_run(role_ui):
    ui = role_ui
    configure_mixed(ui)
    api_lane(ui, "openai")
    ui["backend"]["review_pool"] = {
        "pool": [{"subagent_id": "native", "last_execution": {
            "effective": {"route": "api_model", "model": "gpt-test", "credential_profile_id": "personal"},
            "review_record_id": "rev_42"}}],
        "excluded": [], "last_executions": {}, "config_error": "", "migration": None,
        "row_costs": {"direct": {"usd_per_review": 0.42, "basis": "route_tariff"}},
    }
    page = open_agents(ui)
    rows = page.locator("[data-subagent-row]")
    native, direct, agent = rows.nth(0), rows.nth(1), rows.nth(2)
    page.wait_for_function("() => document.querySelector('[data-subagent-last-review]:not([hidden])')")
    # Cost and history are row facts behind Details & history; a marked API row also prices its delivery.
    assert native.locator("[data-subagent-review-facts]").text_content() == "uses a session seat and time"
    assert native.locator("[data-subagent-last-review] dd").text_content() == (
        "API model · gpt-test · account personal · record rev_42")
    facts = direct.locator("[data-subagent-review-facts]").text_content()
    assert facts == direct.locator("[data-subagent-delivery-cost]").inner_text() == "≈$0.42 per full call (route tariff)"
    assert "reading reviewer" not in facts, "the several-calls clause belongs to a reading row"
    assert agent.locator("[data-subagent-field=\"delivery\"]").count() == 0, "delivery is for API reviewers"
    delivery = direct.locator('[data-subagent-field="delivery"]')
    assert delivery.input_value() == "packet"
    direct.scroll_into_view_if_needed()
    capture(page, "role-reviewer-facts")
    delivery.select_option("native")
    # An edited route is priced only after it is saved: never its old price.
    direct.locator('[data-subagent-field="model"]').fill("gpt-api-next")
    assert "≈$0.42" not in direct.locator("[data-subagent-review-facts]").text_content()
    with page.expect_response("**/api/settings"):
        page.locator("#btn-save-settings").click()
    items = saved_catalog(ui)["items"]
    assert "delivery" not in items[1] and items[1]["review_eligible"] is True


def test_models_does_not_guess_a_credential_family_when_source_mapping_is_unread(role_ui):
    ui = role_ui
    configure_mixed(ui)
    ui["settings"]["OUROBOROS_MODEL_ACCOUNTS"] = {"main": "personal"}
    ui["fixture"]["catalog"]["model_sources"] = []
    page = ui["page"]
    page.goto(ui["url"] + "/#settings")
    page.locator('[data-settings-tab="models"]').click()
    main = page.locator('[data-model-role="main"]')
    account = main.locator('[data-model-role-account]')
    page.wait_for_function("() => document.querySelector('[data-model-role=main] [data-model-role-account]')?.value === 'personal'")
    assert "not checked" in account.locator('option[value="personal"]').inner_text()
    assert account.locator('option[value="work"]').count() == 0
    page.evaluate("""detail => document.dispatchEvent(new CustomEvent(
        'settings-model-catalog:updated', {detail}))""", ui["fixture"]["catalog"])
    assert account.input_value() == "personal"
    assert account.locator('option[value="work"]').count() == 0
    capture(page, "role-models-unknown-mapping")
    page.locator("#btn-save-settings").click()
    page.wait_for_function("() => !document.querySelector('#btn-save-settings').disabled")
    saved = ui["settings"]["OUROBOROS_MODEL_ACCOUNTS"]
    if isinstance(saved, str):
        saved = json.loads(saved)
    assert saved["main"] == "personal"


@pytest.mark.parametrize("consumer", ["Models", "Actor"])
def test_catalog_failure_recovery_and_empty_read_keep_real_editor_nodes_and_draft(role_ui, consumer):
    ui = role_ui
    configure_mixed(ui)
    ui["settings"]["OUROBOROS_MODEL"] = "claudexor::opaque-source=gpt-test"
    ui["settings"]["OUROBOROS_MODEL_ACCOUNTS"] = {"main": "personal"}
    page = open_agents(ui)
    selectors = {
        "Models": ('[data-model-role="main"]', '[data-model-role-source]', '[data-model-role-model]', '[data-model-role-account]'),
        "Actor": ('[data-subagent-row]', '[data-subagent-field="route"]', '[data-subagent-field="model"]', '[data-subagent-field="account"]'),
    }
    if consumer == "Models":
        page.locator('[data-settings-tab="models"]').click()
    row_selector, source_selector, model_selector, account_selector = selectors[consumer]
    row = page.locator(row_selector).first
    field = row.locator(model_selector)
    field.fill("gpt-owner-unsaved")
    field.evaluate("e => { window.catalogDraftNode = e; e.setSelectionRange(4, 9); }")
    row.locator(source_selector).evaluate("e => { window.catalogSourceNode = e; }")
    row.locator(account_selector).evaluate("e => { window.catalogAccountNode = e; }")
    response = {"status": 503, "body": {"error": "catalog temporarily offline"}}
    page.route("**/api/model-catalog*", lambda route: route.fulfill(
        status=response["status"], content_type="application/json", body=json.dumps(response["body"])))
    for status, body in [
        (503, {"error": "catalog temporarily offline"}),
        (200, ui["fixture"]["catalog"]),
        (200, {"items": [], "model_sources": [], "errors": []}),
        (503, {"error": "catalog temporarily offline"}),
    ]:
        response.update(status=status, body=body)
        page.evaluate("async () => (await import('/static/modules/settings_catalog.js')).refreshModelCatalog()")
        assert field.evaluate("e => e === window.catalogDraftNode && document.activeElement === e")
        assert field.evaluate("e => [e.selectionStart, e.selectionEnd]") == [4, 9]
        assert field.input_value() == "gpt-owner-unsaved"
        assert row.locator(source_selector).evaluate("e => e === window.catalogSourceNode")
        assert row.locator(source_selector).input_value() == "subscription:opaque-source"
        assert row.locator(account_selector).evaluate("e => e === window.catalogAccountNode")
        assert row.locator(account_selector).input_value() == "personal"
        if status == 503:
            assert "catalog temporarily offline" in page.locator("#settings-model-catalog-status").text_content()
        else:
            assert "catalog temporarily offline" not in page.locator("#settings-model-catalog-status").text_content()
    capture(page, f"catalog-recovery-draft-{consumer.lower()}")
    with page.expect_response("**/api/settings"):
        page.locator("#btn-save-settings").click()
    saved = [body for path, body in ui["posts"] if path == "/api/settings"][-1]
    if consumer == "Models":
        assert saved["OUROBOROS_MODEL"] == "claudexor::opaque-source=gpt-owner-unsaved"
        assert saved["OUROBOROS_MODEL_ACCOUNTS"]["main"] == "personal"
    else:
        assert saved["OUROBOROS_SUBAGENTS"]["items"][0]["route"]["target_id"] == "claudexor::opaque-source=gpt-owner-unsaved"
        assert saved["OUROBOROS_SUBAGENTS"]["items"][0]["route"]["credential_profile_id"] == "personal"
        assert saved["OUROBOROS_SUBAGENTS"]["items"][0]["review_eligible"] is True
