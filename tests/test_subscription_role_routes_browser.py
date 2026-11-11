"""Settings › Agents through controlled APIs; no runtime or login: the list of heads, and an agent's page
with its own Save.

Reviewers are catalog rows marked Reviewer, so every reviewer is edited on its agent's page."""
from __future__ import annotations

import json

import pytest
from tests import test_subscription_setup_browser as setup_browser
from tests.settings_browser_helpers import open_entry, open_settings, press_editor_save, settings_page

pytestmark = [pytest.mark.ui_browser]
subscription_ui = setup_browser.subscription_ui
capture = setup_browser.capture
def settings_saves(ui):
    """Each entity's write, including a Save carried only by the checker endpoint."""
    return [body.get('settings', {}) if path == '/api/owner/safety-model' else body
            for path, body in ui['posts'] if path in ('/api/settings', '/api/owner/safety-model')]
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


def open_models(ui, slot='main'):
    page = ui['page']
    page.goto(ui['url'] + '/#settings')
    open_settings(page, 'models/role/' + slot)
    page.locator('[data-model-role-model]').wait_for(state='visible')
    return page


def open_agents(ui):
    """Settings › Agents: the list, once its heads are drawn."""
    page = ui["page"]
    page.goto(ui["url"] + "/#settings")
    open_settings(page, "agents").locator("[data-entry-id]").first.wait_for(state="visible")
    return page


def open_agent(ui, agent_id, *, fresh=True):
    """One agent's page, opened from the list; returns the page's form (its `[data-subagent-row]`)."""
    page = open_agents(ui) if fresh else ui["page"]
    if not fresh:
        open_settings(page, "agents").locator("[data-entry-id]").first.wait_for(state="visible")
    return open_entry(page, agent_id).locator("[data-subagent-row]")


def roster_writes(ui):
    """Every roster the owner transaction wrote, in order."""
    return [body["settings"]["OUROBOROS_SUBAGENTS"] for path, body in ui["posts"]
            if path == "/api/owner/safety-model" and "OUROBOROS_SUBAGENTS" in (body.get("settings") or {})]


def saved_catalog(ui):
    actors = roster_writes(ui)[-1]
    return json.loads(actors) if isinstance(actors, str) else actors


def wait_for_switch(page, agent_id, on):
    """An agent's switch on the list shows `on` and its write has settled."""
    page.wait_for_function("""([id, on]) => {
        const input = document.querySelector(`#page-settings [data-entry-id="${id}"] [data-entry-switch]`);
        return input && input.checked === on && input.getAttribute('aria-busy') !== 'true'; }""", arg=[agent_id, on])


@pytest.mark.parametrize("width", [1360, 390])
def test_subscription_accounts_roundtrip_existing_editors(role_ui, width):
    ui = role_ui
    configure_mixed(ui)
    api_lane(ui, "openai")
    ui["page"].set_viewport_size({"width": width, "height": 900})
    page = open_agents(ui)
    level = settings_page(page, "agents")
    assert [row.get_attribute("data-entry-id") for row in level.locator("[data-entry-id]").all()] == ["native", "direct", "agent"]
    assert [chip.inner_text().strip() for chip in level.locator("[data-agents-filter]").all()][:2] == ["All 3", "Reviewers 2"]
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    capture(page, f"role-reviewers-{width}")

    # A subscription model on its agent's page: source, model and the pinned account, then its own Save.
    raw = open_agent(ui, "native", fresh=False)
    page.wait_for_function("""() => document.querySelector('[data-subagent-field="route"]')
        ?.querySelector('option[value="subscription:opaque-source"]')""")
    assert raw.locator('[data-subagent-field="route"]').input_value() == "subscription:opaque-source"
    assert raw.locator('[data-subagent-field="model"]').input_value() == "gpt-test"
    assert raw.locator('[data-subagent-field="account"]').input_value() == "personal"
    assert raw.locator('[data-subagent-field="review_eligible"]').is_checked()
    raw.locator('[data-subagent-field="model"]').fill("gpt-next")
    raw.locator('[data-subagent-field="account"]').select_option("work")
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    capture(page, f"role-actors-{width}")
    assert press_editor_save(page).startswith("Saved")
    assert len(roster_writes(ui)) == 1 and not [body for path, body in ui["posts"] if path == "/api/settings"]

    # A direct API row has no account; a session keeps its pin, and its Reviewer mark writes at once.
    direct = open_agent(ui, "direct", fresh=False)
    assert direct.locator('[data-subagent-field="account"]').count() == 0
    assert direct.locator('[data-subagent-field="delivery"]').input_value() == "packet"
    agent = open_agent(ui, "agent", fresh=False)
    assert agent.locator('[data-subagent-field="account"]').input_value() == "personal"
    mark = agent.locator('[data-subagent-field="review_eligible"]')
    assert not mark.is_checked()
    agent.locator('.agent-use:has([data-subagent-field="review_eligible"]) .ui-switch').click()
    page.wait_for_function("""() => { const input = document.querySelector('#page-settings [data-subagent-field="review_eligible"]');
        return input && input.checked && input.getAttribute('aria-busy') !== 'true'; }""")
    assert len(roster_writes(ui)) == 2
    items = saved_catalog(ui)["items"]
    assert items[0]["route"] == {
        "kind": "api_model", "target_id": "claudexor::opaque-source=gpt-next", "credential_profile_id": "work"}
    assert items[1]["route"] == {"kind": "api_model", "target_id": "openai::gpt-api"}
    assert items[1]["delivery"] == "packet"
    assert items[2]["route"]["kind"] == "agent_session"
    assert [row.get("review_eligible") for row in items] == [True, True, True]
    assert all("OUROBOROS_REVIEWER_SLOTS" not in (body.get("settings") or body) for _path, body in ui["posts"]), (
        "review lanes are never authored")
    raw = open_agent(ui, "native")
    assert raw.locator('[data-subagent-field="account"]').input_value() == "work"
    assert open_agent(ui, "agent", fresh=False).locator('[data-subagent-field="review_eligible"]').is_checked()
    assert not any("login" in path for path, _ in ui["posts"])


def test_source_switch_and_catalog_refresh_keep_focus_and_draft(role_ui):
    ui = role_ui
    configure_mixed(ui)
    lane = api_lane(ui, 'openai')
    row = open_agent(ui, "native")
    page = ui["page"]
    page.wait_for_function("""() => document.querySelector('[data-subagent-field="route"]')
        ?.querySelector('option[value="subscription:opaque-source"]')""")
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
    scroller = settings_page(page, "agents/native")
    before = scroller.evaluate("e => e.scrollTop")
    page.evaluate("""detail => document.dispatchEvent(new CustomEvent(
        'settings-model-catalog:updated', {detail}))""", ui["fixture"]["catalog"])
    assert page.evaluate("document.activeElement.getAttribute('data-subagent-field') === 'model'")
    assert field.input_value() == "gpt-manual"
    assert field.evaluate("e => e.selectionStart") == 3
    assert scroller.evaluate("e => e.scrollTop") == before
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
        "row_costs": {"direct": {"usd_per_review": 0.42, "basis": "observed_reviews", "reviews": 2, "low": 0.4, "high": 0.44}},
    }
    native = open_agent(ui, "native")
    page = ui["page"]
    page.wait_for_function("() => document.querySelector('[data-subagent-last-review]:not([hidden])')")
    # Cost and history are an agent's facts behind Details & history; a marked API row also prices its delivery.
    assert native.locator("[data-subagent-review-facts]").text_content() == "uses a session seat and time"
    assert native.locator("[data-subagent-last-review] dd").text_content() == (
        "API model · gpt-test · account personal · record rev_42")
    agent = open_agent(ui, "agent", fresh=False)
    assert agent.locator("[data-subagent-field=\"delivery\"]").count() == 0, "delivery is for API reviewers"
    direct = open_agent(ui, "direct", fresh=False)
    page.wait_for_function("() => /0\\.42/.test(document.querySelector('[data-subagent-review-facts]')?.textContent || '')")
    facts = direct.locator("[data-subagent-review-facts]").text_content()
    assert facts == direct.locator("[data-subagent-delivery-cost]").inner_text() == (
        "≈$0.42 per review (median of 2 measured, $0.40–$0.44)")
    delivery = direct.locator('[data-subagent-field="delivery"]')
    assert delivery.input_value() == "packet"
    direct.scroll_into_view_if_needed()
    capture(page, "role-reviewer-facts")
    delivery.select_option("native")
    # An edited route is priced only after it is saved: never its old price.
    direct.locator('[data-subagent-field="model"]').fill("gpt-api-next")
    assert "≈$0.42" not in direct.locator("[data-subagent-review-facts]").text_content()
    assert press_editor_save(page).startswith("Saved")
    items = saved_catalog(ui)["items"]
    assert "delivery" not in items[1] and items[1]["review_eligible"] is True


def test_models_does_not_guess_a_credential_family_when_source_mapping_is_unread(role_ui):
    ui = role_ui
    configure_mixed(ui)
    ui["settings"]["OUROBOROS_MODEL_ACCOUNTS"] = {"main": "personal"}
    ui["fixture"]["catalog"]["model_sources"] = []
    page = ui["page"]
    page.goto(ui["url"] + "/#settings")
    open_settings(page, 'models/role/main')
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
    press_editor_save(page)
    saved = ui["settings"]["OUROBOROS_MODEL_ACCOUNTS"]
    if isinstance(saved, str):
        saved = json.loads(saved)
    assert saved["main"] == "personal"


def test_catalog_failure_recovery_and_empty_read_keep_an_agents_form_nodes_and_draft(role_ui):
    """An agent's page while the model catalogue fails, recovers and reads empty: the form keeps its nodes, the
    caret and the unsaved model; the failure is said on the page, and the page's own Save writes the draft."""
    ui = role_ui
    configure_mixed(ui)
    row = open_agent(ui, "native")
    page = ui["page"]
    page.wait_for_function("""() => document.querySelector('[data-subagent-field="route"]')
        ?.querySelector('option[value="subscription:opaque-source"]')""")
    field = row.locator('[data-subagent-field="model"]')
    source, account = row.locator('[data-subagent-field="route"]'), row.locator('[data-subagent-field="account"]')
    field.fill("gpt-owner-unsaved")
    field.evaluate("e => { window.catalogDraftNode = e; e.setSelectionRange(4, 9); }")
    source.evaluate("e => { window.catalogSourceNode = e; }")
    account.evaluate("e => { window.catalogAccountNode = e; }")
    response = {"status": 503, "body": {"error": "catalog temporarily offline"}}
    page.route("**/api/model-catalog*", lambda route: route.fulfill(
        status=response["status"], content_type="application/json", body=json.dumps(response["body"])))
    problem = settings_page(page, "agents/native").locator("[data-subagents-read-problem]")
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
        assert source.evaluate("e => e === window.catalogSourceNode") and source.input_value() == "subscription:opaque-source"
        assert account.evaluate("e => e === window.catalogAccountNode") and account.input_value() == "personal"
        assert ("catalog temporarily offline" in problem.text_content()) is (status == 503)
    capture(page, "catalog-recovery-draft-actor")
    # Not this page's: the Circles page, mounted with the app, re-reads the catalogue on every broadcast and rejects
    # a failed read with nothing listed (circles.js readCatalog, from its listener), unhandled. Anything else counts.
    refused = "The model catalog could not be read."
    assert not [error for error in ui["errors"].unexpected() if refused not in error], ui["errors"].report()
    ui["errors"].clear()
    assert press_editor_save(page).startswith("Saved")
    saved = saved_catalog(ui)["items"][0]
    assert saved["route"] == {"kind": "api_model", "target_id": "claudexor::opaque-source=gpt-owner-unsaved",
                              "credential_profile_id": "personal"}
    assert saved["review_eligible"] is True


def test_catalog_failure_recovery_and_empty_read_keep_real_editor_nodes_and_draft(role_ui):
    from playwright.sync_api import expect
    ui = role_ui
    configure_mixed(ui)
    ui["settings"]["OUROBOROS_MODEL"] = "claudexor::opaque-source=gpt-test"
    ui["settings"]["OUROBOROS_MODEL_ACCOUNTS"] = {"main": "personal"}
    page = open_models(ui)
    row_selector, source_selector, model_selector, account_selector = (
        '[data-model-role="main"]', '[data-model-role-source]', '[data-model-role-model]', '[data-model-role-account]')
    row = page.locator(row_selector).first
    field = row.locator(model_selector)
    field.fill("gpt-owner-unsaved")
    field.evaluate("e => { window.catalogDraftNode = e; e.setSelectionRange(4, 9); }")
    row.locator(source_selector).evaluate("e => { window.catalogSourceNode = e; }")
    row.locator(account_selector).evaluate("e => { window.catalogAccountNode = e; }")
    response = {"status": 503, "body": {"error": "catalog temporarily offline"}}
    page.route("**/api/model-catalog?*", lambda route: route.fulfill(
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
            expect(row.locator("[data-model-role-status]")).to_contain_text("catalog temporarily offline")
        else:
            expect(row.locator("[data-model-role-status]")).not_to_contain_text("catalog temporarily offline")
    capture(page, "catalog-recovery-draft-models")
    page.locator('[data-model-role-model]').press('Escape')
    press_editor_save(page)
    saved = settings_saves(ui)[-1]
    assert saved["OUROBOROS_MODEL"] == "claudexor::opaque-source=gpt-owner-unsaved"
    assert ui["settings"]["OUROBOROS_MODEL_ACCOUNTS"]["main"] == "personal"


def test_image_generation_has_no_settings_default(role_ui):
    """Real Settings render/save keeps generation separate from chat model roles."""
    ui = role_ui
    configure_mixed(ui)
    lane = api_lane(ui)
    page = open_models(ui, 'vision')
    assert page.locator('[data-model-role="image"]').count() == 0
    assert page.locator('[data-model-role="vision"]').count() == 1
    capture(page, "image-settings-model-selected-desktop")
    page.set_viewport_size({"width": 320, "height": 640})
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    capture(page, "image-settings-model-selected-narrow")
    page.locator('[data-model-role-source]').select_option(lane)
    page.locator('[data-model-role-model]').fill('owner-vision')
    page.locator('[data-model-role-model]').press('Escape')
    press_editor_save(page)
    saved = settings_saves(ui)[-1]
    assert saved['OUROBOROS_MODEL_VISION'] == 'owner-vision'
    assert 'OUROBOROS_MODEL_IMAGE' not in saved
