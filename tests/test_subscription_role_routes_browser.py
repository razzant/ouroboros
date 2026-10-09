"""Existing Agents/Reviewers save/reload through controlled APIs; no runtime or login."""
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
            ui["fixture"]["preview"]["reviewer_slots"] = json.loads(payload["OUROBOROS_REVIEWER_SLOTS"])
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
        {"subagent_id": "native", "recommended_use": "Inspect the project.",
         "route": {"kind": "api_model", "target_id": target, "credential_profile_id": "personal"}},
        {"subagent_id": "direct", "recommended_use": "Direct API model.",
         "route": {"kind": "api_model", "target_id": "openai::gpt-api"}},
        {"subagent_id": "agent", "recommended_use": "Existing agent session.",
         "route": {"kind": "agent_session", "target_id": "codex=gpt-test", "credential_profile_id": "personal"}},
    ]}
    slots = {
        "triad": [{"slot_id": "triad_1", "route": {"kind": "api_chat", "target_id": target, "profile_id": "personal"}}],
        "scope": [{"slot_id": "scope_1", "subagent_id": "native"}],
        "advisory": {"enabled": True, "route": {"kind": "api_chat", "target_id": target, "profile_id": "personal"}},
        "deep_review": {"subagent_id": "native"},
    }
    ui["settings"].update(OUROBOROS_SUBAGENTS=actors, OUROBOROS_REVIEWER_SLOTS=json.dumps(slots))
    ui["fixture"]["preview"]["reviewer_slots"] = slots
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
    page.wait_for_function("""() => document.querySelector('[data-slot-route]')
        ?.querySelector('option[value="subscription:opaque-source"]')""")
    return page


def test_reviewer_source_roundtrip_restores_its_own_model_and_account(role_ui):
    configure_mixed(role_ui)
    lane = api_lane(role_ui, 'openai')
    page = open_agents(role_ui)
    for selector, model_field, account_field in [
        ('[data-slot-id="triad_1"]', '[data-slot-custom-api]', '[data-slot-profile]'),
        ('[data-advisory-row]', '[data-advisory-api-model]', '[data-advisory-profile]'),
    ]:
        row = page.locator(selector)
        route = row.locator('[data-slot-route], [data-advisory-route]')
        route.select_option(lane)
        # The chooser holds the model alone; the source select names the provider.
        row.locator(model_field).fill('other-choice')
        route.select_option('subscription:opaque-source')
        assert row.locator(model_field).input_value() == 'gpt-test'
        assert row.locator(account_field).input_value() == 'personal'
        route.select_option(lane)
        assert row.locator(model_field).input_value() == 'other-choice'
        assert route.input_value() == lane
        route.select_option('subscription:opaque-source')
    page.locator('[data-advisory-row]').scroll_into_view_if_needed()
    capture(page, "reviewer-source-roundtrip-restored")
    page.locator('[data-slot-custom-api]').fill('temporary-before-reload')
    with page.expect_response('**/api/reviewer-slots'):
        page.locator('#btn-reload-settings').click()
        page.get_by_role('button', name='Discard and continue', exact=True).click()
    page.locator('[data-advisory-route]').select_option(lane)
    assert page.locator('[data-advisory-api-model]').input_value() == ''


@pytest.mark.parametrize("width", [1360, 390])
def test_subscription_accounts_roundtrip_existing_editors(role_ui, width):
    ui = role_ui
    configure_mixed(ui)
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
    raw.locator('[data-subagent-field="model"]').fill("gpt-next")
    raw.locator('[data-subagent-field="account"]').select_option("work")
    triad = page.locator('[data-slot-id="triad_1"]')
    assert triad.locator('[data-slot-profile]').input_value() == "personal"
    triad.locator('[data-slot-custom-api]').fill("gpt-review")
    triad.locator('[data-slot-profile]').select_option("work")
    advisory = page.locator('[data-advisory-row]')
    advisory.locator('[data-advisory-api-model]').fill("gpt-advisory")
    advisory.locator('[data-advisory-profile]').select_option("work")
    assert page.locator('[data-deep-review-route]').input_value() == "subagent:native"
    assert "native" in page.locator('[data-deep-review-row]').inner_text().lower()
    assert page.locator('[data-deep-review-profile]').count() == 0
    triad.scroll_into_view_if_needed()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    controls = triad.locator(".reviewer-slot-controls")
    boxes = controls.locator("select, input, button").evaluate_all(
        "els => els.filter(e => !e.hidden).map(e => {const b=e.getBoundingClientRect();return {x:b.x,y:b.y,w:b.width,h:b.height}})")
    assert all(b["x"] >= 0 and b["x"] + b["w"] <= width for b in boxes)
    if width > 1000:
        # Direct triad delivery adds a second line; every control remains
        # reachable and no pair overlaps, including the trailing Remove.
        assert all(a["x"] + a["w"] <= b["x"] or b["x"] + b["w"] <= a["x"]
                   or a["y"] + a["h"] <= b["y"] or b["y"] + b["h"] <= a["y"]
                   for i, a in enumerate(boxes) for b in boxes[i + 1:]), boxes
    capture(page, f"role-reviewers-{width}")
    raw.scroll_into_view_if_needed()
    capture(page, f"role-actors-{width}")
    page.locator("#btn-save-settings").click()
    page.wait_for_function("() => !document.querySelector('#btn-save-settings').disabled")
    writes = [body for path, body in ui["posts"] if path == "/api/settings"]
    assert len(writes) == 1
    saved = writes[0]
    saved_actors = saved["OUROBOROS_SUBAGENTS"]
    if isinstance(saved_actors, str):
        saved_actors = json.loads(saved_actors)
    assert saved_actors["items"][0]["route"] == {
        "kind": "api_model", "target_id": "claudexor::opaque-source=gpt-next", "credential_profile_id": "work"}
    assert saved_actors["items"][1]["route"] == {"kind": "api_model", "target_id": "openai::gpt-api"}
    assert saved_actors["items"][2]["route"]["kind"] == "agent_session"
    saved_slots = json.loads(saved["OUROBOROS_REVIEWER_SLOTS"])
    assert saved_slots["triad"][0]["route"] == {
        "kind": "api_chat", "target_id": "claudexor::opaque-source=gpt-review", "profile_id": "work"}
    assert saved_slots["advisory"]["route"]["profile_id"] == "work"
    assert saved_slots["deep_review"] == {"subagent_id": "native"}
    assert saved_slots["scope"][0] == {"slot_id": "scope_1", "subagent_id": "native"}
    open_agents(ui)
    assert page.locator('[data-subagent-row]').nth(0).locator('[data-subagent-field="account"]').input_value() == "work"
    assert page.locator('[data-slot-profile]').input_value() == "work"
    assert page.locator('[data-advisory-profile]').input_value() == "work"
    assert page.locator('[data-deep-review-route]').input_value() == "subagent:native"
    assert "account work" in page.locator('[data-deep-review-row]').inner_text()
    assert not any("login" in path for path, _ in ui["posts"])


def test_source_switch_and_catalog_refresh_keep_focus_and_draft(role_ui):
    ui = role_ui
    configure_mixed(ui)
    lane = api_lane(ui, 'openai')
    page = open_agents(ui)
    triad = page.locator('[data-slot-id="triad_1"]')
    triad.locator('[data-slot-route]').select_option(lane)
    assert triad.locator('[data-slot-profile]').count() == 0
    triad.locator('[data-slot-custom-api]').fill("openai::gpt-api")
    triad.locator('[data-slot-route]').select_option("subscription:opaque-source")
    assert triad.locator('[data-slot-custom-api]').input_value() == "gpt-test"
    assert triad.locator('[data-slot-profile]').input_value() == "personal"
    triad.locator('[data-slot-custom-api]').fill("gpt-manual")
    triad.scroll_into_view_if_needed()
    field = triad.locator('[data-slot-custom-api]')
    field.focus()
    field.evaluate("e => e.setSelectionRange(3, 3)")
    before = page.locator("#content").evaluate("e => e.scrollTop")
    page.evaluate("""detail => document.dispatchEvent(new CustomEvent(
        'settings-model-catalog:updated', {detail}))""", ui["fixture"]["catalog"])
    assert page.evaluate("document.activeElement.hasAttribute('data-slot-custom-api')")
    assert field.input_value() == "gpt-manual"
    assert field.evaluate("e => e.selectionStart") == 3
    assert page.locator("#content").evaluate("e => e.scrollTop") == before
    capture(page, "role-source-refresh-focus")


def test_scope_and_inline_deep_keep_native_delivery_and_auto_account(role_ui):
    ui = role_ui
    configure_mixed(ui)
    slots = ui["fixture"]["preview"]["reviewer_slots"]
    slots["scope"] = [{"slot_id": "scope_1", "route": {
        "kind": "api_chat", "target_id": "claudexor::opaque-source=gpt-scope", "profile_id": "personal"}}]
    slots["deep_review"] = {"route": {
        "kind": "api_chat", "target_id": "claudexor::opaque-source=gpt-deep", "profile_id": "personal"}}
    ui["settings"]["OUROBOROS_REVIEWER_SLOTS"] = json.dumps(slots)
    page = open_agents(ui)
    scope = page.locator('[data-slot-id="scope_1"]')
    scope.locator('[data-slot-profile]').select_option("")
    scope.locator('[data-slot-custom-api]').fill("gpt-scope-next")
    deep = page.locator('[data-deep-review-row]')
    assert "Native inspection episode" in deep.inner_text()
    assert "host read-only tools" in deep.inner_text()
    deep.locator('[data-deep-review-profile]').select_option("work")
    deep.locator('[data-deep-review-api-model]').fill("gpt-deep-next")
    deep.scroll_into_view_if_needed()
    capture(page, "role-inline-deep")
    page.locator("#btn-save-settings").click()
    page.wait_for_function("() => !document.querySelector('#btn-save-settings').disabled")
    saved = json.loads(ui["settings"]["OUROBOROS_REVIEWER_SLOTS"])
    assert saved["scope"][0]["route"] == {
        "kind": "api_chat", "target_id": "claudexor::opaque-source=gpt-scope-next"}
    assert saved["deep_review"] == {"route": {
        "kind": "api_chat", "target_id": "claudexor::opaque-source=gpt-deep-next", "profile_id": "work"}}
    open_agents(ui)
    assert page.locator('[data-slot-id="scope_1"] [data-slot-profile]').input_value() == ""
    assert page.locator('[data-deep-review-profile]').input_value() == "work"
    assert "Native inspection episode" in page.locator('[data-deep-review-row]').inner_text()
    assert "host read-only tools" in page.locator('[data-deep-review-row]').inner_text()


def test_models_does_not_guess_a_credential_family_when_source_mapping_is_unread(role_ui):
    ui = role_ui
    configure_mixed(ui)
    ui["settings"]["OUROBOROS_MODEL_ACCOUNTS"] = {"main": "personal"}
    ui["fixture"]["catalog"]["model_sources"] = []
    page = open_agents(ui)
    page.locator('[data-settings-tab="models"]').click()
    main = page.locator('[data-model-role="main"]')
    account = main.locator('[data-model-role-account]')
    assert account.input_value() == "personal"
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


@pytest.mark.parametrize("consumer", ["Models", "Actor", "Triad", "Advisory"])
def test_catalog_failure_recovery_and_empty_read_keep_real_editor_nodes_and_draft(role_ui, consumer):
    ui = role_ui
    configure_mixed(ui)
    ui["settings"]["OUROBOROS_MODEL"] = "claudexor::opaque-source=gpt-test"
    ui["settings"]["OUROBOROS_MODEL_ACCOUNTS"] = {"main": "personal"}
    page = open_agents(ui)
    selectors = {
        "Models": ('[data-model-role="main"]', '[data-model-role-source]', '[data-model-role-model]', '[data-model-role-account]'),
        "Actor": ('[data-subagent-row]', '[data-subagent-field="route"]', '[data-subagent-field="model"]', '[data-subagent-field="account"]'),
        "Triad": ('[data-slot-id="triad_1"]', '[data-slot-route]', '[data-slot-custom-api]', '[data-slot-profile]'),
        "Advisory": ('[data-advisory-row]', '[data-advisory-route]', '[data-advisory-api-model]', '[data-advisory-profile]'),
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
    assert page.locator('[data-deep-review-route]').input_value() == "subagent:native"
    capture(page, f"catalog-recovery-draft-{consumer.lower()}")
    with page.expect_response("**/api/settings"):
        page.locator("#btn-save-settings").click()
    saved = [body for path, body in ui["posts"] if path == "/api/settings"][-1]
    if consumer == "Models":
        assert saved["OUROBOROS_MODEL"] == "claudexor::opaque-source=gpt-owner-unsaved"
        assert saved["OUROBOROS_MODEL_ACCOUNTS"]["main"] == "personal"
    elif consumer == "Actor":
        assert saved["OUROBOROS_SUBAGENTS"]["items"][0]["route"]["target_id"] == "claudexor::opaque-source=gpt-owner-unsaved"
        assert saved["OUROBOROS_SUBAGENTS"]["items"][0]["route"]["credential_profile_id"] == "personal"
    else:
        slots = json.loads(saved["OUROBOROS_REVIEWER_SLOTS"])
        route = (slots["triad"][0] if consumer == "Triad" else slots["advisory"])["route"]
        assert route["target_id"] == "claudexor::opaque-source=gpt-owner-unsaved"
        assert route["profile_id"] == "personal"


def test_image_model_uses_bare_engine_id_and_saves_without_chat_role_options(role_ui):
    """Real Settings module, rendered input and POST, not a mocked role collector."""
    ui = role_ui
    configure_mixed(ui)
    ui["settings"]["OUROBOROS_MODEL_IMAGE"] = "gpt-image-2"
    page = open_agents(ui)
    page.locator('[data-settings-tab="models"]').click()
    group = page.locator('[data-model-role-group="image"]')
    field = group.locator('[data-model-role-model]')
    assert field.input_value() == "gpt-image-2"
    for selector in ('[data-model-role-source]', '[data-model-role-account]',
                     '[data-model-role-context]', '[data-model-role-processing]'):
        assert group.locator(selector).count() == 0
    group.scroll_into_view_if_needed()
    capture(page, "image-model-bare-desktop")
    page.set_viewport_size({"width": 390, "height": 844})
    group.scroll_into_view_if_needed()
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    capture(page, "image-model-bare-narrow")

    field.fill('claudexor::codex=gpt-image-2')
    page.locator('#btn-save-settings').click()
    assert not [body for path, body in ui['posts'] if path == '/api/settings']
    assert field.get_attribute('aria-invalid') == 'true'
    assert 'bare engine model id' in group.locator('[data-model-role-error]').inner_text()

    field.fill('gpt-image-3')
    with page.expect_response('**/api/settings'):
        page.locator('#btn-save-settings').click()
    saved = [body for path, body in ui['posts'] if path == '/api/settings'][-1]
    assert saved['OUROBOROS_MODEL_IMAGE'] == 'gpt-image-3'
    for key in ('OUROBOROS_MODEL_ACCOUNTS', 'OUROBOROS_MODEL_CONTEXT_WINDOWS',
                'OUROBOROS_MODEL_PROCESSING_PREFERENCES'):
        assert 'image' not in (saved.get(key) or {})
