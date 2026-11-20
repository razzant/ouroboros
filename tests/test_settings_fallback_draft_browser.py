"""A saved fallback's window on its own page (Settings › Models › Fallback model N): an incomplete number is refused beside
Save and nothing is sent, the typed text survives leaving Settings, and a whole number is written as that fallback's
entry of the window map alone."""
from __future__ import annotations

import pytest
from tests import test_subscription_role_routes_browser as roles
from tests.settings_browser_helpers import open_settings, press_editor_save

pytestmark = [pytest.mark.ui_browser]
subscription_ui = roles.subscription_ui
role_ui = roles.role_ui


@pytest.mark.parametrize('ordinal', [1, 2])
def test_saved_fallback_context_keeps_incomplete_number_through_stay_and_auto(role_ui, ordinal):
    from playwright.sync_api import expect
    ui = role_ui
    roles.configure_mixed(ui)
    ui['settings']['OUROBOROS_MODEL_FALLBACKS'] = 'owner/first, owner/second, owner/third'
    ui['settings']['OUROBOROS_MODEL_CONTEXT_WINDOWS'] = {'fallback': [0, 0, 0]}
    page = ui["page"]
    page.goto(ui["url"] + "/#settings")
    level = open_settings(page, f"models/role/fallback/{ordinal + 1}")
    level.locator("[data-editor-save]").wait_for(state="visible")
    # A model role names its destination, apart from a return to it (docs/DESIGN.md "Page header").
    assert page.locator("#page-settings [data-settings-lead-name]").inner_text() == f"Fallback model {ordinal + 1}"
    expect(level.locator("[data-model-role-model]")).to_have_value(['owner/first', 'owner/second', 'owner/third'][ordinal])
    level.locator("details.model-role-details > summary").click()
    field = level.locator('[data-model-role-context]')
    expect(field).to_have_value('')
    expect(level.locator("[data-editor-discard]")).to_be_hidden()
    field.press('1')
    field.press('e')
    assert field.evaluate('input => input.validity.badInput && input.value === ""')
    expect(level.locator("[data-editor-discard]")).to_be_visible()  # the page says it holds unsaved text
    field.evaluate('input => { window.__fallbackDraftNode = input; }')
    page.locator('[data-nav-page="chat"]').click()
    # Leaving keeps the typed text for the return (owner 2026-11-03): no discard question, the same node.
    expect(page.locator('#page-chat')).to_be_visible()
    expect(page.get_by_role('button', name='Stay', exact=True)).to_have_count(0)
    page.locator('[data-nav-page="settings"]').click()
    expect(level).to_be_visible()
    assert field.evaluate('input => input === window.__fallbackDraftNode && input.validity.badInput')
    level.locator("[data-editor-save]").click()
    expect(field).to_have_attribute('aria-invalid', 'true')
    expect(level.locator("[data-editor-result]")).to_contain_text('context window must be a positive whole number, or Auto')
    assert not [path for path, _ in ui['posts'] if path in ('/api/settings', '/api/owner/safety-model')]
    field.fill('')
    assert not field.evaluate('input => input.validity.badInput')
    expect(field).to_have_attribute('aria-invalid', 'false')
    field.fill('32768')
    assert press_editor_save(page).startswith('Saved')
    # One write, of the window map alone, through the checker's transaction: only this fallback's entry changed.
    written = [(path, body) for path, body in ui['posts'] if path in ('/api/settings', '/api/owner/safety-model')]
    expected = [0, 0, 0]
    expected[ordinal] = 32768
    assert [path for path, _ in written] == ['/api/owner/safety-model']
    assert written[0][1]['settings'] == {'OUROBOROS_MODEL_CONTEXT_WINDOWS': {'fallback': expected}}
    expect(level.locator('[data-model-role-context]')).to_have_value('32768')


@pytest.mark.parametrize("interaction", ["mouse", "keyboard", "touch"])
def test_open_fallback_source_menu_follows_discovery(role_ui, interaction):
    """The same opening gains Codex. A press or keyboard highlight on an existing API keeps its identity."""
    from playwright.sync_api import expect

    ui = role_ui
    roles.configure_mixed(ui)
    roles.api_lane(ui, "openrouter")
    roles.api_lane(ui, "openai")
    ui["settings"]["OUROBOROS_MODEL_FALLBACKS"] = "owner/first, owner/second"
    touch = interaction == "touch"
    page = ui["new_page"](has_touch=True, viewport={"width": 390, "height": 780}) if touch else ui["page"]
    page.emulate_media(reduced_motion="reduce")
    pending = []
    page.route("**/api/model-catalog?view=sources", lambda route: pending.append(route))
    page.goto(ui["url"] + "/#settings")
    level = open_settings(page, "models/role/fallback/2")
    trigger = level.get_by_role("combobox", name="Fallback Model source", exact=True)
    source = level.locator("select[data-model-role-source]")
    (trigger.tap if touch else trigger.click)()
    popup = page.get_by_role("listbox", name="Fallback Model source", exact=True)
    expect(popup).to_be_visible()
    expect(trigger).to_be_focused()
    expect(level.locator("[data-model-role-model]")).to_have_value("owner/second")
    earlier = popup.get_by_role("option", name="OpenAI", exact=True)
    if interaction == "mouse":
        earlier.hover()
        before = earlier.bounding_box()
        page.mouse.down()
    elif interaction == "keyboard":
        trigger.press("End")
        expect(trigger).to_have_attribute("aria-activedescendant", earlier.get_attribute("id"))
    expect(popup.get_by_role("option", name="Codex (model)")).to_have_count(0)
    assert pending
    # Several new rows above the API keys move the pressed/highlighted row to a different index.
    sources = ui["fixture"]["catalog"]["model_sources"] + [
        {"id": "another", "label": "Another subscription"}, {"id": "third", "label": "Third subscription"}]
    for route in pending:
        route.fulfill(json={"items": [], "model_sources": sources, "errors": []})
    codex = popup.get_by_role("option", name="Codex (model)", exact=True)
    expect(codex).to_be_visible()
    expect(trigger).to_have_attribute("aria-expanded", "true")
    expect(earlier).to_be_visible()
    roles.capture(page, "fallback-live-source-" + interaction)
    if interaction == "mouse":
        assert earlier.bounding_box()["y"] > before["y"] + before["height"]
        page.mouse.up()
        expect(source).to_have_value("api:openai")
    elif interaction == "keyboard":
        expect(trigger).to_have_attribute("aria-activedescendant", earlier.get_attribute("id"))
        trigger.press("Escape")
        expect(trigger).to_be_focused()
        expect(source).to_have_value("api:openrouter")
        trigger.press("Enter")
        trigger.press("Home")
        trigger.press("Enter")
    else:
        codex.tap()
    expect(popup).to_have_count(0)
    if interaction != "mouse":
        expect(source).to_have_value("subscription:opaque-source")
        field = level.locator("[data-model-role-model]")
        field.fill("gpt")
        option = page.locator('[role="listbox"]:visible [data-model-value="gpt-test"]')
        expect(option).to_be_visible()
        (option.tap if touch else option.click)()
        expect(field).to_have_value("gpt-test")
    assert not [url for url in ui["reads"] if url.endswith("/api/model-catalog")]
    assert not ui["posts"]
    if interaction == "keyboard":
        assert press_editor_save(page).startswith("Saved")
        assert ui["settings"]["OUROBOROS_MODEL_FALLBACKS"] == "owner/first, claudexor::opaque-source=gpt-test"
