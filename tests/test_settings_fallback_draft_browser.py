"""A saved fallback's window on its own page (Settings › Models › Fallback N): an incomplete number is refused beside
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
    assert level.locator(".settings-page-title").inner_text() == f"Fallback {ordinal + 1}"
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
