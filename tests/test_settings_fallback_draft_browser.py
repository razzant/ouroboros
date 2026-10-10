"""Saved repeated model contexts keep native incomplete numbers in the Settings draft."""
from __future__ import annotations

import json
import re
import os
from pathlib import Path

import pytest
from tests import test_subscription_role_routes_browser as roles

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
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
    with page.expect_response(lambda response: response.url.endswith("/api/model-catalog")):
        roles.open_agents(ui)
    page.wait_for_function("() => !document.querySelector('#btn-refresh-model-catalog').disabled")
    page.locator('[data-settings-tab="models"]').click()
    row = page.locator('[data-model-role-group="fallback"] .model-role-row').nth(ordinal)
    row.locator('summary').click()
    field = row.locator('[data-model-role-context]')
    expect(field).to_have_value('')
    assert not field.get_attribute('id').startswith('s-')
    expect(page.locator('#settings-unsaved-indicator')).not_to_have_class(re.compile('is-visible'))
    field.press('1')
    field.press('e')
    assert field.evaluate('input => input.validity.badInput && input.value === ""')
    expect(page.locator('#settings-unsaved-indicator')).to_have_class(re.compile('is-visible'))
    field.evaluate('input => { window.__fallbackDraftNode = input; }')
    page.locator('[data-nav-page="chat"]').click()
    expect(page.get_by_role('button', name='Stay', exact=True)).to_be_visible()
    page.get_by_role('button', name='Stay', exact=True).click()
    expect(page.locator('#page-settings')).to_be_visible()
    assert field.evaluate('input => input === window.__fallbackDraftNode && input.validity.badInput')
    page.locator('#btn-save-settings').click()
    expect(field).to_have_attribute('aria-invalid', 'true')
    assert not [path for path, _ in ui['posts'] if path == '/api/settings']
    field.fill('')
    assert not field.evaluate('input => input.validity.badInput')
    expect(field).to_have_attribute('aria-invalid', 'false')
    expect(page.locator('#settings-unsaved-indicator')).not_to_have_class(re.compile('is-visible'))
    field.fill('32768')
    with page.expect_response('**/api/settings'):
        page.locator('#btn-save-settings').click()
    saved = [body for path, body in ui['posts'] if path == '/api/settings']
    assert len(saved) == 1
    expected = [0, 0, 0]
    expected[ordinal] = 32768
    assert saved[0]['OUROBOROS_MODEL_CONTEXT_WINDOWS']['fallback'] == expected


def test_all_settings_number_fields_are_in_complete_raw_draft_selector(role_ui):
    """Enumerate actual consumers: settings/Agents numbers and all role-context rows."""
    ui = role_ui
    roles.configure_mixed(ui)
    ui['settings']['OUROBOROS_MODEL_FALLBACKS'] = 'owner/first, owner/second, owner/third'
    page = ui["page"]
    with page.expect_response(lambda response: response.url.endswith("/api/model-catalog")):
        roles.open_agents(ui)
    page.wait_for_function("() => !document.querySelector('#btn-refresh-model-catalog').disabled")
    selector = (Path(__file__).resolve().parents[1] / 'web/modules/settings.js').read_text().split('function snapshotSettingsDraft()', 1)[1].split("page.querySelectorAll('", 1)[1].split("')", 1)[0]
    # A maximum response is an exact-route acknowledgement its own Apply writes
    # (`POST /api/owner/capability-ack`), never a Settings draft field.
    facts = page.evaluate('''selector => [...document.querySelectorAll('#page-settings input[type="number"]')]
        .filter(input => !input.closest('[data-extension-settings-form]'))
        .map(input => ({id:input.id, model_context:input.hasAttribute('data-model-role-context'),
            acknowledgement:input.hasAttribute('data-response-limit'), selected:input.matches(selector)}))''', selector)
    assert all(row['selected'] is not row['acknowledgement'] for row in facts), facts
    assert any(row['acknowledgement'] for row in facts)
    assert sum(row['model_context'] for row in facts) >= 7
    output = os.environ.get('OUROBOROS_UI_EVIDENCE_DIR')
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / 'settings-number-consumers.json').write_text(json.dumps(facts, indent=2) + '\n')
