"""Processing intent through existing Settings and onboarding, with no model calls."""
import pytest
from tests import test_subscription_role_routes_browser as roles

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
subscription_ui = roles.subscription_ui
role_ui = roles.role_ui


def test_processing_settings_save_reopen_and_actor_reference(role_ui):
    ui = role_ui
    roles.configure_mixed(ui)
    page = roles.open_agents(ui)
    page.locator('[data-settings-tab="models"]').click()
    assert page.locator('[data-global-processing]').input_value() == ''
    page.locator('[data-global-processing]').select_option('fast')
    main = page.locator('[data-model-role="main"]')
    assert not main.locator('details').evaluate('e => e.open')
    # The collapsed override must be recognizable as a disclosure: the glyph is
    # drawn by CSS, so only a real render can prove it is there and flips open.
    marker = "s => getComputedStyle(s, '::before').content"
    assert main.locator('summary').evaluate(marker) == '"▸ "'
    main.locator('summary').click()
    assert main.locator('summary').evaluate(marker) == '"▾ "'
    main.locator('[data-model-role-processing]').select_option('standard')
    assert 'Standard (override)' in main.locator('summary').text_content()
    main_model = ui['settings']['OUROBOROS_MODEL']
    page.locator('[data-settings-tab="agents"]').click()
    actor = page.locator('[data-subagent-row]').first
    assert 'Fast (from Models)' in actor.locator('[data-processing-summary]').text_content()
    assert actor.locator('[data-processing-details] > summary').evaluate(marker) == '"▸ "'
    actor.locator('[data-processing-details] > summary').click()
    # The first row is a reviewer: its processing is the processing of its reviews.
    assert actor.locator('[data-subagent-field="review_eligible"]').is_checked()
    actor.locator('[data-subagent-field="processing_preference"]').select_option('economy')
    field = actor.locator('[data-subagent-field="model"]')
    field.focus()
    field.evaluate('e => { window.processingModelNode = e; e.setSelectionRange(1, 4); }')
    page.evaluate("""() => document.dispatchEvent(new CustomEvent('settings-model-catalog:updated',
        {detail:{read_state:'transport',errors:[{error:'offline'}]}}))""")
    assert field.evaluate('e => e === window.processingModelNode && document.activeElement === e')
    assert field.evaluate('e => [e.selectionStart, e.selectionEnd]') == [1, 4]
    assert actor.locator('[data-processing-details]').evaluate('e => e.open')
    assert actor.locator('[data-subagent-field="processing_preference"]').input_value() == 'economy'
    roles.capture(page, 'processing-actor-reference-and-inline')
    with page.expect_response('**/api/settings'):
        page.locator('#btn-save-settings').click()
    saved = [body for path, body in ui['posts'] if path == '/api/settings'][-1]
    assert saved['OUROBOROS_PROCESSING_PREFERENCE'] == 'fast'
    assert saved['OUROBOROS_MODEL_PROCESSING_PREFERENCES']['main'] == 'standard'
    assert saved['OUROBOROS_MODEL'] == main_model
    assert saved['OUROBOROS_SUBAGENTS']['items'][0]['processing_preference'] == 'economy'
    assert 'processing_preference' not in saved['OUROBOROS_SUBAGENTS']['items'][1]
    assert 'OUROBOROS_REVIEWER_SLOTS' not in saved
    page = roles.open_agents(ui)
    reopened = page.locator('[data-subagent-row]').first
    assert reopened.locator('[data-subagent-field="processing_preference"]').input_value() == 'economy'
    page.locator('[data-settings-tab="models"]').click()
    assert page.locator('[data-global-processing]').input_value() == 'fast'
    assert page.locator('[data-model-role="main"] [data-model-role-processing]').input_value() == 'standard'


def test_onboarding_processing_draft_returns_and_reaches_finish(subscription_ui):
    ui, page = subscription_ui, subscription_ui['page']
    page.goto(ui['url'] + '/onboarding')
    page.wait_for_selector('#quick-start-btn:not([hidden])')
    page.locator('#next-btn').click()
    page.wait_for_selector('[data-global-processing]')
    page.locator('[data-global-processing]').select_option('fast')
    main = page.locator('[data-model-role="main"]')
    main.locator('summary').click()
    main.locator('[data-model-role-processing]').select_option('standard')
    page.locator('#next-btn').click()
    page.wait_for_selector('[data-reviewers-note]')
    page.locator('#back-btn').click()
    page.wait_for_selector('[data-global-processing]')
    assert page.locator('[data-global-processing]').input_value() == 'fast'
    assert page.locator('[data-model-role="main"] [data-model-role-processing]').input_value() == 'standard'
    for _ in range(3):
        page.locator('#next-btn').click()
    page.wait_for_selector('.summary-card')
    assert 'Processing: Standard (override)' in page.locator('.summary-card').text_content()
    roles.capture(page, 'processing-onboarding-summary')
    with page.expect_response('**/api/onboarding/complete'):
        page.locator('#next-btn').click()
    saved = [body for path, body in ui['posts'] if path == '/api/onboarding/complete'][-1]
    assert saved['OUROBOROS_PROCESSING_PREFERENCE'] == 'fast'
    assert saved['OUROBOROS_MODEL_PROCESSING_PREFERENCES']['main'] == 'standard'
