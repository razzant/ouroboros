"""Processing intent through Settings › Models and onboarding, with no model calls."""
import pytest
from tests import test_subscription_role_routes_browser as roles
from tests.settings_browser_helpers import commit_row, open_settings, press_editor_save

pytestmark = [pytest.mark.ui_browser]
subscription_ui = roles.subscription_ui
role_ui = roles.role_ui
# The shared disclosure caret: a drawn chevron, no glyph, pointing right when closed (-45°) and down when open
# (45°); its turn is a transition, so the angle is awaited.
CARET = """([summary, angle]) => {
    const c = getComputedStyle(summary, '::before');
    const drawn = !['none', 'normal'].includes(c.content) && parseFloat(c.width) > 0
        && parseFloat(c.borderRightWidth) > 0 && parseFloat(c.borderBottomWidth) > 0;
    const m = new DOMMatrixReadOnly(c.transform);
    return drawn && Math.round(Math.atan2(m.b, m.a) * 180 / Math.PI) === angle;
}"""


def caret(summary, angle):
    summary.page.wait_for_function(CARET, arg=[summary.element_handle(), angle])


def test_processing_settings_save_reopen_and_actor_reference(role_ui):
    """Models: the install's Processing is a row of the fold that saves itself, and Main's own choice is a field of
    Main's page, saved by its Save through the checker's transaction. (An agent's line that says "Fast (from
    Models)" is the Agents page's, test_agents_settings_browser.py.)"""
    ui = role_ui
    roles.configure_mixed(ui)
    page = ui["page"]
    page.goto(ui["url"] + "/#settings")
    level = open_settings(page, "models", "OUROBOROS_PROCESSING_PREFERENCE")
    row = level.locator('[data-setting-row="OUROBOROS_PROCESSING_PREFERENCE"]')
    row.wait_for(state="visible")  # the row the page was opened for: its fold is open
    assert row.locator("[data-setting-control]").input_value() == ''
    assert commit_row(page, "OUROBOROS_PROCESSING_PREFERENCE", "fast").startswith("Saved")
    assert ui['posts'][-1] == ('/api/settings', {'OUROBOROS_PROCESSING_PREFERENCE': 'fast'})
    main_model = ui['settings']['OUROBOROS_MODEL']
    level = open_settings(page, "models/role/main")
    main = level.locator('[data-model-role="main"]')
    main.locator("[data-editor-save]").wait_for(state="visible")
    assert not main.locator('details').evaluate('e => e.open')
    # The collapsed fold must be recognizable as a disclosure: the caret is
    # drawn by CSS, so only a real render can prove it is there and flips open.
    caret(main.locator('summary'), -45)
    main.locator('summary').click()
    caret(main.locator('summary'), 45)
    assert 'Fast (from Models)' in main.locator('summary').text_content()
    main.locator('[data-model-role-processing]').select_option('standard')
    assert 'Standard (override)' in main.locator('summary').text_content()
    roles.capture(page, 'processing-main-role-page')
    before = len(ui['posts'])
    assert press_editor_save(page).startswith('Saved')
    # Only what changed is written: Main's entry of the processing map, by the checker; the model is not sent again.
    assert ui['posts'][before:] == [('/api/owner/safety-model', {'settings': {'OUROBOROS_MODEL_PROCESSING_PREFERENCES': {'main': 'standard'}}})]
    assert ui['settings']['OUROBOROS_MODEL'] == main_model
    page.reload()
    level = open_settings(page, "models", "OUROBOROS_PROCESSING_PREFERENCE")
    expect_value = level.locator('[data-setting-row="OUROBOROS_PROCESSING_PREFERENCE"] [data-setting-control]')
    page.wait_for_function("() => document.querySelector('#page-settings [data-setting-row=\"OUROBOROS_PROCESSING_PREFERENCE\"] [data-setting-control]')?.value === 'fast'")
    assert expect_value.input_value() == 'fast'
    level = open_settings(page, "models/role/main")
    level.locator("[data-editor-save]").wait_for(state="visible")
    assert level.locator('[data-model-role="main"] [data-model-role-processing]').input_value() == 'standard'


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


def test_agent_processing_save_reopen_and_catalog_refresh(role_ui):
    ui = role_ui
    roles.configure_mixed(ui)
    ui['settings'].update(OUROBOROS_PROCESSING_PREFERENCE='fast',
                          OUROBOROS_MODEL_PROCESSING_PREFERENCES={'main': 'standard'})
    actor = roles.open_agent(ui, "native")
    page = ui["page"]
    assert 'Fast (from Models)' in actor.locator('[data-processing-summary]').text_content()
    caret(actor.locator('[data-processing-details] > summary'), -45)
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
    # The always-mounted Circles listener rejects this deliberately failed catalogue, independently of the editor.
    refused = 'The model catalog could not be read.'
    assert not [error for error in ui['errors'].unexpected() if refused not in error], ui['errors'].report()
    ui['errors'].clear()
    assert press_editor_save(page).startswith('Saved')
    saved = roles.saved_catalog(ui)
    assert saved['items'][0]['processing_preference'] == 'economy'
    assert 'processing_preference' not in saved['items'][1]
    body = [body['settings'] for path, body in ui['posts'] if path == '/api/owner/safety-model'][-1]
    assert set(body) == {'OUROBOROS_SUBAGENTS'}
    reopened = roles.open_agent(ui, 'native')
    assert reopened.locator('[data-subagent-field="processing_preference"]').input_value() == 'economy'
    assert ui['settings']['OUROBOROS_PROCESSING_PREFERENCE'] == 'fast'
    assert ui['settings']['OUROBOROS_MODEL_PROCESSING_PREFERENCES']['main'] == 'standard'
