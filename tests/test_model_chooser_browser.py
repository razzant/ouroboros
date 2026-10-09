"""Editable model chooser on real Settings/actor/reviewer consumers, no model calls."""
from __future__ import annotations

import pytest

from tests import test_subscription_role_routes_browser as roles

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
subscription_ui = roles.subscription_ui
role_ui = roles.role_ui


def catalog(ui):
    value = ui["fixture"]["catalog"]
    value["items"] = [{"value": f"choice-{i:02}", "id": f"choice-{i:02}"} for i in range(30)]
    return value


def select_model_field(ui, consumer):
    roles.configure_mixed(ui)
    # The catalog above is unprefixed, so every consumer is put on the same
    # OpenRouter API lane the suggestions belong to.
    lane = roles.api_lane(ui)
    data = catalog(ui)
    page = roles.open_agents(ui)
    page.evaluate("""detail => document.dispatchEvent(new CustomEvent(
        'settings-model-catalog:updated', {detail}))""", data)
    if consumer in ["Models", "Fallback"]:
        page.locator('[data-settings-tab="models"]').click()
        if consumer == 'Fallback': page.locator('[data-model-add]').click()
        group = page.locator('[data-model-role="main"]') if consumer == 'Models' else page.locator('[data-model-role-group="fallback"] .model-role-row').first
        group.locator('[data-model-role-source]').select_option(lane)
        return page, group.locator('[data-model-role-model]')
    # Actor: a row marked Reviewer, so the reviewer chooser is this same field.
    row = page.locator('[data-subagent-row]').nth(1)
    row.locator('[data-subagent-field="route"]').select_option(lane)
    return page, row.locator('[data-subagent-field="model"]')


@pytest.mark.parametrize("consumer", ["Models", "Fallback", "Actor"])
def test_chooser_keyboard_free_values_and_catalog_identity(role_ui, consumer):
    page, field = select_model_field(role_ui, consumer)
    field.fill('choice')
    popup = page.locator('[id="' + field.get_attribute('aria-controls') + '"]')
    assert popup.is_visible()
    assert popup.locator('[role="option"]').count() == 30
    field.press('ArrowDown')
    assert field.get_attribute('aria-activedescendant')
    field.press('Escape')
    assert field.input_value() == 'choice'
    assert not popup.is_visible()
    field.press('ArrowDown')
    field.press('Enter')
    assert field.input_value() == 'choice-00'
    assert field.get_attribute('aria-expanded') == 'false'
    field.fill('owner/new-unknown-id')
    field.evaluate("""e => { window.keptModelField = e; e.setSelectionRange(3, 7);
        e.dispatchEvent(new CompositionEvent('compositionstart', {bubbles:true})); }""")
    page.evaluate("""() => document.dispatchEvent(new CustomEvent(
        'settings-model-catalog:updated', {detail:{items:[],model_sources:[]}}))""")
    assert field.evaluate('e => e === window.keptModelField && document.activeElement === e')
    assert field.evaluate('e => [e.selectionStart,e.selectionEnd]') == [3, 7]
    assert field.input_value() == 'owner/new-unknown-id'
    field.evaluate("e => e.dispatchEvent(new CompositionEvent('compositionend', {bubbles:true}))")
    field.press('Tab')
    assert field.input_value() == 'owner/new-unknown-id'
    assert not popup.is_visible()
    roles.capture(page, f"chooser-{consumer.lower()}-unknown-kept")


def test_agent_session_uses_same_chooser_and_keeps_native_short_choices(role_ui):
    roles.configure_mixed(role_ui)
    page = roles.open_agents(role_ui)
    row = page.locator('[data-subagent-row]').nth(2)
    model = row.locator('[data-subagent-field="model"]')
    assert model.evaluate('e => e.tagName') == 'INPUT'
    assert model.get_attribute('role') == 'combobox'
    for field in ['route', 'account', 'effort']:
        assert row.locator(f'[data-subagent-field="{field}"]').evaluate('e => e.tagName') == 'SELECT'
    model.fill('future-model')
    page.evaluate("""detail => document.dispatchEvent(new CustomEvent(
        'settings-model-catalog:updated', {detail}))""", role_ui['fixture']['catalog'])
    assert model.input_value() == 'future-model'
    assert row.locator('[data-subagent-field="account"]').input_value() == 'personal'
    roles.capture(page, 'chooser-session-reference-preserved')


def test_incomplete_reviewer_holds_whole_settings_save_until_corrected(role_ui):
    roles.configure_mixed(role_ui)
    page = roles.open_agents(role_ui)
    reviewer = page.locator('[data-subagent-row]').first
    assert reviewer.locator('[data-subagent-field="review_eligible"]').is_checked()
    model = reviewer.locator('[data-subagent-field="model"]')
    model.fill('')
    page.locator('#btn-save-settings').click()
    assert not [path for path, _ in role_ui['posts'] if path == '/api/settings']
    assert model.input_value() == ''
    assert model.get_attribute('aria-invalid') == 'true'
    model.fill('owner-unknown-review-model')
    assert model.get_attribute('aria-invalid') == 'false'
    with page.expect_response('**/api/settings'):
        page.locator('#btn-save-settings').click()
    writes = [payload for path, payload in role_ui['posts'] if path == '/api/settings']
    assert len(writes) == 1
    roles.capture(page, 'chooser-reviewer-corrected-save')


@pytest.fixture
def touch_role_ui(monkeypatch, request):
    monkeypatch.setenv('OUROBOROS_UI_HAS_TOUCH', '1')
    return request.getfixturevalue('role_ui')


def test_chooser_touch_selection(touch_role_ui):
    page, field = select_model_field(touch_role_ui, 'Actor')
    page.set_viewport_size({'width': 390, 'height': 600})
    field.tap()
    field.fill('choice-')
    popup = page.locator('[id="' + field.get_attribute('aria-controls') + '"]')
    popup.locator('[data-model-value="choice-01"]').tap()
    assert field.input_value() == 'choice-01'
    assert field.get_attribute('aria-expanded') == 'false'
    roles.capture(page, 'chooser-touch-selected')


def test_partial_numeric_context_blocks_save_and_preserves_auto(role_ui):
    roles.configure_mixed(role_ui)
    role_ui['settings']['OUROBOROS_MODEL_CONTEXT_WINDOWS'] = {'main': 8192}
    page = roles.open_agents(role_ui)
    page.locator('[data-settings-tab="models"]').click()
    row = page.locator('[data-model-role="main"]')
    row.locator('summary').click()
    field = row.locator('[data-model-role-context]')
    assert field.input_value() == '8192'
    field.fill('')
    field.press('1')
    field.press('e')
    assert field.evaluate('e => e.validity.badInput')
    page.locator('#btn-save-settings').click()
    assert not [path for path, _ in role_ui['posts'] if path == '/api/settings']
    assert field.get_attribute('aria-invalid') == 'true'
    field.fill('16384')
    assert field.get_attribute('aria-invalid') == 'false'
    with page.expect_response('**/api/settings'):
        page.locator('#btn-save-settings').click()
    saved = [payload for path, payload in role_ui['posts'] if path == '/api/settings'][-1]
    assert saved['OUROBOROS_MODEL_CONTEXT_WINDOWS']['main'] == 16384
    page.wait_for_function("() => !document.querySelector('#btn-save-settings').disabled")
    row.locator('summary').click()
    field.fill('')
    assert not field.evaluate('e => e.validity.badInput')
    with page.expect_response('**/api/settings'):
        page.locator('#btn-save-settings').click()
    saved = [payload for path, payload in role_ui['posts'] if path == '/api/settings'][-1]
    assert saved['OUROBOROS_MODEL_CONTEXT_WINDOWS']['main'] == 0
    roles.capture(page, 'numeric-context-corrected-auto')
    page.wait_for_function("() => !document.querySelector('#btn-save-settings').disabled")
    row.locator('summary').click()
    field.press('1')
    field.press('e')
    assert field.evaluate('e => e.validity.badInput')
    assert 'is-visible' in page.locator('#settings-unsaved-indicator').get_attribute('class')


def test_chooser_scrolls_with_mobile_keyboard_boundary(role_ui):
    page = role_ui['page']
    page.set_viewport_size({'width': 390, 'height': 844})
    page.add_init_script("""(() => {
        const viewport = new EventTarget();
        Object.assign(viewport, {width:390,height:844,offsetLeft:0,offsetTop:0,scale:1});
        Object.defineProperty(window,'visualViewport',{value:viewport,configurable:true});
        window.testViewport = viewport;
    })();""")
    page, field = select_model_field(role_ui, 'Actor')
    field.click()
    field.fill('choice')
    popup = page.locator('[id="' + field.get_attribute('aria-controls') + '"]')
    page.evaluate("""() => { testViewport.height=400;testViewport.dispatchEvent(new Event('resize')); }""")
    page.wait_for_function("document.body.classList.contains('keyboard-open')")
    assert popup.evaluate('e => e.scrollHeight > e.clientHeight')
    def gesture(target):
        return target.evaluate("""node => {
            for (const [type,y] of [['touchstart',220],['touchmove',170]]) {
                const event=new Event(type,{bubbles:true,cancelable:true});
                Object.defineProperty(event,'touches',{value:[{clientX:100,clientY:y}]});
                node.dispatchEvent(event);
                if(type==='touchmove') return event.defaultPrevented;
            }
        }""")
    assert gesture(popup) is False
    assert gesture(page.locator('#page-settings .app-page-header')) is True
    assert field.evaluate('e => document.activeElement === e')
    roles.capture(page, 'chooser-keyboard-touch-scroll')


@pytest.mark.parametrize("width,height", [(320, 480), (390, 600), (640, 360), (641, 540),
    (760, 600), (768, 600), (980, 600), (981, 540), (1440, 900), (1920, 900)])
def test_chooser_popup_reachable_in_narrow_short_view(role_ui, width, height):
    page, field = select_model_field(role_ui, 'Actor')
    page.set_viewport_size({'width': width, 'height': height})
    field.click()
    field.fill('choice')
    popup = page.locator('[id="' + field.get_attribute('aria-controls') + '"]')
    assert popup.is_visible()
    box = popup.bounding_box()
    roles.capture(page, f'chooser-open-{width}-{height}')
    assert box and box['x'] >= 0 and box['y'] >= 0
    assert box['x'] + box['width'] <= width + 1
    assert box['y'] + box['height'] <= height + 1, {'popup': box, 'anchor': field.bounding_box()}
    option = popup.locator('[role="option"]').nth(29)
    option.scroll_into_view_if_needed()
    option.click()
    assert field.input_value() == 'choice-29'
    roles.capture(page, f'chooser-narrow-{width}-{height}')
