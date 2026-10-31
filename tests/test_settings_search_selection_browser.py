"""Settings search edits travel through the real binder, preview and dispatch."""
import json
from types import SimpleNamespace

import pytest

from tests import test_model_route_contribution_browser as routes

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
subscription_ui = routes.subscription_ui
role_ui = routes.role_ui
route_ui = routes.route_ui


@pytest.fixture(autouse=True)
def _anthropic_sdk_present(monkeypatch):
    """The preview and dispatch read the optional Anthropic SDK locally; this suite chooses it."""
    import sys
    import types

    monkeypatch.setitem(sys.modules, 'anthropic', sys.modules.get('anthropic') or types.ModuleType('anthropic'))


@pytest.mark.parametrize('edit', ['untouched', 'model', 'source', 'same_model'])
def test_anthropic_settings_selection_and_untouched_legacy_dispatch(route_ui, monkeypatch, edit):
    from playwright.sync_api import expect
    from ouroboros import llm
    from ouroboros.tools import search

    ui = route_ui
    ui['settings'].update(ANTHROPIC_API_KEY='fixture-key',
                          OUROBOROS_WEBSEARCH_BACKEND='auto' if edit == 'source' else 'anthropic',
                          OUROBOROS_WEBSEARCH_MODEL='claude-opus-4-6')
    page = ui['page']
    page.goto(ui['url'] + '/#settings')
    page.locator('[data-settings-tab="models"]').click()
    preview = page.locator('#s-websearch-preview')
    expect(preview).to_contain_text('claude-sonnet-4-6')
    field = page.locator('#s-websearch-model')
    expect(field).to_have_value('claude-opus-4-6')
    if edit in {'model', 'same_model'}:
        field.fill('claude-future-native' if edit == 'model' else 'claude-opus-4-6')
        field.blur()
    elif edit == 'source':
        page.locator('#s-websearch-source').select_option('anthropic')
    expected = ('claude-sonnet-4-6' if edit == 'untouched' else
                'claude-future-native' if edit == 'model' else 'claude-opus-4-6')
    expect(preview).to_contain_text(f'anthropic · {expected}')
    if edit != 'untouched':
        expect(preview).not_to_contain_text('does not apply')
    if edit == 'model':
        preview.scroll_into_view_if_needed()
        routes.roles.capture(page, 'anthropic-native-edited-preview')
    with page.expect_response('**/api/settings'):
        page.locator('#btn-save-settings').click()
    page.wait_for_function("() => !document.querySelector('#btn-save-settings').disabled")
    saved = [body for path, body in ui['posts'] if path == '/api/settings'][-1]
    assert saved['OUROBOROS_WEBSEARCH_BACKEND'] == 'anthropic'
    assert saved['OUROBOROS_WEBSEARCH_MODEL'] == (
        'claude-opus-4-6' if edit == 'untouched' else f'anthropic::{expected}')
    for key in ('OUROBOROS_WEBSEARCH_BACKEND', 'OUROBOROS_WEBSEARCH_MODEL'):
        monkeypatch.setenv(key, saved[key])
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'fixture-key')
    calls = []
    def server_tool(**kwargs):
        calls.append(kwargs['model'])
        return SimpleNamespace(content=[SimpleNamespace(type='text', text='found')], usage=None, model=kwargs['model'])
    monkeypatch.setattr(llm, 'anthropic_web_search_server_tool', server_tool)
    result = json.loads(search._web_search(SimpleNamespace(pending_events=[], task_metadata={}), 'query'))
    assert calls == [expected] and result['model'] == expected
    page.locator('#btn-reload-settings').click()
    expect(field).to_have_value(saved['OUROBOROS_WEBSEARCH_MODEL'])
    expect(preview).to_contain_text(f'anthropic · {expected}')
    if edit == 'model':
        preview.scroll_into_view_if_needed()
        routes.roles.capture(page, 'anthropic-native-saved-reloaded')
    assert not ui['errors']


@pytest.mark.parametrize('state', ['missing_sdk_and_legacy_url', 'no_key'])
def test_settings_preview_names_why_a_search_source_is_unavailable(route_ui, monkeypatch, state):
    """#1697: the preview the real gateway serves is rendered with each unavailable cause."""
    import sys
    from playwright.sync_api import expect

    ui = route_ui
    ui['settings'].update(OUROBOROS_WEBSEARCH_BACKEND='auto', OUROBOROS_WEBSEARCH_MODEL='gpt-5.2')
    if state == 'no_key':
        ui['settings'].pop('ANTHROPIC_API_KEY', None)
        expected = 'Unavailable: openai (no API key), anthropic (no API key).'
    else:
        monkeypatch.setitem(sys.modules, 'anthropic', None)
        ui['settings'].update(ANTHROPIC_API_KEY='fixture-key', OPENAI_API_KEY='fixture-key',
                              OPENAI_BASE_URL='https://compatible.invalid/v1')
        expected = 'Unavailable: openai (OPENAI_BASE_URL is set), anthropic (package not installed).'
    monkeypatch.setitem(sys.modules, 'ddgs', types_module('ddgs'))
    page = ui['page']
    page.goto(ui['url'] + '/#settings')
    page.locator('[data-settings-tab="models"]').click()
    preview = page.locator('#s-websearch-preview')
    expect(preview).to_contain_text(expected)
    expect(preview).to_contain_text('Eligible routes: openrouter · openai/gpt-5.2 → ddgs.')
    assert 'fixture-key' not in preview.inner_text()
    source = page.locator('#s-websearch-source')
    selection = (source.input_value(), page.locator('#s-websearch-model').input_value())
    preview.scroll_into_view_if_needed()
    routes.roles.capture(page, f'search-preview-{state}')
    assert selection == ('auto', 'gpt-5.2'), 'rendering a cause changes no control or draft'
    assert not [body for path, body in ui['posts'] if path == '/api/settings'], 'a preview saves nothing'
    assert not ui['errors']


def types_module(name):
    import types

    return types.ModuleType(name)
