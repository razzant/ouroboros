"""Settings search edits travel through the real binder, preview and dispatch."""
import json
from types import SimpleNamespace

import pytest

from tests import test_model_route_contribution_browser as routes

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
subscription_ui = routes.subscription_ui
role_ui = routes.role_ui
route_ui = routes.route_ui


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
