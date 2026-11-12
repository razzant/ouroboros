"""Settings search edits travel through the real binder, preview and dispatch."""
import json
import sys
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.search_routes import SEARCH_BACKENDS, SEARCH_MODELS
from tests import test_model_route_contribution_browser as routes
from tests import test_task_settings_snapshot as task_settings
from tests.settings_browser_helpers import open_settings, press_editor_save, commit_row, top_path

pytestmark = [pytest.mark.ui_browser]
subscription_ui = routes.subscription_ui
role_ui = routes.role_ui
route_ui = routes.route_ui
isolated_runtime = task_settings.isolated_runtime  # the shared isolated, sealed root


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
    open_settings(page, 'models/role/websearch')
    preview = page.locator('[data-websearch-preview]')
    expect(preview).to_contain_text(SEARCH_MODELS['anthropic'])
    field = page.locator('#s-websearch-model')
    expect(field).to_have_value('claude-opus-4-6')
    if edit in {'model', 'same_model'}:
        field.fill('claude-future-native' if edit == 'model' else 'claude-opus-4-6')
        field.blur()
    elif edit == 'source':
        page.locator('#s-websearch-source').select_option('anthropic')
    expected = (SEARCH_MODELS['anthropic'] if edit == 'untouched' else
                'claude-future-native' if edit == 'model' else 'claude-opus-4-6')
    expect(preview).to_contain_text(f'Anthropic · {expected}')
    if edit != 'untouched':
        expect(preview).not_to_contain_text('does not apply')
    if edit == 'model':
        preview.scroll_into_view_if_needed()
        routes.roles.capture(page, 'anthropic-native-edited-preview')
    press_editor_save(page)
    saved = ui['settings']
    if edit == 'untouched':
        assert not [body for path, body in ui['posts'] if path == '/api/settings']
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
    open_settings(page, 'models')
    open_settings(page, 'models/role/websearch')
    expect(field).to_have_value(saved['OUROBOROS_WEBSEARCH_MODEL'])
    expect(preview).to_contain_text(f'Anthropic · {expected}')
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
        expected = 'Not available now: OpenAI (no API key), Anthropic (no API key).'
    else:
        monkeypatch.setitem(sys.modules, 'anthropic', None)
        ui['settings'].update(ANTHROPIC_API_KEY='fixture-key', OPENAI_API_KEY='fixture-key',
                              OPENAI_BASE_URL='https://compatible.invalid/v1')
        expected = 'Not available now: OpenAI (OPENAI_BASE_URL is set), Anthropic (package not installed).'
    monkeypatch.setitem(sys.modules, 'ddgs', types_module('ddgs'))
    page = ui['page']
    page.goto(ui['url'] + '/#settings')
    open_settings(page, 'models/role/websearch')
    preview = page.locator('[data-websearch-preview]')
    expect(preview).to_contain_text(expected)
    # The sources by their option words in the main line; the model and the gaps behind its Details.
    expect(preview).to_contain_text('Uses OpenRouter, then DuckDuckGo.')
    expect(preview).to_contain_text('OpenRouter · openai/gpt-5.2.')
    assert 'fixture-key' not in preview.inner_text()
    source = page.locator('#s-websearch-source')
    selection = (source.input_value(), page.locator('#s-websearch-model').input_value())
    preview.scroll_into_view_if_needed()
    routes.roles.capture(page, f'search-preview-{state}')
    assert selection == ('auto', 'gpt-5.2'), 'rendering a cause changes no control or draft'
    assert not [body for path, body in ui['posts'] if path == '/api/settings'], 'a preview saves nothing'
    assert not ui['errors']


# The real page on the real settings endpoints: Save writes the real document, Reload reads it back and a task's
# web_search resolves from it through the real registry. Stood in: the provider send helpers and what reaches past
# this test (the LAN address probe, the supervisor start, a save's hot-reload effects).

_KEYS = {'OPENAI_API_KEY': 'openai-fixture', 'OPENROUTER_API_KEY': 'router-fixture', 'ANTHROPIC_API_KEY': 'anthropic-fixture'}


@pytest.fixture
def saved_ui(role_ui, isolated_runtime, monkeypatch):
    from ouroboros import config
    from ouroboros.gateway import settings as gateway

    monkeypatch.setitem(sys.modules, 'ddgs', types_module('ddgs'))
    monkeypatch.setattr(gateway, '_build_network_meta', lambda *_a, **_k: {})
    monkeypatch.setattr(gateway, '_start_supervisor_if_needed_for_request', lambda *_a, **_k: False)
    monkeypatch.setattr(gateway, '_apply_settings_save_side_effects', lambda *_a, **_k: None)
    app = Starlette(routes=[Route('/api/settings', gateway.api_settings_get, methods=['GET']),
                            Route('/api/settings', gateway.api_settings_post, methods=['POST'])])
    app.state.drive_root, app.state.repo_dir = config.DATA_DIR, config.REPO_DIR
    with TestClient(app) as client:
        def bridge(route):
            url = urlsplit(route.request.url)
            path = url.path + ('?' + url.query if url.query else '')
            response = (client.post(path, json=route.request.post_data_json) if route.request.method == 'POST' else client.get(path))
            route.fulfill(status=response.status_code, content_type='application/json', body=response.text)
        role_ui['page'].route('**/api/settings', bridge)
        role_ui['page'].route('**/api/settings?*', bridge)
        yield role_ui


def _open_models(ui, document, *, theme=''):
    task_settings.save_owner({**_KEYS, 'OUROBOROS_MODEL': 'openrouter::fixture-main', **document})
    page = ui['page']
    if theme:
        page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)})")
    page.goto(ui['url'] + '/#settings')
    open_settings(page, 'models/role/websearch')
    return page, page.locator('[data-websearch-preview]'), page.locator('#s-websearch-model')


def _save_and_reload(page):
    if top_path(page) == 'models/role/websearch':
        press_editor_save(page)
    page.reload()
    open_settings(page, 'models/role/websearch')


def _unrelated_save(page):
    open_settings(page, 'tools', 'OUROBOROS_TOOL_TIMEOUT_SEC')
    assert commit_row(page, 'OUROBOROS_TOOL_TIMEOUT_SEC', 601).startswith('Saved')


def _on_disk():
    from ouroboros import config

    return json.loads(config.SETTINGS_PATH.read_text(encoding='utf-8'))


def _dispatch(monkeypatch, tmp_path, calls_as):
    """``call(**args)`` runs one task's web_search on the saved document; every leg acts as ``calls_as[source]``."""
    from ouroboros import config, subagent_runtime
    from ouroboros.tools import search
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    calls = []

    def leg(source):
        def send(_ctx, _query, **kwargs):
            model = kwargs.get('model', '').removeprefix('anthropic::')
            calls.append((source, model))
            outcome = calls_as.get(source, 'answer')
            if outcome == 'refuse':
                error = RuntimeError('HTTP 400 this model is refused')
                error.status_code = 400
                raise error
            if outcome == 'unknown':
                return search._unknown_provider_outcome(SEARCH_BACKENDS[source])
            return json.dumps({'answer': f'{source} answered', 'answer_type': 'summary', 'sources': [], 'backend': source})
        return send

    for source in ('openai', 'openrouter', 'anthropic'):
        monkeypatch.setattr(search, f'_web_search_{source}', leg(source))
    monkeypatch.setattr(search, '_web_search_ddgs', lambda _query: calls.append(('ddgs', '')) or json.dumps(
        {'answer': 'ddgs answered', 'answer_type': 'summary', 'sources': [], 'backend': 'ddgs'}))
    repo, data = tmp_path / 'dispatch-repo', tmp_path / 'dispatch-data'
    repo.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ToolContext(repo_dir=repo, drive_root=data, task_id='task-search-dispatch'))

    def call(**args):
        calls.clear()
        with config.task_settings_scope(subagent_runtime.apply_task_start_settings()):
            result = registry.execute_result('web_search', {'query': 'q', **args})
        return result, json.loads(result.text), list(calls)
    return call


@pytest.mark.parametrize('theme', ['light', 'dark'])
def test_a_blank_search_model_stays_blank_from_the_real_page_to_dispatch(saved_ui, monkeypatch, tmp_path, theme):
    """Owner decision (1 Nov 2026): an empty Model follows the model Ouroboros recommends for each source. An
    unrelated Save writes it back blank, the reload still shows it blank, and a task then sends the recommended
    models; a model the call names is mandatory, and an unknown outcome is never sent again."""
    from playwright.sync_api import expect

    ui = saved_ui
    page, preview, field = _open_models(ui, {'OUROBOROS_WEBSEARCH_BACKEND': 'auto'}, theme=theme)
    models = [f"OpenAI · {SEARCH_MODELS['openai']} (recommended)", f"OpenRouter · {SEARCH_MODELS['openrouter']} (recommended)",
              f"Anthropic · {SEARCH_MODELS['anthropic']} (recommended)"]
    expect(preview).to_contain_text('Uses OpenAI, then OpenRouter, then Anthropic, then DuckDuckGo.')
    expect(preview).to_contain_text('; '.join(models) + '.')
    expect(field).to_have_value('')
    assert page.evaluate("document.documentElement.dataset.theme || document.documentElement.getAttribute('data-theme')") == theme
    browser = page.context.browser
    print('ENGINE_RECEIPT', json.dumps({'engine': browser.browser_type.name, 'version': browser.version, 'theme': theme,
                                        'user_agent': page.evaluate('navigator.userAgent')}))
    preview.scroll_into_view_if_needed()
    routes.roles.capture(page, f'search-blank-real-{browser.browser_type.name}-{theme}-wide')
    page.set_viewport_size({'width': 390, 'height': 844})
    field.scroll_into_view_if_needed()
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    routes.roles.capture(page, f'search-blank-real-{browser.browser_type.name}-{theme}-narrow')
    page.set_viewport_size({'width': 1360, 'height': 900})
    _unrelated_save(page)
    _save_and_reload(page)
    on_disk = _on_disk()
    assert on_disk.get('OUROBOROS_TOOL_TIMEOUT_SEC') == 601 and on_disk.get('OUROBOROS_WEBSEARCH_MODEL', '') == ''
    expect(field).to_have_value('')
    expect(preview).to_contain_text(f"Anthropic · {SEARCH_MODELS['anthropic']} (recommended)")

    acts = {}
    call = _dispatch(monkeypatch, tmp_path, acts)
    _result, found, sent = call()
    assert sent == [('openai', SEARCH_MODELS['openai'])] and found['answer'] == 'openai answered'
    assert all(leg['model_source'] == 'provider_default' for leg in found['route']['legs'])
    # A named model: a routed one has its one leg; a refusal brings in nothing else.
    acts.update(anthropic='refuse')
    refused, body, sent = call(model='anthropic::named-future')
    assert refused.code == 'TOOL_ERROR' and sent == [('anthropic', 'named-future')] and 'answer' not in body
    # A native id under Auto may use the same model's other provider, never another model or ddgs.
    acts.update(openai='refuse', openrouter='refuse')
    refused, body, sent = call(model='native-future')
    assert refused.code == 'TOOL_ERROR' and sent == [('openai', 'native-future'), ('openrouter', 'openai/native-future')]
    # An unknown outcome is never sent again, on a named call or an omitted one.
    acts.update(openai='unknown', openrouter='answer')
    for args, expected in (({'model': 'native-future'}, ('openai', 'native-future')), ({}, ('openai', SEARCH_MODELS['openai']))):
        _result, body, sent = call(**args)
        assert sent == [expected] and body['reason_code'] == 'provider_outcome_unknown'
    assert not ui['errors'].unexpected(), ui['errors'].report()


@pytest.mark.parametrize('case,document,edit,saved,reloaded_preview,first_sent', [
    # An old shipped value is a saved choice like any other: it survives an unrelated Save and keeps its legs.
    ('legacy', {'OUROBOROS_WEBSEARCH_MODEL': 'gpt-5.2'}, None, 'gpt-5.2', 'OpenAI · gpt-5.2;', ('openai', 'gpt-5.2')),
    # The Anthropic pin keeps its legacy reading of a saved bare id: recommended, and said to be unapplied.
    ('legacy_pin', {'OUROBOROS_WEBSEARCH_BACKEND': 'anthropic', 'OUROBOROS_WEBSEARCH_MODEL': 'claude-opus-4-6'}, None,
     'claude-opus-4-6', 'Anthropic does not apply the saved model claude-opus-4-6', ('anthropic', SEARCH_MODELS['anthropic'])),
    # Clearing the field is the owner's move onto the recommendations; typing authors a choice.
    ('cleared', {'OUROBOROS_WEBSEARCH_MODEL': 'gpt-5.2'}, '', '', f"OpenAI · {SEARCH_MODELS['openai']} (recommended)",
     ('openai', SEARCH_MODELS['openai'])),
    ('edited', {}, 'native-future', 'native-future', 'OpenAI · native-future;', ('openai', 'native-future')),
    # A single-provider install left blank: its own source's recommended model, whatever Main is.
    ('direct_anthropic', {'OPENAI_API_KEY': '', 'OPENROUTER_API_KEY': '', 'OUROBOROS_MODEL': 'anthropic::fixture-main'}, None, '',
     f"Anthropic · {SEARCH_MODELS['anthropic']} (recommended)", ('anthropic', SEARCH_MODELS['anthropic'])),
])
def test_a_saved_search_choice_keeps_its_meaning_from_the_real_page_to_dispatch(
        saved_ui, monkeypatch, tmp_path, case, document, edit, saved, reloaded_preview, first_sent):
    from playwright.sync_api import expect

    ui = saved_ui
    page, preview, field = _open_models(ui, document)
    expect(field).to_have_value(document.get('OUROBOROS_WEBSEARCH_MODEL', ''))
    if edit is not None:
        field.fill(edit)
        field.blur()
        expect(preview).to_contain_text(reloaded_preview)  # the draft preview already reads as the save will
    else:
        _unrelated_save(page)
    _save_and_reload(page)
    assert _on_disk().get('OUROBOROS_WEBSEARCH_MODEL', '') == saved
    expect(field).to_have_value(saved)
    expect(preview).to_contain_text(reloaded_preview)
    if case == 'cleared':
        routes.roles.capture(page, 'search-cleared-real-saved-reloaded')
    _result, found, sent = _dispatch(monkeypatch, tmp_path, {})()
    assert sent == [first_sent] and 'error' not in found
    assert not ui['errors'].unexpected(), ui['errors'].report()


def types_module(name):
    import types

    return types.ModuleType(name)
