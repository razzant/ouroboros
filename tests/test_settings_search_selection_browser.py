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


_DEEPSEEK_KEYS = {'DEEPSEEK_API_KEY': 'deepseek-fixture', 'OUROBOROS_CONNECTIONS': [
    {'kind': 'deepseek', 'name': 'work', 'label': 'Work', 'api_key': 'work-fixture'},
    {'kind': 'deepseek', 'name': 'old', 'api_key': 'old-fixture', 'enabled': False}]}
_POINTER = 'Add a key in Accounts for more'
_MODEL_HELP = 'Only the built-in web search uses it. Empty follows the model Ouroboros recommends for each source.'
_KEY_HELP = 'DeepSeek search takes no model: choose the key it uses.'


def _fields(page):
    """The Model and Key fields of the web search page (each a label with its control and its own help)."""
    return page.locator('label.ui-field:has(#s-websearch-model)'), page.locator('label.ui-field:has(#s-websearch-key)')


def _help_under(field, control):
    """A field's help sits in its own column right under its control (DESIGN.md: help belongs to that field)."""
    box, help_box = control.bounding_box(), field.locator('.ui-field-help').bounding_box()
    assert abs(help_box['x'] - box['x']) < 1 and help_box['y'] >= box['y'] + box['height'], (box, help_box)


@pytest.mark.parametrize('theme', ['light', 'dark'])
def test_an_explicit_deepseek_source_from_the_real_page_to_its_named_connection(saved_ui, monkeypatch, tmp_path, theme):
    """Owner decision (11 Nov 2026): DeepSeek search is only an explicit Source. Auto's preview never names it. Choosing
    it puts a Key select in place of the Model field, with the keys of Accounts labelled as every source select labels
    them (a source is chosen, never spelled: DESIGN.md); Save and Reload keep the chosen key, and a task's web_search
    then sends the query once on that connection's key, never the built-in one."""
    import httpx
    from playwright.sync_api import expect

    from ouroboros import net_transport

    ui = saved_ui
    page, preview, field = _open_models(ui, {'OUROBOROS_WEBSEARCH_BACKEND': 'auto', 'OUROBOROS_WEBSEARCH_MODEL': 'gpt-5.5',
                                             **_DEEPSEEK_KEYS}, theme=theme)
    source, key = page.locator('#s-websearch-source'), page.locator('#s-websearch-key')
    model_field, key_field = _fields(page)
    expect(preview).to_contain_text('Uses OpenAI, then OpenRouter, then Anthropic, then DuckDuckGo.')
    expect(preview).not_to_contain_text('DeepSeek')  # DeepSeek keys alone add nothing to Auto
    expect(key_field).to_be_hidden()
    expect(model_field.locator('.ui-field-help')).to_have_text(_MODEL_HELP)
    _help_under(model_field, field)
    source.select_option('deepseek')
    expect(model_field).to_be_hidden()
    expect(key_field.locator('span').first).to_have_text('Key')
    expect(key_field.locator('.ui-field-help')).to_have_text(_KEY_HELP)
    _help_under(key_field, key)
    expect(key.locator('option')).to_have_text(['DeepSeek', 'DeepSeek · Work', _POINTER])  # the off key is not offered
    expect(key).to_have_value('api:deepseek')
    # The saved model of another source stays as saved and is said not to apply.
    expect(preview).to_contain_text('Uses DeepSeek. DeepSeek takes no model, so the saved model gpt-5.5 is not applied.')
    expect(preview).to_contain_text('DeepSeek · built-in key.')
    key.select_option(label='DeepSeek · Work')
    expect(preview).to_contain_text('DeepSeek · connection work.')  # in Details: the key the search will use
    preview.locator('summary').click()
    key.scroll_into_view_if_needed()
    routes.roles.capture(page, f'search-deepseek-key-chosen-{theme}')
    page.set_viewport_size({'width': 390, 'height': 844})
    key.scroll_into_view_if_needed()
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')  # the Key select fits a phone
    routes.roles.capture(page, f'search-deepseek-key-chosen-{theme}-narrow')
    page.set_viewport_size({'width': 1360, 'height': 900})
    _save_and_reload(page)
    on_disk = _on_disk()
    assert (on_disk.get('OUROBOROS_WEBSEARCH_BACKEND'), on_disk.get('OUROBOROS_WEBSEARCH_MODEL')) == ('deepseek', 'deepseek@work::')
    expect(source).to_have_value('deepseek')
    expect(key).to_have_value('api:deepseek@work')
    expect(field).to_be_hidden()
    expect(preview).to_contain_text('DeepSeek · connection work.')
    routes.roles.capture(page, f'search-deepseek-key-reloaded-{theme}')
    source.select_option('auto')  # the Model field and its own words come back; DeepSeek's key spelling does not
    expect(model_field).to_be_visible()
    expect(field).to_have_value('')
    expect(model_field.locator('.ui-field-help')).to_have_text(_MODEL_HELP)
    expect(key_field).to_be_hidden()
    source.select_option('deepseek')

    sent = []

    def respond(request):
        sent.append((str(request.url), request.headers['authorization'], json.loads(request.content)))
        return httpx.Response(200, json={'results': [{'title': 'Found', 'url': 'https://example.invalid/', 'summary': 'It.'}]})

    monkeypatch.setattr(net_transport, 'env_proxies_configured', lambda: False)
    monkeypatch.setattr(net_transport, 'remote_httpx_transport', lambda *a, **k: httpx.MockTransport(respond))
    result, found, _stood_in = _dispatch(monkeypatch, tmp_path, {})(query='  site:example.invalid found  ')
    assert sent == [('https://api.deepseek.com/v1/tools/search', 'Bearer work-fixture', {'query': '  site:example.invalid found  '})]
    assert 'error' not in found and found['backend'] == 'deepseek_search_tool' and found['route']['legs'][0]['connection'] == 'work'
    assert not ui['errors'].unexpected(), ui['errors'].report()


_UNREADABLE = 'Saved choice (not readable)'
_INVALID_NAME = 'has an invalid connection name; choose its key under Web search in Settings'


@pytest.mark.parametrize('saved,shown,said', [
    ('deepseek@old::', 'DeepSeek · old (off)', 'Not available now: DeepSeek connection old (not configured).'),
    ('deepseek@gone::', 'DeepSeek · gone (removed)', 'Not available now: DeepSeek connection gone (not configured).'),
    ('deepseek@work', _UNREADABLE, 'The saved DeepSeek choice (deepseek@work) names no key'),
    ('deepseek@Work::', _UNREADABLE, f'The saved DeepSeek choice (deepseek@Work::) {_INVALID_NAME}'),
], ids=['off', 'removed', 'misspelt', 'name-case'])
def test_a_saved_deepseek_key_that_cannot_send_stays_chosen_until_the_owner_chooses_another(saved_ui, request, saved, shown,
                                                                                            said):
    """A saved key that is off, removed or misspelt, or whose name the key grammar refuses, is never read as the
    built-in key: it stays chosen and says why, the keys of Accounts are listed once each beside it, a Save before any
    choice leaves the saved value as it was, and the owner's own choice of another key replaces it on Save."""
    from playwright.sync_api import expect

    ui = saved_ui
    page, preview, field = _open_models(ui, {'OUROBOROS_WEBSEARCH_BACKEND': 'deepseek', 'OUROBOROS_WEBSEARCH_MODEL': saved,
                                             **_DEEPSEEK_KEYS})
    key = page.locator('#s-websearch-key')
    expect(key.locator('option:checked')).to_have_text(shown)
    expect(key.locator('option')).to_have_text([shown, 'DeepSeek', 'DeepSeek · Work', _POINTER] if shown == _UNREADABLE
                                                else ['DeepSeek', 'DeepSeek · Work', shown, _POINTER])
    expect(preview).to_contain_text(said)
    expect(field).to_have_value(saved)
    key.scroll_into_view_if_needed()
    routes.roles.capture(page, f'search-deepseek-key-{request.node.callspec.id}')
    assert press_editor_save(page) == 'Nothing to save.'
    assert (_on_disk().get('OUROBOROS_WEBSEARCH_BACKEND'), _on_disk().get('OUROBOROS_WEBSEARCH_MODEL')) == ('deepseek', saved)
    key.select_option(label='DeepSeek')
    expect(preview).to_contain_text('DeepSeek · built-in key.')
    _save_and_reload(page)
    assert (_on_disk().get('OUROBOROS_WEBSEARCH_BACKEND'), _on_disk().get('OUROBOROS_WEBSEARCH_MODEL')) == ('deepseek', '')
    expect(key).to_have_value('api:deepseek')
    expect(key.locator('option')).to_have_text(['DeepSeek', 'DeepSeek · Work', _POINTER])
    assert not ui['errors'].unexpected(), ui['errors'].report()


def test_a_held_deepseek_choice_offers_the_keys_added_in_accounts_meanwhile(saved_ui):
    """The Key select's pointer leads to Accounts while the page holds an unsaved DeepSeek choice. Keys added there
    meanwhile are offered on return; the held choice stays unsaved and the one now chosen is what Save writes."""
    from playwright.sync_api import expect

    ui = saved_ui
    page, preview, _field = _open_models(ui, {'OUROBOROS_WEBSEARCH_BACKEND': 'auto', 'OUROBOROS_WEBSEARCH_MODEL': ''})
    source, key = page.locator('#s-websearch-source'), page.locator('#s-websearch-key')
    source.select_option('deepseek')  # no DeepSeek key yet: the built-in one is said to be missing
    expect(key.locator('option')).to_have_text(['DeepSeek (no key)', _POINTER])
    open_settings(page, 'accounts')  # the unsaved page is put aside as it was left
    task_settings.save_owner({**_KEYS, 'OUROBOROS_MODEL': 'openrouter::fixture-main', **_DEEPSEEK_KEYS})
    open_settings(page, 'models/role/websearch')
    expect(source).to_have_value('deepseek')
    expect(page.locator('[data-settings-page="models/role/websearch"] [data-editor-discard]')).to_be_visible()  # unsaved
    expect(key.locator('option')).to_have_text(['DeepSeek', 'DeepSeek · Work', _POINTER])
    expect(preview).to_contain_text('DeepSeek · built-in key.')
    assert _on_disk().get('OUROBOROS_WEBSEARCH_BACKEND', 'auto') == 'auto' and not _on_disk().get('OUROBOROS_WEBSEARCH_MODEL')
    key.select_option(label='DeepSeek · Work')
    key.scroll_into_view_if_needed()
    routes.roles.capture(page, 'search-deepseek-held-choice-new-key')
    _save_and_reload(page)
    assert (_on_disk().get('OUROBOROS_WEBSEARCH_BACKEND'), _on_disk().get('OUROBOROS_WEBSEARCH_MODEL')) == ('deepseek', 'deepseek@work::')
    expect(key).to_have_value('api:deepseek@work')
    open_settings(page, 'models')  # the same words on the Models page's Web search row
    expect(page.locator('[data-settings-page="models"]')).to_contain_text('DeepSeek runs only when you choose it.')
    routes.roles.capture(page, 'models-page-web-search-row')
    assert not ui['errors'].unexpected(), ui['errors'].report()


# A translated install (the house idiom: the gateway's /api/ui/i18n answer, stubbed). Each field's help is its own
# sentence the overlay translates whole, and the Key select's words translate like any option.
_HELP_RU = 'Используется только встроенным веб-поиском. Пустое поле: модель, которую Ouroboros рекомендует для каждого источника.'
_KEY_HELP_RU = 'Поиску DeepSeek модель не нужна: выберите ключ, который он использует.'
_TRANSLATED = {'language': 'ru', 'english': False, 'revision': 1, 'entries': {
    english: {'text': text, 'provenance': 'imported'} for english, text in (
        (_MODEL_HELP, _HELP_RU), (_KEY_HELP, _KEY_HELP_RU),
        ('Recommended for each source', 'Рекомендуемая для каждого источника'), ('Key', 'Ключ'),
        (_POINTER, 'Добавьте ключ в Accounts, чтобы их стало больше'))}}
_COUNT_HELP_WRITES = """() => {
    window.__helpWrites = 0;
    const observer = new MutationObserver((records) => { window.__helpWrites += records.length; });
    document.querySelectorAll('.models-form .ui-field-help').forEach((help) => observer.observe(help,
        { childList: true, subtree: true, characterData: true }));
}"""


def test_a_translated_search_model_help_changes_with_the_source_and_survives_typing(saved_ui):
    """Each field's help and the Model placeholder are the page's own English, so a translated install shows their
    translation; typing or choosing a key never rewrites them, and switching to DeepSeek and back shows each field's
    own translated help, the Key select's words translated too."""
    from playwright.sync_api import expect

    ui = saved_ui
    ui['page'].route('**/api/ui/i18n', lambda route: route.fulfill(json=_TRANSLATED))
    ui['page'].route('**/api/ui/i18n/missing', lambda route: route.fulfill(json={'ok': True}))
    page, preview, field = _open_models(ui, {'OUROBOROS_WEBSEARCH_BACKEND': 'auto', **_DEEPSEEK_KEYS})
    source, key = page.locator('#s-websearch-source'), page.locator('#s-websearch-key')
    model_field, key_field = _fields(page)
    expect(model_field.locator('.ui-field-help')).to_have_text(_HELP_RU)
    expect(field).to_have_attribute('placeholder', 'Рекомендуемая для каждого источника')
    page.evaluate(_COUNT_HELP_WRITES)
    field.press_sequentially('gpt-5.5')
    expect(preview).to_contain_text('OpenAI · gpt-5.5')  # every keystroke has refreshed the preview
    assert page.evaluate('window.__helpWrites') == 0
    expect(model_field.locator('.ui-field-help')).to_have_text(_HELP_RU)
    field.fill('')
    source.select_option('deepseek')
    expect(key_field.locator('.ui-field-help')).to_have_text(_KEY_HELP_RU)
    expect(key_field.locator('span').first).to_have_text('Ключ')
    expect(key.locator('option')).to_have_text(['DeepSeek', 'DeepSeek · Work', 'Добавьте ключ в Accounts, чтобы их стало больше'])
    page.evaluate(_COUNT_HELP_WRITES)
    key.select_option(label='DeepSeek · Work')
    expect(preview).to_contain_text('DeepSeek · connection work.')
    assert page.evaluate('window.__helpWrites') == 0
    help_ = key_field.locator('.ui-field-help')
    help_.scroll_into_view_if_needed()
    routes.roles.capture(page, 'search-deepseek-translated-help')
    page.set_viewport_size({'width': 390, 'height': 844})
    key.scroll_into_view_if_needed()
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    routes.roles.capture(page, 'search-deepseek-translated-help-narrow')
    page.set_viewport_size({'width': 1360, 'height': 900})
    source.select_option('auto')
    expect(model_field.locator('.ui-field-help')).to_have_text(_HELP_RU)
    expect(field).to_have_attribute('placeholder', 'Рекомендуемая для каждого источника')
    assert not ui['errors'].unexpected(), ui['errors'].report()


def types_module(name):
    import types

    return types.ModuleType(name)
