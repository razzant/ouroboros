"""One resolver's projection is consumed by context, preview and real tool dispatch."""
import json
import sys
from types import SimpleNamespace

import pytest

from ouroboros.search_routes import SEARCH_MODELS, resolve_web_search_route
from ouroboros.tools import search
from tests import test_extensions_api as extension_fixtures
from tests.test_extensions_api import _write_ext

_clean_extensions = extension_fixtures._clean_extensions
RECOMMENDED = SEARCH_MODELS['anthropic']  # the curated table's Anthropic row, never a literal here


@pytest.fixture(autouse=True)
def _anthropic_sdk_present(monkeypatch):
    """The optional Anthropic SDK is a chosen fact here, not whatever this interpreter has:
    a stand-in module makes it discoverable and importable; a test sets None for absence.
    ddgs keeps following each test's own sys.modules entry."""
    import types

    monkeypatch.setitem(sys.modules, "anthropic", sys.modules.get("anthropic") or types.ModuleType("anthropic"))


def test_generic_extension_search_executes_when_builtin_search_is_unavailable(tmp_path, monkeypatch, _clean_extensions):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, find_skill, save_enabled, save_review_state
    from ouroboros.tools.registry import ToolRegistry
    skills_root, drive_root = tmp_path / 'skills', tmp_path / 'drive'
    drive_root.mkdir()
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    monkeypatch.setenv('OUROBOROS_SKILLS_REPO_PATH', str(skills_root))
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_BACKEND', 'openai')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    plugin = ("def search(ctx, query=''):\n    return 'extension result: ' + query\n"
              "def register(api):\n    api.register_tool('search', search, description='Independent search', schema={}, timeout_sec=10)\n")
    path = _write_ext(skills_root, 'research_provider', permissions=['tool'], plugin=plugin)
    save_enabled(drive_root, 'research_provider', True)
    save_review_state(drive_root, 'research_provider', SkillReviewState(status='pass',
        content_hash=compute_content_hash(path, manifest_entry='plugin.py')))
    skill = find_skill(drive_root, 'research_provider', repo_path=str(skills_root))
    assert extension_loader.load_extension(skill, lambda: {}, drive_root=drive_root) is None
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=drive_root)
    registry._ctx.task_metadata = {'work_kind': 'owner'}  # an owner task's minted kind
    name = extension_loader.extension_surface_name('research_provider', 'search')
    assert search._available_web_search_backends() == []
    assert registry.get_schema_by_name(name)['function']['name'] == name
    assert 'extension result: independent query' in registry.execute(name, {'query': 'independent query'})


@pytest.mark.parametrize('source,selected,expected', [
    # A blank saved Model is the recommended one of every source Auto tries.
    ('auto', '', [('openai', SEARCH_MODELS['openai']), ('openrouter', SEARCH_MODELS['openrouter']), ('anthropic', RECOMMENDED)]),
    ('auto', 'future', [('openai', 'future'), ('openrouter', 'openai/future'), ('anthropic', RECOMMENDED)]),
    ('auto', 'vendor/future', [('openrouter', 'vendor/future'), ('anthropic', RECOMMENDED)]),
    ('openai', 'anthropic::future', []), ('openai', 'vendor/future', []),
])
def test_dispatch_uses_resolved_candidates_and_truthful_failed_legs(monkeypatch, source, selected, expected):
    for key in ('OPENAI_API_KEY', 'OPENROUTER_API_KEY', 'ANTHROPIC_API_KEY'):
        monkeypatch.setenv(key, 'synthetic')
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_BACKEND', source)
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_MODEL', selected)
    monkeypatch.setitem(sys.modules, 'ddgs', None)
    calls = []
    def call(name):
        def failed(_ctx, _query, **kwargs):
            model = kwargs['model'].removeprefix('anthropic::')
            calls.append((name, model))
            error = RuntimeError('explicit terminal refusal')
            error.ledger_attempt_ids = [name + '-attempt']
            raise error
        return failed
    for name in ('openai', 'openrouter', 'anthropic'):
        monkeypatch.setattr(search, '_web_search_' + name, call(name))
    route = resolve_web_search_route()
    result = json.loads(search._web_search(SimpleNamespace(task_metadata={}), 'query'))
    assert calls == expected
    assert [(leg['source'], leg['model']) for leg in route['legs']] == expected
    assert result['route'] == route
    assert [(leg['source'], leg['model']) for leg in result['legs_tried']] == expected
    assert all(leg['outcome'] == 'failed' and leg['ledger_attempt_ids'] == [leg['source'] + '-attempt'] for leg in result['legs_tried'])


def test_openrouter_only_legacy_auto_and_context_projection_preserve_saved_default(tmp_path, monkeypatch):
    from ouroboros.context import build_runtime_section
    from tests._context_shared import _make_health_env
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'synthetic')
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_MODEL', 'gpt-5.2')
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_BACKEND', 'auto')
    monkeypatch.setitem(sys.modules, 'ddgs', None)
    route = resolve_web_search_route()
    assert [(leg['source'], leg['model']) for leg in route['legs']] == [('openrouter', 'openai/gpt-5.2')]
    env = _make_health_env(tmp_path)
    payload = json.loads(build_runtime_section(env, {'id': 'search-task', 'type': 'task'}).split('\n\n', 1)[1])
    assert payload['capabilities']['web_search_route'] == route
    assert route['model'] == 'gpt-5.2'
    schema = search.get_tools()[0].schema
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_MODEL', 'anthropic::future')
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_BACKEND', 'anthropic')
    assert search.get_tools()[0].schema == schema
    assert 'skills' in schema['description'].lower() and 'MCP' in schema['description']


@pytest.mark.parametrize('saved,sent,source', [('anthropic/claude-opus-4-6', 'claude-opus-4-6', 'selection'),
                                               ('vendor/future', RECOMMENDED, 'provider_default')])
def test_auto_sends_a_saved_legacy_anthropic_model_on_the_direct_anthropic_leg(monkeypatch, saved, sent, source):
    from ouroboros import llm
    for key in ('OPENAI_API_KEY', 'OPENROUTER_API_KEY', 'OPENAI_BASE_URL'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'synthetic')
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_BACKEND', 'auto')
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_MODEL', saved)
    monkeypatch.setitem(sys.modules, 'ddgs', None)
    requests = []
    def server_tool(**kwargs):
        requests.append(kwargs['model'])
        return SimpleNamespace(content=[SimpleNamespace(type='text', text='found')], usage=None, model=kwargs['model'])
    monkeypatch.setattr(llm, 'anthropic_web_search_server_tool', server_tool)
    route = resolve_web_search_route()
    assert [(leg['source'], leg['model'], leg['model_source']) for leg in route['legs']] == [('anthropic', sent, source)]
    result = json.loads(search._web_search(SimpleNamespace(pending_events=[], task_metadata={}), 'query'))
    assert requests == [sent] and result['answer'] == 'found' and result['model'] == sent
    assert [(leg['source'], leg['outcome']) for leg in result['legs_tried']] == [('anthropic', 'succeeded')]


@pytest.mark.parametrize('saved', ['claudexor::codex=future', 'openai-compatible::local', ''])
def test_a_ddgs_pin_ignores_any_saved_model_and_still_searches(monkeypatch, saved):
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_BACKEND', 'ddgs')
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_MODEL', saved)
    monkeypatch.setitem(sys.modules, 'ddgs', SimpleNamespace())
    monkeypatch.setattr(search, '_web_search_ddgs', lambda query: json.dumps({'answer': 'ddgs: ' + query, 'backend': 'ddgs'}))
    route = resolve_web_search_route()
    assert 'error' not in route and [leg['source'] for leg in route['legs']] == ['ddgs']
    assert search._available_web_search_backends() == ['ddgs']
    result = json.loads(search._web_search(SimpleNamespace(pending_events=[], task_metadata={}), 'q'))
    assert result['answer'] == 'ddgs: q' and [leg['outcome'] for leg in result['legs_tried']] == ['succeeded']


@pytest.mark.parametrize('source,saved', [('auto', 'claudexor::codex=future'), ('auto', 'anthropic::vendor/future'), ('anthropic', 'anthropic::vendor/future')])
def test_an_unsupported_selection_is_disclosed_under_auto_and_refused_by_a_strict_source(monkeypatch, source, saved):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'synthetic')
    monkeypatch.setitem(sys.modules, 'ddgs', None)
    route = resolve_web_search_route(backend=source, model=saved)
    if source == 'auto':
        assert route['unapplied_model'] == saved and 'error' not in route
        assert [(leg['source'], leg['model'], leg['model_source']) for leg in route['legs']] == [
            ('anthropic', RECOMMENDED, 'provider_default')]
    else:
        assert route['error'] and not route['legs'] and 'unapplied_model' not in route


@pytest.mark.parametrize('override,expected,origin', [
    (None, RECOMMENDED, 'provider_default'),
    ('claude-opus-4-6', 'claude-opus-4-6', 'selection'),
    ('claude-future-native', 'claude-future-native', 'selection'),
])
def test_anthropic_native_per_call_model_reaches_provider_while_saved_legacy_stays_default(
        monkeypatch, override, expected, origin):
    from ouroboros import llm
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'synthetic')
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_BACKEND', 'anthropic')
    # Even an identical per-call value is an explicit selection, not saved legacy intent.
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_MODEL', 'claude-opus-4-6')
    calls = []
    def server_tool(**kwargs):
        calls.append(kwargs['model'])
        return SimpleNamespace(content=[SimpleNamespace(type='text', text='found')], usage=None, model=kwargs['model'])
    monkeypatch.setattr(llm, 'anthropic_web_search_server_tool', server_tool)
    kwargs = {} if override is None else {'model': override}
    result = json.loads(search._web_search(SimpleNamespace(pending_events=[], task_metadata={}), 'query', **kwargs))
    assert calls == [expected]
    assert result['model'] == expected and result['answer'] == 'found'
    assert result['route']['legs'][0]['model_source'] == origin
    assert result['legs_tried'][0]['model'] == expected


@pytest.mark.parametrize('override', ['openai::future', 'vendor/future', 'anthropic/vendor/future'])
def test_anthropic_pin_refuses_foreign_per_call_model_without_dispatch(monkeypatch, override):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'synthetic')
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_BACKEND', 'anthropic')
    def forbidden(*args, **kwargs):
        pytest.fail('A conflicting model must not dispatch any provider or fallback')
    for name in ('openai', 'openrouter', 'anthropic', 'ddgs'):
        monkeypatch.setattr(search, '_web_search_' + name, forbidden)
    result = json.loads(search._web_search(SimpleNamespace(task_metadata={}), 'query', model=override))
    assert result['route']['error'] and result['legs_tried'] == []


# A model the call names is mandatory (owner decision, 1 Nov 2026): only legs that send it run, a routed model
# or named connection has exactly one, and a known refusal or an unsupported route never brings in another
# model, connection, source or ddgs. The mind gets the failure and decides what to do next.

def _every_source_ready(monkeypatch, source):
    from ouroboros.provider_models import CONNECTIONS_KEY

    for key in ('OPENAI_API_KEY', 'OPENROUTER_API_KEY', 'ANTHROPIC_API_KEY'):
        monkeypatch.setenv(key, 'synthetic')
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    monkeypatch.setenv(CONNECTIONS_KEY, json.dumps([{'kind': 'openai', 'name': 'work', 'api_key': 'work-key'}]))
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_BACKEND', source)
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_MODEL', 'saved-model')  # the saved choice is no substitute either
    monkeypatch.setitem(sys.modules, 'ddgs', SimpleNamespace())


def _refusing_legs(monkeypatch):
    calls = []

    def leg(name):
        def refused(_ctx, _query, **kwargs):
            calls.append((name, kwargs['model'].removeprefix('anthropic::'), kwargs.get('connection', '')))
            error = RuntimeError('HTTP 400 this model is refused')
            error.status_code = 400
            raise error
        return refused

    for name in ('openai', 'openrouter', 'anthropic'):
        monkeypatch.setattr(search, '_web_search_' + name, leg(name))
    monkeypatch.setattr(search, '_web_search_ddgs', lambda query: calls.append(('ddgs', '', '')) or json.dumps(
        {'answer': 'ddgs answer', 'answer_type': 'summary', 'sources': [], 'backend': 'ddgs'}))
    return calls


@pytest.mark.parametrize('source,named,expected', [
    ('auto', 'anthropic::future', [('anthropic', 'future', '')]),
    ('openrouter', 'future', [('openrouter', 'openai/future', '')]),
    # No built-in transport sends these: refused before any send, never replaced by Auto's other legs.
    ('auto', 'claudexor::codex=future', []),
    ('auto', 'openai::vendor/future', []),
    ('ddgs', 'future', []),  # ddgs takes no model, so a named one is refused rather than dropped
])
def test_a_model_the_call_names_is_mandatory(monkeypatch, source, named, expected):
    _every_source_ready(monkeypatch, source)
    calls = _refusing_legs(monkeypatch)
    result = json.loads(search._web_search(SimpleNamespace(pending_events=[], task_metadata={}), 'query', model=named))
    assert calls == expected
    assert 'error' in result and 'answer' not in result
    assert [(leg['source'], leg['model'], leg.get('connection', '')) for leg in result['legs_tried']] == expected
    assert result['route']['model'] == named and 'This call named its model' in result['route']['note']
    if expected:
        assert all(leg['model_source'] == 'selection' for leg in result['route']['legs'])
    else:
        assert result['route']['error'] and result['route']['legs'] == []


def test_a_named_native_model_may_use_its_other_provider_but_an_unknown_outcome_stops(monkeypatch):
    _every_source_ready(monkeypatch, 'auto')
    calls = []

    def openai_refused(_ctx, _query, **kwargs):
        calls.append(('openai', kwargs['model']))
        error = RuntimeError('HTTP 404 model not found')
        error.status_code = 404
        raise error

    def openrouter_answered(_ctx, _query, **kwargs):
        calls.append(('openrouter', kwargs['model']))
        return json.dumps({'answer': 'the same model elsewhere', 'answer_type': 'summary', 'sources': [],
                           'backend': 'openrouter_server_tool'})

    def forbidden(*_args, **_kwargs):
        pytest.fail('a named model never reaches another model or ddgs')

    monkeypatch.setattr(search, '_web_search_openai', openai_refused)
    monkeypatch.setattr(search, '_web_search_openrouter', openrouter_answered)
    monkeypatch.setattr(search, '_web_search_anthropic', forbidden)
    monkeypatch.setattr(search, '_web_search_ddgs', forbidden)
    ctx = SimpleNamespace(pending_events=[], task_metadata={})
    result = json.loads(search._web_search(ctx, 'q', model='future'))
    assert calls == [('openai', 'future'), ('openrouter', 'openai/future')]
    assert result['answer'] == 'the same model elsewhere'
    assert [(leg['source'], leg['outcome']) for leg in result['legs_tried']] == [('openai', 'failed'), ('openrouter', 'succeeded')]
    calls.clear()
    monkeypatch.setattr(search, '_web_search_openai', lambda _ctx, _query, **kwargs: calls.append(('openai', kwargs['model']))
                        or search._unknown_provider_outcome('openai_responses'))
    result = json.loads(search._web_search(ctx, 'q', model='future'))
    assert calls == [('openai', 'future')] and result['reason_code'] == 'provider_outcome_unknown'
    assert [leg['outcome'] for leg in result['legs_tried']] == ['provider_outcome_unknown']


# #1697: an attempted leg, a failed or unavailable one, and the source that answered stay distinct.

def _discoverable(monkeypatch, **packages):
    from ouroboros import search_routes

    monkeypatch.setattr(search_routes, "_package_discoverable", lambda name: packages.get(name, True))


@pytest.mark.parametrize('settings,expected', [
    ({'ANTHROPIC_API_KEY': 'synthetic'}, {'anthropic': 'package_missing', 'openai': 'no_credentials',
                                          'openrouter': 'no_credentials', 'ddgs': 'package_missing'}),
    ({'OPENAI_API_KEY': 'synthetic', 'OPENAI_BASE_URL': 'https://compatible.invalid/v1'},
     {'openai': 'legacy_openai_base_url', 'openrouter': 'no_credentials', 'anthropic': 'no_credentials',
      'ddgs': 'package_missing'}),
])
def test_unavailable_legs_name_their_locally_known_cause(monkeypatch, settings, expected):
    _discoverable(monkeypatch, anthropic=False, ddgs=False)
    route = resolve_web_search_route(settings={'OUROBOROS_WEBSEARCH_BACKEND': 'auto', **settings})
    assert route['legs'] == []
    assert {leg['source']: leg['unavailable_reason'] for leg in route['unavailable']} == expected


def test_a_discoverable_sdk_keeps_the_anthropic_leg_and_eligible_legs_carry_no_reason(monkeypatch):
    _discoverable(monkeypatch, anthropic=True, ddgs=True)
    route = resolve_web_search_route(settings={'OUROBOROS_WEBSEARCH_BACKEND': 'auto', 'ANTHROPIC_API_KEY': 'k'})
    assert [leg['source'] for leg in route['legs']] == ['anthropic', 'ddgs']
    assert all('unavailable_reason' not in leg for leg in route['legs'])


def test_package_discovery_reads_the_import_system_not_the_network(monkeypatch):
    from ouroboros import search_routes

    monkeypatch.setitem(sys.modules, 'anthropic', None)  # a blocked import is not ready
    assert search_routes._package_discoverable('anthropic') is False
    monkeypatch.delitem(sys.modules, 'anthropic')
    monkeypatch.setattr(search_routes.importlib.util, 'find_spec', lambda name: object() if name == 'anthropic' else None)
    assert search_routes._package_discoverable('anthropic') is True
    monkeypatch.setattr(search_routes.importlib.util, 'find_spec', lambda name: None)
    assert search_routes._package_discoverable('anthropic') is False


def test_settings_preview_endpoint_serves_the_unavailable_causes(monkeypatch):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from ouroboros.gateway import settings as settings_mod

    _discoverable(monkeypatch, anthropic=False, ddgs=True)
    saved = {'OUROBOROS_WEBSEARCH_BACKEND': 'auto', 'OPENROUTER_API_KEY': 'sk-or-secret-7f3', 'ANTHROPIC_API_KEY': 'sk-ant-secret-7f3'}
    monkeypatch.setattr(settings_mod, 'settings_read_view', lambda: (dict(saved), {}))
    monkeypatch.setattr(settings_mod, 'apply_runtime_provider_defaults', lambda s: (dict(s), False, []))
    client = TestClient(Starlette(routes=[Route('/api/settings', endpoint=settings_mod.api_settings_get)]))
    data = client.get('/api/settings', params={'websearch_preview': '1'}).json()
    assert [leg['source'] for leg in data['legs']] == ['openrouter', 'ddgs']
    assert {leg['source']: leg['unavailable_reason'] for leg in data['unavailable']} == {
        'openai': 'no_credentials', 'anthropic': 'package_missing'}
    assert 'secret-7f3' not in json.dumps(data)  # causes never carry a secret value
    # No saved Model: the preview is the resolver's route, each leg on its source's recommended model.
    assert data == resolve_web_search_route(settings=dict(saved)) and data['model'] == ''
    assert {leg['source']: (leg['model'], leg['model_source']) for leg in data['legs'] + data['unavailable']} == {
        source: (SEARCH_MODELS[source], 'provider_default') for source in ('openai', 'openrouter', 'anthropic')
    } | {'ddgs': ('', 'provider_default')}


def _progress_ctx():
    rows = []
    return SimpleNamespace(pending_events=[], task_metadata={}, emit_progress_fn=rows.append), rows


def test_fallback_progress_names_the_failure_and_the_source_that_answered(monkeypatch):
    from ouroboros import llm

    for key in ('OPENAI_API_KEY', 'OPENAI_BASE_URL'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'synthetic')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'synthetic')
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_BACKEND', 'auto')
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_MODEL', '')
    _discoverable(monkeypatch, anthropic=True, ddgs=True)
    monkeypatch.setitem(sys.modules, 'anthropic', None)  # discoverable, yet its import fails at dispatch

    def refused(**kwargs):
        error = RuntimeError('HTTP 403 terminal refusal')
        error.status_code = 403
        raise error

    monkeypatch.setattr(llm, 'openrouter_web_search_server_tool', refused)
    monkeypatch.setattr(search, '_web_search_ddgs', lambda query: json.dumps(
        {'answer': 'a', 'answer_type': 'summary', 'sources': [{'url': 'https://example.invalid/a'}], 'backend': 'ddgs'}))
    for query in ('neutral query', 'second neutral query'):
        ctx, rows = _progress_ctx()
        result = json.loads(search._web_search(ctx, query))
        assert result['backend'] == 'ddgs' and result['sources']
        assert [(leg['source'], leg['outcome']) for leg in result['legs_tried']] == [
            ('openrouter', 'failed'), ('anthropic', 'failed'), ('ddgs', 'succeeded')]
        assert 'anthropic' in result['backend_errors'][1].lower() and '(ModuleNotFoundError)' in result['backend_errors'][1]
        # Owner rows classify and point at Logs; the exception text stays in the result only.
        # The SDK import fails before the helper could claim an Anthropic attempt.
        assert rows == [f'🔍 Trying OpenRouter web search: {query}',
                        '↪ Trying Anthropic after OpenRouter web search failed; details in Logs',
                        '↪ Trying DDGS after Anthropic web search failed; details in Logs',
                        '🔍 Answer from DDGS after 2 failed sources']
        assert 'HTTP 403' in result['backend_errors'][0]
        assert not any(row.startswith('🔍 Searching via') for row in rows)


def test_an_unknown_outcome_stops_without_fallback_progress(monkeypatch):
    from ouroboros import llm
    from ouroboros.usage_accounting import PhysicalAttemptCapture

    for key in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'synthetic')
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_BACKEND', 'auto')
    _discoverable(monkeypatch, ddgs=True)

    def lost(**kwargs):
        error = RuntimeError('connection reset after send')
        error.physical_attempt_capture = PhysicalAttemptCapture('lost-attempt', 'openai/gpt-5.2', 'openrouter', 'dispatched', 'opaque')
        raise error

    monkeypatch.setattr(llm, 'openrouter_web_search_server_tool', lost)
    monkeypatch.setattr('ouroboros.transport_custody.outcome_unknown_on_chain', lambda exc: True)
    monkeypatch.setattr(search, '_web_search_ddgs', lambda query: pytest.fail('no fallback after an unknown outcome'))
    ctx, rows = _progress_ctx()
    result = json.loads(search._web_search(ctx, 'q'))
    assert result['reason_code'] == 'provider_outcome_unknown'
    assert [leg['outcome'] for leg in result['legs_tried']] == ['provider_outcome_unknown']
    assert rows == ['🔍 Trying OpenRouter web search: q',
                    '⚠️ Web search outcome unknown (OpenRouter); no further attempt or fallback was sent']


def test_a_pinned_source_without_its_sdk_reports_the_import_failure_before_any_attempt_claim(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'synthetic')
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_BACKEND', 'anthropic')
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_MODEL', '')
    _discoverable(monkeypatch, anthropic=False)
    monkeypatch.setitem(sys.modules, 'anthropic', None)
    ctx, rows = _progress_ctx()
    result = json.loads(search._web_search(ctx, 'q'))
    assert result['route']['unavailable'][0]['unavailable_reason'] == 'package_missing'
    assert [(leg['source'], leg['outcome']) for leg in result['legs_tried']] == [('anthropic', 'failed')]
    assert '(ModuleNotFoundError)' in result['error'] and result['backend'] == 'anthropic'
    assert rows == ['⚠️ Web search failed (Anthropic); no source answered; details in Logs']


def test_a_deadline_stop_is_said_and_sends_nothing_more(monkeypatch):
    from ouroboros import llm

    for key in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'synthetic')
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_BACKEND', 'auto')
    _discoverable(monkeypatch, ddgs=True)
    checks = iter([False, False, True])  # dispatcher, OpenRouter helper, then the next leg's check
    monkeypatch.setattr(search, '_web_search_deadline_exhausted', lambda ctx: next(checks))

    def refused(**kwargs):
        error = RuntimeError('HTTP 429 rate limited')
        error.status_code = 429
        raise error

    monkeypatch.setattr(llm, 'openrouter_web_search_server_tool', refused)
    monkeypatch.setattr(search, '_web_search_ddgs', lambda query: pytest.fail('no send past the deadline'))
    ctx, rows = _progress_ctx()
    result = json.loads(search._web_search(ctx, 'q'))
    assert result['reason_code'] == 'deadline_exhausted'
    assert [leg['outcome'] for leg in result['legs_tried']] == ['failed']
    assert rows == ['🔍 Trying OpenRouter web search: q',
                    '⚠️ Web search stopped: the owner deadline leaves no dispatch window']


# The registry's availability gate reads the same route causes: a present key whose SDK
# is not installed withholds web_search as package_missing, never as a missing key.

def _task_registry(tmp_path):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    repo, data = tmp_path / 'repo', tmp_path / 'data'
    repo.mkdir()
    data.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ToolContext(repo_dir=repo, drive_root=data, task_id='task-search-cause'))
    return registry


def _search_settings(monkeypatch, backend, **keys):
    for key in ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENROUTER_API_KEY', 'ANTHROPIC_API_KEY'):
        monkeypatch.delenv(key, raising=False)
    for key, value in keys.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_BACKEND', backend)
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_MODEL', '')


def _withheld(registry):
    from ouroboros.tool_policy import format_capability_omissions

    names = {schema['function']['name'] for schema in registry.schemas(core_only=True)}
    rows = [row for row in registry.capability_omissions() if 'web_search' in row.get('tools', [])]
    return 'web_search' not in names, rows, format_capability_omissions(rows, header='')


@pytest.mark.parametrize('backend,keys,packages,reason,detail', [
    # A strict Anthropic source with its key but no SDK: the cause is the package, not the key.
    ('anthropic', {'ANTHROPIC_API_KEY': 'synthetic'}, {'anthropic': False},
     'package_missing', 'web_search_backend: anthropic package_missing'),
    # Control: no key at all keeps the existing credential answer.
    ('anthropic', {}, {'anthropic': False}, 'missing_credential', 'web_search_backend: anthropic no_credentials'),
    # Mixed causes are all named, in route order, never collapsed into one missing key.
    ('auto', {'OPENAI_API_KEY': 'synthetic', 'OPENAI_BASE_URL': 'https://compatible.invalid/v1'},
     {'anthropic': False, 'ddgs': False}, 'legacy_openai_base_url+missing_credential+package_missing',
     'web_search_backend: openai legacy_openai_base_url, openrouter no_credentials, '
     'anthropic no_credentials, ddgs package_missing'),
])
def test_registry_withholds_web_search_under_the_route_cause(tmp_path, monkeypatch, backend, keys, packages, reason, detail):
    _search_settings(monkeypatch, backend, **keys)
    _discoverable(monkeypatch, **packages)
    registry = _task_registry(tmp_path)
    hidden, rows, lines = _withheld(registry)
    assert hidden and [row['reason'] for row in rows] == [reason]
    assert rows[0]['details']['web_search'] == detail
    assert lines == [f'- tools: {reason} ({", ".join(rows[0]["tools"])})'] and 'web_search' in rows[0]['tools']
    refused = registry.execute_result('web_search', {'query': 'q'})
    assert refused.code == 'CAPABILITY_UNAVAILABLE'
    assert f"'web_search' is unavailable: {reason} ({detail})." in refused.text
    if reason == 'package_missing':
        assert 'missing_credential' not in refused.text
    assert registry.get_schema_by_name('web_search') is None


def test_registry_offers_web_search_when_a_leg_is_eligible(tmp_path, monkeypatch):
    _search_settings(monkeypatch, 'anthropic', ANTHROPIC_API_KEY='synthetic')
    _discoverable(monkeypatch, anthropic=True)
    registry = _task_registry(tmp_path)
    hidden, rows, _lines = _withheld(registry)
    assert not hidden and rows == []
    assert registry.get_schema_by_name('web_search') is not None


@pytest.mark.parametrize('source,model,named,leg', [('openai', 'openai/gpt-5.2', 'gpt-5.5', ('openai', 'gpt-5.5')),
                                                   ('openrouter', 'anthropic::future', 'vendor/model', ('openrouter', 'vendor/model'))])
def test_an_unsupported_saved_selection_leaves_web_search_to_a_call_that_names_a_model(
        tmp_path, monkeypatch, source, model, named, leg):
    """An unsupported saved Model no longer withholds the tool while its Source can serve: the omitted call
    reports the selection's error and sends nothing; a call naming a valid model reaches its leg."""
    _search_settings(monkeypatch, source, OPENAI_API_KEY='synthetic', OPENROUTER_API_KEY='synthetic')
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_MODEL', model)
    route = resolve_web_search_route()
    assert route['error'] and not route['legs']
    sent = []

    def answered(_ctx, _query, **kwargs):
        sent.append((leg[0], kwargs['model']))
        return json.dumps({'answer': 'found', 'answer_type': 'summary', 'sources': [], 'backend': leg[0]})

    monkeypatch.setattr(search, '_web_search_' + leg[0], answered)
    registry = _task_registry(tmp_path)
    hidden, rows, _lines = _withheld(registry)
    assert not hidden and rows == [] and registry.get_schema_by_name('web_search') is not None
    omitted = registry.execute_result('web_search', {'query': 'q'})
    assert omitted.code == 'TOOL_ERROR' and route['error'] in omitted.text and sent == []
    found = registry.execute_result('web_search', {'query': 'q', 'model': named})
    assert sent == [leg] and json.loads(found.text)['answer'] == 'found'


def test_registry_withholds_web_search_only_when_no_call_could_run(tmp_path, monkeypatch):
    _search_settings(monkeypatch, 'openai', OPENROUTER_API_KEY='synthetic')  # the pinned source has no key
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_MODEL', 'openai/gpt-5.2')
    route = resolve_web_search_route()
    registry = _task_registry(tmp_path)
    hidden, rows, _lines = _withheld(registry)
    assert hidden and [row['reason'] for row in rows] == ['unsupported_selection+missing_credential']
    assert rows[0]['details']['web_search'] == f"{route['error']}; web_search_backend: openai no_credentials"
    refused = registry.execute_result('web_search', {'query': 'q', 'model': 'gpt-5.5'})
    assert refused.code == 'CAPABILITY_UNAVAILABLE' and route['error'] in refused.text
    assert registry.get_schema_by_name('web_search') is None


# A named connection is a choice only a selection or a call makes (owner decision, 1 Nov 2026): an install whose only
# accounts are named offers the tool to the call that names one, while an omitted call and a bare id stay built-in.

def _named_only(monkeypatch, source, saved, row, **packages):
    from ouroboros.provider_models import CONNECTIONS_KEY

    _search_settings(monkeypatch, source)
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_MODEL', saved)
    monkeypatch.setenv(CONNECTIONS_KEY, json.dumps([row]))
    _discoverable(monkeypatch, **{'anthropic': True, 'ddgs': False, **packages})


_OPENAI_WORK = {'kind': 'openai', 'name': 'work', 'api_key': 'work-key'}


@pytest.mark.parametrize('source,saved,row,named,sent', [
    ('auto', '', _OPENAI_WORK, 'openai@work::future', ('openai', 'future', 'work')),
    ('openai', 'openai/gpt-5.2', _OPENAI_WORK, 'openai@work::future', ('openai', 'future', 'work')),  # saved unusable
    ('anthropic', '', {'kind': 'anthropic', 'name': 'team', 'api_key': 'k'}, 'anthropic@team::future', ('anthropic', 'future', 'team')),
])
def test_a_named_connection_only_install_offers_web_search_to_the_call_that_names_it(
        tmp_path, monkeypatch, source, saved, row, named, sent):
    _named_only(monkeypatch, source, saved, row)
    calls = _refusing_legs(monkeypatch)
    registry = _task_registry(tmp_path)
    hidden, rows, _lines = _withheld(registry)
    assert not hidden and rows == [] and registry.get_schema_by_name('web_search') is not None
    # An omitted call and a bare id never reach the named account; the failure says which one a call may name.
    omitted = registry.execute_result('web_search', {'query': 'q'})
    bare = registry.execute_result('web_search', {'query': 'q', 'model': 'future'})
    assert omitted.code == bare.code == 'TOOL_ERROR' and not any(connection for _s, _m, connection in calls)
    assert json.loads(omitted.text)['named_connections'] == [f"{row['kind']}@{row['name']}"]
    assert 'named_connections' not in json.loads(bare.text)  # the call named its model; that failure is its own
    assert all('connection' not in leg for leg in resolve_web_search_route()['legs'])
    calls.clear()
    refused = registry.execute_result('web_search', {'query': 'q', 'model': named})
    assert calls == [sent] and refused.code == 'TOOL_ERROR'  # sent there and only there, even when refused


def test_a_failed_search_with_a_built_in_route_does_not_advertise_named_connections(tmp_path, monkeypatch):
    _named_only(monkeypatch, 'auto', '', _OPENAI_WORK)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'synthetic')
    _refusing_legs(monkeypatch)
    result = _task_registry(tmp_path).execute_result('web_search', {'query': 'q'})
    assert result.code == 'TOOL_ERROR' and 'named_connections' not in json.loads(result.text)


@pytest.mark.parametrize('source,row,packages', [
    ('anthropic', _OPENAI_WORK, {}),  # a pinned Source refuses another source's connection
    ('ddgs', _OPENAI_WORK, {}),  # ddgs takes no model, so no call could name one
    ('auto', {**_OPENAI_WORK, 'enabled': False}, {}),  # a disabled row cannot serve
    ('auto', {'kind': 'openai', 'name': 'keyless'}, {}),  # nor one without its credential
    ('auto', {'kind': 'openai-compatible', 'name': 'vllm', 'base_url': 'https://compatible.invalid/v1'}, {}),  # no search transport
    ('anthropic', {'kind': 'anthropic', 'name': 'team', 'api_key': 'k'}, {'anthropic': False}),  # its SDK is missing
])
def test_a_named_connection_no_call_could_use_leaves_web_search_withheld(tmp_path, monkeypatch, source, row, packages):
    _named_only(monkeypatch, source, '', row, **packages)
    hidden, rows, _lines = _withheld(_task_registry(tmp_path))
    assert hidden and rows and search._available_web_search_backends() == []


@pytest.mark.parametrize('sdk', [False, True])
def test_the_registry_names_a_named_connections_own_cause(tmp_path, monkeypatch, sdk):
    """Only anthropic@team has a key, under the Anthropic Source with a blank Model. Without its SDK the registry
    names that cause beside the built-in leg's missing key, never a lone missing key; with it, the call runs there."""
    _named_only(monkeypatch, 'anthropic', '', {'kind': 'anthropic', 'name': 'team', 'api_key': 'k'}, anthropic=sdk)
    calls = _refusing_legs(monkeypatch)
    registry = _task_registry(tmp_path)
    hidden, rows, _lines = _withheld(registry)
    result = registry.execute_result('web_search', {'query': 'q', 'model': 'anthropic@team::future'})
    if sdk:
        assert not hidden and rows == [] and calls == [('anthropic', 'future', 'team')] and result.code == 'TOOL_ERROR'
        return
    reason, detail = 'missing_credential+package_missing', 'web_search_backend: anthropic no_credentials, anthropic@team package_missing'
    assert hidden and [(row['reason'], row['details']['web_search']) for row in rows] == [(reason, detail)]
    assert result.code == 'CAPABILITY_UNAVAILABLE' and f"'web_search' is unavailable: {reason} ({detail})." in result.text
    assert calls == []


# A blank Model is the owner's "follow the recommendations" (owner decision, 1 Nov 2026): it stays blank through
# load, an ordinary Settings save and reload, while every saved model, an old shipped one included, stays saved.

@pytest.mark.parametrize('stored,expected', [
    (None, {'OUROBOROS_WEBSEARCH_MODEL': '', 'OUROBOROS_WEBSEARCH_BACKEND': 'auto'}),  # a fresh install
    ({'OUROBOROS_WEBSEARCH_MODEL': 'gpt-5.2'}, {'OUROBOROS_WEBSEARCH_MODEL': 'gpt-5.2', 'OUROBOROS_WEBSEARCH_BACKEND': 'auto'}),
    ({'OUROBOROS_WEBSEARCH_BACKEND': 'anthropic', 'OUROBOROS_WEBSEARCH_MODEL': 'claude-opus-4-6'},
     {'OUROBOROS_WEBSEARCH_MODEL': 'claude-opus-4-6', 'OUROBOROS_WEBSEARCH_BACKEND': 'anthropic'}),
])
def test_settings_keep_a_blank_search_model_blank_and_a_saved_one_saved(tmp_path, monkeypatch, stored, expected):
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import config as cfg
    from ouroboros.gateway import settings as settings_mod
    from tests.test_owner_settings_write_seam import _settings_app

    for key in cfg.SETTINGS_DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / 'data' / 'settings.json'
    path.parent.mkdir()
    monkeypatch.setattr(cfg, 'DATA_DIR', path.parent)
    monkeypatch.setattr(cfg, 'SETTINGS_PATH', path)
    cfg.reset_runtime_mode_baseline_for_tests()
    if stored is not None:
        path.write_text(json.dumps({'OPENROUTER_API_KEY': 'synthetic', **stored}), encoding='utf-8')
    search_keys = tuple(expected)
    app = _settings_app(monkeypatch, path)
    app.router.routes.append(Route('/api/settings', endpoint=settings_mod.api_settings_get, methods=['GET']))
    client = TestClient(app)
    shown = client.get('/api/settings').json()
    assert {key: shown[key] for key in search_keys} == expected
    # The Settings page posts the fields it shows: an unrelated save writes them back as shown.
    saved = client.post('/api/settings', json={'TOTAL_BUDGET': '25', **{key: shown[key] for key in search_keys}})
    assert saved.status_code == 200, saved.text
    on_disk = json.loads(path.read_text(encoding='utf-8'))
    assert on_disk.get('OUROBOROS_WEBSEARCH_MODEL', '') == expected['OUROBOROS_WEBSEARCH_MODEL']
    reloaded = cfg.load_settings()
    assert {key: reloaded[key] for key in search_keys} == expected
    route = resolve_web_search_route(settings=reloaded)
    if expected['OUROBOROS_WEBSEARCH_MODEL'] == '':
        assert all(leg['model'] == SEARCH_MODELS[leg['source']] and leg['model_source'] == 'provider_default'
                   for leg in route['legs'] + route['unavailable'])
    elif expected['OUROBOROS_WEBSEARCH_BACKEND'] == 'anthropic':
        assert route['ignored_legacy_model'] == 'claude-opus-4-6'  # the legacy bare spelling keeps its meaning
    else:
        assert route['legs'][0]['model'] == 'openai/gpt-5.2' and route['legs'][0]['model_source'] == 'selection'
    # Editing the field authors a choice.
    assert client.post('/api/settings', json={'OUROBOROS_WEBSEARCH_MODEL': 'gpt-5.5'}).status_code == 200
    assert json.loads(path.read_text(encoding='utf-8'))['OUROBOROS_WEBSEARCH_MODEL'] == 'gpt-5.5'
    cfg.reset_runtime_mode_baseline_for_tests()


# Credential plans (benchmark isolation, the manifest disclosure) read the search slot as the legs this
# resolver sends for the target's own Source and Model, never as a chat route and never by Main's provider.

@pytest.mark.parametrize(("search", "search_keys", "main"), [
    pytest.param("", {"OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"}, "openrouter/model", id='-search_keys0-openrouter/model'),  # Blank: the recommended model of every source Auto tries, whatever Main is (owner decision, 1 Nov 2026).
    pytest.param("gpt-5.2", {"OPENAI_API_KEY", "OPENROUTER_API_KEY"}, "anthropic::owner-main", id='gpt-5.2-search_keys1-anthropic::owner-main'),  # A bare native id is OpenAI's in search grammar: its OpenAI leg first, then the same model on OpenRouter.
    pytest.param("google/gemini-2.5-pro", {"OPENROUTER_API_KEY"}, "openrouter/model", id='google/gemini-2.5-pro-search_keys2-openrouter/model'),  # An OpenRouter catalog id (a single-model pin) plans that leg only, never Auto's other-model fallback.
    pytest.param("openai@work::gpt-5.5", {"OPENAI_API_KEY", "OPENROUTER_API_KEY", "OUROBOROS_CONNECTIONS"}, "anthropic::owner-main", id='openai@work::gpt-5.5-search_keys3-anthropic::owner-main'),  # A named connection travels as its row; the saved route's same-model OpenRouter leg is kept too.
])
def test_the_web_search_slot_plans_the_legs_its_configured_search_sends(main, search, search_keys):
    """The search Model is no chat route: the plan takes the legs ``search_routes`` resolves for the
    target's own Source and Model, so the isolated run searches in the target's order whatever Main is.
    A blank search is not pinned to the chat slots' provider: its recommended direct legs are planned
    and their keys copied (a run that must stay on one provider pins its search, as the provenance `_LIVE` does)."""
    from devtools.benchmarks.common.server_runner import build_isolated_settings
    from ouroboros.provider_models import provider_credential_plan

    live = {"OUROBOROS_MODEL": main, "OUROBOROS_MODEL_LIGHT": main, "OUROBOROS_WEBSEARCH_MODEL": search,
            "OPENROUTER_API_KEY": "or-value", "OPENAI_API_KEY": "oa-value", "ANTHROPIC_API_KEY": "an-value",
            "OUROBOROS_CONNECTIONS": [{"kind": "openai", "name": "work", "api_key": "work-value"}]}
    chat_only = provider_credential_plan({**live, "OUROBOROS_WEBSEARCH_BACKEND": "ddgs"})["planned_keys"]
    assert set(provider_credential_plan(live)["planned_keys"]) == set(chat_only) | search_keys
    out = build_isolated_settings(live, OUROBOROS_MODEL=main, OUROBOROS_MODEL_LIGHT=main)
    assert {key for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY") if key in out} >= (
        search_keys - {"OUROBOROS_CONNECTIONS"})
    first = resolve_web_search_route(settings=out)["legs"][0]
    target = resolve_web_search_route(settings=live)["legs"][0]
    assert (first["source"], first["model"], first.get("connection", "")) == (
        target["source"], target["model"], target.get("connection", ""))  # the isolated run searches as the target


_NOT_SEARCH_KEYS = ("OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_VISION", "OUROBOROS_MODEL_CONSCIOUSNESS",
                    "OUROBOROS_MODEL_FALLBACKS", "OUROBOROS_MODEL_FALLBACK", "OUROBOROS_MODEL_SAFETY")


_SEARCH_CHOICE = ("OUROBOROS_WEBSEARCH_BACKEND", "OUROBOROS_WEBSEARCH_MODEL")


@pytest.mark.parametrize(("main", "saved", "override", "search_keys"), [
    pytest.param("deepseek::chat", {}, {}, {"OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"}, id='deepseek::chat-saved0-override0-search_keys0'),  # nothing saved: Auto, not this process's Source
    pytest.param("deepseek::chat", {"OUROBOROS_WEBSEARCH_BACKEND": "ddgs", "OUROBOROS_WEBSEARCH_MODEL": "gpt-5.2"}, {}, set(), id='deepseek::chat-saved2-override2-search_keys2'),  # The saved Source travels with its Model: a ddgs pin's stale model never becomes paid Auto search,
    pytest.param("deepseek::chat", {"OUROBOROS_WEBSEARCH_BACKEND": "ddgs", "OUROBOROS_WEBSEARCH_MODEL": "gpt-5.2"}, {"OUROBOROS_WEBSEARCH_BACKEND": "openai"}, {"OPENAI_API_KEY"}, id='deepseek::chat-saved4-override4-search_keys4'),  # A benchmark's explicit Source wins over the saved one.
    pytest.param("deepseek::chat", {"OUROBOROS_WEBSEARCH_MODEL": "vendor/future"}, {}, {"OPENROUTER_API_KEY"}, id='deepseek::chat-saved6-override6-search_keys6'),  # never Auto's other-model fallback
    pytest.param("minimax::chat", {"OUROBOROS_WEBSEARCH_BACKEND": "openai"}, {}, {"OPENAI_API_KEY"}, id='minimax::chat-saved1-override1-search_keys1'),
    pytest.param("minimax::chat", {"OUROBOROS_WEBSEARCH_BACKEND": "anthropic", "OUROBOROS_WEBSEARCH_MODEL": "claude-opus-4-6"}, {}, {"ANTHROPIC_API_KEY"}, id='minimax::chat-saved3-override3-search_keys3'),  # nor does a legacy bare id under the Anthropic pin change meaning (the recommended model is still the one sent).
    pytest.param("minimax::chat", {"OUROBOROS_WEBSEARCH_BACKEND": "openai", "OUROBOROS_WEBSEARCH_MODEL": "openai/gpt-5.2"}, {}, {"OPENAI_API_KEY"}, id='minimax::chat-saved5-override5-search_keys5'),  # An unusable saved Model leaves the Source's recommended leg planned: a call naming its own model meets it.
])
def test_an_isolated_document_searches_as_the_saved_one_whatever_main_or_the_host(monkeypatch, saved, override, search_keys, main):
    """The real isolated document keeps the saved Source with its Model (an explicit override wins), and its grant
    is planned from that pair, whatever Main is and whatever this process's own Source and Model are; the run's
    route then offers exactly the granted legs."""
    from devtools.benchmarks.common.secrets import isolated_credential_grants
    from devtools.benchmarks.common.server_runner import build_isolated_settings
    from ouroboros.provider_models import PROVIDER_CREDENTIAL_GROUPS

    monkeypatch.setenv("OUROBOROS_WEBSEARCH_BACKEND", "anthropic")
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_MODEL", "anthropic::host-search")
    search_group = {"OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"}
    live = {**dict.fromkeys(_NOT_SEARCH_KEYS, main), "OPENAI_API_KEY": "oa", "OPENROUTER_API_KEY": "or", "ANTHROPIC_API_KEY": "an",
            "DEEPSEEK_API_KEY": "ds", "MINIMAX_API_KEY": "mm", **saved}
    out = build_isolated_settings(live, **override)
    assert {key: out.get(key) for key in _SEARCH_CHOICE} == {key: {**saved, **override}.get(key) for key in _SEARCH_CHOICE}
    grants = isolated_credential_grants(out)
    assert set(grants["planned_keys"]) & search_group == search_keys
    assert {key for key in search_group if key in out} == search_keys
    assert set(PROVIDER_CREDENTIAL_GROUPS[main.partition("::")[0]]) <= set(grants["planned_keys"])
    route, saved_route = resolve_web_search_route(settings=out), resolve_web_search_route(settings={**live, **override})

    def meaning(of):  # readiness differs (the isolated document holds only planned keys); the planned legs do not
        return of["source"], of.get("ignored_legacy_model"), sorted((leg["source"], leg["model"]) for leg in of["legs"] + of["unavailable"])
    assert meaning(route) == meaning(saved_route)  # the isolated run searches as the saved document does
    if route.get("error"):  # a call naming its own model is what runs there, on the recommended leg's credential
        route = resolve_web_search_route(settings=out, model="")
    keyed = {"openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
    # Credentials reach what the run will send, and nothing but the target's own legs is eligible.
    assert {keyed[leg["source"]] for leg in route["legs"] if leg["source"] in keyed} == search_keys


def test_a_ddgs_search_plans_no_credential_of_its_own():
    """ddgs ignores a saved model, so the slot's route spelling plans nothing (it once planned its provider)."""
    from ouroboros.provider_models import provider_credential_plan

    settings = {"OUROBOROS_MODEL": "openrouter/model", "OUROBOROS_WEBSEARCH_BACKEND": "ddgs",
                "OUROBOROS_WEBSEARCH_MODEL": "anthropic::saved-search"}
    plan = provider_credential_plan(settings)
    assert "ANTHROPIC_API_KEY" not in plan["planned_keys"] and "anthropic" not in plan["providers"]
