"""One resolver's projection is consumed by context, preview and real tool dispatch."""
import json
import sys
from types import SimpleNamespace

import pytest

from ouroboros.search_routes import resolve_web_search_route
from ouroboros.tools import search
from tests import test_extensions_api as extension_fixtures
from tests.test_extensions_api import _write_ext

_clean_extensions = extension_fixtures._clean_extensions


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
    ('auto', 'future', [('openai', 'future'), ('openrouter', 'openai/future'), ('anthropic', 'claude-sonnet-4-6')]),
    ('auto', 'vendor/future', [('openrouter', 'vendor/future'), ('anthropic', 'claude-sonnet-4-6')]),
    ('auto', 'anthropic/future', [('openrouter', 'anthropic/future'), ('anthropic', 'future')]),
    ('auto', 'anthropic::future', [('anthropic', 'future')]),
    ('auto', 'openai::future', [('openai', 'future'), ('openrouter', 'openai/future')]),
    ('auto', 'openrouter::vendor/future', [('openrouter', 'vendor/future')]),
    ('openai', 'future', [('openai', 'future')]),
    ('openrouter', 'future', [('openrouter', 'openai/future')]),
    ('anthropic', 'old-ignored-setting', [('anthropic', 'claude-sonnet-4-6')]),
    ('anthropic', 'anthropic::future', [('anthropic', 'future')]),
    ('openai', 'anthropic::future', []), ('openai', 'vendor/future', []),
    # Auto keeps the legs that need no model when no built-in transport serves the saved one.
    ('auto', 'claudexor::codex=future', [('anthropic', 'claude-sonnet-4-6')]),
    ('auto', 'openai::vendor/future', [('anthropic', 'claude-sonnet-4-6')]),
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
                                               ('vendor/future', 'claude-sonnet-4-6', 'provider_default')])
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


@pytest.mark.parametrize('saved', ['claudexor::codex=future', 'deepseek::deepseek-chat', 'openai-compatible::local', ''])
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


@pytest.mark.parametrize('source,saved', [('auto', 'claudexor::codex=future'), ('openai', 'claudexor::codex=future'),
                                          ('auto', 'anthropic::vendor/future'), ('anthropic', 'anthropic::vendor/future')])
def test_an_unsupported_selection_is_disclosed_under_auto_and_refused_by_a_strict_source(monkeypatch, source, saved):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'synthetic')
    monkeypatch.setitem(sys.modules, 'ddgs', None)
    route = resolve_web_search_route(backend=source, model=saved)
    if source == 'auto':
        assert route['unapplied_model'] == saved and 'error' not in route
        assert [(leg['source'], leg['model'], leg['model_source']) for leg in route['legs']] == [
            ('anthropic', 'claude-sonnet-4-6', 'provider_default')]
    else:
        assert route['error'] and not route['legs'] and 'unapplied_model' not in route


@pytest.mark.parametrize('override,expected,origin', [
    (None, 'claude-sonnet-4-6', 'provider_default'),
    ('', 'claude-sonnet-4-6', 'provider_default'),
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


@pytest.mark.parametrize('override', ['openai::future', 'vendor/future', 'anthropic::vendor/future', 'anthropic/vendor/future'])
def test_anthropic_pin_refuses_foreign_per_call_model_without_dispatch(monkeypatch, override):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'synthetic')
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_BACKEND', 'anthropic')
    def forbidden(*args, **kwargs):
        pytest.fail('A conflicting model must not dispatch any provider or fallback')
    for name in ('openai', 'openrouter', 'anthropic', 'ddgs'):
        monkeypatch.setattr(search, '_web_search_' + name, forbidden)
    result = json.loads(search._web_search(SimpleNamespace(task_metadata={}), 'query', model=override))
    assert result['route']['error'] and result['legs_tried'] == []


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
    monkeypatch.setattr(settings_mod, 'load_settings', lambda: dict(saved))
    monkeypatch.setattr(settings_mod, 'apply_runtime_provider_defaults', lambda s: (dict(s), False, []))
    client = TestClient(Starlette(routes=[Route('/api/settings', endpoint=settings_mod.api_settings_get)]))
    data = client.get('/api/settings', params={'websearch_preview': '1'}).json()
    assert [leg['source'] for leg in data['legs']] == ['openrouter', 'ddgs']
    assert {leg['source']: leg['unavailable_reason'] for leg in data['unavailable']} == {
        'openai': 'no_credentials', 'anthropic': 'package_missing'}
    assert 'secret-7f3' not in json.dumps(data)  # causes never carry a secret value


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


@pytest.mark.parametrize('source,model', [('openai', 'openai/gpt-5.2'), ('openrouter', 'anthropic::future')])
def test_registry_explains_an_unsupported_selection_without_claiming_a_missing_key(tmp_path, monkeypatch, source, model):
    _search_settings(monkeypatch, source, OPENAI_API_KEY='synthetic', OPENROUTER_API_KEY='synthetic')
    monkeypatch.setenv('OUROBOROS_WEBSEARCH_MODEL', model)
    route = resolve_web_search_route()
    registry = _task_registry(tmp_path)
    hidden, rows, _lines = _withheld(registry)
    assert hidden and [row['reason'] for row in rows] == ['unsupported_selection']
    assert rows[0]['details']['web_search'] == route['error']
    result = registry.execute_result('web_search', {'query': 'q'})
    assert result.code == 'CAPABILITY_UNAVAILABLE' and route['error'] in result.text
    assert 'unsupported_selection' in result.text and 'missing_credential' not in result.text
    assert registry.get_schema_by_name('web_search') is None
