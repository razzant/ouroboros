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
