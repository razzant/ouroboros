"""A route's capability identity: helpers size replies on the endpoint dispatch sends to, and capability snapshots
change no transport payload or unrelated route key."""

import copy
import json

import pytest

from ouroboros import capability_evidence as ce, config, context_fit, llm_routing, response_limits
from ouroboros.llm import LLMClient
from ouroboros.llm_attempt import bound_reply_allowance
from ouroboros.response_limits import dispatch_target


def test_helper_capacity_stays_on_dispatch_endpoint(tmp_path, monkeypatch):
    model = 'openai-compatible::fixture'
    before = {'OPENAI_COMPATIBLE_BASE_URL': 'http://127.0.0.1:18080/v1'}
    after = {'OPENAI_COMPATIBLE_BASE_URL': 'http://127.0.0.1:28080/v1'}
    monkeypatch.setattr(ce, 'canonical_evidence_root', lambda: tmp_path)
    target = dispatch_target(model, before)
    monkeypatch.setattr(ce, '_openai_compatible_metadata_window',
        lambda model, base_url, *args, **kwargs:
            80000 if base_url == before['OPENAI_COMPATIBLE_BASE_URL'] else 200000)
    for base_url in (before['OPENAI_COMPATIBLE_BASE_URL'], after['OPENAI_COMPATIBLE_BASE_URL']):
        observed = ce.probe(tmp_path, provider='openai-compatible', model=model,
                            base_url=base_url, allow_fetch=True)
        assert observed.status == ce.STATUS_CONFIRMED
    monkeypatch.setattr(config, 'runtime_settings', lambda: after)
    monkeypatch.setattr(context_fit, 'bounded_prompt_tokens_for_payload', lambda *a, **k: 35000)
    result = bound_reply_allowance(target, {'messages': [{'role': 'user', 'content': 'intact'}], 'max_tokens': 65536})
    assert result['max_tokens'] == 35000


def test_helper_retains_explicit_openrouter_route_spelling(tmp_path, monkeypatch):
    model = 'openrouter::fixture'
    settings = {'OPENROUTER_API_KEY': 'fixture-key'}
    monkeypatch.setattr(ce, 'canonical_evidence_root', lambda: tmp_path)
    monkeypatch.setattr(config, 'runtime_settings', lambda: settings)
    target = dispatch_target(model, settings)
    catalog_reads = []
    def catalog_window(native_model, *, allow_fetch=True):
        catalog_reads.append((native_model, allow_fetch))
        return 80000
    monkeypatch.setattr(LLMClient, 'openrouter_context_length', staticmethod(catalog_window))
    monkeypatch.setattr(LLMClient, '_CONTEXT_LENGTH_OBSERVED_AT', ce.utc_now_iso())
    monkeypatch.setattr(LLMClient, '_fetch_openrouter_capabilities', classmethod(lambda cls: None))
    observed = ce.probe(tmp_path, provider='openrouter', model=model, allow_fetch=True)
    assert observed.status == ce.STATUS_CONFIRMED
    assert observed.window_tokens == 80000
    assert catalog_reads == [('fixture', True)]
    monkeypatch.setattr(context_fit, 'bounded_prompt_tokens_for_payload', lambda *a, **k: 35000)
    result = bound_reply_allowance(target, {'messages': [{'role': 'user', 'content': 'intact'}], 'max_tokens': 65536})
    assert result['max_tokens'] == 35000


def test_reviewer_scope_stays_on_the_target_probed(tmp_path, monkeypatch):
    from ouroboros import context_window, model_slots, reviewer_window
    model = 'openai-compatible::fixture'
    before = {'OPENAI_COMPATIBLE_BASE_URL': 'http://127.0.0.1:18080/v1'}
    after = {'OPENAI_COMPATIBLE_BASE_URL': 'http://127.0.0.1:28080/v1'}
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(config, 'runtime_settings', lambda: before)
    monkeypatch.setattr(model_slots, 'model_role_option', lambda *a, **kw: 0)
    route = dict(provider='openai-compatible', model=model, base_url=before['OPENAI_COMPATIBLE_BASE_URL'],
                 options={}, use_local=False)
    scope = context_window.route_observation_scope(route, before)
    context_window.record_observation(tmp_path, scope, refused=80000)
    def probe(*args, **kwargs):
        before.update(after)
        return ce.CapabilityEvidence(200000, 'confirmed', ce.SOURCE_PROVIDER_METADATA,
            ce.route_fingerprint(provider=route['provider'], model=model, base_url=route['base_url']),
            model, route['provider'], ts=ce.utc_now_iso())
    monkeypatch.setattr(ce, 'probe', probe)
    result = reviewer_window.resolve_reviewer_window(model, use_local=False)
    assert result.window_record.route_fp == scope
    assert result.sizing_window() == 80000


@pytest.mark.parametrize('model,endpoint,capability_endpoint', [
    ('openai-compatible@lab::fixture', 'http://127.0.0.1:18080/v1', 'http://127.0.0.1:18080/v1'),
    ('openrouter::fixture', 'https://openrouter.ai/api/v1', ''),
    ('openrouter@lab::fixture', 'https://openrouter.ai/api/v1', ''),
    ('openai::fixture', 'https://api.openai.com/v1', 'http://127.0.0.1:18080/v1'),
    ('minimax@lab::fixture', 'https://api.minimaxi.com/v1', 'https://api.minimaxi.com/v1'),
])
def test_helper_uses_one_resolved_identity_without_any_settings_reread(
        tmp_path, monkeypatch, model, endpoint, capability_endpoint):
    from ouroboros import llm_routing, response_limits
    from ouroboros.provider_models import CONNECTIONS_KEY
    rows = [dict(kind='openai-compatible', name='lab', base_url='http://127.0.0.1:18080/v1'),
            dict(kind='openrouter', name='lab', api_key='fixture'),
            dict(kind='minimax', name='lab', api_key='fixture', region='cn_zh')]
    settings = {'OPENROUTER_API_KEY': 'fixture', 'OPENAI_API_KEY': 'fixture',
                'OPENAI_BASE_URL': 'http://127.0.0.1:18080/v1', CONNECTIONS_KEY: rows}
    target = dispatch_target(model, settings)
    expected = dict(provider=target['provider'], model=model, base_url=capability_endpoint,
                    dispatch_base_url=endpoint)
    assert target['capability_route'] == expected
    assert target['base_url'] == target['capability_route']['dispatch_base_url']
    ce.record_owner_ack(tmp_path, provider=target['provider'], model=model,
                        base_url=capability_endpoint, window_tokens=80000)
    monkeypatch.setattr(ce, 'canonical_evidence_root', lambda: tmp_path)
    monkeypatch.setattr(context_fit, 'bounded_prompt_tokens_for_payload', lambda *a, **kw: 35000)
    settings['OPENAI_BASE_URL'] = 'http://127.0.0.1:28080/v1'
    rows[0]['base_url'] = 'http://127.0.0.1:28080/v1'
    rows[2]['region'] = 'global_en'
    def forbidden(*args, **kwargs):
        pytest.fail('a resolved capability lookup must not resolve its route or read Settings again')
    monkeypatch.setattr(config, 'runtime_settings', forbidden)
    monkeypatch.setattr(llm_routing, 'runtime_setting', forbidden)
    monkeypatch.setattr(response_limits, 'dispatch_target', forbidden)
    assert target['capability_route'] == expected
    assert bound_reply_allowance(target, {'messages': [{'role': 'user', 'content': 'intact'}],
                                         'max_tokens': 65536})['max_tokens'] == 35000


def test_reviewer_reply_reserve_keeps_the_window_endpoint(tmp_path, monkeypatch):
    from ouroboros import model_slots, provider_models, response_limits, reviewer_window
    model = 'openai-compatible::fixture'
    settings = {'OPENAI_COMPATIBLE_BASE_URL': 'http://127.0.0.1:18080/v1'}
    before = settings['OPENAI_COMPATIBLE_BASE_URL']
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(ce, 'canonical_evidence_root', lambda: tmp_path)
    monkeypatch.setattr(config, 'runtime_settings', lambda: settings)
    monkeypatch.setattr(model_slots, 'model_role_option', lambda *a, **kw: 0)
    ce.record_owner_ack(tmp_path, provider='openai-compatible', model=model, base_url=before, window_tokens=80000)
    response_limits.record_response_ack(tmp_path, provider='openai-compatible', model=model,
                                        base_url=before, max_output_tokens=12000)
    window = reviewer_window.resolve_reviewer_window(model, use_local=False, allow_fetch=False).sizing_window()
    settings['OPENAI_COMPATIBLE_BASE_URL'] = 'http://127.0.0.1:28080/v1'
    # The old endpoint's response cap must survive both a changed route and stale short-name fields.
    monkeypatch.setattr(response_limits, 'dispatch_target', lambda *a, **kw: dispatch_target(model, settings))
    monkeypatch.setattr(provider_models, 'review_model_uses_local', lambda *_: pytest.fail('resolved lane must stay bound'))
    reserve, _ = reviewer_window.window_scaled_reserves(window, output_reserve=65536,
        tokenizer_margin=10000, model_id=model, binding={'use_local': None})
    assert reserve == 12000


def test_main_probe_does_not_resolve_again_after_metadata_returns(tmp_path, monkeypatch):
    from ouroboros import llm_routing, model_slots, response_limits, server_runtime
    model = 'openai-compatible::fixture'
    settings = {'OPENAI_COMPATIBLE_BASE_URL': 'http://127.0.0.1:18080/v1'}
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(config, 'runtime_settings', lambda: settings)
    monkeypatch.setattr(server_runtime, 'apply_runtime_provider_defaults', lambda value: (value, False, []))
    monkeypatch.setattr(model_slots, 'model_role_option', lambda *a, **kw: 0)
    response_limits.record_response_ack(tmp_path, provider='openai-compatible', model=model,
        base_url=settings['OPENAI_COMPATIBLE_BASE_URL'], max_output_tokens=12000)
    def forbidden(*args, **kwargs):
        pytest.fail('metadata/output facts must use the same already resolved target')
    def metadata(*args, **kwargs):
        settings['OPENAI_COMPATIBLE_BASE_URL'] = 'http://127.0.0.1:28080/v1'
        monkeypatch.setattr(llm_routing, 'runtime_setting', forbidden)
        return 80000
    monkeypatch.setattr(ce, '_openai_compatible_metadata_window', metadata)
    route, evidence = context_fit.resolve_context_fit_route({'model': model}, allow_fetch=True)
    assert route['base_url'] == 'http://127.0.0.1:18080/v1'
    assert evidence.response_limit['max_output_tokens'] == 12000


@pytest.mark.parametrize('model,settings,fields', [
    ('openai-compatible::fixture', {'OPENAI_BASE_URL': 'http://127.0.0.1:18080/v1',
        'OPENAI_API_KEY': 'fixture'}, {'OPENAI_COMPATIBLE_BASE_URL', 'OPENAI_BASE_URL', 'OPENAI_API_KEY'}),
    ('openai-compatible::fixture', {'OPENAI_COMPATIBLE_BASE_URL': 'http://127.0.0.1:18080/v1',
        'OPENAI_COMPATIBLE_API_KEY': 'fixture'}, {'OPENAI_COMPATIBLE_BASE_URL', 'OPENAI_COMPATIBLE_API_KEY'}),
    # Direct OpenAI dispatch reads its key and its connection's API (``OPENAI_API_SURFACE``: Chat or Responses body).
    ('openai::fixture', {'OPENAI_BASE_URL': 'http://127.0.0.1:18080/v1',
        'OPENAI_API_KEY': 'fixture'}, {'OPENAI_API_KEY', 'OPENAI_API_SURFACE'}),
])
def test_target_snapshot_preserves_the_dispatch_read_observer(model, settings, fields):
    from ouroboros.deployment_policy import _Reads
    from ouroboros.provider_models import CONNECTIONS_KEY

    observed = _Reads(settings)
    target = dispatch_target(model, observed)
    assert observed.names == fields | {CONNECTIONS_KEY}
    original_endpoint = target['capability_route']['dispatch_base_url']
    observed.update({key: 'changed' for key in settings})
    assert target['api_key'] == 'fixture'
    assert target['capability_route']['dispatch_base_url'] == original_endpoint
    if model == 'openai::fixture':
        assert target['capability_route']['base_url'] == settings['OPENAI_BASE_URL']


def test_checker_digest_ignores_metadata_but_binds_dispatch_credentials():
    from types import SimpleNamespace
    from ouroboros.deployment_policy import _checker_targets

    policy = SimpleNamespace(checker=SimpleNamespace(routes=lambda: [('openai::fixture', False, '')]))
    settings = {'OPENAI_API_KEY': 'fixture', 'OPENAI_BASE_URL': 'http://127.0.0.1:18080/v1'}
    before = _checker_targets(policy, ({}, settings))
    changed_metadata = {**settings, 'OPENAI_BASE_URL': 'http://127.0.0.1:28080/v1'}
    assert _checker_targets(policy, ({}, changed_metadata)) == before
    changed_credential = {**settings, 'OPENAI_API_KEY': 'different-fixture'}
    assert _checker_targets(policy, ({}, changed_credential)) != before


_CARRIER_ENDPOINT = 'http://127.0.0.1:19876/carrier-only-sentinel'

def test_claudexor_wire_projects_only_transport_fields(monkeypatch):
    from ouroboros import llm_claudexor
    target = llm_routing.resolve_target('claudexor::codex=fixture', {})
    monkeypatch.setattr(llm_claudexor, 'send_account', lambda *a, **kw: ({}, None))
    messages = [{'role': 'user', 'content': 'complete input'}]
    before = llm_claudexor._request(target, messages, None, {})
    target['capability_route']['dispatch_base_url'] = _CARRIER_ENDPOINT
    after = llm_claudexor._request(target, messages, None, {})
    assert after == before
    assert after['messages'] == messages
    assert after['model'] == 'fixture'
    assert 'capability_route' not in json.dumps(after)
    assert _CARRIER_ENDPOINT not in json.dumps(after)


@pytest.mark.parametrize('mutation', ['carrier', 'settings'])
def test_transport_scopes_and_client_caches_ignore_metadata_changes(monkeypatch, mutation):
    from ouroboros import model_concurrency, net_transport, openrouter_cost, vision_routing
    from ouroboros.request_wire_contract import build_request_wire_profile

    settings = {'OPENAI_API_KEY': 'fixture-key', 'OPENAI_BASE_URL': 'http://127.0.0.1:18080/v1'}
    monkeypatch.setattr(config, 'runtime_settings', lambda: settings)
    target = llm_routing.resolve_target('openai::fixture', settings)
    client = LLMClient()
    monkeypatch.setattr(client, '_new_remote_client', lambda target: object())
    monkeypatch.setattr(llm_routing, '_remote_http_client', lambda *a, **kw: (None, {}))
    monkeypatch.setattr(net_transport, 'openai_client_class', lambda *a, **kw: lambda **kwargs: object())
    monkeypatch.setattr(net_transport, 'extra_ca_bundle', lambda: '')

    def identities(target):
        payload = {'model': target['resolved_model'], 'messages': [{'role': 'user', 'content': 'input'}]}
        return (
            model_concurrency._route_digest('openai::fixture', False, target),
            build_request_wire_profile(target, payload, api_surface='chat.completions').fingerprint,
            vision_routing.image_route_scope(target),
            openrouter_cost.binding_for_target({**target, 'provider': 'openrouter'}),
            client._get_remote_client(target), client._get_async_remote_client(target),
        )

    before = identities(target)
    if mutation == 'carrier':
        target['capability_route']['dispatch_base_url'] = _CARRIER_ENDPOINT
    else:
        settings.update(OPENAI_BASE_URL=_CARRIER_ENDPOINT, OPENAI_API_KEY='different-fixture-key')
    assert identities(target) == before
    actual_route_change = {**target, 'base_url': 'http://127.0.0.1:28080/v1'}
    assert all(a != b for a, b in zip(identities(actual_route_change), before))


@pytest.mark.parametrize('mutation', ['carrier', 'settings', 'model'])
def test_route_question_reuses_answer_unless_its_executable_route_changes(monkeypatch, mutation):
    from ouroboros import server_runtime
    settings = {'OPENAI_BASE_URL': 'http://127.0.0.1:18080/v1'}
    monkeypatch.setattr(config, 'runtime_settings', lambda: settings)
    monkeypatch.setattr(server_runtime, 'apply_runtime_provider_defaults', lambda values: (values, False, []))
    dispatch = response_limits.dispatch_target
    metadata = {'endpoint': ''}

    def with_metadata(*args, **kwargs):
        target = dispatch(*args, **kwargs)
        if metadata['endpoint']:
            target['capability_route']['dispatch_base_url'] = metadata['endpoint']
        return target

    monkeypatch.setattr(response_limits, 'dispatch_target', with_metadata)
    calls = []

    def resolver(task, *, allow_fetch):
        calls.append(copy.deepcopy(task))
        return context_fit._context_route(task)[0], object()

    task = {'model': 'claudexor::codex=fixture', 'use_local_model': False}
    question = context_fit.RouteQuestion(task, resolver)
    assert question.asked.wait(5)
    assert len(calls) == 1
    if mutation == 'carrier':
        metadata['endpoint'] = _CARRIER_ENDPOINT
    elif mutation == 'settings':
        settings.update(OPENAI_BASE_URL=_CARRIER_ENDPOINT, OPENAI_API_KEY='unrelated-fixture-key')
    else:
        task['model'] = 'claudexor::codex=other-fixture'
    question(task, allow_fetch=True)
    assert len(calls) == (2 if mutation == 'model' else 1)
    assert question.early == (mutation != 'model')
    assert '_resolved_target' not in json.loads(context_fit._route_key(task))


def test_native_reviewer_inspection_and_episode_facts_exclude_capability_carrier(monkeypatch):
    from types import SimpleNamespace
    from ouroboros import reviewer_window
    from ouroboros.review_native_episode import NativeToolRoundReviewExecutor

    target = llm_routing.resolve_target('openai::fixture', {'OPENAI_API_KEY': 'fixture-key'})
    window = reviewer_window.ReviewerWindow(
        window_tokens=80000, status='confirmed', model='openai::fixture',
        capability_route=target['capability_route'])
    monkeypatch.setattr(reviewer_window, 'resolve_reviewer_window', lambda *a, **kw: window)
    executor = object.__new__(NativeToolRoundReviewExecutor)
    executor.assignment = SimpleNamespace(
        slot=SimpleNamespace(model='openai::fixture', use_local=False, slot_id='fixture',
                             session_profile='', max_tokens=65536),
        request=SimpleNamespace(max_tokens=65536))
    executor._inspection_ctx = SimpleNamespace()
    executor._previous_request = {}
    executor._images = SimpleNamespace(facts={})
    executor._refresh_window()
    facts = executor._context_facts([{'role': 'user', 'content': 'intact'}], [])
    assert facts['window'] == {
        'window_tokens': 80000, 'status': 'confirmed', 'stale': False, 'observed_at': '',
        'model': 'openai::fixture', 'asserted_window_tokens': None,
        'model_route': {}, 'window_record': None,
    }
    assert 'capability_route' not in json.dumps(facts)
    # The sizing result still carries its snapshot for reply allowance lookup.
    assert window.sizing_window().capability_route == target['capability_route']
