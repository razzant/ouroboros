"""Capability snapshots do not change transport payloads or unrelated route keys."""
import copy
import json

import pytest

from ouroboros import config, context_fit, llm_routing, response_limits
from ouroboros.llm import LLMClient


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
