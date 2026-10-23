"""Live window facts belong to the connection that received the provider input."""
import asyncio
from dataclasses import replace

import pytest

from ouroboros import capability_evidence as ce
from ouroboros.context_window import resolve_window
from ouroboros.contracts.live_conversation import LiveOptions
from tests.test_live_conversation import host_world as host_world
from tests.test_live_usage import token_usage


@pytest.mark.parametrize('window,basis,can_raise,accepted', [
    (133554, 'provider', True, 150000),
    (92000, 'provider', False, 150000),
    (80000, 'estimator', False, 0),
])
def test_completed_input_keeps_units_reserve_and_active_account(
        host_world, monkeypatch, window, basis, can_raise, accepted):
    host = host_world.host
    root = host.ctx.drive_root
    monkeypatch.setattr(ce, 'canonical_evidence_root', lambda: root)
    host.model = replace(host.model, facts=replace(host.model.facts,
        conversation_window_tokens=window, basis=basis, acceptance_can_raise=can_raise))
    old_scope = host.turn.observation_fingerprint()
    factory = host_world.provider['session']

    def large_output_bound(*args, **kwargs):
        session = factory(*args, **kwargs)
        session.output_bound_tokens = 18000
        return session

    host_world.provider['session'] = large_output_bound

    async def consume(session):
        assert host._renewal_window.total_equivalent == window
        assert host._renewal_window.reserve == (18000 if can_raise else 10000)
        # A settings edit during the connection cannot reattribute its receipt.
        host_world.settings['TEST_CREDENTIAL'] = 'rotated-synthetic-account'
        host.response_started('accepted')
        for _ in range(2):  # Duplicate provider completions cannot create another fact.
            host.response_done('accepted', usage=token_usage(text_input_tokens=100000,
                audio_input_tokens=50000, cached_input_tokens=80000, text_output_tokens=1000),
                raw_usage={'input_tokens': 150000, 'output_tokens': 1000,
                           'input_token_details': {'cached_tokens': 80000}}, status='completed')
        current = resolve_window(host.plan, 18000, root=root)
        assert current.route_fp == old_scope and current.accepted == accepted
        assert current.reserve == 18000
        assert current.total_equivalent == (168000 if can_raise else window)
        assert current.usable == (150000 if can_raise else window - 18000)
        assert host._renewal_window.total_equivalent == current.total_equivalent
        assert host._renewal_window.reserve == (18000 if can_raise else 10000)
        assert host._renewal_window.accepted == accepted
        rows = ce._load(root).get('main_windows', {}).get(old_scope, {}).get('observations', [])
        assert len(rows) == (1 if accepted else 0)
        if rows:
            assert rows[0]['accepted'] == 150000 and rows[0]['reserve'] == 18000
        host._held_input, host._growth = 140000, 0
        host._history_at_receipt = host._history_estimate()
        with monkeypatch.context() as polling:
            polling.setattr(ce, '_load', lambda *a, **kw: pytest.fail('audio polling read the window store'))
            for _ in range(1000):
                assert host.renewal_due() is (not can_raise)

    async def rotated(session):
        assert session.opened
        assert host.plan.window_record.route_fp != old_scope
        assert host.plan.window_record.accepted == 0
        assert host._renewal_window.total_equivalent == window
        assert host._renewal_window.reserve == (18000 if can_raise else 10000)

    async def run():
        await host.session(consume, LiveOptions())
        await host.session(rotated, LiveOptions())

    asyncio.run(run())


@pytest.mark.parametrize('status,raw', [
    ('failed', {'input_tokens': 200000}),
    ('cancelled', {'input_tokens': 200000}),
    ('interrupted', {'input_tokens': 200000}),
    ('completed', {}),
    ('completed', {'input_tokens': True}),
    ('completed', {'output_tokens': 200000}),
])
def test_only_completed_provider_input_is_window_evidence(host_world, monkeypatch, status, raw):
    host = host_world.host
    monkeypatch.setattr(ce, 'canonical_evidence_root', lambda: host.ctx.drive_root)
    host.model = replace(host.model, facts=replace(host.model.facts,
        conversation_window_tokens=133554, acceptance_can_raise=True))

    async def consume(session):
        host.response_started('unwitnessed')
        host.response_done('unwitnessed', usage=token_usage(text_input_tokens=200000),
                           raw_usage=raw, status=status)
        assert resolve_window(host.plan, 10000, root=host.ctx.drive_root).accepted == 0
        assert host._renewal_window.total_equivalent == 133554

    asyncio.run(host.session(consume, LiveOptions()))


def test_retention_renewal_keeps_the_declared_window_and_growth_reserve(host_world, monkeypatch):
    """A large per-response output allowance is not the session's renewal margin."""
    import json
    from dataclasses import asdict

    host = host_world.host
    root = host.ctx.drive_root
    monkeypatch.setattr(ce, 'canonical_evidence_root', lambda: root)
    host.model = replace(host.model, facts=replace(host.model.facts,
        conversation_window_tokens=92000, working_reserve_tokens=10000, acceptance_can_raise=False))
    factory = host_world.provider['session']

    def large_output_bound(*args, **kwargs):
        session = factory(*args, **kwargs)
        session.output_bound_tokens = 65536
        return session

    host_world.provider['session'] = large_output_bound

    async def consume(session):
        initial = host._renewal_window
        host.response_started('retention-receipt')
        host.response_done('retention-receipt', usage=token_usage(text_input_tokens=38607,
            audio_input_tokens=222, audio_output_tokens=52), raw_usage={
                'promptTokenCount': 38940, 'responseTokenCount': 52,
                'promptTokensDetails': [{'modality': 'TEXT', 'tokenCount': 38607},
                                       {'modality': 'AUDIO', 'tokenCount': 222}]}, status='completed')
        current = host._renewal_window
        print('RETENTION_RENEWAL ' + json.dumps({
            'initial': asdict(initial), 'after_receipt': asdict(current),
            'held': host._held_input, 'growth': host._growth,
            'physical_reserve': host._physical_context.response_reserve_tokens,
            'renewal_due': host.renewal_due()}, sort_keys=True))
        assert initial.total_equivalent == current.total_equivalent == 92000
        assert initial.reserve == current.reserve == 10000
        assert current.accepted == 38940
        assert host._physical_context.response_reserve_tokens == 65536
        assert resolve_window(host.plan, 65536, root=root).reserve == 65536
        rows = ce._load(root)['main_windows'][host.plan.route_fp]['observations']
        assert rows[-1]['accepted'] == 38940 and rows[-1]['reserve'] == 65536
        with monkeypatch.context() as polling:
            polling.setattr(ce, '_load', lambda *a, **kw: pytest.fail('audio polling read the window store'))
            assert not host.renewal_due()
            host._held_input = 81999
            host._growth = 0
            host._history_at_receipt = host._history_estimate()
            assert not host.renewal_due()
            host._held_input = 82000
            assert host.renewal_due()

    asyncio.run(host.session(consume, LiveOptions()))


from contextlib import asynccontextmanager
from types import SimpleNamespace
from tests.test_grok_openai_protocol import providers as providers, deny_network as deny_network, Wire


@pytest.mark.parametrize('rotation', ['before_prepare', 'before_open'])
def test_acceptance_belongs_to_the_account_the_connector_opened(host_world, monkeypatch, rotation, providers):
    from ouroboros import capability_evidence as ce
    from ouroboros.contracts.live_conversation import LiveOptions

    host = host_world.host
    root = host.ctx.drive_root
    monkeypatch.setattr(ce, 'canonical_evidence_root', lambda: root)
    host.model = replace(host.model, facts=replace(host.model.facts,
        conversation_window_tokens=133554, acceptance_can_raise=True))
    first_scope = host.turn.observation_fingerprint()
    factory, check = host_world.provider['session'], host.check
    opened_scopes = []
    checks = 0

    registrations = {}
    providers['openai'].module.register(SimpleNamespace(
        register_realtime_provider=lambda name, **kw: registrations.update(kw),
        get_settings=lambda keys: {'OPENAI_API_KEY': host_world.settings['TEST_CREDENTIAL']}))

    @asynccontextmanager
    async def fake_socket(url, **kwargs):
        # Exercise the shipped connector, including its late get_settings read.
        # The header is synthetic; no real credential or network is involved.
        assert kwargs['extra_headers']['Authorization'] == 'Bearer second-synthetic-account'
        opened_scopes.append(host.turn.observation_fingerprint())
        yield Wire('openai')

    import websockets.legacy.client
    monkeypatch.setattr(websockets.legacy.client, 'connect', fake_socket)

    async def controls_with_settings_edit():
        nonlocal checks
        checks += 1
        await check()
        # session() awaits controls before preparation and again before open.
        # A Settings edit while that second asynchronous check waits is allowed.
        if checks == (1 if rotation == 'before_prepare' else 2):
            host_world.settings['TEST_CREDENTIAL'] = 'second-synthetic-account'

    monkeypatch.setattr(host, 'check', controls_with_settings_edit)
    host_world.provider['session'] = registrations['session']

    async def complete(session):
        host.response_started('account-witness')
        host.response_done('account-witness', usage=token_usage(text_input_tokens=150000),
            raw_usage={'input_tokens': 150000}, status='completed')

    asyncio.run(host.session(complete, LiveOptions()))
    actual_scope, = opened_scopes
    assert actual_scope != first_scope
    records = ce._load(root).get('main_windows', {})
    accepted = lambda scope: max((row.get('accepted', 0) for row in
        records.get(scope, {}).get('observations', [])), default=0)
    # Returning to the first key must not borrow the other account's success.
    from ouroboros.realtime_setup import session_candidate
    host_world.settings['TEST_CREDENTIAL'] = 'test-credential'
    _, returning, _ = session_candidate(host, factory, LiveOptions())
    assert (accepted(first_scope), accepted(actual_scope),
            returning.physical_context.capacity_total_tokens) == (0, 150000, 133554)
