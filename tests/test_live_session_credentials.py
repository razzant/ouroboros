"""Granted connector reads and account evidence share one session capture."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import json

import pytest

from ouroboros import capability_evidence as ce
from ouroboros.contracts.live_conversation import LiveOptions
from ouroboros.extension_plugin_api import PluginAPIImpl, _PluginAPIConfig
from tests.test_grok_openai_protocol import providers as providers, deny_network as deny_network, Wire
from tests._live_world import host_world as host_world
from tests.test_live_usage import token_usage


def test_open_pins_the_reprepared_account_despite_another_settings_edit(host_world, providers, monkeypatch, tmp_path):
    from ouroboros import observability

    host = host_world.host
    root = host.ctx.drive_root
    monkeypatch.setattr(ce, 'canonical_evidence_root', lambda: root)
    host.model = replace(host.model, facts=replace(host.model.facts,
        conversation_window_tokens=133554, acceptance_can_raise=True))
    settings = {'OPENAI_API_KEY': 'synthetic-first'}
    api = PluginAPIImpl(_PluginAPIConfig(skill_name='fixture', permissions=['read_settings', 'realtime_provider'],
        env_allowlist=['OPENAI_API_KEY'], granted_keys=['OPENAI_API_KEY'],
        state_dir=tmp_path / 'api', settings_reader=lambda: dict(settings)))
    providers['openai'].module.register(api)
    registrations = api._staged.realtime_providers['fixture:openai']
    assert registrations['_credential_api'] is api
    host_world.provider.update(session=registrations['session'], _credential_api=registrations['_credential_api'],
        catalog=replace(host_world.provider['catalog'], credential_settings=('OPENAI_API_KEY',)))
    first_scope = host.turn.observation_fingerprint()
    check, persist = host.check, observability.persist_physical_candidate
    checks = preparations = 0
    scopes, opened = {}, []

    async def edit_before_open():
        nonlocal checks
        checks += 1
        await check()
        if checks == 2:
            settings['OPENAI_API_KEY'] = 'synthetic-second'
            scopes['second'] = host.turn.observation_fingerprint()

    def edit_while_reprepared(*args, **kwargs):
        nonlocal preparations
        preparations += 1
        receipt = persist(*args, **kwargs)
        if preparations == 2:
            # This await gap follows the final preparation. The connector must
            # use that preparation's captured key, not this newer Settings value.
            settings['OPENAI_API_KEY'] = 'synthetic-third'
            scopes['third'] = host.turn.observation_fingerprint()
        return receipt

    @asynccontextmanager
    async def socket(url, **kwargs):
        expected = 'synthetic-second' if not opened else 'synthetic-third'
        assert kwargs['additional_headers']['Authorization'] == 'Bearer ' + expected
        opened.append(expected)
        yield Wire('openai')

    import websockets.asyncio.client
    monkeypatch.setattr(websockets.asyncio.client, 'connect', socket)
    monkeypatch.setattr(host, 'check', edit_before_open)
    monkeypatch.setattr(observability, 'persist_physical_candidate', edit_while_reprepared)

    async def complete(session):
        assert host.plan.route_fp == scopes['second']
        host.response_started('account-witness')
        host.response_done('account-witness', usage=token_usage(text_input_tokens=150000),
                           raw_usage={'input_tokens': 150000}, status='completed')
        assert host._renewal_window.accepted == 150000

    async def renewed(session):
        assert host.plan.route_fp == scopes['third']
        assert host._renewal_window.accepted == 0
        assert host._renewal_window.total_equivalent == 133554

    async def run():
        await host.session(complete, LiveOptions())
        await host.session(renewed, LiveOptions())

    asyncio.run(run())
    records = ce._load(root)['main_windows']
    assert first_scope not in records and scopes['third'] not in records
    assert records[scopes['second']]['observations'][0]['accepted'] == 150000
    assert len(opened) == 2 and preparations == 3
    assert api.get_settings(['OPENAI_API_KEY']) == settings  # capture is no longer active


@pytest.mark.parametrize('host_world', [('low', 500_000, 'estimator')], indirect=True)
def test_account_repreparation_starts_from_the_lease_source_not_the_discarded_fit(host_world, monkeypatch):
    from ouroboros import observability
    from ouroboros.context_window import record_observation
    from ouroboros.realtime_setup import measure_session_wire

    host = host_world.host
    source_scope = host.turn.observation_fingerprint()
    schema = {'type': 'function', 'function': {'name': 'fixture_low_tool', 'description': 'Low tool schema. ' * 30_000}}
    host.schemas.append(schema)
    owner_words = {'role': 'user', 'content': 'Preserve these exact synthetic owner words.'}
    host.messages.append(owner_words)
    factory, options = host_world.provider['session'], LiveOptions()
    nano = host.plan.reproject_transcript(host.messages, 'nano')
    _, small, _, _ = measure_session_wire(host, replace(host.plan, initial_mode='nano'), nano,
        host._tool_schemas_for_mode('nano'), factory, options)
    persist = observability.persist_physical_candidate
    preparations = []

    def record(*args, **kwargs):
        receipt = persist(*args, **kwargs)
        attempt = kwargs['attempt_id']
        if attempt.endswith(('-2', '-2-rebound')):
            preparations.append(kwargs['candidate_facts']['fit']['measurement']['rendered_mode'])
            if attempt.endswith('-2'):
                host_world.settings['TEST_CREDENTIAL'] = 'synthetic-new-account'
        return receipt

    monkeypatch.setattr(observability, 'persist_physical_candidate', record)

    async def run():
        ready, stop = asyncio.Event(), asyncio.Event()
        async def serve(session):
            ready.set()
            await stop.wait()

        serving = asyncio.create_task(host.session(serve, options))
        try:
            await asyncio.wait_for(ready.wait(), 30)
            original = host._connection_view
            record_observation(host.ctx.drive_root, source_scope,
                refused=small.prompt_tokens_estimate + 30_000, kind='total', basis='provider_stated')
            lease = await host.prepare_session(options)
            try:
                assert preparations == ['nano', 'low']
                assert host._connection_view is original and not original._context_session.closed
                assert host.plan.initial_mode == 'low' and schema in host.schemas

                async def consume(session):
                    assert not original._context_session.closed
                    assert host.plan.initial_mode == 'low' and host.plan.route_fp != source_scope
                    rows = json.loads(session.selected.messages_json)
                    assert rows.count(owner_words) == 1 and schema in json.loads(session.selected.tools_json)
                    assert sum('Fresh view before this facts row:' in str(row['content']) for row in rows) == 1
                await lease.run(consume)
            finally:
                await lease.close()
        finally:
            stop.set()
            await asyncio.wait_for(serving, 30)
        assert sum(session.opened for session in host_world.sessions) == 2  # first fit was never opened

    asyncio.run(run())


def test_capture_is_private_per_api_and_still_obeys_the_grant(tmp_path):
    from ouroboros.realtime_credentials import SessionCredentials

    settings = {'OPENAI_API_KEY': 'synthetic-current', 'TOTAL_BUDGET': 5}  # and an ordinary, ungranted setting
    def make():
        return PluginAPIImpl(_PluginAPIConfig(skill_name='fixture', permissions=['read_settings'],
            env_allowlist=['OPENAI_API_KEY', 'TOTAL_BUDGET'], granted_keys=['OPENAI_API_KEY'],
            state_dir=tmp_path, settings_reader=lambda: dict(settings)))
    api, other = make(), make()
    capture = SessionCredentials(api, {'OPENAI_API_KEY': 'synthetic-captured'}, 'synthetic-scope')
    assert 'synthetic-captured' not in repr(capture)
    with capture.activate():
        assert api.get_settings(settings) == {**settings, 'OPENAI_API_KEY': 'synthetic-captured'}
        assert other.get_settings(settings) == settings
        api._granted_upper = frozenset()
        assert api.get_settings(settings) == {'TOTAL_BUDGET': 5}
    assert other.get_settings(settings) == settings
