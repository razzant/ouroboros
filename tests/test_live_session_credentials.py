"""Granted connector reads and account evidence share one session capture."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace

from ouroboros import capability_evidence as ce
from ouroboros.contracts.live_conversation import LiveOptions
from ouroboros.extension_plugin_api import PluginAPIImpl, _PluginAPIConfig
from tests.test_grok_openai_protocol import providers as providers, deny_network as deny_network, Wire
from tests.test_live_conversation import host_world as host_world
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
        assert kwargs['extra_headers']['Authorization'] == 'Bearer ' + expected
        opened.append(expected)
        yield Wire('openai')

    import websockets.legacy.client
    monkeypatch.setattr(websockets.legacy.client, 'connect', socket)
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


def test_capture_is_private_per_api_and_still_obeys_the_grant(tmp_path):
    from ouroboros.realtime_credentials import SessionCredentials

    settings = {'OPENAI_API_KEY': 'synthetic-current', 'theme': 'dark'}
    def make():
        return PluginAPIImpl(_PluginAPIConfig(skill_name='fixture', permissions=['read_settings'],
            env_allowlist=['OPENAI_API_KEY', 'theme'], granted_keys=['OPENAI_API_KEY'],
            state_dir=tmp_path, settings_reader=lambda: dict(settings)))
    api, other = make(), make()
    capture = SessionCredentials(api, {'OPENAI_API_KEY': 'synthetic-captured'}, 'synthetic-scope')
    assert 'synthetic-captured' not in repr(capture)
    with capture.activate():
        assert api.get_settings(settings) == {**settings, 'OPENAI_API_KEY': 'synthetic-captured'}
        assert other.get_settings(settings) == settings
        api._granted_upper = frozenset()
        assert api.get_settings(settings) == {'theme': 'dark'}
    assert other.get_settings(settings) == settings
