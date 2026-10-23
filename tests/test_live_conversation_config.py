"""Feature configuration checks credential availability without returning or logging values."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from ouroboros.contracts.realtime import CatalogModel, CatalogVoice, ProviderCatalog, RouteFacts
from skills.live_conversation import plugin


class API:
    def __init__(self, root):
        self.root, self.routes, self.sections = root, {}, []
    def get_state_dir(self): return str(self.root / 'feature')
    def get_runtime_info(self): return {'data_dir': str(self.root)}
    def get_settings(self, _keys): raise AssertionError('availability uses the host and engine API')
    def register_route(self, name, handler, **_kwargs): self.routes[name] = handler
    def register_live_conversation(self, name, handler, **kwargs): self.feature = kwargs
    def register_settings_section(self, *args, **kwargs): self.sections.append((args, kwargs))


@pytest.fixture
def config_api(tmp_path, monkeypatch):
    from ouroboros import extension_registry_state as registry, realtime_setup
    catalog = ProviderCatalog('Fixture voice', (CatalogModel('fixture-model', 'Fixture model', default=True, vision=True,
        voices=(CatalogVoice('fixture-voice', 'Fixture voice'),), honours=frozenset({'pause'}),
        facts=RouteFacts(conversation_window_tokens=12345, source='fixture measured'),
        notes=(('search', 'Fixture does not search.'),)),), ('TEST_CREDENTIAL',), 'Fixture audio stays synthetic.')
    skill = SimpleNamespace(name='fixture-engine')
    providers = {'fixture-engine:voice': {'skill':skill.name, 'catalog':catalog}}
    granted = ['TEST_CREDENTIAL']
    credentials = {'TEST_CREDENTIAL': 'test-credential'}
    engine_api = SimpleNamespace(get_settings=lambda keys: {key: credentials[key] for key in keys
        if key in granted and key in credentials})
    extensions = {skill.name: SimpleNamespace(loaded_skill=skill, api_instances=[engine_api])}
    for module in (registry, realtime_setup):
        monkeypatch.setattr(module, '_realtime_providers', providers)
        monkeypatch.setattr(module, '_extensions', extensions)
        monkeypatch.setattr(module, '_unloading', set())
    api = API(tmp_path)
    plugin.register(api)
    return api, granted, providers, credentials


def request(api, value=None):
    async def body(): return value
    response = asyncio.run(api.routes['config'](SimpleNamespace(method='GET' if value is None else 'POST', json=body)))
    return response.status_code, json.loads(response.body)


def test_offer_requires_a_granted_value_and_never_returns_or_logs_a_secret(config_api, caplog, capsys):
    caplog.set_level('DEBUG')
    api, granted, providers, credentials = config_api
    responses = []
    def checked_request(value=None):
        status, data = request(api, value)
        responses.append(data)
        return status, data
    status, data = checked_request()
    assert status == 200 and data['offered'] is False and data['provider'] == 'gemini_live:gemini'
    assert data['catalogs'][0]['id'] == 'fixture-engine:voice'
    model = data['catalogs'][0]['models'][0]
    assert model['voices'][0]['id'] == 'fixture-voice' and model['facts']['conversation_window_tokens'] == 12345
    value = {'enabled':True, 'provider':'fixture-engine:voice', 'model':'fixture-model', 'voice':'fixture-voice'}
    status, data = checked_request(value)
    assert status == 200 and data['offered'] is True and data['selected_model']['vision'] is True
    credentials.clear()  # auto-granted permission, but no configured value
    assert checked_request()[1]['offered'] is False
    credentials['TEST_CREDENTIAL'] = ''
    assert checked_request()[1]['offered'] is False
    credentials['TEST_CREDENTIAL'] = 'test-credential'
    assert checked_request()[1]['offered'] is True
    granted.clear()
    assert checked_request()[1]['offered'] is False
    granted.append('TEST_CREDENTIAL')
    assert checked_request()[1]['offered'] is True
    assert checked_request({**value, 'enabled':False})[1]['offered'] is False
    assert checked_request(value)[1]['offered'] is True
    providers.clear()
    assert checked_request()[1]['offered'] is False
    assert checked_request({**value, 'enabled':False})[0] == 200
    captured = capsys.readouterr()
    assert 'test-credential' not in json.dumps(responses)
    assert 'test-credential' not in caplog.text + captured.out + captured.err


def test_unknown_catalog_selections_are_refused_and_call_tools_registered(config_api):
    api, _, _, _ = config_api
    for fields in ({'model':'invented'}, {'voice':'invented'}, {'provider':'missing:engine'}):
        status, data = request(api, {'enabled':True, 'provider':'fixture-engine:voice', **fields})
        assert status == 400 and data['error']
    assert {tool.name for tool in api.feature['call_tools']} == {'end_call','set_call_languages'}
    assert all(tool.silent for tool in api.feature['call_tools'])
    from ouroboros.extension_ui_validation import validate_settings_schema
    validate_settings_schema(api.sections[0][1]['schema'])


@pytest.mark.parametrize('auto_grant', [False, True])
def test_default_call_skills_install_without_credentials_and_leave_beta_off(tmp_path, monkeypatch, auto_grant):
    """Default-on call skills never open a call by themselves.

    With the owner's own auto-grant setting on, a reviewed engine gets its key grant like any reviewed skill;
    Calls (beta) stays off either way, so no Talk, media or connection follows from the install.
    """
    from pathlib import Path
    from ouroboros.skill_builtin import default_on_names, install_builtin_skill
    from ouroboros.skill_install_record import read_install_record
    from ouroboros.skill_loader import load_skill, grant_status_for_skill
    from ouroboros import config
    seeds = Path(__file__).resolve().parents[1] / 'skills'
    names = ('live_conversation', 'gemini_live', 'grok_realtime', 'openai_realtime')
    assert set(names) <= set(default_on_names(seeds))
    monkeypatch.setenv('OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS', 'true')
    monkeypatch.setenv('OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS', str(auto_grant).lower())
    monkeypatch.setattr(config, 'SETTINGS_PATH', tmp_path / 'empty-test-settings.json')
    for name in names:
        result = install_builtin_skill(name, drive_root=tmp_path, seed_dir=seeds, turn_on_when_ready=True)
        assert result['ok'], result
        assert read_install_record(tmp_path, name)['turn_on_when_ready'] is True
        skill = load_skill(tmp_path / 'skills' / 'native' / name, tmp_path)
        granted = grant_status_for_skill(tmp_path, skill)['granted_keys']
        if not auto_grant:
            assert result['auto_granted_keys'] == [] and granted == []
    api = API(tmp_path)
    plugin.register(api)
    assert api.sections[0][1]['title'] == 'Calls (beta)'
    assert api.feature['configure']().enabled is False
    assert plugin.configuration_view(api.feature['configure'](), [])['offered'] is False


def test_call_setup_copy_describes_default_on_skills_and_the_owner_flow(config_api):
    from pathlib import Path
    api, _, _, _ = config_api
    intro = ' '.join(component['text'] for component in api.sections[0][1]['schema']['components']
                     if component['type'] == 'markdown')
    skill = ' '.join((Path(__file__).resolve().parents[1] / 'skills/live_conversation/SKILL.md').read_text(encoding="utf-8").split())
    for text in (intro, skill):
        assert 'on by default' in text
        assert 'turn on calls (beta)' in text.lower()
        assert 'choose an engine' in text.lower()
        assert 'add its API key' in text
        assert 'Enable this feature' not in text and 'Choose an enabled voice engine' not in text
        assert 'grant that engine its credential' not in text
    assert 'auto-grant' in skill and 'with it off' in skill
    assert 'Calls are a beta, off by default' in skill


@pytest.mark.parametrize('engine', ['openai', 'gemini_extended'])
@pytest.mark.parametrize('saved', [None, '', 'low', 'high'])
def test_unset_thinking_uses_the_selected_catalog_lowest_level_without_changing_saved_choices(engine, saved):
    from skills.live_conversation.driver import session_options
    from skills.openai_realtime.plugin import CATALOG as OPENAI
    from skills.gemini_live.plugin import CATALOG as GEMINI
    model = OPENAI.models[0] if engine == 'openai' else next(model for model in GEMINI.models if model.thinking_levels)
    raw = {} if saved is None else {'thinking': saved}
    config = plugin.validate(raw)
    host = SimpleNamespace(config=config, model=model, state={})
    assert config.thinking == ('' if saved is None else saved)
    assert session_options(host).thinking == (saved or ('minimal' if engine == 'openai' else 'low'))


def test_the_untouched_beta_switch_is_never_written_so_a_later_default_reaches_this_install(config_api, tmp_path):
    api, _, _, _ = config_api
    path = tmp_path / 'feature' / 'conversation.json'
    choice = {'provider': 'fixture-engine:voice', 'model': 'fixture-model'}
    status, _ = request(api, {**choice, 'enabled': False})
    assert status == 200 and 'enabled' not in json.loads(path.read_text(encoding='utf-8'))
    status, view = request(api, {**choice, 'enabled': True})
    assert status == 200 and view['enabled'] is True and json.loads(path.read_text(encoding='utf-8'))['enabled'] is True
    status, view = request(api, {**choice, 'enabled': False})
    # Once flipped, the owner's own choice stays written, default or not.
    assert status == 200 and view['enabled'] is False and json.loads(path.read_text(encoding='utf-8'))['enabled'] is False
