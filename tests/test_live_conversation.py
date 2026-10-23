"""Isolated consumers only. No credentials, provider sockets or device capture.

Run through scripts/safe_test.py. Echoes prove our consumers, not vendor acceptance.
Raw wire/ledger receipts are exported even when assertions fail.
"""
from __future__ import annotations
import asyncio
import base64
import json
import shutil
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests._extension_loader_shared import _clear_loader_state  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
ENGINES = [('openai_realtime', 'openai'), ('grok_realtime', 'grok'), ('gemini_live', 'gemini')]


def wait_for(predicate, timeout=10):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        value = predicate()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError("timed out waiting for live call evidence")


@pytest.fixture
def packages(tmp_path):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import (find_skill, save_enabled, SkillReviewState,
                                       requested_skill_permissions, save_skill_grants)
    drive = tmp_path / 'packages'
    found = {}
    for name in ['live_conversation', *(name for name, _ in ENGINES)]:
        directory = drive / 'skills/external' / name
        shutil.copytree(ROOT / 'skills' / name, directory)
        skill = find_skill(drive, name)
        save_enabled(drive, name, True)
        skill = replace(skill, enabled=True, manifest=replace(skill.manifest, env_from_settings=[]),
                        review=SkillReviewState(status='pass', content_hash=skill.content_hash))
        permissions = requested_skill_permissions(skill.manifest.permissions, skill.manifest.subscribe_events)
        save_skill_grants(drive, name, [], content_hash=skill.content_hash, requested_keys=[],
                          granted_permissions=permissions, requested_permissions=permissions)
        assert extension_loader.load_extension(skill, lambda: {}, drive_root=drive) is None
        found[name] = skill
    yield drive, found
    for name in found:
        extension_loader.unload_extension(name)


class VoiceWire:
    """Synthetic vendor-dialect fixture with one ordinary read_file tool call."""
    def __init__(self, engine, path):
        self.engine, self.path = engine, path
        self.sent, self.received = [], []
        self.events = asyncio.Queue()
        self.session = {}
        self.invoked = False
        self.result_received = threading.Event()
        self.closed = False

    def put(self, value):
        self.events.put_nowait(value)

    async def recv(self):
        value = await self.events.get()
        self.received.append(value)
        return json.dumps(value)

    def start_tool(self):
        if self.invoked:
            return
        self.invoked = True
        args = {'path': str(self.path)}
        if self.engine == 'gemini':
            self.put({'serverContent': {'inputTranscription': {'text': 'Read the synthetic canary.', 'finished': True}}})
            self.put({'toolCall': {'functionCalls': [{'id': 'read-1', 'name': 'read_file', 'args': args}]}})
        else:
            self.put({'type': 'conversation.item.input_audio_transcription.completed', 'item_id': 'owner-1',
                      'transcript': 'Read the synthetic canary.', 'status': 'completed'})
            self.put({'type': 'response.created', 'response': {'id': 'tools-1'}})
            self.put({'type': 'response.done', 'usage': {'input_audio_duration': 2}, 'response': {
                'id': 'tools-1', 'status': 'completed', 'usage': {'input_tokens': 100, 'output_tokens': 5},
                'output': [{'type': 'function_call', 'call_id': 'read-1', 'name': 'read_file',
                            'arguments': json.dumps(args)}]}})

    def answer(self):
        pcm = base64.b64encode(b'\x00\x00' * 480).decode()
        if self.engine == 'gemini':
            self.put({'serverContent': {'modelTurn': {'parts': [{'inlineData': {
                'mimeType': 'audio/pcm;rate=24000', 'data': pcm}}]},
                'outputTranscription': {'text': 'Synthetic canary read.'}}})
            self.put({'usageMetadata': {'promptTokenCount': 120, 'responseTokenCount': 12},
                      'serverContent': {'turnComplete': True}})
        else:
            self.put({'type': 'response.created', 'response': {'id': 'answer-1'}})
            self.put({'type': 'response.output_audio.delta', 'item_id': 'answer-item', 'delta': pcm})
            self.put({'type': 'response.output_audio_transcript.done', 'item_id': 'answer-item',
                      'transcript': 'Synthetic canary read.'})
            self.put({'type': 'response.done', 'usage': {'input_audio_duration': 3}, 'response': {
                'id': 'answer-1', 'status': 'completed', 'usage': {'input_tokens': 120, 'output_tokens': 12}, 'output': []}})

    async def send(self, raw):
        event = json.loads(raw)
        self.sent.append(event)
        kind = event.get('type')
        if 'setup' in event:
            self.put({'setupComplete': {}})
        elif kind == 'session.update':
            for key, value in event['session'].items():
                if key == 'audio' and key in self.session:
                    for direction, fields in value.items():
                        self.session['audio'].setdefault(direction, {}).update(fields)
                else:
                    self.session[key] = value
            self.put({'type': 'session.updated', 'session': self.session.copy()})
        elif kind == 'conversation.item.create':
            self.put({'type': 'conversation.item.done', 'item': event['item'],
                      'previous_item_id': event.get('previous_item_id')})
            if event['item']['type'] == 'function_call_output':
                self.result_received.set()
        elif kind == 'response.create':
            self.answer()
        elif kind == 'input_audio_buffer.append' or 'audio' in event.get('realtimeInput', {}):
            self.start_tool()
        elif 'toolResponse' in event:
            self.result_received.set()
            self.answer()
        elif kind == 'conversation.item.truncate':
            self.put({'type': 'conversation.item.truncated', 'item_id': event['item_id'], 'audio_end_ms': event['audio_end_ms']})

    @asynccontextmanager
    async def connect(self, model):
        try:
            yield self
        finally:
            self.closed = True



class FakeSession:
    output_bound_tokens = 100
    renewal_ready = True

    def __init__(self, selected, options, *, observe):
        self.selected, self.options, self.observe = selected, options, observe
        self.wire = {'messages': json.loads(selected.messages_json), 'tools': json.loads(selected.tools_json),
                     'instructions': options.instructions + (' Speech recognition should expect: ' + ', '.join(options.languages) + '.' if options.languages else '')}
        self.opened = self.closed = False

    async def open(self):
        self.opened = True

    async def close(self):
        self.closed = True


@pytest.fixture
def host_world(tmp_path, monkeypatch, provider_catalog_offline, request):
    from ouroboros import config, context, extension_registry_state as registry
    from ouroboros.contracts.live_conversation import LiveConfig, CallTool
    from ouroboros.contracts.realtime import CatalogModel, ProviderCatalog, RouteFacts, RealtimeRoute
    from ouroboros.live_conversation import LiveConversationTurn, ConversationHost
    from ouroboros.call_protocol import CallSpec
    from ouroboros.call_peers import LiveCallHub
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.tool_context import ToolContext
    from ouroboros.task_results import write_task_result, STATUS_RUNNING
    from tests._memory_view_context import world
    from tests.test_budget_fence_liveness import _publish
    from ouroboros.model_wait import task_model_wait_scope
    from supervisor import message_bus

    env, memory, rooms = world(tmp_path / 'world')
    owner_mode, window, basis = getattr(request, 'param', ('nano', 200000, 'provider'))
    if hasattr(request, 'param'):
        for relative in ('BIBLE.md', 'prompts/SYSTEM.md'):
            shutil.copyfile(ROOT / relative, env.repo_dir / relative)
        shutil.copytree(ROOT / 'docs', env.repo_dir / 'docs', dirs_exist_ok=True)
        memory.identity_path().write_text(memory._default_identity())
        memory.scratchpad_path().write_text(memory._default_scratchpad() + '\nCurrent project notes. ' * 180)
    monkeypatch.setattr(config, 'DATA_DIR', env.drive_root)
    monkeypatch.setenv('OUROBOROS_DATA_DIR', str(env.drive_root))
    monkeypatch.setattr(context, 'get_context_mode', lambda: owner_mode)
    monkeypatch.setattr(message_bus, 'DATA_DIR', env.drive_root)
    monkeypatch.setattr(message_bus, '_BRIDGE', message_bus.LocalChatBridge())
    settings = {'TEST_CREDENTIAL': 'test-credential'}
    async def language(host, arguments):
        host.state['languages'] = arguments['languages']
        return 'Language selection retained.'
    async def handler(host):
        pass
    call_tool = CallTool('set_test_language', 'Set the test language.',
        {'type': 'object', 'properties': {'languages': {'type': 'array', 'items': {'type': 'string'}}}}, language, silent=True)
    live_config = LiveConfig(enabled=True, provider='fixture:voice')
    feature = dict(skill='fixture', key='fixture:call', extension_generation='generation',
                   content_hash='fixture-content', configure=lambda: live_config, handler=handler, call_tools=(call_tool,))
    catalog = ProviderCatalog('Fixture engine', (CatalogModel('voice-test', 'Voice test',
        RouteFacts(window, 250000, 10000, basis, 'fixture', '2026-10-21'), default=True, vision=True),),
        ('TEST_CREDENTIAL',))
    sessions = []
    def factory(*args, **kw):
        value = FakeSession(*args, **kw)
        sessions.append(value)
        return value
    provider = dict(skill='fixture', extension_generation='generation', content_hash='fixture-content',
        catalog=catalog, route=RealtimeRoute('fixture_live', 'wss://example.invalid/live', 'fixture::'), session=factory)
    bundle = registry._ExtensionRegistrations(generation_digest='generation',
        realtime_providers=['fixture:voice'], conversation_features=['fixture:call'],
        api_instances=[SimpleNamespace(get_settings=lambda keys: dict(settings))])
    monkeypatch.setitem(registry._extensions, 'fixture', bundle)
    monkeypatch.setitem(registry._realtime_providers, 'fixture:voice', provider)
    monkeypatch.setitem(registry._conversation_features, 'fixture:call', feature)
    spec = CallSpec('fixture-call', rooms['alpha'], live_config.provider, '', '', client_message_id='call:fixture-call:open')
    turn = LiveConversationTurn(feature, live_config, spec, publish=lambda event: None)
    task = dict(id='fixture-call-task', type='task', text=turn.opening, _is_direct_chat=True,
        chat_id=spec.chat_id, model=turn.model, use_local_model=False,
        origin_message_ref={'chat_id': spec.chat_id, 'client_message_id': spec.client_message_id},
        metadata={'execution_transport': turn.request, 'host_fact': {'kind': 'call_open'}})
    write_task_result(env.drive_root, task['id'], STATUS_RUNNING, chat_id=spec.chat_id)
    _publish(env.drive_root)
    ctx = ToolContext(repo_dir=env.repo_dir, drive_root=env.drive_root, branch_dev='test', task_id=task['id'],
        current_chat_id=spec.chat_id, is_direct_chat=True, task_metadata=task['metadata'], task_contract={})
    progress = []
    ctx.emit_progress_fn = lambda text, **kw: progress.append({'text': text, **kw})
    tools = ToolRegistry(env.repo_dir, env.drive_root)
    tools.set_context(ctx)
    messages, _ = context.build_llm_messages(env, memory, task, ctx=ctx, tool_schemas=tools.schemas(), route_resolver=turn.resolve_route)
    hub = LiveCallHub(turn.spec)
    with task_model_wait_scope(task=task, drive_root=env.drive_root, event_queue=None, worker_slot_held=False) as owner:
        ctx.model_wait_context, owner.tool_context = owner, ctx
        host = ConversationHost(turn, ctx, messages, tools, task, '', hub)
        yield SimpleNamespace(host=host, env=env, ctx=ctx, settings=settings, config=live_config, provider=provider,
            sessions=sessions, feature=feature, task=task, progress=progress)
        asyncio.run(host.finish())
        hub.close('fixture_done')
        registry._extensions.pop('fixture', None)
        registry._realtime_providers.pop('fixture:voice', None)
        registry._conversation_features.pop('fixture:call', None)


def test_typed_port_default_model_offer_and_unknown_model(host_world):
    from ouroboros.live_conversation import call_offer, LiveConversationTurn
    from ouroboros.contracts.realtime import RealtimeSetupError
    state, host = host_world, host_world.host
    assert host.config is state.config and host.model.id == 'voice-test'
    assert host.state == {} and host.opening
    assert call_offer(state.config)['offered'] is True
    assert call_offer(replace(state.config, enabled=False))['reason'] == 'calls_disabled'
    assert call_offer(replace(state.config, provider='missing:voice'))['reason'] == 'provider_unavailable'
    state.settings.clear()
    assert call_offer(state.config)['reason'] == 'credential'
    state.settings['TEST_CREDENTIAL'] = 'test-credential'
    with pytest.raises(RealtimeSetupError, match='model_unavailable'):
        LiveConversationTurn(state.feature, replace(state.config, model='unknown'), host.turn.spec, publish=lambda _: None)


@pytest.mark.parametrize('host_world', [(mode, window, basis) for mode in ('max', 'low', 'nano')
                                      for window, basis in ((80000, 'estimator'), (92000, 'provider'),
                                                            (500000, 'provider'))], indirect=True)
def test_real_books_call_uses_ordinary_owner_mode_and_window_ladder(host_world):
    from ouroboros.context_fit import estimate_context_prompt_tokens
    from ouroboros.contracts.live_conversation import LiveOptions
    from ouroboros.loop_model_call import _main_context_profile
    from ouroboros.tool_policy import initial_tool_schemas
    from skills.live_conversation.driver import CALL_INSTRUCTIONS, languages_opening_sentence

    host = host_world.host
    original = host.plan
    facts = host.model.facts
    options = LiveOptions(instructions=CALL_INSTRUCTIONS + '\n' + languages_opening_sentence(('en-US',)),
                          languages=('en-US',))
    rows = [*host.messages, {'role': 'system', 'content': options.instructions}]
    args = dict(window_tokens=facts.conversation_window_tokens, known_window=True, ratio=1.0,
                output_reserve=original.output_reserve_tokens, tool_schemas=host.schemas,
                current_messages=rows)
    ordinary = original.reproject_for_route(**args, start_mode=original.initial_mode)
    projected = ordinary.reproject_transcript(host.messages, ordinary.initial_mode)
    if (ordinary.initial_mode == 'max'
            and estimate_context_prompt_tokens(projected, host.schemas) + facts.working_reserve_tokens > facts.conversation_window_tokens):
        projected = [*projected, {'role': 'system', 'content': options.instructions}]
        ordinary = ordinary.reproject_for_route(**{**args, 'current_messages': projected},
                                                start_mode='max', select_mode='low')
        ordinary = ordinary.reproject_for_route(**{**args, 'current_messages': ordinary.reproject_transcript(projected, 'low')},
                                                start_mode='low')
    expected = ordinary.reproject_transcript(host.messages, ordinary.initial_mode)[0]
    expected_tools = initial_tool_schemas(host.tools, context_mode=ordinary.initial_mode)
    expected_tools.extend(schema for schema in host.schemas if schema['function']['name'] in host.call_tools)

    async def consume(session):
        assert session.opened
        assert json.loads(session.selected.messages_json)[0] == expected
        assert json.loads(session.selected.tools_json) == expected_tools
        assert session.selected.mode == ordinary.initial_mode
        assert host._physical_context.profile == _main_context_profile(ordinary, ordinary.initial_mode)
    asyncio.run(host.session(consume, options))
    assert host_world.sessions[-1].closed
    assert host.plan.preferred_mode == original.preferred_mode
    if facts.conversation_window_tokens == 500000:
        assert host.plan.initial_mode == original.preferred_mode


def test_every_session_refits_current_canonical_rows_and_options(host_world, monkeypatch):
    from ouroboros.context_fit import ContextFitPlan
    from ouroboros.contracts.live_conversation import LiveOptions
    host = host_world.host
    refits = []
    original = ContextFitPlan.reproject_for_route
    def refit(plan, **kwargs):
        refits.append(json.loads(json.dumps(kwargs['current_messages'])))
        return original(plan, **kwargs)
    monkeypatch.setattr(ContextFitPlan, 'reproject_for_route', refit)
    async def consume(session):
        assert session.opened
    async def run():
        await host.session(consume, LiveOptions(instructions='Feature then owner instructions', languages=('ru-RU',)))
        await host.observe_context({'content': 'NEW PEER FACT'})
        await host.session(consume, LiveOptions(instructions='Updated instructions'))
    asyncio.run(run())
    assert len(refits) == 2
    assert 'NEW PEER FACT' not in canonical(refits[0]) and 'NEW PEER FACT' in canonical(refits[1])
    assert 'Feature then owner instructions' in canonical(refits[0])
    assert all(session.closed for session in host_world.sessions)
    assert host_world.sessions[0].options.languages == ('ru-RU',)
    assert host.messages[-1]['_host_context_kind'] == 'call_observation'


def canonical(value):
    return json.dumps(value, ensure_ascii=False)


def test_physical_dispatch_minus_one_refuses_and_positive_passes(host_world, monkeypatch):
    from ouroboros.context_fit import ContextFitPlan
    from ouroboros.contracts.live_conversation import LiveOptions
    from ouroboros.contracts.realtime import RealtimeSetupError
    from ouroboros.realtime_setup import session_candidate
    host = host_world.host
    host.model = replace(host.model, facts=replace(host.model.facts, basis='estimator'))
    # An irreducible ordinary projection: the physical guard must still refuse.
    monkeypatch.setattr(ContextFitPlan, 'reproject_for_route', lambda plan, **kw: plan)
    _, request, _ = session_candidate(host, host_world.provider['session'], LiveOptions())
    raw = request.prompt_tokens_bounded_estimate
    host.model = replace(host.model, facts=replace(host.model.facts,
        conversation_window_tokens=raw + 10000 - 1))
    async def consume(session):
        pytest.fail('known overflowing physical request was dispatched')
    with pytest.raises(RealtimeSetupError, match='context_capacity'):
        asyncio.run(host.session(consume, LiveOptions()))
    assert not any(session.opened for session in host_world.sessions)
    host.model = replace(host.model, facts=replace(host.model.facts,
        conversation_window_tokens=raw + 10000 + 1))
    async def allowed(session):
        assert session.opened
        host.response_started('fits')
        host.response_done('fits', usage={}, raw_usage={}, status='completed')
    asyncio.run(host.session(allowed, LiveOptions()))
    from ouroboros.usage_accounting import read_usage_records
    row, = read_usage_records(host.ctx.drive_root)
    assert row['physical_context']['capacity_total_tokens'] == raw + 10001
    assert row['physical_context']['measurement_density'] == 1.0


def test_provider_density_uses_only_selected_text_and_estimator_stays_raw(host_world):
    from ouroboros.capability_evidence import resolve_main_token_density
    from ouroboros.contracts.live_conversation import LiveOptions
    from ouroboros.realtime_setup import session_candidate
    host = host_world.host
    options = LiveOptions(instructions='Feature instructions', languages=('ru-RU', 'en-US'))
    _, cold, _ = session_candidate(host, host_world.provider['session'], options)
    assert cold.physical_context.measurement_density == 1.0
    # A caption describes native audio; it is not an addition to selected TEXT.
    host.messages.append({'role': 'user', 'content': 'Native audio caption ' * 300})
    host.response_started('density')
    estimate = host._response_text_estimates['density']
    assert estimate == host._selected_text_estimate
    host.messages.pop()
    host.response_done('density', usage={'text_input_tokens': estimate * 2,
        'audio_input_tokens': 99999, 'image_input_tokens': 88888, 'video_input_tokens': 77777},
        raw_usage={'input_tokens_details': {'text_tokens': estimate * 2, 'audio_tokens': 99999}}, status='completed')
    assert resolve_main_token_density(host.ctx.drive_root, host.plan.route_fp, host.turn.model, view_kind=host.plan.view_kind)[0] == 2
    _, warm, _ = session_candidate(host, host_world.provider['session'], options)
    assert warm.physical_context.measurement_density == 2
    assert warm.prompt_tokens_estimate == warm.prompt_tokens_bounded_estimate * 2
    host.model = replace(host.model, facts=replace(host.model.facts, basis='estimator'))
    _, raw, _ = session_candidate(host, host_world.provider['session'], options)
    assert raw.physical_context.measurement_density == 1
    assert raw.prompt_tokens_estimate == raw.prompt_tokens_bounded_estimate
    host.response_started('grok')
    host.response_done('grok', usage={'text_input_tokens': estimate * 3},
        raw_usage={'input_tokens_details': {'text_tokens': estimate * 3}}, status='completed')
    host.model = replace(host.model, facts=replace(host.model.facts, basis='provider'))
    host.response_started('audio-only')
    host.response_done('audio-only', usage={'audio_input_tokens': estimate * 4}, raw_usage={}, status='completed')
    assert resolve_main_token_density(host.ctx.drive_root, host.plan.route_fp, host.turn.model, view_kind=host.plan.view_kind)[0] == 2


def test_instructions_and_provider_language_sentence_are_inside_fit(host_world):
    from ouroboros.contracts.live_conversation import LiveOptions
    from ouroboros.realtime_setup import session_candidate
    host = host_world.host
    _, plain, _ = session_candidate(host, host_world.provider['session'], LiveOptions())
    _, instructed, _ = session_candidate(host, host_world.provider['session'], LiveOptions(instructions='Owner instructions ' * 200))
    _, language, _ = session_candidate(host, host_world.provider['session'], LiveOptions(
        instructions='Owner instructions ' * 200, languages=('ru-RU', 'en-US')))
    assert plain.prompt_tokens_bounded_estimate < instructed.prompt_tokens_bounded_estimate < language.prompt_tokens_bounded_estimate


def test_call_tool_silent_replay_and_existing_row_change_receipt(host_world):
    host = host_world.host
    original = host.messages[0]
    async def edit(host, args):
        original['content'] = 'Changed existing system row.'
        return 'done'
    host.call_tools['set_test_language'] = replace(host.call_tools['set_test_language'], handler=edit)
    call = {'id': 'local-1', 'type': 'function', 'function': {'name': 'set_test_language', 'arguments': '{}'}}
    async def run():
        rows = await host.tool_batch([call])
        assert rows.silent == ('local-1',)
        assert any(row['content'] == 'Changed existing system row.' for row in rows)
        assert any(row.get('tool_call_id') == 'local-1' for row in rows)
        replay = await host.tool_batch([call])
        assert replay.silent == ('local-1',) and host.rounds == 1
    asyncio.run(run())


def test_tool_collision_refuses_but_distinct_feature_tools_join(host_world):
    from ouroboros.live_conversation import ConversationHost
    from ouroboros.contracts.realtime import RealtimeSetupError
    host = host_world.host
    assert 'set_test_language' in {s['function']['name'] for s in host.schemas}
    colliding = replace(host.call_tools['set_test_language'], name='read_file')
    host.turn.feature = {**host.turn.feature, 'call_tools': (colliding,)}
    with pytest.raises(RealtimeSetupError, match='tool_collision'):
        ConversationHost(host.turn, host.ctx, host.messages, host.tools, host.task, '', host.hub)


def test_captions_are_crash_durable_but_checkpoint_at_completed_turn(host_world, monkeypatch):
    from ouroboros.dialogue_store import iter_rows
    host = host_world.host
    checkpoints = []
    monkeypatch.setattr(host, 'checkpoint', checkpoints.append)
    async def run():
        await host.caption('owner', 'Durable owner speech', 'owner-1')
        await host.caption('owner', 'Durable owner speech', 'owner-1')
        await host.caption('ouroboros', 'Durable reply', 'answer-1', completed=True)
    asyncio.run(run())
    assert not checkpoints
    rows = list(iter_rows(host.ctx.drive_root, 'chat'))
    assert sum('Durable owner speech' in str(row) for row in rows) == 1
    spoken = [entry.row for entry in iter_rows(host.ctx.drive_root, 'chat') if entry.row.get('spoken')]
    assert len(spoken) == 1 and spoken[0]['text'] == 'Durable reply'
    assert spoken[0]['spoken']['status'] == 'played' and spoken[0]['role'] == 'assistant'
    assert spoken[0]['narration'] is True and spoken[0]['system_type'] == 'model_narration'
    assert not [entry for entry in iter_rows(host.ctx.drive_root, 'progress') if entry.row.get('spoken')]
    # No start event still leaves a settled, unknown-priced response.
    host.response_done('answer-1', usage={}, raw_usage={}, status='completed')
    assert checkpoints == ['ready']


def test_finished_call_both_halves_reach_later_turn_memory(host_world):
    from tests._memory_view_context import room_text
    host = host_world.host
    async def run():
        await host.caption('owner', 'Remember the sapphire telescope?', 'owner-memory')
        await host.caption('ouroboros', 'The sapphire telescope is in the attic.', 'reply-memory', completed=True)
        host.response_done('reply-memory', usage={}, raw_usage={}, status='completed')
        await host.finish()
    asyncio.run(run())
    later = room_text(host.ctx.drive_root, {'id': 'later-text-turn', 'chat_id': host.turn.spec.chat_id})
    assert 'Remember the sapphire telescope?' in later
    assert 'The sapphire telescope is in the attic.' in later


def test_selected_opening_is_one_labelled_host_fact(host_world):
    from ouroboros.contracts.live_conversation import LiveOptions
    from ouroboros.realtime_setup import session_candidate
    host = host_world.host
    session, _, _ = session_candidate(host, host_world.provider['session'], LiveOptions())
    rows = json.loads(session.selected.messages_json)
    openings = [row for row in rows if host.opening in str(row['content'])]
    assert len(openings) == 1
    assert '[Host fact: a live call opened; not the owner\'s words]' in str(openings[0]['content'])
    assert any(row['role'] == 'system' for row in rows)


def test_renewal_uses_reported_held_input_and_growth(host_world):
    host = host_world.host
    host.model = replace(host.model, facts=replace(host.model.facts, conversation_window_tokens=20000, working_reserve_tokens=10000))
    assert host.renewal_due() is False
    host.response_done('one', usage={'text_input_tokens': 4000}, raw_usage={}, status='completed')
    assert host.renewal_due() is False
    host.response_done('two', usage={'text_input_tokens': 8000}, raw_usage={}, status='completed')
    assert host.renewal_due() is True


@pytest.mark.parametrize('surface', ['provider', 'feature'])
@pytest.mark.parametrize('action', ['reload', 'disable'])
def test_active_call_reload_waits_and_owner_disable_cancels(host_world, monkeypatch, surface, action):
    from concurrent.futures import ThreadPoolExecutor
    import contextvars
    from ouroboros import call_peers, extension_loader as loader, extension_registry_state as registry
    from ouroboros.contracts.live_conversation import LiveOptions
    host = host_world.host
    provider_name, feature_name = 'fixture', 'fixture-feature'
    provider_bundle = registry._ExtensionRegistrations(generation_digest='generation',
        realtime_providers=['fixture:voice'], api_instances=registry._extensions[provider_name].api_instances)
    feature_bundle = registry._ExtensionRegistrations(generation_digest='generation', conversation_features=['fixture:call'])
    monkeypatch.setitem(registry._extensions, provider_name, provider_bundle)
    monkeypatch.setitem(registry._extensions, feature_name, feature_bundle)
    monkeypatch.setitem(host.turn.feature, 'skill', feature_name)
    monkeypatch.setattr(call_peers, 'get_hub', lambda call_id: host.hub)
    started, release, ended = threading.Event(), threading.Event(), threading.Event()
    sessions = []
    async def handler(port):
        async def consume(session):
            sessions.append(session)
            started.set()
            while not release.is_set():
                await asyncio.sleep(.01)
        try:
            await port.session(consume, LiveOptions())
        finally:
            ended.set()
    monkeypatch.setitem(host.turn.feature, 'handler', handler)
    name = provider_name if surface == 'provider' else feature_name
    subject = SimpleNamespace(name=name)
    current = {'skill': name, 'enabled': True, 'desired_live': True,
               'loaded_present': True, 'live_loaded': False, 'reason': 'ready'}
    monkeypatch.setattr(loader, 'runtime_state_for_skill_name', lambda *a, **kw: dict(current))
    monkeypatch.setattr(loader, '_finalize_extension_reconcile', lambda *a, **kw: None)
    loaded = []
    monkeypatch.setattr(loader, 'load_extension', lambda *a, **kw: loaded.append(a[0]) or None)
    def reconcile():
        return loader.reconcile_extension(name, host.ctx.drive_root, lambda: {}, skills=[subject], repo_path='')
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(contextvars.copy_context().run, host.turn.run,
            host.ctx, list(host.messages), host.tools, task=host.task, initial_effort='')
        try:
            wait_for(started.is_set, timeout=30)
            if action == 'reload':
                refused = reconcile()
                assert refused['reason'] == 'live_call_active'
                assert refused['action'] == 'extension_reload_pending'
                assert not loaded and not ended.is_set()
                release.set()
                running.result(timeout=10)
                assert reconcile()['action'] == 'extension_loaded'
                assert loaded == [subject]
            else:
                current.update(enabled=False, desired_live=False, reason='disabled')
                assert reconcile()['action'] == 'extension_unloaded'
                result = running.result(timeout=10)
                assert result[1]['reason_code'] == 'cancelled'
                assert name not in registry._extensions and ended.is_set()
            assert sessions[-1].closed
        finally:
            release.set()
            running.result(timeout=10)


def test_selected_text_estimator_excludes_media():
    from ouroboros.realtime_setup import text_input_estimate
    text = {'type': 'text', 'text': 'Selected text'}
    def estimate(parts):
        return text_input_estimate([{'role': 'user', 'content': parts}], [])
    baseline = estimate([text])
    assert estimate([text, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + 'eA==' * 5000}},
                     {'type': 'input_audio', 'input_audio': {'data': 'eA==' * 5000}}]) == baseline
    assert estimate([{'type': 'text', 'text': 'More selected text ' * 100}]) > baseline


def test_call_observations_are_bound_to_the_granted_account(host_world):
    from ouroboros.context_window import record_observation
    from ouroboros.contracts.live_conversation import LiveOptions
    from ouroboros.realtime_setup import session_candidate

    host = host_world.host
    first = host.plan.route_fp
    record_observation(host.ctx.drive_root, first, refused=50_000)
    _, request, _ = session_candidate(host, host_world.provider["session"], LiveOptions())
    assert request.physical_context.capacity_total_tokens == 50_000
    host_world.settings["TEST_CREDENTIAL"] = "another-test-credential"
    _, request, _ = session_candidate(host, host_world.provider["session"], LiveOptions())
    assert host.plan.route_fp != first
    assert request.physical_context.window_observation_fp == host.plan.route_fp
    assert request.physical_context.capacity_total_tokens == host.model.facts.conversation_window_tokens
