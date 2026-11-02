"""Refused fits remain inspectable; the fresh Max/Gemini probe fits offline."""
import asyncio
import json
import socket
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros.contracts.live_conversation import LiveOptions
from tests.test_live_conversation import ROOT, VoiceWire, host_world as host_world


def host_events(host):
    host.wire.flush()
    return [json.loads(row['raw']) for line in host.wire_path.read_text(encoding="utf-8").splitlines()
            if (row := json.loads(line))['direction'] == 'host']


def test_refused_fit_is_durable_before_error_and_fitting_candidate_stays_quiet(host_world, monkeypatch):
    from ouroboros.context_fit import ContextFitPlan
    from ouroboros.contracts.realtime import RealtimeSetupError
    from ouroboros.observability import read_call_payload
    from ouroboros.realtime_setup import session_candidate
    from ouroboros.usage_accounting import read_usage_records

    host = host_world.host
    from ouroboros.context_budget import bound_host_row
    monkeypatch.setattr('ouroboros.context_measurement.room_fact',
        lambda *a, **kw: bound_host_row('ouroboros_context_facts', 'Fixture room facts.'))
    monkeypatch.setattr(ContextFitPlan, 'reproject_for_route', lambda plan, **kwargs: plan)
    _, request, _ = session_candidate(host, host_world.provider['session'], LiveOptions())
    raw = request.prompt_tokens_bounded_estimate
    host.model = replace(host.model, facts=replace(host.model.facts, conversation_window_tokens=raw + 9999))
    async def refused(session):
        pytest.fail('overflowing candidate reached the provider')
    with pytest.raises(RealtimeSetupError, match='context_capacity'):
        asyncio.run(host.session(refused, LiveOptions()))
    host.wire.flush()  # the journal's writer creates the file
    events = host_events(host) if host.wire_path.exists() else []
    refusals = [event for event in events if event['type'] == 'call.fit_refused']
    assert len(refusals) == 1, 'context_capacity lost the fit that it refused'
    event, = refusals
    assert event['reason'] == 'context_capacity' and event['units'] == 'provider'
    assert event['tool_count'] == len(host.schemas)
    measured = event['fit']['measurement']
    assert measured['rendered_mode'] == 'nano'
    assert measured['capacity_total_tokens'] == raw + 9999
    assert measured['response_reserve_tokens'] == 10000
    assert measured['estimated_input_tokens'] == measured['raw_input_tokens']
    free = measured['capacity_total_tokens'] - measured['raw_input_tokens'] - 10000
    assert free < 0 and event['refusal']['free_tokens'] == free and event['deficit_tokens'] == -free + 1
    assert host.trace['fit_refusal'] == event
    manifest, payload, _ = read_call_payload(host.ctx.drive_root, task_id=host.ctx.task_id,
                                            call_id=event['manifest_ref']['call_id'])
    for key in ('fit', 'route_facts', 'tool_count', 'deficit_tokens', 'refusal', 'units'):
        assert manifest[key] == event[key]
    assert payload == host_world.sessions[-1].wire
    assert not any(session.opened for session in host_world.sessions)
    assert read_usage_records(host.ctx.drive_root) == []

    # The new passive row describes this candidate's window, so its own bytes
    # vary with that window and must be included in the fitting control too.
    host.model = replace(host.model, facts=replace(host.model.facts,
        conversation_window_tokens=measured['raw_input_tokens'] + 10000 + 500))
    async def allowed(session):
        assert session.opened
    asyncio.run(host.session(allowed, LiveOptions()))
    assert [event for event in host_events(host) if event['type'] == 'call.fit_refused'] == refusals
    assert host_events(host)[-1]['type'] == 'call.prepared'


def test_fresh_default_max_connects_with_real_gemini_envelope_and_allowance(tmp_path, monkeypatch, provider_catalog_offline):
    from ouroboros import config, context, extension_registry_state as registry
    from ouroboros.agent import Env
    from ouroboros.call_peers import LiveCallHub
    from ouroboros.call_protocol import CallSpec
    from ouroboros.contracts.live_conversation import LiveConfig
    from ouroboros.live_conversation import ConversationHost, LiveConversationTurn
    from ouroboros.memory import Memory
    from ouroboros.model_wait import task_model_wait_scope
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.tool_context import ToolContext
    from skills.gemini_live.plugin import CATALOG, ROUTE, GeminiSession
    from skills.live_conversation.driver import CALL_INSTRUCTIONS, call_tools, session_options
    from tests.test_budget_fence_liveness import _publish

    def no_network(*args, **kwargs):
        pytest.fail('fresh-install fit test attempted a network connection')
    monkeypatch.setattr(socket.socket, 'connect', no_network)
    root = tmp_path / 'fresh'
    root.mkdir()
    monkeypatch.setattr(config, 'DATA_DIR', root)
    monkeypatch.setenv('OUROBOROS_DATA_DIR', str(root))
    # No owner context-mode setting: use the application's actual default.
    monkeypatch.setattr(config, 'runtime_setting', lambda key, default=None: default)
    assert config.get_context_mode() == 'max'
    env, memory = Env(repo_dir=ROOT, drive_root=root), Memory(repo_dir=ROOT, drive_root=root)
    memory.ensure_files()
    live_config = LiveConfig(enabled=True, provider='fixture:gemini')
    feature = dict(skill='fixture', key='fixture:call', extension_generation='generation',
                   call_tools=call_tools(root / 'feature'))
    wire, sessions = VoiceWire('gemini', None), []
    def factory(selected, options, *, observe):
        session = GeminiSession(selected, options, connect=wire.connect, observe=observe)
        sessions.append(session)
        return session
    provider = dict(skill='fixture', extension_generation='generation', catalog=CATALOG, route=ROUTE, session=factory)
    bundle = registry._ExtensionRegistrations(generation_digest='generation', realtime_providers=['fixture:gemini'],
        api_instances=[SimpleNamespace(get_settings=lambda keys: {'GEMINI_API_KEY': 'test-credential'})])
    monkeypatch.setitem(registry._extensions, 'fixture', bundle)
    monkeypatch.setitem(registry._realtime_providers, 'fixture:gemini', provider)
    spec = CallSpec('fresh-gemini', 1, live_config.provider, '', '', client_message_id='call:fresh-gemini:open')
    turn = LiveConversationTurn(feature, live_config, spec, publish=lambda _: None)
    task = dict(id='fresh-gemini-task', type='task', text=turn.opening, chat_id=1, _is_direct_chat=True,
        model=turn.model, use_local_model=False,
        metadata={'execution_transport': turn.request, 'host_fact': {'kind': 'call_open'}},
        origin_message_ref={'chat_id': 1, 'client_message_id': spec.client_message_id})
    ctx = ToolContext(repo_dir=ROOT, drive_root=root, branch_dev='test', task_id=task['id'],
                      current_chat_id=1, is_direct_chat=True, task_metadata=task['metadata'], task_contract={})
    tools = ToolRegistry(ROOT, root)
    tools.set_context(ctx)
    write_task_result(root, task['id'], STATUS_RUNNING, chat_id=1)
    _publish(root)
    messages, _ = context.build_llm_messages(env, memory, task, ctx=ctx,
        tool_schemas=tools.schemas(), route_resolver=turn.resolve_route)
    assert ctx.context_fit_plan.preferred_mode == 'max'
    assert ctx.context_fit_plan.initial_mode == 'nano'  # ordinary physical fallback before sending
    hub = LiveCallHub(turn.spec)
    with task_model_wait_scope(task=task, drive_root=root, event_queue=None, worker_slot_held=False) as owner:
        ctx.model_wait_context, owner.tool_context = owner, ctx
        host = ConversationHost(turn, ctx, messages, tools, task, '', hub)
        async def consume(session):
            assert (await session.receive())[0].kind == 'ready'
            assert session.output_bound_tokens == session.selected.output_allowance_tokens > 8192
            assert session.events[0]['setup']['generationConfig']['maxOutputTokens'] == session.output_bound_tokens
            assert CALL_INSTRUCTIONS in session.options.instructions
            assert {'end_call', 'set_call_languages'} <= {row['function']['name'] for row in host.schemas}
        async def run():
            try:
                await host.session(consume, session_options(host))
            finally:
                await host.finish()
        try:
            asyncio.run(run())
        finally:
            hub.close('fixture_done')
        assert wire.closed and len(wire.sent) == 1
        prepared, = [event for event in host_events(host) if event['type'] == 'call.prepared']
        assert not any(event['type'] == 'call.fit_refused' for event in host_events(host))
        assert prepared['route_facts']['conversation_window_tokens'] == 92000
        assert prepared['route_facts']['working_reserve_tokens'] == 10000
        assert prepared['units'] == 'provider'
        measured = prepared['fit']['measurement']
        assert measured['response_reserve_tokens'] == sessions[-1].output_bound_tokens
        assert measured['estimated_input_tokens'] + measured['response_reserve_tokens'] < 92000
        assert measured['profile'] == 'task_local_nano'
