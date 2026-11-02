"""One measured source view for text and every fresh call candidate; no providers."""
import asyncio
import copy
import json
from dataclasses import replace

import pytest

from ouroboros.contracts.live_conversation import LiveOptions
from tests.test_live_conversation import host_world as host_world, packages as packages


def test_wire_projection_is_recursive_and_leaves_strings_and_canonical_sources_exact():
    from ouroboros.llm_messages import project_wire_messages
    rows = [{'role': 'user', '_host_context_kind': 'owner_words', 'content': [
        {'type': 'text', 'text': '{"_tool_result_record":"owner literal"}', '_context_capsule': {'exact': 'source'},
         'cache_control': {'type': 'ephemeral'}, '_review_image_id': 'host-image',
         'acceptance_observation': {'status': 'seen'}, '_original_image_url': 'host-original'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,YQ=='}, '_source_path': '/exact.png'}]}]
    before = copy.deepcopy(rows)
    assert project_wire_messages(rows) == [{'role': 'user', 'content': [
        {'type': 'text', 'text': '{"_tool_result_record":"owner literal"}'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,YQ=='}}]}]
    assert rows == before


@pytest.mark.parametrize('engine', ['gemini', 'grok', 'openai'])
def test_real_layouts_never_serialize_host_records(host_world, packages, engine):
    from ouroboros.realtime_setup import measure_session_wire, loaded_provider
    from tests.test_context_source_view import _tool
    host = host_world.host
    rows = [*host.messages, *_tool('a', invocation='invocation-a')]
    name = {'gemini': 'gemini_live', 'grok': 'grok_realtime', 'openai': 'openai_realtime'}[engine]
    factory = loaded_provider(name + ':' + engine)['session']
    session, _, _, _ = measure_session_wire(host, host.plan, rows, [], factory, LiveOptions())
    wire = json.dumps(session.wire)
    for key in ('_tool_result_record', '_host_context_kind', '_host_text_sha256', '_stable_prefix_blocks', 'cache_control'):
        assert key not in wire
    assert 'invocation-a' not in wire
    assert '_tool_result_record' in rows[-1]


def test_wire_measurement_recomputes_all_deficits_together(host_world):
    from ouroboros.realtime_setup import measure_session_wire
    from ouroboros.context_measurement import reclaim_low_water_margin
    host = host_world.host
    host.plan = replace(host.plan, window_tokens=100000, output_reserve_tokens=10000)
    rows = [*host.messages, {'role': 'assistant', 'content': 'word ' * 70000}]
    _, _, evidence, _ = measure_session_wire(host, host.plan, rows, host.schemas,
                                            host_world.provider['session'], LiveOptions())
    fit = evidence['fit']; m = fit['measurement']
    total = m['estimated_input_tokens'] + m['response_reserve_tokens']
    target = max(0, total - m['target_total_tokens'])
    capacity = max(0, total - m['capacity_total_tokens'])
    margin = reclaim_low_water_margin(m['target_total_tokens'], m['capacity_total_tokens']) if max(target, capacity) else 0
    assert (m['target_deficit_tokens'], m['capacity_deficit_tokens'], m['low_water_margin_tokens'],
            m['reclaim_goal_tokens'], fit['action']) == (
                target, capacity, margin, max(target, capacity) + margin, 'send_target_miss' if target else 'send')


def test_caption_types_protect_words_but_not_heard_observations(host_world):
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
    from ouroboros.context_compaction import context_units, owner_protected_unit_ids
    from ouroboros.context_renewal import renewal_seed
    host = host_world.host
    async def run():
        await host.caption(speaker='owner', text='Exact owner words.', item_id='owner')
        await host.caption(speaker='ouroboros', text='Exact spoken reply.', item_id='voice', completed=True)
        await host.caption(speaker='owner', text='Participant observation.', item_id='meeting', heard=True)
        await host.caption(speaker='ouroboros', text='Unconfirmed speech.', item_id='unheard')
    asyncio.run(run())
    rows = host.messages[-4:]
    assert [r.get(HOST_CONTEXT_KIND_KEY) for r in rows] == ['owner_words', 'owner_dialogue', None, None]
    units = context_units(host.messages, scope='dialogue')
    protected = owner_protected_unit_ids(host.messages, units)
    assert all(u.unit_id in protected for u in units if host.messages[u.start] in rows[:2])
    seed, _ = renewal_seed(host.messages)
    assert all(r in seed for r in rows[:2])
    from ouroboros import context_compaction as cc
    from tests.test_context_reclaim_materializer import _request
    request = replace(_request(host.messages, 0), working_note='The observed conversation remains exact.',
        expected_view_revision=cc.context_reclaim_transcript_sha256(host.messages), keep_unit_ids=())
    folded, receipt = cc.compact_tool_history_llm(host.messages, request=request,
        observed_messages=host.messages, tool_schemas=[], drive_root=host.ctx.drive_root,
        task_id=host.ctx.task_id, fit_candidate=lambda *_: {'accepted': True})
    assert receipt.status == 'applied'
    assert all(row in folded for row in rows[:2])


def test_address_facts_keep_invocation_and_reader_but_not_trace_blob():
    from ouroboros.context_compaction import context_units
    from ouroboros.context_source_view import _unit_facts
    from tests.test_context_source_view import _tool
    rows = _tool('a', invocation='invocation-a')
    facts = _unit_facts(rows, context_units(rows, scope='tool')[0])
    assert 'invocation invocation-a' in facts and 'source/invocation-a.txt' in facts
    assert 'trace/invocation-a.json' not in facts
    assert 'trace_ref' in rows[-1]['_tool_result_record']


@pytest.mark.parametrize('engine,window', [('gemini', 92000), ('openai', 133554), ('grok', 80000)])
def test_fresh_candidates_outgrow_window_then_continue_by_exact_physical_cut(host_world, packages, engine, window):
    from ouroboros.realtime_setup import loaded_provider, session_candidate
    from ouroboros.context_budget import bound_host_row
    from ouroboros.artifacts import read_actor_source_bytes
    host = host_world.host
    name = {'gemini': 'gemini_live', 'openai': 'openai_realtime', 'grok': 'grok_realtime'}[engine]
    provider = loaded_provider(name + ':' + engine)
    host.model = provider['catalog'].models[0]
    host.turn.native_model = host.model.id
    host._session_opened = True
    original = [bound_host_row(kind, f'{i} exact delivered words. ' * 4000) for i in range(8)
                for kind in ('owner_words', 'owner_dialogue')]
    newest = bound_host_row('owner_words', 'Newest turn stays verbatim.')
    host.messages.extend([*original, newest, {'role': 'assistant', 'content': 'Current tail.'}])
    _, request, evidence = session_candidate(host, provider['session'], LiveOptions())
    assert request.prompt_tokens_estimate + request.working_reserve_tokens <= window, evidence
    assert any(r['rung'] == 'physical_cut' for r in evidence['recovery'])
    record = next(row for row in host.messages if row.get('_host_context_kind') == 'context_physical_cut')
    source = json.loads(record['content'].split('checkpoint JSON array: ')[1].rstrip('.'))
    checkpoint = json.loads(read_actor_source_bytes(host.ctx.drive_root, host.ctx.task_id, source))
    assert all(row in checkpoint for row in original)
    assert newest in host.messages
    tail = [row for row in host.messages if row.get('_host_context_kind') not in {'tool_manifest', 'ouroboros_context_facts'}]
    assert tail[-1]['content'] == 'Current tail.'
    assert sum(row.get('_host_context_kind') == 'tool_manifest' for row in host.messages) == 1  # passive fact follows
    # The untouched newer suffix remains exactly the delivered bytes, not a host retelling.
    remaining = [row for row in host.messages if row in original]
    assert remaining == original[len(original)-len(remaining):]


def test_speculative_capacity_failure_is_not_ready_and_retries_only_changed_view(host_world):
    from skills.live_conversation.driver import _Call
    from ouroboros.contracts.realtime import RealtimeSetupError
    host = host_world.host
    async def run():
        call = _Call(host)
        call.session = type('Serving', (), {'silent_context_updates': True})()
        attempts = []
        async def fail(options):
            attempts.append(options)
            raise RealtimeSetupError('context_capacity', 'irreducible')
        host.prepare_session = fail
        call.request_renewal()
        call.tool_refit.set()
        call.input_open = True
        boundary = asyncio.create_task(call.refit_boundary())
        await asyncio.sleep(0)
        assert not call.preparation_ready()
        await call.check_preparation()
        call.input_open = False
        await asyncio.sleep(.02)
        assert not boundary.done(), 'a cleared failed refit must not end the serving consumer'
        boundary.cancel()
        await asyncio.gather(boundary, return_exceptions=True)
        assert not call.renewing and call.session is not None
        call.request_renewal()
        assert len(attempts) == 1 and call.preparing is None
        host._context_changed()
        call.request_renewal()
        await asyncio.sleep(0)
        assert len(attempts) == 2
        await call.check_preparation()
    asyncio.run(run())


@pytest.mark.parametrize('exposed,rung', [(True, 'bodies'), (False, 'unseen_bodies')])
def test_transcript_remedies_precede_memory_when_exposed_and_follow_it_when_unseen(host_world, exposed, rung):
    from ouroboros.realtime_setup import session_candidate
    from tests.test_context_source_view import _tool, _observation
    host = host_world.host
    host._session_opened = True
    host.model = replace(host.model, facts=replace(host.model.facts, conversation_window_tokens=40000))
    host.messages.extend(_tool('large', invocation='exact-large', text='body ' * 40000))
    before = copy.deepcopy(host.plan.projection('nano').memory_facts)
    if exposed:
        host.ctx._last_context_observation = _observation(host.messages)
    _, _, evidence = session_candidate(host, host_world.provider['session'], LiveOptions())
    assert rung in [r['rung'] for r in evidence['recovery']], evidence
    assert not any(r.get('role') == 'tool' for r in host.messages)
    if exposed:
        assert host.plan.projection('nano').memory_facts == before
        assert [r['rung'] for r in evidence['recovery']] == ['bodies']


def test_exposure_records_only_delivered_rows_on_a_usable_adopted_connection(host_world):
    from types import SimpleNamespace
    from ouroboros.live_context_view import record_live_exposure
    from ouroboros.context_source_view import _exposed_units
    from ouroboros.context_compaction import context_units
    from ouroboros.realtime_setup import rows_for_engine
    from tests.test_context_source_view import _tool
    host = host_world.host
    held = _tool('held', invocation='held-effect')
    queued = _tool('queued', invocation='queued-effect')
    host.messages.extend([*held, *queued])
    view = copy.copy(host)
    view._connection_owner = host
    view._context_session = SimpleNamespace(response_context_rows={'usable': rows_for_engine(held, vision=True)})
    host._connection_view = None
    record_live_exposure(view, 'usable')
    assert not getattr(host.ctx, '_last_context_observation', None), 'speculation is not exposure'
    host._connection_view = view
    record_live_exposure(view, 'usable')
    units = context_units(host.messages, scope='tool')
    exposed = _exposed_units(host.messages, units, host.ctx._last_context_observation)
    assert exposed == {unit.unit_id for unit in units if any(
        call['id'] == 'held' for call in host.messages[unit.start].get('tool_calls', ()))}
    view._context_session.response_context_rows['next'] = rows_for_engine([*held, *queued], vision=True)
    record_live_exposure(view, 'next')
    assert _exposed_units(host.messages, units, host.ctx._last_context_observation) == {u.unit_id for u in units}


def test_seed_is_shared_when_individual_addresses_cannot_save_space(host_world, tmp_path):
    from ouroboros.context_renewal import _is_note
    from ouroboros.realtime_setup import session_candidate, measure_session_wire
    from tests.test_context_renewal import _noted
    from tests.test_context_attention_mind import _tool, _owner
    host = host_world.host
    note = next(row for row in _noted(tmp_path) if _is_note(row))
    end = [note, _owner('Newest turn.')]
    _, request, _, _ = measure_session_wire(host, host.plan, [*host.messages, *end], host.schemas,
                                           host_world.provider['session'], LiveOptions())
    host.model = replace(host.model, facts=replace(host.model.facts,
        conversation_window_tokens=request.prompt_tokens_estimate + 11000))
    host._session_opened = True
    for i in range(100):
        host.messages.extend(_tool(str(i), 'ok'))
    host.messages.extend(end)
    _, _, evidence = session_candidate(host, host_world.provider['session'], LiveOptions())
    assert 'seed' in [r['rung'] for r in evidence['recovery']], evidence
    assert note in host.messages and end[-1] in host.messages
    assert not any(row.get('role') == 'tool' for row in host.messages)


def test_passive_room_fact_is_sent_once_and_counts_its_own_bytes(host_world):
    from ouroboros.realtime_setup import session_candidate, measure_session_wire
    host = host_world.host
    host._session_opened = True
    before = sum(row.get('_host_context_kind') == 'ouroboros_context_facts' for row in host.messages)
    session, request, evidence = session_candidate(host, host_world.provider['session'], LiveOptions())
    assert sum(row.get('_host_context_kind') == 'ouroboros_context_facts' for row in host.messages) == before + 1
    assert 'Fresh view before this facts row:' in host.messages[-1]['content']
    assert 'Largest residents:' in host.messages[-1]['content']
    _, measured, _, _ = measure_session_wire(host, host.plan, host.messages, host.schemas,
        host_world.provider['session'], LiveOptions(), _room_fact=False)
    assert measured.prompt_tokens_estimate == request.prompt_tokens_estimate
    assert json.loads(session.selected.messages_json)[-1]['content'] == host.messages[-1]['content']
    assert not evidence['recovery']


def test_next_fresh_fit_retires_the_old_room_fact_before_memory(host_world):
    from ouroboros.context_budget import bound_host_row
    from ouroboros.realtime_setup import session_candidate
    host = host_world.host
    host._session_opened = True
    host.model = replace(host.model, facts=replace(host.model.facts, conversation_window_tokens=40000))
    old = bound_host_row('ouroboros_context_facts', 'Superseded exact room facts. ' * 10000)
    host.messages.append(old)
    _, _, evidence = session_candidate(host, host_world.provider['session'], LiveOptions())
    assert not evidence['recovery']  # a fresh prefix replaces the passive snapshot before pressure passes
    import re
    from ouroboros.artifacts import task_artifact_dir_path
    path = json.loads(re.search(r"read_file\(root='artifact_store', path=(.+)\)", host.messages[-1]['content'])[1])
    assert old in json.loads((task_artifact_dir_path(host.ctx.drive_root, host.ctx.task_id) / path).read_text())
    assert old not in host.messages
    assert sum(row.get('_host_context_kind') == 'ouroboros_context_facts' for row in host.messages) == 1


def test_passive_fact_needs_no_new_attribute_on_an_external_provider_session(host_world):
    from ouroboros.realtime_setup import session_candidate
    class SlottedSession:
        __slots__ = ('native',)
        def __init__(self, native):
            self.native = native
        def __getattr__(self, name):
            return getattr(self.native, name)
    def factory(*args, **kwargs):
        return SlottedSession(host_world.provider['session'](*args, **kwargs))
    host = host_world.host
    host._session_opened = True
    session, request, _ = session_candidate(host, factory, LiveOptions())
    assert json.loads(session.selected.messages_json)[-1]['content'] == host.messages[-1]['content']
    assert request.prompt_tokens_estimate > 0


@pytest.mark.parametrize('refused', [False, True])
def test_published_candidate_room_reader_always_resolves(host_world, refused):
    import re
    from ouroboros.evidence_packs import handle_bytes
    from ouroboros.observability import read_call_payload
    from ouroboros.realtime_setup import session_candidate
    from ouroboros.contracts.realtime import RealtimeSetupError
    host = host_world.host
    if refused:
        host.model = replace(host.model, facts=replace(host.model.facts, conversation_window_tokens=10001))
        with pytest.raises(RealtimeSetupError, match='context_capacity'):
            session_candidate(host, host_world.provider['session'], LiveOptions())
        event = host.trace['fit_refusal']
        _, wire, _ = read_call_payload(host.ctx.drive_root, task_id=host.ctx.task_id,
                                      call_id=event['manifest_ref']['call_id'])
    else:
        session, _, _ = session_candidate(host, host_world.provider['session'], LiveOptions())
        wire = session.wire
    text = json.dumps(wire)
    # Decode the actual content string, not JSON's quote escapes.
    def strings(value):
        if isinstance(value, str): yield value
        elif isinstance(value, dict):
            for item in value.values(): yield from strings(item)
        elif isinstance(value, list):
            for item in value: yield from strings(item)
    paths = [path for value in strings(wire) for path in re.findall(r"path=(\"[^\"]*room-[^\"]+\")", value)]
    if refused:
        from ouroboros.artifacts import task_artifact_dir_path
        assert not paths and not list(task_artifact_dir_path(host.ctx.drive_root, host.ctx.task_id).rglob('room-*.json'))
        return
    assert paths, text[:100]
    for path in paths:
        assert json.loads(handle_bytes(host.ctx.drive_root, host.ctx.task_id, json.loads(path)))


@pytest.mark.parametrize('all_loaded', [False, True])
def test_fresh_fit_restates_manifest_after_exact_old_row_cut(host_world, monkeypatch, all_loaded):
    from ouroboros.context_budget import bound_host_row
    from ouroboros.context_renewal import physical_window_cut
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.context_fit import estimate_context_prompt_tokens
    from ouroboros.realtime_setup import measure_session_wire
    host = host_world.host
    old = bound_host_row('tool_manifest', 'Old exact manifest. ' * 2000)
    rows = [*host.messages[:2], old, bound_host_row('owner_words', 'Latest question')]
    kept, receipt = physical_window_cut(rows, drive_root=host.ctx.drive_root, task_id=host.ctx.task_id,
        measure=estimate_context_prompt_tokens, usable=estimate_context_prompt_tokens(rows)-1000)
    assert receipt and old not in kept
    assert old in json.loads(read_actor_source_bytes(host.ctx.drive_root, host.ctx.task_id, receipt['source']))
    schemas = host.tools.schemas() if all_loaded else host.schemas
    captured = []
    measure_session_wire(host, host.plan, kept, schemas, host_world.provider['session'], LiveOptions(), canonical_rows=captured)
    notices = [row for row in captured if row.get('_host_context_kind')=='tool_manifest']
    assert bool(notices) is not all_loaded
