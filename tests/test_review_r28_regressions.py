"""R28 integration invariants, synthetic memory only; no providers or native state."""
import asyncio
from dataclasses import replace
import json

import pytest

from ouroboros import memory_floor as mf, memory_view as mv
from tests import _memory_view_synthetic as syn
from tests.test_live_conversation import host_world as host_world, packages as packages


@pytest.mark.parametrize('new_kind', ['page', 'part'])
@pytest.mark.parametrize('saving', [0, 2_000])
def test_page_giveback_keeps_covered_age_order(new_kind, saving, record_property):
    old = {**syn.pages(1, 1_000)[0], 'id': 'old-page', 'floor_order': [0, 1]}
    recent = {**syn.pages(1, 20_000)[0], 'id': 'recent-' + new_kind,
              'kind': new_kind, 'floor_order': [1, 4]}
    snapshot = syn.snapshot(story=[recent, old])  # publication order differs from covered age
    full = mf.view_tokens(mv.render_story(snapshot))
    level = mf.fit_memory_view(snapshot, {'margin': full - saving, 'physical': 10**9, 'budget': None})
    rendered = mv.render_story(snapshot, level)
    old_whole, recent_whole = old['text'] in rendered, recent['text'] in rendered
    record_property('page_giveback', json.dumps(dict(new_kind=new_kind, saving=saving,
        full_tokens=full, selected_tokens=mf.view_tokens(rendered), addressed=level.addressed,
        old_whole=old_whole, recent_whole=recent_whole)))
    if not saving:
        assert level == mv.FULL_VIEW and old_whole and recent_whole
    else:
        assert not recent_whole  # the newer body cannot fit this frame
        assert not old_whole, 'give-back must not restore an older page over a newer shortened retelling'
    assert snapshot.story == (recent, old)  # selected forms never rewrite exact sources


@pytest.mark.parametrize('reader_fails', [False, True])
@pytest.mark.parametrize('provider_id,model_index', [
    ('gemini_live:gemini', 0), ('gemini_live:gemini', 1),
    ('openai_realtime:openai', 0), ('grok_realtime:grok', 0),
])
def test_call_carries_target_waiting_facts_from_real_sources(host_world, packages, provider_id,
                                                            model_index, record_property, monkeypatch, reader_fails):
    from ouroboros import context
    from ouroboros.memory import Memory
    from ouroboros.late_review_marks import publish_late_review
    from ouroboros.tools.chronicle import _chronicle_write
    from ouroboros.context_view_facts import memory_facts
    from ouroboros.memory_waiting import text as waiting_text
    from ouroboros.realtime_setup import loaded_provider, session_candidate
    from ouroboros.contracts.live_conversation import LiveOptions
    from tests.test_late_review_marks import _record

    host = host_world.host
    root = host.ctx.drive_root
    mark = publish_late_review(root, _record(root, 'late-reviewed-work', 1, 'Exact late review.')).mark_id
    note = json.loads(_chronicle_write(host.ctx, kind='note', text='One unsettled thought.'))['node_id']
    assert mark and note
    # Capture through the real initial-context builder after the records exist.
    rows, _ = context.build_llm_messages(host_world.env, Memory(root, host.ctx.repo_dir), host_world.task,
        ctx=host.ctx, tool_schemas=host.tools.schemas(), route_resolver=host.turn.resolve_route)
    host.plan = host.ctx.context_fit_plan
    host.messages[:] = rows
    host._context_changed()
    snapshot = mv.snapshot_from_json(host.plan.core.memory_view_json)
    expected = waiting_text(snapshot.waiting)
    assert snapshot.waiting['late_reviews']['count'] == snapshot.waiting['notes']['count'] == 1
    assert snapshot.waiting['projects']['count'] > 0
    text_facts = memory_facts(host.plan, host.plan.initial_mode, host.messages)
    assert expected and expected in text_facts  # the target's text producer still exposes all three classes

    if reader_fails:
        from ouroboros import artifacts
        store = artifacts.store_actor_source_bytes
        def refuse_reader(*args, **kwargs):
            if kwargs.get('category') == 'context_checkpoints':
                raise OSError('synthetic reader publication failure')
            return store(*args, **kwargs)
        monkeypatch.setattr(artifacts, 'store_actor_source_bytes', refuse_reader)
    provider = loaded_provider(provider_id)
    host.model = provider['catalog'].models[model_index]
    host.turn.native_model = host.model.id
    session, request, evidence = session_candidate(host, provider['session'], LiveOptions())
    delivered = json.loads(session.selected.messages_json)
    def text_content(row):
        body = row.get('content', '')
        return '\n'.join(str(block.get('text', '')) for block in body if isinstance(block, dict)) if isinstance(body, list) else str(body)
    delivered_text = '\n'.join(map(text_content, delivered))
    record_property('waiting_delivery', json.dumps(dict(provider=provider_id, model=host.model.id,
        captured=snapshot.waiting, text_producer=True, call_contains_waiting=expected in delivered_text,
        input_tokens=request.prompt_tokens_estimate, reserve=request.working_reserve_tokens,
        capacity=evidence['fit']['measurement']['capacity_total_tokens'])))
    assert request.prompt_tokens_estimate + request.working_reserve_tokens <= host.model.facts.conversation_window_tokens
    assert expected in delivered_text, 'the call must deliver its already captured waiting census'
    facts = [row for row in delivered if '[CONTEXT_FACTS] Fresh view' in text_content(row)]
    assert len(facts) == 1
    assert ('Exact view before this refresh: read_file' in text_content(facts[0])) is not reader_fails
    if not reader_fails:
        assert text_content(facts[0]).endswith(').'), 'the independent reader stays last'
    # Sealed wire and the admission request include the added facts.
    assert expected in json.dumps(session.wire, ensure_ascii=False)
    from ouroboros.realtime_setup import measure_session_wire
    stripped = [row for row in delivered if row is not facts[0]]
    _, bare, _, _ = measure_session_wire(host, host.plan, stripped, host.schemas,
        provider['session'], LiveOptions(), _room_fact=False)
    assert request.prompt_tokens_estimate > bare.prompt_tokens_estimate


def test_giveback_keeps_a_newer_small_page_when_its_whole_body_fits(record_property):
    old = {**syn.pages(1, 1)[0], 'id': 'old-long-first-sentence', 'floor_order': [0, 1],
           'text': 'First thought ' * 300 + '. ' + 'An old detailed tail. ' * 1500}
    recent = {**syn.pages(1, 1_000)[0], 'id': 'new-small', 'floor_order': [1, 2]}
    snapshot = syn.snapshot(story=[recent, old])
    level = mf.fit_memory_view(snapshot, {'margin': 700, 'physical': 10**9, 'budget': None})
    rendered = mv.render_story(snapshot, level)
    record_property('newest_control', json.dumps(dict(addressed=level.addressed,
        selected_tokens=mf.view_tokens(rendered))))
    assert old['text'] not in rendered and recent['text'] in rendered
    assert mf.view_tokens(rendered) <= 700


@pytest.mark.parametrize('vision', [True, False])
def test_live_image_growth_and_blocked_release_use_delivered_basis(host_world, vision):
    from ouroboros.context_fit import estimate_context_prompt_tokens
    from ouroboros.contracts.live_conversation import ContextUpdate, LiveOptions
    from ouroboros.contracts.realtime import RealtimeSetupError
    from ouroboros.live_context_view import serving_delivery_room
    from ouroboros.realtime_setup import rows_for_engine
    from skills.live_conversation.driver import _Call
    from tests.test_live_image_retention import look
    host = host_world.host
    host.model = replace(host.model, vision=vision)
    async def consume(session):
        # An image already held at the receipt also checks the baseline side.
        await host.observe_context(look(host, 0))
        held = host._renewal_window.total_equivalent - host._renewal_window.reserve - 1100
        host.response_done('receipt', usage={'audio_input_tokens': held}, raw_usage={}, status='completed')
        extra = [{'role': 'system', 'content': host._session_instructions}] if host._session_instructions else []
        receipt = estimate_context_prompt_tokens(
            [*rows_for_engine(host.messages, vision=vision, schemas=host.schemas), *extra], host.schemas)
        if not vision:
            assert host._serving_history_at_receipt == receipt
        row = look(host, 1)
        await host.observe_context(row)
        tiny = [{'role': 'user', 'content': 'tiny room note'}]
        current = rows_for_engine([*host.messages, *tiny, *extra], vision=vision, schemas=host.schemas)
        pending = max(estimate_context_prompt_tokens(current, host.schemas) - receipt,
                      estimate_context_prompt_tokens(tiny))
        room = serving_delivery_room(host, tiny, host.schemas)
        if not vision:
            assert room == 1100 - pending
        call = _Call(host); call.session = session; call.connected.set()
        sent = []
        async def send(command): sent.append(command)
        session.send = send
        call.pending_context_rows = tiny
        call.failed_preparation_revision = host._history_version
        if vision:
            assert room < 0
            with pytest.raises(RealtimeSetupError, match='context_capacity'):
                await call.release_serving()
            assert not sent
        else:
            assert room > 0
            await call.release_serving()
            assert any(isinstance(c, ContextUpdate) for c in sent)
    asyncio.run(host.session(consume, LiveOptions()))


@pytest.mark.parametrize('adopted', [False, True])
def test_six_looks_restore_text_reader_after_call_end(host_world, adopted):
    import copy
    from ouroboros.contracts.live_conversation import LiveOptions
    from ouroboros.llm_messages import project_wire_messages
    from ouroboros.loop_messages import _evict_stale_image_blocks
    from ouroboros.realtime_setup import rows_for_engine
    from tests.test_live_image_retention import look
    host = host_world.host
    host.plan = replace(host.plan, preferred_mode='nano')
    async def run():
        for i in range(6):
            await host.observe_context(look(host, i))
        if adopted:
            async def consume(session):
                pass
            await host.session(consume, LiveOptions())
        await host.finish()
    asyncio.run(run())
    def placeholders(rows):
        return [p['text'] for row in rows if isinstance(row.get('content'), list)
                for p in row['content'] if isinstance(p, dict) and 'image evicted' in str(p.get('text', ''))]
    call_floor = placeholders(project_wire_messages(host.messages))
    assert len(call_floor) == 4 and all('view_image path=' in text for text in call_floor)
    assert '_evicted_image' not in json.dumps(project_wire_messages(host.messages))
    from ouroboros.working_checkpoint import checkpoint_path
    saved = json.loads(checkpoint_path(host.ctx.drive_root, host.ctx.task_id, 1).read_text())
    assert placeholders(project_wire_messages(saved['messages'])) == call_floor
    # A later call still renders against its own catalog, independent of the text checkpoint.
    assert all('view_image' not in text for text in placeholders(
        rows_for_engine(host.messages, vision=True, schemas=[])))
    text = [copy.deepcopy(look(host, 10 + i)) for i in range(12)]
    _evict_stale_image_blocks(text, owner_mode='max')
    assert len(placeholders(text)) == 7
    assert all('view_image path=' in t for t in placeholders(text))


@pytest.mark.parametrize('placeholder', [False, True])
def test_text_estimator_excludes_block_host_metadata(host_world, placeholder):
    import copy
    from ouroboros.context_images import image_placeholder
    from ouroboros.context_fit import estimate_context_prompt_tokens
    from ouroboros.llm_messages import project_wire_messages
    block = image_placeholder({'_caption': 'Exact image caption', '_source_path': '/exact/frame.jpg'}) if placeholder else {
        'type': 'text', 'text': 'An ordinary text block.'}
    rows = [{'role': 'user', 'content': [block]}]
    before = copy.deepcopy(rows)
    assert estimate_context_prompt_tokens(rows) == estimate_context_prompt_tokens(project_wire_messages(rows))
    assert rows == before
    # Same-named keys in opaque tool arguments remain meaningful content.
    arguments = [{'role': 'assistant', 'tool_calls': [{'id': 'one', 'type': 'function',
        'function': {'name': 'example', 'arguments': json.dumps({'_caption': 'x' * 4000})}}]}]
    assert estimate_context_prompt_tokens(arguments) > 1000


@pytest.mark.parametrize('failed_fit_same_revision', [False, True])
def test_native_resume_without_handle_uses_fresh_fit(host_world, monkeypatch, failed_fit_same_revision):
    """A tool boundary outlives its native handle, even when request_renewal returns early."""
    import threading
    from tests._live_stack import stack
    from ouroboros.tools.tool_result import ToolResult
    entered, release = threading.Event(), threading.Event()
    effects = []
    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            execute, batch = live.host.tools.execute_result, live.host.tool_batch
            def blocked(name, args):
                if name != 'list_available_tools':
                    return execute(name, args)
                effects.append(name); entered.set()
                assert release.wait(30)
                return ToolResult(status='ok', code='OK', text='Exact completed work for fresh fit.')
            async def boundary(calls):
                result = await batch(calls)
                # Existing tool-refit boundary meets the native request after the handle vanished.
                live.call.tool_refit.set()
                if failed_fit_same_revision:
                    live.call.failed_preparation_revision = live.host._history_version
                return result
            monkeypatch.setattr(live.host.tools, 'execute_result', blocked)
            monkeypatch.setattr(live.host, 'tool_batch', boundary)
            old = live.wire
            old.tool('fresh-result', 'list_available_tools', {})
            try:
                await live.wait(entered.is_set)
                old.put({'sessionResumptionUpdate': {'resumable': False}})
                old.put({'goAway': {'timeLeft': '30s'}})
                await live.wait(lambda: live.call.renewing and live.call.resume)
                release.set()
                await live.wait(lambda: len(live.provider.connections) == 2 and live.call.connected.is_set()
                                and live.host.connection_number == 2)
                assert not live.call.connection_resumed
                assert live.wire.sent[0]['setup']['sessionResumption'] == {}
                await live.wait(lambda: any(event.get('clientContent', {}).get('turnComplete') for event in live.wire.sent))
                await live.quiet()
                assert 'Exact completed work for fresh fit.' in json.dumps(live.wire.sent[0])
                assert any(event.get('clientContent', {}).get('turnComplete') for event in live.wire.sent)
                assert not live.call.deferred_tool_outputs and not live.call.pending_offers
                assert 'fresh-result' not in old.outputs and 'fresh-result' not in live.wire.outputs
                assert effects == ['list_available_tools']
            finally:
                release.set()
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())
