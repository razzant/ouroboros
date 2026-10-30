"""Observed call prefixes use the ordinary authored view and exact-source custody."""
import asyncio
import copy
import json

from tests.test_live_conversation import host_world as host_world


def invocation(ident, **arguments):
    return {'id': ident, 'type': 'function', 'function': {
        'name': 'compact_context', 'arguments': json.dumps(arguments)}}


def test_first_call_tool_can_inspect_the_accepted_view_and_budget(host_world):
    from ouroboros.contracts.live_conversation import LiveOptions
    host = host_world.host
    assert not getattr(host.ctx, '_last_context_observation', None)

    async def consume(session):
        assert session.opened
        result = await host.tool_batch([invocation('inspect', inspect=True)])
        text = next(row['content'] for row in result if row.get('tool_call_id') == 'inspect')
        assert 'Context view unavailable' not in text
        shown = json.loads(text)
        assert shown['memory_view_status'] == 'observed_snapshot'
        assert {'memory', 'host'} <= shown['inventory']['by_kind'].keys()
        hosts = [row for row in shown['units'] if row['kind'] == 'host']
        assert hosts and all(row['form'] in {'whole', 'operative'} and row['resident_tokens'] >= 0 for row in hosts)
        assert shown['fit_measurement']['estimated_input_tokens'] == host._prepared_view[0]['fit']['measurement']['estimated_input_tokens']
        assert shown['fit_measurement']['response_reserve_tokens'] == 10000
        assert shown['work_room']['first_send'] and shown['work_room']['requested_tokens'] == 25000
        assert shown['resident_memory']['room_tokens'] > 0
    asyncio.run(host.session(consume, LiveOptions(instructions='Measured call instructions.')))


def test_authored_fold_waits_for_quiet_prices_wire_preserves_new_words_and_restores_exact_units(host_world, monkeypatch):
    from ouroboros.context_budget import bound_host_row, OWNER_WORDS_KIND
    from ouroboros.contracts.live_conversation import LiveOptions
    from ouroboros.tools.compact_context import _compact_context
    from skills.live_conversation.driver import _Call

    host = host_world.host
    source = [
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'effect', 'type': 'function', 'function': {
            'name': 'read_file', 'arguments': '{"path":"retained-source"}'}}]},
        {'role': 'tool', 'tool_call_id': 'effect', 'content': 'Exact completed tool evidence. ' * 1800}]
    host.messages.extend(copy.deepcopy(source))
    host._context_changed()
    host.executed['effect'] = ('completed identity', [copy.deepcopy(source[-1])])
    executed = copy.deepcopy(host.executed['effect'])
    owner = bound_host_row(OWNER_WORDS_KIND, 'New owner words arriving after inspect stay whole.')
    options = LiveOptions(instructions='The complete feature frame also occupies the measured wire.')
    saved = {}
    from ouroboros.realtime_setup import measure_session_wire
    measured = []
    def price(*args, **kwargs):
        measured.append(kwargs)
        return measure_session_wire(*args, **kwargs)
    monkeypatch.setattr('ouroboros.realtime_setup.measure_session_wire', price)

    async def first(session):
        shown = json.loads(_compact_context(host.ctx, inspect=True))
        unit = next(row for row in shown['units'] if row['kind'] == 'tool')
        saved['restore'] = dict(checkpoint_ref=shown['restore_checkpoint'],
                                unit_id=unit['unit_id'], raw_sha256=unit['raw_sha256'])
        host.messages.append(owner)
        keep = [row['unit_id'] for row in shown['units'] if 'unit_id' in row and row != unit]
        result = await host.tool_batch([invocation('fold', working_note='The completed read established the retained fact.',
            keep_unit_ids=keep, expected_view_revision=shown['view_revision']),
            {'id': 'load-reader', 'type': 'function', 'function': {
                'name': 'enable_tools', 'arguments': '{"tools":"read_file"}'}}])
        assert 'requested' in next(row['content'] for row in result if row.get('tool_call_id') == 'fold')
        call = _Call(host)
        call.session = session
        call.input_open = True
        peer = asyncio.create_task(call.peers())
        try:
            await asyncio.sleep(.05)
            assert host.context_view_pending() and source[-1] in host.messages
            assert not call.renewing, 'An owner speaking prevents publication and reconnect.'
            call.input_open = False
            await asyncio.wait_for(peer, 10)
        finally:
            peer.cancel()
            await asyncio.gather(peer, return_exceptions=True)
        assert call.renewing and not call.resume and not host.context_view_pending()
        receipt = host.ctx._context_view_receipt
        assert receipt['status'] == 'applied', receipt
        assert receipt['fit_after']['estimated_input_tokens'] < receipt['fit_before']['estimated_input_tokens']
        assert source[-1] not in host.messages and owner in host.messages
        assert host.executed['effect'] == executed
        assert any(row.get('tool_call_id') == 'fold' for row in host.messages)
        assert any(row.get('tool_call_id') == 'load-reader' for row in host.messages)
        saved['receipt'] = copy.deepcopy(receipt)
        assert any(row.get('fresh') is False and row.get('_room_fact') is False for row in measured), measured

    async def second(session):
        receipt = saved['receipt']
        actual = host._prepared_view[0]
        # The authored receipt prices the serving view. A fresh open additionally
        # replaces its room fact and records its own independently measured wire.
        assert receipt['fit_after']['estimated_input_tokens'] > 0
        assert host.ctx._context_view_receipt['sent_fit']['wire_sha256'] == actual['wire_sha256']
        assert not actual['work_room']['first_send'] and actual['work_room']['requested_tokens'] == 0
        assert 'requested' in _compact_context(host.ctx, restore_unit_refs=[saved['restore']])
        assert await host.apply_context_view()
        assert host.ctx._context_view_receipt['status'] == 'applied'
        from ouroboros.context_compaction import context_units
        from ouroboros.artifacts import read_actor_source_bytes
        checkpoint = saved['restore']['checkpoint_ref']
        original = json.loads(read_actor_source_bytes(host.ctx.drive_root, host.ctx.task_id, checkpoint))['messages']
        restored_unit = next(unit for unit in context_units(original, scope='dialogue')
                             if unit.unit_id == saved['restore']['unit_id'])
        assert original[restored_unit.start:restored_unit.end + 1] == source
        # Ordinary restore is the exact source text, not a live protocol call to replay.
        assert any(isinstance(row.get('content'), list) and restored_unit.source_text in row['content'][0].get('text', '')
                   for row in host.messages)
        assert host.executed['effect'] == executed and owner in host.messages

    async def run():
        await host.session(first, options)
        await host.session(second, options)
    asyncio.run(run())


def test_call_note_form_selection_and_exact_restore_survive_reopens(host_world):
    from dataclasses import replace
    from ouroboros.contracts.live_conversation import LiveOptions
    from ouroboros.memory_view import _note_pointer, _note_lines, snapshot_from_json
    from tests.test_memory_view_lever import note_plan

    host = host_world.host
    source, snapshot = note_plan()
    host.plan = host.ctx.context_fit_plan = replace(host.plan, core=source.core, core_sha256=source.core_sha256,
        max_projection=source.max_projection, low_projection=source.low_projection,
        nano_projection=replace(source.max_projection, mode='nano'), output_reserve_tokens=10000)
    host.messages[:] = host.plan.messages_for('nano')
    note = snapshot.room['notes'][0]
    saved = {}

    async def select(session):
        result = await host.tool_batch([invocation('note-line', view_address=['F8:' + note['id']])])
        assert 'requested' in result[-1]['content']
        assert await host.apply_context_view()
        receipt = host.ctx._context_view_receipt
        assert receipt['status'] == 'applied', receipt
        assert receipt['fit_after']['estimated_input_tokens'] < receipt['fit_before']['estimated_input_tokens']
        saved['wire'] = receipt['fit_after']['wire_sha256']

    async def restore(session):
        assert host.ctx._context_view_receipt['sent_fit']['wire_sha256'] == host._prepared_view[0]['wire_sha256']
        wire = json.loads(session.selected.messages_json)
        assert _note_pointer(note) in wire[0]['content'][-1]['text']
        assert note['text'] not in json.dumps(wire)
        result = await host.tool_batch([invocation('note-whole', view_address=['F8:' + note['id'] + ':whole'])])
        assert 'requested' in result[-1]['content']
        assert await host.apply_context_view()
        assert host.ctx._context_view_receipt['status'] == 'applied'

    async def whole(session):
        wire = json.loads(session.selected.messages_json)
        assert _note_lines([note])[0] in wire[0]['content'][-1]['text']
        assert next(n for n in snapshot_from_json(host.plan.core.memory_view_json).room['notes'] if n['id'] == note['id']) == note
        assert {'note-line', 'note-whole'} <= host.executed.keys()

    async def run():
        for consume in (select, restore, whole):
            await host.session(consume, LiveOptions())
    asyncio.run(run())
