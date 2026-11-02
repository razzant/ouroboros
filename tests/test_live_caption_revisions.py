"""Late native caption corrections reach the same durable and selected call row."""
import asyncio
from dataclasses import replace

import pytest

from skills.live_conversation.driver import _Call
from tests.test_gemini_live_boundaries import deny_network as deny_network, gemini as gemini, make_session, receive
from tests.test_gemini_live_captions import THINKING, GREETING, replay_greeting, finals
from tests.test_live_conversation import host_world as host_world


@pytest.mark.parametrize('already_persisted', [False, True])
def test_thinking_caption_tail_reaches_durable_history(host_world, gemini, monkeypatch, already_persisted):
    """Port of Astra R4-1, with identity, no-op replay and distinct-item guards."""
    from ouroboros import dialogue_store as ds
    from supervisor import message_bus
    host = host_world.host
    peer_captions, projections = [], []
    original_send = host.hub.send_caption
    def send_caption(*args, **kwargs):
        peer_captions.append((args, kwargs))
        return original_send(*args, **kwargs)
    monkeypatch.setattr(host.hub, 'send_caption', send_caption)
    monkeypatch.setattr(message_bus.get_bridge(), '_broadcast_fn', projections.append)
    def saved(item):
        return [entry for entry in ds.iter_rows(host.ctx.drive_root, 'chat')
                if entry.row.get('spoken', {}).get('item_id') == item]
    async def run():
        session, wire = make_session(gemini, model=THINKING)
        call = _Call(host)
        writer = asyncio.create_task(call.save_rows())
        try:
            await session.open()
            await session.receive()
            original, = finals(await replay_greeting(session, wire))
            item = original['item_id']
            call.audio_ms[item] = 1000
            if already_persisted:
                call.played[item] = 1000
            before = len(host.messages)
            await call.caption(original)
            await asyncio.wait_for(call.rows.join(), 10)
            first = saved(item)
            message = host.messages[before] if already_persisted else None
            if already_persisted:
                assert first[0].row['text'] == GREETING, 'IDLE must finalize before any tail arrives'
                # A re-fit replaces the system view, preserving this transcript row.
                host.messages[:] = host.plan.reproject_transcript(host.messages, host.plan.initial_mode)
            tokens, version = host._history_estimate(), host._history_version
            revised, = finals(await receive(session, wire, {'serverContent': {'outputTranscription': {
                'text': ' До связи.', 'finished': True, 'startOffset': '8s', 'endOffset': '9s'}}}))
            assert revised['item_id'] == item and revised['replace'] is True
            assert revised['text'] == GREETING + ' До связи.'
            await call.caption(revised)
            if not already_persisted:
                call.played[item] = 1000
                await call.finish_spoken(item, completed=True)
            await asyncio.wait_for(call.rows.join(), 10)
            row, = saved(item)
            assert len(host.messages[before:]) == 1
            assert peer_captions[-1][0][1] == revised['text']
            assert peer_captions[-1][1]['replace'] is True
            assert (row.row['text'], host.messages[before]['content']) == (revised['text'], revised['text'])
            assert host._history_version == version + 1 and host._history_estimate() > tokens
            if already_persisted:
                assert host.messages[before] is message
                assert (row.seq, row.length, row.dispatch) == (first[0].seq, first[0].length, first[0].dispatch)
                assert row.row == {**first[0].row, 'text': revised['text']}
            assert projections[-1]['content'] == revised['text']
            assert projections[-1]['spoken'] == row.row['spoken']
            snapshot = (ds.room(host.ctx.drive_root, host.turn.spec.chat_id),
                        host._history_version, len(projections))
            with monkeypatch.context() as guarded:
                def no_store(*args, **kwargs):
                    raise AssertionError('an identical replay must not touch the store')
                guarded.setattr(ds, 'transaction', no_store)
                await call.caption(revised)
                await asyncio.wait_for(call.rows.join(), 10)
                assert not writer.done(), 'replay attempted storage'
            assert (ds.room(host.ctx.drive_root, host.turn.spec.chat_id),
                    host._history_version, len(projections)) == snapshot
            assert saved(item) == [row]
            await call.caption({**original, 'item_id': 'next-item', 'text': 'A separate response.'})
            call.played['next-item'] = 500
            await call.finish_spoken('next-item', completed=True)
            await asyncio.wait_for(call.rows.join(), 10)
            assert len(saved(item)) == len(saved('next-item')) == 1
            assert saved('next-item')[0].seq != row.seq
            assert [m['content'] for m in host.messages[before:]] == [revised['text'], 'A separate response.']
        finally:
            writer.cancel()
            await asyncio.gather(writer, return_exceptions=True)
            await session.close()
    asyncio.run(run())


@pytest.mark.parametrize('status', ['played', 'interrupted', 'generated'])
def test_replacement_preserves_playback_and_default_replays_do_not_write(host_world, monkeypatch, status):
    from ouroboros import dialogue_store as ds
    host = host_world.host
    async def run():
        await host.caption('ouroboros', 'First words.', 'item', played_ms=123,
                           completed=status == 'played', interrupted=status == 'interrupted')
        first = list(ds.iter_rows(host.ctx.drive_root, 'chat'))[-1]
        message = host.messages[-1]
        prefix = message['content'].removesuffix('First words.')
        with monkeypatch.context() as guarded:
            guarded.setattr(ds, 'transaction', lambda *a, **kw: pytest.fail('default replay touched store'))
            await host.caption('ouroboros', 'First words.', 'item')
            await host.caption('ouroboros', 'Changed without replacement intent.', 'item')
        await host.caption('ouroboros', 'Full corrected words.', 'item', replace=True, played_ms=9999,
                           completed=status != 'played', interrupted=status != 'interrupted')
        row = ds.row_at(host.ctx.drive_root, 'chat', first.seq)
        assert row.row == {**first.row, 'text': 'Full corrected words.'}
        assert row.row['spoken']['status'] == status
        assert message is host.messages[-1] and message['content'] == prefix + 'Full corrected words.'
    asyncio.run(run())


def test_presence_spoken_replacement_keeps_transport_provenance(host_world):
    from ouroboros import dialogue_store as ds
    host = host_world.host
    host.turn.spec = replace(host.turn.spec, presence_binding_id='meeting-binding')
    host.task['metadata']['presence'] = {'binding_id': 'meeting-binding', 'transport_skill': 'test-peer',
        'event': {'source_event_id': 'meeting-1', 'provider': 'test-meeting', 'account_id': 'account-1',
                  'conversation_id': 'room-1', 'thread_id': '', 'conversation_key': 'test-meeting:account-1:room-1',
                  'actor': {'display_name': 'Participant', 'platform_actor_id': 'participant-1'}}}
    async def run():
        await host.caption('ouroboros', 'First words.', 'item', completed=True)
        first = list(ds.iter_rows(host.ctx.drive_root, 'chat'))[-1]
        await host.caption('ouroboros', 'Corrected words.', 'item', replace=True)
        rows = list(ds.iter_rows(host.ctx.drive_root, 'chat'))
        assert rows[-1].seq == first.seq
        assert rows[-1].row == {**first.row, 'text': 'Corrected words.'}
        assert rows[-1].row['presence_provenance']['binding_id'] == 'meeting-binding'
        assert host.messages[-1]['content'] == 'Corrected words.'
    asyncio.run(run())
