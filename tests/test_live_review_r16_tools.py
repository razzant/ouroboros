"""R16: argument failures and unavailable pixels are facts, not call teardown."""
import asyncio
import base64
import json
from dataclasses import replace

import pytest

from ouroboros import call_protocol as cp
from ouroboros.contracts.live_conversation import ContextUpdate, ToolOutputs
from ouroboros.contracts.realtime import CatalogModel, RouteFacts
from ouroboros.tool_call_log import invocation_scope
from skills.live_conversation.attention import Attention, attend_tool
from skills.live_conversation.driver import _Call, engine_sentence
from tests.test_live_conversation import host_world as host_world_fixture
from tests.test_live_conversation_driver import Batch, Host, Session, started, until

host_world = host_world_fixture


@pytest.mark.parametrize('arguments', [{'see': 'invalid'}, {'look': 'yes'}, {'revisit_after_s': -1}])
def test_bad_attend_is_a_tool_row_and_the_next_valid_call_still_runs(host_world, arguments):
    async def scenario():
        host = host_world.host
        host.call_tools['attend'] = attend_tool()
        host.state['attention'] = Attention(host)
        for cid, args in [('bad', arguments), ('good', {'see': 'asked'})]:
            rows = await host.tool_batch([{'id': cid, 'type': 'function', 'function': {
                'name': 'attend', 'arguments': json.dumps(args)}}])
            text = next(r['content'] for r in rows if r.get('tool_call_id') == cid)
            assert rows.silent == (cid,)
            if cid == 'bad':
                assert text.startswith('Call tool argument error: ')
            else:
                assert json.loads(text) == {'mode': 'asked', 'changed': True}
            assert not host.hub.closed
    asyncio.run(scenario())


@pytest.mark.parametrize('error', [TypeError, ValueError])
def test_handler_argument_errors_are_rows_not_teardown(host_world, error):
    async def scenario():
        host = host_world.host
        async def handler(_host, _args):
            raise error('synthetic handler validation')
        host.call_tools['attend'] = replace(attend_tool(), handler=handler)
        rows = await host.tool_batch([{'id': 'bad', 'type': 'function',
            'function': {'name': 'attend', 'arguments': '{}'}}])
        assert next(r['content'] for r in rows if r.get('tool_call_id') == 'bad') == (
            'Call tool argument error: synthetic handler validation')
        assert not host.hub.closed
    asyncio.run(scenario())


@pytest.mark.parametrize('failure', ['empty', 'io', 'typed', 'success'])
@pytest.mark.parametrize('look', [False, True])
def test_blind_frame_retention_reports_failure_and_continues(tmp_path, failure, look):
    from ouroboros.contracts.realtime import RealtimeSetupError
    async def scenario():
        host = Host()
        host.model = replace(host.model, vision=False)
        host.state['greeted'] = True
        async def retain(data, mime, *, source=''):
            if failure == 'io':
                raise OSError('synthetic read-only artifact store')
            if failure == 'typed':
                raise RealtimeSetupError('frame_retention', 'synthetic retention failure')
            if failure == 'empty':
                return ''
            path = tmp_path / 'frame.jpg'
            path.write_bytes(data)
            return str(path)
        host.retain_frame = retain
        async def tools(calls):
            rows = []
            for call in calls:
                with invocation_scope({'tool_call_id': call['id']}):
                    text = await host.state['attention'].attend(json.loads(call['function']['arguments']))
                rows.append({'role': 'tool', 'tool_call_id': call['id'], 'content': text})
            return Batch(rows, silent=[call['id'] for call in calls])
        host.tool_batch = tools
        task, session = await started(host)
        try:
            session.emit('ready', input_rate=24000, output_rate=24000)
            host.hub.events.append(cp.PeerVideo('source', cp.PEER_OWNER, 'image/jpeg', b'exact'))
            await until(lambda: host.state['attention'].frames)
            if look:
                session.emit('tool_call', call_id='look', name='attend', arguments='{"look":true}')
                kind = ToolOutputs
            else:
                session.emit('input_started')
                kind = ContextUpdate
            await until(lambda: any(isinstance(c, kind) for c in session.sent) or task.done())
            if task.done():
                await task
            rows = [r for c in session.sent if isinstance(c, kind) for r in c.messages]
            assert ('frame.jpg' if failure == 'success' else 'shared frame could not be saved; ask what is on the screen') in str(rows)
            assert not task.done() and not host.hub.closed
            if look:
                assert next(c for c in session.sent if isinstance(c, ToolOutputs)).silent == ()
        finally:
            host.hub.ended_by_owner = True
            if not task.done():
                await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_attend_images_follow_call_ids_even_when_results_are_read_out_of_order(tmp_path):
    async def scenario():
        host = Host()
        async def retain(data, mime, *, source=''):
            path = tmp_path / (data.decode() + '.jpg')
            path.write_bytes(data)
            return str(path)
        host.retain_frame = retain
        attention = Attention(host)
        results = {}
        for cid in ('first', 'second'):
            attention.frame(cp.PeerVideo('source', cp.PEER_OWNER, 'image/jpeg', cid.encode()))
            with invocation_scope({'tool_call_id': cid}):
                results[cid] = await attention.attend({'look': True})
        for cid in ('second', 'first'):
            calls = [{'id': cid, 'function': {'name': 'attend'}}]
            rows = Batch([{'role': 'tool', 'tool_call_id': cid, 'content': results[cid]}], silent=[cid])
            images, silent = attention.tool_outputs(calls, rows)
            assert not silent and len(images) == 1
            assert base64.b64decode(images[0]['content'][1]['image_url']['url'].split(',')[1]) == cid.encode()
            assert attention.tool_outputs(calls, rows)[0] == [], 'replayed results do not repeat pixels'
    asyncio.run(scenario())


def test_host_binds_the_actual_call_id_for_an_attend_look(host_world, tmp_path):
    async def scenario():
        host = host_world.host
        host.call_tools['attend'] = attend_tool()
        attention = host.state['attention'] = Attention(host)
        async def retain(data, mime, *, source=''):
            path = tmp_path / 'exact.jpg'
            path.write_bytes(data)
            return str(path)
        host.retain_frame = retain
        attention.frame(cp.PeerVideo('source', cp.PEER_OWNER, 'image/jpeg', b'exact'))
        calls = [{'id': 'actual-id', 'type': 'function', 'function': {'name': 'attend', 'arguments': '{"look":true}'}}]
        changes = await host.tool_batch(calls)
        assert set(attention.looks) == {'actual-id'}
        images, silent = attention.tool_outputs(calls, changes)
        assert len(images) == 1 and silent == () and not attention.looks
    asyncio.run(scenario())


def test_frame_retention_does_not_swallow_cancellation_or_seeing_route_failures():
    async def scenario():
        for vision in (False, True):  # cancellation is never a failed save
            host = Host()
            host.model = replace(host.model, vision=vision)
            async def cancelled(*_args, **_kwargs):
                raise asyncio.CancelledError()
            host.retain_frame = cancelled
            attention = Attention(host)
            attention.frame(cp.PeerVideo('source', cp.PEER_OWNER, 'image/jpeg', b'exact'))
            with pytest.raises(asyncio.CancelledError):
                await attention.attend({'look': True})
            assert not attention.seen and not attention.looks
        # R17 N2: a seeing route's failed save is said in words, not as an argument error, and the pixels still come.
        host = Host()
        host.model = replace(host.model, vision=True)
        async def broken(*_args, **_kwargs):
            raise OSError("disk full")
        host.retain_frame = broken
        attention = Attention(host)
        attention.frame(cp.PeerVideo('source', cp.PEER_OWNER, 'image/jpeg', b'exact'))
        result = json.loads(await attention.attend({'look': True}))
        assert result['frames'] == [{'source': 'source', 'kind': '', 'error': 'The shared frame could not be saved as a file.'}]
        [images] = attention.looks.values()
        assert images and 'not saved' in images[0]['content'][0]['text']
    asyncio.run(scenario())


def test_legacy_catalog_constructor_and_silence_are_independent_of_vendor_and_vision():
    # All pre-existing positional arguments keep their positions and defaults.
    old = CatalogModel('old', 'Old', RouteFacts(), False, False, (), (), frozenset(), ())
    assert old.silent_tool_results is True
    assert 'not guaranteed silent' not in engine_sentence(old)
    renamed = replace(old, id='vendor-free', vision=True, silent_tool_results=False)
    assert 'not guaranteed silent' in engine_sentence(renamed)
    assert old.thinking_default == ''  # owner 28.10: a named default is optional and keyword-only


def test_a_default_thinking_level_must_be_one_of_the_models_levels():
    from ouroboros.contracts.realtime import RealtimeSetupError
    named = CatalogModel('m', 'M', thinking_levels=('minimal', 'low'), honours=frozenset({'thinking'}), thinking_default='low')
    assert named.thinking_default == 'low'
    with pytest.raises(RealtimeSetupError, match='default thinking level'):
        CatalogModel('m', 'M', thinking_levels=('minimal', 'low'), honours=frozenset({'thinking'}), thinking_default='high')


@pytest.mark.parametrize('duration,played,heard', [(None, 0, False), (None, 1, True), (1000, 979, False), (1000, 980, True)])
def test_caption_completion_uses_the_same_heard_rule(duration, played, heard):
    async def scenario():
        call = _Call(Host())
        call.session = Session()
        if duration is not None:
            call.audio_ms['item'] = duration
        call.played['item'] = played
        assert call.heard('item') is heard
        await call.caption(dict(speaker='ouroboros', item_id='item', text='Exact words', final=True))
        assert ('item' not in call.pending_spoken) is heard
        assert (not call.rows.empty()) is heard
    asyncio.run(scenario())
