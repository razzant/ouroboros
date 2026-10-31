"""Discrete observations offer a turn; media and self output never choose attention."""
import asyncio
import base64
import json
from dataclasses import replace
from pathlib import Path

import pytest

from ouroboros import call_protocol as cp
from ouroboros.contracts.live_conversation import ContextUpdate, LiveHost, Video
from ouroboros.tool_call_log import invocation_scope
from skills.live_conversation.attention import Attention, attend_tool
from tests.test_live_conversation_driver import Batch, Host, started, until


def test_facts_are_canonical_at_once_but_offered_once_only_when_quiet():
    async def scenario():
        host = Host()
        host.state['greeted'] = True
        task, session = await started(host)
        try:
            session.emit('ready', input_rate=24000, output_rate=24000)
            session.emit('input_started')
            await asyncio.sleep(.02)
            host.hub.events.extend([
                cp.PeerControl('owner', cp.PEER_OWNER, dict(type='share', sharing=True, kind='screen')),
                cp.PeerJoined(cp.PeerInfo('guest', cp.PEER_MEETING, True, True, 'Reported room')),
                cp.PeerLeft('guest', 'finished'),
            ])
            await until(lambda: len(host.context) == 3)
            assert not any(isinstance(c, ContextUpdate) for c in session.sent)
            session.emit('input_done')
            await until(lambda: any(isinstance(c, ContextUpdate) for c in session.sent))
            updates = [c for c in session.sent if isinstance(c, ContextUpdate)]
            assert len(updates) == 1 and len(updates[0].messages) == 3
            assert all(row['content'].startswith('[Host fact; not owner words]') for row in host.context)
            await asyncio.sleep(.04)
            assert len([c for c in session.sent if isinstance(c, ContextUpdate)]) == 1
        finally:
            host.hub.ended_by_owner = True
            await asyncio.wait_for(task, 2)
    asyncio.run(scenario())



@pytest.mark.parametrize('mode', ['turns', 'stream', 'asked'])
def test_frames_never_offer_and_delivery_is_the_minds_choice(mode):
    async def scenario():
        host = Host()
        host.state['greeted'] = True
        task, session = await started(host)
        try:
            host.state['attention'].mode = mode
            session.emit('ready', input_rate=24000, output_rate=24000)
            host.hub.events.extend([cp.PeerVideo('owner', cp.PEER_OWNER, 'image/jpeg', b'exact'),
                                    cp.PeerVideo('owner', cp.PEER_OWNER, 'image/jpeg', b'exact')])
            await until(lambda: not host.hub.events)
            assert not any(isinstance(c, ContextUpdate) for c in session.sent)
            assert len([c for c in session.sent if isinstance(c, Video)]) == (mode == 'stream')
            session.emit('input_started')
            await asyncio.sleep(.03)
            assert len([c for c in session.sent if isinstance(c, Video)]) == (mode != 'asked')
            session.emit('input_done')
            session.emit('input_started')
            await asyncio.sleep(.03)
            assert len([c for c in session.sent if isinstance(c, Video)]) == (mode != 'asked'), 'no exact duplicate'
            assert not host.context, 'frames, input boundaries and usage are not discrete offers'
        finally:
            host.hub.ended_by_owner = True
            await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


@pytest.mark.parametrize('vision', [True, False])
def test_look_has_exact_image_or_reader_and_continues_only_when_asked(tmp_path, vision):
    async def scenario():
        host = Host()
        host.model = replace(host.model, vision=vision)
        async def retain(data, mime, *, source=''):
            path = tmp_path / 'exact.jpg'
            path.write_bytes(data)
            return str(path)
        host.retain_frame = retain
        attention = host.state['attention'] = Attention(host)
        attention.frame(cp.PeerVideo('peer', cp.PEER_MEETING, 'image/jpeg', b'exact-jpeg'))
        tool = attend_tool()
        unchanged = json.loads(await tool.handler(host, {'see': None, 'look': None}))
        assert unchanged == {'mode': 'turns', 'changed': False} and not attention.looks
        with invocation_scope({'tool_call_id': 'look'}):
            result = await tool.handler(host, {'see': 'asked', 'look': True})
        record = json.loads(result)
        assert record['mode'] == 'asked' and record['changed']
        assert Path(record['frames'][0]['file_path']).read_bytes() == b'exact-jpeg'
        calls = [{'id':'look', 'function': {'name':'attend'}}]
        rows = Batch([{'role':'tool', 'tool_call_id':'look', 'content':result}], silent=['look'])
        images, silent = attention.tool_outputs(calls, rows)
        assert silent == ()
        assert bool(images) == vision
        if vision:
            assert base64.b64decode(images[0]['content'][1]['image_url']['url'].split(',')[1]) == b'exact-jpeg'
        else:
            assert record['reader'] == 'vlm_query(file_path=...)'
        result = await tool.handler(host, {'see': 'asked', 'look': None})
        rows = Batch([{'role':'tool', 'tool_call_id':'look', 'content':result}], silent=['look'])
        assert attention.tool_outputs(calls, rows) == ([], ('look',))
    asyncio.run(scenario())


def test_reported_speaker_not_call_title_and_toggles_offer_nothing():
    async def scenario():
        host = Host()
        host.state['greeted'] = True
        task, session = await started(host)
        try:
            session.emit('ready', input_rate=24000, output_rate=24000)
            host.hub.events.extend([
                cp.PeerControl('meeting', cp.PEER_MEETING, {'type':'participant', 'label':'Reported speaker', 'speaking':True}),
                cp.PeerAudio('meeting', cp.PEER_MEETING, b'\0\0'),
                cp.PeerControl('meeting', cp.PEER_MEETING, {'type':'participant', 'label':'Reported speaker', 'speaking':False}),
            ])
            await until(lambda: not host.hub.events)
            session.emit('caption', speaker='owner', item_id='utterance', text='Their exact words', final=True)
            await until(lambda: host.captions)
            assert host.captions[0]['speaker_label'] == 'Reported speaker'
            assert host.captions[0]['heard']
            assert host.context == [] and not any(isinstance(c, ContextUpdate) for c in session.sent)
        finally:
            host.hub.ended_by_owner = True
            await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_additive_ports_keep_old_callers():
    frame = Video(b'old bytes', 'image/jpeg')
    assert frame.source == ''
    assert asyncio.run(LiveHost.retain_frame(object(), frame.data, frame.mime)) == ''
    assert LiveHost.timing_event(object(), 'unused') is None
    # Older feature code supplies none of the new caption/retention/timing arguments.
    host = Host()
    asyncio.run(host.caption(speaker='owner', text='old words', item_id='old'))
    assert host.captions == [{'speaker':'owner', 'text':'old words', 'item_id':'old'}]


def test_silent_tool_choice_survives_refit_before_first_words(tmp_path):
    async def scenario():
        host = Host()
        host.state['greeted'] = True
        async def tools(calls):
            host.due = True
            return Batch([{'role':'tool', 'tool_call_id':'silent', 'content':'listening'}], silent=['silent'])
        host.tool_batch = tools
        task, first = await started(host)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.emit('tool_call', call_id='silent', name='attend', arguments='{}')
            await until(lambda: len(host.sessions) == 2)
            second = host.sessions[1]
            second.emit('ready', input_rate=24000, output_rate=24000)
            await asyncio.sleep(.04)
            assert second.sent == [], 'renewal must not manufacture first words after deliberate silence'
        finally:
            host.hub.ended_by_owner = True
            await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_refitted_look_does_not_send_its_pixels_twice(tmp_path):
    from skills.live_conversation.driver import _Call
    from tests.test_live_conversation_driver import Session
    async def scenario():
        host = Host()
        async def retain(data, mime, *, source=''):
            path = tmp_path / 'look.jpg'; path.write_bytes(data); return str(path)
        host.retain_frame = retain
        call = _Call(host)
        call.session = Session()
        call.connected.set()
        call.attention.frame(cp.PeerVideo('source', cp.PEER_OWNER, 'image/jpeg', b'exact'))
        with invocation_scope({'tool_call_id': 'look'}):
            await call.attention.attend({'look':True})
        call.pending_offers.extend(call.attention.looks.pop('look'))
        call.attention.seen.clear()  # fresh connection has not seen these images
        await call.offer_pending()
        assert not any(isinstance(command, Video) for command in call.session.sent)
        assert len(call.session.sent) == 1 and isinstance(call.session.sent[0], ContextUpdate)
        assert call.session.sent[0].messages[0]['_call_frame_source'] == 'source'
    asyncio.run(scenario())


def test_route_lines_disclose_actual_silence_support():
    from skills.live_conversation.driver import engine_sentence
    from skills.gemini_live.plugin import CATALOG as gemini
    from skills.grok_realtime.plugin import CATALOG as grok
    assert 'not guaranteed silent' not in engine_sentence(gemini.models[0])
    assert 'not guaranteed silent' in engine_sentence(gemini.models[1])
    assert grok.models[0].silent_tool_results
    assert 'not guaranteed silent' not in engine_sentence(grok.models[0])


def test_speech_during_frame_retention_defers_offer_without_losing_the_fact(tmp_path):
    from skills.live_conversation.driver import _Call
    from tests.test_live_conversation_driver import Session
    async def scenario():
        host = Host()
        host.model = replace(host.model, vision=False)
        call = _Call(host)
        call.session = Session(); call.connected.set()
        async def retain(data, mime, *, source=''):
            call.input_open = True
            path = tmp_path / 'frame.jpg'; path.write_bytes(data); return str(path)
        host.retain_frame = retain
        call.attention.frame(cp.PeerVideo('source', cp.PEER_OWNER, 'image/jpeg', b'exact'))
        await call.offer_fact('The owner started showing the screen.')
        await call.offer_pending()
        assert not call.session.sent and len(call.pending_offers) == 2
        call.input_open = False
        await call.offer_pending()
        assert len(call.session.sent) == 1 and len(call.session.sent[0].messages) == 2
        assert len(host.context) == 2 and not call.pending_offers
    asyncio.run(scenario())


def test_overlapping_frame_deliveries_claim_exact_bytes_once():
    from tests.test_live_conversation_driver import Session
    async def scenario():
        host = Host(); attention = Attention(host)
        attention.frame(cp.PeerVideo('source', cp.PEER_OWNER, 'image/jpeg', b'exact'))
        session = Session(); entered, release = asyncio.Event(), asyncio.Event()
        async def slow(command):
            entered.set(); await release.wait(); session.sent.append(command)
        session.send = slow
        first = asyncio.create_task(attention.deliver(session))
        await entered.wait()
        await attention.deliver(session)
        release.set(); await first
        assert len(session.sent) == 1
    asyncio.run(scenario())


def test_peer_rejoin_cannot_inherit_an_old_share_or_frame():
    async def scenario():
        host = Host(); host.state['greeted'] = True
        task, session = await started(host)
        try:
            session.emit('ready', input_rate=24000, output_rate=24000)
            session.emit('input_started')  # collect facts without offering during speech
            await asyncio.sleep(.02)
            for _ in range(2):
                host.hub.events.extend([
                    cp.PeerJoined(cp.PeerInfo('guest', cp.PEER_MEETING, True, True, 'Reported room', video=True)),
                    cp.PeerVideo('guest', cp.PEER_MEETING, 'image/jpeg', b'exact'),
                    cp.PeerControl('guest', cp.PEER_MEETING, {'type':'share','sharing':True,'kind':'screen'}),
                    cp.PeerLeft('guest', 'left'),
                ])
                await until(lambda: not host.hub.events)
            assert len(host.context) == 6 and sum('started showing' in r['content'] for r in host.context) == 2
            assert not host.state['attention'].frames and not host.state['attention'].kinds
            assert not session.sent
        finally:
            host.hub.ended_by_owner = True
            await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_quiet_native_state_does_not_offer_before_the_port_is_ready():
    from skills.live_conversation.driver import _Call
    from tests.test_live_conversation_driver import Session
    async def scenario():
        call = _Call(Host()); call.session = Session()
        await call.offer_fact('A peer joined.')
        assert call.quiet(), 'ordinary native quiet remains available to the existing refit lane'
        await call.offer_pending()
        assert not call.session.sent and len(call.pending_offers) == 1
        call.connected.set()
        await call.offer_pending()
        assert len(call.session.sent) == 1 and not call.pending_offers
    asyncio.run(scenario())


@pytest.mark.parametrize('resume', [False, True])
def test_a_fresh_fit_points_to_the_facts_it_holds_and_a_resume_sends_them(resume):
    """A share fact queued while the owner spoke is canonical at once. A fresh fit already holds it, so the new
    connection is offered one pointer to it, never the fact a second time; a resumed provider session never saw
    it, so the fact itself goes."""
    async def scenario():
        host = Host()
        host.state['greeted'] = True
        task, first = await started(host)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.emit('resumable', handle='handle')
            first.emit('input_started')
            await asyncio.sleep(.02)
            host.hub.events.append(cp.PeerControl('owner', cp.PEER_OWNER, dict(type='share', sharing=True, kind='screen')))
            await until(lambda: len(host.context) == 1)
            first.emit('renew', within_s=2, reason='rotate', resume=resume)
            first.emit('input_done')
            await until(lambda: len(host.sessions) == 2)
            assert not any(isinstance(c, ContextUpdate) for c in first.sent)
            second = host.sessions[1]
            second.emit('ready', input_rate=24000, output_rate=24000)
            await until(lambda: any(isinstance(c, ContextUpdate) for c in second.sent))
            await asyncio.sleep(.03)
            updates = [c for c in second.sent if isinstance(c, ContextUpdate)]
            assert len(updates) == 1
            texts = [row['content'] for row in updates[0].messages]
            if resume:
                assert texts == [host.context[0]['content']]
            else:
                assert len(texts) == 1 and 'new facts are in the fitted history' in texts[0]
                assert 'started showing the screen' not in texts[0]
        finally:
            host.hub.ended_by_owner = True
            await asyncio.wait_for(task, 2)
    asyncio.run(scenario())
