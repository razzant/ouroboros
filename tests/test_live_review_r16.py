"""Independent R16 boundary probes; only synthetic data and local ports."""
import asyncio
from dataclasses import replace

import pytest

from ouroboros import call_protocol as cp
from ouroboros.call_peers import LiveCallHub
from ouroboros.contracts.live_conversation import ContextUpdate, Video
from skills.live_conversation.driver import _Call
from tests.test_live_conversation_driver import Host, Session, until
from tests.test_live_proactive_host import synthetic_jpeg


@pytest.mark.parametrize('owner_interrupts', [False, True])
def test_blind_revisit_frame_fact_and_requested_offer_survive_owner_input(tmp_path, owner_interrupts):
    async def scenario():
        host = Host()
        host.model = replace(host.model, vision=False)
        call = _Call(host)
        call.session = Session()
        call.connected.set()
        owner_rows = []

        async def retain(data, mime, *, source=''):
            if owner_interrupts:
                # Same sequence as provider(input_started) while disk retention yields.
                call.input_open = True
                rows = await call.attention.deliver(call.session)
                owner_rows.extend(rows)
                if rows:
                    await call.context_update(rows)
            path = tmp_path / 'remote-frame.jpg'
            path.write_bytes(data)
            return str(path)

        host.retain_frame = retain
        call.attention.mode = 'turns'
        call.attention.frame(cp.PeerVideo('remote', cp.PEER_OWNER, 'image/jpeg', b'exact-frame'))
        await call.offer_fact('This is the look you asked for.', requested_look=True)
        await call.offer_pending()
        if owner_interrupts:
            assert not call.session.sent and call.pending_offers, 'speech defers both the frame fact and the look'
        # Finish the owner's turn and try the next physical boundary as well.
        call.input_open = False
        next_rows = await call.attention.deliver(call.session)
        if next_rows:
            await call.context_update(next_rows)
        await call.offer_pending()
        sent = [row for command in call.session.sent if isinstance(command, ContextUpdate)
                for row in command.messages]
        assert any('remote-frame.jpg' in str(row) for row in sent), (
            f'exact frame fact retained={host.context}, sent={sent}, pending={call.pending_offers}, '
            f'claimed={call.attention.seen}, owner_rows={owner_rows}')
        assert sum('remote-frame.jpg' in str(row) for row in sent) == 1
        if owner_interrupts:
            assert sum('look you asked' in str(row) for row in sent) == 1
            assert call.response_open, 'the mind-named look is offered once after speech ends'
    asyncio.run(scenario())


def test_output_flush_preserves_latest_unconsumed_input_frame():
    hub = LiveCallHub(cp.CallSpec('review-frame', 1, 'test', 'test'), rates=(24000, 24000))
    try:
        hub.join(cp.PeerInfo('owner', cp.PEER_OWNER, True, True, video=True), 24000)
        hub.next_event(0)
        data = synthetic_jpeg()
        assert hub.receive_video('owner', cp.VideoFrame(hub.epoch, 0, 'image/jpeg', data))
        hub.flush()  # driver.interrupt uses this for ordinary barge-in
        event = hub.next_event(0)
        assert isinstance(event, cp.PeerVideo), 'accepted latest input is gone; browser now skips identical JPEG forever'
        assert event.data == data
    finally:
        hub.close('test')


def test_output_flush_keeps_sources_independent_and_each_latest_frame():
    hub = LiveCallHub(cp.CallSpec('review-sources', 1, 'test', 'test'), rates=(24000, 24000))
    try:
        for peer, role in [('owner', cp.PEER_OWNER), ('guest', cp.PEER_MEETING)]:
            hub.join(cp.PeerInfo(peer, role, True, True, video=True), 24000)
            hub.next_event(0)
        data = synthetic_jpeg()
        for peer in ('owner', 'guest'):
            assert hub.receive_video(peer, cp.VideoFrame(hub.epoch, 0, 'image/jpeg', data))
            assert hub.receive_video(peer, cp.VideoFrame(hub.epoch, 1, 'image/jpeg', data))
        hub.flush()
        events = [hub.next_event(0), hub.next_event(1.1)]  # retain the hub's one-second cadence
        assert {event.peer_id for event in events} == {'owner', 'guest'}
        assert all(isinstance(event, cp.PeerVideo) and event.data == data for event in events)
        assert hub.next_event(0) is None, 'only one latest accepted frame per source'
    finally:
        hub.close('test')


def test_transport_rejects_frame_from_prior_playback_epoch():
    hub = LiveCallHub(cp.CallSpec('review-epoch', 1, 'test', 'test'), rates=(24000, 24000))
    try:
        hub.join(cp.PeerInfo('owner', cp.PEER_OWNER, True, True, video=True), 24000)
        hub.next_event(0)
        epoch = hub.epoch
        hub.flush()
        assert not hub.receive_video('owner', cp.VideoFrame(epoch, 0, 'image/jpeg', synthetic_jpeg()))
    finally:
        hub.close('test')


@pytest.mark.parametrize('resume', [False, True])
@pytest.mark.parametrize('vision', [False, True])
def test_frame_delivery_across_actual_fresh_fit_and_resume(tmp_path, resume, vision):
    async def scenario():
        host = Host()
        host.model = replace(host.model, vision=vision)
        call = _Call(host)
        call.session = Session()
        call.connected.set()

        async def retain(data, mime, *, source=''):
            path = tmp_path / 'same-frame.jpg'
            path.write_bytes(data)
            return str(path)
        host.retain_frame = retain
        call.attention.frame(cp.PeerVideo('remote', cp.PEER_OWNER, 'image/jpeg', b'same-frame'))
        await call.offer_fact('The owner started showing the screen.')
        # Frame retention finishes, then an input keeps this batch pending until renewal.
        call.pending_offers[:0] = await call.attention.deliver(call.session)
        fitted_history = list(host.context) if not resume else []
        # Exercise the real connection boundary, not a copy of its implementation.
        session = Session()
        call.resume = resume
        call.writer = asyncio.create_task(call.save_rows())
        task = asyncio.create_task(call.consume(session))
        session.emit('ready', input_rate=24000, output_rate=24000)
        try:
            await until(lambda: any(isinstance(c, ContextUpdate) for c in session.sent))
            await call.attention.deliver(session)
        finally:
            host.hub.ended_by_owner = True
            try:
                await asyncio.wait_for(task, 2)
            finally:
                call.writer.cancel()
                await asyncio.gather(call.writer, return_exceptions=True)
        delivered = [row for command in session.sent if isinstance(command, ContextUpdate)
                     for row in command.messages]
        copies = [row for row in fitted_history + delivered if 'same-frame.jpg' in str(row)]
        assert len(copies) == (not vision), f'{len(copies)} copies of one canonical frame fact: {copies}'
        assert len([row for row in host.context if 'same-frame.jpg' in str(row)]) == (not vision)
        assert sum(isinstance(c, Video) for c in session.sent) == (vision and not resume)
    asyncio.run(scenario())
