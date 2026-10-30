"""Optional A7: only a duration named by the mind can offer the floor again."""
import asyncio
from dataclasses import replace

import pytest

from ouroboros import call_protocol as cp
from ouroboros.contracts.live_conversation import ContextUpdate, Video
from skills.live_conversation import attention as module
from skills.live_conversation.driver import _Call
from tests.test_live_conversation_driver import Host, Session, started, until


@pytest.mark.parametrize('consume', ['fire', 'owner', 'offer', 'renew'])
def test_named_revisit_survives_speech_offers_and_renewal_then_fires_once(monkeypatch, consume):
    now = [100.0]
    monkeypatch.setattr(module, 'monotonic', lambda: now[0])
    async def scenario():
        host = Host(); host.state['greeted'] = True
        task, session = await started(host)
        try:
            session.emit('ready', input_rate=24000, output_rate=24000)
            await asyncio.sleep(.02)
            attention = host.state['attention']
            assert attention.revisit_at is None
            now[0] = 1000000
            await asyncio.sleep(.03)
            assert not session.sent, 'there is no host default or fallback'
            await attention.attend({'see':'asked', 'revisit_after_s':7, 'look':None})
            host.hub.events.append(cp.PeerVideo('source', cp.PEER_OWNER, 'image/jpeg', b'current'))
            await until(lambda: not host.hub.events)
            if consume == 'owner':
                session.emit('input_started'); session.emit('input_done')
                await asyncio.sleep(.03)
                assert attention.revisit_at is not None
            elif consume == 'offer':
                host.hub.events.append(cp.PeerControl('owner', cp.PEER_OWNER, {'type':'share','sharing':True,'kind':'screen'}))
                await until(lambda: any(isinstance(c, ContextUpdate) for c in session.sent))
                assert attention.revisit_at is not None
                session.emit('response_done', response_id='share', usage={}, raw_usage={}, status='completed')
            elif consume == 'renew':
                deadline = attention.revisit_at
                host.due = True
                await until(lambda: len(host.sessions) == 2)
                session = host.sessions[1]
                session.emit('ready', input_rate=24000, output_rate=24000)
                await asyncio.sleep(.02)
                assert host.state['attention'] is attention and attention.revisit_at == deadline
            now[0] += 8
            if consume in {'fire', 'renew', 'owner', 'offer'}:
                await until(lambda: any(isinstance(c, ContextUpdate) and 'look you asked' in str(c.messages) for c in session.sent))
                update = next(c for c in session.sent if isinstance(c, ContextUpdate) and 'look you asked' in str(c.messages))
                assert 'look you asked' in str(update.messages)
                assert any(isinstance(c, Video) for c in session.sent), 'a named look gets the current view even in asked mode'
            before = len(session.sent)
            now[0] += 1000000
            session.emit('response_done', response_id='silent-look', usage={}, raw_usage={}, status='completed', continues=False)
            await asyncio.sleep(.04)
            assert len(session.sent) == before and attention.revisit_at is None, 'one shot never re-arms'
        finally:
            host.hub.ended_by_owner = True
            await asyncio.wait_for(task, 2)
            assert host.state['attention'].revisit_at is None
    asyncio.run(scenario())


def test_due_revisit_waits_for_quiet_and_call_end_cancels_it(monkeypatch):
    now = [0.0]; monkeypatch.setattr(module, 'monotonic', lambda: now[0])
    async def scenario():
        host = Host(); host.state['greeted'] = True
        task, session = await started(host)
        try:
            session.emit('ready', input_rate=24000, output_rate=24000)
            await asyncio.sleep(.02)
            attention = host.state['attention']
            await attention.attend({'revisit_after_s':3})
            session.emit('response_started', response_id='still-speaking')
            # Advance the clock only once the driver holds that response open (a loaded machine raced this, 28.10).
            await until(lambda: ('start', 'still-speaking') in host.usage)
            now[0] = 10
            await asyncio.sleep(.04)
            assert not session.sent and not host.context
            session.emit('response_done', response_id='still-speaking', usage={}, raw_usage={}, status='completed', continues=False)
            await until(lambda: any(isinstance(c, ContextUpdate) for c in session.sent))
            await attention.attend({'revisit_after_s':100})
        finally:
            host.hub.ended_by_owner = True
            await asyncio.wait_for(task, 2)
            assert attention.revisit_at is None
    asyncio.run(scenario())


def test_owner_turn_during_saved_look_defers_the_pending_offer_without_losing_its_frame(tmp_path):
    async def scenario():
        host = Host(); host.model = replace(host.model, vision=False)
        call = _Call(host); call.session = Session(); call.connected.set()
        async def retain(data, mime, *, source=''):
            call.input_open = True
            path = tmp_path / 'frame.jpg'; path.write_bytes(data); return str(path)
        host.retain_frame = retain
        call.attention.frame(cp.PeerVideo('source', cp.PEER_OWNER, 'image/jpeg', b'exact'))
        await call.offer_fact('This is the look you asked for.', requested_look=True)
        await call.offer_pending()
        assert call.pending_offers and not call.response_open
        assert not call.session.sent
        call.input_open = False
        await call.offer_pending()
        assert not call.pending_offers
        assert len(call.session.sent) == 1 and isinstance(call.session.sent[0], ContextUpdate)
        assert 'frame.jpg' in str(call.session.sent[0].messages)
        assert 'look you asked' in str(call.session.sent[0].messages)
        call.input_open = False
        assert call.attention.revisit_at is None
    asyncio.run(scenario())


@pytest.mark.parametrize('value', [-1, True, float('inf'), float('nan'), 'later'])
def test_invalid_duration_never_arms_a_host_fallback(value):
    async def scenario():
        attention = module.Attention(Host())
        with pytest.raises(ValueError):
            await attention.attend({'revisit_after_s':value, 'see':'stream'})
        assert attention.revisit_at is None and attention.mode == 'turns'
        await attention.attend({'revisit_after_s':None})
        assert attention.revisit_at is None
    asyncio.run(scenario())
