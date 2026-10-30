"""One call.timing fact spans the real replacement boundaries, never an idle policy."""
import asyncio

from ouroboros.contracts.live_conversation import LiveOptions
from tests.test_live_conversation import host_world as host_world
from tests.test_live_host_fit_evidence import host_events
from tests.test_live_conversation_driver import Host, started, until


def test_host_collects_one_renewal_fact_and_keeps_unknown_audio_unknown(host_world):
    host = host_world.host
    async def consume(session):
        host.timing_event('first_audio', at=103)  # a late old-session boundary is not replacement audio
        assert not (host.wire_path.exists() and [r for r in host_events(host) if r.get('renewal')])
        host.timing_event('ready', at=104)
        host.timing_event('first_audio', at=105)
        host.timing_event('first_audio', at=106)
    host.timing_event('requested', at=100)
    host.timing_event('first_audio', at=101)
    host.timing_event('ready', at=101)  # no replacement fit yet: this is still the retiring session
    assert not (host.wire_path.exists() and [r for r in host_events(host) if r.get('renewal')])
    host.timing_event('quiet', at=102)
    asyncio.run(host.session(consume, LiveOptions()))
    records = [r for r in host_events(host) if r.get('type') == 'call.timing' and r.get('renewal')]
    assert len(records) == 1
    fact = records[0]
    assert fact['requested_at'] == 100 and fact['quiet_at'] == 102
    assert fact['fit_done_at'] is not None and fact['provider_ready_at'] == 104 and fact['first_audio_at'] == 105
    host.timing_event('first_audio', at=107)
    host.timing_event('requested', at=110)
    host.timing_event('quiet', at=111)
    host.timing_event('ready', at=112)
    host.timing_event('requested', at=115)  # silent session ends; retain the missing boundary as unknown
    records = [r for r in host_events(host) if r.get('type') == 'call.timing' and r.get('renewal')]
    assert len(records) == 2 and records[-1]['first_audio_at'] is None
    host.timing_event('closed', at=116)
    host.timing_event('closed', at=117)
    records = [r for r in host_events(host) if r.get('type') == 'call.timing' and r.get('renewal')]
    assert len(records) == 3


def test_driver_records_requested_quiet_ready_audio_once_and_not_initial_open():
    async def scenario():
        host = Host()
        seen = []
        def record(phase, **kw):
            if phase == "first_audio":
                assert host.hub.audio, "timing cannot precede the first queued output"
            seen.append(phase)
        host.timing_event = record
        host.state['greeted'] = True
        task, first = await started(host)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            await until(lambda: 'ready' in seen)
            assert 'requested' not in seen and 'quiet' not in seen
            first.emit('input_started')
            host.due = True
            await until(lambda: 'requested' in seen)
            await asyncio.sleep(.03)
            assert seen.count('requested') == 1 and 'quiet' not in seen
            first.emit('input_done')
            await until(lambda: len(host.sessions) == 2)
            assert seen.count('quiet') == 1
            second = host.sessions[1]
            second.emit('ready', input_rate=24000, output_rate=24000)
            second.emit('audio', item_id='first-new-audio', rate=24000, pcm=b'\0\0')
            await until(lambda: 'first_audio' in seen)
            assert seen.index('requested') < seen.index('quiet') < seen.index('first_audio')
        finally:
            host.hub.ended_by_owner = True
            await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_offer_wait_is_one_end_fact_with_overlap_counted_only_once(host_world):
    host = host_world.host
    host.timing_event('offer_idle', at=0)
    host.timing_event('offer_wait:input_open,session.input_pending', at=10)
    host.timing_event('offer_wait:session.output_pending', at=12)
    host.timing_event('offer_idle', at=19)
    host.timing_event('offer_wait:unheard', at=30)
    assert not host.wire_path.exists(), 'no per-poll or per-interval journal records'
    host.timing_event('offer_wait_done', at=33)
    host.timing_event('offer_wait_done', at=34)
    assert [r for r in host_events(host) if r.get('type') == 'call.offer_wait'] == [{
        'type': 'call.offer_wait', 'pending_seconds': 12.0, 'longest_blocker': 'session.output_pending',
        'seconds_by_blocker': {'input_open': 2.0, 'session.input_pending': 2.0,
                              'session.output_pending': 7.0, 'unheard': 3.0}}]


def test_driver_only_times_an_offer_or_due_revisit_and_records_end():
    from skills.live_conversation.driver import _Call
    from tests.test_live_conversation_driver import Session
    host = Host()
    phases = []
    host.timing_event = phases.append
    call = _Call(host)
    call.session = Session()
    call.input_open = True
    call.track_offer_wait()
    assert phases == ['offer_idle'], 'ordinary listening with no offer is not waiting for the floor'
    call.pending_offers.append({'role': 'user', 'content': 'Synthetic host fact'})
    call.track_offer_wait(); call.track_offer_wait()
    assert phases == ['offer_idle', 'offer_wait:input_open'], 'polling records boundaries, not one event per tick'
    call.pending_offers.clear()
    call.track_offer_wait()
    assert phases[-1] == 'offer_idle'
