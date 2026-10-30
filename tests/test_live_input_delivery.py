"""Accepted microphone input survives consumer lag and connection setup."""
import asyncio
import json
import struct
from types import SimpleNamespace

import pytest

from ouroboros import call_protocol as cp
from ouroboros.call_peers import CallPeerError, LiveCallHub
from ouroboros.contracts.live_conversation import Audio, LiveOptions
from ouroboros.gateway.call_ws import serve_peer
from tests.test_live_conversation import host_world as host_world


@pytest.mark.parametrize('delay', [0, .04])
def test_fifty_frame_capture_reaches_slow_provider_in_order(delay):
    from skills.live_conversation.driver import _Call
    async def run():
        hub = LiveCallHub(cp.CallSpec('input-delivery', 1, 'fixture', 'voice'))
        host = SimpleNamespace(hub=hub, state={}, renewal_due=lambda: False, context_view_pending=lambda: False)
        call = _Call(host)
        call.input_rate = 24000
        received, incoming = [], asyncio.Queue()
        class Socket:
            async def accept(self): pass
            async def receive(self): return await incoming.get()
            async def send_text(self, text): pass
            async def send_bytes(self, data): pass
            async def close(self, **kwargs):
                incoming.put_nowait({'type': 'websocket.disconnect'})
        async def send(command):
            assert isinstance(command, Audio)
            await asyncio.sleep(delay)
            received.append(command.pcm)
        call.session = SimpleNamespace(send=send)
        incoming.put_nowait({'type': 'websocket.receive', 'text': cp.encode_control({
            'type': 'hello', 'protocol': 1, 'role': cp.PEER_OWNER, 'mic': True, 'speaker': True, 'rate': 24000})})
        socket = asyncio.create_task(serve_peer(Socket(), hub, role=cp.PEER_OWNER))
        consumer = asyncio.create_task(call.peers())
        expected = [i.to_bytes(2, 'little') * 480 for i in range(1, 121)]
        try:
            clock = asyncio.get_running_loop()
            start = clock.time()
            for seq, pcm in enumerate(expected):
                await asyncio.sleep(max(0, start + seq / 50 - clock.time()))
                incoming.put_nowait({'type': 'websocket.receive', 'bytes': cp.encode_frame(
                    cp.AudioFrame(cp.FRAME_AUDIO_IN, hub.epoch, seq, 24000, 0, pcm))})
            deadline = clock.time() + 10
            while len(received) < len(expected) and clock.time() < deadline:
                await asyncio.sleep(.01)
            assert received == expected, f'{len(received)}/{len(expected)} frames delivered'
            assert hub.input_audio_evidence()['dropped_samples'] == 0
        finally:
            hub.close('test_finished')
            await asyncio.wait_for(asyncio.gather(socket, consumer), 30)
    asyncio.run(run())


def audio(hub, seq, samples=137, epoch=None):
    return cp.AudioFrame(cp.FRAME_AUDIO_IN, hub.epoch if epoch is None else epoch,
                         seq, 24000, 0, b'\x01\x00' * samples)


def test_rejected_input_and_pending_close_have_distinct_drop_causes():
    hub = LiveCallHub(cp.CallSpec('drop-causes', 1, 'fixture', 'voice'))
    hub.join(cp.PeerInfo('owner', cp.PEER_OWNER, True, True), 24000)
    hub.next_event(0)
    assert hub.receive_audio('owner', audio(hub, 1))
    hub.flush()
    hub.flush()  # two generations back is genuinely stale input
    assert not hub.receive_audio('owner', audio(hub, 2, epoch=0))
    assert hub.receive_audio('owner', audio(hub, 3))
    assert not hub.receive_audio('owner', audio(hub, 3))
    hub.receive_control('owner', {'type': 'mute', 'muted': True})
    assert not hub.receive_audio('owner', audio(hub, 4))
    hub.receive_control('owner', {'type': 'mute', 'muted': False})
    assert hub.receive_audio('owner', audio(hub, 5))
    hub.close('owner_stop')
    evidence = hub.input_audio_evidence()
    assert {name: row['samples'] for name, row in evidence['drop_causes'].items()} == {
        'stale_epoch': 137, 'sequence': 137, 'muted': 411, 'call_closed': 137}
    assert evidence['dropped_samples'] == 822
    assert evidence['dropped_ms'] == pytest.approx(822 * 1000 / 24000)


@pytest.mark.parametrize('frames', [1, 10, 30])
@pytest.mark.parametrize('epoch', [0, 65535])
def test_barge_in_keeps_inflight_microphone_frames_and_retires_old_output(frames, epoch):
    """Audit epoch-repro: 20–600 ms in flight belongs to the new owner turn."""
    hub = LiveCallHub(cp.CallSpec('barge-in', 1, 'fixture', 'voice'))
    hub._epoch = epoch
    hub.join(cp.PeerInfo('owner', cp.PEER_OWNER, True, True), 24000)
    hub.set_engine_rates(24000, 24000)
    hub.next_event(0)
    hub.play(b'\x02\x00' * 480, 'old-answer')
    hub.flush()
    for seq in range(frames):
        assert hub.receive_audio('owner', audio(hub, seq, samples=480, epoch=epoch))
    assert hub.receive_audio('owner', audio(hub, frames, samples=480))
    received = [hub.next_event(.1).pcm for _ in range(frames + 1)]
    assert received == [b'\x01\x00' * 480] * (frames + 1)
    while (outgoing := hub.next_outgoing('owner')) is not None:
        assert isinstance(outgoing, str), 'flush must still retire queued output audio'
    assert hub.input_audio_evidence()['dropped_samples'] == 0
    assert not hub.receive_audio('owner', audio(hub, frames, epoch=epoch))  # duplicate
    assert not hub.receive_audio('owner', audio(hub, frames - 1, epoch=epoch))  # reordered
    assert not hub.receive_audio('owner', audio(hub, frames + 1, epoch=(hub.epoch + 1) & 0xFFFF))
    hub.flush()
    assert not hub.receive_audio('owner', audio(hub, frames + 1, epoch=epoch))
    assert hub.receive_audio('owner', audio(hub, frames + 1))
    hub.close('test_finished')


@pytest.mark.parametrize('leave', [False, True])
def test_accepted_input_survives_peer_leave_and_engine_rate_change(leave):
    hub = LiveCallHub(cp.CallSpec('retained-input', 1, 'fixture', 'voice'))
    hub.join(cp.PeerInfo('owner', cp.PEER_OWNER, True, True), 24000)
    hub.next_event(0)
    assert hub.receive_audio('owner', audio(hub, 0, samples=480))
    if leave:
        hub.leave('owner')
        assert isinstance(hub.next_event(0), cp.PeerLeft)
    hub.set_engine_rates(16000, 24000)
    event = hub.next_event(.1)
    assert isinstance(event, cp.PeerAudio) and event.pcm == b'\x01\x00' * 320
    assert hub.input_audio_evidence()['dropped_samples'] == 0
    hub.close('test_finished')


def test_call_setup_holds_audio_until_provider_ready(host_world):
    host, hub = host_world.host, host_world.host.hub
    hub.join(cp.PeerInfo('owner', cp.PEER_OWNER, True, True), 24000)
    hub.next_event(0)
    assert hub.receive_audio('owner', audio(hub, 0, samples=480))
    async def consume(session):
        await asyncio.sleep(.03)
        assert hub.next_event(0) is None, 'setup consumed audio before the provider declared its rates'
        hub.set_engine_rates(24000, 24000)
        event = hub.next_event(.1)
        assert isinstance(event, cp.PeerAudio) and event.pcm == b'\x01\x00' * 480
    asyncio.run(host.session(consume, LiveOptions()))


def test_capacity_wait_retries_same_frame_without_counting_it_as_a_drop():
    hub = LiveCallHub(cp.CallSpec('pressure', 1, 'fixture', 'voice'), audio_queue_frames=1)
    hub.join(cp.PeerInfo('owner', cp.PEER_OWNER, True, True), 24000)
    hub.next_event(0)
    assert hub.receive_audio('owner', audio(hub, 0, samples=480))
    with pytest.raises(CallPeerError, match='input_backpressure'):
        hub.receive_audio('owner', audio(hub, 1, samples=480))
    assert isinstance(hub.next_event(.1), cp.PeerAudio)
    assert hub.receive_audio('owner', audio(hub, 1, samples=480))
    assert isinstance(hub.next_event(.1), cp.PeerAudio)
    assert hub.input_audio_evidence()['dropped_samples'] == 0
    hub.close('test_finished')


def test_final_evidence_includes_input_discarded_by_close(host_world, monkeypatch):
    from ouroboros import call_peers
    host, hub = host_world.host, host_world.host.hub
    monkeypatch.setitem(call_peers._HUBS, host.turn.spec.call_id, hub)
    hub.join(cp.PeerInfo('owner', cp.PEER_OWNER, True, True), 24000)
    assert hub.receive_audio('owner', audio(hub, 0))
    host.turn.run(host.ctx, host.messages, host.tools, task=host.task, initial_effort='')
    events = [json.loads(json.loads(line)['raw']) for line in host.wire_path.read_text(encoding="utf-8").splitlines()]
    [evidence] = [event for event in events if event.get('type') == 'call.input_audio']
    assert evidence['drop_causes'] == {'call_closed': {'samples': 137, 'ms': 137 * 1000 / 24000}}
    assert hub.closed


def test_delivered_microphone_audio_never_waits_behind_the_mixer_tick(monkeypatch):
    """A delivery gap must not become lasting delay: behind one, the 27.10 OpenAI call lost 7.7 s of barge-in speech."""
    from ouroboros import call_peers
    clock = [100.0]
    monkeypatch.setattr(call_peers.time, "monotonic", lambda: clock[0])
    hub = LiveCallHub(cp.CallSpec('late-burst', 1, 'fixture', 'voice'))
    hub.join(cp.PeerInfo('owner', cp.PEER_OWNER, True, True), 24000)
    hub.set_engine_rates(24000, 24000)
    while hub.next_event(0) is not None:
        pass
    def drain():
        count = 0
        while (event := hub.next_event(0)) is not None:
            count += isinstance(event, cp.PeerAudio)
        return count
    sent = got = 0
    for seq in range(30):  # 600 ms of speech held up upstream, delivered together
        assert hub.receive_audio('owner', audio(hub, seq, samples=480))
        sent += 1
    got += drain()
    assert got >= sent - 1  # before the fix: 0 of 30
    for seq in range(30, 80):  # one more second at real time
        clock[0] += .02
        assert hub.receive_audio('owner', audio(hub, seq, samples=480))
        sent += 1
        got += drain()
    assert got >= sent - 1  # before the fix: 50 of 80, so 600 ms was still queued a second later
    hub.close('test_finished')


def test_mixer_tick_still_gathers_asynchronous_microphones(monkeypatch):
    """Control for the catch-up: two peers whose frames arrive within one tick still mix into one chunk."""
    from ouroboros import call_peers
    clock = [100.0]
    monkeypatch.setattr(call_peers.time, "monotonic", lambda: clock[0])
    hub = LiveCallHub(cp.CallSpec('tick', 1, 'fixture', 'voice'))
    for name, role in (('owner', cp.PEER_OWNER), ('guest', cp.PEER_MEETING)):
        hub.join(cp.PeerInfo(name, role, True, True), 24000)
    hub.set_engine_rates(24000, 24000)
    while hub.next_event(0) is not None:
        pass
    pcm = lambda value: struct.pack('<480h', *[value] * 480)
    assert hub.receive_audio('owner', cp.AudioFrame(cp.FRAME_AUDIO_IN, hub.epoch, 0, 24000, 0, pcm(1000)))
    assert hub.next_event(0) is None
    clock[0] += .005
    assert hub.receive_audio('guest', cp.AudioFrame(cp.FRAME_AUDIO_IN, hub.epoch, 0, 24000, 0, pcm(2000)))
    assert hub.next_event(0) is None
    clock[0] += .016
    assert struct.unpack('<480h', hub.next_event(0).pcm) == (3000,) * 480
    assert hub.next_event(0) is None
    hub.close('test_finished')
