"""Accepted microphone input survives consumer lag and connection setup."""
import asyncio
import json
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
        host = SimpleNamespace(hub=hub, state={}, renewal_due=lambda: False)
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
