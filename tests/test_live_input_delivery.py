"""Accepted microphone input survives consumer lag, connection setup and a planned switch."""
import asyncio
import json
import struct
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from ouroboros import call_protocol as cp
from ouroboros.call_peers import CallPeerError, LiveCallHub
from ouroboros.contracts.live_conversation import Audio, LiveEvent, LiveOptions
from ouroboros.contracts.realtime import RealtimeSetupError
from ouroboros.gateway.call_ws import serve_peer
from ouroboros.realtime_session import JsonSession
from tests._live_world import FakeSession
from tests._live_world import host_world as host_world
from tests.test_live_conversation_driver import Batch, Host, Session, started, until


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


@pytest.mark.parametrize(("epoch", "frames"), [(0, 1), (0, 30), (65535, 10)])
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


@pytest.mark.parametrize('opening', ['sequential', 'prepared'])
def test_call_setup_holds_audio_until_provider_ready(host_world, opening):
    host, hub = host_world.host, host_world.host.hub
    hub.join(cp.PeerInfo('owner', cp.PEER_OWNER, True, True), 24000)
    hub.next_event(0)
    words = [value.to_bytes(2, 'little') * 480 for value in (1, 2, 3)]
    def accept(seq):
        assert hub.receive_audio('owner', cp.AudioFrame(cp.FRAME_AUDIO_IN, hub.epoch, seq, 24000, 0, words[seq]))
    async def consume(session):
        await asyncio.sleep(.03)  # longer than the mixer's tick
        assert hub.next_event(0) is None, 'setup consumed audio before the provider declared its rates'
        hub.set_engine_rates(24000, 24000)
        held = words[1:] if opening == 'prepared' else words
        assert [hub.next_event(.1).pcm for _ in held] == held
    async def run():
        if opening == 'sequential':
            for seq in range(3):
                accept(seq)
            return await host.session(consume, LiveOptions())
        lease = await host.prepare_session(LiveOptions())
        accept(0)
        served = hub.next_event(.1)
        assert isinstance(served, cp.PeerAudio) and served.pcm == words[0], (
            'preparing a successor withheld audio from the connection still serving')
        accept(1), accept(2)
        await lease.run(consume)
        await lease.task
    asyncio.run(run())


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


class Voice(FakeSession):
    """The fixture engine with a scripted reader and a recording writer whose audio write takes finite time."""
    declares_at_open = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sent, self.events = [], None

    async def open(self):
        await super().open()
        self.events = asyncio.Queue()
        if self.declares_at_open:
            self.emit('ready', input_rate=24000, output_rate=24000)

    async def receive(self):
        return await self.events.get()

    async def send(self, command):
        self.sent.append(command)
        if isinstance(command, Audio):
            await asyncio.sleep(.001)

    def final_captions(self):
        return []

    def emit(self, kind, **data):
        self.events.put_nowait([LiveEvent(kind, data)])

    def heard(self):
        return b''.join(command.pcm for command in self.sent if isinstance(command, Audio))


@pytest.mark.parametrize('handoff', ['sequential', 'ready_buffered', 'ready_late', 'prepared_stale',
                                     'ended_before_ready'])
def test_a_microphone_ahead_of_the_provider_loses_nothing_across_a_renewal(host_world, handoff):
    """The real host and driver over a scripted engine. The microphone stays ahead of the provider's finite
    writes, so the hub holds more than one mixed chunk whenever a connection changes. Every accepted sample then
    reaches exactly one connection, in order and after that connection's Ready, or is counted when the call closes."""
    from skills.live_conversation import driver
    host, hub = host_world.host, host_world.host.hub
    opened = []

    class Engine(Voice):
        silent_context_updates = handoff != 'sequential'  # an engine that cannot overlap renews sequentially

        async def open(self):
            self.declares_at_open = not opened or handoff not in ('ready_late', 'ended_before_ready')
            opened.append(self)
            await super().open()

    host_world.provider['session'] = Engine
    hub.join(cp.PeerInfo('owner', cp.PEER_OWNER, True, True), 24000)

    async def scenario():
        accepted, capturing, clock = [], asyncio.Event(), asyncio.get_running_loop()
        capturing.set()

        async def microphone():
            while capturing.is_set():
                pcm = (len(accepted) % 60000 + 1).to_bytes(2, 'little') * 480
                try:
                    assert hub.receive_audio('owner', cp.AudioFrame(cp.FRAME_AUDIO_IN, hub.epoch, len(accepted),
                                                                   24000, 0, pcm))
                except CallPeerError as exc:  # the hub is full: its socket keeps the frame and retries
                    assert exc.code == 'input_backpressure'
                    await asyncio.sleep(.002)
                    continue
                accepted.append(pcm)
                await asyncio.sleep(0)

        async def reached(predicate):  # a hang detector, not a latency claim: every fit runs off this loop
            deadline = clock.time() + 60
            while not predicate():
                assert not call.done(), ('the call ended before the scenario boundary', call)
                assert clock.time() < deadline, ('boundary timed out', len(opened), host.connection_number)
                await asyncio.sleep(.005)

        call = asyncio.create_task(driver.run(host))
        mic = asyncio.create_task(microphone())
        try:
            await reached(lambda: opened and len(opened[0].heard()) >= 20 * 960)
            first, connections = opened[0], 3 if handoff == 'prepared_stale' else 2
            first.renewal_ready = handoff != 'prepared_stale'
            first.emit('renew', within_s=30, resume=False)
            if handoff == 'prepared_stale':  # an authored edit lands after the successor sealed its prefix
                await reached(lambda: host._prepared_successor is not None)
                host.messages[0]['content'] += '\nAn authored edit of the fitted prefix.'
                host._context_changed()
                first.renewal_ready = True
            await reached(lambda: len(opened) == connections and host.connection_number == connections)
            last = opened[-1]
            if not last.declares_at_open:
                await asyncio.sleep(.05)  # the adopted consumer already polls; its Ready is still on the way
            if handoff == 'ended_before_ready':
                capturing.clear()
                await mic
                hub.receive_control('owner', {'type': 'bye'})
                await asyncio.wait_for(call, 30)
                hub.close('owner_end')
                delivered, stream = b''.join(session.heard() for session in opened), b''.join(accepted)
                assert not last.heard() and delivered == stream[:len(delivered)]
                counted = {cause: row['samples'] for cause, row in hub.input_audio_evidence()['drop_causes'].items()}
                assert counted == {'call_closed': (len(stream) - len(delivered)) // 2}, (
                    'accepted audio is delivered or counted, never taken by a connection that cannot send it')
                return
            if not last.declares_at_open:
                assert not last.heard()
                last.emit('ready', input_rate=24000, output_rate=24000)
            await reached(lambda: len(last.heard()) >= 20 * 960)
            capturing.clear()
            await mic
            await reached(lambda: last.heard().endswith(accepted[-1]))
            delivered, stream = b''.join(session.heard() for session in opened), b''.join(accepted)
            assert delivered == stream, f'{len(delivered) // 960} of {len(accepted)} accepted frames reached a connection'
            assert hub.input_audio_evidence()['dropped_samples'] == 0
            assert all(not session.heard() for session in opened[1:-1]), 'a stale successor serves nothing'
        finally:
            capturing.clear()
            if not hub.closed:
                hub.receive_control('owner', {'type': 'bye'})
            await asyncio.gather(mic, call, *host._connection_tasks, return_exceptions=True)
    asyncio.run(scenario())


class Socket:
    """A websocket that takes a frame onto its wire at once, as a transport's write does, and then drains."""

    def __init__(self):
        self.frames, self.drained = [], asyncio.Event()
        self.drained.set()

    async def send(self, raw):
        self.frames.append(json.loads(raw))
        await self.drained.wait()


class Wire(Session):
    """The driver's scripted session over the real JsonSession.write, its send lock included."""

    def __init__(self):
        super().__init__()
        self.socket = Socket()
        self.transport = JsonSession(SimpleNamespace(messages_json='[]'), None, connect=None, observe=lambda *_: None)
        self.transport.ws = self.socket

    async def send(self, command):
        self.sent.append(command)
        if isinstance(command, Audio):
            await self.transport.write({'audio': command.pcm.hex()})


@asynccontextmanager
async def a_result_awaiting_its_fit_behind_a_held_write(monkeypatch):
    """A call whose microphone chunk has left the hub and waits for the transport's send lock behind another
    writer, when a finished tool result asks for a fresh fit."""
    from skills.live_conversation import driver
    calls, wires, host = [], [], Host()

    class Call(driver._Call):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.peer_exits = 0
            calls.append(self)

        async def peers(self):
            try:
                return await super().peers()
            finally:
                self.peer_exits += 1

    async def session(consume, options):
        host.due = False
        wires.append(Wire())
        host.sessions.append(wires[-1])
        return await consume(wires[-1])

    async def batch(requested):
        host.due = True  # the finished result no longer fits the serving connection
        return Batch([{'role': 'tool', 'tool_call_id': call['id'], 'content': 'Done once.'} for call in requested])

    monkeypatch.setattr(driver, '_Call', Call)
    host.session, host.tool_batch = session, batch
    task, first = await started(host)
    chunk, other = b'\x07\x00' * 480, None
    try:
        first.emit('ready', input_rate=24000, output_rate=24000)
        await until(lambda: calls[0].connected.is_set())
        first.socket.drained.clear()
        other = asyncio.create_task(first.transport.write({'other': 1}))  # holds the lock while its frame drains
        await until(lambda: first.socket.frames == [{'other': 1}])
        host.hub.events.append(cp.PeerAudio('owner', cp.PEER_OWNER, chunk))
        await until(lambda: any(isinstance(command, Audio) for command in first.sent))  # off the hub, at the lock
        first.emit('tool_call', call_id='work', name='effect', arguments='{}')
        await until(lambda: calls[0].tool_refit.is_set() or len(wires) > 1)
        yield SimpleNamespace(host=host, call=calls[0], wires=wires, first=first, task=task, chunk=chunk, other=other)
    finally:
        for wire in wires:
            wire.socket.drained.set()
        host.hub.ended_by_owner = True
        await asyncio.wait_for(asyncio.gather(task, *([other] if other else []), return_exceptions=True), 2)


@pytest.mark.parametrize('owner_speaks', ['never', 'during_the_write', 'during_the_row_join'])
def test_a_planned_tool_refit_never_cancels_a_microphone_write_in_flight(monkeypatch, owner_speaks):
    async def scenario():
        async with a_result_awaiting_its_fit_behind_a_held_write(monkeypatch) as live:
            host, call, first, wires = live.host, live.call, live.first, live.wires
            def reconnecting():
                return any(state[0] == cp.STATE_RECONNECTING for state in host.hub.states)
            await asyncio.sleep(.25)
            assert len(wires) == 1 and not reconnecting(), 'the planned switch did not wait for the write in flight'
            saved, entered = asyncio.Event(), asyncio.Event()
            if owner_speaks == 'during_the_write':
                first.emit('input_started')
                await until(lambda: call.input_open)
            elif owner_speaks == 'during_the_row_join':
                save = host.caption
                async def held_caption(**row):
                    entered.set()
                    await saved.wait()
                    await save(**row)
                host.caption = held_caption
                first.emit('caption', speaker='owner', item_id='earlier', text='Earlier words.', final=True)
                await asyncio.wait_for(entered.wait(), 2)
            first.socket.drained.set()
            await live.other
            on_the_wire = [{'other': 1}, {'audio': live.chunk.hex()}]
            await until(lambda: first.socket.frames == on_the_wire)
            if owner_speaks == 'during_the_row_join':
                await until(lambda: call.peer_exits == 1)  # the switch is decided; joining its rows yields
                first.emit('input_started')
                await until(lambda: call.input_open)
                saved.set()
            if owner_speaks != 'never':
                await asyncio.sleep(.1)
                assert len(wires) == 1 and not reconnecting(), 'new native input keeps its reader and the page capturing'
                later = b'\x08\x00' * 480
                host.hub.events.append(cp.PeerAudio('owner', cp.PEER_OWNER, later))
                on_the_wire.append({'audio': later.hex()})
                await until(lambda: first.socket.frames == on_the_wire)
                first.emit('input_done')
            await until(lambda: len(wires) == 2)
            assert reconnecting()
            assert first.socket.frames == on_the_wire and not wires[1].socket.frames, 'each chunk is on exactly one wire'
    asyncio.run(scenario())


@pytest.mark.parametrize('ending', ['stop', 'provider_refusal', 'transport_loss'])
def test_an_urgent_end_never_waits_for_a_held_microphone_write(monkeypatch, ending):
    """Stop and a provider's failure stay able to cancel. The cancelled write's outcome is then unknown: nothing
    replays it, and the call's closing count does not claim it."""
    async def scenario():
        async with a_result_awaiting_its_fit_behind_a_held_write(monkeypatch) as live:
            first, wires, task = live.first, live.wires, live.task
            if ending == 'stop':  # the host's control watcher cancels and joins the feature
                task.cancel()
                await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 1)
            elif ending == 'provider_refusal':
                first.emit('closed', resumable=False, detail='synthetic refusal')
                with pytest.raises(RealtimeSetupError, match='provider_closed'):
                    await asyncio.wait_for(task, 1)
            else:
                first.emit('closed', transport_loss=True)
                await until(lambda: len(wires) == 2)
            assert not live.other.done(), 'nothing waited for the held send lock'
            assert first.socket.frames == [{'other': 1}] and not any(wire.socket.frames for wire in wires[1:])
    asyncio.run(scenario())
