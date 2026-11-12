"""Overlapping fresh sessions over the real gateway, host, driver and native adapters."""
import asyncio
import json

import pytest

from tests._extension_loader_shared import _clear_loader_state  # noqa: F401
from tests._live_stack import Stack, stack
from tests._live_stack_page import Page
from tests._live_stack_wire import NativeWire
from tests._live_world import host_world as host_world
from tests._cases import chosen_rows

ROUTES = [('openai', False), ('grok', False), ('gemini', False), ('gemini', True)]


async def request(run):
    async def renew():
        run.call.request_renewal()
    await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(renew(), run.call.writer.get_loop()))


@pytest.mark.parametrize('engine,thinking', chosen_rows(ROUTES, 0, 1, 3))
def test_successor_opens_while_owner_speaks_and_catches_up_before_answer(host_world, monkeypatch, engine, thinking):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        original = NativeWire.send
        async def gate(wire, raw):
            event = json.loads(raw)
            if wire.number == 2 and ('setup' in event or event.get('type') == 'session.update'):
                entered.set()
                await release.wait()
            return await original(wire, raw)
        monkeypatch.setattr(NativeWire, 'send', gate)
        async with stack(host_world, monkeypatch, engine, thinking=thinking) as run:
            old = run.wire
            old.input_start('overlap-owner')
            await run.wait(lambda: run.call.input_open)
            await request(run)
            await asyncio.wait_for(entered.wait(), 15)
            assert run.host.connection_number == 1, 'preparation may not publish the next fit or connection'
            count = len(old.audio)
            await run.wait(lambda: len(old.audio) > count + 960)
            assert not run.wire.audio and not run.wire.spoken, 'prepared socket receives no media or response'
            # Both a caption and a host fact arrive AFTER the successor's sealed setup.
            run.page.share(True, 'screen')
            await run.wait(lambda: any('started showing the screen' in str(r) for r in run.host.messages))
            run.provider.script.append(('say', 'Answer on the still-serving connection.'))
            old.input_end('overlap-owner', 'Owner words during successor setup.')
            await run.wait(lambda: any('Owner words during successor setup.' in str(r) for r in run.host.messages))
            release.set()
            await run.wait(lambda: run.host.connection_number == 2 and run.call.connected.is_set())
            await run.wait(lambda: bool(run.wire.spoken))
            await run.quiet()
            new = run.wire
            await run.wait(lambda: old.closed)
            assert len(run.provider.connections) == 2
            evidence = run.host._prepared_view[0]
            fit = evidence['fit']['measurement']
            assert evidence['work_room']['remaining_tokens'] == (fit['capacity_total_tokens']
                - fit['estimated_input_tokens'] - fit['response_reserve_tokens'])
            assert any('Owner words during successor setup.' in str(v) for v in new.sent)
            assert sum('started showing the screen' in json.dumps(v) for v in new.sent) == 1
            assert len([s for w in run.provider.connections for s in w.spoken if s == 'Synthetic greeting heard.']) == 1
            assert sum('Owner words during successor setup.' in str(r) for r in run.transcript()) == 1
            run.host.wire.flush()
            facts = [json.loads(r['raw']) for r in run.journal() if r['direction'] == 'host']
            timing = next(f for f in facts if f.get('type') == 'call.timing' and f.get('first_audio_at'))
            assert timing['fit_done_at'] <= timing['prepared_ready_at'] <= timing['switched_at'] <= timing['first_audio_at']
            # No setup starts in the switching gap: it finished before the quiet handoff.
            assert timing['prepared_ready_at'] <= timing['quiet_at'] <= timing['switched_at']
        assert run.result[1]['execution_status'] == 'ok', run.result
        assert all(w.closed for w in run.provider.connections)
    asyncio.run(scenario())


class ExactPage(Page):
    """A 24 kHz microphone whose every 20 ms frame is distinct, so a 24 kHz engine's wire compares byte for byte.
    It never pauses, as frames already in flight do not: what the hub is handed while a connection changes is
    the subject here, not the page's own capture rule."""

    def __init__(self):
        super().__init__(24000)
        self.recording = True

    def capturing(self):
        return self.recording

    def microphone(self):
        return (self.seq % 60000 + 1).to_bytes(2, 'little') * self.chunk


@pytest.mark.parametrize('engine', ['openai', 'grok'])
def test_accepted_microphone_bytes_cross_a_prepared_handoff_exactly(host_world, monkeypatch, engine):
    """Host, driver, gateway and native adapter over a loopback wire; the page and the provider are Python doubles.
    Both engines take 24 kHz PCM, so every accepted sample must be on one provider socket, in order: the old one,
    or its successor once that declared Ready."""
    from ouroboros.call_peers import LiveCallHub
    from ouroboros.live_connections import PreparedConnection
    from tests import _live_stack
    accepted = bytearray()
    receive, adopt = LiveCallHub.receive_audio, PreparedConnection._adopt

    def accepting(hub, peer_id, frame):
        taken = receive(hub, peer_id, frame)
        if taken:
            accepted.extend(frame.pcm)
        return taken

    def buffered(hub):
        with hub._condition:
            return sum(len(track.samples) for peer in hub._audio_peers() for track in peer.tracks.values())

    async def adopt_behind_a_backlog(lease):
        if lease.ahead:  # more than one mixed chunk waits in the hub when the successor is adopted
            async def backlog():
                while buffered(lease.host.hub) < 3 * 480:
                    await asyncio.sleep(.005)
            await asyncio.wait_for(backlog(), 60)
        return await adopt(lease)

    monkeypatch.setattr(LiveCallHub, 'receive_audio', accepting)
    monkeypatch.setattr(PreparedConnection, '_adopt', adopt_behind_a_backlog)
    monkeypatch.setattr(_live_stack, 'Page', ExactPage)

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as run:
            first = run.wire
            await run.wait(lambda: len(first.audio) >= 9600)
            await request(run)
            await run.wait(lambda: run.host.connection_number == 2 and run.call.connected.is_set())
            second = run.wire
            await run.wait(lambda: len(second.audio) >= 9600)
            run.page.recording = False
            await run.wait(lambda: bytes(second.audio[-960:]) == bytes(accepted[-960:]))
            assert first is not second and len(run.provider.connections) == 2
            delivered = bytes(first.audio) + bytes(second.audio)
            assert delivered == bytes(accepted), (
                f'{len(delivered)} of {len(accepted)} accepted bytes reached a provider socket')
            assert run.host.hub.input_audio_evidence()['dropped_samples'] == 0
            assert not any('may not have been heard' in str(row) for row in run.host.messages)
        assert run.result[1]['execution_status'] == 'ok', run.result
    asyncio.run(scenario())


def test_owner_says_stays_on_the_serving_socket_when_a_successor_is_accepted(tmp_path):
    """The call serves its first socket until it is quiet, so an utterance that began there ends there, although
    `Stack.wire` (the latest accepted socket) names the prepared successor by the time the owner's audio arrives."""
    from types import SimpleNamespace

    class Wire:
        def __init__(self, number):
            self.number, self.audio, self.received, self.calls = number, bytearray(), [], []

        def input_start(self, item):
            self.calls.append(('start', item))

        def input_end(self, item, text):
            self.calls.append(('end', item, text))

    async def scenario():
        run = Stack(SimpleNamespace(env=SimpleNamespace(drive_root=tmp_path)), 'grok')
        serving, successor = Wire(1), Wire(2)
        run.provider.connections.append(serving)
        loop, started, ended = asyncio.get_running_loop(), serving.input_start, serving.input_end

        def owner_starts(item):
            started(item)
            loop.call_later(.02, run.provider.connections.append, successor)  # the renewal's successor is accepted first,
            loop.call_later(.1, serving.audio.extend, bytes(960))             # then the late audio reaches the serving socket

        def owner_ends(item, text):
            ended(item, text)
            run.page.controls.append({'text': text, 'final': True})

        serving.input_start, serving.input_end = owner_starts, owner_ends
        await asyncio.wait_for(run.owner_says('Words while a successor is prepared.'), 10)
        assert run.wire is successor, 'the successor was accepted during the utterance'
        assert serving.calls == [('start', 'owner-0'), ('end', 'owner-0', 'Words while a successor is prepared.')]
        assert successor.calls == [], 'the prepared successor serves nothing yet'
    asyncio.run(scenario())


@pytest.mark.parametrize('engine,thinking', chosen_rows(ROUTES, 0, 1, 3))
def test_failed_preparation_falls_back_to_one_sequential_fit(host_world, monkeypatch, engine, thinking):
    async def scenario():
        original = NativeWire.send
        async def fail_second(wire, raw):
            if wire.number == 2:
                await wire.websocket.close(code=1012, reason='synthetic setup restart')
                return
            return await original(wire, raw)
        monkeypatch.setattr(NativeWire, 'send', fail_second)
        async with stack(host_world, monkeypatch, engine, thinking=thinking) as run:
            await request(run)
            await run.wait(lambda: len(run.provider.connections) == 3 and run.host.connection_number == 3)
            await run.quiet()
            run.provider.script.append(('say', 'Conversation after failed preparation.'))
            await run.owner_says('Owner still here after setup restart.')
            await run.quiet()
            assert sum(s == 'Synthetic greeting heard.' for w in run.provider.connections for s in w.spoken) == 1
        assert run.result[1]['execution_status'] == 'ok', run.result
        assert all(w.closed for w in run.provider.connections)
    asyncio.run(scenario())


@pytest.mark.parametrize('ending', ['owner_end', 'owner_stop', 'panic', 'budget_exhausted'])
@pytest.mark.parametrize('engine,thinking', chosen_rows(ROUTES, 0, 2))
def test_owner_end_joins_preparation_and_closes_both_sockets(host_world, monkeypatch, engine, thinking, ending):
    async def scenario():
        entered = asyncio.Event()
        original = NativeWire.send
        async def block_second(wire, raw):
            if wire.number == 2:
                entered.set()
                await wire.websocket.wait_closed()
                return
            return await original(wire, raw)
        monkeypatch.setattr(NativeWire, 'send', block_second)
        async with stack(host_world, monkeypatch, engine, thinking=thinking) as run:
            await request(run)
            await asyncio.wait_for(entered.wait(), 15)
            if ending == 'owner_end':
                run.page.end()
            elif ending == 'budget_exhausted':
                from ouroboros import live_usage
                from ouroboros.usage_accounting import BudgetExceeded
                def exhausted(**kwargs):
                    raise BudgetExceeded('Synthetic exhausted shared budget')
                monkeypatch.setattr(live_usage, 'check_live_budget', exhausted)
            else:
                monkeypatch.setattr(run.host.ctx.model_wait_context, 'control_reason', lambda **kwargs: ending)
            await asyncio.wait_for(asyncio.shield(run.running), 15)
        if ending == 'owner_end':
            assert run.result[1]['execution_status'] == 'ok', run.result
        else:
            assert run.result[1]['reason_code'] == ending, run.result
        assert all(w.closed for w in run.provider.connections)
        assert all(t.done() for t in run.host._connection_tasks)
    asyncio.run(scenario())


@pytest.mark.parametrize('engine,thinking', chosen_rows(ROUTES, 0, 3))
@pytest.mark.parametrize('mail', [False, True])
def test_catchup_does_not_offer_its_own_speech_but_pending_mail_is_answered(host_world, monkeypatch, engine, thinking, mail):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        original = NativeWire.send
        async def gate(wire, raw):
            if wire.number == 2 and not entered.is_set():
                entered.set()
                await release.wait()
            return await original(wire, raw)
        monkeypatch.setattr(NativeWire, 'send', gate)
        async with stack(host_world, monkeypatch, engine, thinking=thinking) as run:
            old = run.wire
            old.input_start('owner-before-handoff')
            await run.wait(lambda: run.call.input_open)
            await request(run)
            await asyncio.wait_for(entered.wait(), 15)
            run.provider.script.append(('say', 'The old connection already answered the owner.'))
            old.input_end('owner-before-handoff', 'A question already answered before the handoff.')
            await run.wait(lambda: any('The old connection already answered the owner.' in str(r) for r in run.host.messages))
            if mail:
                run.mail.put({'text': 'New typed room request during preparation.', 'msg_id': 'during-prep'})
                await run.wait(lambda: any('New typed room request during preparation.' in str(r) for r in run.host.messages))
                run.provider.script.append(('say', 'Answer to the new room request.'))
            release.set()
            await run.wait(lambda: run.host.connection_number == 2)
            await run.quiet()
            if mail:
                await run.wait(lambda: 'Answer to the new room request.' in run.wire.spoken)
                await run.quiet()
                assert run.wire.spoken == ['Answer to the new room request.']
            else:
                assert not run.wire.spoken, 'catching up our own answered conversation never offers another turn'
            assert any('A question already answered before the handoff.' in str(v) for v in run.wire.sent)
        assert run.result[1]['execution_status'] == 'ok', run.result
    asyncio.run(scenario())


@pytest.mark.parametrize('engine,thinking', chosen_rows(ROUTES, 0, 1, 3))
def test_prepared_socket_lost_before_catchup_falls_back_without_losing_words(host_world, monkeypatch, engine, thinking):
    async def scenario():
        async with stack(host_world, monkeypatch, engine, thinking=thinking) as run:
            old = run.wire
            old.input_start('held-owner')
            await run.wait(lambda: run.call.input_open)
            await request(run)
            await run.wait(lambda: run.call.preparing.done())
            await run.wire.websocket.close(code=1012, reason='synthetic prepared transport loss')
            run.provider.script.append(('say', 'Answer before the prepared socket failed.'))
            old.input_end('held-owner', 'Exact words arriving after the prepared fit.')
            await run.wait(lambda: run.host.connection_number == 3)
            await run.quiet()
            assert any('Exact words arriving after the prepared fit.' in str(v) for v in run.wire.sent)
            assert sum('Exact words arriving after the prepared fit.' in str(r) for r in run.transcript()) == 1
        assert run.result[1]['execution_status'] == 'ok', run.result
    asyncio.run(scenario())


@pytest.mark.parametrize('thinking', [False, True])
def test_native_resume_after_a_prepared_handoff_still_waits_for_close(host_world, monkeypatch, thinking):
    import threading
    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini', thinking=thinking) as run:
            await request(run)
            await run.wait(lambda: run.host.connection_number == 2)
            await run.quiet()
            native = run.call.session
            close = native.close
            closing, release = threading.Event(), threading.Event()
            async def slow_close():
                closing.set()
                await asyncio.to_thread(release.wait)
                await close()
            monkeypatch.setattr(native, 'close', slow_close)
            try:
                run.wire.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': 'resume-after-preparation'}})
                await run.wait(lambda: run.host.state.get('resume_handle') == 'resume-after-preparation')
                run.wire.put({'goAway': {'timeLeft': '30s'}})
                await run.wait(closing.is_set)
                assert len(run.provider.connections) == 2, 'a native resume must not overlap its old connection'
            finally:
                release.set()
            await run.wait(lambda: run.host.connection_number == 3)
            await run.quiet()
            assert run.wire.sent[0]['setup']['sessionResumption'] == {'handle': 'resume-after-preparation'}
            assert not run.host.timing['connections'][-1]['prepared_ahead']
    asyncio.run(scenario())
