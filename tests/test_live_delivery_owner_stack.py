"""The whole call, on the real host and adapters: conversation goes on around a long effect, and what waits for a
connection has one sender. Only the provider's far end and the page are simulated; socket closes are real server closes
over loopback, ordered by a gate at the adapter's write boundary. The fake-port layer is ``test_live_delivery_owner.py``.
"""
import asyncio
import json
import threading
import time

import pytest

from ouroboros import call_protocol as cp
from ouroboros.contracts.realtime import RealtimeSetupError
from tests._extension_loader_shared import _clear_loader_state  # noqa: F401
from tests._live_world import host_world as host_world
from tests.test_live_tool_continuity import Held, POINTER, unit_at
from tests.test_live_tool_continuity_stack import answered_while_held, native_output, page_state

MAIL = 'Typed room mail waiting for its connection.'
ROOM_POINTER = 'New room context'
RESTART = dict(code=1012, reason='synthetic service restart')


def live_frames(wire, text):
    """What was written live: a fitted setup re-creating history is not a delivery."""
    return [frame for frame in wire.sent if text in json.dumps(frame) and ('clientContent' in frame or (
            frame.get('type') == 'conversation.item.create' and not str(frame['item'].get('id', '')).startswith('setup_item')))]


def written(live, text):
    return sum(len(live_frames(wire, text)) for wire in live.provider.connections)


def spoken(live):
    return sum(len(wire.spoken) for wire in live.provider.connections)


def position(wire, text):
    return next(i for i, frame in enumerate(wire.sent) if text in json.dumps(frame))


def handle(wire, value='covering'):
    wire.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': value}})


async def typed_and_answered(live, text, answer):
    live.provider.script.append(('say', answer))
    live.mail.put({'text': text, 'msg_id': text})
    await live.wait(lambda: any(v.get('text') == answer and v.get('final') for v in live.page.controls), 30)
    await spoken_out(live)


async def spoken_out(live):
    """The turn has finished and been heard. Gemini's ``quiet`` also waits for its own open call, so it cannot say this."""
    await live.wait(lambda: not live.call.response_open and not live.call.pending_spoken and not live.call.unheard(), 30)


def close_on_write(monkeypatch, live, matches, *, before=None, **close):
    """The first native write that ``matches`` finds its socket really closed by the server, so the real client's own
    write raises. The call runs on its own loop in its own thread; the server stays on the scenario's."""
    from ouroboros.realtime_session import JsonSession
    original, server, fired = JsonSession.write, asyncio.get_running_loop(), []

    async def write(session, payload, *, snapshot=None):
        if not fired and matches(payload):
            fired.append(payload)
            if before is not None:
                before()
            closing = asyncio.run_coroutine_threadsafe(live.wire.websocket.close(**close), server)
            await asyncio.wrap_future(closing)
        return await original(session, payload, snapshot=snapshot)
    monkeypatch.setattr(JsonSession, 'write', write)
    return fired


def refused_speculative_fits(monkeypatch):
    """A real route to the worker: the host's own ahead fit fails for capacity, so nothing new is on its way."""
    from ouroboros.live_connections import PreparedConnection
    fit = PreparedConnection._fit

    async def capacity_failure(lease, credentials, *, rebound=False):
        if lease.ahead:
            raise RealtimeSetupError('context_capacity', 'Synthetic failure of a speculative capacity fit')
        return await fit(lease, credentials, rebound=rebound)
    monkeypatch.setattr(PreparedConnection, '_fit', capacity_failure)


class Measures:
    """The host's real serving measure with a gate behind it for rows the scenario names. One renewal falls due on demand."""

    def __init__(self, monkeypatch, live, holding):
        self.count, self.tasks, self.open, self.verdict = 0, [], threading.Event(), None
        self.due = False
        measure, renewal = live.host.delivery_fits, live.host.renewal_due

        async def gated(rows, schemas):
            self.due = False
            admitted = await measure(rows, schemas)
            if holding in json.dumps(rows, default=str):
                self.count += 1
                self.tasks.append(asyncio.current_task().get_coro().__qualname__)
                while not self.open.is_set():
                    await asyncio.sleep(.001)
            return admitted if self.verdict is None else self.verdict
        monkeypatch.setattr(live.host, 'delivery_fits', gated)
        monkeypatch.setattr(live.host, 'renewal_due', lambda: self.due or renewal())

    def tool_makes_it_due(self, monkeypatch, live):
        batch = live.host.tool_batch

        async def due_at_the_boundary(calls):
            result = await batch(calls)
            self.due = True
            return result
        monkeypatch.setattr(live.host, 'tool_batch', due_at_the_boundary)


@pytest.mark.parametrize('engine,recovery', [('gemini', 'native'), ('gemini', 'fresh'), ('grok', 'fresh')])
def test_the_call_keeps_talking_and_keeps_its_result_once_around_a_long_effect(host_world, monkeypatch, engine, recovery):
    """The owner's whole promise on one call. A long effect runs. The conversation goes both ways on the connection that
    issued it where the engine allows, through a lost socket, and on the replacement, spoken and typed. A fresh recovery
    discloses what may not have been heard. The effect then ends: one effect, its result once, one answer, nothing repeated."""
    from tests._live_stack import stack
    slow = Held(monkeypatch)
    native = recovery == 'native'

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            first = live.wire
            first.tool('slow-call', 'list_available_tools', {})
            try:
                await slow.started()
                if engine == 'gemini':  # OpenAI and Grok answer only once their open call has its output (disclosed)
                    await answered_while_held(live, slow, 'Words before the loss.', 'Answer before the loss.')
                    await typed_and_answered(live, 'Typed before the loss.', 'Typed answer before the loss.')
                if native:
                    handle(first)
                    await live.wait(lambda: live.host.state.get('resume_handle') == 'covering', 30)
                else:
                    heard = len(first.audio)
                    first.input_start('owner-cut-off')
                    await live.wait(lambda: len(first.audio) >= heard + 960 and live.call.input_open, 30)
                await first.websocket.close(**RESTART)
                await live.wait(lambda: len(live.provider.connections) == 2 and live.call.connected.is_set(), 30)
                await live.wait(lambda: page_state(live) == cp.STATE_LIVE, 30)
                second = live.wire
                assert live.call.conversation == (1 if native else 2) and not slow.release.is_set()
                assert second.sent[0].get('setup', {}).get('sessionResumption', {}) == ({'handle': 'covering'} if native else {})
                if not native:
                    await live.wait(lambda: live_frames(second, 'may not have been heard'), 30)
                    await spoken_out(live)
                await answered_while_held(live, slow, 'Words after the loss.', 'Answer after the loss.')
                await typed_and_answered(live, 'Typed after the loss.', 'Typed answer after the loss.')
                assert not native_output(second, 'slow-call') and slow.effects == ['list_available_tools']
                said = spoken(live)
            finally:
                slow.release.set()
            if native:
                await live.wait(lambda: native_output(second, 'slow-call'), 60)
            else:
                await live.wait(lambda: any(POINTER in json.dumps(frame) for frame in live.wire.sent), 60)
            await live.quiet()
            await asyncio.sleep(.2)
            assert spoken(live) == said + 1, 'the result gets one answer'
            pointers = sum(POINTER in json.dumps(frame) for wire in live.provider.connections for frame in wire.sent)
            outputs = sum(native_output(wire, 'slow-call') for wire in live.provider.connections)
            assert (outputs, pointers) == ((1, 0) if native else (0, 1)), 'its result once, natively or by its one pointer'
            messages = live.host.messages
            assert slow.effects == ['list_available_tools'], 'one effect'
            assert messages[unit_at(messages, 'slow-call') + 1].get('tool_call_id') == 'slow-call'
            assert sum(row.get('tool_call_id') == 'slow-call' for row in messages if row.get('role') == 'tool') == 1
            for text in ('Typed before the loss.', 'Typed after the loss.') if engine == 'gemini' else ('Typed after the loss.',):
                assert written(live, text) == 1 and sum(text in str(row.get('content')) for row in messages) == 1
            for answer in ('Answer after the loss.', 'Typed answer after the loss.'):
                assert sum(answer in wire.spoken for wire in live.provider.connections) == 1
            gaps = [row for row in messages if row.get('_host_context_kind') == 'call_fact'
                    and 'may not have been heard' in row['content']]
            assert len(gaps) == int(not native), 'disclosed on fresh recovery only'
            assert not any('_tool_answer' in json.dumps(frame) for wire in live.provider.connections for frame in wire.sent)
            assert not live.call.deferred_tool_outputs and not live.call.pending_context_rows
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


def test_real_mail_measured_across_a_native_resume_reaches_it_once_before_its_pointer(host_world, monkeypatch):
    """The real producer: a typed message, a due renewal, the host's own failed ahead fit. Each connection's worker measures in
    turn while the conversation stays readable; only the live one writes, and its pointer follows the words."""
    from tests._live_stack import stack
    refused_speculative_fits(monkeypatch)

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            first = live.wire
            gate = Measures(monkeypatch, live, MAIL)
            handle(first)
            await live.wait(lambda: live.host.state.get('resume_handle') == 'covering', 30)
            gate.due = True
            said = spoken(live)
            live.mail.put({'text': MAIL, 'msg_id': 'typed-mail'})
            await live.wait(lambda: gate.count == 1, 30)
            assert gate.tasks[0] == '_Call.delivery' and live.call.pending_context_rows
            await first.websocket.close(**RESTART)
            await live.wait(lambda: gate.count == 2 and len(live.provider.connections) == 2, 30)
            second = live.wire
            assert live.call.connection_resumed and gate.tasks[1] == '_Call.delivery'
            await asyncio.sleep(.05)
            assert not live_frames(second, ROOM_POINTER), 'a slow measure cannot let its pointer overtake the words'
            gate.open.set()
            await live.wait(lambda: live_frames(second, ROOM_POINTER), 30)
            await live.quiet()
            await asyncio.sleep(.1)
            assert written(live, MAIL) == 1 and len(live_frames(second, MAIL)) == 1 and not live_frames(first, MAIL)
            assert position(second, MAIL) < position(second, ROOM_POINTER)
            assert sum(MAIL in str(row.get('content')) for row in live.call.session.held_context_rows) == 1
            assert sum(MAIL in str(row.get('content')) for row in live.host.messages) == 1
            assert spoken(live) == said + 1, 'one answer'
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


@pytest.mark.parametrize(("route", "next_connection"), [
    ('waiting-mail', 'resumed'),
    ('typed-mail-while-idle', 'fresh'),
])
def test_a_real_mail_write_meeting_a_closed_socket_keeps_the_call(host_world, monkeypatch, route, next_connection):
    """Whichever worker writes it. The call continues; a resume gets the words once before their pointer, and a fresh
    fit holds them and is still asked to answer."""
    from tests._live_stack import stack
    resumes = next_connection == 'resumed'
    if route == 'waiting-mail':
        refused_speculative_fits(monkeypatch)

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            first = live.wire
            gate = Measures(monkeypatch, live, MAIL)
            gate.open.set()
            if resumes:
                handle(first)
                await live.wait(lambda: live.host.state.get('resume_handle') == 'covering', 30)
            fired = close_on_write(monkeypatch, live, lambda payload: MAIL in json.dumps(payload), **RESTART)
            gate.due = route == 'waiting-mail'
            said = spoken(live)
            live.mail.put({'text': MAIL, 'msg_id': 'typed-mail'})
            await live.wait(lambda: fired and len(live.provider.connections) == 2 and live.call.connected.is_set(), 30)
            second = live.wire
            assert live.call.conversation == (1 if resumes else 2) and not live.running.done()
            await live.wait(lambda: live_frames(second, ROOM_POINTER), 30)
            await live.quiet()
            await asyncio.sleep(.1)
            assert not live_frames(first, MAIL), 'that write never reached the provider'
            assert len(live_frames(second, MAIL)) == int(resumes)
            if resumes:
                assert position(second, MAIL) < position(second, ROOM_POINTER)
            else:
                assert MAIL in json.dumps(second.sent[0]), 'the fresh fit holds the words'
            assert sum(MAIL in str(row.get('content')) for row in live.host.messages) == 1
            assert spoken(live) == said + 1, 'one answer'
            assert not live.call.pending_context_rows
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


def test_mail_queued_in_the_adapter_meeting_a_closed_socket_keeps_the_call(host_world, monkeypatch):
    """The turn is busy, so Gemini holds the words and writes them at turn end, on the reader's own receive, which classifies
    the loss. A positive to keep: the delivery worker is not involved, and the words are in the fresh fit, once."""
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            first = live.wire
            first.say('A long answer still being spoken.', complete=False)
            await live.wait(lambda: live.call.session.turn_busy, 30)
            fired = close_on_write(monkeypatch, live, lambda payload: MAIL in json.dumps(payload), **RESTART)
            live.mail.put({'text': MAIL, 'msg_id': 'typed-mail'})
            await live.wait(lambda: live.call.session.held_context, 30)
            assert not live.call.pending_context_rows and not fired
            first.complete()
            await live.wait(lambda: fired and len(live.provider.connections) == 2 and live.call.connected.is_set(), 30)
            second = live.wire
            await live.quiet()
            assert not live.running.done() and live.call.conversation == 2
            assert not live_frames(first, MAIL) and not live_frames(second, MAIL) and MAIL in json.dumps(second.sent[0])
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


@pytest.mark.parametrize('next_connection', ['resumed', 'fresh'])
def test_a_real_output_write_failing_on_a_resumed_connection_recovers_and_end_still_ends(host_world, monkeypatch, next_connection):
    """Astra's and the critic's self-wait through the real producer. The first measure dies with its connection, the resumed
    connection's worker writes, and that socket has closed. The reader classifies it with the microphone live: a third
    connection, the result once, one answer, and End ends the call."""
    from tests._live_stack import stack
    resumes = next_connection == 'resumed'
    refused_speculative_fits(monkeypatch)

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            first = live.wire
            gate = Measures(monkeypatch, live, 'kept-call')
            gate.tool_makes_it_due(monkeypatch, live)
            first.tool('kept-call', 'get_task_result', {'task_id': 'none'})
            await live.wait(lambda: gate.count == 1 and live.call.deferred_tool_outputs, 30)
            handle(first)
            await live.wait(lambda: live.host.state.get('resume_handle') == 'covering', 30)
            await first.websocket.close(**RESTART)
            await live.wait(lambda: gate.count == 2 and len(live.provider.connections) == 2, 30)
            assert live.call.connection_resumed and gate.tasks == ['_Call.delivery'] * 2
            withdraw = None if resumes else lambda: live.host.state.pop('resume_handle', None)
            fired = close_on_write(monkeypatch, live, lambda payload: 'toolResponse' in payload, before=withdraw, **RESTART)
            said = spoken(live)
            gate.open.set()
            await live.wait(lambda: fired and len(live.provider.connections) == 3 and live.call.connected.is_set(), 30)
            third = live.wire
            assert live.call.conversation == (1 if resumes else 2)
            if resumes:
                await live.wait(lambda: native_output(third, 'kept-call'), 30)
            else:
                await live.wait(lambda: any(POINTER in json.dumps(frame) for frame in third.sent), 30)
            await live.quiet()
            await asyncio.sleep(.1)
            connections = live.provider.connections
            assert sum(native_output(wire, 'kept-call') for wire in connections) == int(resumes)
            assert sum(POINTER in json.dumps(frame) for wire in connections for frame in wire.sent) == int(not resumes)
            assert spoken(live) == said + 1 and live.call.answer_due is None and not live.call.deferred_tool_outputs
            assert sum(row.get('tool_call_id') == 'kept-call' for row in live.host.messages if row.get('role') == 'tool') == 1
            live.page.end()
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline and not live.running.done():
                await asyncio.sleep(.02)
            assert live.running.done(), 'End ends the call'
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


def test_end_during_a_real_release_ends_the_call(host_world, monkeypatch):
    from tests._live_stack import stack
    refused_speculative_fits(monkeypatch)

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            gate = Measures(monkeypatch, live, 'kept-call')
            gate.tool_makes_it_due(monkeypatch, live)
            live.wire.tool('kept-call', 'get_task_result', {'task_id': 'none'})
            await live.wait(lambda: gate.count == 1, 30)
            live.page.end()
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline and not live.running.done():
                await asyncio.sleep(.02)
            done = live.running.done()
            gate.open.set()
            assert done, 'End ends the call while its worker measures'
            assert not native_output(live.wire, 'kept-call')
        assert live.result[1]['execution_status'] == 'ok', live.result
        assert any(row.get('tool_call_id') == 'kept-call' for row in live.host.messages), 'the result stays canonical'
    asyncio.run(scenario())


def test_a_real_refusal_by_the_serving_measure_stays_terminal_and_typed(host_world, monkeypatch):
    from tests._live_stack import stack
    refused_speculative_fits(monkeypatch)

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            gate = Measures(monkeypatch, live, 'kept-call')
            gate.tool_makes_it_due(monkeypatch, live)
            gate.verdict = False
            gate.open.set()

            async def no_fallback(rows, schemas):
                return None
            monkeypatch.setattr(live.host, 'tool_result_fallback', no_fallback)
            live.wire.tool('kept-call', 'get_task_result', {'task_id': 'none'})
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not live.running.done():
                await asyncio.sleep(.02)
            assert live.running.done() and len(live.provider.connections) == 1
            assert not native_output(live.wire, 'kept-call')
        assert live.result[1]['execution_status'] != 'ok' and 'context_capacity' in str(live.result), live.result
    asyncio.run(scenario())


def test_a_real_refusing_close_during_a_release_stays_terminal_and_typed(host_world, monkeypatch):
    """The reader still tells loss from refusal when the worker's write fails first: no reconnection on the provider's own words."""
    from tests._live_stack import stack
    refused_speculative_fits(monkeypatch)

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            gate = Measures(monkeypatch, live, 'kept-call')
            gate.tool_makes_it_due(monkeypatch, live)
            gate.open.set()
            fired = close_on_write(monkeypatch, live, lambda payload: 'toolResponse' in payload,
                                   code=1008, reason='synthetic policy refusal')
            live.wire.tool('kept-call', 'get_task_result', {'task_id': 'none'})
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not live.running.done():
                await asyncio.sleep(.02)
            assert fired and live.running.done() and len(live.provider.connections) == 1
        assert live.result[1]['execution_status'] != 'ok' and 'synthetic policy refusal' in str(live.result), live.result
    asyncio.run(scenario())


@pytest.mark.parametrize('engine', ['openai', 'grok'])
def test_a_real_interruption_meeting_a_closed_socket_is_a_transport_loss(host_world, monkeypatch, engine):
    """OpenAI and Grok write on barge-in. The same commands in the same order, and a closed socket leaves the reader reading."""
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            first = live.wire
            first.say('A long answer the owner cuts off.', complete=False)
            await live.wait(lambda: live.call.active_responses, 30)
            fired = close_on_write(monkeypatch, live, lambda payload: payload.get('type') in {
                'response.cancel', 'conversation.item.truncate'}, **RESTART)
            first.input_start('owner-barge-in')
            await live.wait(lambda: fired and len(live.provider.connections) == 2 and live.call.connected.is_set(), 30)
            assert live.call.conversation == 2 and not live.running.done()
            await live.quiet()
            live.provider.script.append(('say', 'Answered on the replacement.'))
            await live.owner_says('Words on the replacement.')
            await live.wait(lambda: 'Answered on the replacement.' in live.wire.spoken, 30)
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())
