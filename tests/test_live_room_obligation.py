"""A typed room request accepted by a connection stays owed until a response actually answers it.

Typed mail may be written, or only accepted into an adapter's queue, before its connection is lost. A fresh fit holds the
words, which answers nothing: the fresh session gets one neutral pointer, and so does every later fresh session until a
response begun after the request reached the provider finishes. The mind may answer, act or stay silent. An answered
request stays silent; observations are not requests. The real driver over its fake port, then the real host and
adapters over loopback.
"""
import asyncio
import json
import time

import pytest

from ouroboros import call_protocol as cp
from ouroboros.contracts.live_conversation import ContextUpdate
from ouroboros.contracts.realtime import CatalogModel
from tests._extension_loader_shared import _clear_loader_state  # noqa: F401
from tests._live_world import host_world as host_world
from tests.test_live_conversation_driver import Host, until
from tests.test_live_deferred_custody import MAIL, READY, answer, holding_mail, mail_updates, outputs, pointers, successor
from tests.test_live_delivery_owner import ROOM_POINTER, Socket, never, opened, position, room_pointers
from tests.test_live_delivery_owner_stack import RESTART, handle, live_frames, spoken
from tests.test_live_tool_continuity import POINTER
from tests.test_live_tool_continuity_stack import native_output
from tests._live_stack_wire import NativeWire

LATER = 'Later room mail.'
SPEAK = NativeWire.answer  # the loopback provider's own answer, whatever a scenario patches over it


class Busy(Socket):
    """An adapter whose turn is busy: it accepts an update at once and holds it only when it really writes it."""

    def __init__(self, during=None):
        super().__init__(during)
        self.busy, self.queue = True, []

    async def send(self, command):
        if self.busy and isinstance(command, ContextUpdate):
            self.queue.append(command)
            return
        await super().send(command)

    async def flush(self):
        self.busy = False
        queued, self.queue = self.queue, []
        for command in queued:
            await super().send(command)


class Legacy(Socket):
    """An older session, with no delivery record: its send boundary stands for delivery."""

    def __init__(self, during=None):
        super().__init__(during)
        del self.held_context_rows


def mailbox_of(world):
    box = []

    async def mailbox():
        return box.pop(0) if box else []
    world.host.mailbox = mailbox
    return box


def room_mail(text=MAIL):
    return [{'role': 'user', 'content': text}]


async def written_mail(box, session, text=MAIL):
    box.append(room_mail(text))
    await until(lambda: text in str(session.sent))


def respond_ended(session, response, *, status='completed', **more):
    session.emit('response_done', response_id=response, usage={}, raw_usage={}, status=status, **more)


async def resumed(world, session):
    """A native resume. A real adapter's record continues where the last one ended, so this one does too."""
    count = len(world.host.sessions)
    session.emit('resumable', handle='covering')
    session.emit('closed', resumable=True, transport_loss=True, reason='transport_lost')
    await until(lambda: len(world.host.sessions) == count + 1)
    made = world.host.sessions[-1]
    made.held_context_rows[:0] = session.held_context_rows
    made.emit('ready', **READY)
    await until(lambda: world.call.connected.is_set() and world.call.session is made)
    return made


# --- the fake port: what delivers a request, and what answers it --------------------------------------------------------


@pytest.mark.parametrize('answered_on', [1, 2, 3])
def test_a_request_owes_one_pointer_on_each_fresh_session_until_answered(answered_on):
    """Answered before the loss it stays silent; unanswered it follows every fresh session. The words are written once."""
    async def scenario():
        async with opened(holds=never, fit_failed=False) as world:
            box = mailbox_of(world)
            session = world.first
            await written_mail(box, session)
            for number in (1, 2, 3):
                if answered_on == number:
                    await answer(session, f'answer-{number}')
                session = await successor(world, session)
                await asyncio.sleep(.04)
                await until(lambda: room_pointers(session) == int(answered_on > number))
            assert [room_pointers(made) for made in world.host.sessions] == [0, *(int(answered_on > n) for n in (1, 2, 3))]
            assert sum(len(mail_updates(made)) for made in world.host.sessions) == 1, 'a pointer never repeats the words'
    asyncio.run(scenario())


@pytest.mark.parametrize('status,answered', [('completed', True), ('interrupted', True), ('failed', False), ('incomplete', False)])
def test_only_a_finished_response_answers(status, answered):
    async def scenario():
        async with opened(holds=never, fit_failed=False) as world:
            box = mailbox_of(world)
            first = world.first
            await written_mail(box, first)
            first.emit('response_started', response_id='answer')
            respond_ended(first, 'answer', status=status)
            await first.handled()
            second = await successor(world, first)
            await asyncio.sleep(.04)
            assert room_pointers(second) == int(not answered)
    asyncio.run(scenario())


@pytest.mark.parametrize('ends', [False, True], ids=['still-continuing', 'ended'])
def test_a_continuing_response_answers_only_once_it_ends(ends):
    async def scenario():
        async with opened(holds=never, fit_failed=False) as world:
            box = mailbox_of(world)
            first = world.first
            await written_mail(box, first)
            first.emit('response_started', response_id='answer')
            respond_ended(first, 'answer', continues=True)
            if ends:
                respond_ended(first, 'answer')
            await first.handled()
            second = await successor(world, first)
            await asyncio.sleep(.04)
            assert room_pointers(second) == int(not ends)
    asyncio.run(scenario())


@pytest.mark.parametrize('case,owed', [('lost-while-queued', True), ('written-then-no-response', True),
                                       ('response-begun-after-delivery', False)])
def test_adapter_acceptance_is_not_delivery(case, owed):
    """Accepting the words removes their pointer from the offers. Only the adapter's own record of a real write delivers
    them, and only a response begun after that write answers them."""
    async def scenario():
        async with opened(make=lambda during: Busy(during), holds=never, fit_failed=False) as world:
            box = mailbox_of(world)
            first = world.first
            if case == 'response-underway-at-acceptance':
                first.emit('response_started', response_id='underway')
                await first.handled()
            box.append(room_mail())
            await until(lambda: first.queue)
            assert not mail_updates(first) and not world.call.pending_offers, 'accepted: its pointer has left the offers'
            if case == 'response-underway-at-acceptance':
                respond_ended(first, 'underway')
                await first.handled()
            elif case == 'response-begun-while-queued':
                first.emit('response_started', response_id='talk-over')
                await first.handled()
            if case != 'lost-while-queued':
                await first.flush()
                assert len(mail_updates(first)) == 1
            if case == 'response-begun-while-queued':
                respond_ended(first, 'talk-over')
                await first.handled()
            elif case == 'response-begun-after-delivery':
                await answer(first, 'real-answer')
            world.make = Socket
            second = await successor(world, first)
            await asyncio.sleep(.04)
            assert room_pointers(second) == int(owed) and not mail_updates(second)
    asyncio.run(scenario())


def test_a_response_underway_when_words_are_written_does_not_answer_them():
    async def scenario():
        async with opened(holds=never, fit_failed=False) as world:
            box = mailbox_of(world)
            first = world.first
            first.emit('response_started', response_id='underway')
            await first.handled()
            await written_mail(box, first)
            respond_ended(first, 'underway')
            await first.handled()
            second = await successor(world, first)
            await until(lambda: room_pointers(second) == 1)
    asyncio.run(scenario())


def test_a_newer_request_joins_the_older_and_only_its_own_delivery_can_settle_both():
    """A response to the first words finishes while newer words wait in the queue. Neither is settled, so one pointer asks;
    that pointer queued is not delivered either, and one answer after its delivery settles both."""
    async def scenario():
        async with opened(make=lambda during: Busy(during), holds=never, fit_failed=False) as world:
            box = mailbox_of(world)
            first = world.first
            box.append(room_mail())
            await until(lambda: first.queue)
            await first.flush()
            first.emit('response_started', response_id='answers-the-first')
            await first.handled()
            first.busy = True
            box.append(room_mail(LATER))
            await until(lambda: first.queue)
            respond_ended(first, 'answers-the-first')
            await first.handled()
            world.make = lambda during: Busy(during)
            second = await successor(world, first)
            await until(lambda: len(second.queue) == 1 and ROOM_POINTER in str(second.queue))
            second.emit('response_started', response_id='before-the-pointer-is-written')
            respond_ended(second, 'before-the-pointer-is-written')
            await second.handled()
            await second.flush()
            await until(lambda: room_pointers(second) == 1)
            assert not any(LATER in str(command) for command in second.sent), 'its words are in the fit'
            world.make = Socket
            third = await successor(world, second)
            await until(lambda: room_pointers(third) == 1)
            await answer(third, 'answers-both')
            fourth = await successor(world, third)
            await asyncio.sleep(.04)
            assert not room_pointers(fourth)
    asyncio.run(scenario())


def test_waiting_mail_stays_owed_from_retirement_through_each_fresh_session():
    """The worker never wrote it. The first fresh fit retires the rows and its pointer asks, again on each fresh session."""
    async def scenario():
        async with opened(holds=holding_mail) as world:
            box = mailbox_of(world)
            host, first = world.host, world.first
            host.due = True
            box.append(room_mail())
            await asyncio.wait_for(world.entered.wait(), 2)
            second = await successor(world, first)
            await until(lambda: room_pointers(second) == 1)
            third = await successor(world, second)
            await until(lambda: room_pointers(third) == 1)
            await answer(third, 'answer')
            await successor(world, third)
            await asyncio.sleep(.04)
            assert [room_pointers(made) for made in host.sessions] == [0, 1, 1, 0]
            assert not any(mail_updates(made) for made in host.sessions)
    asyncio.run(scenario())


def test_a_turn_begun_after_waiting_mail_is_written_answers_it_and_retires_its_pointer():
    """The owner is speaking, so the pointer waits. The turn that follows began with the words delivered: nothing more is asked."""
    async def scenario():
        async with opened(holds=holding_mail) as world:
            box = mailbox_of(world)
            host, call, first = world.host, world.call, world.first
            first.emit('input_started')
            await first.handled()
            host.due = True
            box.append(room_mail())
            await asyncio.wait_for(world.entered.wait(), 2)
            world.release.set()
            await until(lambda: mail_updates(first))
            assert mail_updates(first)[0].respond is False and any(ROOM_POINTER in str(row) for row in call.pending_offers)
            first.emit('response_started', response_id='owner-turn')
            respond_ended(first, 'owner-turn')
            first.emit('input_done')
            await first.handled()
            await asyncio.sleep(.04)
            assert not room_pointers(first) and not call.pending_offers
            second = await successor(world, first)
            await asyncio.sleep(.04)
            assert not room_pointers(second)
    asyncio.run(scenario())


def test_waiting_mail_asked_for_on_its_own_connection_is_answered_there():
    async def scenario():
        async with opened(holds=never) as world:
            box = mailbox_of(world)
            host, first = world.host, world.first
            host.due = True
            box.append(room_mail())
            await until(lambda: room_pointers(first) == 1)
            assert position(first, MAIL) < position(first, ROOM_POINTER)
            await answer(first, 'answer')
            second = await successor(world, first)
            await asyncio.sleep(.04)
            assert not room_pointers(second)
    asyncio.run(scenario())


def test_a_claimed_drain_is_owed_like_any_other_request():
    """Words durable before a cancelled write leave ``take_mail_answer``'s pointer, which also follows fresh losses."""
    async def scenario():
        host = Host()
        claims = [True]
        host.take_mail_answer = lambda: bool(claims) and claims.pop()
        async with opened(holds=never, fit_failed=False, host=host) as world:
            first = world.first
            await until(lambda: room_pointers(first) == 1)
            second = await successor(world, first)
            await until(lambda: room_pointers(second) == 1)
            await answer(second, 'answer')
            await successor(world, second)
            await asyncio.sleep(.04)
            assert [room_pointers(made) for made in host.sessions] == [1, 1, 0]
    asyncio.run(scenario())


def test_frame_facts_and_peer_observations_owe_no_answer():
    """Not every observation is a request: their own offers are unchanged, and nothing follows them across a fresh loss.
    Neither gets any response here, so an obligation wrongly armed for either would show on the fresh session."""
    async def scenario():
        host = Host()
        host.model = CatalogModel('test', 'Test', vision=False, honours=frozenset({'pause', 'interruptions'}))
        async with opened(holds=never, fit_failed=False, host=host) as world:
            call, first = world.call, world.first
            host.hub.events.append(cp.PeerControl('peer', cp.PEER_OWNER, {'type': 'context', 'text': 'Peer fact.'}))
            await until(lambda: 'Peer fact.' in str(first.sent))
            call.attention.frame(cp.PeerVideo('remote', cp.PEER_OWNER, 'image/jpeg', b'exact-frame'))
            first.emit('input_started')
            await until(lambda: 'could not be saved' in str(first.sent))
            first.emit('input_done')
            await first.handled()
            second = await successor(world, first)
            await asyncio.sleep(.05)
            assert not room_pointers(second) and not pointers(second)
    asyncio.run(scenario())


@pytest.mark.parametrize('answered', [False, True], ids=['unanswered', 'answered'])
def test_an_older_session_uses_its_send_boundary(answered):
    async def scenario():
        async with opened(make=lambda during: Legacy(during), holds=never, fit_failed=False) as world:
            box = mailbox_of(world)
            first = world.first
            await written_mail(box, first)
            if answered:
                await answer(first, 'answer')
            world.make = Socket
            second = await successor(world, first)
            await asyncio.sleep(.04)
            assert room_pointers(second) == int(not answered)
    asyncio.run(scenario())


@pytest.mark.parametrize('case,room,tool', [('neither-answered', 1, 1), ('result-answered-before-the-mail', 1, 0),
                                            ('mail-accepted-during-the-results-answer', 1, 0),
                                            ('one-answer-after-both', 0, 0)])
def test_room_and_tool_obligations_settle_independently(case, room, tool):
    """Each is answered only by a response begun after its own delivery. What is left gets its pointer, in one offer, and
    one response then settles everything left."""
    async def scenario():
        make = (lambda during: Busy(during)) if case == 'mail-accepted-during-the-results-answer' else Socket
        async with opened(make=make, holds=never, fit_failed=False) as world:
            box = mailbox_of(world)
            first = world.first

            async def result():
                first.emit('tool_call', call_id='kept', name='quick_tool', arguments='{}')
                await until(lambda: first.finished == 1)
            if case == 'result-written-during-the-mails-answer':
                await written_mail(box, first)
                first.emit('response_started', response_id='answers-the-mail')
                await first.handled()
                await result()
                respond_ended(first, 'answers-the-mail')
                await first.handled()
            elif case == 'mail-accepted-during-the-results-answer':
                await result()
                first.emit('response_started', response_id='answers-the-result')
                await first.handled()
                box.append(room_mail())
                await until(lambda: first.queue)
                respond_ended(first, 'answers-the-result')
                await first.handled()
            else:
                await result()
                if case == 'result-answered-before-the-mail':
                    await answer(first, 'answers-the-result')
                await written_mail(box, first)
                if case == 'one-answer-after-both':
                    await answer(first, 'answers-both')
            world.make = Socket
            second = await successor(world, first)
            await asyncio.sleep(.05)
            assert (room_pointers(second), pointers(second)) == (room, tool)
            if room and tool:
                assert position(second, POINTER) == position(second, ROOM_POINTER), 'one offer, one turn'
            assert not outputs(second) and not mail_updates(second)
            if room or tool:
                await answer(second, 'answers-what-is-left')
            third = await successor(world, second)
            await asyncio.sleep(.04)
            assert not room_pointers(third) and not pointers(third)
    asyncio.run(scenario())


@pytest.mark.parametrize('answered', [False, True], ids=['unanswered', 'answered-on-the-resumed-conversation'])
def test_a_native_resume_continues_the_conversation_that_holds_the_request(answered):
    """A resume is the same conversation: no pointer there, and the request stays owed until a response answers it. A later
    fresh fit then asks only if it is still owed."""
    async def scenario():
        async with opened(holds=never, fit_failed=False) as world:
            box = mailbox_of(world)
            call, host, first = world.call, world.host, world.first
            await written_mail(box, first)
            second = await resumed(world, first)
            await asyncio.sleep(.04)
            assert call.conversation == 1 and host.options[1].resume_handle == 'covering'
            assert not room_pointers(second) and not mail_updates(second)
            if answered:
                await answer(second, 'answer')
            third = await successor(world, second)
            await asyncio.sleep(.04)
            assert call.conversation == 2 and room_pointers(third) == int(not answered)
    asyncio.run(scenario())


# --- the real host and adapters ------------------------------------------------------------------------------------------


def answers_only_on(monkeypatch, *numbers):
    """The loopback provider speaks only on the connections named; elsewhere a request gets no answer at all."""
    monkeypatch.setattr(NativeWire, 'answer', lambda wire: SPEAK(wire) if wire.number in numbers else None)


def room_frames(wire):
    return live_frames(wire, ROOM_POINTER)


def queued(session):
    return getattr(session, 'held_context', None) or getattr(session, 'context_queue', None)


def canonical(live, text=MAIL):
    return sum(text in str(row.get('content')) for row in live.host.messages)


def no_private_marker_on_any_wire(live):
    return not any('_tool_answer' in json.dumps(frame) for wire in live.provider.connections for frame in wire.sent)


async def fresh_after(live, wire):
    count = len(live.provider.connections)
    await wire.websocket.close(**RESTART)
    await live.wait(lambda: len(live.provider.connections) == count + 1 and live.call.connected.is_set(), 30)
    return live.wire


async def spoken_out(live):
    """The turn has finished and been heard."""
    await live.wait(lambda: not live.call.response_open and not live.call.pending_spoken and not live.call.unheard(), 30)


async def asked(live, wire, owed):
    """An owed pointer goes out at the first quiet moment; otherwise that moment passes with nothing asked."""
    if owed:
        await live.wait(lambda: room_frames(wire), 30)
    else:
        await live.quiet()
    await asyncio.sleep(.2)
    assert len(room_frames(wire)) == int(owed), (wire.number, owed)


@pytest.mark.parametrize(("engine", "answered_on"), [('gemini', 1), ('openai', 3), ('grok', 1)])
def test_real_typed_mail_is_owed_on_each_fresh_session_until_answered(host_world, monkeypatch, engine, answered_on):
    """The real producer, adapters and sockets. Unanswered words follow each fresh session; once answered, on whichever
    connection, later losses stay silent."""
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            answers_only_on(monkeypatch, answered_on)
            wire, said = live.wire, spoken(live)
            live.mail.put({'text': MAIL, 'msg_id': 'typed-mail'})
            await live.wait(lambda: live_frames(wire, MAIL), 30)
            for number in (1, 2, 3):
                if answered_on == number:
                    await live.wait(lambda: spoken(live) == said + 1, 30)
                    await spoken_out(live)
                wire = await fresh_after(live, wire)
                assert MAIL in json.dumps(wire.sent), 'every fresh fit holds the words'
                await asked(live, wire, answered_on > number)
            assert spoken(live) == said + 1, 'one answer'
            assert sum(len(live_frames(w, MAIL)) for w in live.provider.connections) == 1, 'written live once'
            assert canonical(live) == 1 and no_private_marker_on_any_wire(live)
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


@pytest.mark.parametrize(("engine", "case", "owed"), [
    ('gemini', 'lost-while-queued', True),
    ('openai', 'written-as-the-old-turn-ended', True),
    ('grok', 'answered-after-delivery', False),
])
def test_real_adapter_queued_mail_is_owed_until_answered_after_delivery(host_world, monkeypatch, engine, case, owed):
    """A turn is already underway, so the adapter accepts the words and writes them only when it ends. That turn saw
    nothing of them; only a response begun after the write can answer them."""
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            answers_only_on(monkeypatch, *((1,) if case == 'answered-after-delivery' else ()))
            first = live.wire
            first.say('An answer already being spoken.', complete=False)
            await live.wait(lambda: live.call.active_responses, 30)
            live.mail.put({'text': MAIL, 'msg_id': 'typed-mail'})
            await live.wait(lambda: queued(live.call.session), 30)
            assert not live_frames(first, MAIL) and not live.call.pending_context_rows, 'accepted, not written'
            said = spoken(live)
            if case != 'lost-while-queued':
                first.complete()
                await live.wait(lambda: live_frames(first, MAIL), 30)
            if case == 'answered-after-delivery':
                await live.wait(lambda: spoken(live) == said + 1, 30)
                await spoken_out(live)
            answers_only_on(monkeypatch)
            second = await fresh_after(live, first)
            await asked(live, second, owed)
            assert not live_frames(second, MAIL) and canonical(live) == 1 and no_private_marker_on_any_wire(live)
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


def test_real_gemini_silent_turn_ending_as_queued_words_are_written_does_not_answer_them(host_world, monkeypatch):
    """Gemini begins and ends a silent local response, then writes its queued update, in one conversion. That response
    ended before those words reached the provider: its own end record decides, and the newer words stay owed."""
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            answers_only_on(monkeypatch)
            first = live.wire
            live.mail.put({'text': MAIL, 'msg_id': 'first-mail'})
            await live.wait(lambda: live_frames(first, MAIL), 30)
            live.mail.put({'text': LATER, 'msg_id': 'later-mail'})
            await live.wait(lambda: live.call.session.held_context, 30)
            first.complete()  # a usage-only turn end: silence chosen before the later words were written
            await live.wait(lambda: live_frames(first, LATER), 30)
            await asyncio.sleep(.05)
            second = await fresh_after(live, first)
            await asked(live, second, True)
            assert not live_frames(second, LATER) and canonical(live, LATER) == 1
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


def test_real_gemini_silent_choice_after_delivery_is_an_answer(host_world, monkeypatch):
    """Silence is a finished choice: with the words delivered, a turn ending with no output settles them."""
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            answers_only_on(monkeypatch)
            first = live.wire
            live.mail.put({'text': MAIL, 'msg_id': 'typed-mail'})
            await live.wait(lambda: live_frames(first, MAIL), 30)
            first.complete()
            await live.wait(lambda: not live.call.response_open, 30)
            second = await fresh_after(live, first)
            await asked(live, second, False)
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


@pytest.mark.parametrize('engine', ['gemini', 'openai'])
def test_real_tool_and_room_obligations_get_one_answer_and_no_repeat(host_world, monkeypatch, engine):
    """A written result and typed words, both unanswered at the loss: both pointers, one answer, nothing written again and
    the effect not run again; a later loss stays silent."""
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            answers_only_on(monkeypatch, 2)
            first = live.wire
            first.tool('kept-call', 'get_task_result', {'task_id': 'none'})
            await live.wait(lambda: native_output(first, 'kept-call') and not live.call.tools_running, 30)
            live.mail.put({'text': MAIL, 'msg_id': 'typed-mail'})
            # OpenAI still awaits the result's response, so it may only accept the words: owed either way.
            await live.wait(lambda: canonical(live) and (live_frames(first, MAIL) or queued(live.call.session)), 30)
            said = spoken(live)
            second = await fresh_after(live, first)
            await live.wait(lambda: room_frames(second) and spoken(live) == said + 1, 30)
            await spoken_out(live)
            assert len(room_frames(second)) == 1 and sum(POINTER in json.dumps(frame) for frame in second.sent) == 1
            if engine == 'gemini':  # one update carries both
                assert any(POINTER in json.dumps(frame) and ROOM_POINTER in json.dumps(frame) for frame in second.sent)
            assert not native_output(second, 'kept-call') and not live_frames(second, MAIL)
            answers_only_on(monkeypatch)
            third = await fresh_after(live, second)
            await asked(live, third, False)
            assert not any(POINTER in json.dumps(frame) for frame in third.sent) and spoken(live) == said + 1
            assert sum(row.get('tool_call_id') == 'kept-call' for row in live.host.messages if row.get('role') == 'tool') == 1
            assert no_private_marker_on_any_wire(live)
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


@pytest.mark.parametrize('answered', [False, True], ids=['unanswered', 'answered-on-the-resumed-conversation'])
def test_real_native_resume_continues_the_conversation_that_holds_the_request(host_world, monkeypatch, answered):
    """Real Gemini handles, read in order. A handle published after the write resumes the same conversation: no pointer,
    and the request is owed until a response there answers it. A later fresh fit asks only if it still is."""
    from tests._live_stack import stack
    words = 'Answered on the resumed conversation.'

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            answers_only_on(monkeypatch)
            first = live.wire
            live.mail.put({'text': MAIL, 'msg_id': 'typed-mail'})
            await live.wait(lambda: live_frames(first, MAIL), 30)
            handle(first, 'after-mail')
            await live.wait(lambda: live.host.state.get('resume_handle') == 'after-mail', 30)
            second = await fresh_after(live, first)
            assert live.call.conversation == 1 and second.sent[0]['setup']['sessionResumption'] == {'handle': 'after-mail'}
            await asyncio.sleep(.2)
            assert not room_frames(second) and not live_frames(second, MAIL)
            if answered:
                second.say(words)
                await live.wait(lambda: any(v.get('text') == words and v.get('final') for v in live.page.controls), 30)
                await spoken_out(live)
            second.put({'sessionResumptionUpdate': {'resumable': False}})
            await live.wait(lambda: not live.host.state.get('resume_handle'), 30)
            third = await fresh_after(live, second)
            assert live.call.conversation == 2
            await asked(live, third, not answered)
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


def test_real_owed_request_leaves_the_conversation_going_and_end_ends_it(host_world, monkeypatch):
    """With the words owed, the reader and microphone carry on: the owner speaks and is answered, and that turn, begun after
    the pointer reached the provider, settles them. End then ends the call at once."""
    from tests._live_stack import stack
    reply = 'Answered the owner after the pointer.'

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            answers_only_on(monkeypatch)
            first = live.wire
            live.mail.put({'text': MAIL, 'msg_id': 'typed-mail'})
            await live.wait(lambda: live_frames(first, MAIL), 30)
            second = await fresh_after(live, first)
            await live.wait(lambda: room_frames(second), 30)
            answers_only_on(monkeypatch, 2)
            live.provider.script.append(('say', reply))
            await live.owner_says('Words while the room request is owed.')
            await live.wait(lambda: any(v.get('text') == reply and v.get('final') for v in live.page.controls), 30)
            await spoken_out(live)
            third = await fresh_after(live, second)
            await asked(live, third, False)
            live.page.end()
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline and not live.running.done():
                await asyncio.sleep(.02)
            assert live.running.done(), 'End ends the call'
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())
