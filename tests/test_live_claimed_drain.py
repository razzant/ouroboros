"""Typed words whose mailbox worker was cancelled before it received them.

The host's drain is joined, so when a connection's teardown cancels the worker the words are canonical and their answer
is claimed, while the drain's return value has nowhere to go. A fresh fit holds those words. A native resume continues a
conversation that never saw them, so there they take the worker's lane: written once, ahead of their pointer, and owed
until a response begun after that write finishes. The real host's drain, the real driver over its fake port, then the
real host and Gemini adapter over loopback.
"""
import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from ouroboros import call_protocol as cp
from ouroboros.live_conversation import ConversationHost
from ouroboros.realtime_setup import rows_for_engine
from tests._extension_loader_shared import _clear_loader_state  # noqa: F401
from tests._live_world import host_world as host_world
from tests.test_live_conversation_driver import Host, until
from tests.test_live_deferred_custody import MAIL, answer, holding_result, mail_updates, outputs, pointers, successor
from tests.test_live_delivery_owner import ROOM_POINTER, Socket, never, opened, position, room_pointers
from tests.test_live_delivery_owner_stack import RESTART, close_on_write, handle, live_frames, spoken, written
from tests.test_live_host_joined import mailbox_host
from tests.test_live_room_obligation import (LATER, Busy, answers_only_on, asked, canonical, fresh_after, mailbox_of,
                                             no_private_marker_on_any_wire, respond_ended, resumed, room_frames, room_mail,
                                             spoken_out)
from tests.test_live_tool_continuity_stack import native_output

HANDLE = 'published-before-the-mail'
OLDER = 'Older room mail waiting for its measure.'


# --- the real host's drain ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("cancel", "vision"), [
    pytest.param(False, True, id='received-sighted'),
    pytest.param(True, False, id='cancelled-blind'),
])
def test_a_cancelled_waiter_leaves_exactly_what_it_would_have_received(monkeypatch, tmp_path, cancel, vision):
    """The same projection either way: what the engine can take, its source stamp kept and host-only keys gone. Once."""
    async def run():
        host, entered, release, _, row = mailbox_host(monkeypatch, tmp_path)
        host.model.vision = vision
        row.update(_call_frame_source='source-stamp', _host_context_kind='host-only', content=[
            {'type': 'text', 'text': 'Look at this.'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}, '_caption': 'saved.png'}])
        expected = rows_for_engine([row], vision=vision, schemas=[])
        assert expected[0]['_call_frame_source'] == 'source-stamp' and '_host_context_kind' not in expected[0]
        assert ('image_url' in json.dumps(expected)) == vision
        task = asyncio.create_task(host.mailbox())
        assert await asyncio.to_thread(entered.wait, 30)
        if cancel:
            task.cancel()
            await asyncio.sleep(0)
        release.set()
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await task
            assert ConversationHost.take_mail_rows(host) == expected
        else:
            assert await task == expected, 'its waiter has them: nothing is kept'
        assert ConversationHost.take_mail_rows(host) == []
        assert ConversationHost.take_mail_answer(host) is True and ConversationHost.take_mail_answer(host) is False
    asyncio.run(run())


def test_old_host_mail_rows_port_defaults_to_nothing_kept():
    from ouroboros.contracts.live_conversation import LiveHost
    assert LiveHost.take_mail_rows(object()) == []


# --- the real driver over its fake port -----------------------------------------------------------------------------------


class Claiming(Host):
    """The host port after a cancelled drain: its claim and what its waiter would have received, once."""

    def __init__(self):
        super().__init__()
        self.claimed, self.rows = False, []

    def claim(self, text=MAIL):
        self.claimed, self.rows = True, room_mail(text)

    def take_mail_answer(self):
        claimed, self.claimed = self.claimed, False
        return claimed

    def take_mail_rows(self):
        rows, self.rows = self.rows, []
        return rows


async def dropped(world, session):
    """The session drops with a covering handle; its successor opens and has not reported ready."""
    count = len(world.host.sessions)
    session.emit('resumable', handle='covering')
    session.emit('closed', resumable=True, transport_loss=True, reason='transport_lost')
    await until(lambda: len(world.host.sessions) == count + 1)
    return world.host.sessions[-1]


@pytest.mark.parametrize('route', ['no-handle', 'resume-refused-while-opening'])
def test_a_fresh_connection_asks_for_claimed_words_without_writing_them(route):
    """Its fit holds them. A resume refused while it opens spent no words: consume never saw it."""
    async def scenario():
        host = Claiming()
        async with opened(holds=never, fit_failed=False, host=host) as world:
            first = world.first
            host.claim()
            world.stale_resume = route == 'resume-refused-while-opening'
            second = await successor(world, first, resumes=world.stale_resume)
            await until(lambda: room_pointers(second) == 1)
            assert host.options[1].resume_handle == '' and world.call.conversation == 2
            assert not mail_updates(second) and not world.call.pending_context_rows
            await answer(second, 'answer')
            third = await successor(world, second)
            await asyncio.sleep(.04)
            assert not room_pointers(third) and not mail_updates(third)
    asyncio.run(scenario())


@pytest.mark.parametrize('then', ['resumes', 'goes-fresh'])
def test_claimed_words_wait_out_a_second_loss(then):
    """Nothing is written before the resumed connection is ready. Lost first, the words stay with the call: the next resume
    writes them once, and a fresh fit holds them and asks."""
    async def scenario():
        host = Claiming()
        async with opened(holds=never, fit_failed=False, host=host) as world:
            first = world.first
            host.claim()
            second = await dropped(world, first)
            await asyncio.sleep(.04)
            assert not second.sent and world.call.conversation == 1
            if then == 'resumes':
                third = await resumed(world, second)
                await until(lambda: room_pointers(third) == 1)
                assert [update.respond for update in mail_updates(third)] == [False]
                assert position(third, MAIL) < position(third, ROOM_POINTER) and world.call.conversation == 1
            else:
                third = await successor(world, second)
                await until(lambda: room_pointers(third) == 1)
                assert not mail_updates(third) and world.call.conversation == 2
            assert not second.sent
    asyncio.run(scenario())


@pytest.mark.parametrize('case,owed', [('response-begun-while-queued', True), ('response-begun-after-delivery', False)])
def test_claimed_words_are_answered_only_after_their_write(case, owed):
    """Accepted by a busy adapter they have not been delivered, and neither the claim nor the fit stands in for that."""
    async def scenario():
        host = Claiming()
        async with opened(holds=never, fit_failed=False, host=host) as world:
            first = world.first
            host.claim()
            world.make = lambda during: Busy(during)
            second = await resumed(world, first)
            await until(lambda: second.queue)
            assert not mail_updates(second)
            if case == 'response-begun-while-queued':
                second.emit('response_started', response_id='underway')
                await second.handled()
            await second.flush()
            await until(lambda: mail_updates(second))
            if case == 'response-begun-while-queued':
                respond_ended(second, 'underway')
                await second.handled()
            else:
                await answer(second, 'answer')
            world.make = Socket
            third = await successor(world, second)
            await asyncio.sleep(.04)
            assert room_pointers(third) == int(owed) and not mail_updates(third)
    asyncio.run(scenario())


def test_claimed_words_keep_their_place_between_older_and_newer_mail():
    """Older mail was waiting for its measure at the loss, and newer mail arrives during the next one. All before any pointer."""
    async def scenario():
        host = Claiming()
        async with opened(holds=lambda rows: any(row.get('content') == OLDER for row in rows), host=host) as world:
            box = mailbox_of(world)
            first = world.first
            host.due = True
            box.append(room_mail(OLDER))
            await asyncio.wait_for(world.entered.wait(), 2)
            host.due = False
            host.claim()
            second = await resumed(world, first)
            box.append(room_mail(LATER))
            await until(lambda: any(row.get('content') == LATER for row in world.call.pending_context_rows))
            assert not second.sent, 'the measure of the older words is still open'
            world.release.set()
            await until(lambda: LATER in str(second.sent) and room_pointers(second))
            sent = str(second.sent)
            assert sent.index(OLDER) < sent.index(MAIL) < sent.index(LATER) < sent.index(ROOM_POINTER)
            assert [sent.count(text) for text in (OLDER, MAIL, LATER)] == [1, 1, 1]
    asyncio.run(scenario())


def test_a_retained_result_goes_first_then_the_claimed_words_and_one_response_answers_both():
    """The conversation is waiting for that result: native order, its effect not run again, and nothing more to ask."""
    async def scenario():
        host = Claiming()
        async with opened(holds=holding_result, host=host) as world:
            first = world.first
            host.due = True
            first.emit('tool_call', call_id='kept', name='quick_tool', arguments='{}')
            await asyncio.wait_for(world.entered.wait(), 2)
            host.due = False
            world.holds = never
            host.claim()
            second = await resumed(world, first)
            await until(lambda: outputs(second) and mail_updates(second))
            assert second.sent.index(outputs(second)[0]) < second.sent.index(mail_updates(second)[0])
            await answer(second, 'answers-both')
            await asyncio.sleep(.04)
            assert len(outputs(first) + outputs(second)) == 1 and len(mail_updates(second)) == 1
            assert not pointers(second) and not room_pointers(second) and world.call.room_due is None
    asyncio.run(scenario())


def test_end_before_the_resumed_connection_is_ready_writes_nothing():
    async def scenario():
        host = Claiming()
        async with opened(holds=never, fit_failed=False, host=host) as world:
            first = world.first
            host.claim()
            second = await dropped(world, first)
            host.hub.ended_by_owner = True
            await asyncio.wait_for(world.task, 5)
            assert not second.sent
    asyncio.run(scenario())


def test_an_older_host_that_claims_without_rows_still_gets_its_pointer():
    """No rows to write: the pointer it always got, on any connection. The real host always has them."""
    async def scenario():
        host = Host()
        claims = [True]
        async with opened(holds=never, fit_failed=False, host=host) as world:
            first = world.first
            host.take_mail_answer = lambda: bool(claims) and claims.pop()
            second = await resumed(world, first)
            await until(lambda: room_pointers(second) == 1)
            assert world.call.conversation == 1 and not mail_updates(second)
    asyncio.run(scenario())


# --- the real host and Gemini adapter ---------------------------------------------------------------------------------------


@asynccontextmanager
async def held_drain(monkeypatch, live):
    """The host's real drain stops just after committing the words until the connection's own teardown cancels the worker
    around it: that teardown schedules the release in the very step that cancels. The call runs on its own loop."""
    held = SimpleNamespace(entered=threading.Event(), release=threading.Event(), cancelled=threading.Event())
    checkpoint, mailbox, send_state = live.host.checkpoint, live.host.mailbox, live.call.hub.send_state

    def checkpointing(boundary):
        if not held.entered.is_set() and MAIL in str(live.host.messages[-1].get('content')):
            held.entered.set()
            assert held.release.wait(60)
        return checkpoint(boundary)

    async def watched():
        try:
            return await mailbox()
        except asyncio.CancelledError:
            if held.entered.is_set():
                held.cancelled.set()
            raise

    def state(value, reason='', **more):
        send_state(value, reason, **more)
        if not reason and value in {cp.STATE_RECONNECTING, cp.STATE_CLOSING}:
            asyncio.get_running_loop().call_soon(held.release.set)

    monkeypatch.setattr(live.host, 'checkpoint', checkpointing)
    monkeypatch.setattr(live.host, 'mailbox', watched)
    monkeypatch.setattr(live.call.hub, 'send_state', state)
    await asyncio.sleep(.3)  # retires a poll that captured the original method
    try:
        yield held
    finally:
        held.release.set()


async def published(live, wire=None):
    handle(wire or live.wire, HANDLE)
    await live.wait(lambda: live.host.state.get('resume_handle') == HANDLE, 30)


async def committed(live, held):
    live.mail.put({'text': MAIL, 'msg_id': 'typed-mail'})
    await live.wait(held.entered.is_set, 30)
    assert canonical(live) == 1 and not written(live, MAIL), 'canonical, and never written'


async def lost_around_the_drain(live, held):
    """The words are committed, the connection is lost around the drain, and its successor comes up."""
    wire, count = live.wire, len(live.provider.connections)
    await committed(live, held)
    await wire.websocket.close(**RESTART)
    await live.wait(lambda: len(live.provider.connections) > count and live.call.connected.is_set(), 30)
    assert held.cancelled.is_set(), 'the worker really was cancelled around its drain'
    return live.provider.connections[count]


def resumed_setup(wire):
    return wire.sent[0]['setup']['sessionResumption'] == {'handle': HANDLE}


def words_then_pointer(wire):
    words, pointer = live_frames(wire, MAIL), room_frames(wire)
    assert len(words) == 1, 'the words themselves, written once on the conversation that never saw them'
    assert len(pointer) == 1 and wire.sent.index(words[0]) < wire.sent.index(pointer[0])
    assert not words[0]['clientContent'].get('turnComplete'), 'the pointer asks, the words do not'


async def later_fresh_loss(live, monkeypatch, wire, *, owed):
    wire.put({'sessionResumptionUpdate': {'resumable': False}})
    await live.wait(lambda: not live.host.state.get('resume_handle'), 30)
    answers_only_on(monkeypatch)
    fresh = await fresh_after(live, wire)
    assert fresh.sent[0]['setup']['sessionResumption'] == {} and MAIL in json.dumps(fresh.sent[0]), 'its fit holds the words'
    await asked(live, fresh, owed)
    assert written(live, MAIL) == 1 and canonical(live) == 1 and no_private_marker_on_any_wire(live)


@pytest.mark.parametrize('answered', [True, False], ids=['answered-there', 'unanswered'])
def test_real_cancelled_drain_reaches_the_resumed_conversation_as_its_words(host_world, monkeypatch, answered):
    """The gap this module exists for: a pointer alone would send the conversation to words it never received."""
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            answers_only_on(monkeypatch, *((2,) if answered else ()))
            await published(live)
            async with held_drain(monkeypatch, live) as held:
                said = spoken(live)
                second = await lost_around_the_drain(live, held)
            assert resumed_setup(second) and live.call.conversation == 1
            await live.wait(lambda: room_frames(second), 30)
            words_then_pointer(second)
            if answered:
                await live.wait(lambda: spoken(live) == said + 1, 30)
                await spoken_out(live)
            await later_fresh_loss(live, monkeypatch, second, owed=not answered)
            assert spoken(live) == said + int(answered)
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


def test_real_cancelled_drain_is_held_by_a_fresh_fit_and_never_written(host_world, monkeypatch):
    """The opposite ordering: no handle, so the next connection's fit holds the words and one pointer asks. One answer."""
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            answers_only_on(monkeypatch, 2)
            async with held_drain(monkeypatch, live) as held:
                said = spoken(live)
                second = await lost_around_the_drain(live, held)
            assert second.sent[0]['setup']['sessionResumption'] == {} and live.call.conversation == 2
            assert MAIL in json.dumps(second.sent[0]), 'its fit holds the words'
            await live.wait(lambda: room_frames(second) and spoken(live) == said + 1, 30)
            await spoken_out(live)
            assert len(room_frames(second)) == 1 and not written(live, MAIL) and canonical(live) == 1
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


def test_real_words_the_worker_received_are_not_written_again_on_a_resume(host_world, monkeypatch):
    """The other opposite ordering: the waiter got them, wrote them and was then lost. Nothing is kept for the resume."""
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            answers_only_on(monkeypatch)
            first = live.wire
            await published(live)
            live.mail.put({'text': MAIL, 'msg_id': 'typed-mail'})
            await live.wait(lambda: live_frames(first, MAIL), 30)
            handle(first, 'published-after-the-mail')
            await live.wait(lambda: live.host.state.get('resume_handle') == 'published-after-the-mail', 30)
            second = await fresh_after(live, first)
            assert second.sent[0]['setup']['sessionResumption'] == {'handle': 'published-after-the-mail'}
            await asyncio.sleep(.3)
            assert not live_frames(second, MAIL) and not room_frames(second) and live.call.conversation == 1
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


def test_real_turn_left_open_across_the_resume_holds_the_words_and_does_not_answer_them(host_world, monkeypatch):
    """Earlier words left a turn open, so the resumed adapter only accepts the claimed ones. That silent turn ends and they are
    written in one conversion: it ended before them, so they stay owed and a pointer asks. The response to it settles them."""
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            answers_only_on(monkeypatch)
            first = live.wire
            await published(live)
            live.mail.put({'text': LATER, 'msg_id': 'earlier-mail'})
            await live.wait(lambda: live_frames(first, LATER), 30)
            async with held_drain(monkeypatch, live) as held:
                said = spoken(live)
                second = await lost_around_the_drain(live, held)
            assert resumed_setup(second) and live.call.conversation == 1
            await live.wait(lambda: live.call.session.held_context, 30)
            assert not live_frames(second, MAIL) and not live.call.pending_context_rows, 'accepted, not written'
            answers_only_on(monkeypatch, 2)
            second.complete()  # a usage-only turn end
            await live.wait(lambda: room_frames(second) and spoken(live) == said + 1, 30)
            words_then_pointer(second)
            await spoken_out(live)
            await later_fresh_loss(live, monkeypatch, second, owed=False)
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


def test_real_catch_up_lost_with_its_socket_stays_with_the_call(host_world, monkeypatch):
    """The resumed connection dies under the write. The words stay in custody and the next resume writes them, once."""
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            answers_only_on(monkeypatch)
            await published(live)
            fired = close_on_write(monkeypatch, live, lambda payload: 'clientContent' in payload and MAIL in json.dumps(payload),
                                   **RESTART)
            async with held_drain(monkeypatch, live) as held:
                second = await lost_around_the_drain(live, held)
            await live.wait(lambda: fired and len(live.provider.connections) == 3 and live.call.connected.is_set(), 30)
            third = live.wire
            await live.wait(lambda: room_frames(third), 30)
            assert resumed_setup(third) and live.call.conversation == 1 and not live_frames(second, MAIL)
            words_then_pointer(third)
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


def test_real_cancelled_drain_beside_a_written_result_gets_one_answer_and_no_repeat(host_world, monkeypatch):
    """A result written before the loss, words claimed around it, a resume. One response begun after both settles both: the
    result is not written again, its effect is not run again, and a later fresh loss asks nothing."""
    from tests._live_stack import stack
    words = 'Answered the result and the room together.'

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            answers_only_on(monkeypatch)
            first = live.wire
            first.tool('kept-call', 'get_task_result', {'task_id': 'none'})
            await live.wait(lambda: native_output(first, 'kept-call') and not live.call.tools_running, 30)
            await published(live)
            async with held_drain(monkeypatch, live) as held:
                second = await lost_around_the_drain(live, held)
            assert resumed_setup(second) and live.call.conversation == 1
            await live.wait(lambda: live_frames(second, MAIL), 30)
            second.say(words)
            await live.wait(lambda: any(v.get('text') == words and v.get('final') for v in live.page.controls), 30)
            await spoken_out(live)
            assert not native_output(second, 'kept-call') and live.call.answer_due is None and live.call.room_due is None
            assert not room_frames(second), 'answered before it was offered'
            await later_fresh_loss(live, monkeypatch, second, owed=False)
            assert sum(row.get('tool_call_id') == 'kept-call' for row in live.host.messages if row.get('role') == 'tool') == 1
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


def test_real_end_around_a_cancelled_drain_opens_nothing_and_writes_nothing(host_world, monkeypatch):
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini') as live:
            first = live.wire
            await published(live)
            async with held_drain(monkeypatch, live) as held:
                await committed(live, held)
                live.page.end()
                deadline = time.monotonic() + 30
                while not live.running.done() and time.monotonic() < deadline:
                    await asyncio.sleep(.02)
            assert live.running.done() and held.cancelled.is_set()
            assert len(live.provider.connections) == 1 and not live_frames(first, MAIL) and canonical(live) == 1
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())
