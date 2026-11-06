"""One connection-owned worker sends what waited for a connection.

A withheld output or pending mail has one custody holder and one sender across measurement, cancellation, fresh adoption
and native resume. The reader and the microphone never wait on delivery, so a failed write always reaches the reader's
classification and End still ends the call. The real driver over its fake port; ``test_live_delivery_owner_stack.py``
holds the same contract on the real host and adapters.
"""
import asyncio
from contextlib import asynccontextmanager

import pytest
from websockets.exceptions import ConnectionClosed

from ouroboros import call_protocol as cp
from ouroboros.contracts.live_conversation import Audio, ContextUpdate, Interrupt, Respond, ToolOutputs, Video
from ouroboros.contracts.realtime import CatalogModel, RealtimeSetupError
from tests.test_live_conversation_driver import Batch, Host, until
from tests.test_live_deferred_custody import (MAIL, READY, answer, attach, held_write, holding_mail, holding_result,
                                              mail_updates, outputs, pointers, settled, successor, withheld)
from tests.test_live_tool_continuity import Adapter, RECEIPT, look_call

ROOM_POINTER = 'New room context'


class Socket(Adapter):
    """The fake port with a socket that can die: as on a real websocket, every write after that raises."""

    def __init__(self, during=None, *, dies_on=None):
        super().__init__(during)
        self.dies_on, self.dead, self.refused = dies_on, False, []

    async def send(self, command):
        if self.dies_on is not None and self.dies_on(command):
            self.dead = True
        if self.dead:
            self.refused.append(command)
            raise ConnectionClosed(None, None)
        await super().send(command)


def dying_on(kind, text=''):
    return lambda during: Socket(during, dies_on=lambda command: isinstance(command, kind) and text in str(command))


def room_pointers(session):
    return sum(ROOM_POINTER in str(command) for command in session.sent)


def position(session, text):
    return next(i for i, command in enumerate(session.sent) if text in str(command))


def never(rows):
    return False


def done_batch(calls, *, silent=()):
    return asyncio.sleep(0, Batch([{'role': 'tool', 'tool_call_id': call['id'], 'content': f"Done {call['id']}."}
                                   for call in calls], silent=silent))


def one_mail(world, text=MAIL):
    rows = [[{'role': 'user', 'content': text}]]

    async def mailbox():
        return rows.pop() if rows else []
    world.host.mailbox = mailbox


@asynccontextmanager
async def opened(*, make=Socket, holds=holding_mail, fit_failed=True, ready=True, host=None, call=None):
    """A live call on the real driver. ``fit_failed``: a speculative fit already failed for this history, so what
    a due renewal would have held goes to the worker instead, measured by a gate the scenario can hold open."""
    from skills.live_conversation.driver import _Call
    host = host or Host()
    if call is None:
        host.tool_batch = done_batch
        call = _Call(host)
    world = attach(host, call, make=make)
    world.holds = holds
    if fit_failed:
        call.failed_preparation_revision = 0
    world.task = asyncio.create_task(call.run())
    try:
        await until(lambda: host.sessions)
        world.first = host.sessions[0]
        if ready:
            world.first.emit('ready', **READY)
            await until(lambda: call.connected.is_set())
        yield world
    finally:
        world.release.set()
        if world.opening is not None:
            world.opening.set()
        host.hub.ended_by_owner = True
        if not world.task.done():
            await asyncio.wait_for(world.task, 2)


def waiting_mail(world):
    """The real route: a fit is due, and refused for this history. The worker now owns the mail."""
    world.host.due = True
    return world.call.context_update([{'role': 'user', 'content': MAIL}])


@pytest.mark.parametrize('resumed', [False, True])
def test_the_reader_and_the_microphone_go_on_while_a_delivery_is_measured(resumed):
    """A slow measure holds up nothing else: events are read, audio goes up, words are recorded."""
    async def scenario():
        async with opened(holds=holding_result) as world:
            host, _call, session = world.host, world.call, world.first
            host.due = True
            session.emit('tool_call', call_id='kept', name='quick_tool', arguments='{}')
            await asyncio.wait_for(world.entered.wait(), 2)
            if resumed:
                world.entered.clear()
                session = await successor(world, session, resumes=True)
                await asyncio.wait_for(world.entered.wait(), 2)
            session.emit('response_started', response_id='talk')
            session.emit('caption', speaker='owner', item_id='owner-1', text='Words meanwhile.', final=True)
            session.emit('response_done', response_id='talk', **RECEIPT)
            await session.handled()
            host.hub.events.append(cp.PeerAudio('peer', cp.PEER_OWNER, b'\x00\x00' * 480))
            await until(lambda: any(isinstance(command, Audio) for command in session.sent))
            await until(lambda: any(row.get('text') == 'Words meanwhile.' for row in host.captions))
            assert any(entry[:2] == ('done', 'talk') for entry in host.usage) and not outputs(session)
            world.release.set()
            await until(lambda: session.finished == 1)
            assert len(outputs(session)) == 1 and not pointers(session)
    asyncio.run(scenario())


@pytest.mark.parametrize(("item", "close", "next_connection"), [
    ('output', 'writer_first', 'resumed'),
    ('output', 'reader_first', 'resumed'),
    ('mail', 'writer_first_with_microphone', 'fresh'),
])
def test_a_release_on_a_resumed_connection_whose_socket_closes(item, close, next_connection):
    """Whichever side of the driver sees the close first, the reader classifies it and the call goes on. The item stays in
    custody until a write succeeds: a resume gets it once, before any pointer, and a fresh fit holds it and owes one answer."""
    is_output = item == 'output'

    async def scenario():
        async with opened(holds=holding_result if is_output else holding_mail) as world:
            host, call, first = world.host, world.call, world.first
            if is_output:
                host.due = True
                first.emit('tool_call', call_id='kept', name='quick_tool', arguments='{}')
            else:
                await waiting_mail(world)
            await asyncio.wait_for(world.entered.wait(), 2)
            world.entered.clear()
            world.make = dying_on(ToolOutputs) if is_output else dying_on(ContextUpdate, MAIL)
            second = await successor(world, first, resumes=True)
            assert host.options[1].resume_handle == 'covering'
            await asyncio.wait_for(world.entered.wait(), 2)
            world.make = Socket
            resumes = next_connection == 'resumed'
            if close == 'reader_first':
                third = await successor(world, second, resumes=resumes)
                world.release.set()
                await asyncio.sleep(.02)
                assert not second.refused, 'the worker that was measuring died with its connection'
            else:
                world.release.set()
                await until(lambda: second.refused)
                await second.handled()  # the reader is reading again, with nothing left to classify yet
                if close == 'writer_first_with_microphone':
                    host.hub.events.append(cp.PeerAudio('peer', cp.PEER_OWNER, b'\x00\x00' * 480))
                    await until(lambda: any(isinstance(command, Audio) for command in second.refused))
                third = await successor(world, second, resumes=resumes)
            assert not world.task.done()
            assert host.options[2].resume_handle == ('covering' if resumes else '')
            if is_output and resumes:
                await until(lambda: third.finished == 1)
                assert not pointers(third)
                third_outputs = 1
            elif is_output:
                await until(lambda: pointers(third) == 1)
                third_outputs = 0
            else:
                await until(lambda: room_pointers(third) == 1)
                assert len(mail_updates(third)) == int(resumes)
                if resumes:
                    assert mail_updates(third)[0].respond is False
                    assert position(third, MAIL) < position(third, ROOM_POINTER), 'mail precedes its answer pointer'
                third_outputs = 0
            await answer(third, 'answered')
            assert call.answer_due is None and not call.deferred_tool_outputs and not call.pending_context_rows
            await asyncio.sleep(.03)
            assert len(outputs(third)) == third_outputs and pointers(third) <= 1 and room_pointers(third) <= 1
            assert not outputs(first) and not outputs(second) and not mail_updates(first) and not mail_updates(second)
    asyncio.run(scenario())


@pytest.mark.parametrize(("item", "next_connection"), [('output', 'resumed'), ('mail', 'fresh')])
def test_a_first_release_whose_socket_closes_keeps_the_call(item, next_connection):
    """The route that used to escape the call's global tool worker: no resume before it, nothing but a refused fit."""
    is_output, resumes = item == 'output', next_connection == 'resumed'

    async def scenario():
        make = dying_on(ToolOutputs) if is_output else dying_on(ContextUpdate, MAIL)
        async with opened(make=make, holds=never) as world:
            host, call, first = world.host, world.call, world.first
            if is_output:
                host.due = True
                first.emit('tool_call', call_id='kept', name='quick_tool', arguments='{}')
            else:
                await waiting_mail(world)
            await until(lambda: first.refused)
            await first.handled()
            assert not world.task.done() and (call.deferred_tool_outputs or call.pending_context_rows)
            world.make = Socket
            second = await successor(world, first, resumes=resumes)
            if is_output and resumes:
                await until(lambda: second.finished == 1)
            else:
                await until(lambda: pointers(second) + room_pointers(second) == 1)
            assert len(outputs(second)) == int(is_output and resumes)
            assert len(mail_updates(second)) == int(not is_output and resumes)
            assert not call.deferred_tool_outputs and not call.pending_context_rows
    asyncio.run(scenario())


def test_typed_words_do_not_wait_behind_a_measured_output():
    """Typed mail is immediate. Only older pending mail, or a fit it must wait for, ever holds it back."""
    async def scenario():
        async with opened(holds=holding_result) as world:
            host, call, first = world.host, world.call, world.first
            host.due = True
            first.emit('tool_call', call_id='kept', name='quick_tool', arguments='{}')
            await asyncio.wait_for(world.entered.wait(), 2)
            one_mail(world)
            await until(lambda: mail_updates(first))
            assert mail_updates(first)[0].respond is True and not outputs(first) and not call.pending_context_rows
            world.release.set()
            await until(lambda: first.finished == 1)
            assert len(mail_updates(first)) == 1 and not room_pointers(first)
    asyncio.run(scenario())


def test_a_refit_obligation_keeps_what_it_withheld_from_the_retiring_connection():
    """Nothing is measured or written while that obligation stands, whatever else says delivery is due. A fresh fit holds it."""
    async def scenario():
        asked = asyncio.Event()

        class Speculative(Socket):
            silent_context_updates = True

        async def held_fit(options):
            asked.set()
            await asyncio.Future()

        host = Host()
        host.prepare_session = held_fit
        async with opened(make=lambda during: Speculative(during), holds=holding_result, fit_failed=False, host=host) as world:
            call, first = world.call, world.first
            host.due = True
            first.emit('tool_call', call_id='kept', name='quick_tool', arguments='{}')
            await asyncio.wait_for(asked.wait(), 2)
            await until(lambda: call.deferred_tool_outputs and call.tool_refit.is_set() and not call.tools_running)
            host.due = False
            call.delivery_due.set()
            await asyncio.sleep(.05)
            assert not world.entered.is_set() and not outputs(first) and call.deferred_tool_outputs
    asyncio.run(scenario())


@pytest.mark.parametrize('item', ['output', 'mail'])
def test_a_resumed_conversation_is_delivered_to_even_with_a_fit_on_its_way(item):
    """A refit request outlives the resume, and that is no reason to keep the conversation waiting for what it is owed:
    the result reaches it natively at once. Pointers still wait for the fit."""
    is_output = item == 'output'

    async def scenario():
        asked = asyncio.Event()

        class Speculative(Socket):
            silent_context_updates = True

        async def held_fit(options):
            asked.set()
            await asyncio.Future()

        host = Host()
        host.prepare_session = held_fit
        async with opened(make=lambda during: Speculative(during), holds=never, fit_failed=False, host=host) as world:
            call, first = world.call, world.first
            if is_output:
                host.due = True
                first.emit('tool_call', call_id='kept', name='quick_tool', arguments='{}')
            else:
                await waiting_mail(world)
            await asyncio.wait_for(asked.wait(), 2)
            await until(lambda: call.deferred_tool_outputs or call.pending_context_rows)
            host.due = False
            world.make = lambda during: Speculative(during)
            second = await successor(world, first, resumes=True)
            if is_output:
                await until(lambda: second.finished == 1)
            else:
                await until(lambda: mail_updates(second))
            assert call.renewing and not call.resume, 'the fresh fit is still on its way'
            assert len(outputs(second)) == int(is_output) and len(mail_updates(second)) == int(not is_output)
            assert not pointers(second) and not room_pointers(second)
    asyncio.run(scenario())


def test_a_measure_spanning_a_native_resume_writes_its_mail_once():
    """Two connections measure the same mail in turn. Only the one that is alive when its measure ends can write it."""
    async def scenario():
        async with opened() as world:
            await waiting_mail(world)
            await asyncio.wait_for(world.entered.wait(), 2)
            world.entered.clear()
            second = await successor(world, world.first, resumes=True)
            await asyncio.wait_for(world.entered.wait(), 2)
            world.release.set()
            await until(lambda: room_pointers(second) == 1)
            assert [update.messages for update in mail_updates(second)] == [({'role': 'user', 'content': MAIL},)]
            assert len(second.held_context_rows) == 2 and not mail_updates(world.first)
    asyncio.run(scenario())


def test_mail_arriving_during_a_measure_follows_it_in_order_and_each_is_written_once():
    async def scenario():
        async with opened() as world:
            call, first = world.call, world.first
            await waiting_mail(world)
            await asyncio.wait_for(world.entered.wait(), 2)
            later = {'role': 'user', 'content': 'Later mail.'}
            await call.context_update([later])
            assert [row['content'] for row in call.pending_context_rows] == [MAIL, 'Later mail.']
            await asyncio.sleep(.03)
            assert not room_pointers(first), 'no pointer before the mail it points to'
            world.release.set()
            await until(lambda: not call.pending_context_rows and room_pointers(first))
            written = [row['content'] for command in first.sent if isinstance(command, ContextUpdate) and not command.respond
                       for row in command.messages]
            assert written == [MAIL, 'Later mail.']
            assert position(first, 'Later mail.') < position(first, ROOM_POINTER)
    asyncio.run(scenario())


@pytest.mark.parametrize('next_connection', ['resumed', 'fresh'])
def test_a_direct_mail_write_that_fails_returns_to_custody_with_its_pointer(next_connection):
    """Typed words go at once; a closed socket leaves them owed. The mailbox worker is connection-owned, so the reader
    classifies this. A resume gets the words before their pointer, a fresh fit holds them and is still asked to answer."""
    resumes = next_connection == 'resumed'

    async def scenario():
        async with opened(make=dying_on(ContextUpdate, MAIL), holds=never, fit_failed=False) as world:
            call, first = world.call, world.first
            one_mail(world)
            await until(lambda: first.refused)
            await until(lambda: call.pending_context_rows)
            assert not call.response_open and room_pointers(first) == 0
            world.make = Socket
            second = await successor(world, first, resumes=resumes)
            await until(lambda: room_pointers(second) == 1)
            assert len(mail_updates(second)) == int(resumes) and not call.pending_context_rows
            if resumes:
                assert position(second, MAIL) < position(second, ROOM_POINTER)
    asyncio.run(scenario())


@pytest.mark.parametrize('next_connection', ['resumed', 'fresh'])
def test_a_direct_mail_write_cancelled_with_its_connection_is_returned_before_the_next_one(next_connection):
    """The write is in flight when the connection ends. Its rows return while the old connection is joined, so a fresh
    consume clears them for good and a resume writes them once: never restored onto a session whose fit holds them."""
    resumes = next_connection == 'resumed'

    async def scenario():
        entered = asyncio.Event()

        class Stuck(Socket):
            async def send(self, command):
                if isinstance(command, ContextUpdate) and MAIL in str(command):
                    entered.set()
                    await asyncio.Future()
                await super().send(command)

        async with opened(make=lambda during: Stuck(during), holds=never, fit_failed=False) as world:
            call, first = world.call, world.first
            one_mail(world)
            await asyncio.wait_for(entered.wait(), 2)
            assert not call.pending_context_rows, 'only the write holds them'
            world.make = Socket
            second = await successor(world, first, resumes=resumes)
            await until(lambda: room_pointers(second) == 1)
            assert len(mail_updates(second)) == int(resumes) and not call.pending_context_rows
    asyncio.run(scenario())


@pytest.mark.parametrize('next_connection', ['resumed', 'fresh'])
def test_mail_arriving_before_ready_reaches_that_connection_before_its_pointer(next_connection):
    """After adoption and before ready the words are in no fit. A fresh connection needs them written as much as a resumed one."""
    async def scenario():
        async with opened(holds=holding_mail, fit_failed=False) as world:
            host, call, first = world.host, world.call, world.first
            world.opening = asyncio.Event()
            resumes = next_connection == 'resumed'
            if resumes:
                first.emit('resumable', handle='covering')
            first.emit('closed', resumable=resumes, transport_loss=True, reason='transport_lost')
            await until(lambda: call.session is None)
            one_mail(world)
            world.opening.set()
            await until(lambda: len(host.sessions) == 2 and call.pending_context_rows)
            second = host.sessions[1]
            second.emit('ready', **READY)
            await asyncio.wait_for(world.entered.wait(), 2)
            await asyncio.sleep(.03)
            assert not room_pointers(second), 'a slow measure cannot let the pointer overtake the words'
            world.release.set()
            await until(lambda: room_pointers(second) == 1)
            assert len(mail_updates(second)) == 1 and position(second, MAIL) < position(second, ROOM_POINTER)
    asyncio.run(scenario())


@pytest.mark.parametrize('fits', [True, False])
def test_failed_fit_mail_goes_through_the_serving_measure(fits):
    """Words precede their pointer on the serving connection; words that do not fit it end the call by their typed reason."""
    async def scenario():
        async with opened(holds=never) as world:
            world.verdict = fits
            await waiting_mail(world)
            if fits:
                await until(lambda: room_pointers(world.first) == 1)
                assert len(mail_updates(world.first)) == 1
                assert position(world.first, MAIL) < position(world.first, ROOM_POINTER)
            else:
                with pytest.raises(RealtimeSetupError, match='window'):
                    await asyncio.wait_for(world.task, 2)
                assert not mail_updates(world.first) and world.call.pending_context_rows
    asyncio.run(scenario())


@pytest.mark.parametrize('stage', ['measure', 'write'])
def test_end_during_a_release_ends_the_call_at_once(stage):
    """End cancels the worker wherever it is. A stalled native write cannot hold the call open."""
    async def scenario():
        write = held_write(False)
        async with withheld(writes=[write.during], holds=holding_result if stage == 'measure' else never) as world:
            await asyncio.wait_for((world.entered if stage == 'measure' else write.entered).wait(), 2)
            world.host.hub.ended_by_owner = True
            await asyncio.wait_for(world.task, 1)
            assert world.first.finished == 0 and world.call.deferred_tool_outputs
    asyncio.run(scenario())


def test_a_goodbye_writes_no_further_native_result():
    """A failed fit would release the withheld result, but a native result could cut the goodbye short."""
    async def scenario():
        opening, prepared = asyncio.Event(), asyncio.Event()

        class Speculative(Socket):
            silent_context_updates = True

        async def refused_fit(options):
            opening.set()
            await prepared.wait()
            raise RealtimeSetupError('context_capacity', 'Synthetic capacity failure')

        host = Host()
        host.prepare_session = refused_fit
        async with opened(make=lambda during: Speculative(during), holds=never, fit_failed=False, host=host) as world:
            host.due = True
            world.first.emit('tool_call', call_id='kept', name='quick_tool', arguments='{}')
            await asyncio.wait_for(opening.wait(), 2)
            await until(lambda: world.call.deferred_tool_outputs)
            host.due = False
            host.hub.end_by_ouroboros()
            prepared.set()
            await asyncio.wait_for(world.task, 2)
            assert not outputs(world.first)
    asyncio.run(scenario())


def test_a_planned_resume_waits_for_a_write_in_flight_instead_of_guessing():
    """Cancelling this write and repeating it on the resumed socket could repeat a native output. Nothing else waits on it."""
    async def scenario():
        write = held_write(False)
        async with withheld(silent=('kept',), writes=[write.during], holds=never) as world:
            call, first = world.call, world.first
            await asyncio.wait_for(write.entered.wait(), 2)
            first.emit('resumable', handle='after-write-began')
            first.emit('renew', within_s=30, resume=True, reason='go_away')
            await until(lambda: call.renewing)
            await asyncio.sleep(.05)
            assert len(world.host.sessions) == 1 and 'delivering' in call.quiet_blockers()
            write.release.set()
            await until(lambda: len(world.host.sessions) == 2)
            second = world.host.sessions[1]
            second.emit('ready', **READY)
            await asyncio.sleep(.03)
            assert first.finished == 1 and not outputs(second) and not call.deferred_tool_outputs
            assert world.host.options[1].resume_handle == 'after-write-began'
    asyncio.run(scenario())


def test_a_later_result_waits_behind_the_one_being_measured():
    """Native order is the order the batches finished, with one sender for both and one answer owed."""
    async def scenario():
        async with opened(holds=holding_result) as world:
            call, host, first = world.call, world.host, world.first
            async def batch(calls):
                return await done_batch(calls)
            host.tool_batch = batch
            world.holds = lambda rows: any(row.get('tool_call_id') == 'earlier' for row in rows)
            host.due = True
            first.emit('tool_call', call_id='earlier', name='quick_tool', arguments='{}')
            await asyncio.wait_for(world.entered.wait(), 2)
            first.emit('tool_call', call_id='later', name='quick_tool', arguments='{}')
            await until(lambda: len(call.deferred_tool_outputs) == 2 and not call.tools_running)
            assert not outputs(first)
            world.release.set()
            await until(lambda: first.finished == 2)
            assert [command.messages[0]['tool_call_id'] for command in outputs(first)] == ['earlier', 'later']
            await asyncio.sleep(.03)
            assert not pointers(first) and not call.deferred_tool_outputs
            await answer(first, 'answered')
            assert call.answer_due is None
    asyncio.run(scenario())


def test_an_admitted_output_is_not_measured_again_on_its_resume():
    """It needed no fit when its write met the closed socket, and a resume is the same conversation: no second verdict."""
    async def scenario():
        measured = []
        async with opened(make=dying_on(ToolOutputs), holds=never, fit_failed=False) as world:
            call, host, first = world.call, world.host, world.first

            async def spy(rows, schemas):
                measured.append(rows)
                return False
            host.delivery_fits = spy
            first.emit('tool_call', call_id='kept', name='quick_tool', arguments='{}')
            await until(lambda: first.refused and call.deferred_tool_outputs)
            await first.handled()
            assert len(first.refused) == 1, 'no retry on a socket the reader has yet to classify'
            world.make = Socket
            second = await successor(world, first, resumes=True)
            await until(lambda: second.finished == 1)
            await settled(world)
            assert not measured and len(outputs(second)) == 1 and not pointers(second)
    asyncio.run(scenario())


@pytest.mark.parametrize('item', ['output', 'mail'])
def test_a_refusal_while_a_fresh_fit_is_on_its_way_waits_for_that_fit(item):
    """The fit holds what the old connection refused. Only if the fit itself fails does the refusal stand, typed as ever."""
    is_output = item == 'output'

    async def scenario():
        asked, prepared = asyncio.Event(), asyncio.Event()

        class Speculative(Socket):
            silent_context_updates = True

        async def refused_fit(options):
            asked.set()
            await prepared.wait()
            raise RealtimeSetupError('context_capacity', 'Synthetic capacity failure')

        host = Host()
        host.prepare_session = refused_fit
        held = holding_result if is_output else holding_mail
        async with opened(make=lambda during: Speculative(during), holds=held, host=host) as world:
            call, first = world.call, world.first
            if is_output:
                host.due = True
                first.emit('tool_call', call_id='kept', name='quick_tool', arguments='{}')
            else:
                await waiting_mail(world)
            await asyncio.wait_for(world.entered.wait(), 2)
            call.failed_preparation_revision = None  # history moved on: this request earns its fit
            call.request_renewal()
            await asyncio.wait_for(asked.wait(), 2)
            world.verdict = False
            world.release.set()
            await asyncio.sleep(.05)
            assert not world.task.done() and (call.deferred_tool_outputs or call.pending_context_rows)
            assert not outputs(first) and not mail_updates(first)
            prepared.set()
            with pytest.raises(RealtimeSetupError, match='window'):
                await asyncio.wait_for(world.task, 2)
            assert not outputs(first) and not mail_updates(first)
            assert call.deferred_tool_outputs or call.pending_context_rows, 'custody is retained through a typed end'
    asyncio.run(scenario())


def test_a_fresh_session_starts_with_nothing_awaited_even_on_a_withdrawn_resume():
    """A response in flight at the loss can never finish here. Left behind, it would hold quiet shut for the rest of the call."""
    async def scenario():
        async with opened(fit_failed=False) as world:
            host, call, first = world.host, world.call, world.first
            first.emit('response_started', response_id='old-response')
            await first.handled()
            draining, drained = asyncio.Event(), asyncio.Event()

            async def caption(**row):
                draining.set()
                await drained.wait()
            host.caption = caption
            first.emit('caption', speaker='owner', item_id='owner-1', text='Words.', final=True)
            await asyncio.wait_for(draining.wait(), 2)
            first.emit('resumable', handle='covering')
            first.emit('closed', resumable=True, transport_loss=True, reason='transport_lost')
            await until(lambda: call.provider_closed)
            host.state.pop('resume_handle')
            drained.set()
            await until(lambda: len(host.sessions) == 2)
            second = host.sessions[1]
            second.emit('ready', **READY)
            await until(lambda: call.connected.is_set() and call.session is second)
            assert call.conversation == 2 and not call.active_responses
            second.emit('response_started', response_id='new-response')
            second.emit('response_done', response_id='new-response', **RECEIPT)
            await second.handled()
            assert call.quiet()
            host.hub.events.append(cp.PeerControl('peer', cp.PEER_OWNER, {'type': 'context', 'text': 'Peer fact.'}))
            await until(lambda: 'Peer fact.' in str(second.sent))
    asyncio.run(scenario())


@pytest.mark.parametrize('written', [False, True], ids=['never-written', 'written-then-lost'])
def test_a_look_whose_answer_crosses_a_fresh_switch_keeps_asking_for_its_pixels(written):
    """Whichever way the fresh session comes to owe a look's answer, its one pointer still asks for what its fit lacks."""
    async def scenario():
        host = Host()
        call = look_call(host)
        async with opened(host=host, call=call, holds=lambda rows: not written, fit_failed=False) as world:
            first = world.first
            first.emit('tool_call', call_id='look', name='attend', arguments='{"look":true}')
            if written:
                await until(lambda: first.finished == 1)
                world.make = Socket
                second = await successor(world, first)
            else:
                await asyncio.wait_for(world.entered.wait(), 2)
                first.emit('resumable', handle='covering')
                first.emit('closed', resumable=True, transport_loss=True, reason='transport_lost')
                await until(lambda: not call.connected.is_set() and call.resume)
                world.stale_resume = True  # promised, and refused only once the output's turn has passed
                world.release.set()
                await until(lambda: len(host.sessions) == 2)
                second = host.sessions[1]
                second.emit('ready', **READY)
            await until(lambda: pointers(second) == 1)
            assert any('_requested_look' in str(command) for command in second.sent if 'batch returned' in str(command))
            assert not outputs(second) and 'look' not in call.issued
    asyncio.run(scenario())


# --- the reader's own writes -----------------------------------------------------------------------------------------


@pytest.mark.parametrize('resumes', [False, True], ids=['fresh', 'resumed'])
def test_an_interruption_meeting_a_closed_socket_leaves_the_reader_reading(resumes):
    """The page dropped that audio whatever the socket could still be told: local truth first, the loss classified next."""
    async def scenario():
        async with opened(make=dying_on(Interrupt), fit_failed=False) as world:
            host, call, first = world.host, world.call, world.first
            first.emit('audio', item_id='spoken', rate=24000, pcm=b'\x00\x00' * 480)
            first.emit('caption', speaker='ouroboros', item_id='spoken', text='Cut off by the owner.', final=True)
            await first.handled()
            assert 'spoken' in call.pending_spoken
            first.emit('speech_started')
            await first.handled()
            assert first.refused and 'spoken' in call.interrupted and not call.pending_spoken
            await until(lambda: any(row.get('interrupted') for row in host.captions))
            world.make = Socket
            await successor(world, first, resumes=resumes)
            assert not world.task.done() and not call.unheard() and call.conversation == (1 if resumes else 2)
    asyncio.run(scenario())


def test_an_opening_that_never_reached_a_socket_is_given_by_the_replacement():
    async def scenario():
        async with opened(make=dying_on(Respond), fit_failed=False, ready=False) as world:
            host, call, first = world.host, world.call, world.first
            first.emit('ready', **READY)
            await until(lambda: first.refused)
            await first.handled()
            assert not call.connected.is_set() and not host.state.get('greeted')
            world.make = Socket
            second = await successor(world, first)
            assert [command for command in second.sent if isinstance(command, Respond)] == [Respond(host.opening)]
            assert host.state.get('greeted')
    asyncio.run(scenario())


@pytest.mark.parametrize('vision', [True, False], ids=['pixels', 'blind-fact'])
def test_frames_at_an_input_boundary_survive_a_closed_socket(vision):
    """Pixels are offered again at the next boundary; a blind frame fact is mail, in custody until written."""
    async def scenario():
        host = Host()
        host.model = CatalogModel('test', 'Test', vision=vision, honours=frozenset({'pause', 'interruptions'}))
        kind = Video if vision else ContextUpdate
        async with opened(make=dying_on(kind), holds=never, fit_failed=False, host=host) as world:
            call, first = world.call, world.first
            call.attention.frame(cp.PeerVideo('remote', cp.PEER_OWNER, 'image/jpeg', b'exact-frame'))
            first.emit('input_started')
            await until(lambda: first.refused)
            await first.handled()
            assert not world.task.done()
            world.make = Socket
            second = await successor(world, first, resumes=True)
            if vision:
                assert 'remote' not in call.attention.seen
                second.emit('input_started')
                await until(lambda: any(isinstance(command, Video) for command in second.sent))
                assert sum(isinstance(command, Video) for command in second.sent) == 1
            else:
                await until(lambda: room_pointers(second) == 1)
                assert sum('could not be saved' in str(command) for command in second.sent) == 1
                assert position(second, 'could not be saved') < position(second, ROOM_POINTER)
    asyncio.run(scenario())
