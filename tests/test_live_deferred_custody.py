"""A withheld native output belongs to its conversation through every wait of its delivery.

A result can be withheld, measured, perhaps shortened and written while its connection is replaced. A fresh session holds
the unit in its fit and owes the answer by itself, and may finish that answer before an old wait would have returned; a
native resume keeps the conversation and needs the output exactly once. Whatever ends late touches nothing it no longer
owns. Two layers: the real driver over its fake port, and the real host with loopback adapters. The worker that owns this
delivery has its own contract in ``test_live_delivery_owner.py``.
"""
import asyncio
import json
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from websockets.exceptions import ConnectionClosed

from ouroboros import call_protocol as cp
from ouroboros.contracts.live_conversation import ContextUpdate, ToolOutputs
from ouroboros.contracts.realtime import RealtimeSetupError
from tests._extension_loader_shared import _clear_loader_state  # noqa: F401
from tests._live_world import host_world as host_world
from tests.test_live_conversation_driver import Batch, Host, until
from tests.test_live_tool_continuity import Adapter, POINTER, RECEIPT, look_call, reached
from tests.test_live_tool_continuity_stack import native_output

READY = dict(input_rate=24000, output_rate=24000)
MAIL = 'Room mail measured for this connection.'


def pointers(session):
    return sum(POINTER in str(command) for command in session.sent)


def outputs(session):
    return [command for command in session.sent if isinstance(command, ToolOutputs)]


def mail_updates(session):
    return [command for command in session.sent if isinstance(command, ContextUpdate) and MAIL in str(command)]


def holding_mail(rows):
    return any(row.get('content') == MAIL for row in rows)


def holding_result(rows):
    return any(row.get('tool_call_id') == 'kept' for row in rows)


def attach(host, call, writes=(), make=Adapter):
    """Sessions with this driver's delivery record; ``world.opening`` keeps the next one from opening."""
    world = SimpleNamespace(host=host, call=call, writes=list(writes), opening=None, stale_resume=False, make=make,
                            entered=asyncio.Event(), release=asyncio.Event(), verdict=True, holds=holding_result)

    async def session(consume, options):
        if world.opening is not None and host.sessions:
            await world.opening.wait()
        if world.stale_resume and options.resume_handle:
            world.stale_resume = False
            raise RealtimeSetupError('prepared_stale', 'history moved while the resume opened')
        made = world.make(world.writes.pop(0) if world.writes else None)
        host.sessions.append(made)
        host.options.append(options)
        return await consume(made)

    async def delivery_fits(rows, schemas):
        host.due = False  # it was due once, which withheld this delivery
        if world.holds(rows):
            world.entered.set()
            await world.release.wait()
        return world.verdict
    host.session, host.delivery_fits = session, delivery_fits
    call.serving_catalog = lambda: ()
    return world


@asynccontextmanager
async def withheld(*, silent=(), writes=(), deferred=True, holds=holding_result, mail=False, make=Adapter):
    """One result, ``kept``, on the real driver. ``deferred``: a speculative fit already failed for this history, so
    its output waits for a serving measure, which ``holds`` keeps open until the scenario releases it."""
    from skills.live_conversation.driver import _Call
    host = Host()
    host.tool_batch = lambda calls: asyncio.sleep(0, Batch([{'role': 'tool', 'tool_call_id': 'kept', 'content': 'Done.'}],
                                                           silent=silent))
    call = _Call(host)
    world = attach(host, call, writes, make)
    world.holds = holds
    if deferred:
        call.failed_preparation_revision = 0
    world.task = asyncio.create_task(call.run())
    try:
        await until(lambda: host.sessions)
        world.first = host.sessions[0]
        world.first.emit('ready', **READY)
        await until(lambda: call.connected.is_set())
        if mail:
            call.pending_context_rows.append({'role': 'user', 'content': MAIL})
        host.due = deferred
        world.first.emit('tool_call', call_id='kept', name='quick_tool', arguments='{}')
        yield world
    finally:
        world.release.set()
        if world.opening is not None:
            world.opening.set()
        host.hub.ended_by_owner = True
        if not world.task.done():
            await asyncio.wait_for(world.task, 2)


async def successor(world, session, *, resumes=False):
    """The session drops; the next one opens and reports ready."""
    count = len(world.host.sessions)
    if resumes:
        session.emit('resumable', handle='covering')
    session.emit('closed', resumable=resumes, transport_loss=True, reason='transport_lost')
    await until(lambda: len(world.host.sessions) == count + 1)
    made = world.host.sessions[-1]
    made.emit('ready', **READY)
    await until(lambda: world.call.connected.is_set() and world.call.session is made)
    return made


async def answer(session, response):
    session.emit('response_started', response_id=response)
    session.emit('response_done', response_id=response, **RECEIPT)
    await session.handled()


async def settled(world):
    await until(lambda: not world.call.tools_running)
    await asyncio.sleep(.05)
    assert not world.task.done(), world.task.exception() if not world.task.cancelled() else 'the call ended'


@pytest.mark.parametrize('fits', [True, False])
def test_a_result_answered_on_the_successor_is_never_asked_again(fits):
    """The old measure returns after the answer finished, admitting or refusing: nothing re-arms it, nothing ends the call."""
    async def scenario():
        async with withheld() as world:
            shortened = []

            async def fallback(rows, schemas):
                shortened.append(rows)
                return rows
            world.host.tool_result_fallback = fallback
            await asyncio.wait_for(world.entered.wait(), 2)
            second = await successor(world, world.first)
            await until(lambda: pointers(second) == 1)
            await answer(second, 'answered')
            assert world.call.answer_due is None
            world.verdict = fits
            world.release.set()
            await settled(world)
            assert pointers(second) == 1 and not outputs(second) and world.call.answer_due is None
            assert not world.call.response_open and not world.call.pending_offers
            assert not shortened, 'a retired output needs no shortened form'
    asyncio.run(scenario())


@pytest.mark.parametrize('shortened', [True, False])
def test_a_fallback_finishing_after_the_switch_dispatches_nothing(shortened):
    """The fallback is one more wait: its rows, or its refusal, no longer own anything on the successor."""
    async def scenario():
        async with withheld() as world:
            entered, release = asyncio.Event(), asyncio.Event()

            async def fallback(rows, schemas):
                entered.set()
                await release.wait()
                return rows if shortened else None
            world.host.tool_result_fallback = fallback
            world.verdict = False
            await asyncio.wait_for(world.entered.wait(), 2)
            world.release.set()
            await asyncio.wait_for(entered.wait(), 2)
            second = await successor(world, world.first)
            await until(lambda: pointers(second) == 1)
            await answer(second, 'answered')
            release.set()
            await settled(world)
            assert pointers(second) == 1 and not outputs(second) and world.call.answer_due is None
    asyncio.run(scenario())


def test_a_fitting_fallback_still_delivers_natively_on_its_own_conversation():
    async def scenario():
        async with withheld() as world:
            async def fallback(rows, schemas):
                return [{**row, 'content': 'Shortened.'} for row in rows]
            world.host.tool_result_fallback = fallback
            world.verdict = False
            await asyncio.wait_for(world.entered.wait(), 2)
            world.release.set()
            await until(lambda: world.first.finished == 1)
            assert [row['content'] for row in outputs(world.first)[0].messages] == ['Shortened.']
            assert world.call.response_open and not world.call.deferred_tool_outputs
    asyncio.run(scenario())


def test_a_refused_delivery_with_no_fallback_stays_terminal_on_its_own_conversation():
    async def scenario():
        async with withheld() as world:
            world.verdict = False
            await asyncio.wait_for(world.entered.wait(), 2)
            world.release.set()
            with pytest.raises(RealtimeSetupError, match='window'):
                await asyncio.wait_for(world.task, 2)
    asyncio.run(scenario())


@pytest.mark.parametrize('released_on', [2, 3, 4])
def test_an_unanswered_result_owes_one_pointer_per_fresh_session_whenever_its_measure_ends(released_on):
    """Multiple reconnections under one held measure: each fresh session is owed the same one pointer until an answer."""
    async def scenario():
        async with withheld() as world:
            await asyncio.wait_for(world.entered.wait(), 2)
            session = world.first
            for number, spoken in ((2, False), (3, True), (4, False)):
                session = await successor(world, session)
                if spoken or number < 3:
                    await until(lambda: pointers(session) == 1)
                if released_on == number:
                    world.release.set()
                    await settled(world)
                if spoken:
                    await answer(session, 'answered')
                    assert world.call.answer_due is None
                elif number == 2:
                    assert world.call.answer_due is not None, 'written pointer, no answer: still owed'
            await asyncio.sleep(.05)
            assert pointers(session) == 0, 'answered: a later fresh session is not asked again'
            assert [pointers(made) for made in world.host.sessions] == [0, 1, 1, 0]
            assert not any(outputs(made) for made in world.host.sessions)
    asyncio.run(scenario())


@pytest.mark.parametrize('silent', [(), ('kept',)])
def test_a_late_measure_leaves_the_successors_response_state(silent):
    """The successor awaits a response to what it delivered; a stale dispatch must not reopen its quiet."""
    async def scenario():
        async with withheld(silent=silent) as world:
            await asyncio.wait_for(world.entered.wait(), 2)
            # The peer's fact is in the room before the successor opens, so its first quiet offer carries it beside the
            # answer it owes. Appended after the successor reported ready, it raced that offer: where asyncio runs
            # every timer due within one clock resolution at once (Windows: 1/64 s) the owed answer went alone, and
            # the fact rightly waited for the next quiet moment.
            world.opening = asyncio.Event()
            reconnect = asyncio.create_task(successor(world, world.first))
            await until(lambda: world.call.session is None)
            world.host.hub.events.append(cp.PeerControl('peer', cp.PEER_OWNER, {'type': 'context', 'text': 'Peer fact.'}))
            world.opening.set()
            second = await reconnect
            await until(lambda: world.call.response_open and 'Peer fact.' in str(second.sent))
            world.release.set()
            await settled(world)
            assert world.call.response_open and not outputs(second)
            await answer(second, 'answered')
            assert not world.call.response_open
    asyncio.run(scenario())


@pytest.mark.parametrize('silent', [(), ('kept',)])
def test_a_measured_result_stays_native_on_its_own_conversation(silent):
    """No reconnection: one native output, no pointer before or after it, and its answer settles it."""
    async def scenario():
        async with withheld(silent=silent) as world:
            await asyncio.wait_for(world.entered.wait(), 2)
            await asyncio.sleep(.05)  # quiet ticks while it is measured: the output will ask for the answer itself
            assert not pointers(world.first) and not world.call.response_open
            world.release.set()
            await until(lambda: world.first.finished == 1 and not world.call.tools_running)
            await asyncio.sleep(.05)
            assert len(outputs(world.first)) == 1 and not pointers(world.first)
            assert not world.call.deferred_tool_outputs and 'kept' not in world.call.issued
            assert not world.call.pending_offers
            if silent:
                assert world.call.answer_due is None and not world.call.response_open
            else:
                assert world.call.answer_due is not None and world.call.response_open
                await answer(world.first, 'answered')
                assert world.call.answer_due is None and not world.call.response_open
    asyncio.run(scenario())


@pytest.mark.parametrize('resumes', [True, False])
def test_a_measure_ending_while_disconnected_waits_for_readiness_and_changes_nothing(resumes):
    """Between connections there is no session to write to and no response to await. A native resume then gets the
    output once; a fresh session already holds it and is owed one pointer."""
    async def scenario():
        async with withheld() as world:
            await asyncio.wait_for(world.entered.wait(), 2)
            world.opening = asyncio.Event()
            if resumes:
                world.first.emit('resumable', handle='covering')
            world.first.emit('closed', resumable=resumes, transport_loss=True, reason='transport_lost')
            await until(lambda: world.call.session is None)
            offers = list(world.call.pending_offers)
            world.release.set()
            await asyncio.sleep(.05)
            assert not world.call.response_open and not world.task.done()
            assert world.call.pending_offers == offers
            world.opening.set()
            await until(lambda: len(world.host.sessions) == 2)
            second = world.host.sessions[1]
            carried = world.call.answer_due
            second.emit('ready', **READY)
            if resumes:
                await until(lambda: second.finished == 1)
                await settled(world)
                assert len(outputs(second)) == 1 and not outputs(world.first) and not pointers(second)
                assert world.call.response_open
                await answer(second, 'answered')
                assert world.call.answer_due is None
            else:
                await until(lambda: pointers(second) == 1)
                await settled(world)
                assert not outputs(second) and pointers(second) == 1
                assert world.call.answer_due is carried and carried is not None, 'consume carried it; this release arms nothing'
    asyncio.run(scenario())


@pytest.mark.parametrize('resumes', [True, False])
def test_a_refusal_arriving_between_connections_counts_only_for_a_conversation_that_continues(resumes):
    async def scenario():
        async with withheld() as world:
            await asyncio.wait_for(world.entered.wait(), 2)
            world.opening, world.verdict = asyncio.Event(), False
            if resumes:
                world.first.emit('resumable', handle='covering')
            world.first.emit('closed', resumable=resumes, transport_loss=True, reason='transport_lost')
            await until(lambda: world.call.session is None)
            world.release.set()
            await asyncio.sleep(.05)
            assert not world.task.done(), 'the connection that was measuring is gone, and its answer with it'
            world.opening.set()
            await until(lambda: len(world.host.sessions) == 2)
            second = world.host.sessions[1]
            second.emit('ready', **READY)
            if resumes:  # measured again for the conversation that continues, and refused
                with pytest.raises(RealtimeSetupError, match='window'):
                    await asyncio.wait_for(world.task, 2)
                assert not outputs(second)
                return
            await until(lambda: pointers(second) == 1)
            await settled(world)
            assert not outputs(second) and pointers(second) == 1
    asyncio.run(scenario())


def held_write(fails):
    """A native write in flight until the scenario decides how it ends."""
    entered, release = asyncio.Event(), asyncio.Event()

    async def during(adapter):
        entered.set()
        await release.wait()
        if fails:
            raise ConnectionClosed(None, None)
        adapter.deliver()
    return SimpleNamespace(during=during, entered=entered, release=release)


@pytest.mark.parametrize(("fails", "answered", "deferred"), [
    (True, True, False),
    (True, False, True),
    (False, True, False),
])
def test_a_write_ending_after_the_switch_changes_nothing_on_the_successor(fails, answered, deferred):
    """However late the old socket's write ends: consume carried its debt, so a failure offers nothing more, and a
    stale success records no delivery on the conversation that replaced it."""
    async def scenario():
        write = held_write(fails)
        async with withheld(writes=[write.during], deferred=deferred, holds=lambda rows: False) as world:
            await asyncio.wait_for(write.entered.wait(), 2)
            second = await successor(world, world.first)
            await until(lambda: pointers(second) == 1)
            if answered:
                await answer(second, 'answered')
            write.release.set()
            await settled(world)
            assert pointers(second) == 1 and not outputs(second)
            assert (world.call.answer_due is None) == answered
            assert world.call.resume_tool_output is None and world.call.delivered_catalog is None
            assert 'kept' not in world.call.issued
    asyncio.run(scenario())


def test_a_resume_withdrawn_during_a_write_leaves_the_fresh_session_able_to_go_quiet():
    """The write awaits an answer this connection will never give. However a session comes to be fresh, it starts with
    nothing awaited, so its one pointer goes out without waiting for the old socket's write to end."""
    async def scenario():
        write = held_write(True)
        async with withheld(writes=[write.during], deferred=False, holds=lambda rows: False) as world:
            await asyncio.wait_for(write.entered.wait(), 2)
            assert world.call.response_open
            draining, drained = asyncio.Event(), asyncio.Event()

            async def caption(**row):
                draining.set()
                await drained.wait()
            world.host.caption = caption
            world.first.emit('caption', speaker='owner', item_id='owner-1', text='Words.', final=True)
            await asyncio.wait_for(draining.wait(), 2)
            world.first.emit('resumable', handle='covering')
            world.first.emit('closed', resumable=True, transport_loss=True, reason='transport_lost')
            await until(lambda: world.call.provider_closed)
            world.host.state.pop('resume_handle')  # withdrawn while canonical rows drain: the replacement is fresh
            drained.set()
            await until(lambda: len(world.host.sessions) == 2)
            second = world.host.sessions[1]
            second.emit('ready', **READY)
            await until(lambda: pointers(second) == 1)
            write.release.set()
            await settled(world)
            assert world.call.conversation == 2 and pointers(second) == 1 and not outputs(second)
    asyncio.run(scenario())


@pytest.mark.parametrize('deferred', [False, True])
def test_a_failed_write_retried_on_a_resume_reaches_it_once(deferred):
    """The writer that keeps custody retries on the resumed socket. The resumed ready's own release, or any other
    releaser, must not write the same output beside it while that write is in flight."""
    async def scenario():
        retry = held_write(False)

        async def closed(adapter):
            raise ConnectionClosed(None, None)
        async with withheld(writes=[closed, retry.during], deferred=deferred, holds=lambda rows: False) as world:
            await until(lambda: outputs(world.first))
            second = await successor(world, world.first, resumes=True)
            await asyncio.wait_for(retry.entered.wait(), 2)
            await asyncio.sleep(.05)
            assert len(outputs(second)) == 1, 'one effect, one native output'
            retry.release.set()
            await settled(world)
            assert len(outputs(second)) == 1 and not pointers(second) and not world.call.deferred_tool_outputs
            await answer(second, 'answered')
            assert world.call.answer_due is None
    asyncio.run(scenario())


def test_a_direct_result_never_written_before_a_fresh_fit_owes_its_one_pointer():
    """No debt was armed before the switch: this is the only send exit that still offers, once."""
    async def scenario():
        host = Host()
        call = look_call(host)
        world = attach(host, call)
        world.holds = lambda rows: any(row.get('tool_call_id') == 'look' for row in rows)
        task = asyncio.create_task(call.run())
        try:
            await until(lambda: host.sessions)
            first = host.sessions[0]
            first.emit('ready', **READY)
            first.emit('tool_call', call_id='look', name='attend', arguments='{"look":true}')
            await asyncio.wait_for(world.entered.wait(), 2)
            first.emit('resumable', handle='covering')
            first.emit('closed', resumable=True, transport_loss=True, reason='transport_lost')
            await until(lambda: not call.connected.is_set() and call.resume)
            world.stale_resume = True  # a native resume is promised, and refused only once the output's send waits
            world.release.set()
            await until(lambda: len(host.sessions) == 2)
            second = host.sessions[1]
            second.emit('ready', **READY)
            await until(lambda: pointers(second) == 1 and not call.tools_running)
            await asyncio.sleep(.05)
            assert call.conversation == 2 and not outputs(second) and pointers(second) == 1
            assert 'look' not in call.issued
            await answer(second, 'answered')
            assert call.answer_due is None
            await asyncio.sleep(.05)
            assert pointers(second) == 1
        finally:
            world.release.set()
            host.hub.ended_by_owner = True
            if not task.done():
                await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


@pytest.mark.parametrize(("gap", "fits"), [(False, True), (True, False)])
def test_mail_measured_across_a_fresh_switch_stays_in_the_fit(gap, fits):
    """The same worker measures pending mail after its output: a fresh fit holds that mail, and with no connection
    yet there is no session to write it to. Neither ends the call, however the old measure answers."""
    async def scenario():
        async with withheld(holds=holding_mail, mail=True) as world:
            await asyncio.wait_for(world.entered.wait(), 2)
            assert world.first.finished == 1 and not world.call.deferred_tool_outputs
            world.verdict = fits
            if gap:
                world.opening = asyncio.Event()
                world.first.emit('closed', resumable=False, transport_loss=True, reason='transport_lost')
                await until(lambda: world.call.session is None)
                world.release.set()
                await asyncio.sleep(.05)
                assert not world.task.done()
                world.opening.set()
                await until(lambda: len(world.host.sessions) == 2)
                second = world.host.sessions[1]
                second.emit('ready', **READY)
            else:
                second = await successor(world, world.first)
                world.release.set()
            await settled(world)
            assert not mail_updates(second) and not world.call.pending_context_rows
    asyncio.run(scenario())


def test_mail_measured_across_a_native_resume_reaches_it_once():
    async def scenario():
        async with withheld(holds=holding_mail, mail=True) as world:
            await asyncio.wait_for(world.entered.wait(), 2)
            world.opening = asyncio.Event()
            world.first.emit('resumable', handle='covering')
            world.first.emit('closed', resumable=True, transport_loss=True, reason='transport_lost')
            await until(lambda: world.call.session is None)
            world.release.set()
            await asyncio.sleep(.05)
            assert not world.task.done() and world.call.pending_context_rows
            world.opening.set()
            await until(lambda: len(world.host.sessions) == 2)
            second = world.host.sessions[1]
            second.emit('ready', **READY)
            await until(lambda: mail_updates(second))
            await settled(world)
            assert len(mail_updates(second)) == 1 and not mail_updates(world.first)
            assert not world.call.pending_context_rows
    asyncio.run(scenario())


def test_the_private_answer_marker_leaves_no_copy_of_a_row_for_a_wire():
    from ouroboros.llm_messages import project_wire_messages
    row = {'role': 'user', 'content': 'Pointer.', '_tool_answer': ['kept']}
    assert project_wire_messages([row]) == [{'role': 'user', 'content': 'Pointer.'}]
    assert project_wire_messages([row], delivery_marks=True) == [{'role': 'user', 'content': 'Pointer.'}]
    assert row['_tool_answer'] == ['kept'], 'the projection copies; the held row keeps its marker'


# --- the real host and adapters ----------------------------------------------------------------------


@pytest.mark.parametrize(("engine", "answered"), [('openai', True), ('gemini', False)])
def test_native_guard_returning_after_a_successor_took_custody(host_world, monkeypatch, engine, answered):
    """Real host and adapters: one real ahead-fit capacity failure withholds a completed batch for its serving measure,
    and the socket is replaced while that measure runs. One pointer, one spoken answer if the mind gives it, never an
    old output, and an unanswered result still owes the next fresh session its one pointer."""
    from ouroboros.live_connections import PreparedConnection
    from tests._live_stack import stack
    from tests._live_stack_wire import NativeWire
    measuring, release = threading.Event(), threading.Event()
    fit = PreparedConnection._fit
    speak = NativeWire.answer

    async def capacity_failure(lease, credentials, *, rebound=False):
        if lease.ahead:
            raise RealtimeSetupError('context_capacity', 'Synthetic failure of a speculative capacity fit')
        return await fit(lease, credentials, rebound=rebound)
    monkeypatch.setattr(PreparedConnection, '_fit', capacity_failure)

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            batch, measure, renewal = live.host.tool_batch, live.host.delivery_fits, live.host.renewal_due
            suppressed = [False]

            async def failed_fit_state(calls):
                result = await batch(calls)
                suppressed[0] = True
                revision = live.host._history_version
                live.call.request_renewal()
                await reached(lambda: live.call.failed_preparation_revision == revision and not live.call.renewing)
                return result

            async def gated(rows, schemas):
                admitted = await measure(rows, schemas)
                if holding_result(rows):
                    assert admitted
                    measuring.set()
                    while not release.is_set():
                        await asyncio.sleep(.001)
                return admitted

            monkeypatch.setattr(live.host, 'tool_batch', failed_fit_state)
            monkeypatch.setattr(live.host, 'delivery_fits', gated)
            monkeypatch.setattr(live.host, 'renewal_due', lambda: suppressed[0] or renewal())
            monkeypatch.setattr(NativeWire, 'answer', lambda wire: speak(wire) if answered else None)
            first = live.wire
            first.tool('kept', 'get_task_result', {'task_id': 'none'})
            try:
                await live.wait(measuring.is_set, 30)
                suppressed[0] = False
                await first.websocket.close(code=1012, reason='synthetic service restart')
                await live.wait(lambda: live.call.conversation == 2 and live.call.connected.is_set(), 30)
                second = live.wire
                count = lambda wire: sum(POINTER in json.dumps(frame) for frame in wire.sent)
                await live.wait(lambda: count(second) == 1 and (second.spoken or not answered), 30)
                if answered:
                    await live.quiet()
                    assert live.call.answer_due is None
                spoken = list(second.spoken)
                release.set()
                await live.wait(lambda: not live.call.tools_running, 30)
                await asyncio.sleep(.25)
                assert not native_output(second, 'kept') and count(second) == 1, (count(second), second.spoken)
                assert second.spoken == spoken, 'no second spoken answer'
                assert (live.call.answer_due is None) == answered
                if not answered:
                    await second.websocket.close(code=1012, reason='synthetic service restart')
                    await live.wait(lambda: live.call.conversation == 3 and live.call.connected.is_set(), 30)
                    third = live.wire
                    await live.wait(lambda: count(third) == 1, 30)
                    assert not native_output(third, 'kept')
            finally:
                release.set()
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())
