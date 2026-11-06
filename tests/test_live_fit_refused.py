"""When a connection cannot be fitted or reopened, the call says so; it never says something false to keep going.

A batch that finishes on a later conversation has no native lane: only a fresh fit can hold it, and its pointer claims that
fit. If the fit is refused the call ends typed ``context_capacity``, the whole result canonical and nothing repeated, and no
pointer ever reached the serving model. A conversation that issued the call keeps its range fallback. A replacement that
cannot be opened ends typed ``reconnect_failed``, with no retry of its own. The fake port forces the ordering; the real host
and adapters show the same ending under the forced refusal existing tests use. No organic capacity failure is produced here.
"""
import asyncio
import json

import pytest

from ouroboros import call_protocol as cp, realtime_session
from ouroboros.contracts.live_conversation import ContextUpdate, ToolOutputs
from ouroboros.contracts.realtime import RealtimeSetupError
from tests._extension_loader_shared import _clear_loader_state  # noqa: F401
from tests._live_world import host_world as host_world
from tests.test_live_conversation_driver import Host, Session, started, until
from tests.test_live_delivery_owner_stack import RESTART, handle
from tests.test_live_tool_continuity import POINTER, Held, held_batch
from tests.test_live_tool_continuity_stack import page_state

READY = dict(input_rate=24000, output_rate=24000)
BODY = 'Result from one completed effect. ' * 1600


def pointers(session):
    return [command for command in session.sent if isinstance(command, ContextUpdate) and POINTER in str(command.messages)]


def outputs(session):
    return [command for command in session.sent if isinstance(command, ToolOutputs)]


def refused_at(revision):
    error = RealtimeSetupError('context_capacity', 'protected remainder does not fit')
    error.source_revision = revision
    return error


async def fresh_while_it_runs(host, task, first):
    first.emit('ready', **READY)
    first.emit('tool_call', call_id='work', name='slow_tool', arguments='{}')
    await until(lambda: host.batch_running.is_set())
    first.emit('closed', resumable=False, transport_loss=True, reason='transport_lost')
    await until(lambda: len(host.sessions) == 2)
    second = host.sessions[1]
    second.silent_context_updates = True
    second.selected = type('Selected', (), {'tools_json': '[]'})()
    second.emit('ready', **READY)
    await until(lambda: host.sessions[1].events.empty())
    return second


def test_a_refused_fit_for_a_batch_from_an_earlier_conversation_ends_typed_and_tells_no_lie():
    """Opus's P5: the serving room would pass its guard, so the old path sent a pointer no fit or output stood behind."""
    async def scenario():
        host = Host()
        gate, host.batch_running, _ = held_batch(host)
        attempts = []

        async def prepare(options):
            attempts.append(options)
            raise refused_at(0)
        host.prepare_session = prepare

        async def fits(rows, schemas):
            return True
        host.delivery_fits = fits
        task, first = await started(host)
        second = await fresh_while_it_runs(host, task, first)
        gate.set()
        with pytest.raises(RealtimeSetupError, match='context_capacity'):
            await asyncio.wait_for(task, 5)
        assert len(attempts) == 1 and len(host.sessions) == 2
        assert not pointers(second) and not outputs(second), 'neither a pointer to a fit that never was, nor an orphan output'
    asyncio.run(scenario())


def test_a_changed_view_gets_its_fit_before_the_refusal_counts():
    """The refusal named an older revision: the current history earns its own attempt first."""
    async def scenario():
        host = Host()
        gate, host.batch_running, _ = held_batch(host)
        host._history_version = 2
        attempts = []

        async def prepare(options):
            attempts.append(options)
            raise refused_at(1 if len(attempts) == 1 else 2)
        host.prepare_session = prepare

        async def fits(rows, schemas):
            return True
        host.delivery_fits = fits
        task, first = await started(host)
        second = await fresh_while_it_runs(host, task, first)
        gate.set()
        with pytest.raises(RealtimeSetupError, match='context_capacity'):
            await asyncio.wait_for(task, 5)
        assert len(attempts) == 2 and not pointers(second)
    asyncio.run(scenario())


def test_the_pointer_is_true_on_the_fresh_session_that_holds_the_result():
    """The opposite order: no speculative fit on offer, so the fit comes at the next switch. Its session gets the pointer."""
    async def scenario():
        host = Host()
        gate, host.batch_running, batches = held_batch(host)
        task, first = await started(host)
        try:
            second = await fresh_while_it_runs(host, task, first)
            gate.set()
            await until(lambda: len(host.sessions) == 3)
            third = host.sessions[2]
            third.emit('ready', **READY)
            await until(lambda: pointers(third))
            assert not pointers(second) and not outputs(second) and not outputs(third) and batches == [['work']]
            assert host.options[2].resume_handle == ''
        finally:
            host.hub.ended_by_owner = True
            if not task.done():
                await asyncio.wait_for(task, 5)
    asyncio.run(scenario())


def test_a_native_resume_before_the_fit_keeps_waiting_for_it():
    """The fit was wanted, not refused. A native resume of the conversation that issued nothing here comes up busy with
    its own native work and reports ready: its worker finds the unit no native write can deliver while that fit is still
    on its way, so it neither ends the call nor hears the pointer. Once it is quiet, the fresh session that follows does."""
    async def scenario():
        host = Host()
        gate, host.batch_running, batches = held_batch(host)

        async def session(consume, options):
            made = Session()
            made.renewal_ready = not options.resume_handle  # a resumed engine still busy with its own native work
            host.sessions.append(made)
            host.options.append(options)
            return await consume(made)
        host.session = session
        task, first = await started(host)
        try:
            second = await fresh_while_it_runs(host, task, first)
            second.emit('input_started')  # holds the forced switch while the result finishes
            await until(lambda: second.events.empty())
            gate.set()
            await until(lambda: host.state.get('refit'))
            second.emit('resumable', handle='covering')
            second.emit('closed', resumable=True, transport_loss=True, reason='transport_lost')
            await until(lambda: len(host.sessions) == 3)
            resumed = host.sessions[2]
            assert host.options[2].resume_handle == 'covering'
            resumed.emit('ready', **READY)
            await until(lambda: resumed.events.empty())
            await asyncio.sleep(.1)  # its delivery worker has run
            assert not task.done(), 'a wanted fit is not a refused one'
            assert len(host.sessions) == 3 and not pointers(resumed) and not outputs(resumed)
            resumed.renewal_ready = True
            await until(lambda: len(host.sessions) == 4)
            fresh = host.sessions[3]
            assert host.options[3].resume_handle == ''
            fresh.emit('ready', **READY)
            await until(lambda: pointers(fresh))
            assert not task.done() and not outputs(fresh) and batches == [['work']]
        finally:
            host.hub.ended_by_owner = True
            if not task.done():
                await asyncio.wait_for(task, 5)
    asyncio.run(scenario())


@pytest.mark.parametrize(("engine", "serving"), [('openai', 'has-room'), ('gemini', 'full')])
def test_real_refused_fit_after_a_reconnection_ends_typed_with_the_result_whole(host_world, monkeypatch, engine, serving):
    """Under the forced refusal existing tests use. ``has-room`` is the order that used to send the false pointer."""
    from tests._live_stack import stack
    slow = Held(monkeypatch, text=BODY)

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            first = live.wire
            first.tool('large', 'list_available_tools', {})
            try:
                await slow.started()
                await first.websocket.close(**RESTART)
                await live.wait(lambda: live.host.connection_number == 2 and live.call.connected.is_set(), 30)
                await live.wait(lambda: page_state(live) == cp.STATE_LIVE, 30)
                second, host = live.wire, live.host

                async def refuse(options):
                    raise refused_at(host._history_version)
                monkeypatch.setattr(host, 'prepare_session', refuse)
                if serving == 'full':
                    window = host._renewal_window.total_equivalent if host._renewal_window else host.model.facts.conversation_window_tokens
                    reserve = host._renewal_window.reserve if host._renewal_window else host.model.facts.working_reserve_tokens or 0
                    host._held_input = window - reserve - 1200
                    host._history_at_receipt = host._history_estimate()
                    host._serving_history_at_receipt = host._serving_history_estimate()
            finally:
                slow.release.set()
            await asyncio.wait_for(asyncio.shield(live.running), 60)
        assert live.result[1]['reason_code'] == 'context_capacity' and live.result[1]['execution_status'] == 'infra_failed'
        assert not any(POINTER in json.dumps(frame) for frame in second.sent), 'no fit stood behind it'
        assert not any('large' in json.dumps(frame) and ('functionResponses' in json.dumps(frame) or 'function_call_output' in json.dumps(frame))
                       for frame in second.sent)
        assert any(row.get('tool_call_id') == 'large' and BODY in str(row.get('content')) for row in host.messages)
        assert slow.effects == ['list_available_tools']
    asyncio.run(scenario())


@pytest.mark.parametrize('engine', ['openai', 'gemini'])
def test_real_refused_fit_on_the_issuing_conversation_keeps_its_range_fallback(host_world, monkeypatch, engine):
    from tests._live_stack import stack
    slow = Held(monkeypatch, text=BODY)

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            first = live.wire
            first.tool('large', 'list_available_tools', {})
            try:
                await slow.started()
                host = live.host

                async def refuse(options):
                    raise refused_at(host._history_version)
                monkeypatch.setattr(host, 'prepare_session', refuse)
                window = host._renewal_window.total_equivalent if host._renewal_window else host.model.facts.conversation_window_tokens
                reserve = host._renewal_window.reserve if host._renewal_window else host.model.facts.working_reserve_tokens or 0
                host._held_input = window - reserve - 1200
                host._history_at_receipt = host._history_estimate()
                host._serving_history_at_receipt = host._serving_history_estimate()
            finally:
                slow.release.set()
            await live.wait(lambda: 'large' in first.outputs or any('functionResponses' in json.dumps(frame) and 'large' in json.dumps(frame)
                                                                     for frame in first.sent), 60)
            assert not live.running.done()
        assert live.result[1]['reason_code'] == 'finished' and slow.effects == ['list_available_tools']
        assert any(row.get('tool_call_id') == 'large' and BODY in str(row.get('content')) for row in host.messages)
    asyncio.run(scenario())


# --- a replacement that cannot be opened -----------------------------------------------------------------------------------

# The network errors websockets names for an opening attempt; before Python 3.11 its handshake timeout is asyncio's own.
OPEN_FAILURES = {'no-route': lambda: OSError(65, 'No route to host'),
                 'handshake-timeout': lambda: asyncio.TimeoutError('timed out during opening handshake')}


async def ended_with(task):
    """The call's own ending: a genuine hang is not mistaken for a timeout the call raised."""
    done, _ = await asyncio.wait({task}, timeout=5)
    assert task in done, 'the call never ended'
    return task.exception()


@pytest.mark.parametrize('resuming', [False, True], ids=['fresh', 'native'])
@pytest.mark.parametrize('engine', ['openai', 'gemini'])
@pytest.mark.parametrize('failure', sorted(OPEN_FAILURES))
def test_real_replacement_open_failure_ends_with_its_typed_reason(host_world, monkeypatch, failure, engine, resuming):
    """No retry and no timer: one network error on the reopen ends the call, and the owner reads that it was unreachable."""
    from tests._live_stack import stack
    if resuming and engine != 'gemini':
        pytest.skip('only Gemini has a native resume')

    async def scenario():
        async with stack(host_world, monkeypatch, engine) as live:
            first = live.wire
            if resuming:
                handle(first, 'covering')
                await live.wait(lambda: live.host.state.get('resume_handle') == 'covering', 30)

            def unreachable(uri, *, headers):
                raise OPEN_FAILURES[failure]()
            monkeypatch.setattr(realtime_session, 'connect_provider', unreachable)
            await first.websocket.close(**RESTART)
            await asyncio.wait_for(asyncio.shield(live.running), 60)
            connections = len(live.provider.connections)
        assert connections == 1
        assert live.result[1]['reason_code'] == 'reconnect_failed' and live.result[1]['execution_status'] == 'infra_failed'
    asyncio.run(scenario())


def test_only_a_replacement_opening_takes_that_reason(subtests):
    """A first open keeps its ending, and so does the same error raised inside a connection's own lifetime."""
    for failure in sorted(OPEN_FAILURES):
        with subtests.test(failure=failure):
            from skills.live_conversation.driver import run
            error = OPEN_FAILURES[failure]

            def kept(exc):
                assert type(exc) is type(error()), repr(exc)

            async def first_open():
                host = Host()

                async def session(consume, options):
                    raise error()
                host.session = session
                kept(await ended_with(asyncio.create_task(run(host))))

            async def lifetime():
                host = Host()
                task = asyncio.create_task(run(host))
                await until(lambda: host.sessions)
                first = host.sessions[0]
                first.emit('closed', resumable=False, transport_loss=True, reason='transport_lost')
                await until(lambda: len(host.sessions) == 2)
                second = host.sessions[1]

                async def send(command):
                    raise error()
                second.send = send
                second.emit('ready', **READY)
                second.emit('input_started')
                kept(await ended_with(task))

            async def replacement():
                host = Host()
                opened = []

                async def session(consume, options):
                    opened.append(options)
                    if len(opened) == 2:
                        raise error()
                    made = type(host).session
                    return await made(host, consume, options)
                host.session = session
                task = asyncio.create_task(run(host))
                await until(lambda: host.sessions)
                host.sessions[0].emit('closed', resumable=False, transport_loss=True, reason='transport_lost')
                exc = await ended_with(task)
                assert isinstance(exc, RealtimeSetupError) and exc.code == 'reconnect_failed', repr(exc)
            asyncio.run(first_open())
            asyncio.run(lifetime())
            asyncio.run(replacement())


@pytest.mark.parametrize('failure', ['none', 'no-route', 'handshake-timeout'])
def test_stale_prepared_successor_then_replacement_open(host_world, monkeypatch, failure):
    """A prepared fit can stale when a tool finishes: its replacement uses the same opening-error boundary."""
    from tests._live_stack import stack
    from tests.test_live_tool_continuity import Held, POINTER
    from tests.test_live_tool_continuity_stack import fitted, native_output

    slow, opens = Held(monkeypatch), []

    async def scenario():
        async with stack(host_world, monkeypatch, 'openai') as live:
            first = live.wire
            first.tool('finishes-after-fit', 'list_available_tools', {})
            try:
                await slow.started()
                live.host.state['refit'] = True
                await live.wait(lambda: live.call.preparing is not None and live.call.preparing.done(), 30)
                prepared = live.provider.connections[1]
                assert prepared is not first and len(live.provider.connections) == 2
                original = realtime_session.connect_provider

                def open_replacement(uri, *, headers):
                    opens.append(failure)
                    if failure != 'none':
                        raise OPEN_FAILURES[failure]()
                    return original(uri, headers=headers)

                monkeypatch.setattr(realtime_session, 'connect_provider', open_replacement)
            finally:
                slow.release.set()
            if failure == 'none':
                await live.wait(lambda: any(row.get('tool_call_id') == 'finishes-after-fit' for row in fitted(live)), 60)
                await live.wait(lambda: any(POINTER in json.dumps(row) for row in live.wire.sent), 60)
                await live.quiet()
                assert live.wire is not prepared
            else:
                await asyncio.wait_for(asyncio.shield(live.running), 60)
            assert opens == [failure]
            assert any(row.get('tool_call_id') == 'finishes-after-fit' for row in live.host.messages)
            assert not any(native_output(wire, 'finishes-after-fit') for wire in live.provider.connections)
            assert slow.effects == ['list_available_tools']
        assert live.result[1]['reason_code'] == ('finished' if failure == 'none' else 'reconnect_failed')

    asyncio.run(scenario())
