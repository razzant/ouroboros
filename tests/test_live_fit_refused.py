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
        if failure == 'none':  # the owner ended a call that opened: its transcript is in the room
            assert live.result[0] == 'Call ended. Its transcript is in this room.'
            assert live.host.hub._state['reason'] == 'finished' and not live.host.hub._state.get('detail')

    asyncio.run(scenario())


def _run_call(host_world, monkeypatch, handler, *, window=None):
    """Run the admitted call's real terminal (``LiveConversationTurn.run``) with ``handler`` as the feature's body."""
    from dataclasses import replace
    from ouroboros import call_peers

    host = host_world.host
    turn = host.turn
    if window is not None:
        facts = replace(turn.catalog_model.facts, conversation_window_tokens=window)
        monkeypatch.setattr(turn, 'catalog_model', replace(turn.catalog_model, facts=facts))
    monkeypatch.setitem(call_peers._HUBS, turn.spec.call_id, host.hub)
    monkeypatch.setitem(turn.feature, 'handler', handler)
    seen = {}
    real = turn.run.__globals__['ConversationHost']

    class Observed(real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen['host'] = self

    monkeypatch.setitem(turn.run.__globals__, 'ConversationHost', Observed)
    words, usage, _trace = turn.run(host.ctx, list(host.messages), host.tools, task=host.task, initial_effort='')
    return words, usage, seen['host'], dict(host.hub._state)


def _refusing_handler(host_world, *, opened):
    async def handler(port):
        from ouroboros.realtime_setup import session_candidate
        from skills.live_conversation.driver import session_options
        port._session_opened = opened  # True: the refusal of a renewal of a call that is already serving
        await asyncio.to_thread(session_candidate, port, host_world.provider['session'], session_options(port))
    return handler


def test_a_session_setup_refusal_before_the_first_connection_says_the_call_did_not_start(host_world, monkeypatch):
    """Path 2: the ordinary build fitted and the call's own instructions and tools did not. The room's row is the same
    notice path 1 leaves (code, step), with the measured numbers of the refused candidate, and claims no transcript."""
    words, usage, host, state = _run_call(host_world, monkeypatch, _refusing_handler(host_world, opened=False),
                                          window=11_000)
    measured = host.trace['fit_refusal']['fit']['measurement']
    row = usage['host_notice_row']
    assert usage['reason_code'] == 'context_capacity' and usage['execution_status'] == 'infra_failed'
    assert row['notice']['code'] == 'context_capacity' and row['notice']['step'] == 'open_calls'
    assert words == row['record'] and 'transcript' not in words and 'context_capacity' not in words
    # The engine as the call's opening row names it, once; the model id in Details.
    assert row['notice']['values']['engine'] == host.turn.engine and f'Engine: {host.turn.engine},' in host.turn.opening
    assert row['text'].count(host.turn.engine) == 1 and host.turn.native_model in row['notice']['details']
    takes = measured['capacity_total_tokens'] - measured['response_reserve_tokens']
    assert row['notice']['values']['need'] == f"{measured['estimated_input_tokens']:,}"
    assert row['notice']['values']['takes'] == f'{takes:,}'
    for fact in (f"{measured['capacity_total_tokens']:,}", f"{measured['response_reserve_tokens']:,}", 'fixture'):
        assert fact in row['notice']['details'], fact
    assert state['reason'] == 'context_capacity' and state['action'] == 'settings' and state['detail'] == row['text']


def test_a_capacity_refusal_after_the_session_opened_keeps_the_call_ended_words(host_world, monkeypatch):
    """A call that served and then could not renew ended a real call: its words keep the transcript pointer."""
    from ouroboros.realtime_setup import CONTEXT_CAPACITY_DETAIL, call_ending

    words, usage, _host, state = _run_call(host_world, monkeypatch, _refusing_handler(host_world, opened=True),
                                           window=11_000)
    assert (words, usage) == call_ending('context_capacity', 'infra_failed', CONTEXT_CAPACITY_DETAIL)
    assert 'host_notice_row' not in usage and 'action' not in state


@pytest.mark.parametrize('opened', [False, True])
def test_the_owners_end_says_whether_the_call_ever_opened(opened):
    """The one home of a call's ending words (owner 26.11): an End before the call opened has no transcript to point
    at; an End of a call that opened keeps today's words. Both are the owner's ordinary End (``finished``, ``ok``)."""
    from ouroboros.realtime_setup import call_ending

    words, usage = call_ending('finished', 'ok', opened=opened)
    assert words == ('Call ended. Its transcript is in this room.' if opened else 'Call ended before it started.')
    assert usage == {'terminal_origin': 'host_notice', 'execution_status': 'ok', 'reason_code': 'finished'}


def test_end_pressed_before_the_engine_opened_is_the_ordinary_ending(host_world, monkeypatch):
    """The owner's bye reached the hub while the call prepared: the real driver never opens the engine and the call
    ends with the owner's ordinary End, worded for a call that never started, the one a preparation refused after
    that bye also takes (``tests/test_live_preparation_fit_refusal.py``)."""
    from ouroboros.realtime_setup import call_ending
    from skills.live_conversation.driver import run as drive

    hub = host_world.host.hub
    hub.join(cp.PeerInfo('owner', cp.PEER_OWNER, True, True), 24000)
    hub.receive_control('owner', {'type': 'bye'})
    words, usage, host, state = _run_call(host_world, monkeypatch, drive)
    assert (words, usage) == call_ending('finished', 'ok', opened=False) and state['reason'] == 'finished'
    assert words == 'Call ended before it started.'
    assert state['detail'] == words and not state.get('action')  # every connected sheet reads the same ending
    assert not host._session_opened and not host_world.sessions


def test_end_during_the_session_candidate_stays_the_owners_end(host_world, monkeypatch):
    """Path 2 with the owner's bye delivered while the refusing candidate is measured: the owner's End stands, as on
    path 1 (``tests/test_live_preparation_fit_refusal.py``); the refusal's measurement stays in the journal. The other
    direction (no bye: the refusal) is ``test_a_session_setup_refusal_before_the_first_connection_...``."""
    from ouroboros import realtime_setup
    from ouroboros.realtime_setup import call_ending

    hub = host_world.host.hub
    hub.join(cp.PeerInfo('owner', cp.PEER_OWNER, True, True), 24000)
    original = realtime_setup.measure_session_wire

    def bye_while_measuring(*args, **kwargs):
        if not hub.ended_by_owner:
            hub.receive_control('owner', {'type': 'bye'})
        return original(*args, **kwargs)
    monkeypatch.setattr(realtime_setup, 'measure_session_wire', bye_while_measuring)
    words, usage, host, state = _run_call(host_world, monkeypatch, _refusing_handler(host_world, opened=False),
                                          window=11_000)
    assert hub.ended_by_owner and not host._session_opened
    assert (words, usage) == call_ending('finished', 'ok', opened=False) and 'host_notice_row' not in usage
    assert state['reason'] == 'finished' and state['detail'] == 'Call ended before it started.'
    assert not state.get('action')
    assert 'fit_refusal' in host.trace


@pytest.mark.parametrize('bye_at', ['fit', 'open'])
def test_an_end_during_an_admitted_first_fit_never_opens_the_engine(host_world, monkeypatch, bye_at):
    """The owner's bye reaches the hub while the first native session is measured, and that fit is admitted: the
    engine is never opened and the call ends as the owner's End of a call that never started. The other direction
    (``open``: the bye arrives only once the engine opened) opens it once and keeps the ended-call words."""
    from ouroboros import realtime_setup
    from ouroboros.realtime_setup import call_ending
    from skills.live_conversation.driver import run as drive
    from tests._live_world import FakeSession

    hub = host_world.host.hub
    hub.join(cp.PeerInfo('owner', cp.PEER_OWNER, True, True), 24000)
    measure, opened_engine, opens = realtime_setup.measure_session_wire, FakeSession.open, []

    def bye():
        if not hub.ended_by_owner:
            hub.receive_control('owner', {'type': 'bye'})

    def measured(*args, **kwargs):
        if bye_at == 'fit':
            bye()
        return measure(*args, **kwargs)

    async def opened(self):
        opens.append(hub.ended_by_owner)
        await opened_engine(self)
        if bye_at == 'open':
            bye()

    async def silent(self, *args, **kwargs):
        return None
    monkeypatch.setattr(realtime_setup, 'measure_session_wire', measured)
    monkeypatch.setattr(FakeSession, 'open', opened)
    monkeypatch.setattr(FakeSession, 'send', silent, raising=False)
    monkeypatch.setattr(FakeSession, 'final_captions', lambda self: (), raising=False)

    async def bounded(port):
        await asyncio.wait_for(drive(port), timeout=30)
    words, usage, host, state = _run_call(host_world, monkeypatch, bounded)
    assert hub.ended_by_owner and state['reason'] == 'finished' and 'host_notice_row' not in usage
    if bye_at == 'fit':
        assert opens == [] and not host._session_opened
        assert (words, usage) == call_ending('finished', 'ok', opened=False)
        assert state['detail'] == 'Call ended before it started.' and not state.get('action')
    else:
        assert opens == [False] and host._session_opened
        assert (words, usage) == call_ending('finished', 'ok', opened=True)
