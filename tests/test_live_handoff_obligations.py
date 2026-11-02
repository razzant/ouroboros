"""R21 reviewer regressions: the real stack, local providers and disposable stores."""
import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import usage_accounting as ua
from ouroboros.contracts.live_conversation import ContextUpdate, LiveOptions
from ouroboros.contracts.realtime import RealtimeSetupError
from tests._extension_loader_shared import _clear_loader_state  # noqa: F401
from tests._live_stack import stack
from tests._live_stack_wire import NativeWire
from tests.test_live_conversation import host_world as host_world
from tests.test_live_host_settlement import tariff
from tests.test_live_stack_preparation import request, ROUTES
from tests.test_grok_openai_protocol import providers as providers, Wire


@pytest.mark.parametrize('engine,thinking', ROUTES)
def test_preparation_refusal_is_terminal_without_another_attempt(host_world, monkeypatch, engine, thinking):
    async def scenario():
        send = NativeWire.send
        async def refuse(wire, raw):
            if wire.number == 2:
                wire.put({'type': 'error', 'error': {'code': 'invalid_request', 'message': 'Synthetic preparation refusal'}})
                return
            return await send(wire, raw)
        monkeypatch.setattr(NativeWire, 'send', refuse)
        async with stack(host_world, monkeypatch, engine, thinking=thinking) as run:
            await request(run)
            for _ in range(1500):
                if run.running.done() or len(run.provider.connections) >= 3:
                    break
                await asyncio.sleep(.01)
            attempts = len(run.provider.connections)
        assert attempts == 2, f'Provider refused connection 2, but {attempts} connections opened'
        assert run.result[1]['reason_code'] == 'provider_error'
    asyncio.run(scenario())


@pytest.mark.parametrize('engine,thinking', ROUTES)
def test_legacy_provider_without_silent_catchup_keeps_sequential_calls(host_world, monkeypatch, engine, thinking):
    async def scenario():
        async with stack(host_world, monkeypatch, engine, thinking=thinking) as run:
            cls = type(run.call.session)
            # An out-of-repository old implementation accepts ContextUpdate but
            # always asks for an answer, and declares no new capability.
            monkeypatch.setattr(cls, 'silent_context_updates', False, raising=False)
            original = cls.send
            async def legacy_send(session, command):
                if isinstance(command, ContextUpdate):
                    command = replace(command, respond=True)
                return await original(session, command)
            monkeypatch.setattr(cls, 'send', legacy_send)
            old = run.wire
            old.input_start('legacy-owner')
            await run.wait(lambda: run.call.input_open)
            await request(run)
            assert run.call.preparing is None, 'an old provider must keep the sequential path'
            run.provider.script.append(('say', 'Already answered on the old connection.'))
            old.input_end('legacy-owner', 'Words for the old provider.')
            await run.wait(lambda: run.host.connection_number == 2)
            await run.quiet()
            assert not run.wire.spoken, 'catch-up must not repeat an already-answered turn'
            await run.owner_says('A new question on the continued legacy call.')
            await run.wait(lambda: bool(run.wire.spoken))
            await run.quiet()  # wait for the real page's playback receipt before ending this positive call
            assert sum(s == 'Synthetic greeting heard.' for w in run.provider.connections for s in w.spoken) == 1
    asyncio.run(scenario())


@pytest.mark.parametrize('engine,thinking', ROUTES)
@pytest.mark.parametrize('mail_kind', ['room', 'task_result'])
def test_mail_durable_before_switch_keeps_its_unanswered_obligation(host_world, monkeypatch, engine, thinking, mail_kind):
    async def scenario():
        async with stack(host_world, monkeypatch, engine, thinking=thinking) as run:
            old = run.wire
            old.input_start('owner-holds-switch')
            await run.wait(lambda: run.call.input_open)
            await request(run)
            await run.wait(lambda: run.call.preparing.done())
            original = run.host.mailbox
            listening, durable, cancelled = threading.Event(), threading.Event(), threading.Event()
            async def stopped_after_drain():
                rows = await original()
                listening.set()
                if rows and not durable.is_set():
                    durable.set()
                    try:
                        await asyncio.Future()
                    except asyncio.CancelledError:
                        cancelled.set()
                        raise
                return rows
            monkeypatch.setattr(run.host, 'mailbox', stopped_after_drain)
            # Retire any poll that captured the original method before the hook.
            # The next committed rows must traverse the boundary this probe gates.
            await run.wait(listening.is_set)
            text = 'Synthetic typed question committed during preparation.'
            if mail_kind == 'room':
                run.mail.put({'text': text, 'msg_id': 'r21-synthetic-mail'})
            else:
                from ouroboros.owner_mailbox import write_task_message
                assert write_task_message(run.root, text, run.host.ctx.task_id,
                    source_task_id='synthetic-completed-task', provenance='system', msg_id='r21-synthetic-result')
            await run.wait(durable.is_set)
            assert getattr(run.host, '_pending_mail_answer', False), 'the canonical drain must own its pending answer'
            run.provider.script.append(('say', 'The old connection answered the spoken question.'))
            old.input_end('owner-holds-switch', 'The spoken question.')
            await run.wait(lambda: run.host.connection_number == 2)
            await run.quiet()
            assert cancelled.is_set(), 'probe must reach cancellation of the in-flight mailbox drain'
            await run.wait(lambda: any(text in str(row) for row in run.wire.sent))
            # The offer can be in the websocket between the two lists. Observe
            # its actual answer, rather than interpreting that transient as loss.
            await run.wait(lambda: bool(run.wire.spoken))
            await run.quiet()
            assert len(run.wire.spoken) == 1, 'durable typed mail owes exactly one answer opportunity'
    asyncio.run(scenario())


@pytest.mark.parametrize('engine,thinking', ROUTES)
def test_waiting_socket_death_is_closed_and_settled_before_quiet(host_world, monkeypatch, engine, thinking):
    async def scenario():
        async with stack(host_world, monkeypatch, engine, thinking=thinking) as run:
            old = run.wire
            old.input_start('waiting-owner')
            await run.wait(lambda: run.call.input_open)
            await request(run)
            await run.wait(lambda: run.call.preparing.done())
            lease = run.call.preparing.result()
            await run.wire.websocket.close(code=1012, reason='Synthetic waiting socket restart')
            await run.wait(lambda: lease.task.done())
            assert lease.connection.get('closed_at') and not lease.view._connection_accounting
            assert run.host.connection_number == 1 and run.call.input_open, 'death is observed before any handoff'
            run.provider.script.append(('say', 'Old serving answer.'))
            old.input_end('waiting-owner', 'Words after the idle successor died.')
            await run.wait(lambda: run.host.connection_number == 3)
            await run.quiet()
            assert len(run.provider.connections) == 3
            assert any('Words after the idle successor died.' in str(row) for row in run.wire.sent)
    asyncio.run(scenario())


def test_waiting_grok_connection_cost_participates_in_shared_limit(host_world, providers, monkeypatch):
    host = host_world.host
    now = [100.0]
    monkeypatch.setattr(providers['grok'].module, 'monotonic', lambda: now[0])
    host.quote = tariff(input_cost_per_second=.01)
    wires = []
    def factory(selected, options, *, observe):
        wire = Wire('grok')
        wires.append(wire)
        return providers['grok'].session(selected, options, observe=observe, connect=wire.connect)
    host_world.provider['session'] = factory

    async def scenario():
        ready, retired = asyncio.Event(), asyncio.Event()
        async def serve(session):
            ready.set()
            while not retired.is_set():
                await session.receive()
        with ua.usage_scope(ua.UsageScope(drive_root=host.ctx.drive_root, task_id=host.ctx.task_id, global_limit_usd=.9)):
            current = asyncio.create_task(host.session(serve, LiveOptions()))
            await ready.wait()
            prepared = await host.prepare_session(LiveOptions())
            now[0] = 160.0  # both sockets have accrued 60 seconds, $0.60 each
            async def both_receipts():
                while sum(row.get('cost_usd') or 0 for row in ua.read_usage_records(host.ctx.drive_root)) != pytest.approx(1.2):
                    await asyncio.sleep(.01)
            # The ledger is the boundary, not one tick plus 100 ms of scheduler
            # slack on the loaded test host. A missing waiting meter still fails.
            try:
                await asyncio.wait_for(both_receipts(), 120)
            except asyncio.TimeoutError:
                pass  # close both leases below, then assert the unchanged budget obligation
            before = ua.read_usage_records(host.ctx.drive_root)
            refused = False
            try:
                await host.check()
            except RealtimeSetupError as exc:
                assert exc.code == 'budget_exhausted'
                refused = True
            await prepared.close()
            retired.set()
            await current
            total = sum(row.get('cost_usd') or 0 for row in ua.read_usage_records(host.ctx.drive_root))
        assert total == pytest.approx(1.2), 'all cost is eventually settled once'
        observed = sum(row.get('cost_usd') or 0 for row in before)
        assert refused, f'shared $0.90 limit accepted $1.20 spent; only ${observed:.2f} visible before closing preparation'
    asyncio.run(scenario())


def test_prepared_account_change_is_rechecked_at_adoption(host_world):
    host = host_world.host
    async def scenario():
        lease = await host.prepare_session(LiveOptions())
        original = lease.view.plan.route_fp
        host_world.settings['TEST_CREDENTIAL'] = 'synthetic-replacement-account'
        assert host.turn.observation_fingerprint() != original
        try:
            with pytest.raises(RealtimeSetupError, match='prepared_stale'):
                await lease.run(lambda session: asyncio.sleep(0))
        finally:
            await lease.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('engine,thinking', ROUTES)
def test_languages_recorded_after_preparation_reach_next_session(host_world, monkeypatch, engine, thinking):
    async def scenario():
        async with stack(host_world, monkeypatch, engine, thinking=thinking) as run:
            old = run.wire
            old.input_start('language-owner')
            await run.wait(lambda: run.call.input_open)
            await request(run)
            await run.wait(lambda: run.call.preparing.done())
            old.tool('languages-after-fit', 'set_call_languages', {'languages': ['de-DE']})
            await run.wait(lambda: run.host.state.get('languages') == ('de-DE',))
            old.input_end('language-owner', 'Synthetic owner statement after the language update.')
            await run.wait(lambda: run.host.connection_number >= 2)
            await run.quiet()
            assert run.call.session.options.languages == ('de-DE',), (
                run.call.session.options.languages, run.host.state, run.host.connection_number)
    asyncio.run(scenario())


@pytest.mark.parametrize('field,value', [('voice', 'Kore'), ('instructions', 'New exact owner instruction.'),
    ('pause', 'long'), ('interruptions', False), ('thinking', 'high'), ('search', True)])
def test_changed_live_option_discards_preparation_once(host_world, monkeypatch, field, value):
    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini', thinking=True) as run:
            old = run.wire
            old.input_start('changed-option-owner')
            await run.wait(lambda: run.call.input_open)
            await request(run)
            await run.wait(lambda: run.call.preparing.done())
            assert getattr(run.host.config, field) != value
            run.host.config = replace(run.host.config, **{field: value})
            old.input_end('changed-option-owner', 'Owner words while the options changed.')
            await run.wait(lambda: run.host.connection_number == 3)
            await run.quiet()
            actual = getattr(run.call.session.options, field)
            assert (actual.endswith(value) if field == 'instructions' else actual == value)
            assert len(run.provider.connections) == 3 and run.provider.connections[1].closed
    asyncio.run(scenario())


@pytest.mark.parametrize('thinking', [False, True])
@pytest.mark.parametrize('resumable', [False, True])
def test_exact_documented_gemini_expiry_continues(host_world, monkeypatch, thinking, resumable):
    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini', thinking=thinking) as run:
            if resumable:
                run.wire.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': 'synthetic-expiry-handle'}})
                await run.wait(lambda: run.host.state.get('resume_handle') == 'synthetic-expiry-handle')
            await run.wire.websocket.close(code=1011, reason='Deadline expired before operation could complete.')
            for _ in range(1500):
                if run.running.done() or len(run.provider.connections) >= 2:
                    break
                await asyncio.sleep(.01)
            continued = len(run.provider.connections) == 2
        assert continued, run.result
    asyncio.run(scenario())


@pytest.mark.parametrize('engine', ['openai', 'grok'])
def test_heard_but_interrupted_answers_prove_continuation(host_world, monkeypatch, engine):
    async def scenario():
        async with stack(host_world, monkeypatch, engine) as run:
            for number in range(1, 5):
                wire = run.wire
                wire.say(f'Synthetic audible answer before loss {number}.', complete=False)
                item = f'item-{wire.number}-{wire.responses}'
                await run.wait(lambda: run.call.played.get(item, 0) >= 100)
                wire.input_start(f'owner-interrupt-{number}')
                await run.wait(lambda: run.call.input_open and not run.call.active_responses)
                await wire.websocket.close(code=1012, reason='Synthetic transport restart')
                for _ in range(1500):
                    if run.running.done() or run.host.connection_number == number + 1:
                        break
                    await asyncio.sleep(.01)
                if run.running.done():
                    break
                await run.quiet()
            count = len(run.provider.connections)
        assert count == 5, (count, run.result)
    asyncio.run(scenario())


@pytest.mark.parametrize('engine,thinking', [('gemini', False), ('gemini', True), ('openai', False), ('grok', False)])
@pytest.mark.parametrize('break_offer', [False, True])
def test_handoff_mail_offer_keeps_custody_until_its_write_succeeds(host_world, monkeypatch, engine, thinking, break_offer):
    from ouroboros.contracts.live_conversation import ContextUpdate
    from tests.test_live_stack_preparation import request
    async def scenario():
        async with stack(host_world, monkeypatch, engine, thinking=thinking) as run:
            old = run.wire
            original = type(run.call.session).send
            broken = []
            async def send(session, command):
                if (break_offer and not broken and run.host.connection_number == 2
                        and isinstance(command, ContextUpdate) and command.respond
                        and any('New room context is in the fitted history' in str(row) for row in command.messages)):
                    broken.append(True)
                    await session.ws.close(code=1012, reason='Synthetic loss while sending the pending-mail offer')
                return await original(session, command)
            monkeypatch.setattr(type(run.call.session), 'send', send)
            old.input_start('owner-holds-mail-switch')
            await run.wait(lambda: run.call.input_open)
            await request(run)
            await run.wait(lambda: run.call.preparing.done())
            text = 'Synthetic room request whose post-handoff offer meets a closed socket.'
            run.mail.put({'text': text, 'msg_id': 'review-mail-offer-loss'})
            await run.wait(lambda: any('New room context is in the fitted history' in str(row) for row in run.call.pending_offers))
            run.provider.script.append(('say', 'Old connection answered the spoken owner turn.'))
            old.input_end('owner-holds-mail-switch', 'Synthetic spoken turn before the handoff.')
            await run.wait(lambda: run.host.connection_number == (3 if break_offer else 2))
            await run.quiet()
            # Allow the normal offer worker to process an event-loop turn after quiet.
            await asyncio.sleep(.1)
            assert any(text in str(row) for row in run.host.messages)
            assert len(run.wire.spoken) == 1, dict(
                broken=broken, spoken=run.wire.spoken, pending=run.call.pending_offers,
                mail_marker=getattr(run.host, '_pending_mail_answer', False),
                connections=len(run.provider.connections))
    asyncio.run(scenario())


@pytest.mark.parametrize('cancel', [False, True])
def test_failed_offer_restored_before_newer_arrivals(cancel):
    from skills.live_conversation.driver import _Call
    async def scenario():
        older, newer = {'content':'already owned'}, {'content':'arrived during write'}
        class Attention:
            async def deliver(self, session, **kwargs): return []
            def include_images(self, rows): pass
        class Socket:
            async def send(self, command):
                call.pending_offers.append(newer)
                if cancel:
                    raise asyncio.CancelledError()
                raise OSError('lost write')
        call = object.__new__(_Call)
        call.pending_offers, call.session, call.attention = [older], Socket(), Attention()
        call.host = SimpleNamespace(renewal_due=lambda: False, state={})
        call.connected = asyncio.Event()
        call.connected.set()
        call.quiet = lambda: True
        with pytest.raises(asyncio.CancelledError if cancel else OSError):
            await call.offer_pending()
        assert call.pending_offers == [older, newer]
    asyncio.run(scenario())
