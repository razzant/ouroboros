"""Overlapping fresh sessions over the real gateway, host, driver and native adapters."""
import asyncio
import json

import pytest

from tests._extension_loader_shared import _clear_loader_state  # noqa: F401
from tests._live_stack import stack
from tests._live_stack_wire import NativeWire
from tests.test_live_conversation import host_world as host_world

ROUTES = [('openai', False), ('grok', False), ('gemini', False), ('gemini', True)]


async def request(run):
    async def renew():
        run.call.request_renewal()
    await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(renew(), run.call.writer.get_loop()))


@pytest.mark.parametrize('engine,thinking', ROUTES)
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


@pytest.mark.parametrize('engine,thinking', ROUTES)
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
@pytest.mark.parametrize('engine,thinking', ROUTES)
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


@pytest.mark.parametrize('engine,thinking', ROUTES)
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


@pytest.mark.parametrize('engine,thinking', ROUTES)
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
