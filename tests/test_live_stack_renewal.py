"""Renewal and transport loss over the same gateway/host/native-wire harness."""
import asyncio
import json

import pytest

from tests._extension_loader_shared import _clear_loader_state  # noqa: F401
from tests._live_stack import stack
from tests.test_live_conversation import host_world as host_world


@pytest.mark.parametrize('engine,thinking', [('openai', False), ('grok', False), ('gemini', False), ('gemini', True)])
def test_full_stack_host_window_renews_with_pending_canonical_facts(host_world, monkeypatch, engine, thinking):
    async def scenario():
        async with stack(host_world, monkeypatch, engine, thinking=thinking) as run:
            run.wire.say('Synthetic answer before renewal.', complete=False)
            await run.wait(lambda: bool(run.call.active_responses))
            run.page.share(True, 'screen')
            await run.wait(lambda: any('started showing the screen' in str(r) for r in run.host.messages))
            before = len(run.provider.connections)
            if engine == 'grok':
                # The declared Grok window is an estimator window; token receipts
                # cannot change that basis. Add measured canonical text until the
                # real host growth calculation requests renewal.
                text = 'Synthetic pending room evidence. ' * 4500
                run.mail.put({'text': text, 'msg_id': 'window-growth'})
                await run.wait(lambda: any(text in str(r) for r in run.host.messages))
            else:
                run.wire.prompt = run.host.model.facts.conversation_window_tokens - run.host.model.facts.working_reserve_tokens + 1
            # Finish the native caption and receipt; page playback still gates quiet.
            if engine == 'gemini':
                run.wire.put({'serverContent': {'outputTranscription': {'text': ' Finished.', 'finished': True}}})
            else:
                run.wire.put({'type': 'response.output_audio_transcript.done',
                              'item_id': f'item-{run.wire.number}-{run.wire.responses}',
                              'transcript': 'Synthetic answer before renewal. Finished.'})
            run.wire.complete()
            await run.wait(lambda: len(run.provider.connections) > before)
            await run.quiet()
            assert any('started showing the screen' in str(v) for v in run.wire.sent), 'fresh fit must carry the pending fact'
            assert sum('started showing the screen' in str(r) for r in run.host.messages) == 1
            run.provider.script.append(('say', 'Synthetic conversation after renewal.'))
            await run.owner_says('Synthetic owner after renewal.')
            await run.quiet()
        assert run.result[1]['execution_status'] == 'ok', run.result
        assert any('Synthetic owner after renewal.' in str(r) for r in run.transcript())
    asyncio.run(scenario())


@pytest.mark.parametrize('engine', ['grok', 'openai', 'gemini'])
def test_full_stack_unfit_room_mail_ends_typed_instead_of_silently_waiting(host_world, monkeypatch, engine):
    async def scenario():
        async with stack(host_world, monkeypatch, engine) as run:
            text = 'Synthetic pending room evidence. ' * 30000
            run.mail.put({'text': text, 'msg_id': 'physical-overflow'})
            await run.wait(run.running.done)
        assert run.result[1]['reason_code'] == 'context_capacity', run.result
        assert any(text in str(r) for r in run.host.messages), 'the refused delivery stays canonical'
        assert len(run.provider.connections) == 1
        assert not any(text in str(row) for row in run.wire.sent), 'overflow is never written to the serving socket'
    asyncio.run(scenario())


@pytest.mark.parametrize('thinking', [False, True])
@pytest.mark.parametrize('reason', ['audio_drop', 'compression', 'go_away', 'transport_drop'])
def test_full_stack_gemini_retention_or_resume_preserves_conversation(host_world, monkeypatch, thinking, reason):
    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini', thinking=thinking) as run:
            before = len(run.provider.connections)
            if reason in {'audio_drop', 'compression'}:
                run.wire.audio_tokens = 300
                if reason == 'compression':
                    # Intermediate provider sample crosses compression trigger;
                    # settled prompt is small, so the host itself need not renew.
                    run.wire.put({'usageMetadata': {'promptTokenCount': 92000}})
                run.wire.say('Synthetic before token fall.')
                await run.wait(lambda: run.call.session.held_audio == 300)
                await run.quiet()
                if reason == 'compression':
                    run.wire.put({'usageMetadata': {'promptTokenCount': 92000}})
                run.wire.audio_tokens = 50
                run.wire.say('Synthetic after token fall.')
                if reason == 'compression':
                    await run.wait(lambda: run.call.session.held_audio == 50)
                    await run.quiet()
                    assert len(run.provider.connections) == before
                    run.host.wire.flush()
                    assert any('compression_explains_drop' in r['raw'] for r in run.journal())
                else:
                    await run.wait(lambda: len(run.provider.connections) > before)
                    await run.quiet()
                    assert not run.wire.sent[0]['setup']['sessionResumption'], 'audio eviction needs a fresh retelling'
            else:
                run.wire.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': 'synthetic-resume-handle'}})
                await run.wait(lambda: run.host.state.get('resume_handle') == 'synthetic-resume-handle')
                if reason == 'transport_drop':
                    await run.wire.websocket.close(code=1012, reason='synthetic service restart')
                else:
                    run.wire.put({'goAway': {'timeLeft': '30s'}})
                await run.wait(lambda: len(run.provider.connections) > before)
                await run.quiet()
                assert run.wire.sent[0]['setup']['sessionResumption'] == {'handle': 'synthetic-resume-handle'}
                assert not run.host.timing['connections'][-1]['prepared_ahead'], 'native resumption stays sequential'
                assert len([text for wire in run.provider.connections for text in wire.spoken
                            if text == 'Synthetic greeting heard.']) == 1
            run.provider.script.append(('say', 'Synthetic continued conversation.'))
            await run.owner_says('Synthetic owner still here.')
            await run.quiet()
        assert run.result[1]['execution_status'] == 'ok', run.result
        assert any('Synthetic owner still here.' in str(r) for r in run.transcript())
        assert any(json.loads(r['raw']).get('type') == 'call.offer_wait' for r in run.journal() if r['direction'] == 'host')
    asyncio.run(scenario())


@pytest.mark.parametrize('engine', ['openai', 'grok', 'gemini'])
def test_nonresumable_transport_drop_continues_from_fitted_history(host_world, monkeypatch, engine):
    async def scenario():
        async with stack(host_world, monkeypatch, engine) as run:
            run.provider.script.append(('say', 'Answer before network loss.'))
            await run.owner_says('Owner words before network loss.')
            await run.quiet()
            await run.wire.websocket.close(code=1012, reason='synthetic service restart')
            await run.wait(lambda: len(run.provider.connections) == 2 and run.host.connection_number == 2)
            await run.quiet()
            assert any('Owner words before network loss.' in str(v) for v in run.wire.sent)
            run.provider.script.append(('say', 'Answer after network loss.'))
            await run.owner_says('Owner words after network loss.')
            await run.quiet()
            assert sum(s == 'Synthetic greeting heard.' for w in run.provider.connections for s in w.spoken) == 1
        assert run.result[1]['execution_status'] == 'ok', run.result
        assert sum('Owner words before network loss.' in str(r) for r in run.transcript()) == 1
        assert run.journal()
    asyncio.run(scenario())


@pytest.mark.parametrize('resumable', [False, True])
@pytest.mark.parametrize('thinking', [False, True])
def test_full_stack_loss_during_a_tool_result_respects_the_native_handle(host_world, monkeypatch, resumable, thinking):
    """R19-2 through the real stack: the writer that meets the close first does not end a resumable call."""
    from websockets.exceptions import ConnectionClosedError

    async def scenario():
        loop = asyncio.get_running_loop()
        async with stack(host_world, monkeypatch, 'gemini', thinking=thinking) as live:
            native = live.call.session
            send, recv = native.ws.send, native.ws.recv
            if resumable:
                live.wire.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': 'synthetic-tool-handle'}})
                await live.wait(lambda: live.host.state.get('resume_handle') == 'synthetic-tool-handle')
            async def close_before_result(raw):
                if 'toolResponse' in json.loads(raw):
                    closing = asyncio.run_coroutine_threadsafe(
                        live.wire.websocket.close(code=1012, reason='synthetic service restart'), loop)
                    await asyncio.wrap_future(closing)
                await send(raw)
            async def read_later(*args, **kwargs):
                try:
                    return await recv(*args, **kwargs)
                except ConnectionClosedError:
                    await asyncio.sleep(.1)
                    raise
            monkeypatch.setattr(native.ws, 'send', close_before_result)
            monkeypatch.setattr(native.ws, 'recv', read_later)
            original_receive = native.receive
            async def delayed_close():
                events = await original_receive()
                if any(e.kind == 'closed' for e in events):
                    await asyncio.sleep(.1)
                return events
            monkeypatch.setattr(native, 'receive', delayed_close)
            live.wire.tool('inflight-result', 'attend', {'see': None, 'look': None, 'revisit_after_s': None})
            await live.wait(lambda: len(live.provider.connections) == 2 and live.host.connection_number == 2)
            if resumable:
                await live.wait(lambda: 'inflight-result' in live.wire.outputs)
            await live.quiet()
            assert 'inflight-result' in live.host.executed, 'the completed effect is already durable'
            if resumable:
                assert len(live.provider.connections) == 2, (
                    'tool send bypassed native resumption', live.running.result() if live.running.done() else None)
                assert 'inflight-result' in live.wire.outputs
            else:
                assert any('inflight-result' in str(v) for v in live.wire.sent), 'fresh fit carries the committed result'
                assert not live.wire.outputs, 'a fresh session must not receive an orphan native tool result'
            assert sum(c.get('call_id') == 'inflight-result' for c in live.host.trace['tool_calls']) == 1
    asyncio.run(scenario())


@pytest.mark.parametrize('engine', ['openai', 'grok'])
def test_openai_grok_tool_result_write_loss_continues_without_repeating_effect(host_world, monkeypatch, engine):
    """A committed result travels in the fresh fit when its native write loses the connection."""
    from websockets.exceptions import ConnectionClosedError

    async def scenario():
        loop = asyncio.get_running_loop()
        async with stack(host_world, monkeypatch, engine) as live:
            native = live.call.session
            send, recv = native.ws.send, native.ws.recv
            async def close_before_result(raw):
                event = json.loads(raw)
                if event.get('type') == 'conversation.item.create' and event['item'].get('type') == 'function_call_output':
                    closing = asyncio.run_coroutine_threadsafe(
                        live.wire.websocket.close(code=1012, reason='synthetic service restart'), loop)
                    await asyncio.wrap_future(closing)
                await send(raw)
            async def read_later(*args, **kwargs):
                try:
                    return await recv(*args, **kwargs)
                except ConnectionClosedError:
                    await asyncio.sleep(.1)  # the writer demonstrably fails first
                    raise
            monkeypatch.setattr(native.ws, 'send', close_before_result)
            monkeypatch.setattr(native.ws, 'recv', read_later)
            live.wire.tool('inflight-result', 'attend', {'see': None, 'look': None, 'revisit_after_s': None})
            await live.wait(lambda: len(live.provider.connections) == 2 and live.host.connection_number == 2)
            await live.quiet()
            assert 'inflight-result' in live.host.executed
            assert any('inflight-result' in str(v) for v in live.wire.sent)
            items = [v['item'] for v in live.wire.sent if v.get('type') == 'conversation.item.create']
            if engine == 'openai':
                pair = [v['type'] for v in items if v.get('call_id') == 'inflight-result']
                assert pair == ['function_call', 'function_call_output'], 'exactly one atomic fitted pair'
            else:
                assert not live.wire.outputs, 'Grok puts canonical history in its instructions'
            assert sum(c.get('call_id') == 'inflight-result' for c in live.host.trace['tool_calls']) == 1
    asyncio.run(scenario())


@pytest.mark.parametrize('engine', ['openai', 'grok', 'gemini'])
@pytest.mark.parametrize('failure', ['event', 'close'])
def test_provider_refusal_stays_terminal_without_reconnection(host_world, monkeypatch, engine, failure):
    async def scenario():
        async with stack(host_world, monkeypatch, engine) as run:
            if failure == 'event':
                run.wire.put({'type': 'error', 'error': {'code': 'invalid_request', 'message': 'Synthetic refusal'}})
            else:
                await run.wire.websocket.close(code=1008, reason='Synthetic policy refusal')
            await asyncio.wait_for(asyncio.shield(run.running), 15)
        assert run.result[1]['reason_code'] == ('provider_error' if failure == 'event' else 'provider_closed')
        assert len(run.provider.connections) == 1
    asyncio.run(scenario())


@pytest.mark.parametrize('engine', ['openai', 'grok', 'gemini'])
def test_repeated_transport_losses_without_an_answer_reach_existing_bound(host_world, monkeypatch, engine):
    async def scenario():
        async with stack(host_world, monkeypatch, engine) as run:
            for number in range(1, 5):
                await run.wire.websocket.close(code=1012, reason='synthetic restart')
                if number < 4:
                    await run.wait(lambda: len(run.provider.connections) == number + 1 and run.host.connection_number == number + 1)
                    await run.quiet()
                else:
                    await asyncio.wait_for(asyncio.shield(run.running), 15)
        assert run.result[1]['reason_code'] == 'reconnect_failed'
        assert len(run.provider.connections) == 4, 'the existing three unproven reconnects are the bound'
    asyncio.run(scenario())


@pytest.mark.parametrize('engine', ['openai', 'grok', 'gemini'])
def test_transport_replacement_open_refusal_ends_with_its_typed_reason(host_world, monkeypatch, engine):
    from tests._live_stack_wire import NativeWire
    async def scenario():
        send = NativeWire.send
        async def refused(wire, raw):
            if wire.number == 2:
                wire.put({'type': 'error', 'error': {'code': 'invalid_request', 'message': 'Synthetic open refusal'}})
                return
            return await send(wire, raw)
        monkeypatch.setattr(NativeWire, 'send', refused)
        async with stack(host_world, monkeypatch, engine) as run:
            await run.wire.websocket.close(code=1012, reason='synthetic restart')
            await asyncio.wait_for(asyncio.shield(run.running), 15)
        assert run.result[1]['reason_code'] == 'provider_error'
        assert len(run.provider.connections) == 2
        assert all(w.closed for w in run.provider.connections)
    asyncio.run(scenario())


@pytest.mark.parametrize('during_call,handle_after,outcome', [
    (during, after, 'answered')
    for during in ('none', 'cleared', 'predates')
    for after in ('absent', 'with-answer', 'after-answer')
] + [('predates', 'cleared-after', 'answered'),
     ('cleared', 'absent', 'silent'), ('predates', 'absent', 'silent'),
     ('predates', 'after-answer', 'silent'),
     ('cleared', 'absent', 'unanswered'), ('predates', 'absent', 'unanswered'),
     ('cleared', 'absent', 'same-turn'), ('predates', 'absent', 'same-turn')])
@pytest.mark.parametrize('thinking', [False, True])
def test_r28_goaway_while_tool_pending(host_world, monkeypatch, during_call, handle_after, outcome, thinking):
    """Follow direct delivery through renewal: only a post-write handle can resume its result."""
    import threading
    from tests._live_stack_wire import NativeWire
    from ouroboros.tools.tool_result import ToolResult
    entered, release = threading.Event(), threading.Event()
    effects = []
    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini', thinking=thinking) as live:
            execute = live.host.tools.execute_result
            def blocked(name, args):
                if name != 'list_available_tools': return execute(name, args)
                effects.append(name)
                return ToolResult(status='ok', code='OK', text='Completed exactly once.')
            monkeypatch.setattr(live.host.tools, 'execute_result', blocked)
            tool_batch = live.host.tool_batch
            async def pending(calls):
                entered.set()
                while not release.is_set():
                    await asyncio.sleep(.01)
                return await tool_batch(calls)
            monkeypatch.setattr(live.host, 'tool_batch', pending)
            send = NativeWire.send
            async def answer_then_handle(wire, raw):
                is_result = 'toolResponse' in json.loads(raw)
                if is_result and handle_after == 'cleared-after':
                    wire.put({'sessionResumptionUpdate': {'resumable': False}})
                if is_result and handle_after == 'with-answer':
                    # Reissuing the same value still supplies new authority. It
                    # arrives with the response, before its answer completes.
                    wire.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': 'before-call'}})
                await send(wire, raw)
                if is_result and handle_after == 'after-answer':
                    wire.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': 'after-answer'}})
            monkeypatch.setattr(NativeWire, 'send', answer_then_handle)
            old = live.wire
            old.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': 'before-call'}})
            await live.wait(lambda: live.host.state.get('resume_handle') == 'before-call')
            spoken_before = len(old.spoken)
            complete = old.complete
            if outcome in ('unanswered', 'same-turn'):
                old.complete = lambda: None
            if outcome == 'unanswered':
                old.answer = complete  # finish the tool turn, with no answer
            name, args = ('attend', {'see': None, 'look': None, 'revisit_after_s': None}) if outcome == 'silent' else ('list_available_tools', {})
            if outcome == 'silent' and handle_after == 'after-answer':
                write = live.call.session.ws.send
                async def write_then_handle(raw):
                    await write(raw)
                    if 'toolResponse' in json.loads(raw):
                        await live.wait(lambda: live.host.state.get('resume_handle') == 'after-answer')
                monkeypatch.setattr(live.call.session.ws, 'send', write_then_handle)
            old.tool('pending-native', name, args)
            old.complete = complete
            try:
                await live.wait(entered.is_set)
                if during_call == 'cleared':
                    old.put({'sessionResumptionUpdate': {'resumable': False}})
                    await live.wait(lambda: live.host.state.get('resume_handle') == '')
                if during_call != 'none':
                    old.put({'goAway': {'timeLeft': '30s'}})
                    await live.wait(lambda: live.call.renewing and live.call.resume)
                release.set()
                await live.wait(lambda: 'pending-native' in old.outputs or len(live.provider.connections) > 1)
                assert 'pending-native' in old.outputs, 'a fitting result must reach the issuing socket'
                fresh = during_call != 'none' and handle_after in ('absent', 'cleared-after')
                if during_call != 'none':
                    await live.wait(lambda: len(live.provider.connections) == 2 and live.wire.sent)
                    expected = {} if fresh else {'handle': 'before-call' if handle_after == 'with-answer' else 'after-answer'}
                    assert live.wire.sent[0]['setup']['sessionResumption'] == expected
                await live.wait(lambda: not live.call.pending_offers)
                await live.quiet()
                assert effects == ([] if outcome == 'silent' else ['list_available_tools'])
                assert len(old.spoken) - spoken_before == (1 if outcome in ('answered', 'same-turn') else 0)
                assert not live.call.deferred_tool_outputs and not live.call.pending_offers
                assert len(live.provider.connections) == (1 if during_call == 'none' else 2)
                assert sum('toolResponse' in row for row in old.sent) == 1
                if during_call != 'none':
                    assert 'pending-native' not in live.wire.outputs
                    if fresh:
                        fitted = json.loads(live.call.session.selected.messages_json)
                        assert any(row.get('tool_call_id') == 'pending-native' for row in fitted)
                        if outcome != 'silent':
                            assert 'Completed exactly once.' in json.dumps(live.wire.sent[0]['setup'])
                    if fresh and outcome != 'silent':
                        assert any('The requested tool batch returned' in json.dumps(row)
                                   for row in live.wire.sent if 'clientContent' in row)
                        assert len(live.wire.spoken) == 1
                    else:
                        assert not live.wire.spoken
                        assert not any('The requested tool batch returned' in json.dumps(row)
                                       for row in live.wire.sent if 'clientContent' in row)
                    if not fresh:
                        # A later GoAway has no new handle or tool write. The old
                        # connection's marker must not invalidate this checkpoint.
                        live.wire.put({'goAway': {'timeLeft': '30s'}})
                        await live.wait(lambda: len(live.provider.connections) == 3 and live.wire.sent)
                        assert live.wire.sent[0]['setup']['sessionResumption'] == expected
                        await live.quiet()
                        assert not live.wire.outputs and not live.wire.spoken
            finally:
                release.set()
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())


@pytest.mark.parametrize('thinking', [False, True])
@pytest.mark.parametrize('second', ['silent', 'ordinary', 'single'])
def test_ordinary_continuation_survives_later_batches(host_world, monkeypatch, record_property, thinking, second):
    import threading
    entered, release = threading.Event(), threading.Event()

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini', thinking=thinking) as live:
            host, call, old = live.host, live.call, live.wire
            old.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': 'before-tool'}})
            await live.wait(lambda: host.state.get('resume_handle') == 'before-tool')
            tool_batch = host.tool_batch
            batches = []

            async def gated(calls):
                batches.append([c['id'] for c in calls])
                if len(batches) == 1:
                    entered.set()
                    while not release.is_set():
                        await asyncio.sleep(.01)
                return await tool_batch(calls)

            monkeypatch.setattr(host, 'tool_batch', gated)
            complete = old.complete
            old.complete = lambda: None
            old.answer = lambda: None  # the model has not begun any answer to the first result
            old.tool('first-ordinary', 'list_available_tools', {})
            old.complete = complete
            try:
                await live.wait(entered.is_set)
                if second != 'single':
                    # A second native message of the same open model turn (NON_BLOCKING calling).
                    name, args = (('attend', {'see': None, 'look': None, 'revisit_after_s': None})
                                  if second == 'silent' else ('get_task_result', {'task_id': 'none'}))
                    old.put({'toolCall': {'functionCalls': [{'id': 'second-call', 'name': name, 'args': args}]}})
                    await live.wait(lambda: call.tool_queue.qsize() == 1)
                old.put({'goAway': {'timeLeft': '30s'}})
                await live.wait(lambda: call.renewing and call.resume)
                release.set()
                await live.wait(lambda: 'first-ordinary' in old.outputs and (second == 'single' or 'second-call' in old.outputs))
                old.put({'serverContent': {'turnComplete': True,
                    **({'interactionStatus': 'IDLE'} if thinking else {})}})
                await live.wait(lambda: len(live.provider.connections) == 2 and live.wire.sent)
                await live.wait(lambda: not call.pending_offers)
                await live.quiet()
                fitted = json.loads(call.session.selected.messages_json)
                result = dict(probe='B', thinking=thinking, second=second, batches=batches,
                    setup=live.wire.sent[0]['setup']['sessionResumption'], pointer=any('The requested tool batch returned' in json.dumps(row)
                                for row in live.wire.sent if 'clientContent' in row),
                    spoken=live.wire.spoken, old_spoken=old.spoken[1:],
                    in_fit=any(r.get('tool_call_id') == 'first-ordinary' for r in fitted),
                    old_responses=sum('toolResponse' in r for r in old.sent),
                    new_responses=sum('toolResponse' in r for r in live.wire.sent))
                record_property('continuation_batches', json.dumps(result))
                assert result['setup'] == {} and result['in_fit'] and not result['new_responses'], result
                assert result['pointer'] and len(result['spoken']) == 1, result
                assert result['old_responses'] == (1 if second == 'single' else 2), result
                assert batches == [['first-ordinary']] + ([] if second == 'single' else [['second-call']])
                ids = [r.get('tool_call_id', r.get('call_id')) for r in host.trace['tool_calls']]
                assert ids.count('first-ordinary') == 1
                assert ids.count('second-call') == int(second != 'single')
            finally:
                release.set()
        assert live.result[1]['execution_status'] == 'ok', live.result

    asyncio.run(scenario())
