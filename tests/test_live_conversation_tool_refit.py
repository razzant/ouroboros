"""Completed tool effects must fit the live window before native delivery."""
import asyncio
import json
from dataclasses import replace

import pytest

from ouroboros.contracts.live_conversation import ContextUpdate, LiveEvent, LiveOptions, Respond, ToolOutputs
from ouroboros.contracts.realtime import RealtimeSetupError
from ouroboros import call_protocol as cp
from skills.live_conversation.driver import _Call
from tests.test_live_conversation import host_world as host_world  # noqa: F401
from tests.test_live_conversation_driver import Batch, Host, Hub, install_tools, started, until
from tests.test_live_usage import token_usage


@pytest.mark.parametrize('held', [15000, 39000])
def test_tool_result_cannot_overflow_native_held_context(host_world, held):
    """Port of rev-astra R2-1: real host/read_file/fit, fake provider transport."""
    host = host_world.host
    host.model = replace(host.model, facts=replace(host.model.facts,
        conversation_window_tokens=50000, working_reserve_tokens=10000))
    source = host.ctx.repo_dir / 'review-facts.txt'
    source.write_text('A retained source fact for this call.\n' * 800)

    async def consume(session):
        text = host._selected_raw_estimate
        host.response_done('previous', usage=token_usage(text_input_tokens=text,
            audio_input_tokens=held - text), raw_usage={}, status='completed')
        assert not host.renewal_due()
        host.hub = Hub()
        call = _Call(host)
        call.session = session
        call.connected.set()
        sent = []
        async def send(command): sent.append(command)
        session.send = send
        call.tool_queue.put_nowait({'id': 'read-large', 'type': 'function', 'function': {
            'name': 'read_file', 'arguments': json.dumps({'path': str(source), 'max_lines': 1000})}})
        worker = asyncio.create_task(call.tools())
        try:
            await asyncio.wait_for(call.tool_queue.join(), 15)
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        result = next(row for row in host.messages if row.get('tool_call_id') == 'read-large')
        assert 'retained source fact' in result['content'], result
        assert host.renewal_due() is (held == 39000)
        assert any(isinstance(command, ToolOutputs) for command in sent) is (held == 15000), (
            'ToolOutputs crossed the open connection despite a measured window overflow',
            len(result['content']), host._held_input, host._history_estimate() - host._history_at_receipt,
            host.model.facts.conversation_window_tokens)
        assert host.executed['read-large'][1] == [result], 'the execution receipt survives withheld delivery'
    asyncio.run(host.session(consume, LiveOptions()))


@pytest.mark.parametrize('growth', ['result', 'schemas'])
@pytest.mark.parametrize('held', [15000, 39000])
@pytest.mark.parametrize('silent', [False, True])
@pytest.mark.parametrize('completed_receipt', [False, True])
def test_tool_growth_refits_without_a_quiet_boundary_and_answers_from_fitted_history(growth, held, silent, completed_receipt):
    async def scenario():
        host = Host()
        history, executed, fitted = [], [], []
        original_session = host.session
        async def session(consume, options):
            fitted.append((list(history), list(host.schemas)))
            return await original_session(consume, options)
        async def batch(calls):
            executed.extend(call['id'] for call in calls)
            row = {'role': 'tool', 'tool_call_id': 'work', 'content': 'The effect already completed.'}
            history.append(row)
            if growth == 'schemas':
                host.schemas.append({'type': 'function', 'function': {'name': 'newly_enabled', 'description': 'New schema'}})
            host.due = held + 9494 + 10000 >= 50000
            return Batch([row], silent=('work',) if silent else ())
        host.session, host.tool_batch = session, batch
        task, first = await started(host)
        finalized = []
        first.final_captions = lambda: finalized.append(True) or [LiveEvent('caption', {
            'speaker': 'owner', 'item_id': 'tail', 'text': 'The retained tail.', 'final': True, 'incomplete': True})]
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.emit('resumable', handle='must-not-resume-overflow')
            first.emit('response_started', response_id='tool-turn')
            if completed_receipt:
                first.emit('response_done', response_id='tool-turn', usage={}, raw_usage={}, status='completed', continues=True)
            first.renewal_ready = False  # the native pending call cannot finish without its withheld result
            first.emit('tool_call', call_id='work', name='enable_tools' if growth == 'schemas' else 'effect', arguments='{}')
            await until(lambda: len(host.sessions) > 1 or any(isinstance(x, ToolOutputs) for x in first.sent))
            if held == 15000:
                assert len(host.sessions) == 1
                assert not finalized
                assert next(x for x in first.sent if isinstance(x, ToolOutputs)) == ToolOutputs(
                    tuple(history), tuple(host.schemas), silent=('work',) if silent else ())
                assert [x for x in first.sent if isinstance(x, Respond)] == [Respond(host.opening)]
            else:
                assert not any(isinstance(x, ToolOutputs) for x in first.sent)
                assert host.options[1].resume_handle == ''
                assert fitted[1] == (history, host.schemas)
                assert finalized == [True]
                assert host.captions[0]['text'] == '[Partial transcript] The retained tail.'
                second = host.sessions[1]
                assert not second.sent, 'the replacement must be ready before opening a response'
                second.emit('ready', input_rate=24000, output_rate=24000)
                second.emit('ready', input_rate=24000, output_rate=24000)
                # Silence is finished, including before the first spoken words. A non-silent
                # result offers one truthful continuation at the quiet boundary.
                if not silent:
                    await until(lambda: any(isinstance(x, ContextUpdate) for x in second.sent))
                await asyncio.sleep(.04)
                responses = [x for x in second.sent if isinstance(x, ContextUpdate)]
                assert len(responses) == int(not silent)
                if responses:
                    assert 'requested tool batch returned' in str(responses[0].messages)
                    assert host.opening not in str(responses[0].messages)
                    second.emit('response_started', response_id='answer')
                    second.emit('response_done', response_id='answer', usage={}, raw_usage={}, status='completed', continues=False)
                    await until(lambda: any(row[:2] == ('done', 'answer') for row in host.usage))
                assert not any(isinstance(x, ToolOutputs) for x in second.sent), 'the fitted result is not appended twice'
                assert len([row for row in host.usage if row[:2] == ('done', 'tool-turn')]) == int(completed_receipt), (
                    'refitting must not fabricate a receipt for an unfinished response')
                # The old outstanding-tool liveness must not pin subsequent refits.
                host.due = True
                await until(lambda: len(host.sessions) == 3)
                assert executed == ['work'], 'renewal never re-runs a completed effect'
            assert executed == ['work']
            assert history[0]['content'] == 'The effect already completed.'
        finally:
            host.hub.ended_by_owner = True
            await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


@pytest.mark.parametrize('overflow', [False, True])
def test_ending_tool_withholds_overflow_without_cutting_off_the_goodbye(tmp_path, overflow):
    async def scenario():
        host = Host()
        install_tools(host, tmp_path)
        original_batch = host.tool_batch
        async def batch(calls):
            changes = await original_batch(calls)
            host.due = overflow
            return changes
        host.tool_batch = batch
        task, first = await started(host)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.emit('audio', item_id='goodbye', rate=24000, pcm=b'\0' * 48000)
            first.emit('tool_call', call_id='end', name='end_call', arguments='{}')
            await until(lambda: host.hub.ended_by_ouroboros)
            await asyncio.sleep(.05)
            assert any(isinstance(x, ToolOutputs) for x in first.sent) is (not overflow)
            assert not task.done() and len(host.sessions) == 1, 'the owner still needs to hear the goodbye'
            host.hub.events.append(cp.PeerControl('owner', cp.PEER_OWNER,
                {'type': 'played', 'item_id': 'goodbye', 'played_ms': 1000}))
            await asyncio.wait_for(task, 2)
            assert (cp.STATE_CLOSED, 'ouroboros_bye') in host.hub.states
        finally:
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_tool_refit_failure_retains_completed_effect_without_retry():
    async def scenario():
        host = Host()
        executed, history = [], []
        original_session = host.session
        async def session(consume, options):
            if host.sessions: raise RealtimeSetupError('context_capacity', 'The replacement cannot fit.')
            return await original_session(consume, options)
        async def batch(calls):
            executed.extend(call['id'] for call in calls)
            history.append({'role': 'tool', 'tool_call_id': 'effect', 'content': 'Saved once.'})
            host.due = True
            return Batch(history)
        host.session, host.tool_batch = session, batch
        task, first = await started(host)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.renewal_ready = False
            first.emit('tool_call', call_id='effect', name='effect', arguments='{}')
            await until(lambda: task.done() or any(isinstance(x, ToolOutputs) for x in first.sent))
            assert not any(isinstance(x, ToolOutputs) for x in first.sent)
            with pytest.raises(RealtimeSetupError, match='context_capacity'):
                await asyncio.wait_for(task, 2)
            assert executed == ['effect'] and history[0]['content'] == 'Saved once.'
            assert len(host.sessions) == 1
        finally:
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


@pytest.mark.parametrize('overflow', [False, True])
def test_tool_refit_includes_queued_effects_before_fitting_the_replacement(overflow):
    async def scenario():
        host = Host()
        running, release = asyncio.Event(), asyncio.Event()
        history, executed, fitted = [], [], []
        original_session = host.session
        async def session(consume, options):
            fitted.append(list(history))
            return await original_session(consume, options)
        async def batch(calls):
            running.set()
            await release.wait()
            executed.extend(call['id'] for call in calls)
            rows = [{'role': 'tool', 'tool_call_id': call['id'], 'content': 'Completed ' + call['id']} for call in calls]
            history.extend(rows)
            host.due = overflow
            return Batch(rows)
        host.session, host.tool_batch = session, batch
        task, first = await started(host)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.renewal_ready = False
            first.emit('tool_call', call_id='one', name='effect', arguments='{}')
            await running.wait()
            first.emit('tool_call', call_id='two', name='effect', arguments='{}')
            await asyncio.sleep(.03)
            release.set()
            await until(lambda: len(host.sessions) > 1 or len(executed) == 2)
            if overflow:
                await until(lambda: len(host.sessions) == 2)
                assert executed == ['one', 'two']
                assert fitted[1] == history
                assert [row['tool_call_id'] for row in fitted[1]] == ['one', 'two']
                assert not any(isinstance(x, ToolOutputs) for x in first.sent)
                second = host.sessions[1]
                second.emit('ready', input_rate=24000, output_rate=24000)
                await until(lambda: bool(second.sent))
                assert len(second.sent) == 1 and isinstance(second.sent[0], ContextUpdate)
            else:
                await until(lambda: len([x for x in first.sent if isinstance(x, ToolOutputs)]) == 2)
                assert len(host.sessions) == 1
            assert executed == ['one', 'two']
        finally:
            release.set()
            host.hub.ended_by_owner = True
            await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_queued_end_call_preserves_goodbye_while_tool_refit_is_pending():
    async def scenario():
        host = Host()
        running, release = asyncio.Event(), asyncio.Event()
        executed = []
        async def batch(calls):
            running.set()
            await release.wait()
            executed.extend(call['id'] for call in calls)
            if calls[0]['id'] == 'end':
                await asyncio.sleep(.03)  # the real tool lane checkpoints before the effect
                host.hub.end_by_ouroboros()
            host.due = True
            return Batch([{'role': 'tool', 'tool_call_id': call['id'], 'content': 'Done'} for call in calls],
                         silent=['end'] if calls[0]['id'] == 'end' else [])
        host.tool_batch = batch
        task, first = await started(host)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.emit('audio', item_id='goodbye', rate=24000, pcm=b'\0' * 48000)
            first.emit('tool_call', call_id='one', name='effect', arguments='{}')
            await running.wait()
            first.emit('tool_call', call_id='end', name='end_call', arguments='{}')
            await asyncio.sleep(.03)
            release.set()
            await until(lambda: host.hub.ended_by_ouroboros)
            await asyncio.sleep(.05)
            assert not task.done(), 'withheld results must not cut off the queued goodbye'
            assert not any(isinstance(x, ToolOutputs) for x in first.sent)
            host.hub.events.append(cp.PeerControl('owner', cp.PEER_OWNER,
                {'type': 'played', 'item_id': 'goodbye', 'played_ms': 1000}))
            await asyncio.wait_for(task, 2)
            assert len(host.sessions) == 1 and executed == ['one', 'end']
            assert (cp.STATE_CLOSED, 'ouroboros_bye') in host.hub.states
        finally:
            release.set()
            host.hub.ended_by_owner = True
            if not task.done(): await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


@pytest.mark.parametrize('offline_tools', [False, True])
def test_an_engine_that_cannot_change_tools_live_takes_them_at_the_tool_boundary(offline_tools):
    """28.10: Gemini cannot change a live session's declarations. A Gemini call that enabled desktop tools
    kept calling enable_tools for tools it never received, and the call ended 30 s later. The driver renews
    at that tool boundary and answers from the fitted history; an engine that updates in place still gets
    the result on the open connection."""
    async def scenario():
        host = Host()
        async def batch(calls):
            row = {'role': 'tool', 'tool_call_id': 'load', 'content': 'Tools are registered.'}
            host.schemas.append({'type': 'function', 'function': {'name': 'newly_enabled', 'description': 'New schema'}})
            return Batch([row])
        host.tool_batch = batch
        task, first = await started(host)
        if offline_tools:
            first.tools_need_renewal = lambda tools: any(t['function']['name'] == 'newly_enabled' for t in tools)
        try:
            first.emit('ready', input_rate=24000, output_rate=24000)
            first.emit('response_started', response_id='tool-turn')
            first.renewal_ready = False
            first.emit('tool_call', call_id='load', name='enable_tools', arguments='{}')
            await until(lambda: len(host.sessions) > 1 or any(isinstance(x, ToolOutputs) for x in first.sent))
            if offline_tools:
                assert not any(isinstance(x, ToolOutputs) for x in first.sent)
                second = host.sessions[1]
                second.emit('ready', input_rate=24000, output_rate=24000)
                await until(lambda: any(isinstance(x, ContextUpdate) for x in second.sent))
                assert 'requested tool batch returned' in str(next(x for x in second.sent if isinstance(x, ContextUpdate)).messages)
            else:
                assert len(host.sessions) == 1
                assert next(x for x in first.sent if isinstance(x, ToolOutputs)).tools == tuple(host.schemas)
        finally:
            host.hub.ended_by_owner = True
            await asyncio.wait_for(task, 2)
    asyncio.run(scenario())
