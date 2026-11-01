"""R31/R32: fresh-fit fallback offers continuation regardless of old-socket speech."""
import asyncio
import json
import threading

import pytest

from ouroboros import call_protocol as cp
from ouroboros.contracts.live_conversation import Audio, ToolOutputs
from tests._live_stack import stack
from tests.test_live_conversation import host_world as host_world


@pytest.mark.parametrize('thinking', [False, True])
@pytest.mark.parametrize('speech_at', [
    'before-result', 'before-receive-returns', 'after-result', 'after-same-response', 'across-result'])
def test_requested_pixels_keep_fallback_continuation(
        host_world, monkeypatch, record_property, thinking, speech_at):
    entered, release = threading.Event(), threading.Event()
    before_result = speech_at.startswith('before-')

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini', thinking=thinking) as live:
            host, call, old = live.host, live.call, live.wire
            old.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': 'before-tool'}})
            await live.wait(lambda: host.state.get('resume_handle') == 'before-tool')
            retain = host.retain_frame

            async def delayed(data, mime, *, source=''):
                entered.set()
                while not release.is_set():
                    await asyncio.sleep(.01)
                return await retain(data, mime, source=source)

            monkeypatch.setattr(host, 'retain_frame', delayed)
            native, order, seen = call.session, [], {}
            write = native.ws.send
            written = asyncio.Event()
            convert = native.convert

            async def delayed_conversion(event):
                events = await convert(event)
                if (speech_at == 'before-receive-returns' and native.output_caption
                        and native.output_caption[1] == 'I am checking.'
                        and any(e.kind == 'response_done' for e in events)):
                    # Conversion can yield while flushing held context. Exercise
                    # a result written before this pre-write completion returns.
                    await written.wait()
                return events

            monkeypatch.setattr(native, 'convert', delayed_conversion)

            async def timed_write(raw):
                payload = json.loads(raw)
                is_image = 'video' in payload.get('realtimeInput', {})
                is_result = 'toolResponse' in payload
                if is_result:
                    seen['response_at_write'] = native.response
                await write(raw)
                if is_image:
                    order.append('pixels-written')
                if is_result:
                    order.append('function-response-written')
                    written.set()
                if is_image and speech_at == 'across-result':
                    old.say('The view is arriving.', complete=False)
                    old.put({'serverContent': {'outputTranscription': {'text': '', 'finished': True}}})
                    await live.wait(lambda: native.output_caption and native.output_caption[1] == 'The view is arriving.')
                    assert not native.response_complete
                    order.append('speech-started')
                    return
                if ((is_image and before_result) or (is_result and not before_result)):
                    if speech_at == 'across-result':
                        words = 'The view is arriving.'
                        old.complete()  # only completion is after the write, under the same response id
                    else:
                        words = 'I am checking.' if before_result else 'The requested view is here.'
                        old.say(words)
                    await live.wait(lambda: native.output_caption and native.output_caption[1] == words
                                    and native.response_complete and not native.turn_busy
                                    and (speech_at == 'before-receive-returns' or not call.response_open))
                    order.append('speech-completed')
                    seen['answer_response'] = native.response

            monkeypatch.setattr(native.ws, 'send', timed_write)
            call.attention.frame(cp.PeerVideo('screen', cp.PEER_OWNER, 'image/jpeg', b'exact requested frame'))
            complete = old.complete
            if speech_at != 'before-receive-returns':
                old.complete = lambda: None  # keep the issuing tool turn open
            old.tool('unanswered-look', 'attend', {'see': 'asked', 'look': True, 'revisit_after_s': None})
            old.complete = complete
            # Let the old turn settle, without inventing a new spoken answer or
            # a new usage-bearing response after the function response arrives.
            old.answer = lambda: old.put({'serverContent': {'turnComplete': True,
                **({'interactionStatus': 'IDLE'} if thinking else {})}})
            if speech_at in {'after-same-response', 'across-result'}:
                old.answer = lambda: None  # the answer completes the issuing response id
            try:
                await live.wait(entered.is_set)
                old.put({'goAway': {'timeLeft': '30s'}})
                await live.wait(lambda: call.renewing and call.resume)
                release.set()
                await live.wait(lambda: len(live.provider.connections) == 2 and live.wire.sent)
                await live.wait(lambda: not call.pending_offers)
                await live.quiet()
                pointer = any('The requested tool batch returned' in json.dumps(r)
                              for r in live.wire.sent if 'clientContent' in r)
                observed = dict(thinking=thinking, speech_at=speech_at, order=order, **seen,
                    setup=live.wire.sent[0]['setup']['sessionResumption'],
                    old_output='unanswered-look' in old.outputs,
                    successor_output='unanswered-look' in live.wire.outputs,
                    successor_pointer=pointer, successor_spoken=live.wire.spoken,
                    effects=sum(r.get('call_id') == 'unanswered-look' for r in host.trace['tool_calls']))
                record_property('answer_order', json.dumps(observed))
                assert observed['setup'] == {} and observed['old_output']
                assert observed['effects'] == 1 and not observed['successor_output']
                assert pointer, observed
                assert len(observed['successor_spoken']) == 1, observed
                assert sum('toolResponse' in row for row in old.sent) == 1
                if speech_at in {'after-same-response', 'across-result'}:
                    assert seen['answer_response'] == seen['response_at_write']
            finally:
                release.set()
        assert live.result[1]['execution_status'] == 'ok', live.result

    asyncio.run(scenario())


@pytest.mark.parametrize('thinking', [False, True])
@pytest.mark.parametrize('speech_at', ['before-result', 'after-result', 'after-same-response'])
def test_queued_result_keeps_fallback_continuation(
        host_world, monkeypatch, record_property, thinking, speech_at):
    entered, release = threading.Event(), threading.Event()

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini', thinking=thinking) as live:
            host, call, old = live.host, live.call, live.wire
            # Exercise a later batch too, with this result queued behind audio.
            old.tool('earlier-catalog', 'list_available_tools', {})
            await live.wait(lambda: 'earlier-catalog' in old.outputs)
            await live.quiet()
            effects = []
            execute = host.tools.execute_result

            def counted(name, args):
                if name == 'list_available_tools':
                    effects.append(name)
                return execute(name, args)

            monkeypatch.setattr(host.tools, 'execute_result', counted)
            old.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': 'before-tool'}})
            await live.wait(lambda: host.state.get('resume_handle') == 'before-tool')
            tool_batch = host.tool_batch
            tool_loop = None

            async def delayed(calls):
                nonlocal tool_loop
                tool_loop = asyncio.get_running_loop()
                entered.set()
                while not release.is_set():
                    await asyncio.sleep(.01)
                return await tool_batch(calls)

            monkeypatch.setattr(host, 'tool_batch', delayed)
            native, order, seen = call.session, [], {}
            write = native.ws.send
            audio_entered = threading.Event()
            result_queued = threading.Event()
            hold_audio = False
            audio_task = None
            send = native.send

            async def queued_send(command):
                if isinstance(command, ToolOutputs):
                    result_queued.set()
                return await send(command)

            monkeypatch.setattr(native, 'send', queued_send)

            async def speech():
                words = 'Let me check what is available.' if speech_at == 'before-result' else 'The tool catalog is here.'
                seen['result_received_before_speech'] = 'queued-catalog' in old.outputs
                seen['result_written_before_speech'] = 'function-response-written' in order
                old.say(words)
                await live.wait(lambda: native.output_caption and native.output_caption[1] == words
                                and native.response_complete and not call.response_open and not native.turn_busy)
                order.append('speech-completed')
                seen['answer_response'] = native.response

            async def timed_write(raw):
                nonlocal hold_audio
                payload = json.loads(raw)
                is_audio = 'audio' in payload.get('realtimeInput', {})
                is_result = 'toolResponse' in payload
                if is_result:
                    seen['response_at_write'] = native.response
                await write(raw)
                if is_audio and hold_audio:
                    hold_audio = False
                    audio_entered.set()
                    await live.wait(result_queued.is_set)
                    await speech()  # the result is queued behind this writer lock
                if is_result:
                    order.append('function-response-written')
                    if speech_at != 'before-result':
                        await speech()

            monkeypatch.setattr(native.ws, 'send', timed_write)
            complete = old.complete
            old.complete = lambda: None
            old.tool('queued-catalog', 'list_available_tools', {})
            old.complete = complete
            old.answer = lambda: old.put({'serverContent': {'turnComplete': True,
                **({'interactionStatus': 'IDLE'} if thinking else {})}})
            if speech_at == 'after-same-response':
                old.answer = lambda: None
            try:
                await live.wait(entered.is_set)
                old.put({'goAway': {'timeLeft': '30s'}})
                await live.wait(lambda: call.renewing and call.resume)
                if speech_at == 'before-result':
                    hold_audio = True
                    audio_task = asyncio.run_coroutine_threadsafe(
                        native.send(Audio(b'\0\0' * 320, 16000)), tool_loop)
                    await live.wait(audio_entered.is_set)
                release.set()
                await live.wait(lambda: len(live.provider.connections) == 2 and live.wire.sent)
                await live.wait(lambda: not call.pending_offers)
                await live.quiet()
                pointer = any('The requested tool batch returned' in json.dumps(r)
                              for r in live.wire.sent if 'clientContent' in r)
                observed = dict(thinking=thinking, speech_at=speech_at, order=order, **seen,
                    setup=live.wire.sent[0]['setup']['sessionResumption'],
                    old_output='queued-catalog' in old.outputs,
                    successor_output='queued-catalog' in live.wire.outputs,
                    successor_pointer=pointer, successor_spoken=live.wire.spoken,
                    effects=len(effects))
                record_property('queued_answer_order', json.dumps(observed))
                assert observed['setup'] == {} and observed['old_output']
                assert observed['effects'] == 1 and not observed['successor_output']
                assert pointer, observed
                assert len(observed['successor_spoken']) == 1, observed
                assert sum(any(response['id'] == 'queued-catalog' for response in
                              row.get('toolResponse', {}).get('functionResponses', [])) for row in old.sent) == 1
                assert seen['result_written_before_speech'] is (speech_at != 'before-result')
                if speech_at == 'after-same-response':
                    assert seen['answer_response'] == seen['response_at_write']
            finally:
                release.set()
                if audio_task is not None:
                    await live.wait(audio_task.done)
                    audio_task.result()
        assert live.result[1]['execution_status'] == 'ok', live.result

    asyncio.run(scenario())


@pytest.mark.parametrize('thinking', [False, True])
@pytest.mark.parametrize('completion_before_write', [True, False])
def test_native_read_before_conversion_keeps_fallback_continuation(
        host_world, monkeypatch, record_property, thinking, completion_before_write):
    entered, release = threading.Event(), threading.Event()

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini', thinking=thinking) as live:
            host, call, old = live.host, live.call, live.wire
            old.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': 'before-tool'}})
            await live.wait(lambda: host.state.get('resume_handle') == 'before-tool')
            retain = host.retain_frame

            async def delayed(data, mime, *, source=''):
                entered.set()
                while not release.is_set():
                    await asyncio.sleep(.01)
                return await retain(data, mime, source=source)

            monkeypatch.setattr(host, 'retain_frame', delayed)
            native = call.session
            observe, write, convert = native.observe, native.ws.send, native.convert
            completion_read = asyncio.Event()
            seen, order = {}, []
            awaiting_completion = False

            def observed(direction, raw):
                observe(direction, raw)
                if (direction == 'receive' and awaiting_completion
                        and json.loads(raw).get('serverContent', {}).get('turnComplete')):
                    # This is JsonSession.read's actual wire-observation boundary.
                    # Releasing the blocked media write here adds no reader delay;
                    # GeminiSession.receive's existing awaits choose the ordering.
                    seen.setdefault('result_started_at_native_read', 'function-response-write-start' in order)
                    order.append('completion-read')
                    completion_read.set()

            async def converted(event):
                if awaiting_completion and event.get('serverContent', {}).get('turnComplete'):
                    seen.setdefault('result_started_at_conversion', 'function-response-write-start' in order)
                    order.append('completion-converted')
                return await convert(event)

            async def timed_write(raw):
                nonlocal awaiting_completion
                payload = json.loads(raw)
                is_image = 'video' in payload.get('realtimeInput', {})
                is_result = 'toolResponse' in payload
                if is_result:
                    order.append('function-response-write-start')
                await write(raw)
                if ((is_image and completion_before_write) or (is_result and not completion_before_write)):
                    awaiting_completion = True
                    old.say('Let me check.' if completion_before_write else 'The requested view is here.')
                    await completion_read.wait()

            monkeypatch.setattr(native, 'observe', observed)
            monkeypatch.setattr(native, 'convert', converted)
            monkeypatch.setattr(native.ws, 'send', timed_write)
            call.attention.frame(cp.PeerVideo('screen', cp.PEER_OWNER, 'image/jpeg', b'exact requested frame'))
            complete = old.complete
            old.complete = lambda: None
            old.tool('read-boundary-look', 'attend', {'see': 'asked', 'look': True, 'revisit_after_s': None})
            old.complete = complete
            old.answer = lambda: None  # no answer after the response beyond the one explicitly scripted
            try:
                await live.wait(entered.is_set)
                old.put({'goAway': {'timeLeft': '30s'}})
                await live.wait(lambda: call.renewing and call.resume)
                release.set()
                await live.wait(lambda: len(live.provider.connections) == 2 and live.wire.sent)
                await live.wait(lambda: not call.pending_offers)
                await live.quiet()
                pointer = any('The requested tool batch returned' in json.dumps(row)
                              for row in live.wire.sent if 'clientContent' in row)
                observed_result = dict(thinking=thinking, completion_before_write=completion_before_write,
                    order=order, **seen, old_output='read-boundary-look' in old.outputs,
                    successor_output='read-boundary-look' in live.wire.outputs,
                    setup=live.wire.sent[0]['setup']['sessionResumption'],
                    pointer=pointer, spoken=live.wire.spoken,
                    effects=sum(row.get('call_id') == 'read-boundary-look' for row in host.trace['tool_calls']))
                record_property('read_boundary', json.dumps(observed_result))
                assert observed_result['setup'] == {} and observed_result['old_output']
                assert observed_result['effects'] == 1 and not observed_result['successor_output']
                assert seen['result_started_at_native_read'] is not completion_before_write, observed_result
                assert seen['result_started_at_conversion'], observed_result
                assert pointer, observed_result
                assert len(observed_result['spoken']) == 1, observed_result
                assert sum('toolResponse' in row for row in old.sent) == 1
            finally:
                release.set()
        assert live.result[1]['execution_status'] == 'ok', live.result

    asyncio.run(scenario())
