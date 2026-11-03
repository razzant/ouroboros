"""Native result-write ordering and publication of the corresponding resume handle."""
import asyncio
import json
import threading

import pytest

from ouroboros import call_protocol as cp
from ouroboros.contracts.live_conversation import Audio
from tests._live_stack import stack
from tests.test_live_conversation import host_world as host_world


@pytest.mark.parametrize('answer', [False, True])
@pytest.mark.parametrize('thinking', [False, True])
@pytest.mark.parametrize('handle_timing', ['before-result', 'queued-result', 'after-result'])
def test_handle_during_requested_pixels_cannot_cover_later_result(
        host_world, monkeypatch, record_property, thinking, handle_timing, answer):
    entered, release = threading.Event(), threading.Event()

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini', thinking=thinking) as live:
            host, call, old = live.host, live.call, live.wire
            old.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': 'before-look'}})
            await live.wait(lambda: host.state.get('resume_handle') == 'before-look')
            retain = host.retain_frame

            async def delayed(data, mime, *, source=''):
                entered.set()
                while not release.is_set():
                    await asyncio.sleep(.01)
                return await retain(data, mime, source=source)

            monkeypatch.setattr(host, 'retain_frame', delayed)
            native = call.session
            write = native.ws.send
            order, queued = [], []
            revisions = {}

            async def timed_write(raw):
                # Image and function response are separate native writes. Hold the
                # image write's completion until an actual incoming handle event
                # has been processed. This models the writer yielding under load.
                payload = json.loads(raw)
                if 'video' in payload.get('realtimeInput', {}):
                    revisions['before_pixels'] = call.resume_revision
                await write(raw)
                is_image = 'video' in payload.get('realtimeInput', {})
                is_result = 'toolResponse' in payload
                if is_image:
                    order.append('pixels-written')
                    if handle_timing == 'queued-result':
                        # Audio takes the shared writer lock before the function
                        # response, then yields while its handle arrives.
                        queued.append(asyncio.create_task(native.send(Audio(b'\0\0' * 320, 16000))))
                        await asyncio.sleep(0)
                if is_result:
                    order.append('function-response-written')
                if ((is_image and handle_timing == 'before-result')
                        or (is_result and handle_timing == 'after-result')
                        or ('audio' in payload.get('realtimeInput', {}) and queued
                            and 'handle-received' not in order)):
                    handle = handle_timing
                    old.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': handle}})
                    while host.state.get('resume_handle') != handle:
                        await asyncio.sleep(.01)
                    revisions['new_handle'] = call.resume_revision
                    order.append('handle-received')

            monkeypatch.setattr(native.ws, 'send', timed_write)
            call.attention.frame(cp.PeerVideo('screen', cp.PEER_OWNER, 'image/jpeg', b'exact requested frame'))
            complete = old.complete
            if not answer:
                old.complete = lambda: None  # tool turn stays open until its result
                old.answer = complete  # complete that tool turn, without an answer
            old.tool('ordered-look', 'attend', {'see': 'asked', 'look': True, 'revisit_after_s': None})
            old.complete = complete
            try:
                await live.wait(entered.is_set)
                old.put({'goAway': {'timeLeft': '30s'}})
                await live.wait(lambda: call.renewing and call.resume)
                release.set()
                await live.wait(lambda: 'ordered-look' in old.outputs)
                await live.wait(lambda: len(live.provider.connections) == 2 and live.wire.sent)
                setup = live.wire.sent[0]['setup']['sessionResumption']
                observed = dict(thinking=thinking, handle_timing=handle_timing, order=order,
                                successor=setup, revisions=revisions,
                                old_output='ordered-look' in old.outputs,
                                successor_output='ordered-look' in live.wire.outputs,
                                effects=sum(r.get('call_id') == 'ordered-look' for r in host.trace['tool_calls']))
                record_property('write_order', json.dumps(observed))
                if handle_timing != 'after-result':
                    assert order == ['pixels-written', 'handle-received', 'function-response-written']
                    assert setup == {}, 'handle predating function response must use a fresh fit'
                else:
                    assert order == ['pixels-written', 'function-response-written', 'handle-received']
                    assert setup == {'handle': 'after-result'}
                    assert not any('The requested tool batch returned' in json.dumps(r)
                                   for r in live.wire.sent if 'clientContent' in r)
                await live.wait(lambda: not call.pending_offers)
                await live.quiet()
                assert observed['effects'] == 1 and not live.wire.outputs
                pointer = any('The requested tool batch returned' in json.dumps(r)
                              for r in live.wire.sent if 'clientContent' in r)
                assert pointer is (handle_timing != 'after-result')
                assert len(live.wire.spoken) == int(handle_timing != 'after-result')
                assert sum('toolResponse' in row for row in old.sent) == 1
            finally:
                release.set()
                await live.wait(lambda: all(task.done() for task in queued))
                for task in queued:
                    task.result()
        assert live.result[1]['execution_status'] == 'ok', live.result

    asyncio.run(scenario())


@pytest.mark.parametrize('handle_after', [False, True])
def test_legacy_session_keeps_send_boundary_order(handle_after):
    from ouroboros.contracts.live_conversation import ToolOutputs
    from skills.live_conversation.driver import _Call
    from tests.test_live_conversation_driver import Host, Session

    async def scenario():
        host = Host()
        call = _Call(host)
        call.session = Session()  # no additive native-write fact
        call.connected.set()
        host.state['resume_handle'] = 'same-value'
        call.resume_revision = 1
        await call.send(ToolOutputs(({'role': 'tool', 'tool_call_id': 'c', 'content': 'Done'},), ()))
        if handle_after:
            call.resume_revision += 1
        call.request_renewal(resuming=True)
        await call.peers()
        assert call.resume is handle_after
        assert bool(call.pending_offers) is not handle_after
    asyncio.run(scenario())


@pytest.mark.parametrize('thinking', [False, True])
@pytest.mark.parametrize('release_at', ['read', 'conversion'])
def test_resume_uses_the_handle_whose_freshness_was_observed(host_world, monkeypatch, record_property,
                                                          thinking, release_at):
    held = threading.Event()

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini', thinking=thinking) as live:
            host, call, old = live.host, live.call, live.wire
            initial_spoken = list(old.spoken)
            old.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': 'before-tool'}})
            await live.wait(lambda: host.state.get('resume_handle') == 'before-tool')
            native = call.session
            old.answer = old.complete  # ordinary result is written; no spoken continuation yet
            old.tool('publication-result', 'list_available_tools', {})
            await live.wait(lambda: 'publication-result' in old.outputs and not call.tools_running)
            await live.quiet()
            assert call.resume_tool_output and call.resume_tool_output[1]
            assert not native.resume_after_tool_response

            check, observe, convert = call.check_preparation, native.observe, native.convert
            gate = None
            trail = []

            async def checked():
                nonlocal gate
                await check()
                if not held.is_set():
                    # Schedule the existing peer worker at the read boundary.
                    # The native reader and conversion have no inserted delay.
                    gate = asyncio.Event()
                    held.set()
                    await gate.wait()
                    trail.append(dict(at='peer-wakes', covered=native.resume_after_tool_response,
                                      host_handle=host.state.get('resume_handle'),
                                      native_handle=native.resume_handle))

            def observed(direction, raw):
                observe(direction, raw)
                if direction == 'receive' and json.loads(raw).get('sessionResumptionUpdate', {}).get('newHandle') == 'after-tool':
                    trail.append(dict(at='handle-read'))
                    if release_at == 'read':
                        gate.set()

            async def converted(event):
                result = await convert(event)
                if event.get('sessionResumptionUpdate', {}).get('newHandle') == 'after-tool':
                    trail.append(dict(at='handle-converted'))
                    if release_at == 'conversion':
                        gate.set()
                return result

            monkeypatch.setattr(call, 'check_preparation', checked)
            monkeypatch.setattr(native, 'observe', observed)
            monkeypatch.setattr(native, 'convert', converted)
            try:
                await live.wait(held.is_set)
                old.put({'goAway': {'timeLeft': '30s'}})
                await live.wait(lambda: call.renewing and call.resume)
                old.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': 'after-tool'}})
                await live.wait(lambda: len(live.provider.connections) == 2 and live.wire.sent)
                await live.wait(lambda: not call.pending_offers)
                await live.quiet()
                result = dict(thinking=thinking, release_at=release_at, trail=trail,
                    setup=live.wire.sent[0]['setup']['sessionResumption'],
                    pointers=sum('The requested tool batch returned' in json.dumps(row)
                                for row in live.wire.sent if 'clientContent' in row),
                    spoken=live.wire.spoken, old_responses=sum('toolResponse' in row for row in old.sent),
                    new_responses=sum('toolResponse' in row for row in live.wire.sent),
                    effects=sum(row.get('tool_call_id') == 'publication-result' for row in host.trace['tool_calls']))
                record_property('handle_publication', json.dumps(result))
                assert result['effects'] == 1 and result['old_responses'] == 1 and not result['new_responses'], result
                if release_at == 'read':
                    assert result['setup'] == {}, result
                    assert result['pointers'] == 1 and len(result['spoken']) == 1, result
                else:
                    assert result['setup'] == {'handle': 'after-tool'}, result
                    assert result['pointers'] == 0 and not result['spoken'], result
                assert old.spoken == initial_spoken, result
            finally:
                if gate is not None:
                    native_loop = gate._loop
                    if native_loop is not None:
                        native_loop.call_soon_threadsafe(gate.set)
        assert live.result[1]['execution_status'] == 'ok', live.result

    asyncio.run(scenario())


@pytest.mark.parametrize('thinking', [False, True])
@pytest.mark.parametrize('update', ['new-handle', 'cleared'])
def test_update_read_after_write_unconverted_at_quiet_poll(
        host_world, monkeypatch, record_property, thinking, update):
    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini', thinking=thinking) as live:
            host, call, old = live.host, live.call, live.wire
            initial_spoken = list(old.spoken)
            old.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': 'before-tool'}})
            await live.wait(lambda: host.state.get('resume_handle') == 'before-tool')
            native = call.session
            complete = old.complete
            old.complete = lambda: None
            old.answer = lambda: None  # the result is written; no answer has started
            old.tool('after-write', 'list_available_tools', {})
            old.complete = complete
            await live.wait(lambda: 'after-write' in old.outputs and call.response_open and not call.tools_running)

            # Park the 10 ms poll at its next pass and let the native read of the update release it.
            # The reader is not delayed; this only lands the poll's timer inside the existing interval.
            gate, parked = asyncio.Event(), threading.Event()
            observe, convert, check = native.observe, native.convert, call.check_preparation
            armed = False
            trail = []

            async def parked_check():
                if armed and not gate.is_set():
                    parked.set()
                    await gate.wait()
                    trail.append(dict(poll_resumes=True, revision=native.resume_revision,
                                      handle=host.state.get('resume_handle')))
                return await check()

            def observed(direction, raw):
                observe(direction, raw)
                if direction == 'receive' and armed and 'sessionResumptionUpdate' in json.loads(raw):
                    trail.append('update-read')
                    gate.set()

            async def converted(event):
                if armed and 'sessionResumptionUpdate' in event:
                    trail.append('update-converted')
                return await convert(event)

            monkeypatch.setattr(call, 'check_preparation', parked_check)
            monkeypatch.setattr(native, 'observe', observed)
            monkeypatch.setattr(native, 'convert', converted)
            armed = True
            await live.wait(parked.is_set)
            old.put({'goAway': {'timeLeft': '30s'}})
            await live.wait(lambda: call.renewing and call.resume)
            complete()  # the tool turn finishes with no speech: quiet, unobserved by the parked poll
            await live.wait(lambda: not call.response_open and call.quiet())
            frame = ({'resumable': True, 'newHandle': 'after-write'} if update == 'new-handle' else {'resumable': False})
            old.put({'sessionResumptionUpdate': frame})
            await live.wait(lambda: len(live.provider.connections) == 2 and live.wire.sent)
            await live.wait(lambda: not call.pending_offers)
            await live.quiet()
            result = dict(probe='F', thinking=thinking, update=update, trail=trail,
                setup=live.wire.sent[0]['setup']['sessionResumption'],
                pointers=sum('The requested tool batch returned' in json.dumps(row)
                             for row in live.wire.sent if 'clientContent' in row),
                spoken=live.wire.spoken, old_responses=sum('toolResponse' in r for r in old.sent),
                new_responses=sum('toolResponse' in r for r in live.wire.sent),
                effects=sum(row.get('tool_call_id') == 'after-write' for row in host.trace['tool_calls']))
            record_property('handle_publication', json.dumps(result))
            assert result['effects'] == 1 and result['old_responses'] == 1 and not result['new_responses'], result
            assert result['setup'] == {}, result
            assert result['pointers'] == 1 and len(result['spoken']) == 1, result
            assert old.spoken == initial_spoken, result
        assert live.result[1]['execution_status'] == 'ok', live.result

    asyncio.run(scenario())
