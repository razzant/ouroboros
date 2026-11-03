"""Completed tool delivery at a requested native resume, with real host measures."""
import asyncio
import json
from dataclasses import replace

import pytest

from ouroboros import call_protocol as cp
from ouroboros.contracts.live_conversation import LiveOptions, ToolOutputs
from ouroboros.live_context_view import serving_delivery_room
from skills.live_conversation.attention import attend_tool
from skills.live_conversation.driver import _Call
from tests.test_live_conversation import host_world as host_world


@pytest.mark.parametrize('vision,room,successor', [
    (True, 700, 'native'), (True, 4000, 'native'),
    (True, 700, 'fresh'), (False, 700, 'native'),
    (True, 700, 'none'), (True, 4000, 'none'), (False, 700, 'none'),
])
def test_native_attend_measures_the_complete_delivery(host_world, record_property, vision, room, successor):
    host = host_world.host
    host.model = replace(host.model, vision=vision)
    tool = attend_tool()
    host.call_tools[tool.name] = tool
    host.schemas.append({'type': 'function', 'function': {'name': tool.name,
        'description': tool.description, 'parameters': dict(tool.parameters)}})
    async def consume(session):
        call = _Call(host)
        call.session = session
        call.connected.set()
        deliveries = []
        async def send(command):
            if isinstance(command, ToolOutputs):
                deliveries.append((command, serving_delivery_room(host, command.messages, command.tools)))
        session.send = send
        worker = asyncio.create_task(call.tools())
        def request(cid):
            return {'id': cid, 'type': 'function', 'function': {
                'name': 'attend', 'arguments': '{"see":"asked","look":true,"revisit_after_s":null}'}}
        try:
            call.attention.frame(cp.PeerVideo('screen', cp.PEER_OWNER, 'image/jpeg', b'first exact frame'))
            await call.tool_queue.put(request('first-look'))
            await asyncio.wait_for(call.tool_queue.join(), 30)
            assert len(deliveries) == 1
            held = host._renewal_window.total_equivalent - host._renewal_window.reserve - room
            host.response_done('first-look-receipt', usage={'audio_input_tokens': held}, raw_usage={}, status='completed')
            call.response_open = False
            assert not host.renewal_due()
            call.attention.frame(cp.PeerVideo('screen', cp.PEER_OWNER, 'image/jpeg', b'second exact frame'))
            if successor != 'none':
                call.request_renewal(resuming=successor == 'native')
            await call.tool_queue.put(request('second-look'))
            await asyncio.wait_for(call.tool_queue.join(), 30)
            later = deliveries[1:]
            deferred = call.deferred_tool_outputs
            command = later[0][0] if later else deferred[0][0]
            free = later[0][1] if later else serving_delivery_room(host, command.messages, command.tools)
            assert not host.renewal_due(), 'the fresh projection collapses the prior same-source pixels'
            record_property('attend_boundary', json.dumps(dict(vision=vision, room=room, successor=successor,
                serving_free=free, fresh_due=host.renewal_due(), sent=len(later), deferred=len(deferred),
                refit=bool(host.state.get('refit')), native_resume=call.resume,
                second_effects=sum(row.get('call_id') == 'second-look' for row in host.trace['tool_calls']))))
            assert all(free >= 0 for _command, free in later), 'complete requested image exceeds serving room'
            assert sum(row.get('call_id') == 'second-look' for row in host.trace['tool_calls']) == 1
            if vision and room == 700:
                assert not later and deferred
                assert not call.resume and call.pending_offers
            elif successor != 'fresh':
                assert later and not deferred
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await call.close_preparation()
    asyncio.run(host.session(consume, LiveOptions()))


@pytest.mark.parametrize('engine,thinking,successor', [
    ('gemini', False, 'native'), ('gemini', True, 'native'),
    ('gemini', False, 'none'), ('gemini', True, 'none'), ('openai', False, 'none'),
])
@pytest.mark.parametrize('room', [700, 4000])
def test_native_wire_attend_during_goaway(host_world, monkeypatch, record_property, engine, thinking, successor, room):
    from tests._live_stack import stack
    import threading
    entered, release = threading.Event(), threading.Event()
    async def scenario():
        async with stack(host_world, monkeypatch, engine, thinking=thinking) as live:
            host, call, old = live.host, live.call, live.wire
            call.attention.frame(cp.PeerVideo('screen', cp.PEER_OWNER, 'image/jpeg', b'first exact frame'))
            old.tool('first-look', 'attend', {'see': 'asked', 'look': True, 'revisit_after_s': None})
            await live.wait(lambda: 'first-look' in old.outputs)
            await live.quiet()
            # A controlled near-window receipt, keeping the actual native prefix and serving baselines.
            held = host._renewal_window.total_equivalent - host._renewal_window.reserve - room
            for view in (host, host._connection_view):
                view._held_input, view._growth = held, 0
                view._history_at_receipt = view._history_estimate()
                view._serving_history_at_receipt = view._serving_history_estimate()
            old.prompt, old.audio_tokens = held, held  # next tool boundary reports the same held size
            if successor == 'native':
                old.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': 'before-look'}})
                await live.wait(lambda: host.state.get('resume_handle') == 'before-look')
            retain = host.retain_frame
            async def delayed(data, mime, *, source=''):
                entered.set()
                while not release.is_set():
                    await asyncio.sleep(.01)
                return await retain(data, mime, source=source)
            monkeypatch.setattr(host, 'retain_frame', delayed)
            native, sent = call.session, []
            send = native.send
            async def measured(command):
                if isinstance(command, ToolOutputs):
                    sent.append({'room': serving_delivery_room(host, command.messages, command.tools),
                        'fresh_due': host.renewal_due(), 'native_resume': call.resume,
                        'second_result': any(r.get('tool_call_id') == 'second-look' for r in command.messages)})
                return await send(command)
            monkeypatch.setattr(native, 'send', measured)
            initial_video = sum('video' in event.get('realtimeInput', {}) for event in old.sent)
            call.attention.frame(cp.PeerVideo('screen', cp.PEER_OWNER, 'image/jpeg', b'second exact frame'))
            old.tool('second-look', 'attend', {'see': 'asked', 'look': True, 'revisit_after_s': None})
            try:
                await live.wait(entered.is_set)
                if successor == 'native':
                    old.put({'sessionResumptionUpdate': {'resumable': False}})
                    old.put({'goAway': {'timeLeft': '30s'}})
                    await live.wait(lambda: call.renewing and call.resume, timeout=15)
                else:
                    assert not call.renewing
                release.set()
                await live.wait(lambda: 'second-look' in old.outputs or len(live.provider.connections) > 1)
                observed = [item for item in sent if item['second_result']]
                more_video = sum('video' in event.get('realtimeInput', {}) for event in old.sent) - initial_video
                record_property('native_attend_boundary', json.dumps(dict(thinking=thinking, room=room,
                    sends=observed, second_result_on_issuing_wire='second-look' in old.outputs,
                    new_video_on_issuing_wire=more_video, connections=len(live.provider.connections))))
                assert all(item['room'] >= 0 for item in observed), 'native wire received an over-capacity requested image'
                if room == 4000:
                    assert observed and 'second-look' in old.outputs
                    if engine == 'gemini':
                        assert more_video == 1
                else:
                    assert not observed and 'second-look' not in old.outputs and more_video == 0
                    await live.wait(lambda: any('The requested tool batch returned' in json.dumps(row)
                                               for row in live.wire.sent))
                    await live.quiet()
                    if engine == 'gemini':
                        assert live.wire.sent[0]['setup']['sessionResumption'] == {}
                        assert 'second-look' not in live.wire.outputs
                    assert 'second-look' in live.call.session.selected.messages_json
                    assert sum(row.get('call_id') == 'second-look' for row in host.trace['tool_calls']) == 1
            finally:
                release.set()
    asyncio.run(scenario())
