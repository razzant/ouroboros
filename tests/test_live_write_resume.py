"""A send notices transport loss before the reader: native resumption still owns it."""
import asyncio
import json

import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from ouroboros import call_protocol as cp
from skills.live_conversation.driver import run
from tests.test_gemini_live_boundaries import gemini as gemini, make_session
from tests.test_live_conversation_driver import Host, until


@pytest.mark.parametrize('resumable', [False, True])
def test_failed_audio_write_defers_to_the_native_readers_resumption_outcome(gemini, resumable):
    async def scenario():
        host = Host()
        host.state['greeted'] = True
        native, wire = make_session(gemini)
        original_send, original_recv = wire.send, wire.recv
        closed = ConnectionClosedError(Close(1012, 'synthetic restart'), Close(1012, 'synthetic restart'), True)
        async def send(raw):
            if 'audio' in json.loads(raw).get('realtimeInput', {}):
                wire.put(closed)
                raise closed
            await original_send(raw)
        async def recv():
            try: return await original_recv()
            except ConnectionClosedError:
                await asyncio.sleep(.03)  # the writer demonstrably fails first
                raise
        wire.send, wire.recv = send, recv
        async def connection(consume, options):
            current = native if not host.sessions else make_session(gemini, options)[0]
            host.sessions.append(current)
            await current.open()
            try: return await consume(current)
            finally: await current.close()
        host.session = connection
        running = asyncio.create_task(run(host))
        try:
            await until(lambda: host.hub.states)
            if resumable:
                wire.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': 'synthetic-resume'}})
                await until(lambda: host.state.get('resume_handle'))
            host.hub.events.append(cp.PeerAudio('owner', cp.PEER_OWNER, b'\0\0' * 320))
            if resumable:
                await until(lambda: len(host.sessions) == 2 or running.done())
                if running.done(): await running
                assert len(host.sessions) == 2
                assert host.sessions[1].options.resume_handle == 'synthetic-resume'
            else:
                await until(lambda: len(host.sessions) == 2)
                assert not host.sessions[1].options.resume_handle, 'no handle means a fresh fit, never native resumption'
        finally:
            host.hub.ended_by_owner = True
            if not running.done(): await asyncio.wait_for(running, 2)
    asyncio.run(scenario())


@pytest.mark.parametrize('resumable', [False, True])
def test_failed_tool_result_write_defers_to_native_resumption(gemini, resumable):
    """R19-2: a tool result that meets the closed socket first is delivered on the resumed connection."""
    from tests.test_live_conversation_driver import Batch

    async def scenario():
        host = Host()
        host.state['greeted'] = True
        native, wire = make_session(gemini)
        wires = [wire]
        original_send, original_recv = wire.send, wire.recv
        closed = ConnectionClosedError(Close(1012, 'synthetic restart'), Close(1012, 'synthetic restart'), True)
        async def send(raw):
            if 'toolResponse' in json.loads(raw):
                wire.put(closed)
                raise closed
            await original_send(raw)
        async def recv():
            try: return await original_recv()
            except ConnectionClosedError:
                await asyncio.sleep(.03)  # the writer demonstrably fails first
                raise
        wire.send, wire.recv = send, recv
        async def batch(calls):
            return Batch([{'role': 'tool', 'tool_call_id': c['id'], 'content': 'Synthetic completed effect'} for c in calls])
        host.tool_batch = batch
        async def connection(consume, options):
            if host.sessions:
                current, replacement = make_session(gemini, options)
                wires.append(replacement)
            else:
                current = native
            host.sessions.append(current)
            await current.open()
            try: return await consume(current)
            finally: await current.close()
        host.session = connection
        running = asyncio.create_task(run(host))
        try:
            await until(lambda: host.hub.states)
            if resumable:
                wire.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': 'synthetic-resume'}})
                await until(lambda: host.state.get('resume_handle'))
            wire.put({'toolCall': {'functionCalls': [{'id': 'c', 'name': 'read_file', 'args': {}}]}})
            if resumable:
                await until(lambda: len(wires) == 2 and any('toolResponse' in e for e in wires[1].sent) or running.done())
                if running.done(): await running
                assert host.sessions[1].options.resume_handle == 'synthetic-resume'
                [delivered] = [e for e in wires[1].sent if 'toolResponse' in e]
                assert [r['id'] for r in delivered['toolResponse']['functionResponses']] == ['c']
            else:
                await until(lambda: len(host.sessions) == 2)
                assert not host.sessions[1].options.resume_handle, 'no handle means a fresh fit, never native resumption'
        finally:
            host.hub.ended_by_owner = True
            if not running.done(): await asyncio.wait_for(running, 2)
    asyncio.run(scenario())
