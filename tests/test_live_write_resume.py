"""A send notices transport loss before the reader: native resumption still owns it."""
import asyncio
import json
from dataclasses import replace

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
                if resumable:  # issued after the call arrived, the handle resumes a conversation holding it
                    wire.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': 'synthetic-resume'}})
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


def test_failed_tool_write_keeps_the_resume_catalog_and_later_tool_loads(gemini):
    from skills.live_conversation.driver import _Call
    from tests._live_wire import Batch, Wire, selected

    def schema(name):
        return {'type': 'function', 'function': {'name': name, 'parameters': {'type': 'object'}}}

    async def scenario():
        host = Host()
        host.state['greeted'] = True
        keep, removed, new = (schema(name) for name in ('read_file', 'old_low_tool', 'newly_loaded'))
        host.schemas[:] = [keep, removed]
        call = _Call(host)
        handle = 'resume-with-new-catalog'
        wires, effects = [], []

        async def batch(calls):
            ids = tuple(item['id'] for item in calls)
            effects.extend(ids)
            if 'enable' in ids:
                host.schemas.append(new)
            return Batch([{'role': 'tool', 'tool_call_id': ident, 'content': 'Completed once.'} for ident in ids],
                         silent=ids)

        async def connect(consume, options):
            wire = Wire()
            wires.append(wire)
            if len(wires) == 1:
                write = wire.send
                async def fail_tool(raw):
                    if 'toolResponse' in json.loads(raw):
                        restart = Close(1012, 'synthetic restart')
                        loss = ConnectionClosedError(restart, restart, True)
                        wire.put({'sessionResumptionUpdate': {'resumable': True, 'newHandle': handle}})
                        wire.put(loss)
                        raise loss
                    await write(raw)
                wire.send = fail_tool
            elif options.resume_handle:
                host.schemas[:] = [keep]  # the replacement fit's sole catalog change
            mode = 'low' if len(wires) == 1 else 'nano'
            session = gemini.GeminiSession(replace(selected(tools=host.schemas), mode=mode), options,
                                          connect=wire.connect, observe=lambda *_: None)
            host.sessions.append(session)
            await session.open()
            try:
                return await consume(session)
            finally:
                await session.close()

        host.tool_batch, host.session = batch, connect
        running = asyncio.create_task(call.run())
        def request(wire, ident):
            wire.put({'toolCall': {'functionCalls': [{'id': ident, 'name': 'read_file', 'args': {}}]}})

        try:
            await until(lambda: call.connected.is_set())
            request(wires[0], 'once')
            await until(lambda: len(wires) > 1 and any('toolResponse' in event for event in wires[1].sent) or running.done())
            if running.done():
                await running
            resumed = host.sessions[1]
            assert resumed.options.resume_handle == handle and resumed.selected.mode == 'nano'
            declarations, = wires[1].sent[0]['setup']['tools']
            assert [item['name'] for item in declarations['functionDeclarations']] == ['read_file']
            assert call.serving_catalog() == (keep,) and not resumed.tools_need_renewal((keep,))
            assert (resumed.selected.model, handle) not in gemini.RESUMPTIONS
            assert not call.deferred_tool_outputs and not call.renewing

            request(wires[1], 'enable')  # a genuine new declaration still requires a fresh fit
            await until(lambda: len(wires) == 3 and call.connected.is_set() or running.done())
            if running.done():
                await running
            assert not host.sessions[2].options.resume_handle
            assert json.loads(host.sessions[2].selected.tools_json) == [keep, new]
            assert effects == ['once', 'enable']
            assert [item['id'] for wire in wires for event in wire.sent
                    for item in event.get('toolResponse', {}).get('functionResponses', ())] == ['once']
        finally:
            host.hub.ended_by_owner = True
            if not running.done():
                await asyncio.wait_for(running, 2)

    asyncio.run(scenario())
