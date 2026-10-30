"""Proactive commands through all three native adapters; no provider connection."""
import asyncio
import json

import pytest

from ouroboros.contracts.live_conversation import LiveOptions, Respond, Video
from ouroboros.contracts.realtime import RealtimeSetupError, canonical_json
from tests._extension_loader_shared import _clear_loader_state  # noqa: F401
from tests.test_live_conversation import ENGINES, VoiceWire, packages as packages
from tests.test_live_provider_boundaries import selected
from tests.test_live_conversation_driver import Host, Batch, until


class SilentWire(VoiceWire):
    async def send(self, raw):
        event = json.loads(raw)
        if 'toolResponse' in event:
            self.sent.append(event)
            self.result_received.set()
            responses = event['toolResponse']['functionResponses']
            if all(r.get('scheduling') == 'SILENT' for r in responses):
                self.put({'serverContent': {'turnComplete': True}})
            else:
                self.answer()
        elif event.get('type') == 'conversation.item.delete':
            self.sent.append(event)
            self.put({'type':'conversation.item.deleted', 'item_id':event['item_id']})
        else:
            await super().send(raw)


@pytest.mark.parametrize('name,engine', ENGINES)
@pytest.mark.parametrize('look', [False, True])
def test_attend_silent_or_look_continues_through_driver_and_native_session(packages, tmp_path, name, engine, look):
    from dataclasses import replace
    from ouroboros.realtime_setup import loaded_provider
    from skills.live_conversation.driver import run, call_tools
    async def scenario():
        host, wires = Host(), []
        host.state['greeted'] = True
        host.model = replace(host.model, vision=engine != 'grok')
        feature_tools = {t.name: t for t in call_tools(tmp_path)}
        tool = feature_tools['attend']
        host.schemas = [{'type':'function', 'function': {'name':tool.name, 'description':tool.description,
                                                        'parameters':dict(tool.parameters)}}]
        factory = loaded_provider(f'{name}:{engine}')['session']
        async def connection(consume, options):
            wire = SilentWire(engine, None)
            native = factory(replace(selected(), tools_json=canonical_json(host.schemas)),
                             LiveOptions(), connect=wire.connect, observe=lambda *_: None)
            await native.open()
            wires.append(wire)
            host.sessions.append(native)
            try: return await consume(native)
            finally: await native.close()
        async def batch(calls):
            rows = []
            for call in calls:
                from ouroboros.tool_call_log import invocation_scope
                with invocation_scope({'tool_call_id': call['id']}):
                    text = await tool.handler(host, json.loads(call['function']['arguments']))
                rows.append({'role':'tool', 'tool_call_id':call['id'], 'content':text})
            return Batch(rows, silent=[c['id'] for c in calls])
        host.session, host.tool_batch = connection, batch
        async def retain(data, mime, *, source=""):
            path = tmp_path / "look.jpg"
            path.write_bytes(data)
            return str(path)
        host.retain_frame = retain
        task = asyncio.create_task(run(host))
        try:
            await until(lambda: wires or task.done())
            if task.done(): await task
            wire = wires[0]
            if look:
                from ouroboros import call_protocol as cp
                from tests._live_stack_page import jpeg
                pixels = jpeg('synthetic requested look')
                host.hub.events.append(cp.PeerVideo('source', cp.PEER_OWNER, 'image/jpeg', pixels))
                await until(lambda: host.state['attention'].frames)
            args = {'see':None, 'look':look}
            if engine == 'gemini':
                wire.put({'toolCall':{'functionCalls':[{'id':'attend-1','name':'attend','args':args}]}})
            else:
                wire.put({'type':'response.done', 'response': {'id':'decision', 'status':'completed',
                    'output':[{'type':'function_call', 'call_id':'attend-1','name':'attend','arguments':json.dumps(args)}]}})
            await until(lambda: wire.result_received.is_set() or task.done())
            if task.done(): await task
            if look:
                await until(lambda: bool(host.hub.audio))
            await asyncio.sleep(.05)
            assert bool(host.hub.audio) == look, 'silent consideration must not create another spoken response'
            assert len(wires) == 1
            if look:
                assert (tmp_path / 'look.jpg').read_bytes() == pixels
            if look and engine == 'gemini':
                frame_index = next(i for i,v in enumerate(wire.sent) if 'video' in v.get('realtimeInput', {}))
                result_index = next(i for i,v in enumerate(wire.sent) if 'toolResponse' in v)
                assert frame_index < result_index, 'look pixels must arrive before the result can continue the answer'
            if look and engine == 'openai':
                assert any(p.get('type') == 'input_image' for v in wire.sent for p in v.get('item',{}).get('content',[]))
            assert not any('idle_timeout_ms' in json.dumps(event) for event in wire.sent)
        finally:
            host.hub.ended_by_owner = True
            await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


def test_openai_replaces_only_the_requested_sources_image(packages):
    from ouroboros.realtime_setup import loaded_provider
    async def scenario():
        wire = SilentWire('openai', None)
        session = loaded_provider('openai_realtime:openai')['session'](
            selected(), LiveOptions(), connect=wire.connect, observe=lambda *_: None)
        try:
            await session.open(); await session.receive()
            await session.send(Video(b'a', 'image/jpeg', 'screen-peer'))
            await session.receive()
            await session.send(Video(b'b', 'image/jpeg', 'camera-peer'))
            await session.receive()
            first = session.source_images['screen-peer']
            await session.send(Video(b'c', 'image/jpeg', 'screen-peer'))
            deletes = [v for v in wire.sent if v.get('type') == 'conversation.item.delete']
            assert deletes == [{'type':'conversation.item.delete', 'item_id':first}]
            await session.receive(); await session.receive()
            assert not session.requested_deletions
            with pytest.raises(RealtimeSetupError, match='provider_truncation'):
                session.convert({'type':'conversation.item.deleted','item_id':session.source_images['camera-peer']})
            before = len(deletes)
            await session.send(Video(b'legacy', 'image/jpeg'))
            await session.send(Video(b'legacy-next', 'image/jpeg'))
            assert len([v for v in wire.sent if v.get('type') == 'conversation.item.delete']) == before
            assert not any(v.get('type') == 'response.create' for v in wire.sent)
        finally:
            await session.close()
    asyncio.run(scenario())


def test_grok_offer_fact_is_delivered_before_response(packages):
    from ouroboros.realtime_setup import loaded_provider
    async def scenario():
        wire = SilentWire('grok', None)
        session = loaded_provider('grok_realtime:grok')['session'](
            selected(), LiveOptions(), connect=wire.connect, observe=lambda *_: None)
        try:
            await session.open(); await session.receive()
            await session.send(Respond('A reported peer joined.'))
            assert wire.sent[-1]['item']['content'][0]['text'].endswith('A reported peer joined.')
            assert not any(v.get('type') == 'response.create' for v in wire.sent)
            await session.receive()
            assert wire.sent[-1]['type'] == 'response.create'
        finally:
            await session.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('name,engine', ENGINES)
def test_attend_schema_wire_cost_and_nullable_fields(packages, name, engine):
    from dataclasses import replace
    from ouroboros.realtime_setup import loaded_provider
    from ouroboros.context_fit import estimate_context_prompt_tokens
    from skills.live_conversation.attention import attend_tool
    tool = attend_tool()
    params = dict(tool.parameters)
    assert set(params['required']) == set(params['properties'])
    assert all('null' in field['type'] for field in params['properties'].values())
    schema = {'type':'function', 'function':{'name':tool.name,'description':tool.description,'parameters':params}}
    factory = loaded_provider(f'{name}:{engine}')['session']
    def wire(tools):
        return factory(replace(selected(), tools_json=canonical_json(tools)), LiveOptions(),
                       connect=lambda *_: None, observe=lambda *_: None).wire
    baseline, with_tool = wire([]), wire([schema])
    before, after = (estimate_context_prompt_tokens([value]) for value in (baseline, with_tool))
    assert after > before
    print('ATTEND_SCHEMA ' + json.dumps({'engine':engine,'before':before,'after':after,'added_tokens':after-before}))
