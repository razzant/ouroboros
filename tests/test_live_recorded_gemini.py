"""Structural replay of bf43cbf3, not a scripted model decision or real media.

The journal supplies native ordering, PCM lengths, usage and missing caption-final
flags. Its continuous microphone supplies the clock. Private text/images/handles
are absent. The page sends share after its first frame and plays/flushes by samples.
"""
import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace

from ouroboros import call_protocol as cp, call_peers
from ouroboros.gateway.call_ws import serve_peer
from ouroboros.tool_call_log import invocation_scope
from skills.live_conversation import attention, driver
from tests._live_stack_page import Page, jpeg
from tests.test_gemini_live_boundaries import gemini as gemini, make_session
from tests.test_live_conversation_driver import Host, Batch, until


def inflate(value):
    if isinstance(value, list):
        return [inflate(v) for v in value]
    if not isinstance(value, dict):
        return value
    return {('data' if k == '_pcm_bytes' else k):
            (base64.b64encode(bytes(v)).decode() if k == '_pcm_bytes' else inflate(v)) for k, v in value.items()}


def test_recorded_unfinished_captions_release_share_and_named_look_at_quiet(gemini, monkeypatch, record_property):
    timeline = json.loads((Path(__file__).parent / 'fixtures/live/gemini_bf43cbf3.json').read_text())
    clock = [0.0]
    monkeypatch.setattr(attention, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(call_peers, 'time', SimpleNamespace(monotonic=lambda: clock[0]))

    async def scenario():
        host = Host()
        host.model = gemini.CATALOG.models[0]
        host.hub = call_peers.LiveCallHub(cp.CallSpec('wire-replay', 1, 'gemini', 'gemini'))
        tools = {t.name: t for t in driver.call_tools(None)}
        host.schemas = [{'type': 'function', 'function': {'name': t.name, 'description': t.description,
                        'parameters': dict(t.parameters)}} for t in tools.values()]
        session, wire = make_session(gemini, tools=host.schemas)
        async def batch(calls):
            rows = []
            for c in calls:
                with invocation_scope({'tool_call_id': c['id']}):
                    value = await tools[c['function']['name']].handler(host, json.loads(c['function']['arguments']))
                rows.append({'role': 'tool', 'tool_call_id': c['id'], 'content': value})
            return Batch(rows, silent=[c['id'] for c in calls])
        host.tool_batch = batch
        async def connection(consume, options):
            session.options = options
            await session.open()
            try: return await consume(session)
            finally: await session.close()
        host.session = connection
        call = driver._Call(host)
        offered, native_speech = [], [False]
        send = wire.send
        async def observed_send(raw):
            value = json.loads(raw)
            if 'clientContent' in value and any(word in str(value) for word in ('started showing', 'look you asked')):
                offered.append({'seconds': clock[0], 'native_speech': native_speech[0],
                                'share': 'started showing' in str(value), 'revisit': 'look you asked' in str(value)})
            await send(raw)
        wire.send = observed_send
        page = Page()
        gateway = asyncio.create_task(serve_peer(page, host.hub, role='owner', video=True))
        running = asyncio.create_task(call.run())
        snapshots = []
        try:
            await until(lambda: call.connected.is_set())
            pending = iter(timeline['timeline'])
            row = next(pending, None)
            shown = False
            for tick in range(timeline['end_tick'] + 1):
                clock[0] = tick * .02
                page.tick()
                while row and row['tick'] <= tick:
                    if 'video' in row:
                        page.video(jpeg(str(row['video'])))
                        if not shown:
                            page.share(True)
                            shown = True
                    else:
                        activity = row['event'].get('voiceActivity', {}).get('type')
                        if activity:
                            native_speech[0] = activity == 'ACTIVITY_START'
                        wire.put(inflate(row['event']))
                    row = next(pending, None)
                    await asyncio.sleep(.012)
                if tick in (350, 3150, 4500):
                    snapshots.append({'seconds': clock[0], 'quiet': call.quiet(),
                        'driver': {k: bool(getattr(call, k)) for k in
                                   ('response_open', 'input_open', 'tools_running', 'pending_spoken')},
                        'unheard': call.unheard(), 'session': {k: bool(getattr(session, k)) for k in
                            ('input_pending', 'output_pending', 'turn_busy', 'calls', 'held_context')}})
                if running.done(): await running
                await asyncio.sleep(.001)
            record_property('quiet_checkpoints', json.dumps(snapshots))
            record_property('offers', json.dumps(offered))
            offers = [v['clientContent'] for v in wire.sent if 'clientContent' in v][1:]
            assert any('started showing the screen' in str(v) for v in offers), snapshots
            assert sum('look you asked' in str(v) for v in offers) == 1, snapshots
            assert all(not v['native_speech'] for v in offered), offered
            assert any(v['type'] == 'flush' for v in page.controls), 'recorded barge-in must cross page playback'
            assert page.played and len([v for v in host.usage if v[0] == 'done']) == 5
            assert any(r['speaker'] == 'owner' for r in host.captions)
            assert not any(r.get('code', 1000) != 1000 for r in page.closed), page.closed
        finally:
            page.end()
            await asyncio.wait_for(running, 3)
            await asyncio.wait_for(gateway, 3)
            host.hub.close('test_complete')
    asyncio.run(scenario())
