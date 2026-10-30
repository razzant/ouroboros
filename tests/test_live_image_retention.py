"""Exact frame files survive; pixels leave only at a replaced prefix or call end."""
import asyncio
import base64
import copy
import json
import time
from dataclasses import replace

import pytest

from ouroboros.artifacts import task_artifact_dir_path
from ouroboros.context_images import image_positions
from ouroboros.contracts.live_conversation import LiveOptions
from tests.test_live_conversation import host_world as host_world


def look(host, i, *, readable=True):
    data = (f'exact synthetic pixels {i};'.encode() * 8000)[:150000]
    path = task_artifact_dir_path(host.ctx.drive_root, host.ctx.task_id, create=True) / f'frame-{i}.jpg'
    if readable:
        path.write_bytes(data)
    return {'role':'user', '_call_frame_source':'screen', 'content':[
        {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(data).decode()},
         '_capture_source':'screen', **({'_source_path':str(path)} if readable else {})}]}


@pytest.mark.parametrize('count,remaining', [(4,4),(12,2)])
def test_call_end_applies_owner_floor_before_text_continuation(host_world, count, remaining):
    from ouroboros.context_fit import estimate_context_prompt_tokens
    host=host_world.host
    host.plan=replace(host.plan, preferred_mode='nano')
    async def run():
        for i in range(count): await host.observe_context(look(host,i))
        assert len(image_positions(host.messages)) == count
        await host.finish()
    asyncio.run(run())
    assert len(image_positions(host.messages)) == remaining
    assert estimate_context_prompt_tokens(host.messages) + 10000 < 24000
    for i in range(count-remaining):
        assert f'frame-{i}.jpg' in json.dumps(host.messages)


def test_fresh_adoptions_bound_pixels_checkpoints_and_room_readers(host_world, record_property):
    host=host_world.host
    host.plan=replace(host.plan, preferred_mode='nano')
    measurements=[]
    async def run():
        for interval in range(8):
            for i in range(interval*5,(interval+1)*5): await host.observe_context(look(host,i))
            assert len(image_positions(host.messages)) <= 7  # two held + this five-look interval
            before=copy.deepcopy(host.messages)
            lease=await host.prepare_session(LiveOptions())
            try:
                assert host.messages == before  # speculative fitting cannot rewrite the serving prefix
                async def consume(session):
                    assert len(image_positions(host.messages)) == 2
                    size=len(json.dumps(host.messages).encode())
                    start=time.monotonic(); host.checkpoint('ready'); elapsed=time.monotonic()-start
                    readers=list(task_artifact_dir_path(host.ctx.drive_root,host.ctx.task_id).rglob('room-*.json'))
                    measurements.append(dict(looks=(interval+1)*5,canonical_bytes=size,
                        largest_reader=max(p.stat().st_size for p in readers),checkpoint_seconds=round(elapsed,4)))
                    assert size < 500000 and measurements[-1]['largest_reader'] < 500000
                    import re
                    from ouroboros.evidence_packs import handle_bytes
                    wire_rows = json.loads(session.selected.messages_json)
                    paths = [path for row in wire_rows if isinstance(row.get('content'), str)
                             for path in re.findall(r'path=("[^"\n]*room-[^"\n]+")', row['content'])]
                    assert paths
                    for path in paths:
                        assert json.loads(handle_bytes(host.ctx.drive_root, host.ctx.task_id, json.loads(path)))
                await lease.run(consume)
            finally: await lease.close()
        await host.finish()
    asyncio.run(run())
    record_property('pixel_retention',json.dumps(measurements))


@pytest.mark.parametrize('with_words', [False, True])
def test_pending_growth_prices_next_projection_not_unsent_pixels(host_world, with_words):
    from ouroboros.realtime_setup import session_candidate
    host=host_world.host
    session_candidate(host, host_world.provider['session'], LiveOptions())
    host._held_input=80000
    host.model=replace(host.model,facts=replace(host.model.facts,conversation_window_tokens=92000,working_reserve_tokens=10000))
    host._renewal_window=None
    async def run():
        await host.observe_context(look(host,0))
        host._history_at_receipt=host._history_estimate()
        for i in range(1,12): await host.observe_context(look(host,i))
    asyncio.run(run())
    if with_words:
        host.messages.append({'role':'user','content':'Real new words. '*1000}); host._context_changed()
    assert host.renewal_due() is with_words, 'only new content carried on the next wire can force renewal'


def test_canonical_floor_keeps_unreadable_originals(host_world):
    from ouroboros.context_images import evict_stale_image_blocks
    rows=[look(host_world.host,i,readable=i!=0) for i in range(5)]
    original=copy.deepcopy(rows[0])
    assert evict_stale_image_blocks(rows,owner_mode='nano',readable_only=True)
    assert rows[0]==original and len(image_positions(rows))==3
    assert 'frame-1.jpg' in rows[1]['content'][0]['text']


@pytest.mark.parametrize('arrivals,remaining', [(1,4),(2,2)])
def test_adoption_floors_catchup_only_after_snapshot_is_replaced(host_world, arrivals, remaining):
    host=host_world.host
    host.plan=replace(host.plan, preferred_mode='nano')
    async def run():
        for i in range(3): await host.observe_context(look(host,i))
        source=copy.deepcopy(host.messages)
        lease=await host.prepare_session(LiveOptions())
        async def send(command): assert command.respond is False
        lease.session.send=send
        try:
            for i in range(3,3+arrivals): await host.observe_context(look(host,i))
            assert host.messages[:len(source)]==source
            async def consume(session):
                assert lease.caught_up
                assert len(image_positions(host.messages))==remaining
            await lease.run(consume)
        finally: await lease.close()
    asyncio.run(run())


@pytest.mark.parametrize('reader_loaded', [False, True])
def test_image_reader_follows_each_fitted_catalog_after_canonical_floor(host_world, reader_loaded):
    from ouroboros.realtime_setup import rows_for_engine
    host = host_world.host
    host.plan = replace(host.plan, preferred_mode='nano')
    schema = {'type': 'function', 'function': {'name': 'view_image'}}
    async def run():
        for i in range(6): await host.observe_context(look(host, i))
        await host.finish()
    asyncio.run(run())
    assert len(image_positions(host.messages)) == 2
    for loaded in (reader_loaded,):
        rendered = rows_for_engine(host.messages, vision=True, fresh=True, owner_mode='nano', schemas=[schema] if loaded else [])
        text = json.dumps(rendered)
        assert 'frame-0.jpg' in text
        assert ('re-view: view_image' in text) is loaded
        assert '_evicted_image' not in text and '_source_path' not in text
    assert 're-view: view_image' in json.dumps(host.messages)  # call end restores the text reader


@pytest.mark.parametrize('count', [2, 6])
def test_pricing_copy_keeps_canonical_pixels_until_adoption(host_world, count):
    from ouroboros.realtime_setup import measure_session_wire
    host = host_world.host
    host.plan = replace(host.plan, preferred_mode='nano')
    rows = [*host.messages, *(look(host, i) for i in range(count))]
    canonical = []
    session, _, _, _ = measure_session_wire(host, host.plan, rows, host.schemas,
        host_world.provider['session'], LiveOptions(), canonical_rows=canonical)
    assert len(image_positions(canonical)) == count
    assert len(image_positions(json.loads(session.selected.messages_json))) == 1  # same-source projection
