"""Snapshot exposure off the media loop; fitted pixels and canonical sources differ."""
import asyncio
import base64
import copy
import threading
from types import SimpleNamespace

import pytest

from ouroboros.contracts.live_conversation import LiveOptions, Video
from tests.test_live_conversation import host_world as host_world


def test_exposure_hashing_is_off_loop_and_snapshot_owned(host_world, monkeypatch):
    from ouroboros.live_context_view import record_live_exposure, drain_exposure
    from ouroboros.context_compaction import exposed_context_units
    from tests.test_context_source_view import _tool
    host = host_world.host
    view = copy.copy(host)
    view._connection_owner = host
    host._connection_view = view
    held = _tool('held', invocation='held-effect')
    host.messages.extend(held)
    view._context_session = SimpleNamespace(response_context_rows={'answer': copy.deepcopy(host.messages)})
    seen = []
    def hashing(rows, sent):
        seen.append(threading.get_ident())
        return exposed_context_units(rows, sent)
    monkeypatch.setattr('ouroboros.context_compaction.exposed_context_units', hashing)
    async def run():
        loop_thread = threading.get_ident()
        record_live_exposure(view, 'answer')
        host.messages.append({'role':'user','content':'Arrived after receipt'})
        await drain_exposure(host)
        assert seen and all(thread != loop_thread for thread in seen)
        assert host.ctx._last_context_observation['messages'] == host.messages[:-1]
        assert host.ctx._last_context_observation['exposed_units']
    asyncio.run(run())


@pytest.mark.parametrize('carried', [True, False])
def test_fresh_connection_marks_only_fitted_pixels_seen(host_world, carried):
    from skills.live_conversation.driver import _Call
    from hashlib import sha256
    from ouroboros.context_images import fresh_call_images
    host = host_world.host
    payload = b'exact frame'
    row = {'role':'user', '_call_frame_source':'owner', '_call_frame_digest':sha256(payload).hexdigest(), 'content':[
        {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(payload).decode()},
         '_capture_source':'attend:owner','_source_path':'/original.jpeg'}]}
    host.messages.append(row)
    if not carried:
        newer = copy.deepcopy(row)
        newer['_call_frame_digest'] = 'newer'
        host.messages.append(newer)
    host._fitted_frame_rows = fresh_call_images(host.messages)
    async def run():
        call = _Call(host)
        call.attention.frames['owner'] = Video(payload, 'image/jpeg', 'owner')
        async def complete(): return None
        async def wait(): await asyncio.Future()
        call.peers, call.provider, call.mailbox, call.refit_boundary = complete, wait, wait, wait
        call.finalize = complete
        await call.consume(SimpleNamespace(renewal_ready=True))
        sent = []
        async def send(frame): sent.append(frame)
        await call.attention.deliver(SimpleNamespace(send=send))
        assert bool(sent) is not carried
    asyncio.run(run())


def test_prepared_fit_keeps_canonical_pixels_when_next_image_arrives(host_world):
    from tests.test_context_image_floors import images
    from dataclasses import replace
    host = host_world.host
    host.plan = replace(host.plan, preferred_mode='nano')
    async def run():
        for row in images(4, source=''):
            await host.observe_context(row)
        source = copy.deepcopy(host.messages)
        lease = await host.prepare_session(LiveOptions())
        async def send(command):
            assert command.respond is False
        lease.session.send = send
        try:
            await host.observe_context(images(1, source='camera')[0])
            assert host.messages[:len(source)] == source
            async def consume(session):
                assert lease.caught_up
                assert host.fitted_frame_rows()[-1] == host.messages[-1]
            await lease.run(consume)
        finally:
            await lease.close()
    asyncio.run(run())


@pytest.mark.parametrize('fresh', [True, False])
def test_requested_look_offer_is_not_repeated_when_fresh_fit_holds_it(host_world, fresh):
    import json
    from skills.live_conversation.driver import _Call
    from ouroboros.contracts.live_conversation import ToolBatch
    host = host_world.host
    image = {'role':'user', '_call_frame_source':'owner', '_call_frame_digest':'digest', 'content':[
        {'type':'image_url','image_url':{'url':'data:image/png;base64,YQ=='}, '_source_path':'/original.png'}]}
    async def batch(calls):
        return ToolBatch([{'role':'tool','tool_call_id':'look','content':json.dumps({'frames':[]})}], silent=('look',))
    host.tool_batch = batch
    host.renewal_due = lambda: True
    async def run():
        call = _Call(host)
        call.connected.set()
        call.session = SimpleNamespace()
        call.request_renewal = lambda: setattr(call, 'renewing', True)
        call.attention.looks['look'] = [image]
        await call.tool_queue.put({'id':'look','type':'function','function':{'name':'attend','arguments':'{}'}})
        worker = asyncio.create_task(call.tools())
        try:
            await call.tool_queue.join()
            assert call.deferred_tool_outputs and any(row.get('_call_frame_source') for row in call.pending_offers)
            if fresh:
                call.refit_offers()
            assert any(row.get('_call_frame_source') for row in call.pending_offers) is not fresh
            assert any(row.get('_call_frame_source') == 'owner' for row in host.messages)
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
    asyncio.run(run())


@pytest.mark.parametrize('waiting', [True, False])
def test_tool_boundary_joins_exposure_without_blocking_the_media_loop(host_world, waiting):
    host = host_world.host
    async def run():
        gate = asyncio.Event()
        if not waiting:
            gate.set()
        host._exposure_job = asyncio.create_task(gate.wait())
        tool = asyncio.create_task(host.tool_batch([]))
        try:
            await asyncio.sleep(0)  # the media/event loop still advances
            if waiting:
                assert not tool.done(), 'tools can otherwise inspect a stale observed view'
        finally:
            gate.set()
            assert await tool == []
            await host._exposure_job
    asyncio.run(run())
