"""Independent R27 interactions; synthetic actors only, no provider connections."""
import asyncio
import json
from dataclasses import replace

import pytest

from ouroboros.contracts.live_conversation import LiveOptions, ToolOutputs
from tests.test_live_conversation import host_world as host_world


@pytest.mark.parametrize('folded,renewing', [(False, False), (True, False), (True, True)])
def test_full_result_must_wait_when_serving_room_cannot_hold_it(host_world, monkeypatch, record_property, folded, renewing):
    from ouroboros.tools.tool_result import ToolResult
    from ouroboros.tools.compact_context import _compact_context
    from skills.live_conversation.driver import _Call
    host = host_world.host
    source = [
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'prior', 'type': 'function', 'function': {
            'name': 'read_file', 'arguments': '{"path":"exact-prior"}'}}]},
        {'role': 'tool', 'tool_call_id': 'prior', 'content': 'Exact completed prior evidence. ' * 6000}]
    host.messages.extend(source)
    host._context_changed()
    body = 'Result from one completed effect. ' * 1600
    effects = []
    def execute(name, args):
        effects.append(name)
        return ToolResult(status='ok', code='OK', text=body)

    async def consume(session):
        host._held_input = host._renewal_window.total_equivalent - host._renewal_window.reserve - 1200
        host._history_at_receipt = host._history_estimate()
        if folded:
            shown = json.loads(_compact_context(host.ctx, inspect=True))
            unit = next(row for row in shown['units'] if row['kind'] == 'tool')
            keep = [row['unit_id'] for row in shown['units'] if 'unit_id' in row and row != unit]
            result = _compact_context(host.ctx, working_note='The prior exact evidence remains saved.',
                keep_unit_ids=keep, expected_view_revision=shown['view_revision'])
            assert 'requested' in result
            assert await host.apply_context_view()
            assert host.ctx._context_view_receipt['status'] == 'applied'
            assert source[-1] not in host.messages
        monkeypatch.setattr(host.tools, 'execute_result', execute)
        call = _Call(host)
        call.session = session
        call.connected.set()
        sent = []
        async def send(command): sent.append(command)
        session.send = send
        if renewing:
            # The real quiet-boundary consumer requests this successor as soon
            # as the fold publishes. Old native input/tools can continue while
            # it prepares, and tools_running prevents adoption mid-batch.
            call.request_renewal()
            assert call.renewing
        await call.tool_queue.put({'id': 'one', 'type': 'function', 'function': {
            'name': 'list_available_tools', 'arguments': '{}'}})
        worker = asyncio.create_task(call.tools())
        try:
            await asyncio.wait_for(call.tool_queue.join(), 30)
            delivered = [c for c in sent if isinstance(c, ToolOutputs)]
            rows = (delivered[0].messages if delivered else call.deferred_tool_outputs[0][0].messages)
            fits = await host.delivery_fits(rows, tuple(host.schemas))
            record_property('full_result', json.dumps(dict(folded=folded, fits=fits,
                renewing=call.renewing, due=host.renewal_due(), state_refit=host.state.get('refit'),
                delivered=len(delivered), deferred=len(call.deferred_tool_outputs), result_chars=len(body))))
            assert not fits and effects == ['list_available_tools']
            assert not delivered, 'the projector requested a refit but the driver sent the oversized complete result'
            assert call.deferred_tool_outputs and call.preparing is not None
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await call.close_preparation()
    asyncio.run(host.session(consume, LiveOptions()))


@pytest.mark.parametrize('vision,same_source', [(True,False),(True,True),(False,False)])
def test_serving_price_uses_same_basis_as_receipt(host_world, record_property, vision, same_source):
    from tests.test_live_image_retention import look
    from ouroboros.context_fit import estimate_context_prompt_tokens
    from ouroboros.live_context_view import serving_delivery_room
    from skills.live_conversation.driver import _Call
    host = host_world.host
    host.model = replace(host.model, vision=vision,
        facts=replace(host.model.facts, conversation_window_tokens=50000, working_reserve_tokens=10000))
    async def run():
        for i in range(2):
            row = look(host, i)
            row['content'][0]['_capture_source'] = 'same' if same_source else str(i)
            await host.observe_context(row)
        async def consume(session):
            host._held_input = 39000
            host._history_at_receipt = host._history_estimate()
            rows = ({'role':'tool','tool_call_id':'small','content':'Completed once.'},)
            host.messages.extend(rows)
            host._context_changed()
            expected = 1000 - estimate_context_prompt_tokens(rows)
            actual = serving_delivery_room(host, rows, tuple(host.schemas))
            record_property('image_serving_room', json.dumps(dict(vision=vision,same_source=same_source,
                expected_minimum=expected, actual=actual, due=host.renewal_due())))
            call = _Call(host); call.session=session; call.connected.set()
            sent=[]
            async def send(command): sent.append(command)
            session.send=send
            call.deferred_tool_outputs.append((ToolOutputs(rows, tuple(host.schemas)), []))
            # A newly loaded declaration cannot fit the fresh route. The
            # existing connection still has its old catalog and can deliver
            # the tiny completed result. Establish the refusal with the real
            # fitter, not an injected capacity exception.
            host.schemas.append({'type':'function','function':{
                'name':'new_large_capability','description':'Exact capability description. ' * 10000,
                'parameters':{'type':'object','properties':{}}}})
            host._context_changed()
            call.preparing=asyncio.create_task(host.prepare_session(LiveOptions()))
            try:
                await asyncio.gather(call.preparing, return_exceptions=True)
                from ouroboros.contracts.realtime import RealtimeSetupError
                assert isinstance(call.preparing.exception(), RealtimeSetupError)
                assert call.preparing.exception().code == 'context_capacity'
                await call.check_preparation()
            finally:
                await call.close_preparation()
            assert len(sent) == 1, 'already-held canonical pixels are not new delivery growth'
        await host.session(consume, LiveOptions())
    asyncio.run(run())


@pytest.mark.parametrize('looks', [0, 6])
def test_receipted_looks_allow_small_tool_and_blocked_room_release(host_world, monkeypatch, looks):
    from ouroboros.context_fit import estimate_context_prompt_tokens
    from ouroboros.live_context_view import serving_delivery_room
    from ouroboros.tools.tool_result import ToolResult
    from skills.live_conversation.driver import _Call
    from tests.test_live_image_retention import look
    from ouroboros.contracts.live_conversation import ContextUpdate
    host = host_world.host
    host.plan = replace(host.plan, preferred_mode='nano')
    monkeypatch.setattr(host.tools, 'execute_result', lambda *a: ToolResult(status='ok', code='OK', text='Small result.'))
    async def consume(session):
        for i in range(looks):
            await host.observe_context(look(host, i))
        held = host._renewal_window.total_equivalent - host._renewal_window.reserve - 1200
        host.response_done('receipt', usage={'audio_input_tokens': held}, raw_usage={}, status='completed')
        tiny = [{'role': 'user', 'content': 'tiny fact'}]
        assert 0 < serving_delivery_room(host, tiny, host.schemas) <= 1200 - estimate_context_prompt_tokens(tiny)
        assert not host.renewal_due()
        call = _Call(host); call.session = session; call.connected.set()
        sent = []
        async def send(command): sent.append(command)
        session.send = send
        await call.tool_queue.put({'id': 'small', 'type': 'function', 'function': {'name': 'list_available_tools', 'arguments': '{}'}})
        worker = asyncio.create_task(call.tools())
        try:
            await asyncio.wait_for(call.tool_queue.join(), 30)
            assert any(isinstance(c, ToolOutputs) for c in sent)
            assert not host.state.get('refit') and not call.preparing
            call.pending_context_rows = list(tiny)
            call.failed_preparation_revision = host._history_version
            await call.release_serving()
            assert any(isinstance(c, ContextUpdate) and c.messages == tuple(tiny) for c in sent)
        finally:
            worker.cancel(); await asyncio.gather(worker, return_exceptions=True)
            await call.close_preparation()
    asyncio.run(host.session(consume, LiveOptions()))


@pytest.mark.parametrize('new_looks', [0, 3])
def test_live_pixels_since_receipt_are_pending_even_when_next_fit_collapses_them(host_world, new_looks):
    from ouroboros.live_context_view import serving_delivery_room
    from ouroboros.context_fit import estimate_context_prompt_tokens
    from tests.test_live_image_retention import look
    host = host_world.host
    async def consume(session):
        await host.observe_context(look(host, 0))
        held = host._renewal_window.total_equivalent - host._renewal_window.reserve - 1200
        host.response_done('receipt', usage={'audio_input_tokens': held}, raw_usage={}, status='completed')
        receipt = estimate_context_prompt_tokens(host.messages, host.schemas)
        delivered = []
        async def send(command): delivered.extend(command.messages)
        session.send = send
        for i in range(1, new_looks + 1):
            row = look(host, i)
            await host.observe_context(row)
            from ouroboros.contracts.live_conversation import ContextUpdate
            await session.send(ContextUpdate((row,), respond=False))
        from ouroboros.context_images import image_positions
        assert len(image_positions(delivered)) == new_looks
        tiny = [{'role': 'user', 'content': 'tiny fact'}]
        growth = estimate_context_prompt_tokens([*host.messages, *tiny], host.schemas) - receipt
        if new_looks:
            assert serving_delivery_room(host, tiny, host.schemas) == 1200 - max(growth, estimate_context_prompt_tokens(tiny))
        assert (serving_delivery_room(host, tiny, host.schemas) >= 0) is (new_looks == 0)
        assert not host.renewal_due(), 'next fresh fit still collapses the same-source frames'
    asyncio.run(host.session(consume, LiveOptions()))


@pytest.mark.parametrize('successor', ['none', 'native', 'fresh'])
def test_small_result_waits_only_for_a_requested_fresh_successor(host_world, monkeypatch, successor):
    from skills.live_conversation.driver import _Call
    from ouroboros.tools.tool_result import ToolResult
    host = host_world.host
    monkeypatch.setattr(host.tools, 'execute_result', lambda *a: ToolResult(status='ok', code='OK', text='Small result.'))
    async def consume(session):
        call = _Call(host); call.session = session; call.connected.set()
        if successor != 'none':
            call.request_renewal(resuming=successor == 'native')
        sent = []
        async def send(command): sent.append(command)
        session.send = send
        await call.tool_queue.put({'id': 'small', 'type': 'function', 'function': {'name': 'list_available_tools', 'arguments': '{}'}})
        worker = asyncio.create_task(call.tools())
        try:
            await asyncio.wait_for(call.tool_queue.join(), 30)
            assert bool(call.deferred_tool_outputs) is (successor == 'fresh')
            assert call.resume is (successor == 'native')
            assert bool([c for c in sent if isinstance(c, ToolOutputs)]) is (successor != 'fresh')
        finally:
            worker.cancel(); await asyncio.gather(worker, return_exceptions=True)
            await call.close_preparation()
    asyncio.run(host.session(consume, LiveOptions()))
