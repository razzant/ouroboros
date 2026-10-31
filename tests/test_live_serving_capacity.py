"""Serving custody and failed snapshot identity."""
import asyncio
import json
from dataclasses import replace

import pytest

from ouroboros.contracts.live_conversation import LiveOptions, ToolOutputs
from tests.test_live_conversation import host_world as host_world


def test_legacy_session_needs_no_private_tools_attribute_after_capacity_failure(host_world):
    from ouroboros.contracts.realtime import RealtimeSetupError
    from skills.live_conversation.driver import _Call
    host = host_world.host
    async def run():
        call = _Call(host)
        # Implements every member of the public LiveSession contract, including
        # its additive silent-context capability, without a private .tools.
        from ouroboros.realtime_setup import input_from_context_plan
        class LegacySession:
            __slots__ = ('selected', 'wire', 'sent', 'closed')
            silent_context_updates = True
            output_bound_tokens = 100
            renewal_ready = True
            def __init__(self):
                self.selected = input_from_context_plan(host.plan, model=host.turn.native_model,
                    selected_tools=host.schemas, output_allowance_tokens=100)
                self.wire, self.sent, self.closed = {}, [], False
            def final_captions(self): return []
            async def open(self): pass
            async def receive(self): return await asyncio.Future()
            async def send(self, command): self.sent.append(command)
            async def close(self): self.closed = True
        call.session = LegacySession()
        call.connected.set()
        sent = call.session.sent
        async def capacity():
            raise RealtimeSetupError('context_capacity', 'protected remainder does not fit')
        call.preparing = asyncio.create_task(capacity())
        await asyncio.sleep(0)
        output = ToolOutputs(({'role': 'tool', 'tool_call_id': 'finished', 'content': 'Completed once.'},),
                             tuple(host.schemas))
        call.deferred_tool_outputs.append((output, []))
        await call.check_preparation()
        assert len(sent) == 1 and sent[0].messages == output.messages
        assert not call.renewing and not call.deferred_tool_outputs
    asyncio.run(run())


def test_failed_preparation_cannot_flush_a_result_known_not_to_fit_serving_connection(host_world, record_property):
    from ouroboros.context_budget import bound_host_row
    from ouroboros.contracts.realtime import RealtimeSetupError
    from skills.live_conversation.driver import _Call
    host = host_world.host
    host.model = replace(host.model, facts=replace(host.model.facts,
                         conversation_window_tokens=50000, working_reserve_tokens=10000))
    async def consume(session):
        # A native input receipt, plus a chosen result too large for the serving
        # socket, are precisely why the tool path withheld this output.
        host.response_done('review-native-receipt', usage={
            'text_input_tokens': host._selected_raw_estimate,
            'audio_input_tokens': 39000 - host._selected_raw_estimate}, raw_usage={}, status='completed')
        assert host._held_input == 39000
        from tests.test_context_source_view import _tool
        text = 'Exact chosen read result. ' * 6000
        result_rows = _tool('read-chosen', invocation='one-read', text=text)
        result_rows[0]['tool_calls'][0]['function'] = {'name': 'read_file',
            'arguments': json.dumps({'path': 'retained.txt', 'max_chars': len(text)})}
        host.messages.append(bound_host_row('owner_words', 'Newest exact request. '*1000))
        host.messages.extend(result_rows)
        host._context_changed()
        assert host.renewal_due()
        call = _Call(host)
        call.session = session
        sent = []
        async def send(command): sent.append(command)
        call.send = send
        # The real fresh-candidate ladder has to establish the failed fit; no
        # injected exception stands in for the condition being reviewed.
        call.preparing = asyncio.create_task(host.prepare_session(LiveOptions()))
        await asyncio.gather(call.preparing, return_exceptions=True)
        assert isinstance(call.preparing.exception(), RealtimeSetupError)
        assert call.preparing.exception().code == 'context_capacity'
        call.deferred_tool_outputs.append((ToolOutputs(tuple(result_rows), tuple(host.schemas)), []))
        with pytest.raises(RealtimeSetupError, match='context_capacity'):
            await call.check_preparation()
        assert call.deferred_tool_outputs, 'the completed effect retains delivery custody'
        record_property('capacity_fallback', json.dumps({
            'window': host.model.facts.conversation_window_tokens,
            'reserve': host.model.facts.working_reserve_tokens,
            'held': host._held_input,
            'pending': host._history_estimate() - host._history_at_receipt,
            'result_chars': len(text),
            'sent_results': len(sent),
            'preparation_deficit': host.trace['fit_refusal']['deficit_tokens']}))
        assert not sent, 'a known overflowing withheld body was flushed on the old socket'
    asyncio.run(host.session(consume, LiveOptions()))


def test_changed_canonical_view_during_failed_fit_can_retry(host_world, record_property):
    from ouroboros.contracts.realtime import RealtimeSetupError
    from skills.live_conversation.driver import _Call
    host = host_world.host
    host.model = replace(host.model, facts=replace(host.model.facts,
                         conversation_window_tokens=50000, working_reserve_tokens=10000))
    async def consume(session):
        # This input cannot leave while it is the newest owner turn. A later
        # completed exchange makes it eligible for the approved physical cut.
        await host.caption('owner', 'Exact old owner words. '*10000, 'large-old-turn')
        call = _Call(host)
        call.session = session
        failed, release = asyncio.Event(), asyncio.Event()
        prepare = host.prepare_session
        attempts = []
        async def delayed_failure(options):
            attempts.append(host._history_version)
            try:
                return await prepare(options)
            except RealtimeSetupError as exc:
                assert exc.code == 'context_capacity'
                failed.set()
                await release.wait()
                raise
        host.prepare_session = delayed_failure
        call.request_renewal()
        await asyncio.wait_for(failed.wait(), 20)
        await host.caption('ouroboros', 'The serving voice answered.', 'middle-answer', completed=True)
        await host.caption('owner', 'The next owner turn.', 'new-owner-turn')
        latest_revision = host._history_version
        release.set()
        await asyncio.gather(call.preparing, return_exceptions=True)
        await call.check_preparation()
        # Establish with the real candidate builder that this exact NEW view
        # now fits. No fake capacity exception or hypothetical recovery is used.
        lease = await prepare(LiveOptions())
        try:
            assert lease.request.prompt_tokens_estimate + lease.request.working_reserve_tokens <= 50000
            record_property('changed_during_fit', json.dumps({
                'failed_source_revision': attempts[0], 'new_revision': latest_revision,
                'blocked_revision': call.failed_preparation_revision,
                'new_input_tokens': lease.request.prompt_tokens_estimate}))
        finally:
            await lease.close()
        call.request_renewal()
        await asyncio.sleep(0)
        try:
            assert len(attempts) == 2, 'the failed old snapshot suppressed an already changed, fitting view'
        finally:
            await call.close_preparation()
    asyncio.run(host.session(consume, LiveOptions()))


@pytest.mark.parametrize('large', [False, True])
@pytest.mark.parametrize('kind', ['mail', 'offer', 'tools'])
def test_failed_fit_delivery_uses_serving_room_and_keeps_obligation(host_world, large, kind):
    from types import SimpleNamespace
    from ouroboros.contracts.realtime import RealtimeSetupError
    from ouroboros.contracts.live_conversation import ContextUpdate
    from ouroboros.realtime_setup import input_from_context_plan
    from skills.live_conversation.driver import _Call
    host = host_world.host
    host.model = replace(host.model, facts=replace(host.model.facts, conversation_window_tokens=50000))
    host._selected_raw_estimate = 39000
    host._history_at_receipt = host._history_estimate()
    async def run():
        sent = []
        async def send(command): sent.append(command)
        call = _Call(host)
        call.session = SimpleNamespace(selected=input_from_context_plan(host.plan, model=host.turn.native_model,
            selected_tools=host.schemas, output_allowance_tokens=100), send=send, renewal_ready=True)
        call.connected.set()
        rows = [{'role':'user','content':'New exact fact. ' * (10000 if large else 1)}]
        await host.observe_context(rows[0])
        call.failed_preparation_revision = host._history_version
        # A failed new declaration is NOT the serving catalog. Its enormous schema
        # cannot wrongly block a small result which fits the old connection.
        old = tuple(host.schemas)
        host.schemas[:] = [{'type':'function', 'function':{'name':'later', 'description':'big '*100000}}]
        assert host.renewal_due()
        if kind == 'mail':
            action = lambda: call.context_update(rows)
        elif kind == 'offer':
            call.pending_offers.extend(rows)
            action = call.offer_pending
        else:
            call.deferred_tool_outputs.append((ToolOutputs(tuple(rows), tuple(host.schemas)), []))
            action = call.release_serving
        if large:
            with pytest.raises(RealtimeSetupError, match='context_capacity'):
                await action()
            assert not sent
            assert any(rows[0]['content'] in row.get('content', '') for row in host.messages if isinstance(row.get('content'), str))
            assert call.pending_context_rows or call.pending_offers or call.deferred_tool_outputs
        else:
            await action()
            assert len(sent) == 1
            if kind == 'tools':
                assert sent[0].tools == old and not call.deferred_tool_outputs
            else:
                assert isinstance(sent[0], ContextUpdate) and sent[0].messages == tuple(rows)
    asyncio.run(run())


def test_successful_tool_delivery_updates_driver_owned_serving_catalog(host_world):
    from types import SimpleNamespace
    from skills.live_conversation.driver import _Call
    async def run():
        call = _Call(host_world.host)
        async def send(command): pass
        call.session = SimpleNamespace(send=send)
        call.connected.set()
        new = ({'type':'function','function':{'name':'new'}},)
        await call.send(ToolOutputs((), new))
        assert call.serving_catalog() == new
    asyncio.run(run())


@pytest.mark.parametrize('resume', [True, False])
def test_catalog_record_survives_native_resume_but_new_fit_rebinds(host_world, resume):
    from types import SimpleNamespace
    from skills.live_conversation.driver import _Call
    async def run():
        call = _Call(host_world.host)
        call.resume = resume
        old = ({'type':'function','function':{'name':'last_delivered'}},)
        call.delivered_catalog = old
        async def complete(): return None
        async def wait(): await asyncio.Future()
        call.peers, call.provider, call.mailbox, call.refit_boundary = complete, wait, wait, wait
        call.finalize = complete
        await call.consume(SimpleNamespace(renewal_ready=True, selected=SimpleNamespace(tools_json='[]')))
        assert call.delivered_catalog == (old if resume else None)
    asyncio.run(run())


@pytest.mark.parametrize('large', [True, False])
def test_serving_delivery_prices_its_body_even_after_an_authored_view_shrank(host_world, large):
    from ouroboros.context_fit import estimate_context_prompt_tokens
    host = host_world.host
    host.model = replace(host.model, facts=replace(host.model.facts, conversation_window_tokens=50000))
    old = {'role':'assistant', 'content':'old '*5000}
    host._history_at_receipt = estimate_context_prompt_tokens([*host.messages, old], host.schemas)
    # The native receipt includes audio as well as text. A later authored fold
    # changes canonical history; it cannot remove bytes from the serving socket.
    host._held_input = 39000
    rows = [{'role':'tool','tool_call_id':'finished','content':'new '* (2000 if large else 10)}]
    host.messages.extend(rows)
    assert asyncio.run(host.delivery_fits(rows, tuple(host.schemas))) is not large


def test_changed_view_gets_fresh_fit_before_old_socket_refuses_deferred_result(host_world, record_property):
    from ouroboros.contracts.realtime import RealtimeSetupError
    from skills.live_conversation.driver import _Call
    from tests.test_context_source_view import _tool
    host=host_world.host
    host.model=replace(host.model,facts=replace(host.model.facts,conversation_window_tokens=50000,working_reserve_tokens=10000))
    async def consume(session):
        host.response_done("native-held",usage={"text_input_tokens":host._selected_raw_estimate,
            "audio_input_tokens":39000-host._selected_raw_estimate},raw_usage={},status="completed")
        await host.caption("owner", "Large current owner turn. "*6000, "large")
        outputs=_tool("chosen",invocation="chosen-once",text="Chosen exact read. "*5000)
        outputs[0]["tool_calls"][0]["function"]={"name":"read_file","arguments":json.dumps({"path":"synthetic.txt","max_chars":95000})}
        host.messages.extend(outputs);host._context_changed()
        call=_Call(host);call.session=session;call.connected.set()
        actual_prepare=host.prepare_session
        failed,release=asyncio.Event(),asyncio.Event()
        attempts=[]
        async def delayed(options):
            attempts.append(host._history_version)
            try: return await actual_prepare(options)
            except RealtimeSetupError as exc:
                assert exc.code=="context_capacity"
                failed.set();await release.wait();raise
        host.prepare_session=delayed
        call.deferred_tool_outputs.append((ToolOutputs(tuple(outputs),tuple(host.schemas)),[]))
        call.request_renewal()
        await asyncio.wait_for(failed.wait(),30)
        await host.caption("ouroboros","Response before next question.","reply",completed=True)
        await host.caption("owner","Next question makes the old turn cuttable.","new")
        release.set();await asyncio.gather(call.preparing,return_exceptions=True)
        lease=await actual_prepare(LiveOptions())
        try:
            assert lease.request.prompt_tokens_estimate+lease.request.working_reserve_tokens<=50000
            record_property("changed_pending",json.dumps({"source_revision":attempts[0],"current_revision":host._history_version,"fresh_input":lease.request.prompt_tokens_estimate,"old_fits":await host.delivery_fits(outputs,tuple(host.schemas))}))
        finally: await lease.close()
        assert not await host.delivery_fits(outputs,tuple(host.schemas))
        try:
            await call.check_preparation()
            call.request_renewal()
            await asyncio.sleep(0)
            assert len(attempts)==2
        finally: await call.close_preparation()
    asyncio.run(host.session(consume,LiveOptions()))



@pytest.mark.parametrize('pending', [False, True])
def test_native_resume_delivers_room_rows_before_answer_pointer(host_world, pending):
    from types import SimpleNamespace
    from ouroboros.contracts.live_conversation import ContextUpdate, LiveEvent
    from skills.live_conversation.driver import _Call
    async def run():
        host=host_world.host
        host.state['greeted']=True
        call=_Call(host); call.connection_resumed=True
        rows=[{'role':'user','content':'Mail accepted while disconnected.'}] if pending else []
        call.pending_context_rows.extend(rows)
        sent=[]
        async def send(command): sent.append(command)
        batches=iter([[LiveEvent('ready',dict(input_rate=16000,output_rate=24000))]])
        async def receive():
            try: return next(batches)
            except StopIteration: raise asyncio.CancelledError
        call.session=SimpleNamespace(receive=receive, send=send, selected=SimpleNamespace(tools_json='[]'))
        with pytest.raises(asyncio.CancelledError): await call.provider()
        assert not call.pending_context_rows
        assert [list(command.messages) for command in sent if isinstance(command,ContextUpdate)] == ([rows] if pending else [])
    asyncio.run(run())
