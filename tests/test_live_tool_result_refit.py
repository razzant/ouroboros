"""A completed call result gets a fresh full view before an ordinary range fallback."""
import asyncio
import json

import pytest

from ouroboros.contracts.live_conversation import LiveOptions, ToolOutputs
from ouroboros.contracts.realtime import RealtimeSetupError
from tests.test_live_conversation import host_world as host_world


@pytest.mark.parametrize('large,failed,reader_loaded,annotated', [
    (False,False,False,False),(True,False,False,False),
    (True,True,False,False),(True,True,True,False),
    (True,True,False,True),(True,True,True,True)])
def test_call_result_refits_whole_then_falls_back_without_repeating_effect(
        host_world, monkeypatch, large, failed, reader_loaded, annotated):
    from skills.live_conversation.driver import _Call
    from ouroboros.tools.tool_result import ToolResult
    host=host_world.host
    host.schemas[:] = [schema for schema in host.schemas if schema['function']['name'] != 'read_file']
    if reader_loaded:
        host.schemas.extend(schema for schema in host.tools.schemas() if schema['function']['name'] == 'read_file')
    prepare = host.prepare_session
    execute_result = host.tools.execute_result
    body='Result from one completed effect. ' * (1600 if large else 1)
    effects=[]
    def execute(name,args):
        effects.append(name)
        return ToolResult(status='ok',code='OK',text=body + ('\nHost annotation.' if annotated else ''),
                          producer_text=body if annotated else None,
                          host_annotations=('Host annotation.',) if annotated else ())
    monkeypatch.setattr(host.tools, 'execute_result', execute)
    async def consume(session):
        host._held_input=host._renewal_window.total_equivalent-host._renewal_window.reserve-1200
        host._history_at_receipt=host._history_estimate()
        call=_Call(host); call.session=session; call.connected.set()
        sent=[]
        async def send(command): sent.append(command)
        session.send=send
        if failed:
            async def refuse(options):
                error=RealtimeSetupError('context_capacity','Irreducible view')
                error.source_revision=host._history_version
                raise error
            monkeypatch.setattr(host,'prepare_session',refuse)
        await call.tool_queue.put({'id':'one','type':'function','function':{'name':'list_available_tools','arguments':'{}'}})
        worker=asyncio.create_task(call.tools())
        try:
            await asyncio.wait_for(call.tool_queue.join(),30)
            canonical=next(row for row in host.messages if row.get('tool_call_id')=='one')
            assert body in canonical['content'] and effects==['list_available_tools']
            if not large:
                assert call.preparing is None and any(isinstance(c,ToolOutputs) and body in str(c.messages) for c in sent)
                return
            assert not sent and call.deferred_tool_outputs, 'never send the empty first-show view before the fresh fit'
            if failed:
                await asyncio.gather(call.preparing,return_exceptions=True)
                await call.check_preparation()
                output=next(c for c in sent if isinstance(c,ToolOutputs))
                text=next(r['content'] for r in output.messages if r.get('tool_call_id')=='one')
                assert 'RESULT_VIEW_JSON=' in text and body not in text
                view = json.loads(next(line.removeprefix('RESULT_VIEW_JSON=') for line in text.splitlines()
                                       if line.startswith('RESULT_VIEW_JSON=')))
                source = json.loads(next(line.removeprefix('FULL_RESULT_SOURCE_JSON=') for line in text.splitlines()
                                         if line.startswith('FULL_RESULT_SOURCE_JSON=')))
                assert bool(view.get('read_omitted')) is reader_loaded
                assert bool(source.get('read')) is reader_loaded
                if annotated:
                    producer = json.loads(next(line.removeprefix('PRODUCER_RESULT_SOURCE_JSON=') for line in text.splitlines()
                                               if line.startswith('PRODUCER_RESULT_SOURCE_JSON=')))
                    assert bool(producer.get('read')) is reader_loaded
                    assert 'Host annotation.' in text
                if reader_loaded:
                    ref = view['read_omitted']
                    recovered = execute_result('read_file', {key:value for key,value in ref.items() if key != 'tool'})
                    assert recovered.status == 'ok'
                    assert body[ref['start_char']:ref['start_char']+ref['max_chars']] in recovered.text
                else:
                    assert 'no reader is loaded' in text
                assert not call.deferred_tool_outputs
                assert 'one' in host._tool_delivery_fallbacks  # a range still owes the whole delivery
                assert body in canonical['content']
                monkeypatch.setattr(host, 'prepare_session', prepare)
            # Adopt successive real fitted prefixes, including recovery after a
            # range fallback. Later renewals must never inherit that wire view.
            for renewal in range(4):
                lease = await call.preparing if renewal == 0 and not failed else await prepare(LiveOptions())
                async def adopted(session):
                    rows = json.loads(session.selected.messages_json)
                    result = next(row for row in rows if row.get('tool_call_id') == 'one')
                    assert result['content'] == canonical['content']
                    assert body in result['content'] and 'RESULT_VIEW_JSON=' not in result['content']
                    assert next(row for row in host.messages if row.get('tool_call_id') == 'one') == canonical
                    assert not host._tool_delivery_fallbacks
                try:
                    await lease.run(adopted)
                finally: await lease.close()
            assert effects==['list_available_tools']
        finally:
            worker.cancel(); await asyncio.gather(worker,return_exceptions=True)
            await call.close_preparation()
    asyncio.run(host.session(consume,LiveOptions()))


@pytest.mark.parametrize('whole', [False, True])
@pytest.mark.parametrize('queued', [False, True])
def test_successful_send_retires_only_the_whole_result_fallback(host_world, whole, queued):
    import copy
    from skills.live_conversation.driver import _Call
    from ouroboros.live_context_view import _call_tool_row, record_live_exposure, drain_exposure
    from types import SimpleNamespace
    host = host_world.host
    result = {'tool_call_id': 'done', 'fn_name': 'list_available_tools', 'result': 'Exact completed body.'}
    host._tool_delivery_fallbacks = {'done': (result, {'result': 'partial'})}
    row = _call_tool_row(result, result if whole else {'result': 'partial'})
    async def run():
        call = _Call(host); call.connected.set()
        session = SimpleNamespace(held_context_rows=[], response_context_rows={})
        async def send(command):
            if not queued: session.held_context_rows.extend(command.messages)
        session.send = send; call.session = session
        await call.send(ToolOutputs((row,), ()))
        assert ('done' in host._tool_delivery_fallbacks) is (queued or not whole)
        if queued:
            # Native adapters snapshot actual delivered rows before exposing a response.
            session.response_context_rows['answer'] = [row]
            view = copy.copy(host); view._connection_owner = host; view._context_session = session
            host._connection_view = view
            record_live_exposure(view, 'answer')
            await drain_exposure(host)
        assert ('done' in host._tool_delivery_fallbacks) is not whole
    asyncio.run(run())


@pytest.mark.parametrize('thinking', [False, True])
def test_second_tool_refit_waits_for_speech_on_its_active_response(host_world, monkeypatch, record_property, thinking):
    import threading
    from ouroboros.tools.tool_result import ToolResult
    from tests._live_stack import stack

    async def scenario():
        async with stack(host_world, monkeypatch, 'gemini', thinking=thinking) as live:
            host, call, old = live.host, live.call, live.wire
            native = call.session
            complete = old.complete
            old.complete = lambda: None
            old.answer = lambda: None
            old.tool('first-result', 'list_available_tools', {})
            await live.wait(lambda: 'first-result' in old.outputs and not call.tools_running)

            entered, release = threading.Event(), threading.Event()
            batch = host.tool_batch
            body = 'Result from one completed effect. ' * 1600
            monkeypatch.setattr(host.tools, 'execute_result',
                                lambda *args: ToolResult(status='ok', code='OK', text=body))
            refit_facts = {}

            async def second_batch(calls):
                # A normal near-full serving window, as in the existing result-refit
                # tests. The production result projector decides whether to refit.
                host._held_input = host._renewal_window.total_equivalent - host._renewal_window.reserve - 1200
                host._history_at_receipt = host._history_estimate()
                host._serving_history_at_receipt = host._serving_history_estimate()
                changes = await batch(calls)
                refit_facts.update(refit=host.state.get('refit'),
                    fits=await host.delivery_fits(changes, call.serving_catalog()),
                    whole_result=any(body in str(row.get('content', '')) for row in changes))
                entered.set()
                while not release.is_set():
                    await asyncio.sleep(.01)
                return changes

            monkeypatch.setattr(host, 'tool_batch', second_batch)
            old.tool('second-result', 'list_available_tools', {})
            await live.wait(entered.is_set)
            assert refit_facts == {'refit': True, 'fits': False, 'whole_result': True}
            response = native.response
            old.say('The first result means that ', complete=False)
            await live.wait(lambda: native.text.get('ouroboros') and not call.unheard())
            assert response == native.response
            assert call.active_responses and native.turn_busy
            release.set()
            try:
                await live.wait(lambda: not call.tools_running and call.preparation_ready())
                await asyncio.sleep(.15)
                switched_before_completion = call.session is not native
                recorded = dict(thinking=thinking, switched_before_completion=switched_before_completion,
                    refit_facts=refit_facts,
                    native_complete=native.response_complete, native_busy=native.turn_busy,
                    native_output_pending=native.output_pending,
                    old_outputs=list(old.outputs), active=list(call.active_responses),
                    tool_responses=list(call.tool_responses), connections=len(live.provider.connections))
                if not switched_before_completion:
                    old.complete = complete
                    old.complete()
                    await live.wait(lambda: call.session is not native)
                await live.wait(lambda: not call.pending_offers)
                await live.quiet()
                recorded.update(effects=len([row for row in host.trace['tool_calls']
                    if row.get('tool_call_id') in ('first-result', 'second-result')]),
                    successor_outputs=list(live.wire.outputs), successor_spoken=live.wire.spoken,
                    setup=live.wire.sent[0]['setup']['sessionResumption'])
                record_property('refit_speech', json.dumps(recorded))
                assert not switched_before_completion, recorded
                assert len([row for row in host.trace['tool_calls']
                            if row.get('tool_call_id') in ('first-result', 'second-result')]) == 2
                assert 'second-result' not in old.outputs
                assert not live.wire.outputs
            finally:
                release.set()
        assert live.result[1]['execution_status'] == 'ok', live.result
    asyncio.run(scenario())
