"""Handoff guards keep answer custody, refusal meaning and old provider behavior."""
import asyncio

import pytest

from ouroboros.contracts.live_conversation import ContextUpdate
from ouroboros.contracts.realtime import RealtimeSetupError
from skills.live_conversation.driver import _Call
from tests.test_live_conversation_driver import Host, Session, until


def test_old_host_mail_answer_port_defaults_to_no_pending_delivery():
    from ouroboros.contracts.live_conversation import LiveHost
    assert LiveHost.take_mail_answer(object()) is False


def test_stack_wait_allows_slow_progress_but_still_detects_a_hang(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tests._live_stack import Stack
    async def scenario():
        run = Stack(SimpleNamespace(env=SimpleNamespace(drive_root=tmp_path)), 'openai')
        run.provider.connections.append(SimpleNamespace(received=[]))
        loop = asyncio.get_running_loop()
        clock = loop.time
        elapsed = [0]
        # Model load delaying each unit of harness progress by a second. No
        # scenario predicate is changed; the former 15-second detector fires.
        monkeypatch.setattr(loop, 'time', lambda: clock() + elapsed[0])
        def progresses():
            elapsed[0] += 1
            return elapsed[0] == 20
        try:
            await run.wait(progresses)
        finally:
            monkeypatch.setattr(loop, 'time', clock)
        with pytest.raises(AssertionError, match='timed out'):
            await run.wait(lambda: False, timeout=.01)
    asyncio.run(scenario())


def test_room_update_without_a_live_socket_records_an_offer_without_waiting():
    async def scenario():
        call = _Call(Host())
        task = asyncio.create_task(call.context_update([{'role': 'user', 'content': 'New room question'}]))
        await asyncio.sleep(.01)
        try:
            assert task.done(), 'a dead connection must not own the pending answer'
            assert len(call.pending_offers) == 1
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(scenario())


@pytest.mark.parametrize('code', ['authentication', 'rate_limit', 'provider_error', 'context_capacity', 'budget_exhausted'])
def test_prepared_typed_refusal_never_buys_a_sequential_attempt(code):
    async def scenario():
        host = Host()
        call = _Call(host)
        async def refuse():
            raise RealtimeSetupError(code, 'Synthetic prepared refusal')
        call.preparing = asyncio.create_task(refuse())
        with pytest.raises(RealtimeSetupError) as error:
            await call.connections()
        assert error.value.code == code and not host.sessions
    asyncio.run(scenario())


def test_prepared_native_events_wait_for_adoption_in_their_original_order():
    from types import SimpleNamespace
    from ouroboros.live_connections import WaitingReader
    async def scenario():
        host, session = Host(), Session()
        activation = asyncio.get_running_loop().create_future()
        reader = WaitingReader(SimpleNamespace(session=session, activate=activation, host=host))
        session.emit('caption', speaker='owner', text='Buffered native caption')
        session.emit('tool_call', call_id='not-dispatched', name='read_file', arguments='{}')
        try:
            await until(lambda: len(reader.batches) == 2)
            assert not host.captions and not host.context and not host.usage
            activation.set_result(None)
            assert (await reader.receive())[0].kind == 'caption'
            assert (await reader.receive())[0].kind == 'tool_call'
            session.emit('ready')  # the one in-flight native read is delivered, never cancelled at adoption
            assert (await reader.receive())[0].kind == 'ready'
            session.emit('input_started')
            assert (await reader.receive())[0].kind == 'input_started'
        finally:
            await reader.stop()
    asyncio.run(scenario())


@pytest.mark.parametrize('late_response', [False, True])
def test_handoff_rechecks_quiet_after_joining_canonical_rows(late_response):
    async def scenario():
        host = Host()
        host.state['greeted'] = True
        call = _Call(host)
        entered, release = asyncio.Event(), asyncio.Event()
        join = call.rows.join
        async def checkpoint():
            if call.renewing and not entered.is_set():
                entered.set()
                await release.wait()
            await join()
        call.rows.join = checkpoint
        task = asyncio.create_task(call.run())
        try:
            await until(lambda: host.sessions)
            first = host.sessions[0]
            first.emit('ready', input_rate=24000, output_rate=24000)
            await until(call.connected.is_set)
            call.request_renewal()
            await entered.wait()
            if late_response:
                first.emit('response_started', response_id='answer-after-quiet-check')
                await until(lambda: call.response_open)
            release.set()
            await asyncio.sleep(.03)
            assert len(host.sessions) == (1 if late_response else 2)
            if late_response:
                first.emit('response_done', response_id='answer-after-quiet-check', usage={}, raw_usage={},
                           status='completed', continues=False)
                await until(lambda: len(host.sessions) == 2)
                assert any(row[:2] == ('done', 'answer-after-quiet-check') for row in host.usage)
        finally:
            release.set()
            host.hub.ended_by_owner = True
            await asyncio.wait_for(task, 2)
    asyncio.run(scenario())


@pytest.mark.parametrize('status,expected', [('completed', 0), ('interrupted', 0), ('incomplete', 2)])
def test_receipt_proves_an_answer_but_a_synthetic_loss_receipt_does_not(status, expected):
    async def scenario():
        call = _Call(Host())
        call.session = Session()
        call.unproven_resumes = 2
        call.session.emit('response_done', response_id='receipt', usage={}, raw_usage={}, status=status, continues=False)
        task = asyncio.create_task(call.provider())
        try:
            await until(lambda: 'receipt' in call.receipts)
            assert call.unproven_resumes == expected
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(scenario())


@pytest.mark.parametrize('cancel_write', [False, True])
def test_live_room_update_keeps_its_answer_only_until_a_successful_write(cancel_write):
    async def scenario():
        call = _Call(Host())
        call.session = Session()
        call.connected.set()
        entered = asyncio.Event()
        original = call.session.send
        async def send(command):
            entered.set()
            if cancel_write:
                await asyncio.Future()
            await original(command)
        call.session.send = send
        task = asyncio.create_task(call.context_update([{'role': 'user', 'content': 'New room question'}]))
        await entered.wait()
        if cancel_write:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        else:
            await task
            assert len([c for c in call.session.sent if isinstance(c, ContextUpdate)]) == 1
        assert len(call.pending_offers) == int(cancel_write)
    asyncio.run(scenario())
