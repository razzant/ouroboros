"""Completed, interrupted and closing live receipts use the ordinary ledger."""
import asyncio
import json
from dataclasses import replace

import pytest

from ouroboros import usage_accounting as ua
from ouroboros.contracts.live_conversation import LiveOptions
from ouroboros.contracts.realtime import RealtimeSetupError
from tests.test_live_conversation import host_world as host_world
from tests.test_live_usage import token_usage


def tariff(**entry):
    return {'entry': entry, 'fetched_at': '2026-10-21T00:00:00+00:00', 'sha256': 'fixture-digest',
            'catalog_id': 'fixture', 'provider': 'fixture_live'}


def test_response_dedup_pinned_quote_and_shared_limit(host_world):
    host = host_world.host
    host.quote = tariff(input_cost_per_token=.001, output_cost_per_token=.002)
    with ua.usage_scope(ua.UsageScope(drive_root=host.ctx.drive_root, task_id=host.ctx.task_id, global_limit_usd=.01)):
        host.response_started('spoken')
        host.response_started('spoken')
        host.quote['entry']['input_cost_per_token'] = 100  # a refresh cannot reprice an open response
        for _ in range(2):
            host.response_done('spoken', usage=token_usage(text_input_tokens=12), raw_usage={'input_tokens': 12},
                               status='interrupted')
        row, = ua.read_usage_records(host.ctx.drive_root)
        assert row['cost_usd'] == pytest.approx(.012)
        assert row['cost_evidence']['status'] == 'interrupted'
        with pytest.raises(RealtimeSetupError, match='budget_exhausted'):
            host.response_started('next')
        host.response_started('next')  # duplicate notification creates no second refusal row
        for _ in range(2):
            host.response_done('next', usage=token_usage(text_input_tokens=1), raw_usage={}, status='interrupted')
    rows = ua.read_usage_records(host.ctx.drive_root)
    assert len(rows) == 2 and sum(row['cost_usd'] for row in rows) == pytest.approx(100.012)


@pytest.mark.parametrize('ending', ['normal', 'stop', 'budget', 'close_error'])
def test_connected_samples_share_watermark_and_settle_before_close_ends(host_world, ending):
    host = host_world.host
    host.quote = tariff(input_cost_per_second=.01)
    def sample(session, seconds):
        session.observe('usage', json.dumps({'usage': {'connected_seconds': seconds},
            'raw_usage': {'billable_audio_seconds': 999}}))
    async def consume(session):
        for _ in range(2):
            sample(session, 4)
        host.response_started('same-id')
        for _ in range(2):
            host.response_done('same-id', usage={'connected_seconds': 7},
                               raw_usage={'billable_audio_seconds': 999}, status='interrupted')
        async def close():
            assert host._connection_accounting
            for _ in range(2):
                sample(session, 9)
            session.closed = True
            if ending == 'close_error':
                raise RuntimeError('fixture close failed')
        session.close = close
        if ending == 'stop':
            raise asyncio.CancelledError()
        if ending == 'budget':
            host.response_started('over-limit')
    async def run():
        with ua.usage_scope(ua.UsageScope(drive_root=host.ctx.drive_root, task_id=host.ctx.task_id,
                                         global_limit_usd=.06 if ending == 'budget' else None)):
            await host.session(consume, LiveOptions())
    expected = {'stop': (asyncio.CancelledError, None), 'budget': (RealtimeSetupError, 'budget_exhausted'),
                'close_error': (RuntimeError, 'fixture close failed')}
    if ending in expected:
        error, message = expected[ending]
        with pytest.raises(error, match=message):
            asyncio.run(run())
    else:
        asyncio.run(run())
    rows = ua.read_usage_records(host.ctx.drive_root)
    settled = [row for row in rows if row.get('cost_usd') is not None]
    assert [row['cost_usd'] for row in settled] == pytest.approx([.04, .03, .02])
    assert [row['cost_evidence']['usage']['connected_seconds'] for row in settled] == [4, 3, 2]
    assert [row['source'] for row in settled] == ['live_connection_usage', 'live_response', 'live_connection_usage']
    assert all('response_key' not in row['cost_evidence'] for row in (settled[0], settled[2]))
    assert all(row['cost_evidence']['raw_usage']['billable_audio_seconds'] == 999 for row in settled)
    assert host._connection_accounting is False and host_world.sessions[-1].closed

    # A new response belongs to this call, while each physical connection has
    # its own watermark. Its unknown quote stays unknown.
    host.quote = None
    async def next_connection(session):
        host.response_done('next-id', usage={'connected_seconds': 2}, raw_usage={}, status='completed')
        sample(session, 2)
        sample(session, 3)
    asyncio.run(host.session(next_connection, LiveOptions()))
    added = ua.read_usage_records(host.ctx.drive_root)[len(rows):]
    assert len(added) == 2 and all(row['cost_usd'] is None for row in added)
    assert [row['cost_evidence']['usage']['connected_seconds'] for row in added] == [2, 1]


def test_cancel_joins_started_tool_before_connection_closes(host_world):
    host = host_world.host
    checkpoints = []
    original = host.checkpoint
    def checkpoint(boundary):
        original(boundary)
        checkpoints.append(boundary)
    host.checkpoint = checkpoint

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        async def effect(host, arguments):
            entered.set()
            await release.wait()
            return 'Effect finished'
        host.call_tools['set_test_language'] = replace(host.call_tools['set_test_language'], handler=effect)
        call = {'id': 'slow-effect', 'type': 'function',
                'function': {'name': 'set_test_language', 'arguments': '{}'}}
        async def consume(session):
            await host.tool_batch([call])
        running = asyncio.create_task(host.session(consume, LiveOptions()))
        await entered.wait()
        running.cancel()
        await asyncio.sleep(0)
        assert not host_world.sessions[-1].closed and not running.done()
        running.cancel()  # repeated Stop still joins the same effect
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert host_world.sessions[-1].closed
        assert any(row.get('tool_call_id') == 'slow-effect' for row in host.messages)
        assert checkpoints == ['pre_effect', 'post_batch']
    asyncio.run(run())


def test_presence_caption_waits_for_store_and_keeps_participant_identity(host_world, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from ouroboros import dialogue_store
    from ouroboros.presence_call import record_call_utterance

    host = host_world.host
    task = {'id': 'presence-call', 'chat_id': host.turn.spec.chat_id, 'metadata': {'presence': {
        'binding_id': 'fixture-binding', 'transport_skill': 'fixture-transport', 'event': {
            'source_event_id': 'fixture-meeting', 'provider': 'fixture-meeting', 'account_id': 'fixture-account',
            'conversation_id': 'fixture-room', 'thread_id': '', 'conversation_key': 'fixture:meeting:room',
            'actor': {'display_name': 'Opening participant', 'platform_actor_id': 'opening-participant'}}}}}
    append, entered = dialogue_store.append, threading.Event()
    def signalled(*args, **kwargs):
        entered.set()
        return append(*args, **kwargs)
    monkeypatch.setattr(dialogue_store, 'append', signalled)
    root = host.ctx.drive_root
    # Initialize the real store before holding another writer's transaction.
    append(root, 'chat', {'direction': 'out', 'chat_id': task['chat_id'], 'text': 'Fixture setup'})
    with ThreadPoolExecutor(max_workers=1) as executor:
        with dialogue_store.transaction(root, write=True):
            pending = executor.submit(record_call_utterance, root, task, direction='in',
                text='Participant under contention', speaker_label='Meeting participant', utterance_id='heard-1')
            assert entered.wait(5)
            assert not pending.done()
        assert pending.result(timeout=5)
    assert record_call_utterance(root, task, direction='in', text='Participant under contention',
                                speaker_label='Meeting participant', utterance_id='heard-1')
    rows = [entry.row for entry in dialogue_store.iter_rows(root, 'chat') if entry.row.get('task_id') == task['id']]
    assert len(rows) == 1
    assert rows[0]['sender_label'] == 'Meeting participant'
    assert rows[0]['source'] == 'presence:fixture-meeting'
    assert rows[0]['client_message_id'] == 'fixture-meeting:heard:heard-1'


@pytest.fixture
def abandonments(monkeypatch):
    from ouroboros import live_usage
    records = []
    abandon = live_usage.abandon_live_response
    def record(attempt, *, reason):
        records.append((attempt._evidence['response_key'], reason))
        abandon(attempt, reason=reason)
    monkeypatch.setattr(live_usage, 'abandon_live_response', record)
    return records


def test_response_reservation_survives_resumed_connection(host_world, abandonments):
    host = host_world.host
    host.quote = tariff(input_cost_per_token=.001, output_cost_per_token=.002)
    first = {}
    async def begin(session):
        host.response_started('resumed-answer')
        first['row'], = ua.read_usage_records(host.ctx.drive_root)
        first['text_estimate'] = host._selected_raw_estimate  # complete text-only envelope
        return 'fixture-resume-handle'
    async def complete(session):
        assert session.options.resume_handle == 'fixture-resume-handle'
        host.response_started('resumed-answer')  # a replayed start keeps its original reservation
        assert len(ua.read_usage_records(host.ctx.drive_root)) == 1
        assert host._response_text_estimates['resumed-answer'] == first['text_estimate']
        host.response_done('resumed-answer', usage=token_usage(text_input_tokens=12),
                           raw_usage={'input_tokens': 12}, status='completed')
    async def run():
        handle = await host.session(begin, LiveOptions())
        pending, = ua.read_usage_records(host.ctx.drive_root)
        assert pending['state'] == 'dispatched' and not abandonments
        host.quote = tariff(input_cost_per_token=100, output_cost_per_token=200)
        await host.observe_context({'content': 'A new connection refits this later context.'})
        await host.session(complete, LiveOptions(resume_handle=handle))
        await host.finish()
    asyncio.run(run())
    row, = ua.read_usage_records(host.ctx.drive_root)
    assert row['attempt_id'] == first['row']['attempt_id']
    assert row['physical_context'] == first['row']['physical_context']
    assert row['cost_evidence']['response_key'] == 'resumed-answer'
    assert row['state'] == 'settled' and row['cost_usd'] == pytest.approx(.012)
    assert not abandonments and not host.responses
    assert host.connection_number == 2 and all(session.closed for session in host_world.sessions if session.opened)


def test_unfinished_response_is_abandoned_once_at_call_end(host_world, abandonments):
    host = host_world.host
    async def consume(session):
        host.response_started('unfinished')
    async def run():
        await host.session(consume, LiveOptions())
        row, = ua.read_usage_records(host.ctx.drive_root)
        assert row['state'] == 'dispatched' and not abandonments
        await host.finish()
        await host.finish()
    asyncio.run(run())
    row, = ua.read_usage_records(host.ctx.drive_root)
    assert row['state'] == 'unresolved' and row['reason'] == 'call_ended_without_receipt'
    assert abandonments == [('unfinished', 'call_ended_without_receipt')]
    assert not host.responses and not host._response_text_estimates


@pytest.mark.parametrize('failure_stage', ['factory', 'open'])
def test_failed_reopen_abandons_response_during_call_teardown(host_world, abandonments, monkeypatch, failure_stage):
    from ouroboros import call_peers
    from ouroboros.live_conversation import ConversationHost
    host = host_world.host
    observed = {}
    factory = host_world.provider['session']
    def reopen(*args, **kwargs):
        options = args[1]
        if options.resume_handle and failure_stage == 'factory':
            raise RealtimeSetupError('resume_failed', 'fixture failed reopen')
        session = factory(*args, **kwargs)
        if options.resume_handle:
            async def failed_open():
                raise RealtimeSetupError('resume_failed', 'fixture failed reopen')
            session.open = failed_open
        return session
    async def handler(port):
        observed['port'] = port
        async def begin(session):
            port.response_started('unfinished-resume')
            return 'fixture-resume-handle'
        async def unreachable(session):
            pytest.fail('a failed reopen cannot reach its consumer')
        handle = await port.session(begin, LiveOptions())
        observed['pending'], = ua.read_usage_records(port.ctx.drive_root)
        await port.session(unreachable, LiveOptions(resume_handle=handle))
    monkeypatch.setitem(host_world.provider, 'session', reopen)
    monkeypatch.setitem(host.turn.feature, 'handler', handler)
    monkeypatch.setattr(call_peers, 'get_hub', lambda call_id: host.hub)
    monkeypatch.setattr(ConversationHost, 'warm', lambda self: None)
    result = host.turn.run(host.ctx, list(host.messages), host.tools, task=host.task, initial_effort='')
    assert result[1]['reason_code'] == 'resume_failed'
    assert observed['pending']['state'] == 'dispatched'
    row, = ua.read_usage_records(host.ctx.drive_root)
    assert row['state'] == 'unresolved' and row['reason'] == 'call_ended_without_receipt'
    assert abandonments == [('unfinished-resume', 'call_ended_without_receipt')]
    assert not observed['port'].responses
    assert all(session.closed for session in host_world.sessions if session.opened)
    if failure_stage == 'open':
        assert host_world.sessions[-1].closed  # failed open owns cleanup too


@pytest.mark.parametrize('resumed', [False, True])
def test_extended_thinking_idle_receipt_is_not_charged_again(host_world, abandonments, resumed):
    from ouroboros.contracts.live_conversation import LiveEvent
    host = host_world.host
    host.quote = tariff(input_cost_per_token=.001, output_cost_per_token=.002)
    progress = LiveEvent('response_done', {
        'response_id': 'thinking-answer', 'status': 'completed', 'continues': True,
        'usage': token_usage(text_input_tokens=12, text_output_tokens=2, thinking_tokens=3),
        'raw_usage': {'usageMetadata': {
            'promptTokensDetails': [{'modality': 'TEXT', 'tokenCount': 12}],
            'responseTokensDetails': [{'modality': 'TEXT', 'tokenCount': 2}],
            'thoughtsTokenCount': 3}},
    })
    idle = replace(progress, data={**progress.data, 'continues': False})
    first = {}
    def forward(event):
        # The feature handles continues as liveness and forwards this accounting
        # payload through LiveHost's public port without manufacturing a receipt.
        host.response_done(event.data['response_id'], usage=event.data['usage'],
                           raw_usage=event.data['raw_usage'], status=event.data['status'])
    async def begin(session):
        host.response_started(progress.data['response_id'])
        forward(progress)
        first['rows'] = ua.read_usage_records(host.ctx.drive_root)
        return 'fixture-resume-handle'
    async def transition(session):
        for _ in range(2):
            forward(idle)
        host.response_started(idle.data['response_id'])  # a replayed start stays deduplicated too
        assert ua.read_usage_records(host.ctx.drive_root) == first['rows']
        assert not abandonments and not host.responses
        # Equal usage on a genuinely new native id is another billable response.
        following = replace(idle, data={**idle.data, 'response_id': 'next-thinking-answer'})
        host.response_started(following.data['response_id'])
        forward(following)
    async def same_connection(session):
        await begin(session)
        await transition(session)
    async def run():
        if resumed:
            handle = await host.session(begin, LiveOptions())
            await host.session(transition, LiveOptions(resume_handle=handle))
        else:
            await host.session(same_connection, LiveOptions())
        await host.finish()
    asyncio.run(run())
    rows = ua.read_usage_records(host.ctx.drive_root)
    assert [row['cost_evidence']['response_key'] for row in rows] == ['thinking-answer', 'next-thinking-answer']
    assert [row['cost_usd'] for row in rows] == pytest.approx([.022, .022])
    assert all(row['state'] == 'settled' for row in rows) and not abandonments
