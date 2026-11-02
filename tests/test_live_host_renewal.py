"""New canonical rows must fit before a driver forwards them to an open route."""
import asyncio
import queue
from dataclasses import replace

from ouroboros.contracts.live_conversation import LiveOptions
from tests.test_live_conversation import host_world as host_world


def test_pending_context_counts_before_and_after_receipts(host_world):
    host = host_world.host
    async def run():
        for producer in ('mailbox', 'peer', 'tool'):
            async def consume(session):
                # Leave ample room for this connection's initial selection, then
                # tighten the test window to 1,000 tokens of further growth.
                held = host._selected_raw_estimate
                host.model = replace(host.model, facts=replace(host.model.facts,
                    conversation_window_tokens=held + 11000, working_reserve_tokens=10000))
                assert host.renewal_due() is False
                if producer != 'peer':
                    host.response_done(producer, usage={'text_input_tokens': held}, raw_usage={}, status='completed')
                    assert host.renewal_due() is False
                async def add(text, ident):
                    if producer == 'mailbox':
                        host.incoming = queue.Queue()
                        host.incoming.put({'text': text, 'msg_id': ident})
                        assert await host.mailbox()
                    elif producer == 'peer':
                        await host.observe_context({'content': text})
                    else:
                        async def result(port, arguments):
                            return text
                        host.call_tools['set_test_language'] = replace(host.call_tools['set_test_language'], handler=result)
                        await host.tool_batch([{'id': ident, 'type': 'function',
                            'function': {'name': 'set_test_language', 'arguments': '{}'}}])
                await add('Fits inside the window.', producer + '-small')
                assert host.renewal_due() is False
                big = 'Pending canonical context. ' * 1200
                await add(big, producer + '-big')
                assert any(big in str(row.get('content', '')) for row in host.messages)
                assert host.renewal_due() is True
                # Renewal is a decision, not a mutation or silent context cut.
                assert any(big in str(row.get('content', '')) for row in host.messages)
            await host.session(consume, LiveOptions())
            host.model = replace(host.model, facts=replace(host.model.facts, conversation_window_tokens=200000))
    asyncio.run(run())


def test_idle_renewal_reuses_estimate_and_every_context_writer_invalidates(host_world, monkeypatch):
    from ouroboros import context_fit, loop
    from ouroboros.realtime_setup import session_candidate
    host = host_world.host
    estimate = context_fit.estimate_context_prompt_tokens
    measured = []
    def count(*args, **kwargs):
        measured.append(1)
        return estimate(*args, **kwargs)
    async def consume(session):
        monkeypatch.setattr(context_fit, 'estimate_context_prompt_tokens', count)
        for _ in range(1000):
            assert host.renewal_due() is False
        assert measured == [], 'idle audio polls re-estimated unchanged history'

        async def changed(action):
            before = host._history_estimate()
            await action()
            measured.clear()  # tool fitting may measure its own physical candidate
            after = host._history_estimate()
            assert after > before
            assert measured == [1]
            for _ in range(1000):
                assert host.renewal_due() is False
            assert measured == [1]

        async def mailbox():
            host.incoming = queue.Queue()
            host.incoming.put({'text': 'New owner context. ' * 20, 'msg_id': 'cache-mail'})
            await host.mailbox()
        async def enable():
            await host.tool_batch([{'id': 'cache-enable', 'type': 'function', 'function': {
                'name': 'enable_tools', 'arguments': '{"tools":"read_file"}'}}])
            assert 'read_file' in {schema['function']['name'] for schema in host.schemas}
        async def edit():
            def in_place(calls, tools, logs, task_id, executor, messages, trace, emit, **kwargs):
                messages[1]['content'] = 'Changed an existing row. ' * 100
            monkeypatch.setattr(loop, 'handle_tool_calls', in_place)
            await host.tool_batch([{'id': 'cache-edit', 'type': 'function',
                'function': {'name': 'read_file', 'arguments': '{}'}}])
        async def instructions():
            session_candidate(host, host_world.provider['session'], LiveOptions(instructions='New instructions. ' * 100))
        for action in (mailbox,
                       lambda: host.observe_context({'content': 'New peer context. ' * 20}),
                       lambda: host.caption('owner', 'New owner speech. ' * 20, 'cache-owner'),
                       lambda: host.caption('ouroboros', 'New reply. ' * 20, 'cache-reply', completed=True),
                       enable, edit, instructions):
            await changed(action)
    asyncio.run(host.session(consume, LiveOptions()))
