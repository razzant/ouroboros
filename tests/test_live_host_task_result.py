"""A task started by the live voice must return through ordinary chat delivery."""
import asyncio
import json
import queue
import time
from types import SimpleNamespace

from tests.test_live_conversation import host_world as host_world


def test_started_task_result_reaches_live_call_but_other_room_does_not(host_world, monkeypatch):
    from ouroboros import agent_task_pipeline
    from ouroboros.contracts.live_conversation import LiveOptions
    from ouroboros.projects_registry import get_project
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.utils import append_jsonl
    from supervisor import active_activity, message_bus, workers
    from supervisor.events import _handle_send_message, _handle_task_done
    from tests._dialogue import chat_rows

    state, host = host_world, host_world.host
    root, chat_id = state.env.drive_root, state.ctx.current_chat_id
    result_text = 'The call-started task found the synthetic answer: 42.'
    foreign_text = 'Only the unrelated beta room should receive this result.'
    started = []

    class WorkerEvents(queue.Queue):
        def put_nowait(self, event):
            if event['type'] == 'promote_chat_to_task':
                # Fake worker admission only; tool discovery/emission and the
                # matching-token receipt read are real. No result mail is seeded.
                task = dict(id=event['task_id'], type='task', text=event['objective'],
                    chat_id=event['chat_id'], root_task_id=event['task_id'],
                    project_id=event['project_id'],
                    delegation_role='root', objective_author=event['objective_author'],
                    origin_message_ref=event['source_ref'])
                started.append(task)
                write_task_result(root, task['id'], 'scheduled', **{
                    key: value for key, value in task.items() if key != 'id'},
                    promotion_admission={'status': 'scheduled', 'routing_token': event['routing_token']})
            super().put_nowait(event)

    events = WorkerEvents()
    host.incoming, host.event_queue = queue.Queue(), events
    state.ctx.event_queue = events
    state.ctx.task_metadata['origin_message_ref'] = state.task['origin_message_ref']
    actor = SimpleNamespace(_realtime_turn=host.turn, _busy=True, _accepting_owner_messages=True,
        _current_task_id=state.ctx.task_id, _current_chat_id=chat_id,
        _current_task_metadata=state.ctx.task_metadata, _incoming_messages=host.incoming,
        _event_queue=events, tools=host.tools)
    registry = active_activity.DirectActivityRegistry()
    monkeypatch.setattr(active_activity, '_DIRECT_ACTIVITY_REGISTRY', registry)
    registry.register(state.ctx.task_id, chat_id, actor=actor,
                      origin_message_ref=state.task['origin_message_ref'])
    assert workers.direct_chat_turn(state.ctx.task_id)['chat_id'] == chat_id
    # Optional paid post-task processing is outside this result-delivery proof.
    monkeypatch.setattr(agent_task_pipeline, '_run_post_task_processing_async', lambda *a, **kw: None)
    supervisor = SimpleNamespace(DRIVE_ROOT=root, RUNNING={}, PENDING=[], WORKERS={}, EVENT_Q=events,
        bridge=message_bus.get_bridge(), send_with_budget=message_bus.send_with_budget,
        append_jsonl=append_jsonl, persist_queue_snapshot=lambda **kw: None)

    def finish_task(task, text):
        pending = []
        # The ordinary text producer constructs the final send_message and
        # task_done, persists the result and publishes the answer on the live bus.
        agent_task_pipeline.emit_task_results(state.env, None, None, pending, task, text,
            {'rounds': 1}, {'tool_calls': [], 'reasoning_notes': []}, time.time(), root / 'logs',
            event_queue=events)
        live = []
        while not events.empty():
            live.append(events.get_nowait())
        assert any(event['type'] == 'send_message' and event['text'] == text for event in live)
        # Both live and buffered copies traverse the production dedupe/room route.
        for event in live + pending:
            if event['type'] == 'send_message':
                assert _handle_send_message(event, supervisor) is not False
            elif event['type'] == 'task_done':
                _handle_task_done(event, supervisor)
        stored = load_task_result(root, task['id'])
        assert stored['status'] == 'completed' and stored['result'] == text
        delivered = [row for row in chat_rows(root) if row.get('text') == text]
        assert len(delivered) == 1 and delivered[0]['chat_id'] == task['chat_id']

    async def invoke(name, arguments):
        return await host.tool_batch([{'id': 'call-' + name, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments)}}])

    async def converse(session):
        assert session.opened and not session.closed and not host.hub.closed
        assert session.selected.mode == 'nano'
        await invoke('enable_tools', {'tools': 'start_task'})
        assert 'start_task' in {schema['function']['name'] for schema in host.schemas}
        receipt = await invoke('start_task', {'objective': 'Find the synthetic answer while we talk.',
                                            'title': 'Call research', 'project_id': 'alpha', 'workspace': 'none'})
        assert 'accepted and durably scheduled' in str(receipt)
        [task] = started
        assert task['chat_id'] == chat_id
        assert task['objective_author'] == {'kind': 'task', 'task_id': state.ctx.task_id}
        assert task['origin_message_ref'] == state.task['origin_message_ref']
        assert await host.mailbox() == []

        other_chat = get_project(root, 'beta')['chat_id']
        assert other_chat != chat_id
        finish_task(dict(id='unrelated-room-task', type='task', text='Other room work',
                         root_task_id='unrelated-room-task', project_id='beta', chat_id=other_chat), foreign_text)
        assert await host.mailbox() == [], 'another room leaked into the live call'
        assert foreign_text not in str(host.messages)

        finish_task(task, result_text)
        rows = await host.mailbox()
        assert session.opened and not session.closed and not host.hub.closed
        assert foreign_text not in str(rows) and foreign_text not in str(host.messages)
        matching = [row for row in rows if result_text in str(row.get('content'))]
        [row] = matching
        assert row in host.messages and row['role'] == 'user'
        assert task['id'] in row['content'] and '[System task message]' in row['content']
        # A reply from another task is addressed context, never owner authority.
        assert row.get('_host_context_kind') != 'owner_words'
        assert result_text not in str(getattr(state.ctx, '_owner_directives', []))
        assert await host.mailbox() == []

    asyncio.run(host.session(converse, LiveOptions()))
