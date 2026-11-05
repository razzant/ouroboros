"""R9 review boundaries; disposable roots and synthetic provider ports only."""
import queue
import threading
from types import SimpleNamespace

import pytest

from tests.test_delivery_contention import sender as sender
from tests._live_world import host_world as host_world


def actor(monkeypatch, root, task_id='recipient', *, chat_id=1):
    from ouroboros.task_results import write_task_result
    from supervisor import active_activity
    registry = active_activity.DirectActivityRegistry()
    monkeypatch.setattr(active_activity, '_DIRECT_ACTIVITY_REGISTRY', registry)
    value = SimpleNamespace(_busy=True, _accepting_owner_messages=True,
        _current_task_id=task_id, _current_chat_id=chat_id, _current_task_metadata={},
        _owner_message_admission_lock=threading.Lock())
    registry.register(task_id, chat_id, actor=value)
    write_task_result(root, task_id, 'running', chat_id=chat_id)
    return value, registry


@pytest.mark.parametrize('ending', ['returned', 'failed', 'setup_failed'])
def test_call_that_returned_cannot_accept_result_mail(host_world, monkeypatch, ending):
    from ouroboros import call_peers, owner_mailbox as mail
    from ouroboros.contracts.realtime import RealtimeSetupError
    from ouroboros.live_conversation import ConversationHost
    from supervisor import events_chat_delivery as delivery, workers
    state = host_world
    root, tid, room = state.env.drive_root, state.ctx.task_id, state.ctx.current_chat_id
    target, _ = actor(monkeypatch, root, tid, chat_id=room)
    target._realtime_turn = state.host.turn
    monkeypatch.setattr(call_peers, 'get_hub', lambda _: state.host.hub)
    monkeypatch.setattr(call_peers, 'close_hub', lambda _id, reason, **kwargs: state.host.hub.close(reason))
    finish = ConversationHost.finish
    async def finish_after_admission_closes(port):
        # Tool settlement can outlive the receive loop, before run() returns.
        assert target._busy and not target._accepting_owner_messages
        assert workers.direct_chat_turn(tid) is None
        delivery._relay_room_result(root, 'final:source:during-finish', {
            'task_id': 'source', 'chat_id': room, 'text': 'Arrived during call settlement.'})
        assert mail.drain_owner_entries(root, tid) == []
        await finish(port)
    monkeypatch.setattr(ConversationHost, 'finish', finish_after_admission_closes)
    if ending == 'failed':
        async def fail(port):
            raise RealtimeSetupError('fixture_failure', 'Synthetic call failure.')
        monkeypatch.setitem(state.feature, 'handler', fail)
    elif ending == 'setup_failed':
        def fail_warm(port):
            raise RealtimeSetupError('fixture_setup_failure', 'Synthetic setup failure.')
        monkeypatch.setattr(ConversationHost, 'warm', fail_warm)
    # Drive real teardown on normal return and failure, while the agent has not
    # yet stored its terminal result, exactly as on run() return.
    state.host.turn.run(state.ctx, state.host.messages, state.host.tools,
                        task=state.task, initial_effort='', incoming_messages=queue.Queue())
    assert state.host.turn._closed and state.host.hub.closed
    assert target._busy and not target._accepting_owner_messages
    assert workers.direct_chat_turn(tid) is None
    assert not mail.mailbox_drain_ended(root, tid)
    delivery._relay_room_result(root, 'final:source:synthetic', {
        'task_id': 'source', 'chat_id': room, 'text': 'Arrived while the call result was being stored.'})
    assert mail.drain_owner_entries(root, tid) == []
