"""R9 review boundaries; disposable roots and synthetic provider ports only."""
import queue
import threading
from types import SimpleNamespace

import pytest

from tests.test_delivery_contention import sender as sender, owed
from tests.test_live_conversation import host_world as host_world


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


@pytest.mark.parametrize('ended', ['no', 'admission_closed', 'settled'])
def test_relay_checks_admission_and_settlement_and_follows_chat_write(tmp_path, monkeypatch, sender, ended):
    from ouroboros import owner_mailbox as mail
    from ouroboros.task_results import write_task_result
    from supervisor import events_chat_delivery as delivery
    from tests._dialogue import chat_rows
    ctx, frames, _ = sender
    target, _ = actor(monkeypatch, tmp_path)
    event = owed(tmp_path, text='A finished task result.', tid='source')
    if ended == 'admission_closed':
        with target._owner_message_admission_lock:
            target._accepting_owner_messages = False
    elif ended == 'settled':
        write_task_result(tmp_path, 'recipient', 'completed', result='Already finished.')
    write = mail.write_task_message
    observations = []
    def inspect_write(root, text, task_id, **kwargs):
        assert target._owner_message_admission_lock.locked()
        assert len(frames) == 1
        assert [row['text'] for row in chat_rows(root)] == [event['text']]
        observations.append(task_id)
        return write(root, text, task_id, **kwargs)
    monkeypatch.setattr(mail, 'write_task_message', inspect_write)
    assert delivery._handle_send_message(event, ctx)
    assert len(mail.drain_owner_entries(tmp_path, 'recipient')) == int(ended == 'no')
    assert observations == (['recipient'] if ended == 'no' else [])


def test_relay_rechecks_after_waiting_for_admission_lock(tmp_path, monkeypatch, sender):
    from ouroboros import owner_mailbox as mail
    from supervisor import events_chat_delivery as delivery
    ctx, _, _ = sender
    target, _ = actor(monkeypatch, tmp_path)
    event = owed(tmp_path, text='Result races a closing turn.', tid='source')
    entered = threading.Event()
    class Gate:
        def __enter__(self):
            entered.set()
            target_lock.acquire()
        def __exit__(self, *args):
            target_lock.release()
    target_lock = target._owner_message_admission_lock
    target._owner_message_admission_lock = Gate()
    outcome = []
    target_lock.acquire()
    worker = threading.Thread(target=lambda: outcome.append(delivery._handle_send_message(event, ctx)))
    worker.start()
    try:
        assert entered.wait(5)
        target._accepting_owner_messages = False
    finally:
        target_lock.release()
        worker.join(5)
    assert not worker.is_alive() and outcome == [True]
    assert not mail.drain_owner_entries(tmp_path, 'recipient')


def test_mailbox_retry_uses_durable_send_facts_and_keeps_owner_rows_separate(tmp_path, monkeypatch, sender):
    from ouroboros import owner_mailbox as mail, working_checkpoint
    from ouroboros.context_budget import OWNER_WORDS_KIND
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from supervisor import events_chat_delivery as delivery, terminal_delivery as td
    from tests._budget_pause_exact_helpers import _loop_ctx
    from tests._dialogue import chat_rows
    ctx, frames, bus = sender
    target, registry = actor(monkeypatch, tmp_path)
    other = SimpleNamespace(**vars(target))
    other._current_task_id, other._current_chat_id = 'other-room', 2
    other._owner_message_admission_lock = threading.Lock()
    registry.register('other-room', 2, actor=other)
    event = owed(tmp_path, text='Durably sent task answer.', tid='source')
    write = mail.write_task_message
    monkeypatch.setattr(mail, 'write_task_message', lambda *a, **kw: False)
    assert not delivery._handle_send_message(event, ctx)
    assert len(frames) == 1
    [pending] = td.pending_deliveries(tmp_path)
    assert pending['emitted']['chat_id'] == 1 and pending['emitted']['text'] == event['text']
    delivery._DELIVERED_MESSAGE_IDS.clear()
    delivery._DEFERRED_DELIVERIES.clear()
    monkeypatch.setattr(mail, 'write_task_message', write)
    assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == [event['delivery_id']]
    assert len(frames) == 1 and len(chat_rows(tmp_path)) == 1
    assert td.pending_deliveries(tmp_path) == []
    assert not mail.drain_owner_entries(tmp_path, 'other-room')
    [entry] = mail.drain_owner_entries(tmp_path, 'recipient')
    owner_ctx, limit = _loop_ctx(tmp_path, 'recipient', direct=True)
    owner_row = {'role': 'user', '_host_context_kind': OWNER_WORDS_KIND, 'content': 'The owner instruction.'}
    limit.messages, limit.owner_msg_seen = [dict(owner_row)], set()
    injected = queue.Queue()
    _drain_incoming_messages(limit.messages, queue.Queue(), tmp_path, 'recipient', injected,
                             limit.owner_msg_seen, owner_ctx=owner_ctx, defer_content_ack=True)
    assert limit.messages[0] == owner_row and len(limit.messages) == 2
    assert limit.messages[1]['content'].startswith('[System task message]\nTask source in this room has ended; its terminal message:')
    assert '_host_context_kind' not in limit.messages[1]
    assert not owner_ctx._owner_directives and not mail.mail_read_state(tmp_path, 'recipient', entry['msg_id'])
    working_checkpoint.save_ready(limit)
    assert mail.mail_read_state(tmp_path, 'recipient', entry['msg_id'])


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


@pytest.mark.parametrize('placement', ['during_send', 'before_handler', 'after_send', 'main'])
@pytest.mark.parametrize('retry_mailbox', [False, True])
def test_result_relay_uses_the_room_that_actually_received_the_chat_row(tmp_path, monkeypatch, sender,
                                                                     placement, retry_mailbox):
    from ouroboros import owner_mailbox as mail
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.task_results import write_task_result
    from supervisor import events_chat_delivery as delivery, terminal_delivery as td
    from tests._dialogue import chat_rows
    ctx, frames, bus = sender
    alpha = create_project(tmp_path, 'alpha')['chat_id']
    target, registry = actor(monkeypatch, tmp_path, 'main-recipient', chat_id=1)
    project_target = SimpleNamespace(**vars(target))
    project_target._current_task_id, project_target._current_chat_id = 'project-recipient', alpha
    project_target._owner_message_admission_lock = threading.Lock()
    registry.register('project-recipient', alpha, actor=project_target)
    write_task_result(tmp_path, 'project-recipient', 'running', chat_id=alpha)
    write_task_result(tmp_path, 'source', 'running', chat_id=1)
    event = owed(tmp_path, text='The task answer belongs in its newly selected project.', tid='source')
    send = ctx.send_with_budget
    def place():
        bind_task_to_project(tmp_path, 'source', 'alpha', alpha, origin={'absent': 'system'})
    if placement == 'before_handler':
        place()
    def owner_places_task_then_send(*args, **kwargs):
        # Supported "Turn into project" operation, interleaved after the outer
        # route read and before message_bus's authoritative send-time route read.
        if placement == 'during_send':
            place()
        facts = send(*args, **kwargs)
        if placement == 'after_send':
            place()
        return facts
    monkeypatch.setattr(ctx, 'send_with_budget', owner_places_task_then_send)
    room = alpha if placement in {'during_send', 'before_handler'} else 1
    write = mail.write_task_message
    if retry_mailbox:
        monkeypatch.setattr(mail, 'write_task_message', lambda *a, **kw: False)
        assert not delivery._handle_send_message(event, ctx)
        [pending] = td.pending_deliveries(tmp_path)
        assert pending['emitted']['chat_id'] == room
        delivery._DELIVERED_MESSAGE_IDS.clear()
        delivery._DEFERRED_DELIVERIES.clear()
        monkeypatch.setattr(mail, 'write_task_message', write)
        assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == [event['delivery_id']]
    else:
        assert delivery._handle_send_message(event, ctx)
    assert td.pending_deliveries(tmp_path) == []
    [row] = chat_rows(tmp_path)
    assert len(frames) == 1 and row['chat_id'] == room == frames[0]['chat_id']
    actual = {tid: [item['text'] for item in mail.drain_owner_entries(tmp_path, tid)]
              for tid in ['main-recipient', 'project-recipient']}
    expected = {'main-recipient': [], 'project-recipient': []}
    expected['project-recipient' if room == alpha else 'main-recipient'] = [
        'Task source in this room has ended; its terminal message:\n' + event['text']]
    assert actual == expected
