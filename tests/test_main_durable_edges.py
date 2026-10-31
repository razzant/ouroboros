"""Durable edges of Main's owners through the actual passive gateway: rolled-back SQL appends, kept
timestamps, same-batch ask and close, legacy absence, child waits, closure wakes, non-clean finals."""
import json
from tests import _dialogue as dialogue_test
import os
import asyncio
from collections import deque
from types import SimpleNamespace

import pytest

from tests.test_main_terminal_delivery import owed as _owed, _due
from tests.test_quiz_publication_delivery import publication as _publication, _block, _due as _quiz_due
from tests.test_main_state_projection import gateway as _gateway, _project_task, _state
from tests.test_quiz_history_evidence import runtime as _runtime

# Re-export these production-owner fixtures for pytest's name-based injection.
owed, publication, gateway, runtime = _owed, _publication, _gateway, _runtime


def _failed_chat_transaction(monkeypatch):
    from ouroboros import dialogue_store as ds
    original = ds.insert_row

    def failed(conn, stream, *args, **kwargs):
        original(conn, stream, *args, **kwargs)
        if stream == "chat":
            raise ds.StoreError("injected failure after row and room writes")

    monkeypatch.setattr(ds, "insert_row", failed)
    return lambda: monkeypatch.setattr(ds, "insert_row", original)


def _sender(root, monkeypatch):
    from supervisor import message_bus as bus, events_chat_delivery as handler
    from ouroboros.utils import append_jsonl
    frames = []
    bridge = bus.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(bus, 'DATA_DIR', root)
    monkeypatch.setattr(bus, 'load_state', lambda: {'owner_id': 1})
    monkeypatch.setattr(bus, 'get_bridge', lambda: bridge)
    monkeypatch.setattr(bus, 'publish_event', lambda *_a, **_kw: None)
    monkeypatch.setattr(handler, '_DELIVERED_MESSAGE_IDS', deque(maxlen=256))
    return SimpleNamespace(DRIVE_ROOT=root, RUNNING={}, send_with_budget=bus.send_with_budget,
                           append_jsonl=append_jsonl), frames


def _history(root, chat_id=1):
    from ouroboros.gateway.history import _assemble_history_response
    return json.loads(_assemble_history_response(root, chat_id, 80, 0))['messages']


def _consume_payload(root, payload, pattern):
    import pathlib
    import shutil
    from tests.group_run import run_in_group
    path = root / 'main-consumer-history.json'
    path.write_text(json.dumps(payload))
    node = shutil.which('node')
    assert node, 'Node is required for the actual Main consumer regression'
    result = run_in_group([node, '--test', '--test-force-exit', f'--test-name-pattern={pattern}', 'web/tests/main_durable_edges.test.js'],
        cwd=pathlib.Path(__file__).resolve().parents[1], env={**os.environ, 'MAIN_REVIEW_HISTORY_PAYLOAD': str(path)},
        timeout=30)
    (root / 'consumer.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr


def test_failed_main_transaction_retries_readable_history_before_ack(owed, monkeypatch):
    from supervisor import events_chat_delivery as handler, terminal_delivery as delivery
    from ouroboros.task_results import load_task_result
    context, frames = _sender(owed.root, monkeypatch)
    before = dialogue_test.chat_rows(owed.root)
    restore = _failed_chat_transaction(monkeypatch)
    handler._handle_send_message(owed.event, context)
    assert not frames and not delivery.already_delivered(owed.root, owed.event['delivery_id'])
    assert load_task_result(owed.root, 'root')['canonical_terminal_projection_ready']
    assert dialogue_test.chat_rows(owed.root) == before
    restore()
    handler._handle_send_message(owed.event, context)
    assert delivery.already_delivered(owed.root, owed.event['delivery_id'])
    assert any(row.get('delivery_id') == owed.event['delivery_id'] for row in _history(owed.root))
    assert load_task_result(owed.root, 'root')['canonical_terminal_projection_ready'] is None


@pytest.mark.parametrize('kind', ['publication', 'closure', 'answer'])
def test_failed_quiz_transaction_retry_leaves_readable_evidence(publication, monkeypatch, kind):
    from ouroboros import owner_quiz
    from ouroboros.gateway.task_decision import _record_quiz_answer_history
    from supervisor import events_chat_delivery as handler
    if kind != 'publication':
        handler._handle_send_quiz(publication.event, publication.ctx)
        if kind == 'closure':
            owner_quiz.record_closed(publication.root, 't', quiz_id='q', reason='Superseded', actor_task_id='t')
        else:
            owner_quiz.record_answered(publication.root, 't', quiz_id='q', option_index=1, request_id='owner')
    def send():
        if kind == 'publication':
            handler._handle_send_quiz(publication.event, publication.ctx)
        elif kind == 'closure':
            handler._handle_send_quiz_state(publication.event, publication.ctx)
        else:
            _record_quiz_answer_history(publication.root, 't', None, _block(publication), duplicate=False)
    before = dialogue_test.chat_rows(publication.root)
    restore = _failed_chat_transaction(monkeypatch)
    try:
        send()
    except RuntimeError:
        pass
    assert dialogue_test.chat_rows(publication.root) == before
    assert not _block(publication).get('discharged_at')
    restore()
    _quiz_due(publication)
    send()
    expected = {'publication': 'quiz', 'closure': 'quiz_closed', 'answer': 'quiz_answer'}[kind]
    readable = dialogue_test.chat_rows(publication.root)
    assert any(row.get('type') == expected for row in readable)
    state = {'publication': 'open', 'closure': 'closed', 'answer': 'answered'}[kind]
    assert any(row.get('quiz', {}).get('state') == state for row in _history(publication.root))


def test_main_retry_keeps_one_row_with_original_nonempty_timestamp(owed, monkeypatch):
    from ouroboros.terminal_projection import settle_terminal_projection
    from supervisor import events_chat_delivery as handler, message_bus as bus
    first_ts = owed.event.get('ts')
    assert first_ts
    context, frames = _sender(owed.root, monkeypatch)
    real_bridge = bus.get_bridge()
    monkeypatch.setattr(bus, 'get_bridge', lambda: SimpleNamespace(send_message=lambda *_a, **_kw: (False, 'offline')))
    monkeypatch.setattr(bus, 'utc_now_iso', lambda: '2098-01-01T00:00:00+00:00')
    handler._handle_send_message({**owed.event, 'format': ''}, context)
    _due(owed.root)
    settle_terminal_projection(owed.root, 'root')
    monkeypatch.setattr(bus, 'get_bridge', lambda: real_bridge)
    monkeypatch.setattr(bus, 'utc_now_iso', lambda: '2099-01-01T00:00:00+00:00')
    handler._handle_send_message({**owed.queued[-1], 'format': ''}, context)
    rows = dialogue_test.chat_rows(owed.root)
    rows = [row for row in rows if row.get('delivery_id') == owed.event['delivery_id']]
    assert len(rows) == 1 and rows[0]['ts'] == first_ts
    assert len(frames) == 1 and frames[0]['ts'] == first_ts


@pytest.mark.parametrize('lost_queue', [False, True])
def test_same_batch_ask_close_drains_and_maintenance_converges(publication, monkeypatch, lost_queue):
    from ouroboros.gateway.task_decision import recover_question_deliveries
    from ouroboros.gateway.main_decisions import main_decisions_snapshot
    from ouroboros.owner_quiz import record_closed
    from ouroboros.projects_registry import create_project, bind_task_to_project
    from supervisor import events, message_bus
    project = create_project(publication.root, 'build', name='Build')
    bind_task_to_project(publication.root, 't', project['id'], project['chat_id'], origin={'absent': 'system'})
    record_closed(publication.root, 't', quiz_id='q', reason='The action was withdrawn', actor_task_id='t')
    monkeypatch.setattr(message_bus, 'get_bridge', lambda: publication.ctx.bridge)
    if not lost_queue:
        for event in [publication.event, {**publication.event, 'type': 'send_quiz_state', 'state': 'closed'}]:
            events.dispatch_event(event, publication.ctx)
    report = asyncio.run(recover_question_deliveries(publication.root))
    block = _block(publication)
    assert block['publication_chat_id'] == project['chat_id']
    assert block['publication_evidence'] == 'pre_exposure'
    assert block['delivery']['state'] == 'delivered' and block['discharged_at']
    assert not block.get('published_at') and 'answered_at' not in block and 'answered_index' not in block
    rows = dialogue_test.chat_rows(publication.root)
    assert [row['type'] for row in rows] == ['quiz_closed']
    assert publication.frames and all(row.get('state') == 'closed' for row in publication.frames)
    assert not main_decisions_snapshot(publication.root, [], {})['current_questions']
    assert report['pending'] == 0
    assert asyncio.run(recover_question_deliveries(publication.root))['attempted'] == 0


def test_real_same_tool_batch_ask_close_event_drain_leaves_only_closed_history(runtime):
    from tests.test_quiz_answer import _tool_ctx
    from ouroboros.tools.core_artifacts import _escalate
    from ouroboros.owner_quiz import quiz_states
    from ouroboros.gateway.task_decision import recover_question_deliveries
    from ouroboros.utils import append_jsonl
    from ouroboros.task_results import write_task_result
    from supervisor.events import dispatch_event
    write_task_result(runtime.root, runtime.task['id'], 'running', chat_id=1)
    ctx = _tool_ctx(runtime.root, task_id=runtime.task['id'])
    ctx.owner_wait_callback = lambda *_args: None
    assert _escalate(ctx, question='Proceed?', wait_for_answer=True).startswith('OK:')
    qid = ctx._owner_wait_requested
    assert _escalate(ctx, action='close', quiz_id=qid, reason='The branch was withdrawn').startswith('OK:')
    assert not ctx._owner_wait_requested
    host = SimpleNamespace(DRIVE_ROOT=runtime.root, bridge=runtime.bridge, append_jsonl=append_jsonl)
    for event in ctx.pending_events:
        dispatch_event(event, host)
    assert asyncio.run(recover_question_deliveries(runtime.root))['pending'] == 0
    block = quiz_states(runtime.root, runtime.task['id'])[qid]
    assert block['delivery']['state'] == 'delivered' and block['publication_evidence'] == 'pre_exposure'
    assert 'answered_at' not in block and not block.get('published_at')
    rows = dialogue_test.chat_rows(runtime.root)
    assert [row['type'] for row in rows] == ['quiz_closed']


def test_legacy_answer_absence_is_not_replay_authority(runtime, monkeypatch):
    from ouroboros import obligations, owner_quiz, startup_migrations
    from ouroboros.gateway.task_decision import answer_decision, recover_question_deliveries
    from ouroboros.task_results import write_task_result
    from ouroboros import dialogue_store as ds
    from supervisor import queue
    legacy = {'quiz_id': 'old', 'question': 'Old choice?', 'options': ['A', 'B'], 'state': 'answered',
              'asked_at': '2020-01-01T00:00:00+00:00', 'answered_at': '2020-01-02T00:00:00+00:00',
              'answered_index': 1, 'request_id': 'old-owner', 'answered_after_terminal': True, 'chat_id': 1}
    write_task_result(runtime.root, runtime.task['id'], 'completed', chat_id=1, owner_quiz={'old': legacy})
    for kind in ['quiz', 'quiz_answer']:
        ds.append(runtime.root, "chat", {'type': kind, 'direction': 'system', 'chat_id': 1,
            'ts': legacy['answered_at'], 'task_id': runtime.task['id'], 'quiz': legacy})
    before = dialogue_test.chat_rows(runtime.root)
    monkeypatch.setattr(queue, 'RUNNING', {})
    startup_migrations.prepare_startup_state(runtime.root)
    assert asyncio.run(recover_question_deliveries(runtime.root))['attempted'] == 0
    status, reply = asyncio.run(answer_decision(runtime.root, {'decision_id': f"quiz:{runtime.task['id']}:old",
        'request_id': 'old-owner', 'option_index': 1}))
    assert status == 200 and reply['state'] == 'answered'
    assert reply.get('forwarded') is not True
    assert runtime.bridge._inbox.empty()
    assert dialogue_test.chat_rows(runtime.root) == before
    assert owner_quiz.quiz_states(runtime.root, runtime.task['id'])['old'] == legacy
    assert not owner_quiz.question_delivery_projection(legacy)
    assert not owner_quiz.question_delivery_settled(legacy)
    assert runtime.task['id'] not in obligations.members(runtime.root, 'owner_questions')


@pytest.mark.parametrize('bound', [False, True])
def test_child_wait_projects_real_owner_and_control_without_expanding_census(gateway, bound):
    from ouroboros.gateway.task_model_wait import _decide
    from ouroboros.model_wait import mutate_wait
    from ouroboros.owner_mailbox import KIND_MODEL_WAIT, drain_owner_entries
    from ouroboros.projects_registry import bind_task_to_project
    from ouroboros.task_results import write_task_result
    from ouroboros.utils import append_jsonl
    from supervisor.task_model_wait import handle_task_model_wait
    project, root = _project_task(gateway, 'root')
    if bound:
        bind_task_to_project(gateway.root, 'root', project['id'], project['chat_id'], origin={'absent': 'system'})
    child = {'id': 'child', 'parent_task_id': 'root', 'root_task_id': 'root', '_attempt': 3,
             'delegation_role': 'subagent', 'chat_id': 0, 'drive_root': str(gateway.root)}
    write_task_result(gateway.root, 'child', 'running', parent_task_id='root', root_task_id='root')
    for task in [root, child]:
        gateway.running[task['id']] = {'task': task, 'attempt': 3}
    wait = {'wait_id': 'login', 'revision': 3, 'task_attempt': 3, 'state': 'waiting', 'reason': 'auth',
            'model': 'claudexor::codex=test-model', 'source': 'codex', 'credential_harness': 'codex',
            'credential_profile_id': 'test-profile', 'auto_continue': True, 'requires_owner_action': True,
            'worker_slot_held': True}
    mutate_wait(gateway.root, 'child', 'login', lambda _: wait)
    ctx = SimpleNamespace(DRIVE_ROOT=gateway.root, RUNNING=gateway.running, append_jsonl=append_jsonl,
                          bridge=SimpleNamespace(push_log=lambda *_a, **_kw: None))
    handle_task_model_wait({'type': 'task_model_wait', 'task_id': 'child', **wait}, ctx)
    payload = _state(gateway)
    assert {row['activity_id'] for row in payload['active_chat_activities']} == {'root'}
    [attention] = payload['main_attention']
    assert attention['task_id'] == 'child' and attention['root_task_id'] == 'root'
    assert attention['project_id'] == project['id'] and attention['model_waits'] == {'login': wait}
    _consume_payload(gateway.root, payload, 'Main child wait')
    reply = _decide(gateway.root, {'request_id': 'child-auto-off', 'decision_id': 'model_wait:child:login',
                                 'revision': 3, 'action': 'auto_continue', 'auto_continue': False})
    assert reply.status_code == 202, reply.body
    [entry] = drain_owner_entries(gateway.root, 'child', kinds={KIND_MODEL_WAIT})
    assert json.loads(entry['text'])['auto_continue'] is False
    assert not drain_owner_entries(gateway.root, 'root', kinds={KIND_MODEL_WAIT})


def test_legacy_job_quarantines_a_corrupt_task_without_a_fabricated_question(gateway, monkeypatch):
    """The legacy job quarantines unreadable result bytes (log-only visibility, 6.3=B) and
    occupies the identity; no question is reconstructed for it and a GET never scans."""
    from ouroboros import startup_migrations, store, store_import
    from ouroboros.memory import Memory
    root = gateway.root
    (root / 'task_results').mkdir(exist_ok=True)
    (root / 'task_results/broken.json').write_text('{partial')
    startup_migrations.prepare_startup_state(root)
    store_import.run_legacy_reconcile(root)
    assert store_import.occupied_reason(root, 'broken')
    monkeypatch.setattr(store_import, 'ensure_current_store', lambda *_a, **_kw: pytest.fail('GET import'))
    monkeypatch.setattr(Memory, 'read_chat_generations', lambda *_a, **_kw: pytest.fail('GET history scan'))
    with store.open_store(root, write=False) as txn:
        revision = txn.revision()
    payload = _state(gateway)
    assert payload['current_questions'] == [] and payload['current_questions_complete'] is True
    with store.open_store(root, write=False) as txn:
        assert txn.revision() == revision


@pytest.mark.parametrize('source', ['canonical', 'history', 'both'])
def test_wake_reads_explicit_close_reason_after_gc_and_on_bootstrap(tmp_path, source):
    from tests.test_consciousness_wake import _result, _chat, _iso, _observe, T0
    block = {'quiz_id': 'q', 'state': 'closed', 'question': 'Proceed?', 'closed_at': _iso(T0 - 10),
             'closed_actor': 'model', 'closed_by_task_id': 'root', 'closed_reason': 'The action was withdrawn'}
    _result(tmp_path, 'root', ts=T0 - 10000, quizzes={'q': block} if source != 'history' else None)
    if source != 'canonical':
        _chat(tmp_path, {'type': 'quiz_closed', 'direction': 'system', 'ts': block['closed_at'],
                         'task_id': 'root', 'chat_id': 1, 'quiz': block})
    observation = _observe(tmp_path)
    lines = [line for kind, _offset, line in observation.events if kind == 'card_state']
    assert len(lines) == 1
    assert 'model task root' in lines[0] and block['closed_reason'] in lines[0]
    assert 'without an owner answer' in lines[0]
    assert not [event for event in _observe(tmp_path, boundary=observation.boundary).events if event[0] == 'card_state']


def test_native_project_nonclean_final_history_reaches_actual_consumer_once(tmp_path, monkeypatch):
    from ouroboros.projects_registry import create_project
    from ouroboros.task_results import write_task_result
    from ouroboros.terminal_projection import settle_terminal_projection
    from ouroboros.utils import append_jsonl
    from supervisor import events_chat_delivery as handler, message_bus
    project = create_project(tmp_path, 'native-project', name='Partial verification')
    answer = 'The Linux report is ready. The Windows measurements are still missing.'
    rationale = 'The Windows verification was not performed.'
    axes = {'lifecycle': {'status': 'completed'}, 'execution': {'status': 'ok', 'task_completion': {
        'action': 'stop', 'rationale': rationale, 'source': 'finish_task'}},
        'objective': {'status': 'fail', 'source': 'task_completion'}}
    write_task_result(tmp_path, 'native', 'completed', root_task_id='native', project_id=project['id'],
        chat_id=project['chat_id'], result=answer, terminal_origin='model_final', reason_code='final_message',
        outcome_axes=axes, tool_calls=1, completion_tool_calls=1, metadata={'delegation_role': 'root'})
    queued = []
    from tests.test_main_terminal_delivery import lost_bus
    monkeypatch.setattr('supervisor.workers.get_event_q', lambda *, create=True: lost_bus(queued))
    settle_terminal_projection(tmp_path, 'native')
    [event] = queued
    # This journey isolates evidence routing; FIX8 separately tests missing retained time.
    event = {**event, 'ts': event.get('ts') or '2026-10-09T00:00:00+00:00'}
    context, _frames = _sender(tmp_path, monkeypatch)
    handler._handle_send_message(event, context)
    append_jsonl(tmp_path / 'logs/tools.jsonl', {'type': 'tool_call', 'task_id': 'native', 'ts': event['ts'],
        'tool': 'finish_task', 'invocation_id': 'finish-1', 'is_error': False, 'completion_control': True})
    write_task_result(tmp_path, 'independent', 'completed', chat_id=1, result='Independent Main work')
    message_bus.log_chat('out', 1, 1, 'Independent Main work', task_id='independent', ts=event['ts'])
    append_jsonl(tmp_path / 'logs/tools.jsonl', {'type': 'tool_call', 'task_id': 'independent', 'ts': event['ts'],
        'tool': 'read_file', 'invocation_id': 'read-1', 'is_error': False})
    main = _history(tmp_path)
    _consume_payload(tmp_path, {'messages': main}, 'native Project')
    assert not any(row.get('task_id') == 'native' and row.get('tool_evidence') for row in main)
    assert any(row.get('task_id') == 'independent' and row.get('tool_evidence') for row in main)
    assert any(row.get('task_id') == 'native' and row.get('tool_evidence')
               for row in _history(tmp_path, project['chat_id']))
