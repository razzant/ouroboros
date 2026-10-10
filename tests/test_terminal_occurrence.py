"""Executor occurrence and later publication are independent durable facts."""
import json
from types import SimpleNamespace
import pytest

from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.terminal_time import terminal_time_fact, task_attempt_witness

T1 = '2026-09-24T18:18:27+00:00'
T2 = '2026-09-25T15:19:01+00:00'
T3 = '2026-09-26T19:21:43+00:00'


def test_executor_observes_once_enrichment_and_recovery_do_not_restamp(tmp_path, monkeypatch):
    from ouroboros import agent_task_pipeline as pipeline

    write_task_result(tmp_path, 'root', 'running', started_at='2026-09-24T17:00:00Z', task_attempt=0)
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T1)
    pipeline._store_task_result(SimpleNamespace(drive_root=tmp_path), {'id': 'root'},
        'Failed answer', {'rounds': 1}, {'tool_calls': [], 'reasoning_notes': []})
    fact = load_task_result(tmp_path, 'root')['terminal_time']
    assert fact['occurred_at'] == T1 and fact['source'] == 'executor_terminal'
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T3)
    write_task_result(tmp_path, 'root', 'completed', _terminal_observed=True, cost_final=True,
                      terminal_time={**fact, 'occurred_at': T3})
    assert load_task_result(tmp_path, 'root')['terminal_time'] == fact
    write_task_result(tmp_path, 'legacy', 'failed', ts='2026-09-24T17:00:00Z', terminal_time=fact)
    write_task_result(tmp_path, 'legacy', 'failed', _terminal_observed=True)
    assert load_task_result(tmp_path, 'legacy')['terminal_time']['occurred_at'] is None


def test_replica_transfers_only_matching_source_attempt_without_receiver_clock(tmp_path, monkeypatch):
    from ouroboros.post_task_checkpoint import project_replica_task_result_fields

    canonical, child = tmp_path / 'host', tmp_path / 'child'
    for root in (canonical, child):
        write_task_result(root, 'root', 'running', task_attempt=1, started_at=T1)
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T2)
    source = write_task_result(child, 'root', 'failed', _terminal_observed=True)
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T3)
    result = write_task_result(canonical, 'root', 'failed', _terminal_time_source=source)
    assert result['terminal_time'] == source['terminal_time']
    overlay = project_replica_task_result_fields(result, {**source, 'started_at': T3})
    assert terminal_time_fact({**result, **overlay}) == result['terminal_time']
    write_task_result(canonical, 'other', 'running', task_attempt=2, started_at=T3)
    mismatched = write_task_result(canonical, 'other', 'failed', _terminal_time_source=source)
    assert mismatched['terminal_time']['occurred_at'] is None


@pytest.mark.parametrize('known', [False, True])
@pytest.mark.parametrize('patch', [
    {'started_at': T3}, {'task_attempt': 9}, {'_attempt': 9},
    {'metadata': {'attempt': 9, 'note': 'enriched'}},
    {'metadata': {'task_attempt': 9, 'note': 'enriched'}},
    {'metadata': {'note': 'enriched'}}, {'metadata': None},
])
def test_terminal_enrichment_keeps_valid_fact_and_witness(tmp_path, monkeypatch, known, patch):
    from ouroboros.post_task_checkpoint import project_replica_task_result_fields

    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T2)
    original = write_task_result(tmp_path, 'root', 'failed', _terminal_observed=known,
        started_at=T1, task_attempt=1, _attempt=1, metadata={'attempt': 1, 'task_attempt': 1})
    fact = terminal_time_fact(original)
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T3)
    overlay = project_replica_task_result_fields(original, {**original, **patch, 'result': 'enriched'})
    for row in [
        {**original, **overlay},
        write_task_result(tmp_path, 'root', 'failed', **patch, cost_final=True),
        write_task_result(tmp_path, 'root', 'failed', _field_projector=project_replica_task_result_fields,
                          **patch, result='enriched'),
        write_task_result(tmp_path, 'root', 'failed', _terminal_observed=True, cost_final=True),
    ]:
        assert terminal_time_fact(row) == fact
        assert task_attempt_witness(row) == task_attempt_witness(original)
        assert row['status'] == 'failed'
        if isinstance(patch.get('metadata'), dict):
            assert row['metadata']['note'] == 'enriched'
    assert terminal_time_fact(load_task_result(tmp_path, 'root')) == fact


def test_terminal_enrichment_preserves_absent_witness_components(tmp_path):
    from ouroboros.post_task_checkpoint import project_replica_task_result_fields

    original = write_task_result(tmp_path, 'root', 'failed', _terminal_observed=True)
    patch = {'started_at': T3, 'task_attempt': 9, '_attempt': 9,
             'metadata': {'attempt': 9, 'task_attempt': 9, 'note': 'kept'}}
    overlay = project_replica_task_result_fields(original, patch)
    for row in [{**original, **overlay}, write_task_result(tmp_path, 'root', 'failed', **patch)]:
        assert terminal_time_fact(row) == terminal_time_fact(original)
        assert not {'started_at', 'task_attempt', '_attempt'} & row.keys()
        assert row['metadata'] == {'note': 'kept'}


def test_effective_read_copyback_and_followup_keep_canonical_terminal_witness(tmp_path, monkeypatch):
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.headless import copy_child_task_result

    host, child = tmp_path / 'host', tmp_path / 'child'
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T2)
    canonical = write_task_result(host, 'root', 'failed', _terminal_observed=True,
        started_at=T1, task_attempt=1, _attempt=1, metadata={'attempt': 1, 'task_attempt': 1},
        child_drive_root=str(child), headless_child_drive_root=str(child))
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T3)
    write_task_result(child, 'root', 'failed', _terminal_observed=True,
        started_at=T3, task_attempt=9, _attempt=9, metadata={'attempt': 9, 'task_attempt': 9, 'note': 'replica'},
        result='Replica answer')
    effective = load_effective_task_result(host, 'root', materialize_artifacts=False)
    copied = copy_child_task_result(host, {'id': 'root', 'drive_root': str(child)})
    enriched = write_task_result(host, 'root', 'failed', cost_final=True)
    for row in (effective, copied, enriched, load_task_result(host, 'root')):
        assert terminal_time_fact(row) == terminal_time_fact(canonical)
        assert task_attempt_witness(row) == task_attempt_witness(canonical)
        assert row['status'] == 'failed'
        assert row['metadata']['note'] == 'replica'


def test_pause_resume_does_not_mint_an_attempt_or_a_terminal_time(tmp_path, monkeypatch):
    start = write_task_result(tmp_path, 'root', 'running', task_attempt=0, started_at=T1)
    paused = write_task_result(tmp_path, 'root', 'interrupted', _terminal_observed=True)
    resumed = write_task_result(tmp_path, 'root', 'running')
    assert task_attempt_witness(start) == task_attempt_witness(paused) == task_attempt_witness(resumed)
    assert 'terminal_time' not in paused
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T2)
    assert write_task_result(tmp_path, 'root', 'completed', _terminal_observed=True)['terminal_time']['occurred_at'] == T2


def test_reconciled_presence_reopen_and_real_retry_have_their_own_executor_fact(tmp_path, monkeypatch):
    from ouroboros.task_results import reopen_reconciled_presence_placeholder

    write_task_result(tmp_path, 'root', 'running', task_attempt=0, started_at=T1)
    old = write_task_result(tmp_path, 'root', 'failed', metadata={'source':'presence'},
        status_reconciled_from='running', reason_code='orphaned_running_after_worker_restart')
    assert old['terminal_time']['source'] == 'unknown'
    assert reopen_reconciled_presence_placeholder(tmp_path, 'root')
    assert 'terminal_time' not in load_task_result(tmp_path, 'root')
    running = write_task_result(tmp_path, 'root', 'running', task_attempt=1, started_at=T2)
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T3)
    final = write_task_result(tmp_path, 'root', 'completed', _terminal_observed=True)
    assert final['terminal_time']['occurred_at'] == T3
    assert final['terminal_time']['attempt'] == task_attempt_witness(running)
    assert final['terminal_time']['attempt'] != old['terminal_time']['attempt']
    assert final['superseded_placeholder']['terminal_time'] == old['terminal_time']


@pytest.mark.parametrize('canonical_state', ['matching', 'stale_attempt', 'recovered_unknown'])
def test_actual_child_copyback_cannot_authenticate_its_own_attempt(tmp_path, monkeypatch, canonical_state):
    from ouroboros.headless import copy_child_task_result

    host, child = tmp_path / 'host', tmp_path / 'child'
    write_task_result(child, 'root', 'running', task_attempt=1, started_at=T1)
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T2)
    source = write_task_result(child, 'root', 'failed', _terminal_observed=True)
    write_task_result(host, 'root', 'running', task_attempt=2 if canonical_state == 'stale_attempt' else 1,
                      started_at=T3 if canonical_state == 'stale_attempt' else T1)
    if canonical_state == 'recovered_unknown':
        write_task_result(host, 'root', 'failed')
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T3)
    copied = copy_child_task_result(host, {'id': 'root', 'drive_root': str(child)})
    expected = source['terminal_time']['occurred_at'] if canonical_state == 'matching' else None
    assert copied['terminal_time']['occurred_at'] == expected
    final = load_task_result(host, 'root')
    assert final['terminal_time'] == terminal_time_fact(final)
    assert terminal_time_fact(final)['occurred_at'] == expected


@pytest.mark.parametrize('child_state', ['matching', 'stale_attempt', 'recovered'])
def test_orphan_settlement_before_delayed_copyback_keeps_child_occurrence(tmp_path, monkeypatch, child_state):
    """The sweep can settle a split row from its child terminal before copyback lands."""
    import time
    from ouroboros.headless import copy_child_task_result
    from ouroboros.task_status import reconcile_orphaned_running_tasks
    from ouroboros.utils import append_jsonl

    host, child = tmp_path / 'host', tmp_path / 'child'
    write_task_result(host, 'root', 'running', task_attempt=1, started_at=T1, ts=T1,
                      child_drive_root=str(child), headless_child_drive_root=str(child))
    write_task_result(child, 'root', 'running', task_attempt=2 if child_state == 'stale_attempt' else 1,
                      started_at=T1, ts=T1)
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T2)
    if child_state == 'recovered':  # no child terminal: only the queue/worker orphan proof ends it
        monkeypatch.setattr(time, 'time', lambda: 1_800_000_000.0)
        (host / 'state').mkdir(exist_ok=True)
        (host / 'state/queue_snapshot.json').write_text(json.dumps({'ts': '2027-01-15T08:00:00+00:00', 'pending': [], 'running': []}))
        append_jsonl(host / 'logs/events.jsonl', {'ts': T2, 'type': 'worker_boot'})
    else:
        write_task_result(child, 'root', 'completed', _terminal_observed=True, result='Child answer')
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T3)
    assert load_task_result(host, 'root')['status'] == 'running'  # copyback has not delivered it yet
    assert reconcile_orphaned_running_tasks(host) == 1
    rows = [load_task_result(host, 'root')]
    if child_state != 'recovered':
        rows.append(copy_child_task_result(host, {'id': 'root', 'drive_root': str(child)}))
    for row in rows:
        assert row['status'] == ('failed' if child_state == 'recovered' else 'completed')
        assert terminal_time_fact(row)['occurred_at'] == (T2 if child_state == 'matching' else None)
    assert terminal_time_fact(load_task_result(host, 'root')) == terminal_time_fact(rows[0])


@pytest.mark.parametrize('worker_start', [True, False])
def test_split_subagent_start_carries_the_witness_its_copyback_needs(tmp_path, monkeypatch, worker_start):
    """The real subagent route: request receipt, assignment mirror, worker start, child end, copyback."""
    import supervisor.workers as workers
    from ouroboros.agent import OuroborosAgent
    from ouroboros.headless import copy_child_task_result
    from ouroboros.task_status import load_effective_task_result
    from supervisor.worker_assignment import _mirror_assigned_running_status

    host, child = tmp_path / 'host', tmp_path / 'child'
    task = {'id': 'sub', 'delegation_role': 'subagent', 'parent_task_id': 'root', 'root_task_id': 'root',
            'drive_root': str(child), 'child_drive_root': str(child), 'budget_drive_root': str(host),
            'memory_mode': 'forked', '_attempt': 1, 'metadata': {}}
    write_task_result(host, 'sub', 'requested', delegation_role='subagent', parent_task_id='root',
                      drive_root=str(child), child_drive_root=str(child))
    monkeypatch.setattr(workers, 'DRIVE_ROOT', host)
    _mirror_assigned_running_status(task)
    assert task_attempt_witness(load_task_result(host, 'sub'))['started_at'] is None
    actor = SimpleNamespace(env=SimpleNamespace(drive_root=child, budget_drive_root=host),
                            _task_started_ts=1790000000.0)
    if worker_start:
        OuroborosAgent._persist_running_record(actor, task)
    else:  # an install that started the child before the canonical witness existed
        write_task_result(child, 'sub', 'running', task_attempt=1, started_at=T1)
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T2)
    source = write_task_result(child, 'sub', 'failed', _terminal_observed=True, result='Child failed')
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T3)
    effective = load_effective_task_result(host, 'sub', materialize_artifacts=False)
    copied = copy_child_task_result(host, task)
    for row in (effective, copied, load_task_result(host, 'sub')):
        assert row['status'] == 'failed'
        # The canonical start now names the child's attempt; a legacy start
        # without it stays unknown instead of trusting the replica's own clock.
        assert terminal_time_fact(row)['occurred_at'] == (T2 if worker_start else None)
        if worker_start:
            assert task_attempt_witness(row) == task_attempt_witness(source)


def test_create_only_compatibility_import_transfers_trusted_fact(tmp_path, monkeypatch):
    from ouroboros.terminal_projection import append_terminal_projection

    source = write_task_result(tmp_path / 'source', 'root', 'failed', _terminal_observed=True,
                               started_at=T1, chat_id=1, result='Saved result')
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T3)
    assert append_terminal_projection(tmp_path / 'host', 'root', {}, {}, result=source)
    assert load_task_result(tmp_path / 'host', 'root')['terminal_time'] == source['terminal_time']


def test_late_failure_publication_after_other_success_keeps_both_outcomes_and_times(tmp_path, monkeypatch):
    from ouroboros.project_dialogue import append_terminal_task_projection
    from ouroboros.terminal_projection import _already_in_chat

    write_task_result(tmp_path, 'old', 'running', started_at=T1)
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T1)
    old = write_task_result(tmp_path, 'old', 'failed', _terminal_observed=True,
                            chat_id=1, result='Old failure')
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T2)
    new = write_task_result(tmp_path, 'new', 'completed', _terminal_observed=True,
                            chat_id=1, result='Later success')
    monkeypatch.setattr('ouroboros.terminal_projection.utc_now_iso', lambda: T2)
    assert append_terminal_task_projection(tmp_path, 'new', {'id': 'new'}, new, {})
    monkeypatch.setattr('ouroboros.terminal_projection.utc_now_iso', lambda: T3)
    assert append_terminal_task_projection(tmp_path, 'old', {'id': 'old'}, old, {'ts': T1})
    rows = [json.loads(line) for line in (tmp_path / 'logs/chat.jsonl').read_text().splitlines()]
    assert [(row['task_id'], row['ts']) for row in rows] == [('new', T2), ('old', T3)]
    assert rows[-1]['terminal_time']['occurred_at'] == T1
    assert rows[-1]['status'] == 'failed' and rows[0]['status'] == 'completed'
    assert _already_in_chat(tmp_path, rows[-1])['ts'] == T3
    assert load_task_result(tmp_path, 'old')['canonical_terminal_projection']['written_at'] == T3


@pytest.mark.parametrize('observed', [True, False])
def test_project_room_terminal_row_keeps_occurrence_beside_its_publication(tmp_path, monkeypatch, observed):
    """A Project room's own terminal row carries the same end fact as its Main mirror."""
    import asyncio

    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.project_dialogue import append_terminal_task_projection
    from ouroboros.projects_registry import bind_task_to_project, create_project

    project = create_project(tmp_path, 'room-times', name='Room times')
    bind_task_to_project(tmp_path, 'late', project['id'], project['chat_id'], origin={'absent': 'system'})
    write_task_result(tmp_path, 'late', 'running', task_attempt=0, started_at=T1,
                      project_id=project['id'], chat_id=project['chat_id'])
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T1)
    # A recovered or imported terminal has no executor observation: it stays unknown.
    ended = write_task_result(tmp_path, 'late', 'failed', _terminal_observed=observed, result='Late failure')
    monkeypatch.setattr('ouroboros.terminal_projection.utc_now_iso', lambda: T3)
    assert append_terminal_task_projection(tmp_path, 'late', {'id': 'late'}, ended, {})
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(
        SimpleNamespace(query_params={'chat_id': str(project['chat_id'])})))
    rows = [row for row in json.loads(response.body)['messages'] if row.get('system_type') == 'task_summary']
    assert [(row['task_id'], row['ts']) for row in rows] == [('late', T3)], 'publication keeps its own time'
    assert rows[0]['terminal_time'] == ended['terminal_time'] == terminal_time_fact(ended)
    assert rows[0]['terminal_time']['occurred_at'] == (T1 if observed else None)
    assert rows[0]['terminal_time']['source'] == ('executor_terminal' if observed else 'unknown')


def test_unknown_fact_cannot_be_upgraded_by_untyped_or_mismatched_metadata():
    row = {'started_at': T1, 'task_attempt': 1}
    fact = {'v': 1, 'source': 'executor_terminal', 'occurred_at': T2, 'attempt': task_attempt_witness(row)}
    assert terminal_time_fact({**row, 'terminal_time': fact})['occurred_at'] == T2
    for patch in ({'source': 'unknown'}, {'attempt': {}}, {'occurred_at': '2026-09-24'}, {'v': 2}):
        assert terminal_time_fact({**row, 'terminal_time': {**fact, **patch}})['occurred_at'] is None


def published_terminal_fixture(root, monkeypatch):
    """Real producer/outbox replay/bus rows for backend and browser qualification."""
    from queue import Queue
    import supervisor.message_bus as bus
    import supervisor.terminal_delivery as delivery
    from ouroboros.project_dialogue import enqueue_project_completion_summary
    from ouroboros.projects_registry import create_project, bind_task_to_project

    project = create_project(root, 'terminal-times', name='Terminal times')
    queue = Queue()
    bridge = bus.LocalChatBridge({})
    frames = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr('supervisor.workers.get_event_q', lambda: queue)
    monkeypatch.setattr(bus, 'DATA_DIR', root)
    monkeypatch.setattr(bus, 'get_bridge', lambda: bridge)
    monkeypatch.setattr(bus, 'load_state', lambda: {'session_id': 'qualification', 'owner_id': 7})
    monkeypatch.setattr(bus, '_advance_project_visible_revision', lambda *_: None)
    monkeypatch.setattr(bus, 'publish_event', lambda *_a, **_k: None)
    monkeypatch.setattr(delivery, '_REPLAY_MIN_AGE_SEC', 0)
    results = {}
    for task_id, status, ended in [('old', 'failed', T1), ('new', 'completed', T2), ('legacy', 'failed', None)]:
        bind_task_to_project(root, task_id, project['id'], project['chat_id'], origin={'absent':'system'})
        write_task_result(root, task_id, 'running', task_attempt=0, started_at=T1,
                          project_id=project['id'], chat_id=project['chat_id'])
        monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: ended or T1)
        results[task_id] = write_task_result(root, task_id, status, _terminal_observed=bool(ended),
            result=f'{task_id.upper()}_TASK_ANSWER', terminal_origin='model_final')
    for task_id, added in [('new', T2), ('old', T3), ('legacy', T3)]:
        row = results[task_id]
        assert enqueue_project_completion_summary(root, {'status':row['status']}, task_id,
            {'id':task_id, 'project_id':project['id'], 'chat_id':project['chat_id']}, row,
            {'status':row['status']})
        queued = queue.get_nowait()  # lost before delivery, as on restart
        assert queued['progress_meta']['terminal_time'] == row['terminal_time']
        assert delivery.replay_pending_deliveries(root, event_queue=queue) == [queued['delivery_id']]
        replay = queue.get_nowait()
        monkeypatch.setattr(bus, 'utc_now_iso', lambda: added)
        bus.send_with_budget(replay['chat_id'], replay['text'], task_id=task_id,
            role=replay['role'], system_type=replay['system_type'], progress_meta=replay['progress_meta'])
        assert delivery.register_delivery(root, replay['delivery_id'])
    assert not delivery.pending_deliveries(root)
    return project, frames


def test_full_publication_chain_retains_occurrence_and_delivery_times(tmp_path, monkeypatch):
    import asyncio
    from ouroboros.gateway.history import make_chat_history_endpoint
    _, frames = published_terminal_fixture(tmp_path, monkeypatch)
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={'chat_id':'1'})))
    rows = [row for row in json.loads(response.body)['messages'] if row.get('system_type') == 'project_completion_summary']
    assert [row['task_id'] for row in rows] == ['new', 'old', 'legacy']
    assert [row['ts'] for row in rows] == [T2, T3, T3]
    assert [row['terminal_time']['occurred_at'] for row in rows] == [T2, T1, None]
    assert [frame['terminal_time'] for frame in frames if frame.get('type') == 'chat'] == [row['terminal_time'] for row in rows]
    assert load_task_result(tmp_path, 'new')['status'] == 'completed'
    assert load_task_result(tmp_path, 'old')['status'] == 'failed'
