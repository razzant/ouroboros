"""Executor occurrence and later publication are independent durable facts."""
import json
from types import SimpleNamespace
import pytest

from ouroboros.task_results import load_task_result, write_task_result
from tests._dialogue import chat_rows
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


def test_a_supervisor_cancel_is_an_observed_end_named_by_its_observer(tmp_path, monkeypatch):
    """The supervisor cancelling work it holds performs that transition itself, so it observes the end
    (audit 08-11.11 item 6); the fact names its observer. In the other direction nothing else gains a
    clock: a cancel written without the flag (recovery) stays unknown, the flag stamps no other status,
    no field can claim the source, and a terminal the executor observed first is never restamped."""
    def started(task_id):
        monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T1)
        write_task_result(tmp_path, task_id, 'running', task_attempt=1, started_at=T1)
        monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T2)

    started('held')
    cancelled = write_task_result(tmp_path, 'held', 'cancelled', _cancel_observed=True, result='Task cancelled.')
    assert terminal_time_fact(cancelled) == {'v': 1, 'occurred_at': T2, 'source': 'supervisor_cancel',
                                             'attempt': task_attempt_witness(cancelled)}
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T3)
    write_task_result(tmp_path, 'held', 'cancelled', _cancel_observed=True, cost_final=True)  # enrichment
    assert load_task_result(tmp_path, 'held')['terminal_time'] == cancelled['terminal_time']

    started('recovered')
    recovered = write_task_result(tmp_path, 'recovered', 'cancelled', result='was neither queued nor running',
                                  terminal_time={**cancelled['terminal_time'], 'occurred_at': T2})
    assert terminal_time_fact(recovered)['source'] == 'unknown'
    started('failed')
    assert terminal_time_fact(write_task_result(tmp_path, 'failed', 'failed', _cancel_observed=True))['source'] == 'unknown'
    started('done')
    done = write_task_result(tmp_path, 'done', 'completed', _terminal_observed=True, result='ok')
    write_task_result(tmp_path, 'done', 'cancelled', _cancel_observed=True)  # natural completion wins a late cancel
    assert load_task_result(tmp_path, 'done')['terminal_time'] == done['terminal_time']
    assert done['terminal_time']['source'] == 'executor_terminal'
    # Only live transitions claim the observation, including policy cancellation before assignment.
    # Restart reconciliation and the pool's emergency writer leave the time unknown.
    import pathlib

    claims = {path.name: path.read_text(encoding='utf-8').count('_cancel_observed=True')
              for path in (pathlib.Path(__file__).resolve().parents[1] / 'supervisor').glob('*.py')}
    assert {name: count for name, count in claims.items() if count} == {
        'task_lifecycle.py': 2, 'workers.py': 1, 'worker_assignment.py': 2}


def test_only_the_executors_own_end_is_an_answer_time(tmp_path, monkeypatch):
    """A root the supervisor cancelled while its late work was open was stopped, not answered: the
    cancel time is its end and never its answer time, while the executor's own end still is one."""
    from ouroboros.gateway.state import _task_activity_facts

    late = {'post_task_synthesis': 'running'}
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T1)
    for task_id in ('answered', 'stopped'):
        write_task_result(tmp_path, task_id, 'running', task_attempt=1, started_at=T1)
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T2)
    write_task_result(tmp_path, 'answered', 'completed', _terminal_observed=True, root_phase_checkpoint=late)
    write_task_result(tmp_path, 'stopped', 'cancelled', _cancel_observed=True, root_phase_checkpoint=late)
    assert terminal_time_fact(load_task_result(tmp_path, 'stopped'))['occurred_at'] == T2
    answered_at = {task_id: _task_activity_facts(tmp_path, task_id)['display']['late_work']['answered_at']
                   for task_id in ('answered', 'stopped')}
    assert answered_at == {'answered': T2, 'stopped': ''}


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


@pytest.mark.parametrize(("patch", "known"), [
    pytest.param({'started_at': T3}, False, id='patch0-False'),
    pytest.param({'task_attempt': 9}, True, id='patch1-True'),
    pytest.param({'_attempt': 9}, False, id='patch2-False'),
    pytest.param({'metadata': {'attempt': 9, 'note': 'enriched'}}, True, id='patch3-True'),
    pytest.param({'metadata': {'task_attempt': 9, 'note': 'enriched'}}, False, id='patch4-False'),
    pytest.param({'metadata': {'note': 'enriched'}}, True, id='patch5-True'),
    pytest.param({'metadata': None}, False, id='patch6-False'),
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


@pytest.mark.parametrize('host_start,child_start', [
    pytest.param({'task_attempt': 1, '_attempt': 1, 'started_at': T1},
                 {'task_attempt': 9, '_attempt': 9, 'started_at': T3}, id='newer-replica'),
    pytest.param({'task_attempt': 9, 'started_at': T1}, {'task_attempt': 1, 'started_at': T3}, id='older-replica'),
    pytest.param({'task_attempt': 1, 'started_at': T1}, {'task_attempt': 1, 'started_at': T3}, id='conflicting-start'),
    pytest.param({'task_attempt': 1}, {}, id='unnamed-replica'),
    pytest.param({}, {'task_attempt': 1}, id='unnamed-owner'),
    pytest.param({'started_at': T1}, {'started_at': T3}, id='conflicting-legacy-start'),
])
def test_effective_read_copyback_and_followup_keep_canonical_terminal_witness(
        tmp_path, monkeypatch, host_start, child_start):
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.headless import copy_child_task_result

    host, child = tmp_path / 'host', tmp_path / 'child'
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T2)
    canonical = write_task_result(host, 'root', 'failed', _terminal_observed=True,
        **host_start, metadata={'source': 'canonical-task'},
        result='Canonical answer', final_answer='Canonical final', reason_code='canonical_reason',
        child_drive_root=str(child), headless_child_drive_root=str(child))
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T3)
    write_task_result(child, 'root', 'failed', _terminal_observed=True,
        **child_start, metadata={'attempt': child_start.get('task_attempt'), 'note': 'replica', 'source': 'child'},
        result='Replica answer', final_answer='Replica final', reason_code='replica_reason',
        error='foreign error', cancel_origin={'source': 'owner_restart'})
    effective = load_effective_task_result(host, 'root', materialize_artifacts=False)
    effective_full = load_effective_task_result(host, 'root')
    copied = copy_child_task_result(host, {'id': 'root', 'drive_root': str(child)})
    enriched = write_task_result(host, 'root', 'failed', cost_final=True)
    for row in (effective, effective_full, copied, enriched, load_task_result(host, 'root')):
        assert terminal_time_fact(row) == terminal_time_fact(canonical)
        assert task_attempt_witness(row) == task_attempt_witness(canonical)
        assert row['status'] == 'failed'
        assert (row['result'], row['final_answer'], row['reason_code']) == (
            'Canonical answer', 'Canonical final', 'canonical_reason')
        assert not {'error', 'cancel_origin', 'child_ref_promotion'} & row.keys()
        assert row['metadata'] == {'source': 'canonical-task', 'note': 'replica'}


@pytest.mark.parametrize('status', ['failed', 'completed'])
def test_matching_terminal_replica_adopts_its_answer_and_cause_only_for_the_same_outcome(tmp_path, monkeypatch, status):
    from ouroboros.headless import copy_child_task_result
    from ouroboros.task_status import load_effective_task_result

    host, child = tmp_path / 'host', tmp_path / 'child'
    basis = {'task_attempt': 0, 'started_at': T1}
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T2)
    canonical = write_task_result(host, 'root', 'failed', **basis, child_drive_root=str(child),
        _terminal_observed=True, result='Canonical answer', reason_code='canonical_reason',
        wake_summary='Canonical sentence')
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T3)
    write_task_result(child, 'root', status, **basis, _terminal_observed=True,
        result='Refined answer', reason_code='refined_reason', wake_summary='Refined sentence', fold_read=True)
    visible = load_effective_task_result(host, 'root', materialize_artifacts=False)
    copied = copy_child_task_result(host, {'id': 'root', 'drive_root': str(child)})
    for row in (visible, copied, load_task_result(host, 'root')):
        assert row['status'] == 'failed' and terminal_time_fact(row) == terminal_time_fact(canonical)
        assert (row['result'], row['reason_code']) == (('Refined answer', 'refined_reason') if status == 'failed'
                                                     else ('Canonical answer', 'canonical_reason'))
        assert (row.get('wake_summary'), row.get('fold_read')) == (
            ('Refined sentence', True) if status == 'failed' else ('Canonical sentence', None))


@pytest.mark.parametrize('pause,via_sweep', [(None, False), (None, True), ('resumed', False), ('paused', False)])
@pytest.mark.parametrize('status', ['completed', 'cancelled'])
def test_newer_replica_replaces_only_an_open_unheld_mirror(tmp_path, monkeypatch, pause, via_sweep, status):
    from ouroboros.headless import copy_child_task_result
    from ouroboros.owner_continue import continuation_eligibility
    from ouroboros.task_status import execution_owner_record, load_effective_task_result, reconcile_orphaned_running_tasks

    host, child = tmp_path / 'host', tmp_path / 'child'
    old_owner, new_owner = (execution_owner_record(host, {'id': 'root', '_attempt': n}, 'pooled') for n in (1, 2))
    host_row = write_task_result(host, 'root', 'running', task_attempt=1, _attempt=1, started_at=T1,
        child_drive_root=str(child), execution_owner=old_owner, metadata={'attempt': 1, 'note': 'task annotation'},
        result='Old execution', reason_code='old_reason', budget_pause={'state': pause} if pause else None)
    write_task_result(child, 'root', 'running', task_attempt=2, started_at=T2, execution_owner=new_owner)
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T3)
    if status == 'cancelled':
        write_task_result(child, 'root', 'interrupted', reason_code='deadline')
        source = write_task_result(child, 'root', status, _cancel_observed=True, result='New execution',
                                   cancel_origin={'source': 'owner_restart'})
    else:
        source = write_task_result(child, 'root', status, _terminal_observed=True, result='New execution')

    before = load_effective_task_result(host, 'root', materialize_artifacts=False)
    if via_sweep:
        assert reconcile_orphaned_running_tasks(host) == 1
        written = load_task_result(host, 'root')
    else:
        written = copy_child_task_result(host, {'id': 'root', 'drive_root': str(child)})
    replay = copy_child_task_result(host, {'id': 'root', 'drive_root': str(child)})
    for row in (before, written, replay, load_effective_task_result(host, 'root', materialize_artifacts=False)):
        expected = host_row if pause == 'paused' else source
        assert row['status'] == expected['status'] and row['execution_owner'] == expected['execution_owner']
        assert task_attempt_witness(row) == task_attempt_witness(expected)
        assert row['result'] == expected['result'] and row.get('reason_code') == expected.get('reason_code')
        assert terminal_time_fact(row) == terminal_time_fact(expected)
        assert row['metadata']['note'] == 'task annotation'
        if status == 'cancelled' and pause != 'paused':
            assert row['earlier_stop'] == 'deadline'
            assert continuation_eligibility(row, 'root') == {
                'eligible': False, 'cause': 'deadline', 'refusal': 'hard_limit_reached'}


@pytest.mark.parametrize('condition', ['hard-limit', 'owner-stop', 'terminal', 'same-start',
                                       'missing-start', 'invalid-start', 'foreign-owner'])
def test_newer_replica_cancel_cannot_end_a_protected_or_unbound_mirror(tmp_path, condition):
    from ouroboros.headless import copy_child_task_result
    from ouroboros.owner_continue import continuation_eligibility
    from ouroboros.task_status import execution_owner_record, load_effective_task_result

    host, child = tmp_path / 'host', tmp_path / 'child'
    canonical = write_task_result(host, 'root', 'running', task_attempt=1, started_at=T1,
                                  child_drive_root=str(child), result='Current attempt')
    if condition == 'hard-limit':
        canonical = write_task_result(host, 'root', 'interrupted', reason_code='budget_exhausted')
    if condition in {'owner-stop', 'terminal'}:
        status = 'cancelled' if condition == 'owner-stop' else 'completed'
        canonical = write_task_result(host, 'root', status, cancel_origin={'source': 'http_single'})
    owner = execution_owner_record(host, {'id': 'other' if condition == 'foreign-owner' else 'root',
                                          '_attempt': 2}, 'pooled')
    start = {'same-start': T1, 'missing-start': None, 'invalid-start': 'unknown'}.get(condition, T2)
    write_task_result(child, 'root', 'cancelled', task_attempt=2, started_at=start, execution_owner=owner,
                      _cancel_observed=True, cancel_origin={'source': 'owner_restart'}, result='Replica')
    effective = load_effective_task_result(host, 'root', materialize_artifacts=False)
    for key in ('task_attempt', 'status', 'result', 'reason_code', 'cancel_origin', 'earlier_stop', 'terminal_time'):
        assert effective.get(key) == canonical.get(key)
    if condition == 'owner-stop':
        assert continuation_eligibility(effective, 'root')['refusal'] == 'stopped_by_owner'
    assert copy_child_task_result(host, {'id': 'root', 'drive_root': str(child)}) == canonical
    assert load_task_result(host, 'root') == canonical


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


_FIRST_START = {'task_attempt': 0, 'started_at': T1}  # direct execution can start at zero


@pytest.mark.parametrize('child_status', ['failed', 'completed'])
@pytest.mark.parametrize('host_start,child_start,already_settled,accepts', [
    pytest.param(_FIRST_START, _FIRST_START, False, True, id='matching'),
    pytest.param({'task_attempt': 1, 'started_at': T3}, _FIRST_START, False, False, id='stale_attempt'),
    pytest.param({'task_attempt': 0, 'started_at': T3}, _FIRST_START, False, False, id='stale_start'),
    pytest.param({'task_attempt': 1, 'started_at': T3}, {}, False, False, id='unnamed_child'),
    pytest.param({}, {}, False, True, id='legacy_pair'),
    pytest.param(_FIRST_START, _FIRST_START, True, True, id='recovered_unknown'),
])
def test_actual_child_copyback_cannot_authenticate_its_own_attempt(
        tmp_path, monkeypatch, host_start, child_start, already_settled, accepts, child_status):
    from ouroboros.headless import copy_child_task_result
    from ouroboros.task_status import load_effective_task_result

    host, child = tmp_path / 'host', tmp_path / 'child'
    write_task_result(child, 'root', 'running', **child_start)
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T2)
    source = write_task_result(child, 'root', child_status, _terminal_observed=bool(child_start),
                               result='Child answer', total_rounds=3)
    canonical = write_task_result(host, 'root', 'running', child_drive_root=str(child), result='Current work',
                                  **host_start)
    if already_settled:
        canonical = write_task_result(host, 'root', child_status)
    monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: T3)
    visible = load_effective_task_result(host, 'root', materialize_artifacts=False)
    copied = copy_child_task_result(host, {'id': 'root', 'drive_root': str(child)})
    stored = load_task_result(host, 'root')
    if not accepts:
        assert copied == stored == canonical
    expected = source['terminal_time']['occurred_at'] if accepts and not already_settled else None
    for row in (visible, copied, stored, load_effective_task_result(host, 'root', materialize_artifacts=False)):
        assert row['status'] == (child_status if accepts else 'running')
        assert task_attempt_witness(row) == task_attempt_witness(canonical)
        assert row.get('total_rounds') == (3 if accepts else None)
        assert terminal_time_fact(row)['occurred_at'] == expected
        if accepts:
            assert row['terminal_time'] == terminal_time_fact(row)
            if not already_settled:
                assert row['result'] == 'Child answer'
        else:
            assert row['result'] == 'Current work' and 'terminal_time' not in row


@pytest.mark.parametrize('status', ['failed', 'cancelled'])
def test_child_copyback_rechecks_the_attempt_at_publication(tmp_path, monkeypatch, status):
    from ouroboros import headless
    from ouroboros.task_status import load_effective_task_result

    host, child = tmp_path / 'host', tmp_path / 'child'
    write_task_result(host, 'root', 'running', task_attempt=1, started_at=T1, child_drive_root=str(child))
    attempt, started = (2, T2) if status == 'cancelled' else (1, T1)
    write_task_result(child, 'root', status, task_attempt=attempt, started_at=started, result='Stale replica')
    assert load_effective_task_result(host, 'root', materialize_artifacts=False)['status'] == status
    restarted = {}

    def publish_after_restart(*args, **kwargs):
        if not restarted:
            restarted.update(write_task_result(host, 'root', 'running', task_attempt=3, started_at=T3, result='Current'))
        return write_task_result(*args, **kwargs)

    monkeypatch.setattr(headless, 'write_task_result', publish_after_restart)
    copied = headless.copy_child_task_result(host, {'id': 'root', 'drive_root': str(child)})
    assert copied == load_task_result(host, 'root') == restarted
    visible = load_effective_task_result(host, 'root', materialize_artifacts=False)
    assert (visible['status'], visible['task_attempt'], visible['result']) == ('running', 3, 'Current')


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
    rows = chat_rows(tmp_path)
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
    """Real settlement/readiness retry/strict-sender rows for backend and browser qualification."""
    from collections import deque
    from tests._cancel_intents_shared import DeliveryBus
    import supervisor.events_chat_delivery as sender
    import supervisor.message_bus as bus
    import supervisor.terminal_delivery as delivery
    from ouroboros.projects_registry import create_project, bind_task_to_project
    from ouroboros.terminal_projection import SETTLEMENT_DEFERRED, settle_terminal_projection
    from ouroboros.utils import append_jsonl

    project = create_project(root, 'terminal-times', name='Terminal times')
    queue = DeliveryBus()
    bridge = bus.LocalChatBridge({})
    frames = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr('supervisor.workers.get_event_q', lambda *, create=True: queue)
    monkeypatch.setattr(bus, 'DATA_DIR', root)
    monkeypatch.setattr(bus, 'get_bridge', lambda: bridge)
    monkeypatch.setattr(bus, 'load_state', lambda: {'session_id': 'qualification', 'owner_id': 7})
    monkeypatch.setattr(bus, 'publish_event', lambda *_a, **_k: None)
    monkeypatch.setattr(sender, '_DELIVERED_MESSAGE_IDS', deque(maxlen=256))
    context = SimpleNamespace(DRIVE_ROOT=root, RUNNING={}, send_with_budget=bus.send_with_budget,
                              append_jsonl=append_jsonl)
    results = {}
    for task_id, status, ended in [('old', 'failed', T1), ('new', 'completed', T2), ('legacy', 'failed', None)]:
        bind_task_to_project(root, task_id, project['id'], project['chat_id'], origin={'absent':'system'})
        write_task_result(root, task_id, 'running', task_attempt=0, started_at=T1,
                          project_id=project['id'], chat_id=project['chat_id'])
        monkeypatch.setattr('ouroboros.task_results.utc_now_iso', lambda: ended or T1)
        results[task_id] = write_task_result(root, task_id, status, _terminal_observed=bool(ended),
            result=f'{task_id.upper()}_TASK_ANSWER', terminal_origin='model_final')
    for task_id, added in [('new', T2), ('old', T3), ('legacy', T3)]:
        # Settlement freezes the Main event, its address and its notification time.
        monkeypatch.setattr('ouroboros.terminal_projection.utc_now_iso', lambda: added)
        assert settle_terminal_projection(root, task_id) == SETTLEMENT_DEFERRED
        queued = queue.get_nowait()  # lost before delivery, as on restart
        queue.release_delivery(queued['delivery_id'])  # a restart replaces the bus: no membership survives
        assert queued['progress_meta']['terminal_time'] == results[task_id]['terminal_time']
        # Canonical readiness, not the outbox, re-offers the exact retained event once due.
        assert settle_terminal_projection(root, task_id) == SETTLEMENT_DEFERRED
        replay = queue.get_nowait()
        assert replay == queued
        sender._handle_send_message(replay, context)
        # The strict receipt retires the debt, so a later generation has nothing to republish.
        assert load_task_result(root, task_id)['canonical_terminal_projection']['main_disposition'] == 'delivered'
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
