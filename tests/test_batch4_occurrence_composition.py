"""Published occurrence authority composed with Batch4 control, money and custody."""
import copy
from types import SimpleNamespace

import pytest

from ouroboros.task_results import load_task_result, write_task_result
from supervisor import queue, queue_schedules as schedules, schedule_occurrence as occurrence
from tests.test_g1_followup_policy import (  # noqa: F401
    world, register, row, restore, stop, ORIGIN, BINDING, DEADLINE, _assignment_world,
)

pytestmark = pytest.mark.serial


@pytest.mark.parametrize('fault', ['none', 'write', 'readback'])
def test_actual_worker_put_follows_both_verified_dispatch_carriers(world, monkeypatch, fault):  # noqa: F811 - pytest fixture
    world, workers, sent = _assignment_world(world, monkeypatch)
    _, root, pending = world
    scheduled = register(world)
    queue.check_scheduled_tasks()
    [task] = pending
    tid = task['id']
    real_write = schedules._write_scheduled_tasks
    failures = []
    def persist(data, *args, **kw):
        if fault != 'none' and not failures and any((r.get('occurrence') or {}).get('dispatch') == 'possible'
                                                   for r in data['tasks']):
            failures.append(True)
            return False if fault == 'write' else None
        return real_write(data, *args, **kw)
    monkeypatch.setattr(schedules, '_write_scheduled_tasks', persist)
    def put(candidate):
        canonical = load_task_result(root, tid)
        assert canonical['schedule_admission']['dispatch'] == 'possible'
        assert row(root, scheduled['id'])['occurrence']['dispatch'] == 'possible'
        assert canonical['billing_group'] == candidate['metadata']['billing_group'] == BINDING
        assert canonical['deadline_at'] == candidate['deadline_at'] == DEADLINE
        sent.append(copy.deepcopy(candidate))
    workers.WORKERS[0].in_q.put = put
    workers.assign_tasks()
    if fault != 'none':
        assert not sent and pending == [task]
        assert load_task_result(root, tid)['schedule_admission']['dispatch'] == 'possible'
        assert row(root, scheduled['id'])['occurrence'].get('dispatch') != 'possible'
        workers.assign_tasks()  # partial receipt cannot skip the row write on retry
    assert len(sent) == 1 and sent[0]['id'] == tid and not pending


@pytest.mark.parametrize('control', ['pause', 'stop', 'restart'])
def test_control_accepted_during_off_lock_preparation_refuses_admission(world, monkeypatch, control):  # noqa: F811 - pytest fixture
    from ouroboros.owner_pause import install_fence
    from supervisor.followup_policy import record_restart
    _, root, pending = world
    registered = register(world)
    real_prepare = occurrence.prepare
    def prepare(claimed):
        assert not queue._queue_lock._is_owned()
        prepared = real_prepare(claimed)
        if control == 'pause':
            install_fence(root, ORIGIN, request_id='during-prepare')
        elif control == 'stop':
            stop(root)
        else:
            record_restart(root, new=True)
        return prepared
    monkeypatch.setattr(occurrence, 'prepare', prepare)
    queue.check_scheduled_tasks()
    assert not pending
    current = row(root, registered['id'])
    assert not current.get('completed_at') and not current.get('last_task_id')
    assert current.get('followup_hold') or current.get('followup_wait')
    assert not load_task_result(root, current['occurrence']['task_id'])


@pytest.mark.parametrize('evidence', ['none', 'possible', 'missing', 'foreign', 'snapshot_failure'])
def test_resolved_fired_followup_rebinds_only_same_positively_unrun_receipt(world, monkeypatch, evidence):  # noqa: F811 - pytest fixture
    from supervisor.worker_assignment import _claim_worker_launch
    _, root, pending = world
    registered = register(world, 'independent')
    queue.check_scheduled_tasks()
    [task] = pending
    tid = task['id']
    data = schedules.load_schedule_store(root)
    data['tasks'][0].pop('followup_relation')
    schedules._write_scheduled_tasks(data, root)
    # Captured legacy unknown row, already fired with Batch1's exact token.
    saved = load_task_result(root, tid)
    receipt = copy.deepcopy(saved['schedule_admission'])
    receipt['task']['metadata'].pop('followup_relation')
    task['metadata'].pop('followup_relation')
    if evidence == 'possible':
        receipt['dispatch'] = 'possible'
    if evidence == 'foreign':
        receipt['token'] = 'foreign-token'
    write_task_result(root, tid, 'scheduled', schedule_admission={} if evidence == 'missing' else receipt,
                      metadata=copy.deepcopy(task['metadata']))
    original_id = row(root, registered['id'])['occurrence'].copy()
    stop(root)  # an unrecorded relationship is held only by an actual control
    queue.check_scheduled_tasks()
    held = row(root, registered['id'])['followup_hold']['hold_id']
    assert restore(root, registered['id'], held, relation='related')['status'] == 'hold_released'
    real_persist = queue.persist_queue_snapshot
    if evidence == 'snapshot_failure':
        monkeypatch.setattr(queue, 'persist_queue_snapshot', lambda **_kw: False)
    queue.check_scheduled_tasks()
    sent = []
    worker = SimpleNamespace(in_q=SimpleNamespace(put=sent.append))
    if evidence != 'none':
        assert not _claim_worker_launch(queue, task, worker)
        assert not sent and pending == [task]
    if evidence == 'snapshot_failure':
        monkeypatch.setattr(queue, 'persist_queue_snapshot', real_persist)
        queue.check_scheduled_tasks()
    if evidence in {'none', 'snapshot_failure'}:
        saved = load_task_result(root, tid)
        for facet in [task, saved['schedule_admission']['task'], saved]:
            assert facet['metadata']['billing_group'] == BINDING
            assert facet['deadline_at'] == DEADLINE
            assert facet['task_contract']['deadline_at'] == DEADLINE
        assert saved['billing_group'] == BINDING
        assert saved['schedule_admission']['token'] == original_id['token']
        assert _claim_worker_launch(queue, task, worker)
        assert len(sent) == 1 and sent[0]['id'] == tid
    current = row(root, registered['id'])
    assert current['completed_at'] == registered.get('completed_at', current['completed_at'])
    assert current['enabled'] is False and current['last_task_id'] == tid


@pytest.mark.parametrize('owner_hold', [False, True])
def test_saved_schedule_pause_retains_cap_and_unrelated_hold_across_stale_restart(world, monkeypatch, owner_hold):  # noqa: F811 - pytest fixture
    from supervisor.owner_pause_control import request_owner_pause
    from supervisor.queue_snapshot import _retain_snapshot_pending
    from supervisor.followup_policy import record_restart
    world, workers, _sent = _assignment_world(world, monkeypatch)
    _, root, pending = world
    register(world, 'independent')
    queue.check_scheduled_tasks()
    [task] = pending
    tid = task['id']
    assert task['_consciousness_continuation'] is True
    assert request_owner_pause(tid, request_id='saved')['ok']
    # Canonical opacity/continuation survive an older snapshot that lacks them.
    fields = {'_consciousness_continuation': True}
    if owner_hold:
        fields['_owner_hold'] = {'source': 'other-owner', 'revision': 'do-not-release'}
    write_task_result(root, tid, 'scheduled', **fields)
    stale = copy.deepcopy(task)
    stale.pop('_consciousness_continuation', None)
    stale.pop('_owner_hold', None)
    (root / 'state/owner_restart_no_resume.flag').write_text('owner_restart')
    record_restart(root)
    retained, _unseen, consumed, _prior_roots = _retain_snapshot_pending([stale], [], stale=True)
    assert not consumed and len(retained) == 1 and retained[0]['id'] == tid
    assert retained[0]['_consciousness_continuation'] is True
    workers.PENDING[:] = retained
    (root / 'state/owner_restart_no_resume.flag').unlink()
    resumed = queue.resume_budget_paused_task(tid)
    if owner_hold:
        assert not resumed['ok'] and retained[0]['_owner_hold'] == fields['_owner_hold']
    else:
        assert resumed['ok'], resumed
        assert workers.PENDING[0]['id'] == tid


@pytest.mark.parametrize('owner_hold', [False, True])
def test_dispatched_schedule_exact_pause_is_not_ordinary_replay(world, monkeypatch, owner_hold):  # noqa: F811 - pytest fixture
    from supervisor import state
    from supervisor.queue_snapshot import _retain_snapshot_pending
    from supervisor.worker_assignment import _claim_worker_launch
    from tests._budget_pause_exact_helpers import _parked

    world, workers, _sent = _assignment_world(world, monkeypatch)
    _, root, pending = world
    monkeypatch.setattr(state, 'budget_remaining', lambda *_a, **_kw: 5)
    register(world, 'independent')
    queue.check_scheduled_tasks()
    [original] = pending
    tid = original['id']
    assert occurrence.record_dispatch_possible(original)
    pending.clear()
    saved, _pause = _parked(root, monkeypatch, task_id=tid, extra={
        'metadata': original['metadata'], 'admitted_dispatch': 'possible'})
    fields = {'_consciousness_continuation': True}
    if owner_hold:
        fields['_owner_hold'] = {'revision': 'unrelated'}
    write_task_result(root, tid, 'budget_paused', **fields)
    assert occurrence.restore_allowed(copy.deepcopy(original)) is False
    assert occurrence.restore_allowed({'id': tid, 'metadata': {'schedule_id': 'legacy-without-receipt'}}) is False
    restored, _unseen, consumed, _prior_roots = _retain_snapshot_pending([copy.deepcopy(saved)], [], stale=True)
    assert not consumed and len(restored) == 1 and restored[0]['_budget_pause']['exact_continuation']
    workers.PENDING[:] = restored
    resumed = queue.resume_budget_paused_task(tid)
    if owner_hold:
        assert resumed['error'] == 'owner_held'
        return
    assert resumed['ok'], resumed
    assert not queue.resume_budget_paused_task(tid)['ok']  # same single-use grant, never another admission
    sent = []
    assert _claim_worker_launch(queue, restored[0], SimpleNamespace(in_q=SimpleNamespace(put=sent.append)))
    assert [t['id'] for t in sent] == [tid]
    assert sent[0]['_consciousness_continuation'] is True
