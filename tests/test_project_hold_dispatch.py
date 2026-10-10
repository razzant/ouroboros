"""Project recovery must distinguish accepted custody from possible worker dispatch."""
import copy

import pytest

from ouroboros.task_results import load_task_result
from supervisor import queue, workers
from tests.test_project_hold_recovery import accepted, restore_unreadable, resume_after_app_stop, worker
from tests.test_swarm_host_admission import host  # noqa: F401

pytestmark = pytest.mark.serial


def test_failed_running_mirror_and_stale_pending_snapshot_cannot_replay(host, tmp_path, monkeypatch):  # noqa: F811
    from supervisor import worker_assignment

    accepted(host, tmp_path)
    assert queue.persist_queue_snapshot()
    old_snapshot = queue.QUEUE_SNAPSHOT_PATH.read_bytes()
    sent = worker(host, monkeypatch)
    monkeypatch.setattr(worker_assignment, '_mirror_assigned_running_status', lambda _task: None)
    # Physical handoff succeeds; then the process dies before replacing PENDING.
    persist = queue.persist_queue_snapshot
    with monkeypatch.context() as patch:
        patch.setattr(queue, 'persist_queue_snapshot', lambda **kw:
                      False if kw.get('reason') == 'assign_task' else persist(**kw))
        workers.assign_tasks()
    assert [row['id'] for row in sent] == ['held']
    stored = load_task_result(host.root, 'held')
    assert stored['status'] == 'scheduled' and not stored.get('started_at')
    assert stored['admitted_dispatch'] == 'possible'
    queue.RUNNING.clear()
    workers.WORKERS[0].busy_task_id = None
    queue.QUEUE_SNAPSHOT_PATH.write_bytes(old_snapshot)
    queue.restore_pending_from_snapshot()
    # Force the Project outage/recovery composition over that stale row.
    path, original = restore_unreadable(host)
    path.write_bytes(original)
    workers.assign_tasks()
    assert len(sent) == 1
    assert host.pending[0]['_project_admission_restore_hold']


@pytest.mark.parametrize('boundary', ['snapshot', 'snapshot_none', 'result_false', 'result_raise', 'readback'])
def test_failed_pre_handoff_proof_never_sends(host, tmp_path, monkeypatch, boundary):  # noqa: F811
    from supervisor import task_admission

    accepted(host, tmp_path)
    sent = worker(host, monkeypatch)
    with monkeypatch.context() as patch:
        if boundary in {'snapshot', 'snapshot_none'}:
            patch.setattr(queue, 'persist_queue_snapshot', lambda **_kw: False if boundary == 'snapshot' else None)
        elif boundary == 'result_false':
            patch.setattr(task_admission, 'write_task_result', lambda *_a, **_kw: False)
        else:
            def unavailable(*_a, **_kw):
                raise OSError('synthetic pre-handoff failure')
            patch.setattr(task_admission, 'write_task_result' if boundary == 'result_raise'
                          else 'load_task_result', unavailable)
        workers.assign_tasks()
    # A refused claim snapshot sent nothing and keeps its never-sent fact (Batch4:
    # reserved-but-unsent work is not sent); a failed later proof keeps 'possible'.
    assert not sent and host.pending[0]['admitted_dispatch'] == ('none' if boundary.startswith('snapshot') else 'possible')
    assert 'held' not in queue.RUNNING
    # This process still owns the unhanded row; its normal assignment can retry
    # persistence. A restored row below has no such local custody proof.
    workers.assign_tasks()
    assert [row['id'] for row in sent] == ['held']


@pytest.mark.parametrize('proof', ['absent', None, 'possible', ['malformed']])
def test_legacy_or_unknown_dispatch_does_not_backfill_on_restore(host, tmp_path, monkeypatch, proof):  # noqa: F811
    import json

    accepted(host, tmp_path)
    row = host.pending[0]
    if proof == 'absent':
        row.pop('admitted_dispatch', None)
    else:
        row['admitted_dispatch'] = proof
    sent = worker(host, monkeypatch)
    for _ in range(2):
        assert queue.persist_queue_snapshot()
        host.pending.clear()
        queue.restore_pending_from_snapshot()
        workers.assign_tasks()
        assert not sent and host.pending[0]['_project_admission_restore_hold']
        saved = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))['pending'][0]['task']
        if proof == 'absent':
            assert 'admitted_dispatch' not in saved
        else:
            assert saved['admitted_dispatch'] == proof
    assert 'automatic recovery is not authorized' in host.pending[0]['_project_admission_restore_hold']['detail']
    assert host.pending[0]['_project_admission_restore_hold']['reason'] == 'project_dispatch_unconfirmed'


@pytest.mark.parametrize('basis', ['absent', None, ['malformed'], 'legacy', 'incomplete'])
def test_unknown_original_identity_never_recaptures_today(host, tmp_path, monkeypatch, basis):  # noqa: F811
    accepted(host, tmp_path)
    path, original = restore_unreadable(host)
    row = host.pending[0]
    if basis == 'absent':
        row.pop('_project_admission')
    elif basis == 'legacy':
        row['_project_admission']['legacy_basis'] = True
    elif basis == 'incomplete':
        row['_project_admission']['project'].pop('created_at')
    else:
        row['_project_admission'] = basis
    before = copy.deepcopy(row)
    path.write_bytes(original)
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent
    assert row.get('_project_admission') == before.get('_project_admission')
    assert 'original Project identity' in row['_project_admission_restore_hold']['detail']
    assert 'automatic recovery is not authorized' in row['_project_admission_restore_hold']['detail']


@pytest.mark.parametrize('kind', ['crash', 'timeout'])
@pytest.mark.parametrize('outage', [False, True])
def test_retry_owner_admits_but_unknown_handoff_cannot_auto_release(host, tmp_path, monkeypatch, kind, outage):  # noqa: F811
    from ouroboros.task_results import write_task_result
    from supervisor import task_reaper, worker_health
    from tests.test_worker_crash_retry import _make_worker

    task = accepted(host, tmp_path)
    host.pending.clear()
    write_task_result(host.root, 'held', 'running')
    if kind == 'timeout':
        from tests.test_retry_project_binding import _patch_retry_input_handoff
        _patch_retry_input_handoff(monkeypatch)
        retry = task_reaper._enqueue_retry(queue, task, task_id='held', retry_task_id='retry',
            attempt=1, terminal_reason='idle_timeout', recon_fields={})
        assert retry[0], retry
        assert load_task_result(host.root, 'held')['superseded_by'] == 'retry'
        assert load_task_result(host.root, 'retry')['supersedes_task_id'] == 'held'
    else:
        dead = _make_worker(busy_task_id='held', exitcode=1)
        dead.reaping = True
        workers.WORKERS[0] = dead
        meta = {'task': task, 'attempt': 1, 'worker_id': 0}
        queue.RUNNING['held'] = meta
        monkeypatch.setattr(workers, 'QUEUE_MAX_RETRIES', 1)
        monkeypatch.setattr(workers, 'reconstruct_task_cost', lambda *_a, **_kw: {})
        monkeypatch.setattr(workers, 'respawn_worker', lambda *_a: None)
        job = dict(worker=dead, task_id='held', task=task, meta=meta, worker_id=0,
                   exitcode=1, attempt=1, drive_root=str(host.root))
        worker_health._recover_crashed_task_without_terminal(job, queue)
    assert len(host.pending) == 1
    assert host.pending[0]['admitted_dispatch'] == 'possible'
    sent = worker(host, monkeypatch)
    if outage:
        path, original = restore_unreadable(host)
        path.write_bytes(original)
    workers.assign_tasks()
    if outage:
        assert not sent and host.pending[0]['_project_admission_restore_hold']
        assert 'no-dispatch evidence is unconfirmed' in host.pending[0]['_project_admission_restore_hold']['detail']
    else:
        assert [row['id'] for row in sent] == ['retry' if kind == 'timeout' else 'held']


@pytest.mark.parametrize('change', ['chat_id', 'created_at', 'routing_incarnation',
                                  'routing_generation', 'working_dir'])
@pytest.mark.parametrize('frozen', [False, True])
def test_hold_keeps_identity_and_frozen_prepared_resource(host, tmp_path, monkeypatch, change, frozen):  # noqa: F811
    import json

    task = accepted(host, tmp_path)
    task['_project_admission']['frozen'] = frozen
    prepared = copy.deepcopy(task)
    path, original = restore_unreadable(host)
    data = json.loads(original)
    row = data['projects'][0]
    row[change] = row.get(change, 0) + 1 if change in {'chat_id', 'routing_generation'} else 'changed'
    path.write_text(json.dumps(data), encoding="utf-8")
    sent = worker(host, monkeypatch)
    if frozen and change in {'routing_generation', 'working_dir'}:
        resume_after_app_stop(host, 'held')
    workers.assign_tasks()
    if frozen and change in {'routing_generation', 'working_dir'}:
        assert [row['id'] for row in sent] == ['held']
        for key in ('_project_admission', 'workspace_root', 'drive_root', 'text'):
            assert sent[0][key] == prepared[key]
    else:
        assert not sent and host.pending[0]['_terminalization_retry']


def test_benign_activity_keeps_full_tuple_and_prepared_resource(host, tmp_path, monkeypatch):  # noqa: F811
    from ouroboros import projects_registry as registry

    prepared = copy.deepcopy(accepted(host, tmp_path))
    path, original = restore_unreadable(host)
    path.write_bytes(original)
    registry.update_project(host.root, 'target', name='Renamed')
    registry.create_project(host.root, 'neighbour')
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent, 'registry recovery alone must not release work held after Quit/crash'
    resume_after_app_stop(host, 'held')
    workers.assign_tasks()
    assert [row['id'] for row in sent] == ['held']
    assert sent[0]['workspace_root'] == prepared['workspace_root']
    assert sent[0]['_project_admission'] == prepared['_project_admission']
    assert load_task_result(host.root, 'held')['status'] == 'running'
