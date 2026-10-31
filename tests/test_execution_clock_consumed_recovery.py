"""One current durable consumption repairs a captured clock; no retry authority is added."""
import copy

import pytest

from ouroboros import budget_pause
from ouroboros.model_wait import execution_elapsed_seconds
from supervisor import queue, workers
from supervisor.events_budget import fold_consumed_resume
from tests.test_execution_clock_resume_handover import (  # noqa: F401
    _fresh_plan, _granted_after_capacity, _loop_once, _working_then_paused, clock,
)
from tests.test_resume_handover import _DeadProcess
from tests.test_swarm_host_admission import host  # noqa: F401

pytestmark = pytest.mark.serial


def _consumed(host, monkeypatch, clock, *, consume=True):  # noqa: F811
    plan = _fresh_plan(monkeypatch)
    pause = _working_then_paused(host, monkeypatch, clock, 3600, plan)
    task = _granted_after_capacity(host, monkeypatch, clock, pause)
    captured = copy.deepcopy(workers.RUNNING['held'])  # before consumption, like the timeout capture
    if consume:
        _loop_once(host, monkeypatch, task, plan=plan)
    clock[0] += 10
    return captured


@pytest.mark.parametrize('change', ['none', 'unconsumed', 'newer_pause', 'foreign_attempt', 'foreign_root',
    'revoked', 'grant', 'generation', 'pause_generation', 'source', 'unreadable', 'event_attempt', 'event_grant'])
def test_consumption_fold_requires_current_authority_and_runs_once(host, monkeypatch, clock, change):  # noqa: F811
    meta = _consumed(host, monkeypatch, clock, consume=change != 'unconsumed')
    row = budget_pause.budget_pause_row(host.root, 'held')
    event = None
    if change == 'unreadable':
        from tests._store_shared import unreadable_result
        unreadable_result(monkeypatch, 'held')
    elif change.startswith('event_'):
        resume = meta['task']['_budget_pause_resume']
        event = dict(task_attempt=2, pause_id=resume['pause_id'], grant_id=resume['grant_id'])
        event['task_attempt' if change == 'event_attempt' else 'grant_id'] = 99
    elif change not in {'none', 'unconsumed'}:
        if change == 'newer_pause':
            row['pause_id'] = 'newer'
        elif change == 'foreign_attempt':
            row['task_attempt'] += 1
        elif change == 'foreign_root':
            row['root_task_id'] = 'another-root'
        elif change == 'revoked':
            row['grant']['revoked_at'] = 'revoked'
        elif change == 'grant':
            row['grant']['grant_id'] = 'another-grant'
        elif change == 'generation':
            row['grant']['generation'] += 1
        elif change == 'pause_generation':
            row['pause_generation'] += 1
        elif change == 'source':
            row['source_ref']['sha256'] = '0' * 64
        budget_pause.set_budget_pause(host.root, 'held', row)
    before = copy.deepcopy(meta)
    # This addressed row's nested root copy is metadata, not Resume admission.
    folds = change in {'none', 'foreign_root'}
    assert fold_consumed_resume(meta, host.root, event) is folds
    if folds:
        assert execution_elapsed_seconds(meta, clock[0]) == pytest.approx(90)
        assert 'sleep_parked_at' not in meta
        folded = copy.deepcopy(meta)
        assert not fold_consumed_resume(meta, host.root) and meta == folded
    else:
        assert meta == before and execution_elapsed_seconds(meta, clock[0]) == pytest.approx(80)


def test_planned_restart_stamps_consumption_lost_with_worker(host, monkeypatch, clock):  # noqa: F811
    _consumed(host, monkeypatch, clock)
    process = _DeadProcess()
    workers.WORKERS[0].proc = process
    monkeypatch.setattr(workers, 'kill_worker_tree', process.kill)
    monkeypatch.setattr(workers, '_kill_survivors', lambda: None)
    assert workers.kill_workers(force=True, preserve_pending=True, preserve_running_task_ids={'held'})
    [successor] = host.pending
    assert successor['_execution_handoff_sec'] == pytest.approx(90)
    assert successor['_attempt'] == 3
    # The spent selection's existing custody remains; this change does not admit it.
    assert successor['_budget_pause_resume']['grant_id']


def test_assignment_tick_already_folds_lost_consumption(host, monkeypatch, clock):  # noqa: F811
    _consumed(host, monkeypatch, clock)
    assert workers.RUNNING['held'].get('sleep_parked_at')
    workers.assign_tasks()
    assert 'sleep_parked_at' not in workers.RUNNING['held']
    assert execution_elapsed_seconds(workers.RUNNING['held'], clock[0]) == pytest.approx(90)


def test_consumed_worker_death_obeys_the_ordinary_crash_retry_limit(host, monkeypatch, clock):  # noqa: F811
    from supervisor import worker_health
    from ouroboros.task_results import load_task_result
    from tests.test_worker_crash_retry import _make_worker

    _consumed(host, monkeypatch, clock)
    meta = workers.RUNNING['held']
    worker = _make_worker(busy_task_id='held', exitcode=1)
    worker.reaping = True
    workers.WORKERS[0] = worker
    import queue as stdqueue
    jobs = stdqueue.Queue()
    worker.reaping = False
    monkeypatch.setattr(workers, '_LAST_SPAWN_TIME', 0)
    monkeypatch.setattr(workers, 'CRASH_TS', [])
    monkeypatch.setattr(queue, '_reap_queue', jobs)
    monkeypatch.setattr(queue, '_ensure_reaper_started', lambda: None)
    workers.ensure_workers_healthy()
    job = jobs.get_nowait()
    assert job['kind'] == 'confirmed_dead_worker' and job['meta'] is meta
    monkeypatch.setattr(workers, 'respawn_worker', lambda *_a, **_k: None)
    monkeypatch.setattr('ouroboros.headless.prepare_terminal_task_files',
                        lambda *_a, **_k: {'terminal_source_present': False})
    worker_health.recover_confirmed_dead_worker(job)
    result = load_task_result(host.root, 'held')
    assert result['status'] == 'failed' and result['reason_code'] == 'worker_crash_retry_exhausted'
    assert not host.pending


@pytest.mark.parametrize("resume", [None, {"authority": "explicit_resume", "grant_id": "consumed"}])
def test_inapplicable_resume_clock_does_not_read_storage(tmp_path, monkeypatch, resume):
    def unexpected_read(*_args, **_kwargs):
        pytest.fail("an ordinary or already-folded clock needs no durable observation")

    monkeypatch.setattr(budget_pause, "budget_pause_row", unexpected_read)
    meta = {"task": {"id": "ordinary"}, "budget_paused_sec": 12.0}
    if resume is not None:
        meta["task"]["_budget_pause_resume"] = resume
    before = copy.deepcopy(meta)
    assert not fold_consumed_resume(meta, tmp_path) and meta == before
