"""Confirmed worker death retires an exact answer consumer, never its bill."""
import contextlib
import multiprocessing
import os
import queue as stdqueue
import time
import traceback

import pytest

from tests.test_worker_crash_retry import _isolate_worker_crash_state, _reserved_job  # noqa: F401 - autouse isolation fixture

pytestmark = pytest.mark.serial


@pytest.mark.parametrize('evidence', ['exact', 'wrong_birth', 'missing_birth', 'wrong_pid', 'wrong_attempt',
                                     'live', 'replaced_worker', 'write_failed', 'unknown_exit', 'wrong_root', 'wrong_task',
                                     'external'])
def test_confirmed_death_retires_only_exact_consumer(tmp_path, monkeypatch, evidence):
    from ouroboros import usage_accounting as ua, model_wait
    from ouroboros.task_results import write_task_result, load_task_result
    from supervisor import queue, workers, worker_health
    from supervisor.continuation_admission import conflicting_writers
    from tests.test_owner_continue import NONCE
    from supervisor.continuation_admission import admit_continuation

    job, _events = _reserved_job(tmp_path, monkeypatch, exitcode=-9, attempt=7)
    job['skip_respawn'] = True
    task_id = job['task_id']
    handoffs = {'other-work': {'tool': 'opaque', 'state': 'claimed'}} if evidence == 'external' else {}
    write_task_result(tmp_path, task_id, 'running', root_task_id=task_id, chat_id=7,
                      launch_handoffs=handoffs,
                      origin_message_text='Write the Friday report',
                      origin_message_ref={'chat_id': 7, 'client_message_id': 'm-1'},
                      billing_group={'billing_group_id': task_id, 'billing_group_limit_usd': 20.0,
                                     'billing_group_limit_source': 'initial_task_admission',
                                     'billing_group_limit_revision': 'admission-1'})
    monkeypatch.setenv('TOTAL_BUDGET', '1000')
    # The receiver is bound as the worker image would bind it: this pid, another custody session
    # (a row this very image wrote is never retired: tool_custody.written_by_this_image).
    monkeypatch.setattr('ouroboros.tool_custody.owner_proof',
                        lambda: {'pid': os.getpid(), 'kernel_birth': '', 'session': 'the-dead-worker'})
    owner = model_wait.TaskModelWait(task={'id': task_id, '_attempt': 7}, drive_root=tmp_path,
                                    event_queue=None, worker_slot_held=True)
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id=task_id, root_task_id=task_id)):
        with model_wait.operation_wait_scope(owner):
            def send():
                raise RuntimeError('response unknown')
            with pytest.raises(RuntimeError, match='response unknown'):
                ua.execute_physical_attempt(ua.AttemptRequest(model='m', provider='test', reservation_usd=.2), send)
    assert not owner.closed  # Death prevents the ordinary finally from running.
    before = ua.read_usage_records(tmp_path, final_only=True)
    attempt = before[-1]
    assert attempt['state'] == 'unresolved'
    assert attempt['local_answer_consumer_id'] == owner.answer_consumer_id
    birth = attempt.get('local_answer_owner_birth')
    assert birth
    worker = job['worker']
    worker.proc.pid, worker.process_birth, worker.kernel_birth = os.getpid(), birth, ''

    if evidence == 'wrong_birth':
        worker.process_birth = birth + '-other'
    elif evidence == 'missing_birth':
        worker.process_birth = ''
    elif evidence == 'wrong_pid':
        worker.proc.pid += 1
    elif evidence == 'wrong_attempt':
        job['attempt'] += 1
    elif evidence == 'live':
        worker.proc.is_alive.return_value = True
    elif evidence == 'replaced_worker':
        workers.WORKERS[0] = object()
    elif evidence == 'unknown_exit':
        worker.proc.exitcode = None
    elif evidence == 'wrong_root':
        job['task']['root_task_id'] = 'other-root'
    elif evidence == 'wrong_task':
        job['task']['id'] = 'other-task'
    elif evidence == 'write_failed':
        monkeypatch.setattr(model_wait, 'put_handoffs', lambda *_a, **_k: (_ for _ in ()).throw(OSError('disk')))
    monkeypatch.setattr('ouroboros.headless.prepare_terminal_task_files',
                        lambda *_a, **_k: {'terminal_source_present': False})
    worker_health.recover_confirmed_dead_worker(job)
    assert owner.answer_consumer_id not in load_task_result(tmp_path, task_id).get('retired_model_consumers', {}), \
        'the queue-locked door only owes the write'
    from ouroboros import custody_owed
    custody_owed.drain()
    retired = load_task_result(tmp_path, task_id).get('retired_model_consumers', {})
    # A current dead job's joined Process retires every receiver of that process, whatever its task or
    # tree; a stale job, another pid or birth, a live or unjoined process and a failing store do not.
    should_retire = evidence in {'exact', 'external', 'wrong_root', 'wrong_task'}
    assert (owner.answer_consumer_id in retired) == should_retire
    assert load_task_result(tmp_path, task_id)['launch_handoffs'] == handoffs
    assert ua.read_usage_records(tmp_path, final_only=True) == before
    assert any(b['kind'] == 'model_handoff' for b in conflicting_writers(queue, task_id)) != should_retire
    if should_retire:
        # Real Continue admission sees a retired receiver but still unknown money.
        queue.PENDING.clear()
        queue.RUNNING.clear()
        monkeypatch.setattr(queue, 'persist_queue_snapshot', lambda **_k: True)
        monkeypatch.setattr(workers, '_worker_pool_execution_state', lambda: {'available': True})
        result = admit_continuation(task_id, action_nonce=NONCE)
        assert result['ok'] and result['held'] == (evidence == 'external'), result
        assert ua.read_usage_records(tmp_path, final_only=True) == before
        assert not ua.usage_projection(tmp_path, root_task_id=task_id)['cost_final']


def _run_dispatched_consumer(root, task_id):
    """A real spawned worker owns the ledger identity, including in Windows venvs."""
    with (root / 'child.log').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                from ouroboros import usage_accounting as ua, model_wait

                with ua.usage_scope(ua.UsageScope(drive_root=root, task_id=task_id, root_task_id=task_id)):
                    with model_wait.task_model_wait_scope(
                            task={'id': task_id, '_attempt': 3}, drive_root=root,
                            event_queue=None, worker_slot_held=True):
                        def send():
                            (root / 'sent').touch()
                            time.sleep(120)
                        ua.execute_physical_attempt(
                            ua.AttemptRequest(model='m', provider='test', reservation_usd=.2), send)
            except BaseException:
                traceback.print_exc(file=log)
                raise


def test_real_process_death_uses_recorded_birth_and_retains_unknown_money(tmp_path, monkeypatch):
    from ouroboros import usage_accounting as ua
    from ouroboros.task_results import write_task_result, load_task_result
    from supervisor import queue, workers, worker_health, worker_pool_lifecycle
    from supervisor.continuation_admission import conflicting_writers

    task_id = 'physical-consumer'
    write_task_result(tmp_path, task_id, 'running', root_task_id=task_id)
    marker = tmp_path / 'sent'
    monkeypatch.setenv('TOTAL_BUDGET', '1000')
    # Like production Windows workers, spawn bypasses the venv redirector and
    # retains the actual Python child's handle, PID and exit status.
    process = multiprocessing.get_context('spawn').Process(target=_run_dispatched_consumer, args=(tmp_path, task_id))
    process.start()
    try:
        slot = workers.Worker(0, process, stdqueue.Queue(), busy_task_id=task_id)
        monkeypatch.setattr(workers, 'WORKERS', {0: slot})
        task = {'id': task_id, 'root_task_id': task_id, '_attempt': 3, 'type': 'task', 'chat_id': 1}
        workers.RUNNING[task_id] = {'task': task, 'attempt': 3, 'worker_id': 0}
        worker_pool_lifecycle._record_worker_processes()
        assert slot.process_birth
        end = time.monotonic() + 20
        while not marker.exists() and process.is_alive() and time.monotonic() < end:
            time.sleep(.02)
        assert marker.exists(), (tmp_path / 'child.log').read_text(encoding='utf-8')
        before = ua.read_usage_records(tmp_path, final_only=True)
        assert before[-1]['state'] == 'dispatched'
        assert before[-1]['local_answer_owner_pid'] == process.pid
        assert before[-1]['local_answer_owner_birth'] == slot.process_birth
        process.kill()
        process.join(5)
        assert not process.is_alive() and process.exitcode is not None
        workers.ensure_workers_healthy()
        job = queue._reap_queue.get_nowait()
        job['skip_respawn'] = True
        monkeypatch.setattr('ouroboros.headless.prepare_terminal_task_files',
                            lambda *_a, **_k: {'terminal_source_present': False})
        worker_health.recover_confirmed_dead_worker(job)
        from ouroboros import custody_owed
        assert custody_owed.drain() == 0, 'the exit door owed it; the process that knows the death writes it'
        assert before[-1]['local_answer_consumer_id'] in load_task_result(tmp_path, task_id)['retired_model_consumers']
        assert not any(b['kind'] == 'model_handoff' for b in conflicting_writers(queue, task_id))
        assert ua.read_usage_records(tmp_path, final_only=True) == before
    finally:
        if process.is_alive():
            process.kill()
        process.join(5)
        assert not process.is_alive()
        process.close()
