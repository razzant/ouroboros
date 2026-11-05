"""Confirmed worker death retires an exact answer consumer, never its bill."""
import os

import pytest

from tests.test_worker_crash_retry import _isolate_worker_crash_state, _reserved_job  # noqa: F401 - autouse isolation fixture



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
