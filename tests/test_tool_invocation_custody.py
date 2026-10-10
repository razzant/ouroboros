"""Local invocation lifetime and independent physical custody through consumers."""
import multiprocessing
import queue as stdqueue
import subprocess
import sys
import threading
import time

import pytest

from tests.test_batch4_producer_custody import _registry, _docker
from tests.test_worker_crash_retry import _isolate_worker_crash_state  # noqa: F401 - autouse reaper isolation

pytestmark = pytest.mark.serial


def _assert_consumers(root, registry, queue, workers, held):
    from ouroboros.model_sleep import cold_blockers
    from ouroboros.owner_pause import read_fence
    from supervisor.owner_pause_control import request_owner_pause, refresh_owner_pause_tree
    from supervisor.continuation_admission import admit_continuation, conflicting_writers
    from tests.test_owner_continue import NONCE, _interrupted

    assert bool(cold_blockers(registry._ctx)) is held
    assert request_owner_pause('root', request_id='custody-proof')['ok']
    workers.RUNNING.clear()
    refresh_owner_pause_tree('root')
    assert read_fence(root, 'root')['state'] == ('requested' if held else 'paused')
    _interrupted(root, 'root', origin={'source': 'owner_restart', 'reason': 'owner_restart', 'scope': 'single'})
    assert bool(conflicting_writers(queue, 'root')) is held
    ack = admit_continuation('root', action_nonce=NONCE)
    assert ack['ok'] and ack['held'] is held, ack
    assert admit_continuation('root', action_nonce=NONCE)['successor_task_id'] == ack['successor_task_id']


@pytest.mark.parametrize('case', ['nonrepo_diff', 'numeric_argument', 'exit_zero', 'exit_nonzero', 'joined_timeout'])
def test_actual_local_completion_releases_all_control_consumers(tmp_path, monkeypatch, case):
    from ouroboros.task_results import load_task_result

    registry, queue, workers = _registry(tmp_path, monkeypatch)
    if case == 'nonrepo_diff':
        result = registry.execute_result('vcs_diff', {'root': 'active_workspace'})
        assert result.code == 'GIT_ERROR'
    elif case == 'numeric_argument':
        result = registry.execute_result('query_code', {'op': 'symbols', 'limit': 'invalid'})
        assert result.status == 'error'
    else:
        program = {'exit_zero': 'pass', 'exit_nonzero': 'raise SystemExit(7)',
                   'joined_timeout': 'import time; time.sleep(30)'}[case]
        result = registry.execute_result('run_command', {'cmd': [sys.executable, '-c', program], 'timeout_sec': 1})
        assert result.code == {'exit_zero': 'OK', 'exit_nonzero': 'SHELL_EXIT_ERROR', 'joined_timeout': 'TOOL_TIMEOUT'}[case]
    assert not load_task_result(tmp_path, 'root').get('launch_handoffs')
    _assert_consumers(tmp_path, registry, queue, workers, False)


@pytest.mark.parametrize('split_drive', [False, True])
@pytest.mark.parametrize('backend,timeout', [('running', False), ('unreadable', False), ('running', True), ('completed', False)])
def test_backend_receipt_owns_custody_after_local_handler_unwinds(tmp_path, monkeypatch, backend, timeout, split_drive):
    from ouroboros import workspace_executor as executor
    from ouroboros.task_results import load_task_result

    registry, queue, workers = _registry(tmp_path, monkeypatch, 'docker_exec')
    if split_drive:
        registry._ctx.drive_root = tmp_path / 'child-execution'
        registry._ctx.drive_root.mkdir()
        registry._ctx.budget_drive_root = tmp_path
    _docker(monkeypatch, backend=backend, timeout=timeout)
    monkeypatch.setattr(executor, 'kill_process_tree', lambda proc: proc.kill())
    result = registry.execute_result('run_command', {'cmd': ['backend-writer']})
    assert result.code != 'TOOL_ERROR', result
    assert not load_task_result(tmp_path, 'root').get('launch_handoffs')
    _assert_consumers(tmp_path, registry, queue, workers, backend != 'completed')
    records = executor._owned_process_records(tmp_path, 'foreground')
    assert bool(records) == (backend != 'completed')
    if records:
        assert records[0][1]['task_id'] == records[0][1]['root_task_id'] == 'root'
    if split_drive:
        assert not list((registry._ctx.drive_root / 'state' / 'workspace_executor_processes').glob('*.json'))


@pytest.mark.parametrize('command_probe', ['measured', 'missing', 'changed'])
@pytest.mark.parametrize('executor_kind', [None, 'local'])
@pytest.mark.parametrize('split_drive', [False, True])
def test_exception_after_spawn_keeps_live_process_then_releases_after_join(tmp_path, monkeypatch, executor_kind, split_drive, command_probe):
    from ouroboros.tools import shell_process
    from ouroboros.task_results import load_task_result
    from ouroboros.model_sleep import cold_blockers

    registry, queue, workers = _registry(tmp_path, monkeypatch, executor_kind)
    if split_drive:
        registry._ctx.drive_root = tmp_path / 'child-execution'
        registry._ctx.drive_root.mkdir()
        registry._ctx.budget_drive_root = tmp_path
    from ouroboros import workspace_executor as executor
    if command_probe != 'measured':
        monkeypatch.setattr(executor, '_process_command_sha256', lambda _pid: 'recorded-command')
    original = subprocess.Popen
    owned = []
    def popen(cmd, **kw):
        proc = original(cmd, **kw)
        if cmd == [sys.executable, '-c', 'import time; time.sleep(30)']:
            owned.append(proc)
            def fail(**_kwargs):
                raise RuntimeError('host pipe read failed after spawn')
            proc.communicate = fail
        return proc
    monkeypatch.setattr(subprocess, 'Popen', popen)
    try:
        result = registry.execute_result('run_command', {'cmd': [sys.executable, '-c', 'import time; time.sleep(30)']})
        assert result.status == 'error'
        assert owned and owned[0].poll() is None
        assert not load_task_result(tmp_path, 'root').get('launch_handoffs')
        if command_probe != 'measured':
            monkeypatch.setattr(executor, '_process_command_sha256',
                                lambda _pid: '' if command_probe == 'missing' else 'changed-command')
        assert any(b['kind'] == 'workspace_executor' for b in cold_blockers(registry._ctx))
        owned[0].kill()
        owned[0].wait(timeout=5)
        assert not cold_blockers(registry._ctx)
        _assert_consumers(tmp_path, registry, queue, workers, False)
    finally:
        for proc in owned:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            shell_process._active_subprocesses.discard(proc)


@pytest.mark.parametrize('receipt_token', ['exact-token', 'other-token'])
def test_control_event_receipt_survives_unwind_and_reconciles_exact_identity(tmp_path, monkeypatch, receipt_token):
    from ouroboros.tools.control_events import _emit_and_wait_for_routing
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.model_sleep import cold_blockers

    registry, queue, workers = _registry(tmp_path, monkeypatch)
    registry._ctx.event_queue = None
    event = {'type': 'promote_chat_to_task', 'task_id': 'promoted', 'routing_token': 'exact-token',
             'client_message_id': 'message', 'objective': 'child work'}
    def body(ctx, **_kw):
        mode, receipt = _emit_and_wait_for_routing(ctx, event)
        assert mode == 'deferred' and receipt['status'] == 'unconfirmed'
        return 'OK'  # Business success cannot erase the pending host receipt.
    registry.override_handler('knowledge_read', body)
    registry.execute_result('knowledge_read', {'topic': 'x'})
    claim = next(iter(load_task_result(tmp_path, 'root')['launch_handoffs'].values()))
    assert claim['state'] == 'returned' and claim['control_receipts'][0]['routing_token'] == 'exact-token'
    assert cold_blockers(registry._ctx)
    write_task_result(tmp_path, 'promoted', 'scheduled',
                      promotion_admission={'status': 'scheduled', 'routing_token': receipt_token})
    _assert_consumers(tmp_path, registry, queue, workers, receipt_token != 'exact-token')


def test_logical_timeout_does_not_close_a_physically_running_handler(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result
    from ouroboros.model_sleep import cold_blockers
    from ouroboros.tools.tool_result import ToolResult

    registry, _, _ = _registry(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    def body(*_a, **_kw):
        entered.set()
        assert release.wait(5)
        return ToolResult(status='error', code='TOOL_ERROR', text='done',
                          meta={'operation_outcome': 'unknown', 'dynamic_provider': True})
    registry.override_handler('knowledge_read', body)
    thread = threading.Thread(target=registry.execute_result, args=('knowledge_read', {'topic': 'x'}))
    thread.start()
    try:
        assert entered.wait(5)
        thread.join(.01)  # Only the logical caller stopped waiting.
        assert thread.is_alive() and load_task_result(tmp_path, 'root')['launch_handoffs']
        assert cold_blockers(registry._ctx)
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert not load_task_result(tmp_path, 'root')['launch_handoffs']
    assert not cold_blockers(registry._ctx)


def _blocked_worker(root):
    from ouroboros.model_wait import task_model_wait_scope
    from ouroboros.tools.registry import ToolRegistry
    registry = ToolRegistry(repo_dir=root, drive_root=root)
    registry._ctx.task_id = registry._ctx.root_task_id = 'root'
    def body(*_a, **_kw):
        (root / 'inside').touch()
        time.sleep(120)
    registry.override_handler('knowledge_read', body)
    with task_model_wait_scope(task={'id': 'root', '_attempt': 3}, drive_root=root,
                               event_queue=None, worker_slot_held=True):
        registry.execute_result('knowledge_read', {'topic': 'x'})


@pytest.mark.parametrize('door', ['pool_restart', 'cancel_custody'])
def test_real_owner_restart_retires_exact_worker_invocation(tmp_path, monkeypatch, door):
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor import worker_pool_lifecycle
    from supervisor.continuation_admission import admit_continuation
    from tests.test_owner_continue import NONCE

    registry, queue, workers = _registry(tmp_path, monkeypatch)
    task = workers.RUNNING['root']['task']
    write_task_result(tmp_path, 'root', 'running', origin_message_text='Finish the report',
        origin_message_ref={'chat_id': 7, 'client_message_id': 'original'},
        billing_group={'billing_group_id': 'root', 'billing_group_limit_usd': 20.0,
            'billing_group_limit_source': 'initial_task_admission', 'billing_group_limit_revision': 'original'})
    task['_attempt'] = 3
    workers.RUNNING['root'].update(attempt=3, worker_id=0)
    child = multiprocessing.get_context('spawn').Process(target=_blocked_worker, args=(tmp_path,))
    child.start()
    try:
        slot = workers.Worker(0, child, stdqueue.Queue(), busy_task_id='root')
        monkeypatch.setattr(workers, 'WORKERS', {0: slot})
        worker_pool_lifecycle._record_worker_pids()
        end = time.monotonic() + 20
        while not (tmp_path / 'inside').exists() and child.is_alive() and time.monotonic() < end:
            time.sleep(.02)
        assert (tmp_path / 'inside').exists()
        claim_id, before = next(iter(load_task_result(tmp_path, 'root')['launch_handoffs'].items()))
        assert before['local_owner'] == {'pid': child.pid, 'process_birth': slot.process_birth, 'task_attempt': 3}
        monkeypatch.setattr(workers, '_reconcile_confirmed_dead_review_owner', lambda *_a: None)
        if door == 'pool_restart':
            assert workers.kill_workers(result_reason='Owner Restart', terminal_status='cancelled',
                                        stop_source='owner_restart', preserve_pending=True)
        else:
            from ouroboros.cancel_intents import request_cancel
            request_cancel(tmp_path, 'root', source='owner_restart', reason='Owner Restart')
            monkeypatch.setattr(workers, 'respawn_worker', lambda *_a, **_kw: False)
            assert queue.cancel_task_custody('root', deliver=False) == queue.CANCEL_CANCELLED
        child.join(5)
        assert not child.is_alive() and child.exitcode is not None
        row = load_task_result(tmp_path, 'root')
        assert not row['launch_handoffs']
        assert row['retired_tool_invocations'][claim_id]['effect_outcome'] == 'unknown'
        assert row['retired_tool_invocations'][claim_id]['replay_authorized'] is False
        observed = registry.execute_result('get_task_result', {'task_id': 'root'})
        assert claim_id in observed.text and 'unknown' in observed.text
        from ouroboros.tools.join_ledger import _child_result_sha256
        unchanged = registry.execute_result('get_task_result', {
            'task_id': 'root', 'known_result_sha256': _child_result_sha256(row)})
        assert claim_id in unchanged.text and 'unknown' in unchanged.text
        monkeypatch.setattr(workers, '_worker_pool_execution_state', lambda: {'available': True})
        ack = admit_continuation('root', action_nonce=NONCE)
        assert ack['ok'] and not ack['held'], ack
    finally:
        if child.is_alive():
            child.kill()
        child.join(5)
        child.close()


@pytest.mark.parametrize('outcome', ['returned_error', 'raised'])
def test_every_builtin_uses_the_same_actual_handler_lifetime_boundary(tmp_path, monkeypatch, outcome):
    from ouroboros.owner_pause import tool_handoff
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.tool_result import ToolResult

    registry, _, _ = _registry(tmp_path, monkeypatch)
    names = tuple(registry._entries)
    invoked = []
    monkeypatch.setattr(registry, '_invalidate_advisory_if_worktree_changed', lambda *_a: None)
    for name in names:
        def body(*_a, **_kw):
            invoked.append(name)
            if outcome == 'raised':
                raise RuntimeError('handler really unwound')
            return ToolResult(status='error', code='TOOL_ERROR', text='business failed',
                              meta={'operation_outcome': 'unknown'})
        registry.override_handler(name, body)
        with tool_handoff(registry._ctx, name) as claim:
            registry._invoke_builtin_handler(name, registry._entries[name], {}, None, None, None, claim)
        assert not load_task_result(tmp_path, 'root').get('launch_handoffs'), name
    assert invoked == list(names) and len(names) > 100


@pytest.mark.parametrize('evidence', ['exact', 'wrong_birth', 'wrong_attempt', 'wrong_root',
                                     'wrong_pid', 'live', 'unknown_exit', 'replaced_worker', 'replaced_meta', 'legacy'])
def test_death_retires_only_attributed_local_invocation(tmp_path, monkeypatch, evidence):
    import os
    from tests.test_worker_crash_retry import _reserved_job
    from ouroboros.model_wait import TaskModelWait
    from ouroboros.task_results import write_task_result, load_task_result
    from supervisor import workers, worker_health
    from supervisor.continuation_admission import conflicting_writers

    _registry(tmp_path, monkeypatch)
    job, _events = _reserved_job(tmp_path, monkeypatch, exitcode=-9, attempt=3)
    owner = TaskModelWait(task={'id': job['task_id'], '_attempt': 3}, drive_root=tmp_path,
                          event_queue=None, worker_slot_held=True)
    worker = job['worker']
    worker.proc.pid, worker.process_birth = os.getpid(), owner.answer_owner_birth
    identity = {'pid': os.getpid(), 'process_birth': owner.answer_owner_birth, 'task_attempt': 3}
    claim = {'tool': 'local', 'state': 'claimed', 'task_id': job['task_id'],
             'root_task_id': job['task_id'], 'local_owner': identity}
    if evidence == 'legacy':
        claim.pop('local_owner')
    elif evidence in {'wrong_birth', 'wrong_attempt', 'wrong_pid'}:
        field = {'wrong_birth': 'process_birth', 'wrong_attempt': 'task_attempt', 'wrong_pid': 'pid'}[evidence]
        identity[field] = 'foreign' if field == 'process_birth' else identity[field] + 1
    elif evidence == 'wrong_root':
        claim['root_task_id'] = 'other'
    elif evidence == 'live':
        worker.proc.is_alive.return_value = True
    elif evidence == 'unknown_exit':
        worker.proc.exitcode = None
    elif evidence == 'replaced_worker':
        workers.WORKERS[0] = object()
    elif evidence == 'replaced_meta':
        workers.RUNNING[job['task_id']] = dict(job['meta'])
    independent = {**claim, 'local_owner': {**identity, 'process_birth': 'independent'}}
    write_task_result(tmp_path, job['task_id'], 'running', root_task_id=job['task_id'],
                      launch_handoffs={'matching': claim, 'independent': independent})
    with workers._queue_lock:
        worker_health._retire_dead_model_consumers(job)
    after = load_task_result(tmp_path, job['task_id'])
    assert ('matching' not in after['launch_handoffs']) == (evidence == 'exact')
    assert after['launch_handoffs']['independent'] == independent
    workers.RUNNING.clear()
    from supervisor import queue
    assert any(x['kind'] == 'tool_handoff' for x in conflicting_writers(queue, job['task_id']))


def test_executor_census_reports_unreadable_directory(tmp_path, monkeypatch):
    import os
    from ouroboros.model_sleep import cold_blockers

    registry, _, _ = _registry(tmp_path, monkeypatch)
    folder = tmp_path / 'state' / 'workspace_executor_processes'
    assert not cold_blockers(registry._ctx)  # Absent custody store is empty.
    folder.mkdir(exist_ok=True)
    real_scan, real_list = os.scandir, os.listdir
    real_iterdir = type(folder).iterdir
    seen = []
    def denied_scan(path):
        seen.append(os.fspath(path))
        if os.fspath(path) == str(folder):
            raise PermissionError('custody directory unreadable')
        return real_scan(path)
    def denied_list(path):
        seen.append(os.fspath(path))
        if os.fspath(path) == str(folder):
            raise PermissionError('custody directory unreadable')
        return real_list(path)
    def denied_iterdir(path):
        if path == folder:
            raise PermissionError('custody directory unreadable')
        return real_iterdir(path)
    with monkeypatch.context() as patch:
        # Path.iterdir uses a captured accessor in Python 3.10; patch its public
        # seam too, so the same I/O failure reaches both supported implementations.
        patch.setattr(type(folder), 'iterdir', denied_iterdir)
        patch.setattr(os, 'scandir', denied_scan)
        patch.setattr(os, 'listdir', denied_list)
        blockers = cold_blockers(registry._ctx)
        assert any(row['kind'] == 'tree_custody_unreadable' and
                   'workspace_executor_custody_unreadable' in row['detail']
                   for row in blockers), (blockers, seen, str(folder))
    assert not cold_blockers(registry._ctx)  # Readable empty store still works.


@pytest.mark.parametrize("known", [False, True])
def test_effective_retry_read_keeps_predecessor_unknown_effect_history(tmp_path, known):
    from ouroboros.task_results import write_task_result, load_task_result
    from ouroboros.tool_custody import retire_tool_invocations
    from ouroboros.tools.control_task_results import _get_task_result
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.join_ledger import _child_result_sha256

    claim = {"tool": "local-effect", "task_id": "old", "root_task_id": "old", "state": "claimed",
             "local_owner": {"pid": 880001, "process_birth": "owned", "task_attempt": 1}}
    write_task_result(tmp_path, "old", "running", root_task_id="old", launch_handoffs={"original-op": claim})
    retire_tool_invocations(tmp_path, "old", "old", pid=880001, process_birth="owned", task_attempt=1)
    write_task_result(tmp_path, "new", "completed", result="Retry result", original_task_id="old",
                      timeout_retry_from="old", root_task_id="old")
    write_task_result(tmp_path, "old", "failed", reason_code="idle_timeout", superseded_by="new",
                      retry_task_id="new")
    assert "original-op" in load_task_result(tmp_path, "old")["retired_tool_invocations"]
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    data = load_task_result(tmp_path, "new")
    result = _get_task_result(ctx, "old", **({"known_result_sha256": _child_result_sha256(data)} if known else {}))
    assert "original-op" in result and "external effects remain unknown" in result, result
