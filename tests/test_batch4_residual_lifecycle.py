"""Real-consumer discriminators for the remaining Batch4 lifecycle seams."""
from contextlib import contextmanager
from types import SimpleNamespace
import sys

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_batch4_repair_compositions import _running
from tests.test_hurry_initial_lifecycle import pool, _enqueue_origin  # noqa: F401 - pytest fixture
from tests._usage_store_testing import ledger_rows

pytestmark = pytest.mark.serial


@pytest.mark.parametrize('tool,args', [
    ('run_command', {'cmd': ['cd']}),
    ('run_command', {'cmd': ['export', 'X=1']}),
    ('run_command', {'cmd': []}),
    ('run_command', {'cmd': '["unclosed'}),
    ('run_command', {'cmd': [sys.executable], 'env': {'BAD=KEY': 'value'}}),
    ('run_command', {'cmd': ['not-an-installed-batch4-executable']}),
    ('run_command', {'cmd': ['no-such-executable', 'argument']}),
    ('run_command', {'cmd': [sys.executable], 'cwd': 'active_workspace/missing'}),
    ('run_command', {'cmd': [sys.executable], 'scratch': ['.']}),
    ('run_script', {'script': '  '}),
    ('run_script', {'script': 'print(1)', 'cwd': 'active_workspace/missing'}),
    ('run_script', {'script': 'print(1)', 'interpreter': 'not-an-installed-batch4-executable'}),
])
def test_pre_effect_refusal_leaves_sleep_pause_continue_usable(tmp_path, monkeypatch, tool, args):
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.task_results import load_task_result
    from ouroboros.model_sleep import cold_blockers
    from supervisor.continuation_admission import conflicting_writers
    from supervisor.owner_pause_control import request_owner_pause
    q, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id = ctx.root_task_id = 'root'
    ctx.task_attempt = 1
    ctx.owner_wait_callback = lambda *_a: 'unknown'
    result = registry.execute_result(tool, args)
    assert result.status != 'ok', result
    assert not load_task_result(tmp_path, 'root').get('launch_handoffs'), result
    assert not cold_blockers(ctx)
    assert not any(b['kind'] == 'tool_handoff' for b in conflicting_writers(q, 'root'))
    # Corrected input uses the real producer; the refused invocation stays closed.
    corrected = registry.execute_result('run_command', {'cmd': [sys.executable, '-c', 'print("corrected")']})
    assert corrected.status == 'ok'
    sleeping = registry.execute_result('await_messages', {'mode': 'cold', 'wake_after_sec': 3600})
    assert 'sleep_armed' in sleeping.text
    workers.RUNNING.clear()
    workers.PENDING.append({'id': 'root', 'root_task_id': 'root', 'type': 'task', 'chat_id': 1})
    assert request_owner_pause('root', request_id='owner-pause')['state'] == 'paused'


@pytest.mark.parametrize('failure', ['unknown', 'timeout', 'post_spawn_missing'])
def test_unwound_command_error_keeps_error_without_ghost_custody(tmp_path, monkeypatch, failure):
    import subprocess
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools import shell
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import conflicting_writers
    q, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = 'root'
    def ambiguous(*_a, **_k):
        raise {'unknown': OSError('after effect'), 'timeout': subprocess.TimeoutExpired('unknown', 1),
               'post_spawn_missing': FileNotFoundError('after effect')}[failure]
    monkeypatch.setattr(shell, '_tracked_subprocess_run', ambiguous)
    result = registry.execute_result('run_command', {'cmd': ['ambiguous']})
    assert result.status != 'ok'
    assert result.meta.get('operation_outcome') != 'completed_no_effect'
    assert not load_task_result(tmp_path, 'root')['launch_handoffs']
    assert not any(b['kind'] == 'tool_handoff' for b in conflicting_writers(q, 'root'))


@pytest.mark.parametrize('origin', ['review', 'evolution', 'assisted', 'restore', 'plain'])
def test_admitted_queued_root_pause_seeds_exact_lifecycle(pool, monkeypatch, origin):  # noqa: F811 - pytest fixture
    from supervisor import queue
    from supervisor.owner_pause_control import request_owner_pause
    from ouroboros.task_results import load_task_result
    task_id = _enqueue_origin(pool, monkeypatch, origin)
    admitted = queue.PENDING[0]
    result = request_owner_pause(task_id, request_id='pause-before-worker')
    assert result['ok'], result
    row = load_task_result(pool.root, task_id, strict=True)
    assert row['status'] == 'scheduled'
    assert row['root_task_id'] == task_id
    assert row['metadata']['billing_group'] == admitted['metadata']['billing_group']
    assert row['owner_pause']['state'] == 'paused'
    assert row['chat_id'] == admitted['chat_id']
    if origin == 'plain':  # only a receipt-less row is seeded from its admitted queue facts
        assert row['description'] == admitted.get('text', '')
    else:  # a host producer's own scheduled receipt is kept, never rewritten
        assert row['host_admission']['status'] == 'accepted'


@pytest.mark.parametrize('reason,hard', [('cancelled', True), ('cancelled', False), ('deadline', False)])
def test_launch_authority_wait_rejoins_typed_controls(tmp_path, monkeypatch, reason, hard):
    from ouroboros import model_wait, loop_round_limits as limits, owner_pause, cancel_intents
    from ouroboros.llm_attempt import _PhysicalSendNotStarted
    from ouroboros.task_results import write_task_result
    write_task_result(tmp_path, 'root', 'running')
    original = model_wait.ModelWaitInterrupted('owner_launch_authority_unavailable', role='main')
    controls = iter([None, reason])
    @contextmanager
    def unavailable(*_a, **_k):
        raise owner_pause.OwnerPauseRefused('owner_launch_authority_unavailable')
        yield
    monkeypatch.setattr(owner_pause, 'launch_admission', unavailable)
    ctx = SimpleNamespace(status_drive_root=tmp_path, drive_root=tmp_path, task_id='root',
                          accumulated_usage={}, llm_trace={}, messages=[], incoming_messages=None, event_queue=None,
                          owner_msg_seen=set(), tools=SimpleNamespace(_ctx=None))
    monkeypatch.setattr(limits, '_drain_incoming_messages', lambda *_a, **_k: {})
    monkeypatch.setattr(limits, '_handle_deadline', lambda *_a, **_k: None, raising=False)
    if hard:
        cancel_intents.request_cancel(tmp_path, 'root')
    with model_wait.task_model_wait_scope(task={'id': 'root'}, drive_root=tmp_path,
                                         event_queue=None, worker_slot_held=hard) as owner:
        monkeypatch.setattr(owner, 'control_reason', lambda: next(controls))
        if reason == 'deadline':
            monkeypatch.setattr(limits._loop(), '_finalize_forced_services', lambda *_a: None)
            monkeypatch.setattr(limits._loop(), '_forced_fallback_result',
                                lambda *_a, **_k: ('deadline', {}, {}))
            result = limits._handle_model_wait_control(ctx, original)
            assert result[0] == 'deadline'
            assert result[2]['forced_finalization']['control_reason'] == 'deadline'
            assert ctx.accumulated_usage['reason_code'] == 'deadline_local'
        else:
            with pytest.raises(model_wait.ModelWaitInterrupted) as caught:
                limits._handle_model_wait_control(ctx, original)
            assert caught.value.control_reason == reason and caught.value.model_role == 'main'
            control = caught.value
            while isinstance(control.previous_error, model_wait.ModelWaitInterrupted):
                control = control.previous_error
            assert isinstance(control.previous_error, _PhysicalSendNotStarted)
    assert not ledger_rows(tmp_path)


@pytest.mark.parametrize('outcome', ['refused', 'unknown', 'dynamic'])
def test_command_refusal_and_unknown_outcomes_reach_real_continue(tmp_path, monkeypatch, outcome):
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools import shell
    from ouroboros.tools.tool_result import ToolResult
    from tests.test_owner_continue import _interrupted, NONCE
    from supervisor.continuation_admission import admit_continuation
    q, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = 'root'
    if outcome == 'unknown':
        monkeypatch.setattr(shell, '_tracked_subprocess_run', lambda *_a, **_k: (_ for _ in ()).throw(OSError('unknown')))
    elif outcome == 'dynamic':
        registry.override_handler('list_available_tools', lambda *_a, **_k: ToolResult(
            status='ok', code='OK', text='claimed complete',
            meta={'operation_outcome': 'completed_no_effect', 'dynamic_provider': True}))
    if outcome == 'dynamic':
        registry.execute_result('list_available_tools', {})
    else:
        registry.execute_result('run_command', {'cmd': ['cd' if outcome == 'refused' else 'unknown']})
    _interrupted(tmp_path, task_id='root', reason_code='worker_crash_signal')
    workers.RUNNING.clear()
    ack = admit_continuation('root', action_nonce=NONCE)
    assert ack['ok'], ack
    assert not ack['held'], "none of these returned bodies owns an independent physical operation"
    assert len(q.PENDING) == 1


def test_real_agent_context_is_lifecycle_bound_even_without_model_wait(tmp_path, monkeypatch):
    from ouroboros.agent import Env, OuroborosAgent
    from ouroboros.owner_pause import member_fence
    from ouroboros.task_results import task_result_path
    repo, drive = tmp_path / 'repo', tmp_path / 'data'
    repo.mkdir()
    for name in ('state', 'logs'):
        (drive / name).mkdir(parents=True)
    (drive / 'state/state.json').write_text('{"spent_usd": 0.0}')
    monkeypatch.setattr(OuroborosAgent, '_log_worker_boot_once', lambda self: None)
    monkeypatch.setattr('ouroboros.agent.build_llm_messages', lambda **_kw: ([], {}))
    agent = OuroborosAgent(Env(repo_dir=repo, drive_root=drive))
    events = []
    monkeypatch.setattr(agent, '_emit_live_log', lambda *a, **kw: events.append((a, kw)))
    ctx, _messages, _caps = agent._prepare_task_context({'id': 'managed', 'type': 'task', 'text': 'x'})
    assert ctx.task_lifecycle_bound is True and ctx.model_wait_context is None
    assert not any('task_lifecycle_bound' in kw for _, kw in events)
    task_result_path(drive, 'managed').unlink()
    assert member_fence(ctx)['state'] == 'unknown'
    result = agent.tools.execute_result('knowledge_read', {'not_an_argument': 1})
    assert result.code == 'OWNER_PAUSE_NOT_STARTED'
    assert not task_result_path(drive, 'managed').exists()


@pytest.mark.parametrize('corruption', ['invalid_json', 'invalid_fence'])
def test_saved_pause_unreadable_tree_is_unknown_not_paused(tmp_path, monkeypatch, corruption):
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from ouroboros.task_results import write_task_result, task_result_path
    q, _, workers = _install_queue(tmp_path, monkeypatch)
    workers.PENDING.append({'id': 'root', 'root_task_id': 'root', 'type': 'task',
                            '_budget_pause': {'reason': 'owner'}, 'chat_id': 1})
    write_task_result(tmp_path, 'root', 'running', owner_pause={'state': 'not-a-state'})
    if corruption == 'invalid_json':
        task_result_path(tmp_path, 'root').write_text('{broken')
    availability = {'complete': True}
    rows = _chat_activities_snapshot_safe(tmp_path, direct_turns=[], availability=availability)
    assert len(rows) == 1 and rows[0]['phase'] == 'unknown'
    assert availability == {'complete': False}


def test_pause_seed_preserves_concurrent_terminal_and_refuses_unadmitted(pool, monkeypatch):  # noqa: F811 - pytest fixture
    from supervisor import queue
    from supervisor.owner_pause_control import request_owner_pause
    from ouroboros import task_results as results
    task_id = _enqueue_origin(pool, monkeypatch, 'plain')  # seeding needs a receipt-less row
    actual = results.write_task_result
    saved = []
    def terminal_race(*args, **kwargs):
        actual(pool.root, task_id, 'completed', result='natural answer', metadata={'exact': 'terminal'})
        saved.append(results.task_result_path(pool.root, task_id).read_bytes())
        return actual(*args, **kwargs)
    monkeypatch.setattr(results, 'write_task_result', terminal_race)
    assert request_owner_pause(task_id, request_id='race')['error'] == 'task_terminal'
    assert results.task_result_path(pool.root, task_id).read_bytes() == saved[0]
    assert request_owner_pause('unadmitted', request_id='no-task')['error'] == 'task_not_live'
    assert not results.task_result_path(pool.root, 'unadmitted').exists()
    assert not queue.BUDGET_ROOT_FENCES


@pytest.mark.parametrize('failure', ['preparation', 'local_spawn', 'docker_spawn', 'docker_map'])
def test_executor_and_script_preparation_refusals_close_real_handoff(tmp_path, monkeypatch, failure):
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools import shell
    from ouroboros import workspace_executor as executor
    from ouroboros.task_results import load_task_result
    q, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    repo, workspace = tmp_path / 'system', tmp_path / 'workspace'
    repo.mkdir()
    workspace.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id = ctx.root_task_id = 'root'
    ctx.workspace_root, ctx.workspace_mode = workspace, 'external'
    ctx.executor_ref = {'type': 'docker_exec' if failure.startswith('docker') else 'local',
                        'id': 'refused', 'container_name': 'never-contacted', 'network': 'none',
                        'workspace_host_path': str(workspace), 'workspace_backend_path': '/workspace'}
    monkeypatch.setattr('ouroboros.safety.check_safety', lambda *_a, **_k: (True, ''))
    if failure == 'preparation':
        monkeypatch.setattr(shell.tempfile, 'mkdtemp', lambda *_a, **_k: (_ for _ in ()).throw(OSError('preparation')))
    elif failure == 'docker_map':
        monkeypatch.setattr(shell, 'executor_map_host_path', lambda *_a: (_ for _ in ()).throw(ValueError('unmapped')))
    elif failure == 'docker_spawn':
        monkeypatch.setattr(executor, '_assert_docker_network_none', lambda *_a: None)
        popen = executor.subprocess.Popen
        def refuse(cmd, *args, **kwargs):
            if cmd[0] == 'docker':
                raise FileNotFoundError('controlled missing Docker CLI; no daemon contacted')
            return popen(cmd, *args, **kwargs)
        monkeypatch.setattr(executor.subprocess, 'Popen', refuse)
    tool = 'run_script' if failure in {'preparation', 'docker_map'} else 'run_command'
    args = {'script': 'print(1)'} if tool == 'run_script' else {'cmd': ['not-an-installed-batch4-executable']}
    result = registry.execute_result(tool, args)
    assert result.status != 'ok', result
    assert not load_task_result(tmp_path, 'root').get('launch_handoffs'), result
    assert not list(workspace.glob('.ouroboros/tmp_scripts/script_*'))


def test_control_seed_keeps_admitted_source_deadline_and_budget(pool):  # noqa: F811 - pytest fixture
    from supervisor import queue
    from supervisor.owner_pause_control import request_owner_pause
    from ouroboros.task_results import load_task_result
    facts = {'origin_message_text': 'Exact owner words', 'origin_message_ref': {'chat_id': 1, 'client_message_id': 'm'},
             'deadline_at': '2099-01-01T00:00:00+00:00', 'root_cost_ceiling_usd': 1.25,
             'objective_author': {'kind': 'owner'}, 'owner_corpus': [{'content': 'Exact owner words'}]}
    admitted = queue.enqueue_task({'id': 'bound', 'root_task_id': 'bound', 'type': 'task', 'chat_id': 1,
                                  'text': 'host work order', **facts})
    assert not admitted.get('_admission_blocked')
    assert request_owner_pause('bound', request_id='pause')['ok']
    row = load_task_result(pool.root, 'bound', strict=True)
    assert all(row[key] == value for key, value in facts.items())
    assert row['metadata']['billing_group'] == admitted['metadata']['billing_group']


def test_paused_split_root_seed_keeps_the_attempt_its_child_copyback_proves(pool):  # noqa: F811 - pytest fixture
    """A Pause seed writes the host attempt key, so the resumed run's child end time still transfers."""
    from ouroboros.agent import OuroborosAgent
    from ouroboros.headless import copy_child_task_result
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.terminal_time import task_attempt_witness, terminal_time_fact
    from supervisor import queue
    from supervisor.owner_pause_control import request_owner_pause

    host, child = pool.root, pool.root / 'child'
    task = queue.enqueue_task({'id': 'split', 'root_task_id': 'split', 'type': 'task', 'chat_id': 1,
                               'text': 'split work', 'drive_root': str(child), 'child_drive_root': str(child),
                               'budget_drive_root': str(host)})
    assert request_owner_pause('split', request_id='pause-split')['state'] == 'paused'
    assert load_task_result(host, 'split', strict=True)['task_attempt'] == task['_attempt']
    assert queue.resume_budget_paused_task('split')['ok']
    actor = SimpleNamespace(env=SimpleNamespace(drive_root=child, budget_drive_root=host), _task_started_ts=1790000000.0)
    OuroborosAgent._persist_running_record(actor, task)
    source = write_task_result(child, 'split', 'completed', _terminal_observed=True, result='Child answer')
    copied = copy_child_task_result(host, task)
    for row in (copied, load_task_result(host, 'split')):
        assert task_attempt_witness(row) == task_attempt_witness(source)
        assert terminal_time_fact(row) == source['terminal_time']
        assert source['terminal_time']['source'] == 'executor_terminal'
