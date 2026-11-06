"""Real-consumer discriminators for the remaining Batch4 lifecycle seams."""
from contextlib import contextmanager
from types import SimpleNamespace
import sys

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_batch4_repair_compositions import _running
from tests.test_hurry_initial_lifecycle import pool, _enqueue_origin  # noqa: F401 - pytest fixture
from tests._usage_store_testing import ledger_rows



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
    registry._ctx.task_metadata = {'work_kind': 'owner'}
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
        raise owner_pause.LaunchBusy('contention')
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
        monkeypatch.setattr(owner, 'control_reason', lambda **_kw: next(controls))
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
