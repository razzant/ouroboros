"""Discriminators for Pause submission and producer-owned completion facts."""
import sys

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_batch4_repair_compositions import _running



@pytest.mark.parametrize('service', [False, True])
def test_pause_during_process_preparation_prevents_process_start(tmp_path, monkeypatch, service):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools import shell_outputs
    from ouroboros.owner_pause import install_fence
    from ouroboros.task_results import load_task_result

    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = 'root'
    marker = tmp_path / 'effect'
    def preparation(*_a, **_k):
        install_fence(tmp_path, 'root', request_id='during-preparation')
        return {}
    # Both process doors take their output baseline through the shell_outputs owner.
    monkeypatch.setattr(shell_outputs, '_snapshot_declared_outputs', preparation)
    args = {'cmd': [sys.executable, '-c', f'from pathlib import Path; Path({str(marker)!r}).touch()']}
    if service:
        args['name'] = 'held'
    result = registry.execute_result('start_service' if service else 'run_command', args)
    assert not marker.exists(), result
    # The process was not launched, but its tool handler/preparation did run.
    assert result.code == 'OWNER_OPERATION_REFUSED', result
    assert result.meta['owner_pause_not_started'] is False
    assert result.meta.get('operation_outcome') != 'completed_no_effect'
    assert not load_task_result(tmp_path, 'root').get('launch_handoffs')


def test_bound_continue_offer_retries_exact_admission_after_snapshot_failure(tmp_path, monkeypatch):
    from tests.test_owner_continue import _interrupted, NONCE
    from ouroboros.owner_continue import continuation_offer
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation

    q, _, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)
    persist = q.persist_queue_snapshot
    monkeypatch.setattr(q, 'persist_queue_snapshot', lambda **_k: False)
    failed = admit_continuation('pred-1', action_nonce=NONCE)
    assert not failed['ok'] and not workers.PENDING
    offer = continuation_offer(load_task_result(tmp_path, 'pred-1'), 'pred-1')
    assert offer['state'] == 'bound' and offer['action_nonce'] == NONCE
    monkeypatch.setattr(q, 'persist_queue_snapshot', persist)
    ack = admit_continuation('pred-1', action_nonce=offer['action_nonce'])
    assert ack['ok'] and ack['successor_task_id'] == failed['successor_task_id']
    assert len(workers.PENDING) == 1
    admitted = continuation_offer(load_task_result(tmp_path, 'pred-1'), 'pred-1')
    assert admitted['state'] == 'admitted' and 'action_nonce' not in admitted


def test_focused_node_control_consumers():
    import pathlib
    import shutil
    from tests.group_run import run_in_group
    root = pathlib.Path(__file__).resolve().parents[1]
    node = shutil.which('node')
    assert node, 'focused frontend verification requires the provided Node runtime'
    result = run_in_group([node, '--test', 'web/tests/task_continue.test.js',
                           'web/tests/chat_activity_block.test.js', 'web/tests/activity_schedule_time.test.js'],
                          cwd=root, timeout=60)
    print(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr


def test_stop_conflict_refusal_settles_only_its_tool_claim(tmp_path):
    from ouroboros.cancel_intents import request_cancel, active_intent
    from ouroboros.task_results import write_task_result, load_task_result
    from ouroboros.tools.registry import ToolRegistry
    write_task_result(tmp_path, 'parent', 'running')
    write_task_result(tmp_path, 'child', 'running', parent_task_id='parent', root_task_id='parent')
    request_cancel(tmp_path, 'child', source='owner', stop_action_id='retained-action')
    original = active_intent(tmp_path, 'child')
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = 'parent'
    result = registry.execute_result('cancel_task', {'task_id': 'child', 'stop_action_id': 'retained-action'})
    assert result.status == 'blocked' and result.code == 'STOP_ACTION_CONFLICT'
    assert result.meta['operation_outcome'] == 'completed_no_effect'
    assert not load_task_result(tmp_path, 'parent').get('launch_handoffs')
    assert active_intent(tmp_path, 'child') == original
