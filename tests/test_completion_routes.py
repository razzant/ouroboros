"""Completion capability and cold migration preserve host authority and exact bytes."""
import copy
import hashlib
from types import SimpleNamespace

import pytest

from ouroboros.loop_delivery import DeliveryCandidate, completion_schema, selected_completion_text
from ouroboros.owner_wait import continuation_state, restore_continuation_state
from ouroboros.presence_authority import presence_ceiling_payload
from ouroboros.tools.registry import ToolRegistry
from tests.test_presence_runner import _admission


@pytest.mark.parametrize('profile', ['local_readonly_subagent', 'acting_subagent'])
def test_inherited_presence_child_gets_only_local_completion(tmp_path, profile):
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id, ctx.parent_task_id = 'child', 'parent'
    ctx.task_metadata = {'parent_task_id': 'parent', 'delegation_role': 'subagent'}
    ceiling = presence_ceiling_payload(_admission().capability_ceiling)
    ctx.task_contract = {'capability_ceiling': ceiling}
    ctx.task_constraint = {'mode': profile, 'write_surface': 'external_workspace', 'write_root': str(tmp_path)}
    before = copy.deepcopy(ceiling)
    schema = registry.get_schema_by_name('finish_task')
    assert schema is not None
    assert 'pending_review' not in schema['function']['parameters']['properties']
    assert 'acceptance_subject' in schema['function']['parameters']['properties']
    result = registry.execute('finish_task', {'action': 'stop', 'answer': 'Partial', 'rationale': 'Not finished'})
    assert 'completion_requested' in result
    assert ctx.task_contract['capability_ceiling'] == before
    assert 'pending_review is available only on root tasks' in registry.execute('finish_task', {'action': 'finish', 'answer': 'x', 'pending_review': 'finish'})
    assert registry.get_schema_by_name('send_user_message') is None


def test_presence_speaker_and_disabled_tool_keep_authority(tmp_path):
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_contract = {'capability_ceiling': presence_ceiling_payload(_admission().capability_ceiling)}
    ctx.task_metadata = {'presence': {'binding_id': 'a'*32}}
    assert registry.get_schema_by_name('finish_task') is None
    assert 'PRESENCE_CAPABILITY_BLOCKED' in registry.execute('finish_task', {'action': 'finish', 'answer': 'No'})
    ctx.task_metadata = {}
    ctx.task_contract = {'disabled_tools': ['finish_task']}
    assert registry.get_schema_by_name('finish_task') is None
    assert 'completion_requested' not in registry.execute('finish_task', {'action': 'finish', 'answer': 'No'})
    schemas = []
    assert 'finish_task' not in completion_schema(registry, schemas)
    assert schemas == []


def test_nano_materializes_permitted_completion_only_on_hold(tmp_path):
    from ouroboros.tool_policy import initial_tool_schemas
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.messages = []
    schemas = initial_tool_schemas(registry, context_mode='nano')
    assert not any(row['function']['name'] == 'finish_task' for row in schemas)
    original = copy.deepcopy(schemas)
    assert completion_schema(registry, schemas) == 'finish_task'
    assert schemas[:-1] == original and schemas[-1]['function']['name'] == 'finish_task'
    assert registry._ctx.messages[-1]['role'] == 'user'
    completion_schema(registry, schemas)
    assert len(schemas) == len(original) + 1


def test_cold_restore_discards_old_repair_budget_and_retains_exact_selected_sources(tmp_path):
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id, ctx.task_attempt = 'cold', 1
    ctx._delivery_candidate = DeliveryCandidate('kept', hashlib.sha256(b'kept').hexdigest(), 1, 1, 'e', {},
        finalization_control='awaiting_control', control_episode_seen=True)
    held = 'Whole held response'
    ctx._completion_held_sha256 = hashlib.sha256(held.encode()).hexdigest()
    ctx._completion_request = {'action': 'stop', 'answer_sha256': ctx._completion_held_sha256,
        'rationale': 'unfinished', 'observation': {'tool_count': 3}}
    state = continuation_state(ctx, [{'role': 'assistant', 'content': held}], {'tool_calls': []}, {}, 2, [], set())
    state['delivery_candidate']['repair_attempted'] = True
    restored = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    messages, trace, usage, seen = [], {}, {}, set()
    restore_continuation_state(restored, state, messages, trace, usage, seen)
    assert restored._ctx._delivery_control_required is True
    assert not hasattr(restored._ctx._delivery_candidate, 'repair_attempted')
    assert restored._ctx._completion_request == ctx._completion_request
    assert selected_completion_text(restored._ctx._completion_request, restored._ctx._delivery_candidate,
        messages, restored._ctx._completion_held_sha256) == (held, '')
    assert selected_completion_text(restored._ctx._completion_request, restored._ctx._delivery_candidate,
        [], restored._ctx._completion_held_sha256)[0] is None


def test_authored_stop_money_check_uses_real_cap_not_planning_margin(monkeypatch):
    from ouroboros.loop_budget import authored_completion_budget_exhausted
    import ouroboros.loop as loop
    monkeypatch.setattr('ouroboros.loop_budget._wrapup_global_remaining', lambda: 10)
    ctx = SimpleNamespace(active_use_local=False, accumulated_usage={'cost': 1.0})
    cap = SimpleNamespace(root_cap_usd=10.0, ceiling_usd=7.0)
    monkeypatch.setattr(loop, '_loop_tree_accounting', lambda **kw: {'settled_usd': 8.0, 'accounted_usd': 8.0})
    assert authored_completion_budget_exhausted(ctx, 10, cap) is False
    # Open holds are not spending (#1487): $8 known with $5 in flight is not exhaustion.
    monkeypatch.setattr(loop, '_loop_tree_accounting', lambda **kw: {'settled_usd': 8.0, 'accounted_usd': 13.0})
    assert authored_completion_budget_exhausted(ctx, 10, cap) is False
    monkeypatch.setattr(loop, '_loop_tree_accounting', lambda **kw: {'settled_usd': 10.0, 'accounted_usd': 10.0})
    assert authored_completion_budget_exhausted(ctx, 10, cap) is True
    monkeypatch.setattr('ouroboros.loop_budget._wrapup_global_remaining', lambda: 0)
    assert authored_completion_budget_exhausted(ctx, 10, None) is True


def test_resealing_author_stop_retains_queue_generation_comparison():
    from ouroboros.loop_acceptance import _end_task_acceptance_fence
    calls = []
    ctx = SimpleNamespace(_task_acceptance_fence_token='owned', _task_acceptance_fence_generation=7,
        end_acceptance_fence=lambda **kw: calls.append(kw) or {'ok': True, 'status': 'sealed'})
    assert _end_task_acceptance_fence(ctx, outcome='terminal')
    assert ctx._task_acceptance_fence_token is None
    assert _end_task_acceptance_fence(ctx, outcome='author_stop')
    assert calls[-1] == {'token': 'owned', 'outcome': 'author_stop', 'expected_generation': 7}
