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
    ctx = SimpleNamespace(active_use_local=False, active_model=PAID, accumulated_usage={'cost': 1.0})
    cap = SimpleNamespace(root_cap_usd=10.0, ceiling_usd=7.0)
    monkeypatch.setattr(loop, '_loop_tree_accounting', lambda **kw: {'settled_usd': 8.0, 'accounted_usd': 8.0})
    assert authored_completion_budget_exhausted(ctx, 10, cap) is False
    # Open holds are not spending (#1487): $8 known with $5 in flight is not exhaustion.
    monkeypatch.setattr(loop, '_loop_tree_accounting', lambda **kw: {'settled_usd': 8.0, 'accounted_usd': 13.0})
    assert authored_completion_budget_exhausted(ctx, 10, cap) is False
    monkeypatch.setattr(loop, '_loop_tree_accounting', lambda **kw: {'settled_usd': 10.0, 'accounted_usd': 10.0})
    assert authored_completion_budget_exhausted(ctx, 10, cap) is True
    # A route that needs paid admission: the fresh wallet at zero owns its stop.
    monkeypatch.setattr('ouroboros.loop_budget._wrapup_global_remaining', lambda: 0)
    assert authored_completion_budget_exhausted(ctx, 10, None) is True


PAID, SUBSCRIPTION = 'openai/gpt-5.2', 'claudexor::codex=gpt-6-astra'
ROUTES = [('local-model', True, False), (SUBSCRIPTION, False, False), (PAID, False, True)]


def test_the_reached_global_limit_owns_an_authored_stop_only_on_a_paid_route(subtests, monkeypatch):
    """Owner 2A: the limit refuses paid admission only, so on a $0 route (local, an unbilled subscription) neither
    the task's wallet snapshot nor a fresh wallet at zero makes its own stop a budget failure; the root cap binds
    every route."""
    for model, local, pays in ROUTES:
        with subtests.test(model=model, local=local, pays=pays), monkeypatch.context() as monkeypatch:
            from ouroboros.loop_budget import authored_completion_budget_exhausted
            import ouroboros.loop as loop
            ctx = SimpleNamespace(active_use_local=local, active_model=model, accumulated_usage={'cost': 0.0})
            monkeypatch.setattr('ouroboros.loop_budget._wrapup_global_remaining', lambda: 10)
            assert authored_completion_budget_exhausted(ctx, 0, None) is pays  # the snapshot the task started with
            monkeypatch.setattr('ouroboros.loop_budget._wrapup_global_remaining', lambda: 0)
            assert authored_completion_budget_exhausted(ctx, 10, None) is pays  # the wallet reached zero meanwhile
            monkeypatch.setattr(loop, '_loop_tree_accounting', lambda **kw: {'settled_usd': 5.0, 'accounted_usd': 5.0})
            assert authored_completion_budget_exhausted(ctx, 10, SimpleNamespace(root_cap_usd=5.0)) is True


@pytest.mark.parametrize('model, local, pays', ROUTES)
def test_an_authored_stop_at_the_reached_global_limit_completes_unless_its_route_pays(
        tmp_path, monkeypatch, model, local, pays):
    """The real loop at remaining $0: the stop the model authored is published as itself on a $0 route; on a paid
    route the money rail still owns it (``budget_exhausted``)."""
    import json
    import queue
    import ouroboros.loop as loop

    monkeypatch.setenv('OUROBOROS_TASK_REVIEW_MODE', 'off')
    monkeypatch.setenv('USE_LOCAL_MAIN', 'true' if local else '')
    stop = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'stop', 'type': 'function', 'function': {
        'name': 'finish_task', 'arguments': json.dumps({'action': 'stop', 'answer': 'Partial answer.',
                                                        'rationale': 'One check remains.'})}}]}
    calls = []
    monkeypatch.setattr(loop, 'call_llm_with_retry', lambda *_a, **_k: calls.append(1) or (stop, 0.0))
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    text, usage, trace = loop.run_llm_loop(
        [{'role': 'user', 'content': 'go'}], registry, SimpleNamespace(default_model=lambda: model), tmp_path / 'logs',
        lambda *_a, **_k: None, queue.Queue(), task_id='stopper', budget_remaining_usd=0.0,
        drive_root=tmp_path)
    assert calls == [1]
    assert (usage.get('reason_code') == 'budget_exhausted') is pays
    assert (usage.get('execution_status') == 'failed') is pays
    assert ('forced_finalization' in trace) is pays
    assert pays or (trace.get('task_completion') or {}).get('action') == 'stop'
    assert 'Partial answer.' in text


def test_resealing_author_stop_retains_queue_generation_comparison():
    from ouroboros.delivery_seal import _end_task_acceptance_fence
    calls = []
    ctx = SimpleNamespace(_task_acceptance_fence_token='owned', _task_acceptance_fence_generation=7,
        end_acceptance_fence=lambda **kw: calls.append(kw) or {'ok': True, 'status': 'sealed'})
    assert _end_task_acceptance_fence(ctx, outcome='author_stop')
    assert ctx._task_acceptance_fence_token is None
    assert _end_task_acceptance_fence(ctx, outcome='author_stop')
    assert calls[-1] == {'token': 'owned', 'outcome': 'author_stop', 'expected_generation': 7}
