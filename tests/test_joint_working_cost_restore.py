"""Independent public-contract checks of S1 working recovery × S3 cost policy.

Run against an actual integrated source tree with its normal pytest environment.
The current money scope/contract supplies authority; a saved basis label does not.
No provider or installed runtime is contacted. Expectations follow S3's agreed
ordinary/explicit/producer and unknown-authority rules, not observed output.
"""
from dataclasses import asdict
from types import SimpleNamespace as NS

import pytest

from ouroboros import loop, task_pacing
from ouroboros import usage_accounting as ua
from ouroboros import working_checkpoint as wc
from ouroboros.contracts.task_contract import build_task_contract
from ouroboros.task_results import write_task_result
from tests.test_context_fit_v664 import _plan
from tests.test_working_checkpoint import _registry


def prepare(tmp_path, monkeypatch, new_id, saved):
    write_task_result(tmp_path, 'prior-task', 'running')
    prior = _registry(tmp_path, monkeypatch, 'prior-task', 1)
    prior._ctx._cost_ceiling = saved
    limits = NS(tools=prior, messages=_plan().messages_for('max'), llm_trace={'tool_calls': []},
                accumulated_usage={'cost': 1.0}, round_idx=3, tool_schemas=[], owner_msg_seen=set(),
                budget_tail='tool')
    wc.save_round(limits, 'post_batch')
    task = {'id': 'prior-task'}
    assert wc.attach_recovery(tmp_path, task, source_task_id='prior-task', from_attempt=1,
                              cause='worker_crash')
    tid = 'successor-task' if new_id else 'prior-task'
    if new_id:
        write_task_result(tmp_path, tid, 'running')
    registry = _registry(tmp_path, monkeypatch, tid, 2)
    registry._ctx.working_recovery = task['_working_recovery']
    return registry, wc.load_recovery(registry._ctx)


def resume(registry, saved):
    # This calls the real consumer branch, not restore_cost_ceiling directly.
    fresh = task_pacing.CostCeiling(state='active', ceiling_usd=3.0, basis='different-start-wallet')
    route = ('same-model', 'high', False, 'max', 0, registry._ctx.context_fit_plan)
    result = loop._resume_continuation(registry, ({}, {}, saved), [], {}, {}, set(), route, fresh, 1.0)
    assert result[6] is registry._ctx._cost_ceiling, 'the resumed local loop and its context must agree'
    return result[6]


@pytest.mark.parametrize('new_id', [False, True])
@pytest.mark.parametrize('policy', ['ordinary', 'explicit', 'producer'])
def test_working_door_uses_current_policy_not_saved_number(tmp_path, monkeypatch, new_id, policy):
    old = task_pacing.CostCeiling(state='active', ceiling_usd=7.0, root_cap_usd=50.0,
                                basis='legacy-saved-point')
    reg, state = prepare(tmp_path, monkeypatch, new_id, old)
    tid = reg._ctx.task_id
    profile = {'cost_hard_stop_pct': 50} if policy == 'explicit' else {}
    reg._ctx.task_contract = build_task_contract({'id': tid, 'type': 'task',
                                                'task_contract': {'budget_profile': profile}})
    scope = ua.UsageScope(drive_root=tmp_path, task_id=tid, root_task_id=tid,
                          root_limit_usd=50.0, root_cost_ceiling_usd=9.0 if policy == 'producer' else None)
    with ua.usage_scope(scope):
        got = resume(reg, state)
    if policy == 'ordinary':
        assert got.state == 'disabled' and got.ceiling_usd is None
    elif policy == 'explicit':
        assert asdict(got) == asdict(old), 'explicit saved point stays 7, not fresh 3'
    else:
        assert got.state == 'active' and got.ceiling_usd == 9.0, 'producer allowance is 9, without old margin'


@pytest.mark.parametrize('new_id', [False, True])
def test_working_door_rechecks_unknown_then_readable_ordinary_authority(tmp_path, monkeypatch, new_id):
    old = task_pacing.CostCeiling(state='active', ceiling_usd=7.0, root_cap_usd=50.0,
                                basis='legacy-inherited')
    reg, state = prepare(tmp_path, monkeypatch, new_id, old)
    tid = reg._ctx.task_id
    reg._ctx.task_contract = build_task_contract({'id': tid, 'type': 'task'})
    scope = ua.UsageScope(drive_root=tmp_path, task_id=tid, root_task_id='unread-root',
                          parent_task_id='unread-root', root_limit_usd=50.0, root_cost_ceiling_usd=7.0)
    with ua.usage_scope(scope):
        first = resume(reg, state)
        assert first.state == 'active' and first.ceiling_usd == 7.0
        assert first.basis.startswith('legacy_policy_unverified')
        assert task_pacing.cost_stop_authority(reg._ctx) == task_pacing.COST_STOP_UNKNOWN
        write_task_result(tmp_path, 'unread-root', 'running', task_contract=build_task_contract(
            {'id': 'unread-root', 'type': 'task'}))
        reg._ctx._cost_stop_policy = {'authority': task_pacing.COST_STOP_UNKNOWN}
        second = resume(reg, state)
        assert second.state == 'disabled' and second.ceiling_usd is None
        assert task_pacing.cost_stop_authority(reg._ctx) == task_pacing.COST_STOP_NONE
