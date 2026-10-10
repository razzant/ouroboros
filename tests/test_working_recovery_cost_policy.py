"""Working recovery uses the shared S3 restore authority and returns its decision.

Run with the actual integrated task_pacing module. The common restore resolver
is required; a stand-in resolver or a skipped consumer would prove no integration.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ouroboros import loop, task_pacing, usage_accounting as accounting, working_checkpoint as wc
from ouroboros.contracts.task_contract import build_task_contract
from ouroboros.task_results import task_result_path, write_task_result
from tests.test_context_fit_v664 import _plan
from tests.test_working_checkpoint import _registry

def _saved_start(tmp_path, monkeypatch, new_id, *, saved_usd=7.0):
    source_id, target_id = "saved", "retry" if new_id else "saved"
    write_task_result(tmp_path, source_id, "running")
    dead = _registry(tmp_path, monkeypatch, source_id, 1)
    dead._ctx._cost_ceiling = (None if saved_usd is None else task_pacing.CostCeiling(
        state="active", ceiling_usd=saved_usd, root_cap_usd=50.0, basis="old_stop"))
    limit = SimpleNamespace(tools=dead, messages=_plan().messages_for("max"), llm_trace={"tool_calls": []},
                            accumulated_usage={"cost": 1.0}, round_idx=3, tool_schemas=[],
                            owner_msg_seen=set(), budget_tail="tool")
    wc.save_round(limit, "post_batch")
    task = {"id": source_id}
    assert wc.attach_recovery(tmp_path, task, source_task_id=source_id, from_attempt=1, cause="worker_crash")
    if new_id:
        write_task_result(tmp_path, target_id, "running")
    registry = _registry(tmp_path, monkeypatch, target_id, 2)
    registry._ctx.working_recovery = task["_working_recovery"]
    saved = wc.load_recovery(registry._ctx)
    registry._ctx.task_contract = build_task_contract({"id": target_id, "type": "task"})
    return registry, saved


def _restore(registry, saved, scope, *, wallet=100.0):
    # This is the loop's actual consumer: the returned local must agree with
    # ctx, including None from shared restore resolving as at a fresh start.
    stale = task_pacing.CostCeiling(state="active", ceiling_usd=2.0, basis="stale_local")
    registry._ctx._cost_ceiling = stale
    route = ("same-model", "high", False, "max", 0, registry._ctx.context_fit_plan)
    with accounting.usage_scope(scope):
        result = loop._resume_continuation(registry, ({}, {}, saved), [], {}, {}, set(), route, stale, wallet)
    assert result[6] is registry._ctx._cost_ceiling
    assert result[6] is not stale
    return result[6]


@pytest.mark.parametrize("new_id", [False, True])
def test_working_restore_removes_an_ordinary_saved_default(tmp_path, monkeypatch, new_id):
    registry, saved = _saved_start(tmp_path, monkeypatch, new_id)
    scope = accounting.UsageScope(drive_root=tmp_path, task_id=registry._ctx.task_id,
                                  root_task_id=registry._ctx.task_id, root_limit_usd=50.0)
    restored = _restore(registry, saved, scope)
    assert restored.state == "disabled"
    assert restored.basis == "no_default_cost_stop(saved default stop $7.00 removed)"


@pytest.mark.parametrize("new_id", [False, True])
def test_working_restore_returns_the_actual_producer_nine_not_saved_six(tmp_path, monkeypatch, new_id):
    registry, saved = _saved_start(tmp_path, monkeypatch, new_id, saved_usd=6.0)
    scope = accounting.UsageScope(drive_root=tmp_path, task_id=registry._ctx.task_id,
                                  root_task_id=registry._ctx.task_id, root_limit_usd=50.0,
                                  root_cost_ceiling_usd=9.0)
    restored = _restore(registry, saved, scope)
    assert restored == task_pacing.CostCeiling(state="active", ceiling_usd=9.0,
                                              root_cap_usd=50.0, basis="producer_allowance")


@pytest.mark.parametrize("new_id", [False, True])
def test_working_restore_explicit_without_a_saved_point_resolves_like_start(tmp_path, monkeypatch, new_id):
    registry, saved = _saved_start(tmp_path, monkeypatch, new_id, saved_usd=None)
    registry._ctx.task_contract["budget_profile"] = {"cost_hard_stop_pct": 50}
    scope = accounting.UsageScope(drive_root=tmp_path, task_id=registry._ctx.task_id,
                                  root_task_id=registry._ctx.task_id, root_limit_usd=50.0)
    restored = _restore(registry, saved, scope, wallet=20.0)
    assert restored.state == "active" and restored.ceiling_usd == 10.0
    assert task_pacing.cost_stop_authority(registry._ctx) == "explicit"


@pytest.mark.parametrize("new_id", [False, True])
def test_working_restore_rereads_explicit_then_unreadable_then_ordinary_root(tmp_path, monkeypatch, new_id):
    registry, saved = _saved_start(tmp_path, monkeypatch, new_id)
    root_id = "cost-root"
    scope = accounting.UsageScope(drive_root=tmp_path, task_id=registry._ctx.task_id,
                                  root_task_id=root_id, parent_task_id=root_id,
                                  root_limit_usd=50.0, root_cost_ceiling_usd=7.0)
    write_task_result(tmp_path, root_id, "running", task_contract=build_task_contract({
        "id": root_id, "type": "task", "task_contract": {"budget_profile": {"cost_hard_stop_pct": 50}}}))
    explicit = _restore(registry, saved, scope)
    assert explicit.ceiling_usd == 7.0 and explicit.basis == "old_stop"
    assert task_pacing.cost_stop_authority(registry._ctx) == "explicit"

    task_result_path(tmp_path, root_id).write_text("{")
    unknown = _restore(registry, saved, scope)
    assert unknown.ceiling_usd == 7.0 and unknown.basis == "legacy_policy_unverified(old_stop)"
    assert task_pacing.cost_stop_authority(registry._ctx) == "unknown"

    task_result_path(tmp_path, root_id).unlink()
    write_task_result(tmp_path, root_id, "running", task_contract=build_task_contract({
        "id": root_id, "type": "task", "task_contract": {"budget_profile": {}}}))
    ordinary = _restore(registry, saved, scope)
    assert ordinary.state == "disabled" and "saved default stop $7.00 removed" in ordinary.basis
    assert task_pacing.cost_stop_authority(registry._ctx) == "none"
