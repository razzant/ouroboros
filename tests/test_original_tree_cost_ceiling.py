"""The original early threshold survives actual scheduler/payload/scope handoffs."""

import queue
from dataclasses import replace
from pathlib import Path

import pytest

from ouroboros import task_pacing, usage_accounting as accounting
from ouroboros.contracts.task_contract import build_task_contract, normalize_budget_profile
from ouroboros.task_results import load_task_result
from ouroboros.tools import control
from ouroboros.tools.control_scheduling import _schedule_task
from ouroboros.tools.registry import ToolContext
from supervisor.task_dispatch import build_scheduled_task_payload
from tests.test_available_subagents_runtime import _api_row, _settings


def _schedule(ctx):
    result = _schedule_task(ctx, subagent_id="api-builder", objective="continue the work",
                            expected_output="report", memory_mode="empty")
    assert not result.startswith("⚠️"), result
    event = ctx.event_queue.get_nowait()
    stored = load_task_result(ctx.budget_drive_root or ctx.drive_root, event["task_id"])
    assert stored["root_cost_ceiling_usd"] == event["root_cost_ceiling_usd"]
    payload = build_scheduled_task_payload({**event, "tid": event["task_id"],
        "parent_id": ctx.task_id, "text": event["objective"], "desc": event["objective"]})
    assert payload["metadata"]["root_cost_ceiling_usd"] == event["root_cost_ceiling_usd"]
    return payload


@pytest.mark.parametrize("wallet_recovers", [False, True])
@pytest.mark.parametrize("pct", [0, 20, 50])
def test_three_generations_share_the_original_threshold_while_wallet_changes(tmp_path, monkeypatch, wallet_recovers, pct):
    monkeypatch.setattr(control, "load_settings", lambda: _settings(_api_row()))
    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "4")
    profile = normalize_budget_profile({"cost_hard_stop_pct": pct})
    contract = build_task_contract({"id": "root", "type": "task", "task_contract": {"budget_profile": profile}})
    scope = accounting.UsageScope(drive_root=tmp_path, task_id="root", root_task_id="root",
                                  global_limit_usd=200.0, root_limit_usd=500.0)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id, ctx.task_contract, ctx.event_queue = "root", contract, queue.Queue()
    ctx.task_metadata = {"root_task_id": "root", "task_contract": contract}
    ceilings = []
    hold = None
    for depth in range(3):
        if depth == 1:
            with accounting.usage_scope(scope):
                hold = accounting.reserve_attempt(accounting.AttemptRequest(model="fixture", provider="openai",
                    reservation_usd=40.0, task_id="other", root_task_id="other"))
        elif depth == 2:
            with accounting.usage_scope(scope):
                if wallet_recovers:
                    accounting.release_attempt(hold, "controlled never-sent hold")
                accounting.reserve_attempt(accounting.AttemptRequest(model="fixture", provider="openai",
                    reservation_usd=4.0 if wallet_recovers else 20.0, task_id="other2", root_task_id="other2"))
        with accounting.usage_scope(scope):
            wallet = accounting.usage_projection(tmp_path, global_limit_usd=200.0)["remaining_known_usd"]
            ctx._cost_ceiling = task_pacing.resolve_task_cost_ceiling(ctx, wallet)
            ceilings.append(ctx._cost_ceiling)
            if depth < 2:
                payload = _schedule(ctx)
        if depth < 2:
            scope = replace(scope, task_id=payload["id"], parent_task_id=ctx.task_id,
                            root_cost_ceiling_usd=payload["root_cost_ceiling_usd"])
            ctx.task_id, ctx.task_depth = payload["id"], depth + 1
            ctx.task_metadata, ctx.task_contract = payload["metadata"], payload["task_contract"]
            ctx.drive_root, ctx.budget_drive_root = Path(payload["drive_root"]), tmp_path
    assert [c.ceiling_usd for c in ceilings] == [None if pct == 0 else 200.0 * pct / 100] * 3
    assert all(c.state == ("disabled" if pct == 0 else "active") for c in ceilings)
    if pct:
        assert all("root_resolved_ceiling" in c.basis for c in ceilings[1:])


def test_legacy_child_never_labels_its_own_fallback_as_the_original_root(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load_settings", lambda: _settings(_api_row()))
    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "4")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id, ctx.task_depth, ctx.event_queue = "legacy-child", 1, queue.Queue()
    ctx.task_metadata = {"root_task_id": "root"}
    ctx._cost_ceiling = task_pacing.resolve_cost_ceiling(160.0, normalize_budget_profile(None), non_root_member=True)
    assert ctx._cost_ceiling.ceiling_usd == 80.0
    assert "original_root_ceiling_unavailable" in ctx._cost_ceiling.basis
    assert _schedule(ctx)["root_cost_ceiling_usd"] is None
