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
    explicit = normalize_budget_profile({"cost_hard_stop_pct": 50})
    ctx._cost_ceiling = task_pacing.resolve_cost_ceiling(160.0, explicit, non_root_member=True)
    assert ctx._cost_ceiling.ceiling_usd == 80.0
    assert "original_root_ceiling_unavailable" in ctx._cost_ceiling.basis
    assert _schedule(ctx)["root_cost_ceiling_usd"] is None


def test_an_ordinary_root_hands_its_children_no_number(tmp_path, monkeypatch):
    """Owner 2026-10-07: an ordinary root has no early stop, so the real scheduler
    stamps nothing a child could later mistake for an authored ceiling."""
    monkeypatch.setattr(control, "load_settings", lambda: _settings(_api_row()))
    contract = build_task_contract({"id": "root", "type": "task"})
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id, ctx.task_contract, ctx.event_queue = "root", contract, queue.Queue()
    ctx.task_metadata = {"root_task_id": "root", "task_contract": contract}
    scope = accounting.UsageScope(drive_root=tmp_path, task_id="root", root_task_id="root",
                                  global_limit_usd=480.0, root_limit_usd=400.0)
    with accounting.usage_scope(scope):
        ctx._cost_ceiling = task_pacing.resolve_task_cost_ceiling(ctx, 480.0)
        payload = _schedule(ctx)
    assert ctx._cost_ceiling.state == task_pacing.COST_CEILING_DISABLED
    assert payload["root_cost_ceiling_usd"] is None


def _root_row(tmp_path, *, profile=None, metadata=None, contract=True):
    """The canonical RUNNING row a root's start writes (``agent._persist_running_record``)."""
    from ouroboros.task_results import write_task_result

    fields = {"metadata": dict(metadata or {})}
    if contract:
        fields["task_contract"] = build_task_contract(
            {"id": "root", "type": "task", "task_contract": {"budget_profile": profile or {}}})
    write_task_result(tmp_path, "root", "running", **fields)


def _member(tmp_path, inherited):
    """A child scheduled before the upgrade: an inherited number, no percentage of its own."""
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id, ctx.task_contract = "child", build_task_contract({"id": "child", "type": "task"})
    scope = accounting.UsageScope(drive_root=tmp_path, task_id="child", root_task_id="root",
                                  parent_task_id="root", root_limit_usd=400.0, root_cost_ceiling_usd=inherited)
    with accounting.usage_scope(scope):
        return ctx, task_pacing.resolve_task_cost_ceiling(ctx, 480.0)


def test_a_pre_upgrade_child_of_an_ordinary_root_loses_the_removed_default(tmp_path):
    """The #1128 shape: $239.66 was half the wallet the ROOT saw. The root's own
    canonical contract proves it had no explicit percentage and no producer, so the
    child keeps no hidden ceiling; only the hard limits bind."""
    _root_row(tmp_path)
    ctx, ceiling = _member(tmp_path, 239.66)
    assert ceiling.state == task_pacing.COST_CEILING_DISABLED
    assert ceiling.basis == task_pacing.COST_BASIS_NO_DEFAULT_STOP
    assert task_pacing.cost_stop_authority(ctx) == task_pacing.COST_STOP_NONE


def test_a_child_of_an_explicit_root_keeps_the_inherited_number(tmp_path):
    _root_row(tmp_path, profile={"cost_hard_stop_pct": 20})
    ctx, ceiling = _member(tmp_path, 39.0)
    assert ceiling.state == task_pacing.COST_CEILING_ACTIVE and ceiling.ceiling_usd == 39.0
    assert "root_resolved_ceiling" in ceiling.basis
    policy = task_pacing.cost_stop_policy(ctx)
    assert policy["authority"] == task_pacing.COST_STOP_EXPLICIT
    assert policy["profile"]["cost_hard_stop_pct"] == 20  # the Resume refresh's percentage


def test_a_child_of_a_wake_uses_the_producer_allowance_not_an_old_margin(tmp_path):
    """A wake started with $9 of allowance; the pre-upgrade child inherited $6 ($9 minus
    the old margin). The real restriction is the producer's $9, never widened further."""
    _root_row(tmp_path, metadata={"root_cost_ceiling_usd": 9.0, "initiator": "consciousness"})
    ctx, ceiling = _member(tmp_path, 6.0)
    assert ceiling.state == task_pacing.COST_CEILING_ACTIVE and ceiling.ceiling_usd == 9.0
    assert ceiling.planning_margin_usd is None
    assert task_pacing.cost_stop_authority(ctx) == task_pacing.COST_STOP_PRODUCER


@pytest.mark.parametrize("damage", ["missing", "no_contract", "corrupt"])
def test_unreadable_root_authority_keeps_the_inherited_number_provisionally(tmp_path, damage):
    """Neither the default removed on a guess nor an authored stop widened: the
    member keeps the number it already carries, labelled ``legacy_policy_unverified``,
    and the next start re-reads the root once it is readable again."""
    from ouroboros.task_results import task_result_path

    if damage == "no_contract":
        _root_row(tmp_path, contract=False)
    elif damage == "corrupt":
        path = task_result_path(tmp_path, "root")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
    ctx, ceiling = _member(tmp_path, 239.66)
    assert ceiling.state == task_pacing.COST_CEILING_ACTIVE and ceiling.ceiling_usd == 239.66
    assert ceiling.basis == task_pacing.COST_BASIS_POLICY_UNVERIFIED
    assert ceiling.root_cap_usd == 400.0 and ceiling.planning_margin_usd is None
    assert task_pacing.cost_stop_authority(ctx) == task_pacing.COST_STOP_UNKNOWN
    assert "legacy_policy_unverified" in task_pacing.cost_ceiling_disclosure(ceiling)["rule"]
    # Once the root's authority is readable (an ordinary root), the next start drops it.
    task_result_path(tmp_path, "root").unlink(missing_ok=True)
    _root_row(tmp_path)
    _ctx, later = _member(tmp_path, 239.66)
    assert later.state == task_pacing.COST_CEILING_DISABLED


def test_an_unreadable_root_with_no_inherited_number_invents_no_ceiling(tmp_path):
    """Nothing inherited, nothing to keep: the member resolves like any ordinary task
    (no stop), without reading the missing root at all."""
    ctx, ceiling = _member(tmp_path, None)
    assert ceiling.state == task_pacing.COST_CEILING_DISABLED and ceiling.ceiling_usd is None
    assert ceiling.basis == task_pacing.COST_BASIS_NO_DEFAULT_STOP
    assert task_pacing.cost_stop_authority(ctx) == task_pacing.COST_STOP_NONE
