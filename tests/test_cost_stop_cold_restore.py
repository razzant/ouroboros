"""A graceful budget pause saved before the upgrade resumes under its real authority (#1128).

Owner 2026-10-07: ordinary tasks on new AND existing installations lose the default
early stop (half the wallet, the margin before the cap); explicit experiment profiles
keep theirs; a producer allowance stays a real restriction at its actual value; an
explicit Resume never widens what was not authored. Each case drives the real pause,
the real exact-Resume grant and ``budget_pause.resume_paused_loop``.
"""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from ouroboros import task_pacing
from ouroboros import usage_accounting as accounting
from tests import _budget_pause_exact_helpers as helpers
from tests._budget_pause_exact_helpers import _install_queue


def _resume(tmp_path, monkeypatch, task_id, ceiling, *, contract=None, scope=None, ledger=None):
    """Pause on the graceful-ceiling rail with ``ceiling`` saved, grant, and resume."""
    from ouroboros import budget_pause, loop_budget, owner_wait
    from supervisor import workers

    queue, state, _workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 500.0)
    original = helpers._loop_ctx

    def with_saved_policy(*args, **kwargs):
        context, limits = original(*args, **kwargs)
        context._cost_ceiling = ceiling
        if contract is not None:
            context.task_contract = contract
        return context, limits

    monkeypatch.setattr(helpers, "_loop_ctx", with_saved_policy)
    _ctx, _limit, _pause_row = helpers._pause(tmp_path, monkeypatch, task_id=task_id,
                                              rail=budget_pause.RAIL_GRACEFUL_CEILING)
    budget_pause.end_dispatch_fence(task_id)
    row = budget_pause.budget_pause_row(tmp_path, task_id)
    marker = budget_pause.exact_pause_marker(row, default_root=task_id)
    task = {"id": task_id, "type": "task", "chat_id": 0, "root_task_id": task_id, "_attempt": 1,
            "_budget_pause": marker}
    workers.PENDING.append(task)
    budget_pause.set_budget_pause(tmp_path, task_id, {**row, "state": budget_pause.STATE_PAUSED})
    assert queue.resume_budget_paused_task(task_id)["ok"] is True
    ctx, _limit = with_saved_policy(tmp_path, task_id)
    ctx._cost_ceiling = None  # a fresh process: nothing restored yet
    ctx.budget_pause_resume = task["_budget_pause_resume"]
    state_blob = budget_pause.load_budget_pause(ctx)
    assert state_blob["cost_ceiling"] == (None if ceiling is None else asdict(ceiling))
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_k: (None, "max"))
    # The Resume refresh reads the ledger NOW: wallet $500, the tree's known $8 (cap $10).
    monkeypatch.setattr(loop_budget, "_wrapup_global_remaining", lambda: 500.0)
    monkeypatch.setattr(loop_budget, "_loop_tree_accounting", lambda **_k: ledger or {
        "settled_usd": 8.0, "accounted_usd": 30.0, "root_limit_usd": 10.0, "age_sec": 0.0})
    messages, usage = [], {}
    with accounting.usage_scope(scope or accounting.UsageScope(
            drive_root=tmp_path, task_id=task_id, root_task_id=task_id, root_limit_usd=10.0)):
        budget_pause.resume_paused_loop(SimpleNamespace(_ctx=ctx), state_blob, messages, {}, usage, set(),
                                        budget_remaining_usd=500.0)
    return ctx, usage["budget_pause_resume"]["threshold_refresh"], messages[-1]["content"]


def test_an_ordinary_pre_upgrade_pause_resumes_without_the_removed_default(tmp_path, monkeypatch):
    """The #1128 shape: paused at $7 of a $10 cap by cap-minus-margin. Resume removes the
    hidden ceiling instead of re-arming it, and the model is told why."""
    legacy = task_pacing.CostCeiling(state="active", ceiling_usd=7.0, root_cap_usd=10.0,
                                     planning_margin_usd=3.0, basis="min(global_pct, root_cap_minus_margin)")
    ctx, refresh, notice = _resume(tmp_path, monkeypatch, "ordinary-1", legacy)
    assert ctx._cost_ceiling.state == task_pacing.COST_CEILING_DISABLED
    assert refresh == {"refreshed": False, "reason": "no_ceiling", "ceiling_state": "disabled",
                       "ceiling_basis": "no_default_cost_stop(saved default stop $7.00 removed)"}
    assert "not refreshed: no_ceiling (no_default_cost_stop(saved default stop $7.00 removed))" in notice
    assert "Cumulative spend, rounds and elapsed execution time were NOT reset" in notice


def test_an_explicit_profile_keeps_its_saved_point_and_the_q10_refresh(tmp_path, monkeypatch):
    explicit = task_pacing.CostCeiling(state="active", ceiling_usd=7.0, root_cap_usd=10.0,
                                       planning_margin_usd=3.0, basis="min(global_pct, root_cap_minus_margin)")
    ctx, refresh, notice = _resume(tmp_path, monkeypatch, "explicit-1", explicit,
                                   contract={"budget_profile": {"cost_hard_stop_pct": 50}})
    # Spend is the tree's KNOWN $8; the $22 of open holds does not shrink the room.
    assert refresh["refreshed"] is True and refresh["spent_usd"] == 8.0
    assert ctx._cost_ceiling.ceiling_usd == pytest.approx(10.0)  # 8 + min(10 - 8, 50% of 500)
    assert ctx._cost_ceiling.basis.startswith("owner_resume_refresh")
    assert "refreshed to $10.00" in notice


def test_a_producer_allowance_is_restored_at_its_value_and_never_widened(tmp_path, monkeypatch):
    """A wake paused under the old $6 (its $9 allowance minus the margin). The restore
    re-reads the producer's $9; the Resume does not stretch it to the $50 cap."""
    old = task_pacing.CostCeiling(state="active", ceiling_usd=6.0, root_cap_usd=50.0,
                                  planning_margin_usd=3.0, basis="min(root_cap_minus_margin, root_ceiling_minus_margin)")
    scope = accounting.UsageScope(drive_root=tmp_path, task_id="wake-1", root_task_id="wake-1",
                                  root_limit_usd=50.0, root_cost_ceiling_usd=9.0)
    ctx, refresh, notice = _resume(tmp_path, monkeypatch, "wake-1", old, scope=scope)
    assert ctx._cost_ceiling == task_pacing.CostCeiling(
        state="active", ceiling_usd=9.0, root_cap_usd=50.0, basis="producer_allowance")
    assert refresh["refreshed"] is False and refresh["reason"] == "producer_allowance_not_expanded"
    assert ctx._cost_ceiling.ceiling_usd == 9.0
    # Said, not implied: the launch-time allowance still binds even if its window has freed more.
    assert "not refreshed: producer_allowance_not_expanded" in notice
    assert "the $9.00 producer allowance this task was launched with still binds" in notice


def test_a_grouped_producer_allowance_survives_an_unreadable_group_projection(tmp_path, monkeypatch):
    """The same wake carrying a whole-work billing group, whose group projection times
    out at restore. Who authored the stop is copied from the scope BEFORE that fallible
    read, so the producer's $9 is restored (never classified as the removed default and
    dropped), and the cap the scope carries stands in for the unreadable projection."""
    from ouroboros import usage_admission

    def store_timeout(*_args, **_kwargs):
        raise TimeoutError("usage store lock timed out")

    monkeypatch.setattr(usage_admission, "task_money_snapshot", store_timeout)
    old = task_pacing.CostCeiling(state="active", ceiling_usd=6.0, root_cap_usd=50.0,
                                  planning_margin_usd=3.0, basis="min(root_cap_minus_margin, root_ceiling_minus_margin)")
    scope = accounting.UsageScope(drive_root=tmp_path, task_id="wake-2", root_task_id="wake-2",
                                  root_limit_usd=50.0, root_cost_ceiling_usd=9.0, billing_group_id="group-wake",
                                  billing_group_limit_usd=50.0, billing_group_limit_source="fixture")
    ctx, refresh, notice = _resume(tmp_path, monkeypatch, "wake-2", old, scope=scope)
    assert ctx._cost_ceiling == task_pacing.CostCeiling(
        state="active", ceiling_usd=9.0, root_cap_usd=50.0, basis="producer_allowance")
    assert refresh["refreshed"] is False and refresh["reason"] == "producer_allowance_not_expanded"
    assert "removed" not in notice


def test_an_unverified_inherited_number_keeps_its_bound_and_is_not_widened(tmp_path, monkeypatch):
    """A pre-upgrade member whose root row cannot be read: its saved $7 stays, labelled,
    and the Resume refresh reports that it needs authority rather than moving it."""
    saved = task_pacing.CostCeiling(state="active", ceiling_usd=7.0, root_cap_usd=10.0,
                                    basis="min(root_resolved_ceiling)")
    scope = accounting.UsageScope(drive_root=tmp_path, task_id="member-1", root_task_id="lost-root",
                                  parent_task_id="lost-root", root_limit_usd=10.0, root_cost_ceiling_usd=7.0)
    ctx, refresh, _notice = _resume(tmp_path, monkeypatch, "member-1", saved, scope=scope)
    assert ctx._cost_ceiling.state == task_pacing.COST_CEILING_ACTIVE and ctx._cost_ceiling.ceiling_usd == 7.0
    assert ctx._cost_ceiling.basis == "legacy_policy_unverified(min(root_resolved_ceiling))"
    assert refresh == {"refreshed": False, "reason": "policy_unverified_not_expanded",
                       "authority": task_pacing.COST_STOP_UNKNOWN, "ceiling_usd": 7.0}


def test_every_restore_re_reads_the_root_authority_never_a_stashed_policy(tmp_path):
    """The policy a start or an earlier restore stashed on the context is never reused:
    each restore reads the root again. Unreadable -> the saved bound stays, labelled;
    readable ordinary root -> the default it was is removed; readable explicit root ->
    the saved point is kept verbatim."""
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.task_results import task_result_path, write_task_result
    from ouroboros.tools.registry import ToolContext

    def root_row(profile):
        task_result_path(tmp_path, "root-2").unlink(missing_ok=True)
        write_task_result(tmp_path, "root-2", "running", metadata={}, task_contract=build_task_contract(
            {"id": "root-2", "type": "task", "task_contract": {"budget_profile": profile}}))

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id, ctx.task_contract = "member-2", build_task_contract({"id": "member-2", "type": "task"})
    saved = asdict(task_pacing.CostCeiling(state="active", ceiling_usd=7.0, root_cap_usd=10.0,
                                           basis="min(root_resolved_ceiling)"))
    scope = accounting.UsageScope(drive_root=tmp_path, task_id="member-2", root_task_id="root-2",
                                  parent_task_id="root-2", root_limit_usd=10.0, root_cost_ceiling_usd=7.0)
    with accounting.usage_scope(scope):
        first = task_pacing.restore_cost_ceiling(ctx, saved)
        assert (first.state, first.ceiling_usd) == ("active", 7.0)
        assert first.basis == "legacy_policy_unverified(min(root_resolved_ceiling))"
        assert task_pacing.cost_stop_authority(ctx) == task_pacing.COST_STOP_UNKNOWN

        root_row({})
        second = task_pacing.restore_cost_ceiling(ctx, saved)
        assert second.state == task_pacing.COST_CEILING_DISABLED
        assert second.basis == "no_default_cost_stop(saved default stop $7.00 removed)"
        assert task_pacing.cost_stop_authority(ctx) == task_pacing.COST_STOP_NONE

        root_row({"cost_hard_stop_pct": 20})
        ctx._cost_stop_policy = {"authority": task_pacing.COST_STOP_NONE}  # a stale stash is ignored
        third = task_pacing.restore_cost_ceiling(ctx, saved)
        assert third == task_pacing.CostCeiling(**saved)
        assert task_pacing.cost_stop_policy(ctx)["profile"]["cost_hard_stop_pct"] == 20
