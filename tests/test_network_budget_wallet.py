"""The existing final-call price is compared with live shared-wallet evidence."""

from dataclasses import replace

import pytest

from ouroboros import task_pacing, usage_accounting as accounting
from ouroboros.loop_budget import _wrapup_global_remaining


def _request(amount, task="other"):
    return accounting.AttemptRequest(model="fixture-model", provider="openai",
        reservation_usd=amount, task_id=task, root_task_id=task)


def _spend(amount, task="other"):
    """KNOWN spend: the only money the wallet and its fence decide on (#1487)."""
    held = accounting.reserve_attempt(_request(amount, task))
    accounting.mark_dispatched(held)
    accounting.settle_attempt(held, {}, cost_usd=amount, cost_final=True)
    return held


@pytest.mark.parametrize("root_cap", [None, 50.0])
def test_known_global_spend_leaves_one_same_route_final_then_the_atomic_fence_binds(tmp_path, root_cap):
    scope = accounting.UsageScope(drive_root=tmp_path, task_id="main", root_task_id="main",
                                  global_limit_usd=10.0, root_limit_usd=root_cap)
    with accounting.usage_scope(scope):
        _spend(8.0)
        request = _request(1.5, "main")
        remaining = _wrapup_global_remaining()
        assert remaining == 2.0
        args = dict(request=request, root_cap_usd=root_cap, deciding_usd=0.0,
                    global_remaining_usd=remaining)
        # An explicit profile's probe math is unchanged: one final fits, two do not.
        assert task_pacing.wrapup_reservation_fits(**args) is True
        assert task_pacing.wrapup_reservation_fits(**args, reservation_count=2) is False
        final = accounting.reserve_attempt(request)
        assert _wrapup_global_remaining() == 2.0  # the final's own hold is exposure, not spending
        unknown = accounting.reserve_attempt(_request(1.0))
        accounting.mark_dispatched(unknown)
        accounting.mark_unresolved(unknown, "provider outcome unknown")
        assert _wrapup_global_remaining() == 2.0  # an unknown outcome is neither spending nor a refund
        accounting.mark_dispatched(final)
        accounting.settle_attempt(final, {}, cost_usd=2.0, cost_final=False)  # an estimate is known
        assert _wrapup_global_remaining() == 0.0
        with pytest.raises(accounting.BudgetExceeded) as refused:
            accounting.reserve_attempt(request)
        assert refused.value.limit_scope == "global"


@pytest.mark.parametrize("global_remaining,root_cap,deciding,expected", [
    (20.0, 2.0, 1.0, False), (1.0, 50.0, 0.0, False),
    (2.0, None, None, True), (0.0, None, None, False), (None, None, None, None),
])
def test_all_known_remainders_bind_and_unknown_tree_spend_is_not_invented(
    tmp_path, global_remaining, root_cap, deciding, expected,
):
    with accounting.usage_scope(accounting.UsageScope(drive_root=tmp_path, task_id="main", root_task_id="main")):
        assert task_pacing.wrapup_reservation_fits(request=_request(1.5, "main"), root_cap_usd=root_cap,
            deciding_usd=deciding, global_remaining_usd=global_remaining) is expected


def test_a_concurrent_reservation_after_the_check_can_still_refuse_the_final(tmp_path):
    scope = accounting.UsageScope(drive_root=tmp_path, task_id="main", root_task_id="main", global_limit_usd=2.0)
    with accounting.usage_scope(scope):
        request = _request(1.5, "main")
        assert task_pacing.wrapup_reservation_fits(request=request, root_cap_usd=None, deciding_usd=0,
                                                  global_remaining_usd=_wrapup_global_remaining()) is True
        _spend(2.0)  # another task's known spend lands after the observation
        with pytest.raises(accounting.BudgetExceeded):
            accounting.reserve_attempt(request)  # observation did not reserve a share


def test_explicit_scope_limit_and_canonical_root_own_the_wallet_read(tmp_path):
    scope = accounting.UsageScope(drive_root=tmp_path, task_id="main", root_task_id="main", global_limit_usd=10.0)
    with accounting.usage_scope(scope):
        _spend(3.0)
        assert _wrapup_global_remaining() == 7.0
    with accounting.usage_scope(replace(scope, global_limit_usd=20.0)):
        assert _wrapup_global_remaining() == 17.0


def test_projection_failure_and_torn_ledger_are_not_a_known_zero(tmp_path, monkeypatch):
    scope = accounting.UsageScope(drive_root=tmp_path, task_id="main", root_task_id="main", global_limit_usd=10.0)
    with accounting.usage_scope(scope):
        monkeypatch.setattr(accounting, "usage_projection", lambda *_a, **_k: {
            "remaining_known_usd": 0.0, "integrity_degraded": True})
        assert _wrapup_global_remaining() is None
        def fail(*args, **kwargs):
            raise OSError("ledger unavailable")
        monkeypatch.setattr(accounting, "usage_projection", fail)
        assert _wrapup_global_remaining() is None


def test_global_remaining_is_disclosed_without_fabricating_a_tree_amount():
    ceiling = task_pacing.CostCeiling(state="active", ceiling_usd=5.0)
    text = task_pacing.wrapup_last_fit_text(None, ceiling, 2.0)
    assert "spend is unavailable" in text and "$2.00 left across all tasks" in text


def test_the_stop_sentence_opens_with_the_bound_that_binds():
    """An owner whose WALLET ran dry at $125 of a $400 cap must not read a per-task-cap story."""
    capped = task_pacing.CostCeiling(state="active", ceiling_usd=239.66, root_cap_usd=400.0)
    wallet = task_pacing.wrapup_last_fit_text(125.661, capped, 21.834)
    assert wallet.startswith("The shared Total budget is nearly used up: $21.83 left across all tasks.")
    assert "$125.66 of its own $400.00 cap, so the task cap is not what stopped it" in wallet
    assert wallet.endswith("Raise Total budget in Settings for more room.")
    cap = task_pacing.wrapup_last_fit_text(390.0, capped, 500.0)
    assert cap.startswith("This task's tree spent $390.00 of its own $400.00 cap; the shared Total budget still has $500.00.")
    assert "Raise Total budget" not in cap
    assert task_pacing.wrapup_unaffordable_text(125.661, capped, 0.4).startswith("The shared Total budget is nearly used up")


def test_local_final_call_path_does_not_request_an_unneeded_wallet_projection(tmp_path, monkeypatch):
    from ouroboros import loop
    from tests.test_tree_cost_ceiling import _ctx

    reads = []
    monkeypatch.setattr(accounting, "usage_projection", lambda *_a, **_k: reads.append(True) or {})
    monkeypatch.setattr(accounting, "_ROOT_ACCOUNTING_TELEMETRY", {})
    scope = accounting.UsageScope(drive_root=tmp_path, task_id="local", root_task_id="local", global_limit_usd=10.0)
    with accounting.usage_scope(scope):
        accounting._stash_root_accounting("local", {"settled_usd": 0.0, "accounted_usd": 0.0}, None)
        assert loop._check_budget_limits(_ctx(active_use_local=True), 10.0,
            task_pacing.CostCeiling(state="active", ceiling_usd=5.0)) is None
    assert reads == []


def test_local_soft_landing_keeps_the_same_local_affordability_contract(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ouroboros import loop
    from tests.test_tree_cost_ceiling import _ctx

    ctx = _ctx(active_use_local=True, drive_root=tmp_path, drive_logs=tmp_path / "logs", llm=SimpleNamespace())
    monkeypatch.setattr(loop, "_prepare_forced_prompt", lambda _ctx, text, _trace: text)
    monkeypatch.setattr(task_pacing, "prepared_wrapup_candidate", lambda _ctx, messages, **_k: (_request(1.5, "local"), messages))
    monkeypatch.setattr(loop, "_forced_final_answer", lambda *_a, **_k: ("local final", {}, {}))
    ceiling = task_pacing.resolve_cost_ceiling(10.0, {"cost_hard_stop_pct": 50}, root_cap_usd=0.5)
    scope = accounting.UsageScope(drive_root=tmp_path, task_id="local-soft", root_task_id="local-soft", global_limit_usd=0.0)
    with accounting.usage_scope(scope):
        assert loop._soft_land_exhausted_ceiling(ctx, ceiling)[0] == "local final"


def test_forced_prompt_facts_are_priced_before_the_wrap_up_is_admitted(tmp_path, monkeypatch):
    """The typed facts block is added INSIDE ``_prepare_forced_prompt``, so the tokens
    it adds go through the existing wrap-up reservation probe, never around it."""
    from types import SimpleNamespace
    from ouroboros import loop
    from tests.test_tree_cost_ceiling import _ctx

    decision = {"required": True, "status": "advisory_open", "allow": True, "closed": False,
                "outcome": "DEGRADED", "enforcement": "advisory", "reviewer_slots_degraded": True}
    ctx = _ctx(active_use_local=True, drive_root=tmp_path, drive_logs=tmp_path / "logs",
               llm=SimpleNamespace(), llm_trace={"force_plan_decision": decision})
    priced = []
    monkeypatch.setattr(task_pacing, "prepared_wrapup_candidate",
                        lambda _ctx, messages, **_k: (priced.append(messages), (_request(1.5, "local"), messages))[1])
    monkeypatch.setattr(loop, "_forced_final_answer", lambda *_a, **_k: ("local final", {}, {}))
    ceiling = task_pacing.resolve_cost_ceiling(10.0, {"cost_hard_stop_pct": 50}, root_cap_usd=0.5)
    scope = accounting.UsageScope(drive_root=tmp_path, task_id="local-soft", root_task_id="local-soft", global_limit_usd=0.0)
    with accounting.usage_scope(scope):
        assert loop._soft_land_exhausted_ceiling(ctx, ceiling)[0] == "local final"
    (messages,) = priced
    last = next(m for m in reversed(messages) if m["role"] == "user")
    assert "[TASK_STATE_FACTS]" in str(last["content"]) and "plan_review_open=true" in str(last["content"])
    assert "[BUDGET LIMIT]" in str(last["content"])
    # Nothing to state, nothing priced: a closed gate adds no block to the probe.
    priced.clear()
    ctx.llm_trace = {"force_plan_decision": {**decision, "status": "closed", "closed": True}}
    with accounting.usage_scope(scope):
        loop._soft_land_exhausted_ceiling(ctx, ceiling)
    assert "[TASK_STATE_FACTS]" not in str(next(m for m in reversed(priced[0]) if m["role"] == "user")["content"])


@pytest.mark.parametrize("root_cap", [None, 50.0])
def test_live_wallet_triggers_the_existing_final_call_path_without_a_root_cap(tmp_path, monkeypatch, root_cap):
    from types import SimpleNamespace
    from ouroboros import loop
    from tests.test_tree_cost_ceiling import _ctx

    # Controlled reservation cost, shared by prediction and the actual ledger fence.
    monkeypatch.setattr(accounting, "_reservation_cost", lambda request: (
        request.reservation_usd if request.reservation_usd is not None else 1.5))
    scope = accounting.UsageScope(drive_root=tmp_path, task_id="main", root_task_id="main",
                                  global_limit_usd=10.0, root_limit_usd=root_cap)
    ctx = _ctx(task_id="main", drive_root=tmp_path, drive_logs=tmp_path / "logs", llm=SimpleNamespace())
    ceiling = task_pacing.CostCeiling(state="active", ceiling_usd=5.0, root_cap_usd=root_cap)
    prepared = []
    monkeypatch.setattr(loop, "_prepare_forced_prompt", lambda _ctx, text, _trace: (prepared.append(text), text)[1])
    request = _request(1.5, "main")
    monkeypatch.setattr(task_pacing, "prepared_wrapup_candidate", lambda _ctx, messages, **_k: (request, messages))
    admitted = []
    def finish(actual_ctx, **kwargs):
        assert kwargs["_admitted_request"] is request
        admitted.append(accounting.reserve_attempt(request))
        return "verified final", actual_ctx.accumulated_usage, actual_ctx.llm_trace
    monkeypatch.setattr(loop, "_forced_final_answer", finish)
    with accounting.usage_scope(scope):
        _spend(6.0)
        assert loop._check_budget_limits(ctx, 10.0, ceiling) is None
        assert prepared == [] and admitted == []
        _spend(2.0, "another-root")
        result = loop._check_budget_limits(ctx, 10.0, ceiling)
        assert result[0] == "verified final" and len(admitted) == 1 and len(prepared) == 1
        assert "$2.00 left across all tasks" in prepared[0]
        assert ctx.accumulated_usage["cost_stop_rail"] == "wrapup_reservation_last_fit"
        assert _wrapup_global_remaining() == 2.0  # the admitted final's hold is not known spend
        accounting.release_attempt(admitted[0], "controlled final callback did not send")
