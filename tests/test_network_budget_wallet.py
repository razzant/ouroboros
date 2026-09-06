"""The existing final-call price is compared with live shared-wallet evidence."""

from dataclasses import replace

import pytest

from ouroboros import task_pacing, usage_accounting as accounting
from ouroboros.loop_budget import _wrapup_global_remaining


def _request(amount, task="other"):
    return accounting.AttemptRequest(model="fixture-model", provider="openai",
        reservation_usd=amount, task_id=task, root_task_id=task)


@pytest.mark.parametrize("root_cap", [None, 50.0])
def test_actual_global_holds_leave_one_same_route_final_then_the_atomic_fence_binds(tmp_path, root_cap):
    scope = accounting.UsageScope(drive_root=tmp_path, task_id="main", root_task_id="main",
                                  global_limit_usd=10.0, root_limit_usd=root_cap)
    with accounting.usage_scope(scope):
        other = accounting.reserve_attempt(_request(8.0))
        request = _request(1.5, "main")
        remaining = _wrapup_global_remaining()
        assert remaining == 2.0
        args = dict(request=request, root_cap_usd=root_cap, deciding_usd=0.0,
                    global_remaining_usd=remaining)
        assert task_pacing.wrapup_reservation_fits(**args) is True
        assert task_pacing.wrapup_reservation_fits(**args, reservation_count=2) is False
        final = accounting.reserve_attempt(request)
        with pytest.raises(accounting.BudgetExceeded) as refused:
            accounting.reserve_attempt(request)
        assert refused.value.limit_scope == "global"
        assert _wrapup_global_remaining() == 0.5
        accounting.mark_dispatched(other)
        accounting.mark_unresolved(other, "provider outcome unknown")
        assert _wrapup_global_remaining() == 0.5  # a dead/unknown call is not a refund
        accounting.release_attempt(final, "controlled test did not send")
        assert _wrapup_global_remaining() == 2.0


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
        accounting.reserve_attempt(_request(1.0))
        with pytest.raises(accounting.BudgetExceeded):
            accounting.reserve_attempt(request)  # observation did not reserve a share


def test_explicit_scope_limit_and_canonical_root_own_the_wallet_read(tmp_path):
    scope = accounting.UsageScope(drive_root=tmp_path, task_id="main", root_task_id="main", global_limit_usd=10.0)
    with accounting.usage_scope(scope):
        accounting.reserve_attempt(_request(3.0))
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
    assert "spend is unavailable" in text and "global model budget remaining is $2.000" in text


def test_local_final_call_path_does_not_request_an_unneeded_wallet_projection(tmp_path, monkeypatch):
    from ouroboros import loop
    from tests.test_tree_cost_ceiling import _ctx

    reads = []
    monkeypatch.setattr(accounting, "usage_projection", lambda *_a, **_k: reads.append(True) or {})
    monkeypatch.setattr(accounting, "_ROOT_ACCOUNTING_TELEMETRY", {})
    scope = accounting.UsageScope(drive_root=tmp_path, task_id="local", root_task_id="local", global_limit_usd=10.0)
    with accounting.usage_scope(scope):
        accounting._stash_root_accounting("local", 0.0, None)
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
        accounting.reserve_attempt(_request(6.0))
        assert loop._check_budget_limits(ctx, 10.0, ceiling) is None
        assert prepared == [] and admitted == []
        accounting.reserve_attempt(_request(2.0, "another-root"))
        result = loop._check_budget_limits(ctx, 10.0, ceiling)
        assert result[0] == "verified final" and len(admitted) == 1 and len(prepared) == 1
        assert "global model budget remaining is $2.000" in prepared[0]
        assert ctx.accumulated_usage["cost_stop_rail"] == "wrapup_reservation_last_fit"
        assert _wrapup_global_remaining() == 0.5
        accounting.release_attempt(admitted[0], "controlled final callback did not send")
