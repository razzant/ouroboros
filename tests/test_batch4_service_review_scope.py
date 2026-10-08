"""Fresh host review roles preserve actual coordinator caps and inherited work."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from ouroboros import review_custody
from ouroboros import review_substrate as review
from ouroboros import usage_accounting as ua
from ouroboros.gateway.extensions import _ApiReviewCtx
from ouroboros.marketplace.install import _MarketplaceReviewCtx
from tests._usage_store_testing import write_compacted_journal
from tests.test_billing_group import data_root as data_root


def _coordinator_reservation(ctx, monkeypatch, *, skill="one", amount=.1):
    coordinator = review.ReviewCoordinator(llm=SimpleNamespace(), drive_root=ctx.drive_root, usage_ctx=ctx)
    observed = {}
    class Captured(Exception):
        pass
    def paid(*_args, **_kwargs):
        observed["scope"] = ua.current_usage_scope()
        try:
            held = ua.reserve_attempt(ua.AttemptRequest(model="fixture", provider="fixture", reservation_usd=amount))
        except ua.BudgetExceeded as error:
            observed["refusal"] = str(error)
        else:
            observed["attempt"] = held
            ua.mark_dispatched(held)
            ua.settle_attempt(held, {}, cost_usd=amount, cost_final=True)
        raise Captured()
    monkeypatch.setattr(coordinator, "_run_slot", paid)
    monkeypatch.setattr(review_custody, "run_custodied_review_slots",
                        lambda **kw: kw["run_slot"](kw["slots"][0], "existing-operation", {}, time.monotonic()+60, None))
    with pytest.raises(Captured):
        coordinator.run(review.ReviewRequest(surface="skill_review", goal="review " + skill, task_id=ctx.task_id,
                        usage_attribution={"review_skill": skill, "review_wave_id": "wave-" + skill}),
                        [review.ReviewSlot(slot_id="slot", model="fixture")])
    return observed


@pytest.mark.parametrize("ctx_type", [_ApiReviewCtx, _MarketplaceReviewCtx])
def test_actual_host_role_preserves_configured_lifetime_cap(data_root, monkeypatch, ctx_type):
    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", "2")
    monkeypatch.setattr(review, "runtime_setting", lambda key, default=None: "2" if key == "OUROBOROS_PER_TASK_COST_USD" else default)
    ctx = ctx_type(data_root, data_root.parent / "repo")
    # An old root whose imported compacted block disputes its cap (two literals,
    # no carriage): open, so the host role's configured cap applies.
    aggregate = dict(task_id=ctx.task_id, root_task_id=ctx.task_id, parent_task_id="", provider="openai",
                     category="task", source="old", folded_attempt_count=1, cost_usd="0.5", cost_final=True,
                     reservation_upper_bound_usd="1.0", pricing_known=True)
    write_compacted_journal(data_root, [{**aggregate, "attempt_id": "old-first", "model": "z", "root_limit_usd": "2.0"},
                                        {**aggregate, "attempt_id": "old-later", "model": "a", "root_limit_usd": "100.0"}])
    # Known spend: the open block's $1.00, then $0.40 and $0.60 reach the configured $2 cap.
    for skill, amount in (("one", .4), ("two", .6)):
        result = _coordinator_reservation(ctx, monkeypatch, skill=skill, amount=amount)
        assert "attempt" in result
        scope = result["scope"]
        assert scope.non_task_operation and scope.root_limit_usd == 2.0
        assert scope.review_skill == skill and scope.review_wave_id == "wave-"+skill
        assert scope.root_task_id == ctx.task_id
    exhausted = _coordinator_reservation(ctx, monkeypatch, skill="three", amount=.3)
    assert "root model budget exhausted" in exhausted["refusal"]
    assert ua.usage_projection(data_root, root_task_id=ctx.task_id)["accounted_usd"] == pytest.approx(2.0)


@pytest.mark.parametrize("ctx_type", [_ApiReviewCtx, _MarketplaceReviewCtx])
def test_host_role_keeps_global_exhaustion(data_root, monkeypatch, ctx_type):
    ctx = ctx_type(data_root, data_root.parent / "repo")
    with ua.usage_scope(ua.UsageScope(drive_root=data_root, non_task_operation=True, global_limit_usd=.01)):
        spent = ua.reserve_attempt(ua.AttemptRequest(model="fixture", provider="fixture", reservation_usd=.01))
        ua.mark_dispatched(spent)
        ua.settle_attempt(spent, {}, cost_usd=.01, cost_final=True)  # known spend reaches the $0.01 wallet
        result = _coordinator_reservation(ctx, monkeypatch)
    assert result["scope"].global_limit_usd == .01
    assert "global model budget exhausted" in result["refusal"]


@pytest.mark.parametrize("ctx_type", [_ApiReviewCtx, _MarketplaceReviewCtx])
def test_host_hint_never_overrides_inherited_continue_group(data_root, monkeypatch, ctx_type):
    ctx = ctx_type(data_root, data_root.parent / "repo")
    inherited = ua.UsageScope(drive_root=data_root, task_id="helper", root_task_id="successor",
        parent_task_id="successor", billing_group_id="original", billing_group_limit_usd=1.0,
        billing_group_limit_source="ledger_first_row", root_limit_usd=50.0)
    with ua.usage_scope(inherited):
        first = _coordinator_reservation(ctx, monkeypatch, amount=1.0)  # known spend reaches the group's $1
        second = _coordinator_reservation(ctx, monkeypatch, amount=.3)
    assert "attempt" in first
    scope = first["scope"]
    assert not scope.non_task_operation and scope.root_task_id == "successor"
    assert scope.billing_group_id == "original" and scope.billing_group_limit_usd == 1.0
    assert "whole-work budget exhausted for group original" in second["refusal"]


@pytest.mark.parametrize("name", ["api_skill_review", "marketplace_install"])
def test_task_name_alone_confers_no_host_role(data_root, monkeypatch, name):
    ctx = SimpleNamespace(drive_root=data_root, task_id=name, task_lifecycle_bound=True)
    first = _coordinator_reservation(ctx, monkeypatch)
    assert not first["scope"].non_task_operation


@pytest.mark.serial
@pytest.mark.parametrize("door", ["http", "marketplace"])
def test_actual_adapter_produces_non_task_review(data_root, monkeypatch, door):
    from ouroboros import skill_review, skill_review_runner
    from ouroboros.gateway import extensions
    from ouroboros.marketplace import install
    seen = []
    def review_impl(ctx, skill):
        result = _coordinator_reservation(ctx, monkeypatch, skill=skill)
        assert "attempt" in result and result["scope"].non_task_operation
        seen.append(result)
        return SimpleNamespace(status="clean", findings=[], error="")
    monkeypatch.setattr(skill_review, "review_skill", review_impl)
    if door == "marketplace":
        assert install._run_skill_review(data_root, data_root.parent / "repo", "one") == ("clean", [], "")
    else:
        async def lifecycle(ctx, skill, *, source, review_impl):
            return {"status": review_impl(ctx, skill).status}
        monkeypatch.setattr(skill_review_runner, "run_skill_review_lifecycle", lifecycle)
        monkeypatch.setattr(extensions, "_request_drive_root", lambda request: data_root)
        monkeypatch.setattr(extensions, "_request_repo_dir", lambda request: data_root.parent / "repo")
        response = asyncio.run(extensions.api_skill_review(SimpleNamespace(path_params={"skill": "one"})))
        assert response.status_code == 200
    assert len(seen) == 1
