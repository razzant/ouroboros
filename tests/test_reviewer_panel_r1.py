"""Round-one regressions: outcome facts, free composition and adviser disclosures."""
import json

import pytest

from ouroboros import review_ledger as ledger, reviewer_slot_config as slots
from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool
from tests.test_reviewer_panel_policy import roster  # noqa: F401
from tests.test_review_ledger import BOTH, _facts, _raw


@pytest.mark.parametrize("status,operation,pending,expected", [
    ("pending", "in_flight", True, []), ("error", "in_flight", True, []),
    ("pending", "pending_dispatch", True, []), ("answered", "settled", False, []),
    ("unparseable", "settled", False, []), ("failed", "settled", False, ["provider_error"]),
    ("not_dispatched", "not_dispatched", False, ["provider_error"]),
])
def test_not_reached_excludes_awaiting_but_retains_terminal_failures(status, operation, pending, expected):
    row = {"seat_id": "s", "status": status, "operation_state": operation,
           "late_result_pending": pending, "failure_code": "provider_error"}
    assert ledger.panel_not_reached([row]) == [{"seat_id": "s", "cause": cause} for cause in expected]


@pytest.mark.parametrize("answered", [False, True])
def test_late_commit_recomputes_not_reached_without_rewriting_composition(answered):
    prior = ledger.build_commit_gate_record(_facts([
        _raw("s", "m/s", "pending", parts=BOTH, operation_state="in_flight", late_result_pending=True)],
        panel={"reason": "original", "pool": [{"seat_id": "s"}]})).to_dict()
    # An older record can contain the previous dispatch-time label.
    prior["panel"]["not_reached"] = [{"seat_id": "s", "cause": "in_flight"}]
    fresh = ledger.build_commit_gate_record(_facts([
        _raw("s", "m/s", "responded" if answered else "error", parts=BOTH,
             failure_code="" if answered else "provider_error")], panel={"reason": "new"})).to_dict()
    settled = ledger.settle_pending_payload(prior, fresh)
    assert settled["panel"]["not_reached"] == ([] if answered else [{"seat_id": "s", "cause": "provider_error"}])
    assert settled["panel"]["reason"] == "original" and settled["panel"]["pool"] == [{"seat_id": "s"}]
    assert prior["panel"]["not_reached"] == [{"seat_id": "s", "cause": "in_flight"}]


@pytest.mark.parametrize("strictness", ["blocking", "advisory", "cyber_pro"])
@pytest.mark.parametrize("reason", ["", "targeted"])
def test_foreign_change_composes_freely_with_honest_reason_facts(roster, strictness, reason):  # noqa: F811
    panel = slots.compose_review_panel(["b", "adviser"], reason, surface="change", strictness=strictness)
    assert panel.facts["assigned"] == ["b"] and panel.additional == ("adviser",)
    assert panel.facts["reason_missing"] is (not bool(reason))
    if strictness == "advisory" and not reason:
        with pytest.raises(slots.PanelCompositionError, match="Supply reason"):
            slots.compose_review_panel(["b"], surface="own_body", strictness=strictness)


@pytest.mark.parametrize("strictness", ["advisory", "cyber_pro"])
def test_empty_pool_refusal_names_the_main_seat(monkeypatch, strictness):
    set_review_pool(monkeypatch, pool_roster(pool_seat("adviser", "openai/adviser", marked=False)))
    with pytest.raises(slots.PanelCompositionError, match="name.*main"):
        slots.compose_review_panel(["adviser"], "focused", surface="skill", strictness=strictness)
    assert slots.compose_review_panel(["main", "adviser"], surface="skill", strictness=strictness).facts["assigned"] == ["main"]


@pytest.mark.parametrize("additional", [False, True])
def test_skill_adviser_text_is_labelled_and_never_coaches_a_retry(additional):
    from ouroboros.skill_review_output import render_skill_review_block

    text = render_skill_review_block({"skill": "notes", "status": "blockers", "review_round": 2, "findings": [
        {"model": "m/a", "slot_id": "a", "item": "counted-item", "verdict": "FAIL", "severity": "critical", "reason": "fix a"},
        {"model": "m/critic", "slot_id": "critic", "item": "critic-item", "verdict": "FAIL", "severity": "critical",
         "reason": "fix critic", "additional": additional}]})
    assert ("### Reviewer: m/critic [critic] (adviser, not counted)" in text) is additional
    assert "### Reviewer: m/a (adviser" not in text
    assert text.count("counted-item") == 2
    assert text.count("critic-item") == (1 if additional else 2)


@pytest.mark.parametrize("additional", [False, True])
def test_commit_diversity_counts_only_counted_seats(tmp_path, monkeypatch, additional):
    from ouroboros.tools import review
    from tests.test_review_brief_coupling import _dispatch_ctx, _prepared
    from ouroboros.tools.scope_review_contract import SCOPE_REQUIRED_ITEMS

    ctx = _dispatch_ctx(tmp_path)
    prepared = _prepared(ctx.repo_dir, [BOTH, BOTH])
    prepared["row_plan"]["additional"] = [False, additional]
    coupling = [{"item": item, "verdict": "PASS", "severity": "advisory", "reason": "checked"} for item in SCOPE_REQUIRED_ITEMS]
    monkeypatch.setattr(review, "_handle_multi_model_review", lambda *a, **kw: json.dumps({"results": [
        {"slot_id": sid, "model": model, "text": json.dumps({"change": [], "change_clean": True, "coupling": coupling})}
        for sid, model in zip(prepared["row_plan"]["slot_ids"], prepared["models"])]}))
    review._dispatch_unified_review(ctx, "test", prepared)
    assert ("single_reviewer_no_diversity" in getattr(ctx, "_review_degraded_reasons", [])) is additional
