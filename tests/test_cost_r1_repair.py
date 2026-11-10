"""Consumer controls for exact subtotals and bounded custody publication."""
from __future__ import annotations

from decimal import Decimal
from typing import get_type_hints

import pytest

from ouroboros import delegate_terminal


@pytest.mark.parametrize("row,confirmed,estimated", [
    ({"cost_usd": True, "cost_final": True}, 1, 0),
    ({"cost_usd": "2", "cost_final": False}, 0, 2),
    ({"cost_usd": "2", "cost_final": False, "cost_estimated": False}, 0, 2),
    ({"cost_usd": "2", "cost_final": False, "cost_partial": True}, 0, 2),
    ({"cost_usd": "2", "cost_final": False, "cost_partial": True, "cost_estimated": False}, 2, 0),
])
def test_legacy_cash_grammar_is_not_provider_revalidation(row, confirmed, estimated):
    from ouroboros._usage_money import cash_contribution

    cash = cash_contribution({"kind": "subscription_session", "state": "settled", **row})
    assert cash[1:3] == (Decimal(confirmed), Decimal(estimated))


def test_cost_custody_public_type_is_closed_and_nullable():
    from typing import get_args, get_origin

    from typing_extensions import NotRequired

    from ouroboros.delegate_custody_usage import CostCustody
    from ouroboros.gateway.contracts import ChatOutbound, TaskCostBreakdown

    assert set(get_type_hints(CostCustody)) == {"status", "open_run_ids", "unrecorded_run_ids", "closed_unrecorded_run_ids",
                                                "pending_invocation_ids"}
    # Additive: records written before it lack the subset, and absence keeps their earlier reading.
    marked = {name for name, ann in CostCustody.__annotations__.items()
              if (getattr(ann, "__forward_arg__", None) or str(ann)).startswith("NotRequired[")}
    assert marked == {"closed_unrecorded_run_ids"}
    for shape in (ChatOutbound, TaskCostBreakdown):
        annotation = get_type_hints(shape, include_extras=True)["cost_custody"]
        # Key optionality is not nullability; Python 3.10 keeps typing_extensions' wrapper in the hints.
        if get_origin(annotation) is NotRequired:
            (annotation,) = get_args(annotation)
        assert set(get_args(annotation)) == {CostCustody, type(None)}


def test_legacy_evidence_flags_are_quiet_until_cost_knowledge_changes():
    prior = {"subscription_cost_usd": 2}
    stored = {"subagent_envelope": {"execution_evidence": prior}}
    assert not delegate_terminal._stored_evidence_stale(stored, {
        **prior, "subscription_cost_estimated": False, "subscription_cost_partial": False})
    assert delegate_terminal._stored_evidence_stale(stored, {
        **prior, "subscription_cost_estimated": False, "subscription_cost_partial": True})
