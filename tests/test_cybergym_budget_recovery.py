"""Focused CyberGym budget-liability recovery regressions."""

import pytest

from devtools.benchmarks.cybergym.cybergym_adapter import BudgetLedger, run_campaign


def test_nonfinal_measured_spend_above_reservation_is_a_liability_floor(tmp_path):
    def callback(_task, _task_dir):
        return {
            "status": "infra_failed",
            "infra_reason": "partial_cost",
            "cost_usd": 3,
            "cost_estimated": False,
            "cost_final": False,
        }

    root = tmp_path / "nonfinal-cost-above-reservation"
    run_campaign(
        ["arvo:2"],
        run_root=root,
        executor=callback,
        estimated_cost_usd=1,
        budget_cap_usd=5,
    )

    projection = BudgetLedger(root / "claims.jsonl", cap_usd=5).projection()
    assert projection.unresolved_upper_bound_usd == pytest.approx(3)
    assert projection.projected_usd == pytest.approx(3)
