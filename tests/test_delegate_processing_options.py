"""Captured processing intent changes only a supported physical request option."""
from types import SimpleNamespace

import pytest

from ouroboros.tools.delegate import _processing_start_request
from tests._usage_store_testing import ledger_rows


@pytest.mark.parametrize("preference", ["standard", "fast", "economy"])
def test_captured_processing_preference_uses_exact_route_capability(preference):
    gateway = SimpleNamespace(agent_capabilities=lambda: {"harnesses": [
        {"id": "other", "processingPreferences": []},
        {"id": "chosen", "processingPreferences": ["standard", "fast", "economy"]}]})
    body = {"instructions": "captured instructions", "model": "exact-model", "effort": "xhigh"}
    result, facts = _processing_start_request(body, {"processing_preference": preference}, gateway,
                                       SimpleNamespace(route_id="chosen"))
    assert result == {"instructions": "captured instructions", "model": "exact-model", "effort": "xhigh",
                      "processingPreference": preference}
    assert facts["submitted"] == preference and facts["observed"] == "unknown"


def test_old_engine_does_not_receive_unsupported_wire_field():
    gateway = SimpleNamespace(agent_capabilities=lambda: {"harnesses": [{"id": "chosen"}]})
    body = {"instructions": "captured", "model": "exact-model", "effort": "xhigh"}
    result, facts = _processing_start_request(body, {"processing_preference": "economy"}, gateway,
                                       SimpleNamespace(route_id="chosen"))
    assert "processingPreference" not in result
    assert result["instructions"] == "captured"
    assert facts["reason"] == "processing_not_submitted" and facts["submitted"] is None
    assert result["model"] == "exact-model" and result["effort"] == "xhigh"


def test_no_preference_preserves_legacy_body_without_catalog_read():
    body = {"instructions": "legacy", "model": "exact-model"}
    assert _processing_start_request(body, {}, None, None) == (body, {})


@pytest.mark.parametrize("costs,expected,final", [
    ([{"cashUsd": 0, "cashKnowledge": "exact", "valuationUsd": 8}], 0.0, True),
    ([{"cashUsd": None, "cashKnowledge": "unknown", "valuationUsd": 8}], None, False),
    ([{"cashUsd": 1.25, "cashKnowledge": "exact"}, {"cashKnowledge": "unknown"}], 1.25, False),
    ([{"cashUsd": 1.25, "cashKnowledge": "exact"}, {"cashUsd": 0.5, "cashKnowledge": "exact"}], 1.75, True),
])
def test_terminal_and_existing_ledger_use_attempt_cash_not_valuation(tmp_path, costs, expected, final):
    from ouroboros import delegate_custody as custody
    from ouroboros.subagents import DelegatedRunShape
    from ouroboros.tools.delegate_terminal_evidence import _terminal_payload

    attempts = [{"attemptId": str(index), "usageCost": cost} for index, cost in enumerate(costs)]
    detail = {"summary": {"state": "succeeded", "spendUsd": 99}, "attemptExecution": attempts}
    entry = custody.RunCustody(run_id="cash-test", task_id="task", route_id="session",
                              project_persistent=True, ledger_root=str(tmp_path))
    result = custody.settle_run(tmp_path, None, entry, detail)
    assert result["ledger_recorded"]
    rows = ledger_rows(tmp_path)
    row = next(row for row in rows if row.get("kind") == "subscription_session")
    payload = _terminal_payload("cash-test", detail, DelegatedRunShape("readonly", "ask", "live", False))
    assert payload["cost"]["cost_usd"] == row["cost_usd"] == expected
    assert payload["cost"]["cost_final"] == row["cost_final"] == final
    assert payload["attempt_execution"] == attempts
