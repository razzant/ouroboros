"""Reported amounts retain their knowledge independently from reviewer lifecycle."""

import pytest

from ouroboros import review_ledger
from ouroboros.tools.review_change import review_result
from ouroboros.tools.review_response import parse_model_response
from ouroboros.triad_review import _actor_record
from tests.test_review_ledger import _facts


def _through_actor(status, operation_state, usage, *, headers=None):
    result = {
        "slot_id": "seat", "operation_state": operation_state, "usage": usage,
        "choices": [{"message": {"content": "[]\nNO_FINDINGS"}}],
    }
    if status == "error":
        result["error"] = "The answer could not be observed."
    envelope = parse_model_response("example/model", result, headers)
    return _actor_record(
        envelope, idx=0, model_label="example/model", status=status,
        raw_text=envelope["text"],
    ).to_dict()


@pytest.mark.parametrize("status,state", [
    ("error", "in_flight"), ("error", "custody_lost"),
    ("responded", "settled"), ("responded", "late_settled"),
    ("pending", "in_flight"),
])
@pytest.mark.parametrize("amount", [None, 0.0, 1.25])
def test_amount_survives_response_actor_record_and_tool(status, state, amount):
    raw = _through_actor(status, state, {"cost": amount})
    assert raw["cost_usd"] == amount
    record = review_ledger.build_commit_gate_record(_facts([raw])).to_dict()
    assert record["rows"][0]["usd"] == amount
    expected = {"usd": amount if amount is not None else 0.0, "unknown": amount is None}
    assert record["cost"] == expected
    assert review_result(record, reused=False)["cost"] == expected
    assert record["rows"][0]["operation_state"] == state


@pytest.mark.parametrize("status", ["error", "responded"])
def test_estimated_or_missing_amount_is_not_promoted_to_reported_money(status):
    for usage in ({}, {"cost": None, "cost_disclosed_usd": 3.0, "cost_estimated": True},
                  {"cost": 3.0, "cost_estimated": True}):
        raw = _through_actor(status, "settled", usage)
        assert raw["cost_usd"] is None
    assert _through_actor(status, "settled", {"total_cost": 0.75})["cost_usd"] == 0.75
    assert _through_actor(status, "settled", {}, headers={"X-OpenRouter-Cost": "0"})["cost_usd"] == 0.0


def test_missing_amount_changes_cost_knowledge_not_verdict_or_free_reuse(tmp_path):
    known = _through_actor("responded", "settled", {"cost": 1.25})
    absent = _through_actor("responded", "settled", {"cost": None})
    absent["slot_id"] = "other-seat"
    record = review_ledger.build_commit_gate_record(_facts([known, absent])).to_dict()
    assert record["cost"] == {"usd": 1.25, "unknown": True}
    absent["cost_usd"] = 0.0
    priced = review_ledger.build_commit_gate_record(_facts([known, absent])).to_dict()
    assert record["verdict"] == priced["verdict"]
    assert priced["cost"] == {"usd": 1.25, "unknown": False}
    written = review_ledger.write_record(tmp_path, review_ledger.ReviewLedgerRecord.from_dict(record))
    assert review_ledger.load_record(tmp_path, record["record_id"])["cost"] == record["cost"]
    assert review_ledger.recent_records(tmp_path, "task-1", 1)[0]["cost"] == record["cost"]
    assert review_result(written, reused=True)["cost"] == {"usd": 0.0, "unknown": False}
