"""Reported amounts retain their knowledge independently from reviewer lifecycle."""

import json
from types import SimpleNamespace

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


@pytest.mark.parametrize(("amount", "status", "state"), [
    (None, "error", "in_flight"),
    (None, "responded", "late_settled"),
    (0.0, "error", "custody_lost"),
    (0.0, "pending", "in_flight"),
    (1.25, "responded", "settled"),
])
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


@pytest.mark.parametrize("first_final", [True, False])
def test_packet_retry_books_the_empty_reply_and_the_answer_once(tmp_path, monkeypatch, first_final):
    from ouroboros import usage_accounting as ua, usage_store
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

    sent, read_back = [], []
    attempt = usage_store.Txn.attempt

    def read_attempt(txn, attempt_id):
        if not txn.writable:
            read_back.append(attempt_id)
        return attempt(txn, attempt_id)

    monkeypatch.setattr(usage_store.Txn, "attempt", read_attempt)

    class LedgerChat:
        def chat(self, **kwargs):
            first = not sent
            text, cash, final = ("", 0.125, first_final) if first else ("[]\nNO_FINDINGS", 0.0, True)
            message = {"role": "assistant", "content": text}
            usage = {"cost": cash, "cost_final": final, "cost_estimated": not final}
            with ua.capture_attempt_ids() as ids:
                ua.execute_physical_attempt(ua.AttemptRequest(model=kwargs["model"], provider="openai-compatible"),
                    lambda: message, extractor=lambda _: (usage, cash, final))
            # A later carrier can name an earlier attempt again; it is still one bill.
            usage["ledger_attempt_ids"] = [*sent, *ids]
            sent.extend(ids)
            return message, usage

    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, non_task_operation=True)):
        run = run_review_request(ReviewRequest(surface="multi_model_review", goal="Review the fixture"),
                                 slots=[ReviewSlot("seat", "example/model")], drive_root=tmp_path, llm=LedgerChat())
    actor, = run.actors
    assert actor["status"] == "ok" and len(sent) == 2
    assert read_back == ([] if first_final else sent[:1])
    usage = actor["usage"]
    assert usage["ledger_attempt_ids"] == sent == actor["outcome"]["usage_ref"]["ledger_attempt_ids"]
    assert (usage["cost"], usage["cost_partial"], usage["cost_estimated"], usage["cost_final"]) == (
        0.125, False, not first_final, first_final)
    raw = _through_actor("responded", "settled", usage)
    record = review_ledger.build_commit_gate_record(_facts([raw])).to_dict()
    assert record["rows"][0]["usd"] == (0.125 if first_final else None)
    assert record["cost"]["unknown"] is (not first_final)


@pytest.mark.parametrize("prior,release_read", [
    ("", True), ("", False), ("ledger", True), ("capture", True), ("reply", True),
])
def test_released_review_hold_keeps_its_id_without_inventing_a_send(tmp_path, monkeypatch, prior, release_read):
    from ouroboros import delegate_custody_usage as usage, subagent_metric_events as metrics
    from ouroboros.review_execution_projection import review_executions_from_actor_usage
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot

    receipts = {}
    if prior == "ledger":
        receipts["earlier"] = {"state": "settled", "cost_usd": 0.0, "cost_final": True}
    elif prior == "capture":
        receipts["earlier"] = None
    receipts["hold"] = {"state": "released"} if release_read else None
    monkeypatch.setattr(usage, "_review_attempt_rows", lambda ids: {key: receipts[key] for key in ids})
    refusal = RuntimeError("not sent")
    refusal.physical_attempt_capture = SimpleNamespace(attempt_id="hold", state="released")
    refusal.ledger_attempt_ids = list(receipts) if prior == "ledger" else ["hold"]
    events = []
    costs = usage.ReviewAttemptCosts()
    if prior == "capture":
        earlier = RuntimeError("send failed")
        earlier.physical_attempt_capture = SimpleNamespace(attempt_id="earlier", state="unresolved")
        costs.failed(usage.observe_failed_review_send(events.append, earlier))
    elif prior == "reply":
        costs.observe({}, None)  # a legacy adapter can return before a later proven-unsent hold
    costs.failed(usage.observe_failed_review_send(events.append, refusal))
    fact = costs.project({"physical_attempt_state": "released"})
    assert fact["ledger_attempt_ids"] == list(receipts)
    priced = prior == "ledger"
    assert fact["cost"] == (0.0 if priced else None) and fact["cost_final"] is priced
    assert [row["ledger_attempt_ids"] for row in events] == ([["earlier"]] if "earlier" in receipts else [])

    assert review_executions_from_actor_usage([{"usage": fact}]) == ([{"kind": "api"}] if prior else [])
    native = {**fact, "delivery": "native_tool_rounds", "native_rounds": 0}
    assert review_executions_from_actor_usage([{"usage": native}]) == ([{"kind": "native"}] if prior else [])

    emitted = []
    monkeypatch.setattr(metrics, "publish", lambda _root, row: emitted.append(row))
    actor = SimpleNamespace(usage=fact, operation_id="operation", raw_text="", duration_sec=0,
                            failure_code="", awaiting_since="", outcome={"outcome_id": "outcome",
                            "operation_id": "operation", "status": "error",
                            "usage_ref": {"ledger_attempt_ids": list(receipts)}})
    metrics.review_execution(ReviewRequest(surface="plan_review", goal="Review the fixture"),
                             ReviewSlot("seat", "example/model"), actor, None, fallback_root=tmp_path)
    assert emitted[0]["dispatched"] is (True if prior else None)
    assert emitted[0]["actor"]["usage_ref"]["ledger_attempt_ids"] == list(receipts)


@pytest.mark.parametrize("cash,final,partial,estimated", [
    (0.0, True, False, False), (0.125, False, False, True),
    (0.125, False, True, False), (None, False, False, False),
])
def test_read_failure_amount_and_finality_reach_the_usage_event(tmp_path, monkeypatch, cash, final, partial, estimated):
    from ouroboros import delegate_custody_usage as usage
    from ouroboros.tools.review_helpers import emit_review_usage
    from supervisor.events_budget import _handle_llm_usage

    reads = []

    def read(ids):
        reads.extend(ids)
        return {key: {"state": "settled", "cost_usd": cash, "cost_final": final, "cost_partial": partial,
                      "cost_estimated": estimated, "model": "example/model", "provider": "example"} for key in ids}

    monkeypatch.setattr(usage, "_review_attempt_rows", read)
    error = RuntimeError("no review answer")
    error.ledger_attempt_ids = ["failed"]
    ctx = SimpleNamespace(task_id="fixture-review", pending_events=[])
    costs = usage.ReviewAttemptCosts()
    costs.failed(usage.observe_failed_review_send(
        lambda row: emit_review_usage(ctx, model=row["resolved_model"], usage=row, source="review"), error))
    seat = costs.project({})
    assert seat["cost"] == cash and seat["cost_final"] is final and seat["cost_estimated"] is estimated
    assert reads == ["failed"]

    event, = ctx.pending_events
    _handle_llm_usage(event, SimpleNamespace(DRIVE_ROOT=tmp_path))
    persisted = json.loads((tmp_path / "logs" / "events.jsonl").read_text())
    assert persisted["ledger_attempt_ids"] == ["failed"]
    assert (persisted["cost"], persisted["cost_known"], persisted["cost_final"], persisted["cost_estimated"]) == (
        cash, cash is not None, final, estimated)


def test_single_exact_review_reply_keeps_the_no_read_fast_path(monkeypatch):
    from ouroboros import delegate_custody_usage as usage, usage_accounting as ua

    def read(ids):
        assert ids == []  # no attempt-store read back for the exact receipt
        return {}

    monkeypatch.setattr(usage, "_review_attempt_rows", read)
    monkeypatch.setattr(ua, "last_physical_attempt_capture",
                        lambda: SimpleNamespace(attempt_id="ordinary", state="settled"))
    emitted = []
    legacy = {"resolved_model": "example/model", "cost": 0.0}
    costs = usage.ReviewAttemptCosts()
    assert costs.project(legacy) is legacy
    reply = {**legacy, "ledger_attempt_ids": ["ordinary"], "cost_final": True}
    costs.observe(reply, emitted.append)
    assert emitted == [reply]
    assert costs.project(reply) == {**reply, "cost_partial": False, "cost_estimated": False}
