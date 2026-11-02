"""Producer truth survives collection, settlement, receipts and card projection."""
import copy
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from ouroboros import review_ledger as ledger, review_pause, review_waiting, triad_review
from ouroboros.review_projection import compact_review_projection
from ouroboros.review_records import ReviewActorRecord, ReviewRequest, ReviewSlot
from ouroboros.review_seat_outcome import finalize_seat_outcome
from tests.test_review_ledger import _facts
from tests.test_review_seat_outcome import _body, _produce


def _parsed(actor):
    return triad_review.parse_seat_answer({**actor, "text": actor["raw_text"]}, ["change", "coupling"]).to_dict()


def _card(record):
    return compact_review_projection([], [record])["panels"][0]["actors"][0]


@pytest.mark.parametrize("text,verdict,aggregate", [
    (_body(), "PASS", "PASS"),
    (_body(True), "FAIL", "FAIL"),
    (json.dumps({"change": [], "change_clean": True, "coupling": []}), "unanswered", "NOT_PERFORMED"),
])
def test_producer_ledger_and_card_agree_on_all_assigned_parts(tmp_path, text, verdict, aggregate):
    run, _ctx = _produce(tmp_path, text, ["change", "coupling"])
    [actor] = run.actors
    record = ledger.write_record(tmp_path, ledger.build_commit_gate_record(_facts([_parsed(actor)]), drive_root=tmp_path))
    assert actor["outcome"]["verdict"] == record["verdict"]["per_row"]["seat"] == _card(record)["semantic_verdict"] == verdict
    assert compact_review_projection([asdict(run)])["panels"][0]["actors"][0]["semantic_verdict"] == verdict
    assert record["verdict"]["aggregate"] == aggregate
    assert record["rows"][0]["answers"]["change"] == actor["outcome"]["answers"]["change"]
    assert record["verdict"]["quorum"]["responded"] == 1, "partial responses do not change quorum participation"
    assert actor["quorum_contribution"] is _card(record)["quorum_contribution"] is True
    assert actor["enforcement_impact"] == {"PASS": "supports_pass", "FAIL": "veto", "unanswered": "abstains"}[verdict]


@pytest.mark.parametrize("cost", [0.125, 0.0, None])
def test_canonical_pending_settles_late_without_reparse_and_keeps_cost_and_receipt(tmp_path, monkeypatch, cost):
    run, _ctx = _produce(tmp_path, _body(), ["change", "coupling"], cost=cost)
    [actor] = run.actors
    pending = ReviewActorRecord("seat", "requested/model", "error", operation_id=actor["operation_id"],
        operation_state="in_flight", late_result_pending=True, recovery_binding=actor["recovery_binding"])
    finalize_seat_outcome(ReviewRequest(surface="multi_model_review", goal="review", policy={"parts": ["change", "coupling"]}),
                          ReviewSlot("seat", "requested/model"), pending)
    raw = _parsed(asdict(pending))
    record = ledger.write_record(tmp_path, ledger.build_commit_gate_record(_facts([raw], panel={"pool": ["seat"]}), drive_root=tmp_path))
    assert record["rows"][0]["outcome"]["status"] == "pending"
    assert _card(record)["transport_status"] == "awaiting" and not review_waiting.outcome_ids(record)

    # The saved assignment and producer CAS are real; only the attempt lookup is supplied here.
    attempt = SimpleNamespace(root_task_id="", review_contract_fingerprint="")
    monkeypatch.setattr(review_pause, "_left_attempt", lambda _record: attempt)
    monkeypatch.setattr(review_pause, "_left_rows", lambda _attempt: [("multi_model_review", "seat", asdict(pending))])
    def no_parse(*_args, **_kwargs):
        pytest.fail("canonical evidence was reparsed")
    monkeypatch.setattr(triad_review, "parse_two_part_answer", no_parse)
    monkeypatch.setattr(triad_review, "extract_json_array", no_parse)
    left = {"surface": "commit", "task_id": "truth", "retry_key": "", "record_id": record["record_id"], "drive_root": str(tmp_path)}
    _attempt, seats, gap = review_pause._left_seats(left)
    assert not gap and seats[0]["outcome"] == actor["outcome"]
    settled = review_pause._settle_left_record(left, seats, final=False)
    [seat] = settled["rows"]
    assert settled["state"] == "settled" and settled["verdict"]["aggregate"] == "PASS"
    assert seat["outcome"] == actor["outcome"] and seat["answers"] == actor["outcome"]["answers"]
    assert seat["usd"] == cost
    assert settled["cost"] == {"usd": cost or 0.0, "unknown": cost is None}
    card = _card(settled)
    assert (card["transport_status"], card["parse_status"], card["semantic_verdict"]) == ("success", "valid", "PASS")
    assert card["cost"] == {"kind": "unknown" if cost is None else "usd", "usd": cost}
    assert review_pause._seat_words(seats[0]) == "change PASS; coupling PASS"
    live = ledger.build_commit_gate_record(_facts([_parsed(actor)]), record_id=record["record_id"]).to_dict()
    assert review_waiting.outcome_ids(live) == review_waiting.outcome_ids(settled)
    assert review_waiting.outcome_ids(settled)[0].endswith(":" + actor["outcome"]["outcome_id"])


@pytest.mark.parametrize("writer", ["collected", "pending_payload"])
def test_legacy_terminal_replaces_pending_truth_and_uses_saved_parts(tmp_path, writer):
    pending = ReviewActorRecord("seat", "m", "error", operation_id="op", operation_state="in_flight", late_result_pending=True)
    finalize_seat_outcome(ReviewRequest(surface="multi_model_review", goal="g", policy={"parts": ["change", "coupling"]}),
                          ReviewSlot("seat", "m"), pending)
    record = ledger.write_record(tmp_path, ledger.build_commit_gate_record(_facts([_parsed(asdict(pending))]), drive_root=tmp_path))
    # No canonical outcome: the retained two-part assignment must still reject a missing coupling block.
    seat = {"slot_id": "seat", "model": "m", "operation_id": "op", "operation_state": "settled", "status": "ok",
            "parts": ["change", "coupling"], "raw_text": '{"change": [], "change_clean": true}', "cost_usd": .125}
    if writer == "collected":
        settled = review_pause._settle_left_record({"drive_root": str(tmp_path), "record_id": record["record_id"]}, [seat], final=True)
    else:
        fresh = ledger.build_commit_gate_record(_facts([_parsed(seat)]), drive_root=tmp_path).to_dict()
        settled = ledger.revise_record(tmp_path, record["record_id"], lambda now: ledger.settle_pending_payload(now, fresh))
    assert settled["state"] == "settled" and settled["verdict"]["aggregate"] == "NOT_PERFORMED"
    assert not settled["rows"][0].get("outcome")
    assert _card(settled)["semantic_verdict"] == "unanswered" and _card(settled)["transport_status"] != "awaiting"
    assert settled["rows"][0]["usd"] == .125


@pytest.mark.parametrize("canonical", [True, False])
def test_receipt_uses_canonical_identity_or_legacy_digest(tmp_path, canonical):
    run, _ctx = _produce(tmp_path, _body(), ["change", "coupling"])
    [actor] = run.actors
    record = ledger.build_commit_gate_record(_facts([_parsed(actor)])).to_dict()
    if not canonical:
        record["rows"][0].pop("outcome")
    before = review_waiting.outcome_ids(record)
    enriched = copy.deepcopy(record)
    enriched["rows"][0]["source_refs"].append({"role": "another_projection", "ref": "same answer"})
    after = review_waiting.outcome_ids(enriched)
    assert (before == after) is canonical
    if canonical:
        enriched["rows"][0]["outcome_id"] = "stale-top-level-id"
        assert review_waiting.outcome_ids(enriched) == before
    else:
        assert len(before[0].rsplit(":", 1)[-1]) == 64
