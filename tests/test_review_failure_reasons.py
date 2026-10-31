"""Failed seats retain a reason on their ledger row when the full response artifact is pruned.

The reason is the producer's own words (redacted, bounded) and a structural code; an actor record is
never relabelled at construction, so a more specific code a later stage sets still wins, and a seat
that answered or is still awaited gains nothing.
"""

from dataclasses import asdict

import pytest

from ouroboros.review_ledger import build_rows
from ouroboros.review_records import ReviewActorRecord
from ouroboros.triad_review import parse_seat_answers


def _ledger_row(raw):
    row = build_rows({"triad_raw": [raw]})[0]
    row.pop("raw_text", None)
    return row


@pytest.mark.parametrize("status,error", [
    ("error", "Connection reset while reading the daemon response"),
    ("not_dispatched", "The complete request exceeds the transport limit"),
    ("empty", ""),
])
def test_a_failed_seat_keeps_its_reason_on_its_ledger_row(status, error):
    actor = ReviewActorRecord(slot_id="s", model="m", status=status, error=error)
    assert (actor.failure_code, actor.reported_cause) == ("", ""), "the actor record itself is never relabelled"
    row = _ledger_row(asdict(actor))
    assert row.get("reported_cause", "") == error, "the cause is the producer's words, never an invented sentence"
    assert not row.get("failure_code"), "no code is invented for a producer that named none"


@pytest.mark.parametrize("verdict,text", [
    ("ERROR", "Provider connection dropped mid-run"),
    ("UNKNOWN", "I could not produce the required JSON"),
    ("UNKNOWN", ""),
])
def test_parser_reasons_reach_ledger_without_raw_text(verdict, text):
    result = parse_seat_answers({"results": [
        {"slot_id": "s", "model": "m", "verdict": verdict, "text": text},
    ]}, {"s": ["change"]})
    row = _ledger_row(result.actor_records[0].to_dict())
    if text:
        assert text in row["reported_cause"]
    else:
        assert not row.get("reported_cause")


def test_typed_causes_survive_and_success_has_no_failure_facts():
    actor = ReviewActorRecord(slot_id="s", model="m", status="error", error="raw",
        failure_code="credential_pool_exhausted", reported_cause="All accounts exhausted",
        reset_at="2030-01-01T00:00:00Z", http_status=429)
    row = _ledger_row(asdict(actor))
    assert row["failure_code"] == "credential_pool_exhausted"
    assert row["reported_cause"] == "All accounts exhausted"
    assert row["http_status"] == 429 and row["reset_at"]
    result = parse_seat_answers({"results": [
        {"slot_id": "s", "model": "m", "text": "[]"},
    ]}, {"s": ["change"]})
    clean = _ledger_row(result.actor_records[0].to_dict())
    assert clean["status"] == "responded"
    assert not clean.get("failure_code") and not clean.get("reported_cause")


def test_an_awaited_seat_gains_no_failure_facts():
    awaited = {"slot_id": "s", "model_id": "m", "status": "error", "error": "Pending dispatch; in flight",
               "operation_state": "in_flight", "late_result_pending": True}
    row = _ledger_row(awaited)
    assert not row.get("failure_code") and not row.get("reported_cause")
    lost = {**awaited, "operation_state": "custody_lost", "error": "No durable start request was found",
            "failure_code": "review_custody_lost"}
    row = _ledger_row(lost)
    assert row["failure_code"] == "review_custody_lost" and row["reported_cause"] == "No durable start request was found"


def test_fallback_is_short_and_redacted():
    secret = "sk-ant-api03-" + "A1b2C3d4E5f6G7h8" * 4
    actor = ReviewActorRecord(slot_id="s", model="m", status="error",
        error=f"Authentication failed for {secret}\n" + "provider detail " * 200)
    cause = _ledger_row(asdict(actor))["reported_cause"]
    assert cause and len(cause) <= 512
    assert secret not in cause and "\n" not in cause


def test_withheld_packet_seat_keeps_the_size_refusal_on_its_ledger_row():
    from ouroboros.tools.review_admission import triad_not_dispatched_records

    records = triad_not_dispatched_records({"models": ["m"], "slot_ids": ["s"]},
                                          "The complete packet exceeds the configured input window")
    row = _ledger_row(records[0])
    assert row["reported_cause"] == "The complete packet exceeds the configured input window"
