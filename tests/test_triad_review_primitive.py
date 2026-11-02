import json
from types import SimpleNamespace

import pytest

from ouroboros.triad_review import (
    emit_review_model_error_events,
    extract_json_array,
    parse_seat_answers,
)


def test_extract_json_array_handles_fences_and_normalizes():
    raw = "```json\n[{\"item\":\"x\",\"verdict\":\"PASS\",\"severity\":\"critical\",\"reason\":\"ok\"}]\n```"
    parsed = extract_json_array(raw, normalize=True)
    assert parsed[0]["item"] == "x"


def test_extract_json_array_tries_later_fenced_chunks_when_first_is_malformed():
    raw = (
        "```json\n"
        "[{\"item\":\"bad\",\"verdict\":\"FAIL\",}]\n"
        "```\n"
        "```json\n"
        "[{\"item\":\"good\",\"verdict\":\"PASS\",\"severity\":\"critical\",\"reason\":\"ok\"}]\n"
        "```"
    )
    parsed = extract_json_array(raw, normalize=True)
    assert parsed[0]["item"] == "good"


def test_extract_json_array_normalizes_obligation_suffix():
    raw = json.dumps([
        {
            "item": "code_quality (obligation obl-0001)",
            "verdict": "PASS",
            "severity": "critical",
            "reason": "fixed",
        }
    ])
    parsed = extract_json_array(raw, normalize=True)
    assert parsed[0]["item"] == "code_quality"
    assert parsed[0]["obligation_id"] == "obl-0001"


def test_a_single_question_seat_answers_the_findings_array_and_may_omit_items():
    """Contract A for a result or skill seat (owner 21.10, 2A): problems only; a seat that said nothing about an
    item still counts for what it said, and ``[]`` + NO_FINDINGS is a clean answer."""
    one = json.dumps([{"item": "a", "verdict": "FAIL", "severity": "critical", "reason": "bad"}])
    parsed = parse_seat_answers({"results": [
        {"model": "m1", "slot_id": "s1", "text": one},
        {"model": "m2", "slot_id": "s2", "text": "[]\nNO_FINDINGS"},
        {"model": "m3", "slot_id": "s3", "text": "I cannot review this."},
    ]}, {"s1": ("skill",), "s2": ("skill",), "s3": ("skill",)})

    assert parsed.responsive_models == ["m1 [s1]", "m2 [s2]"]
    assert [r.status for r in parsed.actor_records] == ["responded", "responded", "parse_failure"]
    assert parsed.actor_records[0].answers["skill"]["verdict"] == "FAIL"
    assert parsed.actor_records[1].answers["skill"] == {"status": "responded", "verdict": "PASS", "findings": [],
                                                       "critical": 0, "coverage": "n/a"}
    assert [f["item"] for f in parsed.findings] == ["a"]


@pytest.mark.parametrize("part", ["skill", "result", "change"])
def test_a_non_empty_array_with_no_readable_entry_is_no_answer_not_a_clean_one(part):
    """Contract A, every single-question part alike (the rule contract B already applies to its change block):
    an array whose entries carry no ``item`` or no PASS/FAIL ``verdict`` is a parse failure, never a PASS; a
    seat with at least one readable entry keeps it (its junk siblings dropped); ``[]`` + NO_FINDINGS stays clean."""
    unreadable = json.dumps([{"item": "host_token_handling", "severity": "critical", "reason": "prints the token"},
                             {"note": "looks fine"}])
    mixed = json.dumps([{"unexpected": "not a finding"},
                        {"item": "bug_hunting", "verdict": "FAIL", "severity": "critical", "reason": "raises"}])
    parsed = parse_seat_answers({"results": [
        {"model": "m1", "slot_id": "s1", "text": unreadable},
        {"model": "m2", "slot_id": "s2", "text": mixed},
        {"model": "m3", "slot_id": "s3", "text": "[]\nNO_FINDINGS"},
    ]}, {"s1": (part,), "s2": (part,), "s3": (part,)}, severity_policy={"host_token_handling": "critical"})
    assert [r.status for r in parsed.actor_records] == ["parse_failure", "responded", "responded"]
    assert parsed.responsive_models == ["m2 [s2]", "m3 [s3]"]
    refused = parsed.actor_records[0].answers[part]
    assert refused["status"] == "unanswered" and refused["verdict"] == "" and refused["findings"] == []
    assert refused["error"] == "none of the 2 entries has an item and a PASS/FAIL verdict"
    assert parsed.actor_records[1].answers[part]["verdict"] == "FAIL"
    assert [f["item"] for f in parsed.findings] == ["bug_hunting"]
    assert parsed.actor_records[2].answers[part]["verdict"] == "PASS"


def test_a_declared_severity_policy_decides_the_seats_verdict_before_it_is_read():
    """The skill rule (hard-critical items block whatever the seat says; advisory-only items never block;
    an item the policy does not name follows its default) is applied while parsing, so the seat verdict the
    quorum counts already carries it."""
    text = json.dumps([
        {"item": "hard", "verdict": "FAIL", "severity": "advisory", "reason": "seat said advisory"},
        {"item": "soft", "verdict": "FAIL", "severity": "critical", "reason": "seat said critical"},
        {"item": "own", "verdict": "FAIL", "severity": "warning", "reason": "seat keeps its word"},
        {"item": "invented", "verdict": "FAIL", "severity": "critical", "reason": "not a checklist item"},
    ])
    policy = {"hard": "critical", "soft": "advisory", "own": "seat", "*": "advisory"}
    parsed = parse_seat_answers({"results": [{"model": "m1", "slot_id": "s1", "text": text}]},
                                {"s1": ("skill",)}, severity_policy=policy)
    answer = parsed.actor_records[0].answers["skill"]
    assert {f["item"]: f["severity"] for f in answer["findings"]} == {
        "hard": "critical", "soft": "advisory", "own": "advisory", "invented": "advisory"}
    assert (answer["verdict"], answer["critical"]) == ("FAIL", 1)
    # Without a policy the seat's own severities stand (a change seat today).
    plain = parse_seat_answers({"results": [{"model": "m1", "slot_id": "s1", "text": text}]}, {"s1": ("skill",)})
    assert plain.actor_records[0].answers["skill"]["critical"] == 2


def test_quorum_and_degraded_reasons_follow_the_responsive_seats():
    good = json.dumps([{"item": "a", "verdict": "PASS", "severity": "critical", "reason": "ok"}])
    parsed = parse_seat_answers({"results": [
        {"model": "m1", "slot_id": "s1", "verdict": "REVIEW", "text": good},
        {"model": "m2", "slot_id": "s2", "verdict": "REVIEW", "text": good},
        {"model": "m3", "slot_id": "s3", "verdict": "ERROR", "text": "boom"},
    ]}, {})

    assert parsed.quorum_met is True
    assert parsed.degraded_reasons == ["DEGRADED: m3=error (quorum still met)"]


def test_responsive_identity_is_the_stable_slot_when_model_label_drifts():
    good = json.dumps([{"item": "a", "verdict": "PASS", "severity": "critical", "reason": "ok"}])
    parsed = parse_seat_answers({"results": [
        {"model": "model-before", "slot_id": "slot-1", "text": good},
    ]}, {"slot-1": ("change",)})

    assert parsed.responsive_models == ["model-before [slot-1]"]
    assert parsed.actor_records[0].slot_id == "slot-1"


def test_emit_review_model_error_events(tmp_path):
    good = json.dumps([{"item": "a", "verdict": "PASS", "severity": "critical", "reason": "ok"}])
    parsed = parse_seat_answers({
        "results": [
            {"model": "m1", "slot_id": "s1", "verdict": "REVIEW", "text": good},
            {"model": "m2", "slot_id": "s2", "verdict": "ERROR", "text": "boom"},
        ]
    }, {})
    logs = tmp_path / "logs"
    logs.mkdir()
    ctx = SimpleNamespace(drive_logs=lambda: logs)

    emit_review_model_error_events(ctx, parsed, source="skill_review", skill_name="demo")

    data = (logs / "events.jsonl").read_text(encoding="utf-8")
    assert '"review_model_error"' in data
    assert '"skill": "demo"' in data
