import json
from types import SimpleNamespace

from ouroboros.triad_review import (
    emit_review_model_error_events,
    empty_array_is_verified_clean,
    extract_json_array,
    parse_model_review_results,
    strip_leading_reasoning_block,
)

# A MiniMax-M3 / DeepSeek-R1 style response: the whole review lives in a
# passthrough <think> block, then the clean sentinel with no separator.
_REASONING_CLEAN = (
    "<think>\nLet me review this diff. It touches build_touched_file_pack.\n"
    "Checklist: bible_compliance PASS, code_quality PASS. Brackets [x] and "
    "braces {y} in my reasoning must not confuse the parser.\nOK. NO_FINDINGS.\n"
    "</think>\n\n[]NO_FINDINGS"
)
_REASONING_FINDING = (
    "<think>I spotted a real bug at line 5 [note]</think>\n"
    '[{"item": "code_quality", "verdict": "FAIL", "severity": "critical", "reason": "x"}]'
)


def test_strip_leading_reasoning_block_only_removes_a_wellformed_leading_block():
    assert strip_leading_reasoning_block("<think>abc</think>\n[]") == "[]"
    assert strip_leading_reasoning_block("  <think>a</think> tail") == "tail"
    # no block, or a block that is not leading, is left untouched
    assert strip_leading_reasoning_block("[]\nNO_FINDINGS") == "[]\nNO_FINDINGS"
    assert strip_leading_reasoning_block("prefix <think>a</think>") == "prefix <think>a</think>"


def test_reasoning_wrapped_clean_verdict_enters_quorum():
    """A reasoning model whose clean [] / NO_FINDINGS verdict sits behind a
    passthrough <think> block must NOT drop out of quorum (ibl-local-8212d8d13320)."""
    assert empty_array_is_verified_clean(_REASONING_CLEAN) is True
    assert extract_json_array(_REASONING_CLEAN) == []
    parsed = parse_model_review_results({
        "results": [
            {"model": "minimax::MiniMax-M3", "verdict": "REVIEW", "text": _REASONING_CLEAN},
            {"model": "claude", "verdict": "REVIEW", "text": "[]\nNO_FINDINGS"},
        ]
    })
    assert parsed.responsive_models == ["minimax::MiniMax-M3#1", "claude#2"]
    assert parsed.quorum_met is True


def test_reasoning_wrapped_findings_are_still_extracted():
    parsed = extract_json_array(_REASONING_FINDING, normalize=True)
    assert parsed and parsed[0]["item"] == "code_quality" and parsed[0]["verdict"] == "FAIL"


def test_array_inside_a_whole_response_think_block_is_still_recovered():
    """A response that is ENTIRELY one <think>…</think> block (truncation, or a
    route that wraps everything) must not be stripped to nothing — the
    bracket-scan still recovers the array from inside the block."""
    whole = (
        '<think>\nReviewing… my verdict array is '
        '[{"item": "security_issues", "verdict": "FAIL", "severity": "critical", "reason": "y"}]'
        '\n</think>'
    )
    assert strip_leading_reasoning_block(whole) == whole  # nothing survives stripping → keep original
    parsed = extract_json_array(whole, normalize=True)
    assert parsed and parsed[0]["item"] == "security_issues" and parsed[0]["verdict"] == "FAIL"


def test_reasoning_block_does_not_launder_a_refusal():
    """Stripping <think> must not turn a real refusal into a clean verdict:
    prose after the block is still a non-response."""
    refusal = "<think>considering</think>\nI cannot review this diff. []\nNO_FINDINGS"
    assert empty_array_is_verified_clean(refusal) is False


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


def test_parse_model_review_results_enforces_required_items():
    good = json.dumps([
        {"item": "a", "verdict": "PASS", "severity": "critical", "reason": "ok"},
        {"item": "b", "verdict": "PASS", "severity": "critical", "reason": "ok"},
    ])
    partial = json.dumps([
        {"item": "a", "verdict": "PASS", "severity": "critical", "reason": "ok"},
    ])
    parsed = parse_model_review_results({
        "results": [
            {"model": "m1", "verdict": "REVIEW", "text": good},
            {"model": "m2", "verdict": "REVIEW", "text": partial},
        ]
    }, required_items=["a", "b"])

    assert parsed.responsive_models == ["m1#1"]
    assert parsed.actor_records[1].status == "partial"


def test_parse_model_review_results_quorum_and_degraded_reasons():
    good = json.dumps([{"item": "a", "verdict": "PASS", "severity": "critical", "reason": "ok"}])
    parsed = parse_model_review_results({
        "results": [
            {"model": "m1", "verdict": "REVIEW", "text": good},
            {"model": "m2", "verdict": "REVIEW", "text": good},
            {"model": "m3", "verdict": "ERROR", "text": "boom"},
        ]
    }, required_items=["a"])

    assert parsed.quorum_met is True
    assert parsed.degraded_reasons == ["DEGRADED: m3=error (quorum still met)"]


def test_responsive_identity_is_the_stable_slot_when_model_label_drifts():
    good = json.dumps([
        {"item": "a", "verdict": "PASS", "severity": "critical", "reason": "ok"},
    ])
    parsed = parse_model_review_results({"results": [
        {"model": "model-before", "slot_id": "slot-1", "text": good},
        {"model": "model-after", "slot_id": "slot-1", "text": good},
    ]}, required_items=["a"])

    assert parsed.responsive_models == ["model-before [slot-1]"]
    assert len(parsed.actor_records) == 2  # both chunk records remain forensic evidence


def test_emit_review_model_error_events(tmp_path):
    good = json.dumps([{"item": "a", "verdict": "PASS", "severity": "critical", "reason": "ok"}])
    parsed = parse_model_review_results({
        "results": [
            {"model": "m1", "verdict": "REVIEW", "text": good},
            {"model": "m2", "verdict": "ERROR", "text": "boom"},
        ]
    }, required_items=["a"])
    logs = tmp_path / "logs"
    logs.mkdir()
    ctx = SimpleNamespace(drive_logs=lambda: logs)

    emit_review_model_error_events(ctx, parsed, source="skill_review", skill_name="demo")

    data = (logs / "events.jsonl").read_text(encoding="utf-8")
    assert '"review_model_error"' in data
    assert '"skill": "demo"' in data
