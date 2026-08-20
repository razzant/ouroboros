"""Reading reviewer output: what parses, which actors count, and how the checklist aggregates.

Split out of ``tests/test_skill_review.py`` by theme: the fenced blocks and leading prose
``_parse_json_array`` tolerates and the malformed JSON it refuses, the actor findings that
are read, skipped or rejected and how duplicate models are counted by slot, and every
``_aggregate_status`` rule over the critical, soft and advisory checklist items.
"""

from __future__ import annotations

import json

from ouroboros.skill_review import _aggregate_status, _extract_actor_findings, _parse_json_array

from tests._skill_review_shared import (
    _make_actor,
    _pass_array_for_script_skill,
)


def test_parse_json_array_handles_fenced_code_blocks():
    text = "```json\n[{\"item\": \"x\", \"verdict\": \"PASS\"}]\n```"
    assert _parse_json_array(text) == [{"item": "x", "verdict": "PASS"}]


def test_parse_json_array_tolerates_leading_prose():
    text = "Sure! Here is the review:\n\n[{\"item\": \"x\", \"verdict\": \"PASS\"}]\nThanks."
    assert _parse_json_array(text) == [{"item": "x", "verdict": "PASS"}]


def test_parse_json_array_returns_empty_on_malformed_json():
    assert _parse_json_array("not json at all") == []
    assert _parse_json_array("[{broken") == []


def test_extract_actor_findings_reads_flat_text_field():
    """Regression: ``_parse_model_response`` flattens responses to
    ``{"model", "text", ...}`` — extract_actor_findings must read ``text``,
    not ``choices[0].message.content``."""
    result_json = {
        "results": [
            _make_actor("openai/gpt-5.5", _pass_array_for_script_skill()),
            _make_actor("google/gemini-3.5-flash", _pass_array_for_script_skill()),
        ]
    }
    findings, responded = _extract_actor_findings(result_json)
    assert len(findings) == 32
    assert responded == [
        "openai/gpt-5.5#1",
        "google/gemini-3.5-flash#2",
    ]
    assert all(f["verdict"] == "PASS" for f in findings)


def test_extract_actor_findings_skips_error_verdict_actors():
    """Transport errors (verdict=ERROR) must not contribute fake findings."""
    result_json = {
        "results": [
            _make_actor("openai/gpt-5.5", _pass_array_for_script_skill()),
            {
                "model": "google/gemini-3.5-flash",
                "request_model": "google/gemini-3.5-flash",
                "verdict": "ERROR",
                "text": "OpenRouter 404",
                "tokens_in": 0,
                "tokens_out": 0,
            },
        ]
    }
    findings, responded = _extract_actor_findings(result_json)
    assert all(f["model"] == "openai/gpt-5.5" for f in findings)
    assert responded == ["openai/gpt-5.5#1"]


def test_extract_actor_findings_rejects_partial_responses():
    """Phase 3 round 5 regression: a reviewer that returns only a subset
    of the 7 skill checklist items must NOT count toward quorum.

    Otherwise an actor returning just ``[{"item": "manifest_schema",
    "verdict": "PASS"}]`` would hand the pipeline a false PASS on the
    other 6 items simply by omitting them.
    """
    partial_text = json.dumps(
        [
            {"item": "manifest_schema", "verdict": "PASS", "severity": "critical", "reason": "ok"},
        ]
    )
    result_json = {
        "results": [
            _make_actor("openai/gpt-5.5", _pass_array_for_script_skill()),
            _make_actor("google/gemini-3.5-flash", partial_text),
        ]
    }
    findings, responded = _extract_actor_findings(result_json)
    # Partial reviewer must be excluded from both findings and responded set.
    assert "google/gemini-3.5-flash#2" not in responded
    assert responded == ["openai/gpt-5.5#1"]
    for f in findings:
        assert f["model"] == "openai/gpt-5.5"


def test_extract_actor_findings_counts_duplicate_models_by_slot():
    result_json = {
        "results": [
            _make_actor("anthropic/claude-opus-4.6", _pass_array_for_script_skill()),
            _make_actor("anthropic/claude-opus-4.6", _pass_array_for_script_skill()),
        ]
    }

    findings, responded = _extract_actor_findings(result_json)

    assert len(findings) == 32
    assert responded == [
        "anthropic/claude-opus-4.6#1",
        "anthropic/claude-opus-4.6#2",
    ]


def test_aggregate_status_clean_when_all_critical_pass():
    findings = [
        {"item": "manifest_schema", "verdict": "PASS", "severity": "critical"},
        {"item": "permissions_honesty", "verdict": "PASS", "severity": "critical"},
    ]
    assert _aggregate_status(findings, skill_type="script") == "clean"


def test_aggregate_status_blockers_on_critical_fail():
    findings = [
        {"item": "no_repo_mutation", "verdict": "FAIL", "severity": "critical", "reason": "writes to repo"},
    ]
    assert _aggregate_status(findings, skill_type="script") == "blockers"


def test_aggregate_status_blockers_on_critical_item_even_if_mislabeled(monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    findings = [
        {"item": "no_repo_mutation", "verdict": "FAIL", "severity": "advisory", "reason": "writes to repo"},
    ]
    assert _aggregate_status(findings, skill_type="script") == "blockers"


def test_aggregate_status_warnings_on_soft_fail(monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    findings = [
        {"item": "timeout_and_output_discipline", "verdict": "FAIL", "severity": "advisory", "reason": "unbounded loop"},
    ]
    assert _aggregate_status(findings, skill_type="script") == "warnings"


def test_aggregate_status_blockers_on_bug_hunting_fail(monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    findings = [
        {
            "item": "bug_hunting",
            "verdict": "FAIL",
            "severity": "critical",
            "reason": "plugin.py imports a missing module; fix by using the correct relative import",
        },
    ]
    assert _aggregate_status(findings, skill_type="script") == "blockers"


def test_aggregate_status_warnings_on_advisory_bug_hunting_fail(monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    findings = [
        {
            "item": "bug_hunting",
            "verdict": "FAIL",
            "severity": "advisory",
            "reason": "provider sometimes flakes; improve retry diagnostics later",
        },
    ]
    assert _aggregate_status(findings, skill_type="script") == "warnings"


def test_aggregate_status_skill_preflight_is_pending_and_fail_closed(monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    findings = [
        {"item": "skill_preflight", "verdict": "FAIL", "severity": "advisory", "reason": "syntax error"},
    ]
    # A deterministic preflight failure aggregates to PENDING (non-executable under
    # EVERY enforcement mode — stronger than advisory-overridable BLOCKERS) and
    # stays fail-closed even for hash-verified official_hub payloads.
    assert _aggregate_status(findings, skill_type="script") == "pending"
    assert _aggregate_status(findings, skill_type="script", review_profile="official_hub") == "pending"


def test_aggregate_status_no_repo_mutation_stays_hard_critical(monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    findings = [
        {
            "item": "no_repo_mutation",
            "verdict": "FAIL",
            "severity": "advisory",
            "reason": "skill writes to ~/Ouroboros/repo",
        },
    ]
    assert _aggregate_status(findings, skill_type="script") == "blockers"


def test_aggregate_status_extension_namespace_fail_is_critical_only_for_extension(monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    findings = [
        {"item": "extension_namespace_discipline", "verdict": "FAIL", "severity": "critical", "reason": "collides with built-in"},
    ]
    # For non-extension skills the extension_namespace_discipline FAIL is not blocking.
    assert _aggregate_status(findings, skill_type="script") == "warnings"
    # For extension skills it IS blocking.
    assert _aggregate_status(findings, skill_type="extension") == "blockers"


def test_aggregate_status_extension_namespace_advisory_fail_warns(monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    findings = [
        {
            "item": "extension_namespace_discipline",
            "verdict": "FAIL",
            "severity": "advisory",
            "reason": "minor naming cleanup would improve clarity",
        },
    ]
    assert _aggregate_status(findings, skill_type="extension") == "warnings"


def test_aggregate_status_widget_module_safety_fail_is_critical_only_for_module_widgets(monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    findings = [
        {"item": "widget_module_safety", "verdict": "FAIL", "severity": "critical", "reason": "touches localStorage"},
    ]
    assert _aggregate_status(findings, skill_type="script") == "warnings"
    assert _aggregate_status(findings, skill_type="extension", is_module_widget=False) == "blockers"
    assert _aggregate_status(findings, skill_type="extension", is_module_widget=True) == "blockers"


def test_aggregate_status_companion_process_advisory_fail_warns(monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    findings = [
        {
            "item": "companion_process_safety",
            "verdict": "FAIL",
            "severity": "advisory",
            "reason": "transient subprocess would benefit from clearer logging",
        },
    ]
    assert _aggregate_status(findings, skill_type="extension") == "warnings"
