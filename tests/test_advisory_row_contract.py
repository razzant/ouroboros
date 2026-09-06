"""Advisory row validation preserves findings and honest unknown outcomes."""

import json
from types import SimpleNamespace

import pytest

from ouroboros.tools import preflight_review_run as run


@pytest.mark.parametrize("row", [
    {"item": "check", "verdict": "OK", "severity": "critical"},
    {"item": "check", "verdict": "FAILED", "severity": "critical"},
    {"item": "check", "verdict": "FAIL", "severity": "high"},
    {"item": "check", "verdict": "FAIL"},
    {"item": "check", "verdict": "FAIL", "severity": None},
    {"item": "check", "verdict": True, "severity": "critical"},
    {"item": "check", "verdict": "PASS", "severity": "warning"},
    {"item": None, "verdict": "PASS"},
])
def test_invalid_row_rejects_whole_array_without_dropping_findings(row):
    raw = json.dumps([{"item": "valid", "verdict": "PASS"}, row])
    assert run._parse_advisory_output(raw) == []
    assert run._needs_fallback_extraction([], raw)
    assert not run._is_clean_verdict(raw)


def test_enum_case_and_whitespace_are_normalized_once_without_changing_prose():
    rows = [{"item": "check", "verdict": " fail ", "severity": " CRITICAL ",
             "reason": "I withdraw this finding", "obligation_id": "ob-1"},
            {"item": "other", "verdict": " pass "}]
    parsed = run._parse_advisory_output(json.dumps(rows))
    assert parsed == [{**rows[0], "verdict": "FAIL", "severity": "critical"},
                      {**rows[1], "verdict": "PASS"}]


@pytest.mark.parametrize("invalid", [
    {"item": "second", "verdict": "FAIL", "severity": "high"},
    {"item": "second", "severity": "critical"},
    {"unexpected": "row"},
    42,
])
def test_invalid_final_array_cannot_fall_back_to_an_earlier_pass_example(monkeypatch, invalid):
    import ouroboros.llm as llm

    example = [{"item": "example", "verdict": "PASS", "severity": "advisory"}]
    final = [{"item": "real_bug", "verdict": "FAIL", "severity": "critical",
              "reason": "The requested behavior is broken."}, invalid]
    raw = f"Example: {json.dumps(example)}\nFinal findings:\n```json\n{json.dumps(final)}\n```"
    calls = []

    def extract(_self, **kwargs):
        calls.append(kwargs)
        return {"content": "UNEXTRACTABLE"}, {}

    monkeypatch.setattr(llm.LLMClient, "chat", extract)
    items = run._parse_advisory_output(raw)
    assert items == [], "an invalid final array must not select an earlier example"
    assert run._needs_fallback_extraction(items, raw)
    assert run._llm_extract_advisory_items(raw, SimpleNamespace()) == []
    assert len(calls) == 1 and raw in calls[0]["messages"][0]["content"]


@pytest.mark.parametrize("fenced", [False, True])
def test_valid_final_findings_survive_prose_and_unrelated_numeric_arrays(fenced):
    findings = [{"item": "real_bug", "verdict": "FAIL", "severity": "critical"}]
    body = json.dumps(findings)
    if fenced:
        body = f"```json\n{body}\n```"
    raw = f"Counts: [1, 2]\nFindings: {body}\nMore counts: [3, 4]"
    assert run._parse_advisory_output(raw) == findings


def test_malformed_json_array_reaches_existing_extraction_with_the_whole_source(monkeypatch):
    import ouroboros.llm as llm

    raw = json.dumps([{"item": "check", "verdict": "FAILED", "severity": "critical",
                       "reason": "The reviewer found a bug."}])
    extracted = [{"item": "check", "verdict": "FAIL", "severity": "critical",
                  "reason": "The reviewer found a bug."}]
    calls = []

    def chat(_self, **kwargs):
        calls.append(kwargs)
        return {"content": json.dumps(extracted)}, {"cost": 0.001}

    monkeypatch.setattr(llm.LLMClient, "chat", chat)
    assert run._llm_extract_advisory_items(raw, SimpleNamespace()) == extracted
    assert len(calls) == 1
    assert raw in calls[0]["messages"][0]["content"]


@pytest.mark.parametrize("extraction", ["UNEXTRACTABLE", "[]", "unchanged"])
def test_unresolved_severity_never_becomes_clean_or_critical(monkeypatch, extraction):
    import ouroboros.llm as llm

    raw = '[{"item":"check","verdict":"FAIL","severity":"high"}]'
    output = raw if extraction == "unchanged" else extraction
    monkeypatch.setattr(llm.LLMClient, "chat", lambda *a, **k: ({"content": output}, {}))
    assert run._llm_extract_advisory_items(raw, SimpleNamespace()) == []
    assert not run._is_clean_verdict(raw)


def test_transport_to_advisory_consumer_preserves_normalized_findings(tmp_path, monkeypatch):
    from ouroboros.tools import claude_advisory_review as advisory
    from ouroboros import reviewer_slot_config

    raw = '[{"item":"check","verdict":" fail ","severity":" CRITICAL ","reason":"bug"}]'
    monkeypatch.setattr(run, "advisory_review_route", lambda: "api_chat")
    monkeypatch.setattr(reviewer_slot_config, "advisory_slot_config", lambda: SimpleNamespace(effort="low"))
    monkeypatch.setattr("ouroboros.provider_models.model_has_credentials", lambda model: True)
    monkeypatch.setattr(advisory, "_advisory_native_model", lambda: "test/model")
    monkeypatch.setattr(advisory, "_build_advisory_prompt", lambda *a, **k: "review prompt")
    monkeypatch.setattr(advisory, "_predispatch_size_skip", lambda *a: None)
    monkeypatch.setattr(advisory, "_run_advisory_native", lambda *a, **k: (
        SimpleNamespace(success=True, result_text=raw, session_id="test", cost_usd=0,
                        usage={}, error="", stderr_tail=""), "test/model"))
    ctx = SimpleNamespace(task_id="row-contract", task_metadata={})
    items, source, model, _ = run._run_claude_advisory(
        tmp_path, "test", ctx, options={"include_repo_diff": False},
    )
    assert items == [{"item": "check", "verdict": "FAIL", "severity": "critical", "reason": "bug"}]
    assert source == raw and model == "test/model"
