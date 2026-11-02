"""The skill verdict's vocabulary: which checklist items block, and how a persisted verdict reads back.

The severity policy (``skill_severity_policy``) is the checklist's own rule — hard trust-boundary items
block whatever the reviewer said, severity-driven items follow the reviewer's word (advisory for the
extension-only items on a non-extension and for every severity-driven item under ``official_hub``),
advisory-only and unknown items never block — applied to findings before the shared verdict is read
and again at load (``status_from_findings``). A deterministic preflight failure stays ``pending``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros.skill_review_status import (
    EXTENSION_ONLY_ITEMS,
    HARD_CRITICAL_ITEMS,
    SEVERITY_DRIVEN_ITEMS,
    STATUS_PENDING,
    skill_severity_policy,
    status_from_findings,
)


def _fail(item, severity="critical", **extra):
    return {"item": item, "verdict": "FAIL", "severity": severity, "reason": "x", **extra}


def test_the_policy_names_every_checklist_item_and_the_default():
    from ouroboros.tools.review_skill import SKILL_REVIEW_ITEMS

    policy = skill_severity_policy("script")
    assert set(SKILL_REVIEW_ITEMS) >= HARD_CRITICAL_ITEMS | SEVERITY_DRIVEN_ITEMS
    assert all(policy[item] == "critical" for item in HARD_CRITICAL_ITEMS)
    assert {policy[item] for item in SEVERITY_DRIVEN_ITEMS - EXTENSION_ONLY_ITEMS} == {"seat"}
    assert {policy[item] for item in EXTENSION_ONLY_ITEMS} == {"advisory"}, "a script carries no extension surface"
    assert policy["*"] == "advisory", "an item outside the table never blocks"
    extension = skill_severity_policy("extension")
    assert {extension[item] for item in EXTENSION_ONLY_ITEMS} == {"seat"}
    hub = skill_severity_policy("extension", review_profile="official_hub")
    assert {hub[item] for item in SEVERITY_DRIVEN_ITEMS} == {"advisory"} and hub["no_repo_mutation"] == "critical"


def test_status_clean_when_all_critical_pass():
    findings = [{"item": "manifest_schema", "verdict": "PASS", "severity": "critical"},
                {"item": "permissions_honesty", "verdict": "PASS", "severity": "critical"}]
    assert status_from_findings(findings, "script") == "clean"
    assert status_from_findings([], "script") == "clean"


@pytest.mark.parametrize("severity", ["critical", "advisory"])
def test_status_blockers_on_a_hard_item_whatever_the_seat_said(severity):
    assert status_from_findings([_fail("no_repo_mutation", severity)], "script") == "blockers"
    from ouroboros.review_ledger import apply_severity_policy

    judged = apply_severity_policy({"status": "responded", "findings": [_fail("no_repo_mutation", severity)]},
                                   skill_severity_policy("script"))
    assert judged["findings"][0]["severity"] == "critical" and judged["verdict"] == "FAIL", "the ledger's one rule"


def test_status_warnings_on_an_advisory_only_item():
    assert status_from_findings([_fail("timeout_and_output_discipline", "critical")], "script") == "warnings"
    assert status_from_findings([_fail("error_handling", "critical")], "script") == "warnings"


def test_status_follows_the_seat_on_a_severity_driven_item():
    assert status_from_findings([_fail("bug_hunting", "critical")], "script") == "blockers"
    assert status_from_findings([_fail("bug_hunting", "advisory")], "script") == "warnings"
    assert status_from_findings([_fail("bug_hunting", "warning")], "script") == "warnings"
    assert status_from_findings([_fail("bug_hunting", "")], "script") == "blockers", "an unknown severity stays conservative"


def test_status_skill_preflight_is_pending_and_fail_closed():
    findings = [_fail("skill_preflight", "advisory")]
    assert status_from_findings(findings, "script") == STATUS_PENDING
    assert status_from_findings(findings, "script", review_profile="official_hub") == STATUS_PENDING
    admission = [{"item": "plugin_api_admission", "verdict": "FAIL", "severity": "critical", "model": "plugin_api_admission"}]
    assert status_from_findings(admission, "extension") == STATUS_PENDING


def test_extension_only_items_block_only_an_extension():
    finding = [_fail("extension_namespace_discipline", "critical")]
    assert status_from_findings(finding, "script") == "warnings"
    assert status_from_findings(finding, "extension") == "blockers"
    assert status_from_findings([_fail("extension_namespace_discipline", "advisory")], "extension") == "warnings"
    widget = [_fail("widget_module_safety", "critical")]
    assert status_from_findings(widget, "script") == "warnings"
    assert status_from_findings(widget, "extension", is_module_widget=False) == "blockers"
    assert status_from_findings(widget, "extension", is_module_widget=True) == "blockers"


def test_companion_process_advisory_fail_warns():
    assert status_from_findings([_fail("companion_process_safety", "advisory")], "extension") == "warnings"
    assert status_from_findings([_fail("companion_process_safety", "critical")], "extension") == "blockers"


def test_an_unknown_item_never_blocks_and_an_adviser_is_never_counted():
    assert status_from_findings([_fail("invented_item", "critical")], "script") == "warnings"
    assert status_from_findings([_fail("no_repo_mutation", "critical", additional=True)], "script") == "clean"


def test_official_hub_profile_keeps_hard_trust_boundary_blockers_and_downgrades_the_rest():
    hard = [_fail("inject_chat_minimization", "critical")]
    assert status_from_findings(hard, "extension") == "blockers"
    assert status_from_findings(hard, "extension", review_profile="official_hub") == "blockers"
    soft = [_fail("bug_hunting", "critical")]
    assert status_from_findings(soft, "extension") == "blockers"
    assert status_from_findings(soft, "extension", review_profile="official_hub") == "warnings"


def test_a_new_review_mints_no_hub_profile(tmp_path):
    """A review of a hub payload gets the ordinary rule (no profile): a source's trust is an install basis
    read by the gate, never a softer review. The subject carries the policy a new wave is parsed under."""
    from ouroboros.tools.review_skill import FrozenSkillSubject

    subject = FrozenSkillSubject.__new__(FrozenSkillSubject)
    object.__setattr__(subject, "severity_policy", skill_severity_policy("extension", review_profile=""))
    assert subject.severity_policy["bug_hunting"] == "seat"


def test_load_review_state_drops_official_hub_when_sidecar_missing(tmp_path):
    from ouroboros.skill_loader import load_review_state, skill_state_dir

    d = skill_state_dir(tmp_path, "demo")
    d.mkdir(parents=True, exist_ok=True)
    (d / "review.json").write_text(json.dumps({
        "status": "warnings", "content_hash": "h1", "review_profile": "official_hub",
        "findings": [{"item": "bug_hunting", "verdict": "FAIL", "severity": "critical", "model": "m"}]}))
    payload = tmp_path / "payload"
    payload.mkdir()
    state = load_review_state(tmp_path, "demo", skill_type="script", skill_dir=payload)
    assert state.review_profile == "" and state.status == "blockers"
    (payload / ".ouroboroshub.json").write_text("{}", encoding="utf-8")
    state2 = load_review_state(tmp_path, "demo", skill_type="script", skill_dir=payload)
    assert state2.review_profile == "official_hub" and state2.status == "warnings"


def test_load_review_state_preserves_pending_for_preflight_fail(tmp_path):
    from ouroboros.skill_loader import load_review_state, skill_state_dir

    d = skill_state_dir(tmp_path, "demo")
    d.mkdir(parents=True, exist_ok=True)
    (d / "review.json").write_text(json.dumps({
        "status": "pending", "content_hash": "h1",
        "findings": [{"item": "skill_preflight", "verdict": "FAIL", "severity": "critical", "model": "deterministic_preflight"}]}))
    assert load_review_state(tmp_path, "demo", skill_type="script").status == STATUS_PENDING


def test_a_legacy_review_json_still_admits_its_exact_bytes_and_reads_the_fold_fields(tmp_path):
    """A record an earlier version wrote (no record_id, no governance, no seats) keeps admitting the bytes
    it names, by the same rule; the fold's fields read back when present and are absent otherwise."""
    from ouroboros.skill_loader import SkillReviewState, load_review_state, save_review_state, skill_state_dir

    d = skill_state_dir(tmp_path, "legacy")
    d.mkdir(parents=True, exist_ok=True)
    legacy = {"status": "warnings", "content_hash": "h-legacy", "timestamp": "2026-09-01T00:00:00+00:00",
              "reviewer_models": ["openai/gpt-5.5#1"], "prompt_chars": 1010844, "cost_usd": 0.5, "raw_result": "",
              "raw_actor_records": [{"model_id": "openai/gpt-5.5", "status": "responded", "slot_id": "s1", "parsed_items": []}],
              "findings": [{"item": "error_handling", "verdict": "FAIL", "severity": "advisory", "reason": "swallows", "model": "openai/gpt-5.5"},
                           {"item": "manifest_schema", "verdict": "PASS", "severity": "critical", "reason": "ok", "model": "openai/gpt-5.5"}]}
    (d / "review.json").write_text(json.dumps(legacy), encoding="utf-8")
    state = load_review_state(tmp_path, "legacy", skill_type="script")
    assert (state.status, state.content_hash, state.record_id, state.governance, state.seats) == ("warnings", "h-legacy", "", {}, [])
    gate = state.gate_for("h-legacy", enforcement="blocking")
    assert gate["executable_review"] is True and gate["stale"] is False
    assert state.gate_for("h-other", enforcement="blocking")["executable_review"] is False
    projected = SkillReviewState(status="clean", content_hash="h2", timestamp="t", record_id="rl-1",
                                 governance={"owed_chars": 12}, seats=[{"seat_id": "s1"}])
    save_review_state(tmp_path, "legacy", projected)
    back = load_review_state(tmp_path, "legacy", skill_type="script")
    assert (back.record_id, back.governance, back.seats) == ("rl-1", {"owed_chars": 12}, [{"seat_id": "s1"}])


def test_deterministic_preflight_persists_pending_and_is_non_executable(tmp_path, monkeypatch):
    """A deterministic preflight FAILURE is a structural fact, not an LLM verdict: it persists as PENDING,
    non-executable under EVERY enforcement mode, and grants nothing."""
    import ouroboros.skill_review as sr
    import ouroboros.tools.skill_preflight as pf
    from ouroboros.skill_review_status import skill_review_gate

    monkeypatch.setattr(pf, "_handle_skill_preflight",
                        lambda ctx, skill=None, _resolved_binding=None: json.dumps({"ok": False, "error": "bad manifest schema"}))
    granted = []
    monkeypatch.setattr(sr, "auto_grant_if_enabled", lambda drive_root, skill: granted.append(skill))
    monkeypatch.setattr(sr, "save_review_state", lambda *a, **k: None)
    skill = SimpleNamespace(name="demo", review=None,
                            manifest=SimpleNamespace(env_from_settings=["OPENROUTER_API_KEY"], permissions=[], subscribe_events=[]))
    outcome = sr._run_deterministic_preflight(SimpleNamespace(), tmp_path, skill, "hash123", persist=True)
    assert outcome is not None and outcome.status == sr.STATUS_PENDING
    assert skill_review_gate(outcome.status, enforcement="advisory")["executable_review"] is False
    assert skill_review_gate(outcome.status, enforcement="blocking")["executable_review"] is False
    assert granted == []
    assert outcome.requested_keys == ["OPENROUTER_API_KEY"]
