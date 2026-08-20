"""Phase 3 regression tests for the verdict ``ouroboros.skill_review`` persists, and the grant it drives.

These tests mock out ``_handle_multi_model_review`` so no real LLM calls happen. This module
owns the end-to-end verdict: the clean result written to
``data/state/skills/<name>/review.json``, the auto-grant that follows it and the blockers
that stop it under each enforcement mode, the fail returned on a critical finding, and the
distinct fail reasons kept for one item and surfaced in the retry coaching.

The advisory pre-review, the parsing and aggregation layer, the prompt and payload packs,
the rendered review block and the rebuttal ledger were split verbatim into
``tests/test_skill_advisory_pre_review.py``, ``tests/test_skill_review_aggregation.py``,
``tests/test_skill_review_packs.py``, ``tests/test_skill_review_rendering.py`` and
``tests/test_skill_review_rebuttals.py``; the reviewer-array builders, the skill builder and
the context factory they share live in ``tests/_skill_review_shared.py``.
"""

from __future__ import annotations

import json
import pathlib
from unittest.mock import patch

from ouroboros.skill_loader import compute_content_hash, load_review_state
from ouroboros.skill_review import SkillReviewOutcome, render_skill_review_block, review_skill

from tests._skill_review_shared import (
    _NEW_SKILL_REVIEW_PASS_ITEMS,
    _build_skill,
    _make_actor,
    _make_ctx,
    _pass_array_for_script_skill,
    _patch_review,
)


def _fail_array_on_manifest() -> str:
    return json.dumps(
        [
            {"item": "manifest_schema", "verdict": "FAIL", "severity": "critical", "reason": "type does not match payload"},
            {"item": "permissions_honesty", "verdict": "PASS", "severity": "critical", "reason": "ok"},
            {"item": "no_repo_mutation", "verdict": "PASS", "severity": "critical", "reason": "ok"},
            {"item": "path_confinement", "verdict": "PASS", "severity": "critical", "reason": "ok"},
            {"item": "env_allowlist", "verdict": "PASS", "severity": "critical", "reason": "ok"},
            {"item": "timeout_and_output_discipline", "verdict": "PASS", "severity": "advisory", "reason": "ok"},
            {"item": "extension_namespace_discipline", "verdict": "PASS", "severity": "critical", "reason": "ok"},
            {"item": "widget_module_safety", "verdict": "PASS", "severity": "critical", "reason": "ok"},
            *_NEW_SKILL_REVIEW_PASS_ITEMS,
        ]
    )


def _advisory_only_array() -> str:
    return json.dumps(
        [
            {"item": "manifest_schema", "verdict": "PASS", "severity": "critical", "reason": "ok"},
            {"item": "permissions_honesty", "verdict": "PASS", "severity": "critical", "reason": "ok"},
            {"item": "no_repo_mutation", "verdict": "PASS", "severity": "critical", "reason": "ok"},
            {"item": "path_confinement", "verdict": "PASS", "severity": "critical", "reason": "ok"},
            {"item": "env_allowlist", "verdict": "PASS", "severity": "critical", "reason": "ok"},
            {"item": "timeout_and_output_discipline", "verdict": "FAIL", "severity": "advisory", "reason": "unbounded loop"},
            {"item": "extension_namespace_discipline", "verdict": "PASS", "severity": "critical", "reason": "ok"},
            {"item": "widget_module_safety", "verdict": "PASS", "severity": "critical", "reason": "ok"},
            *_NEW_SKILL_REVIEW_PASS_ITEMS,
        ]
    )


def _mark_self_authored(skill_dir: pathlib.Path, drive_root: pathlib.Path) -> None:
    payload = {
        "schema_version": 1,
        "origin": "self_authored",
        "task_id": "task-1",
        "created_at": "2026-05-13T00:00:00+00:00",
    }
    body = json.dumps(payload) + "\n"
    (skill_dir / ".self_authored.json").write_text(body, encoding="utf-8")
    state_dir = drive_root / "state" / "skills" / skill_dir.name
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "self_authored.json").write_text(body, encoding="utf-8")


def test_review_skill_persists_clean_verdict(tmp_path, monkeypatch):
    skills_root = _build_skill(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    ctx = _make_ctx(tmp_path)
    pass_array = _pass_array_for_script_skill()
    canned = json.dumps(
        {
            "results": [
                _make_actor("openai/gpt-5.5", pass_array),
                _make_actor("google/gemini-3.5-flash", pass_array),
                _make_actor("anthropic/claude-opus-4.6", pass_array),
            ]
        }
    )
    with _patch_review(canned):
        outcome = review_skill(ctx, "weather")

    assert isinstance(outcome, SkillReviewOutcome)
    assert outcome.status == "clean"
    assert outcome.error == ""
    assert outcome.reviewer_models[:2] == [
        "openai/gpt-5.5#1",
        "google/gemini-3.5-flash#2",
    ]
    persisted = load_review_state(ctx.drive_root, "weather")
    assert persisted.status == "clean"
    assert persisted.content_hash == outcome.content_hash
    # Content hash must actually match the on-disk snapshot so the
    # stale-review gate stays honest.
    expected_hash = compute_content_hash(skills_root / "weather")
    assert persisted.content_hash == expected_hash


def test_review_skill_auto_grants_after_clean_when_enabled(tmp_path, monkeypatch):
    from ouroboros.skill_loader import load_skill_grants
    import ouroboros.config as config

    skills_root = _build_skill(
        tmp_path,
        env_from_settings=["OPENROUTER_API_KEY"],
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "true")
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "missing_settings.json")
    ctx = _make_ctx(tmp_path)
    _mark_self_authored(skills_root / "weather", ctx.drive_root)
    pass_array = _pass_array_for_script_skill()
    canned = json.dumps(
        {
            "results": [
                _make_actor("openai/gpt-5.5", pass_array),
                _make_actor("openai/gpt-5.5", pass_array),
            ]
        }
    )

    with _patch_review(canned):
        outcome = review_skill(ctx, "weather")

    assert outcome.status == "clean"
    assert outcome.auto_flow is True
    assert outcome.requested_keys == ["OPENROUTER_API_KEY"]
    assert outcome.auto_granted_keys == ["OPENROUTER_API_KEY"]
    grants = load_skill_grants(ctx.drive_root, "weather")
    assert grants["granted_keys"] == ["OPENROUTER_API_KEY"]
    assert grants["content_hash"] == outcome.content_hash


def test_review_skill_auto_grant_skips_blockers_under_blocking(tmp_path, monkeypatch):
    from ouroboros.skill_loader import load_skill_grants
    import ouroboros.config as config

    skills_root = _build_skill(
        tmp_path,
        env_from_settings=["OPENROUTER_API_KEY"],
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "true")
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "missing_settings.json")
    ctx = _make_ctx(tmp_path)
    canned = json.dumps(
        {
            "results": [
                _make_actor("openai/gpt-5.5", _fail_array_on_manifest()),
                _make_actor("google/gemini-3.5-flash", _fail_array_on_manifest()),
            ]
        }
    )

    with _patch_review(canned):
        outcome = review_skill(ctx, "weather")

    assert outcome.status == "blockers"
    assert outcome.requested_keys == ["OPENROUTER_API_KEY"]
    assert outcome.auto_granted_keys == []
    grants = load_skill_grants(ctx.drive_root, "weather")
    assert grants["granted_keys"] == []
    assert not grants.get("content_hash")


def test_review_skill_auto_grants_blockers_under_advisory(tmp_path, monkeypatch):
    from ouroboros.skill_loader import load_skill_grants
    import ouroboros.config as config

    skills_root = _build_skill(
        tmp_path,
        env_from_settings=["OPENROUTER_API_KEY"],
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "true")
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "missing_settings.json")
    ctx = _make_ctx(tmp_path)
    canned = json.dumps(
        {
            "results": [
                _make_actor("openai/gpt-5.5", _fail_array_on_manifest()),
                _make_actor("google/gemini-3.5-flash", _fail_array_on_manifest()),
            ]
        }
    )

    with _patch_review(canned):
        outcome = review_skill(ctx, "weather")

    assert outcome.status == "blockers"
    assert outcome.requested_keys == ["OPENROUTER_API_KEY"]
    assert outcome.auto_granted_keys == ["OPENROUTER_API_KEY"]
    grants = load_skill_grants(ctx.drive_root, "weather")
    assert grants["granted_keys"] == ["OPENROUTER_API_KEY"]
    assert grants["content_hash"] == outcome.content_hash


def test_review_skill_auto_grant_skips_deterministic_preflight_blocker(tmp_path, monkeypatch):
    from ouroboros.skill_loader import load_skill_grants
    import ouroboros.config as config

    skills_root = _build_skill(
        tmp_path,
        env_from_settings=["OPENROUTER_API_KEY"],
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "true")
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "missing_settings.json")
    monkeypatch.setattr(
        "ouroboros.tools.skill_preflight._handle_skill_preflight",
        lambda *_args, **_kwargs: json.dumps({"ok": False, "error": "preflight failed"}),
    )
    ctx = _make_ctx(tmp_path)

    outcome = review_skill(ctx, "weather")

    # Deterministic preflight failures persist as PENDING (non-executable under
    # every enforcement mode), never BLOCKERS (which advisory could override).
    assert outcome.status == "pending"
    assert outcome.requested_keys == ["OPENROUTER_API_KEY"]
    assert outcome.auto_granted_keys == []
    grants = load_skill_grants(ctx.drive_root, "weather")
    assert grants["granted_keys"] == []
    assert not grants.get("content_hash")


def test_render_skill_review_block_shows_auto_granted_keys():
    outcome = SkillReviewOutcome(
        skill_name="weather",
        status="clean",
        content_hash="abc1234567890",
        reviewer_models=["reviewer-a"],
        auto_granted_keys=["OPENROUTER_API_KEY"],
    )

    rendered = render_skill_review_block(outcome)

    assert "Reviewers: reviewer-a" in rendered
    assert "Auto-granted: keys: OPENROUTER_API_KEY" in rendered


def test_review_skill_returns_fail_on_critical_finding(tmp_path, monkeypatch):
    skills_root = _build_skill(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    ctx = _make_ctx(tmp_path)
    canned = json.dumps(
        {
            "results": [
                _make_actor("openai/gpt-5.5", _fail_array_on_manifest()),
                _make_actor("google/gemini-3.5-flash", _fail_array_on_manifest()),
                _make_actor("anthropic/claude-opus-4.6", _fail_array_on_manifest()),
            ]
        }
    )
    with _patch_review(canned):
        outcome = review_skill(ctx, "weather")
    assert outcome.status == "blockers"
    reasons = {f["reason"] for f in outcome.findings if f["verdict"] == "FAIL"}
    assert any("type does not match payload" in r for r in reasons)


def test_review_skill_keeps_distinct_fail_reasons_for_same_item(tmp_path, monkeypatch):
    skills_root = _build_skill(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    ctx = _make_ctx(tmp_path)
    items = [
        item for item in json.loads(_pass_array_for_script_skill())
        if item["item"] != "bug_hunting"
    ]
    items.extend([
        {
            "item": "bug_hunting",
            "verdict": "FAIL",
            "severity": "critical",
            "reason": "plugin.py::run can overflow the retry buffer",
        },
        {
            "item": "bug_hunting",
            "verdict": "FAIL",
            "severity": "critical",
            "reason": "api_client.py::parse_response assumes choices[0]",
        },
    ])
    duplicated_bug_hunting = json.dumps(items)
    canned = json.dumps(
        {
            "results": [
                _make_actor("openai/gpt-5.5", duplicated_bug_hunting),
                _make_actor("google/gemini-3.5-flash", duplicated_bug_hunting),
            ]
        }
    )
    # The advisory pre-review moved to the prompt owner with the per-attempt
    # assembly that calls it; patch it where that caller reads it.
    with patch("ouroboros.skill_review_prompt._run_skill_advisory_pre_review", return_value={"status": "empty"}):
        with _patch_review(canned):
            outcome = review_skill(ctx, "weather")
    bug_reasons = [
        f["reason"] for f in outcome.findings
        if f.get("item") == "bug_hunting" and f.get("verdict") == "FAIL"
    ]
    assert outcome.status == "blockers"
    assert "plugin.py::run can overflow the retry buffer" in bug_reasons
    assert "api_client.py::parse_response assumes choices[0]" in bug_reasons


def test_render_skill_review_block_keeps_same_item_fail_reasons_in_retry_coaching():
    findings = [
        {
            "item": "bug_hunting",
            "verdict": "FAIL",
            "severity": "critical",
            "reason": "plugin.py::run can overflow the retry buffer",
            "model": "model-a",
        },
        {
            "item": "bug_hunting",
            "verdict": "FAIL",
            "severity": "critical",
            "reason": "api_client.py::parse_response assumes choices[0]",
            "model": "model-b",
        },
    ]
    rendered = render_skill_review_block(
        SkillReviewOutcome(
            skill_name="weather",
            status="blockers",
            content_hash="abc",
            findings=findings,
            reviewer_models=["model-a", "model-b"],
        ),
        attempt_idx=2,
    )
    assert "plugin.py::run can overflow the retry buffer" in rendered
    assert "api_client.py::parse_response assumes choices[0]" in rendered


def test_review_skill_returns_advisory_for_soft_only_fail(tmp_path, monkeypatch):
    skills_root = _build_skill(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    ctx = _make_ctx(tmp_path)
    canned = json.dumps(
        {
            "results": [
                _make_actor("openai/gpt-5.5", _advisory_only_array()),
                _make_actor("google/gemini-3.5-flash", _advisory_only_array()),
                _make_actor("anthropic/claude-opus-4.6", _advisory_only_array()),
            ]
        }
    )
    with _patch_review(canned):
        outcome = review_skill(ctx, "weather")
    assert outcome.status == "warnings"


def test_review_skill_returns_warnings_in_advisory_mode(tmp_path, monkeypatch):
    skills_root = _build_skill(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    ctx = _make_ctx(tmp_path)
    canned = json.dumps(
        {
            "results": [
                _make_actor("openai/gpt-5.5", _advisory_only_array()),
                _make_actor("openai/gpt-5.5", _advisory_only_array()),
            ]
        }
    )
    with _patch_review(canned):
        outcome = review_skill(ctx, "weather")
    assert outcome.status == "warnings"
