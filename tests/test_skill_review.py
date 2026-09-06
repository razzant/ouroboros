"""Phase 3 regression tests for the verdict ``ouroboros.skill_review`` persists, and the grant it drives.

These tests mock out ``_handle_multi_model_review`` so no real LLM calls happen. This module
owns the end-to-end verdict: the clean result written to
``data/state/skills/<name>/review.json``, the auto-grant that follows it and the blockers
that stop it under each enforcement mode, the fail returned on a critical finding, the
distinct fail reasons kept for one item and surfaced in the retry coaching, and the
payload-mutation refusals between hashing and the frozen pack.

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
from ouroboros.skill_review import (
    SkillReviewOutcome,
    _extract_actor_findings,
    render_skill_review_block,
    review_skill,
)
from tests._skill_review_shared import (
    _NEW_SKILL_REVIEW_PASS_ITEMS,
    _build_skill,
    _make_actor,
    _make_ctx,
    _pass_array_for_script_skill,
    _patch_review,
)


def test_skill_advisory_pytest_guard_precedes_availability(tmp_path, monkeypatch):
    import ouroboros.skill_review as skill_review
    from ouroboros.tools import claude_advisory_review as advisory

    monkeypatch.setattr(
        advisory,
        "advisory_gate_unavailability_reason",
        lambda: (_ for _ in ()).throw(AssertionError("availability must not be evaluated")),
    )
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "sentinel")

    ctx = _make_ctx(tmp_path)
    assert skill_review._run_skill_advisory_pre_review(
        ctx, skill_name="weather", file_pack="pack"
    ) == {}
    assert not (ctx.drive_root / "logs" / "events.jsonl").exists()


def test_skill_advisory_missing_internal_symbol_is_loud_not_silent(tmp_path, monkeypatch):
    """The retired hasattr probe made a renamed internal silently no-op the
    skill advisory forever. The typed public entry (run_advisory_critic) makes
    that state a VISIBLE fail-open error with a durable warning event."""
    import ouroboros.skill_review as skill_review
    from ouroboros.tools import claude_advisory_review as advisory

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(advisory, "advisory_gate_unavailability_reason", lambda: None)
    monkeypatch.delattr(advisory, "_run_claude_advisory")

    ctx = _make_ctx(tmp_path)
    result = skill_review._run_skill_advisory_pre_review(
        ctx, skill_name="weather", file_pack="pack"
    )
    assert result.get("status") == "error"
    events_path = ctx.drive_root / "logs" / "events.jsonl"
    assert "skill_advisory_pre_review_warning" in events_path.read_text(encoding="utf-8")


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


# ---------------------------------------------------------------------------
# _parse_json_array + _extract_actor_findings
# ---------------------------------------------------------------------------


def test_duplicate_model_slots_keep_stable_identity_across_chunks_and_render_separately():
    actors = []
    for _chunk in range(2):
        for slot_id in ("skill-slot-a", "skill-slot-b"):
            actor = _make_actor("anthropic/claude-fable-5", _pass_array_for_script_skill())
            actor["slot_id"] = slot_id
            actors.append(actor)

    findings, responded = _extract_actor_findings({"results": actors})

    assert responded == [
        "anthropic/claude-fable-5 [skill-slot-a]",
        "anthropic/claude-fable-5 [skill-slot-b]",
    ]
    assert {finding["slot_id"] for finding in findings} == {
        "skill-slot-a", "skill-slot-b",
    }
    rendered = render_skill_review_block(SkillReviewOutcome(
        skill_name="demo", status="clean", findings=findings,
        reviewer_models=responded,
    ))
    assert rendered.count("Reviewer: anthropic/claude-fable-5 [skill-slot-a]") == 1
    assert rendered.count("Reviewer: anthropic/claude-fable-5 [skill-slot-b]") == 1


# ---------------------------------------------------------------------------
# _aggregate_status
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# review_skill end-to-end (mocked LLM)
# ---------------------------------------------------------------------------


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
    history_path = ctx.drive_root / "state" / "skills" / "weather" / "review_history.jsonl"
    terminal = json.loads(history_path.read_text(encoding="utf-8").splitlines()[-1])
    assert terminal["raw_actor_records"]
    assert all(record.get("slot_id") for record in terminal["raw_actor_records"])


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


def test_review_skill_refuses_payload_mutation_between_hash_and_frozen_pack(
    tmp_path, monkeypatch,
):
    import ouroboros.skill_review as sr

    skills_root = _build_skill(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    script = skills_root / "weather" / "scripts" / "fetch.py"
    real_compute = sr.compute_content_hash

    def mutate_after_hash(*args, **kwargs):
        digest = real_compute(*args, **kwargs)
        script.write_text("print('mutated after hash')\n", encoding="utf-8")
        return digest

    monkeypatch.setattr(sr, "compute_content_hash", mutate_after_hash)
    ctx = _make_ctx(tmp_path)
    with patch(
        "ouroboros.tools.review._handle_multi_model_review",
        side_effect=AssertionError("a mismatched frozen payload must not dispatch"),
    ):
        outcome = review_skill(ctx, "weather")

    assert outcome.status == "pending"
    assert "payload changed after hashing" in outcome.error
    assert outcome.paid is False and outcome.wave_id == ""


def test_review_skill_rebinds_manifest_to_the_frozen_payload(tmp_path, monkeypatch):
    import ouroboros.skill_review as sr

    skills_root = _build_skill(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    manifest = skills_root / "weather" / "SKILL.md"
    real_load = sr.load_bound_skill
    loads = 0

    def load_then_replace(binding):
        nonlocal loads
        loaded = real_load(binding)
        loads += 1
        if loads == 1:
            manifest.write_text(
                manifest.read_text(encoding="utf-8").replace(
                    "description: Check the weather.",
                    "description: Rebound frozen description."),
                encoding="utf-8",
            )
        return loaded

    monkeypatch.setattr(sr, "load_bound_skill", load_then_replace)
    captured = {}

    def fake_review(_ctx, *, prompt, **_kwargs):
        captured["prompt"] = prompt
        return json.dumps({"results": [
            _make_actor("reviewer-a", _pass_array_for_script_skill()),
            _make_actor("reviewer-b", _pass_array_for_script_skill()),
        ]})

    with patch("ouroboros.tools.review._handle_multi_model_review", side_effect=fake_review):
        outcome = review_skill(_make_ctx(tmp_path), "weather")

    assert outcome.status == "clean"
    assert loads >= 2
    assert "Rebound frozen description." in captured["prompt"]
    assert "Check the weather." not in captured["prompt"]


def test_review_skill_refuses_mutation_during_deterministic_preflight(tmp_path, monkeypatch):
    import ouroboros.skill_review as sr

    skills_root = _build_skill(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    script = skills_root / "weather" / "scripts" / "fetch.py"
    real_preflight = sr._run_deterministic_preflight

    def mutate_during_preflight(*args, **kwargs):
        result = real_preflight(*args, **kwargs)
        script.write_text("print('changed during preflight')\n", encoding="utf-8")
        return result

    monkeypatch.setattr(sr, "_run_deterministic_preflight", mutate_during_preflight)
    with patch(
        "ouroboros.tools.review._handle_multi_model_review",
        side_effect=AssertionError("a changed preflight payload must not dispatch"),
    ):
        outcome = review_skill(_make_ctx(tmp_path), "weather")

    assert outcome.status == "pending"
    assert "changed during deterministic preflight" in outcome.error
    assert outcome.paid is False and outcome.wave_id == ""


# -----------------------------------------------------------------------------
# v5.18 Skill Review Feedback Overhaul regression tests
# -----------------------------------------------------------------------------


# --- Block C3: structural consecutive-warnings convergence ----------------
