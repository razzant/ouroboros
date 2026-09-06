"""The rendered review block: what the agent is shown, and what it is never shown.

Split out of ``tests/test_skill_review.py`` by theme: the concrete fail reasons the history
section renders and its legacy-signature fallback, the findings grouped by reviewer
verbatim, the self-verification at attempt two and the circuit breaker at attempt three, the
payload dict form, and the raw JSON block the tool result never contains.
"""

from __future__ import annotations

from ouroboros.skill_loader import compute_content_hash
from ouroboros.tools.registry import ToolContext


def test_skill_review_history_section_renders_concrete_fail_reasons():
    from ouroboros.skill_review import _build_skill_review_history_section

    history = [
        {
            "status": "blockers",
            "content_hash": "abcdef123456",
            "fail_findings": [
                {
                    "item": "companion_process_safety",
                    "severity": "critical",
                    "reason_excerpt": "ffmpeg invocation tagged as long-lived",
                    "model": "openai/gpt-5.5",
                },
                {
                    "item": "bug_hunting",
                    "severity": "advisory",
                    "reason_excerpt": "missing exception handling",
                },
            ],
        },
        {
            "status": "blockers",
            "content_hash": "abcdef123456",
            "fail_findings": [
                {
                    "item": "companion_process_safety",
                    "severity": "critical",
                    "reason_excerpt": "still flagged on round 2",
                    "model": "openai/gpt-5.5",
                },
            ],
        },
    ]
    section = _build_skill_review_history_section(history, attempt_idx=3)
    assert "## Previous skill review attempts" in section
    assert "companion_process_safety" in section
    assert "ffmpeg invocation tagged as long-lived" in section
    assert "model=openai/gpt-5.5" in section
    assert "**IMPORTANT RULES FOR THIS REVIEW:**" in section
    assert "Do NOT rephrase prior findings under a different checklist `item` name" in section
    # Convergence rule fires from the 3rd content-hash attempt onward.
    assert "Convergence:" in section or "convergence" in section.lower()


def test_skill_review_history_section_falls_back_to_signature_for_legacy_entries():
    from ouroboros.skill_review import _build_skill_review_history_section

    history = [
        {
            "status": "warnings",
            "content_hash": "old",
            "failure_signature": ["bug_hunting:FAIL:advisory"],
        }
    ]
    section = _build_skill_review_history_section(history)
    assert "Failure signature:" in section
    assert "bug_hunting:FAIL:advisory" in section


def test_render_skill_review_block_groups_findings_by_reviewer_verbatim():
    from ouroboros.skill_review import SkillReviewOutcome, render_skill_review_block

    long_reason = (
        "This skill spawns ffmpeg to transcode a single audio file in the request "
        "handler. The subprocess terminates within the handler scope and does not "
        "outlive the request — it is not a long-lived companion process."
    )
    outcome = SkillReviewOutcome(
        skill_name="demo",
        status="blockers",
        content_hash="abc12345",
        reviewer_models=["openai/gpt-5.5", "google/gemini-3.5-flash"],
        findings=[
            {
                "item": "companion_process_safety",
                "verdict": "FAIL",
                "severity": "critical",
                "reason": long_reason,
                "model": "openai/gpt-5.5",
            },
            {
                "item": "companion_process_safety",
                "verdict": "PASS",
                "severity": "critical",
                "reason": "Transient subprocess, not a long-lived companion.",
                "model": "google/gemini-3.5-flash",
            },
        ],
    )
    markdown = render_skill_review_block(outcome, attempt_idx=1)
    assert "Reviewer: openai/gpt-5.5" in markdown
    assert "Reviewer: google/gemini-3.5-flash" in markdown
    assert long_reason in markdown
    assert "[FAIL critical] companion_process_safety" in markdown
    assert "[PASS] companion_process_safety" in markdown


def test_render_skill_review_block_emits_self_verification_at_attempt_two():
    from ouroboros.skill_review import SkillReviewOutcome, render_skill_review_block

    outcome = SkillReviewOutcome(
        skill_name="demo",
        status="blockers",
        findings=[
            {
                "item": "bug_hunting",
                "verdict": "FAIL",
                "severity": "advisory",
                "reason": "missing error handling",
                "model": "openai/gpt-5.5",
            }
        ],
    )
    markdown_first = render_skill_review_block(outcome, attempt_idx=1)
    assert "Self-verification required" not in markdown_first

    markdown_second = render_skill_review_block(outcome, attempt_idx=2)
    assert "Self-verification required before next skill_review" in markdown_second
    assert "Status: addressed / rebutted / pending" in markdown_second
    assert "Circuit-breaker hint" not in markdown_second


def test_render_skill_review_block_emits_circuit_breaker_at_attempt_three():
    from ouroboros.skill_review import SkillReviewOutcome, render_skill_review_block

    outcome = SkillReviewOutcome(
        skill_name="demo",
        status="blockers",
        findings=[
            {
                "item": "bug_hunting",
                "verdict": "FAIL",
                "severity": "advisory",
                "reason": "missing error handling",
                "model": "openai/gpt-5.5",
            }
        ],
    )
    markdown = render_skill_review_block(outcome, attempt_idx=3)
    assert "Self-verification required" in markdown
    assert "Circuit-breaker hint (attempt 3+)" in markdown
    assert "split the skill pack" in markdown


def test_render_skill_review_block_handles_payload_dict_form():
    from ouroboros.skill_review import render_skill_review_block

    raw_text = "not json but still expensive reviewer output\n```text\nclose fence"
    payload = {
        "skill": "demo",
        "status": "warnings",
        "content_hash": "deadbeefcafe",
        "reviewer_models": ["openai/gpt-5.5"],
        "findings": [
            {
                "item": "error_handling",
                "verdict": "FAIL",
                "severity": "advisory",
                "reason": "best effort",
                "model": "openai/gpt-5.5",
            }
        ],
        "raw_actor_records": [{
            "model_id": "anthropic/claude-opus-4.6",
            "status": "parse_failure",
            "raw_text": raw_text,
        }],
    }
    markdown = render_skill_review_block(payload, attempt_idx=1)
    assert "`demo`" in markdown
    assert "[FAIL advisory] error_handling" in markdown
    assert raw_text in markdown
    assert "````text" in markdown


def test_review_skill_tool_result_has_no_raw_json_block(tmp_path, monkeypatch):
    # C4: the review_skill tool result is rendered-markdown only; the raw JSON
    # payload duplicate (findings + raw_actor_records + raw_result +
    # advisory_result) must not be re-appended into the agent's context.
    import ouroboros.tools.skill_exec as skill_exec_mod
    from ouroboros.skill_review import SkillReviewOutcome

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    skill_dir = skills_root / "alpha"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: alpha\ntype: instruction\nversion: 1.0.0\n---\nDoc.\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        skill_exec_mod,
        "_review_skill_impl",
        lambda _ctx, name, **_kwargs: SkillReviewOutcome(
            skill_name=name, status="clean",
            content_hash=compute_content_hash(skill_dir),
            reviewer_models=["fake/reviewer"], findings=[], error="",
        ),
    )
    out = skill_exec_mod._handle_review_skill(ctx, skill="alpha")
    assert "Raw review payload" not in out
    assert "<details>" not in out
