"""The rendered review block: what the agent is shown, and what it is never shown.

The findings grouped by reviewer verbatim, the retry note from round two (open findings and outcome
duties, no fixed procedure, no invented verdict), the payload dict form the chat detail renders, and
the raw JSON block the tool result never contains.
"""

from __future__ import annotations

from ouroboros.skill_loader import compute_content_hash
from ouroboros.tools.registry import ToolContext


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


def test_render_skill_review_block_emits_retry_note_at_attempt_two():
    from ouroboros.skill_review import SkillReviewOutcome, render_skill_review_block
    from ouroboros.tools.review_prompt_text import REVIEW_REPAIR_JUDGMENT

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
    assert "Before the next skill_review" not in markdown_first

    markdown_second = render_skill_review_block(outcome, attempt_idx=2)
    note = markdown_second.split("Before the next skill_review:", 1)[1]
    assert "Finding: bug_hunting — model=openai/gpt-5.5; missing error handling" in note
    assert REVIEW_REPAIR_JUDGMENT in note
    assert ("An eligible recorded verdict on an unchanged skill pack under the same review "
            "contract is not re-reviewed without a genuinely new review_rebuttal") in " ".join(note.split())
    assert "Status: addressed / rebutted / pending" not in markdown_second
    assert "Do NOT call skill_review" not in markdown_second


def test_render_skill_review_block_later_rounds_keep_one_note_without_circuit_breaker():
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
    assert "Before the next skill_review" in markdown
    assert "split the skill pack" in markdown
    for retired in ("Circuit-breaker", "two concrete fixes", "STOP retrying", "ONE subject line"):
        assert retired not in markdown, retired
    assert markdown.split("Before the next skill_review", 1)[1] == (
        render_skill_review_block(outcome, attempt_idx=2).split("Before the next skill_review", 1)[1])


def test_retry_note_rides_the_series_round_not_the_snapshot():
    from ouroboros.skill_review import render_skill_review_block

    payload = {
        "skill": "demo", "status": "blockers", "review_round": 3,
        "snapshot_attempt": 1, "content_hash": "beefcafe1234",
        "findings": [{"item": "bug_hunting", "verdict": "FAIL", "severity": "critical",
                      "reason": "missing error handling", "model": "reviewer"}],
    }
    markdown = render_skill_review_block(payload, attempt_idx=1)
    assert "Before the next skill_review" in markdown
    assert "Skill review round 3 — snapshot beefcafe1234 (attempt 1)" in markdown
    assert "missing error handling" in markdown
    # The record's ordinal wins even when the caller's legacy fallback is larger.
    payload["review_round"] = 1
    assert "Before the next skill_review" not in render_skill_review_block(payload, attempt_idx=3)


def test_history_detail_shape_shows_the_retry_note_of_its_round():
    from ouroboros.skill_review import render_skill_review_block

    payload = {
        "skill": "demo", "status": "blockers", "review_round": 2,
        "snapshot_attempt": 1, "content_hash": "abc123def456",
        "findings": [{"item": "bug_hunting", "verdict": "FAIL", "severity": "critical",
                      "reason": "missing error handling", "model": "reviewer"}],
    }
    markdown = render_skill_review_block(payload, attempt_idx=1)
    assert "Skill review round 2 — snapshot abc123def456 (attempt 1)" in markdown
    assert "Before the next skill_review" in markdown


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


def test_render_skill_review_block_names_the_record_and_marks_advisers():
    from ouroboros.skill_review import SkillReviewOutcome, render_skill_review_block

    outcome = SkillReviewOutcome(
        skill_name="demo", status="warnings", content_hash="abc12345", record_id="rl-20261021-abc", reused=True,
        reviewer_models=["openai/gpt-5.5 [t1]"],
        findings=[{"item": "error_handling", "verdict": "FAIL", "severity": "advisory", "reason": "swallows",
                   "model": "openai/gpt-5.5", "slot_id": "t1"},
                  {"item": "bug_hunting", "verdict": "FAIL", "severity": "critical", "reason": "typo", "model": "scout",
                   "slot_id": "scout", "additional": True}])
    markdown = render_skill_review_block(outcome)
    assert "Review record: rl-20261021-abc (reused, $0)" in markdown
    assert "### Reviewer: scout (adviser, not counted)" in markdown
    assert "[FAIL advisory] error_handling: swallows" in markdown
