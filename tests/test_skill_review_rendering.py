"""The rendered review block: what the agent is shown, and what it is never shown.

Split out of ``tests/test_skill_review.py`` by theme: the concrete fail reasons the history
section renders and its legacy-signature fallback, the findings grouped by reviewer
verbatim, the retry note from round two (open findings and outcome duties, no fixed
procedure, fix-count STOP or escalating prescription, no invented verdict beside a
quorum miss's partial findings), the payload dict form, and the raw JSON block the tool
result never contains.
"""

from __future__ import annotations

from ouroboros.skill_loader import compute_content_hash
from ouroboros.tools.registry import ToolContext
from tests.review_pool_rosters import set_review_pool


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
    # Convergence rule fires from the 3rd review round of the group onward.
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


def test_history_preserves_real_round_and_snapshot_numbers():
    from ouroboros.skill_review import _build_skill_review_history_section
    history = [{"review_round": n, "snapshot_attempt": n - 5, "snapshot_revised": n == 9,
                "status": "warnings", "content_hash": "abc", "failure_signature": [f"reason-{n}"]}
               for n in range(6, 10)]
    text = _build_skill_review_history_section(history)
    assert "Review round 7, snapshot attempt 2" in text
    assert "Review round 9, snapshot attempt 4" in text
    assert "snapshot_revised=True" in text and "reason-9" in text
    assert "Review round 6" not in text and "Attempt 1:" not in text
    legacy = _build_skill_review_history_section([{"status": "warnings"}])
    assert "Review round unknown, snapshot attempt unknown" in legacy


def test_pending_quorum_note_keeps_partial_fails_and_leaves_retry_to_the_gate(
    tmp_path, monkeypatch,
):
    """Real producer→consumer path: a review_skill round that misses the reviewer
    quorum stays PENDING yet keeps its one responder's FAIL. At round two the
    shared note lists that finding as recorded review state without inventing an
    aggregate verdict, and does not claim the unchanged pack needs a rebuttal:
    an infra fact neither replays nor lapses a verdict, so the identical
    snapshot legitimately reaches the panel again."""
    import json

    import ouroboros.tools.skill_exec as skill_exec_mod
    from ouroboros.skill_review import _load_skill_review_history
    from ouroboros.skill_review_cycles import find_free_replay_row
    from tests._skill_review_shared import (
        _build_skill,
        _make_actor,
        _make_ctx,
        _pass_array_for_script_skill,
        _patch_review,
    )

    skills_root = _build_skill(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    set_review_pool(monkeypatch, ["openai/gpt-5.5", "google/gemini-3.5-flash", "anthropic/claude-opus-4.6"])
    ctx = _make_ctx(tmp_path)
    partial = [
        {**row, "verdict": "FAIL", "reason": "fetch.py writes outside the skill directory"}
        if row["item"] == "path_confinement" else row
        for row in json.loads(_pass_array_for_script_skill())
    ]
    canned = json.dumps({"results": [
        _make_actor("openai/gpt-5.5", json.dumps(partial)),
        *({"model": model, "request_model": model, "verdict": "ERROR",
           "text": "OpenRouter 429", "tokens_in": 0, "tokens_out": 0}
          for model in ("google/gemini-3.5-flash", "anthropic/claude-opus-4.6")),
    ]})
    content_hash = compute_content_hash(skills_root / "weather")

    with _patch_review(canned) as panel:
        first = skill_exec_mod._handle_review_skill(ctx, skill="weather")
        second = skill_exec_mod._handle_review_skill(ctx, skill="weather")

    # The unchanged snapshot had no eligible verdict to replay or refuse on.
    assert panel.call_count == 2
    rows = _load_skill_review_history(ctx.drive_root, "weather", limit=0)
    assert [row["status"] for row in rows] == ["pending", "pending"]
    assert all(row["content_hash"] == content_hash for row in rows)
    assert find_free_replay_row(
        ctx.drive_root, "weather", group_id=str(rows[-1].get("group_id") or ""),
        content_hash=content_hash,
        contract_fingerprint=str(rows[-1].get("review_contract_fingerprint") or ""),
    ) is None

    assert "status=pending" in first and "Before the next skill_review" not in first
    assert "status=pending" in second
    note = second.split("Before the next skill_review:", 1)[1]
    assert "Finding: path_confinement" in note
    assert "fetch.py writes outside the skill directory" in note
    lowered = " ".join(note.lower().split())
    assert "the recorded review state and the individual findings below stand" in lowered
    assert "a missed reviewer quorum neither replays nor lapses a verdict" in lowered
    assert "an unchanged skill pack without an eligible verdict may be reviewed again" in lowered
    for false_claim in ("the recorded verdict and", "another paid review needs"):
        assert false_claim not in lowered, false_claim
