"""Both parts of the existing review brief preserve supplied dispute facts."""
from copy import deepcopy
from pathlib import Path

import pytest

from ouroboros.tools.review_helpers import review_history_with_obligations
from tests.test_review_change_end_to_end import staged_body as _staged_body

staged_body = _staged_body


def _rounds():
    return [
        {"attempt": 1, "subject": {"diff_sha": "old-diff"}, "verdict": "FAIL",
         "critical": [{"item": "parser", "reason": "Original objection", "recommendation": "Use a database"}],
         "author_disposition": {"decision": "reject", "rationale": "The bounded file is the accepted scope."},
         "reviewer_outputs": [{"slot_id": "old-seat", "raw_text": "I disagree because atomic writes need a proof."}]},
        {"attempt": 2, "subject": {"diff_sha": "middle-diff"}, "verdict": "PASS",
         "advisory": [{"item": "naming", "reason": "Check the spelling"}]},
        {"attempt": 3, "subject": {"diff_sha": "new-diff"}, "verdict": "FAIL",
         "critical": [{"item": "parser", "reason": "Consider a database again"}],
         "review_rebuttal": "Now multiple writers are required; reconsider the earlier scope."},
    ]


@pytest.mark.parametrize("route,retrieves,structured", [
    pytest.param("api_chat", False, False, id="api_chat-False"),
    pytest.param("api_chat", True, False, id="api_chat-True"),
    pytest.param("agent_session", True, False, id="agent_session-True"),
    pytest.param("api_chat", False, True, id="api_chat-False-structured"),
    pytest.param("api_chat", True, True, id="api_chat-True-structured"),
    pytest.param("agent_session", True, True, id="agent_session-True-structured"),
])
def test_public_packet_and_two_part_brief_keep_all_supplied_arguments(staged_body, tmp_path, monkeypatch, route, retrieves, structured):
    from ouroboros import reviewer_window, capability_evidence
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.review_admission import build_two_part_brief
    from ouroboros.tools.review_subject import ReviewSubjectSpec, freeze_subject

    # Every capability answer is local; this test assembles inputs and dispatches nothing.
    monkeypatch.setattr(reviewer_window, "reviewer_context_window", lambda *a, **k: 1_000_000)
    monkeypatch.setattr(capability_evidence, "probe", lambda *a, **k: None)
    root = Path(staged_body["repo"])
    ctx = ToolContext(repo_dir=root, drive_root=tmp_path / "data")
    frozen = freeze_subject(ctx, ReviewSubjectSpec(root_kind="system_repo", root=str(root), kind="index",
                                                 governance_root=str(root), surface="change", layer="body"))
    history = _rounds()
    if structured:
        for wave in history:
            wave["verdict"] = {"aggregate": wave["verdict"], "advisory_findings": [
                {"item": "diagnostic", "reason": "Structured verdict evidence stays visible."}]}
    original_history = deepcopy(history)
    coupling = [{"verdict": "FAIL", "blocked": True, "summary": "The coupling rationale stays visible.",
                 "critical_findings": [{"item": "contract", "reason": "Reader mismatch"}],
                 "author_disposition": {"decision": "reject", "rationale": "The reader is migrated together."}}]
    brief = build_two_part_brief(
        frozen, {"slot_id": "fresh", "model": "fake/reviewer", "route": route, "retrieves": retrieves},
        drive_root=ctx.drive_root, task_id="test-task", review_history=history, coupling_history=coupling,
        owner_words="Owner chose file storage for this scope; that approval excludes a database service.")
    text = brief["system"]
    for value in ("The bounded file is the accepted scope.", "I disagree because atomic writes need a proof.",
                  "Use a database", "Check the spelling", "Now multiple writers are required", "old-diff", "new-diff",
                  "approval excludes a database service"):
        assert value in text
    assert history == original_history
    if structured:
        assert "Structured verdict evidence stays visible." in text
    assert brief["parts"] == (["change", "coupling"] if retrieves else ["change"])
    if retrieves:
        assert "The reader is migrated together." in text
        assert "The coupling rationale stays visible." in text
    assert "An author's rejection is not reviewer agreement" in text


def test_missing_durable_obligations_are_an_explicit_gap_with_available_history(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "advisory_review.json").write_text("{broken", encoding="utf-8")
    text = review_history_with_obligations(_rounds(), drive_root=tmp_path, repo_root=tmp_path / "repo")
    assert "REVIEW_HISTORY_SOURCE_UNAVAILABLE" in text
    assert "The bounded file is the accepted scope." in text
    assert "I disagree because atomic writes need a proof." in text


def test_obligation_reason_is_complete_in_shared_history(tmp_path):
    from ouroboros.review_state import AdvisoryReviewState, ObligationItem, make_repo_key, save_state

    reason = "Exact reason and rejected alternative. " * 150 + "DECISIVE_REASON_END"
    state = AdvisoryReviewState()
    state.open_obligations.append(ObligationItem(
        obligation_id="ob-1", item="contract", severity="critical", reason=reason,
        source_attempt_ts="2026-10-09T00:00:00Z", source_attempt_msg="repair", status="still_open",
        repo_key=make_repo_key(tmp_path)))
    save_state(tmp_path, state)
    text = review_history_with_obligations([], drive_root=tmp_path, repo_root=tmp_path)
    assert reason in text and "OMISSION NOTE" not in text


@pytest.mark.parametrize("route,retrieves", [("api_chat", False), ("agent_session", True)])
@pytest.mark.parametrize("structured", [False, True], ids=["scalar", "structured"])
def test_supplied_source_bound_answers_keep_scalar_or_structured_verdict(staged_body, tmp_path, monkeypatch, route, retrieves, structured):
    import json
    from ouroboros import capability_evidence, reviewer_window, review_ledger, review_history_view
    from ouroboros.review_history import review_dispute_history
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.review_admission import build_two_part_brief
    from ouroboros.tools.review_subject import ReviewSubjectSpec, freeze_subject

    monkeypatch.setattr(reviewer_window, "reviewer_context_window", lambda *a, **k: 1_000_000)
    monkeypatch.setattr(capability_evidence, "probe", lambda *a, **k: None)
    root = Path(staged_body["repo"])
    ctx = ToolContext(repo_dir=root, drive_root=tmp_path / "data", task_id="test-task")
    frozen = freeze_subject(ctx, ReviewSubjectSpec(root_kind="system_repo", root=str(root), kind="index",
                                                 governance_root=str(root), surface="change", layer="body"))
    finding = {"item": "parser", "reason": "Exact retained critic argument.", "severity": "critical", "verdict": "FAIL"}
    answer = {"status": "responded", "verdict": "FAIL", "findings": [finding], "critical": 1, "coverage": "n/a"}
    raw = json.dumps(answer)
    source = review_ledger.retain_text_source(ctx.drive_root, ctx.task_id, record_id="prior-record",
        seat_id="old-seat", role="response", part="change", text=raw)
    assert source["status"] == "retained"
    history = [{"review_record_id": "prior-record", "verdict":
        {"aggregate": "FAIL", "critical_findings": [finding]} if structured else "FAIL",
        "reviewers": [{"seat_id": "old-seat", "answers": {"change": answer},
                       "response": {"text": raw, "source": source}}]}]
    original = deepcopy(history)
    brief = build_two_part_brief(frozen,
        {"slot_id": "fresh", "model": "fake/reviewer", "route": route, "retrieves": retrieves},
        drive_root=ctx.drive_root, task_id=ctx.task_id, review_history=history)
    assert "Exact retained critic argument." in brief["system"] and "old-seat" in brief["system"]
    assert brief["parts"] == (["change", "coupling"] if retrieves else ["change"])
    assert history == original and review_ledger.read_source(ctx.drive_root, ctx.task_id, source).decode("utf-8") == raw
    # The reader itself makes this supplied answer a source-bound decision.
    dispute = review_dispute_history(history, drive_root=ctx.drive_root, repo_root=root, task_id=ctx.task_id)
    binding = review_history_view.decision_entries(dispute)[0]["bound_decision"]
    assert binding is not None
    projected, notes, gaps = review_history_view.project_decision_notes(dispute, [
        {"bound_decision": binding, "remark": "Parser concern.", "reason": "Keep its exact retained argument."}])
    assert notes and not gaps
    if structured:
        assert projected["rounds"][0]["verdict"]["aggregate"] == "FAIL"
        assert "authored_view" in projected["rounds"][0]["verdict"]["critical_findings"][0]
    else:
        assert projected["rounds"][0]["verdict"] == "FAIL"
    assert projected["rounds"][0]["reviewers"][0]["answers"]["change"]["critical"] == 1
    assert "authored_view" in projected["rounds"][0]["reviewers"][0]["answers"]["change"]["findings"][0]
