"""Advisory permission never rewrites failed review evidence as a PASS."""

from types import SimpleNamespace

import pytest

from ouroboros.review_state import load_state
from ouroboros.tools import claude_advisory_review as advisory
from ouroboros.tools import git
from ouroboros.tools.parallel_review import aggregate_review_verdict
from ouroboros.tools.review_helpers import build_scope_actor_record
from ouroboros.tools.scope_review import ScopeReviewResult
from tests.test_advisory_inline_freshness import candidate  # noqa: F401


@pytest.mark.parametrize("phase", ["context", "delivery", "format", "window_authority"])
@pytest.mark.parametrize("enforcement", ["blocking", "advisory"])
def test_technical_scope_failure_keeps_full_source_and_status(candidate, monkeypatch, phase, enforcement):  # noqa: F811
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    source = "review source\n" * 500
    result = ScopeReviewResult(
        blocked=True, status="error", failure_phase=phase, failure_code="test_failure",
        block_message="mandatory source unavailable", raw_text=source,
        advisory_findings=[{"item": "received_note", "reason": "keep this finding"}],
    )
    row = build_scope_actor_record(result, slot_id="scope-one")
    candidate._last_scope_raw_results = [row]
    blocked, _, _, _, findings = aggregate_review_verdict(
        None, result, "", [], candidate, "candidate", 0, candidate.repo_dir,
    )
    assert blocked == (enforcement == "blocking")
    assert result.blocked and result.status == "error"
    assert row["raw_text"] == source and row["failure_phase"] == phase
    assert findings[0]["reason"] == "keep this finding"
    if enforcement == "advisory":
        assert "not a PASS" in candidate._review_advisory[-1]


@pytest.mark.parametrize("phase,state,token", [
    ("authority", "settled", ""), ("admission", "not_dispatched", ""),
    ("deadline", "not_dispatched", ""), ("", "settled", ""),
    ("delivery", "in_flight", ""), ("delivery", "custody_lost", ""),
    ("delivery", "settled", "pending-invocation"),
])
def test_independent_failures_remain_blocking(candidate, monkeypatch, phase, state, token):  # noqa: F811
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    result = ScopeReviewResult(blocked=True, status="error", block_message="independent refusal", failure_phase=phase, operation_state=state, pending_invocation_id=token)
    assert aggregate_review_verdict(None, result, "", [], candidate, "candidate", 0, candidate.repo_dir)[0]


def test_mixed_scope_panel_requires_every_failed_origin_to_be_technical(candidate, monkeypatch):  # noqa: F811
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    result = ScopeReviewResult(blocked=True, status="blocked", block_message="mixed failure")
    candidate._last_scope_raw_results = [
        {"status": "error", "failure_phase": "context"},
        {"status": "error", "failure_phase": "authority"},
    ]
    assert aggregate_review_verdict(None, result, "", [], candidate, "candidate", 0, candidate.repo_dir)[0]


def test_scope_context_failure_is_distinct_from_invalid_subject(candidate, monkeypatch):  # noqa: F811
    from ouroboros.tools import scope_review
    from ouroboros.tools.review_admission import prepare_scope_review

    monkeypatch.setattr(scope_review, "_scope_review_skipped_in_low_context", lambda: False)
    monkeypatch.setattr(scope_review, "review_repo_dirs_for", lambda ctx: (ctx.repo_dir, ctx.repo_dir))
    monkeypatch.setattr("ouroboros.tools.review_subject.managed_review_subject", lambda *a: None)
    # Point the canonical checklist reader at the absent fixture source.
    from ouroboros.tools.review_helpers import load_checklist_section
    monkeypatch.setattr(scope_review, "load_checklist_section", lambda name: load_checklist_section(name, candidate.repo_dir / "docs" / "CHECKLISTS.md"))
    prepared, failure = prepare_scope_review(candidate, "candidate", scope_model="test/model")
    assert prepared is None and failure.failure_phase == "context"
    assert failure.status == "error"
    monkeypatch.setattr("ouroboros.tools.review_subject.managed_review_subject", lambda *a: (_ for _ in ()).throw(RuntimeError("unknown managed subject")))
    _, failure = prepare_scope_review(candidate, "candidate", scope_model="test/model")
    assert failure.failure_phase == "authority"


def test_actual_budget_exception_keeps_independent_origin():
    from ouroboros.review_custody import _ReviewAttemptHistory, _review_exception_projection
    from ouroboros.usage_accounting import BudgetExceeded

    usage, _, _, state, _ = _review_exception_projection(BudgetExceeded("no funds"), {}, _ReviewAttemptHistory(), {})
    assert usage["review_failure_phase"] == "admission"
    assert state == "not_dispatched"


def test_preflight_parse_failure_remains_failed_under_advisory(candidate, monkeypatch):  # noqa: F811
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    monkeypatch.setattr(advisory, "_run_advisory_tests", lambda ctx: None)
    monkeypatch.setattr(advisory, "_run_claude_advisory", lambda *a, **kw: ([], "unparsed complete review", "test/model", 20))
    result = git._advisory_and_tests_gate(candidate, "candidate", 0, classification_paths=["change.py"], advisory_paths=["change.py"], skip_advisory_pre_review=False, skip_tests=False)
    assert result is None
    row = load_state(candidate.drive_root).advisory_runs[-1]
    assert row.status == "parse_failure" and row.raw_result == "unparsed complete review"
    assert row.execution["failure_phase"] == "format"
    assert any("not a PASS" in str(item) for item in candidate._review_advisory)


def test_frozen_failed_actor_with_raw_output_stays_failed():
    from ouroboros.review_custody import _frozen_actor

    actor = _frozen_actor({"status": "error", "raw_text": '[{"verdict":"PASS"}]', "failure_phase": "delivery", "error": "result incomplete"}, SimpleNamespace(slot_id="s", model="m"))
    assert actor.status == "error"
    assert actor.raw_text == '[{"verdict":"PASS"}]'
    assert actor.usage["review_failure_phase"] == "delivery"
