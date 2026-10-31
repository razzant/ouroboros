"""Advisory permission never rewrites failed review evidence as a PASS."""

import json
from types import SimpleNamespace

import pytest

from ouroboros.review_execution import ReviewRouteKind
from ouroboros.tools import review as review_mod
from tests.test_git_review_preflight_gate import candidate  # noqa: F401

SOURCE = "review source\n" * 500
PASS_ITEM = json.dumps([{"item": "bible_compliance", "verdict": "PASS", "severity": "critical", "reason": "ok"}])


def _prepared(candidate, parts):  # noqa: F811
    """One wave of ``len(parts)`` api seats; a seat asked ``coupling`` retrieves."""
    n = len(parts)
    retrieves = ["coupling" in p for p in parts]
    models = [f"m/{i + 1}" for i in range(n)]
    from ouroboros.config import get_review_enforcement
    from ouroboros.tools.review_helpers import review_enforcement_blocks

    return {
        "prompt": "PACKET", "stable_prefix_len": 0, "models": models, "routes": [ReviewRouteKind.API_CHAT] * n,
        "target_repo": candidate.repo_dir, "blocking_review": review_enforcement_blocks(get_review_enforcement()),
        "layer": "body", "task_evidence": None,
        "row_plan": {"models": models, "routes": [ReviewRouteKind.API_CHAT] * n,
                     "slot_ids": [f"slot_{i + 1}" for i in range(n)], "parts": [tuple(p) for p in parts],
                     "retrieves": retrieves, "session_tasks": ["BRIEF" if r else "" for r in retrieves],
                     "brief_shas": ["sha" if r else "" for r in retrieves]},
        "governance_manifest": [], "governance_packet_slots": [], "retrieving_manifests": [], "brief_texts": {},
    }


def _dispatch(candidate, monkeypatch, results, parts):  # noqa: F811
    monkeypatch.setattr(review_mod, "_handle_multi_model_review",
                        lambda *a, **kw: json.dumps({"results": results}))
    return review_mod._dispatch_unified_review(candidate, "candidate", _prepared(candidate, parts))


@pytest.mark.parametrize("enforcement", ["blocking", "advisory"])
def test_failed_coupling_seat_keeps_full_source_and_is_never_a_pass(candidate, monkeypatch, enforcement):  # noqa: F811
    """The one seat asked the coupling question fails technically: its record
    keeps the full raw output and the typed failure facts, Part 2 is never
    read as answered, and advisory enforcement only waves the BLOCK through —
    loudly — without rewriting the evidence."""
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    results = [
        {"model": "m/1", "slot_id": "slot_1", "text": PASS_ITEM, "verdict": "UNKNOWN"},
        {"model": "m/2", "slot_id": "slot_2", "verdict": "ERROR", "text": SOURCE,
         "failure_code": "test_failure", "transport_status": "delivery"},
    ]
    review_err = _dispatch(candidate, monkeypatch, results, [("change",), ("change", "coupling")])
    assert (review_err is not None) == (enforcement == "blocking")
    row = candidate._last_triad_raw_results[1]
    assert row["status"] == "error" and row["raw_text"] == SOURCE
    assert row["failure_code"] == "test_failure" and row["transport_status"] == "delivery"
    verdict = candidate._last_review_verdict
    assert verdict["aggregate"] != "PASS" and verdict["per_question"]["coupling"] != "PASS"
    assert candidate._last_coupling_result.status != "responded"
    if enforcement == "advisory":
        assert "explicit author decision is required" in candidate._review_advisory[-1]


@pytest.mark.parametrize("state,token", [("in_flight", ""), ("custody_lost", ""), ("settled", "pending-invocation")])
@pytest.mark.parametrize("enforcement", ["blocking", "advisory"])
def test_pending_seat_is_pending_not_pass_under_either_enforcement(candidate, monkeypatch, state, token, enforcement):  # noqa: F811
    """A seat whose physical operation is unresolved keeps the wave PENDING:
    blocking enforcement blocks on it, advisory records the pending signal and
    the verdict is NOT_PERFORMED (``review_late_result_pending``) — never a PASS
    the gate could settle or reuse."""
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    results = [
        {"model": "m/1", "slot_id": "slot_1", "text": PASS_ITEM, "verdict": "UNKNOWN"},
        {"model": "m/2", "slot_id": "slot_2", "verdict": "ERROR", "text": "", "operation_state": state,
         "pending_invocation_id": token, "late_result_pending": bool(token)},
    ]
    review_err = _dispatch(candidate, monkeypatch, results, [("change",), ("change", "coupling")])
    assert (review_err is not None) == (enforcement == "blocking")
    if review_err:
        assert "REVIEW_PENDING" in review_err
    verdict = candidate._last_review_verdict
    assert verdict["aggregate"] == "NOT_PERFORMED" and verdict["reason"] == "review_late_result_pending"
    assert candidate._last_review_block_reason == "review_late_result_pending"
    if enforcement == "advisory":
        assert any("review is pending" in note for note in candidate._review_advisory)


def test_actual_budget_exception_keeps_independent_origin():
    from ouroboros.review_custody import _ReviewAttemptHistory, _review_exception_projection
    from ouroboros.usage_accounting import BudgetExceeded

    usage, _, _, state, _ = _review_exception_projection(BudgetExceeded("no funds"), {}, _ReviewAttemptHistory(), {})
    assert usage["review_failure_phase"] == "admission"
    assert state == "not_dispatched"


@pytest.mark.parametrize("enforcement", ["blocking", "advisory"])
def test_gate_and_status_keep_the_full_cause_without_changing_enforcement(candidate, monkeypatch, enforcement):  # noqa: F811
    from ouroboros.review_state import CommitAttemptRecord, load_state, make_repo_key, save_state
    from ouroboros.tools.preflight_review import _handle_review_status

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    cause = "Engine refused this saved request. " * 220 + "END: measured 940000 characters; limit 900000 characters"
    results = [{"model": f"m/{i}", "slot_id": f"slot_{i}", "verdict": "ERROR", "text": cause,
                "reported_cause": cause, "failure_code": "native_transcript_cap_exceeded",
                "failure_details": {"measured": 940000, "limit": 900000, "unit": "characters"}}
               for i in (1, 2)]
    error = _dispatch(candidate, monkeypatch, results, [("change",), ("change",)])
    assert (error is not None) == (enforcement == "blocking")
    message = error or "\n".join(candidate._review_advisory)
    assert cause in message and "transient" not in message.lower()
    assert candidate._last_review_verdict["aggregate"] == "QUORUM_FAILED"
    state = load_state(candidate.drive_root)
    state.record_attempt(CommitAttemptRecord(ts="2026-10-25T00:00:00Z", commit_message="candidate", status="blocked",
        task_id=candidate.task_id, repo_key=make_repo_key(candidate.repo_dir), block_reason="review_quorum",
        triad_raw_results=candidate._last_triad_raw_results))
    save_state(candidate.drive_root, state)
    status = json.loads(_handle_review_status(candidate, task_id=candidate.task_id))
    assert cause in status["message"] and "retry the commit" not in status["message"].lower()
    from ouroboros.tools.scope_review_contract import SCOPE_REQUIRED_ITEMS
    coupling = [{"item": item, "verdict": "PASS", "severity": "advisory", "reason": "checked"}
                for item in sorted(SCOPE_REQUIRED_ITEMS)]
    answer = json.dumps({"change": json.loads(PASS_ITEM), "change_clean": True, "coupling": coupling})
    passed = [{"model": f"m/{i}", "slot_id": f"slot_{i}", "text": answer, "verdict": "UNKNOWN"} for i in (1, 2)]
    assert _dispatch(candidate, monkeypatch, passed, [("change", "coupling"), ("change", "coupling")]) is None
    assert candidate._last_review_verdict["aggregate"] == "PASS"


def test_frozen_failed_actor_with_raw_output_stays_failed():
    from ouroboros.review_custody import _frozen_actor

    actor = _frozen_actor({"status": "error", "raw_text": '[{"verdict":"PASS"}]', "failure_phase": "delivery", "error": "result incomplete"}, SimpleNamespace(slot_id="s", model="m"))
    assert actor.status == "error"
    assert actor.raw_text == '[{"verdict":"PASS"}]'
    assert actor.usage["review_failure_phase"] == "delivery"
