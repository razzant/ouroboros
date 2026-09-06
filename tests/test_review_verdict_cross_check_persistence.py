"""The original disputed finding and transcript survive session projection."""

import json

import pytest


@pytest.mark.parametrize("conformed", [True, False])
def test_verdict_result_keeps_disputed_finding_and_full_raw_source(tmp_path, conformed):
    from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAssignment, ReviewRouteKind
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot

    findings = [{"item": "fabricated_import_x1", "verdict": "FAIL",
                 "severity": "critical", "reason": "imports `ouroborosproject_x1`"}]
    raw = json.dumps({"findings": findings} if conformed else findings)
    request = ReviewRequest(surface="scope_review", goal="Review the staged change.",
                            task_id="disputed-finding", session_root=str(tmp_path))
    slot = ReviewSlot(slot_id="slot_x", model="api/model-a", timeout_sec=30,
                      route=ReviewRouteKind.AGENT_SESSION)
    executor = AgentSessionReviewExecutor(ReviewAssignment(request, slot, "call-x"), llm=None)
    executor._session_usage = {}
    executor._deltas = []
    executor._raw_transcript = raw
    executor._conformance_passed = conformed

    result = executor._verdict_result()

    assert json.loads(result.raw_text) == findings
    assert result.usage["verdict_method"] == ("schema" if conformed else "strict")
    assert result.message["session_transcript"] == raw
    assert "cross_check" not in result.usage
