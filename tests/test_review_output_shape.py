"""The review output SHAPE (array | report | two_part | plan_reply) is one form fact per surface.

Retrieving deliveries (delegated session, native tool-round episode) feed their
answer through ``canonicalize_session_verdict``; the shape decides which form the
canonicalizer keeps whole, which contract a retrieving row is asked for and which
session schema the engine enforces, so the ask and the parse can never disagree.
"""

import json

import pytest

from ouroboros.review_execution import (
    REVIEW_SESSION_OUTPUT_SCHEMA,
    review_session_output_schema,
)
from ouroboros.review_verdict_extraction import (
    _strictly_parseable,
    canonicalize_session_verdict,
)
from ouroboros.triad_review import (
    REVIEW_JSON_ARRAY_CONTRACT,
    REVIEW_OUTPUT_SHAPES,
    REVIEW_REPORT_CONTRACT,
    REVIEW_TWO_PART_OBJECT_CONTRACT,
    TWO_PART_SESSION_OUTPUT_SCHEMA,
    default_output_contract,
    review_output_shape,
)

_OBJECT_VERDICT = {
    "verdict": "PASS",
    "outcome_tier": "solved",
    "completion_coach": "",
    "criteria_used": [
        {"criterion": "deliverable exists", "status": "supported",
         "evidence_refs": ["verification_receipts[0]"]},
    ],
    "dialogue_status": "unreachable_here",
    "findings": [],
    "summary": "clean",
}


def test_shape_table_is_form_only_and_defaults_to_array():
    assert REVIEW_OUTPUT_SHAPES == {
        "deep_self_review": "report", "multi_model_review": "two_part", "plan_review": "plan_reply"}
    assert review_output_shape("deep_self_review") == "report"
    # the commit gate's one brief, two parts: a retrieving seat answers contract B
    assert review_output_shape("multi_model_review") == "two_part"
    # the planning organ's reply object is no verdict: passed through whole, never extracted
    assert review_output_shape("plan_review") == "plan_reply"
    for surface in ("skill_review", "advisory_review", ""):
        assert review_output_shape(surface) == "array"


def test_default_output_contract_follows_the_shape():
    """A retrieving row whose surface hands over no `policy["output_contract"]`
    is asked for the SAME form its host parses."""
    assert default_output_contract("array") is REVIEW_JSON_ARRAY_CONTRACT
    assert default_output_contract("report") is REVIEW_REPORT_CONTRACT
    assert default_output_contract("two_part") is REVIEW_TWO_PART_OBJECT_CONTRACT
    from ouroboros.tools.plan_packet import PLAN_REPLY_CONTRACT

    assert default_output_contract("plan_reply") is PLAN_REPLY_CONTRACT
    assert default_output_contract("") is REVIEW_JSON_ARRAY_CONTRACT
    assert "no JSON wrapper" in REVIEW_REPORT_CONTRACT

    from ouroboros.review_execution import ReviewAssignment, ReviewRouteKind, _review_route_executor
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot

    for surface, expected in (("deep_self_review", REVIEW_REPORT_CONTRACT),
                              ("multi_model_review", REVIEW_TWO_PART_OBJECT_CONTRACT), ("skill_review", REVIEW_JSON_ARRAY_CONTRACT),
                              ("plan_review", PLAN_REPLY_CONTRACT)):
        request = ReviewRequest(surface=surface, goal="g", task_id="t", session_root="/tmp", session_task="task", policy={})
        native = _review_route_executor(ReviewAssignment(
            request=request, slot=ReviewSlot(slot_id="n", model="m", effort="low", subagent_id="api-critic"), call_id="c"))
        session = _review_route_executor(ReviewAssignment(
            request=request, slot=ReviewSlot(slot_id="s", model="m", effort="low", route=ReviewRouteKind.AGENT_SESSION), call_id="c"))
        assert native._output_contract() is expected and session._output_contract() is expected


def test_schema_branch_reduces_a_structured_payload_to_its_findings_for_the_array_contract():
    canonical_array, method_array, _ = canonicalize_session_verdict(
        json.dumps({"findings": [{"item": "x", "verdict": "PASS"}]}),
        conformance_passed=True, shape="array")
    assert method_array == "schema"
    assert json.loads(canonical_array) == [{"item": "x", "verdict": "PASS"}]


def test_empty_array_is_the_strict_array_verdict_and_a_whole_object_is_not():
    """``[]`` is the constitutional clean ARRAY verdict; an object is not an array answer."""
    assert _strictly_parseable("[]", "array") is True
    assert _strictly_parseable("[]\nNO_FINDINGS", "array") is True
    assert _strictly_parseable(json.dumps(_OBJECT_VERDICT), "array") is False


def test_report_shape_passes_the_product_through_verbatim():
    report = "# Deep self-review\n\nCRITICAL: loop.py has a race.\n\n- item\n"
    canonical, method, usage = canonicalize_session_verdict(
        report, conformance_passed=True, shape="report", llm=object())
    assert (canonical, method, usage) == (report, "report", {})


def test_session_schema_follows_the_shape():
    assert review_session_output_schema("multi_model_review") is TWO_PART_SESSION_OUTPUT_SCHEMA
    assert review_session_output_schema("advisory_review") is REVIEW_SESSION_OUTPUT_SCHEMA
    # A report is prose: no schema is asked, so the engine is never forced into
    # a findings shape the canonicalizer would then pass through verbatim.
    assert review_session_output_schema("deep_self_review") is None
    assert review_session_output_schema("skill_review")["properties"]["findings"]["minItems"] == 1


@pytest.mark.parametrize("shape", ["array", "two_part"])
def test_extraction_bound_is_shape_independent(shape):
    canonical, method, _ = canonicalize_session_verdict(
        "x" * 400_001, conformance_passed=False, shape=shape, llm=object())
    assert method == "extraction_incomplete"
    assert len(canonical) == 400_001
