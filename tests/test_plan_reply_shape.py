"""The planning organ's reply through the REAL review executors (owner-visible defect found by
W2 on the trial candidate acf980b05): a retrieving participant — a native tool-round seat or an
agent-session seat — answers ONE reply object ``{prose, objections, withdrawn}``. The executors used
to canonicalize it under the findings-array shape (``REVIEW_OUTPUT_SHAPES`` had no ``plan_review``
entry): one light-model extraction per reply, and an obedient ``[]`` replaced the participant's
words, which the organ then recorded as unparseable. The surface now has its own pass-through
shape ``plan_reply``: nothing extracts it, the whole text reaches ``plan_record.parse_reply``.
Every pin here asserts both directions: the plan reply is read whole with no light call, and the
other surfaces keep their shapes (the array shape still extracts the very same text)."""
from __future__ import annotations

import json

from ouroboros.review_execution import (
    REVIEW_SESSION_OUTPUT_SCHEMA, TWO_PART_SESSION_OUTPUT_SCHEMA,
    AgentSessionReviewExecutor, ReviewAssignment, ReviewRouteKind, review_session_output_schema,
)
from ouroboros.review_native_episode import NativeToolRoundReviewExecutor
from ouroboros.review_substrate import ReviewRequest, ReviewSlot
from ouroboros.review_verdict_extraction import canonicalize_session_verdict
from ouroboros.tools import plan_ask
from ouroboros.tools.plan_packet import PLAN_REPLY_CONTRACT, PLAN_REPLY_SESSION_OUTPUT_SCHEMA
from ouroboros.triad_review import default_output_contract, review_output_shape
from tests.test_native_tool_round_executor import _ScriptedLLM, _tool_call

REPLY = {"prose": "Looks sound to me, with one reservation.",
         "objections": [{"id": "o1", "text": "the budget line is missing from done_when"}], "withdrawn": []}
REPLY_TEXT = json.dumps(REPLY)


class _NoLightModel:
    """A transport that must never be reached: an extraction over a plan reply is a wasted paid call."""

    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        raise AssertionError("a light-model extraction was attempted over a plan reply")


class _ObedientLightModel:
    """What the bug did: an obedient light model answering the findings-array contract."""

    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"content": "[]"}, {"prompt_tokens": 5, "completion_tokens": 2, "cost": 0.0001}


def _plan_request(root) -> ReviewRequest:
    return ReviewRequest(surface="plan_review", goal="Weigh the plan before the work starts.", task_id="t-plan",
                         session_root=str(root), session_task="## THE PLAN\n\ngoal: ship the deck\n\nRead notes.md, then answer.",
                         policy={"output_contract": PLAN_REPLY_CONTRACT}, no_proxy=True)


def _organ_reads(result, slot) -> dict:
    return plan_ask.reply_facts_from_row(plan_ask._actor_row(
        {"raw_text": result.raw_text, "status": "ok", "usage": result.usage}, slot))


def test_a_native_retrieving_seat_reply_is_read_whole_with_no_light_call(tmp_path):
    (tmp_path / "notes.md").write_text("budget: not stated\n", encoding="utf-8")
    llm = _ScriptedLLM([{"tool_calls": [_tool_call("read_file", {"path": "notes.md"})]}, {"content": REPLY_TEXT}])
    slot = ReviewSlot(slot_id="s1", model="openai/fake-reviewer", effort="low", route=ReviewRouteKind.API_CHAT,
                      subagent_id="native-reader", native_retrieval_override=True)
    result = NativeToolRoundReviewExecutor(
        ReviewAssignment(request=_plan_request(tmp_path), slot=slot, call_id="op-plan-native"), llm=llm).execute()
    assert len(llm.calls) == 2, "the read round and the answer — never a third (light) send"
    assert result.raw_text == REPLY_TEXT and result.usage["verdict_method"] == "plan_reply"
    assert "extraction" not in result.usage and "capability_delta" not in result.usage
    facts = _organ_reads(result, slot)
    assert facts["status"] == "answered" and facts["reply"]["prose"] == REPLY["prose"]
    assert facts["reply"]["objections"] == REPLY["objections"]


def test_an_agent_session_seat_reply_is_read_whole_even_after_the_schema_conformed(tmp_path):
    llm = _NoLightModel()
    slot = ReviewSlot(slot_id="s2", model="api/model-a", effort="low", timeout_sec=30, route=ReviewRouteKind.AGENT_SESSION)
    executor = AgentSessionReviewExecutor(
        ReviewAssignment(request=_plan_request(tmp_path), slot=slot, call_id="op-plan-session", custody_root=tmp_path), llm=llm)
    # The session route requests PLAN_REPLY_SESSION_OUTPUT_SCHEMA and the engine conformed to it.
    assert review_session_output_schema("plan_review") is PLAN_REPLY_SESSION_OUTPUT_SCHEMA
    executor._raw_transcript, executor._conformance_passed = REPLY_TEXT, True
    result = executor._verdict_result()
    assert result.raw_text == REPLY_TEXT and result.usage["verdict_method"] == "plan_reply" and llm.calls == []
    # The permitted local format resend (plan 5.5) leaves the reply whole as well.
    again = executor.execute()
    assert again.raw_text == REPLY_TEXT and again.usage["verdict_method"] == "plan_reply" and llm.calls == []
    facts = _organ_reads(again, slot)
    assert facts["status"] == "answered" and facts["reply"]["objections"] == REPLY["objections"]
    # A session that did NOT conform (prose around the object) is still handed over whole.
    executor._raw_transcript, executor._conformance_passed = "Here is my answer:\n" + REPLY_TEXT, False
    loose = executor._verdict_result()
    assert loose.raw_text.endswith(REPLY_TEXT) and loose.usage["verdict_method"] == "plan_reply" and llm.calls == []
    assert _organ_reads(loose, slot)["reply"]["prose"] == REPLY["prose"]


def test_the_plan_surface_asks_for_the_reply_contract_on_every_retrieving_route(tmp_path):
    from ouroboros.review_session_preparation import render_review_session_prompt

    assert review_output_shape("plan_review") == "plan_reply"
    assert default_output_contract("plan_reply") is PLAN_REPLY_CONTRACT
    bare = ReviewRequest(surface="plan_review", goal="g", task_id="t", session_root=str(tmp_path), session_task="task", policy={})
    slot = ReviewSlot(slot_id="s", model="m", effort="low", route=ReviewRouteKind.AGENT_SESSION)
    prompt = render_review_session_prompt(bare, slot, "task")
    assert PLAN_REPLY_CONTRACT in prompt and "JSON array" not in prompt
    assert AgentSessionReviewExecutor(ReviewAssignment(request=bare, slot=slot, call_id="c"))._output_contract() is PLAN_REPLY_CONTRACT
    native = ReviewSlot(slot_id="n", model="m", effort="low", subagent_id="api-critic", native_retrieval_override=True)
    assert NativeToolRoundReviewExecutor(ReviewAssignment(request=bare, slot=native, call_id="c"))._output_contract() is PLAN_REPLY_CONTRACT


def test_the_other_surfaces_keep_their_shapes_and_the_array_shape_still_extracts():
    assert review_output_shape("multi_model_review") == "two_part"
    assert review_output_shape("deep_self_review") == "report" and review_output_shape("skill_review") == "array"
    assert review_output_shape("advisory_review") == "array"
    assert review_session_output_schema("multi_model_review") is TWO_PART_SESSION_OUTPUT_SCHEMA
    assert review_session_output_schema("deep_self_review") is None
    assert review_session_output_schema("advisory_review") is REVIEW_SESSION_OUTPUT_SCHEMA
    # The very same text under the findings-array shape is what the bug did: one light call, "[]" replaces the words.
    light = _ObedientLightModel()
    canonical, method, _usage = canonicalize_session_verdict(REPLY_TEXT, conformance_passed=False, shape="array", llm=light)
    assert (canonical, method, len(light.calls)) == ("[]", "light_model_extraction", 1)
    # The plan shape passes the reply through on every branch, the collection-only path included.
    for conformed in (True, False):
        assert canonicalize_session_verdict(REPLY_TEXT, conformance_passed=conformed, shape="plan_reply",
                                            llm=_NoLightModel()) == (REPLY_TEXT, "plan_reply", {})
    assert canonicalize_session_verdict(REPLY_TEXT, conformance_passed=False, shape="plan_reply",
                                        allow_extraction=False) == (REPLY_TEXT, "plan_reply", {})


def test_a_result_wave_on_the_commit_surface_is_asked_and_read_as_the_findings_array(tmp_path):
    """A result wave rides the commit surface (two-part) but names its own shape in its policy: the session
    schema, the prompt contract and the canonicalizer of every retrieving route read the findings array.
    Without it a session seat was handed the change/coupling schema and its result answer never counted."""
    from ouroboros.triad_review import REVIEW_JSON_ARRAY_CONTRACT, request_output_shape

    def request(policy):
        return ReviewRequest(surface="multi_model_review", goal="g", task_id="t", session_root=str(tmp_path),
                             session_task="task", policy=policy)

    result = request({"output_contract": REVIEW_JSON_ARRAY_CONTRACT, "output_shape": "array"})
    assert [request_output_shape(r) for r in (result, request({}), request({"output_shape": "bogus"}))] == [
        "array", "two_part", "two_part"]
    assert review_session_output_schema(result.surface, request_output_shape(result)) is REVIEW_SESSION_OUTPUT_SCHEMA
    assert review_session_output_schema("multi_model_review") is TWO_PART_SESSION_OUTPUT_SCHEMA
    llm = _NoLightModel()
    slot = ReviewSlot(slot_id="r1", model="api/model-a", effort="low", timeout_sec=30, route=ReviewRouteKind.AGENT_SESSION)
    executor = AgentSessionReviewExecutor(
        ReviewAssignment(request=result, slot=slot, call_id="op-result-session", custody_root=tmp_path), llm=llm)
    assert executor._output_contract() is REVIEW_JSON_ARRAY_CONTRACT
    executor._raw_transcript, executor._conformance_passed = '{"findings": []}', True
    clean = executor._verdict_result()
    assert clean.raw_text == "[]" and llm.calls == [], "the conformed findings object is the clean array, no light call"
    native = ReviewSlot(slot_id="n", model="m", effort="low", subagent_id="api-critic", native_retrieval_override=True)
    assert NativeToolRoundReviewExecutor(
        ReviewAssignment(request=result, slot=native, call_id="c"))._output_contract() is REVIEW_JSON_ARRAY_CONTRACT
