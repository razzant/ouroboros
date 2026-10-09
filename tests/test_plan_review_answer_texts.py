"""The mind-facing texts of the answer channel state FACTS in every branch: the open
questions by id, the route an answer takes (recorded at $0, merged by id, the unchanged
envelope asks only the named slots), the escalated question that stays open until its
answer is recorded, and the deadline/cap producers that report state instead of advising
a plan of their own. Each pin fails with the old wording restored."""
from __future__ import annotations

import json

from ouroboros.tools.plan_render import _next_step
from tests.test_plan_review_engine import (  # noqa: F401
    CLEAN, _call, _control, _finding, _state, harness,
)

FP = "f" * 64


def _wave(findings: list, aggregate: str = "REVIEW_REQUIRED", dispositions: list | None = None) -> dict:
    return {"aggregate": aggregate, "closed": False, "request_fingerprint": FP,
            "findings": findings, "dispositions": dispositions or []}


def _q(fid: str, breaks: str = "claim_1", locator: str = "") -> dict:
    return {"finding_id": fid, "slot": fid.split(":")[0], "class": "need_evidence", "breaks": breaks, "locator": locator}


def test_next_step_names_open_questions_by_id():
    for aggregate in ("REVIEW_REQUIRED", "REVISE_PLAN"):
        findings = [_q("s1:q1"), _q("s2:q2", "goal"), _q("s3:e1", "claim_1", "notes.md::tail=16"),
                    {"finding_id": "s1:f1", "slot": "s1", "class": "blocking", "breaks": "claim_1"}]
        text = _next_step(_wave(findings, aggregate), enforcement="blocking", cap=5, cycles_paid=1)
        assert "Open questions to you: s1:q1 (claim_1), s2:q2 (goal). " in text
        assert "Open document requests: s3:e1 (notes.md::tail=16). " in text
        # answered questions leave the line; with none open the line is absent (two-sided)
        answered = [{"finding_id": fid, "decision": "accept", "rationale": "answered"} for fid in ("s1:q1", "s2:q2", "s3:e1")]
        text = _next_step(_wave(findings, aggregate, answered), enforcement="blocking", cap=5, cycles_paid=1)
        assert "Open questions to you" not in text and "Open document requests" not in text
    many = [_q(f"s1:q{i}") for i in range(10)]
    text = _next_step(_wave(many), enforcement="advisory", cap=5, cycles_paid=1)
    assert "s1:q7 (claim_1), +2 more. " in text and "s1:q9" not in text


def test_next_step_states_the_answer_route_in_every_open_branch(monkeypatch):
    from ouroboros.tools import plan_render

    blocking = [{"finding_id": "s1:f1", "slot": "s1", "class": "blocking", "breaks": "claim_1"}]
    for aggregate in ("REVIEW_REQUIRED", "REVISE_PLAN"):
        text = _next_step(_wave(blocking, aggregate), enforcement="blocking", cap=5, cycles_paid=1)
        assert f"plan_task(review_disposition={{review_fingerprint: '{FP}', items: [...]}})" in text
        assert "asks only the slots its items name — one paid cycle 2 of 5" in text
        assert "every other slot keeps its recorded answer at $0" in text and "Whether to buy that cycle is yours" in text
        assert "supersedes this wave (its recorded answers stay); the identical one replays free" in text
        for stale in ("reaches reviewers on the next paid cycle", "rides into the next paid delta cycle",
                      "can no longer be dispositioned", "Swarm", "accept ⇒ change the spec"):
            assert stale not in text, stale
    below = _next_step(_wave(blocking), enforcement="blocking", cap=5, cycles_paid=1)
    assert "stay open after any disposition; they leave the review when the slot that raised them no longer raises it" in below
    advisory = _next_step(_wave(blocking), enforcement="advisory", cap=5, cycles_paid=1)
    assert "a reject with its rationale closes each one; accept or defer keeps it open" in advisory
    # Cyber Pro: the facts and the route ride the branch too, never for a closed wave.
    monkeypatch.setattr(plan_render, "review_enforcement_blocks", lambda _mode: False)
    cyber = _next_step(_wave([_q("s1:q1")]), enforcement="blocking", cap=5, cycles_paid=1)
    assert cyber.startswith("Cyber Pro: Ouroboros decides") and "Open questions to you: s1:q1 (claim_1). " in cyber
    assert "An answer is recorded at $0" in cyber
    closed = _next_step({**_wave([]), "aggregate": "GREEN", "closed": True}, enforcement="blocking", cap=5, cycles_paid=1)
    assert "An answer is recorded" not in closed


def test_cap_reached_texts_report_state_and_authority_without_an_escape_recipe():
    blocking = [{"finding_id": "s1:f1", "slot": "s1", "class": "blocking", "breaks": "claim_1"}]
    text = _next_step(_wave(blocking, "REVISE_PLAN"), enforcement="blocking", cap=2, cycles_paid=2)
    assert "no paid cycle remains to show it to a reviewer: it stays recorded evidence" in text
    assert "finalization is RELEASED" in text and "implementation still held" in text
    assert "owner authority, never reviewer approval" in text and "Swarm" not in text and "unstick" not in text


def test_disposition_schema_states_the_answer_route_and_escalation():
    from ouroboros.tools.plan_review import _DISPOSITION_SCHEMA, get_tools

    description = _DISPOSITION_SCHEMA["description"]
    assert "merged by finding_id" in description
    assert "the unchanged envelope asks again ONLY the slots whose findings the items name" in description
    assert "a changed envelope is reviewed by every slot with the answers in view" in description
    assert "A quiz answer never closes a finding by itself" in description
    assert "defer (rationale names the quiz) while the quiz is open, accept with the owner's decision" in description
    assert "subsequent paid delta review" not in description
    items = _DISPOSITION_SCHEMA["properties"]["items"]
    assert "those slots are asked again" in items["description"]
    assert "the quiz id while it is open (defer)" in items["items"]["properties"]["rationale"]["description"]
    tool = next(t for t in get_tools() if t.name == "plan_task")
    desc = tool.schema["description"]
    assert "your answer re-judged by the originating slot" in desc
    assert "See review_disposition for free answers, closure rules and paid re-asks" in desc
    assert "subsequent paid delta" not in desc
    effort = tool.schema["parameters"]["properties"]["reviewer_effort"]["description"]
    assert "ignored when only collecting a recorded wave" in effort and "answering" not in effort


def test_blocking_reminder_states_the_answer_route():
    from ouroboros.owner_hurry import plan_review_reminder

    held = plan_review_reminder({"outcome": "REVIEW_REQUIRED", "status": "open", "cycles_paid": 1})
    assert "REVIEW_REQUIRED" in held and "the unchanged envelope with your items asks only that slot" in held
    assert "do not rerun reviewers" not in held and "Implementation stays held" in held
    revise = plan_review_reminder({"outcome": "REVISE_PLAN", "status": "open", "cycles_paid": 1})
    assert "asks only the slots those items name" in revise and "a new envelope every slot reviews" in revise
    assert "a task-wide rail releases finalization, never implementation" in revise
    assert "rail fires" not in revise and "rail fires" not in plan_review_reminder({"outcome": "", "status": "open"})


def test_deadline_and_cap_producers_state_facts_not_advice(harness, monkeypatch):  # noqa: F811
    """The deadline rail names remaining time and the absent review; the cap rail names the
    open review, the released finalization and the owner's authority — neither advises the
    mind to proceed with a plan of its own or to seek a reviewer-free escape."""
    from ouroboros.tools import plan_review as pr
    from ouroboros.deadline_utils import utc_now
    from ouroboros.tools.plan_review_runtime import plan_deadline_skip

    ctx = harness.make_ctx()
    ctx.task_metadata["deadline_at"] = (utc_now()).isoformat()
    monkeypatch.setattr(pr, "_plan_deadline_skip", plan_deadline_skip)
    text = plan_deadline_skip(ctx)
    assert text.startswith("PLAN_TASK_SKIPPED_DEADLINE:") and "remaining time 0s" in text
    assert "This call dispatched no reviewer" in text and "Any recorded plan review keeps its state" in text
    assert "no plan review is open" not in text and "proceed with your own" not in text.lower()
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    harness.install({"s1": json.dumps([_finding("f1", "blocking", breaks="claim_1")]), "s2": CLEAN, "s3": CLEAN})
    ctx2 = harness.make_ctx(task_id="task-cap")
    _call(ctx2)
    spent = _call(ctx2, plan="A revised outline.")
    assert "PLAN_REVIEW_CYCLES_EXHAUSTED" in spent and "finalization is RELEASED" in spent
    assert "outcome_tier=blocked_with_evidence" in spent and "owner authority, never reviewer approval" in spent
    assert "Swarm" not in spent and "Do not start the work" not in spent


def test_quiz_answer_never_closes_a_finding(harness):  # noqa: F811
    """A doc-only rule with no host shortcut to revert: an owner quiz answered through
    owner_quiz leaves the plan-review state byte-identical and the question open; the mind's
    later accept citing the decision is what closes it (two-sided)."""
    from ouroboros import owner_quiz
    from ouroboros.task_results import load_plan_review_state
    from ouroboros.tools import plan_review as pr

    harness.install({"s1": json.dumps([_finding("q1", "need_evidence", breaks="claim_1", summary="which?")]),
                     "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    assert _control(_call(ctx)) == {"outcome": "REVIEW_REQUIRED", "closed": False}
    before = json.dumps(load_plan_review_state(harness.drive, ctx.task_id), sort_keys=True)
    fp = _state(harness)["waves"][-1]["request_fingerprint"]
    owner_quiz.record_asked(harness.drive, ctx.task_id, quiz_id="quiz-1", question="which?",
                            options=["the first", "the second"])
    answered = owner_quiz.record_answered(harness.drive, ctx.task_id, quiz_id="quiz-1", option_index=1,
                                          request_id="req-1")
    assert answered["ok"]
    assert json.dumps(load_plan_review_state(harness.drive, ctx.task_id), sort_keys=True) == before
    closed = pr._handle_plan_task(ctx, review_disposition={"review_fingerprint": fp, "items": [
        {"finding_id": "s1:q1", "decision": "accept", "rationale": "owner decided: the second (quiz-1)"}]})
    assert _control(closed) == {"outcome": "GREEN", "closed": True}
