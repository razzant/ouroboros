"""The review substrate itself: slots, quorum, parsing, budgets and refs.

Split by theme out of the original giant of the same name. This module owns the
substrate mechanics: slot transport and duplicate-model independence, quorum
degradation and outcome tiers, fenced JSON parsing, per-slot retries, budget
rails, usage emission and the actor/scope refs the substrate persists.
"""

import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request
from ouroboros.triad_review import parse_model_review_results

from tests._review_substrate_shared import FakeLLM

def test_review_slot_passes_explicit_local_transport_to_llm(tmp_path):
    llm = FakeLLM()
    result = run_review_request(
        ReviewRequest(surface="scope", goal="review locally", task_id="local-review"),
        slots=[ReviewSlot(slot_id="local", model="owner/local-main", use_local=True)],
        drive_root=tmp_path,
        llm=llm,
    )

    assert result.actors
    assert llm.calls and llm.calls[0]["use_local"] is True


class FencedArrayLLM:
    def chat(self, **kwargs):
        body = (
            "Here is the review:\n"
            "```json\n"
            "[{\"verdict\":\"FAIL\",\"severity\":\"critical\",\"item\":\"x\",\"evidence\":\"e\",\"recommendation\":\"r\"}]\n"
            "```"
        )
        return {"content": body}, {"prompt_tokens": 10, "completion_tokens": 5}


class FencedObjectLLM:
    def chat(self, **kwargs):
        body = (
            "Verdict below:\n"
            "```json\n"
            "{\"verdict\":\"PASS\",\"findings\":[]}\n"
            "```"
        )
        return {"content": body}, {"prompt_tokens": 10, "completion_tokens": 5}


class ConcernsLLM:
    def chat(self, **kwargs):
        # Valid JSON, transport ok, but a non-PASS/FAIL/DEGRADED verdict.
        return {"content": "{\"verdict\":\"CONCERNS\",\"findings\":[]}"}, {"prompt_tokens": 10, "completion_tokens": 5}


class ParseDegradedSlotLLM:
    """Two slots PASS; the '-2' slot returns a successful but DEGRADED-verdict
    response (a reviewer doubt, NOT a transport/participation fault)."""

    def chat(self, **kwargs):
        if str(kwargs.get("model", "")).endswith("-2"):
            return {"content": json.dumps({"verdict": "DEGRADED", "findings": [], "summary": "unsure"})}, {}
        return {"content": json.dumps({"verdict": "PASS", "findings": [], "summary": "ok"})}, {}


class ActorErrorSlotLLM:
    """Two slots PASS; the '-2' slot raises (a participation fault)."""

    def chat(self, **kwargs):
        if str(kwargs.get("model", "")).endswith("-2"):
            raise RuntimeError("provider exploded")
        return {"content": json.dumps({"verdict": "PASS", "findings": [], "summary": "ok"})}, {}


def test_degraded_or_errored_acceptance_slot_abstains_when_quorum_remains(tmp_path):
    """T1 (v6.35.0): a single unparseable/DEGRADED-verdict slot must NOT poison a
    clean 2-of-3 PASS quorum. A participation fault also abstains on task
    acceptance when the configured PASS quorum remains; no-quorum is DEGRADED."""
    slots = [ReviewSlot(slot_id=f"s{i}", model=f"m-{i}") for i in range(3)]

    def _req():
        return ReviewRequest(
            surface="task_acceptance", goal="g", subject="done",
            policy={"fail_closed_on_errors": True, "min_successful_slots": 2}, task_id="t",
        )

    ok = run_review_request(_req(), slots=slots, drive_root=tmp_path, llm=ParseDegradedSlotLLM())
    assert ok.aggregate_signal == "PASS"
    assert ok.degraded is False

    bad = run_review_request(_req(), slots=slots, drive_root=tmp_path, llm=ActorErrorSlotLLM())
    assert bad.aggregate_signal == "PASS"
    assert bad.degraded is False


class PassNoTierLLM:
    """PASS verdict but NO outcome_tier — the non-compliant reviewer the required-
    tier contract must catch (a tier-less PASS must not aggregate to a clean PASS)."""

    def chat(self, **kwargs):
        return {"content": json.dumps({"verdict": "PASS", "findings": [], "summary": "ok"})}, {}


class PassWithTierLLM:
    def chat(self, **kwargs):
        body = {"verdict": "PASS", "outcome_tier": "solved", "completion_coach": "ship",
                "criteria_used": [{"criterion": "verified", "status": "supported", "evidence_refs": ["verification_summary"]}], "findings": [], "summary": "ok"}
        return {"content": json.dumps(body)}, {}


class PassTierNoCoachLLM:
    """PASS with a valid outcome_tier but EMPTY completion_coach — still
    non-responsive to the required-tier contract (both keys required)."""

    def chat(self, **kwargs):
        body = {"verdict": "PASS", "outcome_tier": "solved", "completion_coach": "",
                "criteria_used": [{"criterion": "verified", "status": "supported", "evidence_refs": ["verification_summary"]}], "findings": [], "summary": "ok"}
        return {"content": json.dumps(body)}, {}


class PoisonDegradedSlotLLM:
    """Two slots PASS+solved; the '-2' slot returns a DEGRADED verdict carrying a
    BLOCKED outcome_tier + a critical finding — a parse-degraded actor that must
    NOT poison the clean quorum PASS capsule (v6.36.0 review finding)."""

    def chat(self, **kwargs):
        if str(kwargs.get("model", "")).endswith("-2"):
            return {"content": json.dumps({
                "verdict": "DEGRADED",
                "outcome_tier": "blocked_with_evidence",
                "completion_coach": "STOP everything",
                "findings": [{"verdict": "FAIL", "severity": "critical",
                              "item": "poison", "recommendation": "do not ship this"}],
                "summary": "unsure",
            })}, {}
        return {"content": json.dumps({
            "verdict": "PASS", "outcome_tier": "solved", "completion_coach": "ship it",
            "criteria_used": [{"criterion": "verified", "status": "supported",
                               "evidence_refs": ["verification_summary"]}],
            "findings": [], "summary": "ok",
        })}, {}


def test_degraded_actor_does_not_poison_acceptance_capsule(tmp_path):
    """v6.36.0 (scope review finding): aggregate_outcome_tier / build_improvement_
    capsule must draw tier/coach/findings ONLY from actors that contributed to the
    aggregate verdict — a single parse-degraded slot carrying a BLOCKED tier must
    not inject a blocking improvement note into an otherwise-clean quorum PASS.
    v6.55.0 (codex/fable-5 cumulative review): a DELIBERATE minority DEGRADED
    verdict carrying a concrete recommendation now surfaces as ONE labeled
    non-veto [DISSENT] line (the GAIA 3cef3a44 class) — while the mainline
    capsule (tier / coach / bullets) stays unpoisoned exactly as before."""
    from ouroboros.review_substrate import aggregate_outcome_tier, build_improvement_capsule
    slots = [ReviewSlot(slot_id=f"s{i}", model=f"m-{i}") for i in range(3)]
    req = ReviewRequest(
        surface="task_acceptance", goal="g", subject="done",
        policy={"classify_outcome_tier": True, "min_successful_slots": 2}, task_id="t",
    )
    res = run_review_request(req, slots=slots, drive_root=tmp_path, llm=PoisonDegradedSlotLLM())
    assert res.aggregate_signal == "PASS"
    # The degraded '-2' slot's BLOCKED tier / coach must NOT surface.
    assert aggregate_outcome_tier(res) == "solved"
    capsule = build_improvement_capsule(res)
    assert "STOP everything" not in capsule
    assert "blocked" not in capsule.lower()
    # ...but its deliberate DEGRADED verdict + concrete recommendation IS the
    # dissent class: one labeled line, never a mainline bullet.
    assert "[DISSENT — s2 said DEGRADED]" in capsule
    assert "do not ship this" in capsule
    assert "- do not ship this" not in capsule


class ContractDegradedPassLLM:
    """Two slots PASS+solved+coach (contract-valid); the '-2' slot returns
    verdict=PASS but a BLOCKED outcome_tier with an EMPTY completion_coach — a
    CONTRACT-DEGRADED PASS (non-responsive to the required-tier contract). It must
    not contribute its blocked tier / finding to the clean quorum capsule
    (v6.36.0 round-2 scope finding: the live PASS-but-contract-degraded path)."""

    def chat(self, **kwargs):
        if str(kwargs.get("model", "")).endswith("-2"):
            return {"content": json.dumps({
                "verdict": "PASS", "outcome_tier": "blocked_with_evidence", "completion_coach": "",
                "findings": [{"verdict": "FAIL", "severity": "critical",
                              "item": "poison2", "recommendation": "block this hard"}],
                "summary": "x",
            })}, {}
        return {"content": json.dumps({
            "verdict": "PASS", "outcome_tier": "solved", "completion_coach": "ship",
            "criteria_used": [{"criterion": "verified", "status": "supported",
                               "evidence_refs": ["verification_summary"]}],
            "findings": [], "summary": "ok",
        })}, {}


def test_contract_degraded_pass_does_not_poison_capsule(tmp_path):
    """v6.36.0 round-2 scope finding: a verdict=PASS actor that VIOLATES the
    required tier/coach contract is demoted to non-contributing (signal->DEGRADED),
    so it can't feed its blocked tier / finding into the clean quorum PASS capsule —
    the live path the DEGRADED-verdict-only test did not cover."""
    from ouroboros.review_substrate import (
        aggregate_outcome_tier,
        build_improvement_capsule,
        compact_review_projection,
    )
    slots = [ReviewSlot(slot_id=f"s{i}", model=f"m-{i}") for i in range(3)]
    req = ReviewRequest(
        surface="task_acceptance", goal="g", subject="done",
        policy={"classify_outcome_tier": True, "min_successful_slots": 2}, task_id="t",
    )
    res = run_review_request(req, slots=slots, drive_root=tmp_path, llm=ContractDegradedPassLLM())
    assert res.aggregate_signal == "PASS"          # the two contract-valid solved PASS reach quorum
    assert aggregate_outcome_tier(res) == "solved"  # the blocked contract-degraded PASS is excluded
    malformed = next(actor for actor in res.actors if actor["slot_id"] == "s2")
    assert malformed["transport_status"] == "success"
    assert malformed["parse_status"] == "malformed"
    assert malformed["semantic_verdict"] == ""
    assert malformed["quorum_contribution"] is False
    assert "violated the required" in malformed["reason"]
    projected = compact_review_projection([dict(res.__dict__)])["panels"][0]
    projected_malformed = next(
        actor for actor in projected["actors"] if actor["slot_id"] == "s2"
    )
    assert projected_malformed["parse_status"] == "malformed"
    assert projected_malformed["semantic_verdict"] == ""
    capsule = build_improvement_capsule(res)
    assert "block this hard" not in capsule
    assert "blocked" not in capsule.lower()


def test_solved_pass_with_required_coach_does_not_force_reloop(tmp_path):
    """v6.36.0 round-2 cross-module finding: a contract-valid SOLVED review carries
    a required completion_coach, but a coach ALONE must not force a revise round —
    build_improvement_capsule returns '' for a solved/no-findings result."""
    from ouroboros.review_substrate import build_improvement_capsule
    slots = [ReviewSlot(slot_id=f"s{i}", model=f"m-{i}") for i in range(3)]
    req = ReviewRequest(
        surface="task_acceptance", goal="g", subject="done",
        policy={"classify_outcome_tier": True, "min_successful_slots": 2}, task_id="t",
    )
    res = run_review_request(req, slots=slots, drive_root=tmp_path, llm=PassWithTierLLM())
    assert res.aggregate_signal == "PASS"
    assert build_improvement_capsule(res) == ""  # solved + coach, no findings -> finalize, no re-loop


def test_single_configured_reviewer_marks_no_diversity(tmp_path):
    """v6.36.0 (Bible P3, centralized): a one-slot review through the coordinator
    is honored but records single_reviewer_no_diversity durably on EVERY surface —
    so a one-slot acceptance review can never quietly look like an ordinary
    multi-reviewer PASS. v6.74.0 (A6): the note is the TYPED FIELD (an orthogonal
    label projected on the panel), no longer a degraded_reason — the panel reason
    must name the real blocker, not lead with a diversity footnote."""
    one = run_review_request(
        ReviewRequest(surface="task_acceptance", goal="g", subject="d",
                      policy={"min_successful_slots": 1}, task_id="t"),
        slots=[ReviewSlot(slot_id="s0", model="m-0")],
        drive_root=tmp_path, llm=PassWithTierLLM(),
    )
    assert one.single_reviewer_no_diversity is True
    assert "single_reviewer_no_diversity" not in one.degraded_reasons
    from ouroboros.review_substrate import compact_review_projection

    projection = compact_review_projection([
        {**one.__dict__, "authority": "host_root"},
    ])
    assert projection["panels"][0]["single_reviewer_no_diversity"] is True

    three = run_review_request(
        ReviewRequest(surface="task_acceptance", goal="g", subject="d",
                      policy={"min_successful_slots": 2}, task_id="t"),
        slots=[ReviewSlot(slot_id=f"s{i}", model=f"m-{i}") for i in range(3)],
        drive_root=tmp_path, llm=PassWithTierLLM(),
    )
    assert three.single_reviewer_no_diversity is False
    assert "single_reviewer_no_diversity" not in three.degraded_reasons


def test_required_outcome_tier_is_enforced_at_quorum(tmp_path):
    """T1 (v6.35.0): with classify_outcome_tier policy, a PASS WITHOUT a valid
    outcome_tier cannot count toward a clean quorum — the required-tier contract
    is enforced at the parser/quorum level, not just asked for in the prompt.

    v6.46.0 (Q7): on the ADVISORY task-acceptance surface, a SOLVED deliverable has
    no tier-up step, so an empty completion_coach must NOT demote a solved PASS to
    DEGRADED. A tier-LESS PASS is still non-responsive."""
    slots = [ReviewSlot(slot_id=f"s{i}", model=f"m-{i}") for i in range(3)]

    def _req():
        return ReviewRequest(
            surface="task_acceptance", goal="g", subject="done",
            policy={"classify_outcome_tier": True, "min_successful_slots": 2}, task_id="t",
        )

    no_tier = run_review_request(_req(), slots=slots, drive_root=tmp_path, llm=PassNoTierLLM())
    assert no_tier.aggregate_signal == "DEGRADED"  # tier-less PASS is still non-responsive

    # Advisory carve-out: a SOLVED PASS without a coach is RESPONSIVE (nothing to improve).
    no_coach = run_review_request(_req(), slots=slots, drive_root=tmp_path, llm=PassTierNoCoachLLM())
    assert no_coach.aggregate_signal == "PASS"

    with_tier = run_review_request(_req(), slots=slots, drive_root=tmp_path, llm=PassWithTierLLM())
    assert with_tier.aggregate_signal == "PASS"


def test_p3_surfaces_ignore_task_acceptance_tier_policy(tmp_path):
    """Defense in depth: even if a caller accidentally carries the task-only
    classify_outcome_tier flag, commit/scope FAILs remain authoritative vetoes."""
    for surface in ("multi_model_review", "scope_review"):
        result = run_review_request(
            ReviewRequest(
                surface=surface,
                goal="review",
                subject="diff",
                policy={"classify_outcome_tier": True, "min_successful_slots": 1},
                task_id=f"t-{surface}",
            ),
            slots=[ReviewSlot(slot_id="s0", model="m-0")],
            drive_root=tmp_path,
            llm=FencedArrayLLM(),
        )
        assert result.aggregate_signal == "FAIL"
        assert result.actors[0]["signal"] == "FAIL"

def test_review_substrate_treats_duplicate_models_as_independent_slots(tmp_path):
    llm = FakeLLM()
    slots = [
        ReviewSlot(slot_id="triad_a", model="same/model", effort="high"),
        ReviewSlot(slot_id="triad_b", model="same/model", effort="high"),
    ]
    result = run_review_request(
        ReviewRequest(surface="task_acceptance", goal="verify final claim", subject="done", task_id="task-1"),
        slots=slots,
        drive_root=tmp_path,
        llm=llm,
    )

    assert result.aggregate_signal == "PASS"
    assert [actor["slot_id"] for actor in result.actors] == ["triad_a", "triad_b"]
    assert [call["model"] for call in llm.calls] == ["same/model", "same/model"]
    for actor in result.actors:
        assert actor["prompt_ref"]["manifest_ref"]["path"]
        assert actor["response_ref"]["manifest_ref"]["path"]


def test_review_substrate_queues_all_slots_above_concurrency_cap(tmp_path):
    llm = FakeLLM()
    slots = [
        ReviewSlot(slot_id=f"slot_{idx}", model=f"model-{idx}", effort="high")
        for idx in range(10)
    ]
    result = run_review_request(
        ReviewRequest(surface="task_acceptance", goal="verify final claim", subject="done", task_id="task-10"),
        slots=slots,
        drive_root=tmp_path,
        llm=llm,
    )

    assert result.aggregate_signal == "PASS"
    assert [actor["slot_id"] for actor in result.actors] == [slot.slot_id for slot in slots]
    assert {call["model"] for call in llm.calls} == {slot.model for slot in slots}
    assert len(llm.calls) == 10
    assert all(actor["status"] == "ok" for actor in result.actors)

    slow_calls = []
    slow_llm = SimpleNamespace(chat=lambda **kwargs: (
        slow_calls.append(kwargs),
        time.sleep(0.2),
        ({"content": "{\"verdict\":\"PASS\",\"findings\":[],\"summary\":\"late\"}"}, {}),
    )[-1])
    slow_slots = [
        ReviewSlot(slot_id=f"slow_{idx}", model=f"slow-model-{idx}", effort="high", timeout_sec=0.05)
        for idx in range(10)
    ]
    slow_result = run_review_request(
        ReviewRequest(surface="task_acceptance", goal="verify final claim", subject="done", task_id="task-slow"),
        slots=slow_slots,
        drive_root=tmp_path,
        llm=slow_llm,
    )
    assert len(slow_calls) == 10
    assert "Not started before reviewer timeout budget expired" not in "\n".join(slow_result.degraded_reasons)


def test_review_substrate_reports_no_slots_as_degraded(tmp_path):
    result = run_review_request(
        ReviewRequest(surface="plan", goal="review plan", task_id="task-1"),
        slots=[],
        drive_root=tmp_path,
        llm=FakeLLM(),
    )

    assert result.aggregate_signal == "DEGRADED"
    assert result.degraded is True
    assert "no_review_slots" in result.degraded_reasons


def test_review_substrate_emits_usage_when_context_supplied(tmp_path):
    class Ctx:
        task_id = "task-usage"
        pending_events = []

    ctx = Ctx()
    result = run_review_request(
        ReviewRequest(surface="task_acceptance", goal="review claim", task_id="task-usage"),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model")],
        drive_root=tmp_path,
        llm=FakeLLM(),
        usage_ctx=ctx,
    )

    assert result.aggregate_signal == "PASS"
    usage_events = [event for event in ctx.pending_events if event.get("type") == "llm_usage"]
    assert len(usage_events) == 1
    assert usage_events[0]["task_id"] == "task-usage"
    assert usage_events[0]["source"] == "review_substrate:task_acceptance"
    assert usage_events[0]["slot_id"] == "slot_a"


def test_review_usage_preserves_unknown_cost_as_null():
    from ouroboros.tools.review_helpers import emit_review_usage

    ctx = SimpleNamespace(task_id="unknown-review", pending_events=[])
    emit_review_usage(ctx, model="unknown/model", usage={}, source="test")
    event = ctx.pending_events[0]
    assert event["usage"]["cost"] is None
    assert event["usage"]["cost_known"] is False


def test_one_llm_usage_row_per_physical_reviewer_call(tmp_path):
    """A wave of N reviewer slots emits exactly N rows, each naming its slot."""

    class Ctx:
        task_id = "task-wave"
        pending_events = []

    ctx = Ctx()
    run_review_request(
        ReviewRequest(surface="multi_model_review", goal="review claim", task_id="task-wave"),
        slots=[
            ReviewSlot(slot_id="slot_a", model="same/model"),
            ReviewSlot(slot_id="slot_b", model="same/model"),
            ReviewSlot(slot_id="slot_c", model="same/model"),
        ],
        drive_root=tmp_path,
        llm=FakeLLM(),
        usage_ctx=ctx,
    )

    usage_events = [event for event in ctx.pending_events if event.get("type") == "llm_usage"]
    assert len(usage_events) == 3
    assert {event["slot_id"] for event in usage_events} == {"slot_a", "slot_b", "slot_c"}
    assert {event["source"] for event in usage_events} == {"review_substrate:multi_model_review"}


def test_format_repair_emits_one_usage_row_per_send_without_aggregate(tmp_path):
    class RepairLLM:
        def __init__(self):
            self.calls = 0

        def chat(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"content": "malformed"}, {"prompt_tokens": 3, "ledger_attempt_ids": ["a1"]}
            return {"content": "[]"}, {"prompt_tokens": 4, "ledger_attempt_ids": ["a2"]}

    ctx = SimpleNamespace(task_id="repair-usage", event_queue=None, pending_events=[])
    llm = RepairLLM()
    run_review_request(
        ReviewRequest(surface="task_acceptance", goal="review", task_id=ctx.task_id),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model")],
        drive_root=tmp_path, llm=llm, usage_ctx=ctx,
    )

    rows = [event for event in ctx.pending_events if event.get("type") == "llm_usage"]
    assert llm.calls == 2
    assert [row["ledger_attempt_ids"] for row in rows] == [["a1"], ["a2"]]


def test_failed_physical_reviewer_send_still_emits_its_usage_row(tmp_path):
    from ouroboros.usage_accounting import PhysicalAttemptCapture

    class FailedLLM:
        def chat(self, **_kwargs):
            error = RuntimeError("provider failed after dispatch")
            error.physical_attempt_capture = PhysicalAttemptCapture(
                attempt_id="failed-a1", model="same/model", provider="openrouter",
                state="unresolved", candidate_measurement_kind="opaque",
            )
            raise error

    ctx = SimpleNamespace(task_id="failed-usage", event_queue=None, pending_events=[])
    run_review_request(
        ReviewRequest(surface="plan_review", goal="review", task_id=ctx.task_id),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model")],
        drive_root=tmp_path, llm=FailedLLM(), usage_ctx=ctx,
    )

    rows = [event for event in ctx.pending_events if event.get("type") == "llm_usage"]
    assert len(rows) == 1
    assert rows[0]["ledger_attempt_ids"] == ["failed-a1"]


def test_terminal_failed_reviewer_retry_emits_one_row_per_dispatched_attempt(tmp_path):
    from ouroboros.usage_accounting import (
        AttemptRequest, capture_attempt_ids, execute_physical_attempt,
    )

    class FailedRetryLLM:
        def chat(self, **_kwargs):
            with capture_attempt_ids():
                for attempt in range(2):
                    try:
                        execute_physical_attempt(AttemptRequest(
                            model="same/model", provider="openrouter",
                            reservation_usd=0.0,
                        ), lambda: (_ for _ in ()).throw(RuntimeError(f"failed-{attempt}")))
                    except RuntimeError:
                        if attempt:
                            raise

    ctx = SimpleNamespace(task_id="failed-retry-usage", event_queue=None, pending_events=[])
    run_review_request(
        ReviewRequest(surface="plan_review", goal="review", task_id=ctx.task_id),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model")],
        drive_root=tmp_path, llm=FailedRetryLLM(), usage_ctx=ctx,
    )

    rows = [event for event in ctx.pending_events if event.get("type") == "llm_usage"]
    assert len(rows) == 2
    assert all(len(row["ledger_attempt_ids"]) == 1 for row in rows)
    assert rows[0]["ledger_attempt_ids"] != rows[1]["ledger_attempt_ids"]


def test_terminal_budget_refusal_keeps_prior_dispatched_retry_usage(tmp_path):
    from ouroboros.usage_accounting import (
        AttemptRequest, capture_attempt_ids, execute_physical_attempt,
    )

    class BudgetStopsRetryLLM:
        def chat(self, **_kwargs):
            request = AttemptRequest(
                model="same/model", provider="openrouter",
                reservation_usd=1.0, global_limit_usd=1.0,
            )
            with capture_attempt_ids():
                try:
                    execute_physical_attempt(
                        request, lambda: (_ for _ in ()).throw(RuntimeError("dispatched")),
                    )
                except RuntimeError:
                    pass
                execute_physical_attempt(request, lambda: None)

    ctx = SimpleNamespace(task_id="budget-refusal-usage", event_queue=None, pending_events=[])
    run_review_request(
        ReviewRequest(surface="task_acceptance", goal="review", task_id=ctx.task_id),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model")],
        drive_root=tmp_path, llm=BudgetStopsRetryLLM(), usage_ctx=ctx,
    )

    rows = [event for event in ctx.pending_events if event.get("type") == "llm_usage"]
    assert len(rows) == 1
    assert len(rows[0]["ledger_attempt_ids"]) == 1


def test_terminal_attempt_limit_keeps_prior_send_but_excludes_released_hold(tmp_path):
    from ouroboros.usage_accounting import (
        AttemptRequest, capture_attempt_ids, execute_physical_attempt,
        physical_attempt_limit,
    )

    class RailStopsRetryLLM:
        def chat(self, **_kwargs):
            request = AttemptRequest(
                model="same/model", provider="openrouter", reservation_usd=0.0,
            )
            with capture_attempt_ids(), physical_attempt_limit(1):
                try:
                    execute_physical_attempt(
                        request, lambda: (_ for _ in ()).throw(RuntimeError("dispatched")),
                    )
                except RuntimeError:
                    pass
                execute_physical_attempt(request, lambda: None)

    ctx = SimpleNamespace(task_id="rail-refusal-usage", event_queue=None, pending_events=[])
    run_review_request(
        ReviewRequest(surface="task_acceptance", goal="review", task_id=ctx.task_id),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model")],
        drive_root=tmp_path, llm=RailStopsRetryLLM(), usage_ctx=ctx,
    )

    rows = [event for event in ctx.pending_events if event.get("type") == "llm_usage"]
    assert len(rows) == 1
    assert len(rows[0]["ledger_attempt_ids"]) == 1


def test_internal_reviewer_transport_attempts_each_get_one_usage_row(tmp_path):
    class RetriedLLM:
        def chat(self, **_kwargs):
            return {"content": "[]"}, {
                "prompt_tokens": 4, "ledger_attempt_ids": ["wire-a1", "wire-a2"],
            }

    ctx = SimpleNamespace(task_id="wire-retry", event_queue=None, pending_events=[])
    run_review_request(
        ReviewRequest(surface="plan_review", goal="review", task_id=ctx.task_id),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model")],
        drive_root=tmp_path, llm=RetriedLLM(), usage_ctx=ctx,
    )

    rows = [event for event in ctx.pending_events if event.get("type") == "llm_usage"]
    assert [row["ledger_attempt_ids"] for row in rows] == [["wire-a1"], ["wire-a2"]]
    assert [row["usage"]["prompt_tokens"] for row in rows] == [0, 4]


def test_the_single_usage_row_carries_the_reviewer_attribution(tmp_path):
    """The surviving row is traceable to its wave and slot, not just its task.

    The substrate is the only emitter of a reviewer row, so this row is the
    whole projection of that reviewer call: it has to carry the same
    attribution the ledger row does.
    """

    class Ctx:
        task_id = "task-attr"
        pending_events = []

    ctx = Ctx()
    run_review_request(
        ReviewRequest(
            surface="multi_model_review", goal="review claim", task_id="task-attr",
            usage_attribution={"review_wave_id": "wave-77", "review_skill": "commit_triad"},
        ),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model")],
        drive_root=tmp_path,
        llm=FakeLLM(),
        usage_ctx=ctx,
    )

    usage_events = [event for event in ctx.pending_events if event.get("type") == "llm_usage"]
    assert len(usage_events) == 1
    row = usage_events[0]
    assert row["review_wave_id"] == "wave-77"
    assert row["review_slot_id"] == "slot_a"
    assert row["review_skill"] == "commit_triad"


def test_session_row_reports_its_own_route_provider_and_model(tmp_path):
    """A delegated session's own facts outrank an inferred provider."""
    from ouroboros.tools.review_helpers import emit_review_usage

    ctx = SimpleNamespace(task_id="session-review", pending_events=[])
    emit_review_usage(
        ctx,
        model="slot/requested-model",
        provider="claudexor",
        usage={"prompt_tokens": 10, "completion_tokens": 2, "resolved_model": "gpt-5.6-sol"},
        source="review_substrate:multi_model_review",
    )
    api_ctx = SimpleNamespace(task_id="api-review", pending_events=[])
    emit_review_usage(
        api_ctx, model="anthropic/claude-test", usage={"prompt_tokens": 10}, source="test",
    )

    assert ctx.pending_events[0]["provider"] == "claudexor"
    assert api_ctx.pending_events[0]["provider"] == "openrouter"  # inferred, as before


def test_review_substrate_parses_fenced_json_array_findings(tmp_path):
    result = run_review_request(
        ReviewRequest(surface="scope", goal="review diff", task_id="task-json-array"),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model")],
        drive_root=tmp_path,
        llm=FencedArrayLLM(),
    )

    assert result.aggregate_signal == "FAIL"
    assert result.parsed_findings[0]["item"] == "x"
    assert result.actors[0]["parsed"][0]["verdict"] == "FAIL"


def test_review_substrate_parses_fenced_json_object_verdict(tmp_path):
    # A fenced JSON OBJECT (not array) must parse as PASS, not a false DEGRADED.
    result = run_review_request(
        ReviewRequest(surface="task_acceptance", goal="verify claim", subject="done", task_id="task-obj"),
        slots=[
            ReviewSlot(slot_id="slot_a", model="m"),
            ReviewSlot(slot_id="slot_b", model="m"),
        ],
        drive_root=tmp_path,
        llm=FencedObjectLLM(),
    )
    assert result.aggregate_signal == "PASS"
    assert result.degraded is False


def test_review_substrate_degraded_quorum_carries_reason(tmp_path):
    # No FAIL, no PASS quorum, no transport errors -> DEGRADED must still be honest:
    # degraded=True with a non-empty reason (no DEGRADED/degraded=False/empty mismatch).
    result = run_review_request(
        ReviewRequest(
            surface="task_acceptance", goal="verify claim", subject="done", task_id="task-quorum",
            policy={"min_successful_slots": 2},
        ),
        slots=[
            ReviewSlot(slot_id="slot_a", model="m"),
            ReviewSlot(slot_id="slot_b", model="m"),
        ],
        drive_root=tmp_path,
        llm=ConcernsLLM(),
    )
    assert result.aggregate_signal == "DEGRADED"
    assert result.degraded is True
    assert result.degraded_reasons
    assert any("quorum_not_met" in reason for reason in result.degraded_reasons)


def test_p3_commit_actor_retries_same_slot_model_once_then_blocks(tmp_path):
    recovered_llm = Mock()
    recovered_llm.chat.side_effect = [
        TimeoutError("transient timeout"),
        (
            {"content": "{\"verdict\":\"PASS\",\"findings\":[],\"summary\":\"ok\"}"},
            {"prompt_tokens": 1, "completion_tokens": 1},
        ),
    ]
    recovered = run_review_request(
        ReviewRequest(
            surface="multi_model_review", goal="review diff",
            task_id="task-recovered", call_type="multi_model_review",
        ),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model")],
        drive_root=tmp_path,
        llm=recovered_llm,
    )
    assert recovered.aggregate_signal == "PASS"
    assert recovered.actors[0]["status"] == "ok"
    assert recovered_llm.chat.call_count == 2
    assert recovered_llm.chat.call_args_list[0].kwargs == recovered_llm.chat.call_args_list[1].kwargs

    failed_llm = Mock()
    failed_llm.chat.side_effect = [
        TimeoutError("transient timeout"),
        RuntimeError("provider exploded"),
    ]
    result = run_review_request(
        ReviewRequest(
            surface="multi_model_review", goal="review diff",
            task_id="task-error", call_type="multi_model_review",
        ),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model")],
        drive_root=tmp_path,
        llm=failed_llm,
    )

    actor = result.actors[0]
    assert result.aggregate_signal == "DEGRADED"
    assert failed_llm.chat.call_count == 2
    assert actor["status"] == "error"
    assert "provider exploded" in actor["error"]
    assert actor["prompt_ref"]["manifest_ref"]["path"]
    assert actor["response_ref"]["manifest_ref"]["path"]
    manifest = json.loads(open(actor["response_ref"]["manifest_ref"]["path"], encoding="utf-8").read())
    assert manifest["call_type"] == "multi_model_review_error"
    assert manifest["status"] == "error"

    from ouroboros.usage_accounting import _claim_physical_dispatch

    over_limit_llm = SimpleNamespace(chat=lambda **_kwargs: (
        _claim_physical_dispatch(),
        _claim_physical_dispatch(),
        _claim_physical_dispatch(),
    ))
    over_limit = run_review_request(
        ReviewRequest(
            surface="multi_model_review", goal="review diff",
            task_id="task-over-limit", call_type="multi_model_review",
        ),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model")],
        drive_root=tmp_path,
        llm=over_limit_llm,
    )
    assert over_limit.actors[0]["status"] == "error"
    assert "physical attempt limit exhausted (2/2)" in over_limit.actors[0]["error"]


def test_p3_scope_actor_retries_empty_same_slot_model_once_then_blocks(tmp_path, monkeypatch):
    from ouroboros.tools import scope_review

    rows = [
        {
            "item": item,
            "verdict": "PASS",
            "severity": "advisory",
            "reason": "Concrete scope artifact was checked and passes.",
        }
        for item in sorted(scope_review._SCOPE_REQUIRED_ITEMS)
    ]
    recovered_llm = Mock()
    recovered_llm.chat.side_effect = [
        ({"content": ""}, {"prompt_tokens": 0, "completion_tokens": 0}),
        (
            {"content": json.dumps(rows)},
            {"prompt_tokens": 1, "completion_tokens": 1},
        ),
    ]
    monkeypatch.setattr(scope_review, "LLMClient", lambda: recovered_llm)
    monkeypatch.setattr(scope_review, "_build_scope_prompt", lambda *a, **k: ("scope prompt", None))
    monkeypatch.setattr(scope_review, "_scope_window",
                        lambda _model, **_k: scope_review.ReviewerWindow(1_000_000, "confirmed"))
    ctx = SimpleNamespace(
        repo_dir=tmp_path, drive_root=tmp_path,
        task_id="scope-recovered", pending_events=[],
    )
    recovered = scope_review.run_scope_review(ctx, "review scope", scope_model="scope/model")
    assert recovered.status == "responded"
    assert recovered.blocked is False
    assert recovered_llm.chat.call_count == 2
    assert recovered_llm.chat.call_args_list[0].kwargs == recovered_llm.chat.call_args_list[1].kwargs

    empty_llm = Mock()
    empty_llm.chat.side_effect = [
        ({"content": ""}, {"prompt_tokens": 0, "completion_tokens": 0}),
        ({"content": ""}, {"prompt_tokens": 0, "completion_tokens": 0}),
    ]
    monkeypatch.setattr(scope_review, "LLMClient", lambda: empty_llm)
    ctx.task_id = "scope-empty"
    failed = scope_review.run_scope_review(ctx, "review scope", scope_model="scope/model")
    assert failed.blocked is True
    assert failed.status == "empty_response"
    assert failed.operation_id
    assert empty_llm.chat.call_count == 2


def test_review_substrate_persists_timeout_actor_refs(tmp_path):
    import threading
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key

    release = threading.Event()

    def gated_chat(**_kwargs):
        # Holds the call open until the test releases it, so the 0.01s window
        # is GUARANTEED to expire and the timeout actor is the one persisted.
        # The previous 0.2s-sleep-vs-0.01s-window margin was NOT discriminating
        # on a loaded CI host (the heal wave measured a 0.207s poll oversleep):
        # a pre-window settle replaced the asserted timeout actor with PASS
        # (same event gate as
        # test_replayed_late_review_does_not_charge_same_context_twice).
        assert release.wait(10), "test never released the gated review call"
        return {"content": "{\"verdict\":\"PASS\",\"findings\":[],\"summary\":\"late\"}"}, {}

    hanging_llm = SimpleNamespace(chat=gated_chat)
    request = ReviewRequest(surface="scope", goal="review diff", task_id="task-timeout")
    slot = ReviewSlot(slot_id="slot_a", model="same/model", timeout_sec=0.01)
    result = run_review_request(
        request,
        slots=[slot],
        drive_root=tmp_path,
        llm=hanging_llm,
    )

    actor = result.actors[0]
    assert actor["status"] == "error"
    assert "Timeout after" in actor["error"]
    assert actor["prompt_ref"]["manifest_ref"]["path"]
    assert actor["response_ref"]["manifest_ref"]["path"]
    release.set()
    # Drain the released worker before teardown (same drain as
    # test_explicit_retry_key_joins_worker_after_prompt_history_changes).
    key = _attempt_key(request, slot)
    deadline = time.monotonic() + 5.0
    active = True
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            active = key in _ACTIVE
        if not active:
            break
        time.sleep(0.01)
    assert not active


def test_review_substrate_preserves_explicit_zero_budget_rails(tmp_path):
    from ouroboros.usage_accounting import UsageScope, current_usage_scope, usage_scope

    captured = []

    class ScopeCapturingLLM:
        def chat(self, **_kwargs):
            captured.append(current_usage_scope())
            return {
                "content": json.dumps({"verdict": "PASS", "findings": [], "summary": "ok"}),
            }, {}

    with usage_scope(UsageScope(
        drive_root=tmp_path,
        task_id="zero-rail",
        root_task_id="zero-rail",
        global_limit_usd=0.0,
        root_limit_usd=0.0,
    )):
        result = run_review_request(
            ReviewRequest(surface="task_acceptance", goal="review", task_id="zero-rail"),
            slots=[ReviewSlot(slot_id="slot", model="test/model")],
            drive_root=tmp_path,
            llm=ScopeCapturingLLM(),
        )

    assert result.aggregate_signal == "PASS"
    assert len(captured) == 1
    assert captured[0].global_limit_usd == 0.0
    assert captured[0].root_limit_usd == 0.0


def test_triad_actor_records_preserve_review_refs():
    parsed = parse_model_review_results({
        "results": [{
            "model": "m1",
            "text": "[{\"item\":\"x\",\"verdict\":\"PASS\",\"severity\":\"advisory\",\"reason\":\"ok\"}]",
            "prompt_ref": {"manifest_ref": {"path": "prompt.json"}},
            "response_ref": {"manifest_ref": {"path": "response.json"}},
        }]
    })

    actor = parsed.actor_records[0].to_dict()
    assert actor["prompt_ref"]["manifest_ref"]["path"] == "prompt.json"
    assert actor["response_ref"]["manifest_ref"]["path"] == "response.json"


def test_scope_review_result_preserves_substrate_refs(tmp_path, monkeypatch):
    from ouroboros.tools import scope_review
    from ouroboros.tools.review_helpers import build_scope_actor_record

    class FakeScopeLLM:
        def chat(self, **kwargs):
            rows = [
                {
                    "item": item,
                    "verdict": "PASS",
                    "severity": "advisory",
                    "reason": "Fixture confirms scope substrate refs.",
                }
                for item in sorted(scope_review._SCOPE_REQUIRED_ITEMS)
            ]
            return {"content": json.dumps(rows)}, {"prompt_tokens": 10, "completion_tokens": 5}

    ctx = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, task_id="scope-task", pending_events=[])
    monkeypatch.setattr(scope_review, "LLMClient", lambda: FakeScopeLLM())
    monkeypatch.setattr(scope_review, "_build_scope_prompt", lambda *a, **k: ("scope prompt", None))
    monkeypatch.setattr(scope_review, "_get_scope_model", lambda: "test-scope-model")
    # This test isolates durable substrate refs, not the separate P3 authority
    # floor; give its synthetic reviewer explicit >=1M capability evidence.
    monkeypatch.setattr(scope_review, "_scope_window",
                        lambda _model, **_k: scope_review.ReviewerWindow(1_000_000, "confirmed"))

    result = scope_review.run_scope_review(ctx, "commit message")
    record = build_scope_actor_record(result, fallback_model_id="test-scope-model", slot_id="scope_slot_1")

    assert result.status == "responded"
    assert record["prompt_ref"]["manifest_ref"]["path"]
    assert record["response_ref"]["manifest_ref"]["path"]
