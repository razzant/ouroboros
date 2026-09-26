"""Plan-review ENGINE contract tests — quorum reachability under blocking.

Split from ``tests/test_plan_review_engine.py`` at the module-size gate (the
engine module crossed 1600 lines); the shared harness and helpers are re-used
from there so both halves drive the same fake substrate.
"""
from __future__ import annotations

import json

import pytest

from tests import test_plan_review_engine as _engine

CLEAN = _engine.CLEAN
_DEAD_PANEL = _engine._DEAD_PANEL
_call = _engine._call
_control = _engine._control
_patch_health = _engine._patch_health
_state = _engine._state
harness = _engine.harness


@pytest.mark.parametrize("enforcement", ["blocking", "advisory"])
def test_note_only_wave_releases_real_finalization_without_paperwork(harness, enforcement):
    from ouroboros.owner_hurry import force_plan_decision
    from ouroboros.task_results import plan_review_gate_projection
    from ouroboros.tools.plan_review_artifacts import read_wave

    harness.state["enforcement"] = enforcement
    sub = harness.install({"s1": json.dumps([
        _engine._finding("n", "note", summary="A simpler alternative may fit the goal."),
    ])})
    ctx = harness.make_ctx(force_plan=True)
    out = _call(ctx)
    assert _control(out) == {"outcome": "GREEN", "closed": True}  # notes never change the verdict
    state = _state(harness)
    wave = state["waves"][-1]
    assert wave["aggregate"] == "GREEN" and wave["dispositions"] == [] and state["cycles_paid"] == 1
    exact = read_wave(harness.drive, "task-1", wave["wave_artifact"])
    assert exact["findings"][0]["summary"] == "A simpler alternative may fit the goal."
    assert plan_review_gate_projection(state, enforcement)["allow"]
    assert force_plan_decision(ctx, {}, enforcement=enforcement)["allow"]
    assert _control(_call(ctx)) == {"outcome": "GREEN", "closed": True}
    assert len(sub.calls) == 1


@pytest.mark.parametrize("enforcement", ["advisory", "blocking"])
def test_a_reasoned_reject_closes_a_below_quorum_blocking_finding_only_under_advisory(harness, enforcement):
    """The open set, through the engine: one blocking slot below quorum holds the wave
    REVIEW_REQUIRED; a reject with its rationale closes it under advisory (recorded GREEN
    in the hot index and the exact artifact, no reviewer call, no cycle) and the gate
    allows; under blocking the same $0 call leaves it open and the gate still holds."""
    from ouroboros.task_results import plan_review_gate_projection
    from ouroboros.tools import plan_review as pr
    from ouroboros.tools.plan_review_artifacts import read_wave

    harness.state["enforcement"] = enforcement
    sub = harness.install({"s1": json.dumps([_engine._finding("b1", "blocking", breaks="claim_1")]),
                           "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    assert _control(_call(ctx)) == {"outcome": "REVIEW_REQUIRED", "closed": False}
    fp = _state(harness)["waves"][-1]["request_fingerprint"]
    out = pr._handle_plan_task(ctx, review_disposition={"review_fingerprint": fp, "items": [
        {"finding_id": "s1:b1", "decision": "reject", "rationale": "the claim is checked by the demo"}]})
    advisory = enforcement == "advisory"
    assert _control(out) == ({"outcome": "GREEN", "closed": True} if advisory
                             else {"outcome": "REVIEW_REQUIRED", "closed": False})
    state = _state(harness)
    wave = state["waves"][-1]
    exact = read_wave(harness.drive, "task-1", wave["wave_artifact"])
    assert (wave["aggregate"], wave["closed"]) == (exact["aggregate"], exact["closed"])
    assert wave["aggregate"] == ("GREEN" if advisory else "REVIEW_REQUIRED")
    assert ("closed_by_disposition" in " ".join(wave["closure_notes"])) is advisory
    assert ("blocking_finding_below_quorum_stays_open" in out) is not advisory
    assert wave["findings"][0]["class"] == "blocking" and wave["dispositions"][0]["decision"] == "reject"
    assert state["cycles_paid"] == 1 and len(sub.calls) == 1
    assert plan_review_gate_projection(state, enforcement)["allow"] is advisory
    assert plan_review_gate_projection(state, enforcement)["closed"] is advisory


def test_quorum_unreachable_releases_finalization_for_a_blocked_terminal(harness, monkeypatch):
    """B2b blocked finalization: with the quorum structurally unreachable under
    blocking, the gate RELEASES finalization (agent's choice, never auto), the
    review stays OPEN, implementation stays held, and outcomes terminalizes the
    finalized task as blocked_with_evidence with the typed quorum reason."""
    _patch_health(monkeypatch, lambda slots: dict(_DEAD_PANEL))
    harness.install({"s1": CLEAN})
    ctx = harness.make_ctx()
    out = _call(ctx)
    assert "implementation still held" in out
    from ouroboros.owner_hurry import force_plan_decision, plan_review_disclosure
    from ouroboros.task_results import plan_review_gate_projection

    state = _state(harness)
    gate = plan_review_gate_projection(state, "blocking")
    assert gate["status"] == "open" and gate["allow"] is True and gate["closed"] is False
    assert gate["quorum_unreachable"] is True
    assert gate["earliest_reset"] == "2030-01-01T00:00:00+00:00"
    # advisory is untouched: it already proceeded under loud disclosure
    advisory = plan_review_gate_projection(state, "advisory")
    assert advisory["status"] == "advisory_open" and advisory["allow"] is True
    decision = force_plan_decision(ctx, {}, enforcement="blocking")
    assert decision["allow"] is True and decision["quorum_unreachable"] is True
    disclosure = plan_review_disclosure(decision)
    # Owner-readable prose says the ending in words; the ledger identifier stays
    # on the typed objective axis asserted below.
    assert "the task ends blocked with its evidence recorded" in disclosure
    assert "blocked_with_evidence" not in disclosure
    assert "structurally unreachable" in disclosure
    assert "2030-01-01T00:00:00+00:00" in disclosure
    from ouroboros.outcomes import derive_loop_outcome

    outcome = derive_loop_outcome("done", {}, {"force_plan_decision": decision, "tool_calls": []})
    objective = outcome["outcome_axes"]["objective"]
    assert objective["status"] == "fail"
    assert objective["outcome_tier"] == "blocked_with_evidence"
    assert objective["reason"] == "plan_review_quorum_unreachable"
    # the review itself is NOT closed by the release
    assert _state(harness)["waves"][-1]["closed"] is False


def test_reachable_quorum_with_one_dead_slot_still_holds_under_blocking(harness, monkeypatch):
    """One dead slot of three leaves the quorum reachable: no release, the open
    DEGRADED wave holds finalization under blocking exactly as before."""
    _patch_health(monkeypatch, lambda slots: {
        "s3": {"failure_code": "subscription_window_exhausted",
               "reset_at": "2030-01-01T00:00:00+00:00"}})
    prose = "prose only, no findings array"
    harness.install({"s1": prose, "s2": prose})
    ctx = harness.make_ctx()
    out = _call(ctx)
    assert _control(out) == {"outcome": "DEGRADED", "closed": False}
    wave = _state(harness)["waves"][-1]
    assert "quorum_unreachable" not in wave
    from ouroboros.task_results import plan_review_gate_projection

    gate = plan_review_gate_projection(_state(harness), "blocking")
    assert gate["allow"] is False and gate["status"] == "open"
    assert gate["quorum_unreachable"] is False
