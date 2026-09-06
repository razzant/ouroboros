"""Plan-review ENGINE contract tests — quorum reachability under blocking.

Split from ``tests/test_plan_review_engine.py`` at the module-size gate (the
engine module crossed 1600 lines); the shared harness and helpers are re-used
from there so both halves drive the same fake substrate.
"""
from __future__ import annotations

from tests import test_plan_review_engine as _engine

CLEAN = _engine.CLEAN
_DEAD_PANEL = _engine._DEAD_PANEL
_call = _engine._call
_control = _engine._control
_patch_health = _engine._patch_health
_state = _engine._state
harness = _engine.harness


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
    assert "blocked_with_evidence" in disclosure and "structurally unreachable" in disclosure
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
