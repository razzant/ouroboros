"""B2b panel health: what the engine may claim about a lane it could not reach.

Split by theme out of ``tests/test_plan_review_engine.py`` at that module's size
ceiling. This module owns the pre-fan-out health snapshot and everything decided
from it: zero-cost skip rows that stay in the quorum denominator, the structural
vs transient distinction (a daemon that died at dispatch time is UNKNOWN, never
structural), epoch-scoped free replay, and the structurally-unreachable quorum
that releases finalization for a blocked terminal. The engine harness and its
helpers are imported from the shared sibling, so both files drive the identical
fake substrate.
"""
from __future__ import annotations


from ouroboros.tools import plan_review as pr

from tests._plan_review_engine_shared import harness as __harness

# The fixture is requested by name as a test parameter, so it is re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an
# F811 redefinition under the CI ruff gate.
harness = __harness

from tests._plan_review_engine_shared import (
    CLEAN,
    _call,
    _control,
    _slots,
    _state,
)

# ------------------------------------------------------------- B2b: panel health


_DEAD_PANEL = {
    "s2": {"failure_code": "subscription_window_exhausted",
           "reset_at": "2030-01-02T00:00:00+00:00"},
    "s3": {"failure_code": "credential_pool_exhausted",
           "reset_at": "2030-01-01T00:00:00+00:00"},
}


def _patch_health(monkeypatch, snapshot_fn):
    """Patch BOTH snapshot callers: the engine's fan-out seam and the replay seam."""
    import ouroboros.tools.plan_review_runtime as prr

    monkeypatch.setattr(pr, "_plan_panel_health_snapshot", snapshot_fn)
    monkeypatch.setattr(prr, "plan_panel_health_snapshot", snapshot_fn)


def test_health_skip_rows_are_zero_cost_and_stay_in_the_quorum_denominator(harness, monkeypatch):
    """B2b: positive structural evidence turns slots into $0 typed skip rows BEFORE
    dispatch; the quorum denominator never shrinks (BIBLE P3); live slots still
    dispatch even though the dead ones make the quorum unreachable."""
    _patch_health(monkeypatch, lambda slots: dict(_DEAD_PANEL))
    sub = harness.install({"s1": CLEAN})
    out = _call(harness.make_ctx())
    assert _control(out) == {"outcome": "DEGRADED", "closed": False}
    assert [s.slot_id for s in sub.calls[0]["slots"]] == ["s1"]  # only the live slot dispatched
    wave = _state(harness)["waves"][-1]
    rec = {a["slot_id"]: a for a in wave["actors"]}
    assert len(wave["actors"]) == 3  # skip rows stay configured rows
    assert wave["counts"]["configured"] == 3 and wave["counts"]["quorum"] == 2
    assert rec["s2"]["cost"] == 0.0 and rec["s2"]["tokens_in"] == 0
    assert rec["s2"]["failure_code"] == "subscription_window_exhausted"
    assert rec["s2"]["reset_at"] == "2030-01-02T00:00:00+00:00"
    assert rec["s3"]["failure_code"] == "credential_pool_exhausted"
    assert wave["paid"] is True  # s1 was physically dispatched
    # render carries the typed skip and the structural facts
    assert "health_skip[subscription_window_exhausted]" in out
    assert "STRUCTURALLY unreachable" in out and "schedule_followup" in out
    # the wave's own typed rows prove the quorum unreachable: 3 - 2 dead = 1 < 2
    assert wave["quorum_unreachable"] is True
    assert sorted(wave["structurally_dead_slots"]) == ["s2", "s3"]
    assert wave["earliest_reset"] == "2030-01-01T00:00:00+00:00"
    assert wave["health_epoch"] == [
        {"slot": "s2", "code": "subscription_window_exhausted",
         "reset_at": "2030-01-02T00:00:00+00:00"},
        {"slot": "s3", "code": "credential_pool_exhausted",
         "reset_at": "2030-01-01T00:00:00+00:00"},
    ]


def test_unknown_panel_health_dispatches_every_slot(harness, monkeypatch):
    """A failed snapshot (None) is unknown, not structural: every slot dispatches."""
    _patch_health(monkeypatch, lambda slots: None)
    sub = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    out = _call(harness.make_ctx())
    assert _control(out) == {"outcome": "GREEN", "closed": True}
    assert [s.slot_id for s in sub.calls[0]["slots"]] == ["s1", "s2", "s3"]
    wave = _state(harness)["waves"][-1]
    assert wave["health_epoch"] == [] and "quorum_unreachable" not in wave


def test_structural_skip_predicate_requires_positive_evidence():
    """Unknown/undated/stale/transient states DISPATCH; only a dated future window
    exhaustion or a typed dead-pool code is skip evidence (roast pts 4/9)."""
    from ouroboros.tools.plan_review_runtime import _structural_skip_code

    assert _structural_skip_code("", "2030-01-01T00:00:00Z") == "subscription_window_exhausted"
    assert _structural_skip_code("subscription_window_exhausted", "") == ""  # undated
    assert _structural_skip_code("credential_pool_exhausted", "") == "credential_pool_exhausted"
    assert _structural_skip_code("", "2001-01-01T00:00:00Z") == ""  # stale reset
    assert _structural_skip_code("", "not-a-time") == "" and _structural_skip_code("daemon_recovery_only", "") == ""
    assert _structural_skip_code("route_status_disabled", "") == ""  # not window evidence


def test_snapshot_transient_daemon_death_reads_unknown_never_structural(monkeypatch):
    """A ClaudexorUnavailable during the snapshot (daemon_recovery_only, dead socket)
    yields None (unknown, fail-open) — never skip rows, never an epoch entry."""
    import ouroboros.claudexor_daemon as cd
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.tools.plan_review_runtime import plan_panel_health_snapshot

    monkeypatch.setattr(cd, "owned_daemon_provisioned", lambda: True)

    def _dying():
        raise ClaudexorUnavailable("daemon_recovery_only", "daemon is serving recovery only")

    monkeypatch.setattr(cd, "ensure_owned_gateway", _dying)
    assert plan_panel_health_snapshot(_slots(("s1", "m/a"), ("s2", "m/b", "session"))) is None
    # An api_chat-only panel has no route health source: the snapshot trivially ran.
    assert plan_panel_health_snapshot(_slots(("s1", "m/a"))) == {}


def test_epoch_replay_free_while_unchanged_transient_keeps_it_healed_repays(harness, monkeypatch):
    """B2b epoch: an identical envelope replays the recorded open wave free while a
    fresh snapshot matches the recorded epoch; a FAILED snapshot (transient) keeps
    the free replay; a healed lane re-dispatches a NEW paid panel."""
    health = {"evidence": dict(_DEAD_PANEL)}
    _patch_health(monkeypatch, lambda slots: (
        dict(health["evidence"]) if health["evidence"] is not None else None))
    sub = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    first = _call(ctx)
    assert _control(first) == {"outcome": "DEGRADED", "closed": False}
    assert len(sub.calls) == 1 and _state(harness)["cycles_paid"] == 1
    # identical envelope + identical epoch = free replay, zero substrate calls
    second = _call(ctx)
    assert "cached exact review" in second and len(sub.calls) == 1
    assert _state(harness)["cycles_paid"] == 1
    # transient snapshot failure does not change the epoch: still a free replay
    health["evidence"] = None
    third = _call(ctx)
    assert "cached exact review" in third and len(sub.calls) == 1
    # the lanes healed: the epoch moved, the same envelope buys a fresh paid panel
    health["evidence"] = {}
    fourth = _call(ctx)
    assert _control(fourth) == {"outcome": "GREEN", "closed": True}
    assert len(sub.calls) == 2
    assert [s.slot_id for s in sub.calls[1]["slots"]] == ["s1", "s2", "s3"]
    assert _state(harness)["cycles_paid"] == 2


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
