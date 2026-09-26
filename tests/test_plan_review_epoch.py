"""Replay-seam authority rules of the plan-review engine (review-fix batch over
B2b): the DEGRADED-only epoch gate, the empty-epoch no-cache rule, the reviewer
roster identity (effort included), the unpaid quorum-unreachable discovery
landing on the durable record, the advisory-open event's dedup + durability
ordering (memo only after the durable append landed), and the replay decision's
loud fail-open on configuration-resolution failure.

Shares the engine harness with ``test_plan_review_engine`` (real ``ToolContext``,
real ``plan_spec``/``task_results`` v2 code, one fake review substrate). Lives in
its own module because the engine test file sits at its 1500-line ceiling.
"""

from __future__ import annotations

import pytest

import dataclasses
import json
import logging

from tests.test_plan_review_engine import (
    CLEAN, _DEAD_PANEL, _call, _control, _finding, _patch_health, _slots, _state,
)
from tests.test_plan_review_engine import harness as _engine_harness  # shared fixture

harness = _engine_harness  # noqa: F811 — pytest registers the fixture here too


def test_open_review_required_wave_replays_free_even_when_the_epoch_moved(harness, monkeypatch):
    """Review fix 1: the health-epoch re-dispatch check binds ONLY a DEGRADED wave.
    Every other open aggregate replays exactly as before B2b — an epoch change must
    not buy a fresh panel for a wave that already holds real findings."""
    snapshots = {"count": 0, "evidence": {"s3": dict(_DEAD_PANEL["s3"])}}

    def _snap(slots):
        snapshots["count"] += 1
        return dict(snapshots["evidence"])

    _patch_health(monkeypatch, _snap)
    note = json.dumps([_finding("n1", "blocking", breaks="claim_1")])
    sub = harness.install({"s1": note, "s2": CLEAN})
    ctx = harness.make_ctx()
    first = _call(ctx)
    assert _control(first) == {"outcome": "REVIEW_REQUIRED", "closed": False}
    assert len(sub.calls) == 1 and snapshots["count"] == 1
    wave_before = _state(harness)["waves"][-1]
    assert wave_before["health_epoch"] and wave_before["aggregate"] == "REVIEW_REQUIRED"
    # The lane heals: the epoch MOVES — and the identical envelope still replays
    # free: zero substrate calls, zero snapshots, cycles unchanged, wave untouched.
    snapshots["evidence"] = {}
    second = _call(ctx)
    assert "cached exact review" in second
    assert len(sub.calls) == 1 and snapshots["count"] == 1
    assert _state(harness)["cycles_paid"] == 1
    wave_after = _state(harness)["waves"][-1]
    assert wave_after["reviewed_at"] == wave_before["reviewed_at"]  # NOT replaced
    assert "degraded_retries" not in wave_after


def test_degraded_wave_with_matching_epoch_replays_and_a_changed_roster_rediscovers(harness, monkeypatch):
    """Review fix 2: DEGRADED + matching non-empty epoch still replays free; a
    changed reviewer roster (slot target moved) lapses the replay authority and
    re-dispatches a fresh paid panel even under an identical envelope + epoch."""
    _patch_health(monkeypatch, lambda slots: dict(_DEAD_PANEL))
    sub = harness.install({"s1": CLEAN})
    ctx = harness.make_ctx()
    first = _call(ctx)
    assert _control(first) == {"outcome": "DEGRADED", "closed": False}
    assert len(sub.calls) == 1 and _state(harness)["cycles_paid"] == 1
    assert _state(harness)["waves"][-1]["reviewer_config_fingerprint"]
    # Identical envelope + identical epoch + identical roster = free replay.
    second = _call(ctx)
    assert "cached exact review" in second and len(sub.calls) == 1
    # The owner re-points slot s3 at a different target: the roster identity moved,
    # so the SAME envelope re-dispatches a fresh panel (paid).
    harness.state["slots"] = _slots(("s1", "m/a"), ("s2", "m/b"), ("s3", "m/other"))
    third = _call(ctx)
    assert "cached exact review" not in third
    assert len(sub.calls) == 2
    assert _state(harness)["cycles_paid"] == 2


def test_unpaid_all_skip_discovery_lands_the_quorum_fact_on_the_durable_wave(harness, monkeypatch):
    """Review fix 3: an unpaid all-skip attempt that DISCOVERS structural quorum
    unreachability stamps the typed fact onto the paid predecessor D2 preserves —
    so the rendered release and the gate projection AGREE (the old contradiction:
    render said finalization released while the gate still refused)."""
    health = {"evidence": {}}
    _patch_health(monkeypatch, lambda slots: dict(health["evidence"]))
    prose = "prose only, no findings array"
    harness.install({"s1": prose, "s2": prose, "s3": prose})
    ctx = harness.make_ctx()
    first = _call(ctx)
    assert _control(first) == {"outcome": "DEGRADED", "closed": False}
    state = _state(harness)
    assert state["cycles_paid"] == 1 and state["waves"][-1]["paid"] is True
    assert "quorum_unreachable" not in state["waves"][-1]
    from ouroboros.task_results import plan_review_gate_projection

    assert plan_review_gate_projection(state, "blocking")["allow"] is False
    # Every lane is now window-spent: the identical envelope re-dispatches (empty
    # epoch) into an ALL-SKIP unpaid wave that discovers the quorum unreachable.
    health["evidence"] = {
        "s1": {"failure_code": "subscription_window_exhausted",
               "reset_at": "2030-01-03T00:00:00+00:00"},
        **{k: dict(v) for k, v in _DEAD_PANEL.items()},
    }
    second = _call(ctx)
    assert "STRUCTURALLY unreachable" in second
    assert "finalization is RELEASED" in second and "blocked_with_evidence" in second
    state = _state(harness)
    wave = state["waves"][-1]
    # The durable record is still the PAID predecessor — now carrying the fact.
    assert wave["paid"] is True and wave["degraded_retries"] == 1
    assert wave["quorum_unreachable"] is True
    assert sorted(wave["structurally_dead_slots"]) == ["s1", "s2", "s3"]
    assert wave["earliest_reset"] == "2030-01-01T00:00:00+00:00"
    assert state["cycles_paid"] == 1, "an all-skip attempt stays $0"
    gate = plan_review_gate_projection(state, "blocking")
    assert gate["allow"] is True and gate["status"] == "open"
    assert gate["quorum_unreachable"] is True
    assert gate["earliest_reset"] == "2030-01-01T00:00:00+00:00"


def test_three_identical_recalls_emit_one_advisory_open_event(harness, monkeypatch):
    """Review fix 3 (dedup): re-dispatches of an identical envelope under an
    unchanged (fingerprint, epoch) state re-enter the emitter but announce ONCE."""
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "6")
    harness.state["enforcement"] = "advisory"
    _patch_health(monkeypatch, lambda slots: {})
    sub = harness.install({"s1": "", "s2": "", "s3": ""})  # every slot: transport death
    ctx = harness.make_ctx()
    while not harness.events.empty():
        harness.events.get_nowait()
    for _ in range(3):  # empty-epoch DEGRADED: each identical call re-dispatches
        out = _call(ctx)
        assert _control(out) == {"outcome": "DEGRADED", "closed": False}
    assert len(sub.calls) == 3 and _state(harness)["cycles_paid"] == 3
    events = []
    while not harness.events.empty():
        events.append(harness.events.get_nowait())
    typed = [e for e in events if e.get("type") == "log_event"
             and e.get("data", {}).get("type") == "plan_review_advisory_open"]
    assert len(typed) == 1, "one event per (fingerprint, epoch) state, not per call"


def test_advisory_open_event_is_durable_even_with_a_live_queue(harness):
    """Review fix 4: the durable events.jsonl append ALWAYS lands (the live queue
    path persists only task_checkpoint rows); the queue additionally gets the push;
    a second call for the same (fingerprint, epoch) state is deduplicated."""
    from ouroboros.tools.plan_review_runtime import emit_plan_review_advisory_open

    ctx = harness.make_ctx()
    wave = {"request_fingerprint": "e" * 64, "aggregate": "DEGRADED", "cycle_index": 1,
            "paid": True, "health_epoch": [], "actors": []}
    emit_plan_review_advisory_open(ctx, harness.drive, task_id="task-1", wave=wave,
                                   cycles_paid=1, cap=2)
    rows = [json.loads(line) for line in
            (harness.drive / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    mine = [r for r in rows if r.get("type") == "plan_review_advisory_open"
            and r.get("fingerprint") == "e" * 64]
    assert len(mine) == 1 and mine[0]["enforcement"] == "advisory"
    pushed = []
    while not harness.events.empty():
        pushed.append(harness.events.get_nowait())
    assert [e for e in pushed if e.get("data", {}).get("type") == "plan_review_advisory_open"]
    # Same state again: deduplicated — no second durable row, no second push.
    emit_plan_review_advisory_open(ctx, harness.drive, task_id="task-1", wave=wave,
                                   cycles_paid=1, cap=2)
    rows = [json.loads(line) for line in
            (harness.drive / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len([r for r in rows if r.get("fingerprint") == "e" * 64]) == 1
    assert harness.events.empty()


def test_effort_only_roster_change_lapses_replay_authority(harness, monkeypatch):
    """Review fix 3: slot EFFORT is roster identity — it changes what the reviewer
    actually does. Same ids/targets/routes with only the effort moved must
    re-dispatch a fresh paid panel instead of inheriting the free replay."""
    _patch_health(monkeypatch, lambda slots: dict(_DEAD_PANEL))
    sub = harness.install({"s1": CLEAN})
    ctx = harness.make_ctx()
    first = _call(ctx)
    assert _control(first) == {"outcome": "DEGRADED", "closed": False}
    assert len(sub.calls) == 1 and _state(harness)["cycles_paid"] == 1
    # Identical envelope + epoch + roster: free replay.
    second = _call(ctx)
    assert "cached exact review" in second and len(sub.calls) == 1
    # ONLY the effort changes (high -> low) on the same slots: identity moved.
    harness.state["slots"] = [
        dataclasses.replace(slot, effort="low") for slot in harness.state["slots"]]
    third = _call(ctx)
    assert "cached exact review" not in third
    assert len(sub.calls) == 2
    assert _state(harness)["cycles_paid"] == 2


def test_degraded_reminder_promises_free_replay_only_with_structural_epoch(harness, monkeypatch):
    """Round-3: the user-turn DEGRADED reminder mirrors plan_render's
    _degraded_replay_note (the wording SSOT) — a wave WITH a recorded structural
    epoch is promised the free replay with its conditions (unchanged epoch +
    roster); an EMPTY-epoch wave re-dispatches a paid panel, so the old
    unconditional "replays ... at no cost" promise must not appear."""
    from ouroboros.owner_hurry import force_plan_decision, plan_review_reminder

    # Empty epoch: every slot dies at dispatch time, invisible to the snapshot.
    _patch_health(monkeypatch, lambda slots: {})
    harness.install({"s1": "", "s2": "", "s3": ""})
    ctx = harness.make_ctx()
    out = _call(ctx)
    assert _control(out) == {"outcome": "DEGRADED", "closed": False}
    decision = force_plan_decision(ctx, {}, enforcement="blocking")
    assert decision["reviewer_slots_degraded"] and decision["degraded_health_epoch"] == ""
    reminder = plan_review_reminder(decision)
    assert "re-dispatches a fresh panel" in reminder
    assert "no cost" not in reminder and "no further cost" not in reminder
    # Non-empty epoch: structural snapshot evidence recorded — the free replay is
    # promised together with its conditions (epoch + roster stand).
    _patch_health(monkeypatch, lambda slots: dict(_DEAD_PANEL))
    harness.install({"s1": CLEAN})
    ctx2 = harness.make_ctx(task_id="task-epoch")
    out2 = _call(ctx2)
    assert _control(out2) == {"outcome": "DEGRADED", "closed": False}
    decision2 = force_plan_decision(ctx2, {}, enforcement="blocking")
    assert decision2["reviewer_slots_degraded"] and decision2["degraded_health_epoch"]
    reminder2 = plan_review_reminder(decision2)
    assert "at no further cost" in reminder2
    assert "reviewer roster stand" in reminder2
    assert "re-dispatches a fresh panel" not in reminder2


def test_replay_decision_config_failure_keeps_replay_but_logs_loudly(caplog):
    """Review fix 4 (accepted-partial): a configuration-resolution failure keeps the
    recorded free replay (fail-open) but is logged as a WARNING with the exception
    detail — never a silent except."""
    from ouroboros.tools.plan_review_runtime import PLAN_NO_SNAPSHOT, plan_wave_replay_decision

    def _exploding_slots():
        raise RuntimeError("reviewer slot config exploded")

    existing = {"aggregate": "DEGRADED",
                "reviewer_config_fingerprint": "a" * 64,
                "health_epoch": [{"slot": "s1", "code": "subscription_window_exhausted",
                                  "reset_at": "2030-01-01T00:00:00+00:00"}]}
    with caplog.at_level(logging.WARNING, logger="ouroboros.tools.plan_review_runtime"):
        stale, snapshot = plan_wave_replay_decision(_exploding_slots, existing)
    assert stale is False and snapshot is PLAN_NO_SNAPSHOT
    warned = [r for r in caplog.records
              if r.levelno == logging.WARNING and "replay" in r.getMessage()]
    assert warned, "the config-resolution failure must warn loudly"
    assert warned[0].exc_info and "reviewer slot config exploded" in str(warned[0].exc_info[1])


def test_failed_durable_append_is_not_memoized_and_retries(harness, monkeypatch):
    """Review fix 6: the dedup memo is inserted ONLY after the durable append
    succeeded. A failed append is not memoized (and pushes nothing), so the next
    call for the same state retries and lands the event."""
    import ouroboros.utils as utils
    from ouroboros.tools.plan_review_runtime import emit_plan_review_advisory_open

    ctx = harness.make_ctx()
    wave = {"request_fingerprint": "f" * 64, "aggregate": "DEGRADED", "cycle_index": 1,
            "paid": True, "health_epoch": [], "actors": []}
    real_append = utils.append_jsonl

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(utils, "append_jsonl", _boom)
    emit_plan_review_advisory_open(ctx, harness.drive, task_id="task-append", wave=wave,
                                   cycles_paid=1, cap=2)
    events_path = harness.drive / "logs" / "events.jsonl"
    assert not events_path.exists() or "f" * 64 not in events_path.read_text(encoding="utf-8")
    assert harness.events.empty(), "no UI push for an event that never landed durably"
    # The append heals: the SAME state retries and lands exactly once.
    monkeypatch.setattr(utils, "append_jsonl", real_append)
    emit_plan_review_advisory_open(ctx, harness.drive, task_id="task-append", wave=wave,
                                   cycles_paid=1, cap=2)
    rows = [json.loads(line) for line in
            events_path.read_text(encoding="utf-8").splitlines()]
    assert len([r for r in rows if r.get("fingerprint") == "f" * 64]) == 1
    pushed = []
    while not harness.events.empty():
        pushed.append(harness.events.get_nowait())
    assert [e for e in pushed if e.get("data", {}).get("type") == "plan_review_advisory_open"]
    # And now memoized: a third call is a no-op.
    emit_plan_review_advisory_open(ctx, harness.drive, task_id="task-append", wave=wave,
                                   cycles_paid=1, cap=2)
    rows = [json.loads(line) for line in
            events_path.read_text(encoding="utf-8").splitlines()]
    assert len([r for r in rows if r.get("fingerprint") == "f" * 64]) == 1


def _advisory_open_rows(harness, fingerprint):
    path = harness.drive / "logs" / "events.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return [row for row in map(json.loads, lines)
            if row.get("type") == "plan_review_advisory_open" and row.get("fingerprint") == fingerprint]


def test_the_settled_outcome_is_announced_after_the_dispatch_snapshot(harness):
    """The memo keys on the OUTCOME the event announces, not on the wave alone: the
    dispatch snapshot (nothing answered yet) used to mute the settled failures of the
    same fingerprint and epoch forever. Cycle counters stay out of the key, so a
    re-dispatch that ends in the same outcome is still announced once."""
    from ouroboros.tools.plan_review_runtime import emit_plan_review_advisory_open

    words = "Selected model is at capacity. Please try a different model."
    fingerprint = "c" * 64

    def _emit(actors, *, pending, cycle_index=1, cycles_paid=1):
        emit_plan_review_advisory_open(
            ctx, harness.drive, task_id="task-outcome", cycles_paid=cycles_paid, cap=3,
            wave={"request_fingerprint": fingerprint, "aggregate": "DEGRADED", "health_epoch": [],
                  "custody_pending": pending, "paid": not pending, "cycle_index": cycle_index,
                  "actors": actors})

    def _dead(slot_id, cause=words):
        return {"slot_id": slot_id, "ok": False, "failure_code": "run_failed",
                "operation_state": "late_settled", "reported_cause": cause}

    ctx = harness.make_ctx()
    waiting = [{"slot_id": s, "ok": False, "operation_state": "pending_dispatch"} for s in ("s1", "s2", "s3")]
    settled = [_dead("s1"), {"slot_id": "s2", "ok": True}, _dead("s3")]
    _emit(waiting, pending=True, cycles_paid=0)
    assert [[s["failure_code"] for s in row["slots"]] for row in _advisory_open_rows(harness, fingerprint)] == [["", "", ""]]
    _emit(settled, pending=False)
    rows = _advisory_open_rows(harness, fingerprint)
    assert len(rows) == 2, "the settled outcome must not be muted by the dispatch snapshot"
    assert (rows[1]["custody_pending"], rows[1]["paid"]) == (False, True)
    assert [(s["slot_id"], s["ok"], s["failure_code"], s["reported_cause"]) for s in rows[1]["slots"]] == [
        ("s1", False, "run_failed", words), ("s2", True, "", ""), ("s3", False, "run_failed", words)]
    # Other direction — the SAME outcome is never re-announced: not on an identical call,
    # not on the identical snapshot, not when only the cycle counters moved.
    _emit(settled, pending=False)
    _emit(waiting, pending=True, cycles_paid=0)
    _emit(settled, pending=False, cycle_index=2, cycles_paid=2)
    assert len(_advisory_open_rows(harness, fingerprint)) == 2
    # A panel that dies of a DIFFERENT reported cause is a different outcome.
    _emit([_dead("s1", "usage limit reached"), {"slot_id": "s2", "ok": True}, _dead("s3")], pending=False)
    assert len(_advisory_open_rows(harness, fingerprint)) == 3


def test_an_append_that_reports_failure_is_not_memoized_and_the_next_call_emits(harness, monkeypatch, caplog):
    """``append_jsonl`` reports a failed write by RETURNING False, without raising.
    That is a lost event exactly like a raised one: logged, nothing pushed, nothing
    memoized — so the next call for the same outcome lands it. A successful append
    is memoized and the following call stays quiet."""
    import ouroboros.utils as utils
    from ouroboros.tools.plan_review_runtime import emit_plan_review_advisory_open

    ctx = harness.make_ctx()
    fingerprint = "d" * 64
    wave = {"request_fingerprint": fingerprint, "aggregate": "DEGRADED", "cycle_index": 1,
            "paid": True, "health_epoch": [], "actors": []}
    real_append = utils.append_jsonl
    monkeypatch.setattr(utils, "append_jsonl", lambda *args, **kwargs: False)
    with caplog.at_level(logging.WARNING, logger="ouroboros.tools.plan_review_runtime"):
        emit_plan_review_advisory_open(ctx, harness.drive, task_id="task-false", wave=wave,
                                       cycles_paid=1, cap=2)
    assert [r for r in caplog.records if "durable append failed" in r.getMessage()]
    assert _advisory_open_rows(harness, fingerprint) == []
    assert harness.events.empty(), "no UI push for an event that never landed durably"
    monkeypatch.setattr(utils, "append_jsonl", real_append)
    for _ in range(2):  # the retry lands the event; the call after it is memoized
        emit_plan_review_advisory_open(ctx, harness.drive, task_id="task-false", wave=wave,
                                       cycles_paid=1, cap=2)
    assert len(_advisory_open_rows(harness, fingerprint)) == 1
    pushed = []
    while not harness.events.empty():
        pushed.append(harness.events.get_nowait())
    assert len([e for e in pushed if e.get("data", {}).get("type") == "plan_review_advisory_open"]) == 1


def test_cached_replay_of_an_open_wave_retries_a_failed_advisory_open_append(harness, monkeypatch):
    """Post-merge follow-up (sol finding 3): the durable advisory-open append that
    FAILED at record time was unreachable forever — the identical envelope's cached
    replay returned before the emitter. The replay path now re-invokes the emitter
    for the still-open wave: zero substrate calls, and the row lands exactly once."""
    import ouroboros.utils as utils

    harness.state["enforcement"] = "advisory"
    _patch_health(monkeypatch, lambda slots: {})
    note = json.dumps([_finding("n1", "blocking", breaks="claim_1")])
    sub = harness.install({"s1": note, "s2": CLEAN, "s3": CLEAN})
    real_append = utils.append_jsonl

    def _fail_advisory(path, row, *args, **kwargs):
        if isinstance(row, dict) and row.get("type") == "plan_review_advisory_open":
            raise OSError("disk full")
        return real_append(path, row, *args, **kwargs)

    monkeypatch.setattr(utils, "append_jsonl", _fail_advisory)
    ctx = harness.make_ctx()
    first = _call(ctx)
    assert _control(first) == {"outcome": "REVIEW_REQUIRED", "closed": False}
    assert len(sub.calls) == 1
    events_path = harness.drive / "logs" / "events.jsonl"
    durable = events_path.read_text(encoding="utf-8") if events_path.exists() else ""
    assert "plan_review_advisory_open" not in durable, "the append failed: no durable row yet"
    # The disk heals: the IDENTICAL envelope replays from cache — zero further
    # substrate calls — and the replay path retries the durable append.
    monkeypatch.setattr(utils, "append_jsonl", real_append)
    second = _call(ctx)
    assert "cached exact review" in second and len(sub.calls) == 1
    rows = [json.loads(line) for line in
            events_path.read_text(encoding="utf-8").splitlines()]
    mine = [r for r in rows if r.get("type") == "plan_review_advisory_open"]
    assert len(mine) == 1 and mine[0]["aggregate"] == "REVIEW_REQUIRED"
    # A third identical call dedups via the memo: still exactly one durable row.
    third = _call(ctx)
    assert "cached exact review" in third and len(sub.calls) == 1
    rows = [json.loads(line) for line in
            events_path.read_text(encoding="utf-8").splitlines()]
    assert len([r for r in rows if r.get("type") == "plan_review_advisory_open"]) == 1


# --------------------------------------------- panel health is PROFILE-scoped


def _profile_slots(*specs):
    """``specs`` = (slot_id, session_target, profile) → agent_session ``ReviewSlot``s."""
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewSlot

    return [
        ReviewSlot(slot_id=sid, model="delegated", effort="high",
                   role_hint="plan reviewer", route=ReviewRouteKind.AGENT_SESSION,
                   session_target=target, session_profile=profile)
        for sid, target, profile in specs
    ]


def _patch_snapshot_health(monkeypatch, answer):
    """Run the REAL snapshot against a fake daemon; ``answer(route_id, model, pin)``
    returns ``(reason, reset_at)``. Records every ask so memo scoping is provable."""
    import ouroboros.claudexor_daemon as cd
    import ouroboros.subagents as sa

    asked: list[tuple[str, str, str]] = []

    class _Gateway:
        def close(self):
            pass

    def _health(gateway, route_id, shape, *, route_model="", pinned_profile=""):
        asked.append((route_id, route_model, pinned_profile))
        return answer(route_id, route_model, pinned_profile)

    monkeypatch.setattr(cd, "owned_daemon_provisioned", lambda: True)
    monkeypatch.setattr(cd, "ensure_owned_gateway", _Gateway)
    monkeypatch.setattr(sa, "route_health", _health)
    return asked


_SPENT = ("subscription_window_exhausted", "2030-01-01T00:00:00+00:00")
_HEALTHY = ("", "")


def test_snapshot_skips_a_slot_whose_pinned_profile_is_spent(monkeypatch):
    """A slot pinning a spent account is skipped even though the route AGGREGATE
    (the unpinned answer a sibling account still vouches for) reads healthy: the
    pin rides into the health read exactly as it does at dispatch."""
    from ouroboros.tools.plan_review_runtime import plan_panel_health_snapshot

    asked = _patch_snapshot_health(
        monkeypatch, lambda rid, model, pin: _SPENT if pin == "spent-acct" else _HEALTHY)
    slots = _profile_slots(("s1", "codex=gpt-5.6-sol", "spent-acct"))
    assert plan_panel_health_snapshot(slots) == {
        "s1": {"failure_code": "subscription_window_exhausted",
               "reset_at": "2030-01-01T00:00:00+00:00"}}
    assert asked == [("codex", "gpt-5.6-sol", "spent-acct")]


def test_same_route_different_profiles_do_not_share_one_health_verdict(monkeypatch):
    """The memo is keyed by the SUBJECT, pin included: two rows on the same
    harness+model but different accounts are asked separately, so exactly the spent
    one is skipped and the healthy one still dispatches."""
    from ouroboros.tools.plan_review_runtime import (
        plan_health_skip_rows, plan_panel_health_snapshot,
    )

    asked = _patch_snapshot_health(
        monkeypatch, lambda rid, model, pin: _SPENT if pin == "spent-acct" else _HEALTHY)
    slots = _profile_slots(("s1", "codex=gpt-5.6-sol", "spent-acct"),
                           ("s2", "codex=gpt-5.6-sol", "live-acct"))
    evidence = plan_panel_health_snapshot(slots)
    assert set(evidence) == {"s1"}
    assert asked == [("codex", "gpt-5.6-sol", "spent-acct"),
                     ("codex", "gpt-5.6-sol", "live-acct")]
    live, rows = plan_health_skip_rows(slots, evidence)
    assert [s.slot_id for s in live] == ["s2"]
    assert len(rows) == 1 and rows[0]["slot_id"] == "s1"
    assert "'spent-acct'" in rows[0]["error"] and rows[0]["cost"] == 0.0
    # And the memo still WORKS: a repeated identical subject is asked only once.
    asked.clear()
    plan_panel_health_snapshot(slots + _profile_slots(
        ("s3", "codex=gpt-5.6-sol", "spent-acct")))
    assert asked == [("codex", "gpt-5.6-sol", "spent-acct"),
                     ("codex", "gpt-5.6-sol", "live-acct")]


def test_unpinned_slots_keep_the_route_wide_answer(monkeypatch):
    """No regression for rows that pin nothing: the ask carries an empty profile
    (rotation stays Claudexor's business) and the skip row says so plainly."""
    from ouroboros.tools.plan_review_runtime import (
        plan_health_skip_rows, plan_panel_health_snapshot,
    )

    asked = _patch_snapshot_health(monkeypatch, lambda rid, model, pin: _SPENT)
    slots = _profile_slots(("s1", "codex=gpt-5.6-sol", ""))
    evidence = plan_panel_health_snapshot(slots)
    assert set(evidence) == {"s1"} and asked == [("codex", "gpt-5.6-sol", "")]
    _live, rows = plan_health_skip_rows(slots, evidence)
    assert "route-wide" in rows[0]["error"] and "pins no credential profile" in rows[0]["error"]


def test_transient_and_unknown_health_still_fail_open_for_a_pinned_slot(monkeypatch):
    """Fail-open is unchanged by the narrowing: an undated exhaustion, a transient
    daemon state and an unknown reason on a PINNED row all dispatch (no skip row)."""
    from ouroboros.tools.plan_review_runtime import plan_panel_health_snapshot

    for reason, reset in (("subscription_window_exhausted", ""),
                          ("daemon_recovery_only", ""),
                          ("route_status_disabled", ""),
                          ("", "2001-01-01T00:00:00Z")):
        _patch_snapshot_health(monkeypatch, lambda rid, model, pin, r=reason, t=reset: (r, t))
        assert plan_panel_health_snapshot(
            _profile_slots(("s1", "codex=gpt-5.6-sol", "spent-acct"))) == {}, reason


@pytest.mark.parametrize("aggregate", ["REVIEW_REQUIRED", "REVISE_PLAN", "DEGRADED"])
@pytest.mark.parametrize("enforcement", ["blocking", "advisory"])
@pytest.mark.parametrize("epoch", ["", "health"])
@pytest.mark.parametrize("cap", [None, 2, 3])
def test_whole_plan_render_respects_paid_capacity(aggregate, enforcement, epoch, cap):
    from ouroboros.tools.plan_render import _render_wave, _parse_plan_review_control
    import copy

    wave = {"aggregate": aggregate, "closed": False, "request_fingerprint": "fp", "health_epoch": epoch,
            "counts": {"configured": 3, "parseable": 0, "quorum": 2},
            "findings": [{"class": "blocking", "finding_id": "s1:f1"}],
            "closure_notes": ["blocking_finding_below_quorum_stays_open: let the next paid delta cycle judge the rejection",
                              "revise_plan_not_closable_by_disposition: next paid delta cycle",
                              "degraded_not_closable_by_disposition: rerun the wave",
                              "author note: retained unchanged"]}
    before = copy.deepcopy(wave)
    text = _render_wave(wave, cap=cap, cycles_paid=2, enforcement=enforcement)
    assert wave == before
    assert _parse_plan_review_control(text) == (aggregate, False)
    assert "author note: retained unchanged" in text
    assert "rerun the wave" not in text
    # The rendered control owns the current paid-capacity disclosure. The
    # retained closure notes remain provenance; no literal phrase is required.



def test_closed_and_pending_plan_states_do_not_advertise_new_review_at_cap():
    from ouroboros.tools.plan_render import _next_step

    assert _next_step({"aggregate": "GREEN", "closed": True}, enforcement="blocking", cap=2, cycles_paid=2).startswith("Closed: proceed")
    pending = _next_step({"aggregate": "DEGRADED", "custody_pending": True}, enforcement="blocking", cap=2, cycles_paid=2)
    assert "custody reconciliation" in pending and "fresh panel" not in pending


@pytest.mark.parametrize("enforcement", ["blocking", "advisory"])
@pytest.mark.parametrize("cap", [None, 2, 3])
def test_revise_plan_render_names_only_available_paid_cycles_and_exits(enforcement, cap):
    from ouroboros.tools.plan_render import _next_step

    text = _next_step({"aggregate": "REVISE_PLAN", "closed": False},
                      enforcement=enforcement, cap=cap, cycles_paid=2)
    assert "A disposition never closes REVISE_PLAN" in text
    if cap == 2:
        assert "rides into the next paid delta cycle" not in text
        assert "no further paid delta cycle" in text
        if enforcement == "blocking":
            assert "owner unstick (Swarm/hurry)" in text
            assert "once the owner raises OUROBOROS_REVIEW_MAX_CYCLES" in text
            assert "outcome_tier=blocked_with_evidence" in text
    else:
        assert "rides into the next paid delta cycle" in text
        assert "OUROBOROS_REVIEW_MAX_CYCLES" not in text
    if enforcement == "advisory":
        assert "Advisory enforcement: you may proceed" in text
        assert "host discloses" not in text and "your own final answer" in text
    evidence_text = _next_step({"aggregate": "REVIEW_REQUIRED", "closed": False},
                               enforcement=enforcement, cap=cap, cycles_paid=2)
    assert "ONE $0 call" in evidence_text and "no reviewer call, no cycle" in evidence_text
    assert ("reaches reviewers on the next paid cycle" in evidence_text) is (cap != 2)
    assert ("no further paid cycle" in evidence_text) is (cap == 2)


def test_loop_reminder_does_not_repromise_a_spent_panel(monkeypatch):
    from ouroboros.owner_hurry import plan_review_reminder

    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "2")
    for outcome in ("REVISE_PLAN", "REVIEW_REQUIRED", "DEGRADED"):
        text = plan_review_reminder({"outcome": outcome, "status": "open", "cycles_paid": 2,
                                    "reviewer_slots_degraded": outcome == "DEGRADED"})
        assert "cannot dispatch another paid panel" in text
        assert "re-dispatches a fresh panel" not in text
        assert "a changed spec starts the next paid cycle" not in text
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "unlimited")
    assert "re-dispatches a fresh panel" in plan_review_reminder({"outcome": "DEGRADED", "reviewer_slots_degraded": True, "cycles_paid": 2})


def _effort_aware_builder(harness, monkeypatch):
    """A builder stub that honours ``default_effort`` (the engine wraps the builder
    only when the envelope declares an effort; zero-arg stubs stay valid)."""
    from ouroboros.tools import plan_review as pr

    def build(default_effort=""):
        return [dataclasses.replace(slot, effort=default_effort or slot.effort,
                                    declared_effort=default_effort)
                for slot in harness.state["slots"]]

    monkeypatch.setattr(pr, "_plan_review_slots", build)


def test_declared_reviewer_effort_re_dispatches_and_the_same_declaration_replays_free(harness, monkeypatch):
    """Owner batch 2 Q3=A: the envelope declares the panel's strength; effort is
    roster identity, so a changed declaration is a new paid panel while repeating
    the same declaration replays the recorded wave for free."""
    _patch_health(monkeypatch, lambda slots: {})
    _effort_aware_builder(harness, monkeypatch)
    open_finding = json.dumps([_finding("n1", "blocking", breaks="claim_1")])
    sub = harness.install({"s1": open_finding, "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    first = _call(ctx, reviewer_effort="low")
    assert _control(first) == {"outcome": "REVIEW_REQUIRED", "closed": False}
    assert len(sub.calls) == 1 and [s.effort for s in sub.calls[0]["slots"]] == ["low", "low", "low"]
    wave = _state(harness)["waves"][-1]
    assert wave["reviewer_effort"] == "low" and "reviewer effort ordered for this envelope: low" in first
    assert _call(ctx, reviewer_effort="low").count("cached exact review") == 1 and len(sub.calls) == 1
    stronger = _call(ctx, reviewer_effort="max")
    assert "cached exact review" not in stronger and len(sub.calls) == 2
    assert [s.effort for s in sub.calls[1]["slots"]] == ["max", "max", "max"]
    assert _state(harness)["cycles_paid"] == 2
    # Off the scale: a typed argument refusal, no reviewer called, no attempt recorded.
    refused = _call(ctx, reviewer_effort="turbo")
    assert refused.startswith("ERROR: PLAN_SPEC_INVALID") and "reviewer_effort" in refused
    assert len(sub.calls) == 2


def test_a_closed_verdict_from_a_cheap_panel_is_not_reopened_by_a_stronger_declaration(harness, monkeypatch):
    """The disclosed residual named to the owner (batch 2, Q3): a closed verdict is
    earned authority for this envelope; ordering a stronger panel afterwards
    replays the closed wave for free instead of re-dispatching."""
    _patch_health(monkeypatch, lambda slots: {})
    _effort_aware_builder(harness, monkeypatch)
    sub = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx(task_id="task-cheap")
    assert _control(_call(ctx, reviewer_effort="none")) == {"outcome": "GREEN", "closed": True}
    again = _call(ctx, reviewer_effort="max")
    assert _control(again) == {"outcome": "GREEN", "closed": True}
    assert "cached exact review" in again and len(sub.calls) == 1


def test_default_effort_and_omission_share_the_configured_open_wave(harness, monkeypatch):
    _patch_health(monkeypatch, lambda slots: {})
    _effort_aware_builder(harness, monkeypatch)
    finding = json.dumps([_finding("n1", "blocking", breaks="claim_1")])
    transport = harness.install({"s1": finding, "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    assert _control(_call(ctx, reviewer_effort="default"))["closed"] is False
    before = _state(harness)["waves"][-1]
    assert before["reviewer_effort"] == ""
    assert [s.effort for s in transport.calls[0]["slots"]] == [s.effort for s in harness.state["slots"]]
    for effort in ("", "default"):
        assert "cached exact review" in _call(ctx, reviewer_effort=effort)
    after = _state(harness)["waves"][-1]
    assert after["request_fingerprint"] == before["request_fingerprint"]
    assert after["reviewer_config_fingerprint"] == before["reviewer_config_fingerprint"]
    assert len(transport.calls) == _state(harness)["cycles_paid"] == 1


@pytest.mark.parametrize("effort", ["low", "high", "none"])
def test_real_new_plan_keeps_an_explicit_effort(harness, monkeypatch, effort):
    _patch_health(monkeypatch, lambda slots: {})
    _effort_aware_builder(harness, monkeypatch)
    transport = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    assert _control(_call(harness.make_ctx(), reviewer_effort=effort))["closed"]
    assert _state(harness)["waves"][-1]["reviewer_effort"] == effort
    assert [slot.effort for slot in transport.calls[0]["slots"]] == [effort] * 3


def test_no_model_facing_text_promises_that_a_host_will_disclose_the_open_review(harness, monkeypatch):
    """The CLASS is closed, not three instances: every model-facing surface that permits
    proceeding with the review open (the tool description, the rendered next step of a
    settled and of a custody-pending open wave, the spent-cap head) says where the fact
    lives (typed state, the model's own answer) and never that a host will disclose it —
    and a concatenation-flattening scan of the runtime source finds no fourth string."""
    import pathlib
    import re

    from ouroboros.tools import plan_render
    from ouroboros.tools.plan_review import get_tools
    from tests.test_plan_review_engine import DECK_SPEC

    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    harness.state["enforcement"] = "advisory"
    harness.install({"s1": json.dumps([_finding("n1", "note")]), "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    _call(ctx)
    pending = [{"slot_id": s, "model": "m", "ok": False, "error": "Pending dispatch;", "operation_state": "pending_dispatch",
                "late_result_pending": True, "operation_id": f"op-{s}"} for s in ("s1", "s2")]
    surfaces = {
        "tool description": next(t for t in get_tools() if t.name == "plan_task").schema["description"],
        "settled open wave": plan_render._next_step({"aggregate": "REVISE_PLAN", "closed": False, "request_fingerprint": "f" * 64},
                                                    enforcement="advisory", cap=3, cycles_paid=1),
        "custody-pending wave": plan_render._next_step({"aggregate": "DEGRADED", "closed": False, "custody_pending": True,
                                                        "request_fingerprint": "f" * 64, "actors": pending},
                                                       enforcement="advisory", cap=3, cycles_paid=1),
        "spent cap": _call(ctx, spec={**DECK_SPEC, "in_scope": ["a 6-slide deck"]}),
    }
    for name, text in surfaces.items():
        for promise in ("host discloses", "host records and", "discloses it", "discloses that loudly"):
            assert promise not in text, (name, promise)
        assert re.search(r"your own (final )?answer", text), name
    # The class, not the instances: no runtime source (implicit concatenation flattened) still carries one.
    root = pathlib.Path(plan_render.__file__).resolve().parents[1]
    for path in (*root.rglob("*.py"), *(root.parent / "prompts").glob("*.md")):
        flat = re.sub(r'"\s*\n\s*"', "", path.read_text(encoding="utf-8"))
        assert "host discloses" not in flat and "host records and discloses" not in flat, path


def test_an_author_finish_narrates_its_rationale_in_the_models_voice(harness, monkeypatch):
    """The mind's recorded reason reaches the owner as ITS OWN row (``narration=True``),
    verbatim, exactly once per durable author record — never on a refusal, never
    for an empty rationale, and never a second time from the tool's host lines."""
    import ouroboros.review_records as review_records
    from ouroboros.tools import plan_review as pr

    harness.state["enforcement"] = "advisory"
    harness.install({"s1": json.dumps([_finding("b1", "blocking", breaks="claim_1")]), "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    seen: list = []
    ctx.emit_progress_fn = lambda text, **kw: seen.append((text, kw))
    _call(ctx)
    fp = _state(harness)["waves"][-1]["request_fingerprint"]
    rationale = "The blocking finding assumes a chart per slide; the brief fixes one table, so I proceed."
    seen.clear()
    result = pr._handle_plan_task(ctx, review_disposition={
        "review_fingerprint": fp, "items": [], "author_action": "finish",
        "author_disposition": {"disposition": "accepted", "rationale": rationale}})
    assert "Your rationale was shown to the owner in your own voice." in result
    assert [row for row in seen if row[1]] == [(rationale, {"narration": True})]
    assert _state(harness)["current_attempt"]["author_subject"]["author_disposition"]["rationale"] == rationale
    # An empty rationale records the finish and says nothing in the model's voice.
    seen.clear()
    result_empty = pr._handle_plan_task(ctx, review_disposition={
        "review_fingerprint": fp, "items": [], "author_action": "finish",
        "author_disposition": {"disposition": "accepted", "rationale": ""}})
    assert not [row for row in seen if row[1].get("narration")]
    assert "Your rationale was shown to the owner" not in result_empty
    # The disposition path with an author disposition narrates once too; the host line stays host voice.
    seen.clear()
    pr._handle_plan_task(ctx, review_disposition={
        "review_fingerprint": fp, "items": [], "author_disposition": {"disposition": "rejected", "rationale": "Kept as is."}})
    assert [row for row in seen if row[1]] == [("Kept as is.", {"narration": True})]
    assert any(text.startswith("📐 Plan review: findings answered") and not kw for text, kw in seen)
    # A refused finish (reviewers still running) records nothing and narrates nothing.
    monkeypatch.setattr(review_records, "review_outcome_received", lambda *_a, **_kw: False)
    seen.clear()
    refused = pr._handle_plan_task(ctx, review_disposition={
        "review_fingerprint": fp, "items": [], "author_action": "finish",
        "author_disposition": {"disposition": "accepted", "rationale": "Proceed without the reviewers."}})
    assert "reviewers are still running" in refused and seen == []


def _pinned_xhigh_rows_env(monkeypatch):
    """Three api rows the owner pinned `xhigh`, read by the REAL plan builder."""
    from ouroboros.reviewer_slot_config import REVIEWER_SLOTS_ENV
    from ouroboros.tools import plan_review as pr, plan_review_runtime

    payload = {
        "triad": [{"slot_id": sid, "route": {"kind": "api_chat", "target_id": model}, "effort": "xhigh"}
                  for sid, model in (("s1", "m/a"), ("s2", "m/b"), ("s3", "m/c"))],
        "scope": [{"slot_id": "scope-route", "route": {"kind": "api_chat", "target_id": "openai/gpt-5.6-sol"}}],
        "advisory": {"enabled": True, "route": {"kind": "api", "target_id": ""}},
    }
    monkeypatch.setenv(REVIEWER_SLOTS_ENV, json.dumps(payload))
    monkeypatch.setenv("OUROBOROS_EFFORT_REVIEW", "medium")
    monkeypatch.setattr(pr, "_plan_review_slots", plan_review_runtime.plan_review_slots)


def test_the_order_outranks_pinned_rows_and_a_weaker_order_is_named(harness, monkeypatch):
    """Through the real builder on rows pinned `xhigh`: an order `low` runs every seat at
    low (reverted, the pins win), the wave records the effective per-seat effort and the
    owner baseline as one typed `ordered_weaker`, and the verdict names it; the same order
    replays free; `max` on the OPEN wave re-dispatches a paid panel with nothing weaker."""
    _patch_health(monkeypatch, lambda slots: {})
    _pinned_xhigh_rows_env(monkeypatch)
    open_finding = json.dumps([_finding("n1", "blocking", breaks="claim_1")])
    sub = harness.install({"s1": open_finding, "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    first = _call(ctx, reviewer_effort="low")
    assert _control(first) == {"outcome": "REVIEW_REQUIRED", "closed": False}
    assert [s.effort for s in sub.calls[0]["slots"]] == ["low", "low", "low"]
    wave = _state(harness)["waves"][-1]
    assert [(a["effort"], a["declared_effort"]) for a in wave["actors"]] == [("low", "low")] * 3
    assert wave["owner_efforts"] == {"s1": "xhigh", "s2": "xhigh", "s3": "xhigh"}
    assert wave["ordered_weaker"] == {sid: {"effort": "low", "owner_effort": "xhigh"} for sid in ("s1", "s2", "s3")}
    assert "ORDERED WEAKER THAN THE OWNER SETTING on s1 (low < xhigh), s2 (low < xhigh), s3 (low < xhigh)" in first
    assert "· effort low (ordered) ·" in first
    assert "cached exact review" in _call(ctx, reviewer_effort="low") and len(sub.calls) == 1
    stronger = _call(ctx, reviewer_effort="max")
    assert "cached exact review" not in stronger and len(sub.calls) == 2
    assert [s.effort for s in sub.calls[1]["slots"]] == ["max", "max", "max"]
    assert _state(harness)["cycles_paid"] == 2
    assert "ordered_weaker" not in _state(harness)["waves"][-1] and "ORDERED WEAKER" not in stronger


def test_the_owner_baseline_is_recorded_at_dispatch_and_carried_through_collection(harness, monkeypatch):
    """`ordered_weaker` is computed from the request at dispatch and reused by the $0
    collection (`owner_efforts` on the wave), never recomputed from a live setting that
    may have moved while the slots were in flight."""
    from tests.test_plan_review_reconciliation import _collect, _install_barrier_substrate

    _patch_health(monkeypatch, lambda slots: {})
    _pinned_xhigh_rows_env(monkeypatch)
    calls = []
    _install_barrier_substrate(monkeypatch, calls)
    ctx = harness.make_ctx()
    _call(ctx, reviewer_effort="low")
    wave = _state(harness)["waves"][-1]
    assert wave["custody_pending"] is True and wave["owner_efforts"]["s1"] == "xhigh"
    # The owner drops every pin before collection: the recorded baseline still speaks.
    from ouroboros.reviewer_slot_config import REVIEWER_SLOTS_ENV
    payload = json.loads(__import__("os").environ[REVIEWER_SLOTS_ENV])
    for row in payload["triad"]:
        row["effort"] = "low"
    monkeypatch.setenv(REVIEWER_SLOTS_ENV, json.dumps(payload))
    collected = _collect(ctx, wave["request_fingerprint"])
    settled = _state(harness)["waves"][-1]
    assert settled["custody_pending"] is False and settled["owner_efforts"] == wave["owner_efforts"]
    assert settled["ordered_weaker"] == {sid: {"effort": "low", "owner_effort": "xhigh"} for sid in ("s1", "s2", "s3")}
    assert "ORDERED WEAKER THAN THE OWNER SETTING" in collected


def test_a_compound_route_slug_keeps_its_effort_and_discloses_the_unapplied_order(harness, monkeypatch):
    """A seat whose route slug encodes its effort ignores the order (its effort is the
    route's identity) and says so on its actor row; the other seats run the order and
    only they can rank weaker than the owner's setting. Without an order: no disclosure."""
    from ouroboros.tools import plan_review as pr

    _patch_health(monkeypatch, lambda slots: {})

    def build(default_effort=""):  # s2 behaves like a compound slug: its own effort, no declaration
        return [slot if slot.slot_id == "s2" else dataclasses.replace(
            slot, effort=default_effort or slot.effort, declared_effort=default_effort)
                for slot in harness.state["slots"]]

    monkeypatch.setattr(pr, "_plan_review_slots", build)
    sub = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    out = _call(ctx, reviewer_effort="low")
    actors = {a["slot_id"]: a for a in _state(harness)["waves"][-1]["actors"]}
    assert actors["s2"]["effort"] == "high" and actors["s2"]["declared_effort"] == ""
    assert "reviewer_effort_not_applied" in actors["s2"]["disclosures"]
    assert all("reviewer_effort_not_applied" not in actors[s]["disclosures"] for s in ("s1", "s3"))
    assert _state(harness)["waves"][-1]["ordered_weaker"] == {
        "s1": {"effort": "low", "owner_effort": "high"}, "s3": {"effort": "low", "owner_effort": "high"}}
    assert "s2 · m/b · api_chat · effort high · " in out and "disclosures: reviewer_effort_not_applied" in out
    plain = _call(harness.make_ctx(task_id="task-plain"))
    assert "reviewer_effort_not_applied" not in plain and len(sub.calls) == 2
    assert all(a["declared_effort"] == "" for a in _state(harness, "task-plain")["waves"][-1]["actors"])


def test_a_reject_closed_predecessor_carries_nothing_on_a_later_same_spec_wave(harness, monkeypatch):
    """A below-quorum blocking finding closed by a reasoned reject under advisory is earned
    authority: a later same-spec re-dispatch with its objecting seat silent carries nothing
    from that CLOSED predecessor (an OPEN one carries it: the tests below)."""
    from tests.test_plan_review_reconciliation import _collect

    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "5")
    harness.state["enforcement"] = "advisory"
    _patch_health(monkeypatch, lambda slots: {})
    _effort_aware_builder(harness, monkeypatch)
    objection = json.dumps([_finding("n1", "blocking", breaks="claim_1", summary="Friday is impossible")])
    sub = harness.install({"s1": objection, "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    assert _control(_call(ctx, reviewer_effort="low")) == {"outcome": "REVIEW_REQUIRED", "closed": False}
    fp = _state(harness)["waves"][-1]["request_fingerprint"]
    closed = _collect(ctx, fp, items=[{"finding_id": "s1:n1", "decision": "reject", "rationale": "Friday is a hard date"}])
    assert _control(closed) == {"outcome": "GREEN", "closed": True}
    sub.answers = {"s1": "", "s2": CLEAN, "s3": CLEAN}  # a prose revision: a new fingerprint on the same spec
    later = _call(ctx, plan="Outline first, then draft each slide, then rehearse.", reviewer_effort="max")
    assert _control(later) == {"outcome": "GREEN", "closed": True} and len(sub.calls) == 2
    assert not any(f.get("carried_absent_answer") for f in _state(harness)["waves"][-1]["findings"])


def test_an_unparseable_objector_reply_still_carries_its_finding(harness, monkeypatch):
    """Standing findings are judged once ``ok`` is final: an objecting seat that answers prose
    (no findings array) on a same-spec re-dispatch is a non-answer, so its earlier blocking
    finding stays listed and the wave stays REVIEW_REQUIRED; a clean answer retires it."""
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "5")
    _patch_health(monkeypatch, lambda slots: {})
    _effort_aware_builder(harness, monkeypatch)
    objection = json.dumps([_finding("n1", "blocking", breaks="claim_1", summary="Friday is impossible")])
    sub = harness.install({"s1": objection, "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    assert _control(_call(ctx, reviewer_effort="low")) == {"outcome": "REVIEW_REQUIRED", "closed": False}
    sub.answers = {"s1": "I have nothing to add this round.", "s2": CLEAN, "s3": CLEAN}
    prose = _call(ctx, reviewer_effort="max")
    assert _control(prose) == {"outcome": "REVIEW_REQUIRED", "closed": False}
    wave = _state(harness)["waves"][-1]
    s1 = next(a for a in wave["actors"] if a["slot_id"] == "s1")
    assert s1["ok"] is False and s1["carried_findings"] == 1
    assert [f["finding_id"] for f in wave["findings"] if f.get("carried_absent_answer")] == ["s1:n1"]
    assert wave["counts"]["parseable"] == 2 and "did not answer; its earlier finding is still listed" in prose
    sub.answers = {"s1": CLEAN, "s2": CLEAN, "s3": CLEAN}
    assert _control(_call(ctx, reviewer_effort="xhigh")) == {"outcome": "GREEN", "closed": True}


def test_a_seat_awaiting_the_barrier_carries_nothing_until_its_absence_is_terminal(harness, monkeypatch):
    """At the dispatch barrier every fresh row is ``pending_dispatch``: a gap, never a
    non-answer, so no standing finding is carried and nothing says «did not answer»;
    once collection settles the objecting seat as a $0 ``not_dispatched`` refusal, its
    earlier finding is carried and the actor line says «not sent»."""
    from tests.test_plan_review_reconciliation import _collect, _install_barrier_substrate

    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "5")
    _patch_health(monkeypatch, lambda slots: {})
    _effort_aware_builder(harness, monkeypatch)
    objection = json.dumps([_finding("n1", "blocking", breaks="claim_1", summary="Friday is impossible")])
    harness.install({"s1": objection, "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    assert _control(_call(ctx, reviewer_effort="low")) == {"outcome": "REVIEW_REQUIRED", "closed": False}
    calls = []
    _install_barrier_substrate(monkeypatch, calls, refused={"s1"})
    barrier = _call(ctx, reviewer_effort="max")
    assert _control(barrier) == {"outcome": "DEGRADED", "closed": False}
    wave = _state(harness)["waves"][-1]
    assert wave["custody_pending"] is True
    assert not any(f.get("carried_absent_answer") for f in wave["findings"])
    assert all(not a.get("carried_findings") for a in wave["actors"])
    assert "earlier finding is still listed" not in barrier
    settled = _collect(ctx, wave["request_fingerprint"])
    assert _control(settled) == {"outcome": "REVIEW_REQUIRED", "closed": False}
    wave = _state(harness)["waves"][-1]
    s1 = next(a for a in wave["actors"] if a["slot_id"] == "s1")
    assert s1["operation_state"] == "not_dispatched" and s1["carried_findings"] == 1
    assert [f["finding_id"] for f in wave["findings"] if f.get("carried_absent_answer")] == ["s1:n1"]
    assert "not sent; its earlier finding is still listed" in settled


def test_a_changed_order_never_retires_a_silent_objectors_finding(harness, monkeypatch):
    """Standing findings by same spec hash and same seat: after s1 raised a blocking
    finding at `low`, a `max` re-dispatch on the SAME spec where s1 fails to answer
    keeps s1's finding listed (`did not answer; its earlier finding is still listed`)
    and the wave REVIEW_REQUIRED, never GREEN; the silent seat counts as neither
    parseable nor a blocking vote. Positive path: s1 answering clean retires it and the
    wave closes GREEN; a CHANGED spec carries nothing."""
    from tests.test_plan_review_engine import DECK_SPEC

    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "5")
    _patch_health(monkeypatch, lambda slots: {})
    _effort_aware_builder(harness, monkeypatch)
    objection = json.dumps([_finding("n1", "blocking", breaks="claim_1", summary="Friday is impossible")])
    sub = harness.install({"s1": objection, "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    assert _control(_call(ctx, reviewer_effort="low")) == {"outcome": "REVIEW_REQUIRED", "closed": False}
    sub.answers = {"s1": "", "s2": CLEAN, "s3": CLEAN}  # s1 dies on the stronger re-dispatch
    silent = _call(ctx, reviewer_effort="max")
    assert _control(silent) == {"outcome": "REVIEW_REQUIRED", "closed": False} and len(sub.calls) == 2
    wave = _state(harness)["waves"][-1]
    carried = [f for f in wave["findings"] if f.get("carried_absent_answer")]
    assert [(f["finding_id"], f["class"], f["summary"]) for f in carried] == [("s1:n1", "blocking", "Friday is impossible")]
    s1 = next(a for a in wave["actors"] if a["slot_id"] == "s1")
    assert s1["ok"] is False and s1["carried_findings"] == 1 and "findings_carried_absent_answer:1" in s1["disclosures"]
    assert wave["counts"]["parseable"] == 2 and wave["counts"]["blocking_slots"] == 0
    assert "findings_carried_absent_answer:s1:1" in wave["reasons"]
    assert "did not answer; its earlier finding is still listed" in silent and "blocking_below_quorum:0/2" in silent
    # Positive path through the rule: the objector answers and retires its finding.
    sub.answers = {"s1": CLEAN, "s2": CLEAN, "s3": CLEAN}
    assert _control(_call(ctx, reviewer_effort="xhigh")) == {"outcome": "GREEN", "closed": True}
    assert not any(f.get("carried_absent_answer") for f in _state(harness)["waves"][-1]["findings"])
    # A changed spec is a fresh judgement: on a second task whose wave is OPEN on s1's
    # objection, the changed spec with s1 silent carries nothing (the guard is spec-hash
    # equality), while the same silence on the unchanged spec carries it (above).
    other = harness.make_ctx(task_id="task-2")
    sub.answers = {"s1": objection, "s2": CLEAN, "s3": CLEAN}
    assert _control(_call(other, reviewer_effort="low")) == {"outcome": "REVIEW_REQUIRED", "closed": False}
    sub.answers = {"s1": "", "s2": CLEAN, "s3": CLEAN}
    changed = _call(other, spec={**DECK_SPEC, "in_scope": ["a 6-slide deck"]}, reviewer_effort="low")
    assert _control(changed) == {"outcome": "GREEN", "closed": True}
    assert not any(f.get("carried_absent_answer") for f in _state(harness, "task-2")["waves"][-1]["findings"])
    assert "did not answer; its earlier finding is still listed" not in changed


def test_a_compact_wave_carries_the_ordered_weaker_fact():
    from ouroboros.tools.plan_review_artifacts import compact_wave

    wave = {"request_fingerprint": "f" * 64, "aggregate": "GREEN", "closed": True, "paid": True,
            "cycle_index": 1, "findings": [], "ordered_weaker": {"s1": {"effort": "low", "owner_effort": "xhigh"}}}
    assert compact_wave(wave)["ordered_weaker"] == {"s1": {"effort": "low", "owner_effort": "xhigh"}}
    plain = dict(wave); plain.pop("ordered_weaker")
    assert "ordered_weaker" not in compact_wave(plain)
