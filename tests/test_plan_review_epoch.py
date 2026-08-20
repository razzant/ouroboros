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

import dataclasses
import json
import logging

from tests._plan_review_engine_shared import (
    CLEAN, _call, _control, _finding, _slots, _state,
)
from tests.test_plan_review_health import _DEAD_PANEL, _patch_health
from tests._plan_review_engine_shared import harness as _engine_harness  # shared fixture

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
    note = json.dumps([_finding("n1", "note")])
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


def test_cached_replay_of_an_open_wave_retries_a_failed_advisory_open_append(harness, monkeypatch):
    """Post-merge follow-up (sol finding 3): the durable advisory-open append that
    FAILED at record time was unreachable forever — the identical envelope's cached
    replay returned before the emitter. The replay path now re-invokes the emitter
    for the still-open wave: zero substrate calls, and the row lands exactly once."""
    import ouroboros.utils as utils

    harness.state["enforcement"] = "advisory"
    _patch_health(monkeypatch, lambda slots: {})
    note = json.dumps([_finding("n1", "note")])
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
