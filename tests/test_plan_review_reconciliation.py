"""Exact-cycle reconciliation regressions for plan review."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tests.test_plan_review_engine import (
    CLEAN,
    DECK_SPEC,
    _call,
    _control,
    _finding,
    harness as _engine_harness,
    _patch_health,
    _state,
)

# Explicitly re-export the fixture. pytest 8.x does not reliably discover a
# fixture from a test module via ``pytest_plugins`` when the provider is also
# collected, while pytest 9.x happens to do so.
harness = _engine_harness  # noqa: F811 - pytest fixture re-export


def _install_two_turn_substrate(monkeypatch, calls, *, pending_ids=None, texts=None):
    import ouroboros.review_custody as review_custody
    import ouroboros.review_substrate as review_substrate

    pending_ids = set(pending_ids or [])
    texts = dict(texts or {})

    def substrate(request, *, slots, drive_root, llm, usage_ctx=None):
        calls.append((request.retry_key, [slot.slot_id for slot in slots]))
        first = len(calls) == 1
        actors = []
        for slot in slots:
            pending = first and (not pending_ids or slot.slot_id in pending_ids)
            actors.append({
                "slot_id": slot.slot_id, "model": slot.model,
                "status": "error" if pending else "ok",
                "raw_text": "" if pending else texts.get(slot.slot_id, CLEAN),
                "error": "logical wait expired" if pending else "",
                "usage": {"resolved_model": slot.model},
                "prompt_ref": {}, "response_ref": {},
                "operation_id": f"op-{slot.slot_id}",
                "operation_state": "in_flight" if pending else "late_settled",
                "late_result_pending": pending,
            })
        return SimpleNamespace(actors=actors)

    monkeypatch.setattr(review_substrate, "run_review_request", substrate)
    monkeypatch.setattr(review_custody, "review_retry_custody_available", lambda **_kwargs: True)


def test_partial_quorum_stays_open_while_one_paid_slot_is_in_flight(harness, monkeypatch):
    """A 2/3 parseable quorum cannot close over a live paid reviewer worker."""
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    calls = []
    _install_two_turn_substrate(monkeypatch, calls, pending_ids={"s3"})
    ctx = harness.make_ctx()

    first = _call(ctx)
    state = _state(harness)
    wave = state["waves"][-1]
    assert _control(first) == {"outcome": "DEGRADED", "closed": False}
    assert wave["aggregate"] == "DEGRADED"
    assert wave["closed"] is False and wave["custody_pending"] is True
    assert "review_late_result_pending" in wave["reasons"]
    assert "Closed: proceed" not in first

    second = _call(ctx)
    assert _control(second) == {"outcome": "GREEN", "closed": True}


def test_expired_deadline_still_reconciles_existing_paid_wave(harness, monkeypatch):
    """An owner deadline must not strand a reviewer cycle already in flight."""
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    calls = []
    _install_two_turn_substrate(monkeypatch, calls, pending_ids={"s3"})
    ctx = harness.make_ctx()
    ctx.task_metadata["deadline_at"] = (
        datetime.now(timezone.utc) + timedelta(seconds=2000)
    ).isoformat()

    first = _call(ctx)
    assert _control(first) == {"outcome": "DEGRADED", "closed": False}
    assert _state(harness)["waves"][-1]["custody_pending"] is True

    # The second envelope arrives after the task's logical deadline. It must
    # rejoin the exact paid cycle, not return a fresh-deadline skip forever.
    ctx.task_metadata["deadline_at"] = "2000-01-01T00:00:00+00:00"
    second = _call(ctx)

    assert _control(second) == {"outcome": "GREEN", "closed": True}
    assert calls == [calls[0], calls[0]]
    wave = _state(harness)["waves"][-1]
    assert wave["custody_pending"] is False and wave["closed"] is True


def test_resume_keeps_original_dispatched_set_when_skipped_lane_heals(harness, monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    health_calls = []

    def health(_slots):
        health_calls.append(1)
        if len(health_calls) > 1:
            raise AssertionError("in-flight reconciliation re-probed live health")
        return {"s1": {"failure_code": "credential_pool_exhausted", "reset_at": ""}}

    _patch_health(monkeypatch, health)
    calls = []
    _install_two_turn_substrate(monkeypatch, calls)
    ctx = harness.make_ctx()
    _call(ctx)
    out = _call(ctx)

    assert _control(out) == {"outcome": "GREEN", "closed": True}
    assert calls == [(calls[0][0], ["s2", "s3"]), (calls[0][0], ["s2", "s3"])]
    assert health_calls == [1]
    state = _state(harness)
    assert state["cycles_paid"] == 1 and state["waves"][-1]["cycle_index"] == 1
    frozen = {row["slot_id"]: row for row in state["waves"][-1]["actors"]}["s1"]
    assert frozen["failure_code"] == "credential_pool_exhausted" and frozen["cost"] == 0.0


def test_resume_keeps_dispatched_lane_when_live_health_worsens(harness, monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    health_state, health_calls = {"evidence": {}}, []

    def health(_slots):
        health_calls.append(1)
        if len(health_calls) > 1:
            raise AssertionError("in-flight reconciliation re-probed worsened health")
        return dict(health_state["evidence"])

    _patch_health(monkeypatch, health)
    calls = []
    _install_two_turn_substrate(monkeypatch, calls)
    ctx = harness.make_ctx()
    _call(ctx)
    health_state["evidence"] = {
        "s1": {"failure_code": "subscription_window_exhausted",
               "reset_at": "2030-01-01T00:00:00+00:00"},
    }
    out = _call(ctx)

    assert _control(out) == {"outcome": "GREEN", "closed": True}
    assert calls == [
        (calls[0][0], ["s1", "s2", "s3"]),
        (calls[0][0], ["s1", "s2", "s3"]),
    ]
    assert health_calls == [1]
    assert _state(harness)["cycles_paid"] == 1


def test_in_flight_wave_defers_need_evidence_until_terminal_reconciliation(
    harness, monkeypatch,
):
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    requested = json.dumps([_finding(
        "e1", "need_evidence", locator="notes.md", summary="read the notes",
    )])
    calls = []
    _install_two_turn_substrate(
        monkeypatch, calls, pending_ids={"s2", "s3"}, texts={"s1": requested},
    )
    ctx = harness.make_ctx()
    _call(ctx)
    first_state = _state(harness)
    first_fp = first_state["waves"][-1]["request_fingerprint"]
    assert first_state["need_evidence_seen"] == []

    out = _call(ctx)
    state = _state(harness)
    assert _control(out) == {"outcome": "REVIEW_REQUIRED", "closed": False}
    assert [key for key, _slots in calls] == [calls[0][0], calls[0][0]]
    assert state["waves"][-1]["request_fingerprint"] == first_fp
    assert state["waves"][-1]["cycle_index"] == 1 and state["cycles_paid"] == 1
    assert state["need_evidence_seen"] == ["notes.md"]


@pytest.mark.parametrize(
    ("actors", "custody_pending"),
    [
        (["CORRUPT-ROW"], True),
        ([{
            "slot_id": "s1", "operation_id": "op-s1",
            "operation_state": "settled", "late_result_pending": False,
            "status": "error", "error": "unknown custody",
            "usage": {"physical_attempt_state": "future_state"},
        }], False),
    ],
)
def test_malformed_paid_wave_cannot_bypass_resume_validation(
    harness, monkeypatch, actors, custody_pending,
):
    """Malformed exact custody must not fall through to a fresh paid cycle."""
    from ouroboros.tools import plan_review as plan_review_tool

    calls = []
    _install_two_turn_substrate(monkeypatch, calls)
    ctx = harness.make_ctx()
    first = _call(ctx)
    assert _control(first) == {"outcome": "DEGRADED", "closed": False}
    assert len(calls) == 1

    materialize = plan_review_tool._authority_wave

    def malformed_authority(*args, **kwargs):
        wave = dict(materialize(*args, **kwargs))
        wave.update({
            "actors": actors, "paid": True,
            "custody_pending": custody_pending,
            "aggregate": "DEGRADED", "closed": False, "health_epoch": [],
        })
        return wave

    monkeypatch.setattr(plan_review_tool, "_authority_wave", malformed_authority)
    second = _call(ctx)

    assert len(calls) == 1
    assert second.startswith("ERROR: PLAN_REVIEW_CUSTODY_INVALID:")
    assert "Refusing" in second


def test_contradictory_positive_capture_reenters_custody_instead_of_fresh_cycle(
    harness, monkeypatch,
):
    """A synthetic $0 label cannot erase a positive physical-attempt fact."""
    import ouroboros.review_custody as review_custody
    from ouroboros.tools import plan_review as plan_review_tool

    calls = []
    _install_two_turn_substrate(monkeypatch, calls)
    ctx = harness.make_ctx()
    first = _call(ctx)
    assert _control(first) == {"outcome": "DEGRADED", "closed": False}
    assert len(calls) == 1

    materialize = plan_review_tool._authority_wave

    def contradictory_authority(*args, **kwargs):
        wave = dict(materialize(*args, **kwargs))
        wave.update({
            "actors": [{
                "slot_id": "s1", "operation_id": "op-s1",
                "operation_state": "not_dispatched", "late_result_pending": False,
                "status": "not_dispatched", "error": "synthetic refusal",
                "usage": {
                    "physical_attempt_state": "unresolved",
                    "provider_status_code": 503,
                },
            }, {
                "slot_id": "s2", "operation_id": "op-s2-free",
                "operation_state": "not_dispatched", "status": "not_dispatched",
                "error": "frozen $0 refusal",
            }, {
                "slot_id": "s3", "operation_id": "op-s3-free",
                "operation_state": "not_dispatched", "status": "not_dispatched",
                "error": "frozen $0 refusal",
            }],
            "paid": True, "custody_pending": False,
            "aggregate": "DEGRADED", "closed": False, "health_epoch": [],
        })
        return wave

    monkeypatch.setattr(plan_review_tool, "_authority_wave", contradictory_authority)
    monkeypatch.setattr(
        review_custody, "review_retry_custody_available", lambda **_kwargs: False,
    )
    second = _call(ctx)

    assert len(calls) == 1
    assert _control(second) == {"outcome": "DEGRADED", "closed": False}
    assert "process-local custody is unavailable" in second


def test_resume_excludes_synthetic_not_dispatched_operation_ids(tmp_path):
    """A pre-dispatch $0 row has an operation id, but is not a callable lane."""
    from ouroboros.tools.plan_review_artifacts import in_flight_resume_inputs

    slots = [
        SimpleNamespace(slot_id="s1", model="model/one"),
        SimpleNamespace(slot_id="s2", model="model/two"),
    ]
    result = in_flight_resume_inputs(
        {
            "actors": [
                {
                    "slot_id": "s1", "operation_id": "op-paid",
                    "operation_state": "in_flight", "late_result_pending": True,
                    "status": "error", "error": "still running",
                },
                {
                    "slot_id": "s2", "operation_id": "op-free",
                    "operation_state": "not_dispatched", "status": "not_dispatched",
                    "error": "budget admission refused",
                },
            ],
        },
        {}, tmp_path, "mixed-resume", slots,
    )

    assert result["dispatched_slot_ids"] == ["s1"]
    assert [row["slot_id"] for row in result["frozen_rows"]] == ["s2"]


def test_resume_counts_positive_capture_despite_synthetic_not_dispatched(tmp_path):
    """Positive physical custody outranks contradictory synthetic $0 labels."""
    from ouroboros.tools.plan_review_artifacts import in_flight_resume_inputs

    result = in_flight_resume_inputs(
        {
            "actors": [{
                "slot_id": "s1", "operation_id": "op-paid",
                "operation_state": "not_dispatched", "status": "not_dispatched",
                "error": "synthetic refusal",
                "usage": {
                    "physical_attempt_state": "unresolved",
                    "provider_status_code": 503,
                },
            }],
        },
        {}, tmp_path, "contradictory-resume", [
            SimpleNamespace(slot_id="s1", model="model/one"),
        ],
    )

    assert result["dispatched_slot_ids"] == ["s1"]
    assert result["frozen_rows"] == []


def test_resume_rejects_non_object_rows_in_exact_paid_roster(tmp_path):
    """A corrupt durable roster must not lose rows during reconciliation."""
    from ouroboros.tools.plan_review_artifacts import in_flight_resume_inputs

    result = in_flight_resume_inputs(
        {
            "actors": [{
                "slot_id": "s1", "operation_id": "op-paid",
                "operation_state": "in_flight", "late_result_pending": True,
                "status": "error", "error": "still running",
            }, "CORRUPT-ROW"],
        },
        {}, tmp_path, "malformed-roster", [
            SimpleNamespace(slot_id="s1", model="model/one"),
        ],
    )

    assert "error" in result
    assert "drop rows" in result["error"]


def test_resume_rejects_unknown_physical_attempt_state(tmp_path):
    """Unknown custody facts cannot be inferred as a safe retry or refusal."""
    from ouroboros.tools.plan_review_artifacts import in_flight_resume_inputs

    result = in_flight_resume_inputs(
        {
            "actors": [{
                "slot_id": "s1", "operation_id": "op-paid",
                "operation_state": "settled", "status": "error",
                "error": "provider state unavailable",
                "usage": {"physical_attempt_state": "future_state"},
            }],
        },
        {}, tmp_path, "unknown-roster-state", [
            SimpleNamespace(slot_id="s1", model="model/one"),
        ],
    )

    assert "error" in result
    assert "unknown physical-attempt state" in result["error"]


def test_zero_send_route_refusal_does_not_spend_a_plan_cycle(harness, monkeypatch):
    """Callable configuration is not monetary proof when every slot refuses pre-send."""
    import ouroboros.review_substrate as review_substrate

    calls = []

    def zero_send(request, *, slots, drive_root, llm, usage_ctx=None):
        calls.append([slot.slot_id for slot in slots])
        return SimpleNamespace(actors=[{
            "slot_id": slot.slot_id,
            "model": slot.model,
            "status": "not_dispatched",
            "raw_text": "",
            "error": "agent session slot has no session task",
            "failure_code": "session_task_missing",
            "usage": {},
            "prompt_ref": {},
            "response_ref": {},
            "operation_id": f"op-{slot.slot_id}",
            "operation_state": "not_dispatched",
            "late_result_pending": False,
        } for slot in slots])

    monkeypatch.setattr(review_substrate, "run_review_request", zero_send)
    output = _call(harness.make_ctx())
    state = _state(harness)
    wave = state["waves"][-1]

    assert calls == [["s1", "s2", "s3"]]
    assert _control(output) == {"outcome": "DEGRADED", "closed": False}
    assert wave["paid"] is False
    assert state["cycles_paid"] == 0
    assert {row["failure_code"] for row in wave["actors"]} == {
        "session_task_missing",
    }


@pytest.mark.parametrize("row", [
    {"status": "error", "error": "substrate omitted the actor"},
    {"status": "error", "usage": {"physical_attempt_state": "settled"}},
    {"status": "error", "usage": {"physical_attempt_state": "future_state"}},
])
def test_missing_operation_identity_cannot_prove_a_free_wave(row):
    from ouroboros.tools.plan_review_artifacts import _row_has_physical_dispatch

    assert _row_has_physical_dispatch(row) is True


def test_explicit_zero_send_fact_is_the_only_missing_identity_free_proof():
    from ouroboros.tools.plan_review_artifacts import _row_has_physical_dispatch

    assert _row_has_physical_dispatch({
        "status": "not_dispatched", "operation_state": "not_dispatched",
    }) is False


def test_missing_substrate_actor_stays_paid_and_custody_lost(harness, monkeypatch):
    """A dropped actor is unknown physical custody, never a terminal free retry."""
    import ouroboros.review_substrate as review_substrate

    calls = []

    def no_actors(request, *, slots, drive_root, llm, usage_ctx=None):
        calls.append([slot.slot_id for slot in slots])
        return SimpleNamespace(actors=[])

    monkeypatch.setattr(review_substrate, "run_review_request", no_actors)
    first = _call(harness.make_ctx())
    state = _state(harness)
    wave = state["waves"][-1]

    assert calls == [["s1", "s2", "s3"]]
    assert _control(first) == {"outcome": "DEGRADED", "closed": False}
    assert wave["paid"] is True and wave["custody_pending"] is True
    assert state["cycles_paid"] == 1
    assert {row["failure_code"] for row in wave["actors"]} == {
        "review_custody_lost",
    }
    assert {row["operation_state"] for row in wave["actors"]} == {
        "custody_lost",
    }

    second = _call(harness.make_ctx())
    assert _control(second) == {"outcome": "DEGRADED", "closed": False}
    assert "Refusing a duplicate paid send" in second
    assert calls == [["s1", "s2", "s3"]]
    assert _state(harness)["cycles_paid"] == 1


# ------------------------------------------------------------- collection (P1-3)


def _install_barrier_substrate(monkeypatch, calls, *, texts=None, still_pending=(), refused=(), pending_waves=(),
                               pending_by_wave=None):
    """A substrate that honours the event route: a fresh dispatch released at its
    drain deadline returns ``pending_dispatch`` rows; a reconcile returns the settled
    rows (except ``still_pending`` slots, which are still running, and ``refused``
    slots, which settled as typed $0 not_dispatched refusals: nothing was sent)."""
    import ouroboros.review_custody as review_custody
    import ouroboros.review_substrate as review_substrate

    texts = dict(texts or {})

    def substrate(request, *, slots, drive_root, llm, usage_ctx=None):
        calls.append({"retry_key": request.retry_key, "slots": [s.slot_id for s in slots],
                      "reconcile_only": request.reconcile_only, "drain": request.drain_deadline,
                      "_request": request, "_slots": list(slots)})
        fresh = request.drain_deadline is not None and not request.reconcile_only
        wave_fp = str((request.reconciliation_identity or {}).get("subject_hash") or "")
        actors = []
        for slot in slots:
            pending = (fresh or slot.slot_id in still_pending or wave_fp in pending_waves
                       or slot.slot_id in (pending_by_wave or {}).get(wave_fp, ()))
            refuse = not pending and slot.slot_id in refused
            actors.append({
                "slot_id": slot.slot_id, "model": slot.model,
                "status": "not_dispatched" if refuse else ("error" if pending else "ok"),
                "raw_text": "" if (pending or refuse) else texts.get(slot.slot_id, CLEAN),
                "error": ("Pending dispatch; the physical review operation is in flight" if pending
                          else "daemon unreachable before physical review dispatch" if refuse else ""),
                "usage": {"resolved_model": slot.model,
                          **({} if (pending or refuse) else {"physical_attempt_state": "settled"})},
                "prompt_ref": {}, "response_ref": {}, "operation_id": f"op-{wave_fp[:8]}-{slot.slot_id}",
                "operation_state": "pending_dispatch" if pending else ("not_dispatched" if refuse else "settled"),
                "late_result_pending": pending,
            })
        return SimpleNamespace(actors=actors)

    monkeypatch.setattr(review_substrate, "run_review_request", substrate)
    monkeypatch.setattr(review_custody, "review_retry_custody_available", lambda **_kwargs: True)


def _collect(ctx, fingerprint, items=()):
    from ouroboros.tools import plan_review as pr

    return pr._handle_plan_task(ctx, review_disposition={"review_fingerprint": fingerprint, "items": list(items)})


def test_disposition_with_empty_items_collects_the_settled_wave_without_a_second_send(harness, monkeypatch):
    calls = []
    _install_barrier_substrate(monkeypatch, calls)
    ctx = harness.make_ctx()
    first = _call(ctx)
    wave = _state(harness)["waves"][-1]
    assert _control(first) == {"outcome": "DEGRADED", "closed": False}
    assert wave["custody_pending"] is True and wave["paid"] is False
    assert calls[0]["drain"] is not None and calls[0]["reconcile_only"] is False

    collected = _collect(ctx, wave["request_fingerprint"])
    assert _control(collected) == {"outcome": "GREEN", "closed": True}
    state = _state(harness)
    assert state["cycles_paid"] == 1 and state["waves"][-1]["paid"] is True
    # ONE reconcile call: the same retry key, every released slot, no re-dispatch, no wait.
    assert [c["reconcile_only"] for c in calls] == [False, True]
    assert calls[1]["retry_key"] == calls[0]["retry_key"] and calls[1]["slots"] == ["s1", "s2", "s3"]
    assert calls[1]["drain"] is not None
    assert state["current_attempt"]["fingerprint"] == wave["request_fingerprint"]


def test_collection_never_waits_for_a_live_slot_and_stays_free(harness, monkeypatch):
    calls = []
    _install_barrier_substrate(monkeypatch, calls, still_pending={"s3"})
    ctx = harness.make_ctx()
    _call(ctx)
    fingerprint = _state(harness)["waves"][-1]["request_fingerprint"]
    peek = _collect(ctx, fingerprint)
    assert _control(peek) == {"outcome": "DEGRADED", "closed": False}
    wave = _state(harness)["waves"][-1]
    assert wave["custody_pending"] is True
    by_slot = {a["slot_id"]: a for a in wave["actors"]}
    assert by_slot["s1"]["ok"] and by_slot["s2"]["ok"]
    assert by_slot["s3"]["operation_state"] == "pending_dispatch"
    assert calls[-1]["drain"] is not None  # window 0: a peek, never a wait
    # Two settled physical rows prove dispatch: the cycle is paid now, once.
    assert _state(harness)["cycles_paid"] == 1
    # Items on a still-open wave are applied after the collection, never dropped: an
    # unknown finding id is the typed refusal every other disposition path gives.
    again = _collect(ctx, fingerprint, items=[{"finding_id": "x", "decision": "accept", "rationale": "r"}])
    assert again.startswith("ERROR: PLAN_REVIEW_DISPOSITION_INVALID: unknown finding ids x")
    assert _state(harness)["cycles_paid"] == 1


def test_new_envelope_reconciles_the_in_flight_wave_before_superseding(harness, monkeypatch):
    calls = []
    _install_barrier_substrate(monkeypatch, calls)
    ctx = harness.make_ctx()
    _call(ctx)
    old = _state(harness)["waves"][-1]
    assert old["custody_pending"] is True
    second = _call(ctx, spec={**DECK_SPEC, "in_scope": ["a 6-slide deck"]})
    assert _control(second) == {"outcome": "DEGRADED", "closed": False}
    state = _state(harness)
    waves = {w["request_fingerprint"]: w for w in state["waves"]}
    # The old wave was collected (settled rows landed, closed GREEN, paid) BEFORE the
    # new envelope superseded it; the new wave is the current open attempt.
    assert waves[old["request_fingerprint"]]["closed"] is True
    assert waves[old["request_fingerprint"]]["paid"] is True
    new_fp = state["current_attempt"]["fingerprint"]
    assert new_fp != old["request_fingerprint"] and waves[new_fp]["custody_pending"] is True
    assert [c["reconcile_only"] for c in calls] == [False, True, False]
    assert calls[1]["retry_key"] == calls[0]["retry_key"]
    assert state["cycles_paid"] == 1  # the old wave's cycle; the new barrier wave is unpaid


def test_compaction_keeps_an_in_flight_wave_full(tmp_path):
    from ouroboros.task_results import _PLAN_REVIEW_FULL_WAVES, load_plan_review_state, record_plan_review_wave
    from tests.test_plan_review import _wave

    pending = {**_wave("a" * 64, aggregate="DEGRADED"), "paid": False, "custody_pending": True,
               "actors": [{"slot_id": "s1", "operation_state": "pending_dispatch"}]}
    record_plan_review_wave(tmp_path, "t", pending)
    for index in range(_PLAN_REVIEW_FULL_WAVES + 1):
        record_plan_review_wave(tmp_path, "t", _wave(f"{index:064x}", aggregate="GREEN", closed=True))
    waves = load_plan_review_state(tmp_path, "t")["waves"]
    first = waves[0]
    assert first["request_fingerprint"] == "a" * 64
    assert not first.get("compact") and first["custody_pending"] is True
    assert waves[1].get("compact") is True


def test_collect_binds_acceptance_claims_the_same_as_a_synchronous_close(harness, monkeypatch):
    from ouroboros.contracts.task_contract import effective_acceptance_claims
    from ouroboros.task_results import closed_plan_review_wave

    calls = []
    _install_barrier_substrate(monkeypatch, calls)
    ctx = harness.make_ctx()
    _call(ctx)
    state = _state(harness)
    assert closed_plan_review_wave(state) is None
    assert effective_acceptance_claims({}, closed_plan_review_wave(state)) == ([], "")
    _collect(ctx, state["waves"][-1]["request_fingerprint"])
    claims, source = effective_acceptance_claims({}, closed_plan_review_wave(_state(harness)))
    assert source == "plan_review" and [c["claim"] for c in claims] == DECK_SPEC["acceptance_claims"]


def test_two_step_wave_emits_one_advisory_open_event_and_keeps_the_paid_identity(harness, monkeypatch):
    from ouroboros.loop_acceptance_review import acceptance_paid_identity

    harness.state["enforcement"] = "advisory"
    calls = []
    _install_barrier_substrate(monkeypatch, calls, still_pending={"s2"})
    ctx = harness.make_ctx()
    _call(ctx)
    fingerprint = _state(harness)["waves"][-1]["request_fingerprint"]
    _collect(ctx, fingerprint)
    rows = [json.loads(line) for line in
            (harness.drive / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    opens = [r for r in rows if r.get("type") == "plan_review_advisory_open" and r.get("fingerprint") == fingerprint]
    assert len(opens) == 1  # dispatch -> collect is ONE recorded-open state, deduplicated
    # Two plan_task receipts change the acceptance EVIDENCE revision, never the
    # paid identity a panel is bought under (candidate + dispositions only).
    trace = {"tool_calls": [{"plan_review_outcome": "DEGRADED"}, {"plan_review_outcome": "DEGRADED"}],
             "acceptance_obligations": []}
    assert acceptance_paid_identity("cand", trace) == acceptance_paid_identity("cand", {"tool_calls": [], "acceptance_obligations": []})


def test_collection_rebuilds_the_roster_the_wave_was_dispatched_with(harness, monkeypatch):
    """A declared effort is roster identity; the collection reuses the wave's
    recorded declaration, so it finds the same roster instead of refusing."""
    import dataclasses

    from ouroboros.tools import plan_review as pr

    def build(default_effort=""):
        return [dataclasses.replace(slot, effort=default_effort or slot.effort, declared_effort=default_effort)
                for slot in harness.state["slots"]]

    monkeypatch.setattr(pr, "_plan_review_slots", build)
    calls = []
    _install_barrier_substrate(monkeypatch, calls)
    ctx = harness.make_ctx()
    _call(ctx, reviewer_effort="xhigh")
    wave = _state(harness)["waves"][-1]
    assert wave["custody_pending"] is True and wave["reviewer_effort"] == "xhigh"
    collected = _collect(ctx, wave["request_fingerprint"])
    assert _control(collected) == {"outcome": "GREEN", "closed": True}
    assert calls[1]["retry_key"] == calls[0]["retry_key"]


def test_disposition_items_are_recorded_on_a_wave_that_stays_custody_pending(harness, monkeypatch):
    """Fix cycle 1, F2 (P1-3a's letter): collect, THEN apply the items. An author's answer
    to a reviewer's question is recorded even while a sibling slot is still in flight,
    with the closure note the base behaviour carried; nothing is dropped silently."""
    calls = []
    question = json.dumps([_finding("q1", "need_evidence", breaks="claim_1", summary="Why five?")])
    _install_barrier_substrate(monkeypatch, calls, texts={"s1": question}, still_pending={"s3"})
    ctx = harness.make_ctx()
    _call(ctx)
    fingerprint = _state(harness)["waves"][-1]["request_fingerprint"]
    peek = _collect(ctx, fingerprint)
    assert _control(peek) == {"outcome": "DEGRADED", "closed": False}
    answered = _collect(ctx, fingerprint, items=[
        {"finding_id": "s1:q1", "decision": "accept", "rationale": "The board asked for five."}])
    wave = _state(harness)["waves"][-1]
    assert wave["custody_pending"] is True
    assert [(d["finding_id"], d["decision"]) for d in wave["dispositions"]] == [("s1:q1", "accept")]
    assert "The board asked for five." in answered and "degraded_not_closable_by_disposition" in answered
    assert _state(harness)["cycles_paid"] == 1 and len(calls) == 3  # one dispatch, two $0 reconciles
    # The last slot settles: the collection re-synthesizes the wave WITH the recorded
    # answer, and the answered question closes the wave instead of being wiped.
    _install_barrier_substrate(monkeypatch, calls, texts={"s1": question})
    final = _collect(ctx, fingerprint)
    wave = _state(harness)["waves"][-1]
    assert wave["custody_pending"] is False and wave["aggregate"] == "GREEN"  # the answered question emptied the open set
    assert [(d["finding_id"], d["decision"]) for d in wave["dispositions"]] == [("s1:q1", "accept")]
    assert _control(final) == {"outcome": "GREEN", "closed": True}
    assert _state(harness)["cycles_paid"] == 1


# ------------------------------------------------------------- the cap and an in-flight wave (fix cycle 2, 2a)


def _hold_text(text):
    return text.startswith("ERROR: PLAN_REVIEW_IN_FLIGHT:")


def test_a_revised_envelope_at_the_cap_is_held_until_the_in_flight_wave_is_collected_and_a_zero_wave_frees_it(harness, monkeypatch):
    """Under blocking with cap 1: envelope 1 dispatches at the barrier; a revised
    envelope while that wave is still in flight is HELD with a typed refusal that names
    the $0 collection, recorded before any superseding reference (the pending wave
    stays current and collectible); no cycles_exhausted is written for an unproven
    panel, so the gate does not release. When the collection proves no physical
    dispatch (all typed $0 refusals) the cap is untouched and the revised envelope
    dispatches, exactly as the base."""
    from ouroboros.task_results import plan_review_gate_projection

    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    calls = []
    _install_barrier_substrate(monkeypatch, calls, still_pending={"s1", "s2", "s3"})
    ctx = harness.make_ctx()
    _call(ctx)
    first = _state(harness)["waves"][-1]
    assert first["custody_pending"] is True and first["paid"] is False
    revised = {**DECK_SPEC, "in_scope": ["a 6-slide deck"]}
    held = _call(ctx, spec=revised)
    assert _hold_text(held) and first["request_fingerprint"] in held and "review_disposition" in held
    state = _state(harness)
    assert state["current_attempt"]["fingerprint"] == first["request_fingerprint"]  # not superseded
    assert state["current_attempt"].get("status") != "cycles_exhausted"
    assert not any(w.get("cycles_exhausted") for w in state["waves"])
    gate = plan_review_gate_projection(state, "blocking")
    assert gate["allow"] is False and gate["custody_pending"] is True  # no release without a dispatch
    assert [c["reconcile_only"] for c in calls] == [False, True]  # the hold collected at $0, sent nothing
    # Every slot settles as a typed $0 refusal: the collection proves NO dispatch.
    _install_barrier_substrate(monkeypatch, calls, refused={"s1", "s2", "s3"})
    _collect(ctx, first["request_fingerprint"])
    state = _state(harness)
    wave = state["waves"][-1]
    assert wave["custody_pending"] is False and wave["paid"] is False and state["cycles_paid"] == 0
    assert plan_review_gate_projection(state, "blocking")["allow"] is False
    dispatched = _call(ctx, spec=revised)
    assert _control(dispatched) == {"outcome": "DEGRADED", "closed": False}
    state = _state(harness)
    assert state["current_attempt"]["fingerprint"] != first["request_fingerprint"]
    assert calls[-1]["reconcile_only"] is False and calls[-1]["drain"] is not None  # a real new panel
    assert state["cycles_paid"] == 0 and len(calls) == 4


def test_a_revised_envelope_is_exhausted_only_after_the_collection_proves_the_dispatch(harness, monkeypatch):
    from ouroboros.task_results import plan_review_gate_projection

    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    calls = []
    _install_barrier_substrate(monkeypatch, calls, still_pending={"s1", "s2", "s3"})
    ctx = harness.make_ctx()
    _call(ctx)
    first = _state(harness)["waves"][-1]["request_fingerprint"]
    revised = {**DECK_SPEC, "in_scope": ["a 6-slide deck"]}
    assert _hold_text(_call(ctx, spec=revised))
    assert _state(harness)["current_attempt"]["fingerprint"] == first
    _install_barrier_substrate(monkeypatch, calls)  # the reviewers settle: the panel WAS sent
    assert _control(_collect(ctx, first)) == {"outcome": "GREEN", "closed": True}
    assert _state(harness)["cycles_paid"] == 1
    exhausted = _call(ctx, spec=revised)
    assert exhausted.startswith("⚠️ PLAN_REVIEW_CYCLES_EXHAUSTED: 1 of 1 paid plan-review cycles are spent")
    assert plan_review_gate_projection(_state(harness), "blocking")["status"] == "cycles_exhausted"
    assert [c["reconcile_only"] for c in calls] == [False, True, True]  # no new panel was sent


def _answered_mixed_wave(harness, monkeypatch, calls):
    """The state after test_disposition_items_are_recorded_on_a_wave_that_stays_custody_pending:
    s1 asked, the author answered at $0, s2 settled, s3 still pending_dispatch; the
    wave is PAID (two proven sends) and still custody-pending; cycles_paid == 1."""
    question = json.dumps([_finding("q1", "need_evidence", breaks="claim_1", summary="Why five?")])
    _install_barrier_substrate(monkeypatch, calls, texts={"s1": question}, still_pending={"s3"})
    ctx = harness.make_ctx()
    _call(ctx)
    fingerprint = _state(harness)["waves"][-1]["request_fingerprint"]
    _collect(ctx, fingerprint, items=[{"finding_id": "s1:q1", "decision": "accept", "rationale": "The board asked for five."}])
    wave = _state(harness)["waves"][-1]
    assert wave["paid"] is True and wave["custody_pending"] is True and _state(harness)["cycles_paid"] == 1
    return ctx, fingerprint


def test_a_paid_wave_still_in_flight_holds_a_revised_envelope_at_the_cap_and_counts_once(harness, monkeypatch):
    """Fix cycle 3, 3a: a wave that already proved a dispatch (paid) but still has a
    slot in flight occupies its cap slot ONCE (through cycles_paid, not again as
    pending). At cap 1 a revised envelope is held, not exhausted: the pending wave
    stays current and collectible and the gate is not released while a slot runs.
    At the default cap 2 the same revised envelope dispatches a second panel."""
    from ouroboros.task_results import plan_review_gate_projection

    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    calls = []
    ctx, fingerprint = _answered_mixed_wave(harness, monkeypatch, calls)
    revised = {**DECK_SPEC, "in_scope": ["a 6-slide deck"]}
    held = _call(ctx, spec=revised)
    assert held.startswith("ERROR: PLAN_REVIEW_IN_FLIGHT:") and fingerprint in held
    state = _state(harness)
    assert state["current_attempt"]["fingerprint"] == fingerprint
    assert state["current_attempt"].get("status") != "cycles_exhausted"
    assert not any(w.get("cycles_exhausted") for w in state["waves"]) and state["cycles_paid"] == 1
    gate = plan_review_gate_projection(state, "blocking")
    assert gate["status"] != "cycles_exhausted" and gate["allow"] is False and gate["custody_pending"] is True
    assert [c["reconcile_only"] for c in calls] == [False, True, True]  # nothing new was sent
    # The author's answer is still dispositionable on the (still current) wave.
    assert [d["decision"] for d in state["waves"][-1]["dispositions"]] == ["accept"]


def test_a_paid_wave_still_in_flight_leaves_room_for_a_second_panel_under_the_default_cap(harness, monkeypatch):
    calls = []
    ctx, fingerprint = _answered_mixed_wave(harness, monkeypatch, calls)  # the harness cap is 2
    dispatched = _call(ctx, spec={**DECK_SPEC, "in_scope": ["a 6-slide deck"]})
    assert _control(dispatched) == {"outcome": "DEGRADED", "closed": False}
    state = _state(harness)
    assert state["current_attempt"]["fingerprint"] != fingerprint
    assert calls[-1]["reconcile_only"] is False and calls[-1]["drain"] is not None  # a real second panel
    assert state["cycles_paid"] == 1  # the new barrier wave is unproven until its collection


# ------------------------------------------------------------- two waves in flight (fix cycle 3, 3c)


def _two_waves_in_flight(harness, monkeypatch, calls):
    """Cap 2: E1 dispatches W1; while W1 runs, a revised E2 dispatches W2 (room in the cap)."""
    _install_barrier_substrate(monkeypatch, calls, still_pending={"s1", "s2", "s3"})
    ctx = harness.make_ctx()
    _call(ctx)
    w1 = _state(harness)["waves"][-1]["request_fingerprint"]
    _call(ctx, spec={**DECK_SPEC, "in_scope": ["a 6-slide deck"]})
    state = _state(harness)
    w2 = state["current_attempt"]["fingerprint"]
    assert w1 != w2 and {w["request_fingerprint"]: w["custody_pending"] for w in state["waves"]} == {w1: True, w2: True}
    return ctx, w1, w2


def _persist_historical_barrier_outcomes(harness, calls, fingerprint, *, refused=False):
    """The former synthetic collector had no producer CAS; historical recovery
    now needs the same exact source a real released worker always writes."""
    from dataclasses import asdict
    from ouroboros.observability import persist_call
    from ouroboros.review_dispatch import review_operation_binding
    from ouroboros.review_records import ReviewActorRecord

    call = next(c for c in calls if c["_request"].reconciliation_identity["subject_hash"] == fingerprint)
    request = call["_request"]
    for slot in call["_slots"]:
        op = f"op-{fingerprint[:8]}-{slot.slot_id}"
        binding = review_operation_binding(request, slot, op)
        actor = ReviewActorRecord(slot.slot_id, slot.model, status="not_dispatched" if refused else "ok",
                                  error="not started" if refused else "", operation_id=op,
                                  operation_state="not_dispatched" if refused else "settled", recovery_binding=binding)
        persist_call(harness.drive, task_id=request.task_id, call_id=op + "_prompt", call_type="review_prompt",
                     payload={"request": asdict(request), "slot": asdict(slot)},
                     manifest={"review_operation_binding": binding})
        persist_call(harness.drive, task_id=request.task_id, call_id=op + "_response", call_type="review_response",
                     payload={"producer_outcome": asdict(actor), "message": {"content": CLEAN},
                              "usage": {"physical_attempt_state": "released" if refused else "settled"}},
                     manifest={"producer_complete": True, "review_operation_binding": binding})


def test_a_third_envelope_collects_every_in_flight_wave_and_meets_the_cap_when_both_proved_a_dispatch(harness, monkeypatch):
    """Fix cycle 3, 3c (cap 2): E3 collects W1 AND W2 at $0 (not only the current W2);
    both prove a dispatch, so E3 meets the cap as CYCLES_EXHAUSTED instead of looping
    between a hold that names W1 and a STALE disposition on W1."""
    calls = []
    ctx, w1, w2 = _two_waves_in_flight(harness, monkeypatch, calls)
    _persist_historical_barrier_outcomes(harness, calls, w1)
    _install_barrier_substrate(monkeypatch, calls)  # every reviewer of both waves has settled
    third = _call(ctx, spec={**DECK_SPEC, "in_scope": ["a 7-slide deck"]})
    assert third.startswith("⚠️ PLAN_REVIEW_CYCLES_EXHAUSTED: 2 of 2 paid plan-review cycles are spent")
    state = _state(harness)
    by_fp = {w["request_fingerprint"]: w for w in state["waves"]}
    # #789: settlement of a historical wave does not rewrite its original verdict.
    assert not by_fp[w1]["closed"] and by_fp[w1]["aggregate"] == "DEGRADED"
    assert by_fp[w1]["paid"] and by_fp[w2]["closed"] and by_fp[w2]["paid"]
    assert len(by_fp[w1]["historical_supplements"]) == 3
    assert state["cycles_paid"] == 2 and not any(w["custody_pending"] for w in state["waves"])
    assert calls[-1]["reconcile_only"] is True  # nothing new was sent


def test_a_third_envelope_dispatches_when_the_collected_waves_proved_no_dispatch(harness, monkeypatch):
    calls = []
    ctx, w1, w2 = _two_waves_in_flight(harness, monkeypatch, calls)
    _persist_historical_barrier_outcomes(harness, calls, w1, refused=True)
    _install_barrier_substrate(monkeypatch, calls, refused={"s1", "s2", "s3"})  # both waves: typed $0 refusals
    third = _call(ctx, spec={**DECK_SPEC, "in_scope": ["a 7-slide deck"]})
    assert _control(third) == {"outcome": "DEGRADED", "closed": False}
    state = _state(harness)
    assert state["cycles_paid"] == 0 and state["current_attempt"]["fingerprint"] not in {w1, w2}
    assert calls[-1]["reconcile_only"] is False and calls[-1]["drain"] is not None  # a real third panel


def test_the_hold_names_only_the_identical_envelope_route_for_a_wave_that_is_not_current(harness, monkeypatch):
    """W1 keeps running while W2 settles: E3 collects both (W2 closes and pays, W1 stays
    pending), the cap is full, and the hold names W1, which is not the current wave, so it
    offers only the identical-envelope route; that route collects W1 once it settles."""
    calls = []
    ctx, w1, w2 = _two_waves_in_flight(harness, monkeypatch, calls)
    _install_barrier_substrate(monkeypatch, calls, pending_waves={w1})
    held = _call(ctx, spec={**DECK_SPEC, "in_scope": ["a 7-slide deck"]})
    assert held.startswith("ERROR: PLAN_REVIEW_IN_FLIGHT:") and w1 in held
    assert "plan_task(review_disposition=" not in held  # no route the model cannot take
    assert "a review_disposition cannot address it" in held and "resubmit its identical envelope" in held
    state = _state(harness)
    by_fp = {w["request_fingerprint"]: w for w in state["waves"]}
    assert by_fp[w2]["closed"] and by_fp[w2]["paid"] and by_fp[w1]["custody_pending"] is True
    assert state["cycles_paid"] == 1 and not any(w.get("cycles_exhausted") for w in state["waves"])
    _install_barrier_substrate(monkeypatch, calls)  # W1's reviewers settle
    resumed = _call(ctx)  # E1's identical envelope collects W1
    assert _control(resumed) == {"outcome": "GREEN", "closed": True} and _state(harness)["cycles_paid"] == 2
    assert _call(ctx, spec={**DECK_SPEC, "in_scope": ["a 7-slide deck"]}).startswith("⚠️ PLAN_REVIEW_CYCLES_EXHAUSTED")


def test_a_hold_after_collecting_a_wave_that_is_not_current_keeps_the_closed_authority(harness, monkeypatch):
    """Fix cycle 4, J1 (cap 2): E1 dispatches W1 on a slow slot, a revised E2 dispatches
    W2, and W2's $0 collection closes it GREEN. A third revised envelope arrives while W1
    still runs: it collects W1 at $0 first (the engine's resume path over W1's own inputs)
    and is then HELD at the cap. The hold writes nothing, so W2's closed authority must
    still be the current wave: the collection may not leave the pointer on W1, or the gate
    reopens, W2's acceptance claims unbind and the hold offers a disposition route that
    would make E1's superseded spec the closed authority."""
    from ouroboros.contracts.task_contract import effective_acceptance_claims
    from ouroboros.task_results import closed_plan_review_wave, plan_review_gate_projection

    calls = []
    ctx, w1, w2 = _two_waves_in_flight(harness, monkeypatch, calls)
    _install_barrier_substrate(monkeypatch, calls, pending_waves={w1})  # only W2's reviewers settled
    assert _control(_collect(ctx, w2)) == {"outcome": "GREEN", "closed": True}
    state = _state(harness)
    assert state["current_attempt"]["fingerprint"] == w2 and state["cycles_paid"] == 1
    assert plan_review_gate_projection(state, "blocking")["allow"] is True

    held = _call(ctx, spec={**DECK_SPEC, "in_scope": ["a 7-slide deck"]})
    assert held.startswith("ERROR: PLAN_REVIEW_IN_FLIGHT:") and w1 in held
    state = _state(harness)
    assert state["current_attempt"]["fingerprint"] == w2  # the collection left the pointer alone
    gate = plan_review_gate_projection(state, "blocking")
    assert gate["allow"] is True and gate["status"] == "closed" and gate["custody_pending"] is False
    closed = closed_plan_review_wave(state)
    assert closed is not None and closed["request_fingerprint"] == w2
    claims, source = effective_acceptance_claims({}, closed)
    assert source == "plan_review" and [c["claim"] for c in claims] == DECK_SPEC["acceptance_claims"]
    by_fp = {w["request_fingerprint"]: w for w in state["waves"]}
    assert by_fp[w1]["custody_pending"] is True and state["cycles_paid"] == 1
    # W1 is not the current wave, so the hold offers only the identical-envelope route.
    assert "plan_task(review_disposition=" not in held
    assert "a review_disposition cannot address it" in held and "resubmit its identical envelope" in held
