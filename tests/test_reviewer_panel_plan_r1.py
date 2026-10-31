"""Counted council completion leaves advisers' custody and late answers intact."""
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import owner_hurry, plan_record, review_ledger
from ouroboros.review_records import ReviewActorRecord
from ouroboros.tools import plan_ask
from tests.test_plan_task_tool import CLEAN, _record, _slots, _write, harness as _harness
from tests.test_plan_review_event_route import (
    _TASK, _custody_kwargs, _held_worker, _mailbox_entries, _plant_ask, _statuses,
)

harness = _harness
PENDING = {"status": "pending", "raw_text": "", "operation_state": "pending_dispatch", "late_result_pending": True}


def _adviser(harness):
    row = SimpleNamespace(subagent_id="critic", slot_id="critic", target_id="m/critic")
    harness.monkeypatch.setattr("ouroboros.reviewer_slot_config.catalog_review_row", lambda *_: row)
    harness.monkeypatch.setattr("ouroboros.reviewer_slot_config._delivery_slot", lambda *a, **kw: _slots(("critic", "m/critic"))[0])


@pytest.mark.parametrize("counted", [False, True])
@pytest.mark.parametrize("dispatched", [False, True])
def test_ask_dispatch_fact_keeps_its_counted_participant(harness, counted, dispatched):
    _adviser(harness)
    harness.state["slots"] = _slots(("critic" if counted else "s1", "m/a"))
    harness.install({"critic": CLEAN if dispatched else {"status": "not_dispatched", "operation_state": "not_dispatched"}})
    ctx = harness.make_ctx(force_plan=True)
    _write(ctx, ask=["critic"])
    participant = _record(harness)["asks"][0]["participants"][0]
    assert participant["owner_marked"] is counted and participant["dispatched"] is dispatched
    # A subsequent tick change must not promote or demote this ask's participant.
    harness.state["slots"] = _slots(("s1" if counted else "critic", "m/a"))
    decision = owner_hurry.plan_gate_decision(ctx, {}, enforcement="blocking")
    assert decision["allow"] is True
    assert decision["asks_dispatched"] == int(counted and dispatched)


def test_plan_not_reached_tracks_terminal_failure_not_pending(harness):
    sub = harness.install({"s1": PENDING, "s2": {"status": "error", "error": "provider failed"}})
    ctx = harness.make_ctx()
    _write(ctx, ask=["s1", "s2"])
    ask = _record(harness)["asks"][0]
    assert ask["panel"]["not_reached"] == [{"seat_id": "s2", "cause": "provider failed"}]
    actor = ReviewActorRecord(slot_id="s1", model="m/a", status="ok", raw_text=CLEAN, operation_state="settled")
    plan_ask.settle_released_reply(ctx, request=sub.calls[0]["request"], task_id=ctx.task_id,
                                  actor=actor, settled_wave={}, slot=sub.calls[0]["slots"][0])
    assert _record(harness)["asks"][0]["panel"]["not_reached"] == [{"seat_id": "s2", "cause": "provider failed"}]


def test_counted_completion_recovers_custody_answers_and_publishes_once(harness):
    _adviser(harness)
    sub = harness.install({"s1": PENDING, "s2": PENDING, "critic": PENDING})
    ctx = harness.make_ctx()
    _write(ctx, ask=["s1", "s2", "critic"])
    slots = {s.slot_id: s for s in sub.calls[0]["slots"]}
    actors = {sid: ReviewActorRecord(slot_id=sid, model=s.model, status="ok", raw_text=CLEAN,
                                     operation_state="settled") for sid, s in slots.items()}
    kwargs = dict(request=sub.calls[0]["request"], task_id=ctx.task_id, actor=actors["s1"], slot=slots["s1"],
                  settled_wave={}, settled_actors={"s2": {"actor": actors["s2"], "slot": slots["s2"]}})
    plan_ask.settle_released_reply(ctx, **kwargs)
    ask = _record(harness)["asks"][0]
    assert [p["status"] for p in ask["participants"]] == ["answered", "answered", "pending"]
    assert not plan_record.pending_asks(_record(harness))
    assert review_ledger.load_record(harness.drive, ask["record_id"])["state"] == "settled"
    frames = _mailbox_entries(harness.drive, ctx.task_id)
    assert len(frames) == 1 and "counted participants settled" in frames[0]["text"]
    assert "advisers still working: critic" in frames[0]["text"] and "still owed" not in frames[0]["text"]
    plan_ask.settle_released_reply(ctx, **kwargs)
    plan_ask.settle_released_reply(ctx, request=kwargs["request"], task_id=ctx.task_id, actor=actors["critic"],
                                  slot=slots["critic"], settled_wave={sid: "ok" for sid in slots})
    assert len(_mailbox_entries(harness.drive, ctx.task_id)) == 1
    assert _record(harness)["asks"][0]["participants"][-1]["reply"]["prose"] == "Looks sound to me."


def test_completion_frame_uses_the_answers_that_won_the_record_lock(harness):
    sub = harness.install({"s1": PENDING, "s2": PENDING})
    ctx = harness.make_ctx()
    _write(ctx, ask=["s1", "s2"])
    original = plan_record.claim_ask_completion
    def concurrent_reply(root, task_id, ask_id, custody):
        plan_ask.record_reply(None, state_root=root, task_id=task_id, ask_id=ask_id, seat_id="s2",
                              row={"slot_id": "s2", "model": "m/b", "text": CLEAN, "operation_state": "settled"})
        return original(root, task_id, ask_id, custody)
    harness.monkeypatch.setattr(plan_record, "claim_ask_completion", concurrent_reply)
    actor = ReviewActorRecord(slot_id="s1", model="m/a", status="ok", raw_text=CLEAN, operation_state="settled")
    plan_ask.settle_released_reply(ctx, request=sub.calls[0]["request"], task_id=ctx.task_id, actor=actor,
                                  slot=sub.calls[0]["slots"][0], settled_wave={})
    [frame] = _mailbox_entries(harness.drive, ctx.task_id)
    assert "2 answered" in frame["text"] and "still owed" not in frame["text"]


@pytest.mark.parametrize("second_counted", [False, True])
def test_real_custody_wakes_on_counted_completion_and_keeps_adviser_worker(tmp_path, second_counted):
    import ouroboros.review_custody as custody

    slots = _slots(("s1", "m/a"), ("s2", "m/b"))
    _plant_ask(tmp_path, slots)
    plan_record.update_participant(tmp_path, _TASK, ask_id="a1", seat_id="s2", facts={"owner_marked": second_counted})
    release = {s.slot_id: threading.Event() for s in slots}
    entered = {s.slot_id: threading.Event() for s in slots}
    calls, run_slot = _held_worker(slots, {}, release, entered)
    ctx = SimpleNamespace(drive_root=tmp_path, emit_progress_fn=lambda *_: None)
    request, kwargs = _custody_kwargs(tmp_path, surface="plan_review", retry_key=f"plan_ask:{_TASK}:a1",
                                      slots=slots, run_slot=run_slot, ctx=ctx, plan_ask_id="a1")
    request.drain_deadline = time.monotonic()
    try:
        first = custody.run_custodied_review_slots(**kwargs)
        assert all(a.late_result_pending for a in first)
        first_operation = next(a.operation_id for a in first if a.slot_id == "s1")
        assert all(e.wait(10) for e in entered.values())
        release["s1"].set()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with custody._ACTIVE_LOCK:
                first_finished = not any(e.operation_id == first_operation for e in custody._ACTIVE.values())
            if first_finished:
                break
            time.sleep(0.01)
        assert _statuses(tmp_path) == {"s1": "answered", "s2": "pending"}
        assert len(_mailbox_entries(tmp_path, _TASK)) == (0 if second_counted else 1)
        assert custody._RELEASED_WAVES, "the adviser still has physical custody"
        release["s2"].set()
        while time.monotonic() < deadline and custody._RELEASED_WAVES:
            time.sleep(0.01)
    finally:
        for event in release.values():
            event.set()
        # Join the existing operations even on assertion failure; never leave a worker.
        request.drain_deadline = None
        custody.run_custodied_review_slots(**kwargs)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with custody._ACTIVE_LOCK:
                if not any(e.wave_key == custody._wave_key(request) for e in custody._ACTIVE.values()):
                    break
            time.sleep(0.01)
    assert _statuses(tmp_path) == {"s1": "answered", "s2": "answered"}
    assert len(_mailbox_entries(tmp_path, _TASK)) == 1 and sorted(calls) == ["s1", "s2"]
