"""Wave detach and permits follow physical custody, never a logical drain."""
from __future__ import annotations

import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from ouroboros import review_custody
from ouroboros.observability import read_call_payload
from ouroboros.review_records import ReviewActorRecord, ReviewRequest, ReviewSlot
from ouroboros.review_substrate import ReviewCoordinator
from ouroboros.review_wave import ReviewWave, review_wave_scope


def test_one_consumed_mail_event_releases_every_drain_but_keeps_physical_capacity(tmp_path, monkeypatch):
    from ouroboros.review_dispatch import review_operation_binding

    release, first_started = threading.Event(), threading.Event()
    operation = SimpleNamespace(author_detached=False)
    observed, sent = [], []
    ctx = SimpleNamespace()
    monkeypatch.setattr(review_custody, "current_review_operation", lambda: operation)
    monkeypatch.setattr("ouroboros.review_pause.DETACH_POLL_SEC", 0.01)
    def detach(_operation, entries):
        if not first_started.is_set() or observed:
            return False
        observed.append("one addressed event, then consumed")
        for entry in entries:
            entry.author_left_for = "addressed_event"
        return True
    monkeypatch.setattr(review_custody, "author_detaches", detach)
    def run_slot(self, request, slot, **kw):
        operation_id = kw["operation_id"]
        sent.append(slot.slot_id)
        first_started.set()
        assert release.wait(10)
        return ReviewActorRecord(slot.slot_id, slot.model, "ok", raw_text="[]\nNO_FINDINGS",
                                 operation_id=operation_id,
                                 recovery_binding=review_operation_binding(request, slot, operation_id))
    monkeypatch.setattr(ReviewCoordinator, "_run_slot", run_slot)
    requests = [ReviewRequest(surface="multi_model_review", task_id="wave", goal="g", retry_key="wave",
                              policy={"parts": ["result"]}) for _ in range(3)]
    slots = [ReviewSlot(str(i), "test/model", timeout_sec=30) for i in range(3)]
    coordinator = ReviewCoordinator(llm=object(), drive_root=tmp_path, usage_ctx=ctx)
    try:
        with review_wave_scope(1) as wave, ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(contextvars.copy_context().run, coordinator.run, request, [slot])
                       for request, slot in zip(requests, slots)]
            runs = [future.result(timeout=10) for future in futures]
            assert observed == ["one addressed event, then consumed"]
            assert all(run.actors[0]["outcome"]["status"] == "pending" for run in runs)
            assert len(sent) == wave.running == 1
            rejoined = coordinator.run(requests[0], [slots[0]])
            assert rejoined.actors[0]["operation_id"] == runs[0].actors[0]["operation_id"]
            assert len(sent) == wave.running == 1, "free rejoin and detach cannot acquire or release a worker permit"
            release.set()
            with wave.condition:
                assert wave.condition.wait_for(lambda: wave.running == 0 and all(e.event.is_set() for e in wave.entries.values()), timeout=10)
            assert sorted(sent) == ["0", "1", "2"]
            for request, slot, run in zip(requests, slots, runs):
                collected = coordinator.run(request, [slot])
                actor = collected.actors[0]
                assert actor["operation_id"] == run.actors[0]["operation_id"]
                assert actor["outcome"]["status"] == "responded"
                assert read_call_payload(tmp_path, **actor["outcome"]["source_ref"])[0]["producer_complete"]
            assert len(sent) == 3
    finally:
        release.set()


@pytest.mark.parametrize("pause", [True, False])
def test_latch_carries_pause_to_unsent_seats_after_resume(monkeypatch, tmp_path, pause):
    from ouroboros.owner_pause import launch_lock

    operation = SimpleNamespace(wait=object())
    monkeypatch.setattr("ouroboros.review_pause._author_coordinates", lambda _wait: (str(tmp_path), "root", "root"))
    wave = ReviewWave(1)
    initial = review_custody.ActiveReviewAttempt("first", "first", started_at="now")
    wave.register(initial)
    def first_observation(_op, entries):
        for entry in entries:
            entry.author_left_for = "" if pause else "addressed_event"
        return True
    assert wave.observe_detach(operation, first_observation)
    queued = review_custody.ActiveReviewAttempt("queued", "queued", started_at="now")
    wave.register(queued)
    # The owner already resumed; the first observation still governs this unsent episode.
    assert queued.pause_before_launch is pause
    with launch_lock(str(tmp_path), "root"):
        assert not queued.episode
    with wave.worker(queued) as admitted:
        assert admitted is not pause
        queued.event.set()
    assert wave.running == 0
    assert wave.observe_detach(operation, lambda *_a: pytest.fail("mail was consumed; do not peek again"))


def test_queued_seat_opens_its_original_window_when_capacity_arrives(tmp_path, monkeypatch):
    clock = [100.0]
    first_started, release = threading.Event(), threading.Event()
    deadlines = {}
    monkeypatch.setattr(review_custody, "monotonic_now", lambda *_a: clock[0])
    def run_slot(self, request, slot, **kw):
        if slot.slot_id == "first":
            first_started.set()
            assert release.wait(10)
        deadlines[slot.slot_id] = kw["logical_deadline_monotonic"]
        return ReviewActorRecord(slot.slot_id, slot.model, "ok", raw_text="[]\nNO_FINDINGS")
    monkeypatch.setattr(ReviewCoordinator, "_run_slot", run_slot)
    coordinator = ReviewCoordinator(llm=object(), drive_root=tmp_path)
    request = ReviewRequest(surface="multi_model_review", task_id="queued-clock", goal="g", policy={"parts": ["result"]})
    try:
        with review_wave_scope(1) as wave, ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(contextvars.copy_context().run, coordinator.run, request,
                [ReviewSlot("first", "m", timeout_sec=1), ReviewSlot("queued", "m", timeout_sec=1)])
            assert first_started.wait(10)
            # Register the queued seat before advancing the clock; no time threshold decides admission.
            with wave.condition:
                assert wave.condition.wait_for(lambda: len(wave.entries) == 2, timeout=10)
            clock[0] = 200.0
            release.set()
            run = future.result(timeout=10)
            with wave.condition:
                assert wave.condition.wait_for(lambda: wave.running == 0 and all(e.event.is_set() for e in wave.entries.values()), timeout=10)
        assert deadlines["queued"] == 201.0
        assert run.actors[1]["status"] == "ok"
    finally:
        release.set()


def test_async_fanout_shares_the_latch_and_physical_bound(tmp_path, monkeypatch):
    import asyncio
    from ouroboros import review_wave
    from ouroboros.tools import review, review_multi_model
    from ouroboros.tools.registry import ToolContext

    release, started = threading.Event(), threading.Event()
    sent, waves, seen = [], [], []
    current_wave = review_wave.current_review_wave
    def observe_wave():
        wave = current_wave()
        waves.append(wave)
        return wave
    monkeypatch.setattr(review_wave, "current_review_wave", observe_wave)
    monkeypatch.setattr(review_multi_model, "CONCURRENCY_LIMIT", 1)
    monkeypatch.setattr(review_multi_model, "triad_api_messages", lambda *_a, **_kw: ([], ""))
    monkeypatch.setattr(review, "LLMClient", object)
    monkeypatch.setattr(review_custody, "current_review_operation", lambda: SimpleNamespace(author_detached=False))
    monkeypatch.setattr("ouroboros.review_pause.DETACH_POLL_SEC", 0.01)
    def detach(_operation, entries):
        if not started.is_set() or seen:
            return False
        seen.append(True)
        for entry in entries:
            entry.author_left_for = "addressed_event"
        return True
    monkeypatch.setattr(review_custody, "author_detaches", detach)
    def run_slot(self, request, slot, **kw):
        sent.append(slot.slot_id)
        started.set()
        assert release.wait(30)
        return ReviewActorRecord(slot.slot_id, slot.model, "ok", raw_text="[]\nNO_FINDINGS")
    monkeypatch.setattr(ReviewCoordinator, "_run_slot", run_slot)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="fanout")
    try:
        result = asyncio.run(review_multi_model._multi_model_review_async(
            "subject", "review", ["m", "m", "m"], ctx, checking_source={},
            row_plan={"slot_ids": ["a", "b", "c"], "retrieves": [False] * 3,
                      "parts": [["result"]] * 3}))
        assert len(sent) == 1 and seen == [True]
        assert all(row["verdict"] == "PENDING" for row in result["results"])
        wave = waves[0]
        assert wave.capacity == wave.running == 1
        release.set()
        with wave.condition:
            assert wave.condition.wait_for(lambda: wave.running == 0 and all(e.event.is_set() for e in wave.entries.values()), timeout=30)
        assert len(sent) == 3 and all(item is wave for item in waves)
    finally:
        release.set()


@pytest.mark.parametrize("retained", [True, False])
def test_pending_error_projection_uses_the_original_prompt_when_retained(tmp_path, retained):
    from ouroboros.observability import persist_call, read_call_payload
    from ouroboros.review_records import ReviewRequest, ReviewSlot
    from ouroboros.review_substrate import ReviewCoordinator

    request, slot = ReviewRequest(surface="plan_review", task_id="author", goal="review"), ReviewSlot("seat", "model")
    if retained:
        original = persist_call(tmp_path, task_id="author", call_id="paid_prompt", call_type="review_prompt",
                                payload={"original": "assignment"})
    actor = ReviewCoordinator(llm=object(), drive_root=tmp_path)._error_actor(
        request, slot, "still running", operation_id="paid", operation_state="in_flight")
    assert actor.operation_id == "paid" and actor.late_result_pending
    if retained:
        assert actor.prompt_ref["manifest_ref"] == original["manifest_ref"]
        assert read_call_payload(tmp_path, task_id="author", call_id="paid_prompt")[1] == {"original": "assignment"}
    else:
        assert actor.prompt_ref["call_id"] != "paid_prompt"
        assert read_call_payload(tmp_path, task_id="author", call_id=actor.prompt_ref["call_id"])[0]["synthetic"]


def test_measurement_uses_the_context_root_first_and_never_stops_a_pending_drain(tmp_path):
    """A measurement lands in the context's canonical root when it has one and in the
    coordinator's custody root otherwise; a failing measurement of a pending seat leaves
    the drain and the seat's pending custody untouched."""
    from types import SimpleNamespace
    from unittest.mock import patch
    from ouroboros import subagent_metric_events
    from ouroboros.review_records import ReviewActorRecord, ReviewRequest, ReviewSlot
    from ouroboros.review_seat_outcome import finalize_seat_outcome
    from ouroboros.review_substrate import ReviewCoordinator

    request = ReviewRequest(surface="multi_model_review", task_id="metric-task", goal="review", policy={"parts": ["result"]})
    slot = ReviewSlot("seat", "model")
    actor = ReviewActorRecord("seat", "model", "ok", raw_text="[]\nNO_FINDINGS", operation_id="op-measured")
    finalize_seat_outcome(request, slot, actor)
    other = tmp_path / "other"
    with patch("ouroboros.subagent_metric_events.publish") as publish:
        subagent_metric_events.review_execution(request, slot, actor, None, fallback_root=tmp_path)
        assert publish.call_args.args[0] == tmp_path and publish.call_args.args[1]["usable_verdict"] is True
    with patch("ouroboros.subagent_metric_events.publish") as publish:
        subagent_metric_events.review_execution(request, slot, actor, SimpleNamespace(drive_root=other), fallback_root=tmp_path)
        assert publish.call_args.args[0] == other.resolve()
    pending = ReviewActorRecord("seat", "model", "pending", operation_id="op-pending",
                                late_result_pending=True, operation_state="in_flight")
    coordinator = ReviewCoordinator(llm=object(), drive_root=tmp_path)
    with patch.object(coordinator, "_finalize_outcome"), \
            patch("ouroboros.subagent_metric_events.review_execution", side_effect=OSError("metric unavailable")):
        coordinator._settle_actor(request, slot, pending, producer=False)
    assert pending.late_result_pending
