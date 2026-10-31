"""Recorded commit collection joins a live worker."""
from __future__ import annotations

import copy
import queue
import threading
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from ouroboros import model_wait, review_custody as custody, review_operation as operation
from ouroboros.observability import read_call_payload
from ouroboros.review_dispatch import review_operation_binding
from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request
from ouroboros.task_results import write_task_result
from ouroboros.tools.review_change_custody import collect_recorded_commit_actor
from tests.test_review_operation_lifetime import until


# A no-send assertion raised in a review worker must fail the owning test too.
pytestmark = pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")


@pytest.fixture
def live(tmp_path, monkeypatch):
    task = "recorded-live-commit"
    ctx = SimpleNamespace(task_id=task, task_attempt=None, drive_root=tmp_path,
                          task_metadata={}, pending_events=[])
    write_task_result(tmp_path, task, "running")
    request = ReviewRequest(surface="multi_model_review", goal="original instructions",
                            messages=[{"role": "user", "content": "original history"}],
                            task_id=task, retry_key="original-round")
    slot = ReviewSlot(slot_id="original-seat", model="original-model", timeout_sec=10,
                      max_tokens=1234, effort="high")
    sent, release = threading.Event(), threading.Event()
    calls, first = [], {}

    class Held:
        def chat(self, **kwargs):
            calls.append(kwargs)
            sent.set()
            assert release.wait(10)
            return {"content": "[]"}, {"prompt_tokens": 1, "completion_tokens": 1}

    def parent():
        return model_wait.task_model_wait_scope(task={"id": task, "_attempt": 1},
                                                drive_root=tmp_path, event_queue=None,
                                                worker_slot_held=False)

    def original():
        with parent():
            first["result"] = run_review_request(request, slots=[slot], drive_root=tmp_path,
                                                 usage_ctx=ctx, llm=Held())

    thread = threading.Thread(target=original)
    thread.start()
    assert sent.wait(10)
    key = custody._attempt_key(request, slot)
    with custody._ACTIVE_LOCK:
        entry = custody._ACTIVE[key]
    row = dict(slot_id=slot.slot_id, model=slot.model, status="error", operation_id=entry.operation_id,
               operation_state="in_flight", late_result_pending=True,
               awaiting_since=entry.started_at, recovery_binding=dict(entry.recovery_binding))
    ctx._review_frozen_rows = {request.surface: {slot.slot_id: row}}
    ctx._pending_review_attempt = SimpleNamespace(root_task_id=task,
                                                  review_contract_fingerprint=entry.recovery_binding["review_contract"])

    def no_send(*_args, **_kwargs):
        pytest.fail("a recorded live join created another worker or operation")

    monkeypatch.setattr("ouroboros.review_substrate.ReviewCoordinator._run_slot", no_send)
    monkeypatch.setattr(operation, "ReviewOperation", no_send)
    f = SimpleNamespace(ctx=ctx, request=request, slot=slot, entry=entry, row=row, key=key,
                        calls=calls, release=release, parent=parent)
    yield f
    monkeypatch.undo()
    # Restore test-injected racing entries before the original worker settles.
    with custody._ACTIVE_LOCK:
        if not entry.event.is_set():
            custody._ACTIVE[key] = entry
        elif custody._ACTIVE.get(key) is not entry:
            custody._ACTIVE.pop(key, None)
    release.set()
    thread.join(10)
    assert not thread.is_alive()
    until(lambda: entry.operation.closed)
    assert len(calls) == 1


def _collect(f):
    with f.parent():
        return collect_recorded_commit_actor(f.ctx, f.request.surface, f.slot.slot_id, f.request.retry_key)


def _start_collect(f):
    result = {}

    def run():
        try:
            result["actor"] = _collect(f)
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    return thread, result


def test_live_recorded_commit_joins_original_request_owner_and_worker(live, monkeypatch):
    f = live
    before = read_call_payload(f.ctx.drive_root, task_id=f.ctx.task_id,
                               call_id=f"{f.entry.operation_id}_prompt")
    assert not before[1]["request"]["deadline_at"] and before[1]["request"]["task_attempt"] is None
    f.ctx = copy.copy(f.ctx)
    f.ctx.task_metadata = {"root_task_id": "today-root", "deadline_at": "2000-01-01T00:00:00Z"}
    f.ctx.task_attempt = 7
    seen, joined = [], threading.Event()
    drain = custody._run_custodied_review_slots

    def observe(**kwargs):
        seen.append((asdict(kwargs["request"]), asdict(kwargs["slots"][0]),
                     operation.current_review_operation(), model_wait.current_model_wait()))
        # The collector has not joined merely because this observer ran.  Its
        # real drain still has to acquire the original entry and start the
        # relay; releasing the held producer before that point races its
        # settlement and makes the collector correctly report custody_lost.
        thread_start = threading.Thread.start

        def start_and_mark(thread, *args, **kwargs):
            thread_start(thread, *args, **kwargs)
            if thread.name.startswith("ouroboros-review-relay-"):
                joined.set()

        monkeypatch.setattr(threading.Thread, "start", start_and_mark)
        return drain(**kwargs)

    monkeypatch.setattr(custody, "_run_custodied_review_slots", observe)
    thread, result = _start_collect(f)
    try:
        assert joined.wait(5)
        assert not result and thread.is_alive()
        request, slot, owner, waiter = seen[0]
        assert request == {**before[1]["request"], "reconcile_only": True}
        assert slot == before[1]["slot"]
        assert owner is f.entry.operation and waiter is owner.wait
        f.release.set()
        thread.join(5)
        assert "error" not in result, result
        assert result["actor"]["status"] == "ok", result
        assert result["actor"]["operation_id"] == f.entry.operation_id
        assert result["actor"]["recovery_binding"] == {**f.row["recovery_binding"], "pending_invocation_id": ""}
        assert read_call_payload(f.ctx.drive_root, task_id=f.ctx.task_id,
                                 call_id=f"{f.entry.operation_id}_prompt") == before
    finally:
        f.release.set()
        thread.join(5)


@pytest.mark.parametrize("state", ["dead", "unknown", "alive", "foreign", "foreign_binding", "missing_source"])
def test_nonmatching_or_cold_operation_keeps_pure_collection(live, monkeypatch, state):
    f = live
    if state in {"dead", "unknown", "alive"}:
        with custody._ACTIVE_LOCK:
            custody._ACTIVE.pop(f.key)
        f.row["usage"] = {"review_controller": {"pid": 123, "birth": "saved"}}
        monkeypatch.setattr(operation, "controller_state", lambda _identity: state)
    elif state == "foreign":
        foreign = replace(f.entry, operation_id="another-operation")
        foreign.recovery_binding = review_operation_binding(f.request, f.slot, foreign.operation_id)
        monkeypatch.setitem(custody._ACTIVE, f.key, foreign)
    elif state == "foreign_binding":
        monkeypatch.setattr(f.entry, "recovery_binding", {**f.entry.recovery_binding, "root_task_id": "foreign"})
    else:
        f.row["operation_id"] = "missing-source"

    def no_runner(*_a, **_kw):
        pytest.fail("cold/foreign collection entered the waiting runner")

    monkeypatch.setattr("ouroboros.review_substrate.run_review_request", no_runner)
    actor = _collect(f)
    assert actor["operation_id"] == f.row["operation_id"]
    assert actor["operation_state"] == ("in_flight" if state in {"unknown", "alive", "foreign_binding"}
                                        else "custody_lost")


@pytest.mark.parametrize("race", ["disappeared", "replaced", "foreign_cache"])
def test_live_hint_race_cannot_join_foreign_operation_or_start_replacement(live, monkeypatch, race):
    f = live

    def raced(request, **kwargs):
        with custody._ACTIVE_LOCK:
            if race in {"disappeared", "foreign_cache"}:
                custody._ACTIVE.pop(f.key)
                if race == "foreign_cache":
                    from ouroboros.review_records import ReviewActorRecord

                    kwargs["usage_ctx"]._review_settled_attempts = {f.key: ReviewActorRecord(
                        slot_id=f.slot.slot_id, model=f.slot.model, status="ok", operation_id="replacement",
                        recovery_binding=review_operation_binding(f.request, f.slot, "replacement"))}
            else:
                foreign = replace(f.entry, operation_id="replacement")
                foreign.recovery_binding = review_operation_binding(f.request, f.slot, foreign.operation_id)
                custody._ACTIVE[f.key] = foreign
        return run_review_request(request, **kwargs)

    monkeypatch.setattr("ouroboros.review_substrate.run_review_request", raced)
    actor = _collect(f)
    assert actor["operation_state"] == "custody_lost"
    assert actor["operation_id"] == f.row["operation_id"]


@pytest.mark.parametrize("damage", ["none", "foreign_root", "foreign_binding", "missing_live", "not_preparing"])
def test_missing_prompt_only_joins_exact_retained_preparation(live, monkeypatch, damage):
    f = live
    f.row["operation_state"] = "pending_dispatch"
    if damage == "foreign_root":
        f.ctx._pending_review_attempt.root_task_id = "another-root"
    elif damage == "foreign_binding":
        f.row["recovery_binding"]["review_contract"] = "another-contract"
    elif damage == "missing_live":
        monkeypatch.delitem(custody._ACTIVE, f.key)
    elif damage == "not_preparing":
        f.row["operation_state"] = "in_flight"

    def missing(*_args, **_kwargs):
        raise FileNotFoundError("prompt not persisted")

    monkeypatch.setattr(operation, "recorded_review_assignment", missing)
    if damage != "none":
        assert _collect(f)["operation_state"] == "custody_lost"
        return
    thread, result = _start_collect(f)
    try:
        until(lambda: f.entry.operation._drains == 2 or result)
        assert not result and thread.is_alive()
        f.release.set()
        thread.join(5)
        assert result["actor"]["operation_id"] == f.entry.operation_id
    finally:
        f.release.set()
        thread.join(5)


def test_live_rejoin_uses_remaining_original_window_and_does_not_refresh_it(live, monkeypatch):
    f = live
    deadline = f.entry.logical_deadline
    now, waits = [deadline - .15], []

    class ExpiringQueue(queue.Queue):
        def get(self, block=True, timeout=None):
            # Consume only the remaining saved window, including any poll slices.
            assert timeout is not None and 0 < timeout <= deadline - now[0]
            waits.append(timeout)
            now[0] += timeout
            raise queue.Empty

    # Replace only custody's queue reference: the original worker stays held,
    # and its already-created queue and other modules' queues remain real.
    monkeypatch.setattr(custody, "queue", SimpleNamespace(Queue=ExpiringQueue, Empty=queue.Empty))
    monkeypatch.setattr(custody, "monotonic_now", lambda slot=None: now[0])
    actor = _collect(f)
    assert sum(waits) == pytest.approx(.15)
    assert actor["operation_id"] == f.entry.operation_id and actor["operation_state"] == "in_flight"
    waits.clear()
    actor = _collect(f)
    assert actor["operation_id"] == f.entry.operation_id and actor["operation_state"] == "in_flight"
    assert waits == []
    assert f.entry.logical_deadline == deadline
    assert not f.entry.event.is_set()
    assert len(f.calls) == 1


@pytest.mark.parametrize("stop", ["cancel", "owner_restart"])
def test_stop_during_recorded_live_join_reaches_original_wait(tmp_path, monkeypatch, stop):
    from ouroboros import config
    from tests.test_review_operation_lifetime import QuotaModel

    monkeypatch.setattr(config, "CLAUDEXOR_MODEL_POLL_INTERVAL_SEC", .005)
    task = "recorded-stop"
    write_task_result(tmp_path, task, "running")
    ctx = SimpleNamespace(task_id=task, task_attempt=1, drive_root=tmp_path, task_metadata={}, pending_events=[])
    request = ReviewRequest(surface="multi_model_review", goal="original", task_id=task, retry_key="stop-round")
    slot = ReviewSlot(slot_id="original", model="claudexor::codex=exact-model", timeout_sec=10)
    model, original = QuotaModel(), {}

    def parent():
        return model_wait.task_model_wait_scope(task={"id": task, "_attempt": 1}, drive_root=tmp_path,
                                                event_queue=None, worker_slot_held=False)

    def run():
        with parent():
            original["result"] = run_review_request(request, slots=[slot], drive_root=tmp_path, usage_ctx=ctx, llm=model)

    first = threading.Thread(target=run)
    first.start()
    second = None
    try:
        until(lambda: model.calls)
        key = custody._attempt_key(request, slot)
        with custody._ACTIVE_LOCK:
            entry = custody._ACTIVE[key]
        until(lambda: entry.operation.wait.waiting_slots())
        row = dict(slot_id=slot.slot_id, model=slot.model, status="error", operation_id=entry.operation_id,
                   operation_state="in_flight", late_result_pending=True, recovery_binding=entry.recovery_binding)
        ctx._review_frozen_rows = {request.surface: {slot.slot_id: row}}
        f = SimpleNamespace(parent=parent, ctx=ctx, request=request, slot=slot)
        second, result = _start_collect(f)
        until(lambda: entry.operation._drains == 2 or result)
        assert not result
        if stop == "cancel":
            from ouroboros.cancel_intents import request_cancel

            request_cancel(tmp_path, task, reason="owner stop", source="test")
        else:
            from tests._work_stop_helpers import live_restart
            live_restart(tmp_path, monkeypatch)
        second.join(5)
        assert not second.is_alive() and "error" not in result, result
        assert result["actor"]["operation_id"] == entry.operation_id
        assert result["actor"]["status"] == "error"
        assert entry.operation.control() == ("cancelled" if stop == "cancel" else "owner_restart")
        assert len(model.calls) == 1
    finally:
        model.ready.set()
        first.join(5)
        if second is not None:
            second.join(5)


def test_a_disappearing_live_session_does_not_repost_its_pending_start(tmp_path, monkeypatch):
    from tests.test_commit_historical_collection import _CONTRACT, _RETRY, _TASK, _seat

    repo = tmp_path / "repo"
    repo.mkdir()
    row, _prompt = _seat(tmp_path, repo, "multi_model_review", "session", complete=False)
    request, slot = operation.recorded_review_assignment(
        tmp_path, task_id=_TASK, surface="multi_model_review", row=row, retry_key=_RETRY)
    key = custody._attempt_key(request, slot)
    entry = custody.ActiveReviewAttempt(key=key, operation_id=row["operation_id"],
                                        recovery_binding=review_operation_binding(request, slot, row["operation_id"]))
    ctx = SimpleNamespace(drive_root=tmp_path, task_id=_TASK, task_metadata={},
                          _review_frozen_rows={request.surface: {slot.slot_id: row}},
                          _pending_review_attempt=SimpleNamespace(root_task_id=_TASK,
                                                                   review_contract_fingerprint=_CONTRACT))
    monkeypatch.setitem(custody._ACTIVE, key, entry)
    sends = []

    def raced(request, **kwargs):
        custody._ACTIVE.pop(key)
        return run_review_request(request, **kwargs)

    def forbidden_worker(*_a, **_kw):
        sends.append("replacement")
        raise AssertionError("live join attempted an original-token replay")

    monkeypatch.setattr("ouroboros.review_substrate.run_review_request", raced)
    monkeypatch.setattr("ouroboros.review_substrate.ReviewCoordinator._run_slot", forbidden_worker)
    actor = collect_recorded_commit_actor(ctx, request.surface, slot.slot_id, _RETRY)
    assert actor["operation_state"] == "custody_lost"
    assert actor["operation_id"] == row["operation_id"]
    assert sends == []
