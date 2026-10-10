"""Warm owner Pause excludes author time through actual direct/pool consumers."""

from __future__ import annotations

import datetime as dt
import json
import queue as stdqueue
import time

import pytest

from ouroboros import config, deadline_utils, owner_pause, owner_wait
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.model_wait import TaskModelWait, execution_elapsed_seconds
from ouroboros.task_results import load_task_result, write_task_result
from supervisor import queue, worker_owner_wait, workers
from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
from tests.test_owner_wait_pool import pool  # noqa: F401


@pytest.fixture
def clock(monkeypatch):
    stamp = [1_800_000_000.0]
    monkeypatch.setattr(time, "time", lambda: stamp[0])
    monkeypatch.setattr(time, "monotonic", lambda: stamp[0])
    monkeypatch.setattr(deadline_utils, "utc_now", lambda: dt.datetime.fromtimestamp(stamp[0], dt.timezone.utc))
    monkeypatch.setattr(owner_wait, "utc_now_iso", lambda: deadline_utils.utc_now().isoformat())
    monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: 100.0)
    return stamp


def author(root, clock):
    ctx, limit = _loop_ctx(root, "author")
    ctx.task_started_at = clock[0] - 100.0
    ctx.pending_events = []
    ctx._budget_paused_sec = 20.0
    waiter = TaskModelWait(task={"id": "author", "budget_drive_root": str(root)},
                           drive_root=root, event_queue=None, worker_slot_held=False)
    waiter.restore_continuation({"budget_paused_sec": 20.0}, started_at=ctx.task_started_at)
    waiter.tool_context = ctx
    ctx.model_wait_context = waiter
    ctx.owner_wait_callback = owner_wait.direct_owner_wait
    write_task_result(root, "author", "running")
    return ctx, limit


@pytest.mark.parametrize("calendar_deadline", [False, True])
def test_direct_pause_excludes_only_author_interval_and_resumes_once(tmp_path, monkeypatch, clock, calendar_deadline):
    ctx, limit = author(tmp_path, clock)
    waiter = ctx.model_wait_context
    reviewer = TaskModelWait(task={"id": "reviewer", "budget_drive_root": str(tmp_path)},
                             drive_root=tmp_path, event_queue=None, worker_slot_held=False)
    if calendar_deadline:
        waiter.task["deadline_at"] = dt.datetime.fromtimestamp(clock[0] + 50, dt.timezone.utc).isoformat()
    fence, _ = owner_pause.install_fence(tmp_path, "author", request_id="first")

    def advance(_seconds):
        clock[0] += 200
        owner_pause.release_fence(tmp_path, "author", reason="owner_resume")

    monkeypatch.setattr(owner_wait.time, "sleep", advance)
    outcome = owner_wait.park_owner_pause_warm(limit, ctx, fence=fence, detached=["reviewer"])
    assert outcome == ("control:deadline" if calendar_deadline else "control:owner_resume")
    assert waiter.executed_seconds() == 80
    assert reviewer.executed_seconds() == 200, "reviewers keep their independent execution clock"
    assert reviewer.control_reason() == "absolute_ceiling"
    assert ctx._budget_paused_sec == waiter.budget_paused_sec == 220
    assert waiter.sleep_started_monotonic is None
    if calendar_deadline:
        assert waiter.control_reason() == "deadline"
        return
    clock[0] += 5
    assert waiter.executed_seconds() == 85
    fence, _ = owner_pause.install_fence(tmp_path, "author", request_id="second")
    assert owner_wait.park_owner_pause_warm(limit, ctx, fence=fence, detached=["reviewer"]) == "control:owner_resume"
    assert ctx._budget_paused_sec == waiter.budget_paused_sec == 420
    assert waiter.executed_seconds() == 85
    clock[0] += 15
    assert waiter.control_reason() == "absolute_ceiling", "Resume never grants additional active lifetime"


@pytest.mark.parametrize("calendar_deadline", [False, True])
def test_pool_timeout_does_not_charge_pause_and_duplicate_park_cannot_widen_it(pool, monkeypatch, clock, calendar_deadline):  # noqa: F811
    started = clock[0] - 100
    pool.meta.update(started_at=started, budget_paused_sec=20.0, last_heartbeat_at=clock[0])
    wait = {**pool.wait, "started_at": started, "reason": "owner_pause", "quiz_id": "",
            "budget_paused_sec": 20.0, "parked_at": deadline_utils.utc_now().isoformat(),
            "owner_pause": {"parked_at": deadline_utils.utc_now().isoformat()}}
    event = {**pool.event, "checkpoint": wait}
    monkeypatch.setattr(queue, "get_task_abs_ceiling_sec", lambda: 100.0)
    monkeypatch.setattr(queue, "FINALIZATION_GRACE_SEC", 0)
    monkeypatch.setattr(queue, "_ensure_reaper_started", lambda: None)
    reaps = stdqueue.Queue()
    monkeypatch.setattr(queue, "_reap_queue", reaps)
    if calendar_deadline:
        pool.meta["task"]["deadline_at"] = dt.datetime.fromtimestamp(clock[0] + 50, dt.timezone.utc).isoformat()
    # The park's event can wait in transit: its checkpoint dates the interval.
    clock[0] += 5
    worker_owner_wait.handle_owner_wait(event, workers)
    assert pool.original.in_q.get_nowait()["phase"] == "parked"
    clock[0] += 195
    worker_owner_wait.handle_owner_wait(event, workers)  # duplicate delivery must not reset the start
    assert pool.original.in_q.get_nowait()["phase"] == "parked"
    queue._enforce_task_timeouts_locked(workers, clock[0], 0, {})
    if calendar_deadline:
        assert reaps.get_nowait()["terminal_reason"] == "deadline"
        return
    assert reaps.empty(), "Paused authors must survive beyond their remaining active lifetime"
    assert execution_elapsed_seconds(pool.meta, clock[0]) == 80
    worker_owner_wait.handle_owner_wait({**event, "phase": "resume", "resume_reason": "control:owner_resume"}, workers)
    worker_owner_wait.maintain_owner_wait_capacity()
    assert pool.original.in_q.get_nowait()["phase"] == "resume_granted"
    assert pool.meta["budget_paused_sec"] == 220
    assert "sleep_parked_at" not in pool.meta
    snapshot = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())
    assert snapshot["running"][0]["budget_paused_sec"] == 220
    assert snapshot["running"][0]["execution_sec"] == 80
    worker_owner_wait.handle_owner_wait(event, workers)  # a consumed park is stale
    assert pool.original.in_q.empty()
    clock[0] += 5
    wait = {**wait, "wait_id": "second", "budget_paused_sec": 220,
            "owner_pause": {"parked_at": deadline_utils.utc_now().isoformat()}}
    event = {**event, "wait_id": "second", "checkpoint": wait}
    worker_owner_wait.handle_owner_wait(event, workers)
    assert pool.original.in_q.get_nowait()["phase"] == "parked"
    clock[0] += 200
    queue._enforce_task_timeouts_locked(workers, clock[0], 0, {})
    assert reaps.empty()
    worker_owner_wait.handle_owner_wait({**event, "phase": "resume", "resume_reason": "control:owner_resume"}, workers)
    worker_owner_wait.maintain_owner_wait_capacity()
    assert pool.original.in_q.get_nowait()["phase"] == "resume_granted"
    assert pool.meta["budget_paused_sec"] == 420
    assert execution_elapsed_seconds(pool.meta, clock[0]) == 85
    clock[0] += 15
    queue._enforce_task_timeouts_locked(workers, clock[0], 0, {})
    assert reaps.get_nowait()["terminal_reason"] == "absolute_ceiling"


@pytest.mark.parametrize("failure_phase", ["park", "resume"])
def test_failed_pool_snapshot_never_leaks_or_double_counts_pause(pool, monkeypatch, clock, failure_phase):  # noqa: F811
    started = clock[0] - 100
    pool.meta.update(started_at=started, budget_paused_sec=20.0)
    wait = {**pool.wait, "started_at": started, "reason": "owner_pause", "budget_paused_sec": 20.0,
            "owner_pause": {"parked_at": deadline_utils.utc_now().isoformat()}}
    event = {**pool.event, "checkpoint": wait}
    if failure_phase == "park":
        monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **_kw: False)
    worker_owner_wait.handle_owner_wait(event, workers)
    if failure_phase == "park":
        assert pool.original.in_q.get_nowait()["phase"] == "refused"
        assert "sleep_parked_at" not in pool.meta
        clock[0] += 200
        assert execution_elapsed_seconds(pool.meta, clock[0]) == 280
        return
    assert pool.original.in_q.get_nowait()["phase"] == "parked"
    clock[0] += 200
    worker_owner_wait.handle_owner_wait({**event, "phase": "resume"}, workers)
    with monkeypatch.context() as failed:
        failed.setattr(queue, "persist_queue_snapshot", lambda **_kw: False)
        worker_owner_wait.maintain_owner_wait_capacity()
    assert pool.original.in_q.empty()
    assert pool.meta["budget_paused_sec"] == 20
    assert execution_elapsed_seconds(pool.meta, clock[0]) == 80
    # The failed grant's maintenance tick replenished capacity. Its replacement
    # must finish the normal startup handshake before it can lend that slot back.
    workers.WORKERS[1].reaping = False
    worker_owner_wait.maintain_owner_wait_capacity()
    assert pool.original.in_q.get_nowait()["phase"] == "resume_granted"
    assert pool.meta["budget_paused_sec"] == 220
    assert execution_elapsed_seconds(pool.meta, clock[0]) == 80


def test_shutdown_preserves_warm_start_through_actual_budget_resume(tmp_path, monkeypatch, clock):
    from supervisor.restart_retention import park_saved_pause

    q, _state, _workers = _install_queue(tmp_path, monkeypatch)
    ctx, limit = author(tmp_path, clock)
    fence, _ = owner_pause.install_fence(tmp_path, "author", request_id="pause")
    pause_started = clock[0]
    captured = []

    class WorkerExited(Exception):
        pass

    def shutdown(_ctx, checkpoint):
        owner_wait.set_owner_wait(tmp_path, "author", {**checkpoint, "state": "waiting"})
        clock[0] += 200
        task = {"id": "author", "root_task_id": "author", "type": "task", "chat_id": 0, "_attempt": 1}
        parked = park_saved_pause(task, 1, tmp_path, pause_source="shutdown")
        assert parked and parked.get("_budget_pause")
        q.PENDING.append(parked)
        row = load_task_result(tmp_path, "author")["budget_pause"]
        captured.append(row)
        raise WorkerExited

    ctx.owner_wait_callback = shutdown
    with pytest.raises(WorkerExited):
        owner_wait.park_owner_pause_warm(limit, ctx, fence=fence, detached=["reviewer"])
    row = captured[0]
    assert row["paused_at"] == pause_started
    assert row["paused_duration_sec"] == 20
    state = json.loads(read_actor_source_bytes(tmp_path, "author", row["source_ref"]))
    assert state["model_wait"]["budget_paused_sec"] == 20
    clock[0] += 50
    result = q.resume_budget_paused_task("author")
    assert result["ok"], result
    granted = load_task_result(tmp_path, "author")["budget_pause"]["grant"]
    assert granted["paused_duration_sec"] == 270, "prior and warm/cold time are each excluded once"
