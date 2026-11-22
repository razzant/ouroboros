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
from ouroboros.queue_observation import load_queue_document
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
    worker_owner_wait.grant_warm_returns()
    assert pool.original.in_q.get_nowait()["phase"] == "resume_granted"
    assert pool.meta["budget_paused_sec"] == 220
    assert "sleep_parked_at" not in pool.meta
    snapshot = load_queue_document(pool.root)
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
    worker_owner_wait.grant_warm_returns()
    assert pool.original.in_q.get_nowait()["phase"] == "resume_granted"
    assert pool.meta["budget_paused_sec"] == 420
    assert execution_elapsed_seconds(pool.meta, clock[0]) == 85
    clock[0] += 15
    queue._enforce_task_timeouts_locked(workers, clock[0], 0, {})
    assert reaps.get_nowait()["terminal_reason"] == "absolute_ceiling"


@pytest.mark.parametrize("phase", ["park", "resume"])
def test_a_failed_publication_neither_refuses_a_warm_pause_park_nor_withholds_its_grant(pool, monkeypatch, clock, phase):  # noqa: F811
    """The warm owner-Pause park and its grant are store commits that mark the queue (Q spans memory only); a
    publication that fails changes neither: the park stands with its clock facts (sleep, not execution) and the
    grant hands the lent capacity back with the confirmed paused total, never double counted."""
    started = clock[0] - 100
    pool.meta.update(started_at=started, budget_paused_sec=20.0)
    wait = {**pool.wait, "started_at": started, "reason": "owner_pause", "budget_paused_sec": 20.0,
            "owner_pause": {"parked_at": deadline_utils.utc_now().isoformat()}}
    event = {**pool.event, "checkpoint": wait}
    if phase == "park":
        monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **_kw: False)
    worker_owner_wait.handle_owner_wait(event, workers)
    assert pool.original.in_q.get_nowait()["phase"] == "parked"
    assert not pool.original.active_capacity and pool.meta["sleep_parked_at"] == clock[0]
    assert worker_owner_wait.waiting_on_owner(pool.meta, 3)
    assert load_task_result(pool.root, "owner")["owner_wait"]["state"] == "waiting"
    clock[0] += 200
    assert execution_elapsed_seconds(pool.meta, clock[0]) == 80, "the parked interval is sleep, not execution"
    if phase == "park":
        return
    worker_owner_wait.handle_owner_wait({**event, "phase": "resume"}, workers)
    with monkeypatch.context() as failed:
        failed.setattr(queue, "persist_queue_snapshot", lambda **_kw: False)
        worker_owner_wait.grant_warm_returns()
    assert pool.original.in_q.get_nowait()["phase"] == "resume_granted"
    assert pool.original.active_capacity and pool.meta["budget_paused_sec"] == 220
    assert "sleep_parked_at" not in pool.meta
    assert execution_elapsed_seconds(pool.meta, clock[0]) == 80
    assert load_task_result(pool.root, "owner")["owner_wait"]["state"] == "resumed"


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


@pytest.mark.parametrize("bound", ["deadline_passed", "lifetime_exhausted", "still_eligible"])
def test_warm_resume_checks_the_clock_before_accepting_a_lost_start(tmp_path, monkeypatch, clock, bound):
    from ouroboros import delegate_custody
    from supervisor.owner_pause_control import request_owner_pause
    from tests._owner_pause_ended_helpers import _pending, _resume, _tail_tree

    tid = "root-1"
    _, pool_workers, fake_gateway = _tail_tree(tmp_path, monkeypatch, tail=lambda root: _pending(root, tid, "lost"))
    assert request_owner_pause(tid, request_id="fixture-pause")["ok"]
    ctx, limit = _loop_ctx(tmp_path, tid)
    fence = owner_pause.read_fence(tmp_path, tid)
    when = clock[0]
    wait = owner_wait.checkpoint_owner_wait(ctx, limit.messages, limit.llm_trace, limit.accumulated_usage,
        limit.round_idx, limit.tool_schemas, limit.owner_msg_seen,
        pause={"fence_id": fence["fence_id"], "generation": fence["generation"], "detached_reviews": ["review"]})
    meta = pool_workers.RUNNING[tid]
    meta["owner_wait"] = owner_wait.set_owner_wait(tmp_path, tid, {**wait, "state": "waiting"})
    meta.update(started_at=when - 100.0, sleep_parked_at=when, budget_paused_sec=15.0,
                model_wait_quota_clock={"elapsed_sec": 5.0, "observed_at": when})
    expired = [False]
    monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: 80.0 if expired[0] else 100.0)
    if bound == "deadline_passed":
        meta["task"]["deadline_at"] = dt.datetime.fromtimestamp(when + 1, dt.timezone.utc).isoformat()
    choice, = _resume("owner", selected=tid)["acceptable_unknown"]
    clock[0] += 200.0  # Only the calendar deadline runs during the warm wait.
    expired[0] = bound == "lifetime_exhausted"  # A changed ceiling must be read on the confirm.
    resumed = _resume("owner", [choice], selected=tid)
    decisions = delegate_custody.invocation_record(tmp_path, "lost").get("resume_decisions") or {}
    assert not fake_gateway.starts
    if bound == "still_eligible":
        assert resumed["ok"] and resumed["warm"] and decisions[tid]["creation"] == "lookup_only", resumed
    else:
        assert not resumed.get("ok") and resumed["error"] == bound and not decisions, resumed
        assert owner_pause.read_fence(tmp_path, tid) == fence


def test_owner_mail_arriving_during_a_warm_selected_answer_park_holds_the_answer(tmp_path, monkeypatch):
    """A stack that waited under an owner Pause with its selected answer re-enters the final's
    owner-input check on return: my human's words accepted while it was parked hold the retained
    answer for a revision round. Before the fix the loop returned the original answer after one
    model call, ignoring the correction accepted during the park."""
    import threading
    import time
    from types import SimpleNamespace

    from ouroboros import dialogue_store, owner_wait
    from ouroboros.chat_chain import format_address, row_address
    from ouroboros.owner_relay import quote_rows
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.task_results import write_task_result
    from supervisor.owner_pause_control import request_owner_pause
    from supervisor.steering import _handle_steer_task
    from tests._budget_pause_exact_helpers import _install_queue, _supervisor_ctx
    from tests._delivery_candidate_shared import _run_loop
    from tests._owner_pause_ended_helpers import _resume

    def context(root, row):
        metadata = {"origin_message_ref": build_owner_message_ref(
            chat_id=row["chat_id"], client_message_id=row["client_message_id"], ts=row["ts"], text=row["text"])}
        return SimpleNamespace(task_id="parent1", drive_root=root, budget_drive_root=root,
                               current_chat_id=1, task_metadata=metadata, pending_events=[], event_queue=None,
                               drive_logs=lambda: root / "logs")

    tid = "parent1"
    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 5.0)
    task = {"id": tid, "root_task_id": tid, "type": "task", "chat_id": 2}
    write_task_result(tmp_path, tid, "running", root_task_id=tid)
    workers.RUNNING[tid] = {"task": task, "worker_id": 0, "attempt": 1, "started_at": time.time()}
    queue.ACCEPTANCE_FENCES[tid] = {"status": "active", "owner_message_generation": 2}
    parks = []

    def pause_after_selection(*_args):
        if not parks:
            assert request_owner_pause(tid, request_id="answer-park")["ok"]
        return ""

    def resume_with_mail(ctx, checkpoint):
        parks.append(checkpoint["owner_pause"]["resume_point"]["phase"])
        wait = owner_wait.set_owner_wait(tmp_path, tid, {**checkpoint, "state": "waiting"})
        workers.RUNNING[tid]["owner_wait"] = wait
        row = {"chat_id": 1, "direction": "in", "ts": "2026-11-19T00:00:00Z",
               "text": "Replace the selected answer with revised.", "client_message_id": "during-park",
               "attachments": []}
        dialogue_store.append(tmp_path, "chat", row)
        quotes, error = quote_rows(context(tmp_path, row), [format_address(row_address(row))])
        assert not error
        supervisor = _supervisor_ctx(tmp_path, workers, queue, [], [])
        supervisor.bridge = None
        _handle_steer_task({"target_task_id": tid, "message": "New words from the owner", "quotes": quotes,
                            "relay_id": "relay-1", "routing_token": "relay-1", "chat_id": 1,
                            "client_message_id": "during-park", "issuer": {"kind": "task"}}, supervisor)
        assert queue.ACCEPTANCE_FENCES[tid]["owner_message_generation"] == 3
        grant = _resume("owner", selected=tid)
        assert grant["ok"] and grant["warm"], grant
        return "control:owner_resume"

    def configure(ctx):
        ctx.task_attempt, ctx.task_id, ctx.budget_drive_root = 1, tid, tmp_path
        ctx.owner_wait_callback = resume_with_mail
        # As the agent binds every task: the final's owner-input check runs under this admission.
        ctx.owner_message_admission_lock = threading.Lock()
        ctx.owner_message_admission_agent = SimpleNamespace(_busy=True, _current_task_id=tid,
                                                            _accepting_owner_messages=True)

    monkeypatch.setattr("ouroboros.place_reply.routeless_note", pause_after_selection)
    monkeypatch.setattr("ouroboros.review_pause.live_detached_operations", lambda _: [{"operation_id": "critic"}])
    response, _, _, calls = _run_loop(tmp_path, monkeypatch, ["original", "revised"], setup_context=configure)
    assert parks == ["candidate"]
    assert response == "revised" and len(calls) == 2, "the parked answer holds; a round reads the owner's words"
