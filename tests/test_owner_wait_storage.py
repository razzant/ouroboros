"""A current worker's saved clock survives cold Resume and a contended warm park."""
import contextlib
import copy
import os
import queue as stdqueue
import time
from types import SimpleNamespace

import pytest

from ouroboros import budget_pause, model_sleep, owner_wait, store
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.model_wait import (
    ModelWaitInterrupted, TaskModelWait, execution_elapsed_seconds, operation_wait_scope,
)
from ouroboros.task_results import load_task_result, write_task_result
from supervisor import events_budget, worker_owner_wait
from supervisor.restart_retention import retain_sleep_checkpoint
from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
from tests.test_admission_contention import database_reader
from tests.test_execution_clock_resume_handover import clock  # noqa: F401
from tests.test_owner_wait_pool import pool as pool

pytestmark = pytest.mark.serial


def _context(root, task, started):
    ctx, limit = _loop_ctx(root, task["id"])
    ctx.root_task_id, ctx.task_started_at = task["root_task_id"], started
    ctx.pending_events = []
    ctx.task_metadata = {"execution_timeout_sec": task["execution_timeout_sec"]}
    ctx.model_wait_context = TaskModelWait(task=task, drive_root=root, event_queue=None, worker_slot_held=True)
    ctx.model_wait_context.tool_context = ctx
    ctx.model_wait_context.restore_continuation({}, started_at=started)
    return ctx, limit


def _checkpoint(ctx, limit):
    assert model_sleep.request_sleep(ctx, model_sleep.selectors(ctx), "warm")["reason"] == "sleep_armed"
    model_sleep.begin(ctx)
    return owner_wait.checkpoint_owner_wait(ctx, limit.messages, limit.llm_trace, {}, 4, [], set())


def _resumed_sleep(root, monkeypatch, now, *, descendant=False):
    queue, state, workers = _install_queue(root, monkeypatch)
    task_id, root_id = "sleeper", "tree-root" if descendant else "sleeper"
    if descendant:
        write_task_result(root, root_id, "running", root_task_id=root_id)
    write_task_result(root, task_id, "running", root_task_id=root_id)
    task = {"id": task_id, "type": "task", "root_task_id": root_id, "chat_id": 0,
            "drive_root": str(root), "budget_drive_root": str(root), "_attempt": 1,
            "execution_timeout_sec": 600}
    started = now[0] - 100
    ctx, limit = _context(root, task, started)
    original = _checkpoint(ctx, limit)
    owner_wait.set_owner_wait(root, task_id, {**original, "state": "waiting"})
    retained = retain_sleep_checkpoint(root, task_id, 1)
    assert "root_task_id" not in retained  # Exercise the real retained-warm producer.
    sources = {key: (row["source_ref"], read_actor_source_bytes(root, task_id, row["source_ref"]))
               for key, row in (("warm", original), ("cold", retained))}
    task["_budget_pause"] = budget_pause.exact_pause_marker(retained, default_root=root_id)
    workers.PENDING.append(task)
    now[0] += 30
    monkeypatch.setattr(state, "budget_remaining", lambda _state, **_kw: 5.0)
    granted = queue.resume_budget_paused_task(task_id)
    assert granted["ok"], granted
    handoff = copy.deepcopy(task["_budget_pause_resume"])
    assert handoff["authority"] == "explicit_resume"
    meta = {"task": copy.deepcopy(task), "worker_id": 0, "attempt": 1, "started_at": started,
            "budget_paused_sec": handoff["paused_duration_sec"],
            "sleep_parked_at": handoff["sleep_exclusion_since"]}
    ctx, limit = _context(root, task, started)
    ctx.budget_pause_resume = handoff
    saved = budget_pause.load_budget_pause(ctx)
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_kw: (None, "max"))
    now[0] += 20  # Granted but not yet consumed: still paused.
    budget_pause.resume_paused_loop(SimpleNamespace(_ctx=ctx), saved, limit.messages,
                                   limit.llm_trace, {}, set(), budget_remaining_usd=5.0)
    consumed = budget_pause.budget_pause_row(root, task_id)
    assert consumed["state"] == budget_pause.STATE_RESUMED and consumed["grant"]["consumed_at"]
    assert ctx.model_wait_context.budget_paused_sec == consumed["paused_duration_sec"] == 50
    now[0] += 10  # Actual work after Resume; its consumption event is deliberately lost.
    checkpoint = _checkpoint(ctx, limit)
    commands = stdqueue.Queue()
    worker = SimpleNamespace(wid=0, proc=SimpleNamespace(pid=os.getpid()), busy_task_id=task_id,
                             reaping=False, active_capacity=True, in_q=commands)
    workers.WORKERS[0], workers.RUNNING[task_id] = worker, meta
    workers.PENDING.remove(task)
    event = {"type": "owner_wait", "task_id": task_id, "worker_id": 0, "pid": os.getpid(),
             "task_attempt": 1, "wait_id": checkpoint["wait_id"], "phase": "park", "checkpoint": checkpoint}
    consumed_event = {"type": "budget_pause", "phase": "consumed", "task_id": task_id, "task_attempt": 1,
                      "pause_id": consumed["pause_id"], "grant_id": consumed["grant"]["grant_id"]}
    return SimpleNamespace(root=root, queue=queue, workers=workers, task=task, meta=meta, ctx=ctx, limit=limit,
                           event=event, consumed_event=consumed_event, consumed=consumed, sources=sources,
                           worker=worker, commands=commands, handoff=handoff, checkpoint=checkpoint)


@pytest.mark.parametrize("descendant", [False, True], ids=["root", "descendant"])
@pytest.mark.parametrize("nested_root", ["missing", "member", "foreign"])
def test_retained_warm_resume_and_next_park_use_the_current_worker_clock(
        tmp_path, monkeypatch, clock, descendant, nested_root):  # noqa: F811
    env = _resumed_sleep(tmp_path, monkeypatch, clock, descendant=descendant)
    row = copy.deepcopy(env.consumed)
    if nested_root != "missing":
        row["root_task_id"] = env.task["id"] if nested_root == "member" else "unrelated-metadata"
        budget_pause.set_budget_pause(tmp_path, env.task["id"], row)
    captured = copy.deepcopy(env.meta)
    assert events_budget.fold_consumed_resume(captured, tmp_path)
    assert execution_elapsed_seconds(captured, clock[0]) == 110
    assert not events_budget.fold_consumed_resume(captured, tmp_path)
    assert env.checkpoint["budget_paused_sec"] == 50
    worker_owner_wait.handle_owner_wait(env.event, env.workers)
    assert env.commands.get_nowait()["phase"] == "parked" and env.commands.empty()
    assert env.worker.active_capacity is False
    assert env.meta["budget_paused_sec"] == 50
    assert execution_elapsed_seconds(env.meta, clock[0]) == 110
    assert env.meta["task"]["_budget_pause_resume"] == {
        key: value for key, value in env.handoff.items() if key != "sleep_exclusion_since"}
    before_late_event = copy.deepcopy(env.meta)
    events_budget._handle_budget_pause(env.consumed_event, env.workers)
    assert env.meta == before_late_event  # The old notification cannot consume the new sleep.
    clock[0] += 70
    assert execution_elapsed_seconds(env.meta, clock[0]) == 110
    durable = load_task_result(tmp_path, env.task["id"], strict=True)
    assert durable["root_task_id"] == env.task["root_task_id"]
    assert durable["owner_wait"]["wait_id"] == env.checkpoint["wait_id"]
    assert durable["budget_pause"]["grant"] == env.consumed["grant"]
    for source, exact in env.sources.values():
        assert read_actor_source_bytes(tmp_path, env.task["id"], source) == exact


def test_contended_park_returns_without_ack_then_retries_the_same_checkpoint(tmp_path, monkeypatch, clock):  # noqa: F811
    env = _resumed_sleep(tmp_path, monkeypatch, clock)
    calls, original = [], worker_owner_wait.set_owner_wait

    def record(*args, **kwargs):
        assert store._WAIT_BUDGET.get() == 0
        calls.append(copy.deepcopy(args[2]))
        return original(*args, **kwargs)

    monkeypatch.setattr(worker_owner_wait, "set_owner_wait", record)
    before = copy.deepcopy(env.meta)
    with database_reader(tmp_path, exclusive=True):
        worker_owner_wait.handle_owner_wait(env.event, env.workers)
        assert env.worker.active_capacity and env.commands.empty()
        assert env.meta["task"] == before["task"]  # No selection/grant is spent by the live clock projection.
        assert env.meta["sleep_parked_at"] == env.checkpoint["sleep_started_at"]
        assert env.meta["budget_paused_sec"] == 50
        assert worker_owner_wait.has_owner_wait_checkpoint(env.meta, 1)
        with pytest.raises(store.StoreBusy):
            with store.open_store(tmp_path, write=False, wait_sec=0):
                pytest.fail("park returned only after the fixture lock ended")
        worker_owner_wait.retry_pending_owner_waits(env.workers)
        assert env.worker.active_capacity and env.commands.empty()
    worker_owner_wait.retry_pending_owner_waits(env.workers)
    assert env.commands.get_nowait()["phase"] == "parked"
    worker_owner_wait.retry_pending_owner_waits(env.workers)
    assert env.commands.empty() and len(calls) == 3
    assert all(row == calls[0] for row in calls)
    assert env.meta["owner_wait"]["wait_id"] == env.checkpoint["wait_id"]
    assert env.meta["budget_paused_sec"] == 50 and not env.worker.active_capacity
    assert budget_pause.budget_pause_row(tmp_path, env.task["id"])["grant"] == env.consumed["grant"]


@pytest.mark.parametrize("stale", ["worker", "attempt"])
def test_pending_park_cannot_write_for_a_replaced_physical_owner(tmp_path, monkeypatch, clock, stale):  # noqa: F811
    env = _resumed_sleep(tmp_path, monkeypatch, clock)
    with database_reader(tmp_path, exclusive=True):
        worker_owner_wait.handle_owner_wait(env.event, env.workers)
    if stale == "worker":
        env.worker.proc = SimpleNamespace(pid=os.getpid() + 100000)
    else:
        env.meta["attempt"] += 1
    monkeypatch.setattr(worker_owner_wait, "set_owner_wait", lambda *_a, **_kw: pytest.fail("stale park wrote"))
    worker_owner_wait.retry_pending_owner_waits(env.workers)
    assert env.commands.empty() and env.worker.active_capacity
    assert "owner_wait" not in env.meta


@pytest.mark.parametrize("clock_gap", ["absent", "lower"])
def test_missing_or_regressing_checkpoint_clock_stays_pending(tmp_path, monkeypatch, clock, clock_gap, caplog):  # noqa: F811
    env = _resumed_sleep(tmp_path, monkeypatch, clock)
    if clock_gap == "absent":
        env.ctx.model_wait_context = None
        env.event["checkpoint"] = owner_wait.checkpoint_owner_wait(
            env.ctx, env.limit.messages, {}, {}, 4, [], set())
        env.event["wait_id"] = env.event["checkpoint"]["wait_id"]
        assert "budget_paused_sec" not in env.event["checkpoint"]
    else:
        env.event["checkpoint"]["budget_paused_sec"] = env.meta["budget_paused_sec"] - 1
    worker_owner_wait.handle_owner_wait(env.event, env.workers)
    worker_owner_wait.retry_pending_owner_waits(env.workers)
    assert env.worker.active_capacity and env.commands.empty()
    assert "owner_wait" not in env.meta and worker_owner_wait.has_owner_wait_checkpoint(env.meta, 1)
    assert "clock unavailable or contradictory" in caplog.text


def test_consumed_observation_is_nonblocking_outside_queue_lock_and_revalidates_meta(tmp_path, monkeypatch, clock):  # noqa: F811
    env = _resumed_sleep(tmp_path, monkeypatch, clock)
    original = events_budget._resume_observation
    replace = [False]

    def observed(meta, root):
        assert not env.queue._queue_lock._is_owned()
        assert store._WAIT_BUDGET.get() == 0
        result = original(meta, root)
        if replace[0]:
            env.workers.RUNNING[env.task["id"]] = copy.deepcopy(env.meta)
        return result

    monkeypatch.setattr(events_budget, "_resume_observation", observed)
    before = copy.deepcopy(env.meta)
    with database_reader(tmp_path, exclusive=True):
        events_budget._handle_budget_pause(env.consumed_event, env.workers)
        assert env.meta == before
    replace[0] = True
    events_budget._handle_budget_pause(env.consumed_event, env.workers)
    assert env.meta == before and env.workers.RUNNING[env.task["id"]] == before
    replace[0] = False
    events_budget._handle_budget_pause(env.consumed_event, env.workers)
    assert execution_elapsed_seconds(env.workers.RUNNING[env.task["id"]], clock[0]) == 110
    assert "sleep_parked_at" not in env.workers.RUNNING[env.task["id"]]


@pytest.mark.parametrize("control", ["cancelled", "panic", "deadline"])
def test_worker_obeys_existing_controls_before_park_ack(tmp_path, monkeypatch, clock, control):  # noqa: F811
    from ouroboros.cancel_intents import request_cancel

    env = _resumed_sleep(tmp_path, monkeypatch, clock)
    if control == "cancelled":
        request_cancel(tmp_path, env.task["id"], source="pre-ack-test")
    elif control == "panic":
        from ouroboros import model_wait
        monkeypatch.setattr(model_wait, "current_model_wait", lambda: SimpleNamespace(control_reason=lambda: "panic"))
    else:
        env.ctx.model_wait_context.task["deadline_at"] = "2000-01-01T00:00:00+00:00"
    polls = []

    def no_ack(**_kwargs):
        polls.append(True)
        assert len(polls) == 1, "the existing control must end the first pre-ACK poll"
        raise stdqueue.Empty

    events = []

    def deliver(event):
        events.append(event)
        worker_owner_wait.handle_owner_wait(event, env.workers)

    with database_reader(tmp_path, exclusive=True):
        with operation_wait_scope(env.ctx.model_wait_context):
            with pytest.raises(ModelWaitInterrupted) as stopped:
                owner_wait.worker_owner_wait(0, SimpleNamespace(get=no_ack), SimpleNamespace(put=deliver),
                                             env.ctx, env.checkpoint)
        assert stopped.value.control_reason == control
        assert env.worker.active_capacity and env.commands.empty()
        assert len(events) == 1 and events[0]["phase"] == "park"


@pytest.mark.parametrize("consumption_delivered", [False, True], ids=["lost-consumption", "consumption-folded"])
def test_repeated_warm_sleep_excludes_pending_ack_and_keeps_current_closed_total(
        tmp_path, monkeypatch, clock, consumption_delivered):  # noqa: F811
    env = _resumed_sleep(tmp_path, monkeypatch, clock)
    model_sleep.end(env.ctx)  # The helper's prepared checkpoint has not yet parked; no time passed.
    if consumption_delivered:
        events_budget._handle_budget_pause(env.consumed_event, env.workers)
    for task in (env.task, env.meta["task"]):
        task["execution_timeout_sec"] = 130
    env.ctx.task_metadata["execution_timeout_sec"] = 130
    env.worker.proc.is_alive = lambda: True
    monkeypatch.setattr(env.workers, "repo_writer_task_allowed", lambda _task: True)
    original = owner_wait.checkpoint_owner_wait

    def serialize(*args, **kwargs):
        checkpoint = original(*args, **kwargs)
        clock[0] += 5  # Saving the checkpoint is work; sleep starts AFTER it.
        return checkpoint

    monkeypatch.setattr(owner_wait, "checkpoint_owner_wait", serialize)
    parks = []

    def park(_ctx, checkpoint):
        expected = 115 if not parks else 125
        parks.append(checkpoint)
        assert checkpoint["sleep_started_at"] == clock[0]
        event = {**env.event, "wait_id": checkpoint["wait_id"], "checkpoint": checkpoint}
        with database_reader(tmp_path, exclusive=True):
            worker_owner_wait.handle_owner_wait(event, env.workers)
            assert env.worker.active_capacity and env.commands.empty()
            assert execution_elapsed_seconds(env.meta, clock[0]) == expected
            clock[0] += 20  # Real SQLite lock stays held; fake time measures this interval's meaning.
            worker_owner_wait.retry_pending_owner_waits(env.workers)
            assert execution_elapsed_seconds(env.meta, clock[0]) == expected
            assert env.ctx.model_wait_context.executed_seconds() == expected
            assert env.worker.active_capacity and env.commands.empty()
        worker_owner_wait.retry_pending_owner_waits(env.workers)
        assert env.commands.get_nowait()["phase"] == "parked"
        assert not env.worker.active_capacity and env.meta["owner_wait"]["wait_id"] == checkpoint["wait_id"]
        assert execution_elapsed_seconds(env.meta, clock[0]) == expected
        clock[0] += 30
        worker_owner_wait.handle_owner_wait({**event, "phase": "resume"}, env.workers)
        assert worker_owner_wait._grant_resume(env.task["id"], env.meta, env.worker)
        clock[0] += 7  # The original worker has not received the grant/returned from its wait yet.
        assert env.commands.get_nowait()["phase"] == "resume_granted"
        return "mail:peer"

    env.ctx.owner_wait_callback = park
    for expected, paused in ((115, 107), (125, 164)):
        assert model_sleep.request_sleep(env.ctx, model_sleep.selectors(env.ctx), "warm")["reason"] == "sleep_armed"
        owner_wait.wait_after_tools(env.ctx, env.limit.messages, env.limit.llm_trace, {}, 8, [], set())
        assert env.ctx.model_wait_context.executed_seconds() == expected
        assert env.ctx.model_wait_context.budget_paused_sec == paused
        clock[0] += 5
    assert len(parks) == 2 and parks[0]["wait_id"] != parks[1]["wait_id"]
    assert parks[1]["budget_paused_sec"] == 107  # Includes the first worker's 7-second grant handover.
    assert budget_pause.budget_pause_row(tmp_path, env.task["id"])["grant"] == env.consumed["grant"]


@pytest.mark.parametrize("hard_axis", [None, "deadline", "absolute_ceiling"])
def test_pending_park_spares_only_idle_timeout(pool, monkeypatch, hard_axis):
    from supervisor import queue, workers

    # An independent real SQLite connection blocks the park. Release the physical
    # lock before enforcement, but leave the same unacknowledged park pending.
    with contextlib.closing(store.connect(pool.root / "state/current.sqlite", store.TIER_ENFORCED)) as conn:
        conn.execute("BEGIN EXCLUSIVE")
        try:
            worker_owner_wait.handle_owner_wait(pool.event, workers)
            assert pool.original.active_capacity and pool.original.in_q.empty()
            assert "owner_wait" not in pool.meta
        finally:
            conn.execute("ROLLBACK")
    monkeypatch.setattr(queue, "get_task_idle_timeout_sec", lambda: 10)
    monkeypatch.setattr(queue, "get_per_call_timeout_ceiling_sec", lambda: 1)
    monkeypatch.setattr(queue, "get_task_abs_ceiling_sec", lambda: 1000 if hard_axis == "absolute_ceiling" else 10000)
    monkeypatch.setattr(queue, "FINALIZATION_GRACE_SEC", 0)
    reaps = stdqueue.Queue()
    monkeypatch.setattr(queue, "_reap_queue", reaps)
    if hard_axis == "deadline":
        pool.meta["task"]["deadline_at"] = "2000-01-01T00:00:00+00:00"
    queue._enforce_task_timeouts_locked(workers, time.time(), 0, {})
    if hard_axis:
        job = reaps.get_nowait()
        assert job["terminal_reason"] == hard_axis and not {"will_retry", "retry_task_id"} & set(job)
    else:
        assert reaps.empty() and workers.RUNNING["owner"] is pool.meta
        assert pool.original.active_capacity and pool.original.in_q.empty()
