"""A native sleep returned by planned restart keeps one open interval and one clock."""
from __future__ import annotations

import copy
import os
import queue as stdqueue
from types import SimpleNamespace

import pytest

from ouroboros import model_sleep, owner_wait
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.model_wait import TaskModelWait, execution_elapsed_seconds, operation_wait_scope
from ouroboros.task_results import load_task_result, write_task_result
from supervisor import worker_owner_wait
from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
from tests._execution_clock_seams import agent_start
from tests.test_execution_clock_resume_handover import clock  # noqa: F401
from tests.test_resume_handover import _dispatch_one
from tests.test_warm_restart_retention import _handoff, _pool_events, _stop_boot

pytestmark = pytest.mark.serial


def _event(task_id, checkpoint, *, phase="park"):
    return {"type": "owner_wait", "task_id": task_id, "task_attempt": 1,
            "worker_id": 0, "pid": os.getpid(), "wait_id": checkpoint["wait_id"],
            "phase": phase, "checkpoint": checkpoint}


def _return_native(root, monkeypatch, now, *, bounded, reason="sleep"):
    from ouroboros import claudexor_daemon, llm

    def forbidden(*_a, **_kw):
        pytest.fail("native-clock test must not enter a provider or daemon")

    monkeypatch.setattr(llm.LLMClient, "chat", forbidden)
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", forbidden)
    queue, _state, workers = _install_queue(root, monkeypatch)
    _pool_events(workers, monkeypatch)
    monkeypatch.setattr(workers, "repo_writer_task_allowed", lambda _task: True)
    task_id = "saved-warm"  # the peer's real restart helpers share this identity
    task = {"id": task_id, "root_task_id": task_id, "type": "task", "chat_id": 0,
            "_attempt": 1, "_project_scope_none": True, "drive_root": str(root), "budget_drive_root": str(root)}
    if bounded:
        task["execution_timeout_sec"] = 180
    write_task_result(root, task_id, "running", root_task_id=task_id)
    started = now[0] - 100
    ctx, limit = _loop_ctx(root, task_id)
    ctx.task_started_at, ctx.pending_events = started, []
    waiter = ctx.model_wait_context = TaskModelWait(task=task, drive_root=root, event_queue=None, worker_slot_held=True)
    waiter.tool_context = ctx
    waiter.restore_continuation({}, started_at=started)
    messages = copy.deepcopy(limit.messages)
    commands = stdqueue.Queue()
    workers.WORKERS[0] = SimpleNamespace(wid=0, proc=SimpleNamespace(pid=os.getpid()),
        busy_task_id=task_id, reaping=False, active_capacity=True, in_q=commands)
    workers.RUNNING[task_id] = {"task": task, "attempt": 1, "worker_id": 0, "started_at": started}
    captured = []

    class StopTestStack(BaseException):
        pass

    def capture(_ctx, checkpoint):
        worker_owner_wait.handle_owner_wait(_event(task_id, checkpoint), workers)
        assert commands.get_nowait()["phase"] == "parked"
        captured.append(checkpoint)
        raise StopTestStack()

    ctx.owner_wait_callback = capture
    if reason == "sleep":
        assert model_sleep.request_sleep(ctx, model_sleep.selectors(ctx), "warm")["reason"] == "sleep_armed"
    else:
        ctx._owner_wait_requested = "original-quiz"
    with pytest.raises(StopTestStack):
        owner_wait.wait_after_tools(ctx, limit.messages, {}, {}, 4, [], {"seen-owner"})
    [checkpoint] = captured
    source = read_actor_source_bytes(root, task_id, checkpoint["source_ref"])
    now[0] += 60
    selected = _handoff(root, workers, checkpoint, ack=True)
    workers.WORKERS.clear()  # no real worker/process: the stop/restore owners run on retained records
    restored = _stop_boot(queue, workers, selected)
    assert restored["_owner_wait_resume"]["source_ref"] == checkpoint["source_ref"]
    now[0] += 30
    launched = _dispatch_one(SimpleNamespace(root=root, pending=workers.PENDING), monkeypatch)
    meta = workers.RUNNING[task_id]
    with monkeypatch.context() as start_patch:
        fresh_waiter = agent_start(launched, root, start_patch)
    fresh_waiter.closed = False  # The fixture's stop sentinel ended scope; this test now drives that restored stack.
    fresh = fresh_waiter.tool_context
    return SimpleNamespace(root=root, task_id=task_id, queue=queue, workers=workers,
        launched=launched, meta=meta, ctx=fresh, waiter=fresh_waiter, checkpoint=checkpoint,
        source=source, messages=messages, first_started=started, reason=reason)


def _run_native(env, monkeypatch, now):
    from ouroboros.owner_mailbox import write_owner_message

    commands = stdqueue.Queue()
    worker = env.workers.WORKERS[0]
    worker.proc = SimpleNamespace(pid=os.getpid(), is_alive=lambda: True)
    worker.in_q = commands
    phases, at_grant = [], []

    class Inbox:
        advanced = False

        def get(self, timeout):
            if not commands.empty():
                command = commands.get_nowait()
                phases.append(command["phase"])
                return command
            if not self.advanced:
                self.advanced = True
                now[0] += 60
                assert write_owner_message(env.root, "continue", env.task_id, msg_id="wake", kind="owner_text")
                raise stdqueue.Empty
            assert env.meta.get("owner_wait_resume_requested"), "the real worker must request its wake"
            assert worker_owner_wait._grant_resume(env.task_id, env.meta, worker)
            at_grant.append(execution_elapsed_seconds(env.meta, now[0]))
            now[0] += 7  # The worker still waits while the already-issued grant is delivered.
            command = commands.get_nowait()
            phases.append(command["phase"])
            return command

    def callback(ctx, checkpoint):
        return owner_wait.worker_owner_wait(0, Inbox(), SimpleNamespace(
            put=lambda event: worker_owner_wait.handle_owner_wait(event, env.workers)), ctx, checkpoint)

    env.ctx.owner_wait_callback = callback
    env.ctx.pending_events = []
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_kw: (None, "max"))
    saved = owner_wait.load_owner_wait(env.ctx)
    messages, trace, usage, seen = [], {}, {}, set()
    with operation_wait_scope(env.waiter):
        owner_wait.resume_native_loop(SimpleNamespace(_ctx=env.ctx), saved, messages, trace, usage, seen)
    return messages, phases, at_grant


@pytest.mark.parametrize("bounded", [False, True], ids=["unlimited", "explicit-budget"])
@pytest.mark.parametrize("second", ["sleep", "quiz"])
def test_planned_native_sleep_return_then_next_real_park(tmp_path, monkeypatch, clock, bounded, second):  # noqa: F811
    env = _return_native(tmp_path, monkeypatch, clock, bounded=bounded)
    assert env.meta["sleep_parked_at"] == env.checkpoint["sleep_started_at"]
    assert execution_elapsed_seconds(env.meta, clock[0]) == env.waiter.executed_seconds() == 100
    assert not env.waiter.control_reason()  # Even the 180s budget did not spend the 90s sleep/boot gap.
    messages, phases, at_grant = _run_native(env, monkeypatch, clock)
    assert phases == ["parked", "resume_granted"] and at_grant == [100]
    assert env.waiter.budget_paused_sec == env.ctx._budget_paused_sec == 157
    assert env.waiter.executed_seconds() == 100
    assert messages[:len(env.messages)] == env.messages
    assert read_actor_source_bytes(tmp_path, env.task_id, env.checkpoint["source_ref"]) == env.source
    assert not (load_task_result(tmp_path, env.task_id).get("budget_pause") or {}).get("grant")
    assert not env.queue.BUDGET_ROOT_FENCES and not env.ctx.owner_wait_resume
    clock[0] += 10
    env.ctx._loop_mailbox_seen_ids.add("wake")
    if second == "sleep":
        assert model_sleep.request_sleep(env.ctx, model_sleep.selectors(env.ctx, tasks=[env.task_id]), "warm")["reason"] == "sleep_armed"
    else:
        env.ctx._owner_wait_requested = "next-quiz"
    acknowledged = []

    class NextPark(BaseException):
        pass

    def next_park(_ctx, checkpoint):
        assert checkpoint["budget_paused_sec"] == 157
        worker_owner_wait.handle_owner_wait(_event(env.task_id, checkpoint), env.workers)
        command = env.workers.WORKERS[0].in_q.get_nowait()
        acknowledged.append(command["phase"])
        assert execution_elapsed_seconds(env.meta, clock[0]) == 110
        assert not (env.meta.get("owner_wait_park_pending") or {}).get("clock_error")
        raise NextPark()

    env.ctx.owner_wait_callback = next_park
    with pytest.raises(NextPark):
        owner_wait.wait_after_tools(env.ctx, messages, {}, {}, 5, [], {"wake"})
    assert acknowledged == ["parked"]


def test_native_quiz_keeps_explicit_queue_carry_in_both_clock_owners(tmp_path, monkeypatch, clock):  # noqa: F811
    env = _return_native(tmp_path, monkeypatch, clock, bounded=True, reason="owner")
    assert env.launched["_owner_wait_resume"]["budget_paused_sec"] == 30
    assert "sleep_parked_at" not in env.meta and env.waiter.sleep_started_monotonic is None
    assert env.waiter.executed_seconds() == execution_elapsed_seconds(env.meta, clock[0]) == 160
    saved = owner_wait.load_owner_wait(env.ctx)
    owner_wait.restore_continuation_state(SimpleNamespace(_ctx=env.ctx), saved, [], {}, {}, set())
    assert env.ctx._budget_paused_sec == env.waiter.budget_paused_sec == 30


def test_native_sleep_calendar_deadline_keeps_its_own_authority(tmp_path, monkeypatch, clock):  # noqa: F811
    import datetime
    from ouroboros import deadline_utils
    from ouroboros.model_wait import ModelWaitInterrupted

    env = _return_native(tmp_path, monkeypatch, clock, bounded=True)
    monkeypatch.setattr(deadline_utils, "utc_now", lambda: datetime.datetime.fromtimestamp(clock[0], datetime.timezone.utc))
    env.waiter.task["deadline_at"] = datetime.datetime.fromtimestamp(
        clock[0] + 30, datetime.timezone.utc).isoformat()
    events = []
    class ExpiredBeforeAck:
        def get(self, timeout):
            clock[0] += 60
            raise stdqueue.Empty
    env.ctx.pending_events = []
    env.ctx.owner_wait_callback = lambda ctx, checkpoint: owner_wait.worker_owner_wait(
        0, ExpiredBeforeAck(), SimpleNamespace(put=events.append), ctx, checkpoint)
    with operation_wait_scope(env.waiter), pytest.raises(ModelWaitInterrupted) as interrupted:
        owner_wait.resume_native_loop(SimpleNamespace(_ctx=env.ctx), owner_wait.load_owner_wait(env.ctx),
                                      [], {}, {}, set())
    assert interrupted.value.control_reason == "deadline" and [event["phase"] for event in events] == ["park"]
    assert env.waiter.executed_seconds() == 100 and env.waiter.control_reason() == "deadline"
    assert env.ctx._budget_paused_sec == env.waiter.budget_paused_sec == 150
    assert load_task_result(tmp_path, env.task_id)["owner_wait"]["state"] == "waiting"
