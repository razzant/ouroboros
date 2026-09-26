"""Required waiting preserves continuation and never grants itself capacity."""

import json
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.owner_wait import (
    checkpoint_owner_wait, load_owner_wait, restore_owner_wait_allowed,
    set_owner_wait, wait_after_tools, worker_owner_wait,
)
from ouroboros.task_results import write_task_result


def context(tmp_path):
    write_task_result(tmp_path, "root-1", "running")
    return SimpleNamespace(task_id="root-1", task_attempt=1, drive_root=tmp_path,
                           budget_drive_root=str(tmp_path), _owner_wait_requested="q1",
                           _owner_directives=[{"text": "Keep the original form"}],
                           _loop_mailbox_seen_ids=set(), pending_events=[],
                           active_model="test-model", active_effort="high",
                           active_use_local=False, active_context_mode="max",
                           task_started_at=time.time() - 10)


def test_checkpoint_keeps_completed_effects_and_owner_context(tmp_path):
    ctx = context(tmp_path)
    messages = [{"role": "assistant", "tool_calls": [{"id": "call1"}]},
                {"role": "tool", "tool_call_id": "call1", "content": "Created object 42"}]
    trace = {"tool_calls": [{"name": "create", "result": "42"}]}
    block = checkpoint_owner_wait(ctx, messages, trace, {"cost": 2.5}, 4, [], {"m1"})
    saved = json.loads(read_actor_source_bytes(tmp_path, "root-1", block["source_ref"]))
    assert saved["messages"] == messages and saved["trace"] == trace
    assert saved["owner_directives"] == ctx._owner_directives
    assert saved["seen"] == ["m1"] and saved["round_idx"] == 4
    set_owner_wait(tmp_path, "root-1", {**block, "state": "waiting"})
    assert load_owner_wait(ctx, {**block, "restart_transaction_id": "r1"}) == saved
    set_owner_wait(tmp_path, "root-1", {**block, "state": "resumed"}, block["wait_id"])
    with pytest.raises(ValueError, match="not an active"):
        load_owner_wait(ctx, {**block, "restart_transaction_id": "r1"})


def test_optional_question_never_enters_wait(tmp_path):
    ctx = context(tmp_path)
    ctx._owner_wait_requested = ""
    ctx.owner_wait_callback = lambda *_: pytest.fail("optional question parked")
    wait_after_tools(ctx, [], {}, {}, 1, [], set())


def test_wait_projection_does_not_overwrite_a_torn_result(tmp_path):
    from ouroboros.task_results import task_result_path

    path = task_result_path(tmp_path, "root-1")
    path.write_text('{"status":')
    with pytest.raises(ValueError):
        set_owner_wait(tmp_path, "root-1", {"wait_id": "w1", "state": "waiting"})
    assert path.read_text() == '{"status":'


def test_failed_checkpoint_does_not_release_capacity(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    ctx.owner_wait_callback = lambda *_: pytest.fail("capacity released without source")
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr("ouroboros.owner_wait.store_actor_source_bytes", fail)
    with pytest.raises(OSError, match="disk full"):
        wait_after_tools(ctx, [], {}, {}, 1, [], set())


@pytest.mark.serial
def test_deferred_question_flush_and_resume_requires_pool_grant(tmp_path):
    from ouroboros.owner_mailbox import write_owner_message

    ctx = context(tmp_path)
    ctx.pending_events = [{"type": "send_quiz", "task_id": "root-1", "quiz_id": "q1"}]
    commands, events, ended = queue.Queue(), queue.Queue(), threading.Event()
    checkpoint = {"wait_id": "w1", "quiz_id": "q1"}
    thread = threading.Thread(target=lambda: (worker_owner_wait(2, commands, events, ctx, checkpoint), ended.set()))
    thread.start()
    try:
        assert events.get(timeout=2)["type"] == "send_quiz"
        park = events.get(timeout=2)
        assert park["phase"] == "park" and ctx.pending_events == []
        identity = {key: park[key] for key in ("type", "task_id", "task_attempt", "wait_id")}
        commands.put({**identity, "phase": "parked"})
        write_owner_message(tmp_path, "Change the requested destination", task_id="root-1")
        resume = events.get(timeout=3)
        # An owner answer is never labelled a bound expiry.
        assert resume["phase"] == "resume" and resume["resume_reason"] == "owner_text"
        assert not ended.is_set()
        commands.put({**identity, "phase": "resume_granted"})
        thread.join(timeout=2)
        assert ended.is_set()
        assert ctx._loop_mailbox_seen_ids == set()  # ordinary loop drain still owns ACK
    finally:
        commands.put({"type": "owner_wait", "phase": "resume_granted", "task_id": "root-1", "task_attempt": 1, "wait_id": "w1"})
        thread.join(timeout=2)


@pytest.mark.serial
def test_a_spent_bound_asks_the_pool_for_the_same_resume(tmp_path):
    """The pooled lane needs no second scheduler: a spent bound is another
    reason to send the SAME resume request the mailbox already sends, and only
    from inside the parked gate (the supervisor ignores a resume for a wait that
    is not durably waiting yet)."""
    import datetime

    from ouroboros.deadline_utils import utc_now

    ctx = context(tmp_path)
    commands, events, ended = queue.Queue(), queue.Queue(), threading.Event()
    outcome = []
    checkpoint = {"wait_id": "w1", "quiz_id": "q1", "wait_max_minutes": 5,
                  "wait_deadline_at": (utc_now() - datetime.timedelta(seconds=1)).isoformat()}
    thread = threading.Thread(target=lambda: (
        outcome.append(worker_owner_wait(2, commands, events, ctx, checkpoint)), ended.set()))
    thread.start()
    try:
        park = events.get(timeout=2)
        assert park["phase"] == "park"
        identity = {key: park[key] for key in ("type", "task_id", "task_attempt", "wait_id")}
        with pytest.raises(queue.Empty):
            events.get(timeout=1.5)  # not parked yet: the bound must not ask
        commands.put({**identity, "phase": "parked"})
        resume = events.get(timeout=3)
        assert resume["phase"] == "resume" and resume["resume_reason"] == "timeout"
        assert not ended.is_set()  # a request, never a self-granted resume
        commands.put({**identity, "phase": "resume_granted"})
        thread.join(timeout=2)
        assert ended.is_set() and outcome == ["timeout"]
    finally:
        commands.put({"type": "owner_wait", "phase": "resume_granted", "task_id": "root-1",
                      "task_attempt": 1, "wait_id": "w1"})
        thread.join(timeout=2)


def test_the_granted_resume_row_records_the_bound_expiry(tmp_path, monkeypatch):
    """Cosmetic but readable: the worker's resume_reason reaches the durable row
    through the ordinary grant, so the projection can say a bound ended it."""
    from types import SimpleNamespace

    from ouroboros.task_results import load_task_result
    from supervisor import worker_owner_wait as supervisor_wait

    import supervisor.message_bus as mb
    from ouroboros.owner_quiz import quiz_states, record_asked

    record_asked(tmp_path, "root-1", quiz_id="q1", question="Which db?", options=["a", "b"],
                 assumption="a meanwhile", chat_id=1, wait_for_answer=True)
    frames: list = []
    bridge = mb.LocalChatBridge()
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(mb, "get_bridge", lambda: bridge)
    wait = {"wait_id": "w1", "task_attempt": 1, "quiz_id": "q1", "source_ref": "ref"}
    set_owner_wait(tmp_path, "root-1", {**wait, "state": "waiting"})
    worker = SimpleNamespace(in_q=queue.Queue(), busy_task_id="root-1",
                             proc=SimpleNamespace(pid=7), reaping=False, active_capacity=False)
    meta = {"worker_id": 0, "attempt": 1, "owner_wait": {**wait, "state": "waiting"},
            "task": {"id": "root-1", "chat_id": 1}, "started_at": time.time()}
    pool = SimpleNamespace(RUNNING={"root-1": meta}, WORKERS={0: worker}, DRIVE_ROOT=tmp_path)
    supervisor_wait.handle_owner_wait(
        {"type": "owner_wait", "task_id": "root-1", "worker_id": 0, "pid": 7,
         "task_attempt": 1, "wait_id": "w1", "phase": "resume", "resume_reason": "timeout"},
        pool,
    )
    assert meta["owner_wait_resume_requested"] is True
    assert meta["owner_wait"]["resume_reason"] == "timeout"
    monkeypatch.setattr(supervisor_wait, "_pool", lambda: pool, raising=False)
    monkeypatch.setattr(supervisor_wait, "_resume_allowed", lambda *_: True)
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda **_kw: True)
    pool.time = SimpleNamespace(time=time.time)
    assert supervisor_wait._grant_resume("root-1", meta, worker) is True
    row = load_task_result(tmp_path, "root-1")["owner_wait"]
    assert row["state"] == "resumed" and row["resume_reason"] == "timeout"
    # The bound closed: the card's projection and the live card both stop saying "waiting"
    # while the question stays open and answerable (astra round 3).
    block = quiz_states(tmp_path, "root-1")["q1"]
    assert block["state"] == "open" and "wait_for_answer" not in block and block["wait_ended_at"]
    [frame] = [f for f in frames if f.get("type") == "quiz_state"]
    assert frame["quiz_id"] == "q1" and frame["state"] == "open" and frame["wait_for_answer"] is False
    assert frame["chat_id"] == 1


def test_cold_wait_requires_observed_restart_and_current_wait(tmp_path):
    from ouroboros.utils import atomic_write_json

    ctx = context(tmp_path)
    block = checkpoint_owner_wait(ctx, [], {}, {}, 1, [], set())
    set_owner_wait(tmp_path, "root-1", {**block, "state": "waiting"})
    task = {"id": "root-1", "_owner_wait_resume": {**block, "restart_transaction_id": "tx"}}
    assert not restore_owner_wait_allowed(tmp_path, task)
    path = tmp_path / "state/delegate_recovery_transactions/tx.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, {"status": "normal_exit_acknowledged", "task_ids": ["root-1"]})
    assert restore_owner_wait_allowed(tmp_path, task)
    (tmp_path / "state/panic_stop.flag").write_text("panic")
    assert not restore_owner_wait_allowed(tmp_path, task)
