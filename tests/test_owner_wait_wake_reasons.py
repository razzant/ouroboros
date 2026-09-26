"""TZ-2 B2 — a wake is not an answer.

The typed ``resume_reason`` vocabulary, the F10 race guard (a card answered a
second before the wake is never rolled back), the host notice that explains a
non-answer wake (a hurry named as what it is), and the bound facts: the task's
own deadline is hard, an unbounded wait has no general limit.
"""

import datetime
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros.owner_mailbox import (
    KIND_HURRY, KIND_QUIZ_ANSWER, drain_owner_entries, write_owner_message, write_task_message,
)
from ouroboros.owner_quiz import mark_wait_ended, quiz_states, record_answered, record_asked
from ouroboros.owner_wait import (
    _fresh_wake, announce_wait_ended, append_wake_notice, classify_wake, owner_wait_ended_notice,
    owner_wait_timeout_notice, wait_after_tools, worker_owner_wait,
)
from ouroboros.task_results import load_task_result
from tests.test_native_owner_wait import native_context
from tests.test_owner_wait import context


def _bridge(monkeypatch):
    import supervisor.message_bus as mb

    frames: list = []
    bridge = mb.LocalChatBridge()
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(mb, "get_bridge", lambda: bridge)
    return frames


def _quiz_frames(frames):
    return [frame for frame in frames if frame.get("type") == "quiz_state"]


def _hurry(root, task_id):
    assert write_owner_message(root, "owner_hurry", task_id, msg_id="hurry:r1", kind=KIND_HURRY)


def _answer(root, task_id, quiz_id="q1"):
    assert write_owner_message(root, "Owner chose a", task_id,
                               msg_id=f"quiz_answer:{quiz_id}", kind=KIND_QUIZ_ANSWER)


def test_classify_wake_speaks_the_closed_vocabulary():
    answer = {"kind": KIND_QUIZ_ANSWER, "msg_id": "quiz_answer:q1"}
    other = {"kind": KIND_QUIZ_ANSWER, "msg_id": "quiz_answer:q0"}
    text = {"kind": "owner_text", "msg_id": "t"}
    hurry = {"kind": KIND_HURRY, "msg_id": "hurry:r1"}
    mail = {"kind": "task_message", "provenance": "independent_task", "source_task_id": "peer-1"}
    assert classify_wake([], "q1") == "unknown"
    assert classify_wake([answer], "q1") == "answer"
    assert classify_wake([other], "q1") == "owner_text"  # the owner spoke, but not to this card
    assert classify_wake([text], "q1") == "owner_text"
    assert classify_wake([hurry], "q1") == "hurry"
    assert classify_wake([mail], "q1") == "mail:peer-1"
    assert classify_wake([{"kind": "task_message", "provenance": "ancestor_task"}], "q1") == "mail:unknown"
    assert classify_wake([{"kind": "unlabelled"}], "q1") == "mail:unknown"  # closed vocabulary: never bare `mail`
    # Precedence: a control, this card's answer, owner words, hurry, then the mail that woke first.
    assert classify_wake([mail, hurry, text, answer, {"kind": "finalize_now"}], "q1") == "control:finalize_now"
    assert classify_wake([mail, hurry, text, answer], "q1") == "answer"
    assert classify_wake([mail, hurry, text], "q1") == "owner_text"
    assert classify_wake([mail, hurry], "q1") == "hurry"
    assert classify_wake([{**mail, "source_task_id": "peer-2"}, mail], "q1") == "mail:peer-2"


def test_fresh_wake_upgrades_only_on_owner_authority(tmp_path):
    ctx = native_context(tmp_path)
    assert _fresh_wake(ctx, "q1", "timeout") == "timeout"
    _hurry(tmp_path, ctx.task_id)
    assert _fresh_wake(ctx, "q1", "unknown") == "hurry"
    assert _fresh_wake(ctx, "q1", "timeout") == "timeout"  # a hurry does not rewrite why the wait ended
    _answer(tmp_path, ctx.task_id)
    assert _fresh_wake(ctx, "q1", "timeout") == "answer"
    assert _fresh_wake(ctx, "q1", "hurry") == "answer"


def test_a_hurry_wakes_the_direct_wait_but_never_reads_as_an_answer(tmp_path, monkeypatch):
    ctx = native_context(tmp_path)
    ctx.current_chat_id = 1
    record_asked(tmp_path, ctx.task_id, quiz_id="q1", question="Which path?", options=["a", "b"],
                 wait_for_answer=True, chat_id=1, assumption="a meanwhile")
    frames = _bridge(monkeypatch)
    monkeypatch.setattr("ouroboros.owner_wait.time.sleep", lambda _seconds: _hurry(tmp_path, ctx.task_id))
    messages = []
    wait_after_tools(ctx, messages, {}, {}, 1, [], set())
    row = load_task_result(tmp_path, ctx.task_id)["owner_wait"]
    assert row["state"] == "resumed" and row["resume_reason"] == "hurry"
    block = quiz_states(tmp_path, ctx.task_id)["q1"]
    assert block["state"] == "open" and "wait_for_answer" not in block and block["wait_ended_at"]
    [frame] = _quiz_frames(frames)
    assert frame["state"] == "open" and frame["wait_for_answer"] is False and frame["chat_id"] == 1
    [notice] = messages
    assert notice["role"] == "user" and notice["content"].startswith("[SYSTEM NOTICE]")
    assert "owner hurry request" in notice["content"] and "not an answer" in notice["content"]
    assert "not a confirmed owner answer" in notice["content"] and "remains open" in notice["content"]
    # The hurry control stays the loop's to apply; nothing here forged an answer.
    assert [entry["kind"] for entry in drain_owner_entries(tmp_path, ctx.task_id, set())] == [KIND_HURRY]


@pytest.mark.serial
@pytest.mark.parametrize("order", ["answer_before_parked", "answer_after_parked",
                                   "answer_before_parked_with_spent_bound"])
def test_an_answer_racing_the_park_is_reported_as_the_answer(tmp_path, order):
    """Both orders end the pooled wait with ``answer`` — never a bound expiry, never mail."""
    from ouroboros.deadline_utils import utc_now

    ctx = context(tmp_path)
    commands, events, ended = queue.Queue(), queue.Queue(), threading.Event()
    outcome = []
    checkpoint = {"wait_id": "w1", "quiz_id": "q1"}
    if order.endswith("spent_bound"):
        checkpoint.update({"wait_max_minutes": 5,
                           "wait_deadline_at": (utc_now() - datetime.timedelta(seconds=1)).isoformat()})
    thread = threading.Thread(target=lambda: (
        outcome.append(worker_owner_wait(2, commands, events, ctx, checkpoint)), ended.set()))
    thread.start()
    try:
        park = events.get(timeout=2)
        assert park["phase"] == "park"
        identity = {key: park[key] for key in ("type", "task_id", "task_attempt", "wait_id")}
        if order != "answer_after_parked":
            _answer(tmp_path, "root-1")
        commands.put({**identity, "phase": "parked"})
        if order == "answer_after_parked":
            _answer(tmp_path, "root-1")
        resume = events.get(timeout=3)
        assert resume["phase"] == "resume" and resume["resume_reason"] == "answer"
        assert not ended.is_set()  # a request, never a self-granted resume
        commands.put({**identity, "phase": "resume_granted"})
        thread.join(timeout=2)
        assert ended.is_set() and outcome == ["answer"]
    finally:
        commands.put({"type": "owner_wait", "phase": "resume_granted", "task_id": "root-1",
                      "task_attempt": 1, "wait_id": "w1"})
        thread.join(timeout=2)


def test_an_answer_a_second_before_the_wake_is_not_rolled_back(tmp_path, monkeypatch):
    from ouroboros.gateway.task_decision import _quiz_answer_frame

    record_asked(tmp_path, "root-1", quiz_id="q1", question="Which?", options=["a", "b"],
                 wait_for_answer=True, chat_id=1, assumption="a meanwhile")
    assert record_answered(tmp_path, "root-1", quiz_id="q1", option_index=1, request_id="r1")["ok"]
    frames = _bridge(monkeypatch)
    assert mark_wait_ended(tmp_path, "root-1", "q1") is False
    announce_wait_ended(tmp_path, "root-1", "q1", 1)
    block = quiz_states(tmp_path, "root-1")["q1"]
    assert block["state"] == "answered" and block["wait_for_answer"] is True
    assert "wait_ended_at" not in block and not _quiz_frames(frames)
    # The task was waiting when the owner answered: the frame must not claim it moved on.
    assert "You continued under the assumption" not in _quiz_answer_frame(block, 1, "")


def test_a_wait_end_before_the_answer_keeps_the_answer_and_its_audit_stamp(tmp_path, monkeypatch):
    from ouroboros.gateway.task_decision import _quiz_answer_frame

    record_asked(tmp_path, "root-1", quiz_id="q1", question="Which?", options=["a", "b"],
                 wait_for_answer=True, chat_id=1, assumption="a meanwhile")
    frames = _bridge(monkeypatch)
    announce_wait_ended(tmp_path, "root-1", "q1", 1)
    ended = quiz_states(tmp_path, "root-1")["q1"]
    assert "wait_for_answer" not in ended and ended["wait_ended_at"] and len(_quiz_frames(frames)) == 1
    answered = record_answered(tmp_path, "root-1", quiz_id="q1", option_index=0, request_id="r1")
    assert answered["ok"] and answered["block"]["state"] == "answered"
    block = quiz_states(tmp_path, "root-1")["q1"]
    assert block["state"] == "answered" and block["wait_ended_at"] == ended["wait_ended_at"]
    # The task really did continue under its assumption before this answer: the frame says so.
    assert "You continued under the assumption: a meanwhile" in _quiz_answer_frame(block, 0, "")
    # A later wake announces nothing: the answered card is never rolled back to open.
    assert mark_wait_ended(tmp_path, "root-1", "q1") is False
    announce_wait_ended(tmp_path, "root-1", "q1", 1)
    assert quiz_states(tmp_path, "root-1")["q1"]["state"] == "answered"
    assert len(_quiz_frames(frames)) == 1


def test_a_peer_wake_leaves_the_question_open_and_a_new_wait_ends_on_the_answer(tmp_path, monkeypatch):
    ctx = native_context(tmp_path)
    ctx.current_chat_id = 1
    record_asked(tmp_path, ctx.task_id, quiz_id="q1", question="Which path?", options=[],
                 wait_for_answer=True, chat_id=1)
    frames = _bridge(monkeypatch)
    monkeypatch.setattr("ouroboros.owner_wait.time.sleep", lambda _seconds: write_task_message(
        tmp_path, "Here is context", task_id=ctx.task_id, source_task_id="peer-1",
        provenance="independent_task"))
    messages = []
    wait_after_tools(ctx, messages, {}, {}, 1, [], set())
    first = load_task_result(tmp_path, ctx.task_id)["owner_wait"]
    assert first["resume_reason"] == "mail:peer-1" and ctx._owner_wait_requested == ""
    assert quiz_states(tmp_path, ctx.task_id)["q1"]["state"] == "open"
    [notice] = messages
    assert "task mail from peer-1" in notice["content"] and "remains open" in notice["content"]
    assert len(_quiz_frames(frames)) == 1
    # The loop drains and ACKs the peer mail; the model asks again: a NEW card, a NEW wait.
    ctx._loop_mailbox_seen_ids = {entry["msg_id"] for entry in drain_owner_entries(tmp_path, ctx.task_id, set())}
    record_asked(tmp_path, ctx.task_id, quiz_id="q2", question="Still: which path?", options=[],
                 wait_for_answer=True, chat_id=1)
    ctx._owner_wait_requested = "q2"

    def answer(_seconds):
        assert record_answered(tmp_path, ctx.task_id, quiz_id="q2", option_index=None,
                               request_id="r1", comment="Left")["ok"]
        _answer(tmp_path, ctx.task_id, "q2")

    monkeypatch.setattr("ouroboros.owner_wait.time.sleep", answer)
    wait_after_tools(ctx, messages, {}, {}, 2, [], set())
    second = load_task_result(tmp_path, ctx.task_id)["owner_wait"]
    assert second["wait_id"] != first["wait_id"] and second["state"] == "resumed"
    assert second["resume_reason"] == "answer" and len(messages) == 1  # no second notice
    assert quiz_states(tmp_path, ctx.task_id)["q2"]["state"] == "answered"
    assert len(_quiz_frames(frames)) == 1  # the answered card was not re-announced as open


def test_owner_wait_ended_notice_keeps_the_timeout_text_under_both_names(tmp_path):
    ctx = native_context(tmp_path)
    record_asked(tmp_path, ctx.task_id, quiz_id="q1", question="?", options=[],
                 wait_for_answer=True, assumption="keep going")
    checkpoint = {"quiz_id": "q1", "wait_max_minutes": 5}
    timeout = owner_wait_ended_notice(ctx, checkpoint, "timeout")
    assert timeout == owner_wait_timeout_notice(ctx, checkpoint) == owner_wait_ended_notice(ctx, checkpoint)
    assert "No owner answer arrived within 5 minutes" in timeout["content"]
    assert "Proceed under your stated assumption: keep going" in timeout["content"]
    ended = owner_wait_ended_notice(ctx, checkpoint, "not_answered", woke_by="task mail from peer-1")
    assert ended["role"] == "user" and ended["content"].startswith("[SYSTEM NOTICE]")
    assert "question q1 ended after task mail from peer-1, not a confirmed owner answer" in ended["content"]
    with pytest.raises(ValueError, match="unknown owner wait end reason"):
        owner_wait_ended_notice(ctx, checkpoint, "answer")


def test_the_notice_yields_to_rendered_owner_authority_and_to_a_landed_answer(tmp_path):
    ctx = native_context(tmp_path)
    record_asked(tmp_path, ctx.task_id, quiz_id="q1", question="?", options=[], wait_for_answer=True)
    checkpoint = {"quiz_id": "q1"}
    messages = []
    write_task_message(tmp_path, "peer context", task_id=ctx.task_id, source_task_id="peer-1",
                       provenance="independent_task")
    append_wake_notice(ctx, checkpoint, "mail:peer-1", messages)
    assert len(messages) == 1 and "peer-1" in messages[0]["content"]
    assert write_owner_message(tmp_path, "Owner words", ctx.task_id, msg_id="t1")
    append_wake_notice(ctx, checkpoint, "mail:peer-1", messages)
    assert len(messages) == 1  # the owner's own words explain the wake; no host frame beside them
    append_wake_notice(ctx, {**checkpoint, "review_binding": "acceptance"}, "timeout", messages)
    assert len(messages) == 1  # the acceptance park is not an owner question
    append_wake_notice(ctx, checkpoint, None, messages)
    append_wake_notice(ctx, checkpoint, "control:deadline", messages)
    assert len(messages) == 1
    assert record_answered(tmp_path, ctx.task_id, quiz_id="q1", option_index=None,
                           request_id="r", comment="a")["ok"]
    append_wake_notice(ctx, checkpoint, "timeout", messages)
    assert len(messages) == 1  # an answer landed before this round: never claim silence


def test_an_unbounded_wait_has_no_general_limit_and_the_task_deadline_is_hard(tmp_path, monkeypatch):
    from ouroboros.model_wait import TaskModelWait
    from ouroboros.owner_wait import _wait_bound_fields, checkpoint_owner_wait

    ctx = native_context(tmp_path)
    assert _wait_bound_fields(ctx) == {}
    assert "wait_deadline_at" not in checkpoint_owner_wait(ctx, [], {}, {}, 1, [], set())
    polls = []

    def poll(_seconds):
        polls.append(1)
        if len(polls) == 25:
            assert write_owner_message(tmp_path, "Take the left path", ctx.task_id, msg_id="t1")

    monkeypatch.setattr("ouroboros.owner_wait.time.sleep", poll)
    messages = []
    wait_after_tools(ctx, messages, {}, {}, 1, [], set())
    row = load_task_result(tmp_path, ctx.task_id)["owner_wait"]
    assert len(polls) == 25 and row["resume_reason"] == "owner_text" and messages == []
    assert "wait_deadline_at" not in row and "wait_max_minutes" not in row

    # The task's own deadline is a hard axis, consulted before any mailbox fact.
    past = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=1)).isoformat()
    ctx._owner_wait_requested = "q1"
    ctx.model_wait_context = TaskModelWait(
        task={"id": ctx.task_id, "deadline_at": past}, drive_root=tmp_path,
        event_queue=ctx.event_queue, worker_slot_held=False)
    ctx.model_wait_context.tool_context = ctx
    monkeypatch.setattr("ouroboros.owner_wait.time.sleep",
                        lambda _seconds: pytest.fail("a passed deadline must not wait"))
    wait_after_tools(ctx, messages, {}, {}, 2, [], set())
    assert load_task_result(tmp_path, ctx.task_id)["owner_wait"]["resume_reason"] == "control:deadline"
    assert messages == []


def test_a_cold_handoff_is_refused_once_the_task_deadline_passed(tmp_path):
    from ouroboros.deadline_utils import utc_now
    from ouroboros.owner_wait import checkpoint_owner_wait, restore_owner_wait_allowed, set_owner_wait
    from ouroboros.utils import atomic_write_json

    ctx = context(tmp_path)
    block = checkpoint_owner_wait(ctx, [], {}, {}, 1, [], set())
    set_owner_wait(tmp_path, "root-1", {**block, "state": "waiting"})
    path = tmp_path / "state/delegate_recovery_transactions/tx.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, {"status": "normal_exit_acknowledged", "task_ids": ["root-1"]})
    handoff = {**block, "restart_transaction_id": "tx"}
    future = (utc_now() + datetime.timedelta(hours=1)).isoformat()
    past = (utc_now() - datetime.timedelta(seconds=1)).isoformat()
    assert restore_owner_wait_allowed(tmp_path, {"id": "root-1", "_owner_wait_resume": handoff, "deadline_at": future})
    assert not restore_owner_wait_allowed(tmp_path, {"id": "root-1", "_owner_wait_resume": handoff, "deadline_at": past})


@pytest.mark.parametrize("reason,announced", [
    ("hurry", True), ("mail:peer-1", True), ("owner_text", True), ("control:cancelled", False),
])
def test_the_pool_grant_announces_every_non_control_wake(tmp_path, monkeypatch, reason, announced):
    from ouroboros.owner_wait import set_owner_wait
    from supervisor import worker_owner_wait as supervisor_wait

    record_asked(tmp_path, "root-1", quiz_id="q1", question="?", options=["a", "b"],
                 chat_id=1, wait_for_answer=True)
    frames = _bridge(monkeypatch)
    wait = {"wait_id": "w1", "task_attempt": 1, "quiz_id": "q1", "source_ref": "ref"}
    set_owner_wait(tmp_path, "root-1", {**wait, "state": "waiting"})
    worker = SimpleNamespace(in_q=queue.Queue(), busy_task_id="root-1",
                             proc=SimpleNamespace(pid=7), reaping=False, active_capacity=False)
    meta = {"worker_id": 0, "attempt": 1, "owner_wait": {**wait, "state": "waiting"},
            "task": {"id": "root-1", "chat_id": 1}, "started_at": time.time()}
    pool = SimpleNamespace(RUNNING={"root-1": meta}, WORKERS={0: worker}, DRIVE_ROOT=tmp_path,
                           time=SimpleNamespace(time=time.time))
    supervisor_wait.handle_owner_wait(
        {"type": "owner_wait", "task_id": "root-1", "worker_id": 0, "pid": 7,
         "task_attempt": 1, "wait_id": "w1", "phase": "resume", "resume_reason": reason},
        pool,
    )
    monkeypatch.setattr(supervisor_wait, "_pool", lambda: pool, raising=False)
    monkeypatch.setattr(supervisor_wait, "_resume_allowed", lambda *_: True)
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda **_kw: True)
    assert supervisor_wait._grant_resume("root-1", meta, worker) is True
    row = load_task_result(tmp_path, "root-1")["owner_wait"]
    assert row["state"] == "resumed" and row["resume_reason"] == reason
    block = quiz_states(tmp_path, "root-1")["q1"]
    assert block["state"] == "open"
    assert ("wait_for_answer" not in block) is announced and bool(_quiz_frames(frames)) is announced


@pytest.mark.parametrize("outcome,expects", [
    ("hurry", "owner hurry request"), ("mail:peer-1", "task mail from peer-1"),
    ("timeout", "No owner answer arrived within 5 minutes"), ("answer", None), ("control:deadline", None),
])
def test_a_cold_continuation_explains_a_non_answer_wake_like_warm(tmp_path, monkeypatch, outcome, expects):
    from ouroboros import loop
    from ouroboros.owner_wait import checkpoint_owner_wait, load_owner_wait, resume_native_loop, set_owner_wait
    from ouroboros.task_pacing import CostCeiling
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id, ctx.task_attempt = "cold", 1
    ctx.active_model, ctx.active_effort = "m", "high"
    ctx.active_use_local, ctx.active_context_mode = False, "max"
    ctx._owner_wait_requested = "q1"
    ctx._owner_wait_max_minutes, ctx._owner_wait_deadline_at = 5, "2020-01-01T00:00:00+00:00"
    ctx._cost_ceiling = CostCeiling(state="disabled")
    ctx.context_fit_plan = None
    write_task_result(tmp_path, "cold", "running")
    record_asked(tmp_path, "cold", quiz_id="q1", question="?", options=["a", "b"],
                 wait_for_answer=True, assumption="a meanwhile")
    wait = checkpoint_owner_wait(ctx, [{"role": "user", "content": "go"}], {}, {}, 1, [], set())
    set_owner_wait(tmp_path, "cold", {**wait, "state": "waiting"})
    ctx.owner_wait_resume = {**wait, "restart_transaction_id": "confirmed"}
    if outcome == "mail:peer-1":
        write_task_message(tmp_path, "context", task_id="cold", source_task_id="peer-1",
                           provenance="independent_task")
    elif outcome == "hurry":
        _hurry(tmp_path, "cold")
    ctx.owner_wait_callback = lambda *_: outcome
    saved = load_owner_wait(ctx)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *a, **k: (None, "max"))
    messages = []
    resume_native_loop(registry, saved, messages, {}, {}, set())
    assert messages[0]["content"] == "go"
    assert "continued from its saved owner wait" in messages[1]["content"]
    if expects is None:
        assert len(messages) == 2
    else:
        assert len(messages) == 3 and expects in messages[2]["content"]
        assert messages[2]["content"].startswith("[SYSTEM NOTICE]")
