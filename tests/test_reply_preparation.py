"""Which chats a reply is being actively prepared for (DESIGN §9, Telegram typing).

Driven through the projection's real owners: the queue's RUNNING/PENDING rows, the
direct-turn registry, durable task results, the owner-Pause fence, cancel intents,
project bindings, the bridge's reply transports, and the actual producers that
change them (the answer-first final through the supervisor's send seam, the
in-slot wait declarations, the delegated-run supervision loop, a review panel's
seat leases and model waits, the resume announcement). Each scenario keeps its
negative control beside the positive one.
"""

from __future__ import annotations

import json
import os
import queue as queue_module
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros.gateway.reply_preparation import reply_preparation
from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, write_task_result
from tests._budget_pause_exact_helpers import _install_queue

pytestmark = pytest.mark.serial

ORIGIN = {"origin_message_ref": {"chat_id": 1, "client_message_id": "m-1", "ts": "2026-10-17T00:00:00+00:00"}}


@pytest.fixture
def world(tmp_path, monkeypatch):
    from supervisor import message_bus
    from supervisor.active_activity import get_direct_activity_registry

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {})
    registry = get_direct_activity_registry()
    registry.clear()
    yield SimpleNamespace(root=tmp_path, queue=queue, workers=workers, bridge=bridge, registry=registry)
    registry.clear()


def _chats(world):
    return sorted(row["chat_id"] for row in reply_preparation(world.root)["chats"])


def _direct(world, task_id, chat_id=1, *, metadata=ORIGIN, actor=None):
    from supervisor.active_activity import reply_expected

    write_task_result(world.root, task_id, STATUS_RUNNING, chat_id=chat_id, _is_direct_chat=True)
    return world.registry.register(task_id, chat_id, actor=actor,
                                   reply_expected=reply_expected({"metadata": dict(metadata)}))


def _pooled(world, task_id, chat_id=1, *, parent="", metadata=ORIGIN, attempt=1):
    lineage = {"root_task_id": task_id} if not parent else {
        "root_task_id": parent, "parent_task_id": parent, "delegation_role": "subagent"}
    write_task_result(world.root, task_id, STATUS_RUNNING, chat_id=chat_id, **lineage)
    meta = {"task": {"id": task_id, "type": "task", "chat_id": chat_id, "metadata": dict(metadata),
                     "_attempt": attempt, **lineage}, "worker_id": len(world.workers.RUNNING), "attempt": attempt,
            "started_at": time.time()}
    world.workers.RUNNING[task_id] = meta
    return meta


class _Supervisor:
    """A worker's event queue drained synchronously into the supervisor's real handlers."""
    def __init__(self, world):
        from supervisor.cognitive_operations import _handle_cognitive_operation
        from supervisor.task_model_wait import handle_task_model_wait

        self.seen = []
        self.ctx = SimpleNamespace(RUNNING=world.workers.RUNNING, DRIVE_ROOT=world.root, bridge=world.bridge,
                                   append_jsonl=lambda *a, **k: None)
        self.handlers = {"cognitive_operation": _handle_cognitive_operation,
                         "task_model_wait": handle_task_model_wait}

    def put(self, event, *_a, **_k):
        self.seen.append(event)
        handler = self.handlers.get(str(event.get("type") or ""))
        if handler is not None:
            handler(event, self.ctx)

    put_nowait = put

    def kinds(self):
        return [event["type"] for event in self.seen]


def _until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_only_the_owners_root_is_expected_and_children_never_name_a_chat(world):
    _direct(world, "owner-turn", chat_id=1)
    _direct(world, "wake-turn", chat_id=1, metadata={"initiator": "consciousness"})
    _direct(world, "presence-turn", chat_id=7,
            metadata={**ORIGIN, "presence": {"binding_id": "b", "event": {"provider": "x"}}})
    _pooled(world, "api-root", chat_id=0, metadata={"source": "api"})
    _pooled(world, "kid", chat_id=55, parent="owner-turn")
    state = reply_preparation(world.root)
    assert state["complete"] is True
    assert state["roots"] == ["owner-turn"] and _chats(world) == [1]


def test_a_direct_final_ends_preparation_before_the_slow_store_through_the_real_producer(world, monkeypatch):
    """Answer first: the real final crosses the supervisor's send seam while the result
    store is still blocked; from that moment the turn prepares nothing, even when the
    durable outbox could not register it."""
    from ouroboros import agent_task_pipeline as pipeline
    from supervisor import message_bus, state as supervisor_state
    from supervisor.events import _handle_send_message
    from supervisor import terminal_delivery

    _direct(world, "turn-1", chat_id=1)
    assert _chats(world) == [1]
    monkeypatch.setattr(supervisor_state, "reconstruct_task_cost", lambda *a, **k: {})
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)

    def refuse(*_a, **_k):
        raise OSError("outbox unavailable")

    monkeypatch.setattr(terminal_delivery, "register_pending_delivery", refuse)
    store_entered, release = threading.Event(), threading.Event()
    seen_during_store = []
    real_store = pipeline._store_task_result

    def blocked_store(*args, **kwargs):
        try:
            seen_during_store.append(_chats(world))
        finally:
            store_entered.set()
        assert release.wait(10)
        return real_store(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_store_task_result", blocked_store)
    ctx = SimpleNamespace(DRIVE_ROOT=world.root, RUNNING=world.workers.RUNNING,
                          send_with_budget=message_bus.send_with_budget, append_jsonl=lambda *a, **k: None)

    class SupervisorQueue:
        """The live put reaches the supervisor's send seam, as the drain thread would."""
        def put(self, event, *_a, **_k):
            if event.get("type") == "send_message":
                _handle_send_message(event, ctx)

    task = {"id": "turn-1", "type": "task", "chat_id": 1, "text": "q", "_is_direct_chat": True,
            "metadata": dict(ORIGIN)}
    env = SimpleNamespace(drive_root=world.root, repo_dir=world.root)
    worker = threading.Thread(target=pipeline.emit_task_results, args=(
        env, None, None, [], task, "The answer.", {"execution_status": "ok", "reason_code": "final_message"},
        {"tool_calls": [], "reasoning_notes": []}, 0.0, world.root / "logs"),
        kwargs={"event_queue": SupervisorQueue()}, daemon=True)
    worker.start()
    try:
        assert store_entered.wait(10)
        assert seen_during_store == [[]], "the handed-off turn still read as preparing during its store"
        assert world.registry.get("turn-1") is not None, "the turn itself is still alive (post-answer work)"
    finally:
        release.set()
        worker.join(10)


def test_an_intermediate_reply_is_not_an_answer_but_a_pooled_final_is(world):
    from supervisor import message_bus
    from supervisor.events import _handle_send_message
    from supervisor.terminal_delivery import delivery_id_for

    _pooled(world, "root-p", chat_id=1)
    ctx = SimpleNamespace(DRIVE_ROOT=world.root, RUNNING=world.workers.RUNNING,
                          send_with_budget=message_bus.send_with_budget, append_jsonl=lambda *a, **k: None)
    _handle_send_message({"type": "send_message", "chat_id": 1, "task_id": "root-p", "text": "checking…",
                          "system_type": "proactive_message"}, ctx)
    _handle_send_message({"type": "send_message", "chat_id": 1, "task_id": "root-p", "text": "progress",
                          "is_progress": True, "delivery_id": delivery_id_for("root-p", "progress")}, ctx)
    _handle_send_message({"type": "send_message", "chat_id": 1, "task_id": "root-p", "text": "a child's final",
                          "delivery_id": delivery_id_for("someone-else", "done")}, ctx)
    assert _chats(world) == [1]
    _handle_send_message({"type": "send_message", "chat_id": 1, "task_id": "root-p", "text": "done",
                          "delivery_id": delivery_id_for("root-p", "done")}, ctx)
    assert _chats(world) == [] and "root-p" in world.workers.RUNNING


def test_a_failed_direct_turn_is_handed_off_before_its_announce_and_failure_reply(world, monkeypatch):
    from supervisor import message_bus, worker_chat_lane, workers

    entry = _direct(world, "turn-f", chat_id=1)
    observed = []
    monkeypatch.setattr(workers, "send_with_budget", lambda *a, **k: observed.append(("reply", _chats(world))))
    monkeypatch.setattr(message_bus.LocalChatBridge, "send_chat_action",
                        lambda *a, **k: observed.append(("announce", _chats(world))))
    worker_chat_lane._report_direct_chat_error(
        {"task": {"id": "turn-f"}, "chat_id": 1, "kind": "direct_chat", "client_msg_id": "c-1",
         "task_metadata": dict(ORIGIN), "activity": entry}, RuntimeError("boom"))
    assert observed == [("announce", []), ("reply", [])]


def test_owner_pause_fences_before_the_park_and_resume_reopens(world):
    from ouroboros.owner_pause import FENCE_RELEASED, install_fence, read_fence, set_fence_state

    _direct(world, "turn-p", chat_id=1)
    _pooled(world, "root-q", chat_id=2)
    _pooled(world, "kid-q", chat_id=2, parent="root-q")
    assert _chats(world) == [1, 2]
    for root_id in ("turn-p", "root-q"):
        install_fence(world.root, root_id, request_id=f"press-{root_id}")
    assert _chats(world) == [], "a requested Pause renews nothing, its working helper included"
    for root_id in ("turn-p", "root-q"):
        fence = read_fence(world.root, root_id)
        set_fence_state(world.root, root_id, fence_id=fence["fence_id"], state=FENCE_RELEASED,
                        release_reason="owner_resume")
    assert _chats(world) == [1, 2]


def _intent(world, task_id, *, policy, scope="single"):
    from ouroboros import cancel_intents

    path = world.root / "state" / "cancel_intents.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    current = json.loads(path.read_text()) if path.exists() else {"intents": {}}
    current["intents"][task_id] = {"task_id": task_id, "request_id": f"stop-{task_id}", "scope": scope,
                                   "stop_policy": policy, "requested_at": "2026-10-17T00:00:00+00:00",
                                   "state": "active"}
    path.write_text(json.dumps(current))
    assert task_id in cancel_intents.active_intents(world.root, strict=True), "fixture: the intent reads active"


def test_stop_now_fences_its_tree_and_a_graceful_wrap_up_keeps_its_one_summary(world):
    from ouroboros.cancel_intents import STOP_POLICY_FINALIZE, STOP_POLICY_IMMEDIATE

    _pooled(world, "root-s", chat_id=3)
    root_k = _pooled(world, "root-k", chat_id=4)
    _pooled(world, "kid-k", chat_id=4, parent="root-k")
    root_k["owner_wait"] = {"wait_id": "w", "state": "waiting", "task_attempt": 1, "reason": "owner",
                            "source_ref": {"x": 1}}
    assert _chats(world) == [3, 4]
    _intent(world, "root-s", policy=STOP_POLICY_FINALIZE)
    assert _chats(world) == [3, 4], "Wrap up: the summary answer is still being prepared"
    _intent(world, "root-s", policy=STOP_POLICY_IMMEDIATE)
    _intent(world, "kid-k", policy=STOP_POLICY_IMMEDIATE)
    assert _chats(world) == [], "Stop now ends the root's work and a stopped helper sustains nothing"
    (world.root / "state" / "cancel_intents.json").write_text("{not json")
    state = reply_preparation(world.root)
    assert state["complete"] is False and state["chats"] == [], "an unreadable Stop is never a positive"


def test_waits_renew_only_through_a_working_helper(world):
    """An owner question, a model-access wait or a timer/mail-only sleep prepares nothing
    alone; a live descendant whose own work runs does, and an unrelated reviewer's
    access wait never cancels a working author."""
    root = _pooled(world, "root-w", chat_id=3)
    root["owner_wait"] = {"wait_id": "w1", "state": "waiting", "task_attempt": 1, "reason": "owner",
                          "source_ref": {"x": 1}}
    assert _chats(world) == []
    kid = _pooled(world, "kid-w", chat_id=3, parent="root-w")
    assert _chats(world) == [3], "a helper working for the reply sustains it"
    kid["task"]["model_waits"] = {"m1": {"wait_id": "m1", "state": "waiting", "task_attempt": 1, "role": "main"}}
    assert _chats(world) == [], "the helper's own model-access wait prepares nothing"
    kid["task"]["model_waits"]["m1"]["state"] = "resolved"
    grandkid = _pooled(world, "grand-w", chat_id=3, parent="kid-w")
    kid["owner_wait"] = {"wait_id": "w2", "state": "waiting", "task_attempt": 1, "reason": "sleep",
                         "source_ref": {"x": 1}, "sleep": {"mode": "warm", "wake_at": "2099-01-01T00:00:00+00:00"}}
    assert _chats(world) == [3], "a sleeping helper's own working helper still counts"
    grandkid["owner_wait"] = {"wait_id": "w3", "state": "waiting", "task_attempt": 1, "reason": "owner",
                              "source_ref": {"x": 1}}
    assert _chats(world) == []
    root.pop("owner_wait")
    root["task"]["model_waits"] = {"r1": {"wait_id": "r1", "state": "waiting", "task_attempt": 1, "role": "review",
                                          "review_operation": {"owner_id": "op"}}}
    assert _chats(world) == [3], "a reviewer's access wait never cancels the working author"
    root["task"]["model_waits"]["a1"] = {"wait_id": "a1", "state": "waiting", "task_attempt": 1, "role": "main"}
    assert _chats(world) == [], "the author's own access wait does"
    root["task"]["model_waits"]["a1"]["task_attempt"] = 0
    assert _chats(world) == [3], "an earlier attempt's wait row is not the current attempt's"


def test_a_helper_that_handed_off_or_stored_its_answer_sustains_nothing(world):
    """A child's own handoff, stored result or answer checkpoint fences it before it can
    back its waiting parent; an unreadable child is unknown, never a positive."""
    from ouroboros import model_sleep
    from ouroboros.gateway import state as census
    from supervisor.active_activity import mark_reply_handed_off

    supervisor = _Supervisor(world)
    _pooled(world, "root-h", chat_id=5)
    ctx = SimpleNamespace(task_id="root-h", task_attempt=1, event_queue=supervisor, current_chat_id=5)
    model_sleep.declare_in_slot_wait(ctx, "tasks:t", time.time() + 60)  # wait_task on its helper
    assert _chats(world) == []
    _pooled(world, "kid-h", chat_id=5, parent="root-h")
    assert _chats(world) == [5]
    mark_reply_handed_off("kid-h", world.workers.RUNNING)
    assert _chats(world) == [], "a helper past its own answer is post-answer work"
    world.workers.RUNNING["kid-h"].pop("reply_handed_off")
    assert _chats(world) == [5]
    real = census._task_activity_facts
    try:
        census._task_activity_facts = lambda root, tid: {"unavailable": True} if tid == "kid-h" else real(root, tid)
        state = reply_preparation(world.root)
        assert state["chats"] == [] and state["complete"] is False, "unknown never resurrects a helper"
    finally:
        census._task_activity_facts = real
    write_task_result(world.root, "kid-h", STATUS_COMPLETED, parent_task_id="root-h", root_task_id="root-h",
                      delegation_role="subagent", result="done")
    assert _chats(world) == [], "a stored result is an answered helper still alive in its tail"
    _pooled(world, "kid-h2", chat_id=5, parent="root-h")
    assert _chats(world) == [5]
    write_task_result(world.root, "root-h", STATUS_RUNNING, chat_id=5, root_task_id="root-h",
                      root_phase_checkpoint={"post_task_synthesis": "running"})
    assert _chats(world) == [], "the root's own stored answer fences its whole tree"


def test_a_park_request_and_a_slot_held_on_mail_are_waits_not_work(world):
    """A held in-slot wait outranks the wait tool's own ``kind=tool`` lease: a batch with a
    wait tool runs serially (``tool_calls_can_run_parallel``), so no other tool of the same
    round is working beside it."""
    from ouroboros import model_sleep
    from ouroboros.loop_tool_execution import tool_calls_can_run_parallel
    from supervisor.cognitive_operations import _handle_cognitive_operation

    root = _pooled(world, "root-s", chat_id=4)
    root["owner_wait_park_pending"] = {"event": {"checkpoint": {"wait_id": "p", "task_attempt": 1,
                                                                 "source_ref": {"x": 1}}}}
    assert _chats(world) == []
    root.pop("owner_wait_park_pending")
    events = queue_module.Queue()
    ctx = SimpleNamespace(task_id="root-s", task_attempt=1, event_queue=events, current_chat_id=4)
    running = SimpleNamespace(RUNNING=world.workers.RUNNING)
    _handle_cognitive_operation({"type": "cognitive_operation", "task_id": "root-s", "operation_id": "call-1",
                                 "phase": "started", "kind": "tool", "task_attempt": 1}, running)
    assert _chats(world) == [4], "a tool call is the task's own work"
    with model_sleep.in_slot_wait(ctx, 60):
        started = events.get_nowait()
        _handle_cognitive_operation(started, running)
        assert started["kind"] == "in_slot_wait" and _chats(world) == []
    finished = events.get_nowait()
    _handle_cognitive_operation(finished, running)
    assert _chats(world) == [4]
    announced = events.get_nowait()
    assert announced["type"] == "typing_start" and announced["task_id"] == "root-s"
    calls = [{"function": {"name": name}} for name in ("await_messages", "read_file")]
    assert not tool_calls_can_run_parallel(calls), "a wait tool never shares its round with working tools"


class _Ticks:
    """The delegated run as the supervision loop observes it: each tick returns the next
    window payload the test supplies, with the shape ``_delegate_wait`` returns."""
    def __init__(self, run_id):
        self.run_id, self.payloads, self.calls = run_id, queue_module.Queue(), 0

    def __call__(self, _ctx, _run_id, _timeout, _cursor, **_kw):
        self.calls += 1
        return json.dumps(self.payloads.get(timeout=10))

    def tick(self, *, state="running", waiting_on_user=False, status="no_progress", **extra):
        calls = self.calls
        self.payloads.put({"status": status, "run_id": self.run_id, "state": state, "last_seq": 1,
                           "waiting_on_user": waiting_on_user, **extra})
        assert _until(lambda: self.calls > calls), "the loop took the observation and waits for the next"

    def end(self):
        self.payloads.put({"status": "completed", "run_id": self.run_id, "state": "succeeded", "last_seq": 2})


def _hold(ctx, ticks):
    from ouroboros.delegate_supervision import supervised_wait

    thread = threading.Thread(target=supervised_wait, args=(ctx, ticks.run_id), kwargs={"wait_once": ticks},
                              daemon=True)
    thread.start()
    assert _until(lambda: ticks.calls == 1)
    return thread


def test_a_held_delegated_run_prepares_while_observed_running_and_not_while_it_waits(world):
    """The nanny's supervised hold is a wait; the run it holds is a working helper only while
    the supervision owner currently observes it running and not paused on its question."""
    from ouroboros.delegate_progress import emit_external_wait_lease
    from ouroboros.delegate_supervision import _OBSERVATION_CURRENT_SEC

    supervisor = _Supervisor(world)
    root = _pooled(world, "root-n", chat_id=6)
    root["owner_wait"] = {"wait_id": "w", "state": "waiting", "task_attempt": 1, "reason": "owner",
                          "source_ref": {"x": 1}}  # the root waits; only its helper can prepare
    _pooled(world, "nanny", chat_id=6, parent="root-n")
    ctx = SimpleNamespace(task_id="nanny", task_attempt=1, drive_root=world.root, budget_drive_root=str(world.root),
                          task_metadata={"root_task_id": "root-n", "parent_task_id": "root-n",
                                         "delegation_role": "subagent"}, event_queue=supervisor, current_chat_id=6)
    emit_external_wait_lease(ctx, "run-1", time.time() + 60, lease_id="lease-1")
    assert _chats(world) == [6], "the idle-rail lease alone is no wait: its semantics stay the idle rail's"
    ticks = _Ticks("run-1")
    thread = _hold(ctx, ticks)
    try:
        assert reply_preparation(world.root)["complete"] is False and _chats(world) == [], \
            "held before any current observation: unknown, never a positive"
        ticks.tick(state="running")
        assert _chats(world) == [6], "an observed running run works for the reply"
        announced = supervisor.kinds().count("typing_start")
        ticks.tick(state="running", waiting_on_user=True,
                   pending_interactions=[{"interaction_id": "i", "questions": []}])
        assert _chats(world) == [], "a run paused on its question prepares nothing"
        ticks.tick(state="running")
        assert _chats(world) == [6]
        assert supervisor.kinds().count("typing_start") == announced + 1, "working again re-announces"
        ticks.tick(state="queued")
        assert _chats(world) == [], "a run queued for capacity is an access wait"
        ticks.tick(status="observation_pending", state="", reason="daemon_unreachable")
        state = reply_preparation(world.root)
        assert state["complete"] is False and state["chats"] == [], "an unanswered read is unknown"
        ticks.tick(state="running")
        assert _chats(world) == [6]
        path = world.root / "state" / "delegate_supervision" / "nanny.json"
        record = json.loads(path.read_text())
        record["runs"]["run-1"]["observation"]["at_unix"] -= _OBSERVATION_CURRENT_SEC + 1
        path.write_text(json.dumps(record))
        assert reply_preparation(world.root)["complete"] is False and _chats(world) == [], \
            "an observation older than a few ticks says nothing current"
        record["runs"]["run-1"]["observation"]["at_unix"] = time.time()
        record["runs"]["run-1"]["observation"]["attempt"] = "2"
        path.write_text(json.dumps(record))
        assert _chats(world) == [], "another attempt's observation is not this attempt's"
    finally:
        ticks.end()
        thread.join(10)
    assert not thread.is_alive()
    assert _chats(world) == [6], "the hold ended: the nanny's own work goes on"


def test_a_direct_turn_holds_its_delegated_run_on_its_live_context(world):
    ctx = SimpleNamespace(task_id="turn-d", task_attempt=1, drive_root=world.root, budget_drive_root=str(world.root),
                          task_metadata={}, event_queue=None, current_chat_id=1)
    _direct(world, "turn-d", chat_id=1, actor=SimpleNamespace(tools=SimpleNamespace(_ctx=ctx)))
    assert _chats(world) == [1]
    ticks = _Ticks("run-d")
    thread = _hold(ctx, ticks)
    try:
        ticks.tick(state="running", waiting_on_user=True)
        assert _chats(world) == [], "holding a run paused on its question is a wait"
        ticks.tick(state="running")
        assert _chats(world) == [1]
    finally:
        ticks.end()
        thread.join(10)
    assert ctx._in_slot_waits == {} and _chats(world) == [1]
    ctx._in_slot_waits = {"mail:stale": time.time() - 1}
    assert _chats(world) == [1], "an expired mark outlives nothing"
    write_task_result(world.root, "turn-d", STATUS_RUNNING, owner_wait={
        "wait_id": "q", "quiz_id": "quiz-1", "state": "waiting", "task_attempt": 1, "reason": "owner"})
    assert _chats(world) == []
    write_task_result(world.root, "turn-d", STATUS_RUNNING, owner_wait={
        "wait_id": "q", "quiz_id": "quiz-1", "state": "resumed", "task_attempt": 1, "reason": "owner"})
    ctx._owner_wait_requested = "quiz-2"
    assert _chats(world) == [], "a question asked in this batch already waits"


@pytest.fixture
def panel(world, monkeypatch):
    """One real review operation of a root, its seats run by the test."""
    from ouroboros import config, review_operation
    from ouroboros.model_wait import TaskModelWait
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, ActiveReviewAttempt, _emit_operation
    from ouroboros.review_records import ReviewSlot
    from ouroboros.usage_accounting import UsageScope, usage_scope

    monkeypatch.setattr(config, "CLAUDEXOR_MODEL_POLL_INTERVAL_SEC", 0.02)
    released: dict = {}

    def drain(self):
        for row in self.waits.values():
            if row.get("state") == "waiting" and released.get(row.get("_slot_id")):
                row["_action"] = {"action": "retry", "request_id": f"r-{row['_slot_id']}"}

    monkeypatch.setattr(review_operation._OperationWait, "_drain_controls", drain)
    supervisor = _Supervisor(world)
    request = SimpleNamespace(surface="multi_model_review", retry_key="rk", policy={}, task_attempt=1)
    state = SimpleNamespace(threads=[], entries={})

    def build(task_id, chat_id, *, local):
        tool_context = SimpleNamespace(task_id=task_id, task_attempt=1, event_queue=None if local else supervisor,
                                       current_chat_id=chat_id, task_metadata=dict(ORIGIN))
        parent = TaskModelWait(task={"id": task_id, "chat_id": chat_id, "_attempt": 1, "metadata": dict(ORIGIN)},
                               drive_root=world.root, event_queue=tool_context.event_queue, worker_slot_held=True)
        parent.tool_context = tool_context
        operation = review_operation.ReviewOperation(parent=parent, request=request, task_id=task_id,
                                                     result_root=world.root)
        with review_operation._LOCK:
            review_operation._LIVE[operation.owner_id] = operation
        state.parent, state.operation, state.ctx, state.local = parent, operation, tool_context, local
        return operation

    def seat(slot_id):
        owner = state.operation.owner_id
        slot = ReviewSlot(slot_id, "plain-model")
        entry = ActiveReviewAttempt(key=f"k-{owner}-{slot_id}", operation_id=f"op-{owner}-{slot_id}",
                                    assignment=(request, slot), operation=state.operation)
        with _ACTIVE_LOCK:
            _ACTIVE[entry.key] = entry
        state.entries[(owner, slot_id)] = (entry, slot)
        _emit_operation(SimpleNamespace(event_queue=None if state.local else supervisor, execution_id="e"),
                        task_id=state.operation.task_id, request=request, entry=entry, slot=slot, phase="started")

    def settle(slot_id):
        entry, slot = state.entries.pop((state.operation.owner_id, slot_id))
        entry.event.set()
        with _ACTIVE_LOCK:
            _ACTIVE.pop(entry.key, None)
        _emit_operation(SimpleNamespace(event_queue=None if state.local else supervisor, execution_id="e"),
                        task_id=state.operation.task_id, request=request, entry=entry, slot=slot, phase="finished")

    def quota(slot_id):
        def run():
            with usage_scope(UsageScope(review_slot_id=slot_id)):
                state.operation.wait.wait(None, SimpleNamespace(code="subscription_window_exhausted"),
                                          {"model_role": "review", "model": "plain-model"})
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        state.threads.append(thread)
        assert _until(lambda: slot_id in state.operation.wait.waiting_slots())

    def resolve(slot_id):
        released[slot_id] = True
        assert _until(lambda: slot_id not in state.operation.wait.waiting_slots())

    panel = SimpleNamespace(build=build, seat=seat, settle=settle, quota=quota, resolve=resolve, state=state,
                            supervisor=supervisor)
    yield panel
    released.update({slot: True for slot in ("a", "b", "c")})
    for thread in state.threads:
        thread.join(5)
    with _ACTIVE_LOCK:
        for entry, _slot in state.entries.values():
            _ACTIVE.pop(entry.key, None)
    with review_operation._LOCK:
        review_operation._LIVE.clear()


@pytest.mark.parametrize("local", [False, True], ids=["pooled", "direct"])
def test_an_author_draining_its_panel_prepares_only_while_a_seat_works(world, panel, local):
    """The synchronous drain's facts are the panel's own: its live seats against the seats
    its wait owner holds for model access, matched by panel owner and nested seat."""
    from ouroboros.review_pause import author_drain

    if local:
        _direct(world, "root-r", chat_id=6, actor=SimpleNamespace(tools=SimpleNamespace(_ctx=None)))
    else:
        _pooled(world, "root-r", chat_id=6)
    operation = panel.build("root-r", 6, local=local)
    if local:
        world.registry.get("root-r").actor.tools._ctx = panel.state.ctx
    with author_drain(panel.state.parent, operation):
        state = reply_preparation(world.root)
        assert state["chats"] == [] and state["complete"] is False, "no seat yet: unknown, never the drain's work"
        panel.seat("a")
        panel.seat("b")
        panel.quota("a")
        assert _chats(world) == [6], "one seat still works"
        panel.quota("b")
        assert _chats(world) == [], "every live seat waits for model access"
        panel.resolve("b")
        assert _chats(world) == [6]
        panel.settle("b")
        assert _chats(world) == [], "a settled seat is not counted against the waiting one"
        panel.seat("c")
        assert _chats(world) == [6], "a new working seat"
    assert _chats(world) == [6], "the drain returned: the author's own work"


def test_seats_are_matched_by_their_panel_not_by_seat_name(world, panel):
    """Seat ids repeat across panels ("slot-1" in each): another panel's working seat of the
    same name never stands in for the drained panel's waiting one."""
    from ouroboros.review_pause import author_drain

    _pooled(world, "root-x", chat_id=6)
    other = panel.build("root-x", 6, local=False)
    panel.seat("a")  # the other panel's seat "a" works
    drained = panel.build("root-x", 6, local=False)
    with author_drain(panel.state.parent, drained):
        panel.seat("a")
        panel.quota("a")
        assert other.owner_id != drained.owner_id
        assert _chats(world) == [], "the drained panel's only seat waits; the other panel is not it"


def test_an_unattributed_seat_is_never_the_drained_panels_work(world, panel):
    from ouroboros.review_pause import author_drain
    from supervisor.cognitive_operations import _handle_cognitive_operation

    _pooled(world, "root-u", chat_id=6)
    operation = panel.build("root-u", 6, local=False)
    with author_drain(panel.state.parent, operation):
        _handle_cognitive_operation({"type": "cognitive_operation", "task_id": "root-u", "operation_id": "op-old",
                                     "phase": "started", "kind": "review", "task_attempt": 1, "slot_id": "a"},
                                    SimpleNamespace(RUNNING=world.workers.RUNNING))
        state = reply_preparation(world.root)
        assert state["chats"] == [] and state["complete"] is False, "an unattributed seat is unknown, never work"


def test_an_independent_panel_waiting_on_quota_never_hides_the_working_author(world, panel):
    _pooled(world, "root-i", chat_id=7)
    panel.build("root-i", 7, local=False)
    panel.seat("a")
    panel.quota("a")
    assert _chats(world) == [7], "the author is not draining that panel; its own work counts"


def test_a_left_review_park_prepares_only_while_a_seat_of_its_panel_works(world, panel):
    root = _pooled(world, "root-v", chat_id=6)
    root["owner_wait"] = {"wait_id": "v1", "state": "waiting", "task_attempt": 1, "reason": "review",
                          "source_ref": {"x": 1}}
    assert _chats(world) == [], "no seat running: the park is a wait"
    panel.build("root-v", 6, local=False)
    panel.seat("a")
    assert _chats(world) == [6], "reviewers prepare the answer the author waits for"
    panel.quota("a")
    assert _chats(world) == [], "its only seat waits for model access"
    panel.resolve("a")
    panel.settle("a")
    assert _chats(world) == []


def test_a_cold_sleep_beside_a_running_helper_is_not_admitted(world):
    """Corrected evidence: a cold root with a working native helper is no supported shape,
    so the projection is not credited with it (``model_sleep.cold_blockers``)."""
    from ouroboros import model_sleep
    from tests._budget_pause_exact_helpers import _loop_ctx

    _pooled(world, "root-c", chat_id=5)
    _pooled(world, "kid-c", chat_id=5, parent="root-c")
    ctx, _limit = _loop_ctx(world.root, "root-c")
    blockers = model_sleep.cold_blockers(ctx, chosen={"tasks": ["kid-c"], "senders": []})
    assert {"kind": "running_member", "detail": "kid-c"} in blockers, blockers


class _Engine:
    """The Claudexor daemon as the body observer reads it: each held run's scripted state."""
    def __init__(self):
        self.state, self.waiting, self.fail = "running", False, False

    def get_run(self, run_id, timeout_sec=None):
        if self.fail:
            raise OSError("daemon unreachable")
        return {"summary": {"runId": run_id, "state": self.state, "waitingOnUser": self.waiting},
                "lastSeq": 1, "pendingInteractions": []}

    def close(self):
        pass


@pytest.fixture
def body(world, monkeypatch):
    """The real body observer of held runs over a scripted daemon, ticking fast."""
    from ouroboros import delegate_collection, delegate_supervision

    monkeypatch.setattr(delegate_supervision, "_TICK_SEC", 0.05)
    engine = _Engine()
    runtime = delegate_collection.start_body_runtime(world.root, gateway_factory=lambda: engine)
    yield SimpleNamespace(engine=engine, runtime=runtime)
    delegate_collection.stop_body_runtime(world.root, wait=True)


def _observed(world, run_id, **expect):
    from ouroboros.delegate_collection import body_runtime

    return _until(lambda: all((body_runtime(world.root).observation(run_id) or {}).get(key) == value
                              for key, value in expect.items()))


def test_a_warm_sleep_on_its_own_run_types_while_the_body_sees_it_work(world, body, monkeypatch):
    """await_messages(mode=warm, runs=[own run]) through request_sleep, wait_after_tools and the
    pool's real park: the selected run counts while the body currently observes it working."""
    from ouroboros import delegate_custody as custody, model_sleep, owner_wait
    from ouroboros.model_wait import TaskModelWait
    from supervisor import worker_owner_wait
    from tests._budget_pause_exact_helpers import _loop_ctx

    supervisor = _Supervisor(world)
    monkeypatch.setattr(world.workers, "get_event_q", lambda **_kw: supervisor)
    meta = _pooled(world, "root-ws", chat_id=6)
    assert custody.emit(world.root, custody.STARTED, {
        "run_id": "run-ws", "invocation_id": "invoke-ws", "task_id": "root-ws", "root_task_id": "root-ws",
        "route": "route", "source": "delegate_start", "access": "readonly"})
    world.workers.WORKERS[meta["worker_id"]] = SimpleNamespace(
        wid=meta["worker_id"], proc=SimpleNamespace(pid=os.getpid()), busy_task_id="root-ws", reaping=False,
        active_capacity=True, in_q=queue_module.Queue())
    ctx, limit = _loop_ctx(world.root, "root-ws")
    ctx.event_queue, ctx.current_chat_id = supervisor, 6
    waiter = ctx.model_wait_context = TaskModelWait(task=meta["task"], drive_root=world.root, event_queue=None,
                                                    worker_slot_held=True)
    waiter.tool_context = ctx
    waiter.restore_continuation({}, started_at=meta["started_at"])
    assert _observed(world, "run-ws", run_state="running")
    assert _until(lambda: "typing_start" in supervisor.kinds()), "a first read that works is announced too"
    armed = model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, runs=["run-ws"]), "warm")
    assert armed["reason"] == "sleep_armed"
    parked, release = threading.Event(), threading.Event()

    def park(_ctx, checkpoint):
        worker_owner_wait.handle_owner_wait({
            "type": "owner_wait", "task_id": "root-ws", "task_attempt": 1, "worker_id": meta["worker_id"],
            "pid": os.getpid(), "wait_id": checkpoint["wait_id"], "phase": "park", "checkpoint": checkpoint},
            world.workers)
        assert _until(lambda: (worker_owner_wait.retry_pending_owner_waits(world.workers)
                               or "owner_wait" in meta)), "the assignment tick completes a deferred park"
        parked.set()
        assert release.wait(10)
        return "run:run-ws"

    ctx.owner_wait_callback = park
    sleeper = threading.Thread(target=owner_wait.wait_after_tools, args=(ctx, limit.messages, {}, {}, 4, [], set()),
                               daemon=True)
    sleeper.start()
    try:
        assert parked.wait(10)
        assert meta["owner_wait"]["reason"] == "sleep" and meta["owner_wait"]["sleep"]["runs"] == ["run-ws"]
        assert _chats(world) == [6], "the sleep's own run works for the reply"
        body.engine.state = "queued"
        assert _observed(world, "run-ws", run_state="queued")
        assert _chats(world) == [], "a run queued for capacity is an access wait"
        announced = supervisor.kinds().count("typing_start")
        body.engine.state = "running"
        assert _observed(world, "run-ws", run_state="running") and _chats(world) == [6]
        assert _until(lambda: supervisor.kinds().count("typing_start") > announced), "working again is announced"
        seen = body.runtime.observation("run-ws")["at"]
        assert _until(lambda: body.runtime.observation("run-ws")["at"] > seen + 0.25)
        assert supervisor.kinds().count("typing_start") == announced + 1, "once per transition, not per read"
        body.engine.fail = True
        assert _observed(world, "run-ws", answered=False)
        state = reply_preparation(world.root)
        assert state["complete"] is False and state["chats"] == [], "an unanswered read is unknown"
        body.engine.fail = False
        assert _observed(world, "run-ws", answered=True) and _chats(world) == [6]
    finally:
        release.set()
        sleeper.join(10)
    assert not sleeper.is_alive()


def test_a_direct_turns_warm_sleep_counts_only_the_runs_it_selected(world, body):
    """A chat turn's warm sleep, parked by its live actor (``direct_owner_wait``): a selected
    run counts while the body sees it work; a sleep that selected no run waits on mail only."""
    from ouroboros import delegate_custody as custody, model_sleep, owner_wait
    from ouroboros.model_wait import TaskModelWait
    from ouroboros.task_results import load_task_result
    from tests._budget_pause_exact_helpers import _loop_ctx

    ctx, limit = _loop_ctx(world.root, "turn-ws", direct=True)
    ctx.current_chat_id, ctx.pending_events, ctx.owner_wait_callback = 1, [], owner_wait.direct_owner_wait
    _direct(world, "turn-ws", chat_id=1, actor=SimpleNamespace(tools=SimpleNamespace(_ctx=ctx)))
    assert custody.emit(world.root, custody.STARTED, {
        "run_id": "run-wd", "invocation_id": "invoke-wd", "task_id": "turn-ws", "root_task_id": "turn-ws",
        "route": "route", "source": "delegate_start", "access": "readonly"})
    assert _observed(world, "run-wd", run_state="running") and _chats(world) == [1]
    stop = threading.Event()
    for runs, expected in ((["run-wd"], [1]), ([], [])):
        ctx.model_wait_context = TaskModelWait(
            task={"id": "turn-ws", "_attempt": 1}, drive_root=world.root, event_queue=None, worker_slot_held=False,
            owner_control=lambda: "cancelled" if stop.is_set() else None)
        ctx.model_wait_context.tool_context = ctx
        model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, runs=runs), "warm")
        assert _chats(world) == expected, "asked in this batch: it already waits on what it selected"
        stop.clear()
        sleeper = threading.Thread(target=owner_wait.wait_after_tools,
                                   args=(ctx, limit.messages, {}, {}, 4, [], set()), daemon=True)
        sleeper.start()
        try:
            assert _until(lambda: (load_task_result(world.root, "turn-ws") or {}).get("owner_wait", {}).get(
                "state") == "waiting")
            assert _chats(world) == expected, "parked: still only what the sleep selected"
            if runs:
                body.engine.state = "queued"
                assert _observed(world, "run-wd", run_state="queued") and _chats(world) == []
                body.engine.state = "running"
                assert _observed(world, "run-wd", run_state="running") and _chats(world) == [1]
        finally:
            stop.set()
            sleeper.join(10)
        assert not sleeper.is_alive() and _chats(world) == [1], "awake again: the turn's own work"


def test_a_cold_sleep_on_its_own_run_types_while_the_body_holds_it(world, body, monkeypatch):
    """await_messages(mode=cold, runs=[own run]) through request_sleep, the body's exact acceptance,
    the release and the queue's park: the run counts through the transition and while the body
    holds it; another task's observation, a stopped observer or a lost receipt are unknown."""
    from ouroboros import budget_pause as bp, model_sleep, sleep_custody as sc
    from supervisor.events_budget import install_exact_budget_pause
    from tests import test_cold_external_custody as cold
    from tests._budget_pause_exact_helpers import _supervisor_ctx

    env = cold._setup(world.root, monkeypatch)  # re-installs the queue: record events after it
    supervisor = _Supervisor(world)
    monkeypatch.setattr(sc, "_runtime", lambda _root: body.runtime)
    monkeypatch.setattr(world.workers, "get_event_q", lambda **_kw: supervisor)
    env.task.update(chat_id=6, metadata=dict(ORIGIN))
    assert _observed(world, "run-one", run_state="running")
    assert _chats(world) == [6], "the author works before it sleeps"
    model_sleep.request_sleep(env.ctx, model_sleep.selectors(env.ctx, runs=["run-one"]), "cold")
    during = []

    def host_tick(_ctx):
        transfer = lambda: bp.budget_pause_row(world.root, "sleeper").get("external_custody", {}).get("state")
        if transfer() == "requested":
            during.append(("requested", _chats(world)))
        sc.reconcile(env.q, runtime=body.runtime)
        if transfer() == "accepted":
            during.append(("accepted", _chats(world)))
        return ""

    monkeypatch.setattr(bp, "_hold_control_reason", host_tick)
    with pytest.raises(bp.BudgetPauseRequested) as raised:
        bp.enter_cold_sleep(env.limit)
    assert during[:2] == [("requested", [6]), ("accepted", [6])], "releasing the process is no gap"
    row = raised.value.pause
    install_exact_budget_pause(_supervisor_ctx(world.root, env.workers, env.q, [], []), "sleeper",
                               bp.exact_pause_marker(row)["checkpoint"], evt=bp.pause_event(env.task, row))
    bp.end_dispatch_fence("sleeper")
    [parked] = env.workers.PENDING
    assert parked["_budget_pause"]["reason"] == "sleep" and sc.carrier_receipt(parked)["runs"] == [
        ["run-one", "invoke-one"]] and env.workers.RUNNING == {}
    assert _chats(world) == [6], "a cold sleeper's body-held run works for the reply"
    body.engine.state = "queued"
    assert _observed(world, "run-one", run_state="queued") and _chats(world) == []
    announced = supervisor.kinds().count("typing_start")
    body.engine.state = "running"
    assert _observed(world, "run-one", run_state="running") and _chats(world) == [6]
    assert _until(lambda: supervisor.kinds().count("typing_start") > announced), "working again is announced"
    for wrong in ({"task_id": "another"}, {"invocation_id": "invoke-other"}, {"at": time.monotonic() - 60}):
        with body.runtime.lock:
            saved = dict(body.runtime.observed["run-one"])
            body.runtime.observed["run-one"].update(wrong)
            state = reply_preparation(world.root)
            body.runtime.observed["run-one"] = saved
        assert state["chats"] == [] and state["complete"] is False, wrong
    receipt = parked["_budget_pause"].pop("external_custody")
    state = reply_preparation(world.root)
    assert state["chats"] == [] and state["complete"] is True, "no transfer recorded: the selected run is not held"
    parked["_budget_pause"]["external_custody"] = {**receipt, "binding": {"pause_id": "another"}}
    state = reply_preparation(world.root)
    assert state["chats"] == [] and state["complete"] is False, "a receipt that does not bind this park is unknown"
    parked["_budget_pause"]["external_custody"] = receipt
    assert _chats(world) == [6]
    from ouroboros.delegate_collection import stop_body_runtime

    stop_body_runtime(world.root, wait=True)
    state = reply_preparation(world.root)
    assert state["chats"] == [] and state["complete"] is False, "no current observer: unknown, never a positive"


def test_a_releasing_sleep_reads_only_a_transfer_bound_to_its_own_pause(world, body):
    """Negative control on durable rows no owner writes this way (``confirm_release`` binds every
    request it saves): while a pooled sleep releases its process, a requested transfer that binds
    another pause is unknown, never the run's positive."""
    from ouroboros import delegate_custody as custody

    _pooled(world, "root-q", chat_id=6)
    assert custody.emit(world.root, custody.STARTED, {
        "run_id": "run-q", "invocation_id": "invoke-q", "task_id": "root-q", "root_task_id": "root-q",
        "route": "route", "source": "delegate_start", "access": "readonly"})
    assert _observed(world, "run-q", run_state="running")
    pause = {"state": "pausing", "reason": "sleep", "pause_id": "p-1", "task_attempt": 1,
             "sleep": {"sleep_id": "s-1", "mode": "cold", "runs": ["run-q"]}}
    for pause_id, chats, complete in (("p-1", [6], True), ("p-0", [], False)):
        write_task_result(world.root, "root-q", STATUS_RUNNING, chat_id=6, root_task_id="root-q", budget_pause={
            **pause, "external_custody": {"state": "requested", "runs": [["run-q", "invoke-q"]],
                                          "binding": {"pause_id": pause_id, "task_attempt": 1}}})
        state = reply_preparation(world.root)
        assert sorted(row["chat_id"] for row in state["chats"]) == chats and state["complete"] is complete, pause_id


def test_two_roots_share_one_chat_and_unknown_never_cancels_a_known_positive(world, monkeypatch):
    from ouroboros.gateway import state as census

    _direct(world, "turn-a", chat_id=1)
    _pooled(world, "root-b", chat_id=1)
    _direct(world, "turn-u", chat_id=9)
    assert _chats(world) == [1, 9] and len(reply_preparation(world.root)["chats"]) == 2
    real = census._task_activity_facts
    monkeypatch.setattr(census, "_task_activity_facts",
                        lambda root, tid: {"unavailable": True} if tid == "turn-u" else real(root, tid))
    state = reply_preparation(world.root)
    assert state["complete"] is False
    assert sorted(row["chat_id"] for row in state["chats"]) == [1], "unknown makes no positive, cancels none"


def test_the_destination_is_the_answers_own_route_with_that_chats_transport(world):
    from ouroboros import projects_registry as pr

    world.bridge.activate_update_transport({"chat_id": 214251305, "transport": {
        "kind": "telegram", "conversation_id": "214251305"}})
    _direct(world, "turn-t", chat_id=214251305)
    _pooled(world, "root-r", chat_id=1)
    project = pr.create_project(world.root, "room-r", name="room-r")
    pr.bind_task_to_project(world.root, "root-r", project["id"], origin={"absent": "system"})
    chats = {row["chat_id"]: row["transport"] for row in reply_preparation(world.root)["chats"]}
    assert chats == {214251305: {"kind": "telegram", "conversation_id": "214251305"}, project["chat_id"]: {}}


def _supervised(world, events):
    """Drain a worker queue into the supervisor's real cognitive-operation handler."""
    from supervisor.cognitive_operations import _handle_cognitive_operation

    seen = []
    while not events.empty():
        event = events.get_nowait()
        seen.append(event)
        if event.get("type") == "cognitive_operation":
            _handle_cognitive_operation(event, SimpleNamespace(RUNNING=world.workers.RUNNING))
    return seen


def test_real_in_slot_tools_declare_their_wait_and_re_announce_at_return(world):
    from ouroboros.tools.control_task_results import _await_messages, _wait_for_task
    from ouroboros.tools.tool_context import ToolContext

    _pooled(world, "root-i", chat_id=8)
    assert _chats(world) == [8]
    for call in (lambda ctx: _await_messages(ctx, timeout_sec=1),
                 lambda ctx: _wait_for_task(ctx, "never-scheduled", timeout_sec=1)):
        events = queue_module.Queue()
        ctx = ToolContext(repo_dir=world.root, drive_root=world.root)
        ctx.task_id, ctx.task_attempt, ctx.event_queue, ctx.current_chat_id = "root-i", 1, events, 8
        tool = threading.Thread(target=call, args=(ctx,), daemon=True)
        tool.start()
        deadline = time.monotonic() + 5
        while events.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        _supervised(world, events)
        assert tool.is_alive() and _chats(world) == [], "a slot held on mail or a task terminal is no work"
        tool.join(10)
        seen = _supervised(world, events)
        assert _chats(world) == [8]
        assert [event["type"] for event in seen][-1] == "typing_start"


def test_a_resolved_access_wait_re_announces_and_an_interrupted_one_does_not(world, monkeypatch):
    from ouroboros import review_operation
    from ouroboros.model_wait import ModelWaitInterrupted, TaskModelWait

    write_task_result(world.root, "root-m", STATUS_RUNNING, chat_id=1)

    def drain(self):
        for row in self.waits.values():
            if row.get("state") == "waiting":
                row["_action"] = {"action": "retry", "request_id": "r-1"}

    monkeypatch.setattr(TaskModelWait, "_drain_controls", drain)
    monkeypatch.setattr(review_operation._OperationWait, "_drain_controls", drain)
    quota = SimpleNamespace(code="subscription_window_exhausted")
    # The author resolved; the author interrupted; a reviewer of the task's panel resolved;
    # an owner that is neither (a post-task owner, no context and no panel) resolved.
    for closed, owner, announced in ((False, "author", True), (True, "author", False), (False, "panel", True),
                                     (False, "other", False)):
        events = queue_module.Queue()
        waiter = TaskModelWait(task={"id": "root-m", "chat_id": 1}, drive_root=world.root,
                               event_queue=events, worker_slot_held=True)
        if owner == "author":
            waiter.tool_context = SimpleNamespace(event_queue=events, task_id="root-m", current_chat_id=1)
        if owner == "panel":
            waiter = review_operation.ReviewOperation(
                parent=waiter, request=SimpleNamespace(surface="multi_model_review", retry_key="rk", policy={}),
                task_id="root-m", result_root=world.root).wait
        waiter.closed = closed
        try:
            waiter.wait(None, quota, {"model_role": "main", "model": "plain-model"})
        except ModelWaitInterrupted:
            assert closed
        kinds = [event["type"] for event in _supervised(world, events)]
        assert ("typing_start" in kinds) is announced, (owner, kinds)


@pytest.mark.parametrize("outcome,announced", [("control:owner_resume", True), ("control:cancelled", False)])
def test_a_warm_pause_park_re_announces_only_on_the_owners_resume(world, outcome, announced):
    from ouroboros.owner_wait import park_owner_pause_warm
    from tests._budget_pause_exact_helpers import _loop_ctx

    write_task_result(world.root, "root-z", STATUS_RUNNING, chat_id=1)
    ctx, limit = _loop_ctx(world.root, "root-z")
    events = queue_module.Queue()
    ctx.event_queue, ctx.current_chat_id = events, 1
    ctx.owner_wait_callback = lambda _ctx, _checkpoint: outcome
    park_owner_pause_warm(limit, ctx, fence={"fence_id": "f", "generation": 1, "root_task_id": "root-z"},
                          detached=[])
    kinds = [event["type"] for event in _supervised(world, events)]
    assert ("typing_start" in kinds) is announced, kinds
