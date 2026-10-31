"""Telegram typing from real producers to a fake Telegram client (DESIGN §9).

The direct chat lane, the answer-first final, the owner-wait resume seam, the
delegated-run supervision loop, a review panel's seats and the supervisor's
send/typing/lease handlers run for real; their frames cross the real
``LocalChatBridge`` and the global event bus into the Telegram plugin's own
handlers and its one ``TypingCue``, which reads the host's real projection. Only
Telegram itself is a recorder. Negative controls stay beside each positive one.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import queue as queue_module
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx

pytestmark = pytest.mark.serial

PINNED = 777
ORIGIN = {"origin_message_ref": {"chat_id": 1, "client_message_id": "m-1", "ts": "2026-10-17T00:00:00+00:00"}}
TICK = 0.08


def _plugin():
    root = Path(__file__).resolve().parents[1] / "skills" / "telegram"
    pkg = types.ModuleType("tg_flow"); pkg.__path__ = [str(root)]; sys.modules["tg_flow"] = pkg
    spec = importlib.util.spec_from_file_location("tg_flow.plugin", root / "plugin.py")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Telegram:
    """Every Bot API call the bridge makes, in order, with its monotonic time."""
    def __init__(self):
        self.calls, self.lock = [], threading.Lock()

    def record(self, kind, chat):
        with self.lock:
            self.calls.append((kind, chat, time.monotonic()))

    def kinds(self, since=0.0):
        with self.lock:
            return [(kind, chat) for kind, chat, at in self.calls if at >= since]

    def typing_times(self, chat):
        with self.lock:
            return [at for kind, at_chat, at in self.calls if kind == "typing" and at_chat == chat]

    async def send_chat_action(self, chat_id, action="typing"):
        self.record("typing", chat_id)

    async def send_message(self, chat_id, text, parse_mode=""):
        self.record("message", chat_id)
        return len(self.calls)


@pytest.fixture
def flow(tmp_path, monkeypatch):
    from ouroboros import event_bus
    from supervisor import message_bus, state, workers
    from supervisor.active_activity import get_direct_activity_registry
    from supervisor.cognitive_operations import _handle_cognitive_operation
    from supervisor.events_chat_delivery import _handle_send_message, _handle_typing_start
    from supervisor.task_model_wait import handle_task_model_wait

    _queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {})
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, bridge=bridge,
                          send_with_budget=message_bus.send_with_budget, append_jsonl=lambda *a, **k: None)
    handlers = {"send_message": _handle_send_message, "typing_start": _handle_typing_start,
                "cognitive_operation": _handle_cognitive_operation, "task_model_wait": handle_task_model_wait}
    produced = []

    class SupervisorQueue:
        """The supervisor's drain, synchronously: each frame reaches its real handler."""
        def put(self, event, *_a, **_k):
            produced.append(str(event.get("type") or ""))
            handler = handlers.get(str(event.get("type") or ""))
            if handler is not None:
                handler(event, ctx)

        put_nowait = put

    supervisor_queue = SupervisorQueue()
    monkeypatch.setattr(workers, "REPO_DIR", tmp_path / "repo")
    monkeypatch.setattr(workers, "get_event_q", lambda **_kw: supervisor_queue)
    monkeypatch.setattr(workers, "send_with_budget", message_bus.send_with_budget)
    monkeypatch.setattr(state, "load_state", lambda: {"bg_consciousness_enabled": True})
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)
    workers.open_repo_writer_admission()
    registry = get_direct_activity_registry()
    registry.clear()

    plugin = _plugin()
    typing_mod = sys.modules["tg_flow.lib.telegram_typing"]
    monkeypatch.setattr(typing_mod, "_REFRESH_SEC", TICK)
    monkeypatch.setattr(typing_mod, "_MIN_READ_GAP_SEC", 0.01)
    telegram = Telegram()
    monkeypatch.setattr(plugin, "_telegram_client", lambda *_a, **_k: telegram)
    state_dir = tmp_path / "state" / "skills" / "telegram"
    state_dir.mkdir(parents=True)
    api = SimpleNamespace(get_state_dir=lambda: str(state_dir), get_settings=lambda keys: {},
                          get_runtime_info=lambda: {"data_dir": str(tmp_path)}, log=lambda *a, **k: None)

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    bus = event_bus.init_global_event_bus()
    bus.set_loop(loop)
    world = SimpleNamespace(root=tmp_path, workers=workers, bridge=bridge, queue=supervisor_queue, ctx=ctx,
                            plugin=plugin, api=api, telegram=telegram, registry=registry, loop=loop, bus=bus,
                            state_dir=state_dir, cue=None, future=None, reads=[0], produced=produced)

    def settings(**values):
        (state_dir / "settings.json").write_text(json.dumps(values))

    def start():
        cue = plugin._typing_cue(api)
        real_read = cue._read_state

        def counted():
            world.reads[0] += 1
            return real_read()

        cue._read_state = counted
        for sub in list(bus.snapshot()):
            bus.unsubscribe(sub)
        bus.subscribe("telegram", "chat.typing", plugin._make_typing(cue))
        bus.subscribe("telegram", "chat.outbound", plugin._make_outbound(api, cue))
        world.cue, world.future = cue, asyncio.run_coroutine_threadsafe(cue.run(), loop)

    def stop():
        if world.future is not None:
            world.future.cancel()
            time.sleep(0.05)

    world.settings, world.start, world.stop = settings, start, stop
    settings(TELEGRAM_CHAT_ID=str(PINNED), TELEGRAM_MIRROR_MODE="all")
    try:
        yield world
    finally:
        stop()
        # The cue's reads run on the loop's default executor: retire it with the loop.
        asyncio.run_coroutine_threadsafe(loop.shutdown_default_executor(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()
        registry.clear()
        event_bus.init_global_event_bus()


def _until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _running(flow, task_id, *, parent="", chat_id=1, owner_wait=None):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    lineage = {"root_task_id": task_id} if not parent else {
        "root_task_id": parent, "parent_task_id": parent, "delegation_role": "subagent"}
    write_task_result(flow.root, task_id, STATUS_RUNNING, chat_id=chat_id, **lineage)
    meta = {"task": {"id": task_id, "type": "task", "chat_id": chat_id, "_attempt": 1, "metadata": dict(ORIGIN),
                     **lineage}, "attempt": 1, "worker_id": 0, "started_at": time.time()}
    if owner_wait:
        meta["owner_wait"] = {"wait_id": "w", "state": "waiting", "task_attempt": 1, "source_ref": {"x": 1},
                              **owner_wait}
    flow.workers.RUNNING[task_id] = meta
    return meta


def _quiet(flow, chat=PINNED, ticks=4):
    """No action reaches ``chat`` for a few renewal periods."""
    since = time.monotonic()
    time.sleep(TICK * ticks)
    return ("typing", chat) not in flow.telegram.kinds(since=since)


def test_owner_turn_types_until_its_answer_and_post_answer_work_never_revives_it(flow, monkeypatch):
    from ouroboros import agent as agent_module
    from ouroboros import agent_task_pipeline as pipeline
    from supervisor import state as supervisor_state

    monkeypatch.setattr(supervisor_state, "reconstruct_task_cost", lambda *a, **k: {})
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    interim_go, answer_go = threading.Event(), threading.Event()
    store_entered, store_release = threading.Event(), threading.Event()
    real_store = pipeline._store_task_result

    def blocked_store(*args, **kwargs):
        store_entered.set()
        assert store_release.wait(10)
        return real_store(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_store_task_result", blocked_store)
    env = SimpleNamespace(drive_root=flow.root, repo_dir=flow.root)

    class Actor:
        _event_queue = None

        def handle_task(self, task):
            assert interim_go.wait(10)
            self._event_queue.put({"type": "send_message", "chat_id": task["chat_id"], "task_id": task["id"],
                                   "text": "Checking the sources first.", "system_type": "proactive_message"})
            assert answer_go.wait(10)
            pipeline.emit_task_results(env, None, None, [], task, "The answer.",
                                       {"execution_status": "ok", "reason_code": "final_message"},
                                       {"tool_calls": [], "reasoning_notes": []}, 0.0, flow.root / "logs",
                                       event_queue=self._event_queue)
            return []

    monkeypatch.setattr(agent_module, "make_agent", lambda **kw: Actor())
    flow.start()
    turn = threading.Thread(target=flow.workers.handle_chat_direct, args=(1, "question", None),
                            kwargs={"task_metadata": {**ORIGIN, "client_message_id": "m-1"}}, daemon=True)
    turn.start()
    try:
        assert _until(lambda: flow.telegram.kinds().count(("typing", PINNED)) >= 3), flow.telegram.kinds()
        interim_go.set()  # intermediate speech: the reply is still being prepared
        assert _until(lambda: ("message", PINNED) in flow.telegram.kinds())
        spoke = time.monotonic()
        assert _until(lambda: ("typing", PINNED) in flow.telegram.kinds(since=spoke)), "speech ended the cue"
        answer_go.set()
        assert store_entered.wait(10)
        assert _until(lambda: flow.telegram.kinds().count(("message", PINNED)) == 2)
        answered = time.monotonic()
        time.sleep(TICK * 6)  # the turn is alive in its slow store: post-answer work
        assert flow.registry.get(flow.registry.actors()[0].activity_id).reply_handed_off is True
        after = flow.telegram.kinds(since=answered)
        assert ("typing", PINNED) not in after, after
        calls = flow.telegram.kinds()
        assert calls[-1] == ("message", PINNED) and calls.count(("message", PINNED)) == 2
    finally:
        interim_go.set(), answer_go.set(), store_release.set()
        turn.join(10)
    assert flow.registry.actors() == []


def test_a_wake_up_turn_and_a_helper_start_never_flash_typing(flow, monkeypatch):
    """A start event is no cue: a consciousness turn's announce and a delegated child's
    start reach the bridge, and nothing is shown (the bridge used to flash once per event)."""
    from ouroboros import agent as agent_module

    release = threading.Event()

    class Actor:
        def handle_task(self, task):
            assert release.wait(10)
            return []

    monkeypatch.setattr(agent_module, "make_agent", lambda **kw: Actor())
    flow.start()
    receipt = flow.workers.handle_wake_direct(1, "wake", {"initiator": "consciousness",
                                                           "usage_category": "consciousness"})
    assert receipt["admitted"] is True
    flow.workers.RUNNING["kid"] = {"task": {"id": "kid", "chat_id": 1, "parent_task_id": "absent-root",
                                            "root_task_id": "absent-root", "delegation_role": "subagent"},
                                   "attempt": 1, "worker_id": 0}
    reads = flow.reads[0]
    flow.queue.put({"type": "typing_start", "task_id": "kid", "chat_id": 1, "phase": "thinking"})
    assert _until(lambda: flow.reads[0] > reads), "the start woke the cue"
    time.sleep(TICK * 4)
    release.set()
    assert _until(lambda: not any(thread.name.startswith("wake-turn-") and thread.is_alive()
                                  for thread in threading.enumerate()))
    assert flow.telegram.kinds() == []


def test_a_warm_resume_re_announces_without_a_new_turn_and_a_stop_does_not(flow):
    from ouroboros.owner_wait import wait_after_tools

    meta = _running(flow, "root-w", owner_wait={"reason": "owner"})
    flow.start()
    assert _until(lambda: flow.reads[0] >= 1)
    time.sleep(TICK * 4)
    assert flow.telegram.kinds() == [], "a root waiting on the owner renews nothing"
    reads = flow.reads[0]
    time.sleep(TICK * 4)
    assert flow.reads[0] == reads, "and the idle cue reads nothing"

    def resume(outcome):
        def callback(ctx, checkpoint):
            meta["owner_wait"] = {**meta["owner_wait"], "state": "resumed"}  # the pool's grant
            return outcome
        ctx, limit = _loop_ctx(flow.root, "root-w")
        ctx.event_queue, ctx.current_chat_id, ctx.owner_wait_callback = flow.queue, 1, callback
        ctx._owner_wait_requested = "quiz-1"
        wait_after_tools(ctx, limit.messages, limit.llm_trace, limit.accumulated_usage, limit.round_idx,
                         limit.tool_schemas, limit.owner_msg_seen)

    resume("control:finalize_now")
    time.sleep(TICK * 4)
    assert flow.telegram.kinds() == [], "a stopping control is no resumption"
    meta["owner_wait"]["state"] = "waiting"
    resume("answer")
    assert _until(lambda: ("typing", PINNED) in flow.telegram.kinds()), "the same attempt types again"


def test_one_action_per_chat_whatever_backs_it_and_reload_rediscovers_quiet_work(flow):
    for task_id, parent in (("root-1", ""), ("root-2", ""), ("kid-1", "root-1"), ("kid-2", "root-1")):
        _running(flow, task_id, parent=parent)
    flow.start()
    assert _until(lambda: len(flow.telegram.kinds()) >= 2)
    started = time.monotonic()
    time.sleep(TICK * 10)
    actions = [call for call in flow.telegram.kinds(since=started) if call[0] == "typing"]
    # Two roots and two helpers on one chat: one renewal per tick (10 ticks), never one per task.
    assert set(actions) == {("typing", PINNED)} and 2 <= len(actions) <= 12, actions
    flow.stop()  # the skill unloads mid-preparation
    unloaded = time.monotonic()
    time.sleep(TICK * 4)
    assert flow.telegram.kinds(since=unloaded + 0.05) == []
    flow.start()  # reload: no event announces the quiet work, the first read finds it
    assert _until(lambda: ("typing", PINNED) in flow.telegram.kinds(since=time.monotonic() - TICK))


def test_an_event_burst_keeps_one_action_per_chat_per_renewal(flow):
    _running(flow, "root-b")
    _running(flow, "kid-b", parent="root-b")
    flow.start()
    assert _until(lambda: ("typing", PINNED) in flow.telegram.kinds())
    started = time.monotonic()
    while time.monotonic() - started < TICK * 10:
        flow.queue.put({"type": "typing_start", "task_id": "kid-b", "chat_id": 1, "phase": "thinking"})
        time.sleep(0.004)
    times = [at for at in flow.telegram.typing_times(PINNED) if at >= started]
    assert flow.reads[0] > 12, "the burst did wake the cue"
    gaps = [later - earlier for earlier, later in zip(times, times[1:])]
    assert len(times) <= 12 and min(gaps) >= TICK * 0.8, (len(times), gaps)


def test_web_and_project_roots_follow_the_mirror_policy_and_rehome(flow):
    from ouroboros import projects_registry as pr

    for mode, expected in (("all", {PINNED}), ("telegram_only", set())):
        flow.workers.RUNNING.clear()
        flow.settings(TELEGRAM_CHAT_ID=str(PINNED), TELEGRAM_MIRROR_MODE=mode)
        flow.bridge.activate_update_transport({"chat_id": PINNED, "transport": {
            "kind": "telegram", "conversation_id": str(PINNED)}})
        _running(flow, f"root-x-{mode}", chat_id=PINNED)
        flow.start()
        assert _until(lambda: ("typing", PINNED) in flow.telegram.kinds(since=time.monotonic() - TICK)), \
            "a Telegram-born root types in any mode"
        project = pr.create_project(flow.root, f"room-{mode}", name=f"room-{mode}")
        pr.bind_task_to_project(flow.root, f"root-x-{mode}", project["id"], origin={"absent": "system"})
        time.sleep(TICK * 3)  # rehomed: the answer would now go to the project room
        since = time.monotonic()
        time.sleep(TICK * 5)
        assert {chat for kind, chat in flow.telegram.kinds(since=since) if kind == "typing"} == expected
        flow.stop()


def test_a_held_delegated_run_types_while_it_works_and_resumes_by_itself(flow):
    """running → paused on its question → running again: the idle cue is woken by the
    supervision owner when the held run works again, with no new turn or start."""
    from ouroboros.delegate_supervision import supervised_wait

    _running(flow, "root-n", owner_wait={"reason": "owner"})
    _running(flow, "nanny", parent="root-n")
    ctx = SimpleNamespace(task_id="nanny", task_attempt=1, drive_root=flow.root, budget_drive_root=str(flow.root),
                          task_metadata={"root_task_id": "root-n", "parent_task_id": "root-n",
                                         "delegation_role": "subagent"},
                          event_queue=flow.queue, current_chat_id=1)
    payloads, calls = queue_module.Queue(), [0]

    def wait_once(*_a, **_k):
        calls[0] += 1
        return json.dumps(payloads.get(timeout=10))

    def tick(**observed):
        before = calls[0]
        payloads.put({"status": "no_progress", "run_id": "run-1", "state": "running", "last_seq": 1,
                      "waiting_on_user": False, **observed})
        assert _until(lambda: calls[0] > before)

    flow.start()
    hold = threading.Thread(target=supervised_wait, args=(ctx, "run-1"), kwargs={"wait_once": wait_once}, daemon=True)
    hold.start()
    try:
        assert _until(lambda: calls[0] == 1)
        tick()
        assert _until(lambda: ("typing", PINNED) in flow.telegram.kinds()), "the observed running run types"
        tick(waiting_on_user=True)
        time.sleep(TICK * 2)
        assert _quiet(flow) and _quiet(flow), "a run paused on its question renews nothing"
        reads = flow.reads[0]
        time.sleep(TICK * 3)
        assert flow.reads[0] == reads, "the idle cue reads nothing while the run waits"
        tick()
        assert _until(lambda: ("typing", PINNED) in flow.telegram.kinds(since=time.monotonic() - TICK)), \
            "working again types again without any new start"
    finally:
        payloads.put({"status": "completed", "run_id": "run-1", "state": "succeeded", "last_seq": 2})
        hold.join(10)


def test_a_review_panel_types_while_a_seat_works_and_resumes_when_quota_returns(flow, monkeypatch):
    from ouroboros import config, review_operation
    from ouroboros.model_wait import TaskModelWait
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, ActiveReviewAttempt, _emit_operation
    from ouroboros.review_pause import author_drain
    from ouroboros.review_records import ReviewSlot
    from ouroboros.usage_accounting import UsageScope, usage_scope

    monkeypatch.setattr(config, "CLAUDEXOR_MODEL_POLL_INTERVAL_SEC", 0.02)
    released = threading.Event()

    def drain(self):
        for row in self.waits.values():
            if row.get("state") == "waiting" and released.is_set():
                row["_action"] = {"action": "retry", "request_id": "r-1"}

    monkeypatch.setattr(review_operation._OperationWait, "_drain_controls", drain)
    _running(flow, "root-r")
    tool_context = SimpleNamespace(task_id="root-r", task_attempt=1, event_queue=flow.queue, current_chat_id=1,
                                   task_metadata=dict(ORIGIN))
    parent = TaskModelWait(task={"id": "root-r", "chat_id": 1, "_attempt": 1, "metadata": dict(ORIGIN)},
                           drive_root=flow.root, event_queue=flow.queue, worker_slot_held=True)
    parent.tool_context = tool_context
    request = SimpleNamespace(surface="multi_model_review", retry_key="rk", policy={}, task_attempt=1)
    operation = review_operation.ReviewOperation(parent=parent, request=request, task_id="root-r",
                                                 result_root=flow.root)
    slot = ReviewSlot("a", "plain-model")
    entry = ActiveReviewAttempt(key="k-a", operation_id="op-a", assignment=(request, slot), operation=operation)
    usage = SimpleNamespace(event_queue=flow.queue, execution_id="e")

    def reviewer_waits_for_quota():
        with usage_scope(UsageScope(review_slot_id="a")):
            operation.wait.wait(None, SimpleNamespace(code="subscription_window_exhausted"),
                                {"model_role": "review", "model": "plain-model"})

    waiting = threading.Thread(target=reviewer_waits_for_quota, daemon=True)
    with review_operation._LOCK:
        review_operation._LIVE[operation.owner_id] = operation
    with _ACTIVE_LOCK:
        _ACTIVE[entry.key] = entry
    try:
        with author_drain(parent, operation):
            flow.start()
            assert _until(lambda: flow.reads[0] > 0)
            assert _quiet(flow), "no seat exists yet; the first read cannot invent one"
            reads = flow.reads[0]
            time.sleep(TICK * 3)
            assert flow.reads[0] == reads, "unknown without a previous positive leaves the cue idle"
            _emit_operation(usage, task_id="root-r", request=request, entry=entry, slot=slot, phase="started")
            assert _until(lambda: ("typing", PINNED) in flow.telegram.kinds()), "its one seat works"
            waiting.start()
            assert _until(lambda: "a" in operation.wait.waiting_slots())
            time.sleep(TICK * 2)
            assert _quiet(flow) and _quiet(flow), "every live seat waits for model access"
            released.set()  # the quota came back: the reviewer's own wait owner announces it
            assert _until(lambda: ("typing", PINNED) in flow.telegram.kinds(since=time.monotonic() - TICK)), \
                "the seat works again, and the idle cue knew without a new start"
    finally:
        released.set()
        if waiting.ident is not None:
            waiting.join(5)
        with _ACTIVE_LOCK:
            _ACTIVE.pop(entry.key, None)
        with review_operation._LOCK:
            review_operation._LIVE.pop(operation.owner_id, None)


def test_stop_now_ends_the_cue_at_its_next_read(flow):
    from ouroboros.cancel_intents import STOP_POLICY_IMMEDIATE

    _running(flow, "root-s")
    flow.start()
    assert _until(lambda: ("typing", PINNED) in flow.telegram.kinds())
    path = flow.root / "state" / "cancel_intents.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"intents": {"root-s": {"task_id": "root-s", "request_id": "stop-1", "scope": "single",
                                                       "stop_policy": STOP_POLICY_IMMEDIATE, "state": "active",
                                                       "requested_at": "2026-10-17T00:00:00+00:00"}}}))
    time.sleep(TICK * 2)
    assert _quiet(flow) and _quiet(flow), "Stop now: nothing more is shown for that work"


def test_a_warm_sleep_on_its_own_run_types_and_resumes_when_the_body_sees_it_work_again(flow, monkeypatch):
    """await_messages(mode=warm, runs=[own run]): the pool's real park, the body observer's own
    reads, and its working-again announce through the supervisor's typing handler to the cue."""
    from ouroboros import delegate_collection, delegate_custody as custody, delegate_supervision, model_sleep
    from ouroboros import owner_wait
    from ouroboros.model_wait import TaskModelWait
    from supervisor import worker_owner_wait

    class Engine:
        state = "running"

        def get_run(self, run_id, timeout_sec=None):
            return {"summary": {"runId": run_id, "state": self.state}, "lastSeq": 1, "pendingInteractions": []}

        def close(self):
            pass

    engine = Engine()
    monkeypatch.setattr(delegate_supervision, "_TICK_SEC", 0.05)
    meta = _running(flow, "root-s")
    assert custody.emit(flow.root, custody.STARTED, {
        "run_id": "run-s", "invocation_id": "invoke-s", "task_id": "root-s", "root_task_id": "root-s",
        "route": "route", "source": "delegate_start", "access": "readonly"})
    flow.workers.WORKERS[0] = SimpleNamespace(wid=0, proc=SimpleNamespace(pid=os.getpid()), busy_task_id="root-s",
                                              reaping=False, active_capacity=True, in_q=queue_module.Queue())
    runtime = delegate_collection.start_body_runtime(flow.root, gateway_factory=lambda: engine)
    ctx, limit = _loop_ctx(flow.root, "root-s")
    ctx.event_queue, ctx.current_chat_id = flow.queue, 1
    waiter = ctx.model_wait_context = TaskModelWait(task=meta["task"], drive_root=flow.root, event_queue=None,
                                                    worker_slot_held=True)
    waiter.tool_context = ctx
    waiter.restore_continuation({}, started_at=meta["started_at"])
    parked, release = threading.Event(), threading.Event()

    def park(_ctx, checkpoint):
        worker_owner_wait.handle_owner_wait({
            "type": "owner_wait", "task_id": "root-s", "task_attempt": 1, "worker_id": 0, "pid": os.getpid(),
            "wait_id": checkpoint["wait_id"], "phase": "park", "checkpoint": checkpoint}, flow.workers)
        assert _until(lambda: (worker_owner_wait.retry_pending_owner_waits(flow.workers)
                               or "owner_wait" in meta)), "the assignment tick completes a deferred park"
        parked.set()
        assert release.wait(10)
        return "run:run-s"

    ctx.owner_wait_callback = park
    sleeper = threading.Thread(target=owner_wait.wait_after_tools, args=(ctx, limit.messages, {}, {}, 4, [], set()),
                               daemon=True)
    try:
        assert _until(lambda: runtime.observation("run-s").get("run_state") == "running")
        assert model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, runs=["run-s"]), "warm")["reason"] == "sleep_armed"
        sleeper.start()
        assert parked.wait(10) and meta["owner_wait"]["reason"] == "sleep"
        flow.start()
        assert _until(lambda: ("typing", PINNED) in flow.telegram.kinds()), "the sleep's working run types"
        engine.state = "queued"
        assert _until(lambda: runtime.observation("run-s").get("run_state") == "queued")
        time.sleep(TICK * 2)
        assert _quiet(flow) and _quiet(flow), "a run queued for capacity renews nothing"
        reads = flow.reads[0]
        time.sleep(TICK * 3)
        assert flow.reads[0] == reads, "the idle cue reads nothing while the run is queued"
        engine.state = "running"
        assert _until(lambda: ("typing", PINNED) in flow.telegram.kinds(since=time.monotonic() - TICK)), \
            "the body saw it work again and said so: typing resumes with no new start"
    finally:
        release.set()
        sleeper.join(10)
        delegate_collection.stop_body_runtime(flow.root, wait=True)
