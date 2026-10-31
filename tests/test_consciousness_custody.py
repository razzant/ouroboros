"""The own clock follows one wake through binding, callback gaps and carrier changes."""

import threading
from types import SimpleNamespace

import pytest

from ouroboros import consciousness as clock_module
from ouroboros.consciousness import APPOINTMENT_STATE_KEY, BackgroundConsciousness
from supervisor.active_activity import get_direct_activity_registry
from tests import test_consciousness as clocks
from tests.test_consciousness import T0, FLOOR, DEFAULT, _launched, _register

clock = clocks.clock


@pytest.mark.parametrize("shape", ["main", "project", "parked", "auto_resume", "post_answer"])
def test_owner_work_never_holds_its_own_clock(clock, shape):
    registry = get_direct_activity_registry()
    entry = registry.register("owner", 7, project_id="room" if shape == "project" else "")
    if shape != "auto_resume":
        registry.unregister("owner")
        entry = _register("owner")
        entry.project_id = "room" if shape == "project" else ""
    if shape == "parked":
        registry.mark_parked("owner", {"state": "waiting", "quiz_id": "q1"}, drive_root=clock.root)
    if shape == "post_answer":
        entry.reply_handed_off = True
        entry.actor._busy = False
    assert clock.clock.live_wake() == ""
    assert clock.clock.tick(T0 + FLOOR + 1) == "launched"
    assert registry.get("owner") is entry
    assert clock.clock.tick(T0 + FLOOR + 2) == "wake_live"
    assert len(clock.launches) == 1


def test_callback_gap_keeps_custody_until_the_finish_is_persisted(clock, monkeypatch):
    from ouroboros import task_results

    finished = _launched(clock)
    _register("wake0001", initiator="consciousness")
    get_direct_activity_registry().unregister("wake0001")
    entered, release = threading.Event(), threading.Event()
    def read(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return {"status": "completed"}
    monkeypatch.setattr(task_results, "load_task_result", read)
    monkeypatch.setattr(clock_module.time, "time", lambda: T0 + 5000)
    callback = threading.Thread(target=finished, args=("wake0001", True))
    callback.start()
    try:
        assert entered.wait(5)
        assert clock.clock.tick(T0 + 5000) == "wake_live"
        assert len(clock.launches) == 1
    finally:
        release.set()
        callback.join(5)
    assert not callback.is_alive()
    assert clock.clock.live_wake() == ""
    assert clock.store[APPOINTMENT_STATE_KEY] == T0 + 5000 + DEFAULT
    assert clock.clock.tick(T0 + 5000 + DEFAULT - 1) == "not_due"
    clock.receipt["task_id"] = "wake0002"
    assert clock.clock.tick(T0 + 5000 + DEFAULT) == "launched"


@pytest.mark.parametrize("raises", [False, True])
def test_admission_failure_after_binding_releases_direct_custody(clock, monkeypatch, raises):
    from supervisor import workers

    normal = workers.handle_wake_direct
    def refuse(chat_id, text, metadata, *, bind_input, **kwargs):
        bind_input({"id": "refused", "metadata": metadata})
        assert clock.clock.live_wake() == "refused"
        if raises:
            raise RuntimeError("thread start failed")
        return {"admitted": False, "reason": "admission_failed"}
    monkeypatch.setattr(workers, "handle_wake_direct", refuse)
    if raises:
        with pytest.raises(RuntimeError, match="thread start failed"):
            clock.clock.tick(T0 + FLOOR)
    else:
        assert clock.clock.tick(T0 + FLOOR) == "rejected:admission_failed"
    assert clock.clock.live_wake() == ""
    monkeypatch.setattr(workers, "handle_wake_direct", normal)
    assert clock.clock.tick(clock.clock.next_wake_at) == "launched"


def test_another_callback_cannot_release_this_wakes_custody(clock):
    finished = _launched(clock)
    finished("another-id", True)
    assert clock.clock.live_wake() == "wake0001"
    finished("wake0001", True)
    assert clock.clock.live_wake() == ""


@pytest.mark.parametrize("stop_at", ["before_binding", "after_binding", "none"])
def test_stop_during_admission_reaches_the_wake_and_leaves_owner_running(clock, monkeypatch, stop_at):
    import server
    from ouroboros import owner_mailbox
    from ouroboros.task_results import write_task_result
    from supervisor import queue, workers, consciousness_carriers, task_lifecycle
    from tests.test_delivery_forced_finalization import _forced_test_context
    from tests.test_transport_commands import Bridge, Ctx

    monkeypatch.setattr(queue, "DRIVE_ROOT", clock.root)
    monkeypatch.setattr(workers, "DRIVE_ROOT", clock.root)
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=lambda event: None))
    monkeypatch.setattr("ouroboros.config.get_direct_turn_stop_wait_sec", lambda: 0)
    monkeypatch.setattr("supervisor.message_bus.log_chat", lambda *a, **k: None)
    ctx = Ctx(clock.store)
    ctx.consciousness = clock.clock
    owner = _register("owner").actor
    bound, deliver, stopped, intent_written = (threading.Event() for _ in range(4))
    actor = SimpleNamespace(_busy=False, _owner_message_admission_lock=threading.Lock())
    registry = get_direct_activity_registry()
    actual_stop = consciousness_carriers.stop_pooled_wake
    def record_stop(task_id):
        actual_stop(task_id)
        intent_written.set()
    monkeypatch.setattr(consciousness_carriers, "stop_pooled_wake", record_stop)
    def lane(chat_id, text, metadata, *, bind_input, **kwargs):
        registry.register("parent1", chat_id, actor=actor)
        task = {"id": "parent1", "metadata": metadata}
        if stop_at != "before_binding":
            bind_input(task)
        bound.set()
        assert deliver.wait(5)
        if stop_at == "before_binding":
            bind_input(task)
        actor._busy = actor._accepting_owner_messages = True
        actor._current_task_id, actor._current_chat_id = "parent1", chat_id
        actor._current_task_metadata, actor._current_task_text = metadata, text
        write_task_result(clock.root, "parent1", "running", chat_id=chat_id, metadata=metadata, _is_direct_chat=True)
        return {"admitted": True, "task_id": "parent1"}
    monkeypatch.setattr(workers, "handle_wake_direct", lane)
    launched = []
    launch = threading.Thread(target=lambda: launched.append(clock.clock.tick(T0 + FLOOR)))
    def bg_stop():
        server._process_bridge_updates(Bridge([{
            "chat": {"id": 7}, "from": {"id": 7}, "source": "web", "text": "/bg stop",
        }]), 0, ctx)
        stopped.set()
    stop = threading.Thread(target=bg_stop)
    launch.start()
    try:
        assert bound.wait(5)
        assert workers.direct_chat_turn("parent1") is None
        if stop_at != "none":
            stop.start()
            assert stopped.wait(5), "Stop must not wait behind admission's clock lock"
            if stop_at == "after_binding":
                assert intent_written.wait(5)
    finally:
        deliver.set()
        launch.join(5)
        if stop.ident is not None:
            stop.join(5)
    assert launched == ["launched"]
    if stop_at != "none":
        assert intent_written.wait(5)
        assert clock.store["bg_consciousness_enabled"] is False
    task_lifecycle.sweep_cancel_intents()
    controls = owner_mailbox.drain_owner_entries(clock.root, "parent1")
    assert bool(controls) is (stop_at != "none")
    if controls:
        control, = controls
        assert control["kind"] == "finalize_now"
        loop, tool_registry, context, trace = _forced_test_context(clock.root)
        loop._replace_delivery_candidate(tool_registry, context, trace, "Retained wake note.", control="candidate")
        monkeypatch.setattr(loop, "call_llm_with_retry", lambda *a, **k: pytest.fail("stopped wake must make no call"))
        _text, usage, _trace = loop._handle_forced_finalization(context, control["text"])
        assert usage["execution_status"] == "failed"
        assert usage["reason_code"] == "owner_requested_finalization" and "Retained wake note." in _text
        actor._busy = False
        registry.unregister("parent1")
        clock.clock._wake_finished("parent1", True)
        assert clock.clock.live_wake() == ""
    else:
        assert clock.clock.live_wake() == "parent1" and actor._busy
    assert owner._busy and registry.get("owner").actor is owner
    assert owner_mailbox.drain_owner_entries(clock.root, "owner") == []


@pytest.mark.parametrize("witness", ["identity", "metadata", "none"])
def test_restarted_clock_requires_a_carrier_witness(clock, witness):
    _launched(clock)
    if witness == "identity":
        get_direct_activity_registry().register("wake0001", 7)
    elif witness == "metadata":
        clock.store.pop(clock_module.OWN_WAKE_STATE_KEY)
        _register("wake0001", initiator="consciousness")
    restarted = BackgroundConsciousness(clock.root, clock.root / "repo", lambda: 7, now=T0 + FLOOR)
    assert restarted.live_wake() == ("" if witness == "none" else "wake0001")
    assert restarted.tick(restarted.next_wake_at) == ("launched" if witness == "none" else "wake_live")


@pytest.mark.parametrize("outcome", ["allowance_unknown", "allowance_exhausted", "first_conversation", "refused"])
def test_skips_refusals_and_restart_keep_the_appointment(clock, monkeypatch, outcome):
    finished = _launched(clock)
    monkeypatch.setattr(clock_module.time, "time", lambda: T0 + 5000)
    finished("wake0001", True)
    appointment = clock.store[APPOINTMENT_STATE_KEY]
    if outcome.startswith("allowance"):
        monkeypatch.setattr(clock_module, "allowance_window", lambda *a, **k: {"status": outcome})
    elif outcome == "first_conversation":
        clock.store["owner_chat_id"] = None
    else:
        clock.receipt.update(admitted=False, reason="repo_writer_gate_closed")
    assert clock.clock.tick(appointment).startswith(("skipped:", "rejected:"))
    assert clock.clock.next_wake_at > appointment
    assert clock.store[APPOINTMENT_STATE_KEY] == appointment
    restarted = BackgroundConsciousness(clock.root, clock.root / "repo", lambda: 7, now=appointment + 1)
    assert restarted.status_snapshot()["appointment_at"] == clock_module._iso(appointment)


@pytest.mark.parametrize("lateness,word", [(0, "on time"), (59.9, "on time"), (60, "due 1 min ago"),
                                         (5400, "due 1 h 30 min ago"), (None, "")])
def test_launch_reason_reports_appointment_punctuality_only(clock, lateness, word):
    now = T0 + FLOOR
    clock.clock._appointment_at = now - lateness if lateness is not None else 0
    assert clock.clock.tick(now) == "launched"
    launch = clock.launches[-1]
    assert f"[Wake-up · heartbeat{' (' + word + ')' if word else ''}]" in launch["text"]
    assert launch["metadata"]["wake_reason"] == "heartbeat"
    assert "- wake cause: the scheduled wake time; events never move it" in launch["text"]


def test_clock_never_holds_writer_gate_while_waiting_for_queue(clock, monkeypatch):
    from supervisor import queue, workers

    rendering, held, entering_lane = threading.Event(), threading.Event(), threading.Event()
    acquired = []
    render = clock_module.render_wake_message
    def render_while_queue_held(*args, **kwargs):
        rendering.set()
        assert held.wait(5)
        return render(*args, **kwargs)
    def hold_queue():
        assert rendering.wait(5)
        with queue._queue_lock:
            held.set()
            assert entering_lane.wait(5)
            got_gate = workers._repo_writer_gate_lock.acquire(timeout=2)
            acquired.append(got_gate)
            if got_gate:
                workers._repo_writer_gate_lock.release()
    def lane(*args, **kwargs):
        entering_lane.set()
        with queue._queue_lock, workers._repo_writer_gate_lock:
            return {"admitted": False, "reason": "repo_writer_gate_closed"}
    monkeypatch.setattr(clock_module, "render_wake_message", render_while_queue_held)
    monkeypatch.setattr(workers, "handle_wake_direct", lane)
    holder = threading.Thread(target=hold_queue)
    holder.start()
    try:
        assert clock.clock.tick(T0 + FLOOR) == "rejected:repo_writer_gate_closed"
    finally:
        holder.join(5)
    assert not holder.is_alive() and acquired == [True]
