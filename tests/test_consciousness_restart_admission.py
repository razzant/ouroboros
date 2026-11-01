"""A wake commits its start with its registration; a pending Restart before that refuses it (#2489).

The real alarm, direct lane, queue and Restart acceptance run; only the actor, the money
window and the bridge are inert. A Restart intent and its acceptance run on the supervisor
loop, as the tick does, so the reachable race is the loop's next pass landing before the
started wake's thread took the queue lock. The wake committed its start with its registration,
at the boundary ``server_restart._started_restart_work`` reads, so it finishes as existing
work; a Restart before that registration is a typed refusal that starts no thread, so no order
of callback and receipt can move the clock or consume an observation.
"""

from __future__ import annotations

import queue as bus
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import consciousness as clock_module, dialogue_store
from ouroboros.consciousness import (
    APPOINTMENT_STATE_KEY,
    BackgroundConsciousness,
    INTERVAL_STATE_KEY,
    OBSERVATION_STATE_KEY,
    OWN_WAKE_STATE_KEY,
)
from ouroboros.task_results import list_task_results, load_task_result
from supervisor import worker_chat_lane as lane
from supervisor.active_activity import get_direct_activity_registry
from supervisor.consciousness_carriers import queued_wakes
from tests.test_agent_restart_intent import accept, runtime  # noqa: F401  # the shared real-queue fixture
from tests.test_consciousness import AVAILABLE, CEILING, DEFAULT, FLOOR

pytestmark = pytest.mark.serial

OWNER_LINE = "owner line observed during the window"


@pytest.fixture
def race(runtime, monkeypatch):  # noqa: F811
    from supervisor import message_bus, state

    root = runtime.root
    state.save_state({"bg_consciousness_enabled": True, "owner_chat_id": 7})  # an initialized install
    monkeypatch.setenv("OUROBOROS_BG_WAKEUP_MIN", str(FLOOR))
    monkeypatch.setenv("OUROBOROS_BG_WAKEUP_MAX", str(CEILING))
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_AUTONOMY", "act")
    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", "50")
    monkeypatch.setattr(clock_module, "allowance_window", lambda *a, **k: dict(AVAILABLE))
    monkeypatch.setattr(runtime.workers, "get_event_q", lambda: bus.Queue())
    notices, frames, ran, finished = [], [], [], threading.Event()
    monkeypatch.setattr(runtime.workers, "send_with_budget", lambda *a, **k: notices.append(a[1]))
    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(
        broadcast=frames.append, send_chat_action=lambda *a, **k: frames.append({"type": "activity"})))

    def handle_task(task):
        ran.append(task)
        return []

    actor = SimpleNamespace(handle_task=handle_task)
    monkeypatch.setattr(runtime.workers, "_get_chat_agent", lambda: actor)
    get_direct_activity_registry().clear()
    start = time.time()
    clock = BackgroundConsciousness(root, root / "repo", lambda: 7, now=start - FLOOR - 1)
    settle = clock._wake_finished

    def finish(task_id, ok, **kwargs):  # the lane captures the bound method at launch
        settle(task_id, ok, **kwargs)
        finished.set()

    clock._wake_finished = finish
    dialogue_store.append(root, "chat", {"ts": clock_module._iso(start - 1), "direction": "in", "chat_id": 1,
                                         "source": "web", "text": OWNER_LINE})
    yield SimpleNamespace(runtime=runtime, root=root, state=state, clock=clock, start=start, actor=actor,
                          notices=notices, frames=frames, ran=ran, finished=finished)
    get_direct_activity_registry().clear()


def _wake_rows(race):
    return [row["task_id"] for row in list_task_results(race.root) if row["task_id"] != "requester"]


def _refusal_facts(race, *, tick):
    stored = race.state.load_state()
    return {
        "tick": tick, "model_calls": len(race.ran), "notices": race.notices, "frames": race.frames,
        "backoff": race.clock._backoff, "outcome": race.clock.status_snapshot()["last_wake_outcome"],
        "boundary": stored.get(OBSERVATION_STATE_KEY), "own_wake": stored.get(OWN_WAKE_STATE_KEY) or None,
        "last_wake_task_id": race.clock.status_snapshot()["last_wake_task_id"], "live": race.clock.live_wake(),
        "queued": queued_wakes(), "rows": _wake_rows(race), "census": get_direct_activity_registry().snapshot(),
    }


def _refused(reason="restart_pending", backoff=1):
    return {"tick": f"rejected:{reason}", "model_calls": 0, "notices": [], "frames": [],
            "backoff": backoff, "outcome": f"rejected:{reason}", "boundary": None, "own_wake": None,
            "last_wake_task_id": "", "live": "", "queued": [], "rows": [], "census": []}


def _accept_before_registration(race, monkeypatch, then=lambda: None):
    """After the lane's first read, before its registration's locked read; ``then`` runs after acceptance."""
    admit, accepted = lane._admit_chat_task, []

    def accept_then_admit(*args, **kwargs):
        if not accepted:
            accepted.append(accept(race.runtime))
            then()
        return admit(*args, **kwargs)

    monkeypatch.setattr(lane, "_admit_chat_task", accept_then_admit)


def _accept_while_preparing(race, monkeypatch, then=lambda: None):
    """After the turn's registration, in its preparation."""
    accepted = []

    def prepare():
        if not accepted:
            accepted.append(accept(race.runtime))
            then()
        return race.actor

    monkeypatch.setattr(race.runtime.workers, "_get_chat_agent", prepare)


def _join_wake_threads():
    for thread in threading.enumerate():
        if thread.name.startswith("wake-turn-"):
            thread.join(10)
            assert not thread.is_alive()


def test_a_restart_before_the_registration_is_a_refusal_and_leaves_the_window_to_the_next_wake(race, monkeypatch):
    """Nothing it did reads as an owner turn, a failed model run or a consumed observation."""
    executed = []
    execute = lane._execute_chat_task
    monkeypatch.setattr(lane, "_execute_chat_task", lambda admitted: executed.append(admitted) or execute(admitted))
    _accept_before_registration(race, monkeypatch)
    tick = race.clock.tick(race.start)
    _join_wake_threads()
    assert _refusal_facts(race, tick=tick) == _refused()
    assert executed == []  # no thread, so no callback can race the receipt
    assert race.clock.next_wake_at == race.start + DEFAULT  # the ordinary interval, never doubled
    assert APPOINTMENT_STATE_KEY not in race.state.load_state()  # a refusal moves only the next attempt
    events = [row["type"] for row in _events(race.root)]
    assert "consciousness_wake_started" not in events and "consciousness_wake_rejected" in events

    # Once this Restart is released, one wake observes the window the refused one did not consume.
    race.runtime.restart.release_agent_restart(race.runtime.restart._AGENT_RESTART["request_id"], "test_release")
    assert race.clock.tick(race.clock.next_wake_at) == "launched"
    assert race.finished.wait(10)
    task, = race.ran
    assert OWNER_LINE in task["text"] and race.notices == []
    assert race.state.load_state()[OBSERVATION_STATE_KEY]["task_id"] == task["id"]
    assert race.state.load_state()[OWN_WAKE_STATE_KEY]["finished"] is True
    assert queued_wakes() == [] and _wake_rows(race) == [task["id"]]  # one wake, and only the one that ran


def _events(root):
    import json

    path = root / "logs" / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


@pytest.mark.parametrize("window,door_named", [("thread_gap", False), ("thread_gap", True),
                                               ("binding", False), ("preparation", False)])
def test_a_restart_accepted_after_the_registration_finds_a_started_wake(race, monkeypatch, window, door_named):
    """``thread_gap`` is the production order: tick returned, the same loop accepts, then the thread runs.
    The commit shares its Q transaction with the gate, so no later window can bounce it into the owner's
    deferral; a return door named meanwhile closes new starts, and this one already started."""
    entered, release = threading.Event(), threading.Event()
    execute = lane._execute_chat_task

    def delayed(admitted):
        entered.set()
        assert release.wait(10)
        return execute(admitted)

    def accepted():
        accept(race.runtime)
        if door_named:
            race.runtime.queue.RETURNS_NAMED["door"] = "managed_update"

    monkeypatch.setattr(lane, "_execute_chat_task", delayed)
    if window == "preparation":
        _accept_while_preparing(race, monkeypatch)
    elif window == "binding":
        bind = clock_module.bind_wake_observation

        def accept_then_bind(root, task, observation, render):
            accepted()
            return bind(root, task, observation, render)

        monkeypatch.setattr(clock_module, "bind_wake_observation", accept_then_bind)
    assert race.clock.tick(race.start) == "launched"
    assert entered.wait(10)
    if window == "thread_gap":
        accepted()
    before = time.time()
    release.set()
    assert race.finished.wait(10)
    after = time.time()
    task, = race.ran  # it really ran, once, and nothing about it read as an owner's turn
    assert race.runtime.restart._AGENT_RESTART["started"][task["id"]] == task["id"]
    assert not race.runtime.restart.agent_restart_holds(task)
    assert race.notices == [] and queued_wakes() == []
    assert [row["id"] for row in race.runtime.queue.PENDING] == []
    assert race.clock._backoff == 1 and race.clock.status_snapshot()["last_wake_outcome"] != "failed"
    stored = race.state.load_state()
    assert stored[OBSERVATION_STATE_KEY]["task_id"] == task["id"] and stored[OWN_WAKE_STATE_KEY]["finished"] is True
    assert before + DEFAULT <= race.clock.next_wake_at <= after + DEFAULT


@pytest.mark.parametrize("outcome", ["ok", "runner_failure"])
@pytest.mark.parametrize("order", ["callback_first", "receipt_first"])
def test_a_started_wake_settles_once_whichever_of_callback_and_receipt_comes_first(race, monkeypatch, order, outcome):
    """A real failure still backs off, once; the boundary belongs to the wake that ran, in both orders."""
    from supervisor import workers

    receipt_read = threading.Event()
    if outcome == "runner_failure":
        def handle_task(task):
            race.ran.append(task)
            raise RuntimeError("model unreachable")

        race.actor.handle_task = handle_task
    execute = lane._execute_chat_task

    def gated(admitted):
        if order == "receipt_first":
            assert receipt_read.wait(10)
        return execute(admitted)

    monkeypatch.setattr(lane, "_execute_chat_task", gated)
    real_lane = workers.handle_wake_direct

    def read_receipt(*args, **kwargs):
        receipt = real_lane(*args, **kwargs)
        if order == "callback_first":
            assert race.finished.wait(10)
        return receipt

    monkeypatch.setattr(workers, "handle_wake_direct", read_receipt)
    assert race.clock.tick(race.start) == "launched"
    receipt_read.set()
    assert race.finished.wait(10)
    _join_wake_threads()
    task, = race.ran
    backoff = 2 if outcome == "runner_failure" else 1
    assert race.clock._backoff == backoff and race.clock.live_wake() == ""
    stored = race.state.load_state()
    assert stored[OBSERVATION_STATE_KEY]["task_id"] == task["id"] and stored[OWN_WAKE_STATE_KEY]["finished"] is True
    assert race.clock.next_wake_at <= time.time() + DEFAULT * backoff
    assert race.clock.next_wake_at > race.start + DEFAULT * backoff - 1
    assert ("model unreachable" in " ".join(race.notices)) is (outcome == "runner_failure")


@pytest.mark.parametrize("restart", [False, True])
def test_a_real_admission_failure_stays_a_failure_even_beside_a_restart(race, monkeypatch, restart):
    """The lane reported it to the chat: the alarm backs off, and no Restart relabels that."""
    def fail():
        if restart:
            accept(race.runtime)
        raise RuntimeError("no actor")

    monkeypatch.setattr(race.runtime.workers, "_get_chat_agent", fail)
    tick = race.clock.tick(race.start)
    facts = _refusal_facts(race, tick=tick)
    notices = facts.pop("notices")
    assert len(notices) == 1 and "no actor" in notices[0]
    expected = _refused("admission_failed", backoff=2)
    expected.pop("notices")
    assert facts == expected
    assert race.clock.next_wake_at == race.start + 2 * DEFAULT


def test_a_writer_gate_closing_before_the_registration_keeps_its_own_quiet_refusal(race, monkeypatch):
    from supervisor import workers

    admit, closed = lane._admit_chat_task, []

    def close_then_admit(*args, **kwargs):
        closed.append("test-update")
        return admit(*args, **kwargs)

    monkeypatch.setattr(lane, "_admit_chat_task", close_then_admit)
    monkeypatch.setattr(workers, "repo_writer_admission_closed", lambda: closed[0] if closed else "")
    assert _refusal_facts(race, tick=race.clock.tick(race.start)) == _refused("repo_writer_gate_closed")
    assert race.clock.next_wake_at == race.start + FLOOR  # the gate's own quiet retry


def test_a_thread_that_cannot_start_leaves_no_row_owner_or_observation(race, monkeypatch):
    """Nothing ran, so nothing says it did; the next launch observes the same window."""
    start_thread = threading.Thread.start

    def refuse(thread):
        if thread.name.startswith("wake-turn-"):
            raise RuntimeError("can't start new thread")
        start_thread(thread)

    with monkeypatch.context() as patched:
        patched.setattr(threading.Thread, "start", refuse)
        tick = race.clock.tick(race.start)
        facts = _refusal_facts(race, tick=tick)
    assert facts == _refused("admission_failed", backoff=2)
    assert race.clock.next_wake_at == race.start + 2 * DEFAULT
    assert race.clock.tick(race.clock.next_wake_at) == "launched"
    assert race.finished.wait(10)
    assert OWNER_LINE in race.ran[0]["text"]


@pytest.mark.parametrize("fault", ["binder_raises", "source_store_fails"])
def test_a_binding_failure_runs_the_wake_with_its_text_and_moves_no_boundary(race, monkeypatch, fault):
    if fault == "binder_raises":
        monkeypatch.setattr(clock_module, "bind_wake_observation",
                            lambda *a, **k: (_ for _ in ()).throw(OSError("source store unavailable")))
    else:
        monkeypatch.setattr("ouroboros.chat_chain.retain_memory_source",
                            lambda *a, **k: (_ for _ in ()).throw(OSError("source store unavailable")))
    before = time.time()
    assert race.clock.tick(race.start) == "launched"
    assert race.finished.wait(10)
    after = time.time()
    task, = race.ran
    assert OWNER_LINE in task["text"] and race.clock._backoff == 1
    stored = race.state.load_state()
    assert OBSERVATION_STATE_KEY not in stored  # replayed whole by the next wake, never skipped
    assert stored[OWN_WAKE_STATE_KEY]["task_id"] == task["id"] and stored[OWN_WAKE_STATE_KEY]["finished"] is True
    assert before + DEFAULT <= race.clock.next_wake_at <= after + DEFAULT


@pytest.mark.parametrize("window", ["before_registration", "preparation_with_restart", "preparation"])
def test_a_stop_reaches_a_started_wake_and_a_refused_one_has_nothing_to_stop(race, monkeypatch, window):
    stopped, entered, release = [], threading.Event(), threading.Event()
    monkeypatch.setattr(BackgroundConsciousness, "_stop_live_wake", staticmethod(stopped.append))
    if window == "before_registration":
        _accept_before_registration(race, monkeypatch, then=race.clock.stop)
    elif window == "preparation_with_restart":
        _accept_while_preparing(race, monkeypatch, then=race.clock.stop)
    else:
        def prepare():
            race.clock.stop()
            return race.actor

        monkeypatch.setattr(race.runtime.workers, "_get_chat_agent", prepare)
    race.actor.handle_task = lambda task: (entered.set(), release.wait(10), race.ran.append(task), [])[-1]
    tick = race.clock.tick(race.start)
    try:
        if window == "before_registration":
            assert _refusal_facts(race, tick=tick) == _refused()
            assert stopped == []
        else:
            assert tick == "launched" and entered.wait(10)
            assert stopped == [race.clock.live_wake()] and stopped != [""]
    finally:
        release.set()
        _join_wake_threads()


def test_a_panic_barred_wake_never_registers_beside_a_pending_restart(race, monkeypatch):
    accept(race.runtime)
    monkeypatch.setattr(clock_module, "panic_blocks_wake", lambda _root: True)
    monkeypatch.setattr(race.runtime.workers, "_get_chat_agent", lambda: pytest.fail("no actor under Panic"))
    assert race.clock.tick(race.start) == "panic_stop"
    assert _wake_rows(race) == [] and race.clock._backoff == 1 and race.frames == []


def test_owner_main_in_the_same_window_is_still_saved_and_the_wake_is_not(race, monkeypatch):
    """The ordinary Main deferral keeps its receipt, its words and its first durable row."""
    from supervisor import workers

    _accept_while_preparing(race, monkeypatch)
    workers.handle_chat_direct(7, "Main in the window", task_metadata={"client_message_id": "main-1"})
    saved = next(task for task in race.runtime.queue.PENDING if task.get("text") == "Main in the window")
    assert saved["id"] not in queued_wakes()
    row = load_task_result(race.root, saved["id"])
    assert row["status"] == "scheduled" and row["host_admission"]["status"] == "accepted"
    assert race.notices == ["Saved for execution after Restart."]
    assert race.clock.tick(race.start) == "rejected:restart_pending"
    assert race.clock._backoff == 1 and len(race.notices) == 1
    assert list(queued_wakes()) == [] and race.ran == []


def test_no_task_result_or_frame_exists_before_the_wakes_actual_admission(race, monkeypatch):
    """The cause title and the start receipt follow the thread's start, in their pinned order."""
    bind = clock_module.bind_wake_observation
    at_binding = {}

    def witness(root, task, observation, render):
        at_binding.update(row=load_task_result(root, task["id"]), frames=list(race.frames))
        return bind(root, task, observation, render)

    def handle_task(task):
        at_handle = load_task_result(race.root, task["id"])
        assert at_handle["suggested_name"] == "Own time" and at_handle["status"] == "running"
        race.ran.append(task)
        return []

    monkeypatch.setattr(clock_module, "bind_wake_observation", witness)
    race.actor.handle_task = handle_task
    assert race.clock.tick(race.start) == "launched"
    assert race.finished.wait(10)
    assert at_binding == {"row": None, "frames": []}
    task, = race.ran
    assert [frame["type"] for frame in race.frames] == ["task_named", "activity"]
    assert race.frames[0] == {"type": "task_named", "task_id": task["id"], "suggested_name": "Own time"}


def test_the_owners_chosen_interval_survives_the_refusal(race, monkeypatch):
    race.state.update_state(lambda st: st.__setitem__(INTERVAL_STATE_KEY, 2000))
    _accept_before_registration(race, monkeypatch)
    assert race.clock.tick(race.start) == "rejected:restart_pending"
    assert race.clock.next_wake_at == race.start + 2000
    assert race.state.load_state()[INTERVAL_STATE_KEY] == 2000 and race.clock._backoff == 1
