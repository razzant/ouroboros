"""A wake commits its start with its RUNNING row; a pending Restart before that refuses it (#2489).

The real alarm, the lane's door, the queue and Restart acceptance run; the carrier is this test
(``tests.conversation_carrier_support.carry`` runs the worker body here with an inert actor), the
money window and the bridge are stubs. A Restart intent and its acceptance run on the supervisor
loop, as the tick does, so the reachable races are the loop's pass landing between the door's first
gate read and its locked re-read (the seed is minted between them), and after the row is written but
before or during the carrier's first step. The wake committed its start with its row, at the
boundary ``server_restart._started_restart_work`` reads, so it finishes as existing work; a Restart
before that row is a typed refusal that dispatches nothing, so no order of terminal and receipt can
move the clock or consume an observation. The carrier's end reaches the clock as the pooled terminal
the task_done handler delivers (``consciousness_carriers.finish_pooled_wake``) or, when that event is
lost, through the clock's own settlement at the tick that would start over it
(``_settle_unreported_terminal``).
"""

from __future__ import annotations

import queue as bus_module
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import consciousness as clock_module, consciousness_wake, dialogue_store
from ouroboros.consciousness import (
    APPOINTMENT_STATE_KEY,
    BackgroundConsciousness,
    INTERVAL_STATE_KEY,
    OBSERVATION_STATE_KEY,
    OWN_WAKE_STATE_KEY,
)
from ouroboros.task_results import list_task_results, load_task_result, write_task_result
from supervisor import conversation_lane, worker_chat_lane as lane
from supervisor.active_activity import get_direct_activity_registry
from supervisor.consciousness_carriers import finish_pooled_wake, queued_wakes
from tests.conversation_carrier_support import carry, install_lane_spare
from tests.test_agent_restart_intent import accept, runtime  # noqa: F401  # the shared real-queue fixture
from tests.test_consciousness import AVAILABLE, CEILING, DEFAULT, FLOOR


OWNER_LINE = "owner line observed during the window"
OWNER_COUNTED = "1 owner message (counted, not listed: my view holds the words)"  # the wake's text counts it; my view holds the words


def _observed(root, task):
    """A wake's observation as its bound source stores it, every event line whole."""
    from ouroboros.artifacts import read_actor_source_bytes

    source = task["metadata"]["wake_observation"]["source"]
    return read_actor_source_bytes(root, source["task_id"], source).decode("utf-8")


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
    bus = bus_module.Queue()
    monkeypatch.setattr(runtime.workers, "get_event_q", lambda: bus)
    notices, frames, ran, finished = [], [], [], threading.Event()
    monkeypatch.setattr(runtime.workers, "send_with_budget", lambda *a, **k: notices.append(a[1]))
    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(
        broadcast=frames.append, send_chat_action=lambda *a, **k: frames.append({"type": "activity"})))
    monkeypatch.setattr(runtime.workers, "_get_chat_agent", lambda: pytest.fail("the server constructed an agent"))
    spare = install_lane_spare(runtime.workers, 90)  # an idle lane carrier: an admitted wake is handed to it
    install_lane_spare(runtime.workers, 91)  # the owner's reserved spare: a wake never takes the last idle carrier (4A)
    get_direct_activity_registry().clear()
    start = time.time()
    clock = BackgroundConsciousness(root, root / "repo", lambda: 7, now=start - FLOOR - 1)
    settle = clock._wake_finished

    def finish(task_id, ok, **kwargs):  # every settlement of the clock passes here
        settle(task_id, ok, **kwargs)
        finished.set()

    clock._wake_finished = finish
    env = SimpleNamespace(drive_root=root, budget_drive_root=root, drive_path=lambda *parts: root.joinpath(*parts))

    def handle_task(task):  # the inert actor: the terminal row the real agent would write, no model call
        ran.append(task)
        write_task_result(root, task["id"], "completed", result="Whole note.", chat_id=task["chat_id"],
                          _is_direct_chat=True, metadata=task["metadata"], execution_owner=task["_execution_owner"])
        return []

    actor = SimpleNamespace(handle_task=handle_task, env=env)
    dialogue_store.append(root, "chat", {"ts": clock_module._iso(start - 1), "direction": "in", "chat_id": 1,
                                         "source": "web", "text": OWNER_LINE})
    yield SimpleNamespace(runtime=runtime, root=root, state=state, clock=clock, start=start, actor=actor,
                          spare=spare, bus=bus, notices=notices, frames=frames, ran=ran, finished=finished)
    get_direct_activity_registry().clear()


def _carry(race):
    """The carrier: the dispatched wake runs through the real worker body with the inert actor, here."""
    return carry(race.runtime.workers, race.spare, race.actor, out_q=race.bus)


def _settle(race, task_id):
    """The task_done handler's delivery (``events_task_done._finish_task_done_dispatch``): the own-clock
    settlement from the durable terminal, then the RUNNING row's release, in that order."""
    workers = race.runtime.workers
    meta = workers.RUNNING.get(task_id) or {}
    row = load_task_result(race.root, task_id) or {}
    finish_pooled_wake(SimpleNamespace(consciousness=race.clock), task_id, row, dict(meta.get("task") or {}))
    workers.RUNNING.pop(task_id, None)


def _run_wake(race):
    """One launched wake from its carrier to its pooled terminal; the task that ran."""
    _carry(race)
    task = race.ran[-1]
    _settle(race, task["id"])
    return task


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
        "dispatched": list(race.spare.in_q),
    }


def _refused(reason="restart_pending", backoff=1):
    return {"tick": f"rejected:{reason}", "model_calls": 0, "notices": [], "frames": [],
            "backoff": backoff, "outcome": f"rejected:{reason}", "boundary": None, "own_wake": None,
            "last_wake_task_id": "", "live": "", "queued": [], "rows": [], "census": [], "dispatched": []}


def _accept_before_the_row(race, monkeypatch, then=lambda: None):
    """After the door's first gate read, before its locked re-read (the seed is minted between them);
    ``then`` runs after acceptance."""
    seed, accepted = lane._seed_task, []

    def accept_then_seed(*args, **kwargs):
        if not accepted:
            accepted.append(accept(race.runtime))
            then()
        return seed(*args, **kwargs)

    monkeypatch.setattr(lane, "_seed_task", accept_then_seed)


def _accept_while_preparing(race, monkeypatch, then=lambda: None):
    """After the row is written and dispatched, in the carrier's first step."""
    prepare, accepted = lane.prepare_conversation_task, []

    def accept_then_prepare(*args, **kwargs):
        if not accepted:
            accepted.append(accept(race.runtime))
            then()
        return prepare(*args, **kwargs)

    monkeypatch.setattr(lane, "prepare_conversation_task", accept_then_prepare)


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.02)
    return predicate()


def _events(root):
    import json

    path = root / "logs" / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def test_a_restart_before_the_row_is_a_refusal_and_leaves_the_window_to_the_next_wake(race, monkeypatch):
    """Nothing it did reads as an owner turn, a failed model run or a consumed observation."""
    _accept_before_the_row(race, monkeypatch)
    tick = race.clock.tick(race.start)
    assert _refusal_facts(race, tick=tick) == _refused()
    assert race.spare.busy_task_id is None  # no carrier holds anything, so nothing can race the receipt
    assert race.clock.next_wake_at == race.start + DEFAULT  # the ordinary interval, never doubled
    assert APPOINTMENT_STATE_KEY not in race.state.load_state()  # a refusal moves only the next attempt
    events = [row["type"] for row in _events(race.root)]
    assert "consciousness_wake_started" not in events and "consciousness_wake_rejected" in events

    # Once this Restart is released, one wake observes the window the refused one did not consume.
    race.runtime.restart.release_agent_restart(race.runtime.restart._AGENT_RESTART["request_id"], "test_release")
    assert race.clock.tick(race.clock.next_wake_at) == "launched"
    task = _run_wake(race)
    assert race.finished.is_set() and race.ran == [task]
    assert OWNER_COUNTED in task["text"] and OWNER_LINE in _observed(race.root, task) and race.notices == []
    assert race.state.load_state()[OBSERVATION_STATE_KEY]["task_id"] == task["id"]
    assert race.state.load_state()[OWN_WAKE_STATE_KEY]["finished"] is True
    assert queued_wakes() == [] and _wake_rows(race) == [task["id"]]  # one wake, and only the one that ran


@pytest.mark.parametrize("window,door_named", [("carrier_gap", False), ("carrier_gap", True),
                                               ("binding", False), ("preparation", False)])
def test_a_restart_accepted_after_the_row_finds_a_started_wake(race, monkeypatch, window, door_named):
    """``carrier_gap`` is the production order: tick returned with the row written and dispatched, the same
    loop accepts, then the carrier runs. The row shares its Q transaction with the gate, so no later
    window can bounce it into the owner's deferral; a return door named meanwhile closes new starts, and
    this one already started."""
    def accepted():
        accept(race.runtime)
        if door_named:
            race.runtime.queue.RETURNS_NAMED["door"] = "managed_update"

    if window == "preparation":
        _accept_while_preparing(race, monkeypatch)
    elif window == "binding":
        bind = consciousness_wake.bind_wake_observation

        def accept_then_bind(root, task, observation, render):
            accepted()
            return bind(root, task, observation, render)

        monkeypatch.setattr(consciousness_wake, "bind_wake_observation", accept_then_bind)
    assert race.clock.tick(race.start) == "launched"
    if window == "carrier_gap":
        accepted()
    before = time.time()
    task = _run_wake(race)
    after = time.time()
    assert race.finished.is_set() and race.ran == [task]  # it really ran, once, and never read as an owner's turn
    assert race.runtime.restart._AGENT_RESTART["started"][task["id"]] == task["id"]
    assert not race.runtime.restart.agent_restart_holds(task)
    assert race.notices == [] and queued_wakes() == []
    assert [row["id"] for row in race.runtime.queue.PENDING] == []
    assert race.clock._backoff == 1 and race.clock.status_snapshot()["last_wake_outcome"] != "failed"
    stored = race.state.load_state()
    assert stored[OBSERVATION_STATE_KEY]["task_id"] == task["id"] and stored[OWN_WAKE_STATE_KEY]["finished"] is True
    assert before + DEFAULT <= race.clock.next_wake_at <= after + DEFAULT


@pytest.mark.parametrize("delivery", ["task_done_reported", "terminal_unreported"])
def test_a_wake_on_the_degraded_executor_settles_the_clock_like_a_carried_one(race, monkeypatch, delivery):
    """Review M2: a wake no carrier could take runs on the disclosed degraded executor, and its terminal settles
    the clock exactly as a carrier's does (the task_done delivery, or the clock's own read at the tick that would
    start over it). The next tick then waits the interval instead of launching another paid wake in the server."""
    from ouroboros.task_carrier import EXECUTOR_INLINE_DEGRADED
    from tests.conversation_carrier_support import inline_synchronously

    workers = race.runtime.workers
    workers.WORKERS.pop(race.spare.wid)  # only the owner's reserved spare is idle: the wake cannot take it
    monkeypatch.setattr(workers, "_get_chat_agent", lambda: race.actor)  # the degraded executor's in-process actor
    inline_synchronously(monkeypatch)
    assert race.clock.tick(race.start) == "launched"
    task, = race.ran
    assert task["_execution_owner"]["executor"] == EXECUTOR_INLINE_DEGRADED  # it really ran degraded, disclosed
    assert [n for n in race.notices if "No conversation carrier could start" in n]
    if delivery == "task_done_reported":
        _settle(race, task["id"])
    else:
        workers.RUNNING.pop(task["id"])  # the row released, the clock never told
        assert race.clock.tick(race.start) == "not_due"  # settled at the tick that would have started over it
    assert race.finished.is_set() and race.clock.live_wake() == ""
    assert race.clock.tick(race.start + 1) == "not_due" and race.ran == [task]  # no back-to-back wake
    assert race.clock.next_wake_at > race.start + DEFAULT - 1
    assert race.state.load_state()[OWN_WAKE_STATE_KEY]["finished"] is True


@pytest.mark.parametrize(("delivery", "outcome"), [
    ("task_done_reported", "ok"),
    ("terminal_unreported", "runner_failure"),
])
def test_a_started_wake_settles_once_however_its_terminal_reaches_the_clock(race, monkeypatch, delivery, outcome):
    """The task_done handler delivers the terminal (``finish_pooled_wake``), or — the event lost, the row
    released — the clock settles it itself at the tick that would start over it. A real failure still
    backs off, once; the boundary belongs to the wake that ran, either way."""
    if outcome == "runner_failure":
        def handle_task(task):
            race.ran.append(task)
            # The real agent persists its running record (metadata verbatim) before any model call fails.
            write_task_result(race.root, task["id"], "running", chat_id=task["chat_id"], _is_direct_chat=True,
                              metadata=task["metadata"], execution_owner=task["_execution_owner"])
            raise RuntimeError("model unreachable")

        race.actor.handle_task = handle_task
    assert race.clock.tick(race.start) == "launched"
    if outcome == "runner_failure":
        with pytest.raises(RuntimeError, match="model unreachable"):
            _carry(race)
    else:
        _carry(race)
    task, = race.ran
    assert race.clock.live_wake() == task["id"]  # the row is the custody until its terminal is settled
    if delivery == "task_done_reported":
        _settle(race, task["id"])
    else:
        race.runtime.workers.RUNNING.pop(task["id"])  # the row released, the clock never told
        assert not race.finished.is_set()
        assert race.clock.tick(race.start) == "not_due"  # settled here; the next wake is now scheduled
    assert race.finished.is_set()
    backoff = 2 if outcome == "runner_failure" else 1
    assert race.clock._backoff == backoff and race.clock.live_wake() == ""
    stored = race.state.load_state()
    assert stored[OBSERVATION_STATE_KEY]["task_id"] == task["id"] and stored[OWN_WAKE_STATE_KEY]["finished"] is True
    assert race.clock.next_wake_at <= time.time() + DEFAULT * backoff
    assert race.clock.next_wake_at > race.start + DEFAULT * backoff - 1
    row = load_task_result(race.root, task["id"])
    if outcome == "runner_failure":
        assert row["status"] == "failed" and "model unreachable" in row["result"]  # the terminal is the row's
    else:
        assert row["status"] == "completed"
    assert race.notices == []  # an owner notice for a failed terminal is the task_done handler's, not the lane's


@pytest.mark.parametrize("restart", [False, True])
def test_a_real_admission_failure_stays_a_failure_even_beside_a_restart(race, monkeypatch, restart):
    """The lane could not write the row: the alarm backs off like a failed wake, and no Restart relabels that."""
    def fail(task, **kwargs):
        if restart:
            accept(race.runtime)
        raise RuntimeError("no carrier table")

    monkeypatch.setattr(conversation_lane, "admit_turn", fail)
    tick = race.clock.tick(race.start)
    assert _refusal_facts(race, tick=tick) == _refused("admission_failed", backoff=2)
    assert race.clock.next_wake_at == race.start + 2 * DEFAULT


def test_a_writer_gate_closing_before_the_row_keeps_its_own_quiet_refusal(race, monkeypatch):
    from supervisor import workers

    closed: list = []
    seed = lane._seed_task

    def close_then_seed(*args, **kwargs):
        closed.append("test-update")
        return seed(*args, **kwargs)

    monkeypatch.setattr(lane, "_seed_task", close_then_seed)
    monkeypatch.setattr(workers, "repo_writer_admission_closed", lambda: closed[0] if closed else "")
    assert _refusal_facts(race, tick=race.clock.tick(race.start)) == _refused("repo_writer_gate_closed")
    assert race.clock.next_wake_at == race.start + FLOOR  # the gate's own quiet retry


def test_an_admission_that_raises_leaves_no_row_owner_or_observation(race, monkeypatch):
    """Nothing ran, so nothing says it did; the next launch observes the same window."""
    with monkeypatch.context() as patched:
        patched.setattr(conversation_lane, "admit_turn",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no carrier table")))
        tick = race.clock.tick(race.start)
        facts = _refusal_facts(race, tick=tick)
    assert facts == _refused("admission_failed", backoff=2)
    assert race.clock.next_wake_at == race.start + 2 * DEFAULT
    assert race.clock.tick(race.clock.next_wake_at) == "launched"
    task = _run_wake(race)
    assert race.finished.is_set() and OWNER_COUNTED in task["text"] and OWNER_LINE in _observed(race.root, task)


@pytest.mark.parametrize("fault", ["binder_raises", "source_store_fails"])
def test_a_binding_failure_runs_the_wake_with_its_text_and_moves_no_boundary(race, monkeypatch, fault):
    """The carrier's binding fails: the wake still runs on the complete rendered text, and its terminal
    carries no accepted boundary, so the clock keeps the window for the next wake."""
    if fault == "binder_raises":
        monkeypatch.setattr(consciousness_wake, "bind_wake_observation",
                            lambda *a, **k: (_ for _ in ()).throw(OSError("source store unavailable")))
    else:
        monkeypatch.setattr("ouroboros.chat_chain.retain_memory_source",
                            lambda *a, **k: (_ for _ in ()).throw(OSError("source store unavailable")))
    before = time.time()
    assert race.clock.tick(race.start) == "launched"
    task = _run_wake(race)
    after = time.time()
    assert race.finished.is_set()
    assert OWNER_COUNTED in task["text"] and race.clock._backoff == 1  # no source: the text, counts and all
    stored = race.state.load_state()
    assert OBSERVATION_STATE_KEY not in stored  # replayed whole by the next wake, never skipped
    assert stored[OWN_WAKE_STATE_KEY]["task_id"] == task["id"] and stored[OWN_WAKE_STATE_KEY]["finished"] is True
    assert before + DEFAULT <= race.clock.next_wake_at <= after + DEFAULT


@pytest.mark.parametrize("window", ["before_the_row", "preparation_with_restart", "preparation"])
def test_a_stop_reaches_a_started_wake_and_a_refused_one_has_nothing_to_stop(race, monkeypatch, window):
    """``before_the_row``: the stop lands between the door's first gate read and the clock's binding, where
    nothing is live (``stop`` finds no wake); the lane then refuses the id the clock bound, and a refused
    admission has nothing to stop — the clock re-checks Stop only once an admitted row exists."""
    stopped: list = []
    messages: list = []
    monkeypatch.setattr(BackgroundConsciousness, "_stop_live_wake", staticmethod(stopped.append))
    seeded: list = []
    seed = lane._seed_task
    monkeypatch.setattr(lane, "_seed_task", lambda *a, **k: seeded.append(seed(*a, **k)) or seeded[-1])
    stop = lambda: messages.append(race.clock.stop())  # noqa: E731
    if window == "before_the_row":
        _accept_before_the_row(race, monkeypatch, then=stop)
    elif window == "preparation_with_restart":
        _accept_while_preparing(race, monkeypatch, then=stop)
    else:
        prepare = lane.prepare_conversation_task

        def stop_then_prepare(*args, **kwargs):
            stop()
            return prepare(*args, **kwargs)

        monkeypatch.setattr(lane, "prepare_conversation_task", stop_then_prepare)
    tick = race.clock.tick(race.start)
    if window == "before_the_row":
        assert _refusal_facts(race, tick=tick) == _refused()
        assert messages == ["Consciousness off: no own wakes."]  # nothing was live to stop
        refused_id = seeded[0]["id"]
        assert stopped == [] and refused_id not in _wake_rows(race)  # no row ever: nothing to stop
        return
    assert tick == "launched"
    live = race.clock.live_wake()  # the admitted row, before the carrier ran a step
    _carry(race)
    assert _wait_for(lambda: stopped) and stopped == [live] and live == race.ran[0]["id"]
    assert messages == ["Consciousness off: no own wakes. Stopping now: its wake-up."]  # the wake `live`, by its kind


def test_a_panic_barred_wake_never_admits_beside_a_pending_restart(race, monkeypatch):
    accept(race.runtime)
    monkeypatch.setattr(clock_module, "panic_blocks_wake", lambda _root: True)
    monkeypatch.setattr(conversation_lane, "admit_turn", lambda *a, **k: pytest.fail("no admission under Panic"))
    assert race.clock.tick(race.start) == "panic_stop"
    assert _wake_rows(race) == [] and race.clock._backoff == 1 and race.frames == []
    assert queued_wakes() == [] and list(race.spare.in_q) == []


def test_owner_main_in_the_same_window_starts_and_the_wake_is_not(race, monkeypatch):
    """The owner's turn is dialogue (BIBLE P5): accepted in the same window it starts and joins the request's
    started work, while a consciousness wake is still refused until the restart."""
    from supervisor import workers

    _accept_before_the_row(race, monkeypatch)
    workers.handle_chat_direct(7, "Main in the window", task_metadata={"client_message_id": "main-1"})
    assert not any(task.get("text") == "Main in the window" for task in race.runtime.queue.PENDING)
    [main] = [tid for tid, meta in workers.RUNNING.items()
              if isinstance(meta, dict) and (meta.get("task") or {}).get("text") == "Main in the window"]
    assert main not in queued_wakes() and race.runtime.restart._AGENT_RESTART["started"][main] == main
    assert "Saved for execution after Restart." not in race.notices
    assert race.clock.tick(race.start) == "rejected:restart_pending"
    assert race.clock._backoff == 1
    assert list(queued_wakes()) == [] and race.ran == []


def test_no_task_result_or_frame_exists_before_the_carrier_binds_and_the_title_is_its_first_frame(race, monkeypatch):
    """The start receipt is the admission's; the cause title and every frame of the wake are the
    carrier's, in their pinned order: nothing durable or published about the body exists before the
    carrier bound its observation, and the title precedes its first work."""
    bind = consciousness_wake.bind_wake_observation
    at_binding = {}

    def witness(root, task, observation, render):
        at_binding.update(row=load_task_result(root, task["id"]), bus=list(race.bus.queue), frames=list(race.frames))
        return bind(root, task, observation, render)

    def handle_task(task):
        at_handle = load_task_result(race.root, task["id"])
        assert at_handle["suggested_name"] == "Own time" and at_handle["status"] == "running"
        race.ran.append(task)
        write_task_result(race.root, task["id"], "completed", result="Whole note.", chat_id=task["chat_id"],
                          _is_direct_chat=True, metadata=task["metadata"], execution_owner=task["_execution_owner"])
        return []

    monkeypatch.setattr(consciousness_wake, "bind_wake_observation", witness)
    race.actor.handle_task = handle_task
    assert race.clock.tick(race.start) == "launched"
    assert race.frames == [{"type": "activity"}]  # the authoritative start receipt is the door's
    task = _run_wake(race)
    assert at_binding == {"row": None, "bus": [], "frames": [{"type": "activity"}]}
    frames = []
    while not race.bus.empty():
        frames.append(race.bus.get_nowait())
    assert frames[0]["type"] == "task_named" and frames[0]["task_id"] == task["id"]
    assert frames[0]["suggested_name"] == "Own time"


def test_the_owners_chosen_interval_survives_the_refusal(race, monkeypatch):
    race.state.update_state(lambda st: st.__setitem__(INTERVAL_STATE_KEY, 2000))
    _accept_before_the_row(race, monkeypatch)
    assert race.clock.tick(race.start) == "rejected:restart_pending"
    assert race.clock.next_wake_at == race.start + 2000
    assert race.state.load_state()[INTERVAL_STATE_KEY] == 2000 and race.clock._backoff == 1
