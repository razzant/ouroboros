"""A direct turn whose start phase stops advancing is journaled with its own thread's stack.

Each case runs the real ``handle_chat_direct`` against a stand-in actor that passes the
start facts the chat agent emits onto its turn queue (the ``context_building_*`` live logs,
the Main call's ``main_llm_call_state`` start) and blocks where the case says. The liveness
watchdog is the production one: it reads the turn's execution and writes journal rows only,
so every case also proves no owner message, no ``task_checkpoint`` row and no cancel."""
from __future__ import annotations

import queue
import sys
import threading
import time
from types import SimpleNamespace

from supervisor import workers
from supervisor.active_activity import get_direct_activity_registry


class _AheadClock:
    """The ``time`` the watchdog reads: the real monotonic clock pushed ``ahead`` seconds, and a
    sleep that only yields. Turn marks and heartbeats stay on the real clock, so a blocked phase
    has stood a whole interval at the next tick without the test waiting for it."""

    def __init__(self, ahead: float) -> None:
        self.ahead = ahead
        self.ticks = 0

    def __getattr__(self, name):
        return getattr(time, name)

    def monotonic(self) -> float:
        return time.monotonic() + self.ahead

    def sleep(self, _seconds: float) -> None:
        self.ticks += 1
        time.sleep(0.01)


def _wait_until(predicate, budget: float = 10.0) -> bool:
    end = time.monotonic() + budget
    while not predicate() and time.monotonic() < end:
        time.sleep(0.01)
    return bool(predicate())


def _on_stack(thread: threading.Thread, function: str) -> bool:
    frame = sys._current_frames().get(thread.ident)
    try:
        while frame is not None:
            if frame.f_code.co_name == function:
                return True
            frame = frame.f_back
        return False
    finally:
        frame = None


def _lane(monkeypatch, tmp_path):
    """The real direct lane; every way out of a turn (owner chat, event bus, journal) is recorded."""
    from supervisor import message_bus, state

    out = SimpleNamespace(bus=queue.Queue(), sent=[], rows=[])

    def send(chat_id, text, *a, **k):
        out.sent.append((chat_id, text))
        return True, ""

    bridge = SimpleNamespace(send_chat_action=lambda *a, **k: None, broadcast=lambda *a, **k: None,
                             send_message=send)
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "REPO_DIR", tmp_path / "repo")
    monkeypatch.setattr(workers, "get_event_q", lambda: out.bus)
    monkeypatch.setattr(workers, "send_with_budget", send)
    monkeypatch.setattr(message_bus, "send_with_budget", send)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(state, "load_state", lambda: {"owner_chat_id": 7})
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)
    monkeypatch.setattr(state, "append_jsonl", lambda path, row, *a, **k: out.rows.append(dict(row)) or True)
    monkeypatch.setenv("OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC", "90")  # a 15 s watchdog interval
    workers.open_repo_writer_admission()
    return out


def _start_watchdog(monkeypatch, clock: _AheadClock) -> threading.Event:
    from ouroboros import server_liveness

    monkeypatch.setattr(server_liveness, "time", clock)
    stop = threading.Event()
    # The loop's own stamp stays ahead of every clock here: only the turn half has anything to say.
    server_liveness._start_supervisor_liveness_watchdog([time.monotonic() + 10**6], stop)
    return stop


def _stop_watchdog(stop: threading.Event) -> None:
    stop.set()
    for thread in threading.enumerate():
        if thread.name == "supervisor-liveness-watchdog":
            thread.join(timeout=5)


class _Actor:
    """Stands in for the chat agent: ``behave`` passes its start facts and blocks where told."""

    def __init__(self, behave) -> None:
        self.behave = behave
        self.calls: list = []
        self._busy = False
        self._last_activity_ts = None

    def handle_task(self, task):
        self.calls.append(task["id"])
        return self.behave(self, task)


def _pass(actor: _Actor, task: dict, *facts: str) -> None:
    """Start facts, emitted the way the chat agent emits them onto its turn queue."""
    for fact in facts:
        if fact == "main_llm_call_state":
            actor._event_queue.put({"type": fact, "task_id": task["id"], "phase": "started"})
        else:
            actor._event_queue.put({"type": "log_event", "data": {"type": fact, "task_id": task["id"]}})


def _answer(task: dict) -> list:
    return [{"type": "send_message", "chat_id": task["chat_id"], "text": "the answer", "task_id": task["id"]}]


def _answers_at_once(actor: _Actor, task: dict) -> list:
    _pass(actor, task, "context_building_started", "context_building_finished", "main_llm_call_state")
    return _answer(task)


def _turn(monkeypatch, behave):
    from ouroboros import agent as agent_module

    actor = _Actor(behave)
    monkeypatch.setattr(agent_module, "make_agent", lambda **kwargs: actor)
    thread = threading.Thread(target=workers.handle_chat_direct, name="owner-turn-1", args=(1, "hello"),
                              kwargs={"task_metadata": {"client_message_id": "owner-msg-1"}}, daemon=True)
    return actor, thread


def _stall_rows(out) -> list:
    return [row for row in out.rows if str(row.get("type") or "").startswith("direct_turn_stall")]


def _observe_stall(monkeypatch, out, unblock, turn_thread: threading.Thread) -> list:
    """Watch a blocked turn: its onset row and more samples, then unblock it and read its end row."""
    clock = _AheadClock(ahead=20.0)
    stop = _start_watchdog(monkeypatch, clock)
    try:
        assert _wait_until(lambda: _stall_rows(out)), "a phase that stood a whole interval is journaled"
        seen = clock.ticks
        assert _wait_until(lambda: clock.ticks >= seen + 3)
        clock.ahead = 0.0  # the phases after the release pass in milliseconds
        unblock()
        turn_thread.join(timeout=10)
        assert not turn_thread.is_alive()
        assert _wait_until(lambda: any(row["type"] == "direct_turn_stall_end" for row in out.rows))
    finally:
        _stop_watchdog(stop)
    return _stall_rows(out)


def _journal_only(out, actor: _Actor) -> list:
    """No owner message, no checkpoint row, no cancel: the turn ran on to its one answer."""
    bus = []
    while not out.bus.empty():
        bus.append(out.bus.get_nowait())
    assert out.sent == [], out.sent
    assert not [row for row in out.rows if row.get("type") in ("chat_turn_wedge", "task_checkpoint")]
    assert not [item for item in bus if "task_checkpoint" in (item.get("type"), item.get("system_type"))]
    assert len(actor.calls) == 1
    answers = [item for item in bus if item.get("type") == "send_message" and item.get("text") == "the answer"]
    assert len(answers) == 1 and answers[0]["task_id"] == actor.calls[0]
    assert get_direct_activity_registry().executions() == []
    return bus


def test_a_turn_held_before_admission_journals_its_accepted_phase_and_own_stack(monkeypatch, tmp_path):
    """Blocked before its heartbeat and before admission decides it: waiting on the update gate."""
    out = _lane(monkeypatch, tmp_path)
    actor, thread = _turn(monkeypatch, _answers_at_once)
    gate, released = workers._repo_writer_gate_lock, []  # an RLock: its holder must release it

    def unblock() -> None:
        gate.release()
        released.append(True)

    gate.acquire()
    try:
        thread.start()
        assert _wait_until(lambda: _on_stack(thread, "repo_writer_admission_closed"))
        onset, end = _observe_stall(monkeypatch, out, unblock, thread)
    finally:
        if not released:
            gate.release()
    assert onset["type"] == "direct_turn_stall" and onset["phase"] == "accepted"
    assert onset["thread"] == "owner-turn-1" and onset["stood_sec"] >= 15
    assert onset["task_id"] == "" and onset["chat_id"] == 1 and onset["client_message_id"] == "owner-msg-1"
    assert onset["heartbeat_sec"] is None
    assert "workers.py:" in onset["stack"][-1] and onset["stack"][-1].endswith("in repo_writer_admission_closed")
    assert end["type"] == "direct_turn_stall_end" and (end["phase"], end["then"]) == ("accepted", "admitted")
    assert end["execution_id"] == onset["execution_id"] and end["task_id"] == actor.calls[0]
    assert end["samples"] >= 3 and end["top_frames"][0]["frame"].endswith("workers.py:repo_writer_admission_closed")
    assert end["last_stack"][-1].endswith("in repo_writer_admission_closed")
    _journal_only(out, actor)


def test_a_turn_blocked_before_its_heartbeat_journals_its_started_phase(monkeypatch, tmp_path):
    out = _lane(monkeypatch, tmp_path)
    blocked, release = threading.Event(), threading.Event()

    def stalls_before_its_heartbeat(actor, task):
        blocked.set()
        assert release.wait(10)
        return _answers_at_once(actor, task)

    actor, thread = _turn(monkeypatch, stalls_before_its_heartbeat)
    try:
        thread.start()
        assert blocked.wait(10)
        onset, end = _observe_stall(monkeypatch, out, release.set, thread)
    finally:
        release.set()
    assert onset["phase"] == "started" and onset["thread"] == "owner-turn-1"
    assert onset["task_id"] == actor.calls[0] and onset["heartbeat_sec"] is None
    assert any(frame.endswith("in stalls_before_its_heartbeat") for frame in onset["stack"])
    assert (end["phase"], end["then"]) == ("started", "context_building") and end["samples"] >= 3
    assert end["top_frames"][0]["frame"].endswith(":stalls_before_its_heartbeat")
    _journal_only(out, actor)


def test_a_turn_blocked_under_a_live_heartbeat_journals_its_phase_all_the_same(monkeypatch, tmp_path):
    out = _lane(monkeypatch, tmp_path)
    blocked, release = threading.Event(), threading.Event()
    census: list = []

    def stalls_while_its_heartbeat_beats(actor, task):
        beating = threading.Event()

        def beat() -> None:
            while not beating.is_set():
                actor._last_activity_ts = time.monotonic()
                time.sleep(0.01)

        actor._busy, actor._last_activity_ts = True, time.monotonic()
        threading.Thread(target=beat, daemon=True).start()
        try:
            _pass(actor, task, "context_building_started")
            census.extend(get_direct_activity_registry().snapshot())
            blocked.set()
            assert release.wait(10)
            _pass(actor, task, "context_building_finished", "main_llm_call_state")
            return _answer(task)
        finally:
            beating.set()
            actor._busy = False

    actor, thread = _turn(monkeypatch, stalls_while_its_heartbeat_beats)
    try:
        thread.start()
        assert blocked.wait(10)
        onset, end = _observe_stall(monkeypatch, out, release.set, thread)
    finally:
        release.set()
    assert [row["turn_phase"] for row in census] == ["context_building"]
    assert onset["phase"] == "context_building" and onset["thread"] == "owner-turn-1"
    # The heartbeat is live (fresh against the 90 s deadline) and decides nothing.
    assert onset["heartbeat_sec"] is not None and 0 <= onset["heartbeat_sec"] < 90
    assert any(frame.endswith("in stalls_while_its_heartbeat_beats") for frame in onset["stack"])
    assert (end["phase"], end["then"]) == ("context_building", "context_built") and end["samples"] >= 3
    _journal_only(out, actor)


def test_a_reused_thread_opens_a_new_episode_under_its_new_execution(monkeypatch):
    from ouroboros import server_liveness

    rows: list = []
    monkeypatch.setattr("supervisor.state.append_jsonl", lambda path, row, *a, **k: rows.append(dict(row)) or True)
    registry = get_direct_activity_registry()
    watch = server_liveness._TurnPhaseWatch(15)
    first = registry.begin_execution(1, client_message_id="m-1")
    watch.tick(first.marks[-1][1] + 20, {})
    registry.end_execution(first)
    second = registry.begin_execution(1, client_message_id="m-2")  # the same thread, so the same ident
    assert second.thread == first.thread and second.execution_id != first.execution_id
    watch.tick(second.marks[-1][1] + 20, {})
    assert [(row["type"], row["execution_id"]) for row in rows] == [
        ("direct_turn_stall", first.execution_id), ("direct_turn_stall_end", first.execution_id),
        ("direct_turn_stall", second.execution_id)]
    assert rows[1]["then"] == "ended" and rows[2]["client_message_id"] == "m-2"
    watch.tick(second.marks[-1][1] + 21, {})
    assert len(rows) == 3  # an open episode only folds samples until it closes


def test_a_disabled_watchdog_watches_no_turn(monkeypatch, tmp_path):
    out = _lane(monkeypatch, tmp_path)
    monkeypatch.setenv("OUROBOROS_SUPERVISOR_LIVENESS_DEADLINE_SEC", "0")
    blocked, release = threading.Event(), threading.Event()

    def stalls(actor, task):
        blocked.set()
        assert release.wait(10)
        return _answers_at_once(actor, task)

    actor, thread = _turn(monkeypatch, stalls)
    stop = None
    try:
        thread.start()
        assert blocked.wait(10)
        stop = _start_watchdog(monkeypatch, _AheadClock(ahead=20.0))
        assert not [t for t in threading.enumerate() if t.name == "supervisor-liveness-watchdog"]
        release.set()
        thread.join(timeout=10)
    finally:
        release.set()
        if stop is not None:
            _stop_watchdog(stop)
    assert _stall_rows(out) == []
    _journal_only(out, actor)


def test_an_ordinary_turn_reaches_its_model_and_answer_once_and_journals_nothing(monkeypatch, tmp_path):
    out = _lane(monkeypatch, tmp_path)
    clock = _AheadClock(ahead=0.0)
    census: list = []

    def answers(actor, task):
        _pass(actor, task, "context_building_started", "context_building_finished", "main_llm_call_state")
        census.extend(get_direct_activity_registry().snapshot())
        # The call's own time, before its dispatch is seen as well: however long it stands, it is no start stall.
        clock.ahead = 60.0
        seen = clock.ticks
        assert _wait_until(lambda: clock.ticks >= seen + 3)
        return _answer(task)

    actor, thread = _turn(monkeypatch, answers)
    stop = _start_watchdog(monkeypatch, clock)
    try:
        thread.start()
        thread.join(timeout=10)
        assert not thread.is_alive()
    finally:
        _stop_watchdog(stop)
    assert _stall_rows(out) == []
    assert [row["turn_phase"] for row in census] == ["requested"]
    bus = _journal_only(out, actor)
    assert len([item for item in bus if item.get("type") == "main_llm_call_state"]) == 1


def test_a_turn_ending_in_its_stalled_phase_stood_until_its_end_and_a_cut_stack_says_so(monkeypatch):
    from ouroboros import server_liveness

    rows: list = []
    monkeypatch.setattr("supervisor.state.append_jsonl", lambda path, row, *a, **k: rows.append(dict(row)) or True)
    registry = get_direct_activity_registry()
    watch = server_liveness._TurnPhaseWatch(15)
    turn = registry.begin_execution(1, client_message_id="m-1")  # on this test's thread, deeper than the bound
    at = turn.marks[-1][1]
    watch.tick(at + 20, {})  # the watch's clock, a whole interval past the phase's stamp
    registry.end_execution(turn)
    watch.tick(at + 400, {})  # its next look, long after the turn ended
    onset, end = rows
    assert onset["stack_truncated"] is True and "loop_stack_truncated" not in onset
    assert len(onset["stack"]) == server_liveness._STALL_STACK_FRAMES
    assert (end["then"], end["stood_sec"]) == ("ended", round(turn.ended_at - at, 1)) and end["stood_sec"] < 20
    # A whole stack carries no flag.
    began, release, shallow = threading.Event(), threading.Event(), []

    def stands() -> None:
        shallow.append(registry.begin_execution(2))
        began.set()
        release.wait(10)

    thread = threading.Thread(target=stands, daemon=True)
    thread.start()
    try:
        assert began.wait(10)
        watch.tick(shallow[0].marks[-1][1] + 20, {})
    finally:
        release.set()
        thread.join(timeout=10)
    assert rows[-1]["type"] == "direct_turn_stall" and "stack_truncated" not in rows[-1]
    registry.end_execution(shallow[0])


def test_each_watchdog_generation_closes_the_episodes_it_opened(monkeypatch, tmp_path):
    """The supervisor loop's next generation starts its own watchdog while the turn's thread runs on."""
    out = _lane(monkeypatch, tmp_path)
    blocked, release = threading.Event(), threading.Event()

    def stalls(actor, task):
        blocked.set()
        assert release.wait(10)
        return _answers_at_once(actor, task)

    actor, thread = _turn(monkeypatch, stalls)
    clock, stop = _AheadClock(ahead=20.0), None
    try:
        thread.start()
        assert blocked.wait(10)
        stop = _start_watchdog(monkeypatch, clock)
        assert _wait_until(lambda: _stall_rows(out))
        _stop_watchdog(stop)
        assert [(row["type"], row.get("then")) for row in _stall_rows(out)] == [
            ("direct_turn_stall", None), ("direct_turn_stall_end", "unwatched")]
        stop = _start_watchdog(monkeypatch, clock)
        assert _wait_until(lambda: len(_stall_rows(out)) == 3)
        clock.ahead = 0.0
        release.set()
        thread.join(timeout=10)
        assert _wait_until(lambda: len(_stall_rows(out)) == 4)
    finally:
        release.set()
        if stop is not None:
            _stop_watchdog(stop)
    rows = _stall_rows(out)
    assert [(row["type"], row["phase"], row.get("then")) for row in rows] == [
        ("direct_turn_stall", "started", None), ("direct_turn_stall_end", "started", "unwatched"),
        ("direct_turn_stall", "started", None), ("direct_turn_stall_end", "started", "context_building")]
    assert len({row["execution_id"] for row in rows}) == 1 and rows[2]["stood_sec"] >= rows[1]["stood_sec"]
    _journal_only(out, actor)
