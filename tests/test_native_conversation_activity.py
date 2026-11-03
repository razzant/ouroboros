"""Ordinary conversations are admitted to the lane, carried by the pool and stay under its custody."""
from __future__ import annotations

import queue
import threading
from pathlib import Path
from types import SimpleNamespace

from supervisor import workers
from supervisor.active_activity import get_direct_activity_registry
from tests.conversation_carrier_support import carry, isolate_lane_pool, lane_rows


def _lane(monkeypatch, tmp_path, *, spares: int = 1):
    """The real lane: a fresh pool table with ``spares`` ready carriers and one event bus; the server
    constructs no agent (``isolate_lane_pool``), so every body a test wants to run it carries itself."""
    from supervisor import message_bus, state

    bus = queue.Queue()
    lane = SimpleNamespace(bus=bus, spares=isolate_lane_pool(monkeypatch, tmp_path, spares=spares, event_q=bus))
    monkeypatch.setattr(workers, "REPO_DIR", tmp_path / "repo")
    monkeypatch.setattr(workers, "send_with_budget", lambda *a, **kw: None)
    monkeypatch.setattr(state, "load_state", lambda: {})
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(send_chat_action=lambda *a, **kw: None))
    workers.open_repo_writer_admission()
    return lane


def _next_native_phase(phases, expected):
    phase = phases.get()
    if isinstance(phase, BaseException):
        raise phase
    assert phase == expected, (expected, phase)


def test_two_native_turns_are_independently_addressable_and_drainable(monkeypatch, tmp_path):
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.server_routing_context import _addressable_root_tasks
    from supervisor.steering import _handle_steer_task
    from supervisor.worker_chat_lane import conversation_turn_live

    lane = _lane(monkeypatch, tmp_path, spares=2)
    for chat_id in (1, 2):
        workers.handle_chat_direct(chat_id, f"work {chat_id}")
    rows = lane_rows(workers)
    assert len(rows) == 2 and all(not meta["task"].get("_ephemeral_turn") for meta in rows.values())
    assert {meta["worker_id"] for meta in rows.values()} == {spare.wid for spare in lane.spares}
    ids = set(rows)
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=[], bridge=None)
    assert {row["task_id"] for row in _addressable_root_tasks(ctx)} == ids
    first, second = sorted(rows.values(), key=lambda meta: meta["task"]["chat_id"])
    first_id, second_id = first["task"]["id"], second["task"]["id"]
    from tests.test_steer_relay import held_quote
    _handle_steer_task({
        "quotes": held_quote(tmp_path, "Use the blue variant", 2),
        "target_task_id": second_id, "chat_id": 2,
        "message": "Use the blue variant", "client_message_id": "owner-blue",
    }, ctx)
    assert [e["text"] for e in drain_owner_entries(tmp_path, second_id)] == ["Use the blue variant"]
    assert drain_owner_entries(tmp_path, first_id) == []
    # Stop and the update's quiesce reach a carried turn through the pool's custody of its row, never
    # through an actor of this process: none is registered, so the in-process writer drain has nothing
    # to wait for, while the conversation is live for as long as its rows are.
    assert workers.arm_direct_chat_turn(second_id, lambda turn: "control-second") is None
    assert get_direct_activity_registry().actors() == [] and workers.drain_repo_writers(timeout=0) == []
    assert conversation_turn_live()
    workers.close_repo_writer_admission("test-update")
    try:
        workers.handle_chat_direct(3, "must wait for update")
        assert set(lane_rows(workers)) == ids, "the closed gate admits no third turn"
    finally:
        workers.open_repo_writer_admission()
    # Each carrier runs exactly its own turn.
    actors = []

    class Actor:
        def __init__(self):
            self.tasks = []
            actors.append(self)

        def handle_task(self, task):
            self.tasks.append(task)
            return []

    for spare in lane.spares:
        carry(workers, spare, Actor())
    assert sorted(task["id"] for actor in actors for task in actor.tasks) == sorted(ids)
    assert all(len(actor.tasks) == 1 for actor in actors)


def test_the_admission_door_is_a_registered_execution_and_the_row_is_the_turns_custody(monkeypatch, tmp_path):
    """The server's part of a turn is its admission: a registered execution while the door decides (what
    the liveness watchdog reads), then a RUNNING row the pool holds; no writer of this process stays behind."""
    from supervisor import conversation_lane
    from supervisor.worker_chat_lane import conversation_turn_live

    lane = _lane(monkeypatch, tmp_path)
    registry = get_direct_activity_registry()
    deciding, release = threading.Event(), threading.Event()
    real_admit = conversation_lane.admit_turn

    def admit(task, **kwargs):
        deciding.set()
        assert release.wait(10)
        return real_admit(task, **kwargs)

    monkeypatch.setattr(conversation_lane, "admit_turn", admit)
    thread = threading.Thread(target=workers.handle_chat_direct, args=(1, "work"))
    try:
        thread.start()
        assert deciding.wait(10)
        [door] = registry.executions()
        assert door.chat_id == 1 and door.task_id == "" and door.phase == "accepted"
        assert not lane_rows(workers) and registry.actors() == []
    finally:
        release.set()
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert registry.executions() == [] and registry.actors() == []
    [row] = lane_rows(workers).values()
    assert row["worker_id"] == lane.spares[0].wid and conversation_turn_live()
    assert workers.drain_repo_writers(timeout=0) == []  # the row is the pool's to stop, not a writer of this process


def test_restart_census_keeps_native_execution_after_owner_boundary():
    from ouroboros.server_restart import _live_running_task_ids

    registry = get_direct_activity_registry()
    # A settled answer can still be doing post-task work in this actor.
    registry.register("post-task", 1, actor=SimpleNamespace(_busy=False))
    assert _live_running_task_ids(SimpleNamespace(RUNNING={})) == ["post-task"]
    registry.unregister("post-task")
    assert _live_running_task_ids(SimpleNamespace(RUNNING={})) == []


def _mind():
    """An alarm clock that reads liveness off the census alone (no state, no lane)."""
    from ouroboros.consciousness import BackgroundConsciousness

    mind = object.__new__(BackgroundConsciousness)
    mind._last_wake_task_id = ""
    mind._own_wake, mind._in_flight = {}, ""
    return mind


def test_consciousness_ignores_owner_turns_through_native_completion():
    mind = _mind()
    registry = get_direct_activity_registry()
    registry.register("first", 1)
    registry.register("second", 2)
    assert mind.live_wake() == ""
    registry.unregister("first")
    assert mind.live_wake() == ""
    registry.unregister("second")
    assert mind.live_wake() == ""


def test_native_post_task_retains_activity_and_delivers_answer_early(monkeypatch, tmp_path):
    """Actual synthesis dispatch must stay owned after the ordinary final answer."""
    from ouroboros import agent_task_pipeline as pipeline, post_task_evolution
    from ouroboros.gateway.settings import _has_running_agent_tasks, _has_started_agent_tasks
    from ouroboros.post_task_checkpoint import post_task_synthesis_in_flight
    from ouroboros.server_restart import _live_running_task_ids
    from ouroboros.task_results import load_task_result
    from supervisor.worker_chat_lane import conversation_turn_live

    lane = _lane(monkeypatch, tmp_path)
    [spare] = lane.spares
    bus = lane.bus
    phases, release = queue.Queue(), threading.Event()
    synthesis_threads = []

    def consolidate(*args, **kwargs):
        synthesis_threads.append(threading.current_thread())
        phases.put("synthesis")
        release.wait()  # Only the observing test releases this phase, never a timer.

    monkeypatch.setattr(pipeline, "_run_memory_fallback_draft", consolidate)  # the first paid late stage
    monkeypatch.setattr(pipeline, "_record_task_facts", lambda *a, **kw: None)
    monkeypatch.setattr(post_task_evolution, "maybe_promote", lambda *a, **kw: None)
    env = SimpleNamespace(repo_dir=tmp_path / "repo", drive_root=tmp_path)
    mind = _mind()

    class Actor:
        def handle_task(self, task):
            self.task = task
            pending = [
                {"type": "send_message", "task_id": task["id"], "chat_id": task["chat_id"], "text": "done"},
                {"type": "task_done", "task_id": task["id"], "chat_id": task["chat_id"]},
            ]
            pipeline._dispatch_root_post_task(
                env, task, "done", self._event_queue, pending, {}, {}, tmp_path / "logs",
                budget_drive_root="", split_drive=False,
                project_scoped=bool(task.get("project_id")), project_task=False,
                parent_env=None, parent_task=None,
            )
            return pending

    # Both ordinary Main and ordinary Project conversation shapes must use
    # the same lifetime; neither has a headless workspace/task marker.
    for project_id in ("", "room"):
        release.clear()
        synthesis_threads.clear()
        actor = Actor()
        workers.handle_chat_direct(1, "ordinary conversation",
                                   task_metadata={"project_id": project_id, "origin_suppressed": True})
        [task_id] = [tid for tid, meta in lane_rows(workers).items() if meta["worker_id"] == spare.wid]

        def run():
            try:
                carry(workers, spare, actor)  # the carrier: the real worker body, on this thread
            except BaseException as error:
                phases.put(error)
            finally:
                phases.put("returned")

        thread = threading.Thread(target=run)
        try:
            thread.start()
            _next_native_phase(phases, "synthesis")
            assert actor.task["id"] == task_id
            # Synthesis belongs to the carrier that still holds the turn: its row is the custody.
            assert synthesis_threads == [thread]
            assert thread.is_alive()
            assert post_task_synthesis_in_flight(tmp_path, task_id)
            assert spare.busy_task_id == task_id and workers.RUNNING[task_id]["worker_id"] == spare.wid
            assert conversation_turn_live()
            assert _live_running_task_ids(SimpleNamespace(RUNNING=workers.RUNNING)) == [task_id]
            assert _has_running_agent_tasks() and _has_started_agent_tasks()
            assert mind.live_wake() == ""
            # Production early delivery ran before synthesis. The terminal
            # completion stays buffered until post-task work returns.
            early = bus.get_nowait()
            while early["type"] in {"direct_turn_bound", "direct_turn_phase"}:  # the carrier's first-step facts
                early = bus.get_nowait()
            assert early["type"] == "send_message" and early["text"] == "done"
            assert early["task_id"] == task_id and early["delivery_id"]
            assert bus.empty()
        finally:
            release.set()
            thread.join(timeout=10)
            for synthesis_thread in synthesis_threads:
                if synthesis_thread is not thread:
                    synthesis_thread.join(timeout=10)
        _next_native_phase(phases, "returned")
        thread.join()
        assert not thread.is_alive()
        assert get_direct_activity_registry().snapshot() == []
        assert not post_task_synthesis_in_flight(tmp_path, task_id)
        # The carrier released its slot; the row ends with the task_done the supervisor handles
        # (pooled custody, pinned by the queue tests) -- released by hand here.
        assert spare.busy_task_id is None
        workers.RUNNING.pop(task_id)
        assert _live_running_task_ids(SimpleNamespace(RUNNING=workers.RUNNING)) == []
        assert not _has_running_agent_tasks() and not _has_started_agent_tasks()
        assert mind.live_wake() == ""
        assert load_task_result(tmp_path, task_id)["root_phase_checkpoint"]["post_task_synthesis"] == "completed"
        # Retained final and early final use the existing delivery identity;
        # the supervisor deduplicates them. task_done reaches the bus last.
        final, done = bus.get_nowait(), bus.get_nowait()
        assert final["delivery_id"] == early["delivery_id"]
        assert done["type"] == "task_done" and done["task_id"] == task_id
        assert bus.empty()


def test_first_working_tool_call_names_a_main_turn_once(monkeypatch, tmp_path):
    """Owner decision Q7=A (16.09): a direct Main turn is named lazily, by the first
    non-addressing tool call, so a greeting costs no naming call and a working turn
    gets its title as its block becomes the task card; a Project-room turn is named
    by its room and spawns nothing. The carrier hangs the namer on its endpoint."""
    from ouroboros import project_naming

    lane = _lane(monkeypatch, tmp_path)
    [spare] = lane.spares
    calls = []
    monkeypatch.setattr(project_naming, "spawn_turn_namer", lambda *a, **kw: calls.append((a, kw)))

    class Actor:
        def handle_task(self, task):
            frames = self._event_queue
            frames.put_nowait({"type": "log_event", "data": {
                "type": "tool_call_started", "task_id": task["id"], "tool": "promote_chat_to_task",
                "routing_action": "promote_chat_to_task"}})
            assert calls == [], "an addressing call is a receipt, not work"
            for kind, tool in (("tool_call_started", "read_file"), ("tool_call_finished", "read_file"),
                               ("tool_call_started", "run_command")):
                frames.put_nowait({"type": "log_event", "data": {"type": kind, "task_id": task["id"], "tool": tool}})
            return []

    workers.handle_chat_direct(1, "проверь, почему карточка задачи потеряла элементы")
    carry(workers, spare, Actor())
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert Path(args[0]) == tmp_path and args[2] == "проверь, почему карточка задачи потеряла элементы"
    assert callable(kwargs["broadcast"])

    calls.clear()
    workers.handle_chat_direct(1, "и в комнате проекта", task_metadata={"project_id": "room"})
    carry(workers, spare, Actor())
    assert calls == [], "a Project-room turn is named by its room"
