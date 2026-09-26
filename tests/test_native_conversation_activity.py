"""Ordinary conversations execute concurrently and remain under native custody."""
from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

from supervisor import workers
from supervisor.active_activity import get_direct_activity_registry


def _lane(monkeypatch, tmp_path):
    from supervisor import message_bus, state

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "REPO_DIR", tmp_path / "repo")
    monkeypatch.setattr(workers, "get_event_q", lambda: queue.Queue())
    monkeypatch.setattr(workers, "send_with_budget", lambda *a, **kw: None)
    monkeypatch.setattr(state, "load_state", lambda: {})
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(send_chat_action=lambda *a, **kw: None))
    workers.open_repo_writer_admission()


def test_two_native_turns_are_independently_addressable_and_drainable(monkeypatch, tmp_path):
    from ouroboros import agent as agent_module
    from ouroboros.owner_mailbox import drain_owner_messages
    from ouroboros.server_routing_context import _addressable_root_tasks
    from supervisor.steering import _handle_steer_task

    _lane(monkeypatch, tmp_path)
    entered = threading.Barrier(3)
    releases = [threading.Event(), threading.Event()]
    actors = []

    class Actor:
        def __init__(self, **kwargs):
            self.thread = threading.current_thread()
            self.index = len(actors)
            self._owner_message_admission_lock = threading.Lock()
            actors.append(self)

        def handle_task(self, task):
            self.task = task
            self._current_task_id = task["id"]
            self._current_chat_id = task["chat_id"]
            self._current_task_text = task["text"]
            self._current_task_metadata = task.get("metadata", {})
            self._busy = self._accepting_owner_messages = True
            entered.wait(timeout=10)
            assert releases[self.index].wait(10)
            with self._owner_message_admission_lock:
                self._busy = self._accepting_owner_messages = False
            return []

    monkeypatch.setattr(agent_module, "make_agent", Actor)
    threads = [threading.Thread(target=workers.handle_chat_direct, args=(chat_id, f"work {chat_id}")) for chat_id in (1, 2)]
    try:
        for thread in threads:
            thread.start()
        entered.wait(timeout=10)
        assert len(actors) == 2
        ids = {actor.task["id"] for actor in actors}
        assert all(not actor.task.get("_ephemeral_turn") for actor in actors)
        ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, PENDING=[], bridge=None)
        roots = _addressable_root_tasks(ctx)
        assert {row["task_id"] for row in roots} == ids
        first, second = actors
        armed = workers.arm_direct_chat_turn(second.task["id"], lambda turn: "control-second")
        assert armed["stop_control_msg_id"] == "control-second"
        assert "stop_control_msg_id" not in workers.direct_chat_turn(first.task["id"])
        _handle_steer_task({
            "target_task_id": second.task["id"], "chat_id": second.task["chat_id"],
            "message": "Use the blue variant", "client_message_id": "owner-blue",
        }, ctx)
        assert drain_owner_messages(tmp_path, second.task["id"]) == ["Use the blue variant"]
        assert drain_owner_messages(tmp_path, first.task["id"]) == []
        assert second._owner_message_generation == 1
        workers.close_repo_writer_admission("test-update")
        assert set(workers.drain_repo_writers(timeout=0.01)) == ids
        workers.handle_chat_direct(3, "must wait for update")
        assert len(actors) == 2
        releases[first.index].set()
        first.thread.join(timeout=10)
        # Thread scheduling need not match actor allocation.
        remaining = get_direct_activity_registry().snapshot()
        assert {row["activity_id"] for row in remaining} == {second.task["id"]}
        assert workers.drain_repo_writers(timeout=0.01) == [second.task["id"]]
    finally:
        for release in releases:
            release.set()
        for thread in threads:
            thread.join(timeout=10)
        workers.open_repo_writer_admission()
    assert workers.drain_repo_writers(timeout=0) == []
    assert all(not thread.is_alive() for thread in threads)


def test_writer_drain_covers_construction_and_event_delivery(monkeypatch, tmp_path):
    from ouroboros import agent as agent_module

    _lane(monkeypatch, tmp_path)
    constructing, finish_construction = threading.Event(), threading.Event()
    delivering, finish_delivery = threading.Event(), threading.Event()

    class Actor:
        def handle_task(self, task):
            return [{"type": "send_message", "text": "completed"}]

    def construct(**kwargs):
        constructing.set()
        assert finish_construction.wait(10)
        return Actor()

    class Queue:
        def put(self, event):
            delivering.set()
            assert finish_delivery.wait(10)

    monkeypatch.setattr(agent_module, "make_agent", construct)
    monkeypatch.setattr(workers, "get_event_q", Queue)
    thread = threading.Thread(target=workers.handle_chat_direct, args=(1, "work"))
    try:
        thread.start()
        assert constructing.wait(10)
        workers.close_repo_writer_admission("test-update")
        pending = workers.drain_repo_writers(timeout=0.01)
        assert len(pending) == 1
        assert get_direct_activity_registry().get(pending[0]).actor is None
        finish_construction.set()
        assert delivering.wait(10)
        assert workers.drain_repo_writers(timeout=0.01) == pending
    finally:
        finish_construction.set()
        finish_delivery.set()
        thread.join(timeout=10)
        workers.open_repo_writer_admission()
    assert not thread.is_alive()
    assert workers.drain_repo_writers(timeout=0) == []


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
    return mind


def test_consciousness_sees_an_owner_turn_live_until_all_native_work_returns():
    mind = _mind()
    registry = get_direct_activity_registry()
    registry.register("first", 1)
    registry.register("second", 2)
    assert mind.live_turns() == ("", True)
    registry.unregister("first")
    assert mind.live_turns() == ("", True)
    registry.unregister("second")
    assert mind.live_turns() == ("", False)


def test_native_post_task_retains_activity_and_delivers_answer_early(monkeypatch, tmp_path):
    """Actual synthesis dispatch must stay owned after the ordinary final answer."""
    from ouroboros import agent_task_pipeline as pipeline, post_task_evolution
    from ouroboros.gateway.settings import _has_running_agent_tasks, _has_started_agent_tasks
    from ouroboros.post_task_checkpoint import post_task_synthesis_in_flight
    from ouroboros.server_restart import _live_running_task_ids
    from ouroboros.task_results import load_task_result

    _lane(monkeypatch, tmp_path)
    bus = queue.Queue()
    monkeypatch.setattr(workers, "get_event_q", lambda: bus)
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(workers, "PENDING", [])
    entered, release = threading.Event(), threading.Event()
    synthesis_threads = []

    def consolidate(*args, **kwargs):
        synthesis_threads.append(threading.current_thread())
        entered.set()
        assert release.wait(10)

    monkeypatch.setattr(pipeline, "_run_chat_consolidation", consolidate)
    for name in (
        "_run_scratchpad_consolidation", "_record_task_facts", "_run_reflection",
        "_update_improvement_backlog", "_apply_reflection_memory_actions",
    ):
        monkeypatch.setattr(pipeline, name, lambda *a, **kw: None)
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
                env, task, "done", self._event_queue, pending, {}, {}, {}, tmp_path / "logs",
                budget_drive_root="", split_drive=False,
                project_scoped=bool(task.get("project_id")), project_task=False,
                parent_env=None, parent_task=None,
            )
            return pending

    # Both ordinary Main and ordinary Project conversation shapes must use
    # the same lifetime; neither has a headless workspace/task marker.
    for project_id in ("", "room"):
        entered.clear()
        release.clear()
        synthesis_threads.clear()
        actor = Actor()
        thread = threading.Thread(target=workers._run_chat_task, args=(actor, 1, "ordinary conversation"), kwargs={
            "task_metadata": {"project_id": project_id, "origin_suppressed": True},
        })
        try:
            thread.start()
            assert entered.wait(5)
            task_id = actor.task["id"]
            # Use the real callback thread, not a fabricated post-task wait:
            # synthesis belongs to the still-registered direct execution.
            assert synthesis_threads == [thread]
            assert thread.is_alive()
            assert post_task_synthesis_in_flight(tmp_path, task_id)
            registry = get_direct_activity_registry()
            assert registry.get(task_id).actor is actor
            assert workers.drain_repo_writers(0) == [task_id]
            assert _live_running_task_ids(SimpleNamespace(RUNNING={})) == [task_id]
            assert _has_running_agent_tasks() and _has_started_agent_tasks()
            assert mind.live_turns() == ("", True)
            # Production early delivery ran before synthesis. The terminal
            # completion stays buffered until post-task work returns.
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
        assert not thread.is_alive()
        assert get_direct_activity_registry().snapshot() == []
        assert not post_task_synthesis_in_flight(tmp_path, task_id)
        assert workers.drain_repo_writers(0) == []
        assert _live_running_task_ids(SimpleNamespace(RUNNING={})) == []
        assert not _has_running_agent_tasks() and not _has_started_agent_tasks()
        assert mind.live_turns() == ("", False)
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
    by its room and spawns nothing."""
    from ouroboros import agent as agent_module, project_naming

    _lane(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(project_naming, "spawn_turn_namer", lambda *a, **kw: calls.append((a, kw)))

    class Actor:
        def __init__(self, **kwargs):
            self._owner_message_admission_lock = threading.Lock()

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

    monkeypatch.setattr(agent_module, "make_agent", Actor)
    workers.handle_chat_direct(1, "проверь, почему карточка задачи потеряла элементы")
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == tmp_path and args[2] == "проверь, почему карточка задачи потеряла элементы"
    assert callable(kwargs["broadcast"])

    calls.clear()
    workers.handle_chat_direct(1, "и в комнате проекта", task_metadata={"project_id": "room"})
    assert calls == [], "a Project-room turn is named by its room"
