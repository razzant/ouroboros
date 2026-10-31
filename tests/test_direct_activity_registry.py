from __future__ import annotations

import concurrent.futures
import time
import pytest

from supervisor.active_activity import (
    DirectActivityRegistry,
    get_direct_activity_registry,
    track_direct_activity,
)


def test_registry_basic_lifecycle():
    registry = DirectActivityRegistry()
    assert registry.snapshot() == []

    entry = registry.register(
        "act-1",
        chat_id=1,
        client_message_id="msg-101",
        project_id="proj-alpha",
        kind="direct_chat",
        phase="thinking",
    )
    assert entry.activity_id == "act-1"
    assert entry.chat_id == 1
    assert entry.client_message_id == "msg-101"
    assert entry.project_id == "proj-alpha"
    assert entry.kind == "direct_chat"
    assert entry.phase == "thinking"

    snap = registry.snapshot()
    assert len(snap) == 1
    assert snap[0]["activity_id"] == "act-1"
    assert snap[0]["kind"] == "direct_chat"
    # Snapshot shape is the exact ActiveDirectTurn contract — no extra keys.
    assert set(snap[0].keys()) == {
        "activity_id",
        "chat_id",
        "project_id",
        "client_message_id",
        "kind",
        "phase",
        "started_at",
        "turn_phase",
    }
    assert snap[0]["turn_phase"] == "admitted"

    assert registry.get("act-1") is entry

    removed = registry.unregister("act-1")
    assert removed is not None
    assert removed.activity_id == "act-1"
    assert registry.get("act-1") is None
    assert registry.snapshot() == []


def test_registry_multi_chat_isolation():
    registry = DirectActivityRegistry()
    registry.register("act-main", chat_id=1, kind="direct_chat")
    registry.register("act-proj-2", chat_id=2, project_id="proj-2", kind="direct_chat")
    registry.register("act-proj-3", chat_id=3, project_id="proj-3", kind="direct_chat")

    assert len(registry.snapshot()) == 3
    assert len(registry.snapshot(chat_id=1)) == 1
    assert registry.snapshot(chat_id=1)[0]["activity_id"] == "act-main"
    assert len(registry.snapshot(chat_id=2)) == 1
    assert registry.snapshot(chat_id=2)[0]["activity_id"] == "act-proj-2"
    assert len(registry.snapshot(chat_id=3)) == 1
    assert len(registry.snapshot(chat_id=99)) == 0

    registry.unregister("act-proj-2")
    assert len(registry.snapshot()) == 2
    assert len(registry.snapshot(chat_id=2)) == 0
    assert len(registry.snapshot(chat_id=1)) == 1
    assert len(registry.snapshot(chat_id=3)) == 1


def test_track_direct_activity_context_manager():
    registry = get_direct_activity_registry()
    registry.clear()

    assert len(registry.snapshot(chat_id=1)) == 0

    with track_direct_activity(
        "act-cm-1",
        chat_id=1,
        client_message_id="msg-cm",
        kind="direct_chat",
        phase="thinking",
    ) as entry:
        assert entry.activity_id == "act-cm-1"
        assert len(registry.snapshot(chat_id=1)) == 1

    # Guaranteed cleanup on clean exit
    assert len(registry.snapshot(chat_id=1)) == 0

    # Guaranteed cleanup on exception
    with pytest.raises(RuntimeError, match="deliberate failure"):
        with track_direct_activity("act-cm-fail", chat_id=1, kind="direct_chat"):
            assert len(registry.snapshot(chat_id=1)) == 1
            raise RuntimeError("deliberate failure")

    assert len(registry.snapshot(chat_id=1)) == 0


def test_registry_concurrent_operations():
    registry = DirectActivityRegistry()

    def worker(worker_id: int):
        for i in range(50):
            aid = f"worker-{worker_id}-act-{i}"
            registry.register(aid, chat_id=worker_id % 5, kind="direct_chat")
            time.sleep(0.0001)
            registry.unregister(aid)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, i) for i in range(10)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    assert registry.snapshot() == []
    assert registry.executions() == []


def test_a_turn_execution_moves_forward_only_and_ends_with_its_entry(monkeypatch):
    from ouroboros import update_generation_facts

    dispatches = []  # the after-update facts get the first model dispatch of each turn, once
    monkeypatch.setattr(update_generation_facts, "observe_direct_dispatch", lambda **kw: dispatches.append(kw))
    registry = DirectActivityRegistry()
    execution = registry.begin_execution(4, client_message_id="m-4")
    assert execution.phase == "accepted" and registry.executions() == [execution]
    assert registry.snapshot() == []  # begun is not registered: the census lists admitted turns only
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as other:
        elsewhere = other.submit(registry.register, "act-other", 4).result()
    assert elsewhere.execution is not execution  # only the thread that began it adopts it
    registry.unregister("act-other")
    entry = registry.register("act-4", 4, reply_expected=True)
    assert entry.execution is execution and execution.task_id == "act-4"
    assert registry.snapshot()[0]["turn_phase"] == "admitted"
    execution.observe("log_event", {"type": "context_building_started"})
    call = {"llm_call_id": "llm-1"}
    execution.observe("main_llm_call_state", {**call, "phase": "started"})
    assert registry.snapshot()[0]["turn_phase"] == "requested"
    # The engine preparing the request and its queue are no dispatch: the first observation of it going out is.
    for activity in ({"phase": "preparing"}, {"phase": "queued", "blockers": [{"kind": "thread_busy"}]}):
        execution.observe("main_llm_call_state", {**call, "phase": "observed", "model_activity": activity})
    assert execution.phase == "requested" and dispatches == [], "no dispatch fact before the request goes out"
    time.sleep(0.01)
    execution.observe("main_llm_call_state", {**call, "phase": "observed", "model_activity": {"phase": "dispatching"}})
    assert registry.snapshot()[0]["turn_phase"] == "dispatched"
    assert [(d["task_id"], d["accepted_at"] <= d["dispatched_at"]) for d in dispatches] == [("act-4", True)]
    assert dispatches[0]["accepted_at"] == execution.marks[0][1]
    assert dispatches[0]["dispatched_at"] - execution.marks[-2][1] >= 0.01, "the observation's instant, not the start's"
    assert dispatches[0]["attempt_id"] == execution.execution_id
    # A late or repeated fact never moves it back; other frames move nothing.
    assert execution.advance("context_built") is False
    execution.observe("log_event", {"type": "context_building_finished"})
    execution.observe("main_llm_call_state", {**call, "phase": "observed", "model_activity": {"phase": "awaiting_response"}})
    execution.observe("main_llm_call_state", {**call, "phase": "finished"})
    execution.observe("log_event", {"type": "task_started"})
    assert [phase for phase, _ in execution.marks] == [
        "accepted", "admitted", "context_building", "requested", "dispatched"]
    assert len(dispatches) == 1, "a repeated dispatch frame reports nothing again"
    stamps = [at for _, at in execution.marks]
    assert stamps == sorted(stamps)
    registry.unregister("act-4")
    assert registry.executions() == [] and execution.ended_at >= stamps[-1]
    ended = execution.ended_at
    registry.end_execution(execution)  # the thread's own end after its entry's is a no-op
    assert registry.executions() == [] and execution.ended_at == ended


def test_a_call_its_route_never_observes_is_dispatched_at_its_start_once_it_finished(monkeypatch):
    from ouroboros import update_generation_facts

    dispatches = []
    monkeypatch.setattr(update_generation_facts, "observe_direct_dispatch", lambda **kw: dispatches.append(kw))
    registry = DirectActivityRegistry()
    # A route without the engine observes nothing between a call's start and its end; a failed call proves no send.
    execution = registry.register("act-6", 6, reply_expected=True).execution
    execution.observe("main_llm_call_state", {"llm_call_id": "llm-1", "phase": "started"})
    first_start = execution.marks[-1][1]
    execution.observe("main_llm_call_state", {"llm_call_id": "llm-1", "phase": "failed"})
    time.sleep(0.01)
    execution.observe("main_llm_call_state", {"llm_call_id": "llm-2", "phase": "started"})
    second_start = execution.call[1]
    execution.observe("main_llm_call_state", {"llm_call_id": "llm-other", "phase": "finished"})  # not this call
    assert execution.phase == "requested" and dispatches == []
    execution.observe("main_llm_call_state", {"llm_call_id": "llm-2", "phase": "finished"})
    assert execution.marks[-2:] == [("requested", first_start), ("dispatched", second_start)]
    assert [d["dispatched_at"] for d in dispatches] == [second_start] and second_start > first_start
    # A call its route did observe is dispatched only by its own observation, never at its start.
    observed = registry.register("act-7", 7, reply_expected=True).execution
    observed.observe("main_llm_call_state", {"llm_call_id": "llm-3", "phase": "started"})
    observed.observe("main_llm_call_state", {"llm_call_id": "llm-3", "phase": "observed",
                                             "model_activity": {"phase": "queued"}})
    observed.observe("main_llm_call_state", {"llm_call_id": "llm-3", "phase": "finished"})
    assert observed.phase == "requested" and len(dispatches) == 1


def test_a_registered_turn_without_a_begun_execution_gets_one_and_a_replacement_drops_it():
    registry = DirectActivityRegistry()
    first = registry.register("act-5", 5, client_message_id="m-5")
    assert first.execution is not None and first.execution.phase == "admitted"
    assert first.execution.client_message_id == "m-5" and registry.executions() == [first.execution]
    second = registry.register("act-5", 5)
    assert registry.executions() == [second.execution] and second.execution is not first.execution
    registry.clear()
    assert registry.executions() == []


def test_the_turn_queue_shows_its_own_frames_to_its_observer_only():
    import queue

    from supervisor.log_addressing import TurnEventQueue

    seen: list = []
    turn = TurnEventQueue(queue.Queue(), "t-own", 1, on_frame=lambda kind, data: seen.append((kind, data["type"])))
    turn.put({"type": "log_event", "data": {"type": "context_building_started", "task_id": "t-own"}})
    turn.put({"type": "log_event", "data": {"type": "context_building_started", "task_id": "t-other"}})
    turn.put({"type": "main_llm_call_state", "task_id": "t-own", "phase": "started"})
    assert seen == [("log_event", "context_building_started"), ("main_llm_call_state", "main_llm_call_state")]

    def broken(kind, data):
        raise RuntimeError("observer bug")

    inner: queue.Queue = queue.Queue()
    TurnEventQueue(inner, "t-own", 1, on_frame=broken).put({"type": "main_llm_call_state", "task_id": "t-own"})
    assert inner.get_nowait()["type"] == "main_llm_call_state"  # a failing observer never costs the frame


def test_only_an_owner_turn_reports_its_first_dispatch(monkeypatch):
    from ouroboros import update_generation_facts

    dispatches = []
    monkeypatch.setattr(update_generation_facts, "observe_direct_dispatch", lambda **kw: dispatches.append(kw))
    registry = DirectActivityRegistry()
    # A consciousness wake or an auto-resume after restart runs as a direct turn the owner did not send.
    wake = registry.register("wake-1", 4).execution
    for phase in ("started", "finished"):
        wake.observe("main_llm_call_state", {"phase": phase})
    assert wake.phase == "dispatched" and dispatches == []
    registry.unregister("wake-1")
    owner = registry.register("owner-1", 4, reply_expected=True).execution
    for phase in ("started", "finished", "started", "finished"):
        owner.observe("main_llm_call_state", {"phase": phase})
    assert [(d["task_id"], d["attempt_id"]) for d in dispatches] == [("owner-1", owner.execution_id)]
    registry.unregister("owner-1")
