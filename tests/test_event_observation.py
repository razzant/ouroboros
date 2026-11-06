"""The existing bus exposes pending idle inputs without treating them as work."""
from __future__ import annotations

import multiprocessing
import queue

import pytest

from supervisor.event_observation import ObservedEventQueue, idle_input_addresses, pending_idle_observations
from supervisor.event_taxonomy import EVENT_DISPOSITIONS


@pytest.mark.parametrize("kind", [
    "main_llm_call_state", "external_wait_lease",
    "budget_resume_child",
])
def test_idle_input_kinds_are_declared_by_the_event_registry(kind):
    assert EVENT_DISPOSITIONS[kind].idle_input
    assert idle_input_addresses({"type": kind, "task_id": "child", "parent_task_id": "middle",
                                 "root_task_id": "root"}) == {"child", "middle", "root"}


@pytest.mark.parametrize("event", [
    {"type": "task_heartbeat"}, {"type": "send_message", "is_progress": True, "host_narration": True},
    {"type": "log_event", "data": {"type": "llm_usage", "task_id": "t"}},
])
def test_liveness_and_telemetry_are_not_pending_idle_inputs(event):
    bus = ObservedEventQueue()
    bus.put({**event, "task_id": "t"})
    assert bus.qsize() == 1 and pending_idle_observations(bus) == set()
    bus.put({"type": "send_message", "task_id": "t", "is_progress": True})
    assert pending_idle_observations(bus) == {"t"}


def test_pending_index_has_the_same_put_get_boundary_and_no_history():
    bus = ObservedEventQueue(maxsize=2)
    first = {"type": "main_llm_call_state", "task_id": "t"}
    bus.put(first)
    bus.put_nowait({"type": "llm_usage", "task_id": "t"})
    with pytest.raises(queue.Full):
        bus.put_nowait({"type": "llm_usage", "task_id": "refused"})
    assert pending_idle_observations(bus) == {"t"}
    # Local caller mutation cannot corrupt the retained count's address.
    first["task_id"] = "changed-after-put"
    assert bus.get_nowait() is first
    assert pending_idle_observations(bus) == {"t"}
    bus.get()
    with pytest.raises(queue.Empty):
        bus.get_nowait()
    assert pending_idle_observations(bus) == set()


@pytest.mark.parametrize("endpoint", [None, queue.Queue()])
def test_missing_observation_is_unknown_not_an_empty_queue(endpoint):
    assert pending_idle_observations(endpoint) is None


def _produce(bus, accepted, release, phase):
    bus.put({"type": "main_llm_call_state", "task_id": "spawned", "phase": phase,
             "task_attempt": 2, "llm_call_id": "call", "execution_id": "exec",
             "round_id": "exec:1", "call_attempt": 1})
    accepted.set()
    release.wait(30)


def test_original_manager_survives_killed_producer_and_accepts_replacement(tmp_path, monkeypatch):
    from supervisor import workers

    ctx = multiprocessing.get_context("spawn")
    for key, value in (("DRIVE_ROOT", tmp_path), ("_CTX", ctx), ("_EVENT_Q", None),
                       ("_EVENT_Q_MANAGER", None), ("_EVENT_Q_GENERATION", ""),
                       ("_EVENT_Q_SHUTDOWN", False)):
        monkeypatch.setattr(workers, key, value)
    bus = workers.get_event_q()
    manager, generation = workers._EVENT_Q_MANAGER, workers.event_queue_generation()
    children = []
    try:
        for phase in ("started", "finished"):
            accepted, release = ctx.Event(), ctx.Event()
            child = ctx.Process(target=_produce, args=(bus, accepted, release, phase), daemon=True)
            children.append(child)
            child.start()
            assert accepted.wait(15), "spawned producer never reached the existing bus"
            assert pending_idle_observations(bus) == {"spawned"}
            child.kill()
            child.join(10)
            assert not child.is_alive() and child.exitcode != 0
            assert workers.get_event_q() is bus
            assert workers.event_queue_generation() == generation
            assert bus.get(timeout=3)["phase"] == phase
            assert pending_idle_observations(bus) == set()
        with pytest.raises(Exception):  # serialization fails before manager Queue.put
            bus.put({"type": "llm_usage", "task_id": "not-accepted", "value": lambda: None})
        assert pending_idle_observations(bus) == set()
    finally:
        for child in children:
            if child.is_alive():
                child.kill()
                child.join(10)
        workers.shutdown_event_q()
        assert not manager._process.is_alive()
    assert pending_idle_observations(bus) is None
