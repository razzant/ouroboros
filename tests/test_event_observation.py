"""The existing bus exposes pending idle inputs without treating them as work, and is whole or over."""
from __future__ import annotations

import json
import multiprocessing
import os
import queue
import signal
import sys
import time

import pytest

from supervisor.event_observation import ObservedEventQueue, idle_input_addresses, pending_idle_observations
from supervisor.event_taxonomy import EVENT_DISPOSITIONS


@pytest.mark.parametrize("kind", [
    "main_llm_call_state",
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


def _fresh_bus(tmp_path, monkeypatch):
    from supervisor import workers

    ctx = multiprocessing.get_context("spawn")
    for key, value in (("DRIVE_ROOT", tmp_path), ("_CTX", ctx), ("_EVENT_Q", None),
                       ("_EVENT_Q_MANAGER", None), ("_EVENT_Q_GENERATION", ""),
                       ("_EVENT_Q_SHUTDOWN", False)):
        monkeypatch.setattr(workers, key, value)
    return workers, ctx, workers.get_event_q()


def _stopped_rows(tmp_path):
    rows = [json.loads(line) for line in (tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row.get("type") == "event_queue_generation_stopped"]


def test_original_manager_survives_killed_producer_and_accepts_replacement(tmp_path, monkeypatch):
    """The bus is whole or it is over (ARCHITECTURE §5): a killed PRODUCER leaves the bus, which keeps
    serving (quiet); the MANAGER's own death is read on the cached return as a waitpid fact (``EventBusLost``
    with its exit code), so nothing is handed a dead proxy and nothing respawns onto a dead bus."""
    workers, ctx, bus = _fresh_bus(tmp_path, monkeypatch)
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
        assert workers.event_bus_exitcode() is None and workers.get_event_q() is bus  # alive: the cached return
        manager._process.kill()  # the manager itself, by handle: the 05.11 pattern kill
        manager._process.join(10)
        with pytest.raises(workers.EventBusLost) as lost:
            workers.get_event_q()
        assert lost.value.exitcode == manager._process.exitcode and lost.value.exitcode not in (None, 0)
        assert workers.event_bus_exitcode() == lost.value.exitcode and isinstance(lost.value, RuntimeError)
        with pytest.raises(workers.EventBusLost):
            workers.get_event_q(create=False)  # a producer without an endpoint is refused the same way
    finally:
        for child in children:
            if child.is_alive():
                child.kill()
                child.join(10)
        workers.shutdown_event_q()
        assert not manager._process.is_alive()
    assert pending_idle_observations(bus) is None
    stopped = _stopped_rows(tmp_path)
    assert stopped[-1]["exitcode_before_shutdown"] == lost.value.exitcode and stopped[-1]["killed"] is False


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM semantics are POSIX; TerminateProcess cannot be ignored")
def test_the_manager_ignores_the_groups_sigterm_and_shutdown_ends_it_even_when_the_rpc_fails(tmp_path, monkeypatch):
    """The SIGTERM-immune bus (ARCHITECTURE §5): a SIGTERM aimed at the server's group (the launcher's group stop, #1142)
    leaves the bus alive and serving, so ``get_event_q`` keeps answering and a put still lands (quiet);
    ``shutdown_event_q`` ends the manager even when the RPC ``shutdown()`` fails (kill + join: the fire direction, no
    orphan bus), and journals that the manager was alive when the teardown reached it."""
    workers, _ctx, bus = _fresh_bus(tmp_path, monkeypatch)
    manager = workers._EVENT_Q_MANAGER
    try:
        os.kill(manager._process.pid, signal.SIGTERM)
        time.sleep(0.5)
        assert manager._process.is_alive() and workers.get_event_q() is bus and workers.event_bus_exitcode() is None
        bus.put({"type": "task_heartbeat", "task_id": "t"})
        assert bus.get(timeout=5)["task_id"] == "t"
        monkeypatch.setattr(manager, "shutdown", lambda: (_ for _ in ()).throw(RuntimeError("rpc refused")))
    finally:
        workers.shutdown_event_q()
    assert not manager._process.is_alive() and manager._process.exitcode == -signal.SIGKILL
    stopped = _stopped_rows(tmp_path)
    assert stopped[-1]["exitcode_before_shutdown"] is None and stopped[-1]["killed"] is True
    with pytest.raises(RuntimeError):
        workers.get_event_q()  # shut down: nothing creates a second bus in this process
