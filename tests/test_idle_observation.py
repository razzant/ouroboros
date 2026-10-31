"""A bounded event drain is not proof that a task has been observed idle."""
from __future__ import annotations

import multiprocessing
import queue as stdqueue
from types import SimpleNamespace

import pytest

from ouroboros import server_liveness
from supervisor import events, queue, workers
from supervisor.event_observation import EventQueueManager, ObservedEventQueue, pending_idle_observations


@pytest.fixture
def host(tmp_path, monkeypatch):
    clock = [5000.0]
    bus = ObservedEventQueue()
    meta = {"task": {"id": "root", "root_task_id": "root", "_attempt": 1, "chat_id": 7},
            "attempt": 1, "started_at": 1000.0, "last_progress_at": 1000.0, "worker_id": 0}
    running = {"root": meta}
    for mod in (queue, workers):
        monkeypatch.setattr(mod, "DRIVE_ROOT", tmp_path)
        monkeypatch.setattr(mod, "RUNNING", running)
        monkeypatch.setattr(mod, "PENDING", [])
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "get_event_q", lambda **kwargs: bus)
    monkeypatch.setattr(queue, "load_state", lambda: {})
    monkeypatch.setattr(queue, "get_task_idle_timeout_sec", lambda: 900.0)
    monkeypatch.setattr(queue, "get_per_call_timeout_ceiling_sec", lambda: 1800.0)
    monkeypatch.setattr(queue, "get_task_abs_ceiling_sec", lambda: None)
    monkeypatch.setattr(queue, "FINALIZATION_GRACE_SEC", 120)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **kwargs: True)
    monkeypatch.setattr(queue, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(queue, "_reap_queue", stdqueue.Queue())
    monkeypatch.setattr("supervisor.owner_stop.running_owner_stop_tasks", lambda *args, **kw: set())
    monkeypatch.setattr("supervisor.queue_timeouts.time", SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr(events, "time", SimpleNamespace(time=lambda: clock[0]))
    ctx = SimpleNamespace(RUNNING=running, DRIVE_ROOT=tmp_path,
                          append_jsonl=lambda *args: None, send_with_budget=lambda *args, **kw: True)
    liveness = [0.0, {}, 0.0, None]

    def drain(endpoint=None):
        return server_liveness.drain_worker_events(bus if endpoint is None else endpoint, ctx, liveness,
                                                   on_restart=lambda *args: None)

    def tick(endpoint=None):
        queue.enforce_task_timeouts(pending_events=pending_idle_observations(bus if endpoint is None else endpoint))

    return SimpleNamespace(bus=bus, meta=meta, running=running, clock=clock, drain=drain, tick=tick,
                           root=tmp_path)


def _call(task_id="root", attempt=1):
    return {"type": "main_llm_call_state", "task_id": task_id, "phase": "started",
            "task_attempt": attempt, "call_attempt": 1, "llm_call_id": "call",
            "execution_id": "exec", "round_id": "exec:344"}


def _controls(host):
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, drain_owner_entries
    return [row for row in drain_owner_entries(host.root, "root") if row.get("kind") == KIND_FINALIZE_NOW]


def _put_call(endpoint):
    endpoint.put(_call())


@pytest.mark.serial
def test_spawned_call_behind_real_manager_backlog_reaches_the_same_idle_owner(host, monkeypatch):
    monkeypatch.setitem(events.EVENT_HANDLERS, "noise", lambda *args: None)
    context = multiprocessing.get_context("spawn")
    with EventQueueManager(ctx=context) as manager:
        endpoint = manager.Queue()
        for _ in range(100):
            endpoint.put({"type": "noise", "task_id": "foreign"})
        child = context.Process(target=_put_call, args=(endpoint,), daemon=True)
        child.start()
        try:
            child.join(15)
            assert child.exitcode == 0
            assert host.drain(endpoint) is True and endpoint.qsize() == 1
            host.tick(endpoint)
            assert _controls(host) == [] and "active_llm_call" not in host.meta
            assert host.drain(endpoint) is False
            host.tick(endpoint)
            assert host.meta["active_llm_call"]["llm_call_id"] == "call"
            assert _controls(host) == []
        finally:
            if child.is_alive():
                child.kill()
                child.join(10)


def test_call_start_after_bounded_batch_does_not_create_idle_control(host, monkeypatch):
    # Discard unrelated telemetry cheaply; keep the actual main-call reducer.
    monkeypatch.setitem(events.EVENT_HANDLERS, "noise", lambda *args: None)
    for _ in range(100):
        host.bus.put({"type": "noise", "task_id": "other"})
    host.bus.put(_call())
    assert host.drain() is True
    owner_intake = []
    owner_intake.append("owner message")  # ordinary next phase is reached with work queued
    host.tick()
    assert owner_intake == ["owner message"] and host.bus.qsize() == 1
    assert "active_llm_call" not in host.meta and "finalization_requested_at" not in host.meta
    assert _controls(host) == []
    assert host.drain() is False
    assert host.meta["active_llm_call"]["llm_call_id"] == "call"
    host.tick()
    assert _controls(host) == []


def test_time_bound_keeps_pending_task_input_visible(host, monkeypatch):
    mono = [0.0]
    monkeypatch.setattr(server_liveness, "time", SimpleNamespace(monotonic=lambda: mono[0]))
    monkeypatch.setitem(events.EVENT_HANDLERS, "slow", lambda *args: mono.__setitem__(0, 3.0))
    host.bus.put({"type": "slow"})
    host.bus.put(_call())
    assert host.drain() is True
    host.tick()
    assert host.bus.qsize() == 1 and _controls(host) == []


def test_foreign_backlog_and_own_heartbeat_do_not_hide_silence(host):
    for phase in ("initial", "grace_elapsed"):
        for _ in range(101):
            host.bus.put({"type": "llm_usage", "task_id": "foreign"})
            host.bus.put({"type": "task_heartbeat", "task_id": "root"})
            host.bus.put({"type": "send_message", "task_id": "root", "is_progress": True,
                           "host_narration": True})
        host.tick()
        if phase == "initial":
            assert host.meta["silence_attention"]["state"] == "pending_attention"
            host.clock[0] += 121
    assert _controls(host) == [] and queue._reap_queue.empty()
    assert "root" in host.running


@pytest.mark.parametrize("descendant", [True, False])
def test_pending_child_progress_affects_only_its_current_ancestors(host, descendant):
    host.running["child"] = {"task": {"id": "child", "root_task_id": "root" if descendant else "elsewhere",
                                     "parent_task_id": "root" if descendant else "elsewhere"},
                             "started_at": 1000.0, "last_progress_at": 1000.0, "attempt": 1}
    # No parent/root stamp on this event: membership comes from current RUNNING.
    host.bus.put({"type": "llm_usage", "task_id": "child", "usage": {}})
    host.tick()
    assert bool(host.meta.get("silence_attention")) is not descendant


@pytest.mark.parametrize("rail", ["deadline", "absolute_ceiling"])
@pytest.mark.parametrize("unreadable", [False, True])
def test_pending_or_unreadable_observation_never_spares_hard_axes(host, monkeypatch, rail, unreadable):
    if rail == "deadline":
        host.meta["task"]["deadline_at"] = "1970-01-01T01:00:00+00:00"
    else:
        monkeypatch.setattr(queue, "get_task_abs_ceiling_sec", lambda: 1000.0)
    host.bus.put(_call())
    queue.enforce_task_timeouts(pending_events=None if unreadable else pending_idle_observations(host.bus))
    assert _controls(host)[0]["text"] == rail


def test_pending_observation_defers_attention_but_only_folded_progress_resolves(host, monkeypatch):
    host.tick()
    episode = host.meta["silence_attention"]["episode_id"]
    host.clock[0] += 121
    host.bus.put({"type": "send_message", "task_id": "root", "chat_id": 7, "text": "Made progress",
                   "is_progress": True})
    host.tick()
    assert queue._reap_queue.empty() and _controls(host) == []
    assert host.meta["silence_attention"]["episode_id"] == episode
    assert host.meta["silence_attention"]["state"] == "pending_attention"
    assert host.meta["last_progress_at"] == 1000.0
    monkeypatch.setattr("supervisor.events_chat_delivery.time", SimpleNamespace(time=lambda: host.clock[0]))
    host.drain()
    host.tick()
    assert host.meta["last_progress_at"] == host.clock[0]
    assert _controls(host) == [] and "finalization_requested_at" not in host.meta
    assert host.meta["silence_attention"]["state"] == "progress_observed"


def test_unknown_observation_does_not_invent_progress(host):
    queue.enforce_task_timeouts(pending_events=None)
    assert _controls(host) == [] and host.meta["last_progress_at"] == 1000.0
    host.tick()
    assert host.meta["silence_attention"]["state"] == "pending_attention" and _controls(host) == []


def test_old_attempt_input_defers_only_until_its_existing_reducer_rejects_it(host):
    host.bus.put(_call(attempt=2))
    host.tick()
    assert _controls(host) == []
    host.drain()
    host.tick()
    assert "active_llm_call" not in host.meta and _controls(host) == []
    assert host.meta["silence_attention"]["state"] == "pending_attention"
