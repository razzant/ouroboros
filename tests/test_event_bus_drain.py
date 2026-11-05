"""Cost law L11: the supervisor drain takes its batch in ONE bus round trip, lanes unchanged.

``get_nowait`` per event paid a manager proxy round trip each (~940 a minute on 01.11).
``ObservedEventQueue.drain(n)`` serves the same envelopes in the same order that ``n``
consecutive ``get`` calls would (control before telemetry, FIFO inside each lane), with the
same idle-input bookkeeping, and ``restore_front`` hands an unserved tail back to the front of
its lanes with its first put stamps — so a pass cut by its time bound (or by a raise) leaves the
bus exactly as a get-per-event consumer stopping at the same point would have. Every guard is
pinned both ways; the manager-backed shape runs on the real spawn-context manager.
"""
from __future__ import annotations

import multiprocessing
import queue as stdqueue
import time
from types import SimpleNamespace

import pytest

from ouroboros import server_liveness as sl
from supervisor.event_observation import EventQueueManager, ObservedEventQueue
from supervisor.event_taxonomy import CONTROL_LANE, TELEMETRY_LANE


def _mixed(n: int) -> list:
    """Interleaved puts: sends (control, idle inputs for their tasks) between heartbeats and usage rows."""
    events = []
    for i in range(n):
        if i % 3 == 0:
            events.append({"type": "send_message", "n": i, "task_id": f"t{i}", "is_progress": True, "chat_id": 1})
        else:
            events.append({"type": ("task_heartbeat", "llm_usage")[i % 2], "n": i, "task_id": f"t{i}"})
    return events


def _filled(events) -> ObservedEventQueue:
    bus = ObservedEventQueue()
    for event in events:
        bus.put(dict(event))
    return bus


def test_drain_serves_what_consecutive_gets_would_and_keeps_the_idle_index():
    events = _mixed(12)
    by_get, by_drain = _filled(events), _filled(events)
    expected = [by_get.get_nowait()["n"] for _ in range(7)]
    taken = by_drain.drain(7)
    assert [event["n"] for event, _stamp in taken] == expected
    assert expected[:4] == [0, 3, 6, 9], "every control event first, in put order"
    assert by_drain.pending_idle_inputs() == by_get.pending_idle_inputs()
    assert by_drain.qsize() == by_get.qsize() == 5
    assert [by_drain.get_nowait()["n"] for _ in range(5)] == [by_get.get_nowait()["n"] for _ in range(5)]
    assert by_drain.drain(10) == [] and by_drain.drain(0) == []


def test_restore_front_returns_an_unserved_tail_as_if_never_taken(monkeypatch):
    from supervisor import event_observation

    clock = [100.0]
    monkeypatch.setattr(event_observation.time, "monotonic", lambda: clock[0])
    bus = ObservedEventQueue()
    did = "final:t0:0123456789abcdef"
    for event in _mixed(6):
        bus.put({**event, **({"delivery_id": did} if event["n"] == 0 else {})})
        clock[0] += 1.0
    pending_before = bus.pending_idle_inputs()
    taken = bus.drain(6)  # puts were stamped 100..105: n0 100, n1 101, n2 102, n3 103, n4 104, n5 105
    bus.put({"type": "send_message", "n": 99, "task_id": "late", "is_progress": True, "chat_id": 1})
    bus.put({"type": "task_heartbeat", "n": 98, "task_id": "late"})
    bus.restore_front(taken[1:])  # the first one (n0) was served; the rest were not
    clock[0] = 111.0
    facts = bus.lane_facts()
    # Restored envelopes keep their first put stamps: the oldest control is n3 (103), telemetry n1 (101).
    assert (facts[CONTROL_LANE]["oldest_age_sec"], facts[TELEMETRY_LANE]["oldest_age_sec"]) == (8.0, 10.0)
    assert (facts[CONTROL_LANE]["depth"], facts[TELEMETRY_LANE]["depth"]) == (2, 5)
    assert bus.pending_idle_inputs() == (pending_before - {"t0"}) | {"late"}
    assert bus.delivery_outstanding(did) is True, "taking and restoring never touches delivery membership"
    assert [bus.get_nowait()["n"] for _ in range(bus.qsize())] == [3, 99, 1, 2, 4, 5, 98]


class _CountingBus:
    """The real bus behind a counter of the consumer's round trips (each proxy call is one)."""

    def __init__(self, bus: ObservedEventQueue) -> None:
        self.bus, self.calls = bus, []

    def __getattr__(self, name):
        method = getattr(self.bus, name)

        def call(*args, **kwargs):
            self.calls.append(name)
            return method(*args, **kwargs)

        return call


def _ctx():
    return SimpleNamespace(dispatched=[], DRIVE_ROOT=None)


def test_one_pass_takes_its_batch_in_one_round_trip(monkeypatch):
    from supervisor import events as events_mod

    monkeypatch.setattr(events_mod, "dispatch_event", lambda evt, ctx: ctx.dispatched.append(evt["n"]))
    monkeypatch.setattr(sl, "SUPERVISOR_EVENT_BATCH_MAX_EVENTS", 5)
    counting = _CountingBus(_filled(_mixed(12)))
    ctx = _ctx()
    assert sl.drain_worker_events(counting, ctx, [time.monotonic(), {}, 0.0, None],
                                  on_restart=lambda *_: None) is True
    assert counting.calls == ["drain"], "one round trip for the whole pass, never one per event"
    assert ctx.dispatched == [0, 3, 6, 9, 1]
    counting.calls.clear()
    while sl.drain_worker_events(counting, ctx, [time.monotonic(), {}, 0.0, None], on_restart=lambda *_: None):
        pass
    assert set(counting.calls) == {"drain"} and len(counting.calls) == 2
    assert sorted(ctx.dispatched) == list(range(12))


@pytest.mark.parametrize("handler_seconds, expected_served", [(0.03, [0, 3]), (0.06, [0])])
def test_a_pass_cut_by_its_time_bound_restores_its_tail_to_the_front(monkeypatch, handler_seconds, expected_served):
    from supervisor import events as events_mod

    clock = [100.0]
    monkeypatch.setattr(sl, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    late = {"type": "send_message", "n": 50, "task_id": "late", "is_progress": True, "chat_id": 1}

    def dispatch(evt, ctx):
        ctx.dispatched.append(evt["n"])
        if evt["n"] == 3:
            counting.bus.put(dict(late))  # arrives while the pass runs
        clock[0] += handler_seconds

    monkeypatch.setattr(events_mod, "dispatch_event", dispatch)
    monkeypatch.setattr(sl, "SUPERVISOR_EVENT_BATCH_MAX_SEC", 0.05)
    counting = _CountingBus(_filled(_mixed(12)))
    ctx = _ctx()
    assert sl.drain_worker_events(counting, ctx, [clock[0], {}, 0.0, None],
                                  on_restart=lambda *_: None) is True
    served = list(ctx.dispatched)
    assert served == expected_served and counting.calls == ["drain", "restore_front"]
    # What remains is what a get-per-event consumer stopping at the same point leaves queued.
    reference = _filled(_mixed(12))
    for _ in served:
        reference.get_nowait()
    if 3 in served:
        reference.put(dict(late))
    assert [counting.bus.get_nowait()["n"] for _ in range(counting.bus.qsize())] == \
        [reference.get_nowait()["n"] for _ in range(reference.qsize())]


def test_a_raise_out_of_the_pass_restores_the_unserved_tail(monkeypatch):
    from ouroboros import finalization_timing

    def stamp(evt):
        if evt.get("n") == 3:
            raise RuntimeError("stamp failed")

    monkeypatch.setattr(finalization_timing, "stamp_finalization_dequeue", stamp)
    monkeypatch.setattr(sl, "observe_worker_event_lag", lambda *_a: None)
    from supervisor import events as events_mod

    monkeypatch.setattr(events_mod, "dispatch_event", lambda evt, ctx: ctx.dispatched.append(evt["n"]))
    bus = _filled(_mixed(6))
    ctx = _ctx()
    with pytest.raises(RuntimeError):
        sl.drain_worker_events(bus, ctx, [time.monotonic(), {}, 0.0, None], on_restart=lambda *_: None)
    assert ctx.dispatched == [0], "the event whose start raised is consumed, as with get"
    assert [bus.get_nowait()["n"] for _ in range(bus.qsize())] == [1, 2, 4, 5]


def test_a_plain_queue_is_still_read_one_event_at_a_time(monkeypatch):
    from supervisor import events as events_mod

    monkeypatch.setattr(events_mod, "dispatch_event", lambda evt, ctx: ctx.dispatched.append(evt["n"]))
    plain: "stdqueue.Queue" = stdqueue.Queue()
    for n in range(3):
        plain.put({"type": "worker_event", "n": n})
    ctx = _ctx()
    assert sl.drain_worker_events(plain, ctx, [time.monotonic(), {}, 0.0, None], on_restart=lambda *_: None) is False
    assert ctx.dispatched == [0, 1, 2]


def test_drain_and_restore_ride_the_real_manager_proxy():
    context = multiprocessing.get_context("spawn")
    with EventQueueManager(ctx=context) as manager:
        endpoint = manager.Queue()
        for event in _mixed(9):
            endpoint.put(event)
        taken = endpoint.drain(4)
        assert [event["n"] for event, _stamp in taken] == [0, 3, 6, 1]
        endpoint.restore_front(taken[2:])
        assert endpoint.qsize() == 7
        assert [endpoint.get_nowait()["n"] for _ in range(7)] == [6, 1, 2, 4, 5, 7, 8]
