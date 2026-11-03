"""The one supervisor bus serves its control lane before its telemetry lane.

Owner decision (14 Oct 2026): «делим EVENT_Q на две полосы — приоритетную
(сообщения в чат, старт/стоп задач) и фоновую (пульс, статистика токенов), чтобы
телеметрия никогда не задерживала ответы». The lane is declared once on the
taxonomy, the bus routes on put and serves control-first on get, the idle-input
index spans both lanes, and a queued send's delivery identity stays outstanding
from its put until the consumer releases it. Every guard is pinned in both
directions, and the manager-backed shape runs on the real spawn-context
``EventQueueManager`` the server uses.
"""
from __future__ import annotations

import multiprocessing
import queue as stdqueue
from types import SimpleNamespace

import pytest

from supervisor import event_observation
from supervisor.event_observation import (
    EventQueueManager,
    ObservedEventQueue,
    delivery_outstanding,
    event_lane,
    lane_observations,
    pending_idle_observations,
)
from supervisor.event_taxonomy import CONTROL_LANE, TELEMETRY_LANE


def _telemetry(n: int) -> dict:
    kind = ("log_event", "llm_usage", "task_heartbeat")[n % 3]
    if kind == "log_event":
        return {"type": kind, "n": n, "data": {"type": "tool_call_started", "task_id": "worker-task"}}
    return {"type": kind, "n": n, "task_id": "worker-task"}


def _send(n: int = 0, **extra: object) -> dict:
    return {"type": "send_message", "n": n, "task_id": "answer-task", "chat_id": 1, "text": "final", **extra}


@pytest.mark.serial
def test_a_send_behind_five_thousand_telemetry_frames_is_the_first_event_served_on_the_real_manager():
    context = multiprocessing.get_context("spawn")
    with EventQueueManager(ctx=context) as manager:
        endpoint = manager.Queue()
        for n in range(5000):
            endpoint.put(_telemetry(n))
        endpoint.put(_send(5000, delivery_id="final:answer-task:0123456789abcdef"))
        assert endpoint.qsize() == 5001
        lanes = lane_observations(endpoint)  # the proxy exposes the lane facts too
        assert lanes[CONTROL_LANE]["depth"] == 1 and lanes[TELEMETRY_LANE]["depth"] == 5000
        assert lanes[TELEMETRY_LANE]["oldest_age_sec"] >= lanes[CONTROL_LANE]["oldest_age_sec"] >= 0
        first = endpoint.get_nowait()
        assert first["type"] == "send_message" and first["n"] == 5000
        assert endpoint.qsize() == 5000
        # The membership rides the manager too: outstanding until the consumer releases it.
        assert endpoint.delivery_outstanding("final:answer-task:0123456789abcdef") is True
        assert endpoint.put_unless_outstanding(_send(1, delivery_id="final:answer-task:0123456789abcdef")) is False
        assert endpoint.qsize() == 5000
        endpoint.release_delivery("final:answer-task:0123456789abcdef")
        assert endpoint.delivery_outstanding("final:answer-task:0123456789abcdef") is False
        assert delivery_outstanding(endpoint, "final:answer-task:0123456789abcdef") is False
        # Telemetry keeps its own FIFO behind the control lane.
        assert [endpoint.get_nowait()["n"] for _ in range(3)] == [0, 1, 2]


def test_lane_facts_show_a_telemetry_backlog_waiting_behind_control(monkeypatch):
    """Per lane, the depth and the oldest envelope's age at the moment of the read: a
    telemetry backlog queued first keeps aging while control is served ahead of it."""
    clock = [100.0]
    monkeypatch.setattr(event_observation, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    bus = ObservedEventQueue()
    empty = {"depth": 0, "oldest_age_sec": None}
    assert bus.lane_facts() == {CONTROL_LANE: empty, TELEMETRY_LANE: empty}
    for n in range(3):
        bus.put(_telemetry(n))
    clock[0] = 130.0
    bus.put(_send(1))
    bus.put(_send(2))
    clock[0] = 160.0
    assert bus.lane_facts() == {CONTROL_LANE: {"depth": 2, "oldest_age_sec": 30.0},
                                TELEMETRY_LANE: {"depth": 3, "oldest_age_sec": 60.0}}
    assert [bus.get_nowait()["n"] for _ in range(2)] == [1, 2]
    clock[0] = 190.0
    assert bus.lane_facts() == {CONTROL_LANE: empty, TELEMETRY_LANE: {"depth": 3, "oldest_age_sec": 90.0}}
    bus.get_nowait()
    assert bus.lane_facts()[TELEMETRY_LANE] == {"depth": 2, "oldest_age_sec": 90.0}
    assert lane_observations(bus) == bus.lane_facts()


@pytest.mark.parametrize("endpoint", [None, stdqueue.Queue(), SimpleNamespace(lane_facts=lambda: "busy")])
def test_unobservable_lanes_are_unknown_not_empty(endpoint):
    assert lane_observations(endpoint) is None


def test_fifo_inside_each_lane_and_control_first_across_them():
    bus = ObservedEventQueue()
    order = [("c", 1), ("t", 1), ("c", 2), ("t", 2), ("t", 3), ("c", 3)]
    for lane, n in order:
        bus.put(_send(n) if lane == "c" else {**_telemetry(n), "n": n})
    served = []
    while True:
        try:
            evt = bus.get_nowait()
        except stdqueue.Empty:
            break
        served.append((event_lane(evt)[0], evt["n"]))
    assert served == [("c", 1), ("c", 2), ("c", 3), ("t", 1), ("t", 2), ("t", 3)]
    assert bus.qsize() == 0


def test_an_undeclared_kind_rides_the_control_lane():
    bus = ObservedEventQueue()
    bus.put(_telemetry(0))
    bus.put({"type": "kind_nobody_declared", "n": 1})
    assert event_lane({"type": "kind_nobody_declared"}) == CONTROL_LANE
    assert bus.get_nowait()["n"] == 1
    assert event_lane(bus.get_nowait()) == TELEMETRY_LANE


def test_the_idle_input_index_spans_both_lanes_and_follows_the_served_order():
    bus = ObservedEventQueue()
    bus.put({"type": "llm_usage", "task_id": "telemetry-task"})  # idle input, telemetry lane
    bus.put({"type": "schedule_subagent", "task_id": "control-task"})  # idle input, control lane
    bus.put({"type": "task_heartbeat", "task_id": "liveness-only"})  # never an idle input
    assert pending_idle_observations(bus) == {"telemetry-task", "control-task"}
    assert bus.get_nowait()["type"] == "schedule_subagent"
    assert pending_idle_observations(bus) == {"telemetry-task"}
    assert bus.get_nowait()["type"] == "llm_usage"
    assert pending_idle_observations(bus) == set()
    assert bus.get_nowait()["type"] == "task_heartbeat"
    assert bus.qsize() == 0


def test_delivery_membership_counts_every_put_and_releases_one_at_a_time():
    bus = ObservedEventQueue()
    did = "final:answer-task:0123456789abcdef"
    assert bus.delivery_outstanding(did) is False
    bus.put(_send(1, delivery_id=did))
    bus.put_nowait(_send(2, delivery_id=did))  # the worker's original and a queued copy
    assert bus.delivery_outstanding(did) is True and bus.qsize() == 2
    assert bus.put_unless_outstanding(_send(3, delivery_id=did)) is False
    assert bus.qsize() == 2, "a refused copy enqueues nothing"
    bus.get_nowait()  # dequeue does NOT release: the handler may still be sending
    bus.release_delivery(did)
    assert bus.delivery_outstanding(did) is True, "one release retires one put"
    bus.release_delivery(did)
    assert bus.delivery_outstanding(did) is False
    bus.release_delivery(did)  # a surplus release is harmless, never negative
    assert bus.delivery_outstanding(did) is False
    assert bus.put_unless_outstanding(_send(4, delivery_id=did)) is True
    assert bus.delivery_outstanding(did) is True and bus.qsize() == 2
    # Sends without an id and non-send events are enqueued without membership.
    assert bus.put_unless_outstanding(_send(5)) is True
    assert bus.put_unless_outstanding({"type": "task_heartbeat", "task_id": "t", "delivery_id": did}) is True
    assert bus.qsize() == 4 and bus.delivery_outstanding(did) is True


def test_put_unless_outstanding_keeps_the_queue_bound_and_the_idle_index():
    bus = ObservedEventQueue(maxsize=1)
    assert bus.put_unless_outstanding({"type": "llm_usage", "task_id": "t"}) is True
    with pytest.raises(stdqueue.Full):
        bus.put_unless_outstanding({"type": "llm_usage", "task_id": "refused"})
    assert pending_idle_observations(bus) == {"t"}
    assert bus.get_nowait()["task_id"] == "t"
    assert pending_idle_observations(bus) == set()


@pytest.mark.parametrize("endpoint", [None, stdqueue.Queue(), SimpleNamespace(delivery_outstanding=lambda did: "yes")])
def test_an_unobservable_membership_is_unknown_not_absent(endpoint):
    assert delivery_outstanding(endpoint, "final:x:0123456789abcdef") is None
    assert delivery_outstanding(ObservedEventQueue(), "final:x:0123456789abcdef") is False


def test_a_contended_schedule_admission_is_never_re_put_on_the_bus():
    """Producers keep one put and the consumer never re-puts (the bus docstring): a contended
    schedule_subagent is retried by its claim at the head of the preparation generation's own
    lane, so the bus holds no second copy and no deferral owner exists beside the lane."""
    from supervisor import workspace_preparation

    assert not hasattr(workspace_preparation, "defer_schedule_event")
    bus = ObservedEventQueue()
    bus.put({"type": "schedule_subagent", "task_id": "child", "parent_task_id": "root",
             "root_task_id": "root", "chat_id": 1, "delegation_role": "subagent"})
    assert bus.get_nowait()["task_id"] == "child" and bus.qsize() == 0
