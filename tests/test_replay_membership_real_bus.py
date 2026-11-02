"""Replay membership end to end on the real two-lane bus.

The replay sweep (``supervisor.terminal_delivery.replay_pending_deliveries``) and the
bus that owns delivery membership (``supervisor.event_observation.ObservedEventQueue``)
were built against one interface; this pins them together on the spawn-context
``EventQueueManager`` the server uses, so a proxy that does not expose the membership
methods, or a sweep that spends dispatch attempts on queued copies, fails here.

Owner (14 Oct 2026, 18:56): «не сжигать попытки повторной доставки, пока сообщение уже
стоит в очереди».
"""
from __future__ import annotations

import multiprocessing

import pytest

from supervisor import terminal_delivery as td
from supervisor.event_observation import EventQueueManager
from tests.test_delivery_replay_membership import _age, _attempts, _owed


@pytest.mark.serial
def test_a_queued_or_handled_answer_is_not_copied_until_the_consumer_releases_it(tmp_path):
    context = multiprocessing.get_context("spawn")
    with EventQueueManager(ctx=context) as manager:
        bus = manager.Queue()
        event = _owed(tmp_path)
        bus.put(event)  # the worker's own live put

        for _ in range(5):  # queued: many due sweeps, no copy, no attempt
            _age(tmp_path)
            assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == []
        assert _attempts(tmp_path, event["delivery_id"]) == 0
        assert bus.qsize() == 1

        served = bus.get_nowait()  # the consumer took it; the handler is running
        assert served["delivery_id"] == event["delivery_id"]
        _age(tmp_path)
        assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == []
        assert bus.qsize() == 0 and _attempts(tmp_path, event["delivery_id"]) == 0

        bus.release_delivery(event["delivery_id"])  # handler returned without a receipt
        assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == [event["delivery_id"]]
        assert bus.qsize() == 1 and _attempts(tmp_path, event["delivery_id"]) == 0
        copy = bus.get_nowait()
        assert copy["delivery_id"] == event["delivery_id"] and copy["text"] == event["text"]


@pytest.mark.serial
def test_the_same_sweep_without_the_membership_methods_admits_nothing(tmp_path):
    """Control: an endpoint that cannot answer membership is unknown, never 'absent'."""

    class _Plain:
        def __init__(self) -> None:
            self.items: list = []

        def put(self, item) -> None:
            self.items.append(item)

        def put_nowait(self, item) -> None:
            self.items.append(item)

    event = _owed(tmp_path)
    plain = _Plain()
    _age(tmp_path)
    assert td.replay_pending_deliveries(tmp_path, event_queue=plain) == []
    assert plain.items == [] and _attempts(tmp_path, event["delivery_id"]) == 0


@pytest.mark.serial
def test_the_last_replay_copy_is_not_exhausted_while_it_is_still_outstanding(tmp_path, monkeypatch):
    """The replay cap gives a delivery up only once the bus says its last copy left the handler."""
    from supervisor.event_observation import ObservedEventQueue

    monkeypatch.setattr("supervisor.message_bus.send_with_budget", lambda *a, **k: None)
    bus = ObservedEventQueue()
    event = _owed(tmp_path)
    did = event["delivery_id"]
    for attempt in range(td._PENDING_MAX_REPLAYS):
        _age(tmp_path)
        assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == [did]
        assert td.record_replay_dispatch(tmp_path, did)
        if attempt < td._PENDING_MAX_REPLAYS - 1:
            assert bus.get_nowait()["delivery_id"] == did
            bus.release_delivery(did)
    _age(tmp_path)  # the fifth copy is still queued when its backoff expires
    assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == []
    assert bus.delivery_outstanding(did)
    assert td.terminal_answer_receipts(tmp_path, event["task_id"])["state"] == "owed"
    assert bus.get_nowait()["delivery_id"] == did  # the handler ran and returned without a receipt
    bus.release_delivery(did)
    _age(tmp_path)
    assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == []
    assert td.terminal_answer_receipts(tmp_path, event["task_id"])["state"] == "exhausted"
