"""Replay membership lives in the bus (INTERFACES I3): a copy that already stands in the
queue, or is being handled, is never copied again and burns no attempt.

Owner (18:56): «не сжигать попытки повторной доставки, пока сообщение уже стоит в очереди».
LIVE_MEASURE Q6: 331 of 336 replays on 14 Oct were pure duplicates of originals that were
delivered later, and the one false ⚠️ came from attempts burned while the original was
still queued. ``replay_pending_deliveries`` now puts a copy only through the bus's
``put_unless_outstanding``; a still-outstanding id or an unobservable bus admits nothing
and counts nothing; admitted copies only stamp backoff, never dispatch attempts.
"""
from __future__ import annotations

import json
import pathlib
import queue as stdqueue

import pytest

from supervisor import terminal_delivery as td
from tests._cancel_intents_shared import DeliveryBus, _CaptureQueue


def _age(drive, seconds: float = 7200) -> None:
    from datetime import datetime, timedelta, timezone

    store = pathlib.Path(drive) / "state" / "terminal_deliveries.json"
    data = json.loads(store.read_text(encoding="utf-8"))
    old = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
    for row in (data.get("pending") or {}).values():
        row["registered_at"] = old
        if "last_replay_at" in row:
            row["last_replay_at"] = old
    store.write_text(json.dumps(data), encoding="utf-8")


def _owed(drive, tid: str = "root", text: str = "the answer") -> dict:
    event = {"type": "send_message", "chat_id": 1, "task_id": tid, "text": text,
             "delivery_id": td.delivery_id_for(tid, text)}
    assert td.register_pending_delivery(drive, event)
    return event


def _attempts(drive, did: str) -> int:
    rows = {row["delivery_id"]: row for row in td.pending_deliveries(drive)}
    return int(rows[did].get("dispatch_attempts") or 0)


@pytest.mark.parametrize("bus_kind", ["capture", "fifo"])
def test_a_queued_original_is_never_copied_and_burns_no_attempt(tmp_path, bus_kind):
    bus = _CaptureQueue() if bus_kind == "capture" else DeliveryBus()
    event = _owed(tmp_path)
    bus.put(event)  # the worker's own live put
    assert bus.delivery_outstanding(event["delivery_id"])
    for _ in range(25):
        _age(tmp_path)
        assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == []
    assert _attempts(tmp_path, event["delivery_id"]) == 0
    copies = bus.events if bus_kind == "capture" else list(bus.queue)
    assert len(copies) == 1, "the original stands alone in the queue"


def test_after_the_handler_finishes_without_a_receipt_exactly_one_replay_is_due(tmp_path):
    bus = _CaptureQueue()
    event = _owed(tmp_path)
    bus.put(event)
    _age(tmp_path)
    assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == []
    bus.release_delivery(event["delivery_id"])  # the handler returned (its send failed), no receipt
    assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == [event["delivery_id"]]
    assert _attempts(tmp_path, event["delivery_id"]) == 0
    assert len(bus.events) == 2 and bus.events[1]["text"] == "the answer"
    assert "registered_at" not in bus.events[1] and "replay_attempts" not in bus.events[1]
    # The replay copy is outstanding now: the next sweep (due or not) copies nothing more.
    _age(tmp_path)
    assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == []
    assert _attempts(tmp_path, event["delivery_id"]) == 0


def test_admitted_queue_copies_do_not_spend_dispatch_attempts(tmp_path):
    bus = _CaptureQueue()
    event = _owed(tmp_path)
    for _ in range(3):
        _age(tmp_path)
        assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == [event["delivery_id"]]
        assert _attempts(tmp_path, event["delivery_id"]) == 0
        assert td.pending_deliveries(tmp_path)[0]["last_replay_at"]
        bus.release_all()
    assert len(bus.events) == 3


@pytest.mark.parametrize("bus_kind", ["plain_queue", "raising_membership"])
def test_an_unobservable_bus_admits_no_copy_and_counts_nothing(tmp_path, bus_kind):
    class Raising(_CaptureQueue):
        def put_unless_outstanding(self, evt):
            raise OSError("proxy connection lost")

    bus = stdqueue.Queue() if bus_kind == "plain_queue" else Raising()
    event = _owed(tmp_path)
    _age(tmp_path)
    assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == []
    assert _attempts(tmp_path, event["delivery_id"]) == 0
    assert (bus.qsize() if bus_kind == "plain_queue" else len(bus.events)) == 0
    assert [row["delivery_id"] for row in td.pending_deliveries(tmp_path)] == [event["delivery_id"]], "still owed"


def test_a_receipt_that_landed_after_the_rows_were_read_admits_no_copy(tmp_path, monkeypatch):
    bus = _CaptureQueue()
    event = _owed(tmp_path)
    _age(tmp_path)
    actual = td._replay_due

    def deliver_then_answer(row, **kwargs):
        td.register_delivery(tmp_path, event["delivery_id"], emitted=event)  # the original lands now
        return actual(row, **kwargs)

    monkeypatch.setattr(td, "_replay_due", deliver_then_answer)
    assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == []
    assert bus.events == [] and td.pending_deliveries(tmp_path) == []


def test_exhaustion_after_the_cap_is_a_registry_transition_not_a_sixth_copy(tmp_path, monkeypatch):
    notices: list = []
    monkeypatch.setattr("supervisor.message_bus.send_with_budget",
                        lambda chat_id, text, **kw: notices.append((chat_id, text)))
    bus = _CaptureQueue()
    event = _owed(tmp_path, "root-cap", "never lands")
    for _ in range(td._PENDING_MAX_REPLAYS):
        _age(tmp_path)
        assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == [event["delivery_id"]]
        assert td.record_replay_dispatch(tmp_path, event["delivery_id"])
        bus.release_all()
    assert notices == []
    _age(tmp_path)
    assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == []
    assert len(bus.events) == td._PENDING_MAX_REPLAYS
    assert td.pending_deliveries(tmp_path) == []
    receipts = td.terminal_answer_receipts(tmp_path, "root-cap")
    assert receipts["state"] == "exhausted"
    (mark,) = receipts["exhausted"]
    assert mark["attempts"] == td._PENDING_MAX_REPLAYS and mark["reason"] == "replay_exhausted"
    assert pathlib.Path(mark["preserved_path"]).read_text(encoding="utf-8") == "never lands"
    (notice,) = notices
    assert notice[0] == 1 and "was not confirmed" in notice[1]
    # Exhausted rows are never replayed again; a later real receipt of the id clears the mark.
    _age(tmp_path)
    assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == []
    assert td.register_delivery(tmp_path, event["delivery_id"], emitted=event)
    assert td.terminal_answer_receipts(tmp_path, "root-cap")["state"] == "delivered"


def test_the_inflight_replay_set_is_gone():
    assert not hasattr(td, "_INFLIGHT_REPLAY_IDS") and not hasattr(td, "clear_inflight_replay")
