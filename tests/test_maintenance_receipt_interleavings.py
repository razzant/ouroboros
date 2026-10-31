"""Independent maintenance duties meet at real task, outbox and receipt owners.

Events below stop an actual publication between its durable stages. Their wait
timeouts only fail a wedged test; no elapsed-time assumption orders the race.
The sender remains serial, as on the supervisor event thread. External sends
still have their documented at-least-once send/register crash gap.
"""
from __future__ import annotations

from tests import _dialogue as dialogue_test

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from queue import Empty
from threading import Event
from types import SimpleNamespace

import pytest

from ouroboros import cancel_intents, notice_receipts, obligations, pause_notices
from ouroboros import terminal_cost_reconciliation, usage_accounting as usage, usage_store
from ouroboros.budget_pause import set_budget_pause
from ouroboros.startup_migrations import prepare_startup_state
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.utils import append_jsonl
from supervisor import cancel_publication as cancel
from tests._cancel_intents_shared import DeliveryBus
from supervisor import events_chat_delivery, message_bus, queue, terminal_delivery as delivery, workers

pytestmark = pytest.mark.serial


@pytest.fixture
def records(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    sent = []
    bridge = SimpleNamespace(
        send_message=lambda chat, text, **kw: sent.append((chat, text, kw)) or (True, ""),
        push_log=lambda *_: None,
    )
    monkeypatch.setattr(message_bus, "DATA_DIR", root)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 1})
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "try_get_bridge", lambda: bridge)
    monkeypatch.setattr(events_chat_delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    monkeypatch.setattr(queue, "DRIVE_ROOT", root)
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(workers, "WORKERS", {})
    prepare_startup_state(root)
    task = {"id": "task", "chat_id": 9, "root_task_id": "task", "delegation_role": "root"}
    write_task_result(root, "task", "running", **{k: v for k, v in task.items() if k != "id"})
    ctx = SimpleNamespace(DRIVE_ROOT=root, RUNNING={}, append_jsonl=append_jsonl,
                          send_with_budget=message_bus.send_with_budget)
    return SimpleNamespace(root=root, task=task, sent=sent, ctx=ctx)


def _chat_rows(root):
    path = root / "logs" / "chat.jsonl"
    return dialogue_test.log_rows(path)


def _cancelled(records):
    root = records.root
    requested = cancel_intents.request_cancel(root, "task", reason="owner stopped this task", source="http_single")
    claim = cancel_intents.claim_intent(root, "task", owner="maintenance-test")
    assert claim and not claim.get("claim_refused")
    assert claim["request_id"] == requested["request_id"]
    stored = write_task_result(root, "task", "cancelled", **cancel._cancel_result_fields(
        records.task, existing=load_task_result(root, "task"), result="retained intermediate result",
        **cancel._intent_outcome_fields(claim),
    ))
    return claim, stored


def _settle(records, claim):
    cancel._settle_or_reopen_intent(records.ctx, "task", owed_ok=True, intent=claim,
                                  outcome="cancelled", detail="terminal publication owned")
    assert cancel_intents.active_intent(records.root, "task") is None


def _publish(records, event_queue):
    # The same delivery owner called by cancel_publication._publish_cancelled_task.
    return delivery.deliver_unreviewed_salvage(
        records.root, records.task, "task", outcome="stopped", settled_status="cancelled",
        salvaged_text="retained intermediate result", event_queue=event_queue,
    )


def _drain(event_queue, records):
    events = []
    while True:
        try:
            event = event_queue.get_nowait()
        except Empty:
            return events
        events.append(event)
        events_chat_delivery._handle_send_message(event, records.ctx)
        event_queue.release_delivery(str(event.get("delivery_id") or ""))  # the consumer's finally


def _replay_clock(monkeypatch, owed):
    """Drive the existing physical due predicate on both sides of its boundary."""
    stamp = datetime.fromisoformat(owed["registered_at"].replace("Z", "+00:00")).timestamp()
    clock = [stamp + delivery._REPLAY_MIN_AGE_SEC - 1]
    actual = delivery._replay_due
    monkeypatch.setattr(delivery, "_replay_due", lambda row: actual(row, now=clock[0]))
    return clock


@pytest.mark.parametrize("send_replay_first, receipt_available", [
    (False, True), (True, True), (True, False),
], ids=["queued-copy", "delivered-copy", "delivered-with-receipt-deferred"])
def test_cancel_publication_and_replay_keep_one_episode(
    records, monkeypatch, send_replay_first, receipt_available,
):
    claim, stored = _cancelled(records)
    patch_document = delivery.patch_document
    if not receipt_available:
        # The outcome retains the episode while the exact sent fact stays owed
        # until the destination receipt can be recorded off the consumer.
        def unavailable(*_args, **_kwargs):
            raise OSError("cancel receipt storage unavailable")
        monkeypatch.setattr(delivery, "patch_document", unavailable)
    registered, resume = Event(), Event()
    events = DeliveryBus()

    def cancel_owner():
        assert cancel._register_owed_terminal_delivery(
            records.ctx, records.task, "task", stored, deliver=True,
            salvage_text="retained intermediate result",
        )
        registered.set()
        assert resume.wait(10), "test did not release cancel publication"
        _settle(records, claim)
        return _publish(records, events)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(cancel_owner)
        try:
            assert registered.wait(10), "cancel never registered its owed answer"
            (owed,) = delivery.pending_deliveries(records.root)
            expected_id = f"cancel:task:{claim['request_id']}"
            assert owed["delivery_id"] == expected_id
            assert cancel_intents.active_intent(records.root, "task")["state"] == "claimed"
            clock = _replay_clock(monkeypatch, owed)
            assert delivery.replay_pending_deliveries(records.root, event_queue=events) == []
            assert events.empty() and delivery.pending_deliveries(records.root)
            clock[0] += 2
            assert delivery.replay_pending_deliveries(records.root, event_queue=events) == [expected_id]
            observed = _drain(events, records) if send_replay_first else []
            if send_replay_first:
                assert len(records.sent) == 1
                if receipt_available:
                    assert delivery.pending_deliveries(records.root) == []
                    assert load_task_result(records.root, "task")["cancel_receipt"]["delivered_chat_id"] == 9
                else:
                    assert delivery.pending_deliveries(records.root)[0]["emitted"]["chat_id"] == 9
        finally:
            resume.set()
        future.result(timeout=10)

    observed.extend(_drain(events, records))
    assert {event["delivery_id"] for event in observed} == {expected_id}
    assert all((event["chat_id"], event["text"]) == (owed["chat_id"], owed["text"]) for event in observed)
    assert len(records.sent) == len(_chat_rows(records.root)) == 1
    assert records.sent[0][:2] == (9, owed["text"])
    if not receipt_available:
        assert not delivery.already_delivered(records.root, expected_id)
        assert load_task_result(records.root, "task").get("cancel_receipt") is None
        monkeypatch.setattr(delivery, "patch_document", patch_document)
        assert delivery.replay_pending_deliveries(records.root, event_queue=events) == [expected_id]
        assert len(records.sent) == len(_chat_rows(records.root)) == 1 and events.empty()
    assert delivery.pending_deliveries(records.root) == []
    assert delivery.already_delivered(records.root, expected_id)
    receipt = load_task_result(records.root, "task").get("cancel_receipt")
    if receipt_available:
        assert receipt["delivery_id"] == expected_id and receipt["delivered_chat_id"] == 9
    else:
        assert receipt is None  # Recovery must not invent an absent original detail block.


def test_replay_rechecks_owed_row_after_cancel_delivery(records, monkeypatch):
    claim, stored = _cancelled(records)
    assert cancel._register_owed_terminal_delivery(
        records.ctx, records.task, "task", stored, deliver=True, salvage_text="retained intermediate result",
    )
    (owed,) = delivery.pending_deliveries(records.root)
    clock = _replay_clock(monkeypatch, owed)
    clock[0] += 2
    selected, resume = Event(), Event()
    selection = delivery._replay_due  # the clock-aware selection read installed above

    def select_then_wait(row):
        due = selection(row)
        if due and not selected.is_set():
            selected.set()
            assert resume.wait(10), "test did not release replay claim"
        return due

    monkeypatch.setattr(delivery, "_replay_due", select_then_wait)
    replay_queue, publish_queue = DeliveryBus(), DeliveryBus()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(delivery.replay_pending_deliveries, records.root, event_queue=replay_queue)
        try:
            assert selected.wait(10), "replay never read its owed row"
            assert _publish(records, publish_queue)
            assert len(_drain(publish_queue, records)) == 1
            assert len(records.sent) == 1 and delivery.pending_deliveries(records.root) == []
        finally:
            resume.set()
        assert future.result(timeout=10) == []
    assert replay_queue.empty()
    assert delivery.register_pending_delivery(records.root, owed)
    assert delivery.pending_deliveries(records.root) == []
    _settle(records, claim)
    assert load_task_result(records.root, "task")["cancel_receipt"]["delivery_id"] == owed["delivery_id"]


@pytest.mark.parametrize("status", ["completed", "cancelled"])
def test_terminal_delivery_without_stop_keeps_ordinary_final_identity(records, status):
    stored = write_task_result(records.root, "task", status, result="ordinary terminal text")
    events = DeliveryBus()

    def publish():
        if status == "completed":
            return delivery.deliver_completed_result(records.root, records.task, "task", stored, event_queue=events)
        return delivery.deliver_unreviewed_salvage(
            records.root, records.task, "task", outcome="stopped", settled_status="cancelled",
            salvaged_text=stored["result"], event_queue=events,
        )

    assert publish()
    (first,) = _drain(events, records)
    assert first["delivery_id"].startswith("final:task:")
    assert delivery.already_delivered(records.root, first["delivery_id"])
    assert delivery.pending_deliveries(records.root) == []
    assert publish() is False
    assert events.empty() and len(records.sent) == len(_chat_rows(records.root)) == 1
    (receipt,) = delivery.terminal_answer_receipts(records.root, "task")["delivered"]
    assert receipt["delivery_id"] == first["delivery_id"] and receipt["chat_id"] == 9


def test_pause_notice_ack_preserves_concurrent_cancel_cost_and_new_notice(records, monkeypatch):
    root = records.root
    reservation = usage.reserve_attempt(usage.AttemptRequest(
        model="test", provider="openai", drive_root=root, task_id="task", root_task_id="task",
        reservation_usd=1.0, global_limit_usd=100.0,
    ))
    usage.mark_dispatched(reservation)
    set_budget_pause(root, "task", {"pause_id": "first", "state": "paused", "reason": "budget"})
    sent, resume = Event(), Event()
    actual_ack = pause_notices._acknowledge

    def before_ack(*args):
        sent.set()
        assert resume.wait(10), "test did not release notice acknowledgement"
        return actual_ack(*args)

    monkeypatch.setattr(pause_notices, "_acknowledge", before_ack)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(pause_notices.reconcile_pause_notices, root)
        try:
            assert sent.wait(10), "notice did not reach its local receipt"
            old_id, new_id = "pause:task:first", "pause:task:second"
            receipt_id = notice_receipts.pause_receipt_id(9, old_id)
            assert [row["card_row_id"] for row in _chat_rows(root)] == [old_id]
            assert receipt_id in obligations.members(root, "pause_notice_receipts")
            set_budget_pause(root, "task", {"pause_id": "second", "state": "paused", "reason": "budget"})
            current = load_task_result(root, "task")
            write_task_result(root, "task", "cancelled", **cancel._cancel_result_fields(
                records.task, existing=current, result="cancelled with retained work",
                cancel_origin={"source": "http_single", "reason": "owner stopped this task"},
            ))
            usage.settle_attempt(reservation, {"prompt_tokens": 7}, cost_usd=0.4, cost_final=True)
            terminal_cost_reconciliation.reconcile_abandoned_usage(root)
            updated = load_task_result(root, "task")
            assert updated["accounted_upper_bound_usd"] == 0.4 and updated["cost_final"] is True
            assert set(updated["pause_notices"]) == {old_id, new_id}
            with usage_store.read(root) as txn:
                assert txn.dirty_owners() == []
        finally:
            resume.set()
        future.result(timeout=10)

    after_ack = load_task_result(root, "task")
    assert set(after_ack["pause_notices"]) == {new_id}
    assert after_ack["pause_notices"][new_id] == updated["pause_notices"][new_id]
    for field in ("status", "result", "root_task_id", "delegation_role", "budget_pause",
                  "cancel_origin", "accounted_upper_bound_usd", "cost_final"):
        assert after_ack[field] == updated[field]
    assert set(obligations.members(root, "pause_notices")) == {"task"}
    assert receipt_id not in obligations.members(root, "pause_notice_receipts")
    pause_notices.reconcile_pause_notices(root)
    pause_notices.reconcile_pause_notices(root)
    assert [row["card_row_id"] for row in _chat_rows(root)] == [old_id, new_id]
    assert len(records.sent) == 2
    assert not obligations.members(root, "pause_notices")
    assert not obligations.members(root, "pause_notice_receipts")
    assert load_task_result(root, "task")["accounted_upper_bound_usd"] == 0.4
