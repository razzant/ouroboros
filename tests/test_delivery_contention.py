"""Required sends retain exact debt and physical ownership across short store waits."""
import threading
from contextlib import contextmanager
import time
from collections import deque
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_retry, server_liveness, store, utils
from supervisor import events, events_chat_delivery as delivery, message_bus, terminal_delivery as td
from supervisor.event_observation import ObservedEventQueue
from tests._dialogue import chat_rows
from tests.test_dialogue_retry import seed, pinned, bridge, isolated_retry as _isolated_retry
from tests.test_main_terminal_delivery import owed as _main_owed
from tests.test_supervisor_loop_offthread import quiet_tick as _quiet_tick

isolated_retry = _isolated_retry
base_main_owed, quiet_tick = _main_owed, _quiet_tick


@pytest.fixture
def main_owed(monkeypatch, request):
    monkeypatch.setattr(store, "_journal_mode", lambda *_: "PERSIST")
    return request.getfixturevalue("base_main_owed")


@pytest.fixture
def sender(tmp_path, monkeypatch):
    frames = bridge(tmp_path, monkeypatch)
    bus = ObservedEventQueue()
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    monkeypatch.setattr(delivery, "_ACTIVE_DELIVERIES", set())
    monkeypatch.setattr(delivery, "_DEFERRED_DELIVERIES", set())
    monkeypatch.setattr("supervisor.workers.get_event_q", lambda **_kw: bus)
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, EVENT_Q=bus, append_jsonl=utils.append_jsonl,
                          send_with_budget=message_bus.send_with_budget)
    return ctx, frames, bus


def owed(root, text="answer", tid="turn"):
    event = dict(type="send_message", chat_id=1, task_id=tid, text=text, delivery_id=td.delivery_id_for(tid, text))
    assert td.register_pending_delivery(root, event)
    return event


def consume(ctx, bus, monkeypatch):
    controls = []
    monkeypatch.setitem(events.EVENT_HANDLERS, "control_probe", lambda *_: controls.append(time.monotonic()))
    bus.put({"type": "control_probe"})
    start = time.monotonic()
    server_liveness.drain_worker_events(bus, ctx, [start, {}, time.thread_time(), None], on_restart=lambda *_: None)
    assert controls and bus.empty(), "unrelated control remains in the same bounded pass"
    return controls[0] - start


@pytest.mark.parametrize("mode", ["PERSIST", "WAL"])
def test_required_append_returns_to_control_and_duty_finishes_off_consumer(tmp_path, monkeypatch, sender, mode):
    seed(tmp_path, monkeypatch, mode)
    ctx, frames, bus = sender
    event = owed(tmp_path)
    bus.put(event)
    with pinned(tmp_path):
        elapsed = consume(ctx, bus, monkeypatch)
        assert elapsed < 1.25  # a few acquisition slices including cold routing/receipt setup
        if mode == "PERSIST":
            assert frames == [] and td.pending_deliveries(tmp_path)[0]["delivery_deferred"]
            assert .2 <= elapsed
            assert dialogue_retry.snapshot()["rows"] == 0
        else:
            assert len(frames) == 1 and td.pending_deliveries(tmp_path) == []
    consumer = threading.get_ident()
    calls = []
    actual = message_bus.send_with_budget

    def send(*args, **kwargs):
        calls.append(threading.get_ident())
        return actual(*args, **kwargs)

    monkeypatch.setattr(message_bus, "send_with_budget", send)
    results = []
    thread = threading.Thread(target=lambda: results.extend(td.replay_pending_deliveries(tmp_path, event_queue=bus)))
    thread.start()
    thread.join(5)
    assert not thread.is_alive()
    assert calls == ([] if mode == "WAL" else [thread.ident])
    assert consumer not in calls and bus.empty()
    assert len(frames) == 1 and td.pending_deliveries(tmp_path) == []
    assert sum(row.get("delivery_id") == event["delivery_id"] for row in chat_rows(tmp_path)) == 1
    assert td.terminal_answer_receipts(tmp_path, "turn")["state"] == "delivered"


def test_registry_lock_keeps_sent_debt_and_replay_only_registers_receipt(tmp_path, monkeypatch, sender):
    seed(tmp_path, monkeypatch)
    ctx, frames, bus = sender
    event = owed(tmp_path)
    bus.put(event)
    with store.name_lock(tmp_path, "terminal_deliveries.json.lock", timeout_sec=0):
        assert consume(ctx, bus, monkeypatch) < 1.25
        assert len(frames) == 1 and td.pending_deliveries(tmp_path)
        assert not td.already_delivered(tmp_path, event["delivery_id"])
    # The durable marker could not be acquired: the existing memory dedupe still
    # knows the send returned, while the outbox keeps the authoritative text.
    assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == [event["delivery_id"]]
    assert len(frames) == 1 and td.pending_deliveries(tmp_path) == [] and bus.empty()
    assert td.terminal_answer_receipts(tmp_path, "turn")["state"] == "delivered"


def test_capture_wait_uses_the_slice_and_a_restart_replays_only_its_receipt(tmp_path, monkeypatch, sender):
    seed(tmp_path, monkeypatch)
    ctx, frames, bus = sender
    event = owed(tmp_path)
    bus.put(event)
    capture = td._capture_emitted_answer
    budgets = []

    def budgeted_capture(*args, **kwargs):
        # Lane E owns the evidence index lock's budget consumption. Exercise C's
        # scope with an actual held lock using that same acquisition contract.
        budget = store._WAIT_BUDGET.get()
        budgets.append(budget)
        with store.name_lock(tmp_path, "packs_index.lock", timeout_sec=budget):
            return capture(*args, **kwargs)

    monkeypatch.setattr(td, "_capture_emitted_answer", budgeted_capture)
    with store.name_lock(tmp_path, "packs_index.lock", timeout_sec=0):
        assert consume(ctx, bus, monkeypatch) < 1.25
    assert budgets == [.25] and len(frames) == 1
    sent = td.pending_deliveries(tmp_path)[0]["emitted"]
    assert sent["text"] == "answer"
    delivery._DELIVERED_MESSAGE_IDS.clear()
    delivery._DEFERRED_DELIVERIES.clear()
    assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == [event["delivery_id"]]
    assert len(frames) == 1 and td.pending_deliveries(tmp_path) == []
    assert td.terminal_answer_receipts(tmp_path, "turn")["delivered"][0]["sent_at"] == sent["sent_at"]


def test_inflight_physical_send_cannot_be_duplicated_by_replay_or_handler(tmp_path, monkeypatch, sender):
    seed(tmp_path, monkeypatch)
    ctx, frames, bus = sender
    event = owed(tmp_path)
    with pinned(tmp_path):
        assert not delivery._handle_send_message(event, ctx)
    entered, release = threading.Event(), threading.Event()
    actual = message_bus.send_with_budget

    def slow_send(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return actual(*args, **kwargs)

    monkeypatch.setattr(message_bus, "send_with_budget", slow_send)
    thread = threading.Thread(target=lambda: td.replay_pending_deliveries(tmp_path, event_queue=bus))
    thread.start()
    try:
        assert entered.wait(3)
        assert delivery._delivery_key(tmp_path, event) in delivery._ACTIVE_DELIVERIES
        assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == []
        assert not delivery._handle_send_message(event, ctx)
        assert not frames and bus.empty()
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and len(frames) == 1
    assert not delivery._ACTIVE_DELIVERIES and td.pending_deliveries(tmp_path) == []


def test_queued_original_is_skipped_by_direct_continuation(tmp_path, monkeypatch, sender):
    seed(tmp_path, monkeypatch)
    ctx, frames, bus = sender
    event = owed(tmp_path)
    delivery._defer_delivery(ctx, event)
    bus.put(event)
    assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == []
    assert not frames and bus.qsize() == 1
    consume(ctx, bus, monkeypatch)
    assert len(frames) == 1 and td.pending_deliveries(tmp_path) == []


@pytest.mark.parametrize("role,kind,narration", [("system", "reminder", False), ("assistant", "", True)])
def test_write_required_notice_without_an_id_stays_with_its_producer(tmp_path, monkeypatch, sender, role, kind, narration):
    seed(tmp_path, monkeypatch)
    _ctx, frames, bus = sender

    def send():
        message_bus.send_with_budget(1, "Required notice", ts="notice", require_write=True,
                                     role=role, system_type=kind, narration=narration)

    with pinned(tmp_path), store.acquisition_budget(0.2), pytest.raises(RuntimeError, match="could not be persisted"):
        send()  # the producer learns the row is unwritten and keeps its own retry
    assert td.pending_deliveries(tmp_path) == [] and dialogue_retry.snapshot()["rows"] == 0
    assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == [] and not frames
    held = threading.Event()
    reader = threading.Thread(target=_hold_reader, args=(tmp_path, held, 0.6))
    reader.start()
    try:
        assert held.wait(5)
        send()  # its retry keeps the caller's own wait: a reader longer than the slice is waited out
    finally:
        reader.join()
    assert len(frames) == 1 and len(chat_rows(tmp_path)) == 2 and bus.empty()
    assert frames[0]["role"] == role and frames[0].get("system_type", "") == kind
    assert frames[0]["narration"] is narration
    assert chat_rows(tmp_path)[-1]["direction"] == ("system" if role == "system" else "out")


def _hold_reader(root, held, seconds):
    from ouroboros import dialogue_store

    with dialogue_store.transaction(root, write=False):  # the store's own reader, on either lock tier
        held.set()
        time.sleep(seconds)


@pytest.mark.parametrize("failed_receipt", [(False, "offline"), None])
def test_physical_failure_retains_existing_backoff_and_spends_no_busy_retries(tmp_path, monkeypatch, sender, failed_receipt):
    from tests.test_delivery_replay_membership import _age

    seed(tmp_path, monkeypatch)
    ctx, frames, bus = sender
    event = owed(tmp_path)
    with pinned(tmp_path):
        delivery._handle_send_message(event, ctx)
        td.replay_pending_deliveries(tmp_path, event_queue=bus)
    assert td.pending_deliveries(tmp_path)[0]["dispatch_attempts"] == 0
    transport = message_bus.get_bridge()
    actual = transport.send_message
    sends = []
    monkeypatch.setattr(transport, "send_message", lambda *_a, **_k: sends.append(True) or failed_receipt)
    td.replay_pending_deliveries(tmp_path, event_queue=bus)
    assert len(sends) == 1 and td.pending_deliveries(tmp_path)[0]["dispatch_attempts"] == 1
    for _ in range(3):
        td.replay_pending_deliveries(tmp_path, event_queue=bus)
    assert len(sends) == 1
    _age(tmp_path)
    monkeypatch.setattr(transport, "send_message", actual)
    assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == [event["delivery_id"]]
    assert len(frames) == 1 and len(chat_rows(tmp_path)) == 2


def test_required_uncertain_commit_does_not_duplicate_its_durable_row(tmp_path, monkeypatch, sender):
    from ouroboros import dialogue_store as ds

    seed(tmp_path, monkeypatch)
    ctx, frames, bus = sender
    event = owed(tmp_path)
    original = ds.transaction
    failed = []

    @contextmanager
    def uncertain(*args, write, **kwargs):
        with original(*args, write=write, **kwargs) as conn:
            yield conn
        if write and not failed:
            failed.append(True)
            raise store.StoreBusy("uncertain after successful commit")

    monkeypatch.setattr(ds, "transaction", uncertain)
    assert not delivery._handle_send_message(event, ctx)
    assert not frames and len(chat_rows(tmp_path)) == 2
    assert td.replay_pending_deliveries(tmp_path, event_queue=bus) == [event["delivery_id"]]
    assert len(frames) == 1 and len(chat_rows(tmp_path)) == 2


def test_main_ack_wait_keeps_sent_obligation_until_background_recovery(main_owed, monkeypatch, sender):
    import sqlite3
    from contextlib import closing
    from ouroboros.task_results import load_task_result

    ctx, frames, bus = sender
    event = main_owed.event
    root = main_owed.root
    with closing(sqlite3.connect(store.store_path(root), isolation_level=None)) as reader:
        reader.execute("BEGIN")
        reader.execute("SELECT * FROM tasks LIMIT 1").fetchall()
        bus.put(event)
        assert consume(ctx, bus, monkeypatch) < 1.25
        assert len(frames) == 1
        assert td.pending_deliveries(root)[0]["emitted"]["terminal_projection_token"] == event["terminal_projection_token"]
        reader.rollback()
    assert load_task_result(root, "root")["canonical_terminal_projection_ready"]
    assert td.replay_pending_deliveries(root, event_queue=bus) == [event["delivery_id"]]
    assert len(frames) == 1 and td.pending_deliveries(root) == []
    assert load_task_result(root, "root")["canonical_terminal_projection"]["main_disposition"] == "delivered"


def test_buffer_wakes_existing_single_flight_duty_before_regular_cadence(quiet_tick, monkeypatch):
    sm = quiet_tick
    monkeypatch.setattr(sm, "_DELIVERY_SWEEP_LOCK", threading.Lock())
    monkeypatch.setattr(sm, "_DELIVERY_WAKE", threading.Event())
    starts = []

    def start(latch, name, target, args):
        if name == "terminal_delivery_replay":
            starts.append((target, args))
        else:
            latch.release()
        return True

    monkeypatch.setattr(sm, "_start_maintenance_thread", start)
    now = time.time()
    sm._periodic_supervisor_maintenance([now], [now])
    assert starts == []
    dialogue_retry.retain(sm.DATA_DIR, "progress", {"ts": "wake", "chat_id": 1})
    sm._periodic_supervisor_maintenance([now], [now])
    assert len(starts) == 1
    sm._periodic_supervisor_maintenance([now], [now])
    assert len(starts) == 1, "an active existing duty is not replaced"
    target, args = starts[0]
    calls = []
    monkeypatch.setattr(dialogue_retry, "drain", lambda: calls.append("buffer"))
    monkeypatch.setattr(td, "replay_pending_deliveries", lambda *_: calls.append("required"))
    target(*args)
    assert calls == ["buffer", "required"] and not sm._DELIVERY_SWEEP_LOCK.locked()


@pytest.mark.parametrize("old_receipt_blocked", [False, True])
def test_main_new_token_cannot_consume_an_old_sent_fact_or_delivered_id(main_owed, monkeypatch, sender, old_receipt_blocked):
    from copy import deepcopy
    from ouroboros import terminal_projection as projection
    from ouroboros.task_results import load_task_result, write_task_result

    ctx, frames, bus = sender
    root, event = main_owed.root, main_owed.event
    previous = load_task_result(root, "root")
    with monkeypatch.context() as patch:
        if old_receipt_blocked:
            def busy(*_args, **_kwargs):
                raise store.StoreBusy("ack held")
            patch.setattr(projection, "acknowledge_main_delivery", busy)
        delivery._handle_send_message(event, ctx)
    assert len(frames) == 1
    next_event = {**event, "text": "A newer outcome", "terminal_projection_token": "next-publication"}
    ready = deepcopy(previous["canonical_terminal_projection_ready"])
    ready["token"] = next_event["terminal_projection_token"]
    ready["main_delivery"]["event"] = next_event
    marker = {**previous["canonical_terminal_projection"], "token": ready["token"]}
    write_task_result(root, "root", "completed", canonical_terminal_projection_ready=ready,
                      canonical_terminal_projection=marker)
    assert td.register_pending_delivery(root, next_event)
    assert td.pending_deliveries(root)[0]["terminal_projection_token"] == ready["token"]
    # An older physical send may return after the new publication became owed.
    old_sent = {**event, **delivery._DELIVERED_MESSAGE_IDS[-1][1]}
    assert td.register_pending_delivery(root, {**event, "delivery_deferred": True, "emitted": old_sent})
    assert delivery._register_delivered(ctx, event["delivery_id"], emitted=old_sent)
    assert td.pending_deliveries(root)[0]["terminal_projection_token"] == ready["token"]
    with pinned(root):
        assert not delivery._handle_send_message(next_event, ctx)
    assert len(frames) == 1
    assert td.replay_pending_deliveries(root, event_queue=bus) == [event["delivery_id"]]
    assert len(frames) == 2 and frames[-1]["content"] == "A newer outcome"
    assert td.pending_deliveries(root) == []
