"""Replay accounting through the outbox, real bus consumer and message sender.

Only the transport is replaced. Storage, routing, receipt capture and replay run
on private roots; interrupted sends deliberately leave their outcome unknown.
"""
from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import server_liveness, store
from supervisor import events_chat_delivery as delivery, message_bus, terminal_delivery as td
from supervisor.event_observation import ObservedEventQueue
from tests._dialogue import chat_rows, progress_rows
from tests.test_delivery_contention import sender as _sender
from tests.test_delivery_replay_membership import _age
from tests.test_dialogue_retry import isolated_retry as _isolated_retry, pinned, seed
from tests.test_main_terminal_delivery import owed as _main_owed, _due as _main_due

sender, isolated_retry = _sender, _isolated_retry
main_owed = _main_owed


class InterruptedSend(BaseException):
    """The transport entered but never returned a receipt."""


@pytest.fixture
def dispatch(tmp_path, monkeypatch, sender, isolated_retry):
    seed(tmp_path, monkeypatch)
    ctx, _frames, bus = sender
    trace = SimpleNamespace(ctx=ctx, bus=bus, calls=[], notices=[], mode="ok")

    def send(chat_id, text, **kwargs):
        did = (kwargs.get("progress_meta") or {}).get("delivery_id")
        if not did:
            trace.notices.append((chat_id, text))
            return True, "ok"
        row = next(row for row in td.pending_deliveries(tmp_path) if row["delivery_id"] == did)
        history = progress_rows(tmp_path) if kwargs.get("is_progress") else chat_rows(tmp_path)
        assert any(saved.get("delivery_id") == did for saved in history), "history precedes transport entry"
        trace.calls.append((chat_id, text, row.get("dispatch_attempts")))
        if trace.mode == "interrupted":
            raise InterruptedSend()
        return (False, "offline") if trace.mode == "refused" else (True, "ok")

    monkeypatch.setattr(message_bus.get_bridge(), "send_message", send)
    return trace


def owe(root, **fields):
    event = {"type": "send_message", "chat_id": 1, "task_id": "turn", "text": "Exact answer", **fields}
    event.setdefault("delivery_id", td.delivery_id_for(event["task_id"], event["text"]))
    assert td.register_pending_delivery(root, event)
    return event


def drain(trace):
    server_liveness.drain_worker_events(trace.bus, trace.ctx, [time.monotonic(), {}, time.thread_time(), None],
                                       on_restart=lambda *_: pytest.fail("no restart event"))


def sweep(root, trace):
    return td.replay_pending_deliveries(root, event_queue=trace.bus)


def change_row(root, did, **changes):
    path = root / "state" / "terminal_deliveries.json"
    data = json.loads(path.read_text())
    data["pending"][did].update(changes)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize("fmt", ["", "markdown"])
def test_losing_more_than_cap_queued_generations_still_delivers_once(tmp_path, dispatch, fmt):
    event = owe(tmp_path, format=fmt)
    for _ in range(td._PENDING_MAX_REPLAYS + 2):
        _age(tmp_path)
        assert sweep(tmp_path, dispatch) == [event["delivery_id"]]
        assert dispatch.bus.qsize() == 1 and not dispatch.calls
        # A fresh process loses bus membership, retaining only the durable outbox.
        dispatch.bus = ObservedEventQueue()
        dispatch.ctx.EVENT_Q = dispatch.bus
    _age(tmp_path)
    assert sweep(tmp_path, dispatch) == [event["delivery_id"]]
    drain(dispatch)
    assert dispatch.calls == [(1, event["text"], 1)]
    assert td.already_delivered(tmp_path, event["delivery_id"])
    assert not td.pending_deliveries(tmp_path) and not dispatch.notices


@pytest.mark.parametrize("progress", [False, True])
def test_blocked_required_history_consumes_no_replay_entry(tmp_path, dispatch, progress):
    event = owe(tmp_path, is_progress=progress)
    _age(tmp_path)
    assert sweep(tmp_path, dispatch) == [event["delivery_id"]]
    with pinned(tmp_path):
        drain(dispatch)
        for _ in range(td._PENDING_MAX_REPLAYS + 1):
            sweep(tmp_path, dispatch)
    assert dispatch.calls == []
    (row,) = td.pending_deliveries(tmp_path)
    assert row.get("dispatch_attempts", 0) == 0 and row["delivery_deferred"]
    assert sweep(tmp_path, dispatch) == [event["delivery_id"]]
    assert dispatch.calls == [(1, event["text"], 1)]
    assert td.already_delivered(tmp_path, event["delivery_id"])


@pytest.mark.parametrize("mode", ["refused", "interrupted"])
@pytest.mark.parametrize("deferred", [False, True])
def test_every_transport_entry_is_bounded_even_without_a_return(tmp_path, dispatch, mode, deferred):
    event = owe(tmp_path, **({"delivery_deferred": True} if deferred else {}))
    dispatch.mode = mode
    for number in range(1, td._PENDING_MAX_REPLAYS + 1):
        _age(tmp_path)
        if mode == "interrupted":
            with pytest.raises(InterruptedSend):
                sweep(tmp_path, dispatch)
                drain(dispatch)
        else:
            sweep(tmp_path, dispatch)
            drain(dispatch)
        assert dispatch.calls[-1] == (1, event["text"], number)
        assert len(dispatch.calls) == number
        assert sweep(tmp_path, dispatch) == [], "unknown outcomes also keep the retry backoff"
    _age(tmp_path)
    assert sweep(tmp_path, dispatch) == []
    assert not td.pending_deliveries(tmp_path)
    (exhausted,) = td.terminal_answer_receipts(tmp_path, "turn")["exhausted"]
    assert exhausted["dispatch_attempts"] == exhausted["attempts"] == td._PENDING_MAX_REPLAYS
    assert exhausted["text"] == event["text"]
    assert len(dispatch.calls) == td._PENDING_MAX_REPLAYS
    assert dispatch.notices and "recorded replay dispatch entries" in dispatch.notices[0][1]


def test_failed_counter_write_never_enters_transport_then_recovers(tmp_path, dispatch):
    event = owe(tmp_path)
    _age(tmp_path)
    assert sweep(tmp_path, dispatch) == [event["delivery_id"]]
    with store.name_lock(tmp_path, "terminal_deliveries.json.lock", timeout_sec=0):
        drain(dispatch)
    assert dispatch.calls == []
    assert td.pending_deliveries(tmp_path)[0].get("dispatch_attempts", 0) == 0
    assert sweep(tmp_path, dispatch) == [event["delivery_id"]]
    assert dispatch.calls == [(1, event["text"], 1)]
    assert td.already_delivered(tmp_path, event["delivery_id"])


@pytest.mark.parametrize("restart", [False, True])
def test_sent_receipt_repair_at_cap_never_resends_or_spends(tmp_path, monkeypatch, dispatch, restart):
    event = owe(tmp_path, delivery_deferred=True)
    change_row(tmp_path, event["delivery_id"], dispatch_attempts=td._PENDING_MAX_REPLAYS - 1)
    _age(tmp_path)
    capture = td._capture_emitted_answer
    monkeypatch.setattr(td, "_capture_emitted_answer", lambda *_: None)
    assert sweep(tmp_path, dispatch) == []
    assert dispatch.calls == [(1, event["text"], td._PENDING_MAX_REPLAYS)]
    (row,) = td.pending_deliveries(tmp_path)
    assert row["emitted"]["text"] == event["text"]
    if restart:
        delivery._DELIVERED_MESSAGE_IDS.clear()
        delivery._DEFERRED_DELIVERIES.clear()
    monkeypatch.setattr(td, "_capture_emitted_answer", capture)
    assert sweep(tmp_path, dispatch) == [event["delivery_id"]]
    assert len(dispatch.calls) == 1 and not dispatch.notices
    assert td.terminal_answer_receipts(tmp_path, "turn")["state"] == "delivered"


def test_memory_send_fact_repairs_receipt_at_cap_after_emitted_write_failed(tmp_path, monkeypatch, dispatch):
    event = owe(tmp_path, delivery_deferred=True)
    change_row(tmp_path, event["delivery_id"], dispatch_attempts=td._PENDING_MAX_REPLAYS - 1)
    _age(tmp_path)
    register, capture = td.register_pending_delivery, td._capture_emitted_answer
    monkeypatch.setattr(td, "register_pending_delivery", lambda root, row: False if row.get("emitted") else register(root, row))
    monkeypatch.setattr(td, "_capture_emitted_answer", lambda *_: None)
    assert sweep(tmp_path, dispatch) == []
    assert "emitted" not in td.pending_deliveries(tmp_path)[0]
    monkeypatch.setattr(td, "register_pending_delivery", register)
    monkeypatch.setattr(td, "_capture_emitted_answer", capture)
    assert sweep(tmp_path, dispatch) == [event["delivery_id"]]
    assert dispatch.calls == [(1, event["text"], td._PENDING_MAX_REPLAYS)]
    assert td.already_delivered(tmp_path, event["delivery_id"])


@pytest.mark.parametrize("legacy,family", [(False, "final"), (True, "final"), (True, "cancel"), (True, "cascade")])
@pytest.mark.parametrize("route", ["main", "hidden", "project"])
def test_legacy_and_new_debt_keep_identity_metadata_and_routing(tmp_path, monkeypatch, dispatch, legacy, family, route):
    expected_chat = 0 if route == "hidden" else 1
    if route == "project":
        from ouroboros.projects_registry import bind_task_to_project, create_project

        project = create_project(tmp_path, "dispatch-project", name="Dispatch project")
        expected_chat = project["chat_id"]
        bind_task_to_project(tmp_path, "turn", project["id"], expected_chat, origin={"absent": "system"})
    did = td.delivery_id_for("turn", "Exact answer") if family == "final" else f"{family}:turn:episode"
    event = owe(tmp_path, chat_id=0 if route == "hidden" else 1, delivery_id=did, terminal_origin="model_final")
    path = tmp_path / "state" / "terminal_deliveries.json"
    if legacy:
        data = json.loads(path.read_text())
        row = data["pending"][did]
        row.pop("dispatch_attempts")
        row.update(replay_attempts=37, forensic_metadata={"kept": "verbatim"})
        path.write_text(json.dumps(data))
    _age(tmp_path)
    dispatch.mode = "refused"
    assert sweep(tmp_path, dispatch) == [did]
    drain(dispatch)
    assert dispatch.calls == [(expected_chat, event["text"], 1)]
    (row,) = td.pending_deliveries(tmp_path)
    assert row["text"] == event["text"] and row["terminal_origin"] == "model_final"
    if legacy:
        assert row["replay_attempts"] == 37 and row["forensic_metadata"] == {"kept": "verbatim"}
    else:
        assert "replay_attempts" not in row
    _age(tmp_path)
    dispatch.mode = "ok"
    assert sweep(tmp_path, dispatch) == [did]
    assert dispatch.calls[-1] == (expected_chat, event["text"], 2)
    assert td.already_delivered(tmp_path, did)
    # A duplicate queued event after receipt, even after memory loss, never sends.
    delivery._DELIVERED_MESSAGE_IDS.clear()
    dispatch.bus.put({**event, "_delivery_replay": True})
    drain(dispatch)
    assert len(dispatch.calls) == 2


@pytest.mark.parametrize("state", ["exhausted", "withdrawn", "cap"])
def test_stale_replay_cannot_send_or_reopen_retired_debt(tmp_path, dispatch, state):
    event = owe(tmp_path)
    _age(tmp_path)
    assert sweep(tmp_path, dispatch) == [event["delivery_id"]]
    path = tmp_path / "state" / "terminal_deliveries.json"
    data = json.loads(path.read_text())
    did = event["delivery_id"]
    if state == "cap":
        data["pending"][did]["dispatch_attempts"] = td._PENDING_MAX_REPLAYS
    else:
        row = data["pending"].pop(did)
        if state == "exhausted":
            row.pop("dispatch_attempts")
            row.update(replay_attempts=5, attempts=5, reason="replay_exhausted", preserved_path="old-copy.txt")
            data["exhausted"] = {did: row}
    path.write_text(json.dumps(data))
    drain(dispatch)
    assert dispatch.calls == [] and not td.already_delivered(tmp_path, did)
    assert json.loads(path.read_text()) == data
    if state == "exhausted":
        assert td.register_pending_delivery(tmp_path, {**event, "delivery_deferred": True})
        assert sweep(tmp_path, dispatch) == []
        assert json.loads(path.read_text()) == data, "legacy exhausted debt is never auto-reopened or rewritten"


def test_first_live_send_needs_no_dispatch_counter_write(tmp_path, dispatch):
    event = owe(tmp_path)
    dispatch.bus.put(event)
    with store.name_lock(tmp_path, "terminal_deliveries.json.lock", timeout_sec=0):
        drain(dispatch)
    assert dispatch.calls == [(1, event["text"], 0)]
    assert sweep(tmp_path, dispatch) == [event["delivery_id"]]
    assert len(dispatch.calls) == 1 and td.already_delivered(tmp_path, event["delivery_id"])


def test_main_projection_keeps_its_canonical_retry_clock(tmp_path, dispatch, request):
    from ouroboros.task_results import load_task_result

    main = request.getfixturevalue("main_owed")
    dispatch.mode = "refused"
    dispatch.bus.put(main.event)
    drain(dispatch)
    for _ in range(td._PENDING_MAX_REPLAYS + 1):
        before = load_task_result(tmp_path, "root")["canonical_terminal_projection_ready"]["main_delivery"]
        assert sweep(tmp_path, dispatch) == []
        assert load_task_result(tmp_path, "root")["canonical_terminal_projection_ready"]["main_delivery"] == before
        _main_due(tmp_path)
        sweep(tmp_path, dispatch)
        drain(dispatch)
        (row,) = td.pending_deliveries(tmp_path)
        assert "dispatch_attempts" not in row
    assert len(dispatch.calls) == td._PENDING_MAX_REPLAYS + 2
    assert all(count is None for _chat, _text, count in dispatch.calls)
    assert not dispatch.notices


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("value", [None, "unknown", -1])
def test_invalid_dispatch_history_is_not_zero(tmp_path, dispatch, deferred, value):
    event = owe(tmp_path, delivery_deferred=deferred)
    change_row(tmp_path, event["delivery_id"], dispatch_attempts=value)
    _age(tmp_path)
    assert sweep(tmp_path, dispatch) == []
    assert dispatch.calls == []
    assert td.pending_deliveries(tmp_path)[0]["dispatch_attempts"] == value


def test_last_deferred_entry_stays_inflight_past_backoff(tmp_path, monkeypatch, dispatch):
    event = owe(tmp_path, delivery_deferred=True)
    change_row(tmp_path, event["delivery_id"], dispatch_attempts=td._PENDING_MAX_REPLAYS - 1)
    _age(tmp_path)
    entered, release = threading.Event(), threading.Event()
    transport = message_bus.get_bridge()
    send = transport.send_message

    def slow(*args, **kwargs):
        receipt = send(*args, **kwargs)
        entered.set()
        assert release.wait(5)
        return receipt

    monkeypatch.setattr(transport, "send_message", slow)
    results = []
    thread = threading.Thread(target=lambda: results.extend(sweep(tmp_path, dispatch)))
    thread.start()
    try:
        assert entered.wait(3)
        assert not dispatch.bus.delivery_outstanding(event["delivery_id"])
        _age(tmp_path)
        assert sweep(tmp_path, dispatch) == []
        assert td.terminal_answer_receipts(tmp_path, "turn")["state"] == "owed"
        assert not delivery._handle_send_message({**event, "_delivery_replay": True}, dispatch.ctx)
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and results == [event["delivery_id"]]
    assert dispatch.calls == [(1, event["text"], td._PENDING_MAX_REPLAYS)]
    assert not dispatch.notices and td.already_delivered(tmp_path, event["delivery_id"])


@pytest.mark.parametrize("legacy", [False, True])
def test_disclosure_uses_exact_prospective_count_or_unknown(tmp_path, dispatch, legacy):
    event = owe(tmp_path)
    path = tmp_path / "state" / "terminal_deliveries.json"
    data = json.loads(path.read_text())
    row = data["pending"][event["delivery_id"]]
    if legacy:
        row.pop("dispatch_attempts")
        row["replay_attempts"] = 5  # historical queue accounting cannot attest dispatches
    else:
        row["dispatch_attempts"] = 2
    path.write_text(json.dumps(data))
    td._retire_exhausted(tmp_path, [{**row, "delivery_id": event["delivery_id"]}])
    logs = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text().splitlines()]
    (fact,) = [row for row in logs if row["type"] == "terminal_delivery_exhausted"]
    assert fact["attempts"] == (None if legacy else 2), "never inflate to cap or turn absent history into zero"
    assert fact["attempts_basis"] == "recorded_replay_dispatch_entries"
    notice = dispatch.notices[0][1]
    assert ("prior physical send history is unknown" if legacy else "after 2 recorded replay dispatch entries") in notice
    assert "transport failing" not in notice and "crashing" not in notice
