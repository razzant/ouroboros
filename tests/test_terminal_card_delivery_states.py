"""The Done card follows the answer: the outbox's own debt gates it (owner Fork 2, 20:48).

``_delivery_disposition`` reads only the terminal-delivery registry: ``owed``
holds the card (a pending model final, Stop/reap salvage receipt or cascade
digest — debt outranks every older or technical receipt), ``exhausted`` and an
unreadable registry (``unknown``) are SHOWN on the card, never hidden, and
nothing is recomputed from the stored result's prose. The two writers that used
to write their terminal row before owing the salvage answer (pending-deadline
expiry, running cancel) now owe first, so a settle landing between the write
and the registration sees "not yet", never a card without the answer.
"""
from __future__ import annotations

from tests import _dialogue as dialogue_test

import json
import pathlib
import queue as stdqueue
import types
from collections import deque

import pytest

from ouroboros import cancel_intents as ci
from ouroboros.task_results import STATUS_RUNNING, load_task_result, write_task_result
from ouroboros.terminal_projection import (
    SETTLEMENT_DEFERRED, SETTLEMENT_NONE, SETTLEMENT_SETTLED, settle_terminal_projection,
)
from ouroboros.utils import append_jsonl
from supervisor import events_chat_delivery as delivery
from supervisor import terminal_delivery as td
from tests._cancel_intents_shared import _LiveProc, _seed_llm_response  # noqa: F401 - autouse reaper
from tests._cancel_intents_shared import qenv as _qenv

# The fixture is requested by name as a test parameter, so it is re-bound through a
# module attribute (a direct import that reappears as a parameter is an F811 under ruff).
qenv = _qenv
from tests.test_pending_execution_deadlines import add_task, clock, host  # noqa: F401 - the real deadline lane


class _Q:
    def __init__(self):
        self.events = []

    def put(self, evt):
        self.events.append(evt)


def _ctx(root, sent):
    def send(chat, text, **kw):
        sent.append((chat, text))
        append_jsonl(root / "logs" / "chat.jsonl", {"ts": "2026-10-15T00:00:00Z", "chat_id": chat, "text": text,
                                                     "task_id": kw.get("task_id"), "direction": "system"})

    return types.SimpleNamespace(DRIVE_ROOT=root, RUNNING={}, append_jsonl=append_jsonl,
                                 EVENT_Q=stdqueue.Queue(), send_with_budget=send)


def _chat_rows(root):
    path = root / "logs" / "chat.jsonl"
    return [row for row in map(json.loads, filter(str.strip, dialogue_test.log_text(path).splitlines()))]


def _terminal_rows(root, tid):
    return [row for row in _chat_rows(root) if row.get("summary_id") == f"task-terminal:{tid}"]


@pytest.fixture(autouse=True)
def _fresh_dedupe(monkeypatch):
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))


# ------------------------------------------------------------------ the six-case matrix (VERIFY_DELIVERY step 2)

@pytest.mark.parametrize("status,result,origin", [
    ("cancelled", "Running task cancelled and worker terminated.", ""),          # task_lifecycle running cancel
    ("failed", "The explicit task deadline elapsed before worker dispatch.", ""),  # queue_timeouts pending deadline
    ("cancelled", "Rewrote the atlas builder and reran the suite.", "host_salvage"),  # salvage-text root
])
def test_a_host_terminal_root_with_a_delivered_salvage_receipt_publishes_its_card(tmp_path, status, result, origin):
    """DEL-1: the receipt hashes its own framed lines, never the stored prose; once it is
    delivered nothing is owed, so the card is published (the candidate held it forever)."""
    tid = "root-host-1"
    write_task_result(tmp_path, tid, status, _terminal_observed=True, chat_id=7, result=result,
                      **({"terminal_origin": origin} if origin else {}))
    queue = _Q()
    assert td.deliver_unreviewed_salvage(tmp_path, {"chat_id": 7}, tid, outcome=status,
                                         salvaged_text="Partial work: rewrote X.", preserved_path="",
                                         settled_status=status, event_queue=queue) is True
    (evt,) = queue.events
    assert evt["delivery_id"].startswith(f"final:{tid}:")
    assert evt["delivery_id"] != td.delivery_id_for(tid, result)  # the receipt hashes its own lines
    assert td.terminal_answer_receipts(tmp_path, tid)["state"] == "owed"
    assert settle_terminal_projection(tmp_path, tid) == SETTLEMENT_DEFERRED, "owed: the card waits"
    sent = []
    delivery._handle_send_message(evt, _ctx(tmp_path, sent))
    assert sent and td.terminal_answer_receipts(tmp_path, tid)["state"] == "delivered"
    # The delivery ack re-entered the settle owner: the card is already published.
    assert isinstance(load_task_result(tmp_path, tid).get("canonical_terminal_projection"), dict)
    assert settle_terminal_projection(tmp_path, tid) == SETTLEMENT_NONE, "nothing left to settle"
    rows = _chat_rows(tmp_path)
    (card,) = _terminal_rows(tmp_path, tid)
    assert rows.index(card) > rows.index(next(row for row in rows if row.get("text") == evt["text"]))


def test_a_new_owed_primary_is_not_masked_by_an_older_delivered_receipt(tmp_path):
    """Owner 20:48: an old or technical receipt must not hide a NEW owed answer (natural order:
    the primary is registered owed BEFORE the terminal write)."""
    tid = "root-old-receipt"
    old = {"type": "send_message", "chat_id": 1, "task_id": tid, "text": "Earlier technical notice.",
           "delivery_id": td.delivery_id_for(tid, "Earlier technical notice.")}
    assert td.register_delivery(tmp_path, old["delivery_id"], emitted=old)
    new = {"type": "send_message", "chat_id": 1, "task_id": tid, "text": "Current primary answer.",
           "delivery_id": td.delivery_id_for(tid, "Current primary answer.")}
    assert td.register_pending_delivery(tmp_path, new)
    write_task_result(tmp_path, tid, "completed", _terminal_observed=True, chat_id=1, result="Current primary answer.")
    assert td.terminal_answer_receipts(tmp_path, tid)["state"] == "owed"
    assert settle_terminal_projection(tmp_path, tid) == SETTLEMENT_DEFERRED
    assert _terminal_rows(tmp_path, tid) == []          # Done held while the primary is owed
    assert td.register_delivery(tmp_path, new["delivery_id"], emitted=new)
    assert settle_terminal_projection(tmp_path, tid) == SETTLEMENT_SETTLED
    assert len(_terminal_rows(tmp_path, tid)) == 1      # and published once it is delivered


def test_an_unreadable_registry_publishes_the_card_instead_of_hiding_it(tmp_path):
    """DEL-2: unknown is a disclosed state, never an indefinite hold (owner: «неизвестная доставка честно»)."""
    tid = "root-corrupt-registry"
    write_task_result(tmp_path, tid, "completed", _terminal_observed=True, chat_id=1, result="Answer.")
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "terminal_deliveries.json").write_text("{corrupt", encoding="utf-8")
    assert td.terminal_answer_receipts(tmp_path, tid).get("registry") == "unreadable"
    assert settle_terminal_projection(tmp_path, tid) == SETTLEMENT_SETTLED
    assert len(_terminal_rows(tmp_path, tid)) == 1


def test_a_task_without_any_terminal_message_keeps_its_plain_card(tmp_path):
    """No debt known (nothing owed, nothing delivered): the card is published as before."""
    tid = "root-no-message"
    write_task_result(tmp_path, tid, "completed", _terminal_observed=True, chat_id=1, result="Answer.")
    assert td.terminal_answer_receipts(tmp_path, tid)["state"] == "unknown"
    assert settle_terminal_projection(tmp_path, tid) == SETTLEMENT_SETTLED
    (row,) = _terminal_rows(tmp_path, tid)
    assert row["outcome_phase"] == "done" and "terminal_delivery" not in row


# ------------------------------------------------------------------ the id families the debt covers (Astra R4)

def test_a_stop_episode_answer_is_owed_until_its_actual_send(tmp_path):
    """Astra R4: an owner Stop with an active intent mints ``cancel:<tid>:<request_id>`` — the only
    rescued paragraph rides it, so it is terminal-message debt, not an unknown."""
    tid = "stop-answer"
    write_task_result(tmp_path, tid, "running", result="working", chat_id=4)
    intent = ci.request_cancel(tmp_path, tid)
    event = td.build_unreviewed_salvage_event(
        tmp_path, {"chat_id": 4}, tid, outcome="cancelled",
        salvaged_text="the only rescued paragraph", settled_status="cancelled")
    assert event["delivery_id"] == "cancel:stop-answer:" + intent["request_id"]
    assert td._TERMINAL_ANSWER_ID.fullmatch(event["delivery_id"]) is None, "exact-bytes evidence stays narrow"
    assert td.register_pending_delivery(tmp_path, event) is True
    assert td.terminal_answer_receipts(tmp_path, tid)["state"] == "owed"
    write_task_result(tmp_path, tid, "cancelled", _terminal_observed=True, chat_id=4, result="stopped")
    assert settle_terminal_projection(tmp_path, tid) == SETTLEMENT_DEFERRED
    sent = []
    delivery._handle_send_message(event, _ctx(tmp_path, sent))
    assert sent == [(4, event["text"])]
    receipts = td.terminal_answer_receipts(tmp_path, tid)
    assert receipts["state"] == "unknown" and receipts["unverified_delivery_ids"] == [event["delivery_id"]]
    assert settle_terminal_projection(tmp_path, tid) == SETTLEMENT_NONE, "the ack already published the card"
    rows = _chat_rows(tmp_path)
    (card,) = _terminal_rows(tmp_path, tid)
    assert rows.index(card) > rows.index(next(row for row in rows if row.get("text") == event["text"]))


def test_a_cascade_digest_is_debt_while_custody_notice_main_mirror_and_incident_rows_are_not(tmp_path):
    tid = "cascade-root"
    for did in (f"cascade:{tid}:ci_1234",):
        assert td.register_pending_delivery(tmp_path, {"type": "send_message", "chat_id": 1, "task_id": tid,
                                                       "text": "digest", "delivery_id": did})
    assert td.terminal_answer_receipts(tmp_path, tid)["state"] == "owed"
    assert td.register_delivery(tmp_path, f"cascade:{tid}:ci_1234")
    assert td.terminal_answer_receipts(tmp_path, tid)["state"] == "unknown"
    final = td.delivery_id_for(tid, "answer")
    for did in (final + ":custody_notice", final + ":terminal_incident", f"project-completion:{tid}",
                f"cancel:{tid}:ci_1:notice", f"merge-receipt:{tid}:2"):
        assert td.register_pending_delivery(tmp_path, {"type": "send_message", "chat_id": 1, "task_id": tid,
                                                       "text": "row", "delivery_id": did})
    assert td.terminal_answer_receipts(tmp_path, tid)["state"] == "unknown", "only the answer families are debt"
    assert td.terminal_answer_receipts(tmp_path, tid)["owed_delivery_ids"] == []


# ------------------------------------------------------------------ the writers owe BEFORE their terminal row

@pytest.mark.serial
def test_running_cancel_owes_the_salvage_before_its_terminal_row_and_the_card_follows_the_answer(qenv, monkeypatch):
    """A settle landing right after the cancelled row is written sees "not yet": the salvage
    is already owed. After the real send the card is published behind the salvage paragraph."""
    order: list = []
    real_register = td.register_pending_delivery
    monkeypatch.setattr(
        "supervisor.terminal_delivery.register_pending_delivery",
        lambda root, evt: order.append(("owed", str(evt.get("delivery_id") or ""))) or real_register(root, evt),
    )
    from ouroboros import task_results as tr

    real_write = tr.write_task_result
    probes: list = []
    probing: list = []
    task_id = "owed-before-row"

    def write_then_probe(root, tid, status, **kwargs):
        stored = real_write(root, tid, status, **kwargs)
        if tid == task_id and status == "cancelled" and not probing:
            probing.append(True)
            order.append(("write", status))
            # The exact window the old order left open: terminal row written, salvage not yet owed.
            probes.append(settle_terminal_projection(qenv.drive, task_id))
        return stored

    monkeypatch.setattr("ouroboros.task_results.write_task_result", write_then_probe)
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *_a, **_kw: None)
    task = {"id": task_id, "chat_id": 9}
    proc = _LiveProc()
    qenv.workers.WORKERS[0] = types.SimpleNamespace(wid=0, proc=proc, busy_task_id=task_id, reaping=False)
    qenv.q.RUNNING[task_id] = {"task": task, "worker_id": 0}
    real_write(qenv.drive, task_id, STATUS_RUNNING, result="working", chat_id=9)
    _seed_llm_response(qenv.drive, task_id, "the salvaged partial answer")
    intent = ci.request_cancel(qenv.drive, task_id, reason="stop")
    try:
        assert qenv.tl.cancel_task_custody(task_id) == qenv.tl.CANCEL_CANCELLED
    finally:
        proc.terminate()
    expected = f"cancel:{task_id}:{intent['request_id']}"
    assert order.index(("owed", expected)) < order.index(("write", "cancelled")), order
    assert probes == [SETTLEMENT_DEFERRED], "a settle between the row and the send sees the debt"
    assert _terminal_rows(qenv.drive, task_id) == []
    (owed,) = [row for row in td.pending_deliveries(qenv.drive) if row["delivery_id"] == expected]
    assert "the salvaged partial answer" in owed["text"]
    stored = load_task_result(qenv.drive, task_id)
    assert stored["status"] == "cancelled" and stored["cancel_receipt"]["delivery_id"] == expected
    sent = []
    delivery._handle_send_message({**owed, "type": "send_message"}, _ctx(qenv.drive, sent))
    assert sent == [(9, owed["text"])]
    assert settle_terminal_projection(qenv.drive, task_id) == SETTLEMENT_NONE, "the ack already published the card"
    rows = _chat_rows(qenv.drive)
    (card,) = _terminal_rows(qenv.drive, task_id)
    assert rows.index(card) > rows.index(next(row for row in rows if row.get("text") == owed["text"]))
    assert card["outcome_phase"] == "cancelled"


@pytest.mark.serial
def test_pending_deadline_expiry_owes_the_salvage_before_its_terminal_row(host, clock, monkeypatch):  # noqa: F811
    """The queue_timeouts writer: the deadline salvage is owed before the failed row exists;
    a settle right after the write defers; the card follows the delivered salvage."""
    from supervisor import queue
    from ouroboros import task_results as tr

    task = add_task(host, clock)
    order: list = []
    real_register = td.register_pending_delivery
    monkeypatch.setattr(
        "supervisor.terminal_delivery.register_pending_delivery",
        lambda root, evt: order.append(("owed", str(evt.get("delivery_id") or ""))) or real_register(root, evt),
    )
    real_write = tr.write_task_result
    probes: list = []
    probing: list = []

    def write_then_probe(root, tid, status, **kwargs):
        stored = real_write(root, tid, status, **kwargs)
        if tid == "pending" and status == "failed" and not probing:
            probing.append(True)
            order.append(("write", "failed"))
            probes.append(settle_terminal_projection(host.root, "pending"))
        return stored

    monkeypatch.setattr("ouroboros.task_results.write_task_result", write_then_probe)
    clock.now[0] += 100
    queue.enforce_task_timeouts()
    clock.reap()
    assert task not in host.pending
    owed_at = next(index for index, item in enumerate(order) if item[0] == "owed")
    assert owed_at < order.index(("write", "failed")), order
    assert probes == [SETTLEMENT_DEFERRED]
    assert _terminal_rows(host.root, "pending") == []
    (owed,) = td.pending_deliveries(host.root)
    assert owed["task_id"] == "pending" and owed["delivery_id"].startswith("final:pending:")
    sent = []
    delivery._handle_send_message({**owed, "type": "send_message"}, _ctx(host.root, sent))
    assert sent == [(1, owed["text"])]
    assert settle_terminal_projection(host.root, "pending") == SETTLEMENT_NONE, "the ack already published the card"
    rows = _chat_rows(host.root)
    (card,) = _terminal_rows(host.root, "pending")
    assert rows.index(card) > rows.index(next(row for row in rows if row.get("text") == owed["text"]))
    assert card["outcome_phase"] == "error"


def test_no_debt_and_no_readiness_settles_nothing(tmp_path):
    assert settle_terminal_projection(tmp_path, "never-stored") == SETTLEMENT_NONE


# ------------------------------------------------------------------ honest delivery state ON the card (owner 20:48)

def test_an_unreadable_registry_card_says_delivery_could_not_be_confirmed(tmp_path):
    """DEL-2 rendering: the sixth case of the matrix. The warning is row-level only."""
    tid = "root-corrupt-registry-2"
    write_task_result(tmp_path, tid, "completed", _terminal_observed=True, chat_id=1, result="Answer.")
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "terminal_deliveries.json").write_text("{corrupt", encoding="utf-8")
    assert settle_terminal_projection(tmp_path, tid) == SETTLEMENT_SETTLED
    (row,) = _terminal_rows(tmp_path, tid)
    assert row["outcome_phase"] == "warn" and row["outcome"] == "Done with warnings"
    assert row["terminal_delivery"] == {"state": "unknown"}
    assert row["outcome_axes"]["objective"]["warning"] == "terminal_delivery_unknown"
    assert row["reason_detail"] == ("Delivery of the final answer could not be confirmed because the delivery "
                                    "registry could not be read; the result is in task details.")
    assert row["reason_detail"] in row["text"]
    stored = load_task_result(tmp_path, tid)
    assert "warning" not in (stored.get("outcome_axes") or {}).get("objective", {}), "no durable change to the task's axes"
    assert "terminal_delivery" not in stored


def test_an_exhausted_delivery_is_shown_with_its_saved_copy_and_a_late_receipt_clears_the_mark(tmp_path, monkeypatch):
    notices: list = []
    monkeypatch.setattr("supervisor.message_bus.send_with_budget",
                        lambda chat_id, text, **kw: notices.append((chat_id, text, kw)))
    tid = "root-exhausted"
    event = {"type": "send_message", "chat_id": 9, "task_id": tid, "text": "the answer nobody confirmed",
             "delivery_id": td.delivery_id_for(tid, "the answer nobody confirmed")}
    assert td.register_pending_delivery(tmp_path, event)
    write_task_result(tmp_path, tid, "completed", _terminal_observed=True, chat_id=9, result="the answer nobody confirmed")
    assert settle_terminal_projection(tmp_path, tid) == SETTLEMENT_DEFERRED, "owed holds the card"
    # Exhaustion is a transition inside the registry: the payload stays, the copy is preserved,
    # the disclosure is supervisor-owned (this test is the supervisor), and the held card settles.
    for _ in range(td._PENDING_MAX_REPLAYS):
        assert td.record_replay_dispatch(tmp_path, event["delivery_id"])  # transport entered; receipt unknown
    (owed,) = td.pending_deliveries(tmp_path)
    td._retire_exhausted(tmp_path, [owed])
    receipts = td.terminal_answer_receipts(tmp_path, tid)
    assert receipts["state"] == "exhausted" and td.pending_deliveries(tmp_path) == []
    (mark,) = receipts["exhausted"]
    assert mark["text"] == "the answer nobody confirmed" and mark["reason"] == "replay_exhausted"
    assert mark["attempts"] == td._PENDING_MAX_REPLAYS and mark["preserved_path"].endswith(".txt")
    (notice,) = notices
    assert notice[0] == 9 and "was not confirmed" in notice[1] and notice[2]["system_type"] == "terminal_incident"
    (card,) = _terminal_rows(tmp_path, tid)
    assert card["outcome_phase"] == "warn" and card["terminal_delivery"] == {
        "state": "exhausted", "delivery_id": event["delivery_id"], "preserved_path": mark["preserved_path"]}
    assert card["reason_detail"] == ("Sending the final answer was not confirmed after the delivery attempts "
                                     f"ran out; a copy of the answer is saved at {mark['preserved_path']}.")
    # The queued original lands after all: the real receipt retires only that id's mark.
    sent = []
    delivery._handle_send_message(event, _ctx(tmp_path, sent))
    assert sent == [(9, event["text"])]
    receipts = td.terminal_answer_receipts(tmp_path, tid)
    assert receipts["state"] == "delivered" and receipts["exhausted"] == []


def test_a_capacity_eviction_in_a_worker_records_the_mark_and_the_copy_only(tmp_path, monkeypatch):
    """Worker-side eviction: no chat write, no settle from a worker (DEL-5); the registry mark
    plus the preserved copy and the typed event are the disclosure the supervisor reads."""
    notices: list = []
    monkeypatch.setattr("supervisor.message_bus.send_with_budget",
                        lambda chat_id, text, **kw: notices.append((chat_id, text)))
    monkeypatch.setattr("ouroboros.utils.in_worker_process", lambda: True)
    monkeypatch.setattr("ouroboros.terminal_projection.settle_terminal_projection",
                        lambda *_a, **_k: pytest.fail("a worker never settles the card"))
    for index in range(td._PENDING_CAP + 1):
        assert td.register_pending_delivery(tmp_path, {
            "type": "send_message", "chat_id": 5, "task_id": f"cap{index}",
            "text": f"answer {index}", "delivery_id": td.delivery_id_for(f"cap{index}", f"answer {index}")})
    assert notices == []
    receipts = td.terminal_answer_receipts(tmp_path, "cap0")
    assert receipts["state"] == "exhausted"
    (mark,) = receipts["exhausted"]
    assert mark["reason"] == "outbox_capacity" and mark["text"] == "answer 0"
    assert pathlib.Path(mark["preserved_path"]).read_text(encoding="utf-8") == "answer 0"
    rows = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    (event,) = [row for row in rows if row.get("type") == "terminal_delivery_exhausted"]
    assert event["task_id"] == "cap0" and event["reason"] == "outbox_capacity"
    assert len(td.pending_deliveries(tmp_path)) == td._PENDING_CAP


def test_a_failed_copy_leaves_an_honest_empty_pointer(tmp_path, monkeypatch):
    monkeypatch.setattr("supervisor.message_bus.send_with_budget", lambda *_a, **_k: None)

    def refuse(*_args, **_kwargs):
        raise OSError("Too many open files")

    monkeypatch.setattr("ouroboros.observability.preserve_salvaged_output", refuse)
    tid = "root-no-copy"
    event = {"type": "send_message", "chat_id": 9, "task_id": tid, "text": "lost",
             "delivery_id": td.delivery_id_for(tid, "lost")}
    assert td.register_pending_delivery(tmp_path, event)
    write_task_result(tmp_path, tid, "completed", _terminal_observed=True, chat_id=9, result="lost")
    assert settle_terminal_projection(tmp_path, tid) == SETTLEMENT_DEFERRED
    (owed,) = td.pending_deliveries(tmp_path)
    td._retire_exhausted(tmp_path, [owed])
    (mark,) = td.terminal_answer_receipts(tmp_path, tid)["exhausted"]
    assert mark["preserved_path"] == ""
    (card,) = _terminal_rows(tmp_path, tid)
    assert card["reason_detail"].endswith("; no copy of the answer could be saved.")
    assert "saved at" not in card["reason_detail"]
