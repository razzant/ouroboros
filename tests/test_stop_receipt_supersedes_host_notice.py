"""A delivered Stop receipt retires its task's older host notice in that chat; nothing else does.

Real outbox, bus consumer, send handler, receipts and replay on a private root; only the
transport is controlled. Once a Stop receipt is recorded as delivered, the task's earlier
``host_notice`` final that the sender would route now to the chat the receipt reached is
``superseded`` whatever its own earlier sends did (refused, never returned, unknown): its
bytes and send facts stay in the outbox beside the receipt that replaced it, it is neither
delivered nor given up, and no sweep, restart or buffered copy sends it again. A send
already inside the transport may still land; its real receipt is then recorded (a given-up
row's too), from the kept send fact after a restart when the receipt itself could not be
written. A Stop receipt the previous writer (ca791d980) recorded supersedes on the next
generation's first sweep, proven by the task result's receipt. A custody notice the
replaced notice carried is still owed as its own row. Model answers, host salvage,
unstamped legacy rows, other tasks or chats, notices the sender now routes elsewhere (the
task became a Project after the Stop, a row pinned to its own chat) and newer notices stay
owed, and a Stop receipt that was refused, never returned or is merely queued retires
nothing.
"""
from __future__ import annotations

import contextlib
import errno
import json
import pathlib
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from ouroboros import server_liveness, store
from ouroboros.project_dialogue import MAIN_NOTICE_TYPE
from ouroboros.task_finalization import (
    TERMINAL_ORIGIN_HOST_NOTICE, TERMINAL_ORIGIN_HOST_SALVAGE, TERMINAL_ORIGIN_MODEL_FINAL,
)
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.task_store import patch_document
from supervisor import events_chat_delivery as delivery, message_bus, terminal_delivery as td
from supervisor.event_observation import ObservedEventQueue
from supervisor.terminal_supersession import superseded_by
from tests._dialogue import chat_rows
from tests.test_delivery_contention import sender as _sender
from tests.test_dialogue_retry import isolated_retry as _isolated_retry, seed

sender, isolated_retry = _sender, _isolated_retry
NOTICE = "⚠️ Error during processing: OSError: [Errno 23] Too many open files in system"


class Interrupted(BaseException):
    """The process died inside the transport: no outcome of that send is ever recorded."""


@pytest.fixture
def chat(tmp_path, monkeypatch, sender):
    seed(tmp_path, monkeypatch)
    ctx, _frames, bus = sender
    trace = SimpleNamespace(root=tmp_path, ctx=ctx, bus=bus, sends=[], routes={}, notices=[], mode={}, during={})

    def transport(chat_id, text, **kwargs):
        did = (kwargs.get("progress_meta") or {}).get("delivery_id")
        if not did:  # an unnamed host notice, such as an exhausted delivery's warning
            trace.notices.append((chat_id, text))
            return True, "ok"
        trace.sends.append(did)  # the order copies entered the transport
        trace.routes[did] = chat_id
        if hook := trace.during.pop(did, None):
            hook()  # something else happens while this copy is inside the transport
        outcome = trace.mode.get(did)
        if outcome == "interrupted":
            raise Interrupted()
        if outcome == "enfile":
            raise OSError(errno.ENFILE, "Too many open files in system")
        return (False, "offline") if outcome == "refused" else (True, "ok")

    monkeypatch.setattr(message_bus.get_bridge(), "send_message", transport)
    monkeypatch.setattr(td, "_REPLAY_MIN_AGE_SEC", 0.0)  # every owed row is due on the next sweep
    write_task_result(tmp_path, "t1", "cancelled", chat_id=1)  # Stop custody settled the task
    return trace


def registry(root):
    return json.loads((root / "state" / "terminal_deliveries.json").read_text(encoding="utf-8"))


def write_registry(root, data):
    (root / "state" / "terminal_deliveries.json").write_text(json.dumps(data), encoding="utf-8")


def notice(trace, *, task="t1", chat=1, text=NOTICE, origin=TERMINAL_ORIGIN_HOST_NOTICE, **fields):
    """A task's own terminal message, projected and owed the way the answer path does it (not yet queued)."""
    base = {"type": "send_message", "chat_id": chat, "task_id": task, "text": text, "format": "markdown", **fields}
    event = td.project_terminal_result_event(trace.root, {"chat_id": chat}, task, result_text=text,
                                             terminal_origin=origin, base_event=base)
    assert td.register_pending_delivery(trace.root, event)
    return event


def legacy_row(trace, *, origin=None):
    """An owed row as an older generation wrote it: no dispatch count, its legacy replay count, no send facts."""
    did = td.delivery_id_for("t1", NOTICE)
    stamp = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    row = {"chat_id": 1, "task_id": "t1", "text": NOTICE, "log_text": NOTICE, "format": "markdown",
           "delivery_id": did, "ts": stamp, "registered_at": stamp, "replay_attempts": 2,
           **({"role": "system", "terminal_origin": origin} if origin else {})}
    write_registry(trace.root, {"schema_version": 3, "delivered": [], "pending": {did: row}})
    return {**row, "type": "send_message"}


def stop_receipt(trace, *, task="t1", chat=1, delivery_id="cancel:t1:ci_1", outcome="stopped by the owner"):
    """The Stop custody's salvage receipt, built and owed by its own producer (not yet queued)."""
    event = td.build_unreviewed_salvage_event(trace.root, {"chat_id": chat}, task, outcome=outcome,
                                              salvaged_text="The salvaged answer.", settled_status="cancelled",
                                              delivery_id=delivery_id)
    assert td.register_pending_delivery(trace.root, event)
    return event


def drain(trace):
    server_liveness.drain_worker_events(trace.bus, trace.ctx, [time.monotonic(), {}, time.thread_time(), None],
                                       on_restart=lambda *_: pytest.fail("no restart event"))


def deliver(trace, event):
    """One copy on the bus, as its producer or the worker's buffered return puts it, then the consumer."""
    trace.bus.put(dict(event))
    drain(trace)


def sweep(trace):
    """One duty pass (deferred continuations, aged replay), then the consumer sends what it queued."""
    replayed = td.replay_pending_deliveries(trace.root, event_queue=trace.bus)
    drain(trace)
    return replayed


def restart(trace, monkeypatch):
    """A new server generation: an empty bus and process memory; only the durable stores remain."""
    trace.bus = trace.ctx.EVENT_Q = ObservedEventQueue()
    monkeypatch.setattr("supervisor.workers.get_event_q", lambda **_kw: trace.bus)
    for name in ("_ACTIVE_DELIVERIES", "_DEFERRED_DELIVERIES"):
        monkeypatch.setattr(delivery, name, set())
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))


def owed(root):
    return {row["delivery_id"] for row in td.pending_deliveries(root)}


def events(root, kind):
    path = root / "logs" / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return [row for row in rows if row.get("type") == kind]


def without_mark(row):
    return {key: value for key, value in row.items() if key not in ("superseded_by", "superseded_at")}


def test_a_failed_error_notice_is_not_repeated_after_the_delivered_stop(chat, monkeypatch):
    from ouroboros.terminal_projection import _delivery_disposition
    from ouroboros.working_checkpoint import final_answer_registered, registered_answer

    first = notice(chat)
    did = first["delivery_id"]
    chat.mode[did] = "enfile"
    deliver(chat, first)  # the incident: the live first send fails inside the transport
    failed = registry(chat.root)["pending"][did]
    assert failed["transport_failed"] is True and failed["delivery_deferred"] is True and "emitted" not in failed
    history = [row for row in chat_rows(chat.root) if row.get("text") == NOTICE]
    assert len(history) == 1, "the failed attempt's own history row"
    restart(chat, monkeypatch)  # restarts keep only the durable outbox
    assert _delivery_disposition(chat.root, "t1")["state"] == "owed"
    receipt = stop_receipt(chat)
    cancel = receipt["delivery_id"]
    deliver(chat, receipt)
    assert chat.sends == [did, cancel] and chat.routes[cancel] == 1
    assert load_task_result(chat.root, "t1")["cancel_receipt"]["delivered_chat_id"] == 1
    data = registry(chat.root)
    kept = data["superseded"][did]
    assert data["schema_version"] == 4 and did not in data["pending"] and did not in data["delivered"]
    assert without_mark(kept) == failed, "its bytes and earlier send facts stay as they were"
    assert kept["superseded_by"] == cancel and superseded_by(chat.root, did) == cancel
    assert not td.already_delivered(chat.root, did) and "exhausted" not in data
    (event,) = events(chat.root, "terminal_delivery_superseded")
    assert (event["task_id"], event["delivery_id"], event["superseded_by"], event["chat_id"]) == ("t1", did, cancel, 1)
    # Its readers: no debt holds the Done card, and the task's terminal message is the delivered receipt.
    facts = td.terminal_answer_receipts(chat.root, "t1")
    assert facts["owed"] == [] and facts["exhausted"] == [] and facts["unverified_delivery_ids"] == [cancel]
    assert _delivery_disposition(chat.root, "t1")["state"] == ""
    assert final_answer_registered(chat.root, "t1")
    assert {key: registered_answer(chat.root, "t1")[key] for key in ("basis", "delivery_id")} == {
        "basis": "delivered", "delivery_id": cancel}
    # No sweep sends it again, gives up on it or says so; neither does a later generation.
    for _ in range(td._PENDING_MAX_REPLAYS + 2):
        assert sweep(chat) == []
    restart(chat, monkeypatch)
    assert sweep(chat) == []
    deliver(chat, first)  # the worker's buffered copy of the same message
    td.enqueue_terminal_delivery_outcome(chat.root, first, event_queue=chat.bus)  # a producer owing it again
    drain(chat)
    later = notice(chat, task="t2", text="Another task's answer.", origin=TERMINAL_ORIGIN_MODEL_FINAL)
    deliver(chat, later)  # later writes for other messages keep the disposition
    assert chat.sends == [did, cancel, later["delivery_id"]] and chat.notices == []
    assert registry(chat.root)["superseded"][did] == kept and owed(chat.root) == set()
    assert events(chat.root, "terminal_delivery_exhausted") == []
    assert [row for row in chat_rows(chat.root) if row.get("text") == NOTICE] == history, "history is kept, not added to"


@pytest.mark.parametrize("earlier", ["live_send_never_returned", "replay_never_returned", "older_generation_row"])
def test_an_earlier_send_with_an_unknown_outcome_is_superseded_too(chat, monkeypatch, earlier):
    if earlier == "older_generation_row":
        first = legacy_row(chat, origin=TERMINAL_ORIGIN_HOST_NOTICE)
    else:
        first = notice(chat)
        chat.mode[first["delivery_id"]] = "interrupted"
        with pytest.raises(Interrupted):
            if earlier == "live_send_never_returned":
                deliver(chat, first)
            else:  # its live copy was lost; a replay recorded its dispatch entry, then never returned
                assert td.replay_pending_deliveries(chat.root, event_queue=chat.bus) == [first["delivery_id"]]
                drain(chat)
    did = first["delivery_id"]
    row = registry(chat.root)["pending"][did]
    assert not {"emitted", "delivery_deferred", "transport_failed"} & set(row), "the outbox knows no outcome"
    restart(chat, monkeypatch)  # the process that may have shown it is gone
    chat.mode.clear()
    receipt = stop_receipt(chat)
    deliver(chat, receipt)
    kept = registry(chat.root)["superseded"][did]
    assert without_mark(kept) == row and kept["superseded_by"] == receipt["delivery_id"]
    assert sweep(chat) == [] and did not in owed(chat.root)
    assert chat.sends.count(did) == (0 if earlier == "older_generation_row" else 1)


@pytest.mark.parametrize("copy", ["queued_before_the_receipt", "buffered_after_it"])
def test_a_buffered_original_handled_after_the_receipt_never_reaches_the_transport(chat, copy):
    first, receipt = notice(chat), stop_receipt(chat)
    if copy == "queued_before_the_receipt":
        chat.bus.put(dict(first))  # its live copy still waits behind a slow consumer
    assert delivery._handle_send_message(dict(receipt), chat.ctx)  # the duty's own continuation sends the receipt
    if copy == "buffered_after_it":
        chat.bus.put(dict(first))  # the worker's buffered return of the same message
    drain(chat)
    assert chat.sends == [receipt["delivery_id"]]
    assert superseded_by(chat.root, first["delivery_id"]) == receipt["delivery_id"]


@pytest.mark.parametrize("stop", ["refused", "interrupted", "queued_only", "unreadable_outbox"])
def test_a_stop_receipt_not_recorded_as_delivered_retires_nothing(chat, stop):
    first = notice(chat)
    did = first["delivery_id"]
    chat.mode[did] = "refused"
    deliver(chat, first)  # the notice's own first send failed
    receipt = stop_receipt(chat)
    cancel, path = receipt["delivery_id"], chat.root / "state" / "terminal_deliveries.json"
    row = registry(chat.root)["pending"][did]
    if stop == "queued_only":
        assert td.enqueue_terminal_delivery(chat.root, receipt, event_queue=chat.bus)  # admitted, not yet handled
    elif stop == "unreadable_outbox":
        path.write_text("{not json", encoding="utf-8")
        deliver(chat, receipt)  # sent, but no receipt can be recorded
        assert path.read_text(encoding="utf-8") == "{not json", "malformed bytes are kept, never rebuilt"
        assert chat.sends == [did, cancel]
        return
    else:
        chat.mode[cancel] = stop
        if stop == "interrupted":
            with pytest.raises(Interrupted):
                deliver(chat, receipt)
        else:
            deliver(chat, receipt)
    data = registry(chat.root)
    assert data["pending"][did] == row and cancel not in data["delivered"] and "superseded" not in data
    assert events(chat.root, "terminal_delivery_superseded") == []
    chat.mode.clear()
    sent = len(chat.sends)
    sweep(chat)
    assert did in chat.sends[sent:], "the notice is still a retry candidate"


def unwritable(*_args, **_kwargs):
    raise OSError(errno.ENFILE, "Too many open files in system")


@pytest.mark.parametrize("failure", ["result_store_unwritable", "outbox_busy"])
def test_a_sent_stop_receipt_recorded_later_supersedes_before_any_resend(chat, monkeypatch, failure):
    first = notice(chat)
    did = first["delivery_id"]
    chat.mode[did] = "refused"
    deliver(chat, first)  # failed: the next sweep would retry it
    receipt = stop_receipt(chat)
    cancel = receipt["delivery_id"]
    if failure == "result_store_unwritable":
        with monkeypatch.context() as patched:
            patched.setattr(td, "patch_document", unwritable)  # its destination evidence cannot be written
            deliver(chat, receipt)
    else:
        with store.name_lock(chat.root, "terminal_deliveries.json.lock", timeout_sec=0):
            deliver(chat, receipt)  # sent while the outbox is busy for its receipt
    assert chat.sends == [did, cancel] and did in owed(chat.root) and "superseded" not in registry(chat.root)
    chat.mode.clear()
    sweep(chat)  # the duty records the receipt from its send fact first, then retries what is still owed
    assert chat.sends == [did, cancel], "neither the receipt nor the notice it replaced is sent again"
    assert cancel in registry(chat.root)["delivered"] and superseded_by(chat.root, did) == cancel
    assert sweep(chat) == [] and chat.sends == [did, cancel]


def test_a_notice_send_already_in_the_transport_lands_and_its_receipt_is_recorded(chat):
    first, receipt = notice(chat), stop_receipt(chat)
    did, cancel, meanwhile = first["delivery_id"], receipt["delivery_id"], []

    def stop_lands():  # the Stop receipt is delivered while this copy is inside the transport
        assert delivery._handle_send_message(dict(receipt), chat.ctx)
        meanwhile.append(superseded_by(chat.root, did))

    chat.during[did] = stop_lands
    deliver(chat, first)
    assert meanwhile == [cancel], "the outbox stopped owing it; the send already begun was not cancelled"
    data = registry(chat.root)
    assert did in data["delivered"] and did not in data.get("superseded", {}) and did not in data["pending"]
    assert data["receipts"][did]["text_chars"] == len(NOTICE) and chat.sends == [did, cancel]
    assert [row["delivery_id"] for row in td.terminal_answer_receipts(chat.root, "t1")["delivered"]] == [did]
    assert len(events(chat.root, "terminal_delivery_superseded")) == 1, "the transition stays in the event history"


def test_a_notice_send_in_the_transport_that_then_fails_stays_superseded(chat):
    first, receipt = notice(chat), stop_receipt(chat)
    did = first["delivery_id"]
    chat.mode[did] = "refused"
    chat.during[did] = lambda: delivery._handle_send_message(dict(receipt), chat.ctx)
    deliver(chat, first)
    assert superseded_by(chat.root, did) == receipt["delivery_id"] and did not in owed(chat.root)
    chat.mode.clear()
    assert sweep(chat) == [] and chat.sends == [did, receipt["delivery_id"]]


def test_a_notice_whose_send_was_recorded_gets_its_receipt_instead(chat, monkeypatch):
    first = notice(chat)
    did = first["delivery_id"]
    with monkeypatch.context() as patched:
        patched.setattr(td, "_capture_emitted_answer", lambda *_a, **_k: None)  # its byte receipt waits
        deliver(chat, first)  # sent; only the send fact reached the outbox
    assert registry(chat.root)["pending"][did]["emitted"]["text"] == NOTICE
    receipt = stop_receipt(chat)
    deliver(chat, receipt)
    assert did in owed(chat.root) and "superseded" not in registry(chat.root), "a recorded send is repaired"
    sweep(chat)
    data = registry(chat.root)
    assert chat.sends == [did, receipt["delivery_id"]], "the repair sends nothing"
    assert did in data["delivered"] and data["receipts"][did]["text_chars"] == len(NOTICE)


OTHERS = {
    "model_answer": dict(origin=TERMINAL_ORIGIN_MODEL_FINAL, text="The real answer."),
    "model_answer_that_reads_like_an_error": dict(origin=TERMINAL_ORIGIN_MODEL_FINAL),
    "host_salvage": dict(origin=TERMINAL_ORIGIN_HOST_SALVAGE, text="Partial model bytes."),
    "other_task": dict(task="t2"),
    "other_chat": dict(chat=2),
}


@pytest.mark.parametrize("kind", [*sorted(OTHERS), "unstamped_legacy_row", "newer_than_the_receipt"])
def test_every_other_owed_message_stays_owed_and_still_replays(chat, kind):
    if kind == "newer_than_the_receipt":
        receipt = stop_receipt(chat)
        kept = notice(chat)
    else:
        kept = legacy_row(chat) if kind == "unstamped_legacy_row" else notice(chat, **OTHERS[kind])
        receipt = stop_receipt(chat)
    did = kept["delivery_id"]
    payload = registry(chat.root)["pending"][did]
    deliver(chat, receipt)
    assert registry(chat.root)["pending"][did] == payload and "superseded" not in registry(chat.root)
    assert sweep(chat) == [did] and chat.sends == [receipt["delivery_id"], did]


@pytest.mark.parametrize("kind", ["custody_suffix", "main_publication", "stop_receipt_family",
                                  "another_tasks_id", "another_tasks_field"])
def test_rows_outside_the_task_final_family_stay_owed(chat, kind):
    """Rows a typed fact puts outside the receipt's task final, including ones whose id and task disagree."""
    did = {"custody_suffix": td.delivery_id_for("t1", NOTICE) + ":custody_notice",
           "main_publication": td.delivery_id_for("t1", NOTICE), "stop_receipt_family": "cascade:t1:r0",
           "another_tasks_id": td.delivery_id_for("t2", NOTICE), "another_tasks_field": td.delivery_id_for("t1", NOTICE)}[kind]
    stamp = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    row = {"chat_id": 1, "task_id": "t2" if kind == "another_tasks_field" else "t1", "text": NOTICE, "role": "system",
           "terminal_origin": "host_notice", "delivery_id": did, "registered_at": stamp, "dispatch_attempts": 0,
           **({"terminal_projection_token": "publication-1"} if kind == "main_publication" else {})}
    receipt = stop_receipt(chat)
    data = registry(chat.root)
    data["pending"][did] = row
    write_registry(chat.root, data)
    deliver(chat, receipt)
    assert registry(chat.root)["pending"][did] == row and "superseded" not in registry(chat.root)


def to_project(trace, task="t1"):
    """The owner turns the task (or its root) into a Project: the sender routes its messages to that room."""
    from ouroboros.projects_registry import bind_task_to_project

    room = bind_task_to_project(trace.root, task, "room", origin={"absent": "post_hoc_unresolved"})["project_chat_id"]
    assert room not in (0, 1)
    return room


def test_the_task_project_room_is_the_same_chat(chat):
    room = to_project(chat)
    first = notice(chat)  # its row keeps the chat the task was born in
    did = first["delivery_id"]
    chat.mode[did] = "refused"
    deliver(chat, first)
    receipt = stop_receipt(chat)
    deliver(chat, receipt)  # the sender routes both by the task's project binding
    assert chat.routes == {did: room, receipt["delivery_id"]: room} and registry(chat.root)["pending"] == {}
    assert superseded_by(chat.root, did) == receipt["delivery_id"]


@pytest.mark.parametrize("project", ["no_project", "project_before_the_stop", "project_after_the_stop",
                                     "main_pinned_notice"])
def test_a_stop_receipt_recorded_later_supersedes_only_where_the_notice_goes_now(chat, monkeypatch, project):
    """The receipt is recorded from its send fact on the next sweep, when the task may have become a Project.

    No producer gives a host notice final a system type; the Main-pinned row (here and for the previous
    writer) checks that the sender's own routing rule decides even so."""
    first = notice(chat, **({"system_type": MAIN_NOTICE_TYPE} if project == "main_pinned_notice" else {}))
    did = first["delivery_id"]
    chat.mode[did] = "refused"
    deliver(chat, first)  # failed: the next sweep would retry it
    room = to_project(chat) if project in ("project_before_the_stop", "main_pinned_notice") else None
    receipt = stop_receipt(chat)
    cancel = receipt["delivery_id"]
    with monkeypatch.context() as patched:
        patched.setattr(td, "patch_document", unwritable)  # sent; its receipt waits for the next sweep
        deliver(chat, receipt)
    if project == "project_after_the_stop":
        room = to_project(chat)  # the Stop reached chat 1; the room never showed it
    chat.mode.clear()
    sweep(chat)  # the duty records the receipt from its send fact first, then retries what is still owed
    assert chat.routes[cancel] == (1 if project in ("no_project", "project_after_the_stop") else room)
    if project in ("no_project", "project_before_the_stop"):
        assert superseded_by(chat.root, did) == cancel and chat.sends == [did, cancel]
    else:  # the sender takes it where the Stop did not go: still owed there, and sent
        assert superseded_by(chat.root, did) == "" and chat.sends == [did, cancel, did]
        assert chat.routes[did] == (room if project == "project_after_the_stop" else 1)
        assert did in registry(chat.root)["delivered"]


@pytest.mark.parametrize("family", ["cascade_summary", "reap_salvage_without_a_stop"])
def test_only_a_stop_receipt_family_supersedes(chat, family):
    first = notice(chat)
    did = first["delivery_id"]
    chat.mode[did] = "refused"
    deliver(chat, first)
    receipt = (stop_receipt(chat, delivery_id="cascade:t1:ci_1") if family == "cascade_summary"
               else stop_receipt(chat, delivery_id="", outcome="reaped after its deadline"))
    assert receipt["system_type"] == "cancel_receipt" and receipt["delivery_id"].startswith(
        "cascade:" if family == "cascade_summary" else "final:t1:")
    deliver(chat, receipt)
    assert receipt["delivery_id"] in registry(chat.root)["delivered"]
    assert superseded_by(chat.root, did) == (receipt["delivery_id"] if family == "cascade_summary" else "")
    assert (did in owed(chat.root)) is (family != "cascade_summary")


def capture_fails(patched, _trace, did, _held):
    """Retaining this message's sent bytes fails (the incident's ENFILE); every other capture works."""
    from ouroboros import artifacts

    real = artifacts.store_actor_source_bytes

    def capture(root, task_id, **kwargs):
        if json.loads(kwargs["data"]).get("delivery_id") == did:
            raise OSError(errno.ENFILE, "Too many open files in system")
        return real(root, task_id, **kwargs)

    patched.setattr(artifacts, "store_actor_source_bytes", capture)


def outbox_busy(patched, trace, did, held):
    """The outbox is busy for this message's receipt, after its send fact was written."""
    real = td._capture_emitted_answer

    def capture(root, delivery_id, emitted):
        receipt = real(root, delivery_id, emitted)
        if delivery_id == did:
            held.enter_context(store.name_lock(trace.root, "terminal_deliveries.json.lock", timeout_sec=0))
        return receipt

    patched.setattr(td, "_capture_emitted_answer", capture)


@pytest.mark.parametrize("kept, failure", [("superseded", capture_fails), ("superseded", outbox_busy),
                                           ("exhausted", capture_fails)])
def test_a_send_landing_after_its_row_left_the_owed_set_gets_its_receipt_without_a_resend(chat, monkeypatch,
                                                                                         kept, failure):
    first = notice(chat)
    did = first["delivery_id"]
    owed_row = registry(chat.root)["pending"][did]
    receipt = stop_receipt(chat) if kept == "superseded" else None

    def meanwhile():  # while this copy is inside the transport, the outbox stops owing it
        if receipt is not None:
            assert delivery._handle_send_message(dict(receipt), chat.ctx)  # the Stop receipt lands
        else:
            for n in range(td._PENDING_CAP):  # newer owed messages take its slot
                notice(chat, task=f"n{n}", text=f"Newer answer {n}.", origin=TERMINAL_ORIGIN_MODEL_FINAL)

    chat.during[did] = meanwhile
    with contextlib.ExitStack() as held, monkeypatch.context() as patched:
        failure(patched, chat, did, held)
        deliver(chat, first)  # the send returned; its receipt could not be recorded
    row = registry(chat.root)[kept][did]
    assert row["emitted"]["text"] == NOTICE and row["emitted"]["chat_id"] == 1, "the returned send is kept"
    assert {key: row[key] for key in owed_row} == owed_row, "what was owed and earlier sends did stays as it was"
    restart(chat, monkeypatch)  # only the outbox remembers that send now
    assert did in sweep(chat), "the kept send fact is repaired into its receipt"
    data = registry(chat.root)
    assert chat.sends.count(did) == 1, "it is never sent again"
    assert did in data["delivered"] and data["receipts"][did]["text_chars"] == len(NOTICE)
    assert did not in data.get(kept, {}) and did not in data["pending"]
    assert len([row for row in chat_rows(chat.root) if row.get("text") == NOTICE]) == 1
    assert did not in sweep(chat) and chat.sends.count(did) == 1


PARENT_STATE = pathlib.Path(__file__).parent / "fixtures" / "nminus1" / "stop_receipt_ca791d980"


def parent_written(trace, scenario="incident", edit=None):
    """Install what the ca791d980 writer left (the generator is kept with this change's review evidence):
    its outbox (``state/terminal_deliveries.json``: the owed notice, the delivered Stop receipt) and
    the task result recording where and when that receipt went."""
    data = json.loads((PARENT_STATE / scenario / "outbox_registry.json").read_text(encoding="utf-8"))
    stored = json.loads((PARENT_STATE / scenario / "task_result_t1.json").read_text(encoding="utf-8"))
    (did, row), = data["pending"].items()
    if edit:
        edit(data, row, stored["cancel_receipt"])  # a control: one retained fact differs
    write_registry(trace.root, data)
    patch_document(trace.root, "t1", lambda _current: stored)
    return did, data["delivered"][0] if data["delivered"] else "", row


def test_a_stop_receipt_the_previous_writer_recorded_supersedes_on_the_first_sweep(chat, monkeypatch):
    did, cancel, row = parent_written(chat)
    restart(chat, monkeypatch)  # the upgraded generation starts from the parent's durable state
    assert sweep(chat) == [] and chat.sends == [], "the replaced notice is not sent again"
    data = registry(chat.root)
    kept = data["superseded"][did]
    assert without_mark(kept) == row and kept["superseded_by"] == cancel == superseded_by(chat.root, did)
    assert did not in data["pending"] and data["delivered"] == [cancel] and owed(chat.root) == set()
    (event,) = events(chat.root, "terminal_delivery_superseded")
    assert (event["delivery_id"], event["superseded_by"], event["basis"]) == (did, cancel, "task_result_cancel_receipt")
    restart(chat, monkeypatch)
    copy = {key: value for key, value in row.items() if key not in ("registered_at", "dispatch_attempts",
                                                                     "delivery_deferred", "transport_failed")}
    deliver(chat, {**copy, "type": "send_message"})  # a buffered copy of the same message
    assert sweep(chat) == [] and chat.sends == [] and registry(chat.root)["superseded"][did] == kept


def notice_row(edit):
    return lambda data, row, receipt: edit(row, receipt)


UNPROVEN = {
    "stop_not_recorded_as_delivered": lambda data, row, receipt: data.update(delivered=[]),
    "destination_unknown": notice_row(lambda row, receipt: receipt.pop("delivered_chat_id")),
    "another_chat": notice_row(lambda row, receipt: row.update(chat_id=2)),
    "owed_after_the_stop_request": notice_row(lambda row, receipt: row.update(registered_at=receipt["ts"])),
    "stop_request_time_unknown": notice_row(lambda row, receipt: receipt.pop("stop_requested_at")),
    "another_stop_episode": notice_row(lambda row, receipt: receipt.update(delivery_id="cancel:t1:ci_other")),
    "model_answer": notice_row(lambda row, receipt: row.update(terminal_origin=TERMINAL_ORIGIN_MODEL_FINAL)),
    "recorded_send": notice_row(lambda row, receipt: row.update(emitted={**row, "sent_at": row["ts"]})),
}


@pytest.mark.parametrize("kind", sorted(UNPROVEN))
def test_a_previous_writers_notice_without_every_retained_proof_stays_owed(chat, monkeypatch, kind):
    did, _cancel, row = parent_written(chat, edit=UNPROVEN[kind])
    restart(chat, monkeypatch)
    sweep(chat)
    assert "superseded" not in registry(chat.root) and events(chat.root, "terminal_delivery_superseded") == []
    if kind == "recorded_send":  # its send is repaired into a receipt, never resent
        assert chat.sends == [] and did in registry(chat.root)["delivered"]
    else:
        assert chat.sends == [did] and chat.routes[did] == row["chat_id"]


CHILD = {"progress_meta": {"subagent_task_id": "t1", "parent_task_id": "r1", "root_task_id": "r1",
                           "delegation_role": "subagent"}}  # a child's final names its lineage on its row


@pytest.mark.parametrize("project", ["no_project", "project_before_the_stop", "root_project_before_the_stop",
                                     "project_after_the_stop", "main_pinned_notice"])
def test_the_previous_writers_stop_supersedes_only_where_the_notice_goes_now(chat, monkeypatch, project):
    """The notice row kept its birth chat (1); the sender routes it by its task tree's binding now."""
    room = (to_project(chat, "r1") if project == "root_project_before_the_stop"
            else to_project(chat) if project in ("project_before_the_stop", "main_pinned_notice") else None)
    reached = room or 1  # where the previous writer's Stop went: the room the task already had, else chat 1
    fields = (CHILD if project == "root_project_before_the_stop"
              else {"system_type": MAIN_NOTICE_TYPE} if project == "main_pinned_notice" else {})

    def recorded(row, receipt):  # what the previous writer left: the owed row, and where its Stop went
        row.update(fields)
        receipt["delivered_chat_id"] = reached

    did, cancel, _row = parent_written(chat, edit=notice_row(recorded))
    if project == "project_after_the_stop":
        room = to_project(chat)  # the Stop reached chat 1; the room never showed it
    restart(chat, monkeypatch)  # the upgraded generation starts from the parent's durable state
    sweep(chat)
    if project in ("no_project", "project_before_the_stop", "root_project_before_the_stop"):
        assert chat.sends == [] and superseded_by(chat.root, did) == cancel
    else:  # the sender takes it where the Stop did not go: still owed there, and sent
        assert superseded_by(chat.root, did) == "" and chat.sends == [did]
        assert chat.routes[did] == (room if project == "project_after_the_stop" else 1)


def custody_notice(trace):
    """A host-notice final carrying its custody notice (an open delegated run), owed by the completion producer."""
    stored = {"result": NOTICE, "terminal_origin": TERMINAL_ORIGIN_HOST_NOTICE}
    event = td.build_completed_result_event(trace.root, {"chat_id": 1}, "t1", stored, unreconciled_runs=["run-1"])
    assert event["terminal_origin"] == TERMINAL_ORIGIN_HOST_NOTICE and "run-1" in event["terminal_custody_notice"]
    assert td.register_pending_delivery(trace.root, event)
    return event


@pytest.mark.parametrize("stop", ["recorded_now", "recorded_by_the_previous_writer", "split_before_the_stop"])
def test_a_superseded_notice_still_owes_its_separate_custody_notice_once(chat, monkeypatch, stop):
    if stop == "recorded_by_the_previous_writer":
        did, cancel, row = parent_written(chat, "unhandled_custody")
        restart(chat, monkeypatch)
        sweep(chat)  # supersedes it from the retained facts
        sent = []
    else:
        first = custody_notice(chat)
        did = first["delivery_id"]
        if stop == "split_before_the_stop":
            chat.mode[did] = "refused"
            deliver(chat, first)  # its handler owed and sent the custody row; the notice itself failed
        else:
            restart(chat, monkeypatch)  # its only copy was lost before any handler split it
        receipt = stop_receipt(chat)
        cancel = receipt["delivery_id"]
        deliver(chat, receipt)
        sent = chat.sends[:]
    custody = did + ":custody_notice"
    data = registry(chat.root)
    assert data["superseded"][did]["superseded_by"] == cancel
    assert data["superseded"][did]["terminal_custody_notice"].endswith("run-1"), "its payload stays as it was owed"
    assert (custody in data["pending"]) is (stop != "split_before_the_stop"), "owed in the write that replaced it"
    sweep(chat)
    assert chat.sends.count(custody) == 1 and did not in chat.sends[len(sent):], "the custody row lands once"
    (row,) = [row for row in chat_rows(chat.root) if row.get("text", "").endswith("run-1")]
    assert (row.get("direction"), row.get("type")) == ("system", "custody_notice"), "its own typed row"
    assert custody in registry(chat.root)["delivered"] and owed(chat.root) == set()
    assert sweep(chat) == [] and chat.sends.count(custody) == 1


def test_a_model_answer_carrying_a_custody_notice_is_never_superseded(chat, monkeypatch):
    stored = {"result": "The real answer.", "terminal_origin": TERMINAL_ORIGIN_MODEL_FINAL}
    answer = td.build_completed_result_event(chat.root, {"chat_id": 1}, "t1", stored, unreconciled_runs=["run-1"])
    assert answer["terminal_custody_notice"] and td.register_pending_delivery(chat.root, answer)
    restart(chat, monkeypatch)  # its only copy was lost before any handler split it
    receipt = stop_receipt(chat)
    deliver(chat, receipt)
    assert "superseded" not in registry(chat.root) and answer["delivery_id"] in owed(chat.root)
    sweep(chat)  # the answer is sent, and its handler splits and sends the custody row once
    custody = answer["delivery_id"] + ":custody_notice"
    assert chat.sends == [receipt["delivery_id"], answer["delivery_id"], custody]
    assert sweep(chat) == [] and owed(chat.root) == set()
