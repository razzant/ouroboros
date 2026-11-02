"""A relay's authority follows exact retained owner rows, independently of lane."""
import queue
from types import SimpleNamespace

import pytest

from ouroboros.chat_chain import format_address, row_address
from ouroboros.owner_relay import held_owner_messages, quote_rows, write_relay, select_return
from ouroboros import dialogue_store
from ouroboros.loop_round_limits import _drain_incoming_messages
from ouroboros.owner_mailbox import drain_owner_entries
from tests import _dialogue as dialogue_test


def source(tmp_path):
    row = {"chat_id": 1, "direction": "in", "ts": "2026-10-01T00:00:00Z", "text": "  exact\nowner words  ",
           "client_message_id": "owner-1", "attachments": []}
    dialogue_store.append(tmp_path, "chat", row)
    return row, format_address(row_address(row))


def context(tmp_path, row=None):
    from ouroboros.project_dialogue import build_owner_message_ref
    metadata = {} if row is None else {"origin_message_ref": build_owner_message_ref(
        chat_id=row["chat_id"], client_message_id=row["client_message_id"], ts=row["ts"], text=row["text"])}
    return SimpleNamespace(task_id="teller", drive_root=tmp_path, budget_drive_root=tmp_path,
                           current_chat_id=1, task_metadata=metadata, pending_events=[], event_queue=None,
                           drive_logs=lambda: tmp_path / "logs")


def test_only_held_rows_can_be_quoted_and_note_is_never_substituted(tmp_path, monkeypatch):
    from ouroboros.tools.control_relay import _tell_task
    row, handle = source(tmp_path)
    ctx = context(tmp_path, row)
    events = []
    monkeypatch.setattr("ouroboros.tools.control_routing._emit_and_wait_for_routing",
                        lambda ctx, evt: (events.append(evt) or "live", {"status": "delivered"}))
    assert "QUOTE_NOT_HELD" in _tell_task(ctx, "receiver", "note", [handle + "0"])
    assert not events
    _tell_task(ctx, "receiver", "my own note", [handle])
    assert events[0]["message"] == "my own note"
    assert events[0]["quotes"][0]["text"] == row["text"]
    assert events[0]["quotes"][0]["source"]["address"] == handle
    _tell_task(ctx, "receiver", "later note")
    assert events[1]["message"] == "later note" and events[1]["quotes"] == []


def test_atomic_relay_is_idempotent_and_note_never_enters_owner_corpus(tmp_path):
    row, handle = source(tmp_path)
    sender = context(tmp_path, row)
    quotes, error = quote_rows(sender, [handle])
    assert not error
    args = dict(relay_id="r1", note="MODEL NOTE", quotes=quotes, chat_id=1, teller_task_id="teller")
    assert write_relay(tmp_path, "receiver", **args)[1]
    assert not write_relay(tmp_path, "receiver", **args)[1]
    assert len(drain_owner_entries(tmp_path, "receiver")) == 1
    with pytest.raises(ValueError, match="different words"):
        write_relay(tmp_path, "receiver", **{**args, "note": "changed"})
    receiver = context(tmp_path)
    receiver.task_id = "receiver"
    messages = []
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, "receiver", None, set(), owner_ctx=receiver)
    assert [r["content"] for r in receiver._owner_directives] == [row["text"]]
    assert "MODEL NOTE" in str(messages)
    assert handle in held_owner_messages(receiver)
    assert select_return(receiver, "")[0]["source"]["chat_id"] == 1
    assert not drain_owner_entries(tmp_path, "receiver")


def test_cross_room_paused_target_keeps_mail_without_resuming_and_dedupes_generation(tmp_path, monkeypatch):
    from supervisor.steering import _handle_steer_task
    from supervisor import queue as q
    row, handle = source(tmp_path)
    quotes, _ = quote_rows(context(tmp_path, row), [handle])
    target = {"id": "receiver", "chat_id": 42, "drive_root": str(tmp_path), "_owner_hold": True}
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, PENDING=[target],
                          persist_queue_snapshot=lambda **kw: None)
    monkeypatch.setattr(q, "ACCEPTANCE_FENCES", {"receiver": {"status": "active", "owner_message_generation": 2}})
    event = {"target_task_id": "receiver", "message": "note", "quotes": quotes, "relay_id": "r1",
             "routing_token": "r1", "chat_id": 1, "client_message_id": "owner-1", "issuer": {"kind": "task"}}
    _handle_steer_task(event, ctx)
    _handle_steer_task(event, ctx)
    assert q.ACCEPTANCE_FENCES["receiver"]["owner_message_generation"] == 3
    assert ctx.PENDING == [target] and target["_owner_hold"] is True
    assert len(drain_owner_entries(tmp_path, "receiver")) == 1
    relays = [r for r in dialogue_test.chat_rows(tmp_path) if r.get("type") == "relay_entry"]
    assert len(relays) == 1 and relays[0]["text"] == "note"
    assert relays[0]["relay"]["text"] == row["text"]


def test_empty_return_with_nothing_open_is_ordinary_and_stays_ambiguous_among_several(tmp_path):
    """Grok filled reply_to_relay="" on an ordinary send from a call (27.10) and the owner's message was
    refused twice. With no open relay the empty selector means nothing; among several it is ambiguous."""
    from ouroboros.tools.control_runtime import _send_user_message
    ctx = context(tmp_path)
    assert _send_user_message(ctx, "reply", reply_to_relay="").startswith("OK:")
    assert ctx.pending_events[0]["chat_id"] == 1 and ctx.pending_events[0]["system_type"] == "proactive_message"
    relay = {"relay_id": "r", "source": {"chat_id": 1, "address": "h"}}
    ctx._open_relays = {"r": relay, "s": {**relay, "relay_id": "s"}}
    assert "RELAY_RETURN_UNRESOLVED" in _send_user_message(ctx, "reply", reply_to_relay="")
    assert len(ctx.pending_events) == 1


def test_return_survives_teller_end_and_refuses_ambiguous_or_hidden_sender(tmp_path):
    from ouroboros.tools.control_runtime import _send_user_message
    row, handle = source(tmp_path)
    recipient = context(tmp_path)
    recipient.current_chat_id = 42
    recipient.task_metadata = {"origin_suppressed": True}
    relay = {"relay_id": "r", "source": {"chat_id": 1, "address": handle}}
    recipient._open_relays = {"r": relay, "s": {**relay, "relay_id": "s"}}
    assert "RELAY_RETURN_UNRESOLVED" in _send_user_message(recipient, "answer", reply_to_relay="")
    assert not recipient.pending_events
    recipient.current_chat_id = 0
    assert "hidden" in _send_user_message(recipient, "answer", reply_to_relay="r")
    assert not recipient.pending_events and len(recipient._open_relays) == 2
    recipient.current_chat_id = 42
    assert _send_user_message(recipient, "answer", reply_to_relay="r").startswith("OK:")
    [event] = recipient.pending_events
    assert event["chat_id"] == 1 and event["system_type"] == "relay_reply"
    assert event["progress_meta"]["reply_to"]["snippet"] == "exact owner words"
    assert event["progress_meta"]["reply_to_relay"] == "r"
    assert "r" not in recipient._open_relays and "s" in recipient._open_relays


def test_saved_ack_updates_each_quoted_owner_row(tmp_path):
    from ouroboros.owner_relay import retain_owner_message
    from ouroboros.project_dialogue import latest_chat_annotations
    row, handle = source(tmp_path)
    sender = context(tmp_path, row)
    second = {**row, "client_message_id": "owner-2", "text": "second exact words"}
    retain_owner_message(sender, second)
    quotes, error = quote_rows(sender, [handle, format_address(row_address(second))])
    assert not error
    write_relay(tmp_path, "receiver", relay_id="both", note="context", quotes=quotes,
                chat_id=1, teller_task_id="teller")
    receiver = context(tmp_path)
    receiver.task_id = "receiver"
    events = queue.Queue()
    _drain_incoming_messages([], queue.Queue(), tmp_path, "receiver", events, set(), owner_ctx=receiver)
    annotations = latest_chat_annotations(tmp_path)
    assert all(annotations[key]["status"] == "read" for key in ("owner-1", "owner-2"))
    assert all(annotations[key]["target_task_ids"] == ["receiver"] for key in ("owner-1", "owner-2"))
    assert events.get_nowait()["relay_ack"]["relay_id"] == "both"


def test_retained_current_quotes_extend_owner_corpus_without_the_note(tmp_path):
    from ouroboros.task_results import write_task_result
    from ouroboros.owner_words import task_owner_words
    row, handle = source(tmp_path)
    quotes, _ = quote_rows(context(tmp_path, row), [handle])
    write_task_result(tmp_path, "receiver", "running", origin_message_text="Original task")
    write_relay(tmp_path, "receiver", relay_id="r", note="NOT OWNER", quotes=quotes,
                chat_id=1, teller_task_id="teller")
    receiver = context(tmp_path)
    receiver.task_id = "receiver"
    _drain_incoming_messages([], queue.Queue(), tmp_path, "receiver", None, set(), owner_ctx=receiver)
    words, absent = task_owner_words(tmp_path, "receiver")
    assert not absent and [r["text"] for r in words] == ["Original task", row["text"]]
    assert "NOT OWNER" not in str(words)


@pytest.mark.parametrize("changed", [False, True])
def test_quoted_file_is_staged_on_the_recipient_drive_with_its_original_hash(tmp_path, changed):
    from ouroboros.chat_uploads import store_upload, claim_refs, uploads_dir
    from ouroboros.owner_relay import retain_owner_message
    from ouroboros.tools.core import _read_file
    from ouroboros.tools.registry import ToolContext
    _path, ref = store_upload(b"exact attached words", "note.txt", data_dir=tmp_path)
    claim_refs([ref], tmp_path)
    row, _ = source(tmp_path)
    row["attachments"] = [ref]
    sender = context(tmp_path)
    retain_owner_message(sender, row)
    quotes, _ = quote_rows(sender, [format_address(row_address(row))])
    child = tmp_path / "child"
    child.mkdir()
    write_relay(child, "receiver", relay_id="file", note="context", quotes=quotes,
                chat_id=1, teller_task_id="teller", canonical_root=tmp_path)
    if changed:
        (uploads_dir(tmp_path) / ref["upload"]).write_bytes(b"substituted bytes")
    receiver = context(child)
    receiver.task_id = "receiver"
    messages = []
    _drain_incoming_messages(messages, queue.Queue(), child, "receiver", None, set(), owner_ctx=receiver)
    if changed:
        assert "rejected" in str(messages)
        assert "substituted bytes" not in str(messages)
    else:
        artifact = child / "task_results/artifacts/receiver/attachments" / ref["upload"]
        assert artifact.read_bytes() == b"exact attached words"
        reader = ToolContext(repo_dir=tmp_path, drive_root=child, task_id="receiver")
        assert "exact attached words" in _read_file(reader, "attachments/" + ref["upload"], root="artifact_store")
        assert "read_file" in str(messages)


def test_note_carries_the_tellers_summary_and_the_card_names_its_rooms(tmp_path, monkeypatch):
    """The teller writes one line for the owner; the saved row names the room the note came from and the work it was
    written for, by the title it had then (Main's note card reads the receiver from it)."""
    from ouroboros.owner_relay import publish_relay, relay_sender_facts
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.control_relay import _tell_task
    events = []
    monkeypatch.setattr("ouroboros.tools.control_routing._emit_and_wait_for_routing",
                        lambda ctx, evt: (events.append(evt) or "live", {"status": "delivered"}))
    _tell_task(context(tmp_path), "receiver", "a0e1:research-done,no-safe-restore", summary="  Research is done; no safe fix yet  ")
    assert events[0]["summary"] == "Research is done; no safe fix yet"
    entry, _ = write_relay(tmp_path, "receiver", relay_id="s1", note=events[0]["message"], quotes=[], chat_id=1,
                           teller_task_id="teller", summary=events[0]["summary"])
    assert entry["relay"]["summary"] == "Research is done; no safe fix yet"
    assert "summary" not in write_relay(tmp_path, "receiver", relay_id="s2", note="plain", quotes=[], chat_id=1,
                                        teller_task_id="teller")[0]["relay"], "no summary, no key: older readers see today's shape"
    write_task_result(tmp_path, "teller", "running", origin_message_text="Teller work")
    write_task_result(tmp_path, "receiver", "running", origin_message_text="Receiver work", title="Receiver title",
                      suggested_name="Receiver work")
    sent = []
    publish_relay(tmp_path, "receiver", entry, SimpleNamespace(send_message=lambda chat_id, text, **kw: sent.append(kw)))
    relay = sent[0]["progress_meta"]["relay"]
    assert relay["summary"] == "Research is done; no safe fix yet"
    assert relay["sender"] == {"task_id": "teller", "chat_id": 1, "name": "Main"}
    assert relay["target"] == {"task_id": "receiver", "title": "Receiver work"}, "the work's display name, as its card shows it"
    saved = next(found.row for found in dialogue_store.rows_by_client_id(tmp_path, entry["msg_id"]))
    assert saved["relay"]["sender"]["name"] == "Main", "history draws the same card as the live frame"
    assert saved["relay"]["target"] == relay["target"]
    assert relay_sender_facts(tmp_path, "ghost") == {} and relay_sender_facts(tmp_path, "") == {}
    # A row saved before the stamp (or by an older install) publishes as it is: no backfill, no invented receiver.
    older, _ = write_relay(tmp_path, "receiver", relay_id="s3", note="older", quotes=[], chat_id=1, teller_task_id="teller")
    log_chat = __import__("supervisor.message_bus", fromlist=["log_chat"]).log_chat
    stored = log_chat("out", 1, 0, older["text"], ts=older["ts"], task_id="receiver", record_type="relay_entry",
                      client_message_id=older["msg_id"], message_meta={"relay": older["relay"]}, drive_root=tmp_path,
                      require_write=True)
    assert "target" not in stored["relay"]
    sent.clear()
    publish_relay(tmp_path, "receiver", older, SimpleNamespace(send_message=lambda chat_id, text, **kw: sent.append(kw)))
    assert "target" not in sent[0]["progress_meta"]["relay"] and "sender" not in sent[0]["progress_meta"]["relay"]
    # A work without a readable name is stamped by id only: the card says `this work`, never the id or the text.
    nameless, _ = write_relay(tmp_path, "nameless", relay_id="s4", note="to a nameless work", quotes=[], chat_id=1,
                              teller_task_id="teller")
    sent.clear()
    publish_relay(tmp_path, "nameless", nameless, SimpleNamespace(send_message=lambda chat_id, text, **kw: sent.append(kw)))
    assert sent[0]["progress_meta"]["relay"]["target"] == {"task_id": "nameless"}


def test_a_note_from_a_task_turned_into_a_project_returns_to_that_project(tmp_path, monkeypatch):
    """F1: a running task bound after admission keeps its old chat until readmission; its note's return
    room is its room by the room rule now, named on the card, never the stale admitted chat."""
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.control_relay import _tell_task
    alpha = create_project(tmp_path, "alpha", name="Alpha")
    write_task_result(tmp_path, "teller", "running", chat_id=1, title="teller")
    bind_task_to_project(tmp_path, "teller", alpha["id"], alpha["chat_id"], origin={"absent": "system"})
    events = []
    monkeypatch.setattr("ouroboros.tools.control_routing._emit_and_wait_for_routing",
                        lambda ctx, evt: (events.append(evt) or "live", {"status": "delivered"}))
    moved = context(tmp_path)  # current_chat_id stays 1: the admitted chat
    _tell_task(moved, "receiver", "Please give me your result.")
    assert events[0]["chat_id"] == alpha["chat_id"]
    plain = context(tmp_path)
    plain.task_id = "main-teller"  # an unbound Main teller still returns to Main
    _tell_task(plain, "receiver", "A note.")
    assert events[1]["chat_id"] == 1
    entry, _ = write_relay(tmp_path, "receiver", relay_id="r1", note="n", quotes=[], chat_id=alpha["chat_id"],
                           teller_task_id="teller")
    assert entry["relay"]["source"]["chat_id"] == alpha["chat_id"] and entry["relay"]["source"]["label"] == "Alpha"


def test_a_relayed_owner_reply_keeps_what_it_replied_to(tmp_path):
    """F2: the owner's elliptical Reply reaches the told work with the message it answers, labelled as a host
    reference beside the verbatim owner words, never as owner words."""
    from ouroboros.chat_references import reference_from_row
    from ouroboros.owner_relay import inject_relay, owner_quote, relay_envelope
    notice = {"chat_id": 1, "direction": "out", "type": "main_notice", "task_id": "teller",
              "ts": "2026-10-01T00:00:00Z", "text": "A specific project observation."}
    dialogue_store.append(tmp_path, "chat", notice)
    owner = {"chat_id": 1, "direction": "in", "ts": "2026-10-01T00:01:00Z", "client_message_id": "owner-reply",
             "text": "Why is that here?", "reply_to": reference_from_row(notice)}
    dialogue_store.append(tmp_path, "chat", owner)
    target = format_address(row_address(notice))
    quote = owner_quote(tmp_path, owner)
    assert quote["reply_to"]["address"] == target
    envelope = relay_envelope("quoted", [quote], chat_id=1)
    assert envelope["quotes"][0]["reply_to"]["snippet"] == "A specific project observation."
    receiver, messages = context(tmp_path), []
    receiver.task_id = "receiver"
    inject_relay(receiver, {"kind": "relay_entry", "text": "Please answer the owner.", "relay": envelope,
                            "owner_rows": [owner], "source_task_id": "teller"}, messages)
    words = next(str(m["content"]) for m in messages if "Owner words quoted verbatim" in str(m["content"]))
    assert "Why is that here?" in words and target in words and "A specific project observation." in words
    assert words.index("Why is that here?") < words.index("[Host reference") and "not the owner's words" in words
    unrelated = {**owner, "client_message_id": "plain", "ts": "2026-10-01T00:02:00Z", "text": "Plain."}
    unrelated.pop("reply_to")
    assert "reply_to" not in owner_quote(tmp_path, unrelated)  # nothing invented for an ordinary message


def test_a_relay_answer_is_owed_before_the_queue_and_survives_a_failed_send(tmp_path, monkeypatch):
    """F4: the named return is owed in the terminal outbox under a stable id before the queue sees it; a
    send that fails after enqueue leaves it owed (deferred), and the receipt never claims it was sent."""
    from supervisor import events_chat_delivery as delivery, terminal_delivery
    from ouroboros.tools.control_runtime import _send_user_message
    row, handle = source(tmp_path)
    q = queue.Queue()
    recipient = context(tmp_path)
    recipient.task_id, recipient.event_queue = "recipient", q
    recipient._open_relays = {"once": {"relay_id": "once", "source": {"chat_id": 1, "address": handle, "label": "Main"}}}
    receipt = _send_user_message(recipient, "Response after relay", reply_to_relay="once")
    assert receipt.startswith("OK: answer to relay once owed to Main") and " sent " not in receipt
    queued = q.get_nowait()
    assert queued["delivery_id"] == "relay-return:once:recipient" and not recipient._open_relays

    def owed():
        return {row["delivery_id"]: row for row in terminal_delivery.pending_deliveries(tmp_path)}["relay-return:once:recipient"]
    assert owed()["text"] == "Response after relay" and owed()["chat_id"] == 1
    failing = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, append_jsonl=lambda *a, **k: None,
                              send_with_budget=lambda *a, **k: (_ for _ in ()).throw(OSError("injected sender outage")))
    assert delivery._handle_send_message(queued, failing) is False
    assert owed().get("delivery_deferred") is True and owed()["system_type"] == "relay_reply"


def test_an_unrecordable_relay_answer_sends_nothing_and_keeps_the_relay_open(tmp_path, monkeypatch):
    from ouroboros.tools.control_runtime import _send_user_message
    row, handle = source(tmp_path)
    q = queue.Queue()
    recipient = context(tmp_path)
    recipient.task_id, recipient.event_queue = "recipient", q
    recipient._open_relays = {"once": {"relay_id": "once", "source": {"chat_id": 1, "address": handle, "label": "Main"}}}
    monkeypatch.setattr("supervisor.terminal_delivery.register_pending_delivery", lambda *_a, **_k: False)
    receipt = _send_user_message(recipient, "Response after relay", reply_to_relay="once")
    assert "RELAY_RETURN_NOT_OWED" in receipt and q.empty() and "once" in recipient._open_relays


def test_quoteless_relay_to_child_drive_labels_source_from_canonical_registry(tmp_path):
    from ouroboros.projects_registry import create_project

    canonical = tmp_path / "canonical"
    child = tmp_path / "child"
    project = create_project(canonical, "nova", name="Nova")
    args = dict(relay_id="cross-drive", note="A project note", quotes=[], chat_id=project["chat_id"],
                teller_task_id="teller", canonical_root=canonical)
    entry, created = write_relay(child, "receiver", **args)
    assert created
    assert entry["relay"]["source"]["label"] == "Nova"
    assert entry["relay"]["source"]["chat_id"] == project["chat_id"]
    assert entry["relay_canonical_root"] == str(canonical)
    assert write_relay(child, "receiver", **args) == (entry, False)
    assert len(drain_owner_entries(child, "receiver")) == 1
