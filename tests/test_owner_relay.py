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


def test_revoked_place_owner_row_is_not_quoted_or_relayed(tmp_path):
    """A held place row is rechecked when the account is revoked before delivery."""
    from ouroboros import places
    from ouroboros.owner_relay import inject_relay
    from tests.test_place_owner_author import _place_row

    connected = places.connect_place(tmp_path, connector="bridge", provider="chat", account_id="qa",
                                     label="Team chat")
    places.edit_place(tmp_path, connected.place_id, "confirm_owner_account", author="owner",
                      platform_actor_id="anton", account_id="qa")
    row = _place_row(connected.place_id)
    dialogue_store.append(tmp_path, "chat", row)
    stored = next(entry.row for entry in dialogue_store.iter_rows(tmp_path, "chat"))
    handle = format_address(row_address(stored))
    sender = context(tmp_path, stored)
    assert handle in held_owner_messages(sender)

    places.edit_place(tmp_path, connected.place_id, "revoke_owner_account", author="owner",
                      platform_actor_id="anton", account_id="qa")
    assert quote_rows(sender, [handle])[0] == []

    receiver = context(tmp_path)
    receiver.task_id = "receiver"
    inject_relay(receiver, {"relay": {"relay_id": "r1", "quotes": []},
                            "owner_rows": [stored], "source_task_id": "sender", "text": "note"}, [])
    assert not [row for row in getattr(receiver, "_owner_directives", []) if row.get("source") == "owner_mailbox"]
    assert not held_owner_messages(receiver)


def test_cross_root_relay_validates_source_place_but_holds_on_recipient_root(tmp_path):
    """A relay keeps source-place authority when sender and recipient use different data roots."""
    from ouroboros import places
    from ouroboros.owner_relay import inject_relay
    from tests.test_place_owner_author import _place_row

    source_root, recipient_root = tmp_path / "source", tmp_path / "recipient"
    source_root.mkdir()
    recipient_root.mkdir()
    connected = places.connect_place(source_root, connector="bridge", provider="chat", account_id="qa",
                                     label="Source chat")
    places.edit_place(source_root, connected.place_id, "confirm_owner_account", author="owner",
                      platform_actor_id="anton", account_id="qa")
    row = _place_row(connected.place_id)
    dialogue_store.append(source_root, "chat", row)
    stored = next(entry.row for entry in dialogue_store.iter_rows(source_root, "chat"))
    sender = context(source_root, stored)
    handle = format_address(row_address(stored))
    quotes, error = quote_rows(sender, [handle])
    assert not error
    relay, created = write_relay(recipient_root, "receiver", relay_id="cross", note="note", quotes=quotes,
                                 chat_id=1, teller_task_id="sender", canonical_root=source_root)
    assert created

    receiver = context(recipient_root)
    receiver.task_id = "receiver"
    inject_relay(receiver, relay, [])
    assert [row.get("content") for row in receiver._owner_directives] == [stored["text"]]
    assert handle in held_owner_messages(receiver)

    third_root = tmp_path / "third"
    third_root.mkdir()
    forwarded, error = quote_rows(receiver, [handle])
    assert not error and forwarded[0]["text"] == stored["text"]
    forwarded_entry, created = write_relay(third_root, "receiver", relay_id="cross-again", note="note",
                                            quotes=forwarded, chat_id=1, teller_task_id="receiver",
                                            canonical_root=recipient_root)
    assert created
    third = context(third_root)
    third.task_id = "receiver"
    inject_relay(third, forwarded_entry, [])
    assert [row.get("content") for row in third._owner_directives] == [stored["text"]]


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


@pytest.mark.parametrize("shape", ["quiz_row", "late_answer"])
def test_a_relayed_answer_carries_the_card_it_answers(tmp_path, shape):
    """Audit 05.11 item 37: a quoted answer (a quiz row, or the owner row a late answer's head holds) reaches the told
    work with its card as the message it replies to, in the asker's name, beside my human's words."""
    from ouroboros.owner_relay import inject_relay, owner_quote, relay_envelope
    card = {"quiz_id": "q7", "question": "Publish exactly this reply?", "state": "answered", "answered_index": 0,
            "options": ["Publish it", "Not now"]}
    answer = {"chat_id": 1, "direction": "in", "ts": "2026-10-01T00:01:00Z", "client_message_id": "owner-answer",
              "text": "1. Publish it", **({"type": "quiz_answer", "task_id": "asker", "quiz": card} if shape == "quiz_row"
                                           else {"late_answer": {"task_id": "asker", "quiz_id": "q7", "answer_snapshot": {
                                               "task_id": "asker", "quiz": {**card, "answered_after_terminal": True}}}})}
    dialogue_store.append(tmp_path, "chat", answer)
    receiver, messages = context(tmp_path), []
    receiver.task_id = "receiver"
    inject_relay(receiver, {"kind": "relay_entry", "text": "Please act on it.", "owner_rows": [answer],
                            "relay": relay_envelope("quoted", [owner_quote(tmp_path, answer)], chat_id=1),
                            "source_task_id": "teller"}, messages)
    [row] = [row for row in receiver._owner_directives if row["source"] == "owner_mailbox"]
    assert "Publish exactly this reply?" not in str(row["content"])
    assert row["reply_to"] == {"address": "quiz q7 of task asker", "author": "Ouroboros, task asker",
                               "text": "Publish exactly this reply?\n1. Publish it\n2. Not now\n"
                                       "(host record: my human pressed option 1)"}
    words = next(str(m["content"]) for m in messages if "Owner words quoted verbatim" in str(m["content"]))
    assert "[Host question frame]" in words and "Publish exactly this reply?" in words  # the model reads it too


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


def test_pending_text_replay_merges_late_consciousness_origin(tmp_path, monkeypatch):
    from supervisor import terminal_delivery
    from ouroboros import terminal_projection

    monkeypatch.setattr(terminal_projection, "main_delivery_current", lambda *_a, **_k: True)
    pending_deliveries = terminal_delivery.pending_deliveries
    register_pending_delivery = terminal_delivery.register_pending_delivery

    event = {"type": "send_message", "delivery_id": "late-origin", "task_id": "wake",
             "chat_id": 1, "text": "wake", "terminal_projection_token": "token"}
    assert register_pending_delivery(tmp_path, event)
    stamped = {**event, "initiator": "consciousness",
               "progress_meta": {"initiator": "consciousness"}}
    assert register_pending_delivery(tmp_path, stamped)
    row = next(row for row in pending_deliveries(tmp_path) if row["delivery_id"] == "late-origin")
    assert row["initiator"] == "consciousness"
    assert row["progress_meta"]["initiator"] == "consciousness"


def test_pending_reregistration_without_deferral_keeps_original_timestamp(tmp_path, monkeypatch):
    from supervisor.terminal_delivery import pending_deliveries, register_pending_delivery

    first = {"type": "send_message", "delivery_id": "stable-ts", "task_id": "wake", "chat_id": 1,
             "text": "wake", "ts": "2026-11-05T00:00:00+00:00"}
    assert register_pending_delivery(tmp_path, first)
    assert register_pending_delivery(tmp_path, {**first, "ts": None})
    row = next(row for row in pending_deliveries(tmp_path) if row["delivery_id"] == "stable-ts")
    assert row["ts"] == first["ts"]


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


def _carried_entry(canonical, drive, member, quote, *, went_to="", teller="root", earlier_ref=""):
    """One host-carried relay entry as the carriage writes it (``owner_relay._carry_to_member``)."""
    address = quote["source"]["address"]
    return write_relay(drive, member, relay_id=f"{'carry-context' if went_to else 'carry'}:{address}", note="",
                       quotes=[quote], chat_id=1, teller_task_id=teller, canonical_root=canonical,
                       host_carried={"went_to": went_to, **({"earlier_ref": earlier_ref} if earlier_ref else {})})


def _helper(drive, task_id="helper"):
    from ouroboros.tools.registry import ToolContext
    ctx = ToolContext(repo_dir=drive, drive_root=drive, task_id=task_id)
    ctx.task_metadata = {"delegation_role": "subagent", "root_task_id": "root", "parent_task_id": "root"}
    ctx._owner_directives = []
    return ctx


def test_a_carried_entry_renders_the_carried_line_and_holds_the_row_outside_corpus_return_and_receipt(tmp_path):
    """A row the host carries is my human's words shown once under the carried line, held with its fact, its file
    staged on the member's drive; it enters no corpus, opens no return relay, posts no receipt and moves no owner
    hash. A helper gets no handles fact and no runtime handle list; a root with its own row prints both."""
    from ouroboros.chat_uploads import claim_refs, store_upload
    from ouroboros.context import build_runtime_section
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_WORDS_KIND
    from ouroboros.loop_messages import owner_source_sha256
    from ouroboros.owner_relay import owner_quote
    from tests._context_shared import _make_health_env
    import json
    _path, ref = store_upload(b"carried attached words", "plan.txt", data_dir=tmp_path)
    claim_refs([ref], tmp_path)
    row, address = source(tmp_path)
    row["attachments"] = [ref]
    child = tmp_path / "child"
    child.mkdir()
    entry, created = _carried_entry(tmp_path, child, "helper", owner_quote(tmp_path, row, address=address))
    assert created and entry["host_carried"] == {"went_to": ""} and entry["msg_id"] == f"relay:carry:{address}:helper"
    helper = _helper(child)
    hash_before = owner_source_sha256(helper)
    events, messages = queue.Queue(), [{"role": "user", "content": "Your assignment."}]
    controls = _drain_incoming_messages(messages, queue.Queue(), child, "helper", events, set(), owner_ctx=helper)
    assert controls["owner_carried_addressed"] == [address]
    [shown] = messages[1:]
    line = f"[Owner words quoted verbatim; {row['ts']}; Main; carried to this work by the host (reached this work); handle {address}]"
    assert str(shown["content"]).startswith(f"{line}\n{row['text']}\n") and shown[HOST_CONTEXT_KIND_KEY] == OWNER_WORDS_KIND
    assert "[Message from Ouroboros" not in str(shown) and "owner_quote_handles" not in str(messages)
    assert helper._owner_directives == [] and owner_source_sha256(helper) == hash_before
    assert not getattr(helper, "_open_relays", None) and events.empty() and not drain_owner_entries(child, "helper")
    assert held_owner_messages(helper)[address]["carried"] == {"from": "root", "went_to": "", "at": entry["ts"]}
    assert held_owner_messages(helper)[address]["text"] == row["text"]
    assert (child / "task_results/artifacts/helper/attachments" / ref["upload"]).read_bytes() == b"carried attached words"
    assert "read_file" in str(shown["content"])
    env = _make_health_env(tmp_path / "env")
    section = json.loads(build_runtime_section(env, {"id": "helper", "metadata": helper.task_metadata}, ctx=helper).split("\n\n", 1)[1])
    assert "owner_quote_handles" not in section
    # The other direction: a context entry names where the row went and is no closing control; a root prints both facts.
    other = tmp_path / "other"
    other.mkdir()
    _carried_entry(tmp_path, other, "helper", owner_quote(tmp_path, {**row, "attachments": []}, address=address), went_to="sibling")
    peer, peer_messages = _helper(other), []
    assert "owner_carried_addressed" not in _drain_incoming_messages(peer_messages, queue.Queue(), other, "helper", None, set(), owner_ctx=peer)
    assert "carried to this work by the host (went to sibling)" in str(peer_messages[0]["content"])
    root, root_messages = context(tmp_path, row), []
    held_owner_messages(root)  # the turn holds its own row
    _drain_incoming_messages(root_messages, queue.Queue(), tmp_path, "teller", None, set(), owner_ctx=root)
    assert "owner_quote_handles held by this turn" in str(root_messages) and address in str(root_messages)
    section = json.loads(build_runtime_section(env, {"id": "teller", "metadata": root.task_metadata}, ctx=root).split("\n\n", 1)[1])
    assert section["owner_quote_handles"] == [address]


def test_one_address_is_one_entry_and_a_row_held_under_the_same_fact_renders_nothing(tmp_path):
    """Two drains of a root write the member identical bytes under one id, whichever name the project has by then;
    an entry delivered again (a crash before its ack) renders nothing more; a context entry after the addressed one
    renders nothing; an addressed entry after a context one flips the fact and is shown once."""
    from ouroboros.owner_relay import carry_to_members, inject_relay, owner_quote, retain_owner_message
    from ouroboros.projects_registry import create_project, update_project
    from ouroboros.task_results import write_task_result
    project = create_project(tmp_path, "alpha", name="Alpha")
    row = {"chat_id": project["chat_id"], "direction": "in", "ts": "2026-10-01T00:00:00Z", "text": "exact words",
           "client_message_id": "owner-1"}
    dialogue_store.append(tmp_path, "chat", row)
    address = format_address(row_address(row))
    write_task_result(tmp_path, "member", "running", delegation_role="subagent", parent_task_id="root", root_task_id="root")
    root = context(tmp_path)
    root.task_id = "root"
    retain_owner_message(root, row)
    carry_to_members(root, delivered=True)
    update_project(tmp_path, project["id"], name="Beta")
    carry_to_members(root, delivered=True)  # the member holds the id: no second write, whatever the room is called now
    [entry] = drain_owner_entries(tmp_path, "member")
    assert entry["msg_id"] == f"relay:carry:{address}:member" and entry["source_task_id"] == "root"
    assert entry["relay"]["source"]["label"] == "Alpha" and entry["host_carried"] == {"went_to": ""}
    quote = owner_quote(tmp_path, row, address=address)
    member, messages = _helper(tmp_path, "member"), []
    assert inject_relay(member, entry, messages) == [address] and len(messages) == 1
    assert inject_relay(member, entry, messages) == [] and len(messages) == 1  # delivered again: no second line
    context_entry, _ = _carried_entry(tmp_path, tmp_path, "member", quote, went_to="sibling")
    assert inject_relay(member, context_entry, messages) == [] and len(messages) == 1
    assert held_owner_messages(member)[address]["carried"]["went_to"] == ""  # never downgraded to context
    # The other direction: a row held as context, addressed since, is shown once more and its fact flips.
    second = {**row, "client_message_id": "owner-2", "text": "second words", "ts": "2026-10-01T00:02:00Z"}
    dialogue_store.append(tmp_path, "chat", second)
    second_address = format_address(row_address(second))
    second_quote = owner_quote(tmp_path, second, address=second_address)
    as_context, _ = _carried_entry(tmp_path, tmp_path, "member", second_quote, went_to="sibling")
    assert inject_relay(member, as_context, messages) == [] and len(messages) == 2
    assert "went to sibling" in str(messages[1]["content"])
    assert held_owner_messages(member)[second_address]["carried"]["went_to"] == "sibling"
    addressed, _ = _carried_entry(tmp_path, tmp_path, "member", second_quote)
    assert inject_relay(member, addressed, messages) == [second_address] and len(messages) == 3
    assert "reached this work" in str(messages[2]["content"])
    assert held_owner_messages(member)[second_address]["carried"]["went_to"] == ""
    assert inject_relay(member, as_context, messages) == [] and len(messages) == 3


def test_a_failed_write_stops_no_other_member_and_the_next_drain_writes_it_once(tmp_path, monkeypatch):
    """The carriage keeps no delivery state: a write that raises for one member leaves the others written, the next
    drain (of a restarted drainer, with nothing kept in the process) writes exactly the missing entry, a root's
    quiet drain writes a grandchild born below a settled child, and a quiet drain of a helper reads no member while
    a drain that delivered, scheduled a helper or follows a failed write runs the pass."""
    from ouroboros import owner_relay
    from ouroboros.owner_relay import carry_to_members, retain_owner_message
    from ouroboros.task_results import write_task_result
    row, address = source(tmp_path)
    root = context(tmp_path, row)
    root.task_id = "root"
    retain_owner_message(root, row)
    for member in ("a", "b"):
        write_task_result(tmp_path, member, "running", delegation_role="subagent", parent_task_id="root", root_task_id="root")
    write_task_result(tmp_path, "done", "completed", delegation_role="subagent", parent_task_id="root", root_task_id="root")
    real = owner_relay.write_relay

    def failing(drive, task_id, **kwargs):
        if task_id == "a":
            raise OSError("injected mailbox outage")
        return real(drive, task_id, **kwargs)

    monkeypatch.setattr(owner_relay, "write_relay", failing)
    carry_to_members(root, delivered=True)
    assert not drain_owner_entries(tmp_path, "a") and len(drain_owner_entries(tmp_path, "b")) == 1
    assert not drain_owner_entries(tmp_path, "done") and root._carriage_memo["failed"] is True
    monkeypatch.setattr(owner_relay, "write_relay", real)
    restarted = context(tmp_path, row)  # the drainer restarted: a fresh context, no memo, the same retained row
    restarted.task_id = "root"
    retain_owner_message(restarted, row)
    carry_to_members(restarted, delivered=False)
    assert [entry["msg_id"] for entry in drain_owner_entries(tmp_path, "a")] == [f"relay:carry:{address}:a"]
    assert len(drain_owner_entries(tmp_path, "b")) == 1 and restarted._carriage_memo["failed"] is False
    monkeypatch.setattr(owner_relay, "write_relay", lambda *a, **k: (_ for _ in ()).throw(AssertionError("written twice")))
    carry_to_members(root, delivered=True)  # the process that failed, a drain that delivered: nothing is missing
    assert root._carriage_memo["failed"] is False
    # A grandchild born below a child that settled since the last pass: the root's quiet drain writes it, once.
    monkeypatch.setattr(owner_relay, "write_relay", real)
    write_task_result(tmp_path, "b", "completed", delegation_role="subagent", parent_task_id="root", root_task_id="root")
    write_task_result(tmp_path, "grand", "running", delegation_role="subagent", parent_task_id="b", root_task_id="root")
    carry_to_members(root, delivered=False)
    assert [entry["msg_id"] for entry in drain_owner_entries(tmp_path, "grand")] == [f"relay:carry:{address}:grand"]
    assert len(drain_owner_entries(tmp_path, "b")) == 1 and root._carriage_memo["members"] == {"a", "grand"}
    carry_to_members(root, delivered=False)
    assert len(drain_owner_entries(tmp_path, "grand")) == 1
    # The skip rule at a helper, both ways: a quiet drain reads no member; delivery, a new helper or a failed write does.
    reads = []
    monkeypatch.setattr("ouroboros.task_status.find_child_tasks", lambda *a, **k: reads.append(k) or [])
    helper = context(tmp_path, row)
    helper.task_id, helper.task_metadata = "b", {"delegation_role": "subagent", "root_task_id": "root"}
    carry_to_members(helper, delivered=False)  # the attempt's first pass reads
    carry_to_members(helper, delivered=False)
    assert len(reads) == 1 and reads[0]["scope"] == "direct"
    carry_to_members(helper, delivered=True)
    helper._last_scheduled_subagents = [{"task_id": "c"}]
    carry_to_members(helper, delivered=False)
    helper._carriage_memo["failed"] = True
    carry_to_members(helper, delivered=False)
    carry_to_members(helper, delivered=False)
    assert len(reads) == 4


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


@pytest.mark.parametrize("source_count", [1, 2])
@pytest.mark.parametrize("revoked", [False, True])
def test_forwarded_rows_keep_each_source_through_pending_read_delivery_and_continue(tmp_path, source_count, revoked):
    """A→B→C may combine sources. Each row keeps its own authority, full reply and attached bytes at every hop."""
    from ouroboros import places, owner_words
    from ouroboros.chat_references import reference_from_row
    from ouroboros.chat_uploads import store_upload, claim_refs
    from ouroboros.owner_continue import owner_sources, owner_corpus_rows, declared_inputs
    from ouroboros.owner_relay import inject_relay, owner_row_address
    from tests.test_place_owner_author import _place_row

    middle_root, final_root = tmp_path / "middle", tmp_path / "final"
    middle, rows, roots, handles, uploads = context(middle_root), [], [], [], []
    for index in range(source_count):
        root = tmp_path / f"source-{index}"
        place = places.connect_place(root, connector="bridge", provider="chat", account_id="qa", label="Source")
        places.edit_place(root, place.place_id, "confirm_owner_account", author="owner",
                          platform_actor_id="anton", account_id="qa")
        target = {"chat_id": 1, "direction": "out", "ts": "2026-09-30T05:40:00Z",
                  "text": f"Proposal {index}. " * 100, "client_message_id": "proposal"}
        dialogue_store.append(root, "chat", target)
        _path, upload = store_upload(f"attachment {index}".encode(), "note.txt", data_dir=root)
        claim_refs([upload], root)
        row = _place_row(place.place_id, text=f"owner words {index}", client_message_id=f"owner-{index}",
                         reply_to=reference_from_row(target), attachments=[upload])
        dialogue_store.append(root, "chat", row)
        stored = next(item.row for item in dialogue_store.rows_by_client_id(root, f"owner-{index}"))
        handle = owner_row_address(stored)
        quotes, error = quote_rows(context(root, stored), [handle])
        assert not error
        entry, _ = write_relay(middle_root, middle.task_id, relay_id=f"first-{index}", note="context", quotes=quotes,
                                chat_id=1, teller_task_id="first", canonical_root=root)
        inject_relay(middle, entry, [])
        rows.append((stored, target)); roots.append((root, place)); handles.append(handle); uploads.append(upload)
    quotes, error = quote_rows(middle, handles)
    assert not error and len(quotes) == source_count
    entry, _ = write_relay(final_root, "final", relay_id="forwarded", note="context", quotes=quotes,
                            chat_id=1, teller_task_id=middle.task_id, canonical_root=middle_root)
    if revoked:
        root, place = roots[0]
        places.edit_place(root, place.place_id, "revoke_owner_account", author="owner",
                          platform_actor_id="anton", account_id="qa")
    expected = rows[1:] if revoked else rows
    pending, _ = owner_words.task_owner_words(final_root, "final", task={"id": "final"})
    assert [r["text"] for r in pending if r["source"] != owner_words.REPLY_TARGET] == [r[0]["text"] for r in expected]
    assert [r["text"] for r in pending if r["source"] == owner_words.REPLY_TARGET] == [r[1]["text"] for r in expected]
    record = {"origin_message_text": "Original work", "origin_message_ref": {},
              "owner_mailbox": {"rows": [], "read_complete": True}}
    before_drain = owner_sources(final_root, record, "final")
    inputs = declared_inputs("final", record, before_drain)
    assert len(inputs) == len(expected)
    for item, index in zip(inputs, range(1 if revoked else 0, source_count)):
        assert item["status"] == "staged"
        from pathlib import Path
        assert Path(item["abs_path"]).read_bytes() == f"attachment {index}".encode()
    final = context(final_root); final.task_id = "final"
    events = queue.Queue()
    _drain_incoming_messages([], queue.Queue(), final_root, "final", events, set(), owner_ctx=final)
    assert [r["content"] for r in getattr(final, "_owner_directives", [])] == [r[0]["text"] for r in expected]
    if expected:
        kept_return, error = select_return(final, "forwarded")
        assert not error
        assert [q["source"]["address"] for q in kept_return["quotes"]] == handles[1 if revoked else 0:]
        assert kept_return["source"]["address"] == handles[1 if revoked else 0]
    else:
        assert not select_return(final, "forwarded")[0]
    for index in range(1 if revoked else 0, source_count):
        delivered = final_root / "task_results/artifacts/final/attachments" / uploads[index]["upload"]
        assert delivered.read_bytes() == f"attachment {index}".encode()
    continued = owner_sources(final_root, {"origin_message_text": "Original work", "origin_message_ref": {},
                                         "owner_mailbox": {"rows": [], "read_complete": True}}, "final")
    assert not continued["gaps"]
    assert [r["content"] for r in continued["later"]] == [r[0]["text"] for r in expected]
    assert [r["reply_to"]["text"] for r in continued["later"]] == [r[1]["text"] for r in expected]
    assert [r["content"] for r in owner_corpus_rows(continued)][1:] == [r[0]["text"] for r in expected]
    if not revoked:
        from ouroboros.project_dialogue import chat_annotation_receipt
        from ouroboros.owner_relay import publish_relay_ack
        for index, (root, _place) in enumerate(roots):
            assert chat_annotation_receipt(root, f"owner-{index}", "forwarded")["status"] == "read"
        saved_ack = events.get_nowait()["relay_ack"]
        sent = []
        publish_relay_ack(final_root, "final", saved_ack, SimpleNamespace(send_routing_ack=lambda *a, **kw: sent.append(kw)))
        assert {row["client_message_id"] for row in sent} == {f"owner-{i}" for i in range(source_count)}


@pytest.mark.parametrize("revoked", ["before", "after", "none"])
def test_relay_ack_and_return_name_only_owner_rows_that_entered_the_transcript(tmp_path, revoked):
    from ouroboros import places
    from ouroboros.owner_mailbox import acknowledge_transcript_entry
    from ouroboros.owner_relay import inject_relay, publish_relay_ack
    from ouroboros.project_dialogue import chat_annotation_receipt
    from tests.test_place_owner_author import _place_row

    source_root, target_root = tmp_path / "source", tmp_path / "recipient"
    place = places.connect_place(source_root, connector="bridge", provider="chat", account_id="qa", label="Team")
    places.edit_place(source_root, place.place_id, "confirm_owner_account", author="owner",
                      platform_actor_id="anton", account_id="qa")
    row = _place_row(place.place_id)
    dialogue_store.append(source_root, "chat", row)
    stored = next(item.row for item in dialogue_store.iter_rows(source_root, "chat"))
    quotes, error = quote_rows(context(source_root, stored), [format_address(row_address(stored))])
    assert not error
    entry, _ = write_relay(target_root, "receiver", relay_id="revoke", note="Peer note", quotes=quotes,
                            chat_id=1, teller_task_id="teller", canonical_root=source_root)
    def revoke():
        places.edit_place(source_root, place.place_id, "revoke_owner_account", author="owner",
                          platform_actor_id="anton", account_id="qa")
    if revoked == "before":
        revoke()
    receiver = context(target_root); receiver.task_id = "receiver"
    messages = []
    inject_relay(receiver, entry, messages)
    if revoked == "after":
        revoke()  # actual prior delivery remains a fact even if authority changes before acknowledgement
    assert acknowledge_transcript_entry(target_root, "receiver", entry)
    annotation = chat_annotation_receipt(source_root, stored["client_message_id"], "revoke")
    if revoked == "before":
        assert not getattr(receiver, "_owner_directives", []) and not annotation
        assert not receiver._open_relays
        assert "Peer note" in str(messages)
    else:
        assert [r["content"] for r in receiver._owner_directives] == [stored["text"]]
        assert annotation["status"] == "read" and "revoke" in receiver._open_relays
    sent = []
    publish_relay_ack(target_root, "receiver", entry.get("_relay_delivery", entry["relay"]),
                       SimpleNamespace(send_routing_ack=lambda *a, **kw: sent.append(kw)))
    assert len(sent) == (0 if revoked == "before" else 1)


def test_cross_root_return_reference_survives_continue(tmp_path):
    from ouroboros.owner_continue import owner_sources
    from ouroboros.tools.control_runtime import _send_user_message

    source_root, target_root = tmp_path / "source", tmp_path / "target"
    row, handle = source(source_root)
    quotes, error = quote_rows(context(source_root, row), [handle])
    assert not error
    write_relay(target_root, "receiver", relay_id="return", note="context", quotes=quotes,
                chat_id=1, teller_task_id="teller", canonical_root=source_root)
    receiver = context(target_root); receiver.task_id = "receiver"
    _drain_incoming_messages([], queue.Queue(), target_root, "receiver", None, set(), owner_ctx=receiver)
    sources = owner_sources(target_root, {"origin_message_text": "Work", "origin_message_ref": {},
                                          "held_owner_messages": receiver._held_owner_messages,
                                          "open_relays": receiver._open_relays,
                                          "owner_mailbox": {"rows": [], "read_complete": True}}, "receiver")
    successor = context(target_root); successor.task_id = "successor"
    successor.task_metadata = {"continuation": {"owner_sources": sources}}
    held_owner_messages(successor)
    assert _send_user_message(successor, "Done", reply_to_relay="return").startswith("OK:")
    [event] = successor.pending_events
    assert event["progress_meta"]["reply_to"]["snippet"] == "exact owner words"


def test_running_work_content_contract_delivers_its_room_to_root_child_and_nanny(tmp_path, monkeypatch):
    """Ingress -> addressed mailbox -> next drain -> every member, without cross-room broadcast.

    The first-request contract lives in test_memory_view_content_contract. This is
    the already-running half: real routing and drains, exact words and provenance,
    append-only requests and a quiet repeat. Marks have a separate contract.
    """
    from copy import deepcopy

    from ouroboros import server_owner_routing
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.projects_registry import create_project
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.registry import ToolContext
    from supervisor import queue as supervisor_queue
    from tests.test_project_routing_v664 import _ctx

    room = create_project(tmp_path, "contract-running", name="Running compass")
    other = create_project(tmp_path, "contract-unrelated", name="Unrelated orchard")
    tasks = {
        "root": {"id": "running-root", "root_task_id": "running-root", "delegation_role": "root"},
        "child": {"id": "running-child", "parent_task_id": "running-root", "root_task_id": "running-root",
                  "delegation_role": "subagent", "configured_subagent": {"id": "child", "route": {"kind": "api_model"}}},
        "nanny": {"id": "running-nanny", "parent_task_id": "running-root", "root_task_id": "running-root",
                  "delegation_role": "subagent", "configured_subagent": {"id": "nanny", "route": {"kind": "agent_session"}}},
        "other": {"id": "unrelated-root", "root_task_id": "unrelated-root", "delegation_role": "root"},
    }
    actors, messages, seen = {}, {}, {}
    prefix = [{"role": "system", "content": "Captured start view stays unchanged."},
              {"role": "user", "content": "The original assignment."},
              {"role": "assistant", "content": "I have started the work."}]
    for role, task in tasks.items():
        project = other if role == "other" else room
        task.update(chat_id=project["chat_id"], project_id=project["id"], drive_root=str(tmp_path))
        write_task_result(tmp_path, task["id"], "running", **{key: value for key, value in task.items() if key != "id"})
        actors[role] = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=task["id"],
                                  current_chat_id=task["chat_id"], project_id=task["project_id"], task_metadata=task)
        messages[role], seen[role] = deepcopy(prefix), set()
    running = {task["id"]: {"task": task, "started_at": 1.0} for task in tasks.values()}
    host = _ctx(tmp_path, running=running, direct=lambda *_a, **_kw: pytest.fail("a sole room reader takes this message"))
    monkeypatch.setattr(supervisor_queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(supervisor_queue, "ACCEPTANCE_FENCES", {})

    def arrives(project, ident, text, stamp):
        row = {"chat_id": project["chat_id"], "client_message_id": ident, "text": text,
               "ts": stamp, "direction": "in"}
        assert dialogue_store.append(tmp_path, "chat", row) is not None
        server_owner_routing._route_owner_message(object(), host, {
            "chat_id": row["chat_id"], "text": text, "log_text": text, "client_message_id": ident,
            "origin_message_ref": build_owner_message_ref(chat_id=row["chat_id"], client_message_id=ident,
                                                          ts=stamp, text=text),
        })
        return row

    def drain(role):
        _drain_incoming_messages(messages[role], queue.Queue(), tmp_path, tasks[role]["id"],
                                 None, seen[role], owner_ctx=actors[role])
        assert messages[role][:len(prefix)] == prefix
        return "\n".join(message["content"] for message in messages[role][len(prefix):])

    foreign = arrives(other, "late-orchard", "Leave the unrelated orchard open.", "2026-11-10T10:01:00+00:00")
    assert "Leave the unrelated orchard open." in drain("other")  # ingress really delivered somewhere
    for role in ("root", "child", "nanny"):
        assert drain(role) == "" and not held_owner_messages(actors[role])
    own = arrives(room, "late-compass", "  Keep north exactly here; do not rotate it.  ", "2026-11-10T10:02:00+00:00")
    address = format_address(row_address(own))
    root_text = drain("root")  # this very drain carries the row to the children
    assert "[Message from my human]" in root_text and own["text"] in root_text
    assert address in root_text  # the printed source address carries the exact room and timestamp
    assert f"row:{room['chat_id']}@2026-11-10T10:02:00+00:00#" in root_text
    for role in ("child", "nanny"):
        text = drain(role)
        assert (f"[Owner words quoted verbatim; 2026-11-10T10:02:00+00:00; Running compass; "
                f"carried to this work by the host (reached this work); handle {address}]\n{own['text']}") in text
    for role in ("root", "child", "nanny"):
        text = "\n".join(message["content"] for message in messages[role])
        assert text.count(own["text"]) == 1 and foreign["text"] not in text
        held = held_owner_messages(actors[role])[address]
        assert (held["text"], held["chat_id"], held["ts"]) == (own["text"], room["chat_id"], own["ts"])
        before = deepcopy(messages[role])
        drain(role)
        assert messages[role] == before
    assert own["text"] not in drain("other")
