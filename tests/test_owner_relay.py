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


def test_a_note_quoting_several_messages_keys_its_receipt_on_one_quote_s_own_room_and_id(tmp_path, monkeypatch):
    """The note's receipt names ONE quoted message: an id-less first quote never lends its room to a later one's id."""
    from ouroboros.tools.control_relay import _tell_task
    quotes = [{"text": "Words without an id", "source": {"chat_id": 1, "client_message_id": "", "address": "first"}},
              {"text": "Words in the room", "source": {"chat_id": 42, "client_message_id": "owner-2", "address": "second"}}]
    monkeypatch.setattr("ouroboros.owner_relay.quote_rows", lambda ctx, handles: (quotes, ""))
    events = []
    monkeypatch.setattr("ouroboros.tools.control_routing._emit_and_wait_for_routing",
                        lambda ctx, evt: (events.append(evt) or "live", {"status": "delivered"}))
    _tell_task(context(tmp_path), "receiver", "context", ["first", "second"])
    assert (events[0]["chat_id"], events[0]["client_message_id"]) == (42, "owner-2")


def test_each_quoted_message_gets_one_receipt_under_its_own_room_and_id(tmp_path):
    """Two rooms' messages may share an id: a steer quoting both writes each its own pair's receipt, the act's own
    message none twice, and a second quote of one message adds none; a quote that names no room gets none in Main."""
    import json
    from supervisor.steering import _steer_receipt
    quotes = [{"source": {"chat_id": 1, "client_message_id": "owner-1"}},
              {"source": {"chat_id": 42, "client_message_id": "owner-1"}},
              {"source": {"chat_id": 42, "client_message_id": "owner-1"}},
              {"source": {"client_message_id": "owner-unplaced"}}]
    evt = {"routing_token": "r1", "relay_id": "r1", "chat_id": 1, "client_message_id": "owner-1", "quotes": quotes,
           "issuer": {"kind": "owner_turn"}}
    _steer_receipt(SimpleNamespace(DRIVE_ROOT=tmp_path), evt, "receiver", target_label="Receiver", status="delivered")
    rows = [json.loads(line) for line in (tmp_path / "logs" / "chat_annotations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sorted((row["source_chat_id"], row["client_message_id"]) for row in rows) == [(1, "owner-1"), (42, "owner-1")]
    assert {row["routing_token"] for row in rows} == {"r1"}


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
    # A press without words is my human's act, never their words: typed so in the corpus and in the quote.
    [row] = [row for row in receiver._owner_directives if row["source"] == "owner_press"]
    assert row["content"] == "1. Publish it" and "Publish exactly this reply?" not in str(row["content"])
    assert row["reply_to"] == {"address": "quiz q7 of task asker", "author": "Ouroboros, task asker",
                               "text": "Publish exactly this reply?\n1. Publish it\n2. Not now\n"
                                       "(host record: my human pressed option 1)"}
    words = next(str(m["content"]) for m in messages if "Owner card press" in str(m["content"]))
    assert "[Host question frame]" in words and "Publish exactly this reply?" in words  # the model reads it too
    assert ("\n(host record: my human pressed «1. Publish it» and wrote no words; the label is the text of task asker)"
            "\n") in words and "\n1. Publish it\n" not in words
    assert "The label «Publish it» is the text of task asker" in words


@pytest.mark.parametrize("shape", ["words", "file_only", "card_answer"])
def test_a_held_row_names_its_source_by_the_words_its_checksum_covers(tmp_path, shape):
    """A start opened by a held row (``start_task`` ``owner_source``) carries that row's identity by value and the
    words its checksum covers: an inbound row's stored text, a file-only row's host placeholder included, a card
    answer's recorded choice without its frame. The pair finds the row again and passes a cross-room binding's
    integrity check; one text rule for every shape does not (a file-only row hashed by its owner words is lost)."""
    from ouroboros.owner_quiz import quiz_answer_frame
    from ouroboros.owner_relay import owner_row_source, owner_text
    from ouroboros.project_dialogue import build_owner_message_ref, resolve_owner_message_source
    from ouroboros.projects_registry import _validated_origin

    block = {"quiz_id": "q-palette", "question": "Which colors should the table legend use?",
             "options": ["Use blue and gold", "Use gray and green"], "state": "answered",
             "answered_index": 0, "asked_at": "2024-04-09T10:05:00+00:00", "answered_at": "2024-04-09T10:14:00+00:00"}
    row = {"words": {"chat_id": 1, "direction": "in", "ts": "2024-04-08T13:45:00+00:00", "client_message_id": "owner-b",
                     "text": "  Check the seed table;\nverify its totals  "},
           "file_only": {"chat_id": 1, "direction": "in", "ts": "2024-04-08T13:46:00+00:00", "client_message_id": "owner-f",
                         "text": "(file attached)", "text_placeholder": True,
                         "attachments": [{"upload": "0" * 32 + "_seed-counts.csv", "name": "seed-counts.csv", "kind": "file"}]},
           "card_answer": {"chat_id": 412, "direction": "system", "ts": block["answered_at"], "type": "quiz_answer",
                           "source": "owner_quiz_answer", "task_id": "asker", "quiz": block,
                           "client_message_id": "quiz_answer:asker:q-palette",
                           "text": quiz_answer_frame(block, 0, "")}}[shape]
    dialogue_store.append(tmp_path, "chat", row)
    ref, words = owner_row_source(row)
    assert words == {"words": row["text"], "file_only": "(file attached)",
                     "card_answer": "1. Use blue and gold"}[shape]
    assert "[Owner quiz answer]" not in words
    assert resolve_owner_message_source(tmp_path, ref)["client_message_id"] == row["client_message_id"]
    assert _validated_origin({"ref": ref, "text": words}, 4242)["source_ref"] == ref  # another room: the copy's check
    lost = build_owner_message_ref(chat_id=row["chat_id"], client_message_id=row["client_message_id"], ts=row["ts"],
                                   text=owner_text(row))  # the owner's words alone, for every shape
    assert (resolve_owner_message_source(tmp_path, lost) is None) == (shape == "file_only")


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


@pytest.mark.parametrize("selected", [False, True], ids=["plain", "selected-reply"])
def test_a_carried_row_told_on_keeps_the_address_its_store_answers_to(tmp_path, selected):
    """A row the host carried is held with its carriage fact beside it. Told on to another work, and on again, it
    is the same stored row: each holder keeps it under the address the store's sha-checked read answers to, as a
    row that reached that work, and a selected reply keeps its passage."""
    from ouroboros.chat_chain import resolve_row
    from ouroboros.owner_relay import inject_relay, owner_quote, owner_row_address

    excerpt = {"text": "owner", "start_utf16": 6}
    _, replied = source(tmp_path)
    reply = {"reply_to": {"address": replied, "role": "user", "author_label": "You", "excerpt": excerpt}} if selected else {}
    stored = dialogue_store.append(tmp_path, "chat", {"chat_id": 1, "direction": "in", "ts": "2026-10-01T00:01:00Z",
                                                     "text": "Carry this on.", "client_message_id": "owner-2", **reply}).row
    address = format_address(row_address(stored))
    assert resolve_row(tmp_path, address)[0] == stored
    child = tmp_path / "child"
    child.mkdir()
    _carried_entry(tmp_path, child, "helper", owner_quote(tmp_path, stored, address=address), went_to="sibling")
    holder = _helper(child)
    _drain_incoming_messages([], queue.Queue(), child, "helper", None, set(), owner_ctx=holder)
    carried = held_owner_messages(holder)[address]
    assert carried["carried"]["went_to"] == "sibling" and owner_row_address(carried) == address
    for hop in ("second", "third"):
        quotes, error = quote_rows(holder, [address])
        assert not error
        entry, created = write_relay(tmp_path, hop, relay_id=f"onward-{hop}", note="note", quotes=quotes, chat_id=1,
                                     teller_task_id=holder.task_id)
        assert created
        assert all("carried" not in owned for owned in entry["owner_rows"]), "where it reached the sender stays with the sender"
        holder, frames = context(tmp_path), []
        holder.task_id = hop
        inject_relay(holder, entry, frames)
        held = held_owner_messages(holder)
        assert list(held) == [address] and "carried" not in held[address]
        assert [row["msg_id"] for row in holder._owner_directives] == [address]
        assert resolve_row(tmp_path, next(iter(held)))[0] == stored, "the address is the stored row's own"
        assert f"handle {address}]" in frames[0]["content"]
        if selected:
            assert holder._owner_directives[0]["reply_to"]["excerpt"] == excerpt
            assert f"Sender-selected passage of You at {replied}" in frames[0]["content"]


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


def _dispatch_ctx(tmp_path, task_id="asker"):
    """A work's loop context as the agent builds one at dispatch: its dispatch metadata carries no held map."""
    from tests._budget_pause_exact_helpers import _loop_ctx

    ctx, _limit = _loop_ctx(tmp_path, task_id)
    ctx.budget_drive_root, ctx.task_metadata, ctx._loop_mailbox_seen_ids = tmp_path, {"root_task_id": task_id}, set()
    ctx.current_chat_id, ctx.pending_events, ctx.drive_logs = 1, [], lambda: tmp_path / "logs"
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    return ctx


def test_a_new_dispatch_of_a_work_keeps_its_held_words_and_open_relays_on_its_record(tmp_path, monkeypatch):
    """The work holds the owner's answer to its card and another work's relay quoting an owner row; both sit on its
    record. It parks and resumes in a new dispatch of the same id: the agent builds the context first
    (``build_runtime_section``: ``owner_quote_handles``), then the loop restores the continuation. The record must still
    hold both rows and the open relay at the end, and a Continue of the work must inherit them: a first build that
    replaced the record from the dispatch metadata alone emptied the quote handles of three works on 13 Nov."""
    import json

    from ouroboros import owner_wait
    from ouroboros.context import build_runtime_section
    from ouroboros.owner_continue import owner_sources
    from ouroboros.owner_relay import flush_held_state
    from ouroboros.task_results import load_task_result
    from ouroboros.working_checkpoint import flush_content_acks
    from tests._context_shared import _make_health_env
    from tests.test_question_answer_routing import answer, runtime

    app, _bridge, _q, _task = runtime(tmp_path, monkeypatch)
    row, relayed = source(tmp_path)
    quotes, error = quote_rows(context(tmp_path, row), [relayed])
    assert not error
    write_relay(tmp_path, "asker", relay_id="r1", note="Another work's note", quotes=quotes, chat_id=1,
                teller_task_id="teller")
    first = _dispatch_ctx(tmp_path)
    assert answer(app).status_code == 200
    _drain_incoming_messages([], queue.Queue(), tmp_path, "asker", None, first._loop_mailbox_seen_ids,
                             owner_ctx=first, defer_content_ack=True)
    flush_content_acks(first)
    held = set(first._held_owner_messages)
    assert relayed in held and len(held) == 2 and "r1" in first._open_relays
    assert set(load_task_result(tmp_path, "asker")["held_owner_messages"]) == held, "baseline: the durable copy"
    saved = json.loads(json.dumps(owner_wait.continuation_state(first, [], {}, {}, 1, [], set())))  # the park's source

    second = _dispatch_ctx(tmp_path)  # the same task id in a new dispatch: a fresh context, no held map in metadata
    build_runtime_section(_make_health_env(tmp_path / "env"), {"id": "asker", "metadata": second.task_metadata},
                          ctx=second)  # agent._prepare_task_context -> build_llm_messages, before the loop runs
    at_build = load_task_result(tmp_path, "asker") or {}
    assert set(at_build.get("held_owner_messages") or {}) == held and "r1" in (at_build.get("open_relays") or {})
    owner_wait.restore_continuation_state(SimpleNamespace(_ctx=second), saved, [], {}, {}, set())
    assert set(second._held_owner_messages) == held and "r1" in second._open_relays, "memory is whole again"
    flush_held_state(second)  # what the terminal save (working_checkpoint.save_terminal) adds

    after = load_task_result(tmp_path, "asker") or {}
    assert set(after.get("held_owner_messages") or {}) == held and "r1" in (after.get("open_relays") or {})
    assert set(owner_sources(tmp_path, after, "asker")["held_owner_messages"]) == held, "a Continue inherits them"


@pytest.mark.parametrize("parks", [False, True])
def test_an_answered_relay_stays_closed_when_the_same_continue_successor_starts_again(tmp_path, monkeypatch, parks):
    """A Continue successor inherits an open relay (``metadata.continuation.owner_sources``) and answers it, so the
    relay leaves its map and its record. It then starts again under the same id (a cold resume or a restart): its own
    present, empty map is its state, so the unchanged metadata does not reopen the relay; the quoted row stays held.
    A start whose first read of the record fails does not replace it from the inherited view (``_settle_seed``), in
    its own process or after it parks with that view and resumes (the continuation keeps the view partial)."""
    from ouroboros.owner_continue import owner_sources
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools.control_runtime import _send_user_message

    row, handle = source(tmp_path)
    quotes, error = quote_rows(context(tmp_path, row), [handle])
    assert not error
    write_relay(tmp_path, "receiver", relay_id="return", note="context", quotes=quotes, chat_id=1,
                teller_task_id="teller")
    receiver = context(tmp_path); receiver.task_id = "receiver"
    _drain_incoming_messages([], queue.Queue(), tmp_path, "receiver", None, set(), owner_ctx=receiver)
    sources = owner_sources(tmp_path, {"origin_message_text": "Work", "origin_message_ref": {},
                                       "held_owner_messages": receiver._held_owner_messages,
                                       "open_relays": receiver._open_relays,
                                       "owner_mailbox": {"rows": [], "read_complete": True}}, "receiver")
    assert "return" in sources["open_relays"]
    write_task_result(tmp_path, "successor", "running")

    def dispatch():
        ctx = context(tmp_path)
        ctx.task_id, ctx.task_metadata = "successor", {"continuation": {"owner_sources": sources}}
        held_owner_messages(ctx)  # this dispatch's first context build
        return ctx

    first = dispatch()
    assert "return" in first._open_relays
    assert _send_user_message(first, "Done", reply_to_relay="return").startswith("OK:")
    assert (load_task_result(tmp_path, "successor") or {}).get("open_relays") == {}, "answered on the record"
    again = dispatch()
    assert again._open_relays == {}, "not reopened from the unchanged metadata"
    assert set(again._held_owner_messages) == set(first._held_owner_messages) and again._held_owner_messages

    # The same start with the store too busy to read the record: its first build sees only the inherited view, so
    # no write may replace the record from it; once a read lands the record keeps its rows and its closed relay.
    from ouroboros import task_store
    from ouroboros.owner_relay import flush_held_state, retain_owner_message
    from ouroboros.store import StoreBusy

    real_read, busy = task_store.read_document, {"on": True}

    def read(root, task_id, **kwargs):
        if busy["on"] and task_id == "successor":
            raise StoreBusy("controlled busy read", reason="test")
        return real_read(root, task_id, **kwargs)

    monkeypatch.setattr(task_store, "read_document", read)
    blind = dispatch()
    assert "return" in blind._open_relays, "the inherited view only"
    later = {"chat_id": 1, "direction": "in", "ts": "2026-10-02T00:00:00Z", "text": "later owner words",
             "client_message_id": "owner-2", "attachments": []}
    dialogue_store.append(tmp_path, "chat", later)
    assert retain_owner_message(blind, later)
    busy["on"] = False
    record = load_task_result(tmp_path, "successor") or {}
    assert record.get("open_relays") == {} and set(record["held_owner_messages"]) == set(first._held_owner_messages)
    if parks:  # it parks with that partial view and resumes in a new dispatch that can read the record
        import json

        from ouroboros import owner_wait

        saved = json.loads(json.dumps(owner_wait.continuation_state(blind, [], {}, {}, 1, [], set())))
        blind = dispatch()
        owner_wait.restore_continuation_state(SimpleNamespace(_ctx=blind), saved, [], {}, {}, set())
    flush_held_state(blind)  # the terminal save: the record reads now
    record = load_task_result(tmp_path, "successor") or {}
    assert record.get("open_relays") == {} == blind._open_relays, "the answered relay stays closed"
    assert set(record["held_owner_messages"]) == set(first._held_owner_messages) | set(blind._held_owner_messages)
    assert len(record["held_owner_messages"]) == len(first._held_owner_messages) + 1, "the later row joins them"


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


@pytest.mark.parametrize("comment", ["", "  when the tide turns  "])
def test_history_of_a_relay_preserves_the_act_or_exact_words(tmp_path, comment):
    from ouroboros.dialogue_provenance import relation_text
    from ouroboros.owner_relay import owner_quote, relay_envelope

    card = {"quiz_id": "tide-card", "state": "answered", "question": "Which chart?",
            "options": ["Use the tide chart"], "answered_index": 0, "comment": comment}
    row = {"type": "quiz_answer", "task_id": "chart-maker", "quiz": card, "chat_id": 1, "ts": "2040-02-11T12:00:00Z"}
    quote = owner_quote(tmp_path, row)
    relay = relay_envelope("chart-handoff", [quote], chat_id=1)
    assert relay["text"] == (comment or "1. Use the tide chart")
    frame = relation_text({"relay": relay})
    assert quote["source"]["address"] in frame
    if comment:
        assert "Owner words quoted" in frame and frame.endswith("\n" + comment)
    else:
        assert "pressed" in frame and "Use the tide chart" in frame and "chart-maker" in frame
        assert "Owner words quoted" not in frame


def test_relay_puts_last_local_words_on_each_press_and_excludes_carried_words(tmp_path):
    import queue
    from ouroboros.answer_facts import last_own_words
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_relay import owner_quote, write_relay

    ctx = context(tmp_path)
    local = {"source": "direct_incoming", "content": "Keep the blue bowl", "client_message_id": "local-blue",
             "chat_id": "1", "ts": "2040-01-01T01:00:00Z"}
    other = {**local, "client_message_id": "other-blue", "carried": True, "ts": "2040-01-02T01:00:00Z"}
    ctx._owner_directives = [other]
    assert last_own_words(ctx) == ""
    ctx._owner_directives = [local, other]
    line = last_own_words(ctx)
    assert "local-blue" in line
    card = {"quiz_id": "bowl", "state": "answered", "question": "Which bowl?", "options": ["Blue"], "answered_index": 0}
    quotes = []
    for index, comment in enumerate(("", "", "  here  ")):
        row = {"type": "quiz_answer", "direction": "system", "source": "owner_quiz_answer", "task_id": "bowl-maker",
               "quiz": {**card, "comment": comment}, "chat_id": 1, "ts": local["ts"], "client_message_id": f"bowl-{index}"}
        quotes.append(owner_quote(tmp_path, row))
    assert write_relay(tmp_path, ctx.task_id, relay_id="bowl-delivery", note="Read the cards",
                       quotes=quotes, chat_id=1, teller_task_id="bowl-runner")[1]
    messages = []
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, ctx.task_id, None, set(), owner_ctx=ctx)
    owner_rows = [m for m in messages if m.get("_host_context_kind") == "owner_words"]
    assert [line in m["content"] for m in owner_rows] == [True, True, False]
    assert all(line not in m["content"] for m in messages if m not in owner_rows)


def test_a_carried_corpus_answer_keeps_its_card_without_a_room_row(tmp_path):
    from ouroboros.owner_relay import _rows_to_carry, inject_relay
    from ouroboros.owner_words import answered_card_reply

    ctx = context(tmp_path)
    card = {"quiz_id": "reed-option", "question": "Which reed?", "options": ["Tall reed"],
            "answered_index": 0}
    target = answered_card_reply(card, "reed-maker")
    ctx._owner_directives = [
        {"source": "owner_press", "content": "1. Tall reed", "reply_to": target, "source_task_id": "reed-maker",
         "quiz_id": "reed-option", "client_message_id": "reed-press", "chat_id": "1"},
        {"source": "owner_mailbox", "content": "  before dusk  ", "client_message_id": "reed-text", "chat_id": "1"}]
    child = _helper(tmp_path, "reed-reader")
    frames = []
    for _, _, quote, _, _ in _rows_to_carry(ctx, tmp_path):
        assert quote["owner_row"]["_owner_corpus_rebuilt"] is True
        entry, created = _carried_entry(tmp_path, tmp_path, child.task_id, quote, went_to=ctx.task_id)
        assert created
        inject_relay(child, entry, frames)
    assert len(frames) == 2
    press, words = (row["content"] for row in frames)
    assert "Owner words quoted verbatim" not in press and "pressed" in press
    assert "Which reed?" in press and "reed-maker" in press
    assert "Owner words quoted verbatim" in words and words.endswith("  before dusk  ")
    assert not getattr(child, "_owner_directives", None)


def test_a_relay_built_before_act_labels_still_renders_a_press_as_a_press(tmp_path):
    from ouroboros.owner_relay import inject_relay, owner_quote, relay_envelope
    from ouroboros.dialogue_provenance import relation_text

    card = {"quiz_id": "paper-choice", "question": "Which paper?", "state": "answered",
            "options": ["Green paper"], "answered_index": 0}
    row = {"type": "quiz_answer", "source": "owner_quiz_answer", "direction": "system", "task_id": "paper-maker",
           "quiz": card, "chat_id": 1, "ts": "2040-01-01T00:00:00Z"}
    quote = owner_quote(tmp_path, row)
    quote.pop("input_source", None)
    envelope = relay_envelope("paper-mail", [quote], chat_id=1)
    messages = []
    ctx = context(tmp_path)
    inject_relay(ctx, {"kind": "relay_entry", "relay": envelope, "source_task_id": "paper-courier",
                       "owner_rows": [row], "text": "From the maker"}, messages)
    assert "Owner card press" in messages[0]["content"] and "Owner words quoted verbatim" not in messages[0]["content"]
    assert "paper-maker" in messages[0]["content"] and "pressed" in relation_text({"relay": envelope})
    [act] = ctx._owner_directives
    assert act["source_task_id"] == "paper-maker" and act["relayed_from_task_id"] == "paper-courier"


@pytest.mark.parametrize("kind", ["press", "comment", "unavailable"])
def test_reference_only_answer_uses_its_source_store_for_history_and_receipt(tmp_path, kind):
    from ouroboros.dialogue_provenance import quote_words, relation_text
    from ouroboros.owner_quiz import record_answered, record_asked
    from ouroboros.owner_relay import inject_relay, owner_quote
    from ouroboros.owner_words import owner_words_contain

    source_root, asker, quiz = tmp_path / "source", "reed-desk", "reed-choice"
    label, comment = "Use the reed map", "  after dusk  " if kind == "comment" else ""
    # The recipient has a different card under the same task and quiz ids.
    for root, value in ((tmp_path, "unrelated answer"), (source_root, comment)):
        if root == source_root and kind == "unavailable":
            continue
        record_asked(root, asker, quiz_id=quiz, question="Which map?", options=[label])
        record_answered(root, asker, quiz_id=quiz, option_index=0, comment=value, request_id="reed-answer")
    row = {"direction": "in", "text": comment or "1. " + label, "chat_id": 1, "ts": "2040-01-01T03:00:00Z",
           "client_message_id": "reed-late", "late_answer": {"task_id": asker, "quiz_id": quiz},
           "_owner_source_root": str(source_root)}
    quote = owner_quote(tmp_path, row)
    entry, created = write_relay(tmp_path, "reed-reader", relay_id="reed-relay", note="Read the choice",
                                 quotes=[quote], teller_task_id="reed-teller", chat_id=1)
    assert created
    history = relation_text({"relay": entry["relay"]})
    ctx, frames = context(tmp_path), []
    inject_relay(ctx, entry, frames)
    [received] = ctx._owner_directives
    assert quote_words(row, source_root) == comment
    assert not owner_words_contain(ctx, label) and not owner_words_contain(ctx, "unrelated answer")
    assert ("Owner words quoted" in history) == bool(comment)
    expected = {"press": "owner_press", "comment": "owner_mailbox", "unavailable": "owner_card_input"}[kind]
    assert received["source"] == quote["input_source"] == expected
    if kind != "unavailable":
        assert received["content"] == row["text"] and received["reply_to"]["address"] == "quiz reed-choice of task reed-desk"
        assert "Which map?" in quote["question_frame"] and "Which map?" in frames[0]["content"]


def test_a_lost_submission_stays_an_input_through_carriage_and_onward_relay(tmp_path):
    from ouroboros.loop_messages import owner_input_revision, owner_reply_owed
    from ouroboros.owner_mailbox import KIND_CARD_INPUT, write_owner_message
    from ouroboros.owner_relay import _rows_to_carry, inject_relay, owner_quote
    from ouroboros.owner_words import cited_owner_basis, owner_words_contain

    ctx = context(tmp_path)
    assert write_owner_message(tmp_path, "An old card frame", ctx.task_id, kind=KIND_CARD_INPUT,
                               msg_id="card_input:card_0123456789abcdef:elm")
    _drain_incoming_messages([], queue.Queue(), tmp_path, ctx.task_id, None, set(), owner_ctx=ctx)
    [lost] = ctx._owner_directives
    words = lost["content"]
    assert owner_input_revision(ctx) == 1 and owner_reply_owed(ctx) and not owner_words_contain(ctx, words)
    child = _helper(tmp_path, "elm-reader")
    [(_, _, quote, _, _)] = _rows_to_carry(ctx, tmp_path)
    entry, created = _carried_entry(tmp_path, tmp_path, child.task_id, quote, went_to=ctx.task_id)
    assert created
    frames = []
    inject_relay(child, entry, frames)
    assert "Owner card input" in frames[0]["content"] and "Owner words quoted" not in frames[0]["content"]
    receiver = context(tmp_path)
    held = next(iter(held_owner_messages(child).values()))
    from ouroboros.dialogue_provenance import quote_words
    assert quote_words(held, tmp_path) == ""
    ordinary, _ = source(tmp_path)
    onward, created = write_relay(tmp_path, "elm-receiver", relay_id="elm-onward", note="Two inputs",
                                  quotes=[owner_quote(tmp_path, held), owner_quote(tmp_path, ordinary)],
                                  teller_task_id=child.task_id, chat_id=1)
    assert created
    inject_relay(receiver, onward, [])
    assert receiver._owner_directives[0]["source"] == "owner_card_input"
    assert not owner_words_contain(receiver, words) and not cited_owner_basis(receiver, words)[0]
    assert owner_words_contain(receiver, ordinary["text"])


@pytest.mark.parametrize("first_available", [False, True], ids=["gap-first", "answer-first"])
@pytest.mark.parametrize("comment", ["", "  after moonrise  "], ids=["press", "comment"])
def test_reference_only_relay_quotes_one_read_of_the_askers_store(tmp_path, monkeypatch, first_available, comment):
    from ouroboros import owner_quiz
    from ouroboros.dialogue_provenance import relation_text
    from ouroboros.owner_relay import owner_quote, relay_envelope

    source_drive, asker, quiz = tmp_path / "elm-room", "elm-desk", "reed-map"
    label, question = "File the reed map", "Which map?"
    owner_quiz.record_asked(source_drive, asker, quiz_id=quiz, question=question, options=[label])
    owner_quiz.record_answered(source_drive, asker, quiz_id=quiz, option_index=0, comment=comment, request_id="elm-answer")
    row = {"direction": "in", "chat_id": 3, "client_message_id": "elm-late", "text": comment or "1. " + label,
           "late_answer": {"task_id": asker, "quiz_id": quiz}, "_owner_source_root": str(source_drive)}
    original, reads = owner_quiz.quiz_states, []

    def alternating(drive, task):
        reads.append((str(drive), task))
        return original(drive, task) if (len(reads) == 1) == first_available else {}

    monkeypatch.setattr(owner_quiz, "quiz_states", alternating)
    quote = owner_quote(tmp_path / "recipient", row)
    history = relation_text({"relay": relay_envelope("reed-relay", [quote], chat_id=3)})
    if first_available:
        assert quote["input_source"] == ("owner_mailbox" if comment else "owner_press")
        assert quote["text"] == row["text"] and question in quote["question_frame"]
        assert ("Owner words quoted" in history) is bool(comment)
        assert (comment or label) in history
    else:
        assert quote["input_source"] == "owner_card_input" and "question_frame" not in quote
        assert "accepted answer is unavailable" in quote["text"] and quote["text"] in history
        assert "Owner words quoted" not in history
    assert reads == [(str(source_drive), asker)]


def test_a_held_selection_with_only_a_preview_takes_a_later_source_once_and_an_addressed_flip_keeps_it(tmp_path):
    """A carried entry that completes a held selected reply keys on the source itself: a held ``_owner_reply_target``
    with preview fields only is completed like one with none, by a copy of the same address and passage alone, once.
    Another passage's source, or an address this work never held, changes nothing; a context row an addressed copy
    flips keeps the source it already held. The row's standing never moves with its source."""
    from ouroboros.owner_relay import inject_relay, owner_quote, owner_row_address, relay_envelope

    excerpt = {"text": "North", "start_utf16": 0}
    reference = {"address": "row:1@source#0123456789ab", "author_label": "Mira", "role": "user",
                 "snippet": "North trail…", "excerpt": excerpt}
    owner = {"chat_id": 1, "direction": "in", "source": "web", "user_id": 1, "ts": "2026-11-04T09:01:00Z",
             "client_message_id": "owner-held", "text": "Which trail?", "reply_to": reference}
    address = owner_row_address(owner)
    files = [{"name": "trail.pdf", "mime": "application/pdf", "size": 55}]
    full = {"address": reference["address"], "author": "Mira", "excerpt": excerpt, "role": "user",
            "text": "North trail. The full body.", "attachments": files}
    preview = {"address": reference["address"], "author": "Mira", "excerpt": excerpt, "role": "user", "snippet": "North trail…"}
    standing = {"from": "root", "went_to": "routed:head", "at": "2026-11-04T09:02:00Z"}

    def entry(*, went, enrich, target):
        quote = owner_quote(tmp_path, {**owner, "_owner_source_root": str(tmp_path),
                                       **({"_owner_reply_target": target} if target else {})}, address=address)
        return {"kind": "relay_entry", "msg_id": f"relay:carry-source:{address}:member", "ts": "2026-11-04T09:03:00Z", "text": "",
                "relay": relay_envelope("carry-source", [quote], chat_id=1), "owner_rows": [quote["owner_row"]],
                "source_task_id": "root", "relay_canonical_root": str(tmp_path),
                "host_carried": {"went_to": went, **({"enrich": True} if enrich else {})}}

    member = context(tmp_path)
    member.task_id = "member"
    member._held_owner_messages = {address: {**owner, "_owner_reply_target": preview, "carried": dict(standing)}}
    messages = []
    inject_relay(member, entry(went="", enrich=True, target=full), messages)
    held = member._held_owner_messages[address]
    assert held["_owner_reply_target"] == full and held["carried"] == standing, "the source moved; the standing did not"
    assert messages == [], "no words are shown again"
    inject_relay(member, entry(went="routed:head", enrich=True, target={**full, "text": "A different body."}), messages)
    assert member._held_owner_messages[address]["_owner_reply_target"] == full, "once: the held source is not replaced"

    member._held_owner_messages = {address: {**owner, "carried": dict(standing)}}
    other_passage = {**full, "excerpt": {"text": "trail", "start_utf16": 6}}
    inject_relay(member, entry(went="routed:head", enrich=True, target=other_passage), messages)
    assert "_owner_reply_target" not in member._held_owner_messages[address], "another passage's source completes nothing"

    member._held_owner_messages = {}
    inject_relay(member, entry(went="", enrich=True, target=full), messages)
    assert member._held_owner_messages == {} and messages == [], "an address this work never held: nothing held or shown"

    member._held_owner_messages = {address: {**owner, "_owner_reply_target": full, "carried": dict(standing)}}
    inject_relay(member, entry(went="", enrich=False, target=None), messages)
    flipped = member._held_owner_messages[address]
    assert flipped["carried"]["went_to"] == "" and flipped["_owner_reply_target"] == full, "addressed now, source kept"
    assert "Which trail?" in str(messages), "the flip shows its words once, as it always did"


# --- files of the message a work answers: inherited by value, named when they cannot be copied ------------------

def _held_file(root, name, data, **row):
    from ouroboros.chat_uploads import store_upload

    _path, ref = store_upload(data, name, data_dir=root)
    return {"chat_id": 1, "direction": "in", "ts": "2024-04-08T13:45:00+00:00", "client_message_id": "owner-b",
            "text": "Check the totals in the seed table", "attachments": [ref], **row}


def test_a_file_of_another_store_is_inherited_only_from_the_callers_own_verified_copy(tmp_path):
    """B lives in another data root: the snapshot a start carries names its file without an upload id. Work the
    first work starts receives the first work's own staged copy of exactly those bytes, and the work that one
    starts receives the second's (a chain); a later file of the caller never travels. Without such a copy, with one
    of other bytes, or with one gone by the admission, the file is named ``outside_store``."""
    import pathlib

    from ouroboros.artifacts import attachment_manifest_projection, stage_task_attachments
    from ouroboros.chat_uploads import store_upload
    from ouroboros.owner_relay import origin_snapshot, source_upload_specs, upload_specs

    teller, recipient = tmp_path / "teller", tmp_path / "recipient"
    row = _held_file(teller, "seed-counts.csv", b"bed,seeds\nwest,3\n", _owner_source_root=str(teller))
    [ref] = origin_snapshot(row, recipient)["attachments"]
    assert ref["unavailable"] == "outside_store" and "upload" not in ref
    later_path, _later = store_upload(b"bed,seeds\neast,1\n", "seed-counts.csv", data_dir=recipient)
    first = stage_task_attachments(recipient, "first-work", [*upload_specs(row, teller), {"path": str(later_path)}])
    assert [item["status"] for item in first] == ["staged", "staged"]

    def caller(task_id, manifest):
        return SimpleNamespace(drive_root=recipient, task_id=task_id,
                               task_contract=attachment_manifest_projection(recipient, task_id, manifest))

    holder = first
    for task_id, new_id in (("first-work", "second-work"), ("second-work", "third-work")):
        [spec] = source_upload_specs(caller(task_id, holder), [ref])
        assert pathlib.Path(spec["path"]).samefile(holder[0]["abs_path"])
        assert (spec["label"], spec["inherited"], spec["path_only"]) == ("seed-counts.csv", True, True)
        holder = stage_task_attachments(recipient, new_id, [spec])
        assert [(item["status"], item["label"]) for item in holder] == [("staged", "seed-counts.csv")]
        assert pathlib.Path(holder[0]["abs_path"]).read_bytes() == b"bed,seeds\nwest,3\n"
    [spec] = source_upload_specs(caller("third-work", holder), [ref])
    pathlib.Path(spec["path"]).unlink()  # the caller's copy is gone by the new work's admission
    [gone] = stage_task_attachments(recipient, "fourth-work", [spec])
    assert (gone["status"], gone["reason"]) == ("rejected", "outside_store")
    for bare in (SimpleNamespace(drive_root=recipient, task_id="other-work", task_contract={}),
                 caller("third-work", holder)):  # no copy declared; then a declared copy that is gone
        [spec] = source_upload_specs(bare, [ref])
        assert spec["path"] == "" and spec["unavailable"] == "outside_store"
        [named] = stage_task_attachments(recipient, "fifth-work", [spec])
        assert (named["status"], named["reason"], named["label"]) == ("rejected", "outside_store", "seed-counts.csv")
    pathlib.Path(first[0]["abs_path"]).write_bytes(b"other bytes under the same name")
    [spec] = source_upload_specs(caller("first-work", first), [ref])
    assert spec["path"] == "", "a copy of other bytes is not that file"


def test_a_local_upload_is_inherited_by_its_id_and_a_missing_one_is_named(tmp_path):
    """The snapshot's ref of a file saved in this store stages from that upload by its id and saved identity; a ref
    that names no saved upload is named ``missing``; nothing is looked up by the message's text or client id."""
    import pathlib

    from ouroboros.artifacts import stage_task_attachments
    from ouroboros.chat_uploads import unavailable_ref, uploads_dir
    from ouroboros.owner_relay import origin_snapshot, source_upload_specs

    row = _held_file(tmp_path, "seed-counts.csv", b"bed,seeds\nnorth,8\n")
    [ref] = origin_snapshot(row, tmp_path)["attachments"]
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="first-work")
    saved, lost = source_upload_specs(ctx, [ref, unavailable_ref("old-notes.txt")])
    assert pathlib.Path(saved["path"]).samefile(uploads_dir(tmp_path) / ref["upload"]) and "unavailable" not in saved
    assert (lost["path"], lost["unavailable"], lost["inherited"]) == ("", "missing", True)
    staged, named = stage_task_attachments(tmp_path, "second-work", [saved, lost])
    assert staged["status"] == "staged" and pathlib.Path(staged["abs_path"]).read_bytes() == b"bed,seeds\nnorth,8\n"
    assert (named["status"], named["reason"], named["label"]) == ("rejected", "missing", "old-notes.txt")
    assert stage_task_attachments(tmp_path, "third-work", [{"path": "", "label": "bare"}])[0]["reason"] == "invalid_path"


def test_a_helper_copies_every_row_it_copied_before_and_leaves_only_the_displaced_source_named(tmp_path):
    """At a helper's birth the files of its tree rows, held rows of both kinds, room rows, Project origin rows and
    marked rows are staged on its own drive as before. Only the message its root's start displaced (carried in
    beside the one the work answers) keeps its file named with the host's mark and the act that gives it."""
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.chat_uploads import store_upload
    from ouroboros.owner_relay import NOT_GIVEN, mark_context_files, stage_birth_files
    from ouroboros.owner_words import attachments_line

    def row(carrier, source, name, data):
        _path, ref = store_upload(data, name, data_dir=tmp_path)
        return {"text": f"words of {name}", "source": source, "carrier": carrier, "task_id": "root", "ts": "t",
                "ref": f"chat 1 / {name}", "attachments": [ref]}

    copied = [row("ctx", "origin_message", "source.csv", b"source"), row("held", "owner_mailbox", "held.csv", b"held"),
              row("went to other-work", "owner_mailbox", "context.csv", b"context"),
              row("room", "room_message", "room.csv", b"room"), row("origin", "origin_message", "project.csv", b"project"),
              row("marked", "marked_message", "marked.csv", b"marked")]
    displaced = row("ctx", "origin_message", "displaced.csv", b"displaced")
    opening = {"client_message_id": "owner-b", "chat_id": "1", "ts": "2024-04-08T13:45:00+00:00", "row_sha256": "b" * 64}
    item = {"source": "origin_message", "client_message_id": "owner-a", "chat_id": "1", "carried": True,
            "ts": "2024-04-08T13:20:00+00:00", "row_sha256": "a" * 64}
    mark_context_files([({**opening, "source": "origin_message", "carried": True}, copied[0]), (item, displaced)], opening)
    assert "staged" not in copied[0]["attachments"][0], "the message the work answers is never marked"
    handle = "row:1@2024-04-08T13:20:00+00:00#aaaaaaaaaaaa"
    assert displaced["attachments"][0]["staged"] == (
        f"{NOT_GIVEN}: owner_source='{handle}' at start_task, or tell_task(quote=['{handle}']) to a running work, gives it")
    held = {**item, "row_sha256": "c" * 64}
    reached = row("ctx", "origin_message", "relayed.csv", b"relayed")
    mark_context_files([(held, reached)], opening, {"c" * 64})
    assert "staged" not in reached["attachments"][0], "a row the work holds had its files staged here"
    stage_birth_files(tmp_path, {"id": "helper", "drive_root": str(tmp_path)}, "helper", [*copied, displaced])
    store = task_artifact_dir_path(tmp_path, "helper") / "attachments"
    assert sorted(path.read_bytes() for path in store.iterdir()) == sorted(
        [b"source", b"held", b"context", b"room", b"project", b"marked"])
    assert all(item["attachments"][0]["staged"].startswith("read_file(root='artifact_store', path='attachments/")
               for item in copied)
    line = attachments_line(displaced["attachments"])
    assert line.startswith("[Host-retained attachments: displaced.csv (") and NOT_GIVEN in line


def test_a_helper_holds_its_roots_inherited_file_once_and_a_rejected_root_row_as_a_gap(tmp_path):
    """The root's input manifest reaches a helper through the existing parent-manifest inheritance: the staged copy
    once, a row the root could not stage as the same named gap, never a refused start. The helper's birth staging of
    the source row then finds that copy and adds no second file."""
    from ouroboros.artifacts import materialize_inherited_attachment_manifest, stage_task_attachments, task_artifact_dir_path
    from ouroboros.owner_relay import origin_snapshot, source_upload_specs, stage_birth_files

    row = _held_file(tmp_path, "seed-counts.csv", b"bed,seeds\nnorth,8\n")
    refs = [*origin_snapshot(row, tmp_path)["attachments"], {"name": "old-notes.txt", "unavailable": "outside_store"}]
    root = stage_task_attachments(tmp_path, "root-work", source_upload_specs(
        SimpleNamespace(drive_root=tmp_path, task_id="caller"), refs))
    assert [(item["status"], item["reason"]) for item in root] == [("staged", ""), ("rejected", "outside_store")]
    copied, error = materialize_inherited_attachment_manifest(root, tmp_path, "helper")
    assert not error and [(item["status"], item["reason"]) for item in copied] == [("staged", ""), ("rejected", "outside_store")]
    source = {"text": row["text"], "source": "origin_message", "carrier": "ctx", "task_id": "root-work", "ts": row["ts"],
              "ref": "chat 1 / owner-b", "attachments": [dict(row["attachments"][0])]}
    stage_birth_files(tmp_path, {"id": "helper", "drive_root": str(tmp_path)}, "helper", [source])
    store = task_artifact_dir_path(tmp_path, "helper") / "attachments"
    assert [path.read_bytes() for path in store.iterdir()] == [b"bed,seeds\nnorth,8\n"], "one copy, not a second"
    assert source["attachments"][0]["staged"] == f"read_file(root='artifact_store', path='{copied[0]['relpath']}')"


def test_a_relayed_quote_that_carried_only_a_file_keeps_its_descriptor_in_the_corpus(tmp_path):
    """The quote's row enters the recipient's corpus with its file named by saved identity, no path; carried on
    into work the recipient starts, that file is named as not given."""
    from ouroboros.chat_uploads import claim_refs, store_upload
    from ouroboros.owner_relay import NOT_GIVEN, mark_context_files, retain_owner_message
    from ouroboros.owner_words import carried_owner_corpus
    _path, ref = store_upload(b"bed,seeds\nwest,3\n", "seed-counts.csv", data_dir=tmp_path)
    claim_refs([ref], tmp_path)
    row = {"chat_id": 1, "direction": "in", "ts": "2026-10-01T00:00:00Z", "text": "(file attached)",
           "text_placeholder": True, "client_message_id": "owner-file", "attachments": [ref]}
    dialogue_store.append(tmp_path, "chat", row)
    sender = context(tmp_path)
    retain_owner_message(sender, row)
    quotes, error = quote_rows(sender, [format_address(row_address(row))])
    assert not error
    write_relay(tmp_path, "receiver", relay_id="file-only", note="context", quotes=quotes, chat_id=1, teller_task_id="teller")
    receiver = context(tmp_path)
    receiver.task_id = "receiver"
    _drain_incoming_messages([], queue.Queue(), tmp_path, "receiver", None, set(), owner_ctx=receiver)
    [kept] = receiver._owner_directives
    assert (kept["source"], kept["content"], kept["carried"]) == ("owner_mailbox", "", True)
    [named] = kept["attachments"]
    assert (named["name"], named["upload"]) == ("seed-counts.csv", ref["upload"]) and not {"path", "abs_path"} & set(named)
    [carried] = carried_owner_corpus(receiver)
    assert carried["content"] == "" and carried["attachments"] == kept["attachments"]
    shown = {"attachments": [dict(named)]}
    mark_context_files([(carried, shown)], {"client_message_id": "owner-b", "chat_id": "1", "row_sha256": "b" * 64})
    assert shown["attachments"][0]["staged"].startswith(NOT_GIVEN)
