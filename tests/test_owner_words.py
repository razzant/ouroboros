"""The owner's words that caused a work tree (``ouroboros/owner_words.py``).

Every rule is checked in both directions: the rule fires on its case and stays quiet
on the neighbouring one, so deleting the rule turns a test red.
"""
from __future__ import annotations

import ast
import json
import pathlib
import queue

import pytest

from ouroboros import owner_words as ow
from ouroboros.loop_messages import _initialize_owner_directives
from ouroboros.loop_round_limits import _drain_incoming_messages
from ouroboros.owner_mailbox import (KIND_CARD_INPUT, KIND_OWNER_TEXT, KIND_QUIZ_ANSWER, KIND_TASK_MESSAGE,
                                     write_owner_message, write_task_message)
from ouroboros.project_dialogue import build_owner_message_ref
from ouroboros.tools.tool_context import ToolContext

REPO = pathlib.Path(__file__).resolve().parents[1]
ROOT = "root0001"
TS = "2026-09-30T05:41:45+00:00"
MIND = {"kind": "mind", "task_id": ROOT, "focus": {"role": "root", "task_id": ROOT}}


def _ctx(tmp_path, *, directives=None, metadata=None, task_id=ROOT, task_type="task", chat_id=None):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=task_id, current_chat_id=chat_id,
                      task_metadata=dict(metadata or {}), current_task_type=task_type)
    if directives is not None:
        ctx._owner_directives = [dict(row) for row in directives]
    return ctx


def _at(second: int) -> str:
    return f"2026-09-30T05:41:{second:02d}+00:00"


def _project(tmp_path, task_id=ROOT, *, origin=None) -> int:
    """A registered Project room the task is bound to; ``origin``: the owner message that started it, by ref
    and text (the binding's retention-proof copy), else a typed absence."""
    from ouroboros.projects_registry import bind_task_to_project, create_project

    room = create_project(tmp_path, "quotas", name="Quotas")["chat_id"]
    bind_task_to_project(tmp_path, task_id, "quotas", room, origin=origin or {"absent": "system"})
    return room


def _raising(exc: Exception):
    def raise_it(*_args, **_kwargs):
        raise exc
    return raise_it


def _stamps(rows):
    return [(row["carrier"], row["ref"], row["ts"], row["text"]) for row in rows]


def _append(path: pathlib.Path, *rows: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _in_row(text, client_id, *, ts=TS, chat_id=1, **extra):
    return {"ts": ts, "direction": "in", "chat_id": chat_id, "text": text, "client_message_id": client_id,
            "task_id": "", **extra}


def _chat(tmp_path, *rows):
    from ouroboros import dialogue_store as ds

    for row in rows:
        ds.append(tmp_path, "chat", row)


def _annotate(tmp_path, client_id, *, action="promote_chat_to_task", status="scheduled", target=ROOT, token="t1"):
    _append(tmp_path / "logs" / "chat_annotations.jsonl", {
        "ts": TS, "type": "chat_annotation", "client_message_id": client_id, "action": action,
        "target": target, "status": status, "routing_token": token})


def _bind(tmp_path, task_id=ROOT, **origin):
    path = tmp_path / "state" / "project_task_bindings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"bindings": {}}
    data["bindings"][task_id] = {"task_id": task_id, "project_id": "p", "project_chat_id": -7, **origin}
    path.write_text(json.dumps(data), encoding="utf-8")


def _texts(rows):
    return [row["text"] for row in rows]


# --- carried by value ---------------------------------------------------------------------------------------

def test_root_corpus_carries_owner_rows_and_never_initial_text(tmp_path):
    corpus = [{"source": "initial_user", "content": "Build the report"},
              {"source": "owner_mailbox", "content": "Add charts", "msg_id": "m-2"},
              {"source": "initial_text", "content": "A wake or a work order"}]
    words = ow.governing_words_for_schedule(_ctx(tmp_path, directives=corpus))
    assert _texts(words[ow.FIELD]) == ["Build the report", "Add charts"]
    assert {row["carrier"] for row in words[ow.FIELD]} == {"ctx"}
    assert words[ow.FIELD][1]["ref"] == "msg m-2"
    # The other side: a corpus with only the non-owner first turn carries nothing of it.
    only_text = ow.governing_words_for_schedule(_ctx(tmp_path, directives=corpus[2:]))
    assert ow.FIELD not in only_text and only_text[ow.ABSENT_FIELD] == "task_type=task"


def test_child_passes_on_exactly_its_inherited_words(tmp_path):
    inherited = [{"text": "Root words", "source": "initial_user", "carrier": "ctx", "task_id": ROOT,
                  "ts": TS, "ref": "chat 1 / msg-1"}]
    child_meta = {"delegation_role": "subagent", "root_task_id": ROOT, ow.FIELD: inherited}
    child = _ctx(tmp_path, task_id="child01", metadata=child_meta,
                 directives=[{"source": "initial_text", "content": "Parent's assignment"}])
    assert ow.governing_words_for_schedule(child) == {ow.FIELD: inherited}
    absent_child = _ctx(tmp_path, task_id="child02", metadata={"delegation_role": "subagent",
                                                               ow.ABSENT_FIELD: "initiator=consciousness"})
    assert ow.governing_words_for_schedule(absent_child) == {ow.ABSENT_FIELD: "initiator=consciousness"}
    # A root holding the same metadata key hands those words on beside its own corpus.
    root = _ctx(tmp_path, metadata={ow.FIELD: inherited}, directives=[{"source": "initial_user", "content": "Mine"}])
    assert _texts(ow.governing_words_for_schedule(root)[ow.FIELD]) == ["Root words", "Mine"]


def test_absence_carries_the_raw_run_origin_marker(tmp_path):
    conscious = _ctx(tmp_path, metadata={"initiator": "consciousness"},
                     directives=[{"source": "initial_text", "content": "wake"}])
    assert ow.governing_words_for_schedule(conscious) == {ow.ABSENT_FIELD: "initiator=consciousness"}
    bare = _ctx(tmp_path, task_type="", directives=[])
    assert ow.governing_words_for_schedule(bare) == {ow.ABSENT_FIELD: ow.NOT_RECORDED}
    spoken = _ctx(tmp_path, metadata={"initiator": "consciousness"},
                  directives=[{"source": "owner_mailbox", "content": "Owner wrote in"}])
    assert _texts(ow.governing_words_for_schedule(spoken)[ow.FIELD]) == ["Owner wrote in"]


def test_main_turn_origin_is_carried_byte_for_byte(tmp_path):
    message = "  Сделай отчёт:\n\tцифры за неделю  \n"
    ref = build_owner_message_ref(chat_id=1, client_message_id="msg-1790746905659-12", ts=TS, text=message)
    ctx = _ctx(tmp_path, metadata={"origin_message_ref": ref, "origin_message_text": message})
    _initialize_owner_directives(ctx, [{"role": "user", "content": message}])
    rows = ow.governing_words_for_schedule(ctx)[ow.FIELD]
    assert rows == [{"text": message, "source": "initial_user", "carrier": "ctx", "task_id": ROOT, "ts": TS,
                     "ref": "chat 1 / msg-1790746905659-12"}]
    assert ow.safety_sources(ctx) == [{"source": "initial_user", "content": message, "ts": TS}]  # dated by its door
    assert ctx._owner_directives == [{"source": "initial_user", "content": message}]
    unstamped = _ctx(tmp_path)
    _initialize_owner_directives(unstamped, [{"role": "user", "content": message}])
    assert ow.FIELD not in ow.governing_words_for_schedule(unstamped)
    assert ow.safety_sources(unstamped) == [{"source": "initial_text", "content": message}]


def test_project_first_message_is_never_added_to_the_words(tmp_path):
    project_words = "Start the Quotas project"
    _bind(tmp_path, source_text=project_words)
    in_corpus = _ctx(tmp_path, metadata={"project_id": "p"},
                     directives=[{"source": "initial_user", "content": project_words}])
    assert _texts(ow.governing_words_for_schedule(in_corpus)[ow.FIELD]) == [project_words]
    elsewhere = _ctx(tmp_path, metadata={"project_id": "p"},
                     directives=[{"source": "initial_user", "content": "Fix the quota bar"}])
    assert _texts(ow.governing_words_for_schedule(elsewhere)[ow.FIELD]) == ["Fix the quota bar"]


def test_schedule_never_reads_dialogue_and_a_render_reads_the_room_once(tmp_path, monkeypatch):
    """A native helper's field is computed from what the run holds: no dialogue is read. A handoff's render
    (``work_words``) reads the bound room once, whatever the set's size; the history resolver does read the store."""
    import ouroboros.chat_chain as chat_chain
    import ouroboros.dialogue_store as ds
    import ouroboros.project_dialogue as project_dialogue

    refuse = _raising(AssertionError("the dialogue store was read"))
    for module, name in ((ds, "iter_probes"), (project_dialogue, "resolve_owner_message_source"),
                         (chat_chain, "room_rows"), (project_dialogue, "project_origin_rows")):
        monkeypatch.setattr(module, name, refuse)
    room = _project(tmp_path)
    ref = build_owner_message_ref(chat_id=room, client_message_id="msg-9", ts=TS, text="Go")
    ctx = _ctx(tmp_path, metadata={"origin_message_ref": ref, "root_task_id": ROOT},
               directives=[{"source": "initial_user", "content": "Go"}, {"source": "owner_mailbox", "content": "On"}])
    assert _texts(ow.governing_words_for_schedule(ctx)[ow.FIELD]) == ["Go", "On"]
    reads: list = []
    monkeypatch.setattr(chat_chain, "room_rows", lambda root, room_id, **_kw: reads.append(int(room_id)) or iter(()))
    monkeypatch.setattr(project_dialogue, "project_origin_rows", lambda root, chat: reads.append(int(chat)) or [])
    assert "\nGo\n" in ow.owner_words_text(ctx) and reads == [room, room]
    # The substitution is live: the history resolver does read the store, and trips it.
    _annotate(tmp_path, "msg-9")
    with pytest.raises(AssertionError, match="dialogue store was read"):
        ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT})


def test_a_handoff_renders_the_rooms_rows_live_dated_in_the_unions_order_and_every_ingress_branch(tmp_path, monkeypatch):
    """The set a handoff renders, in order: the tree's rows, the rows held as context, the bound room's rows read
    here, the Project's origin row, the rows marks in force target — each under its reference and the time its room
    accepted it. Two «да» on two dates are two rows; a row's file is named under its words; a room that cannot be
    read is one typed gap line beside the rows that could. Then one row from each branch that records my human's
    words, dated as its room accepted it, else ``received`` (when the host got it), else undated: never invented."""
    from ouroboros import dialogue_store as ds
    from ouroboros.chat_chain import row_address
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.owner_relay import inject_relay, stamped_refs, words_unseen_in

    origin = build_owner_message_ref(chat_id=1, client_message_id="msg-o", ts=_at(0), text="Start the Quotas project")
    room = _project(tmp_path, origin={"ref": origin, "text": "Start the Quotas project"})
    file = {"upload": "u1", "name": "quota.png", "mime": "image/png", "kind": "image", "size": 1234, "sha256": "ab" * 32}
    _chat(tmp_path, _in_row("да", "cm-d1", ts=_at(1), chat_id=room), _in_row("да", "cm-d2", ts=_at(2), chat_id=room),
          _in_row("See the file", "cm-f", ts=_at(3), chat_id=room, attachments=[file]),
          {**_in_row("Ouroboros says", "out-1", ts=_at(4), chat_id=room), "direction": "out"},
          _in_row("Main room words", "cm-main", ts=_at(5)))
    door = build_owner_message_ref(chat_id=room, client_message_id="cm-door", ts=_at(1), text="Fix the quota bar")
    ctx = _ctx(tmp_path, metadata={"origin_message_ref": door, "root_task_id": ROOT},
               directives=[{"source": "initial_user", "content": "Fix the quota bar"}])
    ctx._held_owner_messages = {"row:ctx": {**_in_row("Went to another work", "cm-ctx", ts=_at(6), chat_id=room),
                                            "carried": {"went_to": "other01"}}}
    marked = dict(ds.accepted_row(tmp_path, 1, "cm-main").row)
    assert ChronicleStore(tmp_path).mark(row_address(marked), "keep this in view", MIND, room_id=str(room)).ok
    rows, absent, gaps = ow.work_words(ctx)
    assert _stamps(rows) == [
        ("ctx", f"chat {room} / cm-door", _at(1), "Fix the quota bar"),
        ("went to other01", f"chat {room} / cm-ctx", _at(6), "Went to another work"),
        ("room", f"chat {room} / cm-d1", _at(1), "да"), ("room", f"chat {room} / cm-d2", _at(2), "да"),
        ("room", f"chat {room} / cm-f", _at(3), "See the file"),
        ("origin", "chat 1 / msg-o", _at(0), "Start the Quotas project"),
        ("marked", "chat 1 / cm-main", _at(5), "Main room words")] and (absent, gaps) == ("", [])
    text = ow.render_owner_words(rows, audience="child", root_task_id=ROOT, gaps=gaps)
    assert f"[{_at(3)} · owner · room_message · chat {room} / cm-f]\nSee the file\n[Host-retained attachments: quota.png (image/png, 1234 bytes)]\n" in text
    assert text.count("\nда\n") == 2 and "Ouroboros says" not in text
    # A held context row's stamp names where it went, between its source and its reference; a room row's has no
    # such part; every parser still takes the reference last: the held row is seen by it, and unseen without it.
    held = f"[{_at(6)} · owner · owner_mailbox of task {ROOT} · went to other01 · chat {room} / cm-ctx]"
    assert f"{held}\nWent to another work\n" in text and text.count(" · went to ") == 1
    assert {f"chat {room} / cm-ctx", f"chat {room} / cm-f"} <= stamped_refs(text) and "went to other01" not in stamped_refs(text)
    assert stamped_refs(ow.words_stamp({"source": ow.REPLY_TARGET, "author": "Ann", "ref": "row:abc"})) == set()
    assert words_unseen_in(ctx, text) == [] and [row["ref"] for row in words_unseen_in(
        ctx, text.replace(f"chat {room} / cm-ctx", f"chat {room} / cm-gone"))] == [f"chat {room} / cm-ctx"]
    # The other side: a room that cannot be read is one gap line; the tree's, the held, the origin and the marked
    # rows stay. An origin read that fails is its own line; a room read that may lack rows says so.
    monkeypatch.setattr("ouroboros.chat_chain.room_rows", _raising(OSError("locked")))
    rows, absent, gaps = ow.work_words(ctx)
    assert ([row["carrier"] for row in rows], absent) == (["ctx", "went to other01", "origin", "marked"], "")
    assert gaps == [f"[Host fact: my human's rows of room {room} could not be read here (OSError); chat_history reads that room.]"]
    assert ow.render_owner_words(rows, audience="child", gaps=gaps).endswith("Main room words\n\n" + gaps[0])
    monkeypatch.setattr("ouroboros.chat_chain.room_rows",
                        lambda root, room_id, gaps=None, **_kw: gaps.add("appearances_deriving") or iter(()))
    monkeypatch.setattr("ouroboros.project_dialogue.project_origin_rows", _raising(KeyError("binding")))
    rows, _absent, gaps = ow.work_words(ctx)
    assert [row["carrier"] for row in rows] == ["ctx", "went to other01", "marked"] and gaps == [
        f"[Host fact: my human's rows of room {room} read here may lack some (Rows this Project shows but that were "
        "written elsewhere are still being indexed); chat_history reads that room.]",
        f"[Host fact: the origin rows the Project retained for room {room} could not be read here (KeyError); "
        "chat_history reads that room.]"]
    monkeypatch.undo()
    # Every ingress branch, through its own seam. A root in Main: the door; owner mail with its chat row held and
    # without one; an answer with its row; a card frame; a direct delivery with a dated payload, an undated one
    # and plain text; a relayed quote. A confirmed place account's words mid-turn are recorded dated as
    # provenance (``PLACE_CONFIRMED``, outside the set): an order exists only through a verified cite.
    main = build_owner_message_ref(chat_id=1, client_message_id="cm-door", ts=_at(1), text="Build it")
    root = _ctx(tmp_path, task_id="main0001", metadata={"origin_message_ref": main, "origin_message_text": "Build it"})
    messages = [{"role": "user", "content": "Build it"}]
    _initialize_owner_directives(root, messages)
    answer = _in_row("1. Publish it", "cm-ans", ts=_at(3))
    _chat(tmp_path, _in_row("Add charts", "cm-mail", ts=_at(2)), answer, _in_row("Direct too", "direct-1", ts=_at(4)))
    write_owner_message(tmp_path, "Add charts", "main0001", msg_id="mail-1", client_message_id="cm-mail")
    write_owner_message(tmp_path, "Mailbox only", "main0001", msg_id="mail-2")
    card = {"quiz_id": "q7", "question": "Publish?", "state": "answered", "answered_index": 0,
            "options": ["Publish it", "Not now"], "delivery": {"answer_row": answer}}
    write_owner_message(tmp_path, "[Owner quiz answer] quiz q7", "main0001", msg_id="qa-7", kind=KIND_QUIZ_ANSWER,
                        quiz_delivery={"task_id": "asker", "quiz_id": "q7", "answer_snapshot": card})
    write_owner_message(tmp_path, "[Card] typed into the card", "main0001", msg_id="card-1", kind=KIND_CARD_INPUT)
    write_task_message(tmp_path, "[Place event]", "main0001", source_task_id="place-run", provenance="place_event",
                       msg_id="pl-1", sender_origin={"conversation_key": "tg:123"},
                       place_arrival={"owner_words": "Place words mid-turn", "source_event_id": "ev-2", "message": {}})
    incoming: queue.Queue = queue.Queue()
    incoming.put({"text": "Direct too", "client_message_id": "direct-1", "ts": _at(4)})
    incoming.put({"text": "Undated payload", "client_message_id": "direct-2"})
    incoming.put("Plain direct text")
    _drain_incoming_messages(messages, incoming, tmp_path, "main0001", None, set(), owner_ctx=root)
    quoted = _in_row("Quoted by the teller", "cm-relay", ts=_at(5))
    inject_relay(root, {"owner_rows": [quoted], "text": "note", "relay": {"relay_id": "r1"}, "source_task_id": "t"}, messages)
    received = {row["content"]: row.get("received") for row in root._owner_directives}
    assert all(received[text] for text in ("Undated payload", "Plain direct text", "Mailbox only",
                                           "[Card] typed into the card", "Place words mid-turn"))
    from ouroboros.presence_runner import presence_chat_id
    rows = [row for row in ow.work_words(root)[0] if row["source"] != ow.REPLY_TARGET]
    assert _stamps(rows) == [
        ("ctx", "chat 1 / cm-door", _at(1), "Build it"),
        ("ctx", "chat 1 / direct-1", _at(4), "Direct too"),
        ("ctx", "msg direct-2", f"received {received['Undated payload']}", "Undated payload"),
        ("ctx", "", f"received {received['Plain direct text']}", "Plain direct text"),
        ("ctx", "chat 1 / cm-mail", _at(2), "Add charts"),
        ("ctx", "msg mail-2", f"received {received['Mailbox only']}", "Mailbox only"),
        ("ctx", "chat 1 / cm-ans", _at(3), "1. Publish it"),
        ("ctx", "msg card-1", f"received {received['[Card] typed into the card']}", "[Card] typed into the card"),
        ("ctx", "chat 1 / cm-relay", _at(5), "Quoted by the teller")]
    assert ow.render_owner_words(rows, audience="child").count("[received ") == 4
    place = [item for item in root._owner_directives if item.get("content") == "Place words mid-turn"]
    assert [(item["source"], item.get("msg_id"), item.get("carried")) for item in place] == [
        (ow.PLACE_CONFIRMED, f"chat {presence_chat_id('tg:123')} / ev-2", True)]
    # A place turn's opening: the confirmed account's words are provenance under the event's reference, in no
    # set; a presence contact's late answer (the card's ``answered_at``); and a row recorded before the update
    # (as recorded, read under its held row's room).
    focus = {"addressed": True, "actor": {"owner_confirmed": True}, "room_chat_id": 555}
    opening = _ctx(tmp_path, task_id="pres0001", task_type="presence", metadata={
        "focus": focus, "presence": {"observed_text": "Place words at the opening", "event": {"source_event_id": "ev-1"}}})
    _initialize_owner_directives(opening, [{"role": "user", "content": "[event frame]"}])
    assert _stamps(ow.work_words(opening)[0]) == [] and [
        (item["source"], item.get("msg_id"), item.get("carried")) for item in opening._owner_directives
        if item.get("source") == ow.PLACE_CONFIRMED] == [(ow.PLACE_CONFIRMED, "chat 555 / ev-1", True)]
    late_card = {"quiz_id": "q9", "question": "Go?", "state": "answered", "answered_index": 1, "options": ["No", "Yes"],
                 "presence_contact": True, "answered_after_terminal": True, "answered_at": _at(6)}
    late = _ctx(tmp_path, task_id="late0001", metadata={"late_answer": {
        "task_id": "asker", "quiz_id": "q9", "answer_snapshot": {"task_id": "asker", "quiz": late_card}}})
    _initialize_owner_directives(late, [{"role": "user", "content": "[late answer frame]"}])
    assert _stamps(ow.work_words(late)[0]) == [("ctx", "", _at(6), "2. Yes"),
                                               ("ctx", "quiz q9 of task asker", "", "Go?\n1. No\n2. Yes\n(host record: my human pressed option 2)")]
    before = _ctx(tmp_path, task_id="pre0001", metadata={"objective_author": {"kind": "task", "task_id": ROOT}, "owner_corpus": [
        {"source": "owner_mailbox", "content": "Add charts", "msg_id": "mail-1", "client_message_id": "cm-mail"},
        {"source": "owner_mailbox", "content": "Never held", "msg_id": "mail-9", "client_message_id": "cm-9"}]})
    _initialize_owner_directives(before, [{"role": "user", "content": "a drafted objective"}])
    before._held_owner_messages = {"row:held": _in_row("Add charts", "cm-mail", ts=_at(2))}
    assert _stamps(ow.work_words(before)[0]) == [("ctx", "chat 1 / cm-mail", _at(2), "Add charts"),
                                                 ("ctx", "msg mail-9", "", "Never held")]
    assert ow._earlier_refs(before) == {"chat 1 / cm-mail": "msg mail-1"}


def test_a_helper_in_the_room_shows_a_drained_message_once_and_one_message_is_one_row_under_any_carrier(tmp_path):
    """A root bound to a Project drains one room message and schedules a helper there: the field carries the row
    under its room and client id, and the helper's first request — the field, then the room read once — shows the
    message once (a mailbox-only row keeps its ``msg`` reference). A corpus row recorded before the update (a client
    id, no room) renders under the chat reference of the row this focus holds for it, paired by client id, by quiz
    id or by a direct delivery's id; an answer to a card and a direct delivery each render once, the card beside
    the answer under the asking task's name, whichever carrier holds the row. A root bound to the room after its
    admission (its admitted chat stays) still holds a drained message's chat row, found in the bound room."""
    room = _project(tmp_path)
    _chat(tmp_path, _in_row("Add charts", "cm-mail", ts=_at(2), chat_id=room))
    write_owner_message(tmp_path, "Add charts", ROOT, msg_id="m-2", client_message_id="cm-mail")
    write_owner_message(tmp_path, "Mailbox only", ROOT, msg_id="m-3")
    door = build_owner_message_ref(chat_id=room, client_message_id="cm-door", ts=_at(1), text="Fix the quota bar")
    root = _ctx(tmp_path, metadata={"origin_message_ref": door, "root_task_id": ROOT}, chat_id=room)
    messages = [{"role": "user", "content": "Fix the quota bar"}]
    _initialize_owner_directives(root, messages)
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, ROOT, None, set(), owner_ctx=root)
    field = ow.governing_words_for_schedule(root)[ow.FIELD]
    received = next(row["received"] for row in root._owner_directives if row.get("msg_id") == "m-3")
    assert [(row["ref"], row["ts"]) for row in field] == [
        (f"chat {room} / cm-door", _at(1)), (f"chat {room} / cm-mail", _at(2)), ("msg m-3", f"received {received}")]
    child = {"id": "child01", "parent_task_id": ROOT, "root_task_id": ROOT,
             "metadata": {"delegation_role": "subagent", ow.FIELD: field}}
    from ouroboros.owner_relay import task_words_text

    first = task_words_text(tmp_path, child)
    assert first.count("Add charts") == first.count("Mailbox only") == first.count("Fix the quota bar") == 1
    assert f"[{_at(2)} · owner · owner_mailbox of task {ROOT} · chat {room} / cm-mail]\nAdd charts\n" in first
    helper = _ctx(tmp_path, task_id="child01", metadata={"delegation_role": "subagent", "parent_task_id": ROOT,
                                                         "root_task_id": ROOT, ow.FIELD: field})
    assert ow.owner_words_text(helper, audience="child") == first
    # Rows recorded before the update, carried into a drafted start under their mailbox ids; the start holds the
    # chat rows (an owner mail, an answered card, a direct delivery) and the room holds the mail too: one row each.
    card = {"quiz_id": "q7", "question": "Publish?", "state": "answered", "answered_index": 0,
            "options": ["Publish it", "Not now"]}
    answer = {**_in_row("1. Publish it", "cm-ans", ts=_at(3), chat_id=room), "type": "quiz_answer", "task_id": "asker",
              "quiz": card}
    before = _ctx(tmp_path, task_id="pre0001", metadata={
        "parent_task_id": ROOT, "root_task_id": ROOT, "objective_author": {"kind": "task", "task_id": ROOT},
        "owner_corpus": [{"source": "owner_mailbox", "content": "Add charts", "msg_id": "m-2", "client_message_id": "cm-mail"},
                         {"source": "owner_quiz_answer", "content": "1. Publish it", "msg_id": "qa-7", "quiz_id": "q7",
                          "source_task_id": "asker", "reply_to": ow.answered_card_reply(card, "asker")},
                         {"source": "direct_incoming", "content": "Direct too", "msg_id": "direct-1"}]})
    _initialize_owner_directives(before, [{"role": "user", "content": "a drafted objective"}])
    before._held_owner_messages = {"row:mail": _in_row("Add charts", "cm-mail", ts=_at(2), chat_id=room), "row:answer": answer,
                                   "row:direct": _in_row("Direct too", "direct-1", ts=_at(4), chat_id=room)}
    rows = ow.work_words(before)[0]
    assert _stamps(rows) == [
        ("ctx", f"chat {room} / cm-mail", _at(2), "Add charts"),
        ("ctx", f"chat {room} / cm-ans", _at(3), "1. Publish it"),
        ("ctx", "quiz q7 of task asker", "", "Publish?\n1. Publish it\n2. Not now\n(host record: my human pressed option 1)"),
        ("ctx", f"chat {room} / direct-1", _at(4), "Direct too")]
    assert rows[2]["author"] == "Ouroboros, task asker" and rows[2]["source"] == ow.REPLY_TARGET
    assert ow._earlier_refs(before) == {f"chat {room} / cm-mail": "msg m-2", f"chat {room} / cm-ans": "msg qa-7",
                                        f"chat {room} / direct-1": "msg direct-1"}
    # Unpaired, named: a pre-update row whose chat row no focus holds keeps its mailbox reference, once, its card
    # beside it as its corpus row carries it.
    before._held_owner_messages = {"row:mail": _in_row("Add charts", "cm-mail", ts=_at(2), chat_id=room)}
    assert [row["ref"] for row in ow.work_words(before)[0]] == [
        f"chat {room} / cm-mail", "msg qa-7", "quiz q7 of task asker", "msg direct-1"]
    # An answer recorded before its card rode with it (no ``reply_to``) takes the card of the held twin: the card
    # is the next row, not a row after the rest of the corpus.
    before._held_owner_messages["row:answer"] = answer
    next(row for row in before._owner_directives if row["source"] == "owner_quiz_answer").pop("reply_to")
    assert [row["ref"] for row in ow.work_words(before)[0]] == [
        f"chat {room} / cm-mail", f"chat {room} / cm-ans", "quiz q7 of task asker", "msg direct-1"]
    # A room answer read live (no focus holds it): once, its card beside it under the asking task's name.
    _chat(tmp_path, answer)
    rows = ow.work_words(helper)[0]
    assert _stamps(rows)[-2:] == [("room", f"chat {room} / cm-ans", _at(3), "1. Publish it"), (
        "room", "quiz q7 of task asker", "", "Publish?\n1. Publish it\n2. Not now\n(host record: my human pressed option 1)")]
    assert rows[-1]["author"] == "Ouroboros, task asker" and rows[-1]["source"] == ow.REPLY_TARGET
    assert [row["ref"] for row in rows].count(f"chat {room} / cm-ans") == 1
    # A root bound to the room after admission keeps its admitted chat (1): the chat row of a message it drains is
    # found in the bound room, so the message is one row under its room and accept time, and its mailbox
    # reference stays an identity of the same row.
    _chat(tmp_path, _in_row("Bind me late", "cm-late", ts=_at(5), chat_id=room))
    write_owner_message(tmp_path, "Bind me late", ROOT, msg_id="m-5", client_message_id="cm-late")
    stale = _ctx(tmp_path, metadata={"origin_message_ref": door, "root_task_id": ROOT}, chat_id=1)
    messages = [{"role": "user", "content": "Fix the quota bar"}]
    _initialize_owner_directives(stale, messages)
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, ROOT, None, set(), owner_ctx=stale)
    assert [(row["ref"], row["ts"]) for row in ow.work_words(stale)[0] if row["text"] == "Bind me late"] == [
        (f"chat {room} / cm-late", _at(5))]
    assert ow._earlier_refs(stale) == {f"chat {room} / cm-late": "msg m-5"}


def test_task_governing_words_reads_both_payload_shapes(tmp_path):
    rows = [{"text": "Words", "source": "initial_user", "carrier": "ctx", "task_id": ROOT, "ts": "", "ref": ""}]
    assert ow.task_governing_words({ow.FIELD: rows}) == (rows, "")
    assert ow.task_governing_words({"metadata": {ow.FIELD: rows}}) == (rows, "")
    assert ow.task_governing_words({"metadata": {ow.ABSENT_FIELD: "task_type=presence"}}) == ([], "task_type=presence")
    assert ow.task_governing_words({"metadata": {}}) == ([], "")


def test_dedup_by_normalized_text_checksum(tmp_path):
    corpus = [{"source": "initial_user", "content": "Do  the\nthing"},
              {"source": "owner_mailbox", "content": "Do the thing "},
              {"source": "owner_mailbox", "content": "Do the other thing"}]
    rows = ow.governing_words_for_schedule(_ctx(tmp_path, directives=corpus))[ow.FIELD]
    assert _texts(rows) == ["Do  the\nthing", "Do the other thing"]


# --- history carriers, in order -----------------------------------------------------------------------------

def _all_carriers(tmp_path):
    origin_ref = build_owner_message_ref(chat_id=1, client_message_id="msg-o", ts=TS, text="origin text")
    task = {"id": ROOT, "origin_message_ref": origin_ref, "origin_message_text": "origin text"}
    _bind(tmp_path, source_text="binding text",
          source_ref=build_owner_message_ref(chat_id=1, client_message_id="msg-b", ts=TS, text="binding text"))
    _chat(tmp_path, _in_row("annotated text", "msg-a"))
    _annotate(tmp_path, "msg-a")
    write_owner_message(tmp_path, "mailbox text", ROOT, msg_id="mb-1", kind=KIND_OWNER_TEXT)
    return task


def test_history_carriers_fall_through_in_order(tmp_path):
    task = _all_carriers(tmp_path)
    rows, absent = ow.task_owner_words(tmp_path, ROOT, task=task)
    assert (_texts(rows), rows[0]["carrier"], absent) == (["origin text"], "origin_message", "")
    rows, _ = ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT})
    assert (_texts(rows), rows[0]["carrier"]) == (["binding text"], "binding")
    (tmp_path / "state" / "project_task_bindings.json").unlink()
    rows, _ = ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT})
    assert (_texts(rows), rows[0]["carrier"], rows[0]["source"]) == (["annotated text"], "annotation",
                                                                     "promote_chat_to_task")
    (tmp_path / "logs" / "chat_annotations.jsonl").unlink()
    rows, _ = ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT})
    assert (_texts(rows), rows[0]["carrier"], rows[0]["ref"]) == (["mailbox text"], "mailbox", "msg mb-1")
    for path in (tmp_path / "memory" / "owner_mailbox").iterdir():
        path.unlink()
    assert ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT}) == ([], ow.NOT_RECORDED)
    assert ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT, "metadata": {"initiator": "consciousness"}}) == (
        [], "initiator=consciousness")


def test_origin_and_binding_refs_resolve_through_the_chain(tmp_path):
    message = "Ref-only origin"
    ref = build_owner_message_ref(chat_id=1, client_message_id="msg-r", ts=TS, text=message)
    assert ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT, "origin_message_ref": ref}) == ([], ow.NOT_RECORDED)
    _chat(tmp_path, _in_row(message, "msg-r"))
    rows, _ = ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT, "metadata": {"origin_message_ref": ref}})
    assert rows == [{"text": message, "source": "origin_message", "carrier": "origin_message", "task_id": ROOT,
                     "ts": TS, "ref": "chat 1 / msg-r"}]
    _bind(tmp_path, source_ref=ref)
    assert ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT})[0][0]["carrier"] == "binding"


def test_manual_target_annotation_is_not_a_carrier(tmp_path):
    _chat(tmp_path, _in_row("Which task?", "msg-m"), _in_row("Steer this", "msg-s"))
    _annotate(tmp_path, "msg-m", action="steer_task", status="needs_manual_target")
    assert ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT}) == ([], ow.NOT_RECORDED)
    _annotate(tmp_path, "msg-s", action="steer_task", status="delivered", token="t2")
    assert _texts(ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT})[0]) == ["Steer this"]


def test_annotation_counts_only_through_an_inbound_row(tmp_path):
    agent_id = "agent-steer:tok-1"
    _annotate(tmp_path, agent_id, action="steer_task", status="delivered")
    _chat(tmp_path, {**_in_row("Ouroboros relayed this", agent_id), "direction": "out"})
    assert ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT}) == ([], ow.NOT_RECORDED)
    _chat(tmp_path, _in_row("The owner's own steer", agent_id))
    rows, _ = ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT})
    assert (_texts(rows), rows[0]["source"]) == (["The owner's own steer"], "steer_task")


@pytest.mark.parametrize("injection", [{"system_type": "proactive_message"}, {"presence": {"room": "x"}},
                                       {"source": "skill_repair"}])
def test_non_owner_inbound_rows_are_cut(tmp_path, injection):
    _chat(tmp_path, _in_row("Injected", "msg-i", **injection))
    _annotate(tmp_path, "msg-i")
    assert ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT}) == ([], ow.NOT_RECORDED)
    _chat(tmp_path, _in_row("Plain owner row", "msg-p"))
    _annotate(tmp_path, "msg-p", token="t2")
    assert _texts(ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT})[0]) == ["Plain owner row"]


def test_mailbox_carrier_takes_owner_text_only(tmp_path):
    write_owner_message(tmp_path, "a sibling's message", ROOT, msg_id="tm-1", kind=KIND_TASK_MESSAGE)
    assert ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT}) == ([], ow.NOT_RECORDED)
    write_owner_message(tmp_path, "owner follow-up", ROOT, msg_id="mb-2", kind=KIND_OWNER_TEXT)
    assert _texts(ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT})[0]) == ["owner follow-up"]


def test_child_resolves_its_root_unless_it_carries_the_field(tmp_path):
    _bind(tmp_path, source_text="root's origin")
    child = {"id": "child01", "root_task_id": ROOT, "delegation_role": "subagent"}
    rows, _ = ow.task_owner_words(tmp_path, "child01", task=child)
    assert (_texts(rows), rows[0]["task_id"]) == (["root's origin"], ROOT)
    carried = [{"text": "carried", "source": "initial_user", "carrier": "ctx", "task_id": ROOT, "ts": "", "ref": ""}]
    assert ow.task_owner_words(tmp_path, "child01", task={**child, "metadata": {ow.FIELD: carried}}) == (carried, "")


# --- rendering ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("audience", ["child", "session", "writer"])
def test_render_is_verbatim_and_whole_for_every_audience(audience):
    long_text = ("Слово владельца. " * 700)[:10_250]
    assert len(long_text) == 10_250
    row = {"text": long_text, "source": "initial_user", "carrier": "ctx", "task_id": "9e0a326d", "ts": TS,
           "ref": "chat 1 / msg-1790746905659-12"}
    text = ow.render_owner_words([row], audience=audience, root_task_id="9e0a326d")
    assert long_text in text and text.endswith(long_text)
    assert f"[{TS} · owner · initial_user of task 9e0a326d · chat 1 / msg-1790746905659-12]\n" in text
    assert text.count(long_text) == 1


def test_render_absence_is_one_line_and_unknown_audience_is_refused():
    line = ow.render_owner_words([], "initiator=consciousness", audience="child")
    assert line == "No words of my human are recorded for this work (host marker: initiator=consciousness)."
    assert "\n" not in line
    assert ow.render_owner_words([], "", audience="session") == ""
    with pytest.raises(ValueError):
        ow.render_owner_words([], "x", audience="everyone")


def test_owner_words_text_renders_the_scheduled_words(tmp_path):
    ctx = _ctx(tmp_path, metadata={"root_task_id": ROOT}, directives=[{"source": "initial_user", "content": "Ask"}])
    text = ow.owner_words_text(ctx)
    assert text.startswith("## My human's words for this work (verbatim, dated, host-attested)\n") and "\nAsk" in text
    assert ow.owner_words_text(_ctx(tmp_path, metadata={"initiator": "consciousness"}, directives=[])) == (
        "No words of my human are recorded for this work (host marker: initiator=consciousness).")


# --- no model ----------------------------------------------------------------------------------------------

def _llm_imports(source: str) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(str(node.module or ""))
    return sorted(name for name in names if name.startswith("ouroboros.llm"))


def test_module_imports_no_model_client():
    source = (REPO / "ouroboros" / "owner_words.py").read_text(encoding="utf-8")
    assert _llm_imports(source) == []
    assert _llm_imports("def f():\n    from ouroboros.llm import LLMClient\nimport ouroboros.llm_openai_compatible\n") == [
        "ouroboros.llm", "ouroboros.llm_openai_compatible"]


def test_live_speech_is_citable_but_not_an_owner_source(tmp_path):
    from ouroboros.owner_continue import CORPUS_INITIAL_SOURCES, CORPUS_OWNER_SOURCES
    from ouroboros.loop_messages import owner_reply_owed
    from ouroboros.safety import _owner_yes

    ctx = _ctx(tmp_path, directives=[{"source": "live_call", "content": "Keep all these words. Choose the other model.", "msg_id": "spoken"}])
    before = [dict(row) for row in ctx._owner_directives]
    assert ow.cited_owner_basis(ctx, "Choose the other model.")[0] is None
    basis, error = ow.cited_owner_basis(ctx, "Choose the other model.", call_speech=True)
    assert not error and basis == {"source": "live_call", "cited": "Choose the other model.", "msg_id": "spoken"}
    assert ctx._owner_directives == before
    assert "live_call" not in ow.OWNER_SOURCES | CORPUS_INITIAL_SOURCES | CORPUS_OWNER_SOURCES
    assert ow.carried_owner_corpus(ctx) == [] and ow.directive_owner_rows(ctx) == []
    assert not owner_reply_owed(ctx)
    assert _owner_yes(ctx, lambda: pytest.fail("a caption never supplies an answered approval card")) == ""
    assert ow.cited_owner_basis(ctx, "invented", call_speech=True)[0] is None
    # Even a caption alongside a card is not the answered-card approval channel.
    ctx._owner_directives[0]["reply_to"] = {"address": "quiz q1", "text": "Publish these words.\nroom-a"}
    subject = lambda: ("Publish these words.", ["room-a"], False, [])
    assert _owner_yes(ctx, subject) == ""
    ctx._owner_directives[0]["source"] = "owner_mailbox"
    assert _owner_yes(ctx, subject) == "spoken"
    assert owner_reply_owed(ctx)
