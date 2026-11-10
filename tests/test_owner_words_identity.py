"""One message of my human is one row wherever its words are read back, every carrier of them is read, and each
row keeps its own author (review RV1W: Opus F1, Astra F4/F5, workflow #4/#5/#6/#11).

Each guard is checked both ways: one source under two carriers is one row while two messages with the same words
stay two; a carried corpus is read while its absence still says so; a reply target keeps its author in the
by-reference form while my human's row keeps theirs; one place snapshot serves a whole read.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_store as ds, owner_words as ow
from ouroboros.chat_chain import format_address, row_address
from ouroboros.project_dialogue import build_owner_message_ref

ROOT = "root0001"
TS = "2026-09-30T05:41:45+00:00"
TARGET = {"address": "row:1@2026-09-30T05:40:00+00:00#0123456789ab", "author": "Ouroboros",
          "text": "The proposed plan, step by step."}


def _stored(tmp_path, text, client_id, *, ts=TS):
    ds.append(tmp_path, "chat", {"ts": ts, "direction": "in", "chat_id": 1, "text": text,
                                 "client_message_id": client_id, "task_id": ""})
    return next(entry.row for entry in ds.iter_probes(tmp_path, "chat", [("client_message_id", client_id)]))


def _root_holding(tmp_path, origin, held):
    ref = build_owner_message_ref(chat_id=1, client_message_id=origin["client_message_id"], ts=origin["ts"],
                                  text=origin["text"])
    return {"id": ROOT, "metadata": {"origin_message_ref": ref},
            "held_owner_messages": {format_address(row_address(row)): row for row in held}}


def test_a_message_its_root_holds_and_names_as_its_origin_is_one_row(tmp_path):
    """Opus F1 E1: the page writer read the origin's row and the held copy of it under two references, the second
    stamped as a mailbox delivery that never happened."""
    words = "Please rebuild the index and tell me what changed."
    origin = _stored(tmp_path, words, "msg-o")
    rows, absent = ow.task_owner_words(tmp_path, ROOT, task=_root_holding(tmp_path, origin, [origin]))
    assert [(row["source"], row["ref"], row["text"]) for row in rows] == [("origin_message", "chat 1 / msg-o", words)]
    assert absent == ""


@pytest.mark.parametrize("carrier", ["origin_ref", "origin_text", "binding", "corpus"])
@pytest.mark.parametrize("source_state", ["readable", "missing", "error"])
def test_legacy_origin_and_held_copy_share_identity_without_collapsing_another_message(tmp_path, monkeypatch, carrier, source_state):
    from ouroboros.owner_relay import held_owner_messages, retain_owner_message
    from tests.test_owner_relay import context

    for ts in (TS, "2026-09-30T05:42:00+00:00"):
        ds.append(tmp_path, "chat", {"chat_id": 1, "direction": "in", "ts": ts, "text": "yes"})
    origin, other = [item.row for item in ds.iter_rows(tmp_path, "chat")]
    ref = build_owner_message_ref(chat_id=1, client_message_id="", ts=TS, text="yes")
    ctx = context(tmp_path); ctx.task_metadata = {"origin_message_ref": ref}
    held_owner_messages(ctx)
    retain_owner_message(ctx, other)
    if source_state != "readable":
        def unavailable(*_args):
            if source_state == "error":
                raise OSError("source unavailable")
            return None
        monkeypatch.setattr("ouroboros.project_dialogue.resolve_owner_message_source", unavailable)
    record = {"id": ROOT, "held_owner_messages": ctx._held_owner_messages}
    if carrier == "binding":
        monkeypatch.setattr("ouroboros.projects_registry.project_task_bindings", lambda _root: {
            ROOT: {"source_text": "yes", "source_ref": ref}})
    else:
        record["metadata"] = {"origin_message_ref": ref}
        if carrier == "origin_text":
            record["metadata"]["origin_message_text"] = "yes"
        elif carrier == "corpus":
            record["metadata"]["owner_corpus"] = [{"source": "origin_message", "content": "yes"}]
    rows, absent = ow.task_owner_words(tmp_path, ROOT, task=record)
    assert not absent
    assert [(r["ref"], r["text"]) for r in rows] == [(format_address(row_address(origin)), "yes"),
                                                     (format_address(row_address(other)), "yes")]


@pytest.mark.parametrize("author", ["door", "task", "continuation"])
def test_a_legacy_origin_quoted_back_into_the_live_turn_is_one_known_source(tmp_path, author):
    from ouroboros.loop_messages import _initialize_owner_directives
    from ouroboros.owner_relay import inject_relay, owner_quote, relay_envelope
    from tests.test_owner_relay import context

    row = {"chat_id": 1, "direction": "in", "ts": TS, "text": "yes"}
    ds.append(tmp_path, "chat", row)
    stored = next(item.row for item in ds.iter_rows(tmp_path, "chat"))
    ctx = context(tmp_path)
    ctx.task_metadata = {"origin_message_ref": build_owner_message_ref(chat_id=1, client_message_id="", ts=TS, text="yes"),
                         "origin_message_text": "yes"}
    if author != "door":
        ctx.task_metadata.update(objective_author={"kind": author}, owner_corpus=[
            {"source": "origin_message", "content": "yes"},
            {"source": "origin_message", "content": "yes", "chat_id": "1", "client_message_id": "another"}])
    _initialize_owner_directives(ctx, [{"role": "user", "content": "yes"}])
    inject_relay(ctx, {"relay": relay_envelope("same", [owner_quote(tmp_path, stored)], chat_id=1),
                       "owner_rows": [stored], "text": "context", "source_task_id": "other"}, [])
    expected = ["yes"] if author == "door" else ["yes", "yes"]
    assert [r["text"] for r in ow.directive_owner_rows(ctx)] == expected
    assert [r["text"] for r in ow.governing_words_for_schedule(ctx)[ow.FIELD]] == expected


def test_missing_origin_identity_does_not_infer_a_match_from_equal_words(tmp_path):
    first = _stored(tmp_path, "yes", "one")
    second = _stored(tmp_path, "yes", "two", ts="2026-09-30T05:42:00+00:00")
    rows, _ = ow.task_owner_words(tmp_path, ROOT, task={"origin_message_text": "yes",
        "held_owner_messages": {format_address(row_address(r)): r for r in (first, second)}})
    assert {r["ref"] for r in rows} >= {"chat 1 / one", "chat 1 / two"}
    assert len(rows) == 3, "without an origin reference, two possible identical sources cannot be guessed into one"


def test_two_messages_with_the_same_words_stay_two_rows(tmp_path):
    origin = _stored(tmp_path, "yes", "msg-a")
    other = _stored(tmp_path, "yes", "msg-b", ts="2026-09-30T05:42:00+00:00")
    rows, _absent = ow.task_owner_words(tmp_path, ROOT, task=_root_holding(tmp_path, origin, [origin, other]))
    assert [(row["source"], row["ref"]) for row in rows] == [("origin_message", "chat 1 / msg-a"),
                                                             ("owner_mailbox", "chat 1 / msg-b")]


def test_the_page_writer_reads_the_words_a_follow_up_carries(tmp_path):
    corpus = [{"source": "owner_mailbox", "content": "Check staging every Monday.", "msg_id": "m1", "reply_to": TARGET},
              {"source": "initial_text", "content": "a work order, not my human's words"}]
    record = {"id": "followup-root", "metadata": {"objective_author": {"kind": "task", "task_id": "creator"},
                                                  "owner_corpus": corpus}}
    rows, absent = ow.task_owner_words(tmp_path, "followup-root", task=record)
    assert [(row["source"], row["text"]) for row in rows] == [("owner_mailbox", "Check staging every Monday."),
                                                              (ow.REPLY_TARGET, TARGET["text"])]
    assert absent == "" and rows[1]["author"] == "Ouroboros"
    bare, marker = ow.task_owner_words(tmp_path, "followup-root", task={"id": "followup-root", "metadata": {}})
    assert bare == [] and marker == ow.NOT_RECORDED, "without a carried corpus the absence is still named"


def test_the_by_reference_form_keeps_each_rows_own_author(tmp_path, monkeypatch):
    from ouroboros import memory_fallback

    rows = [ow._row("Do the migration tonight.", source="origin_message", carrier="origin_message", task_id="t1",
                    ts=TS, ref="chat 1 / msg-o"),
            ow._row("", source=ow.REPLY_TARGET, carrier="ctx", task_id="t1", ref=TARGET["address"], author="Sasha")]
    monkeypatch.setattr(ow, "task_owner_words", lambda root, task: (rows, ""))
    item = memory_fallback._Row(({}, {"root_task_id": "t1"}, 0), "t1", "a line", "")
    _group, whole, pointer = memory_fallback._owner_words(tmp_path, [item])["t1"]
    mine, target = pointer.splitlines()[1:]
    assert mine.startswith(f"[{TS} · owner · origin_message of task t1 · chat 1 / msg-o] (25 chars")
    assert target == (f"[Sasha: the message my human replied to, not my human's words · {TARGET['address']}] "
                      f"{ow.UNREAD_TARGET}"), "a reply target is never signed as my human"
    assert "· owner ·" not in target and whole.count(ow.words_stamp(rows[1])) == 1, "one stamp in both forms"


def test_one_place_snapshot_serves_a_whole_read(tmp_path, monkeypatch):
    from ouroboros import places
    from tests.test_place_owner_author import _place_row

    connected = places.connect_place(tmp_path, connector="bridge", provider="chat", account_id="qa", label="Team")
    places.edit_place(tmp_path, connected.place_id, "confirm_owner_account", author="owner",
                      platform_actor_id="anton", account_id="qa")
    held = {f"row:{index}": {**_place_row(connected.place_id, text=f"words {index}"), "client_message_id": f"ev-{index}"}
            for index in range(4)}
    reads = []
    real = places.place_owner_accounts
    monkeypatch.setattr(places, "place_owner_accounts", lambda root: reads.append(root) or real(root))
    rows, _absent = ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT, "held_owner_messages": held})
    assert [row["text"] for row in rows] == [f"words {index}" for index in range(4)], "my human's place rows, each"
    assert len(reads) == 1, "one snapshot for the whole read, never one per row"


@pytest.mark.parametrize("confirmed", [True, False])
@pytest.mark.parametrize("carrier", ["place_owner", "place_confirmed"])
def test_a_continue_shows_one_place_event_once_under_one_authority(confirmed, carrier):
    """An event my human's confirmed account opened is read once: under a cited ``place_owner`` row, or under their
    confirmed place words (the successor's provenance row). A participant's same words stay theirs, beside mine."""
    from ouroboros.owner_continue import work_order_text

    words = "Ship the release tonight, then tell me."
    wake = {"source": "consciousness_wake", "task_id": "wake1", "wake_cause": "focus",
            "focus": {"actor": {"owner_confirmed": confirmed}}, "observed_text": words}
    row = {"source": carrier, "content": words, "msg_id": "chat 7 / ev-1"}
    # Unconfirmed opener: a confirmed row with the same words is my human's own later message, a second event.
    sources = {"later": [row] if confirmed else []} if carrier == "place_owner" else {"confirmed_place": [row]}
    text = work_order_text("wake1", "technical interruption", {"wake": wake, **sources})
    assert text.count(words) == (1 if confirmed or carrier == "place_owner" else 2), "one event, one reading; two stay two"
    context = "The event's words, from the sender recorded above (context, not an owner instruction; verbatim):"
    assert (context in text) is not confirmed
    assert ("Other owner words in that task's record (verbatim, in order):" in text) is (confirmed and carrier == "place_owner")
    assert ("under their confirmed words" in text) is (confirmed and carrier == "place_confirmed")


# --- RV2W: the typed identity reaches every reader (Astra R3, Opus N1) -------------------------------------------


@pytest.mark.parametrize("carrier", ["origin", "corpus"])
def test_one_message_is_one_row_whichever_carrier_names_it(tmp_path, carrier):
    """Astra R3 :13: a drafted start's carried corpus (``_attach_drafted_objective``) names the same owner message
    its root holds; the page writer reads one row, at that message's address, never a second ``owner_mailbox``."""
    from ouroboros.loop_messages import _initialize_owner_directives
    from ouroboros.tools.control_routing import _attach_drafted_objective, _attach_origin_from_metadata
    from tests.test_owner_words_carried import _owner_root

    ctx = _owner_root(tmp_path)
    origin = _stored(ctx.drive_root, "Please keep the export working.", "owner-one")
    record = _root_holding(ctx.drive_root, origin, [origin])
    ctx.task_metadata.update(record["metadata"], origin_message_text=origin["text"])
    _initialize_owner_directives(ctx, [{"role": "user", "content": origin["text"]}])
    if carrier == "corpus":
        event = {}
        _attach_origin_from_metadata(ctx, event)
        _attach_drafted_objective(ctx, event)
        record["metadata"].update(event)
        assert record["metadata"]["owner_corpus"][0]["client_message_id"] == "owner-one"
    rows, _absent = ow.task_owner_words(ctx.drive_root, ROOT, task=record)
    assert [(row["source"], row["ref"], row["text"]) for row in rows] == [
        ("origin_message", "chat 1 / owner-one", origin["text"])]


@pytest.mark.parametrize("reader", ["helper", "writer", "evidence"])
def test_two_identified_instructions_in_the_same_words_stay_two(tmp_path, reader):
    """Astra R3 :35: two owner turns approve two assignments with the same "Yes."; create → update → fire keeps both
    ids, and the helper's words, the page writer and the terminal evidence each keep two rows."""
    from ouroboros.loop_messages import _initialize_owner_directives
    from ouroboros.review_evidence_sections import _accept_owner_directives
    from ouroboros.tools.followup import _handle_schedule_followup, _manage_schedules
    from supervisor import queue as schedules
    from tests.test_owner_words_carried import _owner_root

    ctx = _owner_root(tmp_path, {"origin_message_ref": {"chat_id": 1, "client_message_id": "first-yes"}})
    _initialize_owner_directives(ctx, [{"role": "user", "content": "Yes."}])
    assert _handle_schedule_followup(ctx, relation="independent", objective="Check staging.",
                                     cron="0 9 * * 1").startswith("FOLLOWUP_SCHEDULED")
    [registered] = schedules.list_scheduled_tasks(ctx.drive_root)["tasks"]
    ctx.task_metadata["origin_message_ref"]["client_message_id"] = "second-yes"
    del ctx._owner_directives
    _initialize_owner_directives(ctx, [{"role": "user", "content": "Yes."}])
    assert json.loads(_manage_schedules(ctx, action="update", schedule_id=registered["id"],
                                       objective="Check production.", reason="Owner approved the change."))["ok"]
    [updated] = schedules.list_scheduled_tasks(ctx.drive_root)["tasks"]
    successor = schedules._task_from_schedule(updated)
    assert [row["client_message_id"] for row in successor["metadata"]["owner_corpus"]] == ["first-yes", "second-yes"]
    live = SimpleNamespace(task_id="next-run", task_metadata=successor["metadata"])
    _initialize_owner_directives(live, [{"role": "user", "content": successor["text"]}])
    if reader == "helper":
        words = [(row["text"], row["ref"]) for row in ow.governing_words_for_schedule(live)[ow.FIELD]]
    elif reader == "evidence":
        words = [(row["content"], row["client_message_id"]) for row in _accept_owner_directives(live, None, "next-run")]
    else:
        words = [(row["text"], row["ref"]) for row in ow.task_owner_words(ctx.drive_root, "next-run", task=successor)[0]]
    assert [text for text, _identity in words] == ["Yes.", "Yes."] and len({identity for _text, identity in words}) == 2


def test_rows_without_any_id_still_repeat_only_by_their_words():
    """The other side of the identity rule: rows that name no id at all keep the earlier dedup by their words."""
    from ouroboros.review_evidence_sections import _accept_owner_directives

    ctx = SimpleNamespace(task_id="run", _owner_directives=[
        {"source": "owner_mailbox", "content": "Yes."}, {"source": "owner_mailbox", "content": "Yes."},
        {"source": "owner_mailbox", "content": "Yes.", "msg_id": "m-1"}])
    assert [row.get("msg_id") for row in _accept_owner_directives(ctx, None, "run")] == [None, "m-1"]


def test_legacy_held_rows_without_client_ids_keep_their_canonical_handles(tmp_path):
    """Legacy held rows have no client id, so their retained row addresses must remain distinct."""
    held = {
        "row:1@2026-09-30T05:40:00+00:00#aaaaaaaaaaaa": {
            "direction": "in", "chat_id": 1, "text": "first legacy instruction",
            "ts": "2026-09-30T05:40:00+00:00"
        },
        "row:1@2026-09-30T05:41:00+00:00#bbbbbbbbbbbb": {
            "direction": "in", "chat_id": 1, "text": "second legacy instruction",
            "ts": "2026-09-30T05:41:00+00:00"
        },
    }
    rows, absent = ow.task_owner_words(tmp_path, ROOT, task={"id": ROOT, "held_owner_messages": held})
    assert absent == ""
    assert [(row["text"], row["ref"]) for row in rows] == [
        ("first legacy instruction", next(iter(held))),
        ("second legacy instruction", list(held)[1]),
    ]


@pytest.mark.parametrize("carrier", ["origin_text", "origin_ref", "binding", "annotation", "held", "mailbox"])
@pytest.mark.parametrize("target_kind", ["message", "task_card"])
def test_fallback_carriers_keep_the_whole_reply_target(tmp_path, monkeypatch, carrier, target_kind):
    """The same stored reply reaches the page writer through each supported fallback, not only its live corpus."""
    from ouroboros.chat_references import reference_from_row
    from ouroboros.owner_relay import owner_quote, write_relay
    from ouroboros.project_dialogue import append_chat_annotation

    proposal = {"chat_id": 1, "direction": "out", "ts": "2026-09-30T05:40:00Z",
                "text": "A whole proposal. " * 100, "client_message_id": "proposal"}
    ds.append(tmp_path, "chat", proposal)
    reference = (reference_from_row(proposal) if target_kind == "message"
                 else {"address": "task:some-work", "snippet": "A work card, not a message."})
    owner = {"chat_id": 1, "direction": "in", "ts": TS, "text": "Yes, do that.",
             "client_message_id": "reply", "reply_to": reference}
    ds.append(tmp_path, "chat", owner)
    stored = next(item.row for item in ds.rows_by_client_id(tmp_path, "reply"))
    ref = build_owner_message_ref(chat_id=1, client_message_id="reply", ts=TS, text=owner["text"])
    record = {"id": ROOT}
    if carrier.startswith("origin"):
        record["origin_message_ref"] = ref
        if carrier == "origin_text":
            record["origin_message_text"] = owner["text"]
    elif carrier == "binding":
        monkeypatch.setattr("ouroboros.projects_registry.project_task_bindings", lambda _root: {
            ROOT: {"source_text": owner["text"], "source_ref": ref}})
    elif carrier == "annotation":
        append_chat_annotation(tmp_path, "reply", action="steer_task", target=ROOT, status="read", routing_token="r")
    elif carrier == "held":
        record["held_owner_messages"] = {format_address(row_address(stored)): stored}
    else:
        write_relay(tmp_path, ROOT, relay_id="pending", note="context", quotes=[owner_quote(tmp_path, stored)],
                    chat_id=1, teller_task_id="teller", canonical_root=tmp_path)
    rows, absent = ow.task_owner_words(tmp_path, ROOT, task=record)
    assert not absent and [row["text"] for row in rows if row["source"] != ow.REPLY_TARGET] == [owner["text"]]
    if target_kind == "task_card":
        assert not [row for row in rows if row["source"] == ow.REPLY_TARGET], "a task card is not someone's words"
        return
    [target] = [row for row in rows if row["source"] == ow.REPLY_TARGET]
    assert target["text"] == proposal["text"] and target["author"] != "owner"


@pytest.mark.parametrize("carrier", ["origin_message", "binding"])
def test_retained_owner_text_survives_unreadable_reply_source(tmp_path, monkeypatch, carrier):
    def unavailable(*_args, **_kw):
        raise OSError("source temporarily unreadable")
    monkeypatch.setattr("ouroboros.project_dialogue.resolve_owner_message_source", unavailable)
    rows = ow._message_rows(tmp_path, ROOT, "Exact retained owner words", {"chat_id": 1,
                            "client_message_id": "original"}, carrier)
    assert [r["text"] for r in rows] == ["Exact retained owner words"]


def _drafted_corpus(tmp_path, stored):
    from ouroboros.tools.tool_context import ToolContext

    ref = build_owner_message_ref(chat_id=1, client_message_id=stored["client_message_id"], ts=stored["ts"],
                                  text=stored["text"])
    main = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="main0001", current_task_type="task",
                       task_metadata={"origin_message_ref": ref, "origin_message_text": stored["text"]})
    main._owner_directives = [{"source": "initial_user", "content": stored["text"]}]
    return ref, ow.carried_owner_corpus(main, stored["text"])  # tools/control_routing._attach_drafted_objective


@pytest.mark.parametrize("successor", ["drafted", "continue"])
def test_a_drafted_start_and_a_continue_successor_read_their_origin_once(tmp_path, successor):
    """Opus N1 (E3, E4): the corpus row of the origin stands at the record's own origin address and time, so the
    copy the root's context holds (``owner_relay.held_owner_messages``) is the same row, not a false mailbox one."""
    from ouroboros.owner_continue import owner_corpus_rows
    from ouroboros.owner_relay import held_owner_messages
    from ouroboros.tools.tool_context import ToolContext

    words = "Please rebuild the index and tell me what changed."
    stored = _stored(tmp_path, words, "msg-o")
    ref, corpus = _drafted_corpus(tmp_path, stored)
    if successor == "continue":
        corpus = owner_corpus_rows({"original": {"source": "origin_message", "content": words, "origin_message_ref": ref},
                                    "later": []})
    author = {"kind": "task" if successor == "drafted" else "continuation", "task_id": "main0001"}
    metadata = {"origin_message_ref": ref, "origin_message_text": words, "objective_author": author,
                "owner_corpus": corpus}
    worker = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="root2", task_metadata=metadata,
                         current_task_type="task")
    record = {"id": "root2", "metadata": metadata, "held_owner_messages": dict(held_owner_messages(worker))}
    assert record["held_owner_messages"], "the root's context holds its origin row"
    rows, _absent = ow.task_owner_words(tmp_path, "root2", task=record)
    assert [(row["source"], row["ref"], row["ts"]) for row in rows] == [("origin_message", "chat 1 / msg-o", stored["ts"])]


def test_the_stored_drafted_root_reads_its_origin_once_through_the_writers_loader(tmp_path):
    """Opus N1 (E3s): the page writer's own record read (``_task_record``) of a drafted start whose context build
    persisted the held copy."""
    from ouroboros.owner_relay import held_owner_messages
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.tool_context import ToolContext

    words = "Please rebuild the index and tell me what changed."
    stored = _stored(tmp_path, words, "msg-o")
    ref, corpus = _drafted_corpus(tmp_path, stored)
    author = {"kind": "task", "task_id": "main0001"}
    write_task_result(tmp_path, "drafted02", "running", origin_message_ref=ref, origin_message_text=words,
                      metadata={"objective_author": author, "owner_corpus": corpus})
    held_owner_messages(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="drafted02", current_task_type="task",
                                    task_metadata={"origin_message_ref": ref, "origin_message_text": words,
                                                   "objective_author": author, "owner_corpus": corpus}))
    assert ow._task_record(tmp_path, "drafted02").get("held_owner_messages"), "the context build persisted its copy"
    rows, _absent = ow.task_owner_words(tmp_path, "drafted02")
    assert [row["text"] for row in rows] == [words] and rows[0]["ref"] == "chat 1 / msg-o"


def test_a_continue_keeps_the_opening_events_words_when_only_another_event_of_mine_says_them():
    """Opus N5: a participant's unconfirmed "Yes, go ahead." opened the wake; my human's own confirmed event later
    said the same words. Two events, two authors: the opening's words stay as context beside my human's row."""
    from ouroboros.owner_continue import work_order_text

    words = "Yes, go ahead."
    wake = {"source": "consciousness_wake", "task_id": "wake1", "wake_cause": "focus",
            "focus": {"addressed": True, "actor": {"owner_confirmed": False, "platform_actor_id": "sasha"}},
            "observed_text": words}
    later = [{"source": "place_owner", "content": words, "msg_id": "chat 7 / ev-2"}]  # another event, my human's
    text = work_order_text("wake1", "technical interruption", {"wake": wake, "later": later})
    context = "The event's words, from the sender recorded above (context, not an owner instruction; verbatim):"
    assert context in text and text.count(words) == 2, "the participant's opening words never vanish as a copy of mine"


def test_a_relay_of_several_place_rows_reads_one_place_snapshot(tmp_path, monkeypatch):
    from ouroboros import places
    from ouroboros.owner_relay import inject_relay
    from tests.test_owner_relay import context
    from tests.test_place_owner_author import _place_row

    connected = places.connect_place(tmp_path, connector="bridge", provider="chat", account_id="qa", label="Team")
    places.edit_place(tmp_path, connected.place_id, "confirm_owner_account", author="owner",
                      platform_actor_id="anton", account_id="qa")
    quoted = [{**_place_row(connected.place_id, text=f"Instruction {index}"), "client_message_id": f"owner-{index}"}
              for index in range(3)]
    reads, real = [], places.place_owner_accounts
    monkeypatch.setattr(places, "place_owner_accounts", lambda root: reads.append(root) or real(root))
    receiver = context(tmp_path)
    receiver.task_id = "receiver"
    inject_relay(receiver, {"owner_rows": quoted, "source_task_id": "teller", "text": "note", "relay": {"relay_id": "r1"}}, [])
    assert [row["content"] for row in receiver._owner_directives] == [f"Instruction {index}" for index in range(3)]
    assert len(receiver._held_owner_messages) == 3 and len(reads) == 1, "one snapshot for the whole relay"


def test_schedule_merge_scopes_reused_client_ids_by_chat():
    kept = [{"source": "owner_mailbox", "content": "one", "client_message_id": "42", "chat_id": "100"}]
    other_room = [{"source": "owner_mailbox", "content": "two", "client_message_id": "42", "chat_id": "200"}]
    same_room = [{"source": "owner_mailbox", "content": "updated", "client_message_id": "42", "chat_id": "100"}]
    assert [row["content"] for row in ow.merged_corpus(kept, other_room)] == ["one", "two"]
    assert [row["content"] for row in ow.merged_corpus(kept, same_room)] == ["one"]


def test_two_replies_to_one_message_keep_one_target_row_beside_the_first():
    first = [ow._row("Yes, this one.", source="owner_mailbox", carrier="ctx", ref="msg m-1"),
             ow._row(TARGET["text"], source=ow.REPLY_TARGET, carrier="ctx", ref=TARGET["address"], author="Ouroboros")]
    second = [ow._row("And this too.", source="owner_mailbox", carrier="ctx", ref="msg m-2"),
              ow._row(TARGET["text"], source=ow.REPLY_TARGET, carrier="ctx", ref=TARGET["address"], author="Ouroboros")]
    assert [row["ref"] for row in ow._deduped(first + second)] == ["msg m-1", TARGET["address"], "msg m-2"]


@pytest.mark.parametrize("rooms, kept", [(("100", "200"), ["room one", "room two"]), (("100", "100"), ["room one"]),
                                         (("100", ""), ["room one"])])
def test_one_client_id_in_two_rooms_is_two_messages(rooms, kept):
    """A client id is unique only within its room (a place's message ids repeat across its chats): two rows naming two
    rooms are two messages; the same room, or a carrier that names none (the mailbox), is one message, kept once."""
    from types import SimpleNamespace

    from ouroboros.loop_messages import _initialize_owner_directives
    from ouroboros.owner_continue import owner_corpus_rows

    rows = owner_corpus_rows({"later": [
        {"source": "owner_mailbox", "content": "room one", "client_message_id": "42", "chat_id": rooms[0]},
        {"source": "owner_mailbox", "content": "room two", "client_message_id": "42", "chat_id": rooms[1]}]})
    ctx = SimpleNamespace(task_metadata={"objective_author": {"kind": "continuation"}, "owner_corpus": rows}, task_id="t")
    _initialize_owner_directives(ctx, [{"role": "user", "content": "work order"}])
    assert [row["content"] for row in ctx._owner_directives] == kept


# --- a cite is verified by provenance alone ------------------------------------------------------------------------


def test_cited_owner_basis_accepts_only_my_humans_bytes_that_reached_this_run():
    """Both directions at the verifier's seam: my human's exact words from their confirmed account (the opening event,
    a later arrival, a Main message into the run) verify, emphasis markers aside; a colleague's bytes, a paraphrase,
    the message they replied to and words that never reached this run refuse, typed. The host reads no meaning."""
    ctx = SimpleNamespace(task_id="focus", task_metadata={}, _owner_directives=[
        {"source": ow.PLACE_CONFIRMED, "content": "Проверь экономику **сегодня**", "carried": True, "msg_id": "chat 7 / e1",
         "reply_to": {"address": "platform message m-1 in this room", "author": "", "text": "Deploy the hotfix now"}},
        {"source": "owner_mailbox", "content": "And tell me the cost", "msg_id": "mail-1"},
        {"source": "place_event_context", "content": "<@U2> please update my experiment"},
        {"source": "initial_text", "content": "[FOCUS] frame"}])
    target = {"address": "platform message m-1 in this room", "author": "", "text": "Deploy the hotfix now"}
    place = ("place_owner", "Проверь экономику **сегодня**", "chat 7 / e1", target)
    accepted = {"Проверь экономику сегодня": place, "Проверь экономику **сегодня**": place,
                "And tell me the cost": ("owner_mailbox", "And tell me the cost", "msg mail-1", None)}
    for words, (source, whole, ref, reply_to) in accepted.items():  # the whole message, the span named, the real source
        basis, error = ow.cited_owner_basis(ctx, words)
        assert error == "" and basis == {"source": source, "content": whole, "cited": words, "basis": "cited", "carried": True,
                                         "msg_id": ref, **({"reply_to": reply_to} if reply_to else {})}, words
    for words in ("please update my experiment", "Deploy the hotfix now", "Проверь экономику завтра", "[FOCUS] frame",
                  "Words nobody wrote here"):
        basis, error = ow.cited_owner_basis(ctx, words)
        assert basis is None and error.startswith("owner_words were not found verbatim"), words
    assert ow.cited_owner_basis(ctx, "") == (None, "") and ow.cited_owner_basis(SimpleNamespace(), "x")[0] is None
    assert ow.PLACE_CONFIRMED not in ow.OWNER_SOURCES and "place_owner" in ow.OWNER_SOURCES
