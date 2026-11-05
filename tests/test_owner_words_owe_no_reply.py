"""Carried words owe no reply: a quote another task relayed, a drafted objective's corpus and an addressed
place turn's event keep my human's exact words, yet only my human's own new delivery to this run asks it to
answer before finishing silently. One message reaching a run by mail and by relay is one row.
"""
from __future__ import annotations

from types import SimpleNamespace

from ouroboros.loop_messages import (_initialize_owner_directives, _record_owner_directive, owner_input_revision,
                                     owner_reply_owed, owner_source_sha256)
from ouroboros.owner_words import FIELD, governing_words_for_schedule

OWNER_ROW = {"ts": "2026-11-04T10:00:00+00:00", "direction": "in", "chat_id": 1, "text": "Ship it after the review.",
             "client_message_id": "owner-55"}


def _ctx(**metadata):
    return SimpleNamespace(task_id="t-1", task_metadata=metadata, drive_root=None, current_task_type="task")


def _relay(ctx):
    from ouroboros.owner_relay import inject_relay

    entry = {"kind": "relay_entry", "msg_id": "relay:r1:t-1", "text": "Context from the sender.",
             "relay": {"relay_id": "r1"}, "owner_rows": [dict(OWNER_ROW)], "source_task_id": "sender-9"}
    inject_relay(ctx, entry, [])


def test_a_relayed_quote_keeps_the_words_and_owes_no_reply():
    ctx = _ctx(initiator="consciousness")
    _initialize_owner_directives(ctx, [{"role": "user", "content": "[Wake-up]"}])
    before = owner_source_sha256(ctx)
    _relay(ctx)
    [_frame, quote] = ctx._owner_directives
    assert quote["content"] == OWNER_ROW["text"] and quote["carried"] is True and quote["source_task_id"] == "sender-9"
    assert owner_source_sha256(ctx) != before, "the words are in the corpus every reviewer reads"
    assert owner_input_revision(ctx) == 0 and not owner_reply_owed(ctx), "a quote asks this run for no reply"
    assert [row["text"] for row in governing_words_for_schedule(ctx)[FIELD]] == [OWNER_ROW["text"]]


def test_my_humans_own_mail_still_owes_a_reply():
    ctx = _ctx(initiator="consciousness")
    _initialize_owner_directives(ctx, [{"role": "user", "content": "[Wake-up]"}])
    _record_owner_directive(ctx, source="owner_mailbox", content="Answer me first.", msg_id="mail-2")
    assert owner_input_revision(ctx) == 1 and owner_reply_owed(ctx)


def test_one_message_by_mail_and_by_relay_is_one_row_that_owes_its_reply():
    mailed = _ctx(initiator="consciousness")
    _initialize_owner_directives(mailed, [{"role": "user", "content": "[Wake-up]"}])
    _record_owner_directive(mailed, source="owner_mailbox", content=OWNER_ROW["text"], msg_id="mail-55",
                            origin={"client_message_id": "owner-55"})
    _relay(mailed)
    assert [row.get("carried") for row in mailed._owner_directives[1:]] == [None]
    assert owner_input_revision(mailed) == 1
    relayed_first = _ctx(initiator="consciousness")
    _initialize_owner_directives(relayed_first, [{"role": "user", "content": "[Wake-up]"}])
    _relay(relayed_first)
    assert owner_input_revision(relayed_first) == 0
    _record_owner_directive(relayed_first, source="owner_mailbox", content=OWNER_ROW["text"], msg_id="mail-55",
                            origin={"client_message_id": "owner-55"})
    rows = relayed_first._owner_directives[1:]
    assert len(rows) == 1 and "carried" not in rows[0], "the direct delivery of the same words owes the reply"
    assert owner_input_revision(relayed_first) == 1


def test_a_drafted_objective_and_a_place_event_carry_words_without_a_debt():
    drafted = _ctx(objective_author={"kind": "task", "task_id": "root-1"},
                   owner_corpus=[{"source": "origin_message", "content": "Build the report."}])
    _initialize_owner_directives(drafted, [{"role": "user", "content": "Drafted objective."}])
    assert [row.get("carried") for row in drafted._owner_directives] == [True]
    assert owner_input_revision(drafted) == 0 and not owner_reply_owed(drafted)
    place = _ctx(focus={"room_chat_id": 7, "addressed": True, "actor": {"owner_confirmed": True}},
                 presence={"observed_text": "Look at this when you can.", "event": {"source_event_id": "e1"}})
    _initialize_owner_directives(place, [{"role": "user", "content": "[FOCUS]"}])
    assert place._owner_directives[0]["carried"] is True and owner_input_revision(place) == 0


def test_the_mailbox_and_a_relay_of_one_owner_message_drain_into_one_owed_row(tmp_path):
    import queue

    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.owner_relay import quote_rows, write_relay
    from tests.test_owner_relay import context, source

    row, handle = source(tmp_path)
    quotes, error = quote_rows(context(tmp_path, row), [handle])
    assert not error
    write_relay(tmp_path, "receiver", relay_id="r1", note="note", quotes=quotes, chat_id=1, teller_task_id="teller")
    write_owner_message(tmp_path, row["text"], "receiver", msg_id="steer-1", client_message_id=row["client_message_id"])
    receiver = context(tmp_path)
    receiver.task_id = "receiver"
    _drain_incoming_messages([], queue.Queue(), tmp_path, "receiver", None, set(), owner_ctx=receiver)
    owner = [item for item in receiver._owner_directives if item["source"] == "owner_mailbox"]
    assert len(owner) == 1 and "carried" not in owner[0], owner
    assert owner_input_revision(receiver) == 1
