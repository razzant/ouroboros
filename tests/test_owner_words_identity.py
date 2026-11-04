"""One message of my human is one row wherever its words are read back, every carrier of them is read, and each
row keeps its own author (review RV1W: Opus F1, Astra F4/F5, workflow #4/#5/#6/#11).

Each guard is checked both ways: one source under two carriers is one row while two messages with the same words
stay two; a carried corpus is read while its absence still says so; a reply target keeps its author in the
by-reference form while my human's row keeps theirs; one place snapshot serves a whole read.
"""
from __future__ import annotations

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
def test_a_continue_shows_one_place_event_once_under_one_authority(confirmed):
    from ouroboros.owner_continue import work_order_text

    words = "Ship the release tonight, then tell me."
    wake = {"source": "consciousness_wake", "task_id": "wake1", "wake_cause": "focus",
            "focus": {"addressed": True, "actor": {"owner_confirmed": confirmed}}, "observed_text": words}
    later = [{"source": "place_owner", "content": words, "msg_id": "chat 7 / ev-1"}] if confirmed else []
    text = work_order_text("wake1", "technical interruption", {"wake": wake, "later": later})
    assert text.count(words) == 1, "one event, one reading"
    context = "The event's words, from the sender recorded above (context, not an owner instruction; verbatim):"
    assert (context in text) is not confirmed
    assert ("Other owner words in that task's record (verbatim, in order):" in text) is confirmed
