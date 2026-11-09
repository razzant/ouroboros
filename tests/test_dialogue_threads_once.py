"""Provider threads stored once in the dialogue journal (``dialogue_store``).

A Slack or GitHub row carries the whole thread it belongs to in ``transport.conversation``; a thread of a
hundred messages stored its first message a hundred times. New rows keep every dict element of a list
reached through dicts there once, by reference, in the store's ``shared`` table; the row's identity, its
addresses and every reader that consumes the thread see the whole row, byte for byte.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from ouroboros import chat_chain, dialogue_import as di, dialogue_store as ds, store
from tests import _dialogue as dialogue_test
from tests._cases import case_folders


def _whole_digest(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()


# ---- identity without expansion ---------------------------------------------------------------

def test_a_stored_identity_answers_for_its_row_and_a_row_without_one_is_hashed_as_before():
    row = {"ts": "2026-11-04T10:00:00+00:00", "chat_id": 7, "direction": "in", "text": "hello"}
    assert chat_chain.source_row_id(row) == _whole_digest(row)  # an old row: unchanged identity
    stored = "a" * 64
    assert chat_chain.source_row_id({**row, ds.ROW_SHA_KEY: stored}) == stored
    # Only a whole digest is an identity; anything else is content and is hashed with the row.
    odd = {**row, ds.ROW_SHA_KEY: "not-a-digest"}
    assert chat_chain.source_row_id(odd) == _whole_digest(odd)


def test_the_identity_marker_and_the_room_stamp_are_never_a_callers_content():
    row = {"ts": "t", "chat_id": 1, "text": "x", "room_revisions": {"1": 3}, ds.ROW_SHA_KEY: "b" * 64}
    assert ds._unstamped(row) == {"ts": "t", "chat_id": 1, "text": "x"}


# ---- real-shaped provider rows ----------------------------------------------------------------

TS = "2026-11-04T10:00:00+00:00"
KEY = "slack:T1:C1:1730700000.000000"
FENCE = "`" * 3


def _slack_message(i):
    """One ``conversations.replies`` message as a connector copies it: ~1.16 KB, the measured size of one on the
    owner's rows (a 104-message Slack row was 121 KB, d1_minimal appendix A3)."""
    return {"type": "message", "user": f"U{i % 7}", "text": f"message {i}: ünïcödé \"quoted\" \\ back\nslash / tab\t",
            "ts": f"1730700000.{i:06d}", "thread_ts": "1730700000.000000",
            "client_msg_id": f"{i:08d}-aaaa-bbbb-cccc-dddddddddddd", "team": "T1", "reply_count": 0, "edited": None,
            "score": 1.5e-7 * i, "is_locked": False,
            "blocks": [{"type": "rich_text", "block_id": f"b{i}", "elements": [{"type": "rich_text_section", "elements": [
                {"type": "text", "text": f"message {i} " + "words " * 77}, {"type": "user", "user_id": "U1"}]}]}],
            "reactions": [{"name": "eyes", "count": 2, "users": ["U1", "U2"]}],
            "user_profile": {"real_name": f"Person {i % 7}", "display_name": f"p{i % 7}",
                             "image_72": "https://a.example/x.png", "team": "T1", "is_restricted": False}}


def slack(n):
    return {"title": "#general", "thread_history": {
        "status": "complete", "source": "conversations.replies", "observed_at": TS, "scope": "current_accessible_thread",
        "data": {"channel_id": "C1", "thread_ts": "1730700000.000000", "limit": 200,
                 "messages": [_slack_message(i) for i in range(n)], "complete": True, "next_cursor": "", "pages_read": 1},
        "completed_at": TS}}


def github(n):
    rows = [{"kind": "issue" if not i else "comment", "id": 1000 + i, "user": {"login": f"dev{i % 3}", "type": "User"},
             "body": f"comment {i} on the change\n{FENCE}py\nprint({i})\n{FENCE}", "created_at": TS,
             "labels": [{"name": "bug", "color": "d73a4a"}], "reactions": {"+1": i, "total_count": i}} for i in range(n)]
    return {"title": "razzant/ouroboros#2654", "url": "https://github.com/razzant/ouroboros/pull/2654",
            "history": {"complete": True, "observed_at": TS, "rows": rows}}


def transport_queue(n, cmid="slack:T1:C1:1730700000.000099"):
    return {"title": "#general", "transport_queue": {
        "schema_version": 1, "conversation_key": KEY, "after_source_event_id": cmid, "observed_at": TS,
        "events": [{"source_event_id": f"queued-{i}", "text": f"queued words {i}", "actor": {"platform_actor_id": f"U{i}"}}
                   for i in range(n)], "complete": True, "pending_count": n, "omitted_count": 0, "source": "slack-bridge"}}


SHAPES = {"slack": slack, "github": github, "transport_queue": transport_queue}


def presence_row(conversation, *, cmid="slack:T1:C1:1730700000.000099", ts=TS, chat_id=912345, text="the newest message",
                 task_id="turn-1", **extra):
    """An inbound place arrival in ``presence_runner._log_dialogue``'s shape."""
    return {"ts": ts, "session_id": "s-1", "direction": "in", "type": "place_arrival", "chat_id": chat_id, "user_id": 77,
            "text": text, "format": "", "source": "presence:slack", "sender_label": "Ann", "sender_session_id": "U1",
            "client_message_id": cmid, "presence_event_identity": "e" * 64,
            "transport": {"provider": "slack", "account_id": "T1", "conversation_id": "C1", "thread_id": "1730700000.000000",
                          "conversation_key": KEY, "actor": {"platform_actor_id": "U1", "display_name": "Ann", "is_bot": False},
                          "conversation": conversation, "message": {"message_id": cmid, "sent_at": ts},
                          "identity": {"sends_as": {"kind": "bot", "id": "B1"}}},
            "presence_provenance": {"place_id": "place-1", "binding_id": "binding-1", "source_event_id": cmid,
                                    "conversation_key": KEY, "provider": "slack", "account_id": "T1"},
            "task_id": task_id, "place_id": "place-1", "source_order": {"time": ts, "order": None}, **extra}


def _elements(value):
    """The dict elements at the rule's positions (lists reached through dicts), in order."""
    for item in value.values():
        if isinstance(item, dict):
            yield from _elements(item)
        elif isinstance(item, list):
            yield from (entry for entry in item if isinstance(entry, dict))


def _stored(root, seq):
    with ds.transaction(root, write=False) as conn:
        return conn.execute("SELECT body FROM dialogue WHERE stream='chat' AND seq=?", (seq,)).fetchone()["body"]


def _shared_count(root):
    with ds.transaction(root, write=False) as conn:
        held = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='shared'").fetchone()
        return None if held is None else conn.execute("SELECT count(*) FROM shared").fetchone()[0]


# ---- the writer: each element once, the row whole on every read --------------------------------

def test_a_thread_row_holds_each_element_once_and_reads_back_as_its_exact_line(subtests, tmp_path):
    case_paths = case_folders(tmp_path)
    for shape in sorted(SHAPES):
        tmp_path = next(case_paths)
        with subtests.test(shape=shape):
            row = presence_row(SHAPES[shape](12))
            receipt = ds.append(tmp_path, "chat", row)
            line = ds.encode(row)
            assert receipt.row == row and not receipt.duplicate and receipt.length == len(line.encode("utf-8")) + 1
            body = _stored(tmp_path, receipt.seq)
            stored = ds.decode(body)
            with ds.transaction(tmp_path, write=False) as conn:
                assert ds.whole_line(conn, body) == line  # byte for byte: the line the chain held
                assert ds.whole_body(conn, body) == row
            elements = list(_elements(row["transport"]["conversation"]))
            assert len(body) < len(line) and _shared_count(tmp_path) == len({ds.encode(e) for e in elements})
            references = ds.thread_references(stored["transport"]["conversation"])
            assert references == [hashlib.sha256(ds.encode(e).encode("utf-8")).hexdigest() for e in elements]
            # Nothing outside the thread's elements changes, and the identity is the whole row's, last.
            assert list(stored)[-1] == ds.ROW_SHA_KEY
            assert stored[ds.ROW_SHA_KEY] == chat_chain.source_row_id(row) == chat_chain.source_row_id(stored)
            assert {k: v for k, v in stored.items() if k not in ("transport", ds.ROW_SHA_KEY)} == {
                k: v for k, v in row.items() if k != "transport"}
            assert {k: v for k, v in stored["transport"].items() if k != "conversation"} == {
                k: v for k, v in row["transport"].items() if k != "conversation"}
            assert stored["transport"]["conversation"]["title"] == row["transport"]["conversation"]["title"]
            # Every address of the row is its whole row's, read from the small stored form.
            address = chat_chain.format_address(chat_chain.row_address(row))
            found, resolution = chat_chain.resolve_row(tmp_path, address)
            assert resolution["status"] == "ok" and chat_chain.source_row_id(found) == chat_chain.source_row_id(row)
            assert dialogue_test.chat_rows(tmp_path) == [row] and dialogue_test.lines(tmp_path) == [line]


def test_a_hundred_message_thread_row_shrinks_by_an_order_of_magnitude_and_each_message_is_held_once(tmp_path):
    whole = stored = 0
    for n in range(1, 101):  # each new message's row carries the thread so far, as a connector copies it
        receipt = ds.append(tmp_path, "chat", presence_row(slack(n), cmid=f"slack:T1:C1:1730700000.{n:06d}",
                                                           ts=f"2026-11-04T10:{n // 60:02d}:{n % 60:02d}+00:00"))
        whole, stored = receipt.length - 1, len(_stored(tmp_path, receipt.seq).encode("utf-8"))
    assert stored * 10 <= whole, (stored, whole)  # the 100-message row: an order of magnitude smaller
    assert _shared_count(tmp_path) == 100  # linear in what was observed: every message once, not 5,050 times


def test_a_row_without_a_thread_and_a_row_written_before_stay_whole_and_read_as_today(tmp_path):
    plain = {"ts": TS, "direction": "in", "chat_id": 5, "text": "telegram words", "client_message_id": "tg-1",
             "transport": {"provider": "telegram", "conversation": {"title": "DM", "id": 5}}}
    receipt = ds.append(tmp_path, "chat", plain)
    assert _stored(tmp_path, receipt.seq) == ds.encode(plain) and _shared_count(tmp_path) is None  # no table until used
    # A row an earlier body wrote (or the importer brought in) keeps its whole body; every reader takes it as is.
    old = presence_row(slack(3), cmid="old-1", ts="2026-11-04T09:00:00+00:00")
    body = ds.encode(old)
    with ds.transaction(tmp_path, write=True) as conn:
        marks = conn.execute("SELECT next_seq, live, next_ordinal, next_pos FROM streams WHERE stream='chat'").fetchone()
        ds.insert_row(conn, "chat", int(marks["next_seq"]), old, body, length=len(body.encode("utf-8")) + 1,
                      segment=int(marks["live"]), ordinal=int(marks["next_ordinal"]), pos=int(marks["next_pos"]))
        conn.execute("UPDATE streams SET next_seq=next_seq+?, next_ordinal=next_ordinal+1, next_pos=next_pos+1 "
                     "WHERE stream='chat'", (len(body.encode("utf-8")) + 1,))
        assert ds.whole_line(conn, body) == body and ds.whole_body(conn, body) == old
    entry = ds.row_at(tmp_path, "chat", int(marks["next_seq"]))
    assert entry.row == old and ds.whole_rows(tmp_path, [entry]) == [entry]
    assert chat_chain.source_row_id(entry.row) == _whole_digest(old)


def test_content_that_already_reads_as_a_reference_keeps_its_row_whole(tmp_path):
    conversation = {**slack(2), "meta": {"$shared": "c" * 64}}  # in the thread, outside its positions
    row = presence_row(conversation)
    receipt = ds.append(tmp_path, "chat", row)
    assert _stored(tmp_path, receipt.seq) == ds.encode(row) and dialogue_test.chat_rows(tmp_path) == [row]
    elsewhere = presence_row(slack(2), cmid="tok-1", ts="2026-11-04T09:59:59+00:00", meta={"$shared": "e" * 64})
    receipt = ds.append(tmp_path, "chat", elsewhere)  # beside the thread, in the row's own fields
    assert _stored(tmp_path, receipt.seq) == ds.encode(elsewhere) and dialogue_test.chat_rows(tmp_path)[-1] == elsewhere
    # At a position it is simply an element: held once like any other, given back exactly.
    tokenish = presence_row({"history": {"rows": [{"$shared": "d" * 64}, {"id": 1}]}}, cmid="tok-2",
                            ts="2026-11-04T10:00:01+00:00")
    receipt = ds.append(tmp_path, "chat", tokenish)
    assert ds.ROW_SHA_KEY in ds.decode(_stored(tmp_path, receipt.seq))
    assert dialogue_test.chat_rows(tmp_path)[-1] == tokenish


def test_expansion_by_value_is_the_decoded_byte_splice_for_awkward_json(tmp_path):
    awkward = [{"s": "\u0000\u001f   \U0001F600 \\\" '", "f": [0.1, -0.0, 1e308, 5e-324, 123456789012345678901234567890]},
               {"nested": [[{"inner": "a list in a list stays inside"}], {"d": {"deep": [1, {"x": None}]}}]},
               {}, {"emoji\U0001F642": True, "": ""}]
    row = presence_row({"history": {"rows": awkward, "extra": [1, "two", None, [3]]}, "n": {"m": {"l": awkward[:2]}}})
    receipt = ds.append(tmp_path, "chat", row)
    body = _stored(tmp_path, receipt.seq)
    with ds.transaction(tmp_path, write=False) as conn:
        line = ds.whole_line(conn, body)
        assert line == ds.encode(row) and ds.decode(line) == ds.whole_body(conn, body) == row
    stored = ds.decode(body)
    assert stored["transport"]["conversation"]["history"]["extra"] == [1, "two", None, [3]]  # no dict element: as written


def test_a_reference_the_table_does_not_answer_is_an_unreadable_row_never_a_shorter_thread(tmp_path):
    receipt = ds.append(tmp_path, "chat", presence_row(slack(4)))
    body = _stored(tmp_path, receipt.seq)
    with ds.transaction(tmp_path, write=True) as conn:
        conn.execute("DELETE FROM shared WHERE sha256=(SELECT min(sha256) FROM shared)")
    with ds.transaction(tmp_path, write=False) as conn:
        with pytest.raises(store.StoreLost, match="thread element"):
            ds.whole_line(conn, body)
        with pytest.raises(store.StoreLost, match="thread element"):
            ds.whole_body(conn, body)
    with pytest.raises(store.StoreLost, match="thread element"):
        ds.whole_rows(tmp_path, [ds.row_at(tmp_path, "chat", receipt.seq)])


def test_the_same_event_appended_again_is_a_duplicate_with_and_without_a_room_stamp(tmp_path):
    unnamed = presence_row(slack(5), cmid="")  # no client id: the content comparison alone decides
    first = ds.append(tmp_path, "chat", unnamed, idempotent=True)
    again = ds.append(tmp_path, "chat", unnamed, idempotent=True)
    assert again.duplicate and again.seq == first.seq and again.row == unnamed
    stamped = presence_row(slack(6), cmid="", chat_id=4242, ts="2026-11-04T10:00:02+00:00")
    first = ds.append(tmp_path, "chat", stamped, idempotent=True, feed=True, room_revisions=True)
    assert first.row["room_revisions"] == {"4242": 1}
    again = ds.append(tmp_path, "chat", stamped, idempotent=True, feed=True, room_revisions=True)
    assert again.duplicate and again.seq == first.seq and again.row == first.row  # whole, stamp and all
    assert len(dialogue_test.chat_rows(tmp_path)) == 2
    # A named inbound duplicate returns the stored row whole, as its fresh receipt did.
    named = presence_row(slack(7), cmid="named-1", ts="2026-11-04T10:00:03+00:00")
    fresh = ds.append(tmp_path, "chat", named)
    repeat = ds.append(tmp_path, "chat", presence_row(slack(8), cmid="named-1", ts="2026-11-04T10:00:03+00:00"))
    assert repeat.duplicate and repeat.seq == fresh.seq and repeat.row == named


def test_a_text_revision_keeps_the_thread_by_reference_and_names_the_revised_row(tmp_path):
    row = presence_row(slack(9))
    receipt = ds.append(tmp_path, "chat", row)
    before = ds.decode(_stored(tmp_path, receipt.seq))
    revised = ds.revise_text(tmp_path, "chat", receipt.seq, "revised words")
    assert revised == {**row, "text": "revised words"}
    after = ds.decode(_stored(tmp_path, receipt.seq))
    assert (ds.thread_references(after["transport"]["conversation"])
            == ds.thread_references(before["transport"]["conversation"]))
    assert after[ds.ROW_SHA_KEY] == _whole_digest(revised) != before[ds.ROW_SHA_KEY] and list(after)[-1] == ds.ROW_SHA_KEY
    with ds.transaction(tmp_path, write=False) as conn:
        assert ds.whole_line(conn, _stored(tmp_path, receipt.seq)) == ds.encode(revised)
    found, resolution = chat_chain.resolve_row(tmp_path, chat_chain.format_address(chat_chain.row_address(revised)))
    assert resolution["status"] == "ok" and resolution["seq"] == receipt.seq


def test_the_sql_paths_readers_extract_answer_the_same_on_the_stored_form(tmp_path):
    from types import SimpleNamespace

    from ouroboros import place_arrivals

    row = presence_row(slack(10), historical_import={"authorship": "participant", "version": 1})
    receipt = ds.append(tmp_path, "chat", row)
    body = _stored(tmp_path, receipt.seq)
    paths = ("$.chat_id", "$.place_id", "$.ts", "$.type", "$.transport.identity", "$.transport.account_id",
             "$.sender_session_id", "$.sender_label", "$.transport.actor.is_bot", "$.transport.actor.kind",
             "$.historical_import.authorship", "$.historical_import", "$.source_order.time", "$.source_order.order",
             "$.transport.conversation.title", "$.transport.conversation.name")
    probe = sqlite3.connect(":memory:")
    for path in paths:
        expression = "SELECT json_type(?, ?), json_extract(?, ?)"
        assert probe.execute(expression, (body, path, body, path)).fetchone() == probe.execute(
            expression, (ds.encode(row), path, ds.encode(row), path)).fetchone(), path
    place = SimpleNamespace(place_id="place-1", legacy_binding_ids=())
    rooms = place_arrivals.room_index(tmp_path, place, details=True)
    assert [(r["room_chat_id"], r["label"], r["conversation_id"], r["thread_id"]) for r in rooms] == [
        (912345, "#general", "C1", "1730700000.000000")]
    assert place_arrivals.account_candidates(tmp_path, "place-1") == [{"platform_actor_id": "U1", "label": "Ann"}]


def test_the_rollback_export_writes_the_exact_lines_and_a_re_cutover_takes_them_back(tmp_path):
    from ouroboros.utils import iter_jsonl_objects

    rows = [presence_row(SHAPES[shape](6), cmid=f"export-{shape}", ts=f"2026-11-04T10:00:0{i}+00:00")
            for i, shape in enumerate(sorted(SHAPES))]
    rows.append({"ts": TS, "direction": "out", "chat_id": 912345, "text": "a reply"})
    for row in rows:
        ds.append(tmp_path, "chat", row)
    report = di.export_for_older_body(tmp_path)[str(tmp_path)]["chat"]
    live = tmp_path / "logs" / "chat.jsonl"
    assert live.read_bytes() == b"".join(ds.encode(row).encode("utf-8") + b"\n" for row in rows)
    assert report["rows"] == len(rows) and report["bytes"] == len(live.read_bytes())
    assert list(iter_jsonl_objects(live)) == rows, "an older body reads every thread whole"
    # The older body appends; the newer one boots again: the exported rows are its own, none imported twice,
    # and the file continues the store at its own place (``_continues_store`` compares the whole rows), so
    # the older body's row keeps its chain coordinate and the chain and the store end together.
    exported = len(live.read_bytes())
    older = {"ts": TS, "direction": "in", "chat_id": 1, "text": "older body", "client_message_id": "older-1"}
    dialogue_test.seed(tmp_path, live={"chat": [older]})
    di.cutover(tmp_path)
    assert dialogue_test.chat_rows(tmp_path) == [*rows, older]
    assert ds.accepted_row(tmp_path, 1, "older-1").seq == exported
    chain = sum(path.stat().st_size for path in (tmp_path / "archive").glob("chat_*.jsonl"))
    assert ds.stream_end(tmp_path, "chat") == chain == exported + len(ds.encode(older).encode("utf-8")) + 1
    assert di.export_for_older_body(tmp_path)[str(tmp_path)]["chat"]["rows"] == 0, "the files hold everything again"


# ---- every reader of the thread expands it or never hands it on --------------------------------

def test_a_captured_arrival_is_rebuilt_with_its_whole_thread(tmp_path):
    from ouroboros import place_intake

    row = presence_row(transport_queue(3))
    ds.append(tmp_path, "chat", row)
    task, event = place_intake._captured_arrival(tmp_path, {"focus": {"place_id": "place-1"}}, "turn-1", "e" * 64, None)
    assert event.conversation == row["transport"]["conversation"] and event.text == row["text"]
    assert task["id"] == "turn-1" and event.conversation_key == KEY


_RUNTIME = ("ouroboros", "supervisor")
_EXPANDERS = ("whole_body(", "whole_rows(", "whole_row(", "whole_line(", "_whole_row(")
# Every function that reads a ``conversation`` key, and what it reads it from. A stored dialogue row's thread is
# consumed only where it is expanded; the others never see a stored row.
_CONVERSATION_READERS = {
    "ouroboros/dialogue_store.py::_thread": "the store's own reference walk",
    "ouroboros/place_intake.py::_captured_arrival": "expands: the rebuilt event reaches the model",
    "ouroboros/place_arrivals.py::_room_details": "stored row, scalar title/name only",
    "ouroboros/presence_events.py::parse_presence_event": "the connector's request payload, every live door",
    "ouroboros/presence_history.py::_event": "the history import's request payload",
    "ouroboros/presence_call.py::_event_from_task": "the task document's presence event",
    "ouroboros/presence_carrier.py::_saved_event": "the task document's presence event",
    "ouroboros/presence_observations.py::record_transport_queue.update": "the task document's presence event",
    "ouroboros/presence_owner_contact.py::contact_origin": "the task document's presence event",
    "ouroboros/presence_context.py::_author_directory": "the task document's presence event: author names",
    "ouroboros/presence_context.py::_thread_snapshot": "the task document's presence event: the connector's capture, whole",
    "ouroboros/tool_effects.py::_routed": "a tool effect's declared conversation (a string)",
    "ouroboros/safety_evidence.py::lean_event": "the task document's presence event for the Supervisor: its capture as lines",
    "ouroboros/dialogue_provenance.py::RoomLabelResolver.label": "whether a row's facts carry their conversation's scalars",
    "ouroboros/dialogue_provenance.py::RoomLabelResolver._place_label": "tail facts or a row: the scalar name/audience only",
    "ouroboros/memory_view_conversations.py::_scalars": "tail facts or a row: the conversation's scalars only, for a label",
    "ouroboros/tools/chronicle.py::publish_covered_page": "a page's own covers.conversation flag, never a row's conversation",
}
# Every function that decodes a stored dialogue body itself (``Row`` readers go through ``project_source``).
_BODY_DECODERS = {
    "ouroboros/dialogue_store.py::whole_body": "expands",
    "ouroboros/dialogue_store.py::revise_text": "expands: refreshes the identity, returns the row as written",
    "ouroboros/place_arrivals.py::project_source": "every Row reader: identity and words; whole_rows hands rows on",
    "ouroboros/place_arrivals.py::heard_replay": "stored row, identity and import marker only",
    "ouroboros/place_arrivals.py::record_heard": "stored row, identity only",
    "ouroboros/place_arrivals.py::_room_details": "stored row, scalar title/name only",
    "ouroboros/place_arrivals.py::retain_focus_source": "stored row, its address only",
}
# Readers that hand a stored row on whole (a receipt, a comparison, a rebuilt source, the exports) expand it.
_WHOLE_ROW_READERS = ("ouroboros/dialogue_store.py::append", "ouroboros/dialogue_store.py::_receipt",
                      "ouroboros/dialogue_import.py::_continues_store", "ouroboros/dialogue_import.py::_export_stream",
                      "ouroboros/dialogue_import.py::_expand_stream",
                      "ouroboros/chat_chain.py::recorded_context", "ouroboros/presence_history.py::_record_one",
                      "ouroboros/place_intake.py::_captured_arrival")


def _functions():
    """``{"<path>::<qualified name>": (source, node)}`` of every runtime function, nested ones by their path."""
    import ast
    import pathlib
    from tests._source_scan import parse

    root = pathlib.Path(__file__).resolve().parents[1]
    found = {}
    for path in sorted(p for package in _RUNTIME for p in (root / package).rglob("*.py")):
        text = path.read_text(encoding="utf-8")

        def visit(node, scope):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    name = ".".join(scope + [child.name])
                    if not isinstance(child, ast.ClassDef):
                        found[f"{path.relative_to(root).as_posix()}::{name}"] = (ast.get_source_segment(text, child), child)
                    visit(child, scope + [child.name])
        visit(parse(text), [])
    return found


def _own_nodes(node):
    """The nodes of one function without the bodies of the functions nested in it."""
    import ast

    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)) and child is not node:
            if isinstance(child, ast.Lambda):
                yield from ast.walk(child)
            continue
        yield child
        yield from _own_nodes(child)


def test_every_reader_of_a_thread_expands_it_or_never_hands_it_on():
    import ast

    functions = _functions()

    def reads_conversation(node):
        return any((isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and n.slice.value == "conversation")
                   or (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and n.args
                       and isinstance(n.args[0], ast.Constant) and n.args[0].value == "conversation")
                   for n in _own_nodes(node))

    def decodes_a_body(name, node):
        def decoder(call):
            func = call.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                return func.attr == "decode" and func.value.id in ("dialogue_store", "ds")
            return isinstance(func, ast.Name) and func.id == "decode" and name.startswith("ouroboros/dialogue_store.py::")
        return any(isinstance(n, ast.Call) and decoder(n) for n in _own_nodes(node))

    assert sorted(name for name, (_text, node) in functions.items() if reads_conversation(node)) == sorted(_CONVERSATION_READERS)
    assert sorted(name for name, (_text, node) in functions.items() if decodes_a_body(name, node)) == sorted(_BODY_DECODERS)
    for name, how in {**_CONVERSATION_READERS, **_BODY_DECODERS}.items():
        if how.startswith("expands"):
            assert any(call in functions[name][0] for call in _EXPANDERS), name
    for name in _WHOLE_ROW_READERS:
        assert any(call in functions[name][0] for call in _EXPANDERS), name


def test_rows_read_as_written_keep_a_readers_projection_and_their_identity(tmp_path):
    from ouroboros.place_arrivals import SourceProjection

    row = presence_row(slack(4))
    receipt = ds.append(tmp_path, "chat", row)
    ref = {"kind": "task_source", "path": "context_checkpoints/presence-event-x.json", "task_id": "turn-1"}
    with ds.transaction(tmp_path, write=True) as conn:
        conn.execute("UPDATE dialogue SET dispatch_input=? WHERE stream='chat' AND seq=?",
                     (json.dumps({"place_context_source_ref": ref}), receipt.seq))
    stored = ds.row_at(tmp_path, "chat", receipt.seq)
    assert isinstance(stored.row, SourceProjection) and ds.thread_references(stored.row["transport"]["conversation"])
    whole, = ds.whole_rows(tmp_path, [stored])
    assert isinstance(whole.row, SourceProjection) and whole.row.source_body == row
    assert whole.row == {**row, "context_source_ref": ref}
    assert chat_chain.source_row_id(whole.row) == chat_chain.source_row_id(stored.row) == chat_chain.source_row_id(row)


# ---- an unexpanded thread never reaches the model ----------------------------------------------

def _event(conversation):
    from ouroboros.presence_events import PresenceTurnEvent

    return PresenceTurnEvent(source_event_id="slack:T1:C1:1", provider="slack", account_id="T1", conversation_id="C1",
                             thread_id="1730700000.000000", conversation_key=KEY, actor={"platform_actor_id": "U1"},
                             conversation=conversation, message={}, text="words")


def test_an_event_still_naming_a_stored_reference_is_refused_typed_and_a_whole_one_is_not():
    from ouroboros.presence_events import PresenceTurnError

    whole = slack(3)
    assert _event(whole).conversation == whole
    assert _event({"meta": {"$shared": "f" * 64}}).conversation  # a value of that shape, not where a thread keeps one
    with pytest.raises(PresenceTurnError) as refused:
        _event({"title": "#general", "history": {"rows": [{"$shared": "a" * 64}, {"id": 2}]}})
    assert (refused.value.code, refused.value.field) == ("presence_event_unexpanded", "event.conversation")


def test_a_rebuilt_arrival_whose_reader_skipped_the_expansion_is_refused_before_the_model(tmp_path, monkeypatch):
    from ouroboros import place_intake
    from ouroboros.presence_events import PresenceTurnError

    ds.append(tmp_path, "chat", presence_row(slack(3)))
    monkeypatch.setattr(ds, "whole_body", lambda conn, body: ds.decode(body))  # a reader that forgot to expand
    with pytest.raises(PresenceTurnError, match="presence_event_unexpanded"):
        place_intake._captured_arrival(tmp_path, {"focus": {"place_id": "place-1"}}, "turn-1", "e" * 64, None)


# ---- the model opens a thread by its row's address ----------------------------------------------

def test_a_rows_recorded_context_is_read_by_its_address_exactly_as_its_copy_held_it(tmp_path):
    from ouroboros.dialogue_provenance import dialogue_text
    from ouroboros.place_arrivals import SourceProjection

    row = {**presence_row(slack(30)), "context_source_ref": dict(chat_chain.ROW_CONTEXT)}
    del row["type"]  # a non-focus Presence event's row names itself as its context
    receipt = ds.append(tmp_path, "chat", row)
    entry = ds.row_at(tmp_path, "chat", receipt.seq)
    address = chat_chain.context_address(row)
    assert isinstance(entry.row, SourceProjection) and entry.row["context_source_ref"] == address
    assert address == {"kind": "chat_row", "chat_id": 912345, "ts": TS, "row_sha256": chat_chain.source_row_id(row)}
    pointer = dialogue_text(entry.row).split("memory_read(source_ref=", 1)[1]
    assert json.JSONDecoder().raw_decode(pointer)[0] == address  # what every row reader prints
    copy = json.dumps({"recorded_at": row["ts"], "source_event_id": row["presence_provenance"]["source_event_id"],
                       "text": row["text"], "transport": row["transport"]},
                      ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    stored, document = chat_chain.recorded_context(tmp_path, address)
    assert document == copy and stored["row_sha256"] == address["row_sha256"]  # its thread whole, byte for byte
    assert chat_chain.recorded_context(tmp_path, chat_chain.format_address(address))[1] == copy
    from ouroboros.tools.chronicle import _source_texts

    assert any(text.startswith("message 7: ") for text in _source_texts(tmp_path, address))  # a mark's quote finds it
    with pytest.raises(chat_chain.RowAddressError):
        chat_chain.recorded_context(tmp_path, {**address, "row_sha256": "0" * 64})


def test_the_public_log_and_task_event_readers_hand_clients_the_whole_thread(tmp_path):
    """``/api/logs/chat`` and the task-event streams give API clients whole dialogue rows (docs/CLIENT_CONTRACT.md):
    a thread stored once reads back whole there too, never as references with a trailing identity marker."""
    from ouroboros.gateway import logs
    from ouroboros.gateway.task_events import _TaskEventFollower

    row = presence_row(slack(12))
    receipt = ds.append(tmp_path, "chat", row)
    assert ds.thread_references(ds.decode(_stored(tmp_path, receipt.seq))["transport"]["conversation"])  # by reference

    assert [served for _line, served in logs._dialogue_tail(tmp_path, "chat", 10, None, lambda _row: True)] == [row]
    delta = _TaskEventFollower(tmp_path, row["task_id"])._read_dialogue_delta(tmp_path, "chat")
    assert [served for served, _line in delta] == [row]
