"""Appearance certification waits for authority and never skips older-body rows.

Regression witnesses from the independent membership review. Only disposable stores are used;
PERSIST contention is real, rather than inferred from a mocked result or exception message.
"""
from __future__ import annotations

import pytest

from ouroboros import dialogue_store as ds, room_appearances as ra
from ouroboros.chat_chain import room_rows
from ouroboros.projects_registry import RoomResolver, create_project
from ouroboros.task_results import write_task_result


def _world(root):
    chat = create_project(root, "nova", name="Nova")["chat_id"]
    write_task_result(root, "nova-root", "running", chat_id=chat, title="Nova root")
    ds.append(root, "chat", {"chat_id": 1, "direction": "in", "ts": "seed", "text": "prime dialogue store"})
    return chat


def _notice():
    return {"ts": "2026-10-27T01:00:00Z", "direction": "out", "chat_id": 1,
            "type": "main_notice", "task_id": "nova-root", "text": "Notice before restored archive"}


def test_real_task_store_contention_preserves_accepted_notice_for_catchup(tmp_path, monkeypatch):
    import sqlite3
    from ouroboros import projects_registry as pr, store
    from supervisor.message_ingress import log_chat

    monkeypatch.setattr(store, "_journal_mode", lambda *_args: "PERSIST")
    chat = _world(tmp_path)
    pr._TASK_LINEAGE.pop((str(tmp_path), "nova-root"), None)
    pr._ROOM_ANSWERS.clear()
    before = ra.mark(tmp_path)
    blocker = sqlite3.connect(store.store_path(tmp_path), isolation_level=None)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        with store.acquisition_budget(0.05):
            stored = log_chat("out", 1, 0, "Notice during real task-store contention", record_type="main_notice",
                              task_id="nova-root", drive_root=tmp_path, require_write=True, feed=True)
    finally:
        blocker.rollback()
        blocker.close()
    assert stored, "accepting speech must not depend on deriving its secondary appearance"
    assert ra.mark(tmp_path) == before and ra.owed(tmp_path)
    assert RoomResolver(tmp_path)("nova-root") == chat
    assert ra.run(tmp_path, lambda: False)["state"] == "done"
    assert stored["text"] in [entry.row.get("text") for entry in room_rows(tmp_path, chat)]
    assert not ra.owed(tmp_path)


def test_unavailable_project_registry_leaves_derivation_pending(tmp_path, monkeypatch):
    from ouroboros import projects_registry as pr
    from supervisor.message_ingress import log_chat

    chat = _world(tmp_path)
    before = ra.mark(tmp_path)
    # The production reader distinguishes an unavailable registry from a known empty one.
    # Its strict mode is what lets appearance certification retain that distinction.
    original = pr.reserved_project_chat_ids

    def unavailable(root, *, strict=False):
        if strict:
            raise RuntimeError("registry temporarily unavailable")
        return set()

    with monkeypatch.context() as blocked:
        blocked.setattr(pr, "reserved_project_chat_ids", unavailable)
        stored = log_chat("out", 1, 0, "Notice during registry unavailability", record_type="main_notice",
                          task_id="nova-root", drive_root=tmp_path, require_write=True, feed=True)
        assert stored and ra.mark(tmp_path) == before and ra.owed(tmp_path)
        with pytest.raises(RuntimeError):
            ra.run(tmp_path, lambda: False)
        assert ra.mark(tmp_path) == before
    assert original(tmp_path) == {chat}
    assert ra.run(tmp_path, lambda: False)["state"] == "done"
    assert stored["text"] in [entry.row.get("text") for entry in room_rows(tmp_path, chat)]


def test_restored_archive_import_keeps_preceding_unindexed_live_rows_owed(tmp_path):
    from tests._dialogue import seed, import_all

    chat = _world(tmp_path)
    before = ra.mark(tmp_path)
    row = _notice()
    # An older writer appends a row without advancing the appearance mark.
    stored = ds.append(tmp_path, "chat", row, feed=True, appearances=None)
    assert ra.mark(tmp_path) == before and ra.owed(tmp_path)
    seed(tmp_path, archives={"chat_20261026T000000.jsonl": [
        {"ts": "2026-10-26T01:00:00Z", "direction": "in", "chat_id": 1, "text": "restored older conversation"}]})
    import_all(tmp_path)
    assert ra.mark(tmp_path) == before and ra.owed(tmp_path)
    assert ra.run(tmp_path, lambda: False)["state"] == "done"
    assert row["text"] in [entry.row.get("text") for entry in room_rows(tmp_path, chat)]
    assert ds.row_at(tmp_path, "chat", stored.seq).row == stored.row
    assert not ra.owed(tmp_path)


def _written_by_the_parent_body(root, row):
    """What the parent body's own append leaves: the keys its older rule gave the row (none for private contact
    or a Reply to it), the mark moved past it, and this body's coverage where it was."""
    covered = ds.meta(root, ds.APPEARANCES_COVERED)
    receipt = ds.append(root, "chat", row, feed=row["direction"] == "out", appearances=[])
    with ds.transaction(root, write=True) as conn:
        ds._set_meta(conn, ds.APPEARANCES_COVERED, covered)
    return receipt


def test_rows_a_parent_body_wrote_after_this_one_are_a_gap_until_exactly_they_are_derived_again(tmp_path):
    """The supported rollback round trip: this body's pass is done, the parent body (it moves the mark, never this
    body's coverage) writes private contact and the owner's Reply to it, and this body returns. Its Project reads
    name the gap at once; the job owes the pass exactly the rows from the coverage to the mark, never the history
    below, and derives them unchanged; this body's next append needs nothing."""
    from ouroboros.chat_references import reference_from_row
    from tests.test_room_appearances import _read, _texts

    chat = _world(tmp_path)
    assert ra.run(tmp_path, lambda: False)["state"] == "done" and not ra.owed(tmp_path)
    covered = ds.meta(tmp_path, ds.APPEARANCES_COVERED)
    assert covered == ra.mark(tmp_path) == ds.stream_end(tmp_path, "chat")
    target = {"ts": "2026-10-28T05:00:00Z", "direction": "out", "chat_id": 1, "task_id": "nova-root",
              "type": "main_notice", "text": "Private contact during rollback",
              "presence_contact": {"task_id": "nova-root", "binding_id": "place-1", "source_event_id": "evt-1"}}
    rows = [_written_by_the_parent_body(tmp_path, target), _written_by_the_parent_body(tmp_path, {
        "ts": "2026-10-28T05:01:00Z", "direction": "in", "chat_id": 1, "client_message_id": "rollback-reply",
        "text": "Owner reply during rollback", "reply_to": {**reference_from_row(target), "target_home": "none"}})]
    texts, end = {receipt.row["text"] for receipt in rows}, ds.stream_end(tmp_path, "chat")
    assert ra.mark(tmp_path) == end and ra.owed(tmp_path)
    bodies = [ds.row_at(tmp_path, "chat", receipt.seq) for receipt in rows]
    revisions = {room["chat_id"]: room["revision"] for room in ds.rooms(tmp_path)}
    assert not texts & ({entry.row.get("text") for entry in room_rows(tmp_path, chat)} | set(_texts(tmp_path, chat)))
    assert "chat_appearances_deriving" in _read(tmp_path, chat)[1]["window"]["truncated_by"]  # absent, and said so
    ra._owe_uncovered(tmp_path)
    assert ds.meta(tmp_path, ds.APPEARANCES_CONTACT_REPLIES) == {"next": covered, "upper": end}  # just those rows
    assert ds.meta(tmp_path, ds.APPEARANCES_COVERED) == end
    assert ra.run(tmp_path, lambda: False)["state"] == "done" and not ra.owed(tmp_path)
    assert texts <= {entry.row.get("text") for entry in room_rows(tmp_path, chat)} and texts <= set(_texts(tmp_path, chat))
    assert "chat_appearances_deriving" not in _read(tmp_path, chat)[1]["window"]["truncated_by"]
    for receipt, before in zip(rows, bodies):  # bytes, address and coordinate unchanged; nothing arrives unread
        after = ds.row_at(tmp_path, "chat", receipt.seq)
        assert (after.row, after.length) == (before.row, before.length)
    assert {room["chat_id"]: room["revision"] for room in ds.rooms(tmp_path)} == revisions
    assert ra.run(tmp_path, lambda: False) == {"state": "done", "items": 0}  # once
    ds.append(tmp_path, "chat", {"ts": "2026-10-28T06:00:00Z", "direction": "in", "chat_id": 1, "text": "fresh"},
              appearances=[])
    assert ds.meta(tmp_path, ds.APPEARANCES_COVERED) == ds.stream_end(tmp_path, "chat") and not ra.owed(tmp_path)


def _planned_archive(root):
    """This body cuts over three archive segments of Main rows, each a private contact notice of Nova's work, and
    imports none of them yet: ``(Project chat, pending segments newest first)``."""
    from ouroboros import dialogue_import
    from tests._dialogue import seed

    contact = {"task_id": "nova-root", "binding_id": "place-1", "label": "Archived chat"}
    seed(root, archives={f"chat_2026100{n}T000000.jsonl": [
        {"ts": f"2026-10-0{n}T01:00:00Z", "direction": "out", "chat_id": 1, "task_id": "nova-root", "type": "main_notice",
         "text": f"contact S{n}", "presence_contact": {**contact, "source_event_id": f"e{n}"}}] for n in (1, 2, 3)},
         live={"chat": [{"ts": "2026-10-04T00:00:00Z", "direction": "in", "chat_id": 1, "text": "live tail"}]})
    dialogue_import.cutover(root)  # the plan, and the live tail imported here
    chat = create_project(root, "nova", name="Nova")["chat_id"]
    write_task_result(root, "nova-root", "running", chat_id=chat, title="Nova root")
    return chat, [row for row in dialogue_import.pending_segments(root) if row["stream"] == "chat"]


def test_an_archive_this_body_imports_itself_needs_no_second_pass(tmp_path):
    """Its plan owes the archive's rows to the pass; each import of its own derives a segment and takes it off the
    top, newest first, so the finished import leaves nothing owed and the job reads nothing again."""
    from ouroboros import dialogue_import

    chat, pending = _planned_archive(tmp_path)
    assert ds.meta(tmp_path, ds.APPEARANCES_CONTACT_REPLIES) == {"next": pending[-1]["base"],
                                                                "upper": pending[0]["base"] + pending[0]["size"]}
    dialogue_import.run_dialogue_import(tmp_path)
    state = ds.meta(tmp_path, ds.APPEARANCES_CONTACT_REPLIES)
    assert state["next"] == state["upper"] and not ra.owed(tmp_path)
    assert ra.run(tmp_path, lambda: False) == {"state": "done", "items": 0}
    assert {"contact S1", "contact S2", "contact S3"} <= {entry.row.get("text") for entry in room_rows(tmp_path, chat)}


def test_an_archive_this_body_planned_but_another_body_imported_is_a_gap_until_exactly_those_rows_are_derived(
        tmp_path, monkeypatch):
    """Rollback while this body's archive import is unfinished: it imported the newest segment, then the parent body
    imported the rest by its own rule. That import leaves what the parent's own transaction does: the keys its rule
    gives (none for private contact), its awaiting entry retired, the ledger done, and the pass untouched. On return
    the Project names the gap; the job derives exactly the rows the pass still owes, changing no row or revision."""
    from ouroboros import dialogue_import
    from tests.test_room_appearances import _read, _texts

    chat, pending = _planned_archive(tmp_path)
    dialogue_import.import_segment(tmp_path, "chat", pending[0]["name"])  # this body: the newest segment, then rollback
    with monkeypatch.context() as parent:
        parent.setattr(dialogue_import, "import_keys", lambda *_args, **_kwargs: [])
        parent.setattr(ra, "imported_here", lambda *_args, **_kwargs: None)
        dialogue_import.run_dialogue_import(tmp_path)
    assert not ds.meta(tmp_path, ds.APPEARANCES_AWAITING)  # the parent retired its entries as it imported
    owed = {"next": pending[-1]["base"], "upper": pending[1]["base"] + pending[1]["size"]}
    assert ds.meta(tmp_path, ds.APPEARANCES_CONTACT_REPLIES) == owed and ra.owed(tmp_path)  # exactly the parent's
    shown = {entry.row.get("text") for entry in room_rows(tmp_path, chat)} | set(_texts(tmp_path, chat))
    assert "contact S3" in shown and not {"contact S1", "contact S2"} & shown
    assert "chat_appearances_deriving" in _read(tmp_path, chat)[1]["window"]["truncated_by"]  # absent, and said so
    with ds.transaction(tmp_path, write=False) as conn:
        bodies = [tuple(record) for record in conn.execute("SELECT seq, body, length FROM dialogue ORDER BY stream, seq")]
    revisions = {room["chat_id"]: room["revision"] for room in ds.rooms(tmp_path)}
    assert ra.run(tmp_path, lambda: False)["state"] == "done" and not ra.owed(tmp_path)
    shown = {entry.row.get("text") for entry in room_rows(tmp_path, chat)} | set(_texts(tmp_path, chat))
    assert {"contact S1", "contact S2", "contact S3"} <= shown
    assert "chat_appearances_deriving" not in _read(tmp_path, chat)[1]["window"]["truncated_by"]
    with ds.transaction(tmp_path, write=False) as conn:
        assert [tuple(record) for record in conn.execute("SELECT seq, body, length FROM dialogue ORDER BY stream, seq")] \
            == bodies
    assert {room["chat_id"]: room["revision"] for room in ds.rooms(tmp_path)} == revisions
    assert ra.run(tmp_path, lambda: False) == {"state": "done", "items": 0}  # once
    ds.append(tmp_path, "chat", {"ts": "2026-10-28T06:00:00Z", "direction": "in", "chat_id": 1, "text": "fresh"},
              appearances=[])
    assert ds.meta(tmp_path, ds.APPEARANCES_COVERED) == ds.stream_end(tmp_path, "chat") and not ra.owed(tmp_path)


def test_an_older_bodys_job_cannot_erase_a_gap_this_rule_names_and_a_readable_target_closes_both(tmp_path, monkeypatch):
    """An owner's Reply stamped "none" by the older rule quotes a private contact whose stored line is torn: this rule
    cannot decide it and keeps it behind both gap prefixes. The parent body's job, whose rule takes the stamp as
    settled, erases the legacy ``gap:`` keys (its ``_rederive_gaps`` transaction for such a row); this rule's reader
    still names the gap, and this rule's job puts the erased keys back. Once the target's row is stored again the
    reader decides the Reply at once, and the job removes both prefixes, reading only the rows behind them: no row,
    revision or read changes."""
    import json

    from ouroboros.chat_references import reference_from_row
    from tests._dialogue import import_all, seed
    from tests.test_room_appearances import _read, _texts

    nova = create_project(tmp_path, "nova", name="Nova")["chat_id"]
    write_task_result(tmp_path, "nova-root", "running", chat_id=nova, title="Nova root")
    target = {"ts": "2026-10-01T05:00:00Z", "direction": "out", "chat_id": 1, "task_id": "nova-root",
              "type": "main_notice", "text": "A private question.",
              "presence_contact": {"task_id": "nova-root", "binding_id": "place-1", "source_event_id": "evt-1"}}
    reply = {"ts": "2026-10-01T06:00:00Z", "direction": "in", "chat_id": 1, "client_message_id": "torn",
             "text": "Owner words torn", "reply_to": {**reference_from_row(target), "target_home": "none"}}
    history = [{"ts": f"2026-10-02T00:{n // 60:02d}:{n % 60:02d}Z", "direction": "in", "chat_id": 1, "text": f"row {n}"}
               for n in range(120)]
    seed(tmp_path, archives={"chat_20261001T000000.jsonl": [json.dumps(target)[:48] + "\n", reply, *history]})
    import_all(tmp_path)
    assert ra.run(tmp_path, lambda: False)["state"] == "done"
    seq = ds.rows_by_identity(tmp_path, 1, reply["ts"])[0].seq
    for room in ds.rooms(tmp_path):  # the owner has read every room
        ds.mark_room_read(tmp_path, room["chat_id"], room["revision"])

    def gap_keys():
        with ds.transaction(tmp_path, write=False) as conn:
            return {record["key"] for record in conn.execute(
                "SELECT key FROM room_appearances WHERE stream='chat' AND seq=? AND key LIKE 'gap%'", (seq,))}

    def named():
        return ra.REPLY_TARGET_UNREADABLE in ra.read_gaps(tmp_path, nova, [f"room:{nova}", "task:nova-root"], 0,
                                                          ds.stream_end(tmp_path, "chat"))

    def stored():
        with ds.transaction(tmp_path, write=False) as conn:
            return ([tuple(record) for record in conn.execute("SELECT seq, body, length FROM dialogue WHERE seq<=?", (seq,))],
                    [(room["chat_id"], room["revision"], room["read_revision"]) for room in ds.rooms(tmp_path)])

    both = {"gap:task:nova-root", "gap2:task:nova-root"}
    assert gap_keys() >= both and named() and "Owner words torn" not in _texts(tmp_path, nova)
    before = stored()
    with ds.transaction(tmp_path, write=True) as conn:  # the parent's job: its rule says "none", so no gap
        conn.execute("DELETE FROM room_appearances WHERE stream='chat' AND seq=? AND key>='gap:' AND key<'gap;'", (seq,))
    assert not gap_keys() & {"gap:task:nova-root"} and named()  # this rule's prefix still names it
    assert "chat_reply_target_unreadable" in _read(tmp_path, nova)[1]["window"]["truncated_by"]
    ra._rederive_gaps(tmp_path, *ra._rule(tmp_path))  # this rule's catch-up: still unknown, both prefixes again
    assert gap_keys() >= both and named()
    ds.append(tmp_path, "chat", dict(target), appearances=None)  # the repair: the target's exact row stored again
    assert not named() and "Owner words torn" in _texts(tmp_path, nova)  # the reader decides it at once
    loaded, rows_at, iter_rows = [], ds.rows_at, ds.iter_rows  # every row the job reads, by either reader

    def counted_rows_at(*args, **kwargs):
        found = rows_at(*args, **kwargs)
        loaded.extend(found)
        return found

    monkeypatch.setattr(ds, "rows_at", counted_rows_at)
    monkeypatch.setattr(ds, "iter_rows", lambda *a, **k: (loaded.append(entry) or entry for entry in iter_rows(*a, **k)))
    assert ra.run(tmp_path, lambda: False)["state"] == "done" and not ra.owed(tmp_path)
    monkeypatch.undo()
    assert len(loaded) <= 3 < len(history)  # the repaired row and the gap row, never the history
    assert not gap_keys() and "task:nova-root" in {key for key in _index_keys_of(tmp_path, seq)}
    assert not named() and "Owner words torn" in _texts(tmp_path, nova)
    assert stored() == before  # every earlier row's bytes, every room's revision and read, as they were


def _index_keys_of(root, seq):
    with ds.transaction(root, write=False) as conn:
        return [record["key"] for record in conn.execute("SELECT key FROM room_appearances WHERE stream='chat' AND seq=?",
                                                         (seq,))]
