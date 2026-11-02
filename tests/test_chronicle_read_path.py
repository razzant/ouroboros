"""Chronicle reads: lock-free while the index is current, the publisher's locked catch-up otherwise."""
import contextlib
import json
import os
import pathlib
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ouroboros import chronicle_store, memory_inventory
from ouroboros.chronicle_store import ChronicleStore, source_time_span

MIND = {"kind": "mind", "task_id": "t1", "focus": {"role": "root", "task_id": "t1"}}


def _note(store, text, room="r"):
    result = store.write_note(room_id=room, task_id="t1", text=text, author=MIND)
    assert result.ok, result
    return result.record


def _counted_locks(monkeypatch):
    taken, real = [], chronicle_store.acquire_exclusive_file_lock
    monkeypatch.setattr(chronicle_store, "acquire_exclusive_file_lock",
                        lambda *args, **kwargs: taken.append(1) or real(*args, **kwargs))
    return taken


def _reads(store, record_id):
    """Every reader, once: what a capture asks the chronicle."""
    return (store.get(record_id)["text"], [r["id"] for r in store.room_records("r")], [r["id"] for r in store.records()],
            store.scan_state(), store.active_marks("r"), store.room_head("r"), store.sealed_row_refs("r"),
            store.pages_of_room("r"), store.observation_snapshot(None)[1]["record_id"])


def test_reads_take_no_lock_and_write_nothing_while_the_index_is_current(tmp_path, monkeypatch):
    store = ChronicleStore(tmp_path)
    kept = _note(store, "kept")
    index = store.index_path.stat()
    taken = _counted_locks(monkeypatch)
    assert _reads(store, kept["id"]) == ("kept", [kept["id"]], [kept["id"]], {}, [], kept["sequence"], set(), [],
                                         kept["id"])
    assert taken == [] and not store.lock_path.exists()
    assert store.index_path.stat().st_mtime_ns == index.st_mtime_ns
    _note(store, "published")  # a publication keeps its lock
    assert taken == [1]


def test_a_read_after_an_unprojected_append_catches_up_under_the_lock(tmp_path, monkeypatch):
    store = ChronicleStore(tmp_path)
    _note(store, "first")
    tx = {"kind": "transaction", "schema": 1, "scan_state": None,
          "records": [{"id": "late", "kind": "note", "room_id": "r", "task_id": "t1", "text": "appended",
                       "author": MIND, "ts": "2026-10-25T00:00:00Z"}]}
    with store.log_path.open("a", encoding="utf-8") as stream:  # a publisher stopped between append and projection
        stream.write(json.dumps(tx) + "\n")
    taken = _counted_locks(monkeypatch)
    assert store.get("late")["text"] == "appended" and taken == [1]
    assert store.get("late")["text"] == "appended" and taken == [1]  # caught up: the next read is lock-free again


def test_without_enforced_kernel_locks_every_read_keeps_the_lock(tmp_path, monkeypatch):
    store = ChronicleStore(tmp_path)
    kept = _note(store, "kept")
    monkeypatch.setattr(chronicle_store, "kernel_file_locks_enforced", lambda *_a, **_k: False)  # the name tier
    taken = _counted_locks(monkeypatch)
    assert store.get(kept["id"])["text"] == "kept" and store.room_records("r") and taken == [1, 1]


@pytest.mark.skipif(os.name == "nt", reason="POSIX advisory locks: any close of a descriptor drops them")
def test_a_raw_read_of_the_index_inside_a_lock_free_read_keeps_its_snapshot_locked(tmp_path):
    """A Files preview of ``index.sqlite3`` (``store.read_file_bytes``) during a lock-free read: the reader's
    connection is counted on the inode like every store connection, so the preview's descriptor stays open until it
    closes and another process cannot take the write lock under the snapshot (``store._borrow``)."""
    from ouroboros import store

    chronicle = ChronicleStore(tmp_path)
    _note(chronicle, "kept")
    probe = ("import sqlite3, sys\ndb = sqlite3.connect(sys.argv[1], timeout=0, isolation_level=None)\n"
             "try:\n    db.execute('BEGIN EXCLUSIVE')\n    print('taken')\nexcept sqlite3.OperationalError:\n"
             "    print('locked')")

    def another_process_writes() -> str:
        return subprocess.run([sys.executable, "-I", "-c", probe, str(chronicle.index_path)], capture_output=True,
                              text=True, timeout=60).stdout.strip()

    with chronicle._index() as db:  # current: the read-only connection's transaction holds SQLite's shared lock
        assert db.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 1
        assert store.read_file_bytes(chronicle.index_path, limit=16) == b"SQLite format 3\x00"
        assert another_process_writes() == "locked"
    assert another_process_writes() == "taken"


def test_a_writer_meeting_a_lock_free_reader_waits_its_own_bound_and_is_told_busy(tmp_path):
    """Lock-free readers hold SQLite's read lock, not the publication lock: the tail facts' zero-wait writer gives up
    at once (its rows wait for the next writer) and a publication past its wait is a typed ``TimeoutError``, never
    SQLite's own five seconds nor an untyped error; once the reader is gone both write."""
    import time

    chronicle = ChronicleStore(tmp_path)
    _note(chronicle, "kept")
    expected = chronicle.tail_state()
    with chronicle._index() as db:
        assert db.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 1
        started = time.monotonic()
        assert chronicle.extend_tail(expected, [], {"complete": False}, wait_sec=0) is False
        with pytest.raises(TimeoutError, match="busy"):
            with chronicle._publishing(wait_sec=0.2) as writer:
                writer.execute("INSERT INTO state VALUES ('probe', '1')")
                writer.commit()
        assert time.monotonic() - started < 3
    assert chronicle.extend_tail(expected, [], {"complete": False}, wait_sec=0) is True


def test_a_busy_index_is_contention_never_a_corrupt_file_to_delete(tmp_path):
    """Another connection holding the index's write lock makes the publisher's schema step busy: a typed
    ``TimeoutError``, and the index file stays; only a file that does not decode is deleted and rebuilt."""
    import sqlite3

    chronicle = ChronicleStore(tmp_path)
    kept = _note(chronicle, "kept")
    inode = chronicle.index_path.stat().st_ino
    holder = sqlite3.connect(chronicle.index_path, isolation_level=None)
    holder.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(TimeoutError, match="busy"):
            with chronicle._publishing(wait_sec=0):
                pass
        assert chronicle.index_path.stat().st_ino == inode
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert chronicle.get(kept["id"])["text"] == "kept"


def test_the_index_is_locked_before_the_authority_append_so_a_durable_write_never_reports_busy(tmp_path, monkeypatch):
    """A lock-free reader that began between the trial and the journal append could hold the index past the
    publisher's wait, and a write already durable in ``records.jsonl`` was reported busy (a retry wrote it twice). The
    publisher takes the index's write lock before it appends: a reader begun then waits for the commit, and a busy
    index refuses before anything is appended."""
    import sqlite3

    chronicle = ChronicleStore(tmp_path)
    _note(chronicle, "first")
    seen, real_append = [], chronicle_store.append_jsonl

    def append_while_a_reader_begins(path, *args, **kwargs):
        reader = sqlite3.connect(f"{chronicle.index_path.absolute().as_uri()}?mode=ro", uri=True, timeout=0)
        try:
            reader.execute("BEGIN")
            reader.execute("SELECT COUNT(*) FROM records").fetchone()
            seen.append("read")
        except sqlite3.OperationalError as exc:
            seen.append("locked" if "locked" in str(exc) else str(exc))
        finally:
            reader.close()
        return real_append(path, *args, **kwargs)

    monkeypatch.setattr(chronicle_store, "append_jsonl", append_while_a_reader_begins)
    result = chronicle.write_note(room_id="r", task_id="t1", text="second", author=MIND)
    assert result.ok and result.reason == "saved" and seen == ["locked"]
    assert [record["text"] for record in chronicle.room_records("r")] == ["first", "second"]


_JSON_ARROW = re.compile(r"->>?\s*'\$")  # SQLite's JSON operators: 3.38+, above the 3.24 floor (store.py)


def test_no_sql_uses_the_json_arrow_operators_the_sqlite_floor_lacks():
    """``->`` and ``->>`` need SQLite 3.38: on an install whose Python links an older SQLite (Ubuntu 22.04's
    3.37.2) the statement does not prepare, and a capture that reads it fails whole. ``json_extract`` is at the floor."""
    root = pathlib.Path(chronicle_store.__file__).resolve().parents[1]
    sources = [*(root / "ouroboros").rglob("*.py"), *(root / "supervisor").rglob("*.py"), root / "server.py"]
    assert [str(path.relative_to(root)) for path in sources if _JSON_ARROW.search(path.read_text(encoding="utf-8"))] == []
    assert _JSON_ARROW.search("body->'$.metadata.candidate'") and _JSON_ARROW.search("body ->> '$.a'")
    assert not _JSON_ARROW.search("json_extract(body,'$.metadata.candidate')")


def test_a_publication_between_two_reads_is_visible_to_the_second(tmp_path):
    store = ChronicleStore(tmp_path)
    first = _note(store, "first")
    assert [r["id"] for r in store.room_records("r")] == [first["id"]]
    second = _note(ChronicleStore(tmp_path), "second")  # another writer instance, same authority
    assert [r["id"] for r in store.room_records("r")] == [first["id"], second["id"]]


# --- one capture: one state of the index for every reader --------------------------------------------

LIGHT = {"kind": "helper", "route": "configured-light"}
LEGACY = {"kind": "legacy_helper", "writer": "old_consolidator"}
SOURCE = {"store": [7, 7], "frontier": 0}  # what the stored tail facts below were projected from


def _page(store, *refs, at, author=MIND, **kwargs):
    result = store.publish_page(room_id="r", text=f"page over {refs}", author=author,
                                covers={"rows": list(refs), "stream_span": [at, at + len(refs) - 1]}, **kwargs)
    assert result.ok, result
    return result.record["id"]


def _part(store, *members):
    result = store.publish_part(room_id="r", text=f"part over {members}", member_ids=list(members), author=MIND,
                                expected_sequence=store.room_head("r"))
    assert result.ok, result
    return result.record["id"]


def _every_kind(store):
    """A chronicle with every record kind and every derived fact the readers interpret."""
    assert store.ensure_activated()["kind"] == "activation"
    legacy = store.publish([{"id": "legacy-b01-rr", "kind": "legacy", "room_id": "r", "text": "retelling",
                             "author": LEGACY, "metadata": {"legacy_type": "era", "legacy_block": 1}},
                            {"id": "gap-s", "kind": "gap", "room_id": "s", "author": {"kind": "host"},
                             "metadata": {"source_gap": "test"}}]).record["id"]
    pages = [_page(store, f"a{n}", at=n) for n in range(4)]
    nested = _part(store, _part(store, *pages[:2]), _part(store, *pages[2:]))
    for at, accepted in ((10, True), (11, False)):
        assert store.decide(_page(store, f"b{at}", at=at, author=LIGHT), accepted, MIND, "judged").ok
    _page(store, "c1", at=20, author=LIGHT, metadata={"candidate": True})  # pending
    assert store.decide(_page(store, "c2", at=21, author=LIGHT, metadata={"candidate": True}), True, MIND, "adopt").ok
    kept = _note(store, "sealed note")
    sealing = _page(store, "note:" + kept["id"], "a9", at=30)
    assert store.correct(sealing, "said better", MIND).ok
    account = store.publish_account(room_id="r", text="across", sources=[sealing, {"id": legacy}], author=MIND)
    assert store.select_account(account.record["id"], replaces=[sealing], author=MIND, reason="told").ok
    seen = store.mark({"kind": "chronicle", "id": sealing}, "keep", MIND, room_id="r").record["id"]
    store.mark({"kind": "task", "task_id": "t1"}, "for all", MIND, room_id="s", scope="global")
    gone = store.mark({"kind": "task", "task_id": "t1"}, "later", MIND, room_id="r").record["id"]
    assert store.set_mark_view(seen, "meaning", MIND, "enough").ok and store.release_mark(gone, MIND, "done").ok
    with store.log_path.open("a", encoding="utf-8") as stream:
        stream.write('{"kind": "transaction", "records": [\n')  # a torn transaction: an addressed journal gap
    assert store.publish([], scan_state={"probe": {"k": 1}}).ok
    store.mark({"kind": "task", "task_id": "t1"}, "the room's newest record", MIND, room_id="r")  # not a head kind
    _helper_parts(store)
    rows = [[0, 0, "a" * 64, None, {"type": "x"}, None], [1, 5, "b" * 64, [3, 4], {"type": "task_summary"}, "done"],
            [2, 9, "c" * 64, None, {"direction": "out"}, ""]]
    assert store.extend_tail(None, rows, {"source": SOURCE, "upper": 10, "anchor": [9, "c" * 64], "complete": True})
    return nested


def _helper_parts(store):
    """Room s: dated pages, and a helper's part that acts, one the mind rejected and a candidate (``_status``)."""
    pages = []
    for n in range(3):
        covers = {"rows": [f"s{n}"], "stream_span": [n, n], "ts_span": {"start": f"2026-09-0{n + 1}T00:00:00+00:00",
                                                                       "end": f"2026-09-0{n + 1}T01:00:00+00:00"}}
        page = store.publish_page(room_id="s", text=f"page of s over s{n}", author=LIGHT if n else MIND, covers=covers)
        assert page.ok, page
        pages.append(page.record["id"])
    for n, metadata in ((0, {}), (1, {}), (2, {"candidate": True})):
        part = store.publish_part(room_id="s", text=f"part over s{n}", member_ids=[pages[n]], author=LIGHT,
                                  expected_sequence=store.room_head("s"), metadata=metadata)
        assert part.ok, part
        if n == 1:
            assert store.decide(part.record["id"], False, MIND, "not this one").ok


def _readers(store, nested):
    ids = [record["id"] for record in store.records()] + ["missing"]
    return ([store.get(record_id) for record_id in ids], store.records(), store.records("r"),
            store.records(kinds=("page", "part")), store.records("r", kinds=("note", "mark"), after_seq=3),
            store.room_records("r"), store.room_records("r", after_seq=8), store.room_records("s"),
            store.pages_of_room("r"), store.interpreted(kinds=("page",), author_kind="helper"), store.interpreted("r", kinds=("page",), author_kind="helper"), store.accounts(),
            store.folded_members(nested), store.sealed_row_refs("r"), store.sealed_row_refs("x"),
            store.legacy_pointer_rows(), store.active_marks(None), store.active_marks("r"),
            store.active_marks("r", include_global=False), store.active_marks("x"), store.scan_state(),
            store.activation(), store.room_head("r"), store.room_head("s"), store.room_head("x"), store.journal_size(),
            store.room_records(None), store.room_records(None, kinds=("note",)),
            store.room_records("r", kinds=("legacy",)), store.acting_pages(), store.acting_pages("s"),
            store.acting_pages("x"), store.acting_parts(), store.tail_state(), store.tail_facts(SOURCE),
            store.tail_facts(SOURCE, terminal_only=True), store.tail_facts({"store": [0], "frontier": 0}))


def test_every_reader_inside_a_capture_answers_as_the_index_and_enters_it_once(tmp_path, monkeypatch):
    store = ChronicleStore(tmp_path)
    nested = _every_kind(store)
    outside = _readers(store, nested)
    assert outside[13] and outside[9] and outside[11] and outside[12] and any(r["kind"] == "gap" for r in outside[1])
    entered, real = [], ChronicleStore._index
    monkeypatch.setattr(ChronicleStore, "_index", lambda self: entered.append(1) or real(self))
    with chronicle_store.capture(tmp_path):
        inside = _readers(store, nested)
    assert entered == [1]  # the capture's own read; its readers ask the snapshot
    assert inside == outside


@pytest.mark.parametrize("scope, held, entered", [("terminal", [1, 2], 3), ("none", [], 4)])
def test_a_capture_holds_the_tail_rows_its_view_reads_and_answers_every_reader_as_the_index(
        tmp_path, monkeypatch, scope, held, entered):
    store = ChronicleStore(tmp_path)
    nested = _every_kind(store)
    outside = _readers(store, nested)
    assert outside[-3][1] and outside[-2][1] and outside[-1] is None  # the readers ask the stored tail facts
    count, real = [], ChronicleStore._index
    monkeypatch.setattr(ChronicleStore, "_index", lambda self: count.append(1) or real(self))
    with chronicle_store.capture(tmp_path, tail=scope):
        assert [row[0] for row in store._pinned().tail_rows] == held  # the rows its view reads, no more
        inside = _readers(store, nested)
    assert inside == outside and len(count) == entered  # each tail read it does not hold asks the index


def _counts_from_bodies(store):
    """The reference: every room's acting pages and parts read whole, as the counts were read before."""
    counts = {"pages_by_me": 0, "latest_by_me": "", "helper_pages": 0, "parts": 0}
    for room in sorted({str(record["room_id"]) for record in store.records(kinds=("page", "part"))}):
        for record in store.pages_of_room(room):
            if record["kind"] == "part":
                counts["parts"] += 1
            elif (record.get("author") or {}).get("kind") == "mind":
                counts["pages_by_me"] += 1
                counts["latest_by_me"] = max(counts["latest_by_me"], str(record.get("ts") or "")[:10])
            else:
                counts["helper_pages"] += 1
    return counts


def test_the_story_counts_and_a_room_s_pages_are_counted_in_the_index_without_reading_a_body(tmp_path, monkeypatch):
    store = ChronicleStore(tmp_path)
    _every_kind(store)
    expected, rooms = _counts_from_bodies(store), {}
    for room in ("r", "s", "x"):
        pages = [record for record in store.pages_of_room(room) if record["kind"] == "page"]
        ends = [end for record in pages if (end := (record["covers"].get("ts_span") or {}).get("end"))]
        rooms[room] = (len(pages), str(source_time_span(ends).get("end") or "") if ends else "")
    assert expected["helper_pages"] and expected["parts"] == 4 and rooms["s"] == (3, "2026-09-03T01:00:00+00:00")
    parsed, real = [], json.loads
    monkeypatch.setattr(chronicle_store, "json", SimpleNamespace(dumps=json.dumps,
                                                                 loads=lambda text: parsed.append(text) or real(text)))
    for held in (contextlib.nullcontext, lambda: chronicle_store.capture(tmp_path)):
        with held():
            parsed.clear()
            assert memory_inventory.story_counts(store) == expected
            assert {room: (facts["pages"], facts["last_covered"]) for room in rooms
                    for facts in [memory_inventory.room_facts(store, room, [], {}, 0)]} == rooms
        assert not [text for text in parsed if " over " in text]  # no page or part read whole


def test_a_capture_reads_one_state_and_holds_no_read_transaction_while_its_readers_run(tmp_path):
    store = ChronicleStore(tmp_path)
    first = _note(store, "first")
    with chronicle_store.capture(tmp_path):
        second = _note(ChronicleStore(tmp_path), "second")  # a publisher commits while the capture is pinned
        assert [r["id"] for r in store.room_records("r")] == [first["id"]] and store.get(second["id"]) is None
        assert store.room_head("r") == first["sequence"]  # every reader of the capture: the one state it read
    assert [r["id"] for r in store.room_records("r")] == [first["id"], second["id"]]
