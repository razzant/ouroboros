"""The task card's display facts live in their own row: a tool result or a narration moves one small
row, never the body and never the document signature; a stale whole-document write and an import
cannot regress them; the census takes them from the hot row; export and re-import keep them."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from types import SimpleNamespace

import pytest

from ouroboros import store, store_hot_facts as hot, store_import, store_switch, store_task_cards as cards, task_store
from ouroboros.gateway import state
from ouroboros.store import document_text
from ouroboros.task_activity_facts import record_narration, record_tool_count
from ouroboros.task_pacing import record_tool_activity
from ouroboros.task_results import load_task_result, write_task_result
from tests._store_journal_shared import use_journal
from tests.test_admission_contention import database_reader
from tests.test_store_import import result, write_result


@pytest.fixture(autouse=True)
def release_connections():
    store.close_connections()
    yield
    store.close_connections()


def _context(root, task_id="work", calls=1):
    usage = {"_pacing_tool_activity": {"read_file": {"calls": calls, "error_calls": 0}}}
    return SimpleNamespace(task_id=task_id, drive_root=root, _accumulated_usage=usage)


def _stored(root, task_id):
    """``(body text, write_seq, updated_at, terminal_writes, card row)`` as the store holds them."""
    with store.open_store(root, write=False) as txn:
        row = cards.read(txn, task_id)
        return (txn.body_text(task_id), *txn.row(task_id, "write_seq", "updated_at"), txn.meta(store.TERMINAL_WRITES),
                dict(row) if row is not None else None)


def _legacy_row(txn, document):
    """A row an older writer left: the whole document in the body, card copies included, no card row."""
    task_id = str(document["task_id"])
    columns = {**store.typed_columns(document), "write_seq": (txn.row(task_id, "write_seq") or 0) + 1}
    txn.conn.execute("INSERT OR REPLACE INTO task_bodies VALUES (?,?)", (task_id, document_text(document)))
    names = ("task_id", *columns)
    txn.conn.execute(f"INSERT OR REPLACE INTO tasks ({','.join(names)}) VALUES ({','.join('?' * len(names))})",
                     (task_id, *columns.values()))


@pytest.mark.parametrize("mode", ["PERSIST", "WAL"])
def test_card_writes_move_no_body_and_no_signature_and_reach_the_census(tmp_path, monkeypatch, mode):
    use_journal(monkeypatch, mode)
    write_task_result(tmp_path, "work", "running", chat_id=1)
    before = _stored(tmp_path, "work")
    assert before[4] is None
    memo_id = (str(tmp_path.resolve()), "work")
    facts = state._task_activity_facts(tmp_path, "work")
    key = state._FINALIZING_MEMO[memo_id][0]
    assert facts["card"]["tool_calls"] == 0 and facts["card"]["last_line"] == ""
    context = _context(tmp_path, calls=0)
    for _ in range(200):
        record_tool_activity(context, {"fn_name": "read_file", "is_error": False})
    for number in range(1, 51):
        record_narration(tmp_path, {"task_id": "work", "narration": True, "text": f"💬 line {number}",
                                    "ts": f"2026-10-20T10:{number:02d}:00Z"})
    after = _stored(tmp_path, "work")
    assert after[:4] == before[:4], "a card write rewrote the body, moved the signature or counted a terminal write"
    assert after[4] == {"tool_calls": 200, "last_line": "line 50", "last_line_at": "2026-10-20T10:50:00Z"}
    document = load_task_result(tmp_path, "work", strict=True)
    assert (document["tool_calls"], document["last_line"], document["last_line_at"]) == (
        200, "line 50", "2026-10-20T10:50:00Z")
    assert "tool_calls" not in json.loads(after[0]), "the body must not carry a stale copy"
    row = hot.read(tmp_path, "work")
    assert (row["tool_calls"], row["last_line"], row["write_seq"]) == (200, "line 50", before[1])
    facts = state._task_activity_facts(tmp_path, "work")
    assert facts["card"]["tool_calls"] == 200 and facts["card"]["last_line"] == "line 50"
    assert state._FINALIZING_MEMO[memo_id][0] == key, "the memo key moved although no document changed"
    assert state._FINALIZING_MEMO[memo_id][1]["card"]["tool_calls"] == 0, "the memoized dict was overwritten in place"


def test_a_stale_whole_document_write_never_regresses_a_card_and_an_import_creates_the_row(tmp_path):
    write_task_result(tmp_path, "work", "running", tool_calls=3, last_line="first", last_line_at="2026-10-20T10:00:00Z")
    body, *_rest, row = _stored(tmp_path, "work")
    assert row == {"tool_calls": 3, "last_line": "first", "last_line_at": "2026-10-20T10:00:00Z"}
    assert not {"tool_calls", "last_line", "last_line_at"} & set(json.loads(body))
    record_tool_count(_context(tmp_path, calls=9), {"read_file": {"calls": 9}})
    record_narration(tmp_path, {"task_id": "work", "narration": True, "text": "second", "ts": "2026-10-20T10:05:00Z"})
    # An older count, an older line: the merge keeps the newer card whoever writes.
    write_task_result(tmp_path, "work", "running", tool_calls=5, last_line="stale", last_line_at="2026-10-20T10:01:00Z")
    record_narration(tmp_path, {"task_id": "work", "narration": True, "text": "older", "ts": "2026-10-20T09:59:00Z"})
    task_store.patch_document(tmp_path, "work", lambda current: {
        key: value for key, value in current.items() if key not in cards.FIELDS})  # a mutator that drops them
    document = load_task_result(tmp_path, "work", strict=True)
    assert (document["tool_calls"], document["last_line"], document["last_line_at"]) == (
        9, "second", "2026-10-20T10:05:00Z")
    # A newer line and a higher count from a whole-document write do land (the same rule in both directions).
    write_task_result(tmp_path, "work", "running", tool_calls=12, last_line="third", last_line_at="2026-10-20T10:06:00Z")
    assert _stored(tmp_path, "work")[4] == {"tool_calls": 12, "last_line": "third", "last_line_at": "2026-10-20T10:06:00Z"}
    # An import of a file carrying card values in the middle of the document creates the row, strips the
    # body, and its content baseline hashes the shape ``load`` returns: no false conflict against itself.
    root = tmp_path / "imported"
    write_result(root, {**result("filed", "completed", tool_calls=7), "last_line": "done",
                        "last_line_at": "2026-10-20T11:00:00Z", "result": "r"})
    store_import.ensure_current_store(root, with_children=False)
    stored = _stored(root, "filed")
    assert stored[4] == {"tool_calls": 7, "last_line": "done", "last_line_at": "2026-10-20T11:00:00Z"}
    assert not {"tool_calls", "last_line", "last_line_at"} & set(json.loads(stored[0]))
    with store.open_store(root, write=False) as txn:
        loaded = txn.load("filed")
        assert list(loaded)[-3:] == ["tool_calls", "last_line", "last_line_at"]
        projected = hashlib.sha256(document_text(loaded).encode("utf-8")).hexdigest()
        assert txn.meta("baseline")["filed"]["body_sha256"] == projected


def test_a_card_write_for_an_absent_task_creates_nothing_and_the_row_goes_with_the_task(tmp_path):
    write_task_result(tmp_path, "work", "running")
    record_narration(tmp_path, {"task_id": "missing", "narration": True, "text": "Hello", "ts": "2026-10-20T10:00:00Z"})
    record_tool_count(_context(tmp_path, task_id="missing", calls=4), {"read_file": {"calls": 4}})
    with store.open_store(tmp_path, write=True) as txn:
        assert cards.merge(txn, "missing", {"tool_calls": 1}) == 0
        assert cards.merge(txn, "work", {"tool_calls": "7"}) == 0  # nothing admissible: no row either
        assert txn.conn.execute("SELECT count(*) FROM task_cards").fetchone()[0] == 0
    assert load_task_result(tmp_path, "missing") is None
    record_tool_count(_context(tmp_path, calls=4), {"read_file": {"calls": 4}})
    assert _stored(tmp_path, "work")[4]["tool_calls"] == 4
    assert task_store.delete_document(tmp_path, "work")
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.conn.execute("SELECT count(*) FROM task_cards").fetchone()[0] == 0


@pytest.mark.parametrize("mode", ["PERSIST", "WAL"])
def test_a_card_write_waits_one_slice_then_keeps_the_previous_card(tmp_path, monkeypatch, mode):
    use_journal(monkeypatch, mode)
    write_task_result(tmp_path, "work", "running")
    record_tool_count(_context(tmp_path, calls=1), {"read_file": {"calls": 1}})
    with database_reader(tmp_path, write=True) as release:  # the journal mode's own way of contending a write
        started = time.monotonic()
        record_tool_count(_context(tmp_path, calls=2), {"read_file": {"calls": 2}})  # raises nothing
        waited = time.monotonic() - started
        assert waited < 3.0, f"the card write waited {waited:.1f}s: more than its acquisition slice"
        assert _stored(tmp_path, "work")[4]["tool_calls"] == 1
        release()
    record_tool_count(_context(tmp_path, calls=2), {"read_file": {"calls": 2}})
    assert _stored(tmp_path, "work")[4]["tool_calls"] == 2


def _schema_six(root, monkeypatch, document):
    """A real schema-6 store (hot columns, no card table) holding one legacy row with card copies in its body."""
    with monkeypatch.context() as patch:
        patch.setattr(store, "_SCHEMA", store._SCHEMA.replace(cards.SCHEMA, ""))
        patch.setattr(store, "SCHEMA_VERSION", 6)
        with store.open_store(root, write=True) as txn:
            _legacy_row(txn, document)
    store.close_connections(store.store_path(root))
    with store.open_store(root, write=False) as txn:
        assert not cards.present(txn) and txn.meta("schema_version") == 6


def test_a_schema_six_store_upgrades_without_decoding_bodies_and_the_export_keeps_the_card(tmp_path, monkeypatch):
    from ouroboros.store_import import upgrade_current_schema

    legacy = {"task_id": "legacy", "status": "running", "_schema_version": 1, "tool_calls": 4, "last_line": "old",
              "last_line_at": "2026-10-20T09:00:00Z", "result": "r"}
    _schema_six(tmp_path, monkeypatch, legacy)
    assert "tool_calls" not in hot.read(tmp_path, "legacy")  # no card row to read on this schema
    assert load_task_result(tmp_path, "legacy")["tool_calls"] == 4  # the legacy body copy stays visible
    assert json.loads(task_store.display_documents(tmp_path, ["legacy"])["legacy"])["last_line"] == "old"
    with store.open_store(tmp_path, write=False) as txn:
        job = txn.meta("job:" + hot.JOB)
    with monkeypatch.context() as patch:
        patch.setattr(store.Txn, "body_text", lambda *_: pytest.fail("the schema upgrade decoded a task body"))
        upgrade_current_schema(tmp_path)
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.meta("schema_version") == store.SCHEMA_VERSION and cards.present(txn)
        assert txn.meta("job:" + hot.JOB) == job, "a card upgrade must not reset the hot-facts job"
    assert load_task_result(tmp_path, "legacy")["tool_calls"] == 4
    record_tool_count(_context(tmp_path, task_id="legacy", calls=9), {"read_file": {"calls": 9}})
    document = load_task_result(tmp_path, "legacy", strict=True)
    assert (document["tool_calls"], document["last_line"]) == (9, "old")  # the row wins key by key over the copy
    assert hot.read(tmp_path, "legacy")["tool_calls"] == 9 and hot.read(tmp_path, "legacy")["last_line"] is None
    assert json.loads(_stored(tmp_path, "legacy")[0])["tool_calls"] == 4  # the stale copy goes with the next body write
    write_task_result(tmp_path, "legacy", "completed", result="done")
    body, *_rest, row = _stored(tmp_path, "legacy")
    assert "tool_calls" not in json.loads(body) and row == {"tool_calls": 9, "last_line": "old",
                                                            "last_line_at": "2026-10-20T09:00:00Z"}
    # Export → re-import: the file carries the card (last), the rebuilt store holds it in the row again.
    with store.open_store(tmp_path, write=True, service=True) as txn:
        txn.set_meta("phase", store.EXPORTING)
    store_import.export_facts(tmp_path)
    exported = json.loads((tmp_path / "task_results" / "legacy.json").read_text(encoding="utf-8"))
    assert list(exported)[-3:] == ["tool_calls", "last_line", "last_line_at"] and exported["tool_calls"] == 9
    store_switch._retire(tmp_path)
    store.close_connections()
    assert store_import.ensure_current_store(tmp_path, with_children=False)["status"] == "completed"
    body, *_rest, row = _stored(tmp_path, "legacy")
    assert "tool_calls" not in json.loads(body) and row["tool_calls"] == 9
    assert load_task_result(tmp_path, "legacy", strict=True)["last_line"] == "old"


def test_listings_and_display_views_project_the_card_and_keep_inadmissible_values_in_the_body(tmp_path):
    write_task_result(tmp_path, "modern", "running", chat_id=1)
    write_task_result(tmp_path, "odd", "running", tool_calls="7", last_line=["not", "a", "line"])
    with store.open_store(tmp_path, write=True) as txn:
        _legacy_row(txn, {"task_id": "legacy", "status": "running", "_schema_version": 1, "tool_calls": 4,
                          "last_line": "old", "last_line_at": "2026-10-20T09:00:00Z", "parent_task_id": "modern",
                          "root_task_id": "modern", "delegation_role": "subagent"})
    record_tool_count(_context(tmp_path, task_id="legacy", calls=9), {"read_file": {"calls": 9}})
    record_narration(tmp_path, {"task_id": "modern", "narration": True, "text": "live", "ts": "2026-10-20T10:00:00Z"})
    expected = {"legacy": (9, "old"), "modern": (None, "live"), "odd": ("7", ["not", "a", "line"])}
    readers = {
        "list": {row["task_id"]: row for row in task_store.list_documents(tmp_path)},
        "by_ids": task_store.documents_by_ids(tmp_path, list(expected)),
        "lineage": {row["task_id"]: row for row in task_store.select_lineage(tmp_path, "modern", "modern")},
        "read": {tid: task_store.read_document(tmp_path, tid) for tid in expected},
        "display": {tid: json.loads(text) for tid, text in task_store.display_documents(tmp_path, list(expected)).items()},
    }
    for name, documents in readers.items():
        for tid, (calls, line) in expected.items():
            if tid not in documents:
                assert name == "lineage" and tid != "legacy"
                continue
            assert (documents[tid].get("tool_calls"), documents[tid].get("last_line")) == (calls, line), (name, tid)
    assert "tool_calls" not in readers["display"]["modern"], "a NULL column must not become a JSON null"
    assert readers["display"]["modern"]["last_line_at"] == "2026-10-20T10:00:00Z"
    assert "tool_calls" not in task_store.read_document(tmp_path, "modern")
    body, *_rest, row = _stored(tmp_path, "odd")
    assert json.loads(body)["tool_calls"] == "7" and row is None  # inadmissible: not a card, nothing lost
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.load("odd")["tool_calls"] == "7"


def test_the_merge_sql_runs_on_the_engine_floor(tmp_path):
    """Plain SQL: the merge needs no JSON function (the export runs the store modules bare)."""
    conn = sqlite3.connect(tmp_path / "bare.sqlite", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(store._SCHEMA)
    txn = store.Txn(conn, tmp_path)
    conn.execute("INSERT INTO tasks (task_id, status) VALUES ('t', 'running')")
    assert cards.merge(txn, "t", {"tool_calls": 2}) == 1
    assert cards.merge(txn, "t", {"tool_calls": 1, "last_line": "a", "last_line_at": "2"}) == 1
    assert cards.merge(txn, "t", {"last_line": "b", "last_line_at": "1"}) == 1  # older line: the update keeps "a"
    assert cards.merge(txn, "t", {"last_line_at": "9"}) == 0  # a stamp without its line is no card fact
    assert dict(cards.read(txn, "t")) == {"tool_calls": 2, "last_line": "a", "last_line_at": "2"}
    conn.close()
