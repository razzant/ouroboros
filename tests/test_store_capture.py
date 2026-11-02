"""The capture core (``ouroboros.store_capture``): one SQLite step per statement, values exact, decode after release."""
from __future__ import annotations

import json
import pathlib
import sqlite3

import pytest

from ouroboros.store_capture import CaptureTypeError, capture

TEXTS = ["", "plain", "é😀 中文", "nul\x00inside", 'quote " and \\ back', '{"json": [1, 2.5, true, null]}', "\n\t\r"]
INTEGERS = [0, 1, -1, 2 ** 63 - 1, -(2 ** 63), 2 ** 53 + 1]


def _db(directory: pathlib.Path, journal: str = "PERSIST") -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "capture.sqlite"
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute(f"PRAGMA journal_mode={journal}")
    conn.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, txt TEXT, num INTEGER, maybe)")
    conn.executemany("INSERT INTO t VALUES (?, ?, ?, ?)", [
        (index, TEXTS[index % len(TEXTS)], INTEGERS[index % len(INTEGERS)], None if index % 3 else TEXTS[1])
        for index in range(1, 60)])
    conn.close()
    return path


def test_text_integer_and_null_travel_exactly_and_json_text_stays_text(tmp_path):
    conn = sqlite3.connect(_db(tmp_path), isolation_level=None)
    direct = conn.execute("SELECT id, txt, num, maybe FROM t ORDER BY id").fetchall()
    captured = capture(conn, "SELECT id, txt, num, maybe FROM t", (), ("id", "txt", "num", "maybe"))
    assert sorted(captured.tuples()) == direct  # every value equal: NUL, emoji, 2**63-1 and NULL included
    assert all(isinstance(row["txt"], str) for row in captured.rows())  # JSON text stays a string
    ordered = captured.rows(key=lambda row: row["id"], reverse=True)
    assert [row["id"] for row in ordered] == [row[0] for row in reversed(direct)]


def test_an_empty_answer_and_names_that_are_the_selects_own(tmp_path):
    conn = sqlite3.connect(_db(tmp_path), isolation_level=None)
    assert capture(conn, "SELECT id FROM t WHERE id < 0", (), ("id",)).rows() == []
    with pytest.raises(sqlite3.OperationalError):  # a typo fails in SQLite instead of reading a literal string
        capture(conn, "SELECT id FROM t", (), ("idd",))
    for names in ((), ('id"; DROP TABLE t; --',), ("1id",)):
        with pytest.raises(ValueError):
            capture(conn, "SELECT id FROM t", (), names)
    assert conn.execute("SELECT count(*) FROM t").fetchone()[0] == 59


def test_a_real_is_refused_at_decode_and_a_blob_inside_sqlite(tmp_path):
    conn = sqlite3.connect(_db(tmp_path), isolation_level=None)
    conn.execute("UPDATE t SET num = 1.5 WHERE id = 2")  # a dynamic REAL in an INTEGER column
    conn.execute("UPDATE t SET maybe = 1.0 WHERE id = 3")  # an integral REAL where no affinity converts it
    for row_id, column in ((2, "num"), (3, "maybe")):
        with pytest.raises(CaptureTypeError):
            capture(conn, f"SELECT {column} FROM t WHERE id = ?", (row_id,), (column,)).rows()
    assert len(capture(conn, "SELECT id, num, maybe FROM t WHERE id > 3", (), ("id", "num", "maybe")).rows()) == 56
    with pytest.raises(sqlite3.Error):
        capture(conn, "SELECT x'00ff' AS raw", (), ("raw",))


def test_one_statement_hands_python_one_row_however_many_rows_it_answers(tmp_path):
    conn = sqlite3.connect(_db(tmp_path), isolation_level=None)
    statements, crossed = [], []

    class Counted(sqlite3.Row):
        def __new__(cls, cursor, data):
            crossed.append(1)
            return super().__new__(cls, cursor, data)

    conn.set_trace_callback(statements.append)
    conn.row_factory = Counted
    rows = capture(conn, "SELECT id, txt FROM t", (), ("id", "txt")).rows()
    assert (len(rows), len(statements), len(crossed)) == (59, 1, 1)
    crossed.clear()
    assert len(conn.execute("SELECT id, txt FROM t").fetchall()) == 59 and len(crossed) == 59  # the negative control


def test_a_writer_commits_while_the_captured_answer_is_decoded_but_not_while_rows_are_stepped(tmp_path):
    """Rollback journal: a writer's COMMIT fails at once (no busy wait) while a read steps rows inside its
    transaction, and commits at once after the capture's transaction ended and before its decode."""
    path = _db(tmp_path)
    reader = sqlite3.connect(path, isolation_level=None)
    writer = sqlite3.connect(path, isolation_level=None, timeout=0)

    def write(value: str) -> None:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("UPDATE t SET txt=? WHERE id=1", (value,))
        try:
            writer.execute("COMMIT")
        except sqlite3.OperationalError:
            writer.execute("ROLLBACK")
            raise

    reader.execute("BEGIN")
    reader.execute("SELECT id, txt FROM t").fetchone()  # a row-by-row read in progress holds SHARED
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        write("blocked")
    reader.execute("COMMIT")
    reader.execute("BEGIN")
    captured = capture(reader, "SELECT id, txt FROM t", (), ("id", "txt"))
    reader.execute("COMMIT")
    write("after")  # nothing holds SHARED while the answer is decoded
    assert next(row["txt"] for row in captured.rows() if row["id"] == 1) == TEXTS[1]  # the read's own snapshot
    assert reader.execute("SELECT txt FROM t WHERE id=1").fetchone()[0] == "after"


def test_captures_inside_one_transaction_are_one_snapshot(tmp_path):
    """WAL, where a writer can commit beside an open read: both captures of that read see one snapshot."""
    path = _db(tmp_path, "WAL")
    reader = sqlite3.connect(path, isolation_level=None)
    writer = sqlite3.connect(path, isolation_level=None, timeout=0)
    reader.execute("BEGIN")
    first = capture(reader, "SELECT id, txt FROM t WHERE id=1", (), ("id", "txt"))
    writer.execute("UPDATE t SET txt='changed' WHERE id=1")
    second = capture(reader, "SELECT id, txt FROM t WHERE id=1", (), ("id", "txt"))
    reader.execute("COMMIT")
    assert first.rows() == second.rows() == [{"id": 1, "txt": TEXTS[1]}]
    assert json.loads(capture(reader, "SELECT txt FROM t WHERE id=1", (), ("txt",)).text) == [["changed"]]
