"""A read of an absent store answers from one reused, read-only empty copy of the schema (``store_empty``).

The copy replaces a schema build per read: under a busy server every DDL statement of a build waited for
the GIL, and a history page reads one absent store per settled child drive it names.
"""

from __future__ import annotations

import sqlite3
import threading

import pytest

from ouroboros import dialogue_store as ds, store, store_empty
from tests._store_shared import seed_result


@pytest.fixture
def builds(monkeypatch):
    """Every new in-memory copy this thread makes (each is one schema build), from a clean slate."""
    monkeypatch.setattr(store_empty, "_IDLE", threading.local())
    made, real = [], sqlite3.connect

    def counted(database, *args, **kwargs):
        if database == ":memory:":
            made.append(threading.get_ident())
        return real(database, *args, **kwargs)

    monkeypatch.setattr(store_empty.sqlite3, "connect", counted)
    return made


def _absent_reads(roots):
    seen = []
    for root in roots:
        with store.open_store(root, write=False) as txn:
            assert txn.task_ids() == []
            seen.append(id(txn.conn))
        with ds.transaction(root, write=False) as conn:
            assert conn.execute("SELECT COUNT(*) FROM dialogue").fetchone()[0] == 0
            seen.append(id(conn))
    return seen


def test_absent_stores_read_one_reused_copy_per_schema(tmp_path, builds):
    roots = [tmp_path / f"settled-child-{n}" for n in range(25)]  # removed drives: no directory at all
    seen = _absent_reads(roots)
    assert len(builds) == 2, "one current-store copy and one dialogue copy, however many roots are absent"
    assert len(set(seen)) == 2
    _absent_reads(roots[:3])
    assert len(builds) == 2


def test_an_empty_copy_refuses_writes_and_stays_empty(tmp_path, builds):
    with store.open_store(tmp_path / "gone", write=False) as txn:
        with pytest.raises(sqlite3.OperationalError):
            txn.conn.execute("INSERT INTO meta (key, value) VALUES ('phase', '\"x\"')")
    with store.open_store(tmp_path / "gone", write=False) as txn:
        assert txn.conn.execute("SELECT COUNT(*) FROM meta").fetchone()[0] == 0
    assert len(builds) == 1, "a refused write leaves the copy reusable"


def test_a_nested_read_takes_its_own_copy(tmp_path, builds):
    with store.open_store(tmp_path / "a", write=False) as outer:
        with store.open_store(tmp_path / "b", write=False) as inner:
            assert inner.conn is not outer.conn and outer.conn.in_transaction and inner.conn.in_transaction
        assert outer.conn.in_transaction, "the nested read never ends the caller's transaction"
    assert len(builds) == 2
    _absent_reads([tmp_path / "c"])
    assert len(builds) == 3, "the dialogue schema builds its own copy; the two current copies are reused"


def test_a_failed_read_returns_a_clean_copy(tmp_path, builds):
    with pytest.raises(RuntimeError):
        with store.open_store(tmp_path / "gone", write=False):
            raise RuntimeError("the reader failed")
    with store.open_store(tmp_path / "gone", write=False) as txn:
        assert txn.task_ids() == []
    assert len(builds) == 1


def test_a_store_published_later_is_read_from_its_file(tmp_path, builds):
    root = tmp_path / "child"
    with store.open_store(root, write=False) as txn:
        assert txn.task_ids() == []
    seed_result(root, "task-a", {"task_id": "task-a", "status": "completed"})
    with store.open_store(root, write=False) as txn:
        assert txn.task_ids() == ["task-a"], "absence is asked on every open, never remembered"


def test_each_thread_and_a_forked_child_build_their_own_copy(tmp_path, builds):
    _absent_reads([tmp_path / "main"])
    worker = threading.Thread(target=_absent_reads, args=([tmp_path / "worker"],))
    worker.start()
    worker.join()
    assert len(builds) == 4 and len(set(builds)) == 2, "a SQLite connection never crosses threads"
    store_empty.forget()
    _absent_reads([tmp_path / "after-fork"])
    assert len(builds) == 6
