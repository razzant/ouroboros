"""The WAL checkpoint job (``store_checkpoint``): a log's backlog of committed frames wakes it, its PASSIVE pass
copies the log off the interactive path and records SQLite's numbers, and its look reads only the logs' headers and
rests while nothing new was committed. The connection ceiling itself is held in tests/test_store_journal.py."""
from __future__ import annotations

import builtins
import io
import json
import os
import sqlite3
import threading
import time

import pytest

from ouroboros import (dialogue_store, maintenance, maintenance_jobs, store, store_checkpoint, store_journal, store_timing,
                       usage_store)
from tests._store_journal_shared import use_journal
from tests.test_store import publish


@pytest.fixture(autouse=True)
def release_connections():
    store.close_connections()
    yield
    store.close_connections()


def _root(tmp_path, monkeypatch, mode):
    """A data root whose three stores were built in MODE, as a fresh install builds them."""
    use_journal(monkeypatch, mode)
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    publish(data)
    dialogue_store.create(data)
    usage_store.migrate_from_journal(data)
    return data


@pytest.fixture
def root(tmp_path, monkeypatch):
    return _root(tmp_path, monkeypatch, "WAL")


def _backlog(root, frames=None, key="filler"):
    """One commit leaving at least FRAMES (SQLite's own threshold) committed frames in the current store's log."""
    with store.open_store(root, write=True) as txn:
        txn.set_meta(key, "x" * (4096 * (frames or store_checkpoint.inline_frames())))


def _job(root):
    return maintenance_jobs.job(root, store_checkpoint.NAME) or {}


def _passes(root):
    """The passes' rows of the store-timing journal."""
    journal = root / "logs" / store_timing.JOURNAL
    lines = journal.read_text(encoding="utf-8").splitlines() if journal.exists() else []
    return [row for row in map(json.loads, lines) if row["type"] == store_checkpoint.NAME]


def test_a_backlog_past_sqlite_s_own_threshold_wakes_the_job_and_its_pass_copies_the_log(root):
    """The look is due once a log holds SQLite's own threshold of committed frames no pass saw. The pass copies
    them (``checkpointed`` reaches ``log``; the database file takes the pages), records SQLite's numbers and the log
    it saw per store kind, and one row of the store-timing journal, none in the supervisor journal whose tail every
    context reads; the job then rests (its own row write is no backlog) until the log grows by the threshold again,
    in the log the next writer reset."""
    path = store.store_path(root)
    assert store_checkpoint.inline_frames() == 1000  # SQLite's own default on this build
    assert not store_checkpoint.due(root, {}, {}, False), "building the stores left no backlog"
    _backlog(root)
    assert store_checkpoint.due(root, {}, {}, False)
    seen, size = store.log_identity(path)[:2], path.stat().st_size
    report = maintenance.run_job(root, store_checkpoint.NAME)
    current = _job(root)["stores"]["current"]
    assert report["state"] == "done" and report["items"] >= 1
    assert current["busy"] == 0 and current["checkpointed"] == current["log"] == seen[1] >= 1000
    assert current["wal"] == seen and current["seconds"] >= 0 and current["at"]
    assert path.stat().st_size > size, "the pass moved the log's pages into the database file"
    assert store_timing.JOURNAL == "store_sites.jsonl"
    rows = [json.loads(line) for line in (root / "logs" / store_timing.JOURNAL).read_text(encoding="utf-8").splitlines()]
    assert [row["stores"]["current"] for row in rows if row["type"] == store_checkpoint.NAME] == [
        {key: current[key] for key in ("busy", "log", "checkpointed", "seconds", "wal")}], "its history keeps the log"
    supervisor = root / "logs" / "supervisor.jsonl"
    assert not supervisor.exists() or store_checkpoint.NAME not in supervisor.read_text(encoding="utf-8")
    assert maintenance.run_job(root, store_checkpoint.NAME) is None
    assert store.log_identity(path)[0] != seen[0], "the job's own row write began a reset log"
    _backlog(root)
    assert store_checkpoint.due(root, _job(root), {}, False), "a reset log holding the threshold is due again"


def test_a_reader_s_snapshot_leaves_the_pass_partial_and_the_pass_waits_for_no_writer(root):
    """PASSIVE copies up to the oldest snapshot a reader holds and waits for nobody: with a reader on an older
    snapshot and another connection holding the write lock, the pass copies what it may while that writer still
    holds the lock (the database file grows), and the writer then commits without waiting (busy timeout 0); the
    pass records ``checkpointed`` below ``log``."""
    path = store.store_path(root)
    _backlog(root, 8, key="before")  # pages the reader's snapshot holds whose newest copy it saw: the pass may copy them
    reader = sqlite3.connect(path, isolation_level=None)
    writer = sqlite3.connect(path, isolation_level=None, timeout=0, check_same_thread=False)
    worker = threading.Thread(target=maintenance.run_job, args=(root, store_checkpoint.NAME), daemon=True)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT COUNT(*) FROM meta").fetchone()
        _backlog(root)
        size = path.stat().st_size
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('during', '1')")
        worker.start()
        deadline = time.monotonic() + 30
        while path.stat().st_size == size and time.monotonic() < deadline:
            time.sleep(0.01)
        assert path.stat().st_size > size, "the pass waited for the writer instead of copying what it may"
        writer.execute("COMMIT")  # timeout 0: nothing the pass holds keeps a writer waiting
    finally:
        worker.join(30)
        reader.execute("ROLLBACK")
        for conn in (reader, writer):
            conn.close()
    assert not worker.is_alive()
    current = _job(root)["stores"]["current"]
    assert current["busy"] == 0 and 0 < current["checkpointed"] < current["log"], current


def test_the_look_reads_the_logs_headers_and_opens_neither_a_store_nor_its_index(root, monkeypatch):
    """The due check runs on the server's lane thread beside live transactions: it reads each log's header and
    frame headers and a database header beside a log, and never opens a SQLite connection or the ``-shm`` index,
    whose descriptor's close would drop this process's locks on it (tests/test_store_wal_index.py)."""
    _backlog(root)
    store_checkpoint.inline_frames()  # read once per process from a memory database
    connects, opened = [], []

    def spy(real, seen):
        def call(target, *args, **kwargs):
            seen.append(target if isinstance(target, int) else os.fspath(target))
            return real(target, *args, **kwargs)
        return call

    monkeypatch.setattr(sqlite3, "connect", spy(sqlite3.connect, connects))
    for module, name in ((builtins, "open"), (io, "open"), (os, "open")):
        monkeypatch.setattr(module, name, spy(getattr(module, name), opened))
    assert store_checkpoint.due(root, {}, {}, False)
    assert connects == [] and not [name for name in opened if str(name).endswith("-shm")], (connects, opened)
    assert str(store.store_path(root)) + "-wal" in opened


def test_a_store_that_fails_its_pass_keeps_its_error_at_the_log_it_met_and_waits_for_the_log_to_move(root, monkeypatch):
    """A store whose open or checkpoint fails records its error class at the log it met, beside the stores that
    passed: no look is due again (no spawn on every look) until that log grows by the threshold."""
    real = store_journal.root_stores

    def unreadable_tier():
        raise store.StoreLost("current store unreadable")

    def stores(drive):
        (path, _tier, lock), *rest = real(drive)
        return ((path, unreadable_tier, lock), *rest)

    monkeypatch.setattr(store_journal, "root_stores", stores)
    _backlog(root)
    seen = store.log_identity(store.store_path(root))[:2]
    assert maintenance.run_job(root, store_checkpoint.NAME)["items"] >= 1
    current = _job(root)["stores"]["current"]
    assert current == {"error": "StoreLost", "wal": seen, "at": current["at"]}
    assert not store_checkpoint.due(root, _job(root), {}, False)
    _backlog(root)
    assert store_checkpoint.due(root, _job(root), {}, False)


@pytest.mark.parametrize("held", [lambda: store.StoreBusy("current store held", reason="contention"),
                                  lambda: sqlite3.OperationalError("database is locked")], ids=["StoreBusy", "locked"])
def test_a_store_found_held_keeps_its_last_entry_and_stays_due(root, monkeypatch, held):
    """Contention is no fault (``maintenance._contention``): a store whose open meets a held lock keeps the entry its
    last pass left, so the next look is due for it again, and the pass says ``pending`` and writes no row it would
    not change. Once the store is free, a pass records it and the job rests. (A fault other than contention records
    its error at the log it met: the test above.)"""
    real = store_journal.root_stores

    def busy():
        raise held()

    _backlog(root)
    maintenance.run_job(root, store_checkpoint.NAME)
    before = _job(root)
    _backlog(root, key="more")
    monkeypatch.setattr(store_journal, "root_stores", lambda drive: (
        (real(drive)[0][0], busy, real(drive)[0][2]), *real(drive)[1:]))
    assert maintenance.run_job(root, store_checkpoint.NAME) == {"state": "pending", "items": 1}
    assert _job(root) == before and store_checkpoint.due(root, before, {}, False), "the held store is due again"
    monkeypatch.setattr(store_journal, "root_stores", real)
    assert maintenance.run_job(root, store_checkpoint.NAME)["state"] == "done"
    assert _job(root)["stores"]["current"]["wal"] != before["stores"]["current"]["wal"]
    assert not store_checkpoint.due(root, _job(root), {}, False)


@pytest.mark.parametrize("mode,tier", [("WAL", store.TIER_ENFORCED), ("PERSIST", store.TIER_ENFORCED),
                                       ("PERSIST", store.TIER_NAME)], ids=["wal", "persist", "name-tier"])
def test_the_dispatcher_s_look_opens_no_store_until_a_log_holds_the_threshold(tmp_path, monkeypatch, mode, tier):
    """The server's lane thread asks ``maintenance.due(root, lane="checkpoint")`` at every look: while no log holds
    SQLite's own threshold of committed frames it reads the logs' headers and nothing else (no SQLite connection, no
    name-tier lock), whatever the job's row says, so a quiet install, or one whose stores keep a rollback journal or
    sit on the name tier, opens no store for it. A log holding the threshold makes the look read the row and find
    the job due."""
    if tier == store.TIER_NAME:
        use_journal(monkeypatch, mode)
        data = tmp_path / "data"
        data.mkdir()
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
        publish(data, tier=tier)
    else:
        data = _root(tmp_path, monkeypatch, mode)
    store_checkpoint.inline_frames()  # read once per process from a memory database, never a store
    held = sqlite3.connect(store.store_path(data), isolation_level=None) if mode == "WAL" else None
    try:
        if held is not None:  # another open connection: no last close copies the log away
            held.execute("SELECT 1 FROM meta").fetchone()
        store.close_connections()  # no handle cached here: any store read must open one
        connects, locks = [], []
        real_connect, real_lock = sqlite3.connect, store.name_lock
        monkeypatch.setattr(sqlite3, "connect", lambda *args, **kw: connects.append(args[0]) or real_connect(*args, **kw))
        monkeypatch.setattr(store, "name_lock", lambda *args, **kw: locks.append(args[1]) or real_lock(*args, **kw))
        assert maintenance.due(data, lane="checkpoint") == [] and connects == [] and locks == [], (connects, locks)
        if held is not None:
            _backlog(data)
            store.close_connections()
            connects.clear()
            assert maintenance.due(data, lane="checkpoint") == [store_checkpoint.NAME] and connects, "the row is read"
    finally:
        if held is not None:
            held.close()


def test_each_look_notes_how_long_the_logs_were_for_the_minute_row_from_memory(root, monkeypatch):
    """The lane's looks (the dispatcher's, on its own thread) note each store kind's committed frames; the
    ``supervisor_load`` row of each busy minute drains the most one look saw (``wal_frames``), so a long COMMIT stands
    beside how long its store's log was, whether or not the job was due. The drain runs on the supervisor loop: it
    reads no file. A minute nobody looked says ``looks`` 0, never an empty log (the section's place in the row:
    tests/test_supervisor_load_row.py)."""
    store_checkpoint.drain_frames()
    assert store_checkpoint.drain_frames() == {"looks": 0, "max": {}}
    _backlog(root, 8, key="short")
    assert maintenance.due(root, lane="checkpoint") == []
    short = store.log_identity(store.store_path(root))[1]
    _backlog(root)
    assert maintenance.due(root, lane="checkpoint") == [store_checkpoint.NAME]
    long = store.log_identity(store.store_path(root))[1]
    opened = []
    with monkeypatch.context() as patch:
        for module, name in ((builtins, "open"), (io, "open"), (os, "open"), (os, "stat")):
            real = getattr(module, name)
            patch.setattr(module, name, lambda *args, real=real, **kw: opened.append(args[0]) or real(*args, **kw))
        window = store_checkpoint.drain_frames()
    assert opened == [] and short < 1000 <= long
    assert window["looks"] == 2 and window["max"]["current"] == long, window
    assert store_checkpoint.drain_frames() == {"looks": 0, "max": {}}


def test_the_real_checkpoint_child_copies_the_log_and_the_lane_rests_without_a_supervisor_row(tmp_path, monkeypatch):
    """The lane's real child (``python -m ouroboros.maintenance --lane checkpoint``, the custodied spawn) over a
    backlog: it copies the log and writes its pass row to the store-timing journal and its entry to the job row; the
    lane looks again, finds the job at rest and returns after one spawn, and its clean run leaves no ``maintenance``
    row in the supervisor journal. The child is another process of this interpreter, which no monkeypatch reaches: a
    SQLite that must not share WAL (CI's journal-persist leg) has it refuse a WAL store (``store._kept_journal``), so
    the root is built in the interpreter's own choice. There the stores keep a rollback journal and no log: the lane
    rests without a spawn, and nothing is recorded."""
    from tests.test_evidence_packs import REPO

    own = store._journal_mode(tmp_path, store.TIER_ENFORCED)  # the child's choice, read before any monkeypatch
    root = _root(tmp_path, monkeypatch, own)
    real_spawn, spawns = maintenance.spawn_maintenance_job, []
    monkeypatch.setattr(maintenance, "spawn_maintenance_job",
                        lambda *args, **kw: spawns.append(kw.get("args")) or real_spawn(*args, **kw))
    _backlog(root)
    (root / "logs").mkdir(exist_ok=True)
    seen = store.log_identity(store.store_path(root))
    unit = maintenance.MaintenanceChild("checkpoint")
    assert unit.tick(root, REPO)
    unit._thread.join(240)
    assert not unit._thread.is_alive() and unit.live_children() == []
    supervisor = root / "logs" / "supervisor.jsonl"
    assert not supervisor.exists() or '"maintenance"' not in supervisor.read_text(encoding="utf-8")
    if own != "WAL":
        assert seen is None and spawns == [] and _job(root) == {} and _passes(root) == []
        return
    seen = seen[:2]
    assert spawns == [("--lane", "checkpoint")], "one pass, then a look that found the job at rest"
    current = _job(root)["stores"]["current"]
    assert current["wal"] == seen and current["checkpointed"] == current["log"] == seen[1], current
    assert [row["stores"]["current"]["wal"] for row in _passes(root)] == [seen]
    assert not store_checkpoint.due(root, _job(root), {}, False)


def test_a_rollback_journal_store_has_no_log_and_the_job_leaves_it_alone(tmp_path, monkeypatch):
    """PERSIST (and the name tier) keep no WAL log: whatever is committed, the look is never due and a pass opens
    nothing and records nothing."""
    data = _root(tmp_path, monkeypatch, "PERSIST")
    _backlog(data)
    _backlog(data)
    assert not store.wal_file(store.store_path(data)) and not store_checkpoint.due(data, {}, {}, False)
    assert store_checkpoint.run(data, lambda: False) == {"state": "done", "items": 0}
    assert maintenance_jobs.job(data, store_checkpoint.NAME) is None
