"""Persistent SQLite connections: cost, lifetime, durability and old-body compatibility."""
import contextlib
import json
import multiprocessing
import os
import sqlite3
import subprocess
import sys
import threading
import time
import types

import pytest

from ouroboros import store, store_import, store_switch, usage_store
from tests.test_store import publish


@pytest.fixture(autouse=True)
def release_connections():
    store.close_connections()
    yield
    store.close_connections()


@pytest.mark.parametrize("money", [False, True])
@pytest.mark.parametrize("tier", [store.TIER_ENFORCED, store.TIER_NAME])
def test_fifty_reads_open_one_connection_per_thread(tmp_path, monkeypatch, money, tier):
    monkeypatch.setattr(store, "_journal_mode", lambda *_: "PERSIST")
    if money:
        from ouroboros import platform_layer
        monkeypatch.setattr(platform_layer, "kernel_file_locks_enforced", lambda _path: tier == store.TIER_ENFORCED)
        usage_store.migrate_from_journal(tmp_path)
    else:
        publish(tmp_path, tier=tier)
    store.close_connections()
    opened, statements, observed, failures = [], [], [], []
    real = sqlite3.connect

    def counted(*args, **kwargs):
        conn = real(*args, **kwargs)
        opened.append(conn)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(sqlite3, "connect", counted)
    def read():
        try:
            conns = []
            for _ in range(50):
                with usage_store.read(tmp_path) if money else store.open_store(tmp_path, write=False) as txn:
                    assert txn.meta("lock_tier") == tier
                    conns.append(txn.conn)
                assert not txn.conn.in_transaction
            assert all(conn is conns[0] for conn in conns)
            observed.append(conns[0])
        except BaseException as exc:
            failures.append(exc)

    read()
    worker = threading.Thread(target=read)
    worker.start()
    worker.join(10)
    assert not worker.is_alive() and failures == []
    assert len(opened) == 2 and observed[0] is not observed[1]
    assert statements.count("PRAGMA synchronous = FULL") == 2
    assert statements.count("PRAGMA journal_mode = PERSIST") == 2
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        observed[1].execute("SELECT 1")  # the thread's end disposes its connections
    assert observed[0].execute("SELECT 1").fetchone()[0] == 1


def test_a_thread_that_touches_many_stores_keeps_only_its_recent_handles_open(tmp_path):
    """A long-lived thread (a server worker, a test process) touches many stores over its life:
    child drives, other roots. It keeps the recently leased handles, the hot one included, and
    closes the least recently used idle ones instead of holding a descriptor per store forever."""
    hot = tmp_path / "hot"
    publish(hot)
    with store.open_store(hot, write=False) as txn:
        hot_conn = txn.conn
    seen = []
    for index in range(3 * store._THREAD_POOL_LIMIT):
        root = tmp_path / f"child-{index}"
        publish(root)
        with store.open_store(root, write=False) as txn:
            seen.append(txn.conn)
        with store.open_store(hot, write=False) as txn:  # the root store is read between them
            assert txn.conn is hot_conn
    pool = store._LOCAL_CONNECTIONS.pool
    assert len(pool.entries) <= store._THREAD_POOL_LIMIT
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        seen[0].execute("SELECT 1")  # the least recently used idle handle was closed
    assert seen[-1].execute("SELECT 1").fetchone()[0] == 1 and hot_conn.execute("SELECT 1").fetchone()[0] == 1


def test_connection_reuse_keeps_revision_barrier_and_rollback_semantics(tmp_path):
    publish(tmp_path)
    with store.open_store(tmp_path, write=True) as txn:
        txn.set_meta("value", 1)
        conn = txn.conn
    for barrier in (False, True):
        with store.open_store(tmp_path, write=True, barrier=barrier) as txn:
            assert txn.conn is conn and txn.revision() == 1
    with pytest.raises(ValueError):
        with store.open_store(tmp_path, write=True) as txn:
            txn.set_meta("value", 2)
            raise ValueError("rollback")
    with pytest.raises(store.StoreError, match="barrier transaction wrote"):
        with store.open_store(tmp_path, write=True, barrier=True) as txn:
            txn.set_meta("value", 3)
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.meta("value") == 1 and txn.revision() == 1
    assert not conn.in_transaction


def test_reused_connection_obeys_each_acquisition_budget(tmp_path):
    publish(tmp_path)
    for budget, expected in ((None, 10000), (0, 0), (0.125, 125), (None, 10000)):
        with store.acquisition_budget(budget) if budget is not None else contextlib.nullcontext():
            with store.open_store(tmp_path, write=False) as txn:
                assert txn.conn.execute("PRAGMA busy_timeout").fetchone()[0] == expected
    with store.admission_acquisition():
        with store.open_store(tmp_path, write=False) as txn:
            assert txn.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 0
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 10000


def test_idle_reader_sees_a_later_independent_commit(tmp_path):
    path = publish(tmp_path)
    with store.open_store(tmp_path, write=False) as txn:
        conn = txn.conn
        assert txn.meta("later") is None
    with contextlib.closing(sqlite3.connect(path, isolation_level=None, timeout=0)) as other:
        other.execute("INSERT INTO meta VALUES ('later', '42')")
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.conn is conn and txn.meta("later") == 42


def test_delete_and_persist_writers_interleave_with_full_durability(tmp_path, monkeypatch):
    """An older body (SQLite's default DELETE journal) and this one (PERSIST) write the same store in
    turn. Two bodies never run at once (the PID lock admits one), and within one install every process
    picks the same mode (a boot lock-tier re-record no longer leaves the server in DELETE mode: see
    test_store_import), so on Windows each turn of the older writer comes after this process let its
    handles go, as its exit does. There a PERSIST connection keeps its journal open for as long as it
    lives (the VFS reports UNDELETABLE_WHEN_OPEN and opens files without FILE_SHARE_DELETE), and the
    DELETE-mode commit, which deletes the journal, fails with a disk I/O error (SQLITE_IOERR_DELETE)
    while any such handle is open. POSIX closes that journal when the lock drops, so there this
    process's cached writer stays open across the foreign DELETE commits, which the test keeps covering."""
    monkeypatch.setattr(store, "_journal_mode", lambda *_: "PERSIST")
    path = publish(tmp_path)
    with contextlib.closing(sqlite3.connect(path, isolation_level=None)) as old:
        assert old.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
        for number in range(5):
            old.execute("INSERT OR REPLACE INTO meta VALUES ('number', ?)", (str(number),))
            with store.open_store(tmp_path, write=True) as txn:
                assert txn.meta("number") == number
                assert txn.conn.execute("PRAGMA journal_mode").fetchone()[0] == "persist"
                assert txn.conn.execute("PRAGMA synchronous").fetchone()[0] == 2
                assert txn.conn.execute("PRAGMA journal_size_limit").fetchone()[0] == store.JOURNAL_SIZE_LIMIT
                txn.set_meta("reply", number)
            if os.name == "nt":  # this body's turn ends as its exit ends it there: its handles let go
                store.close_connections(path)
            assert old.execute("SELECT value FROM meta WHERE key='reply'").fetchone()[0] == str(number)
        assert old.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


@pytest.mark.parametrize("wait_sec", [5.0, 0.0], ids=["budget", "no_budget"])
def test_an_explicit_wal_selection_waits_out_another_writer_within_its_budget(tmp_path, monkeypatch, wait_sec):
    """Only a lifecycle owner turns an existing rollback-journal file into WAL (``store_journal``), and
    that switch takes the write lock from inside a read transaction, where SQLite skips the busy handler:
    another connection writing at that moment would fail it at once with "database is locked". The
    selection waits out the writer within its own budget; with no budget it still refuses at once."""
    monkeypatch.setattr(store, "_journal_mode", lambda *_: "PERSIST")
    path = publish(tmp_path)
    store.close_connections()
    monkeypatch.setattr(store, "_journal_mode", lambda *_: "WAL")
    writer = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    writer.execute("BEGIN IMMEDIATE")  # another connection holds the write lock
    released = threading.Timer(0.3, lambda: writer.execute("ROLLBACK"))
    released.start()
    started = time.monotonic()
    try:
        if not wait_sec:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                store.connect(path, store.TIER_ENFORCED, wait_sec=wait_sec, journal_mode="WAL")
            assert time.monotonic() - started < 0.2, "no budget, no wait"
            return
        conn = store.connect(path, store.TIER_ENFORCED, wait_sec=wait_sec, journal_mode="WAL")
        try:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        finally:
            conn.close()
        assert time.monotonic() - started >= 0.25, "it waited for the writer instead of failing"
    finally:
        released.join(5)
        writer.close()


def _trace_wal_waits(monkeypatch, budget):
    """Observe the helper's clock and the timeout SQLite actually receives, without replacing waits."""
    trace = {"clock": [], "sleeps": [], "attempts": []}

    def clock():
        sample = time.monotonic()
        trace["clock"].append(sample)
        return sample

    def remaining():
        return trace["clock"][0] + budget - trace["clock"][-1]

    def sleep(seconds):
        before = time.monotonic()
        left = remaining()
        time.sleep(seconds)
        trace["sleeps"].append({"remaining": left, "granted": seconds,
                                "elapsed": time.monotonic() - before})

    class Connection(store._StoreConnection):
        def execute(self, sql, *args, **kwargs):
            if not sql.startswith("PRAGMA journal_mode = "):
                return super().execute(sql, *args, **kwargs)
            attempt = {"remaining": remaining(), "started": time.monotonic(),
                       "timeout_ms": super().execute("PRAGMA busy_timeout").fetchone()[0]}
            try:
                return super().execute(sql, *args, **kwargs)
            finally:
                attempt["elapsed"] = time.monotonic() - attempt["started"]
                trace["attempts"].append(attempt)

    monkeypatch.setattr(store, "time", types.SimpleNamespace(monotonic=clock, sleep=sleep))
    monkeypatch.setattr(store, "_StoreConnection", Connection)
    return trace


def test_an_explicit_wal_selection_never_waits_past_its_budget(tmp_path, monkeypatch):
    """The waits of an explicit WAL selection add up to its budget, not more: retries that a writer refuses at
    once, then an attempt that a remaining reader holds in SQLite's busy handler, which may wait only
    for what is left (the connection's own busy timeout would allow the whole budget again). The writer
    lets go halfway. Check waits granted against the helper's sampled deadline: under macOS CPU
    contention SQLite's busy handler can take much longer than the timeout it was given. Setup
    and native scheduling time are diagnostics, not additional waits granted by the helper."""
    monkeypatch.setattr(store, "_journal_mode", lambda *_: "PERSIST")
    path = publish(tmp_path)
    store.close_connections()
    monkeypatch.setattr(store, "_journal_mode", lambda *_: "WAL")
    reader = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    reader.execute("BEGIN")
    reader.execute("SELECT * FROM meta").fetchall()  # a reader that stays through the budget
    writer = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    writer.execute("BEGIN IMMEDIATE")  # a writer that lets go halfway through it
    released_at = []
    released = threading.Timer(2.0, lambda: (writer.execute("ROLLBACK"), released_at.append(time.monotonic())))
    released.start()
    budget = 4.0
    trace = _trace_wal_waits(monkeypatch, budget)
    started = time.monotonic()
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            store.connect(path, store.TIER_ENFORCED, wait_sec=budget, journal_mode="WAL")
        finished = time.monotonic()
        phases = json.dumps({"setup_sec": trace["clock"][0] - started,
                             "total_sec": finished - started, "released_at": released_at, **trace})
        # Only a writer that let go before the open ended put the reader's wait to the test.
        assert released_at and released_at[0] < finished, phases
        assert len(trace["attempts"]) > 1 and trace["sleeps"], phases
        assert all(0 < wait["granted"] <= wait["remaining"] for wait in trace["sleeps"]), phases
        # SQLite's timeout is an integer millisecond; a final sub-ms remainder may grant 1 ms.
        assert all(0 <= attempt["timeout_ms"] <= attempt["remaining"] * 1000 + 1
                   for attempt in trace["attempts"]), phases
        assert trace["attempts"][-1]["timeout_ms"] < budget * 1000, phases
    finally:
        released.join(10)
        writer.close()
        reader.execute("ROLLBACK")
        reader.close()


def test_leaving_wal_with_another_wal_connection_open_still_refuses_at_once(tmp_path, monkeypatch):
    """Only the switch INTO WAL waits. Leaving WAL needs every other WAL connection gone, and each keeps its
    shared lock for its life, so a lifecycle request for PERSIST on a WAL store (an older SQLite, an unknown
    filesystem) with another connection idle in it refuses at once, instead of waiting out its whole budget
    for the same refusal. The 1 s line sits far below the 4 s budget and far above a refusal at once."""
    monkeypatch.setattr(store, "_journal_mode", lambda *_: "PERSIST")
    path = publish(tmp_path)
    store.close_connections()
    idle = sqlite3.connect(path, isolation_level=None)
    try:
        assert idle.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"  # another body's WAL store
        idle.execute("SELECT * FROM meta").fetchall()  # and its connection, idle and open for its life
        started = time.monotonic()
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            store.connect(path, store.TIER_ENFORCED, wait_sec=4.0, journal_mode="PERSIST")
        assert time.monotonic() - started < 1.0, "leaving WAL refuses at once"
    finally:
        idle.close()


def test_handoff_closes_other_threads_idle_connections_before_rename(tmp_path):
    path = publish(tmp_path)
    ready, release = threading.Event(), threading.Event()
    held, failures = [], []
    def reader():
        try:
            with store.open_store(tmp_path, write=False) as txn:
                held.append(txn.conn)
            ready.set()
            assert release.wait(10)
        except BaseException as exc:
            failures.append(exc)
    worker = threading.Thread(target=reader)
    worker.start()
    try:
        assert ready.wait(5)
        store.prepare_handoff(path, store.TIER_ENFORCED)
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            held[0].execute("SELECT 1")
        assert store.held_descriptor(path) is None
        retired = path.with_name("retired.sqlite")
        os.replace(path, retired)
        with contextlib.closing(sqlite3.connect(retired)) as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive() and failures == []


def test_handoff_refuses_active_transaction_and_succeeds_after_release(tmp_path):
    path = publish(tmp_path)
    with store.open_store(tmp_path, write=True) as txn:
        txn.set_meta("kept", True)
        with pytest.raises(store.StoreBusy):
            store.prepare_handoff(path, store.TIER_ENFORCED)
        assert txn.meta("kept") is True and txn.conn.in_transaction
    store.prepare_handoff(path, store.TIER_ENFORCED)
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.meta("kept") is True


@pytest.mark.parametrize("spelling", ["real", "linked"])
def test_a_folder_about_to_move_releases_every_threads_handles_inside_it_and_keeps_the_rest(tmp_path, spelling):
    """Windows moves no folder while a file below it is open: every thread's idle handles on
    stores inside the folder close, an active lease inside it refuses, and a store outside it
    keeps its handle. A linked spelling of the folder names the same files."""
    drive, other = tmp_path / "drive", tmp_path / "drive2"  # a sibling sharing the name's prefix
    publish(drive)
    publish(other)
    leased = drive
    if spelling == "linked":
        leased = tmp_path / "linked"
        try:
            leased.symlink_to(drive, target_is_directory=True)
        except OSError as exc:  # Windows without the symlink privilege
            pytest.skip(f"no symlinks here: {exc}")
    ready, release = threading.Event(), threading.Event()
    held, failures = {}, []

    def reader():
        try:
            for name, root in (("drive", leased), ("other", other)):
                with store.open_store(root, write=False) as txn:
                    held[name] = txn.conn
            ready.set()
            assert release.wait(10)
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=reader)
    worker.start()
    try:
        assert ready.wait(5)
        with store.open_store(drive, write=True) as txn:
            with pytest.raises(store.StoreBusy):
                store.close_connections_under(drive)
            txn.set_meta("kept", True)
        store.close_connections_under(drive)
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            held["drive"].execute("SELECT 1")
        assert held["other"].execute("SELECT 1").fetchone()[0] == 1
        moved = tmp_path / "moved"
        os.replace(drive, moved)
        with store.open_store(moved, write=False) as txn:
            assert txn.meta("kept") is True
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive() and failures == []


@pytest.mark.parametrize("money", [False, True])
def test_export_closes_connections_and_leaves_delete_file(tmp_path, money):
    if money:
        usage_store.migrate_from_journal(tmp_path)
        with usage_store.read(tmp_path) as txn:
            conn = txn.conn
        result = usage_store.export_journal(tmp_path)
        path = tmp_path / "state" / result["store_retired_as"]
    else:
        publish(tmp_path)
        with store.open_store(tmp_path, write=True) as txn:
            conn = txn.conn
            txn.set_meta("phase", store.EXPORTING)
        store_import.export_facts(tmp_path)
        path = tmp_path / "state" / store_switch._retire(tmp_path)
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        conn.execute("SELECT 1")
    with contextlib.closing(sqlite3.connect(path)) as exported:
        assert exported.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert exported.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert not path.with_name(path.name + "-journal").exists()


def test_replaced_file_drops_the_old_cached_connection(tmp_path, monkeypatch):
    # Only a rollback-journal file is replaced under an idle handle: an idle WAL handle keeps a shared
    # lock that every handoff out of WAL waits for, and a WAL file renamed over leaves its log to the
    # next file of that name (sqlite.org/howtocorrupt.html, "mispairing database files and hot journals").
    monkeypatch.setattr(store, "_journal_mode", lambda *_: "PERSIST")
    path = publish(tmp_path)
    replacement = publish(tmp_path / "new")
    with store.open_store(tmp_path, write=True) as txn:
        old = txn.conn
        txn.set_meta("old", True)
    if os.name == "nt":
        store.close_connections(path)  # Windows cannot replace an open database
    os.replace(replacement, path)
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.conn is not old and txn.meta("old") is None
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        old.execute("SELECT 1")


@pytest.mark.serial
@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_closes_parent_cache_and_child_opens_its_own_connection(tmp_path):
    publish(tmp_path)
    with store.open_store(tmp_path, write=False) as txn:
        inherited = txn.conn
    ctx = multiprocessing.get_context("fork")
    receiving, sending = ctx.Pipe(duplex=False)
    def child():
        try:
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                inherited.execute("SELECT 1")
            with store.open_store(tmp_path, write=True) as txn:
                assert txn.conn is not inherited
                txn.set_meta("child", os.getpid())
            sending.send((os.getpid(), None))
        except BaseException as exc:
            sending.send((os.getpid(), repr(exc)))
        finally:
            sending.close()
    process = ctx.Process(target=child)
    process.start()
    sending.close()
    try:
        assert receiving.poll(10)
        pid, failure = receiving.recv()
        process.join(10)
        assert process.exitcode == 0 and failure is None and pid != os.getpid()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            inherited.execute("SELECT 1")
        with store.open_store(tmp_path, write=False) as txn:
            assert txn.conn is not inherited and txn.meta("child") == pid
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)
        receiving.close()


@pytest.mark.serial
def test_interpreter_exit_closes_both_stores_before_clearing_module_globals(tmp_path):
    publish(tmp_path)
    usage_store.migrate_from_journal(tmp_path)
    script = """
import atexit, pathlib, sqlite3, sys
connections = []
def verify_closed():
    for conn in connections:
        try:
            conn.execute('SELECT 1')
        except sqlite3.ProgrammingError:
            continue
        raise AssertionError('cached connection still open at shutdown')
    print('closed')
atexit.register(verify_closed)
from ouroboros import store, usage_store
root = pathlib.Path(sys.argv[1])
for lease in (store.open_store(root, write=True), usage_store.hold(root)):
    with lease as txn:
        txn.set_meta('exit_check', True)
        connections.append(txn.conn)
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0 and result.stdout.strip() == "closed", result.stderr
    assert "Exception ignored" not in result.stderr
    for lease in (store.open_store(tmp_path, write=False), usage_store.read(tmp_path)):
        with lease as txn:
            assert txn.meta("exit_check") is True


def _dialogue_root(root):
    from ouroboros import dialogue_store

    dialogue_store.append(root, "chat", {"ts": "2026-10-25T00:00:00Z", "chat_id": 1, "type": "message"})
    return dialogue_store


def _tiers_leased(monkeypatch) -> list:
    leased, real = [], store.connection
    monkeypatch.setattr(store, "connection", lambda path, tier, **kwargs: leased.append(tier) or real(path, tier, **kwargs))
    return leased


@pytest.mark.parametrize("kind", ["dialogue", "current"])
def test_a_warm_read_lease_states_its_file_outside_the_pool_guard_and_reads_no_header(tmp_path, monkeypatch, kind):
    """A warm read pays its stats before the process-wide pool guard, reads the header tier once per file
    identity, and issues the lease budget, BEGIN, the snapshot's SELECT 1, its query and COMMIT, nothing
    else: on a busy server every syscall and statement is a GIL hand-off."""
    if kind == "dialogue":
        dialogue_store = _dialogue_root(tmp_path)
        lease, query = (lambda: dialogue_store.transaction(tmp_path, write=False)), "SELECT count(*) FROM dialogue"
        tier = dialogue_store.header_tier(tmp_path)
        stats_wanted = [True] if tier == store.TIER_ENFORCED else [True, True]  # a name-tier lease states it behind its lock
    else:
        publish(tmp_path)
        lease, query = (lambda: store.open_store(tmp_path, write=False)), "SELECT count(*) FROM tasks"
        tier, stats_wanted = store.TIER_ENFORCED, [True, True]  # the probe's and the lease's
    with lease() as held:
        warm = held if kind == "dialogue" else held.conn
    stats, reads, statements = [], [], []
    real_identity, real_read = store.file_identity, store._read_header

    def guard_free() -> bool:
        free = []

        def attempt():
            if store._CONNECTION_GUARD.acquire(blocking=False):
                store._CONNECTION_GUARD.release()
                free.append(True)
        helper = threading.Thread(target=attempt)
        helper.start()
        helper.join(5)
        return bool(free)

    monkeypatch.setattr(store, "file_identity", lambda path: stats.append(guard_free()) or real_identity(path))
    monkeypatch.setattr(store, "_read_header", lambda *args, **kwargs: reads.append(args) or real_read(*args, **kwargs))
    warm.set_trace_callback(statements.append)
    try:
        with lease() as held:
            conn = held if kind == "dialogue" else held.conn
            assert conn is warm
            conn.execute(query).fetchone()
    finally:
        warm.set_trace_callback(None)
    assert stats == stats_wanted and reads == []
    budget = f"PRAGMA busy_timeout = {int(store.STORE_WAIT_SEC * 1000) if tier == store.TIER_ENFORCED else 0}"
    expected = [budget, "BEGIN", "SELECT 1 FROM meta LIMIT 1", query, "COMMIT"]
    assert [text for text in statements if not text.startswith("SELECT value FROM meta")] == expected


def test_a_read_takes_its_snapshot_when_its_lease_begins_not_at_its_first_query(tmp_path):
    """The read's ``SELECT 1`` is its snapshot moment: a commit another connection attempts after the lease
    began is refused under the rollback journal (the read's shared lock) or kept out of the read under WAL,
    so the read's first query answers the state at entry; the next read sees the commit. The other
    connection is this install's (``store.connect``: the file's journal, as every writer keeps it): a raw
    DELETE-mode one could not commit on Windows while this thread's cached PERSIST handle keeps the
    journal open (SQLITE_IOERR_DELETE; test_delete_and_persist_writers_interleave_with_full_durability)."""
    dialogue_store = _dialogue_root(tmp_path)
    if dialogue_store.header_tier(tmp_path) != store.TIER_ENFORCED:
        pytest.skip("kernel locks are what a foreign connection meets here")
    write = "INSERT OR REPLACE INTO meta (key, value) VALUES ('entry_probe', '1')"
    probe = "SELECT value FROM meta WHERE key='entry_probe'"
    outcome = []
    with dialogue_store.transaction(tmp_path, write=False) as conn:
        with contextlib.closing(store.connect(dialogue_store.path(tmp_path), store.TIER_ENFORCED)) as other:
            try:
                other.execute(write)
                outcome.append("committed")
            except sqlite3.OperationalError as exc:
                assert store.busy(exc), exc
                outcome.append("refused")
        assert conn.execute(probe).fetchone() is None, outcome
    if outcome == ["refused"]:
        with contextlib.closing(store.connect(dialogue_store.path(tmp_path), store.TIER_ENFORCED, wait_sec=5)) as other:
            other.execute(write)
    with dialogue_store.transaction(tmp_path, write=False) as conn:
        assert conn.execute(probe).fetchone()["value"] == "1"


@pytest.mark.parametrize("kind", ["dialogue", "current"])
def test_a_nested_lease_that_fails_rolls_back_only_itself(tmp_path, kind):
    """A nested access on the same thread runs on a temporary connection: its exception ends its own
    transaction and leaves the caller's open, which then commits what it wrote."""
    if kind == "dialogue":
        dialogue_store = _dialogue_root(tmp_path)
        outer_lease = lambda: dialogue_store.transaction(tmp_path, write=True)  # noqa: E731
        inner_lease = lambda: dialogue_store.transaction(tmp_path, write=False)  # noqa: E731
        conn_of = lambda held: held  # noqa: E731
    else:
        publish(tmp_path)
        outer_lease = lambda: store.open_store(tmp_path, write=True)  # noqa: E731
        inner_lease = lambda: store.open_store(tmp_path, write=False)  # noqa: E731
        conn_of = lambda held: held.conn  # noqa: E731
    with outer_lease() as outer:
        conn_of(outer).execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('outer', '1')")
        with pytest.raises(ValueError, match="inner"):
            with inner_lease() as inner:
                assert conn_of(inner) is not conn_of(outer)
                raise ValueError("inner")
        assert conn_of(outer).in_transaction
    with inner_lease() as held:
        assert conn_of(held).execute("SELECT value FROM meta WHERE key='outer'").fetchone()["value"] == "1"


@pytest.mark.parametrize("finish", ["fails_after_the_header", "commits"])
def test_a_retier_in_place_never_leaves_the_old_tier_kept_for_its_file(tmp_path, monkeypatch, finish):
    """A header changes under one file identity only by a retier in place (the boot's). It marks the file
    read-every-time before it writes and records the new tier once it committed: after a retier that failed
    past its header write the next leases follow the header, and after a committed one they use the new
    tier without reading the header again."""
    dialogue_store = _dialogue_root(tmp_path)
    with dialogue_store.transaction(tmp_path, write=False):
        pass  # the tier is kept for this file now
    before = dialogue_store.header_tier(tmp_path)
    after = store.TIER_NAME if before == store.TIER_ENFORCED else store.TIER_ENFORCED
    monkeypatch.setattr(dialogue_store, "probed_tier", lambda _root: after)
    if finish == "commits":
        assert dialogue_store.retier(tmp_path) == before
    else:
        def commit_fails(conn):
            raise sqlite3.OperationalError("disk I/O error")

        with monkeypatch.context() as patched:
            patched.setattr(store, "commit", commit_fails)  # the header is written before the meta row's commit
            with pytest.raises(store.StoreError, match="disk I/O error"):
                dialogue_store.retier(tmp_path)
    assert dialogue_store.header_tier(tmp_path) == after
    leased, reads, real_read = _tiers_leased(monkeypatch), [], store._read_header
    monkeypatch.setattr(store, "_read_header", lambda *args, **kwargs: reads.append(args) or real_read(*args, **kwargs))
    for _ in range(2):
        with dialogue_store.transaction(tmp_path, write=False) as conn:
            assert conn.execute("SELECT count(*) FROM dialogue").fetchone()[0] == 1
    assert leased == [after, after]
    assert len(reads) == (0 if finish == "commits" else 2)


def test_a_released_drive_moves_and_the_files_that_take_its_path_bring_their_own_tier(tmp_path, monkeypatch):
    """The tier kept per file holds no handle and answers for that file only: once the worker that read a
    drive released its idle handles (``_release_drive_stores``; Windows moves no folder with an open file
    below it), the drive moves, its files keep their tier there, and the stores that later take the old
    path are new files whose own headers name theirs."""
    from ouroboros import dialogue_store
    from supervisor.worker_process import _release_drive_stores

    drive, moved = tmp_path / "drive", tmp_path / "moved"
    publish(drive)
    dialogue_store.create(drive, tier=store.TIER_ENFORCED)
    with store.open_store(drive, write=False), dialogue_store.transaction(drive, write=False):
        pass  # both tiers are kept for these files now
    _release_drive_stores(drive)
    os.replace(drive, moved)
    publish(drive, tier=store.TIER_NAME)
    dialogue_store.create(drive, tier=store.TIER_NAME)
    leased = _tiers_leased(monkeypatch)
    for root in (drive, moved):
        with store.open_store(root, write=False) as txn:
            assert txn.meta("lock_tier") == leased[-1]
        with dialogue_store.transaction(root, write=False) as conn:
            assert conn.execute("SELECT count(*) FROM dialogue").fetchone()[0] == 0
    assert leased == [store.TIER_NAME, store.TIER_NAME, store.TIER_ENFORCED, store.TIER_ENFORCED]


@pytest.mark.skipif(os.name == "nt", reason="Windows replaces no file a cached connection holds open")
def test_a_name_tier_lease_states_its_file_after_the_lock_it_waited_for(tmp_path, monkeypatch):
    """A name-tier read waits for the name lock; its holder may replace the store meanwhile (a lifecycle
    owner publishing a new file by rename). The lease states the file once it holds the lock, so it reads
    the file that is there now and not the one its thread's idle cached connection still has open."""
    from ouroboros import dialogue_store

    old, new = tmp_path / "root", tmp_path / "staged"
    for root, text in ((old, "old"), (new, "new")):
        dialogue_store.create(root, tier=store.TIER_NAME)
        with dialogue_store.transaction(root, write=True) as conn:
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('which', ?)", (json.dumps(text),))
    warmed, holding, waiting = threading.Event(), threading.Event(), threading.Event()
    seen, failures, real_lock = [], [], store.name_lock

    def name_lock(*args, **kwargs):  # the reader asks for the lock only after its lease's first stat
        if threading.current_thread() is worker and holding.is_set():
            waiting.set()
        return real_lock(*args, **kwargs)

    monkeypatch.setattr(store, "name_lock", name_lock)

    def reader():
        try:
            with dialogue_store.transaction(old, write=False) as conn:  # warm: an idle cached connection stays
                seen.append(json.loads(conn.execute("SELECT value FROM meta WHERE key='which'").fetchone()["value"]))
            warmed.set()
            assert holding.wait(10)
            with dialogue_store.transaction(old, write=False) as conn:  # waits for the lock held below
                seen.append(json.loads(conn.execute("SELECT value FROM meta WHERE key='which'").fetchone()["value"]))
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=reader)
    worker.start()
    assert warmed.wait(10)
    with real_lock(old, dialogue_store.LOCK_NAME, timeout_sec=10):
        holding.set()
        assert waiting.wait(10)  # the reader stated its file and now waits for this lock
        os.replace(dialogue_store.path(new), dialogue_store.path(old))
    worker.join(30)
    assert not worker.is_alive() and failures == [] and seen == ["old", "new"]
