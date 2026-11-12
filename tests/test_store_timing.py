"""Store transactions report their lock-hold seconds per store kind, in memory; the hook never raises."""
from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_store, store, store_timing, usage_store
from ouroboros.task_results import load_task_result, write_task_result


@pytest.fixture(autouse=True)
def clean_window():
    store.close_connections()
    store_timing.drain()
    store_timing.drain_sites()
    yield
    store_timing.drain()
    store_timing.drain_sites()
    store.close_connections()


def test_the_window_counts_holds_per_kind_and_side_and_drain_resets_it(tmp_path):
    write_task_result(tmp_path, "t", "running")
    assert load_task_result(tmp_path, "t")["status"] == "running"
    with dialogue_store.transaction(tmp_path, write=True):
        pass
    with usage_store.hold(tmp_path, write=True):
        pass
    with pytest.raises(RuntimeError, match="refused"):  # a rolled-back hold counts too
        with store.open_store(tmp_path, write=True):
            raise RuntimeError("refused")
    window = store_timing.drain()
    assert {"current", "dialog", "usage"} <= set(window)
    assert window["current"]["write"]["count"] >= 2 and window["current"]["read"]["count"] >= 1
    assert window["dialog"]["write"]["count"] >= 1 and window["usage"]["write"]["count"] >= 1
    for sides in window.values():
        for bucket in sides.values():
            assert bucket["count"] >= 1 and 0 <= bucket["max_sec"] <= bucket["total_sec"] and bucket["max_site"]
            assert bucket["max_site"].split(":")[0] not in {"ouroboros.store", "ouroboros.task_store", "contextlib"}
    assert store_timing.drain() == {}
    with store.open_store(tmp_path, write=True):
        time.sleep(0.07)  # the window's longest hold names the frame that opened it (a Windows tick's margin)
    longest = store_timing.drain()["current"]["write"]
    assert longest["max_sec"] >= 0.05 and longest["max_site"].startswith("tests.test_store_timing:test_the_window")
    assert "sites" not in longest and "sites_omitted" not in longest, "the minute's row keeps its size: the table is its own"


def test_a_failure_inside_the_hook_never_reaches_the_transaction(tmp_path, monkeypatch):
    def broken():
        raise RuntimeError("site walk failed")

    with monkeypatch.context() as patch:
        patch.setattr(store_timing, "_site", broken)
        write_task_result(tmp_path, "t", "running")  # the commit stands
        with pytest.raises(RuntimeError, match="refused"):  # so does the caller's own exception
            with store.open_store(tmp_path, write=True):
                raise RuntimeError("refused")
    assert load_task_result(tmp_path, "t")["status"] == "running"
    window = store_timing.drain()
    assert window["current"]["write"]["count"] >= 2 and window["current"]["write"]["max_site"] == ""
    write_task_result(tmp_path, "t", "completed")  # the healthy hook records the site again
    assert store_timing.drain()["current"]["write"]["max_site"]


def test_a_clock_too_coarse_to_see_a_short_hold_still_names_its_site(tmp_path, monkeypatch):
    """Windows' ``time.monotonic`` ticks at ~15.6 ms: a hold read by such a clock measures 0.0 and must still name its
    site, as the window's longest hold so far."""
    monkeypatch.setattr(store_timing, "clock", lambda: 1000.0)  # every hold measures 0.0
    write_task_result(tmp_path, "t", "running")
    with dialogue_store.transaction(tmp_path, write=True):
        pass
    window = store_timing.drain()
    for kind in ("current", "dialog"):
        bucket = window[kind]["write"]
        assert bucket["count"] >= 1 and bucket["max_sec"] == 0.0 and bucket["max_site"], (kind, bucket)


def test_a_metered_scope_counts_its_own_transactions_and_chronicle_opens_and_nothing_else(tmp_path):
    """The receipt's counts are one scope's: per store kind and side with summed and longest holds,
    the chronicle's locked opens by name; a transaction on another thread, or after the scope, is
    not in them, and two concurrent scopes keep their own."""
    import concurrent.futures
    import threading

    from ouroboros.chronicle_store import ChronicleStore

    write_task_result(tmp_path, "t", "running")
    with dialogue_store.transaction(tmp_path, write=True):
        pass
    with store_timing.metered() as meter:
        load_task_result(tmp_path, "t")
        for _ in range(2):
            with dialogue_store.transaction(tmp_path, write=False):
                pass
        ChronicleStore(tmp_path).room_head("1")
        other = threading.Thread(target=lambda: load_task_result(tmp_path, "t"))
        other.start()
        other.join()
        counts = store_timing.snapshot(meter)
    load_task_result(tmp_path, "t")  # after the scope: not counted
    assert {key: value for key, value in counts.items() if not key.endswith("_sec") and not key.endswith(".sec")} == {
        "current.read": 1, "dialog.read": 2, "chronicle.locked": 1}
    assert 0.0 <= counts["dialog.read.max_sec"] <= counts["dialog.read.sec"]
    assert store_timing.snapshot(meter) == counts and store_timing.snapshot(None) == {}
    assert store_timing.count("outside") is None  # no scope here: nothing to count, nothing raised

    def scoped(reads: int) -> dict:
        with store_timing.metered() as own:
            for _ in range(reads):
                load_task_result(tmp_path, "t")
            return store_timing.snapshot(own)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        one, three = pool.submit(scoped, 1), pool.submit(scoped, 3)
        assert (one.result()["current.read"], three.result()["current.read"]) == (1, 3)


def test_a_commit_that_does_not_complete_is_recorded_failed_with_its_site_and_cpu(tmp_path, monkeypatch):
    """Rollback journal (the gate's answer for 3.50.4 here, whatever SQLite runs the suite): an open read's
    SHARED lock fails a zero-wait COMMIT; that hold is in the window as failed, not missing, and its error
    names the end it met."""
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 50, 4))
    write_task_result(tmp_path, "t", "running")
    store_timing.drain()
    store_timing.drain_sites()
    with store.open_store(tmp_path, write=False) as reader:
        reader.row("t", "status")
        with pytest.raises(store.StoreBusy) as raised:
            with store.acquisition_budget(0), store.open_store(tmp_path, write=True) as writer:
                writer.conn.execute("UPDATE tasks SET status='completed' WHERE task_id='t'")
    assert " [commit: elapsed " in str(raised.value) and str(raised.value).endswith(", busy_timeout 0 s]")
    assert raised.value.__cause__.store_phase == "commit"
    window = store_timing.drain()
    write, read = window["current"]["write"], window["current"]["read"]
    assert (write["count"], write["failed"], write["refused"], read["failed"]) == (1, 1, 0, 0)
    site = store_timing.drain_sites()["current"]["write"][0]
    assert site["site"].startswith("tests.test_store_timing:test_a_commit") and site["failed"] == 1
    for bucket in (write, read):
        assert 0.0 <= bucket["cpu_sec"] and bucket["max_sec"] <= bucket["total_sec"]
    assert load_task_result(tmp_path, "t")["status"] == "running"


def _short_read(root):
    with store.open_store(root, write=False) as txn:
        txn.row("t", "status")


def _long_read(root):
    with store.open_store(root, write=False):
        time.sleep(0.03)


def _third_read(root):
    with store.open_store(root, write=False) as txn:
        txn.row("t", "status")


def _fourth_read(root):
    with store.open_store(root, write=False) as txn:
        txn.row("t", "status")


def test_the_table_holds_every_site_most_summed_hold_first(tmp_path):
    """No cut: the table is bounded by the sites the code has (40 to 48 of 43 to 51 were hidden behind a top three)."""
    write_task_result(tmp_path, "t", "running")
    store_timing.drain_sites()
    for reader in (_short_read, _short_read, _short_read, _long_read, _third_read, _fourth_read):
        reader(tmp_path)
    sites = store_timing.drain_sites()["current"]["read"]
    assert len(sites) == 4 and sites[0]["site"].startswith("tests.test_store_timing:_long_read"), sites
    totals = [site["total_sec"] for site in sites]
    assert totals == sorted(totals, reverse=True) and sum(site["count"] for site in sites) == 6
    assert store_timing.drain_sites() == {}, "drained means empty"


def test_the_two_windows_are_drained_apart(tmp_path):
    """The minute's row empties its window and leaves the table's, which has its own cadence, whole."""
    write_task_result(tmp_path, "t", "running")
    store_timing.drain()
    store_timing.drain_sites()
    _short_read(tmp_path)
    assert store_timing.drain()["current"]["read"]["count"] == 1
    _short_read(tmp_path)
    assert store_timing.drain()["current"]["read"]["count"] == 1
    [site] = store_timing.drain_sites()["current"]["read"]
    assert site["count"] == 2, site


@pytest.fixture
def publish(monkeypatch):
    """One busy minute's publication of this test's store work, with the table's cadence as the test says."""
    from ouroboros import server_liveness as liveness
    from supervisor import loop_load

    monkeypatch.setattr(loop_load, "CONSUMER", loop_load.ConsumerWindow())
    monkeypatch.setattr(liveness.BOOT, "_load", None)
    monkeypatch.setattr(liveness, "SUPERVISOR_LOAD_WINDOW_SEC", 0.0)
    monkeypatch.setattr(liveness, "_load_sections", lambda: {"store_locks": store_timing.drain})
    monkeypatch.setattr(liveness, "_STORE_SITES_STARTED", [time.monotonic()])

    def once(root, *, table_due: bool):
        monkeypatch.setattr(liveness, "STORE_SITES_WINDOW_SEC", 0.0 if table_due else 3600.0)
        loop_load.CONSUMER.note("task_heartbeat", "telemetry", 0.001, None)
        return liveness.publish_supervisor_load(SimpleNamespace(), root)

    return once


def _journal(root, name, row_type):
    path = root / "logs" / name
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return [row for row in rows if row.get("type") == row_type]


def test_the_minute_s_row_keeps_its_size_and_the_table_lands_in_its_own_journal(tmp_path, publish):
    """A table of 36 to 42 KB (measured on the stand) in the journal whose 256 KB tail every context reads would leave
    that tail six minutes; a row a minute of it would write 58 MB a day. The row keeps what a reader of the minute needs."""
    from ouroboros.tools.query_data import _FILES

    write_task_result(tmp_path, "t", "running")
    store_timing.drain()
    store_timing.drain_sites()
    _short_read(tmp_path)
    alone = publish(tmp_path, table_due=False)
    assert not _journal(tmp_path, "store_sites.jsonl", "store_sites"), "the table waits for its own cadence"
    for reader in (_short_read, _long_read, _third_read, _fourth_read):
        reader(tmp_path)
    many = publish(tmp_path, table_due=True)
    assert _journal(tmp_path, "supervisor.jsonl", "supervisor_load") == [alone, many]
    for row in (alone, many):
        assert set(row["store_locks"]["current"]["read"]) == {
            "count", "failed", "total_sec", "cpu_sec", "max_sec", "commit_sec", "acquire_sec", "max_acquire_sec", "refused",
            "max_commit_sec", "max_site"}, row
    [table] = _journal(tmp_path, "store_sites.jsonl", "store_sites")
    sites = table["store_locks"]["current"]["read"]
    assert table["window_sec"] > 0 and len(sites) == 4 and sum(site["count"] for site in sites) == 5, table
    assert all(site["chains"] for site in sites), "both minutes' work, each site with who called it"
    assert _FILES["store_sites_log"] == "logs/store_sites.jsonl", "the mind reads it as it reads the other journals"


def test_a_table_with_nothing_in_it_writes_no_row(tmp_path, publish):
    store_timing.drain_sites()
    row = publish(tmp_path, table_due=True)
    assert row["store_locks"] == {} and not _journal(tmp_path, "store_sites.jsonl", "store_sites")


def test_a_refused_table_never_costs_the_minute_s_row(tmp_path, publish, monkeypatch):
    from ouroboros import server_liveness as liveness

    write_task_result(tmp_path, "t", "running")
    real = liveness.append_jsonl

    def refuse_the_table(path, row, **kwargs):
        if row["type"] == "store_sites":
            raise OSError("journal unavailable")
        return real(path, row, **kwargs)

    monkeypatch.setattr(liveness, "append_jsonl", refuse_the_table)
    row = publish(tmp_path, table_due=True)
    assert row and _journal(tmp_path, "supervisor.jsonl", "supervisor_load") == [row]
    assert not _journal(tmp_path, "store_sites.jsonl", "store_sites")


def test_the_table_s_journal_rotates_into_the_archive_like_the_others(tmp_path, publish):
    write_task_result(tmp_path, "t", "running")
    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / "logs" / "store_sites.jsonl").write_text(json.dumps({"ts": "old"}) + "\n" + "{}\n" * 400_000, encoding="utf-8")
    assert publish(tmp_path, table_due=True)
    assert [path.name.partition("_")[0] for path in (tmp_path / "archive").glob("store_sites_*.jsonl")] == ["store"]
    assert (tmp_path / "logs" / "store_sites.jsonl").stat().st_size < 800_000


def _the_census(root):
    _short_read(root)


def _the_request(root):
    _short_read(root)


@pytest.fixture
def named_callers(monkeypatch):
    """This module's frames count as first-party callers (production names its own packages)."""
    monkeypatch.setattr(store_timing, "_OWN_PACKAGES", store_timing._OWN_PACKAGES | {"tests"})


def test_a_site_says_who_called_it_by_call_chain(tmp_path, named_callers):
    """One site reached by two callers on two kinds of thread: ``_short_read``'s row says how much each spent, and two
    instances of one thread class are one chain."""
    import threading

    write_task_result(tmp_path, "t", "running")
    store_timing.drain_sites()

    def on(name, call, times):
        for instance in range(times):
            thread = threading.Thread(target=call, args=(tmp_path,), name=f"{name}_{instance}")
            thread.start()
            thread.join()

    on("owner_pause_settlement", _the_census, 3)
    on("asyncio", _the_request, 2)
    [site] = [row for row in store_timing.drain_sites()["current"]["read"] if ":_short_read:" in row["site"]]
    chains = {row["chain"]: row["count"] for row in site["chains"]}
    assert chains == {"owner_pause_settlement: test_store_timing:_the_census": 3,
                      "asyncio: test_store_timing:_the_request": 2}, site
    assert site["count"] == sum(chains.values())
    assert sum(row["total_sec"] for row in site["chains"]) == pytest.approx(site["total_sec"], abs=1e-3)


def _deep_caller(root):
    _the_census(root)


def _deeper_caller(root):
    with contextlib.ExitStack():  # a stdlib call between two first-party callers names nobody
        _deep_caller(root)


def test_a_call_chain_names_the_nearest_first_party_callers_only(tmp_path, named_callers):
    """The tag stays short at any depth, its caller frames skip whatever is not first-party, and a thread's name
    without first-party callers still tags the hold."""
    write_task_result(tmp_path, "t", "running")
    store_timing.drain_sites()
    _deeper_caller(tmp_path)
    [site] = [row for row in store_timing.drain_sites()["current"]["read"] if ":_short_read:" in row["site"]]
    [chain] = site["chains"]
    assert chain["chain"] == "MainThread: test_store_timing:_the_census < test_store_timing:_deep_caller", site
    assert ":_deeper_caller" not in chain["chain"] and store_timing.CHAIN_CALLERS == 2


@pytest.mark.parametrize("thread_name, thread_class", [
    ("asyncio_12", "asyncio"), ("Thread-123 (_run_direct)", "Thread (_run_direct)"), ("wake-turn-8f3a", "wake-turn")])
def test_a_thread_class_drops_instance_tokens_and_keeps_a_duty_s_name(thread_name, thread_class):
    assert store_timing._INSTANCE_TOKEN.sub("", thread_name) == thread_class


_PHASES = ("count", "refused", "acquire_sec", "max_acquire_sec", "total_sec", "commit_sec", "max_sec", "max_commit_sec")


class _Timed:
    """A connection whose statements cost what the test says on the hold clock and fail where it says: the phases
    measured exactly, with no second connection and no sleep."""
    _store_kind = "current"

    def __init__(self, now, *, cost=None, fail=None, busy_timeout_ms=5000):
        self.now, self.cost, self.fail, self.busy_timeout_ms = now, cost or {}, fail or {}, busy_timeout_ms
        self.in_transaction, self.statements = False, []

    def execute(self, sql):
        verb = sql.split()[0]
        self.statements.append(sql)
        self.now[0] += self.cost.get(verb, 0.0)
        if verb in self.fail:
            raise self.fail[verb]
        if verb == "BEGIN":
            self.in_transaction = True
        elif verb in {"COMMIT", "ROLLBACK"}:
            self.in_transaction = False
        return self

    def fetchone(self):
        return (self.busy_timeout_ms,) if self.statements[-1] == "PRAGMA busy_timeout" else (1,)


@pytest.fixture
def hold_clock(monkeypatch):
    """The hold clock in the test's hands (the holding thread's CPU at zero)."""
    now = [100.0]
    monkeypatch.setattr(store_timing, "clock", lambda: now[0])
    monkeypatch.setattr(store_timing, "cpu_clock", lambda: 0.0)
    return now


def _timed_hold(conn, now, body=0.0):
    with store.transaction(conn, write=True):
        now[0] += body


def test_a_hold_splits_into_its_wait_its_body_and_its_end_on_one_clock(hold_clock):
    """The wait for the lock is not the hold; the end (COMMIT) is the hold's part ``commit_sec``; the longest hold
    names its own end, not the window's longest end. No statement is added on the way."""
    first = _Timed(hold_clock, cost={"BEGIN": 0.5, "COMMIT": 0.25})
    _timed_hold(first, hold_clock, body=2.0)
    second = _Timed(hold_clock, cost={"COMMIT": 1.0})  # a shorter hold with a longer end
    _timed_hold(second, hold_clock)
    assert first.statements == second.statements == ["BEGIN IMMEDIATE", "COMMIT"]
    write = store_timing.drain()["current"]["write"]
    assert {key: write[key] for key in _PHASES} == {
        "count": 2, "refused": 0, "acquire_sec": 0.5, "max_acquire_sec": 0.5, "total_sec": 3.25, "commit_sec": 1.25,
        "max_sec": 2.25, "max_commit_sec": 0.25}
    assert write["max_site"].startswith("tests.test_store_timing:_timed_hold:")
    [site] = store_timing.drain_sites()["current"]["write"]
    assert {key: site[key] for key in _PHASES[:-1]} == {
        "count": 2, "refused": 0, "acquire_sec": 0.5, "max_acquire_sec": 0.5, "total_sec": 3.25, "commit_sec": 1.25,
        "max_sec": 2.25}


def test_a_lock_refused_busy_is_a_counted_refusal_with_its_phase_wait_and_budget(hold_clock):
    """No hold, no failed hold: a refusal, its wait counted where it was asked for; the error says the phase, the
    wait and the ``busy_timeout`` read on the refusal only; the error itself and its rollback fact stand."""
    locked = sqlite3.OperationalError("database is locked")
    conn = _Timed(hold_clock, cost={"BEGIN": 0.5}, fail={"BEGIN": locked}, busy_timeout_ms=500)
    with store_timing.metered() as meter, pytest.raises(sqlite3.OperationalError) as raised:
        _timed_hold(conn, hold_clock)
    assert raised.value is locked and locked.rollback_confirmed is True
    assert conn.statements == ["BEGIN IMMEDIATE", "PRAGMA busy_timeout"]
    assert (locked.store_phase, locked.store_elapsed_sec, locked.store_busy_timeout_sec) == ("acquire", 0.5, 0.5)
    note = " [acquire: elapsed 0.500 s, busy_timeout 0.5 s]"
    assert str(store._typed(locked)) == "current store busy: database is locked" + note
    assert str(dialogue_store._typed(locked)) == "dialogue store busy: database is locked" + note
    assert str(usage_store._typed(locked)) == "usage store busy: database is locked" + note
    write = store_timing.drain()["current"]["write"]
    assert {key: write[key] for key in _PHASES} == {
        "count": 0, "refused": 1, "acquire_sec": 0.5, "max_acquire_sec": 0.5, "total_sec": 0.0, "commit_sec": 0.0,
        "max_sec": 0.0, "max_commit_sec": 0.0}
    assert (write["failed"], write["max_site"]) == (0, "")
    [site] = store_timing.drain_sites()["current"]["write"]
    assert (site["site"].split(":")[1], site["count"], site["refused"], site["acquire_sec"], site["chains"]) == (
        "_timed_hold", 0, 1, 0.5, [])
    assert store_timing.snapshot(meter) == {"current.write.acquire_sec": 0.5}, "no hold in a turn's counts"


def test_a_commit_refused_busy_names_the_end_and_its_wait(hold_clock):
    locked = sqlite3.OperationalError("database is locked")
    conn = _Timed(hold_clock, cost={"COMMIT": 0.25}, fail={"COMMIT": locked}, busy_timeout_ms=0)
    with pytest.raises(sqlite3.OperationalError) as raised:
        _timed_hold(conn, hold_clock, body=1.0)
    assert raised.value is locked and locked.rollback_confirmed is True
    assert conn.statements == ["BEGIN IMMEDIATE", "COMMIT", "ROLLBACK", "PRAGMA busy_timeout"]
    assert (locked.store_phase, locked.store_elapsed_sec, locked.store_busy_timeout_sec) == ("commit", 0.25, 0.0)
    assert str(store._typed(locked)).endswith(" [commit: elapsed 0.250 s, busy_timeout 0 s]")
    write = store_timing.drain()["current"]["write"]
    assert {key: write[key] for key in ("count", "failed", "refused", "total_sec", "commit_sec", "max_commit_sec")} == {
        "count": 1, "failed": 1, "refused": 0, "total_sec": 1.25, "commit_sec": 0.25, "max_commit_sec": 0.25}


def test_a_busy_statement_in_the_body_names_the_body_and_leaves_wait_and_budget_unknown(hold_clock):
    """Which connection a statement inside ran on, the transaction cannot tell: no wait or budget is claimed for it."""
    locked = sqlite3.OperationalError("database is locked")
    conn = _Timed(hold_clock, cost={"ROLLBACK": 0.125})
    with pytest.raises(sqlite3.OperationalError) as raised:
        with store.transaction(conn, write=True):
            hold_clock[0] += 1.0
            raise locked
    assert raised.value is locked and conn.statements == ["BEGIN IMMEDIATE", "ROLLBACK"]
    assert (locked.store_phase, locked.store_elapsed_sec, locked.store_busy_timeout_sec) == ("body", None, None)
    assert str(store._typed(locked)).endswith(" [body: elapsed unknown, busy_timeout unknown]")
    write = store_timing.drain()["current"]["write"]
    assert (write["count"], write["failed"], write["total_sec"], write["commit_sec"]) == (1, 1, 1.125, 0.125)


def test_an_inner_transaction_s_marks_stand_and_the_outer_hold_still_counts(hold_clock):
    locked = sqlite3.OperationalError("database is locked")
    outer = _Timed(hold_clock)
    inner = _Timed(hold_clock, cost={"BEGIN": 0.5}, fail={"BEGIN": locked}, busy_timeout_ms=200)
    with pytest.raises(sqlite3.OperationalError):
        with store.transaction(outer, write=True), store.transaction(inner, write=True):
            pass
    assert (locked.store_phase, locked.store_elapsed_sec, locked.store_busy_timeout_sec) == ("acquire", 0.5, 0.2)
    assert outer.statements == ["BEGIN IMMEDIATE", "ROLLBACK"]
    write = store_timing.drain()["current"]["write"]
    assert (write["count"], write["failed"], write["refused"]) == (1, 1, 1)


def test_an_acquisition_error_that_is_not_contention_counts_and_marks_nothing(hold_clock):
    missing = sqlite3.OperationalError("no such table: meta")
    conn = _Timed(hold_clock, fail={"SELECT": missing})
    with pytest.raises(sqlite3.OperationalError):
        with store.transaction(conn, write=False):
            pass
    assert not hasattr(missing, "store_phase") and conn.statements == ["BEGIN", "SELECT 1 FROM meta LIMIT 1", "ROLLBACK"]
    assert type(store._typed(missing)) is store.StoreError
    assert store_timing.drain() == {} and store_timing.drain_sites() == {}
    unmarked = sqlite3.OperationalError("database is locked")  # met outside a transaction: its wording as it was
    assert str(store._typed(unmarked)) == "current store busy: database is locked"


def test_an_unreadable_budget_is_unknown_and_the_refusal_stands(hold_clock):
    locked = sqlite3.OperationalError("database is locked")
    closed = sqlite3.ProgrammingError("Cannot operate on a closed database.")
    conn = _Timed(hold_clock, fail={"BEGIN": locked, "PRAGMA": closed})
    with pytest.raises(sqlite3.OperationalError) as raised:
        _timed_hold(conn, hold_clock)
    assert raised.value is locked and (locked.store_phase, locked.store_busy_timeout_sec) == ("acquire", None)
    assert str(store._typed(locked)).endswith(" [acquire: elapsed 0.000 s, busy_timeout unknown]")
    assert store_timing.drain()["current"]["write"]["refused"] == 1


def _held_write(root, held, release):
    with store.open_store(root, write=True):
        held.set()
        release.wait(5)


@pytest.fixture
def enforced(monkeypatch):
    """SQLite's own locks decide (the name tier's file lock would refuse a second writer first)."""
    from ouroboros import platform_layer

    monkeypatch.setattr(platform_layer, "kernel_file_locks_enforced", lambda _path, **_known: True)


def test_a_real_writer_refused_at_a_held_lock_says_so_and_both_published_rows_count_it(tmp_path, enforced, publish):
    """Another thread holds the write lock: a zero budget is refused at once, a 0.2 s one after its wait; neither is a
    hold. The typed error says the phase, the wait and the budget; the minute's row and the site table carry the
    refusals at the refusing site, the turn's meter their wait only."""
    import re
    import threading

    write_task_result(tmp_path, "t", "running")
    store_timing.drain()
    store_timing.drain_sites()
    held, release, errors = threading.Event(), threading.Event(), []
    holder = threading.Thread(target=_held_write, args=(tmp_path, held, release))
    holder.start()
    try:
        assert held.wait(5)
        with store_timing.metered() as meter:
            for budget in (0, 0.2):
                with pytest.raises(store.StoreBusy) as raised, store.acquisition_budget(budget):
                    with store.open_store(tmp_path, write=True):
                        pytest.fail("a held write lock admits no second writer")
                errors.append(raised.value)
    finally:
        release.set()
        holder.join(5)
    for error, budget in zip(errors, ("0", "0.2")):
        assert re.fullmatch(rf"current store busy: database is locked \[acquire: elapsed \d+\.\d{{3}} s, "
                            rf"busy_timeout {re.escape(budget)} s\]", str(error)), str(error)
        assert error.__cause__.store_phase == "acquire" and error.rollback_confirmed is True
    assert errors[1].__cause__.store_elapsed_sec >= 0.15 > errors[0].__cause__.store_elapsed_sec
    assert set(store_timing.snapshot(meter)) == {"current.write.acquire_sec"}
    row = publish(tmp_path, table_due=True)
    write = row["store_locks"]["current"]["write"]
    assert (write["count"], write["failed"], write["refused"]) == (1, 0, 2) and write["max_acquire_sec"] >= 0.15
    assert write["max_site"].startswith("tests.test_store_timing:_held_write:")
    [table] = _journal(tmp_path, "store_sites.jsonl", "store_sites")
    sites = {site["site"].rsplit(":", 1)[0]: site for site in table["store_locks"]["current"]["write"]}
    refusing = sites["tests.test_store_timing:" + test_a_real_writer_refused_at_a_held_lock_says_so_and_both_published_rows_count_it.__name__]
    assert (refusing["count"], refusing["refused"], refusing["chains"]) == (0, 2, []) and refusing["acquire_sec"] >= 0.15
    assert sites["tests.test_store_timing:_held_write"]["count"] == 1


def test_a_real_wait_that_ends_in_the_lock_is_measured_apart_from_the_hold(tmp_path, enforced):
    import threading

    write_task_result(tmp_path, "t", "running")
    held, release = threading.Event(), threading.Event()
    holder = threading.Thread(target=_held_write, args=(tmp_path, held, release))
    holder.start()
    try:
        assert held.wait(5)
        store_timing.drain()
        store_timing.drain_sites()
        threading.Timer(0.2, release.set).start()
        with store.acquisition_budget(5), store.open_store(tmp_path, write=True):
            pass
    finally:
        release.set()
        holder.join(5)
    write = store_timing.drain()["current"]["write"]
    assert (write["count"], write["refused"]) == (2, 0) and write["max_acquire_sec"] >= 0.15
    sites = {site["site"].rsplit(":", 1)[0]: site for site in store_timing.drain_sites()["current"]["write"]}
    elapsed = sites["tests.test_store_timing:" + test_a_real_wait_that_ends_in_the_lock_is_measured_apart_from_the_hold.__name__]
    assert elapsed["max_acquire_sec"] >= 0.15 > elapsed["max_sec"], "the wait is not the hold"
    assert sites["tests.test_store_timing:_held_write"]["max_sec"] >= 0.15


def test_a_commit_the_body_runs_ends_its_hold_and_never_a_later_transaction_s(hold_clock):
    """A body that commits on its own connection (``usage_store.Txn.commit``: the dispatch claim's zero-wait
    COMMIT under ``launch_admission``) ends the transaction there: that COMMIT is its commit phase, counted once,
    and the hold ends where it returned (what the body does after it holds no lock; ``finish`` has nothing left).
    The next transaction on that connection, ended by a ``finish`` of its own (a barrier's ROLLBACK), keeps its
    own end."""
    conn = _Timed(hold_clock, cost={"COMMIT": 0.25, "ROLLBACK": 0.125})
    with store.transaction(conn, write=True, finish=lambda: None):
        hold_clock[0] += 1.0
        store.commit(conn)
        hold_clock[0] += 0.5
    with store.transaction(conn, write=True, finish=lambda: conn.execute("ROLLBACK")):
        hold_clock[0] += 2.0
    assert conn.statements == ["BEGIN IMMEDIATE", "COMMIT", "BEGIN IMMEDIATE", "ROLLBACK"]
    write = store_timing.drain()["current"]["write"]
    assert {key: write[key] for key in _PHASES} == {
        "count": 2, "refused": 0, "acquire_sec": 0.0, "max_acquire_sec": 0.0, "total_sec": 3.375, "commit_sec": 0.375,
        "max_sec": 2.125, "max_commit_sec": 0.125}


@pytest.mark.parametrize("ending", ["body", "finish"])
def test_a_usage_hold_s_commit_is_its_end_whoever_runs_it(tmp_path, hold_clock, ending):
    """The real money store: the dispatch claim's own ``view.commit(wait_sec=0)`` and a COMMIT left to the hold's
    ``finish`` are each its end, 0.25 s on the hold clock while SQLite runs it, counted once."""
    with usage_store.hold(tmp_path, write=True):  # the store
        pass
    store_timing.drain()

    def cost(sql):
        if sql == "COMMIT":
            hold_clock[0] += 0.25

    with usage_store.hold(tmp_path, write=True) as view:
        view.conn.set_trace_callback(cost)
        hold_clock[0] += 1.0  # the claim's checks and row
        view.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('timing_probe', ?)", (json.dumps(ending),))
        if ending == "body":
            view.commit(wait_sec=0)
        hold_clock[0] += 0.5  # the claim's summaries: after its own COMMIT, outside the lock
    view.conn.set_trace_callback(None)
    held = 1.25 if ending == "body" else 1.75
    write = store_timing.drain()["usage"]["write"]
    assert {key: write[key] for key in ("count", "failed", "total_sec", "commit_sec", "max_sec", "max_commit_sec")} == {
        "count": 1, "failed": 0, "total_sec": held, "commit_sec": 0.25, "max_sec": held, "max_commit_sec": 0.25}
    with usage_store.hold(tmp_path, write=False) as txn:
        assert txn.conn.execute("SELECT value FROM meta WHERE key='timing_probe'").fetchone()[0] == json.dumps(ending)


@pytest.mark.parametrize("ending", ["body", "finish"])
def test_a_usage_commit_refused_busy_keeps_its_phase_through_the_money_error(tmp_path, enforced, monkeypatch, ending):
    """Rollback journal: an open read's SHARED lock refuses a money write's COMMIT, the dispatch claim's own
    zero-wait one or the ``finish``'s after its wait. ``usage_store._commit`` types it inside the transaction; the
    money error still says the phase, the wait and the budget, its SQLite cause carries them, the window's commit
    phase is that COMMIT, and the rollback stands."""
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 50, 4))
    monkeypatch.setattr(usage_store, "USAGE_LOCK_TIMEOUT_SEC", 0.05)  # the finish's wait for readers
    with usage_store.hold(tmp_path, write=True):  # the store, in the rollback journal
        pass
    store_timing.drain()
    with usage_store.hold(tmp_path, write=False):
        with pytest.raises(usage_store.UsageLockUnavailable) as raised:
            with usage_store.hold(tmp_path, write=True, timeout_sec=0.5) as view:
                view.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('timing_probe', '1')")
                if ending == "body":
                    view.commit(wait_sec=0)
    error, cause = raised.value, raised.value.__cause__
    budget = "0" if ending == "body" else "0.05"
    assert str(error).startswith("usage store busy: database is locked [commit: elapsed ")
    assert str(error).endswith(f" s, busy_timeout {budget} s]") and error.rollback_confirmed is True
    assert isinstance(cause, sqlite3.OperationalError) and cause.rollback_confirmed is True
    assert (cause.store_phase, cause.store_busy_timeout_sec) == ("commit", float(budget))
    assert cause.store_elapsed_sec >= (0.0 if ending == "body" else 0.04)
    write = store_timing.drain()["usage"]["write"]
    assert (write["count"], write["failed"], write["refused"]) == (1, 1, 0)
    assert write["commit_sec"] == pytest.approx(cause.store_elapsed_sec, abs=1e-5)
    with usage_store.hold(tmp_path, write=False) as txn:
        assert txn.conn.execute("SELECT value FROM meta WHERE key='timing_probe'").fetchone() is None
