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
    SHARED lock fails a zero-wait COMMIT; that hold is in the window as failed, not missing."""
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 50, 4))
    write_task_result(tmp_path, "t", "running")
    store_timing.drain()
    store_timing.drain_sites()
    with store.open_store(tmp_path, write=False) as reader:
        reader.row("t", "status")
        with pytest.raises(store.StoreBusy):
            with store.acquisition_budget(0), store.open_store(tmp_path, write=True) as writer:
                writer.conn.execute("UPDATE tasks SET status='completed' WHERE task_id='t'")
    window = store_timing.drain()
    write, read = window["current"]["write"], window["current"]["read"]
    assert (write["count"], write["failed"], read["failed"]) == (1, 1, 0)
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
        assert set(row["store_locks"]["current"]["read"]) == {"count", "failed", "total_sec", "cpu_sec", "max_sec", "max_site"}, row
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
