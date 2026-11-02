"""Store transactions report their lock-hold seconds per store kind, in memory; the hook never raises."""
from __future__ import annotations

import sqlite3
import time

import pytest

from ouroboros import dialogue_store, store, store_timing, usage_store
from ouroboros.task_results import load_task_result, write_task_result


@pytest.fixture(autouse=True)
def clean_window():
    store.close_connections()
    store_timing.drain()
    yield
    store_timing.drain()
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
    assert len(longest["sites"]) == 1 and longest["sites_omitted"] == 0, "nothing cut, nothing omitted"


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
    with store.open_store(tmp_path, write=False) as reader:
        reader.row("t", "status")
        with pytest.raises(store.StoreBusy):
            with store.acquisition_budget(0), store.open_store(tmp_path, write=True) as writer:
                writer.conn.execute("UPDATE tasks SET status='completed' WHERE task_id='t'")
    window = store_timing.drain()
    write, read = window["current"]["write"], window["current"]["read"]
    assert (write["count"], write["failed"], read["failed"]) == (1, 1, 0)
    site = write["sites"][0]
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


def test_the_window_keeps_each_site_and_publishes_the_top_sites_by_summed_hold(tmp_path):
    write_task_result(tmp_path, "t", "running")
    store_timing.drain()
    for reader in (_short_read, _short_read, _short_read, _long_read, _third_read, _fourth_read):
        reader(tmp_path)
    read = store_timing.drain()["current"]["read"]
    assert read["count"] == 6 and len(read["sites"]) == store_timing.TOP_SITES < 4  # four sites held, three published
    assert read["sites_omitted"] == 1, "the cut is disclosed in band"
    assert read["sites"][0]["site"].startswith("tests.test_store_timing:_long_read")
    totals = [site["total_sec"] for site in read["sites"]]
    assert totals == sorted(totals, reverse=True) and sum(site["count"] for site in read["sites"]) <= 6
