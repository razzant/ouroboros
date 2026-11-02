"""The cost contract's doors that read a product idiom (tests/_cost_meter.py, tests/_cost_driver.py).

Each counts work only as long as the code it watches keeps one idiom: a lock file is created by an
exclusive ``os.open`` (``platform_layer``), a stored dialogue row is decoded by ``dialogue_store.decode``,
a host name is resolved through ``socket.getaddrinfo``, a maintenance pass names refused work in its report
(``maintenance.run``). These checks live apart from
tests/test_cost_contract.py so that a change to those modules re-runs them in seconds (the commit
gate selects a test module by its direct imports) instead of booting the contract's two installs.
"""

from __future__ import annotations

import json

import pytest

from tests._cost_meter import CostMeter, tagged


@pytest.mark.parametrize("contended", [False, True])
def test_the_meter_counts_one_lock_acquisition_however_contended(tmp_path, contended):
    """A lock taken after its holder let go is one acquisition: the waiter's refused creates and its
    probes of the holder are contention (more opens of the lock file), never more acquisitions."""
    import threading

    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    lock = tmp_path / "state" / "chronicle.lock"
    releaser = threading.Timer(0.3, release_exclusive_file_lock,
                               (lock, acquire_exclusive_file_lock(lock, timeout_sec=1.0) if contended else None))

    def take():
        fd = acquire_exclusive_file_lock(lock, timeout_sec=30.0)
        assert fd is not None
        release_exclusive_file_lock(lock, fd)

    with CostMeter(tmp_path) as meter:
        if contended:
            releaser.start()
        tagged("take", take)
    if contended:
        releaser.join()
    counts = meter.snapshot("take")["counts"]
    assert counts.get("lock_acquired") == 1 and (counts.get("os_open:lock", 0) > 1) is contended, counts


def test_the_meter_counts_stored_dialogue_rows_decoded_and_no_other_json(tmp_path):
    """``rows_decoded`` is the stored dialogue row bodies a boundary decodes; any other JSON is not a row."""
    from ouroboros import dialogue_store

    with CostMeter(tmp_path) as meter:
        tagged("read", lambda: (dialogue_store.decode('{"text": "a stored row"}'), json.loads('{"other": 1}')))
    counts = meter.snapshot("read")["counts"]
    assert (counts.get("json_loads"), counts.get("rows_decoded")) == (2, 1), counts


def test_the_driver_refuses_and_counts_every_lookup_beyond_this_machine(tmp_path):
    """The driver's resolver: this machine's names resolve, any other host is refused and counted under
    the boundary that asked (a turn's capability fetch, for one), so a measurement never waits on a network."""
    from tests._cost_driver import loopback_only

    with CostMeter(tmp_path) as meter:
        lookup = loopback_only(meter, lambda host, *_args, **_kwargs: [host])
        assert tagged("turn", lambda: (lookup("127.0.0.1", 0), lookup("localhost", 0))) == (["127.0.0.1"], ["localhost"])
        with pytest.raises(OSError, match="no network"):
            tagged("turn", lambda: lookup("openrouter.ai", 443))
    snap = meter.snapshot("turn")
    assert snap["counts"].get("network_refused") == 1 and snap["names"] == {"network_refused openrouter.ai": 1}


def test_the_turn_law_s_batched_rows_are_the_rows_append_stores(tmp_path):
    """The turn law lays its rows in one transaction (tests/_turn_shapes.append_rows); each must be stored as
    ``dialogue_store.append`` stores it, positions and room facts included, or the law measures rows no
    install holds."""
    import sqlite3

    from ouroboros import dialogue_store
    from tests._turn_shapes import append_rows

    rows = [{"chat_id": chat_id, "direction": direction, "text": f"row {index}",
             "ts": f"2025-03-0{index}T00:00:00+00:00", **({"task_id": "w1"} if direction == "out" else {})}
            for index, (chat_id, direction) in enumerate([(1, "in"), (1, "out"), (3000, "out"), (1, "in")], start=1)]
    for row in rows:
        dialogue_store.append(tmp_path / "public", "chat", row)
    append_rows(tmp_path / "batched", rows)
    stored = []
    for root in (tmp_path / "public", tmp_path / "batched"):
        conn = sqlite3.connect(dialogue_store.path(root))
        try:
            stored.append([conn.execute(query).fetchall() for query in (
                "SELECT * FROM dialogue ORDER BY seq", "SELECT * FROM streams ORDER BY stream",
                "SELECT * FROM rooms ORDER BY chat_id")])
        finally:
            conn.close()
    assert stored[0] == stored[1] and len(stored[0][0]) == len(rows)


@pytest.mark.parametrize("pass_", ["bound", "unbound", "returned_failed", "left_pending"])
def test_the_driver_reads_the_work_a_maintenance_pass_reports_it_did_not_do(tmp_path, monkeypatch, pass_):
    """``refused`` (tests/_cost_driver.py) reads ``maintenance.run``'s own report: a pass pointed at another install has
    every job refused (``StoreForeign``, #2465); a job may return ``failed`` without raising while the pass reports
    ``completed``; a job left ``pending`` is owed work, refused only at rest. A bound pass at rest refuses nothing."""
    import contextlib

    from ouroboros import maintenance
    from tests._cost_driver import refused
    from tests._store_shared import as_install

    (root := tmp_path / "data").mkdir()

    def run(bound: bool) -> dict:
        with as_install(root) if bound else contextlib.nullcontext():
            return maintenance.run(root, should_stop=lambda: False)
    assert not refused(run(True))  # bound: the root mints its own identity, as at an install's first boot
    if pass_ in ("returned_failed", "left_pending"):  # returned, not raised: custody_trash while a drive stays stuck
        outcome = {"state": "failed" if pass_ == "returned_failed" else "pending", "items": 0}
        monkeypatch.setitem(maintenance.JOBS, "custody_trash", (lambda *_: True, lambda *_: outcome))
    report = run(pass_ != "unbound")
    if pass_ == "unbound":
        assert refused(report) == ["pass failed", *(f"{name}: failed" for name in maintenance.JOBS)], report
    else:
        named = ["custody_trash: failed"] if pass_ == "returned_failed" else []
        assert report["status"] == "completed" and refused(report) == named, report
        assert refused(report, rest=True) == (["custody_trash: pending"] if pass_ == "left_pending" else named)
