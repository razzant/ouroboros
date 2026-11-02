"""The cost contract's doors that read a product idiom (tests/_cost_meter.py, tests/_cost_driver.py).

Each counts work only as long as the code it watches keeps one idiom: a lock file is created by an
exclusive ``os.open`` (``platform_layer``), a stored dialogue row is decoded by ``dialogue_store.decode``,
a host name is resolved through ``socket.getaddrinfo``, a maintenance pass names refused work in its report
(``maintenance.run``), the scale root's budget fact is the budget its settings resolve to (``usage_projection``).
These checks live apart from
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


@pytest.mark.parametrize("k", [1, 30, 100])
def test_the_scale_root_s_budget_fact_follows_the_configured_budget_and_the_settled_spend(tmp_path, monkeypatch, k):
    """``_budget`` (tests/_scale_root.py) hands ``usage_projection`` no limit, so the fact is the budget the root's
    settings resolve to over the spend its ledger settled: ``spend_limit_usd(k)`` as ``build`` writes it, and the
    lowered budget once the owner changes it (the echo it replaced kept the authored limit)."""
    from ouroboros import config, usage_store
    from tests._scale_root import _budget, money, spend_limit_usd

    (root := tmp_path / "data").mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setattr(config, "SETTINGS_PATH", root / "settings.json")  # as the build's interpreter names it

    def configure(limit: float) -> None:
        (root / "settings.json").write_text(json.dumps({"TOTAL_BUDGET": limit}), encoding="utf-8")
    try:
        configure(spend_limit_usd(k))
        for task_id in ("settled-1", "settled-2"):
            money(root, task_id, limit=spend_limit_usd(k))
        built = _budget(root)
        assert built["limit_usd"] == pytest.approx(spend_limit_usd(k), abs=1e-6), built  # the projection rounds to micro-dollars
        assert built["settled_usd"] == pytest.approx(0.8) and built["remaining_known_usd"] == pytest.approx(
            built["limit_usd"] - built["settled_usd"]), built
        configure(5.0)
        assert _budget(root) == {"limit_usd": 5.0, "settled_usd": built["settled_usd"],
                                 "remaining_known_usd": pytest.approx(5.0 - built["settled_usd"])}
    finally:
        usage_store.forget(root)


def test_the_scale_root_build_runs_only_in_a_process_that_reads_the_roots_own_settings(tmp_path, monkeypatch):
    """The budget fact is read through the process's settings, so ``main`` refuses a process whose settings are another
    document (it would answer with ITS budget, or the product default) before it builds anything."""
    from ouroboros import config
    from tests import _scale_root

    (root := tmp_path / "data").mkdir()
    built = []
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setattr(_scale_root, "build", lambda path, k: built.append((path, k)) or {"k": k})
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "elsewhere" / "settings.json")
    with pytest.raises(SystemExit, match="OUROBOROS_SETTINGS_PATH must name the settings.json of the root"):
        _scale_root.main([str(root), "1", str(tmp_path / "refused.json")])
    assert not built and not (tmp_path / "refused.json").exists()
    monkeypatch.setattr(config, "SETTINGS_PATH", root / "settings.json")
    assert _scale_root.main([str(root), "1", str(tmp_path / "facts.json")]) == 0 and built == [(root, 1)]


def test_the_steady_state_is_not_reached_by_a_pass_the_handed_in_judge_names_refused(tmp_path, monkeypatch):
    """``steady_state`` (tests/_turn_shapes.py) judges every maintenance pass by the ``refused`` it is handed (the
    driver's, which runs as ``__main__`` and so is not imported by name). A job that failed once and is no longer due
    leaves the tail complete and nothing due: only the pass's own report says it did nothing, and the install must
    not be called steady for it. The message names the whole state, so no other cause can stand in for the refusal."""
    from ouroboros import maintenance
    from ouroboros.chronicle_store import ChronicleStore
    from tests._cost_driver import refused
    from tests._store_shared import as_install
    from tests._turn_shapes import steady_state

    (root := tmp_path / "data").mkdir()
    owed = [True]

    def failing(*_args) -> dict:
        owed[0] = False
        return {"state": "failed", "items": 0}
    monkeypatch.setitem(maintenance.JOBS, "custody_trash", (lambda *_args: owed[0], failing))
    with as_install(root):
        ChronicleStore(root).ensure_activated(wait=True)
        with pytest.raises(RuntimeError) as breach:
            steady_state(root, refused=refused)
    named = str(breach.value)
    assert "'complete': True" in named and "still due []" in named and "refused ['custody_trash: failed']" in named, named
