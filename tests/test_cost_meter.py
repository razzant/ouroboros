"""The cost contract's doors that read a product idiom (tests/_cost_meter.py, tests/_cost_driver.py).

Each counts work only as long as the code it watches keeps one idiom: a lock file is created by an
exclusive ``os.open`` (``platform_layer``), a stored dialogue row is decoded by ``dialogue_store.decode``,
a host name is resolved through ``socket.getaddrinfo``, a maintenance pass names refused work in its report
(``maintenance.run``; the live set's and the stand's counts are read with it, tests/_live_set.py), the scale
root's budget fact is the budget its settings resolve to (``usage_projection``).
These checks live apart from
tests/test_cost_contract.py so that a change to those modules re-runs them in seconds (the commit
gate selects a test module by its direct imports) instead of booting the contract's two installs.
"""

from __future__ import annotations

import json
import os

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


@pytest.mark.parametrize("runs, refusals", [
    ("failed done", ["L:child-jobs refused: ['custody_trash: failed']"]),
    ("pending done", []), ("done", [])], ids=["failed_then_done", "pending_then_done", "done"])
def test_the_live_set_s_preparation_reads_each_maintenance_run(tmp_path, monkeypatch, runs, refusals):
    """``_child_jobs_done`` (tests/_live_set.py, the live set's and the stand's preparation) runs the child's pass while
    a job is due. Each run's report is read as the contract's own preparation reads it: one refused run is a driver
    error (``measure`` refuses the whole result), though the run after it is clean; pending is owed work a later run did."""
    from ouroboros import maintenance
    from tests._live_set import _child_jobs_done

    left = runs.split()

    def run(_root, **_kwargs):
        state = left.pop(0)
        return None if state == "none" else {"status": "completed", "jobs": {"custody_trash": {"state": state, "items": 0}}}
    monkeypatch.setattr(maintenance, "due", lambda _root: bool(left))
    monkeypatch.setattr(maintenance, "run", run)
    result: dict = {"errors": [], "live_set": {}}
    with CostMeter(tmp_path) as meter:
        _child_jobs_done(meter, tmp_path, result)
    assert result["errors"] == refusals and not left, (runs, result["errors"], left)


# What each run's own maintenance child reported, by the run's stage, pass and phase (a run without a line spawned no
# child), and what each law's read of a count must then say: the run its refusal names, "owed" (accepted, the pending
# named) or "" (accepted, nothing named). ``M`` dispatches the child itself; ``R`` is a pass that requests a look of it.
M, R = "maintenance-child", "reconcile-maintenance"
LIVE_SET_STORIES = {
    "failed_settling_under_a_healthy_rest": (
        {f"held:{M}:first": "done", f"held:{M}:settle0": "failed", f"held:{M}:rest": "done"},
        {("held", M, "rest"): f"held:{M}:settle0", ("markers", M, "rest"): f"held:{M}:settle0",
         ("pending", M, "rest"): "", ("smallest", M, "rest"): "", ("markers", R, "rest"): ""}),
    "pending_at_the_counted_rest": (
        {f"markers:{M}:first": "done", f"markers:{M}:rest": "pending"},
        {("markers", M, "rest"): f"markers:{M}:rest", ("markers", M, "first"): "", ("schedules", M, "rest"): ""}),
    "pending_then_done": (
        {f"markers:{M}:first": "pending", f"markers:{M}:rest": "done"},
        {("markers", M, "first"): "owed", ("markers", M, "rest"): ""}),
    "smallest_rest_pending_its_steady_rest_done": (
        {f"smallest:{M}:rest": "pending", f"smallest-steady:{M}:rest": "done"},
        {("smallest", M, "rest"): "", ("markers", M, "rest"): ""}),
    "smallest_steady_rest_pending": (
        {f"smallest-steady:{M}:rest": "pending"},
        {("smallest", M, "rest"): f"smallest-steady:{M}:rest", ("running", M, "rest"): ""}),
    "smallest_steady_settling_raised": (
        {f"smallest-steady:{M}:first": "done", f"smallest-steady:{M}:settle0": "raised"},
        {("smallest", M, "rest"): f"smallest-steady:{M}:settle0", ("smallest", M, "first"): "",
         ("running", M, "rest"): f"smallest-steady:{M}:settle0"}),
    "another_pass_s_child_failed": (
        {f"held:{M}:first": "pending", f"markers:{R}:rest": "failed"},
        {("held", M, "first"): "owed", ("held", M, "rest"): "", ("markers", M, "rest"): "",
         ("markers", R, "rest"): f"markers:{R}:rest", ("schedules", R, "rest"): ""}),
}


@pytest.mark.parametrize("children, reads", [pytest.param(*story, id=name) for name, story in LIVE_SET_STORIES.items()])
def test_a_live_set_count_answers_for_the_maintenance_runs_behind_it(tmp_path, monkeypatch, recwarn, children, reads):
    """The live set's passes as ``_one_stage`` (tests/_live_set.py) runs them at every stage; ``_drain`` runs the child
    each run spawned and keeps its report under that run's tag; the result, as the driver hands it over, is read by both
    laws' consumers: ``_rest`` (the live-set axis, every stage) and ``_pass`` (the stand, the smallest and largest set).
    A refusal under any run behind a count fails the read, a settling run's under a healthy rest and an earlier stage's
    too; pending fails only the counted rest itself (the smallest set's is its steady one) and is named where owed; a
    pass answers for its own children only, and a run that spawned none has nothing to read."""
    import re
    import sqlite3
    import threading
    import time
    from types import SimpleNamespace

    from ouroboros import maintenance, server_maintenance
    from tests._cost_meter import tag_of
    from tests._live_set import STAGES, _one_stage
    from tests.test_cost_live_set import _rest
    from tests.test_stand_counts import _pass

    result: dict = {"errors": [], "child_spawns": [], "_started": time.monotonic(), "live_set": {"stages": {}}}

    def told() -> str:
        return children.get(tag_of().split(":", 1)[1], "")

    def run(_root, **_kwargs):
        if told() == "raised":
            raise RuntimeError("the child's pass died")
        return {"status": "completed", "jobs": {"custody_trash": {"state": told(), "items": 0}}}

    def one_pass() -> None:
        stage, name, phase = tag_of().split(":")[1:]
        if phase == "first" and f"{stage}:{name}:settle0" in children:  # it moved its own inputs: a settling run follows
            conn = sqlite3.connect(tmp_path / "inputs.sqlite")
            conn.execute("CREATE TABLE IF NOT EXISTS moved (id INTEGER)")
            conn.execute("INSERT INTO moved VALUES (1)")
            conn.commit()
            conn.close()
        if told():
            result["child_spawns"].append(tag_of())  # as the driver records the child's spawn (``_stub_duty_effects``)

    monkeypatch.setattr(maintenance, "run", run)
    monkeypatch.setattr(maintenance, "child", SimpleNamespace(owed=lambda: False, request=lambda: None))
    monkeypatch.setattr(server_maintenance, "_DELIVERY_WAKE", threading.Event())
    with CostMeter(tmp_path) as meter:
        for stage in STAGES:
            result["live_set"]["stages"][stage] = _one_stage(meter, stage, {M: one_pass, R: one_pass}, tmp_path, {},
                                                             threading.Event(), result)
    result.pop("_started")
    measured = json.loads(json.dumps(result, sort_keys=True, default=str))  # as the driver writes it for the laws
    assert measured["errors"] == [] and sorted(tag.split(":", 1)[1] for tag in measured["child_spawns"]) == sorted(children)
    readers = {"live set": lambda stage, name, phase: _rest({1: measured}, 1, stage, name, phase),
               "stand": lambda stage, name, phase: _pass(measured, stage, name, phase)}
    for (stage, name, phase), verdict in reads.items():
        for reader, read in readers.items():
            if reader == "stand" and stage not in (STAGES[0], STAGES[-1]):
                continue  # the stand's laws compare the smallest and the largest set only
            owed = len([w for w in recwarn if "owed work left pending" in str(w.message)])
            if verdict not in ("", "owed"):
                with pytest.raises(RuntimeError, match=f"did not do its work: .*{re.escape(f'L:{verdict}')}"):
                    read(stage, name, phase)
            else:
                assert read(stage, name, phase) is measured["live_set"]["stages"][stage][name][phase]
            named = len([w for w in recwarn if "owed work left pending" in str(w.message)]) > owed
            assert named is (verdict == "owed"), (reader, stage, name, phase, [str(w.message) for w in recwarn])


@pytest.mark.parametrize("failure", ["tests._scale_root", "tests._cost_driver", "reported_error"])
def test_a_broken_measurement_is_not_a_performance_assertion(tmp_path, monkeypatch, failure):
    """Expected performance breaches may be xfailed; a child or fixture that never measured them may not."""
    from pathlib import Path
    from types import SimpleNamespace

    from tests import _stand_cache as cache, group_run, test_cost_contract as contract

    # The inline R_late allowance must not bypass the generic gap's error distinction.
    late = next(row for mark in contract.test_room_history_is_flat.pytestmark if mark.name == "parametrize"
                for row in mark.args[1] if getattr(row, "values", None) == ("R_late",))
    assert next(mark.kwargs.get("raises") for mark in late.marks if mark.name == "xfail") is contract.KnownBreach

    def child(args, **_kwargs):
        module = args[2]
        if module == failure:
            return SimpleNamespace(returncode=1, stdout="", stderr="controlled measurement refusal")
        result = {"k": 1} if module == "tests._scale_root" else {"errors": ["controlled measurement refusal"]}
        Path(args[-1]).write_text(json.dumps(result), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(group_run, "run_in_group", child)
    monkeypatch.setenv("OUROBOROS_TEST_TEMP_ROOT", str(tmp_path))  # a stand of this test's own
    with pytest.raises(RuntimeError, match="controlled measurement refusal"), cache.measuring(1):
        pass


def _stand_build(built):
    def build(home, k, journal=""):
        built.append((k, journal))
        (home / "data" / "archive").mkdir(parents=True)
        (home / "data" / "archive" / "segment.jsonl").write_text("sealed\n", encoding="utf-8")
        (home / "data" / "logs.jsonl").write_text("as built\n", encoding="utf-8")
        (home / "data" / "gone.json").write_text("{}", encoding="utf-8")
        (home / "data" / "state.json").write_text("{}", encoding="utf-8")
        (home / "data" / "object").write_text("read only\n", encoding="utf-8")
        os.chmod(home / "data" / "object", 0o444)
        os.symlink("logs.jsonl", home / "data" / "link")
    return build


def _drive_it(home):
    """What a consumer leaves behind: a rotated log, an atomic write, a file deleted, one appended to, a new folder, a moved link."""
    data = home / "data"
    os.replace(data / "logs.jsonl", data / "archive" / "logs_rotated.jsonl")
    (data / "logs.jsonl").write_text("a consumer's row\n", encoding="utf-8")
    (data / "state.tmp").write_text('{"driven": true}', encoding="utf-8")
    os.replace(data / "state.tmp", data / "state.json")
    (data / "gone.json").unlink()
    with (data / "archive" / "segment.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("appended\n")
    (data / "added" / "deep").mkdir(parents=True)
    (data / "added" / "deep" / "file").write_text("new\n", encoding="utf-8")
    os.utime(data / "object")  # as Git freshens an object it was asked to write again
    (data / "link").unlink()
    os.symlink("archive", data / "link")


def _identity(path):
    seen = os.stat(path, follow_symlinks=False)
    return seen.st_ino, seen.st_size, seen.st_mtime_ns


def test_a_stand_is_built_once_and_each_consumer_finds_it_back_in_place(tmp_path, monkeypatch):
    """Paths and inodes stay: a root's records name both, so even a rotated or atomically replaced file comes back as itself."""
    from tests import _stand_cache as cache

    built = []
    monkeypatch.setenv("OUROBOROS_TEST_TEMP_ROOT", str(tmp_path))
    monkeypatch.setattr(cache, "build", _stand_build(built))
    with cache.stand(1) as first:
        data = first / "data"
        named = [data / name for name in ("logs.jsonl", "state.json", "gone.json", "object", "archive", "archive/segment.jsonl")]
        pristine = [_identity(path) for path in named]
        _drive_it(first)
        assert _identity(named[0]) != pristine[0] and _identity(named[4]) != pristine[4]
    with cache.stand(1) as second:
        assert second == first and built == [(1, "")]
        assert [_identity(path) for path in named] == pristine
        assert (data / "logs.jsonl").read_text(encoding="utf-8") == "as built\n"
        assert (data / "state.json").read_text(encoding="utf-8") == "{}"
        assert (data / "archive" / "segment.jsonl").read_text(encoding="utf-8") == "sealed\n"
        assert sorted(path.name for path in data.iterdir()) == ["archive", "gone.json", "link", "logs.jsonl", "object", "state.json"]
        assert os.readlink(data / "link") == "logs.jsonl" and os.listdir(data / "archive") == ["segment.jsonl"]
    with cache.stand(1, "WAL") as other, cache.stand(30) as larger:
        assert len({first, other, larger}) == 3 and built == [(1, ""), (1, "WAL"), (30, "")]


def test_a_stand_in_use_makes_nobody_wait_and_a_killed_build_publishes_nothing(tmp_path, monkeypatch):
    from tests import _stand_cache as cache

    built = []
    monkeypatch.setenv("OUROBOROS_TEST_TEMP_ROOT", str(tmp_path))
    monkeypatch.setattr(cache, "build", _stand_build(built))
    with cache.stand(1) as held, cache.stand(1) as private:
        assert private != held and built == [(1, ""), (1, "")] and len(private.parts) == len(held.parts)
        (private / "data" / "logs.jsonl").write_text("private\n", encoding="utf-8")
        assert (held / "data" / "logs.jsonl").read_text(encoding="utf-8") == "as built\n"

    def killed(home, _k, _journal=""):
        (home / "data").mkdir(parents=True)
        raise RuntimeError("killed half way")

    monkeypatch.setattr(cache, "build", killed)
    with pytest.raises(RuntimeError, match="killed half way"), cache.stand(2):
        pass
    monkeypatch.setattr(cache, "build", _stand_build(built))
    with cache.stand(2) as whole:
        assert (whole / "data" / "logs.jsonl").is_file() and built[-1] == (2, "")


def _table(tmp_path, rows: int):
    """A store of one table of ``rows`` rows, written before any meter is installed."""
    import sqlite3

    db = tmp_path / "db.sqlite"
    seed = sqlite3.connect(db)
    seed.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    seed.executemany("INSERT INTO t VALUES (?, ?)", ((i, f"value-{i}") for i in range(rows)))
    seed.commit()
    seed.close()
    return db


def test_vm_steps_are_exact_whatever_the_connection_ran_before(tmp_path):
    """#2691: the same statements count the same after any earlier work on their connection, and a statement shorter
    than a progress period is not read as nothing. With a period of N, SQLite credits each prepared statement at its own
    multiples of N, a phase it keeps across executions and boundaries: 27 steps read 0 or 100 by what ran before."""
    import sqlite3

    db, counted = _table(tmp_path, 200), []
    for earlier in range(1, 5):  # the connection's first statement also loads the schema: every count starts after one
        with CostMeter(tmp_path) as meter:
            conn = sqlite3.connect(db)
            tagged("before", lambda: [conn.execute("SELECT v FROM t WHERE id = ?", (i,)).fetchall()
                                      for i in range(earlier)])
            tagged("same", lambda: [conn.execute("SELECT v FROM t WHERE id = ?", (i,)).fetchall() for i in (5, 6, 7)])
            conn.close()
        counted.append(meter.snapshot("same")["counts"].get("vm_steps", 0))
    assert counted[0] > 0 and len(set(counted)) == 1, counted


def test_vm_steps_go_to_the_reader_that_steps_the_statement(tmp_path):
    """A scan's steps stay its own while it queries again row by row (the shape a hidden scan takes): ``callers``
    names the reader stack stepping each instruction, so a breach names the reader that grew, not the statement
    started last."""
    import collections
    import sqlite3

    db, readers = _table(tmp_path, 300), {}

    def scan(conn, inner=None):
        for index, _row in enumerate(conn.execute("SELECT v FROM t")):
            if inner is not None and index % 100 == 0:
                inner()

    def look(conn):
        conn.execute("SELECT count(*) FROM t WHERE v LIKE '%7%'").fetchone()

    for name, run in (("apart", lambda conn: (scan(conn), [look(conn) for _ in range(3)])),
                      ("interleaved", lambda conn: scan(conn, lambda: look(conn)))):
        with CostMeter(tmp_path) as meter:
            conn = sqlite3.connect(db)
            conn.execute("SELECT 1 FROM t LIMIT 1").fetchall()  # the schema load, before the count
            tagged(name, lambda: run(conn))
            conn.close()
        readers[name] = collections.Counter()
        for key, steps in meter.snapshot(name)["callers"].items():
            if key.startswith("vm_steps <- "):
                readers[name][key.split(" <- ")[1].rsplit(".", 1)[-1]] += steps  # by the innermost reader
    assert readers["interleaved"] == readers["apart"] and set(readers["apart"]) == {"scan", "look"}, readers


def test_one_reader_called_by_two_callers_in_turn_credits_each_its_own_steps(tmp_path):
    """A returned frame's address serves the next call in CPython, so the same helper stepping for first() and then
    for second() can sit at one address: a reader is known by the codes its stack runs through, never by a frame."""
    import collections
    import sqlite3

    db = _table(tmp_path, 200)

    def helper(conn):
        conn.execute("SELECT count(*) FROM t WHERE v LIKE '%7%'").fetchone()

    def first(conn):
        helper(conn)

    def second(conn):
        helper(conn)

    with CostMeter(tmp_path) as meter:
        conn = sqlite3.connect(db)
        conn.execute("SELECT 1 FROM t LIMIT 1").fetchall()  # the schema load, before the count
        helper(conn)  # and the statement's first run, which steps once more than its cached reruns
        tagged("turns", lambda: (first(conn), second(conn)))
        conn.close()
    steps = collections.Counter()
    for key, count in meter.snapshot("turns")["callers"].items():
        if key.startswith("vm_steps <- "):
            steps[key.split(" <- ")[2].rsplit(".", 1)[-1]] += count  # by the reader's caller
    assert steps["first"] == steps["second"] > 0 and set(steps) == {"first", "second"}, steps


def test_naming_a_reader_keeps_none_of_its_locals_alive(tmp_path):
    """The meter keeps a reader's codes, not its frame: what the reader held (an open cursor's statement, a file) goes
    when it returns, in the measured thread and in a worker that outlives the meter."""
    import sqlite3
    import threading
    import weakref

    db, kept, measured = _table(tmp_path, 50), [], threading.Event()
    release = threading.Event()

    class Held:
        pass

    def reader(conn):
        held = Held()
        conn.execute("SELECT count(*) FROM t").fetchone()
        return weakref.ref(held)

    def worker():
        conn = sqlite3.connect(db)
        kept.append(reader(conn))
        conn.close()
        measured.set()
        release.wait(30)  # this thread, and whatever it cached, outlives the meter

    thread = threading.Thread(target=worker, name="reader-worker")
    try:
        with CostMeter(tmp_path) as meter:
            conn = sqlite3.connect(db)
            ref = tagged("held", lambda: reader(conn))
            assert ref() is None
            thread.start()
            assert measured.wait(30)
            conn.close()
        assert kept and kept[0]() is None and thread.is_alive()
        assert meter.snapshot("held")["counts"].get("vm_steps", 0) > 0
    finally:
        release.set()
        if thread.is_alive():  # an assertion can end the test before the worker starts, and join would then raise
            thread.join(30)
