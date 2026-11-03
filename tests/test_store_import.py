"""The importer and exporter of the current-facts store (``ouroboros/store_import.py``):
publication and its crash points, child roots, quarantine at import, the byte-identical
export round trip, and the per-row reconciliation of an out-of-band change."""
from __future__ import annotations

import errno
import hashlib
import json
import os
import pathlib
import shutil
import sqlite3

import pytest

from ouroboros import obligations as o
from ouroboros import store, store_import
from ouroboros.store import GENERATION, GENERATION_KEY, StoreLost
from ouroboros.store_import import ensure_current_store, export_facts, set_file_path
from ouroboros.task_results import task_result_path
from ouroboros.utils import append_jsonl, atomic_write_json


def result(task_id: str, status: str = "completed", **fields) -> dict:
    return {"task_id": task_id, "status": status, "ts": "2026-10-10T00:00:00+00:00",
            "updated_at": "2026-10-10T00:00:01+00:00", "_schema_version": 1, **fields}


def write_result(root: pathlib.Path, row: dict) -> pathlib.Path:
    path = task_result_path(root, row["task_id"])
    atomic_write_json(path, row)
    return path


def add_set_member(root: pathlib.Path, name: str, identity: str, facts: dict) -> None:
    """A pre-store root's set FILE (``state/obligations/<name>.json``): importer input only."""
    path = set_file_path(root, name)
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    rows[identity] = facts
    atomic_write_json(path, rows)


def fixture_root(tmp_path: pathlib.Path) -> pathlib.Path:
    """Three results (one running with a child drive holding a retry row, whose own child
    drive holds a grandchild row), two set files, one unstamped file (quarantined at
    import) and one id-mismatched file (refused)."""
    root = tmp_path / "data"
    write_result(root, result("root1", "running", parent_task_id=None, root_task_id="root1"))
    write_result(root, result("child1", parent_task_id="root1", root_task_id="root1", delegation_role="child",
                              nested={"k": [1, 2.5, "ü"]}, pause_notices=[{"episode": 1}]))
    write_result(root, result("done1", "failed", superseded_by="done1-retry"))
    atomic_write_json(task_result_path(root, "legacy"), {"task_id": "legacy", "status": "completed"})  # unstamped
    atomic_write_json(task_result_path(root, "mismatch"), result("other"))
    add_set_member(root, "nonterminal", "root1", {"task_id": "root1"})
    add_set_member(root, "pending_drives", "root1", {"task_id": "root1"})
    add_set_member(root, "custody_open", "run:r1", {"task_id": "root1", "custody": "{}"})
    add_set_member(root, "custody_open", "invocation:i1", {"task_id": "root1", "request": {"x": 1}})
    add_set_member(root, "pause_notices", "child1", {"task_id": "child1"})
    for name in o.BOOT_SETS:  # every file-era boot classification left every set file behind
        if not set_file_path(root, name).exists():
            atomic_write_json(set_file_path(root, name), {})
    drive = root / "state" / "headless_tasks" / "root1" / "data"
    write_result(drive, result("root1", "completed", retry_task_id="root1-r2"))
    write_result(drive, result("root1-r2", "running"))
    write_result(drive / "task_drives" / "root1-r2", result("root1-r2-sub", "running", parent_task_id="root1-r2"))
    return root


def drives_of(root: pathlib.Path) -> list:
    """The fixture's child drive and its grandchild drive, parents first."""
    child = root / "state" / "headless_tasks" / "root1" / "data"
    return [child, child / "task_drives" / "root1-r2"]


def snapshot(root: pathlib.Path) -> dict:
    files = {}
    for path in sorted((root / "task_results").glob("*.json")) + sorted((root / "state" / "obligations").glob("*.json")):
        files[str(path.relative_to(root))] = path.read_bytes()
    return files


def reconcile_legacy(root):
    """An owner-requested fresh pass, using the same resumable job as maintenance."""
    from ouroboros import maintenance_jobs
    maintenance_jobs.record(root, "legacy_reconcile", state="pending", cursor=None, done=0, total=None)
    return store_import.run_legacy_reconcile(root, progress=None, should_stop=lambda: False)["root_report"]


def meta(root: pathlib.Path, key: str):
    with store.open_store(root, write=False) as txn:
        return txn.meta(key)


def load(root: pathlib.Path, task_id: str):
    with store.open_store(root, write=False) as txn:
        return txn.load(task_id)


def rows(path: pathlib.Path, kind: str) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if json.loads(line).get("type") == kind]


# ---- publication ------------------------------------------------------------------

def test_the_import_publishes_stamps_quarantines_and_covers_child_roots(tmp_path):
    root = fixture_root(tmp_path)
    before = snapshot(root)
    report = ensure_current_store(root)
    assert report["status"] == "completed" and report["results"] == 3
    assert report["quarantined"] == 1 and report["refused"] == 1
    assert store.probe(root) == ("present", store.TIER_ENFORCED)
    assert store.generation_stamped(root)
    provenance = meta(root, "import")
    assert provenance["refused"] == {"mismatch": "task_id_mismatch"} and provenance["lock_tier"] == "enforced"
    baseline = meta(root, "baseline")
    assert set(baseline) == {"root1", "child1", "done1", "mismatch"}
    assert baseline["mismatch"] == {**baseline["mismatch"], "refused": "task_id_mismatch"}
    for task_id, entry in baseline.items():
        raw = task_result_path(root, task_id).read_bytes()
        assert entry["sha256"] == hashlib.sha256(raw).hexdigest() and entry["size"] == len(raw)
    assert load(root, "child1")["nested"] == {"k": [1, 2.5, "ü"]}
    with store.open_store(root, write=False) as txn:
        assert txn.set_rows("custody_open") == {"run:r1": {"task_id": "root1", "custody": "{}"},
                                                "invocation:i1": {"task_id": "root1", "request": {"x": 1}}}
        row = txn.conn.execute("SELECT * FROM tasks WHERE task_id='root1'").fetchone()
        assert (row["status"], row["pending_drive"], row["root_task_id"]) == ("running", 1, "root1")
        assert txn.conn.execute("SELECT has_pause_notices FROM tasks WHERE task_id='child1'").fetchone()[0] == 1
    # the unstamped file moved to quarantine, once, with one durable event; the mismatch stayed in place
    assert not task_result_path(root, "legacy", create=False).exists()
    assert (root / "task_results" / "quarantine" / "legacy.json").exists()
    assert task_result_path(root, "mismatch", create=False).exists()
    events = rows(root / "logs" / "events.jsonl", "task_results_quarantined")
    assert len(events) == 1 and events[0]["count"] == 1
    # child drives down the tree: their own store by the same code, retry rows included
    drive, grandchild = drives_of(root)
    assert report["drives"] == {str(drive): "completed", str(grandchild): "completed"}
    with store.open_store(drive, write=False) as txn:
        assert txn.task_ids() == ["root1", "root1-r2"]
    with store.open_store(grandchild, write=False) as txn:
        assert txn.task_ids() == ["root1-r2-sub"]
    assert [row["phase"] for row in rows(root / "logs" / "supervisor.jsonl", "current_store_import")] == ["completed"]
    # the frozen files are untouched by the import
    after = snapshot(root)
    assert {k: v for k, v in before.items() if "legacy" not in k} == after
    # idempotent: a second boot finds the store and nothing to reconcile
    assert ensure_current_store(root)["status"] == "unchanged"


def test_a_crash_before_publication_is_redone_from_scratch(tmp_path, monkeypatch):
    root = fixture_root(tmp_path)
    build = store_import._build

    def crash(tmp, *args, **kwargs):
        build(tmp, *args, **kwargs)
        raise KeyboardInterrupt("the process died before the rename")

    monkeypatch.setattr(store_import, "_build", crash)
    with pytest.raises(KeyboardInterrupt):
        ensure_current_store(root, with_children=False)
    assert store.probe(root) == ("absent", None) and not store.generation_stamped(root)
    stale = root / "state" / "current.sqlite.import-99999-deadbeef"
    stale.write_bytes(b"a sibling a crashed process left behind")
    monkeypatch.setattr(store_import, "_build", build)
    assert ensure_current_store(root, with_children=False)["status"] == "completed"
    assert not stale.exists()


def test_a_crash_after_publication_before_the_stamp_is_closed_by_the_next_boot(tmp_path, monkeypatch):
    from ouroboros import startup_migrations

    root = fixture_root(tmp_path)
    real_stamp = startup_migrations.stamp

    def no_generation_stamp(target, **facts):
        if facts.get(GENERATION_KEY) == GENERATION:
            raise KeyboardInterrupt("died after publishing, before the stamp")
        return real_stamp(target, **facts)

    monkeypatch.setattr(startup_migrations, "stamp", no_generation_stamp)
    with pytest.raises(KeyboardInterrupt):
        ensure_current_store(root, with_children=False)
    assert store.probe(root)[0] == "present" and not store.generation_stamped(root)
    monkeypatch.setattr(startup_migrations, "stamp", real_stamp)
    assert ensure_current_store(root, with_children=False)["status"] == "unchanged"  # accepted by its own meta
    assert store.generation_stamped(root)


def test_a_missing_store_beside_its_stamp_is_lost_until_the_owner_clears_the_stamp(tmp_path):
    from ouroboros.startup_migrations import stamp

    root = fixture_root(tmp_path)
    ensure_current_store(root, with_children=False)
    store.store_path(root).unlink()
    with pytest.raises(StoreLost):
        ensure_current_store(root, with_children=False)
    with pytest.raises(StoreLost):
        ensure_current_store(root, with_children=False)  # no silent rebuild on a later boot either
    stamp(root, **{GENERATION_KEY: None})  # the owner's explicit decision: the files are the authority again
    assert ensure_current_store(root, with_children=False)["status"] == "completed"
    assert store.probe(root)[0] == "present"


def test_the_cancel_latch_migration_runs_first_so_a_latched_row_is_imported_not_quarantined(tmp_path):
    root = fixture_root(tmp_path)
    atomic_write_json(task_result_path(root, "wedged"), {"task_id": "wedged", "status": "cancel_requested"})
    report = ensure_current_store(root, with_children=False)
    assert report["quarantined"] == 1  # still only the legacy row
    assert load(root, "wedged")["status"] == "cancel_requested"
    assert "wedged" in (root / "state" / "cancel_intents.json").read_text(encoding="utf-8")


def test_the_name_tier_is_recorded_from_the_boot_probe(tmp_path, monkeypatch):
    from ouroboros import platform_layer

    monkeypatch.setattr(platform_layer, "kernel_file_locks_enforced", lambda path, **kw: False)
    root = fixture_root(tmp_path)  # built under the same tier the probe reports (one tier per directory)
    assert ensure_current_store(root, with_children=False)["lock_tier"] == store.TIER_NAME
    assert store.probe(root) == ("present", store.TIER_NAME)
    assert load(root, "done1")["status"] == "failed"  # readable through the name lock


def test_the_boot_probe_retiers_an_existing_store_both_ways(tmp_path, monkeypatch):
    """The tier the kernel enforces today governs a store built under the other one: the
    header and ``meta.lock_tier`` follow it in place, the rows stay, the open mode follows."""
    from ouroboros import platform_layer

    root = fixture_root(tmp_path)
    monkeypatch.setattr(platform_layer, "kernel_file_locks_enforced", lambda path, **kw: True)
    assert ensure_current_store(root, with_children=False)["lock_tier"] == store.TIER_ENFORCED
    store.update_task(root, "done1", lambda current: {**current, "db_era": True})
    identity = store_import._mount_identity(root / "state")
    monkeypatch.setattr(store_import, "_mount_identity", lambda path: {**identity, "st_dev": identity["st_dev"] + 1})
    monkeypatch.setattr(platform_layer, "kernel_file_locks_enforced", lambda path, **kw: False)  # the directory moved to NFS
    report = ensure_current_store(root, with_children=False)
    assert (report["status"], report["lock_tier"], report["retiered_from"]) == ("unchanged", store.TIER_NAME, store.TIER_ENFORCED)
    assert store.probe(root) == ("present", store.TIER_NAME) and meta(root, "lock_tier")["tier"] == store.TIER_NAME
    assert load(root, "done1")["db_era"] is True and meta(root, "phase") == store.PUBLISHED
    locks: list = []
    real_name_lock = store.name_lock
    monkeypatch.setattr(store, "name_lock", lambda root_, name, **kw: locks.append(name) or real_name_lock(root_, name, **kw))
    store.update_task(root, "done1", lambda current: {**current, "name_era": True})
    assert locks == [store.LOCK_NAME]  # the write went through the name lock: the open mode followed the tier
    monkeypatch.setattr(store_import, "_mount_identity", lambda path: identity)
    monkeypatch.setattr(platform_layer, "kernel_file_locks_enforced", lambda path, **kw: True)  # and back
    report = ensure_current_store(root, with_children=False)
    assert (report["lock_tier"], report["retiered_from"]) == (store.TIER_ENFORCED, store.TIER_NAME)
    assert store.probe(root) == ("present", store.TIER_ENFORCED) and meta(root, "lock_tier")["tier"] == store.TIER_ENFORCED
    locks.clear()
    assert load(root, "done1")["name_era"] is True and locks == []  # kernel locks again: no name lock taken
    assert "retiered_from" not in ensure_current_store(root, with_children=False)


_SIBLING_WRITE = """
import sys
from ouroboros import store
with store.open_store(sys.argv[1], write=True) as txn:
    txn.set_meta("sibling_wrote", True)
    print(txn.conn.execute("PRAGMA journal_mode").fetchone()[0])
"""


@pytest.mark.parametrize("boot", [False, True], ids=["first_touch", "boot"])
@pytest.mark.parametrize("body_sha", [None, "b" * 40], ids=["no_body_sha", "body_sha"])
@pytest.mark.parametrize("before, after", [(store.TIER_ENFORCED, store.TIER_NAME), (store.TIER_NAME, store.TIER_ENFORCED)],
                         ids=["enforced_to_name", "name_to_enforced"])
def test_a_boot_retier_leaves_this_process_choosing_the_journal_its_siblings_choose(tmp_path, monkeypatch, before, after,
                                                                                     boot, body_sha):
    """A lock-tier re-record converts the file the way an export hands it off, but this process keeps
    running: its next write keeps the journal a sibling process (a worker) keeps, while this process's
    cached connection stays open across the sibling's write. Without a boot that is the rollback
    journal both open with connection-local PERSIST; a boot then chooses this runtime's journal for the
    new tier (``store_journal``), and both use that. Left in the handed-off set it wrote in DELETE mode
    for its whole life while the workers chose PERSIST, and on Windows a worker's open PERSIST journal
    then fails every DELETE commit of the server."""
    import sys

    from ouroboros import platform_layer
    from tests.group_run import run_in_group

    root = fixture_root(tmp_path)
    monkeypatch.setattr(platform_layer, "kernel_file_locks_enforced", lambda path, **kw: before == store.TIER_ENFORCED)
    ensure_current_store(root, with_children=False)
    identity = store_import._mount_identity(root / "state")
    monkeypatch.setattr(store_import, "_mount_identity", lambda path: {**identity, "st_dev": identity["st_dev"] + 1})
    monkeypatch.setattr(platform_layer, "kernel_file_locks_enforced", lambda path, **kw: after == store.TIER_ENFORCED)
    report = ensure_current_store(root, with_children=False, body_sha=body_sha, initialize_journal=boot)
    assert report["retiered_from"] == before  # the volume changed
    expected = store._journal_mode(store.store_path(root), after).lower() if boot else "persist"
    with store.open_store(root, write=True) as txn:
        txn.set_meta("server_wrote", True)
        mine = txn.conn.execute("PRAGMA journal_mode").fetchone()[0]
    try:
        sibling = run_in_group([sys.executable, "-c", _SIBLING_WRITE, str(root)], timeout=120,
                               cwd=str(pathlib.Path(__file__).resolve().parents[1]), env=dict(os.environ))
    finally:
        store.close_connections()
    assert sibling.returncode == 0, sibling.stdout + sibling.stderr
    assert mine == sibling.stdout.strip().splitlines()[-1] == expected


# ---- export -------------------------------------------------------------------------


@pytest.mark.parametrize("initialize_journal", [False, True], ids=["helper", "boot"])
def test_raced_newer_publication_keeps_older_boot_journal_authority(tmp_path, monkeypatch, initialize_journal):
    """A newer helper publishes after the old boot's absent probe, before its import lock.
    Only the explicit boot may demote that WAL file before opening it.
    """
    import contextlib
    import sqlite3
    import sys

    from tests.group_run import run_in_group

    root = tmp_path / "raced"
    path = store.store_path(root)
    path.parent.mkdir(parents=True)
    if store._journal_mode(path.parent, store.TIER_ENFORCED) != "WAL":
        pytest.skip("requires a native WAL-eligible SQLite runtime and local filesystem")
    real_lock = store.name_lock
    publications = []
    script = (
        "import sys; from ouroboros import store, store_import; "
        "store_import.ensure_current_store(sys.argv[1], with_children=False); "
        "print(store.wal_file(store.store_path(sys.argv[1]))); store.close_connections()"
    )

    @contextlib.contextmanager
    def lock(root_, name, **kwargs):
        if name == store_import.IMPORT_LOCK_NAME and not publications:
            result = run_in_group([sys.executable, "-c", script, str(root)], timeout=30,
                                  cwd=str(pathlib.Path(__file__).resolve().parents[1]), env=dict(os.environ))
            publications.append(result)
            assert result.returncode == 0, result.stdout + result.stderr
            assert result.stdout.strip().splitlines()[-1] == "True"
            assert store.wal_file(path)
        with real_lock(root_, name, **kwargs):
            yield

    monkeypatch.setattr(store, "name_lock", lock)
    # The parent models the older opening runtime; its helper above keeps the
    # actual eligible SQLite facts in an independent process.
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 50, 4))
    if initialize_journal:
        report = ensure_current_store(root, with_children=False, body_sha="raced-body",
                                      initialize_journal=True)
        assert report["journal_selected"] and not store.wal_file(path)
        with store.open_store(root, write=True) as txn:
            txn.set_meta("after_raced_boot", True)
            assert txn.meta("last_store_body")["sha"] == "raced-body"
            assert txn.conn.execute("PRAGMA journal_mode").fetchone()[0] == "persist"
    else:
        with pytest.raises(store.StoreError, match="journal is WAL"):
            ensure_current_store(root, with_children=False, body_sha="raced-body", initialize_journal=False)
        assert store.wal_file(path)
    assert len(publications) == 1

def test_the_export_reproduces_every_file_byte_for_byte(tmp_path):
    root = fixture_root(tmp_path)
    ensure_current_store(root)
    drive = root / "state" / "headless_tasks" / "root1" / "data"
    expected, expected_child = snapshot(root), snapshot(drive)
    for path in list((root / "task_results").glob("*.json")) + list((root / "state" / "obligations").glob("*.json")):
        if path.stem != "mismatch":
            path.unlink()
    report = export_facts(root)
    child = export_facts(drive)
    assert (report["results"], report["written"]) == (3, 3) and child["results"] == 2
    assert snapshot(root) == expected
    # a child drive never booted a classification, so it gains the full set of files; its rows are exact
    assert {k: v for k, v in snapshot(drive).items() if k.startswith("task_results")} == expected_child
    nonterminal = json.loads((drive / "state" / "obligations" / "nonterminal.json").read_text(encoding="utf-8"))
    assert list(nonterminal) == ["root1-r2"]
    assert export_facts(root)["written"] == 0  # identical bytes are left alone
    assert ensure_current_store(root)["status"] == "unchanged"  # the baseline followed the export


def test_derived_set_file_rows_are_never_imported_as_owner_held(tmp_path):
    """M7: a stale member of a derived set file (a task no longer running, a synthesis no
    document owes) imports nothing — membership comes from the current predicates only; the
    owner-held sets, the closing receipts of ``delegated_runs`` and the pool's ``pending_drives``
    rows import; the same at P6' after a result change."""
    root = fixture_root(tmp_path)
    add_set_member(root, "nonterminal", "done1", {"task_id": "done1"})  # stale: done1 failed
    add_set_member(root, "synthesis", "done1", {"task_id": "done1"})
    add_set_member(root, "pause_notices", "done1", {"task_id": "done1"})
    add_set_member(root, "delegated_runs", "done1", {"task_id": "done1"})  # derived member, no receipt
    add_set_member(root, "delegated_runs", "child1", {"task_id": "child1", "closed_runs": {"r9": {"settled": True}}})
    add_set_member(root, "unknowns", "task:x", {"reason": "cancel_latch_migration_failed"})
    ensure_current_store(root, with_children=False)

    def held(name):
        with store.open_store(root, write=False) as txn:
            return txn.set_rows(name)

    assert held("nonterminal") == held("synthesis") == held("pause_notices") == {}
    assert set(o.members(root, "nonterminal")) == {"root1"} and o.members(root, "synthesis") == {}
    assert set(o.members(root, "pause_notices")) == {"child1"}  # the document's own notices, from its column
    assert held("delegated_runs") == {"child1": {"task_id": "child1", "closed_runs": {"r9": {"settled": True}}}}
    assert "root1" in held("pending_drives") and held("unknowns") == {"task:x": {"reason": "cancel_latch_migration_failed"}}
    write_result(root, result("fresh2"))  # P6': the old body ran, the set files are re-read
    assert reconcile_legacy(root)["status"] == "reconciled"
    assert held("nonterminal") == held("synthesis") == {} and set(o.members(root, "nonterminal")) == {"root1"}


def test_p6_keeps_the_store_only_set_rows_and_moves_only_what_the_old_body_changed(tmp_path):
    """B4: the owner-held sets reconcile row by row against their own baseline — a row the store
    published since the import survives, a retirement the store made is not undone by a stale
    file, and only the rows the old body added or retired move."""
    root = fixture_root(tmp_path)  # custody_open file: run:r1, invocation:i1
    add_set_member(root, "unknowns", "task:x", {"reason": "cancel_latch_migration_failed"})
    ensure_current_store(root, with_children=False)
    o.add(root, "custody_open", "invocation:new", {"task_id": "root1", "request": {"n": 1}})  # store-only fact
    o.remove(root, "custody_open", "invocation:i1")  # a retirement this store published
    add_set_member(root, "custody_open", "invocation:old", {"task_id": "root1", "request": {"o": 1}})  # the old body's
    atomic_write_json(set_file_path(root, "unknowns"), {})  # the old body retired task:x
    write_result(root, result("fresh2"))
    report = reconcile_legacy(root)
    assert report["status"] == "reconciled" and report["set_rows"] == 2, report
    with store.open_store(root, write=False) as txn:
        assert set(txn.set_rows("custody_open")) == {"run:r1", "invocation:new", "invocation:old"}
        assert txn.set_rows("unknowns") == {}
    write_result(root, result("fresh3"))  # the files unchanged since: nothing moves, the store's rows stand
    assert reconcile_legacy(root)["set_rows"] == 0
    with store.open_store(root, write=False) as txn:
        assert set(txn.set_rows("custody_open")) == {"run:r1", "invocation:new", "invocation:old"}


def test_a_rebuild_does_not_readmit_a_pending_invocation_the_chain_has_since_bound_or_refused(tmp_path):
    """The exported ``custody_open`` set file is a snapshot; a request the retained chain bound to
    a run or definitely refused after the export (an old body's append) stays resolved at the
    rebuild — the chain decides, a request the chain never mentions keeps its file row."""
    from ouroboros import delegate_custody as c
    from ouroboros.store_switch import _retire

    root = fixture_root(tmp_path)
    ensure_current_store(root, with_children=False)
    for iid in ("inv-bound", "inv-refused", "inv-open"):
        assert c.emit(root, c.START_REQUESTED, {"invocation_id": iid, "task_id": "root1", "request": {"p": iid}})
    export_facts(root)
    chain = c.event_log_path(root)
    append_jsonl(chain, {"ts": "2026-01-02T00:00:00+00:00", "type": c.STARTED, "run_id": "run-old", "task_id": "root1",
                         "invocation_id": "inv-bound"})
    append_jsonl(chain, {"ts": "2026-01-02T00:00:01+00:00", "type": c.START_FAILED, "task_id": "root1",
                         "invocation_id": "inv-refused", "definite": True, "error": "refused"})
    add_set_member(root, "custody_open", "invocation:inv-file-only", {"request": {"type": c.START_REQUESTED,
                   "invocation_id": "inv-file-only", "task_id": "root1", "request": {"p": "f"}}})
    _retire(root)
    assert ensure_current_store(root, with_children=False)["status"] == "completed"
    assert sorted(row["invocation_id"] for row in c.pending_invocations(root)) == ["inv-file-only", "inv-open"]
    assert c.invocation_record(root, "inv-bound")["run_id"] == "run-old"
    assert c.invocation_record(root, "inv-refused")["state"] == "failed_definite"


def test_p6_applies_the_custody_events_an_old_body_appended_after_the_boundary_once(tmp_path):
    """B4: the import records the chain boundary and every own append moves it; a P6' reduces
    the legacy events after it (event-id dedup, a replayed request re-opens nothing) and moves
    the boundary again, so neither this store's rows nor an applied legacy row replays."""
    from ouroboros import delegate_custody as c
    from ouroboros.delegate_custody_current import CHAIN_BOUNDARY

    root = fixture_root(tmp_path)
    path = c.event_log_path(root)
    append_jsonl(path, {"ts": "2026-01-01T00:00:00+00:00", "type": c.STARTED, "run_id": "run-a", "task_id": "root1"})
    ensure_current_store(root, with_children=False)
    assert c.emit(root, c.STARTED, {"run_id": "run-b", "task_id": "root1"})  # this store's own row
    with store.open_store(root, write=False) as txn:
        assert txn.meta(CHAIN_BOUNDARY) == path.stat().st_size
    legacy = {"ts": "2026-01-02T00:00:00+00:00", "type": c.SETTLED, "event_id": "L" * 32, "run_id": "run-a",
              "task_id": "root1", "state": "succeeded", "cost_usd": 0.25, "spend_disclosed": True}
    append_jsonl(path, legacy)
    append_jsonl(path, legacy)  # the old body's own replay
    append_jsonl(path, {"ts": "2026-01-02T00:00:01+00:00", "type": c.START_REQUESTED, "invocation_id": "inv-z",
                        "task_id": "root1", "request": {"p": 1}})
    write_result(root, result("fresh2"))
    report = reconcile_legacy(root)
    assert report["status"] == "reconciled" and report["legacy_custody"] == 2, report
    runs = c.replay(root)
    assert runs["run-a"].settled and runs["run-a"].cost_usd == 0.25 and not runs["run-b"].settled
    assert [row["invocation_id"] for row in c.pending_invocations(root)] == ["inv-z"]
    assert sorted(run.run_id for run in c.open_runs(root)) == ["run-b"]
    write_result(root, result("fresh3"))
    assert reconcile_legacy(root)["legacy_custody"] == 0  # the boundary moved past them
    with store.open_store(root, write=False) as txn:
        assert txn.meta(CHAIN_BOUNDARY) == path.stat().st_size


def test_the_export_writes_derived_sets_in_full_from_the_columns(tmp_path):
    root = fixture_root(tmp_path)
    ensure_current_store(root, with_children=False)
    store.update_task(root, "done1", lambda current: {**current, "status": "running"})
    (root / "state" / "obligations" / "nonterminal.json").unlink()
    export_facts(root)
    nonterminal = json.loads((root / "state" / "obligations" / "nonterminal.json").read_text(encoding="utf-8"))
    assert list(nonterminal) == ["done1", "root1"]  # the derived members, from the columns (no stored row)
    pending = json.loads((root / "state" / "obligations" / "pending_drives.json").read_text(encoding="utf-8"))
    assert set(pending) == {"root1", "done1"}


def test_the_export_writes_the_whole_document_once_fields_live_in_rows(tmp_path):
    root = fixture_root(tmp_path)
    claim = {"tool": "write_file", "task_id": "done1", "state": "claimed"}
    write_result(root, result("done1", "failed", launch_handoffs={"op1": claim}, model_waits={}, tail=1))
    original = task_result_path(root, "done1").read_bytes()
    ensure_current_store(root, with_children=False)
    with store.open_store(root, write=False) as txn:
        body = json.loads(txn.body_text("done1"))
        assert body["launch_handoffs"] == {} and body["model_waits"] == {}  # the keys keep their place, emptied
        assert txn.handoffs("done1") == {"launch_handoffs": {"op1": claim}, "model_waits": {},
                                         "retired_model_consumers": {}, "swarm_fanouts": {}}
        assert txn.load("done1")["launch_handoffs"] == {"op1": claim}
        assert "retired_model_consumers" not in txn.load("done1")  # a key the file lacked stays absent
        assert "swarm_fanouts" not in txn.load("done1")
    for task_id in ("root1", "child1", "done1"):
        task_result_path(root, task_id).unlink()
    export_facts(root)
    assert task_result_path(root, "done1").read_bytes() == original  # rows back in place: byte for byte
    for task_id in ("root1", "child1"):
        written = json.loads(task_result_path(root, task_id).read_text(encoding="utf-8"))
        assert written == load(root, task_id)  # the exact inverse of load


def test_the_export_removes_stale_admissible_files_and_keeps_the_evidence(tmp_path):
    root = fixture_root(tmp_path)
    ensure_current_store(root, with_children=False)
    with store.open_store(root, write=True) as txn:
        txn.delete_task("done1")  # a row the store no longer holds: its file is stale
    atomic_write_json(task_result_path(root, "stray"), result("stray"))  # admissible, unknown to the store
    report = export_facts(root)
    assert report["removed"] == 2 and meta(root, "export_manifest")["removed"] == ["done1", "stray"]
    assert not task_result_path(root, "done1", create=False).exists()
    assert not task_result_path(root, "stray", create=False).exists()
    assert task_result_path(root, "mismatch", create=False).exists()  # refused: evidence, kept
    assert (root / "task_results" / "quarantine" / "legacy.json").exists()
    assert "done1" not in meta(root, "baseline") and "mismatch" in meta(root, "baseline")
    assert ensure_current_store(root, with_children=False)["status"] == "unchanged"


def test_the_export_manifest_is_durable_before_the_first_write(tmp_path, monkeypatch):
    root = fixture_root(tmp_path)
    ensure_current_store(root, with_children=False)
    store.update_task(root, "done1", lambda current: {**current, "db_era": True})
    for path in (root / "task_results").glob("*.json"):
        path.unlink()
    monkeypatch.setattr(store_import, "write_text_atomic", lambda *a, **k: (_ for _ in ()).throw(OSError("disk went away")))
    with pytest.raises(OSError):
        export_facts(root)
    manifest = meta(root, "export_manifest")
    assert manifest["status"] == "writing" and sorted(manifest["results"]) == ["child1", "done1", "root1"]
    expected = store.document_text(load(root, "done1")).encode("utf-8")
    assert manifest["results"]["done1"] == {"size": len(expected), "sha256": hashlib.sha256(expected).hexdigest()}
    assert set(manifest["sets"]) >= set(o.BOOT_SETS) and manifest["removed"] == []
    # the abandonment: nothing was written, nothing is adopted; the baseline stays as it was
    with store.open_store(root, write=True) as txn:
        assert store_import.absorb_own_export(root, txn) == {"adopted": 0, "planned": 3}
    assert meta(root, "export_manifest")["status"] == "abandoned"


# ---- per-row reconciliation (P6') -------------------------------------------------------

def test_a_changed_file_replaces_its_row_and_a_touched_file_does_not(tmp_path):
    root = fixture_root(tmp_path)
    ensure_current_store(root, with_children=False)
    path = task_result_path(root, "done1")
    os.utime(path, ns=(1, 1))  # timestamps alone prove nothing
    assert reconcile_legacy(root)["status"] == "unchanged"
    atomic_write_json(path, result("done1", "completed", note="an old body wrote this"))
    report = reconcile_legacy(root)
    assert report["status"] == "reconciled" and report["changed"] == 1 and report["conflicts"] == 0
    assert load(root, "done1")["note"] == "an old body wrote this"
    assert meta(root, "baseline")["done1"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert reconcile_legacy(root)["status"] == "unchanged"


def test_the_reconciliation_reads_only_stat_changed_files_and_compares_them_by_content(tmp_path, monkeypatch):
    root = fixture_root(tmp_path)
    ensure_current_store(root)  # publish children before measuring the whole-tree maintenance pass
    read: list = []
    real_read = store_import._read_result
    monkeypatch.setattr(store_import, "_read_result", lambda path: read.append(path.stem) or real_read(path))
    assert reconcile_legacy(root)["status"] == "unchanged"
    assert read == []  # the budget: one listing plus one stat per file, no reads while nothing moved
    path = task_result_path(root, "done1")
    os.utime(path, ns=(5, 5))  # touched, same bytes: read once, compared by digest, the stored stat refreshed
    touched_mtime = path.stat().st_mtime_ns  # the filesystem may round to its own resolution
    assert reconcile_legacy(root)["status"] == "unchanged"
    assert read == ["done1"] and meta(root, "baseline")["done1"]["mtime_ns"] == touched_mtime
    read.clear()
    assert reconcile_legacy(root)["status"] == "unchanged" and read == []
    text = path.read_text(encoding="utf-8")
    assert json.loads(text)["status"] == "failed" and text.count('"failed"') == 1
    path.write_text(text.replace('"failed"', '"passed"'), encoding="utf-8")  # same length, new mtime, any JSON layout
    report = reconcile_legacy(root)
    assert report["status"] == "reconciled" and report["changed"] == 1 and read == ["done1"]
    assert load(root, "done1")["status"] == "passed"


def test_a_read_error_aborts_the_import_typed_and_stamps_nothing(tmp_path, monkeypatch):
    root = fixture_root(tmp_path)
    real_read_bytes = pathlib.Path.read_bytes

    def exhausted(self):
        if self.name == "done1.json":
            raise OSError(errno.EMFILE, "Too many open files")
        return real_read_bytes(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", exhausted)
    with pytest.raises(store.StoreBusy, match="EMFILE"):
        ensure_current_store(root, with_children=False)
    assert store.probe(root)[0] == "absent" and not store.generation_stamped(root)
    assert not list((root / "state").glob("current.sqlite.import-*"))
    monkeypatch.setattr(pathlib.Path, "read_bytes", real_read_bytes)
    report = ensure_current_store(root, with_children=False)
    assert report["status"] == "completed" and report["results"] == 3 and report["refused"] == 1  # only the schema refusal
    # the same on a published store (the file touched, so it is read): the reconciliation aborts typed, the rows stay
    os.utime(task_result_path(root, "done1"), ns=(7, 7))
    monkeypatch.setattr(pathlib.Path, "read_bytes", exhausted)
    with pytest.raises(store.StoreBusy):
        reconcile_legacy(root)
    monkeypatch.setattr(pathlib.Path, "read_bytes", real_read_bytes)
    assert load(root, "done1")["status"] == "failed"


def test_a_new_file_is_imported_and_a_deleted_file_moves_its_row_to_quarantine(tmp_path):
    root = fixture_root(tmp_path)
    ensure_current_store(root, with_children=False)
    write_result(root, result("fresh1"))
    task_result_path(root, "child1").unlink()
    report = reconcile_legacy(root)
    assert (report["new"], report["deleted"]) == (1, 1)
    assert load(root, "fresh1")["status"] == "completed" and load(root, "child1") is None
    copies = list((root / "task_results" / "quarantine").glob("child1.deleted-*.json"))
    assert len(copies) == 1 and json.loads(copies[0].read_text(encoding="utf-8"))["nested"] == {"k": [1, 2.5, "ü"]}
    events = rows(root / "logs" / "events.jsonl", "task_results_quarantined")
    assert events[-1]["reasons"] == {"deleted_by_old_body": 1}
    assert "child1" not in meta(root, "baseline")


def test_a_row_changed_on_both_sides_lets_the_file_win_and_keeps_the_db_version(tmp_path):
    root = fixture_root(tmp_path)
    ensure_current_store(root, with_children=False)
    store.update_task(root, "done1", lambda current: {**current, "db_side": True})
    atomic_write_json(task_result_path(root, "done1"), result("done1", "completed", file_side=True))
    report = reconcile_legacy(root)
    assert report["conflicts"] == 1
    assert load(root, "done1") == result("done1", "completed", file_side=True)
    conflict = meta(root, "conflicts")["done1"]
    assert json.loads(conflict["db_body"])["db_side"] is True
    # a db-only change with an untouched file is kept (no conflict, no replacement)
    store.update_task(root, "child1", lambda current: {**current, "db_only": True})
    assert reconcile_legacy(root)["status"] == "unchanged"
    assert load(root, "child1")["db_only"] is True


def test_p6_compares_and_preserves_the_whole_projected_document(tmp_path):
    """B5: a handoff row the store wrote is part of its version — a file change over it is a
    conflict whose copy carries the rows; a file an old body deleted leaves a quarantine copy of
    the whole document (rows included), saved before the task and its handoffs go; an
    unchanged document with handoffs is no false conflict."""
    claim = {"kind": "launch", "ts": "2026-01-01T00:00:00+00:00"}
    root = fixture_root(tmp_path)
    write_result(root, result("done1", "failed", launch_handoffs={"op1": claim}))
    write_result(root, result("child1", "completed", launch_handoffs={"op1": claim}))
    ensure_current_store(root, with_children=False)
    with store.open_store(root, write=True) as txn:  # a row-only mutation: the body text is untouched
        txn.put_handoff("done1", "launch_handoffs", "op2", {**claim, "second": True})
        txn.touch("done1")
    atomic_write_json(task_result_path(root, "done1"), result("done1", "failed", launch_handoffs={"op1": claim}, file_side=True))
    atomic_write_json(task_result_path(root, "child1"), result("child1", "completed", launch_handoffs={"op1": claim}, file_side=True))
    task_result_path(root, "root1").unlink()
    report = reconcile_legacy(root)
    assert (report["changed"], report["deleted"], report["conflicts"]) == (2, 1, 1), report
    conflict = meta(root, "conflicts")["done1"]
    assert set(json.loads(conflict["db_body"])["launch_handoffs"]) == {"op1", "op2"}
    assert "child1" not in meta(root, "conflicts")  # the file's change over an unchanged document: no conflict
    assert load(root, "done1")["launch_handoffs"] == {"op1": claim}
    copies = list((root / "task_results" / "quarantine").glob("root1.deleted-*.json"))
    assert len(copies) == 1 and json.loads(copies[0].read_text(encoding="utf-8"))["task_id"] == "root1"
    write_result(root, result("fresh2", launch_handoffs={"op1": claim}))
    reconcile_legacy(root)
    with store.open_store(root, write=True) as txn:
        txn.put_handoff("fresh2", "launch_handoffs", "op2", claim)
        txn.touch("fresh2")
    task_result_path(root, "fresh2").unlink()
    reconcile_legacy(root)
    copies = list((root / "task_results" / "quarantine").glob("fresh2.deleted-*.json"))
    assert len(copies) == 1 and set(json.loads(copies[0].read_text(encoding="utf-8"))["launch_handoffs"]) == {"op1", "op2"}


def test_a_refused_or_quarantined_identity_stays_occupied_for_strict_reads_and_create_only_writes(tmp_path):
    """ABI-2 (M10): the import records the identities it refused or quarantined; a strict read
    refuses them and a create-only write does not readmit them as a fresh absence — from the
    store's record, not a quarantine-directory scan (the directory is gone here). A free
    identity is still created; a plain read still sees absence; an admissible file at P6'
    re-opens the identity."""
    from ouroboros.task_results import load_task_result, write_task_result

    root = fixture_root(tmp_path)  # legacy: quarantined at import; mismatch: refused
    ensure_current_store(root, with_children=False)
    shutil.rmtree(root / "task_results" / "quarantine")
    for task_id in ("legacy", "mismatch"):
        assert load_task_result(root, task_id) is None
        with pytest.raises(ValueError, match="unreadable or invalid"):
            load_task_result(root, task_id, strict=True)
        with pytest.raises(ValueError, match="unreadable or invalid"):
            write_task_result(root, task_id, "scheduled", create_only=True, strict_existing_dict=True)
        assert load(root, task_id) is None
    assert write_task_result(root, "fresh9", "scheduled", create_only=True, strict_existing_dict=True)["status"] == "scheduled"
    write_result(root, result("legacy", "completed"))  # the old body wrote an admissible file under the identity
    assert reconcile_legacy(root)["new"] == 1
    assert load_task_result(root, "legacy", strict=True)["status"] == "completed"
    atomic_write_json(task_result_path(root, "done1"), {"task_id": "done1", "status": "completed"})  # unstamped: quarantined at P6'
    assert reconcile_legacy(root)["quarantined"] == 1
    with pytest.raises(ValueError, match="unreadable or invalid"):
        write_task_result(root, "done1", "scheduled", create_only=True, strict_existing_dict=True)


def test_set_file_edits_alone_are_not_detected_but_follow_a_result_change(tmp_path):
    root = fixture_root(tmp_path)
    ensure_current_store(root, with_children=False)
    add_set_member(root, "custody_open", "run:r2", {"task_id": "done1"})
    assert reconcile_legacy(root)["status"] == "unchanged"  # the named limit
    with store.open_store(root, write=False) as txn:
        assert "run:r2" not in txn.set_rows("custody_open")
    write_result(root, result("fresh2"))
    reconcile_legacy(root)
    with store.open_store(root, write=False) as txn:
        assert "run:r2" in txn.set_rows("custody_open")


def test_an_inadmissible_changed_file_is_quarantined_with_its_row(tmp_path):
    root = fixture_root(tmp_path)
    ensure_current_store(root, with_children=False)
    path = task_result_path(root, "done1")
    path.write_text("{not json", encoding="utf-8")
    report = reconcile_legacy(root)
    assert report["quarantined"] == 1 and load(root, "done1") is None
    assert not path.exists() and (root / "task_results" / "quarantine" / "done1.json").exists()


# ---- a full rebuild only after a proven complete export ---------------------------------

def test_an_unfinished_export_keeps_the_store_as_authority(tmp_path):
    root = fixture_root(tmp_path)
    ensure_current_store(root, with_children=False)
    with store.open_store(root, write=True) as txn:
        txn.set_meta("phase", store.EXPORTING)
        txn.set_meta("export", {"status": "exporting"})
    write_result(root, result("fresh3"))  # a change an old body could never have made here
    report = ensure_current_store(root, with_children=False)
    assert (report["status"], report["phase"]) == ("switch_pending", "exporting")
    assert store.probe(root)[0] == "present" and load(root, "fresh3") is None


def test_a_completed_export_whose_retirement_never_finished_is_set_aside_and_rebuilt(tmp_path):
    root = fixture_root(tmp_path)
    ensure_current_store(root, with_children=False)
    with store.open_store(root, write=True) as txn:
        txn.set_meta("phase", store.RETIRED)
        txn.set_meta("export", {"status": "completed"})
    write_result(root, result("fresh4"))
    report = ensure_current_store(root, with_children=False)
    assert report["status"] == "completed" and report["results"] == 4
    assert load(root, "fresh4")["status"] == "completed"
    aside = list((root / "state").glob("current.sqlite.exported-*"))
    assert len(aside) == 1
    with sqlite3.connect(aside[0]) as conn:
        assert json.loads(conn.execute("SELECT value FROM meta WHERE key='phase'").fetchone()[0]) == "retired"
