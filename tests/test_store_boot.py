"""Boot SQL work ignores frozen history; compatibility checks only child file headers."""
from __future__ import annotations

import pytest

import json
import os
import pathlib
import shutil
import sqlite3
from collections import Counter

from ouroboros import store, store_import
from tests.test_store_import import load, result, write_result


def _boot_fixture(root, count, children, template):
    from ouroboros import startup_migrations

    directory = root / "task_results"
    directory.mkdir(parents=True)
    for index in range(count):
        (directory / f"old-{index}.json").write_text(json.dumps(result(f"old-{index}")), encoding="utf-8")
    for index in range(children):
        state = root / "task_drives" / f"drive-{index:03}" / "state"
        state.mkdir(parents=True)
        for filename in ("current.sqlite", "migrations.json"):
            shutil.copyfile(template / "state" / filename, state / filename)
    startup_migrations.prepare_startup_state(root, initialize_journal=True)
    with store.open_store(root, write=False) as txn:
        sha = txn.meta("last_store_body")["sha"]
    (root / "state/state.json").write_text(json.dumps({"current_sha": sha}), encoding="utf-8")


def _boot_cost(root, monkeypatch):
    from ouroboros import platform_layer, startup_migrations

    counts = Counter(connects=0, child_connects=0, legacy_stats=0, legacy_scans=0,
                     write_transactions=0, write_commits=0, commits=0, vm_steps=0, lock_probes=0)
    connect, stat, scan, opening = sqlite3.connect, os.stat, os.scandir, os.open

    def measured_connect(path, *args, **kwargs):
        counts["connects"] += 1
        counts["child_connects"] += int("/task_drives/" in str(path))
        conn = connect(path, *args, **kwargs)
        def trace(sql):
            counts["write_transactions"] += int(sql == "BEGIN IMMEDIATE")
            if sql == "COMMIT":
                counts["commits"] += 1
                counts["write_commits"] += int(conn.total_changes > 0)
        conn.set_trace_callback(trace)
        conn.set_progress_handler(lambda: counts.update(vm_steps=1) or 0, 1)
        return conn

    def measured_stat(path, *args, **kwargs):
        counts["legacy_stats"] += int("task_results" in pathlib.Path(path).parts) if not isinstance(path, int) else 0
        return stat(path, *args, **kwargs)

    def measured_scan(path):
        counts["legacy_scans"] += int("task_results" in pathlib.Path(path).parts)
        return scan(path)

    def measured_open(path, *args, **kwargs):
        counts["lock_probes"] += int(".kernel-lock-probe." in str(path) and pathlib.Path(path).parent == root / "state")
        return opening(path, *args, **kwargs)

    # Discard process-local probe state and cached connections, as on a cold process start.
    platform_layer._KERNEL_LOCK_TIER.clear()
    store.close_connections()
    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", measured_connect)
        patch.setattr(os, "stat", measured_stat)
        patch.setattr(os, "scandir", measured_scan)
        patch.setattr(os, "open", measured_open)
        if accessor := getattr(pathlib.Path, "_accessor", None):  # Python 3.10 keeps bound OS functions here.
            patch.setattr(accessor, "stat", measured_stat)
            patch.setattr(accessor, "scandir", measured_scan)
        report = startup_migrations.prepare_startup_state(root, initialize_journal=True)
    assert report["status"] == "unchanged" and report["drives"] == {}
    assert report["opened_ms"] >= 0
    return counts


def test_second_boot_has_constant_sql_work_and_no_history_or_writes(tmp_path, monkeypatch):
    template = tmp_path / "template"
    store_import.ensure_current_store(template, with_children=False)
    costs = []
    for count, children in ((50, 20), (500, 200)):
        root = tmp_path / str(count)
        _boot_fixture(root, count, children, template)
        costs.append(_boot_cost(root, monkeypatch))
    assert costs[0] == costs[1], costs
    assert costs[0]["connects"] == costs[0]["commits"] == 1
    assert all(costs[0][name] == 0 for name in (
        "child_connects", "legacy_stats", "legacy_scans", "write_transactions", "write_commits", "lock_probes")), costs


@pytest.mark.parametrize("tier", [store.TIER_NAME, store.TIER_ENFORCED])
def test_unchanged_mount_never_reprobes_even_in_a_fresh_process(tmp_path, monkeypatch, tier):
    from ouroboros import platform_layer

    real_lock = platform_layer.file_lock_exclusive_nb
    if tier == store.TIER_NAME:
        import errno
        def unsupported(_fd):
            raise OSError(errno.ENOTSUP, "synthetic mount without kernel locks")
        monkeypatch.setattr(platform_layer, "file_lock_exclusive_nb", unsupported)
    store_import.ensure_current_store(tmp_path, with_children=False)
    monkeypatch.setattr(platform_layer, "file_lock_exclusive_nb", real_lock)
    platform_layer._KERNEL_LOCK_TIER.clear()
    opening = os.open
    def no_probe(path, *args, **kwargs):
        assert ".kernel-lock-probe." not in str(path)
        return opening(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(os, "open", no_probe)
        assert store_import.ensure_current_store(tmp_path, with_children=False)["lock_tier"] == tier


@pytest.mark.parametrize("changed", ["st_dev", "mount_point"])
def test_changed_mount_reprobes_despite_process_cache(tmp_path, monkeypatch, changed):
    import errno
    from ouroboros import platform_layer

    store_import.ensure_current_store(tmp_path, with_children=False)
    identity = store_import._mount_identity(tmp_path / "state")
    moved = {**identity, changed: identity[changed] + 1 if changed == "st_dev" else str(tmp_path)}
    monkeypatch.setattr(store_import, "_mount_identity", lambda _: moved)
    opening, probes = os.open, []
    def measured(path, *args, **kwargs):
        if ".kernel-lock-probe." in str(path) and pathlib.Path(path).parent == tmp_path / "state":
            probes.append(path)
        return opening(path, *args, **kwargs)
    def unsupported(_fd):
        raise OSError(errno.ENOLCK, "synthetic changed mount")
    monkeypatch.setattr(os, "open", measured)
    monkeypatch.setattr(platform_layer, "file_lock_exclusive_nb", unsupported)
    report = store_import.ensure_current_store(tmp_path, with_children=False)
    assert report["retiered_from"] == store.TIER_ENFORCED and report["lock_tier"] == store.TIER_NAME
    assert len(probes) == 1
    platform_layer._KERNEL_LOCK_TIER.clear()
    store_import.ensure_current_store(tmp_path, with_children=False)
    assert len(probes) == 1  # the new identity is now durable, including its refusal errno
    assert platform_layer._KERNEL_LOCK_TIER[os.path.realpath(tmp_path / "state")] == (False, errno.ENOLCK)


def test_unprobeable_mount_is_not_persisted_as_verified(tmp_path, monkeypatch):
    (tmp_path / "state").mkdir()
    opening = os.open
    def unavailable(path, *args, **kwargs):
        if ".kernel-lock-probe." in str(path):
            raise PermissionError("injected probe-file failure")
        return opening(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(os, "open", unavailable)
        unverified = store_import._lock_tier(tmp_path)
    assert "st_dev" not in unverified and "mount_point" not in unverified
    verified = store_import._lock_tier(tmp_path, unverified)
    assert verified["tier"] == store.TIER_ENFORCED
    assert {key: verified[key] for key in ("st_dev", "mount_point")} == store_import._mount_identity(tmp_path / "state")


def test_flag_probe_work_does_not_grow_with_classified_history(tmp_path):
    work = []
    for count in (50, 500):
        root = tmp_path / str(count)
        store_import.ensure_current_store(root, with_children=False)
        with store.open_store(root, write=True) as txn:
            for index in range(count):
                txn.store_task(result(f"old-{index}"))
        with store.open_store(root, write=False) as txn:
            steps = []
            txn.conn.set_progress_handler(lambda: steps.append(1) or 0, 1)
            assert not store_import._flags_need_derivation(txn)
            txn.conn.set_progress_handler(None, 0)
            work.append(len(steps))
    assert work[0] == work[1]


def test_clean_flags_and_schema_skip_derivation_and_ddl(tmp_path, monkeypatch):
    write_result(tmp_path, result("task"))
    store_import.ensure_current_store(tmp_path)
    monkeypatch.setattr(store_import, "_derive_flag_columns", lambda *_a, **_kw: pytest.fail("derived clean flags"))
    monkeypatch.setattr(store_import, "_upgrade_review_collection", lambda *_: pytest.fail("upgraded current schema"))
    assert store_import.ensure_current_store(tmp_path)["status"] == "unchanged"


@pytest.mark.parametrize("missing", ["index", "column", "null"])
def test_old_writer_flags_are_prepared_before_use(tmp_path, missing):
    if missing == "column":
        from tests.test_review_collection_index import _legacy_store
        _legacy_store(tmp_path, result("task"), owned_root=tmp_path)
    else:
        write_result(tmp_path, result("task"))
        store_import.ensure_current_store(tmp_path)
        with store.open_store(tmp_path, write=True) as txn:
            txn.conn.execute("DROP INDEX tasks_underived_flags" if missing == "index" else
                             "UPDATE tasks SET has_owner_questions=NULL")
    with store.open_store(tmp_path, write=False) as txn:
        assert store_import._flags_need_derivation(txn)
    store_import.ensure_current_store(tmp_path)
    with store.open_store(tmp_path, write=False) as txn:
        assert not store_import._flags_need_derivation(txn)
        assert txn.row("task", "has_owner_questions") == 0
        assert txn.meta("schema_version") == store.SCHEMA_VERSION
    assert load(tmp_path, "task") == result("task")
