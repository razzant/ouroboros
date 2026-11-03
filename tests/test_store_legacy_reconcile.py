"""Old-body detection, resumable reconciliation and the explicit foreground command."""
from __future__ import annotations

import ast
import json
import os
import pathlib
import subprocess
import sys

import pytest

from ouroboros import maintenance_jobs, startup_migrations, store, store_import
from ouroboros.utils import atomic_write_json
from tests.test_store_import import load, meta, result, write_result


def _body(root, sha="store-head"):
    atomic_write_json(root / "state/state.json", {"current_sha": sha})


def _published(root):
    _body(root)
    store_import.ensure_current_store(root, body_sha="store-head")
    with store.open_store(root, write=True) as txn:
        txn.conn.execute("DELETE FROM meta WHERE key='job:legacy_reconcile'")


@pytest.mark.parametrize("case", ["matching", "different", "missing_marker", "unreadable_state", "missing_state"])
def test_previous_body_detector_both_directions(tmp_path, monkeypatch, case):
    _published(tmp_path)
    before = meta(tmp_path, "last_store_body")
    if case == "different":
        _body(tmp_path, "old-body")
    elif case == "missing_marker":
        with store.open_store(tmp_path, write=True) as txn:
            txn.conn.execute("DELETE FROM meta WHERE key='last_store_body'")
    elif case == "missing_state":
        (tmp_path / "state/state.json").unlink()
    elif case == "unreadable_state":
        read = pathlib.Path.read_text
        def unreadable(path, *args, **kwargs):
            if path == tmp_path / "state/state.json":
                raise PermissionError("injected unreadable state")
            return read(path, *args, **kwargs)
        monkeypatch.setattr(pathlib.Path, "read_text", unreadable)
    report = store_import.ensure_current_store(tmp_path, body_sha="store-head")
    job = maintenance_jobs.job(tmp_path, "legacy_reconcile")
    if case == "matching":
        assert job is None and report["legacy_reconcile"] == "not_needed"
        assert meta(tmp_path, "last_store_body") == before
    else:
        assert job["state"] == "pending" and report["legacy_reconcile"] == "scheduled"
        assert job["done"] == 0 and job["cursor"] is None
    assert meta(tmp_path, "last_store_body")["sha"] == "store-head"


@pytest.mark.parametrize("state_doc", [{"current_sha": None}, {}, {"current_sha": "stale-desktop-body"}])
def test_a_bootstrap_that_never_stamps_reconciles_once_not_every_boot(tmp_path, state_doc):
    """Docker images, Colab and source checkouts start without stamping ``current_sha``: every boot
    shows the same unknown (or stale) stamp, and one finished reconcile covers it (owner answer Q2)."""
    atomic_write_json(tmp_path / "state/state.json", {"session_id": "boot", **state_doc})
    seen = []
    for _boot in range(3):
        report = store_import.ensure_current_store(tmp_path, body_sha="same-head")
        seen.append(report["legacy_reconcile"])
        if (maintenance_jobs.job(tmp_path, "legacy_reconcile") or {}).get("state") == "pending":
            store_import.run_legacy_reconcile(tmp_path)
    assert seen == ["scheduled", "not_needed", "not_needed"]
    before = maintenance_jobs.job(tmp_path, "legacy_reconcile")
    store_import.ensure_current_store(tmp_path, body_sha="same-head")
    assert maintenance_jobs.job(tmp_path, "legacy_reconcile") == before, "a steady boot writes no job row"
    _body(tmp_path, "old-body")  # a launcher-managed old body stamped its own SHA meanwhile
    assert store_import.ensure_current_store(tmp_path, body_sha="same-head")["legacy_reconcile"] == "scheduled"
    assert maintenance_jobs.job(tmp_path, "legacy_reconcile")["observed_sha"] == "old-body"


@pytest.mark.parametrize("between", ["no_boot", "a_steady_boot"])
def test_a_second_excursion_to_the_same_older_body_is_reconciled_too(tmp_path, between):
    """Older body X runs, the store body boots (its bootstrap stamps itself), then X runs again, with
    or without another boot of the store body in between: the second visit is a new excursion, not
    the observation the first reconcile covered; a steady boot writes no job row."""
    write_result(tmp_path, result("task", note="initial"))
    _published(tmp_path)
    for visit in (1, 2):
        write_result(tmp_path, result("task", note=f"older body visit {visit}"))
        _body(tmp_path, "old-body")
        assert store_import.ensure_current_store(tmp_path, body_sha="store-head")["legacy_reconcile"] == "scheduled"
        _body(tmp_path, "store-head")  # the bootstrap after the detector: the store body stamps itself
        store_import.note_current_stamp(tmp_path)
        assert store_import.run_legacy_reconcile(tmp_path)["state"] == "done"
        assert load(tmp_path, "task")["note"] == f"older body visit {visit}"
        if between == "a_steady_boot":
            before = maintenance_jobs.job(tmp_path, "legacy_reconcile")
            assert store_import.ensure_current_store(tmp_path, body_sha="store-head")["legacy_reconcile"] == "not_needed"
            store_import.note_current_stamp(tmp_path)
            assert maintenance_jobs.job(tmp_path, "legacy_reconcile") == before, "a steady boot writes no job row"


def test_regular_update_changes_body_stamp_without_scheduling_legacy(tmp_path):
    _published(tmp_path)
    report = store_import.ensure_current_store(tmp_path, body_sha="new-store-head")
    assert report["legacy_reconcile"] == "not_needed"
    assert maintenance_jobs.job(tmp_path, "legacy_reconcile") is None
    assert meta(tmp_path, "last_store_body")["sha"] == "new-store-head"


def test_old_body_edits_wait_for_job_and_reach_root_and_child(tmp_path, monkeypatch):
    child = tmp_path / "task_drives/child"
    for drive in (tmp_path, child):
        write_result(drive, result("task", note="frozen"))
    _published(tmp_path)
    for drive in (tmp_path, child):
        write_result(drive, result("task", note="old body edited this"))
    _body(tmp_path, "old-body")
    report = store_import.ensure_current_store(tmp_path, body_sha="store-head")
    assert report["legacy_reconcile"] == "scheduled"
    assert all(load(drive, "task")["note"] == "frozen" for drive in (tmp_path, child))
    progress = []
    job = store_import.run_legacy_reconcile(tmp_path, progress=progress.append, should_stop=lambda: False)
    assert job["state"] == "done" and job["done"] == job["total"] == 2
    assert progress[-1] == job
    assert all(load(drive, "task")["note"] == "old body edited this" for drive in (tmp_path, child))
    monkeypatch.setattr(store_import, "descendant_drives", lambda *_: pytest.fail("repeated a finished job"))
    assert store_import.run_legacy_reconcile(tmp_path) == job


@pytest.mark.parametrize("interruption", ["stop", "error"])
def test_stop_after_three_drives_resumes_without_reopening_finished_drives(tmp_path, monkeypatch, interruption):
    drives = [tmp_path, *(tmp_path / "task_drives" / f"child-{index}" for index in range(5))]
    for drive in drives:
        write_result(drive, result("task", note="frozen"))
    _published(tmp_path)
    for drive in drives:
        write_result(drive, result("task", note="changed"))
    _body(tmp_path, "old-body")
    store_import.ensure_current_store(tmp_path, body_sha="store-head")
    reconciled, progress = [], []
    reconcile = store_import._reconcile
    def tracked(root, txn):
        if interruption == "error" and root == drives[3] and len(reconciled) == 3:
            raise store.StoreBusy("injected drive unavailable")
        reconciled.append(root)
        return reconcile(root, txn)
    monkeypatch.setattr(store_import, "_reconcile", tracked)
    if interruption == "error":
        with pytest.raises(store.StoreBusy, match="injected"):
            store_import.run_legacy_reconcile(tmp_path, progress=progress.append)
        row = maintenance_jobs.job(tmp_path, "legacy_reconcile")
        assert row["state"] == "failed" and row["waiting_on"] == {"drive": str(drives[3])}
    else:
        row = store_import.run_legacy_reconcile(tmp_path, progress=progress.append,
            should_stop=lambda: bool(progress and progress[-1]["done"] == 3))
        assert row["state"] == "pending"
    assert row["done"] == 3
    assert row["cursor"] == drives[2].relative_to(tmp_path).as_posix()
    # Boot before git bootstrap still sees the same old-body SHA. It preserves the cursor.
    store_import.ensure_current_store(tmp_path, body_sha="store-head")
    assert maintenance_jobs.job(tmp_path, "legacy_reconcile")["cursor"] == row["cursor"]
    interruption = "repaired"
    opened, opening = [], store.open_store
    from contextlib import contextmanager
    @contextmanager
    def tracked_open(root, **kwargs):
        opened.append(pathlib.Path(root))
        with opening(root, **kwargs) as txn:
            yield txn
    with monkeypatch.context() as patch:
        patch.setattr(store, "open_store", tracked_open)
        row = store_import.run_legacy_reconcile(tmp_path, progress=progress.append, should_stop=lambda: False)
    assert row["state"] == "done" and row["done"] == row["total"] == len(drives)
    assert row["error"] is None and row["waiting_on"] is None
    assert reconciled == drives
    assert not set(drives[1:3]) & set(opened)
    assert all(load(drive, "task")["note"] == "changed" for drive in drives)


def test_cli_prints_progress_and_explicitly_runs_again(tmp_path):
    write_result(tmp_path, result("task"))
    _published(tmp_path)
    for index in range(2):
        write_result(tmp_path, result("task", edit=index))
        command = [sys.executable, "-m", "ouroboros.store_import", "--reconcile-legacy"]
        if index == 0:
            command += ["--root", str(tmp_path)]
        completed = subprocess.run(command, capture_output=True, text=True,
                                   env={**os.environ, "OUROBOROS_DATA_DIR": str(tmp_path)})
        assert completed.returncode == 0, completed.stderr
        lines = [json.loads(line) for line in completed.stdout.splitlines()]
        assert lines[0]["state"] == "running" and lines[-1]["state"] == "done"
        assert lines[-1]["done"] == lines[-1]["total"] == 1
        assert load(tmp_path, "task")["edit"] == index


def test_current_migration_stamps_read_no_history(tmp_path, monkeypatch):
    from ouroboros import obligations
    startup_migrations.prepare_startup_state(tmp_path)
    sha = meta(tmp_path, "last_store_body")["sha"]
    _body(tmp_path, sha)
    monkeypatch.setattr(startup_migrations, "boot_admissions", lambda *_: pytest.fail("read boot history"))
    monkeypatch.setattr(startup_migrations, "_legacy_question_publications", lambda *_: pytest.fail("read chat history"))
    monkeypatch.setattr(obligations, "members", lambda *_: pytest.fail("read current rows despite generation stamp"))
    report = startup_migrations.prepare_startup_state(tmp_path)
    assert report["status"] == "unchanged" and report["question_publications"] == {}


def test_supervisor_bootstrap_waits_for_store_detector():
    repo = pathlib.Path(__file__).resolve().parents[1]
    tree = ast.parse((repo / "server.py").read_text(encoding="utf-8"))
    supervisor = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_supervisor")
    calls = [(getattr(node.func, "attr", getattr(node.func, "id", "")), node.lineno)
             for node in ast.walk(supervisor) if isinstance(node, ast.Call)]
    assert dict(calls)["require_published"] < dict(calls)["_bootstrap_supervisor_repo"]
