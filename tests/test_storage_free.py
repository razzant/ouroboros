"""The free-space action (B1-B12 of the storage plan): a pure preview, a run through the existing
retirers with the wait set to zero for the previewed items only, honest removed/kept/deleting."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import headless, storage, task_custody
from ouroboros import store as task_store
from ouroboros import subagent_worktrees as sw
from ouroboros.storage_free import FreeSpace
from tests._store_journal_shared import held_open, tree_state
from tests.test_store import document, publish

OLD = "2026-10-01T00:00:00+00:00"


def _recent() -> str:
    return datetime.fromtimestamp(time.time() - 3600, tz=timezone.utc).isoformat()


@contextlib.contextmanager
def _guard(stop=None):
    yield not (callable(stop) and stop())


def _drives(root: Path, rows):
    """ROWS: (task_id, layout, status, ts[, probe]) → task rows and empty drives in both layouts."""
    documents, probes = [], {}
    for task_id, layout, status, ts, *rest in rows:
        if layout == "headless":
            drive = headless.prepare_task_drive(root, task_id, "empty")
        else:
            drive = root / "task_drives" / task_id
            drive.mkdir(parents=True, exist_ok=True)
        (drive / "scratch.bin").write_bytes(b"s" * 4096)
        if status:
            documents.append(document(task_id, status, child_drive_root=str(drive), ts=ts))
        probes[task_id] = rest[0] if rest else False
    if task_store.store_path(root).exists():
        _add(root, documents)  # tasks finishing after an earlier batch (after a preview, say)
    else:
        publish(root, *documents)
    return probes


def _add(root: Path, documents) -> None:
    conn = task_store.connect(task_store.store_path(root), task_store.TIER_ENFORCED)
    try:
        conn.execute("BEGIN IMMEDIATE")
        txn = task_store.Txn(conn)
        for row in documents:
            txn.store_task(row)
        task_store.commit(conn)
    finally:
        conn.close()


def _fake_sizer(monkeypatch, *, hold: threading.Event = None):
    """The sizing child in-process: the same walker, the same lines, no subprocess."""
    import ouroboros.startup_historical_audit as audit

    spawned = []

    class Child:
        def __init__(self):
            self.pid, self.killed, self._paths, self._lines = 4242, False, b"", None
            self.stdin, self.stdout = self, self

        def write(self, raw):
            self._paths += raw

        def close(self):
            if self._lines is None:
                self._paths.decode("ascii")  # the request is ASCII on the pipe whatever the code page
                paths = [path for path in map(storage._sized_path, self._paths.decode("ascii").split("\n")) if path]
                self._lines = [json.dumps({"event": "size", **row}).encode() + b"\n"
                               for row in storage.size_paths(paths, sleep=lambda _s: None)]
                self._lines.append(b'{"status":"finished","io_priority":"unchanged","paths":%d}\n' % len(paths))

        def readline(self, _limit=0):
            if hold is not None and not hold.is_set() and not self.killed:
                hold.wait(10)
            if self.killed or not self._lines:
                return b""
            return self._lines.pop(0)

        def wait(self):
            return -9 if self.killed else 0

        def poll(self):
            return None if self._lines else 0

        def kill(self):
            self.killed = True

    def spawn(data_root, repo_dir, purpose, *, args=(), **popen):
        assert purpose == "storage" and tuple(args) == ("--size-paths",) and popen.get("stdin") is subprocess.PIPE
        child = Child()
        spawned.append(child)
        return child

    monkeypatch.setattr(audit, "spawn_maintenance_job", spawn)
    return spawned


def _owner(probes, **kwargs):
    return FreeSpace(live=lambda task_id: probes.get(task_id, False), guard=_guard, **kwargs)


def _join(owner: FreeSpace):
    thread = owner._thread
    if thread is not None:
        thread.join(timeout=60)
        assert not thread.is_alive()


def _preview(owner, root, tmp_path):
    assert owner.preview(root, tmp_path) is True
    _join(owner)
    return owner.projection(root)


def _group(view, group_id):
    return next(group for group in view["groups"] if group["id"] == group_id)


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for args in (("init", "-q"), ("config", "user.email", "t@t"), ("config", "user.name", "t")):
        subprocess.run(["git", *args], cwd=str(path), check=True, capture_output=True)
    (path / "a.txt").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=str(path), check=True, capture_output=True)


# ------------------------------------------------------------------ B1, B2


def test_preview_lists_finished_not_live_folders_and_copies_and_names_the_rest(tmp_path, monkeypatch):
    root = tmp_path / "data"
    probes = _drives(root, [
        ("done-old", "direct", "completed", OLD), ("done-old", "headless", "completed", OLD),
        ("done-new", "direct", "failed", _recent()), ("running", "direct", "running", OLD, True),
        ("unknown", "headless", "completed", OLD, None), ("rowless", "direct", "", OLD),
    ])
    repo, wtroot = tmp_path / "repo", tmp_path / "wtroot"
    _init_repo(repo)
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(wtroot))
    for task_id in ("copy-done", "copy-live"):
        sw.provision_worktree(repo_dir=repo, task_id=task_id, worktree_root=wtroot, data_dir=root)
    publish_rows = [document("copy-done", "completed"), document("copy-live", "completed")]
    from tests._store_shared import seed_result

    for row in publish_rows:
        seed_result(root, row["task_id"], row)
    probes["copy-live"] = True
    _fake_sizer(monkeypatch)
    owner = _owner(probes)

    view = _preview(owner, root, tmp_path)
    assert view["state"] == "ready" and view["plan"] and view["error"] is None
    folders = _group(view, "work_folders")
    assert {item["id"] for item in folders["items"]} == {"direct:done-old", "headless:done-old", "direct:done-new"}
    assert (folders["count"], folders["inside_wait"], folders["in_use"], folders["unreadable"]) == (3, 1, 2, 1)
    assert folders["bytes_complete"] is True and folders["bytes"] >= 3 * 4096
    assert all(item["bytes"] >= 4096 for item in folders["items"])
    copies = _group(view, "working_copies")
    assert [item["name"] for item in copies["items"]] == ["copy-done"]
    assert [item["id"].rsplit("-", 1)[0] for item in copies["items"]] == ["plain:copy-done"]  # one object, not its task
    assert (copies["count"], copies["in_use"]) == (1, 1)
    assert view["done"] == view["total"] == 4


def test_preview_is_pure(tmp_path, monkeypatch):
    root = tmp_path / "data"
    probes = _drives(root, [("done", "direct", "completed", OLD), ("helper", "headless", "completed", OLD)])
    _fake_sizer(monkeypatch)
    calls = []
    for module, name in ((task_custody, "settle_child_drive"), (headless, "prune_task_drives"), (headless, "prune_headless_task_drives"),
                         (sw, "prune_orphans"), (sw, "prune_execution_snapshots")):
        monkeypatch.setattr(module, name, lambda *a, _name=name, **k: calls.append(_name))
    from ouroboros import body_candidate, task_results

    monkeypatch.setattr(body_candidate, "retention_verdict", lambda *a, **k: calls.append("retention_verdict"))
    monkeypatch.setattr(task_results, "write_task_result", lambda *a, **k: calls.append("write_task_result"))
    with held_open(task_store.store_path(root)):  # SQLite's journal set-up falls outside: every mtime is a write's
        before = tree_state(root, mtimes=True)

        view = _preview(_owner(probes), root, tmp_path)

        assert view["state"] == "ready" and _group(view, "work_folders")["count"] == 2
        assert calls == []
        assert tree_state(root, mtimes=True) == before
    assert not (root / "logs" / "events.jsonl").exists()


def test_the_preview_reads_a_legacy_row_without_filling_its_retention_projection(tmp_path, monkeypatch):
    """An install upgraded from an older body holds rows with a NULL retention projection: the
    preview classifies them in memory (read-only reader) and writes nothing; the drive pass fills it."""
    from ouroboros import store

    root = tmp_path / "data"
    probes = _drives(root, [("legacy", "direct", "completed", OLD)])

    def projection():
        with store.open_store(root, write=False) as txn:
            return txn.conn.execute("SELECT drive_retention_at FROM tasks WHERE task_id='legacy'").fetchone()[0]

    with store.open_store(root, write=True) as txn:
        txn.conn.execute("UPDATE tasks SET drive_retention_at=NULL WHERE task_id='legacy'")
    assert projection() is None
    _fake_sizer(monkeypatch)
    before = sorted((str(path), path.stat().st_mtime_ns) for path in root.rglob("*"))

    view = _preview(_owner(probes), root, tmp_path)

    folders = _group(view, "work_folders")
    assert (folders["count"], folders["unreadable"], folders["inside_wait"]) == (1, 0, 0), "classified from its body, in memory"
    assert projection() is None, "the preview fills no projection"
    assert sorted((str(path), path.stat().st_mtime_ns) for path in root.rglob("*")) == before
    headless.prune_task_drives(root, retention_days=365, live=lambda _t: False, guard=_guard)
    assert projection() is not None, "the drive pass fills it"


def test_a_folder_the_sizing_cannot_list_keeps_an_unknown_size_never_zero(tmp_path, monkeypatch):
    if os.name == "nt" or os.geteuid() == 0:
        pytest.skip("needs a folder this user cannot list")
    root = tmp_path / "data"
    probes = _drives(root, [("locked", "direct", "completed", OLD), ("open", "direct", "completed", OLD)])
    _fake_sizer(monkeypatch)
    (root / "task_drives" / "locked").chmod(0)
    try:
        view = _preview(_owner(probes), root, tmp_path)
    finally:
        (root / "task_drives" / "locked").chmod(0o700)
    folders = _group(view, "work_folders")
    assert view["state"] == "ready" and folders["count"] == 2
    items = {item["id"]: item["bytes"] for item in folders["items"]}
    assert "direct:locked" not in items and items["direct:open"] >= 4096, "an unknown size is never a largest item"
    assert folders["bytes"] == items["direct:open"] and folders["bytes_complete"] is False


# ------------------------------------------------------------- B3, B4, B6, B7


def _run(owner, root, tmp_path, groups=("work_folders",)):
    view = owner.projection(root)
    started = owner.start(root, tmp_path, view["plan"], list(groups))
    assert started == {"started": True, "reason": ""}, started
    _join(owner)
    return owner.projection(root)


def test_the_run_frees_a_folder_inside_the_wait_that_the_ordinary_pass_keeps(tmp_path, monkeypatch):
    root = tmp_path / "data"
    probes = _drives(root, [("fresh", "direct", "completed", _recent())])
    ordinary = headless.prune_task_drives(root, live=lambda _t: False, guard=_guard)
    assert ordinary["skipped"] == [{"task_id": "fresh", "reason": "younger_than_retention"}]
    _fake_sizer(monkeypatch)
    owner = _owner(probes)
    _preview(owner, root, tmp_path)

    view = _run(owner, root, tmp_path)
    assert view["state"] == "done"
    run = view["run"]
    assert (run["selected"], run["done"], run["stopped"], run["error"]) == (1, 1, False, None)
    assert not (root / "task_drives" / "fresh").exists()
    assert run["removed"] + run["deleting"] == 1, run
    assert isinstance(run["free_before"], int) and isinstance(run["free_after"], int)
    rows = [json.loads(line) for line in (root / "logs" / "supervisor.jsonl").read_text().splitlines()]
    assert [(row["type"], row["phase"], row["selected"]) for row in rows] == [("storage_free", "done", 1)]


def test_the_run_names_the_folders_it_freed_in_its_event(tmp_path, monkeypatch):
    """A drive freed by the owner's run left only counts, so a task's missing folder had no remover on record;
    the event names each folder by its preview id, and a kept folder is never named among them."""
    root = tmp_path / "data"
    probes = _drives(root, [("gone", "direct", "completed", OLD), ("busy", "direct", "completed", OLD)])
    monkeypatch.setattr("ouroboros.server_maintenance.DATA_DIR", root)
    _fake_sizer(monkeypatch)
    owner = _owner(probes)
    _preview(owner, root, tmp_path)
    probes["busy"] = True  # in use again between preview and run
    _run(owner, root, tmp_path)
    rows = [json.loads(line) for line in (root / "logs" / "events.jsonl").read_text().splitlines()]
    report, = [row["report"] for row in rows if row["type"] == "storage_free"]
    assert report["removed_ids"] + report["deleting_ids"] == ["direct:gone"] and report["stuck_ids"] == []
    assert report["removed"] + report["deleting"] == 1 and report["kept"] == [{"reason": "in_use", "count": 1}]


def test_a_folder_not_previewed_is_untouched_and_a_task_live_again_is_kept(tmp_path, monkeypatch):
    root = tmp_path / "data"
    probes = _drives(root, [("first", "direct", "completed", OLD), ("second", "direct", "completed", OLD)])
    _fake_sizer(monkeypatch)
    owner = _owner(probes)
    _preview(owner, root, tmp_path)
    _drives(root, [("late", "direct", "completed", OLD)])  # finished after the preview
    probes["second"] = True  # in use again between preview and run

    view = _run(owner, root, tmp_path)
    run = view["run"]
    assert not (root / "task_drives" / "first").exists()
    assert (root / "task_drives" / "late").is_dir() and (root / "task_drives" / "second").is_dir()
    assert run["kept"] == [{"reason": "in_use", "count": 1}]
    assert run["removed"] + run["deleting"] == 1 and run["selected"] == 2


@pytest.mark.parametrize("trash_deleted", [False, True])
def test_a_folder_the_cadence_settled_first_counts_once_and_raises_nothing(tmp_path, monkeypatch, trash_deleted):
    root = tmp_path / "data"
    probes = _drives(root, [("shared", "direct", "completed", OLD)])
    _fake_sizer(monkeypatch)
    owner = _owner(probes)
    _preview(owner, root, tmp_path)
    assert headless.prune_task_drives(root, live=lambda _t: False, guard=_guard)["pruned"][0]["task_id"] == "shared"
    if trash_deleted:
        import shutil

        shutil.rmtree(root / "state" / "custody_trash")

    view = _run(owner, root, tmp_path)
    run = view["run"]
    assert view["state"] == "done" and run["error"] is None
    assert run["removed"] + run["deleting"] == 1 and run["kept"] == []
    assert (run["removed"], run["deleting"]) == ((1, 0) if trash_deleted else (0, 1))


def test_a_kept_folder_names_its_reason_and_the_next_run_frees_it(tmp_path, monkeypatch):
    root = tmp_path / "data"
    probes = _drives(root, [("pending", "direct", "completed", OLD)])
    _fake_sizer(monkeypatch)
    real = task_custody.settle_child_drive
    once = {"left": 1}

    def retained_once(canonical_root, task_id, drive, **kwargs):
        if once["left"]:
            once["left"] -= 1
            report = kwargs.get("report")
            if report is not None:
                report.setdefault("skipped", []).append({"task_id": task_id, "reason": "child_refs_pending"})
            return {"status": "retained", "reason": "child_refs_pending", "published": 0}
        return real(canonical_root, task_id, drive, **kwargs)

    monkeypatch.setattr(task_custody, "settle_child_drive", retained_once)
    monkeypatch.setattr(headless, "settle_child_drive", retained_once, raising=False)
    owner = _owner(probes)
    _preview(owner, root, tmp_path)
    view = _run(owner, root, tmp_path)
    assert view["run"]["kept"] == [{"reason": "results_pending", "count": 1}]
    assert (root / "task_drives" / "pending").is_dir()

    _preview(owner, root, tmp_path)
    view = _run(owner, root, tmp_path)
    assert view["run"]["kept"] == [] and view["run"]["removed"] + view["run"]["deleting"] == 1
    assert not (root / "task_drives" / "pending").exists()


@pytest.mark.parametrize("code, reason", [
    ("unread_mailbox_uncustodied", "results_pending"),  # late mail: the next pass unions and carries it
    ("call_inventory_changed", "results_pending"),  # call records moved under the pass: re-armed next time
    ("unread_mailbox_unreadable", "unreadable_record"),  # a torn mailbox is a read failure, not pending work
    ("task_result_unreadable", "unreadable_record"),
    ("custody_error: ValueError('torn')", "unreadable_record"),
    ("generation_closed", "restart"),
    ("store_busy", "busy"),  # a store still leased inside the drive: busy right now, tried next pass
    ("not_own_drive", "other"),  # a code with no owner-facing group stays "for another reason"
])
def test_the_fail_closed_mailbox_and_call_inventory_codes_name_their_owner_facing_reason(code, reason):
    from ouroboros.storage_free import _reason

    assert _reason(code) == reason


# ------------------------------------------------------------------ B8, B10


def test_a_second_run_while_one_runs_is_refused_and_a_replaced_plan_is_outdated(tmp_path, monkeypatch):
    root = tmp_path / "data"
    probes = _drives(root, [("slow", "direct", "completed", OLD)])
    _fake_sizer(monkeypatch)
    owner = _owner(probes)
    _preview(owner, root, tmp_path)
    plan = owner.projection(root)["plan"]
    assert owner.start(root, tmp_path, "someone-elses-plan", ["work_folders"]) == {"started": False, "reason": "preview_outdated"}
    assert owner.start(root, tmp_path, plan, ["../x"]) == {"started": False, "reason": "unknown_group"}
    assert owner.start(root, tmp_path, plan, []) == {"started": False, "reason": "nothing_selected"}
    assert (root / "task_drives" / "slow").is_dir(), "nothing ran"

    release = threading.Event()
    real = task_custody.settle_child_drive

    def slow(*args, **kwargs):
        assert release.wait(30)
        return real(*args, **kwargs)

    monkeypatch.setattr(task_custody, "settle_child_drive", slow)
    assert owner.start(root, tmp_path, plan, ["work_folders"])["started"] is True
    assert owner.start(root, tmp_path, plan, ["work_folders"]) == {"started": False, "reason": "busy"}
    assert owner.preview(root, tmp_path) is False
    release.set()
    _join(owner)
    assert owner.projection(root)["state"] == "done"
    assert owner.start(root, tmp_path, plan, ["work_folders"]) == {"started": False, "reason": "preview_outdated"}


def test_a_migrating_store_offers_nothing_and_nothing_runs(tmp_path, monkeypatch):
    root = tmp_path / "data"
    probes = _drives(root, [("done", "direct", "completed", OLD)])
    from ouroboros import store, task_store

    def migrating(*_args, **_kwargs):
        raise store.StoreMigrating("upgrading")

    monkeypatch.setattr(task_store, "drive_retention_facts", migrating)
    _fake_sizer(monkeypatch)
    view = _preview(_owner(probes), root, tmp_path)
    folders = _group(view, "work_folders")
    assert (folders["count"], folders["unreadable"]) == (0, 1)
    assert (root / "task_drives" / "done").is_dir()


# ------------------------------------------------------------------- B9


def test_removed_bytes_follow_the_trash_and_the_row_carries_the_same_numbers(tmp_path, monkeypatch):
    free_bytes = {"now": 100}
    monkeypatch.setattr("ouroboros.storage_free._free_bytes", lambda _root: free_bytes["now"])
    root = tmp_path / "data"
    probes = _drives(root, [("gone", "direct", "completed", OLD), ("waiting", "direct", "completed", OLD)])
    _fake_sizer(monkeypatch)
    owner = _owner(probes)
    view = _preview(owner, root, tmp_path)
    sizes = {item["id"]: item["bytes"] for item in _group(view, "work_folders")["items"]}
    requested = []
    from ouroboros import maintenance

    monkeypatch.setattr(maintenance.child, "request", lambda: requested.append(True))

    view = _run(owner, root, tmp_path)
    run = view["run"]
    assert requested == [True]
    assert (run["removed"], run["deleting"]) == (0, 2), "moved aside: the maintenance child has not deleted them yet"
    assert run["deleting_bytes"] == sizes["direct:gone"] + sizes["direct:waiting"] and run["removed_bytes"] == 0
    assert run["free_before"] == run["free_after"] == 100
    # The child deletes one entry; the projection follows the trash on the next read.
    trash = root / "state" / "custody_trash"
    gone = next(entry for entry in trash.iterdir() if entry.name.startswith("gone-"))
    import shutil

    shutil.rmtree(gone)
    free_bytes["now"] = 500
    view = owner.projection(root)
    assert (view["run"]["removed"], view["run"]["deleting"]) == (1, 1)
    assert view["run"]["removed_bytes"] == sizes["direct:gone"] and view["run"]["deleting_bytes"] == sizes["direct:waiting"]
    assert view["run"]["free_before"] == 100 and view["run"]["free_after"] == 500
    row = json.loads((root / "logs" / "supervisor.jsonl").read_text().splitlines()[-1])
    assert (row["type"], row["selected"], row["deleting"], row["removed"]) == ("storage_free", 2, 2, 0)
    assert row["free_after"] == 100, "the event records completion; the UI reads current free space"


_REAL_SCANDIR = os.scandir


class _WindowsListing:
    """``os.scandir`` of one folder as Windows hands it out: ``DirEntry.stat`` there carries no inode,
    device or link count, so an identity read from the listing never equals an ``lstat`` of the entry."""

    def __init__(self, path):
        self._listing = _REAL_SCANDIR(path)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self._listing.close()

    def __iter__(self):
        return self

    def __next__(self):
        entry = next(self._listing)
        info = entry.stat(follow_symlinks=False)
        listed = SimpleNamespace(st_mode=info.st_mode, st_size=info.st_size, st_mtime_ns=info.st_mtime_ns,
                                 st_ino=0, st_dev=0, st_nlink=0)
        return SimpleNamespace(name=entry.name, path=entry.path, is_dir=entry.is_dir, is_symlink=entry.is_symlink,
                               stat=lambda follow_symlinks=True: listed)


def _windows_listing_of(monkeypatch, folder):
    """Every ``os.scandir`` of FOLDER returns a Windows-shaped listing; any other folder its real one."""
    monkeypatch.setattr(os, "scandir", lambda path=".": _WindowsListing(path) if isinstance(path, (str, os.PathLike))
                        and Path(path) == folder else _REAL_SCANDIR(path))


def _stuck_trash_entry(tmp_path, monkeypatch):
    """A freed drive whose trash entry the maintenance child could not remove (a file in it is held
    open): the job records the entry's identity as it saw it, and both the job's due check and the
    run view compare the entry against that very record. The trash folder is listed the way Windows
    lists it (no inode), so an identity read from the listing would never match. Returns the data
    root, the storage owner and the real removal (the hold released)."""
    from ouroboros import maintenance

    root = tmp_path / "data"
    probes = _drives(root, [("gone", "direct", "completed", OLD)])
    _fake_sizer(monkeypatch)
    monkeypatch.setattr(maintenance.child, "request", lambda: None)
    owner = _owner(probes)
    _preview(owner, root, tmp_path)
    view = _run(owner, root, tmp_path)
    assert (view["run"]["removed"], view["run"]["deleting"], view["run"]["stuck"]) == (0, 1, 0)

    def held_open(entry):
        raise PermissionError(13, "a file in it is held open", str(entry))

    released = task_custody._remove_entry
    _windows_listing_of(monkeypatch, root / "state" / "custody_trash")
    monkeypatch.setattr(task_custody, "_remove_entry", held_open)
    assert maintenance.run_job(root, "custody_trash") == {"state": "failed", "items": 0}
    assert "custody_trash" not in maintenance.due(root), "a stuck entry is no reason to spawn the child"
    view = owner.projection(root)
    assert (view["run"]["removed"], view["run"]["deleting"], view["run"]["stuck"]) == (0, 0, 1)
    return root, owner, released


def test_a_stuck_trash_entry_is_retried_whenever_the_child_runs(tmp_path, monkeypatch):
    """A released handle changes neither the entry's inode nor its time (Windows, an antivirus
    scan), so the entry stays honestly stuck and is no reason to spawn the child; the child still
    retries it whenever it runs for another job, and then it is removed."""
    from ouroboros import maintenance, maintenance_jobs

    root, owner, released = _stuck_trash_entry(tmp_path, monkeypatch)
    # Still held: the child's retry walks the entry again but commits nothing (the store does not move,
    # so no job that waits on a store write is re-armed by it).
    revision, row = maintenance._store_facts(root)["revision"], maintenance_jobs.job(root, "custody_trash")
    assert maintenance.run_job(root, "custody_trash") == {"state": "failed", "items": 0}
    assert maintenance._store_facts(root)["revision"] == revision and maintenance_jobs.job(root, "custody_trash") == row
    monkeypatch.setattr(task_custody, "_remove_entry", released)  # the hold is gone; nothing in the entry changed
    assert "custody_trash" not in maintenance.due(root)
    assert owner.projection(root)["run"]["stuck"] == 1
    assert maintenance.run_job(root, "custody_trash") == {"state": "done", "items": 1}
    view = owner.projection(root)
    assert (view["run"]["removed"], view["run"]["deleting"], view["run"]["stuck"]) == (1, 0, 0)
    assert maintenance.run_job(root, "custody_trash") is None, "an empty trash rests"


def test_a_trash_entry_the_child_could_not_delete_is_stuck_until_it_changes(tmp_path, monkeypatch):
    from ouroboros import maintenance

    root, owner, _released = _stuck_trash_entry(tmp_path, monkeypatch)
    # The entry changed (the owner released a file): the child will retry, so it is deleting again.
    [name] = [entry.name for entry in (root / "state" / "custody_trash").iterdir()]
    (root / "state" / "custody_trash" / name / "released.bin").write_bytes(b"x")
    assert "custody_trash" in maintenance.due(root)
    view = owner.projection(root)
    assert (view["run"]["removed"], view["run"]["deleting"], view["run"]["stuck"]) == (0, 1, 0)
    row = json.loads((root / "logs" / "supervisor.jsonl").read_text().splitlines()[-1])
    assert (row["type"], row["deleting"], row["stuck"]) == ("storage_free", 1, 0)
    # Still held, the retry fails again; it records the entry's new identity, so the job rests instead of
    # spawning the child on every maintenance tick (a state-only write condition would keep the old one).
    from ouroboros import maintenance as maintenance_module
    assert maintenance_module.run_job(root, "custody_trash") == {"state": "failed", "items": 0}
    assert "custody_trash" not in maintenance_module.due(root), "a retried stuck entry is no reason to spawn again"


# ------------------------------------------------------------------- B11


def test_a_closed_generation_stops_the_run_between_settlements(tmp_path, monkeypatch):
    root = tmp_path / "data"
    probes = _drives(root, [("one", "direct", "completed", OLD), ("two", "direct", "completed", OLD)])
    _fake_sizer(monkeypatch)
    owner = _owner(probes)
    _preview(owner, root, tmp_path)
    from ouroboros import server_maintenance

    closed = {"now": False}
    monkeypatch.setattr(server_maintenance, "_stop_requested", lambda *_a, **_k: closed["now"])
    real = task_custody.settle_child_drive

    def settle_then_close(*args, **kwargs):
        outcome = real(*args, **kwargs)
        closed["now"] = True  # a restart lands after the first move
        return outcome

    monkeypatch.setattr(task_custody, "settle_child_drive", settle_then_close)
    view = _run(owner, root, tmp_path)
    assert view["state"] == "stopped" and view["run"]["stopped"] is True
    assert view["run"]["removed"] + view["run"]["deleting"] == 1 and view["run"]["done"] == 1
    assert sum(1 for name in ("one", "two") if (root / "task_drives" / name).is_dir()) == 1, "nothing half-moved"
    row = json.loads((root / "logs" / "supervisor.jsonl").read_text().splitlines()[-1])
    assert row["type"] == "storage_free" and row["phase"] == "stopped"


def test_cancel_during_listing_drops_the_plan_and_kills_the_sizing_child(tmp_path, monkeypatch):
    root = tmp_path / "data"
    probes = _drives(root, [("done", "direct", "completed", OLD)])
    hold = threading.Event()
    spawned = _fake_sizer(monkeypatch, hold=hold)
    owner = _owner(probes)
    assert owner.preview(root, tmp_path) is True
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not spawned:
        time.sleep(0.01)
    assert owner.projection(root)["state"] == "listing" and owner.live_children()
    assert owner.cancel() is True
    hold.set()
    _join(owner)
    assert spawned[0].killed
    view = owner.projection(root)
    assert (view["state"], view["plan"], view["groups"]) == ("idle", None, [])
    assert owner.cancel() is False


# ------------------------------------------------------------------- B12


def test_working_copies_of_finished_work_are_freed_and_a_running_ones_stays(tmp_path, monkeypatch):
    root = tmp_path / "data"
    repo, wtroot = tmp_path / "repo", tmp_path / "wtroot"
    _init_repo(repo)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(wtroot))
    root.mkdir()
    from tests._store_shared import seed_result

    paths = {}
    for task_id in ("finished", "busy"):
        paths[task_id] = Path(sw.provision_worktree(repo_dir=repo, task_id=task_id, worktree_root=wtroot, data_dir=root).path)
        seed_result(root, task_id, document(task_id, "completed"))
    _fake_sizer(monkeypatch)
    owner = _owner({"finished": False, "busy": True})
    view = _preview(owner, root, tmp_path)
    copies = _group(view, "working_copies")
    assert [item["id"].rsplit("-", 1)[0] for item in copies["items"]] == ["plain:finished"] and copies["in_use"] == 1

    view = _run(owner, root, tmp_path, groups=("working_copies",))
    assert view["state"] == "done"
    assert not paths["finished"].exists() and paths["busy"].is_dir()
    assert view["run"]["removed"] == 1 and view["run"]["kept"] == []
    assert {row["task_id"] for row in sw.list_worktrees(data_dir=root)} == {"busy"}


def test_a_delegated_snapshot_is_offered_only_by_the_retirers_own_proof(tmp_path, monkeypatch):
    """The preview offers exactly what ``prune_execution_snapshots`` would remove, by the one proof
    (``subagent_worktrees.snapshot_released``): a snapshot whose task is proven over with custody
    closed is listed and freed; one whose owner is live, unknown (no task row yet: the provisioning
    window) or whose custody is open is in use, unlisted and untouched by the run."""
    from ouroboros import delegate_custody as custody
    from tests._store_shared import seed_result

    root = tmp_path / "data"
    repo, wtroot = tmp_path / "repo", tmp_path / "wtroot"
    _init_repo(repo)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(wtroot))
    root.mkdir()
    handles = {}
    for task_id in ("over", "live", "unknown", "held"):
        handles[task_id] = sw.provision_execution_snapshot(target_root=repo, task_id=task_id, snapshot_id=f"snap-{task_id}",
                                                           worktree_root=wtroot, data_dir=root)
        if task_id != "unknown":
            seed_result(root, task_id, document(task_id, "completed"))
    custody.record_started(root, custody.RunCustody(run_id="run-held", task_id="held", snapshot_id="snap-held",
                                                    execution_root=handles["held"].path))
    # The run frees by the probe the preview offered by (the owner's), handed to the retirer; the
    # supervisor's own answers unknown here, as it does outside the supervisor process.
    monkeypatch.setattr("supervisor.queue.task_settlement_liveness", lambda _task: None)
    _fake_sizer(monkeypatch)
    owner = _owner({"live": True})
    view = _preview(owner, root, tmp_path)
    copies = _group(view, "working_copies")
    assert [item["id"] for item in copies["items"]] == ["snapshot:snap-over"]
    assert (copies["count"], copies["in_use"], copies["unreadable"]) == (1, 3, 0)

    view = _run(owner, root, tmp_path, groups=("working_copies",))
    assert view["state"] == "done" and (view["run"]["removed"], view["run"]["kept"]) == (1, [])
    assert not Path(handles["over"].path).exists()
    assert all(Path(handles[task_id].path).is_dir() for task_id in ("live", "unknown", "held"))
    assert {row["snapshot_id"] for row in sw.list_worktrees(data_dir=root)} == {"snap-live", "snap-unknown", "snap-held"}


@pytest.mark.parametrize("kind", ["plain", "delegated"])
def test_stop_between_working_copies_leaves_the_rest_and_reports_a_stopped_run(tmp_path, monkeypatch, kind):
    """Each copy is one retirer call with a stop check before it: a Stop that lands while the first
    copy goes leaves the second and third as they were, and the run reports stopped."""
    root = tmp_path / "data"
    repo, wtroot = tmp_path / "repo", tmp_path / "wtroot"
    _init_repo(repo)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(wtroot))
    root.mkdir()
    from tests._store_shared import seed_result

    paths = {}
    for task_id in ("first", "second", "third"):
        if kind == "plain":
            paths[task_id] = Path(sw.provision_worktree(repo_dir=repo, task_id=task_id, worktree_root=wtroot, data_dir=root).path)
        else:
            paths[task_id] = Path(sw.provision_execution_snapshot(target_root=repo, task_id=task_id, snapshot_id=f"snap-{task_id}",
                                                                  worktree_root=wtroot, data_dir=root).path)
        seed_result(root, task_id, document(task_id, "completed"))  # both retirers need the task proven over
    _fake_sizer(monkeypatch)
    owner = _owner({})
    view = _preview(owner, root, tmp_path)
    assert _group(view, "working_copies")["count"] == 3
    retirer = "prune_orphans" if kind == "plain" else "prune_execution_snapshots"
    real, calls = getattr(sw, retirer), []

    def retire_then_stop(*args, **kwargs):
        outcome = real(*args, **kwargs)
        calls.append(list(kwargs.get("only") or kwargs["only_rows"]))
        if len(calls) == 1:
            assert owner.cancel() is True  # Stop lands as the first copy's retirer returns
        return outcome

    monkeypatch.setattr(sw, retirer, retire_then_stop)
    view = _run(owner, root, tmp_path, groups=("working_copies",))
    assert view["state"] == "stopped" and view["run"]["stopped"] is True
    assert len(calls) == 1 and len(calls[0]) == 1, "one copy per retirer call"
    assert (view["run"]["removed"], view["run"]["done"], view["run"]["kept"]) == (1, 1, [])
    assert not paths["first"].exists() and paths["second"].is_dir() and paths["third"].is_dir()
    assert {row["task_id"] for row in sw.list_worktrees(data_dir=root)} == {"second", "third"}


# ------------------------------------------------- G3-G7: leftovers no background pass removes

LEFTOVER_GROUPS = ("review_evidence", "review_checkouts", "lost_copies", "service_folders")


def _seat(root: Path, task_id: str, operation: str, *, area: bool = False, marker: bool = True) -> Path:
    from ouroboros import review_checking as rc

    home = rc.seat_dir(root, task_id, operation)
    (home / "evidence").mkdir(parents=True)
    (home / "evidence" / "summary.json").write_text('{"status": "retained"}', encoding="utf-8")
    (home / "evidence" / "venv.bin").write_bytes(b"v" * 8192)
    if marker:
        (home / "checking.json").write_text(json.dumps({"schema": 1, "task_id": task_id, "operation_id": operation,
                                                        "source": {"kind": "workspace", "root": str(root)}}),
                                            encoding="utf-8")
    if area:
        (home / "area").mkdir()
    return home


def _leftovers(root: Path, wtroot: Path):
    """One of each leftover the new groups offer, beside the twin each must keep."""
    from tests._store_shared import seed_result

    for task_id in ("seat-done", "seat-live", "checkout-live"):
        seed_result(root, task_id, document(task_id, "completed"))
    seats = {"done": _seat(root, "seat-done", "op-1"), "copy": _seat(root, "seat-done", "op-2", area=True),
             "live": _seat(root, "seat-live", "op-1"), "unreadable": _seat(root, "seat-done", "op-3", marker=False)}
    base = root / "state" / "review_checkouts"
    checkouts = {"orphan": base / "orphan", "owned": base / "owned"}
    for entry in checkouts.values():
        (entry / "repo").mkdir(parents=True)
        (entry / "repo" / "f.txt").write_text("x", encoding="utf-8")
    (checkouts["owned"] / "checkout.json").write_text(json.dumps({"schema_version": 1, "task_id": "checkout-live"}),
                                                      encoding="utf-8")
    folders = {"lost": wtroot / "dlg_lost-task_0123456789abcdef", "lost_live": wtroot / "dlg_seat-live_0123456789abcdef",
               "env": wtroot / "gone.env", "run_done": wtroot / ".test-evidence" / "run-done",
               "run_live": wtroot / ".test-evidence" / "run-live"}
    for folder in folders.values():
        folder.mkdir(parents=True)
        (folder / "f.bin").write_bytes(b"f" * 4096)
    (folders["run_done"] / "result.json").write_text("{}", encoding="utf-8")
    (folders["run_live"] / "owner.json").write_text(json.dumps({"supervisor_pid": os.getpid()}), encoding="utf-8")
    return seats, checkouts, folders


def test_leftover_groups_offer_only_proven_leftovers_and_the_run_frees_exactly_them(tmp_path, monkeypatch):
    """A seat home of a task proven over whose copy is gone, a review checkout no task owns, a folder
    no copy record names, a gone copy's environment and a finished test run are offered and freed;
    a seat that keeps its copy, a live task's seat/checkout/folder, an unreadable marker and a running
    test run are named and stay. Seat homes go through the custody trash, the rest at once."""
    from ouroboros import maintenance

    root = tmp_path / "data"
    repo, wtroot = tmp_path / "repo", tmp_path / "wtroot"
    _init_repo(repo)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(wtroot))
    root.mkdir()
    recorded = Path(sw.provision_worktree(repo_dir=repo, task_id="copy-live", worktree_root=wtroot, data_dir=root).path)
    (recorded.with_name(recorded.name + ".env")).mkdir()  # an environment beside its live copy
    seats, checkouts, folders = _leftovers(root, wtroot)
    probes = {"seat-live": True, "checkout-live": True, "copy-live": True}
    _fake_sizer(monkeypatch)
    owner = _owner(probes)
    view = _preview(owner, root, tmp_path)
    offered = {group["id"]: (sorted(item["id"] for item in group["items"]), group["in_use"], group["unreadable"])
               for group in view["groups"] if group["id"] in LEFTOVER_GROUPS}
    assert offered == {
        "review_evidence": ([f"seat:{seats['done'].name}"], 2, 1),
        "review_checkouts": (["checkout:orphan"], 1, 0),
        "lost_copies": ([f"lost:{sw._record_name(folders['lost'])}"], 1, 0),  # the full path: one name may stand in two roots
        "service_folders": ([f"env:{sw._record_name(folders['env'])}", f"test_run:{sw._record_name(folders['run_done'])}"], 1, 0),
    }
    requested = []
    monkeypatch.setattr(maintenance.child, "request", lambda: requested.append(True))

    view = _run(owner, root, tmp_path, groups=LEFTOVER_GROUPS)
    assert view["state"] == "done" and view["run"]["kept"] == [] and requested == [True]
    assert (view["run"]["removed"], view["run"]["deleting"]) == (4, 1), "the seat home waits in the custody trash"
    assert not seats["done"].exists() and all(seats[name].is_dir() for name in ("copy", "live", "unreadable"))
    [trashed] = list((root / "state" / "custody_trash").iterdir())
    assert trashed.name.startswith(seats["done"].name + "-") and (trashed / "evidence" / "venv.bin").is_file()
    assert not checkouts["orphan"].exists() and checkouts["owned"].is_dir()
    assert not folders["lost"].exists() and folders["lost_live"].is_dir() and recorded.is_dir()
    assert not folders["env"].exists() and recorded.with_name(recorded.name + ".env").is_dir()
    assert not folders["run_done"].exists() and folders["run_live"].is_dir()
    task_custody.remove_custody_trash(root)  # the maintenance child's job
    assert owner.projection(root)["run"]["removed"] == 5


def test_leftover_listing_is_pure_and_a_run_re_checks_each_item(tmp_path, monkeypatch):
    """The listers take no lock and write nothing; a seat whose task came back to life between the
    preview and the run, and a seat a worker holds, are kept with their reasons."""
    from ouroboros import review_checking as rc

    root = tmp_path / "data"
    wtroot = tmp_path / "wtroot"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(wtroot))
    root.mkdir()
    seats, _checkouts, _folders = _leftovers(root, wtroot)
    second = _seat(root, "seat-done", "op-4")
    _fake_sizer(monkeypatch)
    probes = {"seat-live": True, "checkout-live": True}
    owner = _owner(probes)
    with held_open(task_store.store_path(root)):  # SQLite's journal set-up falls outside: every mtime is a write's
        before, outside = tree_state(root, mtimes=True), tree_state(wtroot, mtimes=True)
        view = _preview(owner, root, tmp_path)
        assert tree_state(root, mtimes=True) == before and tree_state(wtroot, mtimes=True) == outside
    assert _group(view, "review_evidence")["count"] == 2
    probes["seat-done"] = True  # the task is live again: nothing of it goes
    with rc._seat_locked(second):  # a worker holds the other seat
        view = _run(owner, root, tmp_path, groups=("review_evidence",))
    assert sorted((row["reason"], row["count"]) for row in view["run"]["kept"]) == [("busy", 1), ("in_use", 1)]
    assert seats["done"].is_dir() and second.is_dir()


@pytest.mark.parametrize("state", ["readable", "unreadable"])
def test_the_checkout_run_reads_the_review_state_once_and_unknown_custody_keeps_every_checkout(tmp_path, monkeypatch, state):
    """The run's own listing reads the review state, and the markerless fallback one more time, not once per
    checkout; a state that turns unreadable keeps every checkout, named, as the listing would."""
    from ouroboros import review_checking_prune as rp

    root = tmp_path / "data"
    base = root / "state" / "review_checkouts"
    entries = [base / name for name in ("first", "second", "third")]
    for entry in entries:
        (entry / "repo").mkdir(parents=True)
        (entry / "repo" / "f.txt").write_text("x", encoding="utf-8")
    _fake_sizer(monkeypatch)
    owner = _owner({}, offered=("review_checkouts",))
    assert _group(_preview(owner, root, tmp_path), "review_checkouts")["count"] == 3
    real, calls = rp._review_rounds, []

    def counted(data_root):
        calls.append(data_root)
        if state == "unreadable" and len(calls) > 1:
            raise ValueError("the review state changed shape under the run")
        return real(data_root)

    monkeypatch.setattr(rp, "_review_rounds", counted)
    view = _run(owner, root, tmp_path, groups=("review_checkouts",))
    assert view["state"] == "done" and view["run"]["error"] is None
    if state == "readable":
        assert len(calls) == 2 and view["run"]["removed"] == 3 and not any(entry.exists() for entry in entries)
    else:
        assert view["run"]["kept"] == [{"reason": "unreadable_record", "count": 3}] and all(entry.is_dir() for entry in entries)


def test_result_pins_of_a_finished_writer_without_a_disposition_are_offered_and_dropped(tmp_path, monkeypatch):
    """A writer whose task is over keeps only its two pins once its checkout was released: they are
    offered with the size of what only they reach, and the run drops both while the record and the
    saved patch stay (a later apply is a plain apply, a continuation starts fresh)."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_continuation import _adopted_snapshot
    from tests._store_shared import seed_result

    root = tmp_path / "data"
    repo, wtroot = tmp_path / "repo", tmp_path / "wtroot"
    _init_repo(repo)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(wtroot))
    root.mkdir()
    handle = sw.provision_execution_snapshot(target_root=repo, task_id="writer", snapshot_id="snap-w",
                                             worktree_root=wtroot, data_dir=root)
    (Path(handle.path) / "a.txt").write_text("changed by the writer\n", encoding="utf-8")
    patch = tmp_path / "saved" / "workspace.patch"
    sw.capture_result_commit("snap-w", patch, data_dir=root)
    assert sw.release_snapshot_checkout("snap-w", worktree_root=wtroot, data_dir=root)
    seed_result(root, "writer", document("writer", "completed"))
    custody.record_started(root, custody.RunCustody(run_id="run-w", task_id="writer", snapshot_id="snap-w",
                                                    execution_root=handle.path))
    _fake_sizer(monkeypatch)
    owner = _owner({})
    view = _preview(owner, root, tmp_path)
    pins = _group(view, "result_pins")
    assert [item["id"] for item in pins["items"]] == ["pins:snap-w"] and pins["items"][0]["bytes"] > 0
    assert _group(view, "working_copies")["count"] == 0, "a released checkout is no working copy"

    view = _run(owner, root, tmp_path, groups=("result_pins",))
    assert view["state"] == "done" and (view["run"]["removed"], view["run"]["kept"]) == (1, [])
    refs = subprocess.run(["git", "for-each-ref", "refs/ouroboros/"], cwd=str(repo), capture_output=True, text=True)
    assert refs.stdout == ""
    record = sw.find_execution_snapshot("snap-w", root)
    assert record["pins_released"] and not {"baseline_ref", "result_ref", "result_sha"} & set(record)
    assert "changed by the writer" in patch.read_text(encoding="utf-8")
    entry = SimpleNamespace(snapshot_id="snap-w", patch_disposed=False, execution_root=handle.path, baseline_sha="b",
                            run_id="run-w")
    assert _adopted_snapshot(entry, SimpleNamespace(access="workspace_write"))[0] is None
    assert _group(_preview(owner, root, tmp_path), "result_pins")["count"] == 0, "nothing left to offer"


# ---------------------------------------------------------------- routes


def test_free_routes_preview_run_and_cancel(tmp_path, monkeypatch):
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from ouroboros.gateway.storage import storage_routes

    root = tmp_path / "data"
    probes = _drives(root, [("done", "direct", "completed", OLD)])
    _fake_sizer(monkeypatch)
    owner = _owner(probes)
    routes = storage_routes(root, repo_dir=tmp_path, free_owner=owner)
    assert {route.path for route in routes} >= {"/api/storage/free/preview", "/api/storage/free", "/api/storage/free/cancel"}
    with TestClient(Starlette(routes=routes)) as client:
        idle = client.get("/api/storage").json()["free"]
        assert idle["state"] == "idle" and idle["groups"] == []
        answer = client.post("/api/storage/free/preview")
        assert answer.status_code == 202 and answer.json()["started"] is True
        _join(owner)
        ready = client.get("/api/storage").json()["free"]
        assert ready["state"] == "ready" and ready["groups"][0]["count"] == 1
        refused = client.post("/api/storage/free", json={"plan": "stale", "groups": ["work_folders"]})
        assert refused.status_code == 202 and refused.json()["started"] is False and refused.json()["reason"] == "preview_outdated"
        malformed = client.post("/api/storage/free", json={"plan": ready["plan"], "groups": "work_folders"})
        assert malformed.json()["started"] is False
        started = client.post("/api/storage/free", json={"plan": ready["plan"], "groups": ["work_folders"]})
        assert started.status_code == 202 and started.json()["started"] is True
        _join(owner)
        done = client.get("/api/storage").json()["free"]
        assert done["state"] == "done" and done["run"]["selected"] == 1
        cancelled = client.post("/api/storage/free/cancel")
        assert cancelled.status_code == 202 and cancelled.json()["free"]["state"] == "done"
