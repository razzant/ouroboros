"""The storage count: complete, saved, gentle, in a child; passive routes (A1-A16 of the storage plan)."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import storage
from ouroboros.gateway import contracts
from ouroboros.gateway.storage import storage_routes
from ouroboros.storage import StorageCounter, load_saved, measure, save_count, with_growth

REPO = Path(__file__).resolve().parent.parent


def _place(place_id, *roots, group="work"):
    return {"id": place_id, "group": group, "roots": [root if isinstance(root, tuple) else (str(root), ()) for root in roots]}


def _table(data: Path, **external):
    """The real place table resolved for a test data folder, with configured roots overridden."""
    from ouroboros import config

    rows = []
    for place_id, group, spec in storage.PLACES:
        if isinstance(spec, str):
            configured = external.get(spec, "")
            if configured == "":
                configured = getattr(config, spec) if spec == "REPO_DIR" else ""
            roots = [(os.path.realpath(os.fspath(configured)), ())] if configured else []
        else:
            roots = [(os.path.realpath(str(data)), tuple(parts)) for parts in spec]
        rows.append({"id": place_id, "group": group, "roots": roots})
    return rows


def _count(data: Path, **external):
    return measure(data, places=_table(data, **external), sleep=lambda _s: None)


def _row(record, place_id):
    return next(row for row in record["places"] if row["id"] == place_id)


def _write(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes([size % 251]) * size)
    return path


def test_first_count_progress_does_not_call_an_undecided_folder_zero(tmp_path):
    progress = []
    known = tmp_path / "known"
    known.mkdir()
    _write(known / "result", 3)
    unknown = {"id": "project_folders", "group": "work", "roots": [], "undecided": "Choose the project folder"}
    ticks = iter(range(1000))  # every traversal tick crosses the reporting interval
    count = measure(tmp_path, places=[unknown, _place("task_results", known)], progress=progress.append,
                    clock=lambda: float(next(ticks)))
    assert _row(count, "project_folders")["bytes"] is None
    assert progress and any(row["places_done"] == 1 for row in progress)
    assert all(part["name"] != "project_folders" for row in progress for part in row["done"])


@pytest.fixture(params=["openat", "by-path"])
def walk(request, monkeypatch):
    """Both traversals: directory-relative no-follow opens, and the by-path fallback a platform
    without them (Windows) takes. Neither may follow a link."""
    if request.param == "openat" and not getattr(storage, "_RELATIVE", True):
        pytest.skip("this platform has no directory-relative no-follow open")
    monkeypatch.setattr(storage, "_RELATIVE", request.param == "openat", raising=False)
    return request.param


# ------------------------------------------------------------- A1: no bound


def test_a_deep_and_wide_tree_is_counted_completely(tmp_path, walk):
    """The old walk stopped at 32 levels and 250 000 entries; the complete count has no bound."""
    data = tmp_path / "data"
    deep = data / "memory"
    for level in range(60):
        deep = deep / f"l{level}"
    _write(deep / "leaf.bin", 7)
    wide = data / "observability"
    wide.mkdir(parents=True)
    for index in range(2_000):
        (wide / f"f{index}").write_bytes(b"x")
    record = _count(data)
    assert record["unreadable"] == 0
    assert (_row(record, "memory_skills")["files"], _row(record, "memory_skills")["logical_bytes"]) == (1, 7)
    assert (_row(record, "model_calls")["files"], _row(record, "model_calls")["logical_bytes"]) == (2_000, 2_000)
    assert record["files"] == 2_001


# --------------------------------------------------------- A2, A3: identity


def test_hard_links_count_once_and_equal_files_twice(tmp_path, walk):
    data = tmp_path / "data"
    first = _write(data / "observability" / "call.json", 12)
    (data / "task_results").mkdir(parents=True)
    (data / "logs").mkdir(parents=True)
    os.link(first, data / "task_results" / "same-bytes.json")
    os.link(first, data / "logs" / "same-again.json")
    _write(data / "task_results" / "twin-a.bin", 12)
    _write(data / "task_results" / "twin-b.bin", 12)
    record = _count(data)
    assert (_row(record, "model_calls")["files"], _row(record, "model_calls")["logical_bytes"]) == (1, 12)
    assert (_row(record, "task_results")["files"], _row(record, "task_results")["logical_bytes"]) == (2, 24)
    assert (_row(record, "data_other")["files"], _row(record, "data_other")["logical_bytes"]) == (0, 0)
    assert record["files"] == 3 and record["logical_bytes"] == 36


def _identityless(info, *, nlink=0):
    """A Windows-shaped stat: ``DirEntry.stat()`` there carries no inode, device or link count."""
    extra = {name: getattr(info, name) for name in ("st_file_attributes", "st_reparse_tag") if hasattr(info, name)}
    return os.stat_result((info.st_mode, 0, 0, nlink, info.st_uid, info.st_gid, info.st_size,
                           int(info.st_atime), int(info.st_mtime), int(info.st_ctime)), extra)


def test_a_listing_without_identity_asks_each_file_and_folder_itself(tmp_path, monkeypatch):
    """The Windows path: links still count once, and a folder that is another place's root is
    still recognised by the identity the by-path stat reports."""
    data = tmp_path / "data"
    first = _write(data / "task_results" / "a.bin", 4)
    os.link(first, data / "task_results" / "b.bin")
    _write(data / "task_results" / "single.bin", 3)
    _write(data / "state" / "headless_tasks" / "t" / "drive.bin", 100)
    _write(data / "state" / "own.bin", 5)
    listing_stat = storage._listing_stat
    monkeypatch.setattr(storage, "_listing_stat", lambda entry, fd: _identityless(listing_stat(entry, fd)))
    record = _count(data)
    assert (_row(record, "task_results")["files"], _row(record, "task_results")["logical_bytes"]) == (2, 7)
    assert (_row(record, "app_state")["files"], _row(record, "app_state")["logical_bytes"]) == (1, 5)
    assert (_row(record, "work_folders")["files"], _row(record, "work_folders")["logical_bytes"]) == (1, 100)


def test_links_inside_a_place_are_never_counted_or_walked(tmp_path, walk):
    data, outside = tmp_path / "data", tmp_path / "outside"
    _write(outside / "big.bin", 1000)
    _write(data / "task_results" / "own.bin", 3)
    (data / "task_results" / "file-link").symlink_to(outside / "big.bin")
    (data / "task_results" / "dir-link").symlink_to(outside, target_is_directory=True)
    row = _row(_count(data), "task_results")
    assert (row["logical_bytes"], row["files"], row["unreadable"]) == (3, 1, 0)


def test_a_linked_root_is_unknown_and_its_target_is_not_walked(tmp_path, walk):
    data, outside = tmp_path / "data", tmp_path / "outside"
    _write(outside / "secret.bin", 14)
    data.mkdir()
    (data / "observability").symlink_to(outside, target_is_directory=True)
    row = _row(_count(data), "model_calls")
    assert (row["present"], row["bytes"], row["unreadable"]) == (True, None, 1)


# ------------------------------------------------------------ A4: nesting


def test_nested_roots_count_once_in_their_own_place_and_siblings_land_in_the_remainder(tmp_path, walk):
    data = tmp_path / "data"
    _write(data / "state" / "headless_tasks" / "t1" / "d.bin", 11)
    _write(data / "state" / "skills" / "s" / "s.bin", 13)
    _write(data / "state" / "x.json", 2)
    _write(data / "state" / "review_checkouts" / "r.bin", 17)  # a sibling that is no root
    worktrees = data / "copies"  # a configured root inside the data folder
    _write(worktrees / "wt1" / "w.bin", 19)
    _write(data / "logs" / "l.log", 23)
    record = _count(data, get_subagent_worktree_root=str(worktrees))
    sizes = {row["id"]: row["logical_bytes"] for row in record["places"]}
    assert sizes["work_folders"] == 11
    assert sizes["memory_skills"] == 13
    assert sizes["app_state"] == 2 + 17, "state minus the roots other places own"
    assert sizes["working_copies"] == 19
    assert sizes["data_other"] == 23, "the data folder minus every root above"
    assert record["logical_bytes"] == sum(sizes.values()) == 11 + 13 + 2 + 17 + 19 + 23
    assert [part["name"] for part in _row(record, "data_other")["parts"]] == ["logs"]
    assert [part["name"] for part in _row(record, "memory_skills")["parts"]] == ["state/skills/s"]


def test_the_engine_lists_its_named_parts_whatever_their_rank_and_other_places_do_not(tmp_path, walk):
    """The client's three-part engine line reads `projects` and `profiles` by name, so the walker lists
    them beyond the largest eight instead of folding them into the remainder; under any other place
    the same names rank like every other part."""
    data = tmp_path / "data"
    for base in (data / storage.ENGINE_DIR, data / "state"):
        for index in range(storage._LARGEST_PARTS):
            _write(base / f"big{index}" / "b.bin", 20_000 + index)
        _write(base / "projects" / "p.bin", 5)
        _write(base / "profiles" / "q.bin", 3)
    record = _count(data)
    engine = _row(record, "engine")
    names = [part["name"] for part in engine["parts"]]
    assert len(names) == storage._LARGEST_PARTS + 2 and set(names[storage._LARGEST_PARTS:]) == {"projects", "profiles"}
    assert engine["rest"] == {"count": 0, "bytes": 0}
    assert sum(part["bytes"] for part in engine["parts"]) == engine["bytes"]
    state = _row(record, "app_state")
    assert len(state["parts"]) == storage._LARGEST_PARTS and not {"projects", "profiles"} & {part["name"] for part in state["parts"]}
    assert state["rest"]["count"] == 2 and state["rest"]["bytes"] == state["bytes"] - sum(part["bytes"] for part in state["parts"])


def test_two_places_resolving_to_one_folder_walk_it_once(tmp_path, walk):
    data = tmp_path / "data"
    _write(data / "projects" / "p" / "p.bin", 31)
    record = _count(data, get_subagent_projects_root=str(data / "projects"))
    assert _row(record, "memory_skills")["logical_bytes"] == 31
    assert (_row(record, "project_folders")["logical_bytes"], _row(record, "project_folders")["roots"]) == (0, [])
    assert record["logical_bytes"] == 31


# ------------------------------------------------- A5, A6: unreadable, vanished


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs a folder this user cannot list")
def test_an_unreadable_folder_is_counted_as_such_and_the_walk_finishes(tmp_path, walk):
    data = tmp_path / "data"
    _write(data / "observability" / "open" / "a.bin", 5)
    closed = data / "observability" / "closed"
    _write(closed / "hidden.bin", 50)
    closed.chmod(0)
    try:
        record = _count(data)
    finally:
        closed.chmod(stat.S_IRWXU)
    row = _row(record, "model_calls")
    assert (row["bytes"] is not None, row["logical_bytes"], row["unreadable"]) == (True, 5, 1)
    assert record["unreadable"] == 1 and record["logical_bytes"] == 5, "the headline exists"


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs a folder this user cannot list")
def test_a_root_that_cannot_be_listed_is_unknown_never_zero(tmp_path, walk):
    data = tmp_path / "data"
    root = data / "archive"
    _write(root / "a.bin", 9)
    root.chmod(0)
    try:
        row = _row(_count(data), "archives")
    finally:
        root.chmod(stat.S_IRWXU)
    assert (row["present"], row["bytes"], row["unreadable"]) == (True, None, 1)


def test_a_folder_vanishing_mid_walk_counts_as_empty(tmp_path, monkeypatch, walk):
    data = tmp_path / "data"
    victim = data / "task_results" / "victim"
    _write(victim / "gone.bin", 40)
    _write(data / "task_results" / "own.bin", 3)
    real = storage._listing_stat

    def vanish_after_stat(entry, directory_fd):
        info = real(entry, directory_fd)
        if entry.name == "victim" and victim.exists():
            (victim / "gone.bin").unlink()
            victim.rmdir()
        return info

    monkeypatch.setattr(storage, "_listing_stat", vanish_after_stat)
    row = _row(_count(data), "task_results")
    assert (row["logical_bytes"], row["files"], row["unreadable"]) == (3, 1, 0)


def _swap_mid_walk(tmp_path, monkeypatch, listing=lambda info: info):
    """Count ``task_results`` while its folder ``victim`` turns into a link to an outside folder just
    after the walk read its listing entry; LISTING shapes what that entry reports."""
    data, outside = tmp_path / "data", tmp_path / "outside"
    _write(outside / "secret.bin", 100)
    victim = data / "task_results" / "victim"
    victim.mkdir(parents=True)
    _write(data / "task_results" / "own.bin", 3)
    real = storage._listing_stat

    def swap_after_stat(entry, directory_fd):
        info = real(entry, directory_fd)
        if entry.name == "victim" and not victim.is_symlink():
            victim.rmdir()
            victim.symlink_to(outside, target_is_directory=True)
        return listing(info)

    monkeypatch.setattr(storage, "_listing_stat", swap_after_stat)
    return _row(_count(data), "task_results")


def test_a_folder_swapped_for_a_link_mid_walk_is_never_listed_through_it(tmp_path, monkeypatch, walk):
    row = _swap_mid_walk(tmp_path, monkeypatch)
    assert (row["logical_bytes"], row["unreadable"]) == (3, 1)


def _windows_listing(info):
    """A Windows listing entry (``os.DirEntry.stat``): type, size and attributes, but no inode,
    device, link count or blocks, so the walk asks the entry itself for those."""
    return SimpleNamespace(st_mode=info.st_mode, st_size=info.st_size, st_ino=0, st_dev=0, st_nlink=0,
                           st_file_attributes=getattr(info, "st_file_attributes", 0))


def test_a_windows_listing_counts_a_folder_turned_link_unreadable_and_still_walks_the_rest(tmp_path, monkeypatch):
    """A listing without identity sends the by-path walk to each folder itself; a folder that became
    a link since its listing entry is unreadable, as at the recheck before a listing, never a quietly
    skipped link, and an untouched folder beside it is still walked."""
    monkeypatch.setattr(storage, "_RELATIVE", False, raising=False)
    _write(tmp_path / "data" / "task_results" / "kept" / "inner.bin", 5)
    row = _swap_mid_walk(tmp_path, monkeypatch, listing=_windows_listing)
    assert (row["logical_bytes"], row["unreadable"]) == (8, 1)


# ----------------------------------------------------- A7, A8, A9: the file


def test_first_count_saves_latest_only_then_previous_with_growth(tmp_path):
    data = tmp_path / "data"
    _write(data / "observability" / "a.bin", 10)
    first = _count(data)
    save_count(data, first)
    saved = load_saved(data)
    assert saved["latest"]["finished_at"] == first["finished_at"] and saved["previous"] is None
    assert "growth_bytes" not in with_growth(saved["latest"], saved["previous"])

    _write(data / "observability" / "b.bin", 30)
    _write(data / "task_results" / "r.bin", 5)
    second = _count(data)
    second["unreadable"] = 1
    _row(second, "task_results")["unreadable"] = 1
    save_count(data, second)
    saved = load_saved(data)
    assert saved["previous"]["finished_at"] == first["finished_at"]
    assert set(saved["previous"]) == {"finished_at", "bytes", "files", "unreadable", "places"}
    count = with_growth(saved["latest"], saved["previous"])
    assert count["growth_since"] == first["finished_at"]
    assert count["growth_bytes"] == second["bytes"] - first["bytes"]
    assert count["growth_flag"] == "unreadable"
    calls = next(row for row in count["places"] if row["id"] == "model_calls")
    assert calls["growth_bytes"] == calls["bytes"] - _row(first, "model_calls")["bytes"] and "growth_flag" not in calls
    results = next(row for row in count["places"] if row["id"] == "task_results")
    assert results["growth_flag"] == "unreadable"

    third = _count(data)
    save_count(data, third)
    assert load_saved(data)["previous"]["finished_at"] == second["finished_at"], "only the last two counts are kept"


def test_a_count_that_raises_or_cannot_write_leaves_the_file_byte_identical(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _write(data / "observability" / "a.bin", 10)
    save_count(data, _count(data))
    path = data / storage.MEASUREMENT_FILE
    before = path.read_bytes()

    def boom(*_args, **_kwargs):
        raise OSError("disk went away")

    monkeypatch.setattr(storage, "_walk_roots", boom)
    with pytest.raises(OSError):
        _count(data)
    assert path.read_bytes() == before
    monkeypatch.undo()
    monkeypatch.setattr(storage, "atomic_write_json", boom)
    with pytest.raises(OSError):
        save_count(data, _count(data))
    assert path.read_bytes() == before


@pytest.mark.parametrize("content", [
    b"not json", b"[]", b'{"_schema_version": 2, "latest": {}}', b'{"_schema_version": 1}',
    b'{"_schema_version": 1, "latest": {"finished_at": "x", "bytes": "many", "files": 1, "unreadable": 0, "places": [], "started_at": "x"}}',
    b'{"_schema_version": 1, "latest": {"finished_at": "x", "bytes": 1, "files": 1, "unreadable": 0, "places": [1], "started_at": "x"}}',
])
def test_a_corrupt_or_foreign_file_is_no_count_yet(tmp_path, content):
    data = tmp_path / "data"
    (data / "state").mkdir(parents=True)
    (data / storage.MEASUREMENT_FILE).write_bytes(content)
    assert load_saved(data) is None
    counter = StorageCounter()
    projection = counter.projection(data)
    assert projection["count"] is None and projection["error"] is None and projection["counting"] is None


@pytest.mark.parametrize("previous", [{}, "a summary", 7, {"finished_at": "2026-10-16T15:10:05Z", "bytes": 1, "files": 1, "unreadable": 0, "places": "none"},
                                      {"finished_at": "2026-10-16T15:10:05Z", "bytes": 1, "files": 1, "unreadable": 0, "places": [{"id": 3}]}])
def test_a_malformed_previous_loses_growth_never_the_count_or_the_reader(tmp_path, previous):
    """`previous` is validated like `latest`: a hand-edited or foreign summary is dropped, so the
    saved count still shows (without growth) and no reader raises."""
    from ouroboros.utils import atomic_write_json

    data = tmp_path / "data"
    _write(data / "observability" / "a.bin", 10)
    latest = _count(data)
    atomic_write_json(data / storage.MEASUREMENT_FILE, {"_schema_version": storage.SCHEMA_VERSION, "latest": latest, "previous": previous})
    saved = load_saved(data)
    assert saved == {"latest": latest, "previous": None}
    count = with_growth(saved["latest"], saved["previous"])
    assert count["bytes"] == latest["bytes"] and "growth_bytes" not in count and "growth_since" not in count
    assert StorageCounter().projection(data)["count"]["bytes"] == latest["bytes"]


# --------------------------------------------------- A10, A11, A14: the owner


def _fake_spawn(monkeypatch, lines, *, exit_code=0, hold=None, started=None):
    """A child that prints LINES; HOLD (an Event) keeps it alive until set; STARTED is set on spawn."""
    import ouroboros.startup_historical_audit as audit

    spawned = []

    class Child:
        def __init__(self):
            self.pid = 4242
            self._killed = threading.Event()
            self.stdout = self
            self._lines = list(lines)

        def readline(self, _limit=0):
            if self._lines:
                return self._lines.pop(0)
            if hold is not None:
                while not hold.wait(0.01):
                    if self._killed.is_set():
                        return b""
            return b""

        def close(self):
            pass

        def wait(self):
            return -9 if self._killed.is_set() else exit_code

        def poll(self):
            return None if (hold is not None and not hold.is_set() and not self._killed.is_set()) else self.wait()

        def kill(self):
            self._killed.set()

    def spawn(data_root, repo_dir, purpose, **_kwargs):
        child = Child()
        spawned.append((Path(data_root), Path(repo_dir), purpose, child))
        if started is not None:
            started.set()
        return child

    monkeypatch.setattr(audit, "spawn_maintenance_job", spawn)
    return spawned


def _join(counter: StorageCounter):
    thread = counter._thread
    if thread is not None:
        thread.join(timeout=30)
        assert not thread.is_alive()


def _observed(monkeypatch, data: Path, operation):
    """OPERATION's own effect on DATA: (its result, what ``os.scandir`` listed while it ran, the data
    folder's entries with their mtimes before and after). The recorder sees OPERATION alone: on some
    Pythons pathlib's own walk lists through ``os.scandir``, so both snapshots are taken without it."""
    def entries():
        return sorted((str(path), path.stat().st_mtime_ns) for path in data.rglob("*"))

    before, listings, real = entries(), [], os.scandir
    with monkeypatch.context() as observing:
        observing.setattr(os, "scandir", lambda path=".": listings.append(path) or real(path))
        result = operation()
    return result, listings, before, entries()


def test_get_projection_is_passive_and_fresh_free_space_is_read_each_time(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _write(data / "observability" / "a.bin", 10)
    counter = StorageCounter()
    volumes = []

    def disk_usage(path):  # a volume whose free space falls between two reads
        volumes.append(path)
        return SimpleNamespace(total=4_000, free=4_000 - 100 * len(volumes))

    monkeypatch.setattr(shutil, "disk_usage", disk_usage)
    projections, listings, before, after = _observed(
        monkeypatch, data, lambda: [counter.projection(data) for _ in range(2)])
    for projection in projections:
        assert projection == {"counting": None, "error": None, "volume": projection["volume"], "retention_days": projection["retention_days"],
                              "count": None, "free": projection["free"], "removers": projection["removers"]}
        assert projection["removers"]["app_state"][0] == {"id": "runtime_retirement", "state": "never", "at": None}
        assert projection["free"]["state"] == "idle"
        assert isinstance(projection["volume"]["free_bytes"], int) and isinstance(projection["retention_days"], int)
    assert [projection["volume"]["free_bytes"] for projection in projections] == [3_900, 3_800]
    assert volumes == [str(data)] * 2
    assert listings == [] and after == before


@pytest.mark.parametrize("intrusion", ["walk", "write"])
def test_the_passive_read_observer_sees_a_real_walk_and_a_persistent_write(tmp_path, monkeypatch, intrusion):
    """The observer above is live whichever Python runs it: a projection followed by the storage walker
    over the data folder is seen listing it, one followed by a new file under the data folder changes
    its entries, and each intrusion leaves the other half of the oracle quiet."""
    data = tmp_path / "data"
    _write(data / "observability" / "a.bin", 10)
    counter = StorageCounter()

    def intruding():
        projection = counter.projection(data)
        if intrusion == "walk":
            measure(data, places=[_place("data", data)], sleep=lambda _s: None)
        else:
            _write(data / "state" / "written.json", 2)
        return projection

    _, listings, before, after = _observed(monkeypatch, data, intruding)
    if intrusion == "walk":
        assert listings and after == before
    else:
        assert listings == [] and after != before


def test_sixteen_concurrent_starts_spawn_one_child_and_the_finished_count_is_read_back(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _write(data / "observability" / "a.bin", 10)
    release, spawned_at = threading.Event(), threading.Event()
    record = _count(data)
    progress = {"event": "progress", "place": "engine", "current": "projects", "places_done": 1, "places_total": 12,
                "bytes": 55, "files": 2, "done": [{"name": "model_calls", "bytes": 55, "files": 2}]}
    final = {"status": "finished", "finished_at": record["finished_at"], "bytes": record["bytes"], "files": record["files"],
             "unreadable": 0, "duration_seconds": 0.1, "io_priority": "utility"}
    spawned = _fake_spawn(monkeypatch, [json.dumps(progress).encode() + b"\n", json.dumps(final).encode() + b"\n"],
                          hold=release, started=spawned_at)
    counter = StorageCounter()
    gate, answers = threading.Barrier(16), []

    def request():
        gate.wait(30)
        answers.append(counter.start(data, tmp_path))

    threads = [threading.Thread(target=request) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert sorted(answers) == [False] * 15 + [True]
    try:
        assert spawned_at.wait(10) and len(spawned) == 1
        assert spawned[0][2] == "storage" and spawned[0][0] == data
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and counter.projection(data)["counting"]["place"] != "engine":
            time.sleep(0.01)
        counting = counter.projection(data)["counting"]
        assert (counting["place"], counting["current"], counting["places_done"], counting["bytes"]) == ("engine", "projects", 1, 55)
        assert counting["done"] == [{"name": "model_calls", "bytes": 55, "files": 2}]
        assert counter.live_children() == [spawned[0][3]]
        save_count(data, record)  # what the real child writes before its final line
    finally:
        release.set()
        _join(counter)
    projection = counter.projection(data)
    assert projection["counting"] is None and projection["error"] is None
    assert projection["count"]["finished_at"] == record["finished_at"] and projection["count"]["bytes"] == record["bytes"]
    assert counter.live_children() == []
    rows = [json.loads(line) for line in (data / "logs" / "supervisor.jsonl").read_text().splitlines()]
    assert [(row["type"], row["phase"], row["bytes"]) for row in rows] == [("storage_count", "finished", record["bytes"])]


def test_a_count_token_omits_an_unchanged_count(tmp_path):
    data = tmp_path / "data"
    _write(data / "observability" / "a.bin", 10)
    record = _count(data)
    save_count(data, record)
    counter = StorageCounter()
    assert counter.projection(data)["count"]["finished_at"] == record["finished_at"]
    assert "count" not in counter.projection(data, count_token=record["finished_at"])
    assert counter.projection(data, count_token="2000-01-01T00:00:00+00:00")["count"]["finished_at"] == record["finished_at"]
    counter_without = StorageCounter()
    (data / storage.MEASUREMENT_FILE).unlink()
    assert counter_without.projection(data, count_token=record["finished_at"])["count"] is None


def test_the_saved_file_is_re_read_only_when_its_identity_changes(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _write(data / "observability" / "a.bin", 10)
    save_count(data, _count(data))
    counter = StorageCounter()
    reads = []
    real = storage.load_saved
    monkeypatch.setattr(storage, "load_saved", lambda root: reads.append(root) or real(root))
    counter.projection(data)
    counter.projection(data)
    assert len(reads) == 1, "an unchanged file is not parsed again"
    _write(data / "observability" / "b.bin", 20)
    time.sleep(0.01)
    save_count(data, _count(data))  # rolls the previous count: its own read is not the owner's
    baseline = len(reads)
    assert _row(counter.projection(data)["count"], "model_calls")["files"] == 2
    counter.projection(data)
    assert len(reads) == baseline + 1, "a changed file is parsed once"


def test_a_failed_or_killed_child_keeps_the_saved_count_and_says_so(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _write(data / "observability" / "a.bin", 10)
    record = _count(data)
    save_count(data, record)
    counter = StorageCounter()
    _fake_spawn(monkeypatch, [json.dumps({"status": "failed", "message": "The data folder could not be found."}).encode() + b"\n"], exit_code=1)
    assert counter.start(data, tmp_path) is True
    _join(counter)
    projection = counter.projection(data)
    assert projection["error"] == {"at": projection["error"]["at"], "message": "The data folder could not be found.", "stopped": False}
    assert projection["count"]["finished_at"] == record["finished_at"], "the saved count stays"
    _fake_spawn(monkeypatch, [], exit_code=137)
    assert counter.start(data, tmp_path) is True
    _join(counter)
    assert counter.projection(data)["error"]["message"] == "The count ended without a result."
    rows = [json.loads(line)["phase"] for line in (data / "logs" / "supervisor.jsonl").read_text().splitlines()]
    assert rows == ["failed", "failed"]


# ------------------------------------------------------------ A13: Stop


def test_cancel_kills_the_child_and_stop_refuses_a_new_count(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _write(data / "observability" / "a.bin", 10)
    save_count(data, _count(data))
    hold, spawned_at = threading.Event(), threading.Event()
    spawned = _fake_spawn(monkeypatch, [b'{"event":"progress","place":"model_calls","current":"","places_done":0,"places_total":12,"bytes":1,"files":1,"done":[]}\n'],
                          hold=hold, started=spawned_at)
    counter = StorageCounter()
    assert counter.cancel() is False, "nothing runs"
    assert counter.start(data, tmp_path) is True
    assert spawned_at.wait(10)
    child = spawned[0][3]
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and counter.live_children() != [child]:
            time.sleep(0.01)  # the owner thread publishes the child right after the spawn returns
        assert counter.live_children() == [child]
        assert counter.cancel() is True
    finally:
        hold.set()
        _join(counter)
    assert child._killed.is_set()
    assert counter.live_children() == []
    projection = counter.projection(data)
    assert projection["counting"] is None and projection["error"]["stopped"] is True
    assert projection["error"]["message"] == "Stopped before it finished." and projection["count"] is not None
    assert [json.loads(line)["phase"] for line in (data / "logs" / "supervisor.jsonl").read_text().splitlines()] == ["stopped"]
    counter.stop()
    assert counter.start(data, tmp_path) is False, "after Stop nothing starts"


def test_stop_reaches_a_child_spawned_moments_before(tmp_path, monkeypatch):
    """The owner may Stop between the thread start and the spawn: the spawned child is killed."""
    data = tmp_path / "data"
    hold = threading.Event()
    counter = StorageCounter()
    spawned = _fake_spawn(monkeypatch, [], hold=hold)
    import ouroboros.startup_historical_audit as audit

    real_spawn = audit.spawn_maintenance_job

    def spawn_after_stop(*args, **kwargs):
        counter.stop()  # the owner's Stop lands while the spawn is in flight
        return real_spawn(*args, **kwargs)

    monkeypatch.setattr(audit, "spawn_maintenance_job", spawn_after_stop)
    assert counter.start(data, tmp_path) is True
    _join(counter)
    assert spawned[0][3]._killed.is_set()
    assert counter.projection(data)["error"]["stopped"] is True


def test_the_two_stop_sites_and_the_census_name_the_storage_count_and_the_free_owner():
    server = (REPO / "server.py").read_text(encoding="utf-8")
    control = (REPO / "ouroboros" / "server_control.py").read_text(encoding="utf-8")
    assert "storage_count.stop()" in server and "storage_count.live_children()" in server
    assert "storage_free.stop()" in server and server.count("storage_free.live_children()") == 2, "restart adoption and the exit census"
    assert 'attempt("storage-count", storage_count.stop, native=True)' in control
    assert 'attempt("storage-free", storage_free.stop, native=True)' in control


def test_the_production_spawn_forwards_the_configured_work_roots_to_the_child(tmp_path, monkeypatch):
    """`spawn_maintenance_job` builds a narrow environment; the three work roots this process
    resolved (Docker keeps them under /ouroboros/work, outside HOME) reach the child, and a fresh
    interpreter with exactly that environment resolves the same roots."""
    from ouroboros import config, process_custody
    from ouroboros.startup_historical_audit import spawn_maintenance_job

    keys = ("OUROBOROS_SUBAGENT_WORKTREE_ROOT", "OUROBOROS_SUBAGENT_PROJECTS_ROOT", "OUROBOROS_DELIVERABLES_ROOT")
    for key, leaf in zip(keys, ("copies", "projects", "deliverables")):
        (tmp_path / leaf).mkdir()
        monkeypatch.setenv(key, str(tmp_path / leaf))
    expected = [config.get_subagent_worktree_root(), config.get_subagent_projects_root(), config.get_deliverables_root()]
    assert expected == [str(tmp_path / leaf) for leaf in ("copies", "projects", "deliverables")]
    captured = {}
    monkeypatch.setattr(process_custody, "spawn_supervised", lambda argv, **kwargs: captured.update(argv=argv, **kwargs) or object())
    spawn_maintenance_job(tmp_path / "data", REPO, "storage")
    assert captured["argv"][1:3] == ["-m", "ouroboros.storage"]
    env = captured["env"]
    assert [env[key] for key in keys] == expected
    code = ("import json; from ouroboros import config; print(json.dumps([config.get_subagent_worktree_root(), "
            "config.get_subagent_projects_root(), config.get_deliverables_root()]))")
    run = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(REPO), capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]
    assert json.loads(run.stdout.strip().splitlines()[-1]) == expected


# ------------------------------------------------------------ A12: the child


def test_the_count_child_streams_progress_writes_the_file_and_reports_its_priority(tmp_path):
    data = tmp_path / "data"
    _write(data / "observability" / "a.bin", 10)
    _write(data / "task_results" / "b.bin", 20)
    env = {**os.environ, "OUROBOROS_DATA_DIR": str(data), "OUROBOROS_REPO_DIR": str(REPO),
           "OUROBOROS_SETTINGS_PATH": str(tmp_path / "settings.json"), "PYTHONPATH": str(REPO),
           "OUROBOROS_SUBAGENT_WORKTREE_ROOT": str(tmp_path / "wt"), "OUROBOROS_SUBAGENT_PROJECTS_ROOT": str(tmp_path / "projects"),
           "OUROBOROS_DELIVERABLES_ROOT": str(tmp_path / "deliverables")}
    run = subprocess.run([sys.executable, "-m", "ouroboros.storage", "--data-root", str(data)],
                         capture_output=True, text=True, env=env, cwd=str(REPO), timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]
    lines = [json.loads(line) for line in run.stdout.splitlines() if line.strip()]
    assert lines and lines[-1]["status"] == "finished"
    final = lines[-1]
    assert all(line.get("event") == "progress" for line in lines[:-1])
    assert final["io_priority"] and isinstance(final["io_priority"], str)
    saved = load_saved(data)
    assert saved is not None and saved["latest"]["finished_at"] == final["finished_at"]
    assert saved["latest"]["io_priority"] == final["io_priority"]
    assert saved["latest"]["bytes"] == final["bytes"] >= 30
    assert {place["id"] for place in saved["latest"]["places"]} == {place_id for place_id, _group, _spec in storage.PLACES}
    assert next(place for place in saved["latest"]["places"] if place["id"] == "program")["roots"] == [os.path.realpath(str(REPO))]

    sizing = subprocess.run([sys.executable, "-m", "ouroboros.storage", "--data-root", str(data), "--size-paths"],
                            input=f"{json.dumps(str(data / 'task_results'))}\n{json.dumps(str(tmp_path / 'absent'))}\n",
                            capture_output=True, text=True, env=env, cwd=str(REPO), timeout=120)
    assert sizing.returncode == 0, sizing.stderr[-2000:]
    rows = [json.loads(line) for line in sizing.stdout.splitlines() if line.strip()]
    sizes = {row["path"]: row for row in rows if row.get("event") == "size"}
    assert sizes[str(data / "task_results")]["files"] == 1 and sizes[str(data / "task_results")]["bytes"] >= 20
    assert sizes[str(tmp_path / "absent")]["bytes"] == 0 and rows[-1] == {"status": "finished", "io_priority": final["io_priority"], "paths": 2}
    assert load_saved(data)["latest"]["finished_at"] == final["finished_at"], "sizing saves nothing"


@pytest.mark.parametrize("encoding", ["utf-8", "cp1252"])
def test_the_sizing_child_matches_a_non_ascii_path_whatever_the_pipe_code_page(tmp_path, encoding):
    """Windows decodes a pipe in the ANSI code page; the request and the answer are JSON (ASCII on
    the pipe), so a path with a non-ASCII user name sizes and matches under cp1252 as under UTF-8."""
    data = tmp_path / "data"
    target = data / "Пользователь José" / "task_drives" / "done-old"
    _write(target / "scratch.bin", 4096)
    env = {**os.environ, "OUROBOROS_DATA_DIR": str(data), "OUROBOROS_REPO_DIR": str(REPO), "PYTHONPATH": str(REPO),
           "OUROBOROS_SETTINGS_PATH": str(tmp_path / "settings.json"), "PYTHONIOENCODING": encoding}
    run = subprocess.run([sys.executable, "-m", "ouroboros.storage", "--data-root", str(data), "--size-paths"],
                         input=(json.dumps(str(target)) + "\n").encode("ascii"), capture_output=True, env=env, cwd=str(REPO), timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]
    rows = [json.loads(line) for line in run.stdout.decode("ascii").splitlines() if line.strip()]
    sizes = {row["path"]: row for row in rows if row.get("event") == "size"}
    assert sizes[str(target)]["files"] == 1 and sizes[str(target)]["bytes"] >= 4096 and sizes[str(target)]["unreadable"] == 0
    assert rows[-1]["status"] == "finished" and rows[-1]["paths"] == 1


def test_the_child_reports_a_missing_data_folder_as_failed_and_writes_nothing(tmp_path):
    env = {**os.environ, "OUROBOROS_DATA_DIR": str(tmp_path / "absent"), "OUROBOROS_REPO_DIR": str(REPO),
           "OUROBOROS_SETTINGS_PATH": str(tmp_path / "settings.json"), "PYTHONPATH": str(REPO)}
    run = subprocess.run([sys.executable, "-m", "ouroboros.storage", "--data-root", str(tmp_path / "absent")],
                         capture_output=True, text=True, env=env, cwd=str(REPO), timeout=120)
    assert run.returncode == 1
    assert json.loads(run.stdout.strip().splitlines()[-1]) == {"status": "failed", "message": "The data folder could not be found."}
    assert not (tmp_path / "absent").exists()


def test_the_count_child_lowers_its_disk_priority_and_reports_only_what_took():
    """The positive call runs in a child of its own (it changes a process's policy for good) and
    the policy is read back: on macOS IOPOL_UTILITY (4), never IOPOL_STANDARD (5), the tier the
    recorded "utility" must mean."""
    code = ("import ctypes, json, sys\nfrom ouroboros.storage import _lower_io_priority\n"
            "reported = _lower_io_priority()\n"
            "policy = ctypes.CDLL(None, use_errno=True).getiopolicy_np(0, 0) if sys.platform == 'darwin' else None\n"
            "print(json.dumps({'reported': reported, 'disk_policy': policy}))")
    run = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": str(REPO)}, cwd=str(REPO),
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]
    facts = json.loads(run.stdout.strip().splitlines()[-1])
    assert facts["reported"] in {"utility", "best-effort-low", "background", "unchanged"}
    if sys.platform == "darwin":
        assert facts == {"reported": "utility", "disk_policy": 4}


def test_disk_priority_is_best_effort_and_never_fatal(monkeypatch):
    import ctypes

    def refuse(*_args, **_kwargs):
        raise OSError("no such library")

    monkeypatch.setattr(ctypes, "CDLL", refuse)
    monkeypatch.setattr(ctypes, "windll", None, raising=False)
    assert storage._lower_io_priority() == "unchanged"


# ---------------------------------------------------------------- routes


def test_storage_route_surface(tmp_path):
    routes = storage_routes(tmp_path, repo_dir=tmp_path)
    assert {(route.path, frozenset(route.methods or ())) for route in routes} == {
        ("/api/storage", frozenset({"GET", "HEAD"})),
        ("/api/storage/measure", frozenset({"POST"})),
        ("/api/storage/measure/cancel", frozenset({"POST"})),
        ("/api/storage/free/preview", frozenset({"POST"})),
        ("/api/storage/free", frozenset({"POST"})),
        ("/api/storage/free/cancel", frozenset({"POST"})),
        ("/api/storage/free/items", frozenset({"GET", "HEAD"})),
    }


def test_mounted_storage_routes_read_passively_until_the_owner_measures(tmp_path, monkeypatch):
    """The routes as gateway.router mounts them: GET never lists a folder or writes one; the POST
    starts one child, its answer and the GET carry the same envelope, cancel kills it."""
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from ouroboros.gateway.router import collect_routes

    data = tmp_path / "data"
    _write(data / "task_results" / "r.bin", 10)
    hold = threading.Event()
    spawned = _fake_spawn(monkeypatch, [b'{"event":"progress","place":"engine","current":"","places_done":1,"places_total":12,"bytes":1,"files":1,"done":[]}\n'], hold=hold)
    monkeypatch.setattr(storage, "count", StorageCounter())
    routes = [route for route in collect_routes(data_dir=data) if getattr(route, "path", "").startswith("/api/storage")]
    assert len(routes) == 7
    # The owner's cancel writes one supervisor row (logs/); nothing else under the data folder may change.
    snapshot = lambda: sorted((str(path), path.stat().st_mtime_ns) for path in data.rglob("*")  # noqa: E731
                              if "logs" not in path.relative_to(data).parts)
    before = snapshot()
    listings = []
    real = os.scandir
    monkeypatch.setattr(os, "scandir", lambda path: listings.append(path) or real(path))

    with TestClient(Starlette(routes=routes)) as client:
        for _ in range(2):
            response = client.get("/api/storage")
            assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
            body = response.json()
            assert body["count"] is None and body["counting"] is None and body["error"] is None
            assert set(body) == {"count", "counting", "error", "volume", "retention_days", "free", "removers"}
            assert body["free"]["state"] == "idle"
        assert listings == [], "a GET must not walk the data folder"

        started = client.post("/api/storage/measure")
        assert started.status_code == 202 and started.json()["started"] is True
        joined = client.post("/api/storage/measure")
        assert joined.status_code == 202 and joined.json()["started"] is False
        assert len(spawned) == 1
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not client.get("/api/storage").json()["counting"]["place"]:
            time.sleep(0.01)
        assert client.get("/api/storage").json()["counting"]["place"] == "engine"
        cancelled = client.post("/api/storage/measure/cancel")
        assert cancelled.status_code == 202
        _join(storage.count)
        after = client.get("/api/storage").json()
        assert after["counting"] is None and after["error"]["stopped"] is True
        assert spawned[0][3]._killed.is_set()
    assert listings == []
    monkeypatch.undo()
    assert snapshot() == before
    assert [json.loads(line)["type"] for line in (data / "logs" / "supervisor.jsonl").read_text().splitlines()] == ["storage_count"]


# ------------------------------------------------------------- contract


WEB = REPO / "web" / "modules"


def test_storage_envelopes_have_a_field_identical_browser_twin():
    """The gateway re-exports the domain-owned envelopes; storage_types.js is their twin."""
    from tests.test_gateway_parity import _js_typedef_fields

    text = (WEB / "storage_types.js").read_text(encoding="utf-8")
    from ouroboros import storage_free

    names = [name for name in contracts.__all__ if name.startswith("Storage")]
    assert names == ["StorageCount", "StorageCounting", "StorageError", "StoragePart", "StoragePlace", "StorageResponse",
                     "StorageRest", "StorageVolume", "StorageRemover", "StorageFree", "StorageFreeEntry", "StorageFreeGroup",
                     "StorageFreeItem", "StorageFreeItemsResponse", "StorageFreeKept", "StorageFreePool", "StorageFreeRequest",
                     "StorageFreeRun"]
    for name in names:
        home = storage_free if name.startswith("StorageFree") else storage
        assert getattr(contracts, name) is getattr(home, name)
        assert _js_typedef_fields(text, name) == set(getattr(contracts, name).__annotations__), name
    assert "storage_types.js" in (WEB / "api_types.js").read_text(encoding="utf-8").splitlines()[0]
    assert not re.search(r"@typedef \{Object\} Storage", (WEB / "api_types.js").read_text(encoding="utf-8"))
    client = (WEB / "api_client.js").read_text(encoding="utf-8")
    assert "import('./storage_types.js').StorageResponse" in client
    assert "/api/storage/measure/cancel" in client


def test_a_real_count_has_exactly_the_contract_fields(tmp_path):
    data = tmp_path / "data"
    _write(data / "observability" / "a.bin", 1)
    record = _count(data)
    required = {key for key, kind in storage.StorageCount.__annotations__.items() if "NotRequired" not in str(kind)}
    assert set(record) == required
    place_required = {key for key, kind in storage.StoragePlace.__annotations__.items() if "NotRequired" not in str(kind)}
    assert all(set(row) == place_required for row in record["places"])
    assert all(set(part) == set(storage.StoragePart.__annotations__) for row in record["places"] for part in row["parts"])
    assert all(set(row["rest"]) == set(storage.StorageRest.__annotations__) for row in record["places"])
    assert set(record["volume"]) == set(storage.StorageVolume.__annotations__)
    assert [row["id"] for row in record["places"]] == [place_id for place_id, _group, _spec in storage.PLACES]
    assert {row["group"] for row in record["places"]} <= set(storage.GROUPS)


def test_the_engine_home_name_is_the_owned_engine_directory():
    from ouroboros.claudexor_daemon import owned_config_dir

    assert owned_config_dir().name == storage.ENGINE_DIR
    assert next(spec for place_id, _group, spec in storage.PLACES if place_id == "engine") == ((storage.ENGINE_DIR,),)
