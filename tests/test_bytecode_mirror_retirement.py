"""Bytecode mirrors under ``state/pycache.noindex`` retire when their source directory is gone.

Under ``PYTHONPYCACHEPREFIX`` CPython writes ``<prefix>/<source directory>/<name>.<tag>.pyc`` (the
drive stripped on Windows), so the mirror names its source: the ``bytecode`` step of the
``runtime_retirement`` job removes a mirror whose source is gone everywhere it could be, keeps every
other one, and owes the rest of a run past its unlink budget to the next run.
"""
from __future__ import annotations

import ctypes
import importlib.util
import os
import pathlib
import sys
import threading
from types import SimpleNamespace

import pytest

from ouroboros import maintenance_jobs, maintenance_retirement as retirement


@pytest.fixture
def data(tmp_path, monkeypatch):
    root = tmp_path / "data"
    (root / "state" / "pycache.noindex").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    return root


def _compiled(data, source_file: pathlib.Path) -> pathlib.Path:
    """Where CPython itself puts SOURCE_FILE's bytecode under this data root's prefix."""
    saved = sys.pycache_prefix
    sys.pycache_prefix = str(data / "state" / "pycache.noindex")
    try:
        cached = pathlib.Path(importlib.util.cache_from_source(str(source_file)))
    finally:
        sys.pycache_prefix = saved
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"\x00" * 64)
    return cached


def _tree(base: pathlib.Path, *modules: str) -> list:
    """Source files under BASE and their mirrors' files, one per module path."""
    files = []
    for module in modules:
        path = base / module
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x = 1\n", encoding="utf-8")
        files.append(path)
    return files


def test_a_gone_source_loses_its_mirror_whole_and_a_live_one_stays(data, tmp_path):
    live = _tree(tmp_path / "live", "pkg/a.py", "pkg/sub/b.py")
    gone = _tree(tmp_path / "gone", "pkg/a.py", "pkg/sub/b.py")
    live_pyc = [_compiled(data, path) for path in live]
    gone_pyc = [_compiled(data, path) for path in gone]
    import shutil

    shutil.rmtree(tmp_path / "gone")
    report = retirement.prune_stale_bytecode(data)
    assert all(path.exists() for path in live_pyc), "a live source keeps every mirror file"
    assert not any(path.exists() for path in gone_pyc)
    mirror = gone_pyc[0].parents[1]  # the mirror of <tmp>/gone
    assert not mirror.exists() and mirror.parent.is_dir(), "the gone source's mirror went whole, its parent stays"
    assert report["removed"] == 1 and report["errors"] == 0 and "next_due" not in report
    again = retirement.prune_stale_bytecode(data)
    assert (again["removed"], again["unlinked"]) == (0, 0), "a second run removes nothing"
    assert all(path.exists() for path in live_pyc)


def test_a_vanished_subdirectory_under_a_live_tree_goes_alone(data, tmp_path):
    files = _tree(tmp_path / "src", "top.py", "keep/a.py", "drop/b.py", "drop/deeper/c.py")
    compiled = {path.relative_to(tmp_path / "src").as_posix(): _compiled(data, path) for path in files}
    import shutil

    shutil.rmtree(tmp_path / "src" / "drop")
    report = retirement.prune_stale_bytecode(data)
    assert compiled["top.py"].exists() and compiled["keep/a.py"].exists()
    assert not compiled["drop/b.py"].parent.exists(), "only the vanished subtree's mirror went"
    assert report["removed"] == 1


@pytest.mark.skipif(sys.platform == "win32" or getattr(os, "geteuid", lambda: 1)() == 0,
                    reason="POSIX search permission; root reads through it")
def test_an_unreadable_source_keeps_its_mirror(data, tmp_path):
    [source] = _tree(tmp_path / "locked" / "inner", "pkg/a.py")
    pyc = _compiled(data, source)
    (tmp_path / "locked").chmod(0o000)  # the source can no longer be asked: unknown, never gone
    try:
        report = retirement.prune_stale_bytecode(data)
    finally:
        (tmp_path / "locked").chmod(0o755)
    assert pyc.exists() and report["removed"] == 0 and report["unknown"] >= 1


def test_the_budget_leaves_the_rest_owed_and_the_next_run_finishes(data, tmp_path):
    gone = _tree(tmp_path / "gone", *(f"pkg{index}/m{n}.py" for index in range(3) for n in range(4)))
    pyc = [_compiled(data, path) for path in gone]
    import shutil

    shutil.rmtree(tmp_path / "gone")
    first = retirement.prune_stale_bytecode(data, budget=5)
    assert first["unlinked"] == 5 and first["next_due"] is not None, "the rest is owed through next_due"
    assert any(path.exists() for path in pyc)
    second = retirement.prune_stale_bytecode(data)
    assert not any(path.exists() for path in pyc) and "next_due" not in second
    assert not (data / "state" / "pycache.noindex" / str(tmp_path / "gone").lstrip(os.sep)).exists()



def test_a_legacy_tree_an_older_body_wrote_again_goes_whole_within_the_budget(data, tmp_path, monkeypatch):
    [source] = _tree(tmp_path / "live", "pkg/a.py")
    live = _compiled(data, source)
    legacy = data / "state" / "pycache" / "Applications" / "Ouroboros.app"
    legacy.mkdir(parents=True)
    for n in range(6):
        (legacy / f"m{n}.cpython-310.pyc").write_bytes(b"\x00")
    monkeypatch.setattr(sys, "pycache_prefix", str(data / "state" / "pycache.noindex"))
    first = retirement.prune_stale_bytecode(data, budget=4)
    assert first["unlinked"] == 4 and first["next_due"] is not None and legacy.exists(), "the budget holds"
    second = retirement.prune_stale_bytecode(data)
    assert not (data / "state" / "pycache").exists() and second["removed"] == 1, "the old pruner never looked there"
    assert live.exists(), "the current prefix keeps every live mirror"
    legacy.mkdir(parents=True)
    (legacy / "own.cpython-310.pyc").write_bytes(b"\x00")
    monkeypatch.setattr(sys, "pycache_prefix", str(data / "state" / "pycache"))  # an older launcher's process
    assert retirement.prune_stale_bytecode(data)["unlinked"] == 0 and (legacy / "own.cpython-310.pyc").exists()

def test_a_worker_writing_into_a_live_mirror_loses_nothing(data, tmp_path):
    [source] = _tree(tmp_path / "live", "pkg/a.py")
    mirror = _compiled(data, source).parent
    written, stop = [], threading.Event()

    def worker():
        index = 0
        while not stop.is_set():
            path = mirror / f"w{index}.cpython-310.pyc"
            path.write_bytes(b"\x00")
            written.append(path)
            index += 1

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        reports = [retirement.prune_stale_bytecode(data) for _ in range(20)]
    finally:
        stop.set()
        thread.join()
    assert written and all(path.exists() for path in written)
    assert all(report["removed"] == report["unlinked"] == report["errors"] == 0 for report in reports)


def test_userbase_and_links_are_never_touched(data, tmp_path):
    userbase = data / "state" / "python-userbase" / "lib" / "site.py"
    userbase.parent.mkdir(parents=True)
    userbase.write_text("real dependency\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.pyc").write_bytes(b"\x00")
    [source] = _tree(tmp_path / "gone", "pkg/a.py")
    pyc = _compiled(data, source)
    os.symlink(outside, pyc.parent / "link", target_is_directory=True)
    import shutil

    shutil.rmtree(tmp_path / "gone")
    report = retirement.prune_stale_bytecode(data)
    assert userbase.read_text(encoding="utf-8") == "real dependency\n"
    assert (outside / "keep.pyc").exists() and (pyc.parent / "link").is_symlink(), "a link is never followed or removed"
    assert not pyc.exists() and report["removed"] == 0 and report["errors"] == 0, "the folder holding the link stays"
    prefix = data / "state" / "pycache.noindex"
    prefix.rename(prefix.with_name("real-pycache"))
    os.symlink(prefix.with_name("real-pycache"), prefix, target_is_directory=True)
    assert retirement.prune_stale_bytecode(data)["unlinked"] == 0, "a linked prefix is not walked"


def test_windows_mirrors_are_stale_only_when_missing_on_every_drive(data, tmp_path):
    """CPython strips the drive, so ``<prefix>/proj/pkg`` is the mirror of ``X:\\proj\\pkg`` for any X."""
    drives = [tmp_path / "C", tmp_path / "D", tmp_path / "E"]
    for drive in drives:
        drive.mkdir()
    (drives[1] / "proj" / "pkg").mkdir(parents=True)
    mirror = data / "state" / "pycache.noindex" / "proj" / "pkg"
    mirror.mkdir(parents=True)
    (mirror / "a.cpython-310.pyc").write_bytes(b"\x00")
    assert retirement.prune_stale_bytecode(data, sources=drives)["removed"] == 0, "one drive has it: kept"
    import shutil

    shutil.rmtree(drives[1] / "proj")
    (drives[2] / "proj").mkdir()
    if sys.platform != "win32" and getattr(os, "geteuid", lambda: 1)() != 0:
        (drives[2] / "proj").chmod(0o000)  # one drive cannot answer for proj/pkg: unknown keeps it
        try:
            assert retirement.prune_stale_bytecode(data, sources=drives)["removed"] == 0 and mirror.exists()
        finally:
            (drives[2] / "proj").chmod(0o755)
    shutil.rmtree(drives[2] / "proj")
    assert retirement.prune_stale_bytecode(data, sources=drives)["removed"] == 1 and not mirror.exists()
    mirror.mkdir(parents=True)
    (mirror / "a.cpython-310.pyc").write_bytes(b"\x00")
    report = retirement.prune_stale_bytecode(data, sources=[])
    assert mirror.exists() and report["removed"] == 0, "no drive answering proves no absence"


def test_windows_drive_list_comes_from_the_system_and_each_root_is_asked_once(monkeypatch):
    asked = []
    monkeypatch.setattr(retirement, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(retirement, "_answers", lambda root: asked.append(root) or root != "Q:\\")
    monkeypatch.setattr(os, "listdrives", lambda: ["C:\\", "Q:\\"], raising=False)
    assert retirement._source_roots() == ["C:\\"] and asked == ["C:\\", "Q:\\"]
    monkeypatch.delattr(os, "listdrives", raising=False)
    asked.clear()
    kernel32 = SimpleNamespace(GetLogicalDrives=lambda: 0b101)  # A: and C: only, never a probe of B:..Z:
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=kernel32), raising=False)
    assert retirement._source_roots() == ["A:\\", "C:\\"] and asked == ["A:\\", "C:\\"]
    monkeypatch.setattr(retirement, "sys", SimpleNamespace(platform="linux"))
    assert retirement._source_roots() == [os.sep]


def test_a_drive_refusing_its_root_is_still_a_place_a_source_can_be(data, tmp_path, monkeypatch):
    """A refused drive root is there (the per-mirror access rule keeps what it may hold); only an absent
    one (no media, a lost share) is dropped, so another drive's absence never retires a live mirror."""
    drives = [tmp_path / "C", tmp_path / "D"]
    for drive in drives:
        drive.mkdir()
    (drives[1] / "project").mkdir()
    mirror = data / "state" / "pycache.noindex" / "project"
    mirror.mkdir()
    (mirror / "m.cpython-310.pyc").write_bytes(b"\x00")
    monkeypatch.setattr(retirement, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(os, "listdrives", lambda: [str(drive) for drive in drives], raising=False)
    real = os.stat

    def refused(path, *args, **kwargs):
        if pathlib.Path(path) == drives[1]:
            raise PermissionError(13, "access to the drive root refused")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", refused)
    assert retirement.prune_stale_bytecode(data)["removed"] == 0 and mirror.exists()
    assert retirement._answers(str(tmp_path / "Q")) is False, "an absent root is no place a source can be"


def test_the_step_runs_in_the_job_and_a_source_removal_makes_it_due(data, tmp_path, monkeypatch):
    from ouroboros import task_custody, worktree_removal

    retirement.record_startup(data, gaps=True)
    pyc = [_compiled(data, path) for path in _tree(tmp_path / "gone", *(f"pkg/m{n}.py" for n in range(4)))][0]
    import shutil

    shutil.rmtree(tmp_path / "gone")
    monkeypatch.setattr(retirement, "BYTECODE_UNLINKS_PER_RUN", 2)

    def row():
        return maintenance_jobs.job(data, retirement.JOB) or {}

    assert retirement.run(data, lambda: False)["state"] == "done"
    assert row()["next_due"] is not None and retirement.due(data, row(), {}, False), "the rest is owed: due again"
    monkeypatch.setattr(retirement, "BYTECODE_UNLINKS_PER_RUN", 10_000)
    retirement.run(data, lambda: False)
    assert not pyc.parent.exists() and row()["next_due"] is None
    assert not retirement.due(data, row(), {}, False), "nothing owed: at rest"
    worktree_removal.retired({"path": str(tmp_path / "copy")}, data)
    assert retirement.due(data, row(), {}, False), "a retired copy makes the step due at once"
    retirement.run(data, lambda: False)
    assert not retirement.due(data, row(), {}, False)
    trash = data / "state" / "custody_trash" / "drive-1"
    trash.mkdir(parents=True)
    assert task_custody.remove_custody_trash(data)["removed"] == 1
    assert retirement.due(data, row(), {}, False), "a removed drive makes the step due at once"
