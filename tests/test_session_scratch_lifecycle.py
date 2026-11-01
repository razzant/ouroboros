"""A writable session's disposable scratch, exercised whole through its real producer and retirers.

``workspace_preflight.session_service_directories`` makes it for an engine-granted ``workspace_write``
start and, owned but never granted, for a full-access one. It holds a real environment: a virtual
environment made inside it, with a package installed offline from a local wheel, read from a recipe in the
copy. The task finishes; its drive retires through Free's exact item and through the background prune under
the shared retention term; the existing trash removal releases the scratch, and a folder no producer
recorded (an environment made before this declaration) stays. A later session gets fresh scratch and
reinstalls only because it chooses to, from the surviving recipe and cache: nothing reinstalls by itself,
and a cache that no longer holds the package fails visibly. A scratch its producer retained, and a task the
owner kept, are never destroyed.
"""

from __future__ import annotations

import base64
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from ouroboros import headless, owner_keep, task_custody, test_environment
from tests.test_session_service_directories import GRANTED, _Catalog, _prepare, _synthetic_temp_parent  # noqa: F401
from tests.test_storage_free import OLD, _drives, _fake_sizer, _guard, _join, _owner, _preview

pytest.importorskip("ensurepip")  # the environment's own pip comes from it, offline
ACCESSES = ("workspace_write", "full")


def _wheel(cache: Path) -> Path:
    """A tiny pure-Python wheel, built here so the install needs no network."""
    files = {"tinypkg/__init__.py": b'VALUE = "installed"\n',
             "tinypkg-1.0.dist-info/METADATA": b"Metadata-Version: 2.1\nName: tinypkg\nVersion: 1.0\n",
             "tinypkg-1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"}
    record = "".join(f"{name},sha256={base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()},"
                     f"{len(data)}\n" for name, data in files.items()) + "tinypkg-1.0.dist-info/RECORD,,\n"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "tinypkg-1.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr("tinypkg-1.0.dist-info/RECORD", record)
    return path


def _python(scratch: Path) -> Path:
    return scratch / "env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _install(scratch: Path, recipe: Path, cache: Path) -> subprocess.CompletedProcess:
    """What a session chooses to run: an isolated environment made inside its scratch, then that
    environment's own package manager, offline. Never the interpreter running the tests, which a
    required virtualenv (``PIP_REQUIRE_VIRTUALENV``) rightly refuses."""
    subprocess.run([sys.executable, "-m", "venv", str(scratch / "env")], check=True, capture_output=True,
                   timeout=300)
    return subprocess.run([str(_python(scratch)), "-m", "pip", "install", "--no-index", "--find-links", str(cache),
                           "--no-cache-dir", "--disable-pip-version-check", "-r", str(recipe)],
                          capture_output=True, text=True, timeout=120)


def _imports(scratch: Path) -> str:
    if not _python(scratch).exists():
        return ""
    done = subprocess.run([str(_python(scratch)), "-c", "import tinypkg; print(tinypkg.VALUE)"],
                          capture_output=True, text=True, timeout=60)
    return done.stdout.strip() if done.returncode == 0 else ""


@pytest.fixture
def world(tmp_path, monkeypatch):
    data, copy, cache = tmp_path / "data", tmp_path / "copy", tmp_path / "downloads"
    copy.mkdir()
    (copy / "requirements.txt").write_text("tinypkg==1.0\n", encoding="utf-8")
    (copy / "edit.py").write_text("print('unique work')\n", encoding="utf-8")
    _wheel(cache)
    _fake_sizer(monkeypatch)
    monkeypatch.setattr("ouroboros.maintenance.child.request", lambda: None)
    return tmp_path, data, copy, cache


def _scratch(world, task_id, access="workspace_write"):
    tmp_path, _data, copy, _cache = world
    made = _prepare(tmp_path, _Catalog(GRANTED), access=access, task_id=task_id, key=f"k-{task_id}", execution=copy)
    assert made.scratch and Path(made.scratch).is_dir(), made  # never an empty path, the current folder
    assert bool(made.roots) is (access == "workspace_write")  # full access owns it, nothing is granted
    return Path(made.scratch)


@pytest.mark.parametrize("access", ACCESSES)
def test_install_finish_retire_both_ways_then_an_explicit_reinstall_from_the_surviving_recipe(world, access):
    tmp_path, data, copy, cache = world
    probes = _drives(data, [("manual", "direct", "completed", OLD), ("auto", "direct", "completed", OLD)])
    recipe_digest = hashlib.sha256((copy / "requirements.txt").read_bytes()).hexdigest()
    scratches = {}
    for task_id in ("manual", "auto"):
        scratches[task_id] = _scratch(world, task_id, access)
        assert _install(scratches[task_id], copy / "requirements.txt", cache).returncode == 0
        assert _imports(scratches[task_id]) == "installed"
    legacy = scratches["auto"].parent / "ob-review-legacy-venv"  # made before the declaration: no record names it
    legacy.mkdir()
    (legacy / "pyvenv.cfg").write_text("home = /legacy\n", encoding="utf-8")

    owner = _owner(probes)  # by hand: Free's exact item of the finished task's work folder
    _preview(owner, data, tmp_path)
    assert owner.start(data, tmp_path, owner.projection(data)["plan"], None, ["direct:manual"])["started"]
    _join(owner)
    report = headless.prune_task_drives(data, live=lambda _t: False, guard=_guard)  # by itself: the shared term
    assert "auto" in {row["task_id"] for row in report["pruned"]}
    assert task_custody.remove_custody_trash(data)["removed"] == 2
    assert not scratches["manual"].exists() and not scratches["auto"].exists()
    assert (copy / "edit.py").read_text(encoding="utf-8") == "print('unique work')\n", "the copy was never scratch"
    assert (legacy / "pyvenv.cfg").is_file(), "only a producer's recorded scratch is retired"

    fresh = _scratch(world, "next", access)  # a later session: nothing installed for it
    assert fresh not in scratches.values() and _imports(fresh) == ""
    assert hashlib.sha256((copy / "requirements.txt").read_bytes()).hexdigest() == recipe_digest
    assert _install(fresh, copy / "requirements.txt", cache).returncode == 0  # its own explicit choice
    assert _imports(fresh) == "installed"


def test_a_cache_that_lost_the_package_fails_the_reinstall_visibly(world):
    _tmp_path, data, copy, cache = world
    _drives(data, [("later", "direct", "completed", OLD)])
    for wheel in cache.iterdir():
        wheel.unlink()
    failed = _install(_scratch(world, "later"), copy / "requirements.txt", cache)
    assert failed.returncode != 0 and "tinypkg" in failed.stderr
    assert "No matching distribution" in failed.stderr or "Could not find a version" in failed.stderr


def test_a_scratch_its_producer_retained_is_never_destroyed(world):
    _tmp_path, data, _copy, _cache = world
    _drives(data, [("held", "direct", "completed", OLD)])
    scratch = _scratch(world, "held")
    (scratch / "only-here.txt").write_text("unique input\n", encoding="utf-8")
    test_environment.retain_tree(scratch, "the producer could not prove its process ended")
    assert task_custody.settle_child_drive(data, "held", data / "task_drives" / "held",
                                           live=lambda _t: False)["status"] == "removed"
    for _ in range(2):
        assert task_custody.remove_custody_trash(data)["stuck"]
        assert (scratch / "only-here.txt").read_text(encoding="utf-8") == "unique input\n"
    (scratch / test_environment.RETENTION_MARKER).unlink()  # the producer resolves its own retention
    assert task_custody.remove_custody_trash(data)["removed"] == 1 and not scratch.exists()


@pytest.mark.parametrize("access", ACCESSES)
def test_a_task_the_owner_kept_keeps_its_drive_and_scratch(world, access):
    _tmp_path, data, _copy, _cache = world
    probes = _drives(data, [("kept", "direct", "completed", OLD)])
    scratch = _scratch(world, "kept", access)
    owner_keep.set_owner_keep(data, "kept", keep=True, source={"basis": "test"})
    report = headless.prune_task_drives(data, live=lambda _t: False, guard=_guard)
    assert [row["reason"] for row in report["skipped"] if row["task_id"] == "kept"] == ["owner_keep"]
    task_custody.remove_custody_trash(data)
    assert scratch.is_dir() and (data / "task_drives" / "kept").is_dir()
    assert probes == {"kept": False}
