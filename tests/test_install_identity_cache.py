"""Ownership checks memoize stable paths while identity changes remain visible."""
import json
import os
import pathlib
import shutil
import time

import pytest

from ouroboros import install_identity as identity, store


@pytest.mark.parametrize("case,write", [("own_read", False), ("own_write", True),
                                       ("child_write", True), ("bare_write", True)])
def test_steady_ownership_check_cost(tmp_path, monkeypatch, case, write):
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    identity.initialize_install()
    target = {"child_write": root / "task_drives" / "task",
              "bare_write": tmp_path / "isolated"}.get(case, root)
    check = identity.check_store_root
    check(target, write=write)
    calls = 5000
    started = time.perf_counter()
    for _ in range(calls):
        check(target, write=write)
    microseconds = (time.perf_counter() - started) * 1_000_000 / calls
    print(json.dumps({"case": case, "calls": calls, "microseconds_per_call": microseconds}))
    # A warm check is at most one stat (pinned structurally below); Windows' stat alone costs tens of microseconds,
    # while the regressions this bounds (a JSON read, an ancestor walk) cost hundreds.
    assert microseconds < 150, f"{case}: {microseconds:.2f} us/check"


def test_warm_checks_only_stat_the_canonical_identity(tmp_path, monkeypatch):
    """The own canonical root needs no identity read at all; every other warm check stats it once."""
    foreign, own, bare = (tmp_path / name for name in ("foreign", "own", "bare"))
    for root in (foreign, own):
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
        identity.initialize_install()
    paths = (own, own / "task_drives" / "task", bare, foreign / "state/headless_tasks/task/data")

    def check(path, write):
        if path == paths[-1]:
            with pytest.raises(identity.StoreForeign):
                identity.check_store_root(path, write=write)
        else:
            identity.check_store_root(path, write=write)

    for path in paths:
        check(path, True)
    calls, original = [], os.stat

    def stat(path, *args, **kwargs):
        calls.append(os.fspath(path))
        return original(path, *args, **kwargs)

    def cold(*args, **kwargs):
        pytest.fail("a warm ownership check repeated initialization, JSON IO or path resolution")

    with monkeypatch.context() as patch:
        patch.setattr(identity.os, "stat", stat)
        patch.setattr(pathlib.Path, "read_text", cold)
        patch.setattr(pathlib.Path, "resolve", cold)
        patch.setattr(identity, "initialize_install", cold)
        for write in (False, True):
            for path in paths:
                check(path, write)
    assert calls == [str(own / identity.IDENTITY_PATH)] * 6  # three non-root paths, read and write
    assert not bare.exists()


def test_initialization_is_once_per_canonical_root_and_reads_create_nothing(tmp_path, monkeypatch):
    calls, original = [], identity.initialize_install

    def initialize():
        calls.append(identity.canonical_root())
        return original()

    monkeypatch.setattr(identity, "initialize_install", initialize)
    roots = [tmp_path / "A", tmp_path / "B"]
    for root in roots:
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
        identity.check_store_root(root, write=False)
        assert not root.exists()
        for _ in range(3):
            identity.check_store_root(root, write=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(roots[0]))
    identity.check_store_root(roots[0], write=True)
    assert calls == roots
    with pytest.raises(identity.StoreForeign):
        identity.check_store_root(roots[1], write=True)


@pytest.mark.parametrize("changed", ["inode", "mtime", "size"])
def test_identity_record_reloads_only_when_stat_witness_changes(tmp_path, monkeypatch, changed):
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    first = identity.initialize_install()
    path = root / identity.IDENTITY_PATH
    raw, before = path.read_text(), path.stat()
    assert identity._read(root) == first
    replacement = raw.replace(first["id"], "x" * len(first["id"]))
    if changed == "size":
        replacement += " "
    target = path.with_suffix(".new") if changed == "inode" else path
    target.write_text(replacement)
    modified = before.st_mtime_ns + 1_000_000_000 if changed == "mtime" else before.st_mtime_ns
    os.utime(target, ns=(before.st_atime_ns, modified))
    if changed == "inode":
        target.replace(path)
    calls, original = [], pathlib.Path.read_text

    def read(path, *args, **kwargs):
        calls.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", read)
    for _ in range(2):
        assert identity._read(root)["id"] == "x" * len(first["id"])
    assert calls == [path]
    path.write_text("broken")
    with pytest.raises(store.StoreError, match="identity unreadable"):
        identity.check_store_root(root / "task_drives" / "task", write=False)
    identity.check_store_root(root, write=False)  # the own canonical root stays usable


def test_move_mints_identity_and_invalidates_preboot_verdict(tmp_path, monkeypatch):
    old, moved, bare = (tmp_path / name for name in ("old", "moved", "bare"))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(old))
    first = identity.initialize_install()
    identity.check_store_root(old, write=False)
    old.rename(moved)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(moved))
    # Before boot, the vanished old path is bare; this creation-free read is cached.
    identity.check_store_root(old, write=False)
    identity.check_store_root(moved, write=False)
    second = identity.initialize_install()
    assert second["id"] != first["id"] and second["copied_from"] == [str(old)]
    for write in (False, True):
        with pytest.raises(identity.StoreForeign):
            identity.check_store_root(old, write=write)
        identity.check_store_root(moved, write=write)
        identity.check_store_root(bare, write=write)
    assert not old.exists() and not bare.exists()


def test_preboot_copy_cannot_reuse_original_roots_own_verdict(tmp_path, monkeypatch):
    original, copied = tmp_path / "original", tmp_path / "copied"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(original))
    record = identity.initialize_install()
    identity.check_store_root(original, write=False)
    shutil.copytree(original, copied)
    before = {str(p.relative_to(copied)): p.read_bytes() if p.is_file() else None
              for p in copied.rglob("*")}
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(copied))
    with pytest.raises(identity.StoreForeign):
        identity.check_store_root(original, write=False)
    identity.check_store_root(copied, write=False)
    assert json.loads((copied / identity.IDENTITY_PATH).read_text()) == record
    assert {str(p.relative_to(copied)): p.read_bytes() if p.is_file() else None
            for p in copied.rglob("*")} == before
