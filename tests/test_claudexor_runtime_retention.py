"""Old managed engine program copies are retired on the custody pass (owner Q6 = A).

Kept: the pinned install, every tree of the version the owned daemon proved serving, the Node trees
and the pinned archive. Removed: other version trees and archives, and an interrupted install's
staging once older than an hour. Nothing is removed while the serving version is unproven or while
an install holds the lock.
"""
from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest

from ouroboros import claudexor_runtime, claudexor_runtime_retention as retention

PINNED, SERVING = "4.2.0-aaaaaaaaaaaa", "4.1.1"


@pytest.fixture
def cx(tmp_path, monkeypatch):
    """A managed runtime root holding this install's history of engine updates."""
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager", lambda: SimpleNamespace(
        pin=SimpleNamespace(install_name=PINNED, archive_name="claudexor-4.2.0.tar.gz")))
    monkeypatch.setattr("ouroboros.claudexor_daemon.owned_engine_version", lambda: SERVING)
    root = tmp_path / "state" / "cx"
    for rel in (PINNED, "4.1.1-bbbbbbbbbbbb", "4.1.1-cccccccccccc", "3.9.0-dddddddddddd", "4.0.0-eeeeeeeeeeee",
                "node/24.1.0-darwin-arm64", "node/.tmp-11-abcd1234", ".tmp-93452-abcd1234", ".old-deadbeef",
                "unrecognized"):
        (root / rel / "lib").mkdir(parents=True)
        (root / rel / "lib" / "index.js").write_text("module.exports = 1\n", encoding="utf-8")
    (root / "cache").mkdir()
    for name in ("claudexor-4.2.0.tar.gz", "claudexor-3.9.0.tar.gz", ".claudexor-4.2.0.tar.gz.tmp.11.22.ab"):
        (root / "cache" / name).write_bytes(b"archive")
    outside = tmp_path / "outside-runtime"
    (outside / "lib").mkdir(parents=True)
    try:  # a link that names a version is never followed (Windows may refuse to create one)
        os.symlink(outside, root / "4.0.1-ffffffffffff", target_is_directory=True)
    except (OSError, NotImplementedError):
        pass
    return root


def _names(root):
    """Every entry of ``state/cx``, ``node/`` and ``cache/`` (the lock file aside)."""
    return sorted([path.name for path in root.iterdir() if path.name != "install.lock"]
                  + [f"{folder}/{path.name}" for folder in ("node", "cache") for path in (root / folder).iterdir()])


def test_unpinned_versions_go_running_and_pinned_stay_and_young_staging_waits(cx):
    report = retention.prune_unpinned_runtimes()
    assert sorted(report["removed"]) == ["3.9.0-dddddddddddd", "4.0.0-eeeeeeeeeeee", "claudexor-3.9.0.tar.gz"], report
    assert not report["errors"]
    kept = _names(cx)
    for name in (PINNED, "4.1.1-bbbbbbbbbbbb", "4.1.1-cccccccccccc", "node/24.1.0-darwin-arm64",
                 "cache/claudexor-4.2.0.tar.gz", "unrecognized"):
        assert name in kept, name
    for staging in (".tmp-93452-abcd1234", ".old-deadbeef", "node/.tmp-11-abcd1234",
                    "cache/.claudexor-4.2.0.tar.gz.tmp.11.22.ab"):
        assert staging in kept, f"{staging}: younger than an hour, an install may still own it"
    assert (cx.parent.parent / "outside-runtime" / "lib").is_dir(), "a link is never followed"

    later = retention.prune_unpinned_runtimes(now=time.time() + 7200)
    assert sorted(later["removed"]) == [".claudexor-4.2.0.tar.gz.tmp.11.22.ab", ".old-deadbeef",
                                        ".tmp-11-abcd1234", ".tmp-93452-abcd1234"], later
    assert (cx / PINNED / "lib" / "index.js").is_file() and (cx / "4.1.1-bbbbbbbbbbbb" / "lib").is_dir()


def test_a_newer_engine_kept_over_a_lower_pin_keeps_its_tree(cx, monkeypatch):
    """A planned downgrade keeps the newer engine serving (server_restart): both trees stay."""
    (cx / "4.3.1-ffffffffffff" / "lib").mkdir(parents=True)
    monkeypatch.setattr("ouroboros.claudexor_daemon.owned_engine_version", lambda: "4.3.1")
    report = retention.prune_unpinned_runtimes()
    assert "4.3.1-ffffffffffff" in _names(cx) and PINNED in _names(cx)
    assert sorted(report["removed"]) == ["3.9.0-dddddddddddd", "4.0.0-eeeeeeeeeeee", "4.1.1-bbbbbbbbbbbb",
                                         "4.1.1-cccccccccccc", "claudexor-3.9.0.tar.gz"], report


def test_an_unproven_serving_version_removes_nothing(cx, monkeypatch):
    before = _names(cx)
    monkeypatch.setattr("ouroboros.claudexor_daemon.owned_engine_version", lambda: "")
    report = retention.prune_unpinned_runtimes(now=time.time() + 7200)
    assert report["skipped"] == "serving_version_unproven" and not report["removed"]
    assert _names(cx) == before


def test_an_install_holding_the_lock_is_never_raced(cx):
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    before, lock = _names(cx), cx / "install.lock"
    fd = acquire_exclusive_file_lock(lock, timeout_sec=1)
    try:
        report = retention.prune_unpinned_runtimes()
    finally:
        release_exclusive_file_lock(lock, fd)
    assert report["skipped"] == "install_in_progress" and _names(cx) == before
    assert "3.9.0-dddddddddddd" in retention.prune_unpinned_runtimes()["removed"]
