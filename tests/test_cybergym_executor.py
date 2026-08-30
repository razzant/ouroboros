"""Dependency-injected CyberGym executor tests.

No Docker daemon, upstream package, or provider credential is used here.  The
tests exercise the immutable task body and the exact command/HTTP boundaries;
the live smoke remains an operator action documented by the benchmark.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import threading

import pytest

from devtools.benchmarks.cybergym import cybergym_executor as executor_module
from devtools.benchmarks.cybergym.cybergym_adapter import final_poc_record
from devtools.benchmarks.cybergym.cybergym_executor import (
    CommandResult,
    CyberGymExecutor,
    ExecutorConfig,
    ExecutorFailure,
    _bind_container_image,
    _install_workspace_backend_alias,
    _reuse_directory_observation,
    _safe_extract,
)
from devtools.benchmarks.cybergym.cybergym_sidecar import required_resource_labels
from ouroboros.headless import write_workspace_patch_artifacts
from ouroboros.tools.registry import ToolContext, ToolRegistry

# ``_safe_extract`` refuses wholesale — by design, before validating or
# touching anything — on platforms without descriptor-safe publish/cleanup
# primitives (Windows: ``os.supports_dir_fd`` is empty).  Tests that need the
# extraction/validation pass to run are guarded on the exact refusal
# predicate, mirroring the in-file capability skips below.
_DESCRIPTOR_SAFE_EXTRACT = bool(
    executor_module._ARCHIVE_RENAME_DIR_FD and executor_module._ARCHIVE_CLEANUP_DIR_FD
)

# The CyberGym sidecar mount contract requires POSIX host paths: the rootless
# Docker daemon is a unix socket and the server/workspace host directories are
# bind-mounted at the *identical* absolute path inside Linux containers
# (see ``ExecutorConfig`` docstring and ``SidecarCommandSpec.__post_init__``).
# A ``C:\`` drive path cannot satisfy that contract, so these flows are
# Linux/POSIX-exclusive rather than portable.
_requires_posix_mount_paths = pytest.mark.skipif(
    sys.platform == "win32",
    reason="CyberGym mounts POSIX host paths at identical absolute container paths (rootless Linux Docker contract)",
)


def _config(tmp_path: pathlib.Path, **overrides):
    source = tmp_path / "source"
    data = tmp_path / "data"
    run = tmp_path / "run"
    server = tmp_path / "server"
    for path in (source, data, run, server):
        path.mkdir(exist_ok=True)
    mask = server / "mask_map.json"
    mask.write_text("{}", encoding="utf-8")
    values = dict(
        campaign_id="test-campaign",
        source_root=source,
        data_root=data,
        mask_map=mask,
        run_root=run,
        server_root=server,
        server_image="cybergym/server:pin",
        server_image_digest="sha256:" + "1" * 64,
        workspace_image="ouroboros/workspace:pin",
        workspace_image_digest="sha256:" + "2" * 64,
        ouroboros_url="http://127.0.0.1:8765",
        docker_host="unix:///run/user/1006/docker.sock",
        provider_probe=False,
    )
    values.update(overrides)
    return ExecutorConfig(**values)


def _write_archive(path: pathlib.Path, entries: list[tuple[str, str, str]]) -> None:
    """Build a tiny tarball with explicit member types for extraction tests."""
    with tarfile.open(path, "w:gz") as archive:
        for name, kind, value in entries:
            member = tarfile.TarInfo(name)
            if kind == "dir":
                member.type = tarfile.DIRTYPE
                member.mode = 0o755
                archive.addfile(member)
            elif kind == "file":
                payload = value.encode("utf-8")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            elif kind == "symlink":
                member.type = tarfile.SYMTYPE
                member.linkname = value
                archive.addfile(member)
            elif kind == "hardlink":
                member.type = tarfile.LNKTYPE
                member.linkname = value
                archive.addfile(member)
            elif kind == "fifo":
                member.type = tarfile.FIFOTYPE
                archive.addfile(member)
            else:  # pragma: no cover - test helper misuse
                raise AssertionError(kind)


def test_safe_extract_preserves_confined_relative_symlinks(tmp_path):
    if not _DESCRIPTOR_SAFE_EXTRACT:
        pytest.skip("platform has no descriptor-safe archive primitives")
    archive = tmp_path / "repo-vul.tar.gz"
    _write_archive(
        archive,
        [
            ("src-vul", "dir", ""),
            ("src-vul/lib.la", "file", "library"),
            ("src-vul/.libs", "dir", ""),
            ("src-vul/.libs/lib.la", "symlink", "../lib.la"),
            ("src-vul/README.md", "file", "readme"),
            ("src-vul/README", "symlink", "README.md"),
        ],
    )
    destination = tmp_path / "workspace"
    _safe_extract(archive, destination)
    assert (destination / "src-vul/.libs/lib.la").is_symlink()
    assert (destination / "src-vul/.libs/lib.la").read_text(encoding="utf-8") == "library"
    assert (destination / "src-vul/README").read_text(encoding="utf-8") == "readme"
    assert all(
        destination.resolve() in path.resolve().parents
        for path in destination.rglob("*")
        if path.is_symlink()
    )


def test_safe_extract_resolves_symlink_components(tmp_path):
    if not _DESCRIPTOR_SAFE_EXTRACT:
        pytest.skip("platform has no descriptor-safe archive primitives")
    archive = tmp_path / "component-links.tar.gz"
    _write_archive(
        archive,
        [
            ("root", "dir", ""),
            ("root/real", "dir", ""),
            ("root/real/file", "file", "payload"),
            ("root/dirlink", "symlink", "real"),
            ("root/filelink", "symlink", "dirlink/file"),
        ],
    )
    destination = tmp_path / "workspace"
    _safe_extract(archive, destination)
    assert (destination / "root/filelink").is_symlink()
    assert (destination / "root/filelink").read_text(encoding="utf-8") == "payload"


def test_safe_extract_rolls_back_staging_on_extraction_error(tmp_path, monkeypatch):
    if not _DESCRIPTOR_SAFE_EXTRACT:
        pytest.skip("platform has no descriptor-safe archive primitives")
    archive = tmp_path / "partial.tar.gz"
    _write_archive(archive, [("root", "dir", ""), ("root/file", "file", "payload")])
    destination = tmp_path / "workspace"
    destination.mkdir()
    sentinel = destination / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    original_extract = tarfile.TarFile.extract

    def fail_after_extract(self, *args, **kwargs):
        original_extract(self, *args, **kwargs)
        raise OSError("injected extraction failure")

    monkeypatch.setattr(tarfile.TarFile, "extract", fail_after_extract)
    with pytest.raises(ExecutorFailure, match="extraction failed"):
        _safe_extract(archive, destination)
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (destination / "root").exists()
    assert not list(tmp_path.glob(".workspace.extract-*"))


def test_safe_extract_publish_dirfd_survives_destination_replacement(tmp_path, monkeypatch):
    if os.rename not in os.supports_dir_fd:
        pytest.skip("platform has no dirfd-safe rename")
    archive = tmp_path / "race.tar.gz"
    _write_archive(archive, [("root", "dir", ""), ("root/file", "file", "payload")])
    destination = tmp_path / "workspace"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_sentinel = outside / "sentinel"
    outside_sentinel.write_text("untouched", encoding="utf-8")
    backup = tmp_path / "workspace-before-race"
    original_rename = os.rename
    replaced = False

    def replace_destination_once(src, dst, *args, **kwargs):
        nonlocal replaced
        if not replaced and kwargs.get("dst_dir_fd") is not None:
            original_rename(destination, backup)
            destination.symlink_to(outside, target_is_directory=True)
            replaced = True
        return original_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "rename", replace_destination_once)
    _safe_extract(archive, destination)
    assert replaced
    assert outside_sentinel.read_text(encoding="utf-8") == "untouched"
    assert (backup / "root/file").read_text(encoding="utf-8") == "payload"


def test_safe_extract_rollback_dirfd_does_not_follow_replaced_destination(
    tmp_path, monkeypatch
):
    if os.rename not in os.supports_dir_fd:
        pytest.skip("platform has no dirfd-safe rename")
    archive = tmp_path / "rollback-race.tar.gz"
    _write_archive(
        archive,
        [
            ("a-root", "dir", ""),
            ("a-root/file", "file", "first"),
            ("b-root", "dir", ""),
            ("b-root/file", "file", "second"),
        ],
    )
    destination = tmp_path / "workspace"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_entry = outside / "a-root"
    outside_entry.mkdir()
    outside_sentinel = outside_entry / "sentinel"
    outside_sentinel.write_text("must-survive", encoding="utf-8")
    backup = tmp_path / "workspace-before-race"
    original_rename = os.rename
    publishes = 0

    def replace_then_fail(src, dst, *args, **kwargs):
        nonlocal publishes
        if kwargs.get("dst_dir_fd") is None:
            return original_rename(src, dst, *args, **kwargs)
        publishes += 1
        if publishes == 1:
            result = original_rename(src, dst, *args, **kwargs)
            original_rename(destination, backup)
            destination.symlink_to(outside, target_is_directory=True)
            return result
        raise OSError("injected publish failure")

    monkeypatch.setattr(os, "rename", replace_then_fail)
    with pytest.raises(ExecutorFailure, match="extraction failed"):
        _safe_extract(archive, destination)

    assert outside_sentinel.read_text(encoding="utf-8") == "must-survive"
    assert not (outside / "a-root/file").exists()
    assert not (backup / "a-root").exists()
    assert destination.is_symlink()


def test_safe_extract_rejects_path_only_publish_abi(
    tmp_path, monkeypatch
):
    """The non-dirfd publish ABI is refused before any staging write."""
    if not executor_module._ARCHIVE_CLEANUP_DIR_FD:  # noqa: SLF001 - capability seam
        pytest.skip("platform has no descriptor-safe cleanup primitives")
    archive = tmp_path / "path-rollback-race.tar.gz"
    _write_archive(
        archive,
        [
            ("a-root", "dir", ""),
            ("a-root/file", "file", "first"),
            ("b-root", "dir", ""),
            ("b-root/file", "file", "second"),
        ],
    )
    destination = tmp_path / "workspace"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_entry = outside / "a-root"
    outside_entry.mkdir()
    outside_sentinel = outside_entry / "sentinel"
    outside_sentinel.write_text("must-survive", encoding="utf-8")
    backup = tmp_path / "workspace-before-race"
    original_replace = os.replace
    publishes = 0

    def replace_then_fail(src, dst):
        nonlocal publishes
        if pathlib.Path(src).parent.name.startswith(".workspace.extract-"):
            publishes += 1
            if publishes == 1:
                result = original_replace(src, dst)
                original_replace(destination, backup)
                destination.symlink_to(outside, target_is_directory=True)
                return result
            raise OSError("injected publish failure")
        return original_replace(src, dst)

    monkeypatch.setattr(executor_module, "_ARCHIVE_RENAME_DIR_FD", False)
    monkeypatch.setattr(os, "replace", replace_then_fail)
    with pytest.raises(ExecutorFailure, match="descriptor-safe publish and cleanup"):
        _safe_extract(archive, destination)

    assert outside_sentinel.read_text(encoding="utf-8") == "must-survive"
    assert not (outside / "a-root/file").exists()
    assert not backup.exists()
    assert destination.is_dir()


def test_safe_extract_rejects_destination_replacement_before_open(tmp_path, monkeypatch):
    if not (executor_module._ARCHIVE_RENAME_DIR_FD and executor_module._ARCHIVE_CLEANUP_DIR_FD):
        pytest.skip("platform has no descriptor-safe archive primitives")
    archive = tmp_path / "pre-open.tar.gz"
    _write_archive(archive, [("root", "dir", ""), ("root/file", "file", "payload")])
    destination = tmp_path / "workspace"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    backup = tmp_path / "workspace-before-open"
    original_open = os.open
    replaced = False

    def replace_before_destination_open(path, flags, *args, **kwargs):
        nonlocal replaced
        if not replaced and kwargs.get("dir_fd") is not None and str(path) == destination.name:
            os.rename(destination, backup)
            destination.symlink_to(outside, target_is_directory=True)
            replaced = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_before_destination_open)
    with pytest.raises(ExecutorFailure):
        _safe_extract(archive, destination)
    assert replaced
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (outside / "root").exists()
    assert not (backup / "root").exists()


def test_safe_extract_rejects_parent_replacement_before_open(tmp_path, monkeypatch):
    if not (executor_module._ARCHIVE_RENAME_DIR_FD and executor_module._ARCHIVE_CLEANUP_DIR_FD):
        pytest.skip("platform has no descriptor-safe archive primitives")
    archive = tmp_path / "parent-open.tar.gz"
    _write_archive(archive, [("root", "dir", ""), ("root/file", "file", "payload")])
    parent = tmp_path / "parent"
    parent.mkdir()
    destination = parent / "workspace"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    backup = tmp_path / "parent-before-open"
    original_open = os.open
    replaced = False

    def replace_before_parent_open(path, flags, *args, **kwargs):
        nonlocal replaced
        if not replaced and kwargs.get("dir_fd") is None and pathlib.Path(path) == parent:
            os.rename(parent, backup)
            parent.symlink_to(outside, target_is_directory=True)
            replaced = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_before_parent_open)
    with pytest.raises(ExecutorFailure):
        _safe_extract(archive, destination)
    assert replaced
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (outside / "workspace").exists()
    assert not (backup / "workspace/root").exists()


def test_safe_extract_anchors_staging_open_to_admitted_parent(tmp_path, monkeypatch):
    if not (executor_module._ARCHIVE_RENAME_DIR_FD and executor_module._ARCHIVE_CLEANUP_DIR_FD):
        pytest.skip("platform has no descriptor-safe archive primitives")
    archive = tmp_path / "staging-race.tar.gz"
    _write_archive(archive, [("root", "dir", ""), ("root/file", "file", "trusted")])
    parent = tmp_path / "parent"
    parent.mkdir()
    destination = parent / "workspace"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    backup = tmp_path / "parent-before-staging-race"
    original_open = os.open
    original_fstat = os.fstat
    parent_fd = None
    replaced = False

    def track_parent_open(path, flags, *args, **kwargs):
        nonlocal parent_fd
        fd = original_open(path, flags, *args, **kwargs)
        if kwargs.get("dir_fd") is None and pathlib.Path(path) == parent:
            parent_fd = fd
        return fd

    def replace_after_parent_admission(fd):
        nonlocal replaced
        info = original_fstat(fd)
        if fd == parent_fd and not replaced:
            staging = next(parent.glob(".workspace.extract-*"))
            attacker = outside / staging.name
            (attacker / "root").mkdir(parents=True)
            (attacker / "root/file").write_text("attacker", encoding="utf-8")
            os.rename(parent, backup)
            parent.symlink_to(outside, target_is_directory=True)
            replaced = True
        return info

    monkeypatch.setattr(os, "open", track_parent_open)
    monkeypatch.setattr(os, "fstat", replace_after_parent_admission)
    _safe_extract(archive, destination)

    assert replaced
    assert (backup / "workspace/root/file").read_text(encoding="utf-8") == "trusted"
    attacker_staging = next(outside.glob(".workspace.extract-*"))
    assert (attacker_staging / "root/file").read_text(encoding="utf-8") == "attacker"


def test_safe_extract_rejects_replaced_staging_inode(tmp_path, monkeypatch):
    if not (executor_module._ARCHIVE_RENAME_DIR_FD and executor_module._ARCHIVE_CLEANUP_DIR_FD):
        pytest.skip("platform has no descriptor-safe archive primitives")
    archive = tmp_path / "staging-inode-race.tar.gz"
    _write_archive(archive, [("root", "dir", ""), ("root/file", "file", "trusted")])
    parent = tmp_path / "parent"
    parent.mkdir()
    destination = parent / "workspace"
    destination.mkdir()
    original_open = os.open
    original_fstat = os.fstat
    parent_fd = None
    swapped = False

    def track_parent_open(path, flags, *args, **kwargs):
        nonlocal parent_fd
        fd = original_open(path, flags, *args, **kwargs)
        if kwargs.get("dir_fd") is None and pathlib.Path(path) == parent:
            parent_fd = fd
        return fd

    def replace_staging_after_parent_admission(fd):
        nonlocal swapped
        info = original_fstat(fd)
        if fd == parent_fd and not swapped:
            swapped = True
            staging = next(parent.glob(".workspace.extract-*"))
            saved = parent / (staging.name + ".saved")
            os.rename(staging, saved)
            replacement = parent / staging.name
            (replacement / "root").mkdir(parents=True)
            (replacement / "root/file").write_text("attacker", encoding="utf-8")
        return info

    monkeypatch.setattr(os, "open", track_parent_open)
    monkeypatch.setattr(os, "fstat", replace_staging_after_parent_admission)
    with pytest.raises(ExecutorFailure, match="staging (?:changed|directory was replaced)"):
        _safe_extract(archive, destination)

    assert swapped
    assert not (destination / "root").exists()
    replacement = next(
        path for path in parent.glob(".workspace.extract-*")
        if not path.name.endswith(".saved")
    )
    assert (replacement / "root/file").read_text(encoding="utf-8") == "attacker"


def test_safe_extract_closes_publish_fds_if_staging_cleanup_raises(tmp_path, monkeypatch):
    if not (executor_module._ARCHIVE_RENAME_DIR_FD and executor_module._ARCHIVE_CLEANUP_DIR_FD):
        pytest.skip("platform has no descriptor-safe archive primitives")
    archive = tmp_path / "cleanup-fd.tar.gz"
    _write_archive(archive, [("root", "dir", ""), ("root/file", "file", "payload")])
    destination = tmp_path / "workspace"
    destination.mkdir()
    original_open = os.open
    original_close = os.close
    original_remove = executor_module._remove_archive_entry_at
    parent_fd = None
    destination_fd = None
    closed: list[int] = []

    def capture_open(path, flags, *args, **kwargs):
        nonlocal parent_fd, destination_fd
        fd = original_open(path, flags, *args, **kwargs)
        if kwargs.get("dir_fd") is None and pathlib.Path(path) == destination.parent:
            parent_fd = fd
        elif kwargs.get("dir_fd") == parent_fd and str(path) == destination.name:
            destination_fd = fd
        return fd

    def capture_close(fd):
        closed.append(fd)
        return original_close(fd)

    def fail_staging_cleanup(dir_fd, name, expected_identity=None):
        if str(name).startswith(".workspace.extract-"):
            raise OSError("injected staging cleanup failure")
        return original_remove(dir_fd, name, expected_identity=expected_identity)

    monkeypatch.setattr(os, "open", capture_open)
    monkeypatch.setattr(os, "close", capture_close)
    monkeypatch.setattr(executor_module, "_remove_archive_entry_at", fail_staging_cleanup)
    with pytest.raises(ExecutorFailure, match="staging cleanup failed"):
        _safe_extract(archive, destination)

    assert parent_fd is not None and destination_fd is not None
    assert parent_fd in closed
    assert destination_fd in closed
    with pytest.raises(OSError):
        os.fstat(parent_fd)
    with pytest.raises(OSError):
        os.fstat(destination_fd)


def test_safe_extract_requires_descriptor_cleanup_capability(tmp_path, monkeypatch):
    archive = tmp_path / "no-cleanup.tar.gz"
    _write_archive(archive, [("root", "dir", ""), ("root/file", "file", "payload")])
    destination = tmp_path / "workspace"
    monkeypatch.setattr(executor_module, "_ARCHIVE_CLEANUP_DIR_FD", False)
    with pytest.raises(ExecutorFailure, match="descriptor-safe publish and cleanup"):
        _safe_extract(archive, destination)
    assert not list(tmp_path.glob(".workspace.extract-*"))


@pytest.mark.parametrize(
    ("linkname", "message"),
    [
        ("/tmp/cybergym-outside", "must be relative"),
        ("../../cybergym-outside", "escapes its workspace"),
        ("missing", "broken symlink"),
    ],
)
def test_safe_extract_rejects_absolute_escaping_and_broken_links(tmp_path, linkname, message):
    if not _DESCRIPTOR_SAFE_EXTRACT:
        pytest.skip("platform has no descriptor-safe archive primitives")
    archive = tmp_path / "bad.tar.gz"
    _write_archive(archive, [("root", "dir", ""), ("root/link", "symlink", linkname)])
    destination = tmp_path / "workspace"
    outside = tmp_path / "cybergym-outside"
    outside.write_text("untouched", encoding="utf-8")
    with pytest.raises(ExecutorFailure, match=message):
        _safe_extract(archive, destination)
    assert outside.read_text(encoding="utf-8") == "untouched"
    assert not (destination / "root/link").exists()


@pytest.mark.parametrize("kind", ["hardlink", "fifo"])
def test_safe_extract_rejects_special_link_members(tmp_path, kind):
    if not _DESCRIPTOR_SAFE_EXTRACT:
        pytest.skip("platform has no descriptor-safe archive primitives")
    archive = tmp_path / "special.tar.gz"
    _write_archive(archive, [("root", "dir", ""), ("root/member", kind, "root/target")])
    with pytest.raises(ExecutorFailure, match="special member"):
        _safe_extract(archive, tmp_path / "workspace")


def test_safe_extract_rejects_link_parent_conflict_and_destination_symlink(tmp_path):
    if not _DESCRIPTOR_SAFE_EXTRACT:
        pytest.skip("platform has no descriptor-safe archive primitives")
    archive = tmp_path / "conflict.tar.gz"
    _write_archive(
        archive,
        [
            ("root", "dir", ""),
            ("root/link", "symlink", "."),
            ("root/link/payload", "file", "must not write"),
        ],
    )
    with pytest.raises(ExecutorFailure, match="parent is not a directory"):
        _safe_extract(archive, tmp_path / "workspace")

    outside = tmp_path / "outside"
    outside.mkdir()
    redirected = tmp_path / "redirected"
    redirected.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ExecutorFailure, match="must not traverse a symlink"):
        _safe_extract(archive, redirected / "nested")


def test_executor_rejects_non_rootless_or_missing_digest(tmp_path):
    with pytest.raises(ExecutorFailure):
        _config(tmp_path, docker_host="unix:///var/run/docker.sock")
    with pytest.raises(ExecutorFailure):
        _config(tmp_path, workspace_image_digest="latest")


def test_container_image_binding_rejects_cached_digest_for_wrong_container():
    digest = "sha256:" + "1" * 64
    with pytest.raises(ExecutorFailure, match="identity does not match"):
        _bind_container_image(
            {
                "Image": "sha256:" + "2" * 64,
                "Config": {"Image": "cyber/server@" + digest},
            },
            {"Id": "sha256:" + "3" * 64, "RepoDigests": ["cyber/server@" + digest]},
            digest,
            "server",
        )


def test_generated_workspace_git_anchor_tracks_controls_without_source_blobs(tmp_path):
    config_root = tmp_path / "config"
    config_root.mkdir()
    host = CyberGymExecutor(_config(config_root)).host

    def generated(name: str) -> pathlib.Path:
        workspace = tmp_path / name
        workspace.mkdir()
        (workspace / "README.md").write_text("task readme\n", encoding="utf-8")
        (workspace / "description.txt").write_text("find the bug\n", encoding="utf-8")
        (workspace / "submit.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (workspace / "repo-vul.tar.gz").write_bytes(b"archive-input" * 10_000)
        source = workspace / "src-vul"
        source.mkdir()
        (source / "large.c").write_bytes(b"int vulnerable;\n" * 10_000)
        (workspace / "submissions").mkdir()
        return workspace

    first = generated("first")
    second = generated("second")
    first_anchor = executor_module._initialize_generated_workspace_git(
        first, runner=executor_module.run_command, host=host
    )
    second_anchor = executor_module._initialize_generated_workspace_git(
        second, runner=executor_module.run_command, host=host
    )
    assert first_anchor == second_anchor
    assert len(first_anchor) == 40

    tracked = subprocess.run(
        ["git", "-C", str(first), "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert tracked == ["README.md", "description.txt", "submit.sh"]
    assert subprocess.run(
        ["git", "-C", str(first), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout == ""

    (first / "src-vul" / "large.c").write_text("changed benchmark input\n", encoding="utf-8")
    (first / "final.poc").write_text("poc\n", encoding="utf-8")
    status = subprocess.run(
        ["git", "-C", str(first), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert status == ["?? final.poc"]


def test_workspace_backend_alias_is_confined_and_git_ignored(tmp_path):
    workspace = tmp_path / "generated"
    workspace.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", str(workspace)],
        check=True,
        capture_output=True,
        text=True,
    )

    alias = _install_workspace_backend_alias(workspace)
    assert alias == workspace / "workspace"
    assert alias.is_symlink()
    assert os.readlink(alias) == "."
    assert alias.resolve(strict=False) == workspace.resolve(strict=False)
    assert "/workspace" in (
        workspace / ".git" / "info" / "exclude"
    ).read_text(encoding="utf-8").splitlines()

    status = subprocess.run(
        ["git", "-C", str(workspace), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""

    marker = workspace / "final.poc"
    marker.write_bytes(b"adapter-alias-poc")
    record = final_poc_record(workspace)
    assert record.path == str(marker.resolve(strict=False))
    assert record.sha256 == hashlib.sha256(marker.read_bytes()).hexdigest()
    assert (alias / "final.poc").samefile(marker)
    _, patch_manifest = write_workspace_patch_artifacts(
        workspace, tmp_path / "artifacts", task={}
    )
    assert patch_manifest["status"] == "ready_with_changes"
    assert "workspace" not in patch_manifest["untracked_included"]
    assert "final.poc" in patch_manifest["untracked_included"]

    collision = tmp_path / "collision"
    collision.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", str(collision)],
        check=True,
        capture_output=True,
        text=True,
    )
    reserved = collision / "workspace"
    reserved.mkdir()
    with pytest.raises(ExecutorFailure, match="reserved backend alias"):
        _install_workspace_backend_alias(collision)
    assert reserved.is_dir() and not reserved.is_symlink()

    temp_collision = tmp_path / "temp-collision"
    temp_collision.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", str(temp_collision)],
        check=True,
        capture_output=True,
        text=True,
    )
    sentinel = temp_collision / ".git" / "info" / f".exclude.tmp.{os.getpid()}"
    sentinel.write_text("foreign temporary content\n", encoding="utf-8")
    _install_workspace_backend_alias(temp_collision)
    assert sentinel.read_text(encoding="utf-8") == "foreign temporary content\n"


@pytest.mark.parametrize(
    "target_kind",
    ["external", "dangling"],
    ids=["external_symlink", "dangling_symlink"],
)
def test_workspace_backend_alias_rejects_symlinked_git_info(tmp_path, target_kind):
    workspace = tmp_path / target_kind
    workspace.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", str(workspace)],
        check=True,
        capture_output=True,
        text=True,
    )
    info = workspace / ".git" / "info"
    (info / "exclude").unlink()
    info.rmdir()
    if target_kind == "external":
        target = tmp_path / "external-info"
        target.mkdir()
    else:
        target = tmp_path / "missing-info"
    os.symlink(target, info, target_is_directory=True)

    with pytest.raises(ExecutorFailure, match="local git info directory"):
        _install_workspace_backend_alias(workspace)
    assert not os.path.lexists(workspace / "workspace")


def test_workspace_backend_alias_create_race_never_unlinks_replacement(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "create-race"
    workspace.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", str(workspace)],
        check=True,
        capture_output=True,
        text=True,
    )
    exclude = workspace / ".git" / "info" / "exclude"
    alias = workspace / "workspace"
    original_replace = executor_module.os.replace
    original_symlink = executor_module.os.symlink
    original_unlink = pathlib.Path.unlink
    unlink_calls: list[pathlib.Path] = []

    def replace_then_publish(src, dst, *args, **kwargs):
        result = original_replace(src, dst, *args, **kwargs)
        if pathlib.Path(dst) == exclude:
            # Simulate a child winning the alias name after all metadata checks
            # but before the final symlink syscall.
            original_symlink("./", alias, target_is_directory=True)
        return result

    def track_unlink(path, *args, **kwargs):
        if path == alias:
            unlink_calls.append(path)
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(executor_module.os, "replace", replace_then_publish)
    monkeypatch.setattr(pathlib.Path, "unlink", track_unlink)
    with pytest.raises(ExecutorFailure, match="unable to install workspace backend alias"):
        _install_workspace_backend_alias(workspace)
    assert alias.is_symlink() and os.readlink(alias) == "./"
    assert unlink_calls == []
    monkeypatch.undo()
    alias.unlink()


def test_workspace_backend_alias_temp_replacement_is_not_unlinked(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "temp-replacement"
    workspace.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", str(workspace)],
        check=True,
        capture_output=True,
        text=True,
    )
    exclude = workspace / ".git" / "info" / "exclude"
    alias = workspace / "workspace"
    original_replace = executor_module.os.replace
    foreign_temp: dict[str, pathlib.Path] = {}

    def replace_then_swap_temp(src, dst, *args, **kwargs):
        if pathlib.Path(dst) == exclude:
            temp_path = pathlib.Path(src)
            os.unlink(temp_path)
            temp_path.write_text("foreign temporary content\n", encoding="utf-8")
            foreign_temp["path"] = temp_path
            raise OSError("injected metadata replace failure")
        return original_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(executor_module.os, "replace", replace_then_swap_temp)
    with pytest.raises(ExecutorFailure, match="unable to install workspace backend alias"):
        _install_workspace_backend_alias(workspace)
    temp_path = foreign_temp["path"]
    assert temp_path.read_text(encoding="utf-8") == "foreign temporary content\n"
    assert not os.path.lexists(alias)
    assert "/workspace" not in exclude.read_text(encoding="utf-8").splitlines()


def test_workspace_backend_alias_keeps_immediate_post_create_replacement(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "post-create-replacement"
    workspace.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", str(workspace)],
        check=True,
        capture_output=True,
        text=True,
    )
    alias = workspace / "workspace"
    original_symlink = executor_module.os.symlink
    original_unlink = pathlib.Path.unlink

    def publish_then_replace(target, link, *args, **kwargs):
        original_symlink(target, link, *args, **kwargs)
        if pathlib.Path(link) == alias:
            original_unlink(alias)
            # Replace the just-created alias before the helper returns.  Since
            # no post-create check or rollback exists, the replacement remains
            # untouched and the helper does not race with it.
            original_symlink("./", alias, target_is_directory=True)

    monkeypatch.setattr(executor_module.os, "symlink", publish_then_replace)
    returned = _install_workspace_backend_alias(workspace)
    assert returned == alias
    assert alias.is_symlink() and os.readlink(alias) == "./"
    monkeypatch.undo()
    alias.unlink()


def test_workspace_backend_alias_bridges_structured_absolute_path(tmp_path):
    system = tmp_path / "system"
    data = tmp_path / "data"
    workspace = tmp_path / "generated"
    for path in (system, data, workspace):
        path.mkdir()
    (workspace / "README.md").write_text("hello from workspace\n", encoding="utf-8")
    subprocess.run(
        ["git", "init", "--quiet", str(workspace)],
        check=True,
        capture_output=True,
        text=True,
    )
    _install_workspace_backend_alias(workspace)

    context = ToolContext(
        repo_dir=system,
        drive_root=data,
        system_repo_dir=system,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="cybergym-alias-test",
        executor_ref={
            "type": "docker_exec",
            "id": "container-id",
            "container_name": "container-id",
            "network": "none",
            "workspace_host_path": str(workspace),
            "workspace_backend_path": "/workspace",
        },
    )
    registry = ToolRegistry(repo_dir=system, drive_root=data)
    registry.set_context(context)

    write_result = registry.execute(
        "write_file",
        {"root": "active_workspace", "path": "/workspace/final.poc", "content": "POC"},
    )
    assert "Written 1 file(s)" in write_result
    marker = workspace / "final.poc"
    assert marker.read_text(encoding="utf-8") == "POC"
    assert (workspace / "workspace" / "final.poc").samefile(marker)

    read_result = registry.execute(
        "read_file",
        {"root": "active_workspace", "path": "/workspace/final.poc"},
    )
    assert "POC" in read_result
    assert final_poc_record(workspace).sha256 == hashlib.sha256(b"POC").hexdigest()


def test_task_body_is_opaque_and_preserves_network_contract(tmp_path):
    config = _config(tmp_path)
    executor = CyberGymExecutor(config)
    task_dir = config.run_root / "task"
    task_dir.mkdir()
    (task_dir / "description.txt").write_text("Find the crash", encoding="utf-8")
    container_name = "cybergym-workspace-agent-" + "a" * 24
    executor._task_containers[container_name] = "b" * 64
    body = executor._task_body(  # noqa: SLF001 - pure boundary assertion
        type("Task", (), {"task_id": "arvo:1", "metadata": {}})(),
        task_dir,
        container_name,
        "attempt-1",
    )
    assert body["task_id"].startswith("cybergym-")
    assert ":" not in body["task_id"]
    assert body["allowed_resources"] == {"network": True, "web": True, "internet": True}
    assert body["resource_policy"]["network_declaration"] == (
        "private_cybergym_sidecar_with_unrestricted_outbound_internet"
    )
    assert body["resource_policy"]["trajectory_audit"] == {
        "required": True,
        "promotion_gate": True,
        "scope": "all_trajectories",
    }
    assert body["executor_ref"]["network"] == "host"
    assert body["executor_ref"]["workspace_backend_path"] == "/workspace"
    assert body["executor_ref"]["id"] == "b" * 64
    assert body["executor_ref"]["container_name"] == "b" * 64
    assert "arvo:1" not in body["metadata"]
    guidance = body["description"]
    assert "structured file tools" in guidance
    assert "do not give them '/workspace/...' paths" in guidance
    assert "do not set cwd='/workspace'" in guidance
    assert '["bash", "./submit.sh", "./final.poc"]' in guidance
    assert "Internet access is available for general technical documentation" in guidance
    assert "issue tracker or bug reports" in guidance
    assert "changelog, commit history, release notes" in guidance
    assert "published patch" in guidance
    assert "ready-made PoC" in guidance
    assert "prior CyberGym solutions" in guidance
    assert "recorded tool and model trajectory is subject to mandatory audit" in guidance
    assert "missing or incomplete evidence makes the result unreviewable" in guidance
    assert "arvo:1" not in guidance
    assert str(task_dir) not in guidance


def test_reused_directory_observation_rechecks_small_manifest_only(tmp_path):
    payload_root = tmp_path / "payload"
    payload_root.mkdir()
    source = tmp_path / "run_manifest.json"
    source.write_text('{"receipt":true}', encoding="utf-8")
    digest = "a" * 64
    observation = {
        "path": str(payload_root),
        "sha256": digest,
        "expected_sha256": digest,
        "files": 2,
        "bytes": 123,
        "attestation_mode": "reused_manifest_observation",
        "attestation_source_manifest": str(source),
        "attestation_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }

    reused = _reuse_directory_observation(
        observation,
        path=payload_root,
        expected_sha256=digest,
        label="CyberGym data root",
    )
    assert reused["status"] == "passed"
    assert reused["path"] == str(payload_root.resolve())

    source.write_text('{"receipt":false}', encoding="utf-8")
    with pytest.raises(ExecutorFailure, match="manifest changed"):
        _reuse_directory_observation(
            observation,
            path=payload_root,
            expected_sha256=digest,
            label="CyberGym data root",
        )


def test_task_body_requires_immutable_workspace_id(tmp_path):
    config = _config(tmp_path)
    executor = CyberGymExecutor(config)
    task_dir = config.run_root / "task"
    task_dir.mkdir()
    (task_dir / "description.txt").write_text("Find the crash", encoding="utf-8")
    with pytest.raises(ExecutorFailure, match="immutable container id"):
        executor._task_body(  # noqa: SLF001 - boundary contract assertion
            type("Task", (), {"task_id": "arvo:1", "metadata": {}})(),
            task_dir,
            "cybergym-workspace-agent-" + "a" * 24,
            "attempt-1",
        )


@_requires_posix_mount_paths
def test_start_uses_same_absolute_server_root_and_docs_probe(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setenv("CYBERGYM_API_KEY", "test-secret-value")
    calls = []

    def command(argv, *, cwd=None, env=None, timeout=None):
        calls.append(list(argv))
        if "network" in argv and "create" in argv:
            return CommandResult(0, "network-id\n", "")
        if "inspect" in argv and "network" in argv:
            return CommandResult(0, '[{"Name":"cybergym-internal","Id":"network-id","Internal":false,"Driver":"bridge","Labels":{"com.ouroboros.campaign":"test-campaign"}}]', "")
        if "run" in argv:
            return CommandResult(0, "server-container-id\n", "")
        if "inspect" in argv and "container" in argv:
            return CommandResult(0, '[{"Name":"/cybergym-server-test-campaign","Id":"server-container-id","State":{"Running":true},"HostConfig":{"NetworkMode":"cybergym-internal"},"NetworkSettings":{"Networks":{"cybergym-internal":{"Aliases":["cybergym-server-test-campaign"],"NetworkID":"network-id"}}},"Config":{"Labels":{},"Image":"sha256:' + "1" * 64 + '"}}]', "")
        return CommandResult(0, "", "")

    seen_http = []

    def http(method, url, **kwargs):
        seen_http.append((method, url))
        return {
            "openapi": "3.0.0",
            "paths": {
                "/submit-vul": {},
                "/submit-fix": {},
                "/query-poc": {},
                "/verify-agent-pocs": {},
            },
        }

    executor = CyberGymExecutor(dataclasses_replace(config, command_runner=command, http_runner=http, provider_probe=False))
    executor.start()
    assert any("--mount" in call and str(config.server_root) in " ".join(call) for call in calls)
    assert seen_http == [("GET", "http://127.0.0.1:8667/openapi.json")]


@_requires_posix_mount_paths
def test_readiness_rejects_openapi_without_private_submit_fix(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setenv("CYBERGYM_API_KEY", "test-secret-value")
    import devtools.benchmarks.cybergym.cybergym_executor as executor_module

    ticks = iter((0.0, 121.0))
    monkeypatch.setattr(executor_module.time, "monotonic", lambda: next(ticks))

    def command(argv, *, cwd=None, env=None, timeout=None):
        if "network" in argv and "create" in argv:
            return CommandResult(0, "network-id\n", "")
        if "inspect" in argv and "network" in argv:
            return CommandResult(
                0,
                '[{"Name":"cybergym-internal","Id":"network-id","Internal":false,"Driver":"bridge","Labels":{"com.ouroboros.campaign":"test-campaign"}}]',
                "",
            )
        if "run" in argv:
            return CommandResult(0, "server-container-id\n", "")
        if "inspect" in argv and "container" in argv:
            return CommandResult(
                0,
                '[{"Name":"/cybergym-server-test-campaign","Id":"server-container-id","State":{"Running":true},"HostConfig":{"NetworkMode":"cybergym-internal"},"NetworkSettings":{"Networks":{"cybergym-internal":{"Aliases":["cybergym-server-test-campaign"],"NetworkID":"network-id"}}},"Config":{"Labels":{},"Image":"sha256:'
                + "1" * 64
                + '"}}]',
                "",
            )
        return CommandResult(0, "", "")

    def http(method, url, **kwargs):
        return {"openapi": "3.0.0", "paths": {"/submit-vul": {}, "/query-poc": {}, "/verify-agent-pocs": {}}}

    executor = CyberGymExecutor(
        dataclasses_replace(config, command_runner=command, http_runner=http, provider_probe=False)
    )
    with pytest.raises(ExecutorFailure, match="documented route"):
        executor.start()


def test_runtime_attestation_reinspects_immutable_ids_before_gateway_boundary(tmp_path, monkeypatch):
    config = _config(tmp_path)
    executor = CyberGymExecutor(config)
    agent_id = "agent-" + "a" * 24
    workspace_name = "cybergym-workspace-" + agent_id
    plan = executor._task_network_plan("arvo:1", agent_id)
    executor.network_id = "network-123"
    executor.server_id = "server-123"
    executor.server_name = "cybergym-server-test-campaign"
    executor._task_containers = {workspace_name: "workspace-123"}

    server = {
        "Id": "server-123",
        "Name": "/" + executor.server_name,
        "Config": {
            "Labels": required_resource_labels(plan, "server"),
            "RepoDigests": ["cybergym/server@" + config.server_image_digest],
        },
        "State": {"Pid": 101, "Running": True},
        "HostConfig": {"NetworkMode": "cybergym-internal"},
        "NetworkSettings": {
            "Networks": {
                "cybergym-internal": {
                    "NetworkID": "network-123",
                    "Aliases": [plan.server_alias],
                }
            },
            # The sidecar has no host-published port; private calls use the
            # server's immutable-id exec path.
            "Ports": {"8666/tcp": None},
        },
        "Mounts": [
            {
                "Source": "/run/user/1006/docker.sock",
                "Destination": "/var/run/docker.sock",
            }
        ],
    }
    workspace = {
        "Id": "workspace-123",
        "Name": "/" + workspace_name,
        "Config": {
            "Labels": required_resource_labels(plan, "workspace"),
            "RepoDigests": ["ouroboros/workspace@" + config.workspace_image_digest],
        },
        "State": {"Pid": 202, "Running": True},
        "HostConfig": {"NetworkMode": "cybergym-internal"},
        "NetworkSettings": {
            "Networks": {
                "cybergym-internal": {
                    "NetworkID": "network-123",
                    "Aliases": [plan.workspace_alias],
                }
            }
        },
        "Mounts": [],
    }
    network = {
        "Name": "cybergym-internal",
        "Id": "network-123",
        "Internal": False,
        "Driver": "bridge",
        "Labels": {"com.ouroboros.campaign": config.campaign_id},
    }
    executor._server_observation = server
    executor._workspace_observations[workspace_name] = workspace
    inspected = []

    def inspect(kind, name):
        inspected.append((kind, name))
        if kind == "network":
            return network
        if name == "server-123":
            return server
        if name == "workspace-123":
            return workspace
        raise AssertionError((kind, name))

    monkeypatch.setattr(executor, "_inspect", inspect)
    monkeypatch.setattr(
        executor,
        "_connectivity_observation",
        lambda plan, workspace_id, api_key: {
            "agent_to_server": True,
            "verifier_to_private": {"reachable": True},
            "agent_to_public": True,
            "agent_to_verifier": False,
            "agent_socket_visible": False,
            "agent_hidden_artifacts": {
                "/cybergym-server-data": True,
                "/cybergym-mask-map.json": True,
                "/cybergym-poc.db": True,
                "/cybergym-fixed": True,
            },
            "agent_secret_env_absent": True,
            "agent_probe_tools": True,
        },
    )
    report = executor._attest_runtime(  # noqa: SLF001 - boundary contract assertion
        type("Task", (), {"task_id": "arvo:1"})(),
        "attempt-1",
        plan,
        workspace_name,
        "valid-key",
    )
    assert report["ok"] is True
    assert ("container", "server-123") in inspected
    assert ("container", "workspace-123") in inspected
    assert ("network", "network-123") in inspected
    assert (config.run_root / "attestations" / "arvo__1" / "attempt-1" / "sidecar_attestation.json").is_file()

    # Keep the foreign-container guard covered while the registry is now
    # synchronized with concurrent workspace startup.
    network["Containers"] = {"foreign-container-id": {}}
    with pytest.raises(ExecutorFailure, match="unknown container"):
        executor._attest_runtime(  # noqa: SLF001 - ownership guard assertion
            type("Task", (), {"task_id": "arvo:1"})(),
            "attempt-1",
            plan,
            workspace_name,
            "valid-key",
        )


@_requires_posix_mount_paths
def test_workspace_registration_and_attestation_share_registry_lock(tmp_path, monkeypatch):
    """An attached workspace is registered before another lane snapshots Docker."""
    import devtools.benchmarks.cybergym.cybergym_executor as executor_module

    config = _config(tmp_path)
    executor = CyberGymExecutor(config)
    executor.started = True
    executor.network_id = "network-id"
    executor.server_id = "server-id"
    executor.server_name = "cybergym-server-test-campaign"
    server = {
        "Id": executor.server_id,
        "Name": "/" + executor.server_name,
        "Config": {"Image": config.server_image_digest},
        "State": {"Running": True, "Pid": 101},
        "NetworkSettings": {"Networks": {"cybergym-internal": {}}},
    }
    executor._server_observation = server

    agent_a = "agent-" + "a" * 24
    agent_b = "agent-" + "b" * 24
    plan_a = executor._task_network_plan("task-a", agent_a)
    plan_b = executor._task_network_plan("task-b", agent_b)
    name_a = "cybergym-workspace-" + agent_a
    name_b = "cybergym-workspace-" + agent_b
    id_a = "a" * 64
    id_b = "b" * 64
    workspace_a = {
        "Id": id_a,
        "Name": "/" + name_a,
        "Config": {"Image": config.workspace_image_digest},
        "State": {"Running": True, "Pid": 202},
        "NetworkSettings": {"Networks": {"cybergym-internal": {}}},
    }
    workspace_b = {
        "Id": id_b,
        "Name": "/" + name_b,
        "Config": {"Image": config.workspace_image_digest},
        "State": {"Running": True, "Pid": 303},
        "NetworkSettings": {"Networks": {"cybergym-internal": {}}},
    }
    executor._task_containers = {name_a: id_a}
    executor._workspace_observations = {name_a: workspace_a}
    attached = {executor.server_id, id_a}
    b_attached = threading.Event()
    release_b = threading.Event()
    a_entered = threading.Event()
    a_done = threading.Event()
    network_snapshots: list[bool] = []
    errors: list[tuple[str, BaseException]] = []

    def command(argv, *, cwd=None, env=None, timeout=None):
        if "run" in argv and name_b in argv:
            attached.add(id_b)
            b_attached.set()
            assert release_b.wait(5), "test barrier was not released"
            return CommandResult(0, id_b + "\n", "")
        raise AssertionError(argv)

    def inspect(kind, target):
        if kind == "network":
            network_snapshots.append(name_b in executor._task_containers)
            return {
                "Name": "cybergym-internal",
                "Id": executor.network_id,
                "Internal": False,
                "Driver": "bridge",
                "Labels": {"com.ouroboros.campaign": config.campaign_id},
                "Containers": {container_id: {} for container_id in attached},
            }
        if target == executor.server_id:
            return server
        if target == id_a:
            return workspace_a
        if target == name_b or target == id_b:
            return workspace_b
        raise AssertionError((kind, target))

    monkeypatch.setattr(executor, "_inspect", inspect)
    monkeypatch.setattr(
        executor,
        "_connectivity_observation",
        lambda plan, workspace_id, api_key: {
            "agent_to_server": True,
            "verifier_to_private": {"reachable": True},
            "agent_to_public": True,
            "agent_to_verifier": False,
            "agent_socket_visible": False,
            "agent_hidden_artifacts": {"hidden": True},
            "agent_secret_env_absent": True,
            "agent_probe_tools": True,
        },
    )
    monkeypatch.setattr(executor_module, "attest_sidecar_runtime", lambda *args, **kwargs: {"ok": True})
    executor.config = dataclasses_replace(config, command_runner=command)

    def start_workspace():
        try:
            executor._workspace(  # noqa: SLF001 - concurrency seam assertion
                type("Task", (), {"task_id": "task-b"})(),
                config.run_root / "task-b",
                plan_b,
            )
        except BaseException as exc:  # pragma: no cover - assertion reports the cause
            errors.append(("workspace", exc))

    def attest_workspace():
        a_entered.set()
        try:
            executor._attest_runtime(  # noqa: SLF001 - concurrency seam assertion
                type("Task", (), {"task_id": "arvo:task-a"})(),
                "attempt-a",
                plan_a,
                name_a,
                "valid-key",
            )
        except BaseException as exc:
            errors.append(("attestation", exc))
        finally:
            a_done.set()

    workspace_thread = threading.Thread(target=start_workspace)
    attestation_thread = threading.Thread(target=attest_workspace)
    workspace_thread.start()
    try:
        assert b_attached.wait(2)
        attestation_thread.start()
        assert a_entered.wait(2)
        assert not a_done.wait(0.1), "attestation crossed the attach-to-custody barrier"
    finally:
        # Always release the worker, including when running this regression
        # against an intentionally broken mutant.
        release_b.set()
        workspace_thread.join(5)
        attestation_thread.join(5)
    assert not workspace_thread.is_alive()
    assert not attestation_thread.is_alive()
    assert errors == []
    assert executor._task_containers[name_b] == id_b
    assert network_snapshots and all(network_snapshots)


@_requires_posix_mount_paths
def test_workspace_start_error_recovers_name_custody_by_inspect(tmp_path):
    config = _config(tmp_path)
    executor = CyberGymExecutor(config)
    executor.network_id = "network-id"
    agent_id = "agent-" + "c" * 24
    plan = executor._task_network_plan("task-c", agent_id)
    name = "cybergym-workspace-" + agent_id
    container_id = "c" * 64
    observed = {
        "Id": container_id,
        "Name": "/" + name,
        "Config": {
            "Image": config.workspace_image_digest,
            "Labels": {
                "com.ouroboros.campaign": config.campaign_id,
                "com.ouroboros.role": "workspace",
                "com.ouroboros.agent_id": plan.opaque_agent_id,
            },
        },
        "NetworkSettings": {
            "Networks": {"cybergym-internal": {"NetworkID": executor.network_id}}
        },
    }

    removed = []

    def command(argv, *, cwd=None, env=None, timeout=None):
        if "inspect" in argv and "container" in argv:
            target = argv[-1]
            if target in removed:
                return CommandResult(1, "", f"Error: No such container: {target}")
            if target in {name, container_id}:
                return CommandResult(0, json.dumps([observed]), "")
        if "rm" in argv and container_id in argv:
            removed.append(container_id)
            removed.append(name)
            return CommandResult(0, "", "")
        if "run" in argv and name in argv:
            raise ExecutorFailure("docker run transport timeout")
        raise AssertionError(argv)

    executor.config = dataclasses_replace(config, command_runner=command)
    with pytest.raises(ExecutorFailure, match="transport timeout"):
        executor._workspace(  # noqa: SLF001 - startup custody assertion
            type("Task", (), {"task_id": "task-c"})(),
            config.run_root / "task-c",
            plan,
        )
    assert name not in executor._task_containers
    assert name not in executor._unresolved_workspace_custody
    assert container_id in removed
    assert not executor._workspace_starting


@_requires_posix_mount_paths
def test_workspace_starts_remain_concurrent_while_registry_publishes_atomically(tmp_path):
    config = _config(tmp_path)
    executor = CyberGymExecutor(config)
    executor.network_id = "network-id"
    starts_entered = []
    both_entered = threading.Event()
    release = threading.Event()
    errors = []
    observations = {}
    plans = {}
    names = {}
    ids = {}
    for suffix in ("d", "e"):
        agent_id = "agent-" + suffix * 24
        plans[suffix] = executor._task_network_plan("task-" + suffix, agent_id)
        names[suffix] = "cybergym-workspace-" + agent_id
        ids[suffix] = suffix * 64
        observations[suffix] = {
            "Id": ids[suffix],
            "Name": "/" + names[suffix],
            "Config": {"Image": config.workspace_image_digest},
            "NetworkSettings": {
                "Networks": {"cybergym-internal": {"NetworkID": executor.network_id}}
            },
        }

    def command(argv, *, cwd=None, env=None, timeout=None):
        if "run" in argv:
            suffix = next(key for key, name in names.items() if name in argv)
            starts_entered.append(suffix)
            if len(starts_entered) == 2:
                both_entered.set()
            assert release.wait(5), "concurrent-start barrier was not released"
            return CommandResult(0, ids[suffix] + "\n", "")
        raise AssertionError(argv)

    def inspect(kind, target):
        if kind == "container":
            suffix = next(key for key, name in names.items() if target == name)
            return observations[suffix]
        raise AssertionError((kind, target))

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(executor, "_inspect", inspect)
    executor.config = dataclasses_replace(config, command_runner=command)

    def start(suffix):
        try:
            executor._workspace(  # noqa: SLF001 - concurrency seam assertion
                type("Task", (), {"task_id": "task-" + suffix})(),
                config.run_root / ("task-" + suffix),
                plans[suffix],
            )
        except BaseException as exc:  # pragma: no cover - assertion reports cause
            errors.append(exc)

    threads = [threading.Thread(target=start, args=(suffix,)) for suffix in ("d", "e")]
    for thread in threads:
        thread.start()
    try:
        assert both_entered.wait(2), "workspace starts were serialized"
    finally:
        release.set()
        for thread in threads:
            thread.join(5)
        monkeypatch.undo()
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert set(starts_entered) == {"d", "e"}
    assert {executor._task_containers[name] for name in names.values()} == set(ids.values())
    assert not executor._workspace_starting


def test_settled_workspace_cleanup_uses_exact_id_and_postcondition(tmp_path, monkeypatch):
    config = _config(tmp_path)
    executor = CyberGymExecutor(config)
    executor.network_id = "network-123"
    name = "cybergym-workspace-agent-" + "a" * 24
    container_id = "workspace-123"
    executor._task_containers = {name: container_id}
    observed = {
        "Id": container_id,
        "Name": "/" + name,
        "Config": {
            "Labels": {
                "com.ouroboros.campaign": config.campaign_id,
                "com.ouroboros.role": "workspace",
            }
        },
        "NetworkSettings": {
            "Networks": {
                "cybergym-internal": {"NetworkID": executor.network_id}
            }
        },
    }
    inspect_calls = []
    removed = False

    def inspect_optional(kind, target):
        nonlocal removed
        inspect_calls.append((kind, target))
        if kind != "container":
            raise AssertionError((kind, target))
        if target == container_id and not removed:
            return observed
        return None

    docker_calls = []

    def docker(*args, **kwargs):
        nonlocal removed
        docker_calls.append(args)
        removed = True
        return CommandResult(0, "", "")

    monkeypatch.setattr(executor, "_inspect_optional", inspect_optional)
    monkeypatch.setattr(executor, "_docker", docker)
    report_path = config.run_root / "cleanup.json"
    report = executor._cleanup_workspace_container(  # noqa: SLF001 - custody assertion
        name, "arvo:1", "attempt-1", report_path
    )
    assert report["ok"] is True
    assert docker_calls == [("rm", "--force", container_id)]
    assert ("container", container_id) in inspect_calls
    assert all(name not in call for call in docker_calls)
    assert report_path.is_file()
    assert name not in executor._task_containers


def dataclasses_replace(config, **changes):
    import dataclasses

    return dataclasses.replace(config, **changes)


def test_task_body_states_wall_clock_budget_derived_from_ceiling(tmp_path):
    config = _config(tmp_path)
    executor = CyberGymExecutor(config)
    task_dir = config.run_root / "task"
    task_dir.mkdir()
    (task_dir / "description.txt").write_text("Find the crash", encoding="utf-8")
    container_name = "cybergym-workspace-agent-" + "a" * 24
    executor._task_containers[container_name] = "b" * 64  # noqa: SLF001 - boundary fixture
    body = executor._task_body(  # noqa: SLF001 - pure boundary assertion
        type("Task", (), {"task_id": "arvo:1", "metadata": {}})(),
        task_dir,
        container_name,
        "attempt-1",
    )
    guidance = body["description"]
    assert "Time budget: you have at most 2 hours of wall time" in guidance
    assert "best-effort /workspace/final.poc before the deadline" in guidance


def test_deadline_guidance_formats_hours_minutes_and_seconds():
    guidance = executor_module._deadline_guidance  # noqa: SLF001 - pure helper
    assert "at most 2 hours of wall time" in guidance(7200)
    assert "at most 1 hour of wall time" in guidance(3600)
    assert "at most 90 minutes of wall time" in guidance(5400)
    assert "at most 1 minute of wall time" in guidance(60)
    assert "at most 45 seconds of wall time" in guidance(45)


def test_isolate_data_root_extends_telemetry_allowed_roots(tmp_path):
    external = tmp_path / "nvme" / "ouroboros-data"
    config = _config(tmp_path, isolate_data_root=external)
    executor = CyberGymExecutor(config)
    assert executor._telemetry_allowed_roots() == (  # noqa: SLF001 - boundary assertion
        config.run_root,
        external.resolve(),
    )
    with pytest.raises(ExecutorFailure, match="isolate_data_root"):
        _config(tmp_path, isolate_data_root=pathlib.Path("/"))
