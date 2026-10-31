"""A private execution copy is git's own checkout of its snapshot commit.

The snapshot commit B records the source's current files; ``git worktree add
--no-checkout`` plus ``reset --hard`` writes them, so the copy holds git's
representation of B (attribute conversions apply, the executable bit and links
ride the commit) and an untouched copy captures as an empty result. A source edit
after the snapshot never reaches the copy and refuses nothing; a copy whose
provisioning has not returned is kept by GC; an inherited sparse cone keeps its
outside entries in the snapshot and in the result.
"""
from __future__ import annotations

import os
import pathlib
import stat
import subprocess
import threading

import pytest

from ouroboros import body_candidate
from ouroboros import subagent_worktrees as trees
from ouroboros.unindexed import COPIES

_WAIT_SEC = 30.0


def git(root, *args, check=True):
    return subprocess.run(["git", *args], cwd=root, check=check, capture_output=True)


def source_tree(tmp_path):
    root = tmp_path / "target"
    root.mkdir()
    for args in (("init", "-q"), ("config", "user.name", "Fixture"), ("config", "user.email", "f@example.invalid"),
                 ("config", "core.autocrlf", "false")):
        git(root, *args)
    (root / ".gitattributes").write_bytes(b"*.txt text eol=crlf\n*.bin -text\n")
    files = {"lines.txt": b"alpha\nbeta\n", "late.txt": b"a" * (1024 * 1024 + 7) + b"\nend\n",
             "tool.sh": b"#!/bin/sh\necho hi\n", "locked.cfg": b"keep\n", "empty.cfg": b"",
             "dirty.cfg": b"v1\n", "blob.bin": bytes(range(256)) * 3, "gone.cfg": b"deleted before the snapshot\n"}
    for name, content in files.items():
        (root / name).write_bytes(content)
    if os.name != "nt":
        os.symlink("tool.sh", root / "link")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "fixture")
    (root / "dirty.cfg").write_bytes(b"v2 dirty\n")  # an unstaged edit is the copy's input
    (root / "gone.cfg").unlink()
    os.chmod(root / "tool.sh", 0o755)
    os.chmod(root / "locked.cfg", 0o444)
    return root


def provision(tmp_path, target, *, snapshot_id="fill"):
    return trees.provision_execution_snapshot(
        target_root=target, task_id="fill-task", snapshot_id=snapshot_id,
        worktree_root=tmp_path / "snapshots", data_dir=tmp_path / "data")


def nothing_left(tmp_path, target, snapshot_id):
    assert trees.find_execution_snapshot(snapshot_id, tmp_path / "data") is None
    assert not git(target, "for-each-ref", "refs/ouroboros/delegated/").stdout
    assert not [*(tmp_path / "snapshots").glob("dlg_*"), *(tmp_path / "snapshots").glob(f"{COPIES}/dlg_*")]


def test_the_copy_is_git_s_checkout_of_the_snapshot_and_an_untouched_capture_is_empty(tmp_path):
    target = source_tree(tmp_path)

    handle = provision(tmp_path, target)

    copy = pathlib.Path(handle.path)
    # Git's representation of B: the eol attribute applies on checkout, raw files stay raw.
    assert (copy / "lines.txt").read_bytes() == b"alpha\r\nbeta\r\n" != (target / "lines.txt").read_bytes()
    assert (copy / "late.txt").read_bytes().endswith(b"\r\nend\r\n")
    assert git(copy, "show", f"{handle.baseline_sha}:lines.txt").stdout == b"alpha\nbeta\n"
    for name in ("blob.bin", "empty.cfg", "locked.cfg", "tool.sh"):
        assert (copy / name).read_bytes() == (target / name).read_bytes(), name
    assert (copy / "dirty.cfg").read_bytes() == b"v2 dirty\n", "the unstaged edit is the copy's input"
    assert not (copy / "gone.cfg").exists()
    if os.name != "nt":
        assert stat.S_IMODE(os.stat(copy / "tool.sh").st_mode) & 0o111, "the executable bit rides the commit"
        assert os.path.islink(copy / "link") and os.readlink(copy / "link") == "tool.sh"
    assert git(copy, "status", "--porcelain=v1", "-z").stdout == b""
    git(copy, "diff", "--quiet", "--no-ext-diff", handle.baseline_sha, "--")

    patch = tmp_path / "capture" / "workspace.patch"
    result = trees.capture_result_commit("fill", patch, data_dir=tmp_path / "data")
    assert result["result_tree"] == handle.baseline_tree and patch.read_bytes() == b"", \
        "conversions on checkout never show up as changes"
    (copy / "lines.txt").write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")
    result = trees.capture_result_commit("fill", patch, data_dir=tmp_path / "data")
    assert result["result_tree"] != handle.baseline_tree
    assert git(copy, "show", f"{result['result_sha']}:lines.txt").stdout == b"alpha\nbeta\ngamma\n"


@pytest.mark.parametrize("edit", [b"v3 dirty\n", b"v2 dirty and longer\n", b"v2\n"], ids=["same_size", "longer", "shorter"])
def test_a_source_edit_during_population_never_reaches_the_copy_and_refuses_nothing(tmp_path, monkeypatch, edit):
    target = source_tree(tmp_path)
    real_git = trees._git

    def edit_at_checkout(repo_dir, *args, **kwargs):
        if args[:1] == ("reset",):  # the baseline is pinned and recorded; the checkout has not begun
            (target / "dirty.cfg").write_bytes(edit)
        return real_git(repo_dir, *args, **kwargs)

    monkeypatch.setattr(trees, "_git", edit_at_checkout)
    handle = provision(tmp_path, target)

    copy = pathlib.Path(handle.path)
    assert (target / "dirty.cfg").read_bytes() == edit, "the edit happened"
    assert (copy / "dirty.cfg").read_bytes() == b"v2 dirty\n", "the copy keeps the snapshot's bytes"
    assert git(copy, "status", "--porcelain=v1", "-z").stdout == b""
    assert trees.find_execution_snapshot("fill", tmp_path / "data")["provisioning_sec"] == handle.provisioning_sec > 0


def test_a_copy_whose_provisioning_has_not_returned_is_kept_until_its_task_is_over(tmp_path, monkeypatch):
    target = source_tree(tmp_path)
    parked, release = threading.Event(), threading.Event()
    real_check_out = trees._check_out

    def park_after_checkout(*args, **kwargs):
        real_check_out(*args, **kwargs)
        parked.set()  # files are written; the final record is not
        assert release.wait(_WAIT_SEC)

    monkeypatch.setattr(trees, "_check_out", park_after_checkout)
    outcome = {}

    def run():
        try:
            outcome["handle"] = provision(tmp_path, target)
        except Exception as exc:  # surfaced by the assertions below
            outcome["error"] = exc

    worker = threading.Thread(target=run, name="test-provision", daemon=True)
    worker.start()
    try:
        assert parked.wait(_WAIT_SEC)
        row = trees.find_execution_snapshot("fill", tmp_path / "data")
        copy = pathlib.Path(row["path"])
        # The provisional record is light (no completion facts), and it already names the copy.
        assert row["provisioning_sec"] == 0 and copy.is_dir()
        # Custody is not open yet, and the task has not been proved over: GC keeps it.
        assert trees.prune_execution_snapshots(set(), worktree_root=tmp_path / "snapshots",
                                               data_dir=tmp_path / "data") == {"removed": [], "kept": ["fill"]}
    finally:
        release.set()
        worker.join(_WAIT_SEC)
    from ouroboros.task_results import write_task_result
    assert "error" not in outcome and outcome["handle"].path == str(copy)
    assert trees.find_execution_snapshot("fill", tmp_path / "data")["provisioning_sec"] > 0
    write_task_result(tmp_path / "data", "fill-task", "completed")
    monkeypatch.setattr("supervisor.queue.task_settlement_liveness", lambda _task: False)
    monkeypatch.setattr(body_candidate, "live_lineage_processes", lambda *_args: [])
    assert trees.prune_execution_snapshots(set(), worktree_root=tmp_path / "snapshots",
                                           data_dir=tmp_path / "data")["removed"] == ["fill"]
    nothing_left(tmp_path, target, "fill")


@pytest.mark.parametrize("flow", ["snapshot", "task_copy"])
def test_an_entry_outside_an_inherited_sparse_cone_stays_in_the_snapshot_and_its_result(tmp_path, flow):
    root = tmp_path / "sparse"
    (root / "docs").mkdir(parents=True)
    for args in (("init", "-q"), ("config", "user.name", "Fixture"), ("config", "user.email", "f@example.invalid")):
        git(root, *args)
    (root / "a.txt").write_bytes(b"in cone\n")
    (root / "docs" / "b.txt").write_bytes(b"committed\n")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "fixture")
    (root / "docs" / "b.txt").write_bytes(b"dirty outside the cone\n")
    if git(root, "sparse-checkout", "set", "--no-cone", "/a.txt", check=False).returncode != 0:
        pytest.skip("this git has no sparse-checkout")
    if not (root / "docs" / "b.txt").exists():
        pytest.skip("this git removed the modified entry outside the cone")
    copies = (tmp_path / "snapshots").resolve()
    if flow == "snapshot":
        handle = provision(tmp_path, root, snapshot_id="sparse")
    else:
        handle = trees.provision_worktree(repo_dir=root, task_id="sparse-task", worktree_root=copies,
                                          data_dir=tmp_path / "data", source_is_system_repo=False)
    copy = pathlib.Path(handle.path).resolve()
    baseline = handle.baseline_sha if flow == "snapshot" else handle.base_sha
    assert git(copy, "show", f"{baseline}:docs/b.txt").stdout == b"dirty outside the cone\n"
    assert (copy / "a.txt").read_bytes() == b"in cone\n"
    if (copy / "docs" / "b.txt").exists():  # this git did not carry the sparse patterns into the worktree
        assert (copy / "docs" / "b.txt").read_bytes() == b"dirty outside the cone\n"
    else:  # the copy keeps its source's cone: the entry is in B, outside the work tree
        assert git(copy, "ls-files", "-v", "docs/b.txt").stdout == b"S docs/b.txt\n"
    assert git(copy, "status", "--porcelain=v1", "-z").stdout == b"", "no entry reads as deleted"
    if flow == "snapshot":
        patch = tmp_path / "capture" / "workspace.patch"
        result = trees.capture_result_commit("sparse", patch, data_dir=tmp_path / "data")
        assert result["result_tree"] == handle.baseline_tree and patch.read_bytes() == b""
