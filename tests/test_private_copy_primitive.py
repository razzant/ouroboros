"""One private-copy primitive, on real Git.

Every private Git copy starts from a snapshot commit B that git makes from the
source's REAL current files (a private index copied from the source's index; the
source's own index, HEAD and files are never touched) and is a linked worktree
checked out at B. A writer's result is commit C (parent B) plus a plumbing patch;
the checkout can be released and recreated at C; removal drops the checkout, both
pins and the copy's one record file. Each test builds its repositories under
``tmp_path`` and never touches a live installation.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import time

import pytest

from ouroboros import body_candidate
from ouroboros import subagent_worktrees as sw
from ouroboros.unindexed import COPIES
from ouroboros.task_results import write_task_result

SIX_MIB = 6 * 1024 * 1024


def git(cwd, *args, check=True):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=check)


def commit(repo, message="seed"):
    git(repo, "add", "-A")
    git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", message)


def make_source(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True)
    git(path, "init", "-q")
    (path / "a.txt").write_text("one\n", encoding="utf-8")
    (path / "sub").mkdir()
    (path / "sub" / "b.txt").write_text("two\n", encoding="utf-8")
    (path / ".gitignore").write_text("ignored/\n*.log\n", encoding="utf-8")
    commit(path)
    return path


def provision(source, tmp_path, snap="snap1", task_id="t1"):
    return sw.provision_execution_snapshot(target_root=source, task_id=task_id, snapshot_id=snap,
                                           worktree_root=tmp_path / "copies", data_dir=tmp_path / "data")


def events(data: pathlib.Path):
    path = data / "logs" / "events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def refs(repo) -> str:
    return git(repo, "for-each-ref", "--format=%(refname)", "refs/ouroboros/").stdout


# --------------------------------------------------------------------------- #
# Provisioning: what the copy holds, and what the source keeps
# --------------------------------------------------------------------------- #
def test_a_dirty_tree_is_copied_exactly_and_the_source_is_untouched(tmp_path):
    source = make_source(tmp_path / "source")
    (source / "a.txt").write_text("one changed\n", encoding="utf-8")       # tracked edit
    (source / "new.txt").write_text("untracked\n", encoding="utf-8")       # untracked, not ignored
    blob = os.urandom(SIX_MIB)
    (source / "big.bin").write_bytes(blob)                                 # large untracked binary
    (source / "ignored").mkdir()
    (source / "ignored" / "x.txt").write_text("hidden\n", encoding="utf-8")
    (source / "run.log").write_text("hidden\n", encoding="utf-8")
    status_before = git(source, "status", "--porcelain").stdout  # may refresh the real index: read it after
    index_before = (source / ".git" / "index").read_bytes()
    head_before = git(source, "rev-parse", "HEAD").stdout.strip()

    handle = provision(source, tmp_path)

    # The source: its index bytes, HEAD, files and status are what they were.
    assert (source / ".git" / "index").read_bytes() == index_before
    assert git(source, "rev-parse", "HEAD").stdout.strip() == head_before
    assert git(source, "status", "--porcelain").stdout == status_before
    assert (source / "a.txt").read_text(encoding="utf-8") == "one changed\n"
    assert (source / "ignored" / "x.txt").is_file() and (source / "run.log").is_file()
    # The copy: the source's current files, untracked ones included, ignored ones left out.
    copy = pathlib.Path(handle.path)
    assert (copy / "a.txt").read_text(encoding="utf-8") == "one changed\n"
    assert (copy / "new.txt").read_text(encoding="utf-8") == "untracked\n"
    assert (copy / "big.bin").read_bytes() == blob
    assert (copy / "sub" / "b.txt").read_text(encoding="utf-8") == "two\n"
    assert not (copy / "ignored").exists() and not (copy / "run.log").exists()
    # A linked worktree, clean at B; B's parent is the source's HEAD and a ref pins B.
    assert (copy / ".git").is_file()
    assert git(copy, "rev-parse", "HEAD").stdout.strip() == handle.baseline_sha
    assert git(copy, "status", "--porcelain").stdout == ""
    assert git(source, "rev-parse", f"{handle.baseline_sha}^").stdout.strip() == head_before
    assert git(source, "rev-parse", handle.baseline_ref).stdout.strip() == handle.baseline_sha
    assert handle.baseline_ref == "refs/ouroboros/delegated/snap1" and handle.target_head == head_before
    assert handle.git_dir == str((source / ".git").resolve()) and handle.excluded_untracked == ()
    # One record file names the copy; no per-file maps ride it.
    record = sw._record_file(handle.path, tmp_path / "data")
    assert record.parent == tmp_path / "data" / "state" / "private_copies"
    assert re.fullmatch(re.escape(copy.name) + r"-[0-9a-f]{12}\.json", record.name)
    row = json.loads(record.read_text(encoding="utf-8"))
    assert row["kind"] == "delegated_exec" and row["path"] == handle.path and row["snapshot_id"] == "snap1"
    assert not {"file_baseline", "untracked_baseline", "capture_warnings"} & set(row)


def test_a_source_edit_after_the_snapshot_never_reaches_the_copy_and_refuses_nothing(tmp_path, monkeypatch):
    source = make_source(tmp_path / "source")
    (source / "a.txt").write_text("one changed\n", encoding="utf-8")
    real = sw.snapshot_commit

    def then_edit(*args, **kwargs):
        made = real(*args, **kwargs)
        (source / "a.txt").write_text("edited after the snapshot\n", encoding="utf-8")
        (source / "late.txt").write_text("late\n", encoding="utf-8")
        return made

    monkeypatch.setattr(sw, "snapshot_commit", then_edit)
    handle = provision(source, tmp_path)

    copy = pathlib.Path(handle.path)
    assert (source / "a.txt").read_text(encoding="utf-8") == "edited after the snapshot\n"  # the edit happened
    assert (copy / "a.txt").read_text(encoding="utf-8") == "one changed\n"                  # and stayed out
    assert (source / "late.txt").is_file() and not (copy / "late.txt").exists()
    assert git(copy, "status", "--porcelain").stdout == ""
    assert sw.find_execution_snapshot("snap1", tmp_path / "data")["baseline_sha"] == handle.baseline_sha


def test_a_same_size_edit_in_the_index_s_own_second_reaches_the_snapshot(tmp_path, monkeypatch):
    """Git re-reads an entry whose time is not older than its index ("racily clean"): the
    private index keeps the real index's time, so a same-size rewrite that leaves every stat
    field Git trusts unchanged still reaches B. With a fresh index time it would not."""
    source = make_source(tmp_path / "source")
    git(source, "config", "core.trustctime", "false")
    edited, stamp = source / "a.txt", int(time.time()) - 100
    os.utime(edited, (stamp, stamp))
    git(source, "update-index", "--refresh")  # the index records a.txt at that second
    edited.write_bytes(edited.read_bytes().replace(b"one", b"two"))  # same size and line end, same inode
    os.utime(edited, (stamp, stamp))
    os.utime(source / ".git" / "index", (stamp, stamp))  # and the index was written in that second
    index_before = (source / ".git" / "index").read_bytes()

    with monkeypatch.context() as fresh:  # a copy with a fresh time: Git trusts the stat
        fresh.setattr(sw.shutil, "copy2", shutil.copyfile)
        stale, *_ = sw.snapshot_commit(source, message="fresh index time")
    assert git(source, "show", f"{stale}:a.txt").stdout == "one\n"

    snapshot, *_ = sw.snapshot_commit(source, message="kept index time")

    assert git(source, "show", f"{snapshot}:a.txt").stdout == "two\n"
    assert (source / ".git" / "index").read_bytes() == index_before, "the real index is never written"
    assert git(source, "status", "--porcelain").stdout == " M a.txt\n"  # git's own view agrees


def test_a_shallow_clone_is_a_source_like_any_other(tmp_path):
    origin = make_source(tmp_path / "origin")
    (origin / "a.txt").write_text("second\n", encoding="utf-8")
    commit(origin, "second")
    clone = tmp_path / "shallow"
    git(tmp_path, "clone", "-q", "--depth", "1", origin.as_uri(), str(clone))
    assert git(clone, "rev-parse", "--is-shallow-repository").stdout.strip() == "true"
    (clone / "a.txt").write_text("dirty in the shallow clone\n", encoding="utf-8")
    (clone / "u.txt").write_text("untracked\n", encoding="utf-8")

    handle = provision(clone, tmp_path)

    copy = pathlib.Path(handle.path)
    assert (copy / "a.txt").read_text(encoding="utf-8") == "dirty in the shallow clone\n"
    assert (copy / "u.txt").read_text(encoding="utf-8") == "untracked\n"
    assert git(copy, "rev-parse", "HEAD^").stdout == git(clone, "rev-parse", "HEAD").stdout
    assert git(copy, "status", "--porcelain").stdout == ""


def test_a_blobless_partial_clone_is_a_source_like_any_other(tmp_path):
    origin = make_source(tmp_path / "origin")
    (origin / "sub" / "b.txt").write_text("two, second version\n", encoding="utf-8")
    commit(origin, "second")
    git(origin, "config", "uploadpack.allowFilter", "true")
    clone = tmp_path / "blobless"
    git(tmp_path, "clone", "-q", "--no-local", "--filter=blob:none", origin.as_uri(), str(clone))
    assert git(clone, "config", "--get", "remote.origin.partialclonefilter").stdout.strip() == "blob:none"
    (clone / "a.txt").write_text("dirty in the partial clone\n", encoding="utf-8")

    handle = provision(clone, tmp_path)

    copy = pathlib.Path(handle.path)
    assert (copy / "a.txt").read_text(encoding="utf-8") == "dirty in the partial clone\n"
    assert (copy / "sub" / "b.txt").read_text(encoding="utf-8") == "two, second version\n"
    assert git(copy, "status", "--porcelain").stdout == ""


def test_assume_unchanged_and_skip_worktree_edits_are_carried_and_the_flags_stay(tmp_path):
    source = make_source(tmp_path / "source")
    (source / "sub" / "c.txt").write_text("three\n", encoding="utf-8")
    commit(source, "c")
    (source / "a.txt").write_text("hidden by assume-unchanged\n", encoding="utf-8")
    git(source, "update-index", "--assume-unchanged", "a.txt")
    (source / "sub" / "b.txt").write_text("hidden by skip-worktree\n", encoding="utf-8")
    git(source, "update-index", "--skip-worktree", "sub/b.txt", "sub/c.txt")
    (source / "sub" / "c.txt").unlink()  # absent from the work tree, as a sparse checkout leaves it
    assert git(source, "status", "--porcelain").stdout == "", "git's own view hides both edits"

    handle = provision(source, tmp_path)

    copy = pathlib.Path(handle.path)
    assert (copy / "a.txt").read_text(encoding="utf-8") == "hidden by assume-unchanged\n"
    assert (copy / "sub" / "b.txt").read_text(encoding="utf-8") == "hidden by skip-worktree\n"
    assert (copy / "sub" / "c.txt").read_text(encoding="utf-8") == "three\n", "an absent entry keeps its blob"
    flags = git(source, "ls-files", "-v", "a.txt", "sub/b.txt", "sub/c.txt").stdout.splitlines()
    assert sorted(flags) == ["S sub/b.txt", "S sub/c.txt", "h a.txt"], "the source's flags stay"
    assert not (source / "sub" / "c.txt").exists()


def test_an_unborn_branch_snapshots_and_a_nested_repository_is_named_not_copied(tmp_path):
    source = tmp_path / "unborn"
    source.mkdir()
    git(source, "init", "-q")
    (source / "f.txt").write_text("f\n", encoding="utf-8")
    inner = source / "inner"
    inner.mkdir()
    git(inner, "init", "-q")
    (inner / "x.txt").write_text("x\n", encoding="utf-8")

    handle = provision(source, tmp_path)

    copy = pathlib.Path(handle.path)
    assert handle.target_head == ""
    assert git(copy, "rev-list", "--parents", "-n", "1", "HEAD").stdout.split() == [handle.baseline_sha]
    assert (copy / "f.txt").read_text(encoding="utf-8") == "f\n"
    assert not (copy / "inner").exists() and (inner / "x.txt").is_file()
    assert list(handle.excluded_untracked) == [{"path": "inner", "reason": "nested_repository"}]
    assert sw.find_execution_snapshot("snap1", tmp_path / "data")["excluded_untracked"] == [
        {"path": "inner", "reason": "nested_repository"}]


def test_a_linked_worktree_source_is_copied_from_its_own_files(tmp_path):
    main = make_source(tmp_path / "main")
    linked = tmp_path / "linked"
    git(main, "worktree", "add", "-q", "--detach", str(linked))
    (linked / "a.txt").write_text("edit in the linked worktree\n", encoding="utf-8")
    (linked / "only-here.txt").write_text("linked\n", encoding="utf-8")

    handle = provision(linked, tmp_path)

    copy = pathlib.Path(handle.path)
    assert (copy / "a.txt").read_text(encoding="utf-8") == "edit in the linked worktree\n"
    assert (copy / "only-here.txt").read_text(encoding="utf-8") == "linked\n"
    assert (main / "a.txt").read_text(encoding="utf-8") == "one\n" and not (main / "only-here.txt").exists()
    assert handle.git_dir == str((main / ".git").resolve())
    assert git(main, "rev-parse", handle.baseline_ref).stdout.strip() == handle.baseline_sha


def test_a_failed_provisioning_carries_git_s_own_words_and_leaves_nothing(tmp_path):
    source = make_source(tmp_path / "source")
    git(source, "update-ref", "refs/ouroboros/delegated", "HEAD")  # a ref where the pin namespace must be

    with pytest.raises(sw.GitFailed) as failed:
        provision(source, tmp_path)

    text = str(failed.value)
    assert isinstance(failed.value, subprocess.CalledProcessError)
    assert text.startswith("git update-ref refs/ouroboros/delegated/snap1 exited ") and "cannot lock ref" in text
    assert sw.find_execution_snapshot("snap1", tmp_path / "data") is None
    assert not [*(tmp_path / "copies").glob("dlg_*"), *(tmp_path / "copies").glob(f"{COPIES}/dlg_*")]
    assert refs(source) == "refs/ouroboros/delegated\n"
    assert not list((tmp_path / "data" / "state" / "private_copies").glob("*.json"))
    git(source, "update-ref", "-d", "refs/ouroboros/delegated")
    handle = provision(source, tmp_path)  # once git can write the pin, the same call succeeds
    assert pathlib.Path(handle.path).is_dir() and refs(source) == "refs/ouroboros/delegated/snap1\n"


# --------------------------------------------------------------------------- #
# Records: one file per copy, the legacy registry, unreadable records
# --------------------------------------------------------------------------- #
def test_each_copy_has_its_own_record_even_when_two_roots_share_a_directory_name(tmp_path):
    data = tmp_path / "data"
    first, second = tmp_path / "one" / "dlg_same", tmp_path / "two" / "dlg_same"
    assert sw._record_file(first, data) == sw._record_file(str(first) + "/", data)  # one path, one record
    assert sw._record_file(first, data) != sw._record_file(second, data)
    sw._put_record({"path": str(first), "task_id": "a"}, data)
    sw._put_record({"path": str(second), "task_id": "b"}, data)
    assert sorted(row["task_id"] for row in sw.list_worktrees(data)) == ["a", "b"]
    sw._drop_record(first, data)
    assert [row["task_id"] for row in sw.list_worktrees(data)] == ["b"]
    assert sw._record_at(first, data) is None and sw._record_at(second, data)["task_id"] == "b"


def test_the_legacy_registry_is_imported_once_with_a_receipt(tmp_path):
    data = tmp_path / "data"
    kept = {"path": str(tmp_path / "copies" / "acting-new"), "task_id": "acting", "note": "already a record"}
    sw._put_record(kept, data)  # a record written before the import is never replaced by a legacy row
    rows = [{"kind": "delegated_exec", "snapshot_id": "old-run", "task_id": "t-old",
             "path": str(tmp_path / "copies" / "dlg_old")},
            {"task_id": "acting", "path": kept["path"], "note": "legacy"},
            {"task_id": "nameless", "path": ""}]
    raw = json.dumps({"worktrees": rows}).encode("utf-8")
    legacy = data / "state" / "subagent_worktrees.json"
    legacy.write_bytes(raw)

    found = sw.find_execution_snapshot("old-run", data)

    assert found is not None and found["task_id"] == "t-old"
    assert not legacy.exists()
    receipt = json.loads((data / "state" / "private_copies" / ".legacy-import.json").read_text(encoding="utf-8"))
    assert receipt["rows"] == 3 and receipt["moved"] == 1
    assert receipt["legacy_sha256"] == hashlib.sha256(raw).hexdigest()
    assert sw._record_at(kept["path"], data)["note"] == "already a record"
    assert sorted(row["task_id"] for row in sw.list_worktrees(data)) == ["acting", "t-old"]


@pytest.mark.parametrize("payload", [b'{"worktrees": [', b'{"worktrees": "nope"}', b"\xff\xfe"])
def test_an_unreadable_legacy_registry_is_kept_reported_and_blocks_no_new_copy(tmp_path, payload):
    source = make_source(tmp_path / "source")
    data = tmp_path / "data"
    legacy = data / "state" / "subagent_worktrees.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(payload)

    assert sw.list_worktrees(data) == []
    handle = provision(source, tmp_path)

    assert sw.find_execution_snapshot("snap1", data)["path"] == handle.path
    assert legacy.read_bytes() == payload, "the legacy bytes are kept"
    assert not (data / "state" / "private_copies" / ".legacy-import.json").exists()
    reported = [row for row in events(data) if row.get("type") == "private_copy_record_unreadable"]
    assert [(row["op"], row["record"]) for row in reported] == [("legacy_import", str(legacy))]


def test_an_unreadable_record_is_skipped_kept_and_refuses_only_its_own_copy(tmp_path, monkeypatch):
    source = make_source(tmp_path / "source")
    data, copies = tmp_path / "data", tmp_path / "copies"
    broken = provision(source, tmp_path, snap="broken")
    healthy = provision(source, tmp_path, snap="healthy")
    record = sw._record_file(broken.path, data)
    record.write_bytes(b"{not json")

    assert sw.find_execution_snapshot("broken", data) is None  # skipped by every listing
    assert sw.find_execution_snapshot("healthy", data)["path"] == healthy.path
    with pytest.raises(sw.SubagentWorktreeRegistryCorrupt):
        sw._record_at(broken.path, data)
    with pytest.raises(sw.SubagentWorktreeRegistryCorrupt):
        provision(source, tmp_path, snap="broken")  # the same copy refuses before anything moves
    third = provision(source, tmp_path, snap="third")  # another copy is not blocked
    write_task_result(data, "t1", "completed")
    monkeypatch.setattr(body_candidate, "live_lineage_processes", lambda *_args: [])
    report = sw.prune_execution_snapshots(set(), worktree_root=copies, data_dir=data, live=lambda _task: False)
    assert sorted(report["removed"]) == ["healthy", "third"]
    assert not pathlib.Path(healthy.path).exists() and not pathlib.Path(third.path).exists()
    assert pathlib.Path(broken.path).is_dir(), "the copy an unreadable record names stays"
    assert git(source, "rev-parse", broken.baseline_ref).stdout.strip() == broken.baseline_sha
    assert record.read_bytes() == b"{not json", "its bytes are kept"
    reported = [row for row in events(data) if row.get("type") == "private_copy_record_unreadable"]
    assert reported and {row["record"] for row in reported} == {str(record)}


# --------------------------------------------------------------------------- #
# The result commit C, release, recreate, remove
# --------------------------------------------------------------------------- #
def test_the_result_patch_ignores_porcelain_diff_config_and_keeps_named_evidence_out(tmp_path):
    source = make_source(tmp_path / "source")
    (source / ".review-evidence").mkdir()
    (source / ".review-evidence" / "kept.txt").write_text("baseline evidence\n", encoding="utf-8")
    commit(source, "tracked evidence")
    git(source, "config", "diff.noprefix", "true")
    data = tmp_path / "data"
    handle = provision(source, tmp_path)
    copy = pathlib.Path(handle.path)
    (copy / "a.txt").write_text("child edit\n", encoding="utf-8")
    (copy / "sub" / "c.txt").write_text("child new\n", encoding="utf-8")
    (copy / ".review-evidence" / "shot.png").write_bytes(b"\x89PNG evidence")
    (copy / ".review-evidence" / "kept.txt").write_text("baseline evidence, edited\n", encoding="utf-8")
    patch = tmp_path / "capture" / "workspace.patch"

    result = sw.capture_result_commit("snap1", patch, held_out=".review-evidence", data_dir=data)

    text = patch.read_bytes()
    assert result["patch_size"] == len(text) and text.startswith(b"diff --git a/")
    assert b"child edit" in text and b"b/sub/c.txt" in text
    assert b"shot.png" not in text, "new named evidence stays out of the product"
    assert b"b/.review-evidence/kept.txt" in text, "a file the baseline holds stays product"
    tree = git(source, "ls-tree", "-r", "--name-only", result["result_sha"]).stdout.split()
    assert "sub/c.txt" in tree and ".review-evidence/kept.txt" in tree and ".review-evidence/shot.png" not in tree
    assert {"path": ".review-evidence", "reason": "named_evidence"} in result["excluded"]
    assert git(source, "rev-parse", f"{result['result_sha']}^").stdout.strip() == handle.baseline_sha
    assert result["result_ref"] == "refs/ouroboros/delegated-result/snap1"
    assert git(source, "rev-parse", result["result_ref"]).stdout.strip() == result["result_sha"]
    row = sw.find_execution_snapshot("snap1", data)
    assert (row["result_sha"], row["result_tree"], row["result_ref"]) == (
        result["result_sha"], result["result_tree"], result["result_ref"])
    # The patch applies to B elsewhere; the porcelain diff of the same two commits does not.
    elsewhere = tmp_path / "elsewhere"
    git(source, "worktree", "add", "-q", "--detach", str(elsewhere), handle.baseline_sha)
    assert git(elsewhere, "apply", "--check", str(patch), check=False).returncode == 0
    porcelain = tmp_path / "porcelain.patch"
    porcelain.write_bytes(subprocess.run(["git", "diff", "--binary", handle.baseline_sha, result["result_sha"]],
                                         cwd=str(source), capture_output=True, check=True).stdout)
    assert not porcelain.read_bytes().startswith(b"diff --git a/")
    assert git(elsewhere, "apply", "--check", str(porcelain), check=False).returncode != 0


@pytest.mark.parametrize("held_in_baseline", [False, True], ids=["new_entry", "baseline_entry"])
def test_an_entry_standing_where_the_evidence_folder_belongs_follows_the_same_rule(tmp_path, held_in_baseline):
    source = make_source(tmp_path / "source")
    if held_in_baseline:  # the source already tracks a FILE where a run's evidence folder would go
        (source / ".review-evidence").write_text("tracked note\n", encoding="utf-8")
        commit(source, "tracked note")
    data = tmp_path / "data"
    handle = provision(source, tmp_path)
    copy = pathlib.Path(handle.path)
    (copy / ".review-evidence").write_text("written by the run\n", encoding="utf-8")
    (copy / "c.txt").write_text("child new\n", encoding="utf-8")
    patch = tmp_path / "capture" / "workspace.patch"

    result = sw.capture_result_commit("snap1", patch, held_out=".review-evidence", data_dir=data)

    tree = git(source, "ls-tree", "-r", "--name-only", result["result_sha"]).stdout.split()
    assert "c.txt" in tree and b"b/c.txt" in patch.read_bytes()
    if held_in_baseline:  # what B already holds stays product, edits included
        assert ".review-evidence" in tree and b"written by the run" in patch.read_bytes()
    else:  # a new entry under that name is named evidence, never product
        assert ".review-evidence" not in tree and b"written by the run" not in patch.read_bytes()


def test_release_keeps_the_result_and_recreate_restores_its_files_and_index_with_head_at_the_baseline(tmp_path):
    source = make_source(tmp_path / "source")
    data, copies = tmp_path / "data", tmp_path / "copies"
    handle = provision(source, tmp_path)
    copy = pathlib.Path(handle.path)
    (copy / "a.txt").write_text("child edit\n", encoding="utf-8")
    (copy / "c.txt").write_text("child new\n", encoding="utf-8")
    (copy / "sub" / "b.txt").unlink()
    result = sw.capture_result_commit("snap1", tmp_path / "capture" / "workspace.patch", data_dir=data)

    assert sw.release_snapshot_checkout("snap1", worktree_root=copies, data_dir=data) is True
    assert not copy.exists() and str(copy) not in git(source, "worktree", "list", "--porcelain").stdout
    released = sw.find_execution_snapshot("snap1", data)
    assert released["checkout_released"] and released["result_sha"] == result["result_sha"]
    assert git(source, "rev-parse", handle.baseline_ref).stdout.strip() == handle.baseline_sha
    assert git(source, "rev-parse", result["result_ref"]).stdout.strip() == result["result_sha"]

    restored = sw.recreate_snapshot_checkout("snap1", data_dir=data)

    assert "checkout_released" not in restored and "checkout_released" not in sw.find_execution_snapshot("snap1", data)
    assert (copy / "a.txt").read_text(encoding="utf-8") == "child edit\n"
    assert (copy / "c.txt").read_text(encoding="utf-8") == "child new\n"
    assert not (copy / "sub" / "b.txt").exists()
    assert git(copy, "rev-parse", "HEAD").stdout.strip() == handle.baseline_sha  # HEAD at B
    assert git(copy, "write-tree").stdout.strip() == result["result_tree"]  # the index of C
    assert git(copy, "diff", "--quiet", check=False).returncode == 0  # and its files
    again = sw.capture_result_commit("snap1", tmp_path / "capture2" / "workspace.patch", data_dir=data)
    assert again["result_tree"] == result["result_tree"], "a continuation measures the whole run from B"


def test_a_file_c_holds_stays_tracked_once_the_checkout_is_recreated_even_when_git_ignores_it(tmp_path):
    source = make_source(tmp_path / "source")  # its .gitignore hides ignored/ and *.log
    data, copies = tmp_path / "data", tmp_path / "copies"
    copy = pathlib.Path(provision(source, tmp_path).path)
    (copy / "expected.log").write_text("a fixture the run delivers\n", encoding="utf-8")
    (copy / "build.log").write_text("a log it does not\n", encoding="utf-8")
    git(copy, "add", "-f", "expected.log")
    first = sw.capture_result_commit("snap1", tmp_path / "c1" / "workspace.patch", data_dir=data)
    assert sw.release_snapshot_checkout("snap1", worktree_root=copies, data_dir=data)

    sw.recreate_snapshot_checkout("snap1", worktree_root=copies, data_dir=data)
    again = sw.capture_result_commit("snap1", tmp_path / "c2" / "workspace.patch", data_dir=data)

    names = git(source, "ls-tree", "-r", "--name-only", again["result_sha"]).stdout.split()
    assert again["result_tree"] == first["result_tree"] and "expected.log" in names and "build.log" not in names


@pytest.mark.parametrize("remnant", ["release", "restore"])
def test_a_remnant_of_an_interrupted_release_or_restore_is_checked_out_again_whole(tmp_path, remnant):
    source = make_source(tmp_path / "source")
    data, copies = tmp_path / "data", tmp_path / "copies"
    handle = provision(source, tmp_path)
    copy = pathlib.Path(handle.path)
    (copy / "a.txt").write_text("child edit\n", encoding="utf-8")
    result = sw.capture_result_commit("snap1", tmp_path / "capture" / "workspace.patch", data_dir=data)
    assert sw.checkout_whole(sw.find_execution_snapshot("snap1", data))
    if remnant == "release":  # the removal stopped part-way after the record said so
        sw._update_record(handle.path, lambda row: {**row, "checkout_released": "earlier"}, data, op="test")
        (copy / "sub" / "b.txt").unlink()
    else:  # a restore stopped after `worktree add --no-checkout`, before any file
        assert sw.release_snapshot_checkout("snap1", worktree_root=copies, data_dir=data)
        sw._update_record(handle.path, lambda row: {**row, "checkout_released": "earlier"}, data, op="test")
        git(source, "worktree", "add", "--detach", "--no-checkout", str(copy), result["result_sha"])
    assert copy.is_dir() and not sw.checkout_whole(sw.find_execution_snapshot("snap1", data))

    restored = sw.recreate_snapshot_checkout("snap1", worktree_root=copies, data_dir=data)

    assert "checkout_released" not in restored and sw.checkout_whole(sw.find_execution_snapshot("snap1", data))
    assert (copy / "a.txt").read_text(encoding="utf-8") == "child edit\n" and (copy / "sub" / "b.txt").is_file()
    assert git(copy, "rev-parse", "HEAD").stdout.strip() == handle.baseline_sha
    assert git(copy, "write-tree").stdout.strip() == result["result_tree"]
    assert git(source, "worktree", "list", "--porcelain").stdout.count(f"worktree {copy.as_posix()}") == 1


def test_a_release_outside_its_root_deletes_nothing_and_never_marks_the_checkout(tmp_path):
    source = make_source(tmp_path / "source")
    data, copies = tmp_path / "data", tmp_path / "copies"
    handle = provision(source, tmp_path)
    copy = pathlib.Path(handle.path)
    (copy / "a.txt").write_text("child edit\n", encoding="utf-8")
    sw.capture_result_commit("snap1", tmp_path / "capture" / "workspace.patch", data_dir=data)

    assert sw.release_snapshot_checkout("snap1", worktree_root=tmp_path / "elsewhere", data_dir=data) is False

    assert copy.is_dir() and sw.checkout_whole(sw.find_execution_snapshot("snap1", data))
    assert sw.release_snapshot_checkout("snap1", worktree_root=copies, data_dir=data) is True  # its own root
    assert not copy.exists() and not sw.checkout_whole(sw.find_execution_snapshot("snap1", data))


def test_removing_a_copy_drops_only_its_own_worktree_registration(tmp_path):
    """A global ``worktree prune`` would also drop the source's own missing worktree (one on an
    unplugged disk); a copy's removal names its one path."""
    source = make_source(tmp_path / "source")
    data, copies = tmp_path / "data", tmp_path / "copies"
    own = tmp_path.resolve() / "usb" / "feature"  # Git records a worktree's real path
    git(source, "worktree", "add", "-q", "--detach", str(own))
    shutil.rmtree(tmp_path / "usb")  # the disk is unplugged: git still lists it, as missing
    handle = provision(source, tmp_path)

    assert sw.remove_execution_snapshot("snap1", worktree_root=copies, data_dir=data) is True

    listed = git(source, "worktree", "list", "--porcelain").stdout
    assert pathlib.Path(handle.path).as_posix() not in listed and f"worktree {own.as_posix()}" in listed


def test_a_released_copy_without_a_captured_result_is_never_recreated_at_b(tmp_path):
    source = make_source(tmp_path / "source")
    data, copies = tmp_path / "data", tmp_path / "copies"
    handle = provision(source, tmp_path, snap="uncaptured")
    assert sw.release_snapshot_checkout("uncaptured", worktree_root=copies, data_dir=data) is True

    with pytest.raises(FileNotFoundError, match="no captured result"):
        sw.recreate_snapshot_checkout("uncaptured", data_dir=data)

    assert not pathlib.Path(handle.path).exists()
    assert sw.find_execution_snapshot("uncaptured", data)["checkout_released"]


@pytest.mark.parametrize("released", [False, True], ids=["checked_out", "released"])
def test_remove_drops_the_checkout_both_pins_and_the_record(tmp_path, released):
    source = make_source(tmp_path / "source")
    data, copies = tmp_path / "data", tmp_path / "copies"
    handle = provision(source, tmp_path)
    other = provision(source, tmp_path, snap="other")
    (pathlib.Path(handle.path) / "a.txt").write_text("child edit\n", encoding="utf-8")
    result = sw.capture_result_commit("snap1", tmp_path / "capture" / "workspace.patch", data_dir=data)
    if released:
        assert sw.release_snapshot_checkout("snap1", worktree_root=copies, data_dir=data)
    assert git(source, "rev-parse", "--verify", "-q", result["result_ref"], check=False).returncode == 0

    assert sw.remove_execution_snapshot("snap1", worktree_root=copies, data_dir=data) is True

    assert not pathlib.Path(handle.path).exists()
    for ref in (handle.baseline_ref, result["result_ref"]):
        assert git(source, "rev-parse", "--verify", "-q", ref, check=False).returncode != 0, ref
    assert sw.find_execution_snapshot("snap1", data) is None and not sw._record_file(handle.path, data).exists()
    assert pathlib.Path(handle.path).as_posix() not in git(source, "worktree", "list", "--porcelain").stdout
    # The sibling copy keeps its checkout, its pin and its record.
    assert pathlib.Path(other.path).is_dir() and refs(source) == f"{other.baseline_ref}\n"
    assert sw.find_execution_snapshot("other", data)["path"] == other.path


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions that bind this user")
def test_a_folder_git_cannot_open_is_named_never_silently_skipped(tmp_path):
    source = make_source(tmp_path / "source")
    locked = source / "locked"
    locked.mkdir()
    (locked / "x.txt").write_text("x\n", encoding="utf-8")
    locked.chmod(0)
    try:
        handle = provision(source, tmp_path)
    finally:
        locked.chmod(0o755)
    assert {"path": "locked", "reason": "unlistable_directory"} in handle.excluded_untracked
    readable = provision(source, tmp_path, snap="readable")  # once it opens, its files are simply in B
    assert all(row["reason"] != "unlistable_directory" for row in readable.excluded_untracked)
    assert git(source, "show", f"{readable.baseline_sha}:locked/x.txt").stdout == "x\n"


def test_a_leaked_repository_location_never_redirects_a_copy(tmp_path, monkeypatch):
    source = make_source(tmp_path / "source")
    victim = make_source(tmp_path / "victim")
    (source / "only-in-source.txt").write_text("source\n", encoding="utf-8")
    with monkeypatch.context() as leaked:  # a host process started from a Git hook, say
        leaked.setenv("GIT_DIR", str(victim / ".git"))
        leaked.setenv("GIT_INDEX_FILE", str(victim / ".git" / "index"))
        handle = provision(source, tmp_path)
    assert (pathlib.Path(handle.path) / "only-in-source.txt").read_text(encoding="utf-8") == "source\n"
    assert git(victim, "cat-file", "-e", handle.baseline_sha, check=False).returncode != 0
    assert git(source, "cat-file", "-e", handle.baseline_sha, check=False).returncode == 0
