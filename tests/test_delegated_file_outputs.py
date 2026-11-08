"""File-only and mixed delegated results cross the actual capture/disposition seam.

A binary or large result rides the B..C patch as a binary diff (no separate file
outputs); the copy's checkout goes once the capture is saved, and the result
commit C keeps the run's bytes until apply or reject. Delivered files stage by the
target's index membership before the apply, and a failed staging is reversed."""

import json
import subprocess
from pathlib import Path

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.subagent_worktrees import find_execution_snapshot
from ouroboros.tools.delegate import _capture_terminal_patch
from ouroboros.tools.subagent_integration import _integrate_delegated_patch
from tests.test_delegated_run_isolation import TestCaptureAndIntegrate as _CaptureFixture, _git, _isolated_entry


@pytest.mark.parametrize("mixed", [False, True])
def test_binary_result_is_applied_staged_and_survives_snapshot_cleanup(tmp_path, monkeypatch, mixed):
    target, ctx, handle = _CaptureFixture()._provisioned(tmp_path, monkeypatch)
    execution = Path(handle.path)
    binary = b"\x00\xff\x01binary-result" * 1000
    (execution / "report.bin").write_bytes(binary)
    if mixed:
        (execution / "tracked.txt").write_text("new result\n")
    entry = _isolated_entry(ctx, target, handle)
    try:
        capture = _capture_terminal_patch(ctx, entry)
        assert capture["status"] == "ready_with_changes", capture
        assert not (target / "report.bin").exists()
        manifest = json.loads(Path(capture["manifest_artifact"]).read_text(encoding="utf-8"))
        assert manifest["file_outputs"] == [] and "GIT binary patch" in Path(capture["patch_artifact"]).read_text()
        assert capture["checkout_released"] is True and not execution.exists()
        result = _integrate_delegated_patch(ctx, entry.run_id, "apply", "accept complete file result")
        assert "✅ Integrated" in result, result
        assert (target / "report.bin").read_bytes() == binary
        assert "report.bin" in _git(target, "diff", "--cached", "--name-only").stdout
        assert find_execution_snapshot(handle.snapshot_id) is None
        assert not execution.exists()
        assert Path(capture["manifest_artifact"]).is_file()
    finally:
        custody._CUSTODY.clear()


def test_binary_destination_concurrent_edit_preserves_both_results(tmp_path, monkeypatch):
    target, ctx, handle = _CaptureFixture()._provisioned(tmp_path, monkeypatch)
    execution = Path(handle.path)
    (execution / "report.bin").write_bytes(b"\x00prepared")
    entry = _isolated_entry(ctx, target, handle)
    try:
        capture = _capture_terminal_patch(ctx, entry)
        (target / "report.bin").write_bytes(b"owner edit")
        result = _integrate_delegated_patch(ctx, entry.run_id, "apply", "")
        assert "INTEGRATE_CONFLICT" in result, result
        assert (target / "report.bin").read_bytes() == b"owner edit"
        # The run's result survives its released checkout as commit C in the target's own Git.
        assert not execution.exists()
        result_blob = subprocess.run(["git", "cat-file", "blob", f"{capture['result_commit']}:report.bin"],
                                     cwd=target, check=True, capture_output=True).stdout
        assert result_blob == b"\x00prepared"
        assert not entry.patch_disposed
        assert Path(capture["manifest_artifact"]).is_file()
    finally:
        custody._CUSTODY.clear()


def test_binary_input_baseline_crosses_real_capture_and_apply(tmp_path, monkeypatch):
    from ouroboros.subagent_worktrees import provision_execution_snapshot
    from tests.test_delegated_run_isolation import _seed_target, _nanny_ctx
    target = _seed_target(tmp_path)
    (target / "report.docx").write_bytes(b"\x00original document")
    ctx = _nanny_ctx(tmp_path, target, monkeypatch)
    handle = provision_execution_snapshot(target_root=target, task_id=ctx.task_id, snapshot_id="binary-input")
    execution = Path(handle.path)
    assert (execution / "report.docx").read_bytes() == b"\x00original document"
    (execution / "report.docx").write_bytes(b"\x00revised document")
    entry = _isolated_entry(ctx, target, handle)
    try:
        capture = _capture_terminal_patch(ctx, entry)
        assert capture["status"] == "ready_with_changes", capture
        result = _integrate_delegated_patch(ctx, entry.run_id, "apply", "revised document")
        assert "✅ Integrated" in result, result
        assert (target / "report.docx").read_bytes() == b"\x00revised document"
        assert not execution.exists()
    finally:
        custody._CUSTODY.clear()


@pytest.mark.parametrize("kind", ["tracked_in_ignored_folder", "new_ignored"])
def test_ignored_paths_are_delivered_and_staged_by_index_membership(tmp_path, monkeypatch, kind):
    """#2446: a run's edit of a file the target tracks under an ignored folder stages; a new file the
    target's ignore rules exclude is delivered, left out of the index and named. Neither reverses the
    ordinary edit beside it, and the target's own staged work stays staged."""
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.subagent_worktrees import provision_execution_snapshot
    from tests.test_delegated_run_isolation import _nanny_ctx, _seed_target

    target = _seed_target(tmp_path)
    (target / ".gitignore").write_text(".build/\ndist/\n", encoding="utf-8", newline="\n")
    (target / ".build").mkdir()
    (target / ".build" / "pinned.cfg").write_text("base\n", encoding="utf-8", newline="\n")
    _git(target, "add", "-f", "--", ".gitignore", ".build/pinned.cfg")
    _git(target, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "rules",
         "--", ".gitignore", ".build/pinned.cfg")
    ctx = _nanny_ctx(tmp_path, target, monkeypatch)
    handle = provision_execution_snapshot(target_root=target, task_id=ctx.task_id, snapshot_id="snapX")
    custody._CUSTODY.clear()
    execution = Path(handle.path)
    (execution / "newfile.py").write_text("print('hi')\n", encoding="utf-8", newline="\n")
    rel = ".build/pinned.cfg" if kind == "tracked_in_ignored_folder" else "dist/new.bin"
    (execution / rel).parent.mkdir(parents=True, exist_ok=True)
    (execution / rel).write_text("run output\n", encoding="utf-8", newline="\n")
    _git(execution, "add", "-f", "--", rel)
    entry = _isolated_entry(ctx, target, handle)
    try:
        assert _capture_terminal_patch(ctx, entry)["status"] == "ready_with_changes"
        out = _integrate_delegated_patch(ctx, entry.run_id, "apply", "")
        assert "✅ Integrated" in out and entry.patch_disposed == "applied", out
        assert (target / rel).read_text(encoding="utf-8") == "run output\n"
        assert _git(target, "show", ":newfile.py").stdout == "print('hi')\n"
        assert _git(target, "show", ":staged.txt").stdout == "staged\n"
        if kind == "tracked_in_ignored_folder":
            assert _git(target, "show", f":{rel}").stdout == "run output\n"
            assert "file(s), staged)" in out and "Delivered, not indexed" not in out, out
            assert "Changes are staged but NOT committed" in out, out
        else:
            assert _git(target, "ls-files", "--", rel).stdout == ""
            assert "Delivered, not indexed" in out and rel in out, out
            assert "Changes are staged" not in out and "the files named above are not staged at all" in out, out
            verdict = json.loads((task_artifact_dir_path(ctx.drive_root, ctx.task_id)
                                  / f"subagent_patch_verdict_run_{entry.run_id}.json").read_text(encoding="utf-8"))
            assert verdict["unindexed"] == [rel]
    finally:
        custody._CUSTODY.clear()


def test_an_assume_unchanged_entry_really_stages_and_keeps_its_flag(tmp_path, monkeypatch):
    """The delegated working-tree apply: plain `add -u` skips this entry silently, which would leave the run's
    edit unstaged under a "staged" receipt."""
    from ouroboros.subagent_worktrees import provision_execution_snapshot
    from tests.test_delegated_run_isolation import _nanny_ctx, _seed_target

    target = _seed_target(tmp_path)
    (target / "flagged.txt").write_text("base\n", encoding="utf-8", newline="\n")
    _git(target, "add", "--", "flagged.txt")
    _git(target, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "flagged", "--", "flagged.txt")
    _git(target, "update-index", "--assume-unchanged", "--", "flagged.txt")
    ctx = _nanny_ctx(tmp_path, target, monkeypatch)
    handle = provision_execution_snapshot(target_root=target, task_id=ctx.task_id, snapshot_id="snapX")
    custody._CUSTODY.clear()
    (Path(handle.path) / "flagged.txt").write_text("run output\n", encoding="utf-8", newline="\n")
    entry = _isolated_entry(ctx, target, handle)
    try:
        assert _capture_terminal_patch(ctx, entry)["status"] == "ready_with_changes"
        out = _integrate_delegated_patch(ctx, entry.run_id, "apply", "")
        assert "✅ Integrated" in out and "file(s), staged)" in out, out
        assert _git(target, "show", ":flagged.txt").stdout == "run output\n"
        assert _git(target, "ls-files", "-v", "--", "flagged.txt").stdout.startswith("h ")
        assert _git(target, "show", ":staged.txt").stdout == "staged\n"
    finally:
        custody._CUSTODY.clear()


def test_a_staging_failure_reverses_and_restores_only_the_entries_it_changed(tmp_path, monkeypatch):
    """A real index failure after the apply (another Git process holds index.lock): the tree is reversed and
    exactly the index entries the attempt staged return, so the target's own staged state survives and the
    retry lane stays open."""
    from ouroboros.tools import patch_staging as staging

    target, ctx, handle = _CaptureFixture()._provisioned(tmp_path, monkeypatch)
    execution = Path(handle.path)
    (execution / "newfile.py").write_text("print('hi')\n", encoding="utf-8", newline="\n")
    (execution / "staged.txt").write_text("staged\nby the run\n", encoding="utf-8", newline="\n")
    entry = _isolated_entry(ctx, target, handle)

    def state():
        files = {name: (target / name).read_bytes() for name in ("tracked.txt", "staged.txt")}
        return (_git(target, "ls-files", "--stage").stdout, _git(target, "diff", "--cached").stdout,
                files, (target / "newfile.py").exists())

    real = staging._git_paths

    def locked(root, args, paths, prefix=b""):
        if args[1:2] == ["add"] and not {"-u", "-N", "--refresh"} & set(args):  # staged.txt is staged by now
            lock = Path(root) / ".git" / "index.lock"
            lock.write_bytes(b"")
            try:
                return real(root, args, paths, prefix)
            finally:
                lock.unlink()
        return real(root, args, paths, prefix)

    try:
        assert _capture_terminal_patch(ctx, entry)["status"] == "ready_with_changes"
        before = state()
        monkeypatch.setattr(staging, "_git_paths", locked)
        out = _integrate_delegated_patch(ctx, entry.run_id, "apply", "")
        assert "INTEGRATE_APPLIED_UNSTAGED" in out and "reversed" in out and "index.lock" in out, out
        assert state() == before and entry.patch_disposed == ""
        monkeypatch.setattr(staging, "_git_paths", real)
        out = _integrate_delegated_patch(ctx, entry.run_id, "apply", "")
        assert "✅ Integrated" in out, out
        assert _git(target, "show", ":staged.txt").stdout == "staged\nby the run\n"
    finally:
        custody._CUSTODY.clear()
