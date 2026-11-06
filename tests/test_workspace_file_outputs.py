"""File-only and mixed Git results preserve complete bytes and conflict semantics."""
from hashlib import sha256
import os
from pathlib import Path
import subprocess

import pytest

from ouroboros import workspace_patch_capture as capture
from ouroboros.workspace_file_outputs import (
    file_output_changes, prepare_file_outputs, verify_file_outputs,
)


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout


def repository(root, *, unborn=False):
    root.mkdir()
    git(root, "init")
    if not unborn:
        git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
            "commit", "--allow-empty", "-m", "base")
    return root


def commit(root):
    git(root, "add", ".")
    git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "files")
    return git(root, "rev-parse", "HEAD").decode().strip()


@pytest.mark.parametrize("unborn", [False, True])
def test_binary_file_only_result_applies_and_rolls_back_without_touching_index(tmp_path, unborn):
    source = repository(tmp_path / "source", unborn=unborn)
    if unborn:
        target = repository(tmp_path / "target", unborn=True)
    else:
        target = tmp_path / "target"
        git(tmp_path, "clone", str(source), str(target))
    (source / "nested").mkdir()
    (source / "nested" / "answer.bin").write_bytes(b"answer\x00bytes")
    (target / "unrelated.txt").write_text("owner dirty content")
    git(target, "add", "unrelated.txt")
    index_before = git(target, "ls-files", "--stage", "-z")
    _, manifest = capture.write_workspace_patch_artifacts(source, tmp_path / "capture", task={})
    assert manifest["status"] == "ready_with_changes", manifest["errors"]
    assert manifest["patch_size"] == 0
    assert not (tmp_path / "capture" / "workspace.patch").exists()
    rows = file_output_changes(manifest, tmp_path / "capture")
    assert len(rows) == 1 and rows[0]["before"] is None
    with prepare_file_outputs(rows, target, baseline_sha=manifest["base_head"]) as prepared:
        assert prepared.paths == ["nested/answer.bin"]
        prepared.apply()
        assert prepared.verify_applied()
        assert (target / "nested" / "answer.bin").read_bytes() == b"answer\x00bytes"
        prepared.rollback()
    assert not (target / "nested" / "answer.bin").exists()
    assert (target / "unrelated.txt").read_text() == "owner dirty content"
    assert git(target, "ls-files", "--stage", "-z") == index_before


@pytest.mark.parametrize("operation", ["modify", "delete", "symlink"])
def test_tracked_large_changes_have_exact_preimages_and_complete_apply(tmp_path, monkeypatch, operation):
    source = repository(tmp_path / "source")
    original = b"large baseline\x00content"
    (source / "data.bin").write_bytes(original)
    (source / "data.bin").chmod(0o755)
    baseline = commit(source)
    target = tmp_path / "target"
    git(tmp_path, "clone", str(source), str(target))
    index_before = git(target, "ls-files", "--stage", "-z")
    if operation == "modify":
        (source / "data.bin").write_bytes(b"new binary\x00result")
        (source / "data.bin").chmod(0o644)
    elif operation == "delete":
        (source / "data.bin").unlink()
    elif operation == "rename":
        git(source, "mv", "data.bin", "renamed.bin")
    else:
        (source / "data.bin").unlink()
        try:
            (source / "data.bin").symlink_to("other-target.bin")
        except OSError as exc:
            pytest.skip(str(exc))
    monkeypatch.setattr(capture, "_PATCH_FILE_REFERENCE_BYTES", 8)
    _, manifest = capture.write_workspace_patch_artifacts(source, tmp_path / "capture", task={})
    assert manifest["status"] == "ready_with_changes", manifest["errors"]
    assert manifest["patch_size"] == 0
    rows = file_output_changes(manifest, tmp_path / "capture")
    prior = next(row["before"] for row in rows if row["path"] == "data.bin")
    assert prior["size"] == len(original) and prior["sha256"] == sha256(original).hexdigest()
    with prepare_file_outputs(rows, target, baseline_sha=baseline) as prepared:
        prepared.apply()
        assert verify_file_outputs(rows, target)
        if operation == "modify":
            assert (target / "data.bin").read_bytes() == b"new binary\x00result"
            if os.name != "nt":
                assert (target / "data.bin").stat().st_mode & 0o777 == 0o644
        elif operation == "rename":
            assert (target / "renamed.bin").read_bytes() == original
            assert not (target / "data.bin").exists()
        elif operation == "symlink":
            assert (target / "data.bin").is_symlink()
            assert os.readlink(target / "data.bin") == "other-target.bin"
        else:
            assert not (target / "data.bin").exists()
        prepared.rollback()
    assert (target / "data.bin").read_bytes() == original
    assert git(target, "ls-files", "--stage", "-z") == index_before
    assert not git(target, "status", "--porcelain")


def two_file_capture(tmp_path):
    source = repository(tmp_path / "source")
    target = repository(tmp_path / "target")
    for name in ("one.bin", "two.bin"):
        (source / name).write_bytes(name.encode() + b"\x00answer")
    cap_dir = tmp_path / "capture"
    _, manifest = capture.write_workspace_patch_artifacts(source, cap_dir, task={})
    return target, manifest, cap_dir


def test_all_target_paths_checked_before_any_file_apply(tmp_path):
    target, manifest, cap_dir = two_file_capture(tmp_path)
    rows = file_output_changes(manifest, cap_dir)
    with prepare_file_outputs(rows, target) as prepared:
        (target / "two.bin").write_bytes(b"concurrent edit")
        with pytest.raises(ValueError, match="changed before apply"):
            prepared.apply()
    assert not (target / "one.bin").exists()
    assert (target / "two.bin").read_bytes() == b"concurrent edit"


def test_each_target_checked_again_when_an_editor_changes_later_file_during_copy(tmp_path, monkeypatch):
    from ouroboros import artifacts
    target, manifest, cap_dir = two_file_capture(tmp_path)
    rows = file_output_changes(manifest, cap_dir)
    with prepare_file_outputs(rows, target) as prepared:
        original_copy = artifacts.copy_artifact_file
        def copy_with_other_editor(source, destination, **kwargs):
            if Path(destination) == target / "one.bin":
                (target / "two.bin").write_bytes(b"concurrent edit during first copy")
            return original_copy(source, destination, **kwargs)
        monkeypatch.setattr(artifacts, "copy_artifact_file", copy_with_other_editor)
        with pytest.raises(ValueError, match="changed during apply: two.bin"):
            prepared.apply()
        assert (target / "one.bin").read_bytes() == b"one.bin\x00answer"
        assert (target / "two.bin").read_bytes() == b"concurrent edit during first copy"
        prepared.rollback()
    assert not (target / "one.bin").exists()
    assert (target / "two.bin").read_bytes() == b"concurrent edit during first copy"


def test_malformed_side_returns_explicit_value_error(tmp_path):
    target, manifest, cap_dir = two_file_capture(tmp_path)
    manifest["file_output_changes"][0]["before"] = "unobserved"
    with pytest.raises(ValueError, match="no complete identity"):
        file_output_changes(manifest, cap_dir)


def test_corrupt_second_member_never_applies_first_file(tmp_path):
    import zipfile
    from ouroboros.artifacts import stream_artifact_file
    target, manifest, cap_dir = two_file_capture(tmp_path)
    archive = next(row for row in manifest["file_outputs"] if row["kind"] == "workspace_file_outputs")
    with zipfile.ZipFile(cap_dir / archive["name"], "w") as zipped:
        zipped.writestr("one.bin", b"one.bin\x00answer")
        zipped.writestr("two.bin", b"wrong bytes")
    archive.update(stream_artifact_file(cap_dir / archive["name"]))
    rows = file_output_changes(manifest, cap_dir)
    with pytest.raises(OSError, match="verification"):
        with prepare_file_outputs(rows, target):
            pytest.fail("corrupt source must not reach apply")
    assert not (target / "one.bin").exists()
    assert not (target / "two.bin").exists()


def test_rollback_preserves_concurrent_post_apply_edits(tmp_path):
    target, manifest, cap_dir = two_file_capture(tmp_path)
    with prepare_file_outputs(file_output_changes(manifest, cap_dir), target) as prepared:
        prepared.apply()
        (target / "one.bin").write_bytes(b"owner newer version")
        with pytest.raises(ValueError, match="preserved concurrent changes"):
            prepared.rollback()
    assert (target / "one.bin").read_bytes() == b"owner newer version"
    assert not (target / "two.bin").exists()


def test_target_parent_escape_and_incomplete_old_capture_fail_explicitly(tmp_path):
    target, manifest, cap_dir = two_file_capture(tmp_path)
    rows = file_output_changes(manifest, cap_dir)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (target / "nested").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(str(exc))
    rows[0]["path"] = "nested/one.bin"
    with pytest.raises(ValueError, match="escapes target"):
        with prepare_file_outputs(rows, target):
            pytest.fail("escaped path")
    del manifest["file_output_changes"]
    with pytest.raises(ValueError, match="baseline unavailable"):
        file_output_changes(manifest, cap_dir)


def test_full_large_payload_is_streamed_through_capture_prepare_and_apply(tmp_path, monkeypatch):
    from ouroboros.artifacts import stream_artifact_file
    source = repository(tmp_path / "source")
    target = repository(tmp_path / "target")
    original_read = Path.read_bytes
    def no_large_read(path):
        assert path.stat().st_size <= 50 * 1024 * 1024, "large file read into one bytes object"
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", no_large_read)
    with (source / "large.bin").open("wb") as handle:
        for _ in range(52):
            handle.write(b"\x00payload" * (1024 * 128))
    expected = stream_artifact_file(source / "large.bin")
    cap_dir = tmp_path / "capture"
    _, manifest = capture.write_workspace_patch_artifacts(source, cap_dir, task={})
    assert manifest["status"] == "ready_with_changes", manifest["errors"]
    with prepare_file_outputs(file_output_changes(manifest, cap_dir), target) as prepared:
        prepared.apply()
        assert stream_artifact_file(target / "large.bin") == expected


def test_text_patch_and_file_result_deliver_together_with_explicit_git_staging(tmp_path):
    source = repository(tmp_path / "source")
    (source / "code.txt").write_text("old code\n")
    baseline = commit(source)
    target = tmp_path / "target"
    git(tmp_path, "clone", str(source), str(target))
    (source / "code.txt").write_text("new code\n")
    (source / "answer.bin").write_bytes(b"complete\x00answer")
    cap_dir = tmp_path / "capture"
    _, manifest = capture.write_workspace_patch_artifacts(source, cap_dir, task={})
    assert manifest["status"] == "ready_with_changes" and manifest["patch_size"] > 0
    with prepare_file_outputs(file_output_changes(manifest, cap_dir), target, baseline_sha=baseline) as prepared:
        git(target, "apply", "--index", str(cap_dir / "workspace.patch"))
        prepared.apply()
        git(target, "add", "--", *prepared.paths)
    assert (target / "code.txt").read_text() == "new code\n"
    assert git(target, "show", ":answer.bin") == b"complete\x00answer"
    assert not git(target, "diff", "--name-only")
    assert set(git(target, "diff", "--cached", "--name-only").decode().splitlines()) == {"answer.bin", "code.txt"}


def test_legacy_file_results_under_ignore_rules_land_beside_the_text_patch(tmp_path, monkeypatch):
    """#2446 on the copy-less legacy integration: the text patch applies with --3way --index and a file result
    the target already tracks under an ignored folder stages; a new file its ignore rules exclude, from the
    text patch or a file result alike, is delivered, kept out of the index and named in the same receipt."""
    import json
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    target = repository(tmp_path / "target")
    (target / ".gitignore").write_bytes(b".build/\ndist/\n")
    (target / "code.txt").write_bytes(b"old\n")
    (target / ".build").mkdir()
    (target / ".build" / "pinned.bin").write_bytes(b"pinned\x00baseline")
    git(target, "add", "-f", "--", ".build/pinned.bin")
    base = commit(target)
    child = tmp_path / "child"
    git(tmp_path, "clone", "-q", str(target), str(child))
    (child / "code.txt").write_bytes(b"new\n")
    (child / ".build" / "pinned.bin").write_bytes(b"pinned\x00by the child")
    (child / "dist").mkdir()
    (child / "dist" / "new.bin").write_bytes(b"new\x00file result")
    (child / "dist" / "notes.txt").write_bytes(b"n\n")  # small text: it rides the text patch
    git(child, "add", "-f", "--", "dist/new.bin", "dist/notes.txt")
    monkeypatch.setattr(capture, "_PATCH_FILE_REFERENCE_BYTES", 8)
    artifacts = task_artifact_dir_path(data, "child", create=True)
    _, manifest = capture.write_workspace_patch_artifacts(
        child, artifacts, task={"metadata": {"workspace_preflight": {"git": {"head": base}}}})
    assert manifest["status"] == "ready_with_changes" and manifest["patch_size"] > 0, manifest
    assert {row["path"] for row in file_output_changes(manifest, artifacts)} == {".build/pinned.bin", "dist/new.bin"}
    write_task_result(data, "child", "completed", parent_task_id="parent",
                      task_constraint={"mode": "acting_subagent", "surface": "self_worktree"})
    out = _integrate_subagent_patch(ToolContext(repo_dir=target, drive_root=data, task_id="parent"), task_id="child")
    assert "✅ Integrated" in out and "4 file(s), 2 staged)" in out, out
    assert "Delivered, not indexed" in out and "dist/new.bin, dist/notes.txt" in out, out
    assert "Changes are staged" not in out and "the files named above are not staged at all" in out, out
    assert git(target, "show", ":code.txt") == b"new\n"
    assert git(target, "show", ":.build/pinned.bin") == b"pinned\x00by the child"
    assert (target / "dist" / "new.bin").read_bytes() == b"new\x00file result"
    assert (target / "dist" / "notes.txt").read_text() == "n\n"
    assert git(target, "ls-files", "--", "dist") == b""
    verdict = json.loads((task_artifact_dir_path(data, "parent") / "subagent_patch_verdict_child.json").read_text())
    assert (verdict["outcome"], verdict["unindexed"]) == ("applied", ["dist/new.bin", "dist/notes.txt"])


@pytest.mark.parametrize("failing", ["new_file_add", "flag_remark"])
def test_a_failed_legacy_staging_undoes_text_patch_file_results_and_index(tmp_path, monkeypatch, failing):
    """A real index failure while the copy-less legacy integration stages (another Git process holds
    index.lock), first at a new file's add, then at the last step that marks a staged entry assume-unchanged
    again: the text patch, the file results and every touched index entry and flag go back, the parent's own
    unrelated staged and unstaged work stays as it was, and the same captured result then lands."""
    import json
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.task_results import write_task_result
    from ouroboros.tools import patch_staging as staging
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.subagent_integration import _integrate_subagent_patch

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    target = repository(tmp_path / "target")
    (target / ".gitignore").write_bytes(b"dist/\n")
    for name, body in (("code.txt", b"old\n"), ("flag.txt", b"flag\n"), ("other.txt", b"base\n")):  # LF on every OS
        (target / name).write_bytes(body)
    (target / "pinned.bin").write_bytes(b"pinned\x00baseline")
    base = commit(target)
    child = tmp_path / "child"
    git(tmp_path, "clone", "-q", str(target), str(child))
    (target / "other.txt").write_bytes(b"base\nparent staged\n")  # the parent's own work, untouched by the child
    git(target, "add", "--", "other.txt")
    (target / "other.txt").write_bytes(b"base\nparent staged\nparent unstaged\n")
    git(target, "update-index", "--assume-unchanged", "--", "flag.txt")
    (child / "code.txt").write_bytes(b"new\n")
    (child / "flag.txt").write_bytes(b"flag2\n")
    (child / "fresh.txt").write_bytes(b"f\n")
    (child / "pinned.bin").write_bytes(b"pinned\x00by the child")
    (child / "dist").mkdir()
    (child / "dist" / "notes.txt").write_bytes(b"n\n")
    (child / "dist" / "new.bin").write_bytes(b"new\x00file result")
    git(child, "add", "-f", "--", "dist/new.bin", "dist/notes.txt")
    monkeypatch.setattr(capture, "_PATCH_FILE_REFERENCE_BYTES", 8)
    artifacts = task_artifact_dir_path(data, "child", create=True)
    _, manifest = capture.write_workspace_patch_artifacts(
        child, artifacts, task={"metadata": {"workspace_preflight": {"git": {"head": base}}}})
    assert manifest["status"] == "ready_with_changes" and manifest["patch_size"] > 0, manifest
    assert {row["path"] for row in file_output_changes(manifest, artifacts)} == {"pinned.bin", "dist/new.bin"}
    write_task_result(data, "child", "completed", parent_task_id="parent",
                      task_constraint={"mode": "acting_subagent", "surface": "self_worktree"})

    def state():
        files = {name: (target / name).read_bytes() if (target / name).exists() else None
                 for name in ("code.txt", "flag.txt", "other.txt", "pinned.bin", "fresh.txt",
                              "dist/notes.txt", "dist/new.bin")}
        return (git(target, "ls-files", "--stage", "-v", "-z"), git(target, "diff", "--cached", "--binary"),
                git(target, "diff-files", "--name-only", "-z"), files)

    before, real, calls = state(), staging._git_paths, []

    def locked(root, args, paths, prefix=b""):
        calls.append(args)
        step = (args[1:2] == ["add"] and not {"-u", "-N", "--refresh"} & set(args) if failing == "new_file_add"
                else "--assume-unchanged" in args and sum("--assume-unchanged" in seen for seen in calls) == 1)
        if step:
            lock = Path(root) / ".git" / "index.lock"
            lock.write_bytes(b"")
            try:
                return real(root, args, paths, prefix)
            finally:
                lock.unlink()
        return real(root, args, paths, prefix)

    ctx = ToolContext(repo_dir=target, drive_root=data, task_id="parent")
    monkeypatch.setattr(staging, "_git_paths", locked)
    out = _integrate_subagent_patch(ctx, task_id="child")
    assert "INTEGRATE_CONFLICT" in out and "index.lock" in out and "nothing is half-applied" in out, out
    assert state() == before
    verdict = json.loads((task_artifact_dir_path(data, "parent") / "subagent_patch_verdict_child.json").read_text())
    assert (verdict["outcome"], verdict["applied"]) == ("conflict", False)
    monkeypatch.setattr(staging, "_git_paths", real)  # the lock is gone: the same captured result now lands
    out = _integrate_subagent_patch(ctx, task_id="child")
    assert "✅ Integrated" in out and "6 file(s), 4 staged)" in out, out
    assert git(target, "show", ":code.txt") == b"new\n" and git(target, "show", ":fresh.txt") == b"f\n"
    assert git(target, "show", ":flag.txt") == b"flag2\n" and git(target, "ls-files", "-v", "--", "flag.txt")[:1] == b"h"
    assert git(target, "show", ":pinned.bin") == b"pinned\x00by the child"
    assert git(target, "ls-files", "--", "dist") == b"" and (target / "dist" / "new.bin").exists()
    assert git(target, "show", ":other.txt") == b"base\nparent staged\n"
    assert (target / "other.txt").read_text() == "base\nparent staged\nparent unstaged\n"
