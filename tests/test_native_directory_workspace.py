"""Ordinary native folders: direct results retain outputs without a Git baseline."""
import json
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros.artifacts import copy_directory_to_task_artifacts, copy_file_to_task_artifacts
from ouroboros.headless import finalize_task_artifacts, task_artifacts_dir
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.task_status import load_effective_task_result
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.subagent_integration import _integrate_subagent_patch
from supervisor.events_subagent_admission import _resolve_subagent_constraint


def _setup(tmp_path, monkeypatch):
    workspace, repo, drive = (tmp_path / name for name in ("project", "system", "data"))
    for path in (workspace, repo, drive):
        path.mkdir()
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "1")
    ctx = SimpleNamespace(REPO_DIR=repo, DRIVE_ROOT=drive)
    constraint, root, mode, error = _resolve_subagent_constraint(
        ctx, tid="child", requested_constraint={
            "mode": "acting_subagent", "surface": "external_workspace", "write_root": str(workspace),
        }, workspace_root=str(workspace), workspace_mode="external", base_sha="parent-system-sha",
        parent_task_id="parent",
    )
    assert not error
    assert root == str(workspace) and mode == "external_workspace"
    assert constraint["base_sha"] == ""
    task = {"id": "child", "workspace_root": root, "workspace_mode": mode, "task_constraint": constraint}
    write_task_result(drive, "child", "completed", parent_task_id="parent", root_task_id="parent",
                      delegation_role="subagent", task_constraint=constraint,
                      workspace_root=root, result="Task performed its requested action")
    parent = ToolContext(repo_dir=repo, drive_root=drive, task_id="parent",
                         workspace_root=workspace, workspace_mode="external")
    output_ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id="child", workspace_root=workspace,
                             workspace_mode="external")
    return workspace, drive, task, parent, output_ctx


def test_native_directory_retains_and_verifies_binary_output(tmp_path, monkeypatch):
    workspace, drive, task, parent, output_ctx = _setup(tmp_path, monkeypatch)
    output = workspace / "animation.bin"
    payload = b"\x00\xff\x81" * 1024
    output.write_bytes(payload)
    record = copy_file_to_task_artifacts(output_ctx, output, kind="process_output")
    # Direct operation has no full tree inventory or baseline copy.
    monkeypatch.setattr("os.walk", lambda *a, **k: pytest.fail("direct finalization scanned the workspace"))
    # The artifact scanner uses rglob, not a source-tree walk.
    finalize_task_artifacts(drive, task)
    manifest_path = task_artifacts_dir(drive, "child") / "workspace_patch.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["capture_kind"] == "directory_direct"
    assert manifest["before"] == "unknown" and not manifest["complete"]
    assert manifest["registered_outputs"][0]["sha256"] == record["sha256"]
    assert load_task_result(drive, "child")["artifact_status"] == "ready"
    assert not (workspace / ".git").exists()
    assert not (manifest_path.parent / "workspace.patch").exists()
    message = _integrate_subagent_patch(parent, task_id="child")
    assert "Verified 1 registered file postimage(s)" in message
    assert output.read_bytes() == payload
    assert load_effective_task_result(drive, "child")["child_result_disposition"] == "integrated"
    verdict = json.loads((task_artifacts_dir(drive, "parent") / "subagent_patch_verdict_child.json").read_text())
    assert verdict["outcome"] == "verified_registered_outputs" and not verdict["applied"]


@pytest.mark.parametrize("directory", [False, True])
def test_store_conversion_is_not_a_false_workspace_postimage(tmp_path, monkeypatch, directory):
    from pathlib import Path
    from ouroboros.tools.registry import ToolRegistry
    import sys

    workspace, drive, task, parent, output_ctx = _setup(tmp_path, monkeypatch)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    source = workspace / "result"
    if directory:
        source.mkdir()
        (source / "page.txt").write_text("original")
        record = copy_directory_to_task_artifacts(output_ctx, source)[1]
    else:
        source.write_text("original")
        record = copy_file_to_task_artifacts(output_ctx, source)
    registry = ToolRegistry(repo_dir=parent.repo_dir, drive_root=drive)
    registry.set_context(output_ctx)
    converted = registry.execute_result("run_command", {
        "cmd": [sys.executable, "-c", f"open({record['name']!r}, 'ab').write(b'converted')"],
        "cwd": "artifact_store", "outputs": [record["name"]],
    })
    assert converted.code == "OK" and converted.meta.get("artifact_registered") is True, converted.text
    finalize_task_artifacts(drive, task)
    manifest = json.loads((task_artifacts_dir(drive, "child") / "workspace_patch.json").read_text())
    message = _integrate_subagent_patch(parent, task_id="child")
    assert "INTEGRATE_DIRECTORY_OUTPUT_MISMATCH" not in message, message
    assert all(item["path"] != record["path"] for item in manifest["registered_outputs"])
    assert Path(record["path"]).read_bytes().endswith(b"converted")
    assert (source / "page.txt" if directory else source).read_text() == "original"


def test_native_directory_preserves_nonfile_result_extent(tmp_path, monkeypatch):
    workspace, drive, task, parent, _ = _setup(tmp_path, monkeypatch)
    # Opaque shell/GUI effects are real but have no registered output evidence.
    (workspace / "shell-output.txt").write_text("action already happened")
    finalize_task_artifacts(drive, task)
    result = load_task_result(drive, "child")
    assert result["artifact_status"] == "ready"
    assert result["result"] == "Task performed its requested action"
    message = _integrate_subagent_patch(parent, task_id="child", reason="Accepted the observed action")
    assert "no file postimages were verified" in message
    assert "complete changed-file set remain unknown" in message
    assert "nothing to apply" not in message
    verdict = json.loads((task_artifacts_dir(drive, "parent") / "subagent_patch_verdict_child.json").read_text())
    assert verdict["outcome"] == "direct_result_observed"


@pytest.mark.parametrize("changed", ["source", "artifact"])
def test_native_directory_refuses_changed_postimage(tmp_path, monkeypatch, changed):
    workspace, drive, task, parent, output_ctx = _setup(tmp_path, monkeypatch)
    source = workspace / "out.bin"
    source.write_bytes(b"captured\x00")
    record = copy_file_to_task_artifacts(output_ctx, source)
    finalize_task_artifacts(drive, task)
    from pathlib import Path
    (source if changed == "source" else Path(record["path"])).write_bytes(b"changed")
    message = _integrate_subagent_patch(parent, task_id="child")
    assert "INTEGRATE_DIRECTORY_OUTPUT_MISMATCH" in message
    assert not load_task_result(drive, "child").get("child_result_disposition")


def test_native_directory_bundle_verifies_members(tmp_path, monkeypatch):
    workspace, drive, task, parent, output_ctx = _setup(tmp_path, monkeypatch)
    output = workspace / "rendered"
    output.mkdir()
    (output / "frame.png").write_bytes(b"image\x00")
    copy_directory_to_task_artifacts(output_ctx, output)
    finalize_task_artifacts(drive, task)
    assert "Verified 1 registered file postimage(s)" in _integrate_subagent_patch(parent, task_id="child")
    (output / "frame.png").write_bytes(b"later")
    assert "INTEGRATE_DIRECTORY_OUTPUT_MISMATCH" in _integrate_subagent_patch(parent, task_id="child")


@pytest.mark.parametrize("context_kind", ["supervisor", "tool"])
def test_native_directory_admission_keeps_git_geometry(tmp_path, monkeypatch, context_kind):
    workspace, drive, _, parent, _ = _setup(tmp_path, monkeypatch)
    from supervisor.events_subagent_admission import _validate_external_workspace
    ctx = (SimpleNamespace(REPO_DIR=parent.repo_dir, DRIVE_ROOT=drive) if context_kind == "supervisor"
           else ToolContext(repo_dir=parent.repo_dir, drive_root=drive))
    assert "overlap" in _validate_external_workspace(ctx, str(parent.repo_dir))
    assert "overlap" in _validate_external_workspace(ctx, str(drive))
    (workspace / ".git").write_text("gitdir: /nonexistent/project-git")
    def admission():
        return _resolve_subagent_constraint(
            ctx, tid="bad", requested_constraint={"mode": "acting_subagent", "surface": "external_workspace",
                                                  "write_root": str(workspace)},
            workspace_root="", workspace_mode="", base_sha="", parent_task_id="parent",
        )
    assert "Git worktree could not be resolved" in admission()[3]
    (workspace / ".git").unlink()
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    constraint, root, _, error = admission()
    assert not error and constraint["base_sha"] == "(unborn)" and root == str(workspace)


def test_native_directory_reject_does_not_undo_effects(tmp_path, monkeypatch):
    workspace, drive, task, parent, _ = _setup(tmp_path, monkeypatch)
    output = workspace / "out.txt"
    output.write_text("already here")
    finalize_task_artifacts(drive, task)
    message = _integrate_subagent_patch(parent, task_id="child", decision="reject")
    assert "rejecting this result does not undo" in message
    assert output.read_text() == "already here"


def test_native_directory_parent_cannot_verify_another_folder(tmp_path, monkeypatch):
    workspace, drive, task, parent, _ = _setup(tmp_path, monkeypatch)
    finalize_task_artifacts(drive, task)
    other = tmp_path / "other"
    other.mkdir()
    assert "TARGET_MISMATCH" in _integrate_subagent_patch(parent, task_id="child", target_root=str(other))
    assert not (workspace / ".git").exists()


@pytest.mark.parametrize("surface,mixed,drift", [("self_worktree", False, ""), ("self_worktree", True, ""),
                                                 ("external_workspace", False, ""),
                                                 ("self_worktree", False, "worktree"),
                                                 ("self_worktree", False, "index")])
def test_native_git_receives_complete_file_outputs(tmp_path, monkeypatch, surface, mixed, drift):
    from ouroboros import workspace_patch_capture as capture
    monkeypatch.setattr(capture, "_PATCH_FILE_REFERENCE_BYTES", 4)
    source, target, drive = (tmp_path / name for name in ("source", "target", "data"))
    source.mkdir()
    def git(path, *args):
        return subprocess.run(["git", *args], cwd=path, check=True, capture_output=True).stdout
    git(source, "init", "-q")
    git(source, "config", "user.name", "Test")
    git(source, "config", "user.email", "test@example.invalid")
    (source / "large.bin").write_bytes(b"original\x00data")
    (source / "text.txt").write_text("a\n")
    git(source, "add", ".")
    git(source, "commit", "-qm", "Initial")
    base = git(source, "rev-parse", "HEAD").decode().strip()
    git(tmp_path, "clone", "-q", str(source), str(target))
    payload = b"updated\x00\xffbinary"
    (source / "large.bin").write_bytes(payload)
    if mixed:
        (source / "text.txt").write_text("b\n")
    art = task_artifacts_dir(drive, "child-git")
    constraint = {"mode": "acting_subagent", "surface": surface,
                  "write_root": str(source), "base_sha": base}
    artifacts, manifest = capture.write_workspace_patch_artifacts(
        source, art, task={"task_constraint": constraint},
    )
    assert manifest["status"] == "ready_with_changes"
    assert manifest["file_output_changes"] and bool(manifest["patch_size"]) is mixed
    if not mixed:
        (art / "workspace.patch").unlink(missing_ok=True)
    write_task_result(drive, "child-git", "completed", parent_task_id="parent", root_task_id="parent",
                      delegation_role="subagent", task_constraint=constraint, workspace_root=str(source), artifacts=artifacts)
    ctx = ToolContext(repo_dir=target, drive_root=drive, task_id="parent",
                      workspace_root=source if surface == "external_workspace" else None,
                      workspace_mode="external" if surface == "external_workspace" else "")
    if drift == "unmerged":
        oid = git(target, "rev-parse", "HEAD:large.bin").decode().strip()
        conflict = (f"0 {'0' * len(oid)}\tlarge.bin\n"
                    + "".join(f"100644 {oid} {stage}\tlarge.bin\n" for stage in (1, 2, 3)))
        subprocess.run(["git", "update-index", "--index-info"], cwd=target,
                       input=conflict.encode(), check=True, capture_output=True)
        index_before = git(target, "ls-files", "--stage", "-z")
        assert git(target, "ls-files", "--unmerged")
        run = subprocess.run

        def no_apply(args, *a, **kw):
            if args[:2] == ["git", "apply"] and "--3way" in args:
                pytest.fail("an unmerged index must be reported before patch apply")
            return run(args, *a, **kw)

        monkeypatch.setattr(subprocess, "run", no_apply)
    elif drift == "stage_failure":
        run = subprocess.run
        def fail_stage(args, *a, **kw):
            if args[:1] == ["git"] and "add" in args[1:3] and not {"--refresh", "-N"} & set(args):  # staging only
                return subprocess.CompletedProcess(args, 1, b"", b"index write failed")
            return run(args, *a, **kw)
        monkeypatch.setattr(subprocess, "run", fail_stage)
    elif drift:
        (target / "large.bin").write_bytes(b"parent change")
        if drift == "index":
            git(target, "add", "large.bin")
            (target / "large.bin").write_bytes(b"original\x00data")
    output = _integrate_subagent_patch(ctx, task_id="child-git")
    if drift:
        assert "INTEGRATE_CONFLICT" in output
        if drift == "unmerged":
            assert "No file or patch apply was attempted" in output
            assert "3-way apply" not in output and "vcs_restore" not in output
            assert git(target, "ls-files", "--stage", "-z") == index_before
            assert (target / "text.txt").read_text() == "a\n"
        if drift == "stage_failure":  # our own staging failure undoes the text patch and file results alike
            assert (target / "text.txt").read_text() == "a\n" and not git(target, "diff", "--cached", "--name-only")
            verdict = json.loads((task_artifacts_dir(drive, "parent") / "subagent_patch_verdict_child-git.json").read_text())
            assert (verdict["outcome"], verdict["applied"]) == ("conflict", False)
            assert "nothing is half-applied" in output
        assert (target / "large.bin").read_bytes() == (b"parent change" if drift == "worktree" else b"original\x00data")
        if drift == "index":
            assert git(target, "show", ":large.bin") == b"parent change"
        assert not load_effective_task_result(drive, "child-git").get("child_result_disposition")
        return
    if surface == "self_worktree":
        assert "Integrated subagent patch" in output
        assert (target / "large.bin").read_bytes() == payload
        assert b"large.bin" in git(target, "diff", "--cached", "--name-only")
        assert (target / "text.txt").read_text() == ("b\n" if mixed else "a\n")
    else:
        assert "Verified external_workspace child" in output
        assert (source / "large.bin").read_bytes() == payload
        assert (target / "large.bin").read_bytes() == b"original\x00data"
    assert load_effective_task_result(drive, "child-git")["child_result_disposition"] == "integrated"
