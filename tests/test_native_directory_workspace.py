"""Ordinary native folders: direct results retain outputs without a Git baseline."""
import hashlib
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
from tests._workspace_git import publish_capture


def _setup(tmp_path, monkeypatch, *, status="completed"):
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
    write_task_result(drive, "child", status, parent_task_id="parent", root_task_id="parent",
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


@pytest.mark.parametrize("missing", [False, True])
def test_direct_manifest_published_before_held_output_copies(tmp_path, monkeypatch, missing):
    from pathlib import Path
    from ouroboros.artifacts import artifact_record

    workspace, drive, _, parent, output_ctx = _setup(tmp_path, monkeypatch)
    source = workspace / "answer.bin"
    source.write_bytes(b"\x00retained direct output")
    output = copy_file_to_task_artifacts(output_ctx, source, kind="process_output")
    manifest = {"schema_version": 1, "workspace_root": str(workspace), "status": "ready",
                "capture_kind": "directory_direct", "apply_state": "already_applied", "before": "unknown",
                "complete": False, "evidence_extent": "registered_outputs_only", "registered_outputs": [output]}
    path = task_artifacts_dir(drive, "child") / "workspace_patch.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    write_task_result(drive, "child", "completed", artifacts=[output, artifact_record(path, kind="workspace_patch_manifest")],
                      artifact_status="ready", artifact_finalized_at="2026-11-01T12:00:00Z")
    if missing:
        Path(output["path"]).unlink()
    result = _integrate_subagent_patch(parent, task_id="child")
    if missing:
        assert "INTEGRATE_DIRECTORY_OUTPUT_MISMATCH" in result
        verdict = json.loads((task_artifacts_dir(drive, "parent") / "subagent_patch_verdict_child.json").read_bytes())
        assert verdict["outcome"] == "direct_output_mismatch" and not verdict["applied"]
    else:
        assert "Verified 1 registered file postimage(s)" in result
    assert source.read_bytes() == b"\x00retained direct output"


@pytest.mark.parametrize("held, case", [
    (False, ""), (True, ""),
    (False, "hash"), (True, "size"), (True, "conflict"),
    pytest.param(False, "retry", id="retry-original-registration"),
    pytest.param(True, "retry", id="retry-held-registration"),
])
def test_registered_output_after_drive_retirement(tmp_path, monkeypatch, held, case):
    from pathlib import Path
    from ouroboros import cli, headless, workspace_file_outputs
    from ouroboros.artifacts import artifact_record
    from ouroboros.tools.subagent_integration import _locate_child_patch
    from tests._store_shared import add_artifact_conflict
    from tests._task_file_http import task_file_client, use_cli_http
    from tests._workspace_git import retire_published_capture

    folder, drive, task, parent, ctx = _setup(tmp_path, monkeypatch)
    damage = case if case != "retry" else ""
    drive_owner = "predecessor" if case == "retry" else "child"
    child = (headless.prepare_task_drive(drive, drive_owner, "empty") if case == "retry" and held
             else drive / "task_drives" / drive_owner)
    child.mkdir(parents=True, exist_ok=True)
    if case == "retry":
        write_task_result(drive, drive_owner, "failed", child_drive_root=str(child), retry_task_id="child")
        task.update(original_task_id=drive_owner, timeout_retry_from=drive_owner)
    task.update(parent_task_id="parent", root_task_id="parent", delegation_role="subagent")
    ctx.drive_root = child
    source = folder / "report.bin"
    source.write_bytes(b"\x00postimage")
    output = copy_file_to_task_artifacts(ctx, source, kind="process_output")
    if not held:
        # Before d19da6b09e5 the same directory writer emitted the registrations
        # themselves. Exercise its original layout without making a held copy.
        monkeypatch.setattr(workspace_file_outputs, "_held_outputs", lambda outputs, _: outputs)
        writer = headless._directory_direct_artifacts

        def direct_registrations(*args, **kwargs):
            rows, manifest = writer(*args, **kwargs)
            return [row for row in rows if row["kind"] != "workspace_registered_output"], manifest

        monkeypatch.setattr(headless, "_directory_direct_artifacts", direct_registrations)
    original, manifest = publish_capture(child, task)
    [member] = manifest["registered_outputs"]
    relative = Path(member["path"]).relative_to(original)
    assert relative == (Path("registered_outputs") / output["name"] if held else Path(output["name"]))
    original_manifest = (original / "workspace_patch.json").read_bytes()
    if case == "retry":
        # Older result fields can still describe the canonical projection, not the drive occupied by this retry.
        write_task_result(child, "child", "completed", child_drive_root=str(drive),
                          headless_child_drive_root=str(drive))
    retire_published_capture(drive, child, "child", drive_owner=drive_owner)
    store = task_artifacts_dir(drive, "child")
    assert (store / "workspace_patch.json").read_bytes() == original_manifest
    placed = store / relative
    if damage == "hash":
        placed.write_bytes(b"X" * placed.stat().st_size)
    elif damage == "size":
        placed.write_bytes(b"")
    elif damage == "conflict":
        add_artifact_conflict(drive, "child", artifact_record(placed))
    elif held:
        (store / output["name"]).write_bytes(b"mutable output changed after publication")

    _, view, _ = _locate_child_patch(parent, "child")
    with task_file_client(drive) as http:
        reader, requests = use_cli_http(monkeypatch, http)
        result = reader.request("GET", "/api/tasks/child")
        if damage:
            assert view["status"] == "failed"
            with pytest.raises(cli.PatchCLIError):
                cli._patch_bytes_from_result(reader, "child", result, strict=False)
        else:
            assert view["verified_registered_paths"] == {output["name"]: str(placed)}
            assert cli._patch_bytes_from_result(reader, "child", result, strict=False) == b""
            assert ("HEAD", f"/api/tasks/child/artifacts/{output['name']}") in requests
    message = _integrate_subagent_patch(parent, task_id="child")
    assert ("Verified 1 registered file postimage(s)" in message) == (not damage), message


def test_registered_output_waits_for_canonical_settlement(tmp_path, monkeypatch):
    from ouroboros.task_custody import settle_child_drive

    folder, drive, task, _, ctx = _setup(tmp_path, monkeypatch, status="running")
    child = drive / "task_drives" / "child"
    child.mkdir(parents=True)
    ctx.drive_root = child
    source = folder / "report.bin"
    payload = b"\x00postimage"
    source.write_bytes(payload)
    copy_file_to_task_artifacts(ctx, source, kind="process_output")
    original, manifest = publish_capture(child, task)
    manifest_path = original / "workspace_patch.json"
    before = manifest_path.read_bytes()
    assert load_task_result(child, "child")["status"] == "completed"
    assert manifest["status"] == "ready"

    outcome = settle_child_drive(drive, "child", child, live=lambda _: False)

    assert outcome == {"status": "retained", "reason": "canonical_not_settled", "published": 0}
    assert load_task_result(drive, "child")["status"] == "running"
    assert manifest_path.read_bytes() == before
    [member] = manifest["registered_outputs"]
    assert (original / "registered_outputs" / member["name"]).read_bytes() == payload
    assert not (task_artifacts_dir(drive, "child", create=False) / "workspace_patch.json").exists()


@pytest.mark.parametrize("held", [False, True], ids=["original-registration", "held-registration"])
def test_registered_output_address_must_match_its_capture(held):
    from pathlib import PurePosixPath
    from ouroboros.workspace_capture_result import _registered_address

    # Check just the reader's address binding, without a disk store or a retirement.
    manifest_path = "/fixture/recorded/workspace_patch.json"
    publication = {"held_manifest" if held else "source_manifest": manifest_path}
    files = {"registered_outputs/report.bin": "digest"} if held else {}
    with pytest.raises(ValueError, match="registered output is outside its recorded capture"):
        _registered_address({"name": "report.bin", "path": "/fixture/unbound/report.bin"},
                            PurePosixPath("."), publication, {"path": manifest_path}, files)


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
    converted = hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()  # held copies keep identities
    assert all(item["sha256"] != converted for item in manifest["registered_outputs"])
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


@pytest.mark.parametrize("changed", ["source", "artifact", "store_copy"])
def test_native_directory_refuses_changed_postimage(tmp_path, monkeypatch, changed):
    """The capture holds its own copy of each registered output (#3248): that copy changed, or the folder's file, is
    refused; a later write of the store's mutable copy changes nothing the capture published."""
    workspace, drive, task, parent, output_ctx = _setup(tmp_path, monkeypatch)
    source = workspace / "out.bin"
    source.write_bytes(b"captured\x00")
    record = copy_file_to_task_artifacts(output_ctx, source)
    finalize_task_artifacts(drive, task)
    from pathlib import Path
    [held] = json.loads((task_artifacts_dir(drive, "child") / "workspace_patch.json").read_text())["registered_outputs"]
    assert held["path"] != record["path"] and held["sha256"] == record["sha256"]
    {"source": source, "artifact": Path(held["path"]), "store_copy": Path(record["path"])}[changed].write_bytes(b"changed")
    message = _integrate_subagent_patch(parent, task_id="child")
    if changed == "store_copy":
        assert "Verified 1 registered file postimage(s)" in message, message
        return
    assert "INTEGRATE_DIRECTORY_OUTPUT_MISMATCH" in message
    assert not load_task_result(drive, "child").get("child_result_disposition")


def test_cli_and_comparison_verify_the_held_directory_output(tmp_path, monkeypatch):
    from pathlib import Path
    from ouroboros import cli
    from ouroboros.tools.subagent_integration import _compare_subagent_patches
    from tests._task_file_http import task_file_client, use_cli_http

    workspace, drive, task, parent, output_ctx = _setup(tmp_path, monkeypatch)
    source = workspace / "out.bin"
    source.write_bytes(b"recorded output")
    copy_file_to_task_artifacts(output_ctx, source)
    finalize_task_artifacts(drive, task)
    result = load_task_result(drive, "child")
    held = next(row for row in result["artifacts"] if row["kind"] == "workspace_registered_output")
    with task_file_client(drive) as http:
        reader, requests = use_cli_http(monkeypatch, http)
        public = reader.request("GET", "/api/tasks/child")
        assert cli._patch_from_result(reader, "child", public, strict=False) == ""
        assert ("HEAD", f"/api/tasks/child/artifacts/{held['name']}") in requests
        assert "patch status: ready |" in _compare_subagent_patches(parent, ["child"])
        Path(held["path"]).write_bytes(b"altered output")
        with pytest.raises(cli.PatchCLIError):
            cli._patch_from_result(reader, "child", public, strict=False)
        assert "patch status: failed" in _compare_subagent_patches(parent, ["child"])
    assert source.read_bytes() == b"recorded output"


@pytest.mark.parametrize("no_space", [False, True])
def test_native_directory_capture_fails_when_it_cannot_hold_an_output(tmp_path, monkeypatch, no_space):
    """A registered output the capture cannot hold (ENOSPC copying it into ``registered_outputs/``) has its bytes only
    in the mutable store (#3248): manifest, row and bundle say failed and name it, so neither apply nor an explicit
    acceptance takes the result. The healthy control is held, ready and verified."""
    import errno
    from pathlib import Path

    from ouroboros import artifacts

    workspace, drive, task, parent, output_ctx = _setup(tmp_path, monkeypatch)
    source = workspace / "out.bin"
    source.write_bytes(b"captured\x00")
    record = copy_file_to_task_artifacts(output_ctx, source)
    copy = artifacts.copy_artifact_file

    def copy_or_fail(src, destination, **kwargs):
        if no_space and Path(destination).parent.name == "registered_outputs":
            raise OSError(errno.ENOSPC, "No space left on device", str(destination))
        return copy(src, destination, **kwargs)

    monkeypatch.setattr(artifacts, "copy_artifact_file", copy_or_fail)
    finalize_task_artifacts(drive, task)
    manifest = json.loads((task_artifacts_dir(drive, "child") / "workspace_patch.json").read_text())
    row = load_task_result(drive, "child")
    [named] = manifest["registered_outputs"]
    statuses = (manifest["status"], row["artifact_status"], row["artifact_bundle"]["status"])
    if not no_space:
        assert statuses == ("ready",) * 3 and named["path"] != record["path"] and not row.get("artifact_error")
        assert "Verified 1 registered file postimage(s)" in _integrate_subagent_patch(parent, task_id="child")
        return
    assert statuses == ("failed",) * 3
    assert named["status"] == "missing" and named["sha256"] == record["sha256"]  # named with its identity, unheld
    assert f"registered output {named['name']}" in row["artifact_error"] and "No space left" in row["artifact_error"]
    assert row["artifact_bundle"]["errors"] == [row["artifact_error"]]
    for decision in ("apply", "accept"):
        message = _integrate_subagent_patch(parent, task_id="child", decision=decision, reason="Read the findings")
        assert "INTEGRATE_CAPTURE_FAILED" in message and f"registered output {named['name']}" in message, message
    assert not load_effective_task_result(drive, "child").get("child_result_disposition")


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


@pytest.mark.parametrize("failure,reason", [
    ("missing", "executable or working directory not found"),
    ("timeout", "timed out after 5 seconds"),
    ("nonzero", "exited 128: fatal: invalid Git metadata"),
    ("oserror", "could not start git (PermissionError, errno 13)"),
    ("incomplete", "did not report a worktree root"),
])
def test_self_worktree_provisioning_reports_external_git_probe_failure(tmp_path, monkeypatch, failure, reason):
    from ouroboros import subagent_worktrees
    from tests._workspace_git import repository

    workspace = repository(tmp_path / "external")
    ctx = SimpleNamespace(REPO_DIR=tmp_path / "system", DRIVE_ROOT=tmp_path / "data")
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "1")
    original_run = subprocess.run
    sensitive_url = "https://fixture-user:fixture-password@example.invalid/path"
    stderr = f"fatal: invalid Git metadata\n{sensitive_url}\t\x1b" + "noise " * 250
    seen = []

    def fail_probe(args, **kwargs):
        if args != ["git", "rev-parse", "--is-bare-repository", "--show-toplevel"]:
            return original_run(args, **kwargs)
        seen.append(kwargs["cwd"])
        if failure == "missing":
            raise FileNotFoundError(sensitive_url)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, kwargs["timeout"], stderr=stderr)
        if failure == "oserror":
            raise PermissionError(13, sensitive_url)
        if failure == "incomplete":
            return subprocess.CompletedProcess(args, 0, "false\n", stderr)
        return subprocess.CompletedProcess(args, 128, "", stderr)

    monkeypatch.setattr(subprocess, "run", fail_probe)
    monkeypatch.setattr(subagent_worktrees, "provision_worktree", lambda **_: pytest.fail("created a copy after refusal"))
    constraint, root, mode, detail = _resolve_subagent_constraint(
        ctx, tid="child", requested_constraint={"mode": "acting_subagent", "surface": "self_worktree"},
        workspace_root=str(workspace), workspace_mode="external", base_sha="", parent_task_id="parent",
    )
    assert seen == [str(workspace.resolve())]
    assert constraint["mode"] == "local_readonly_subagent" and (root, mode) == (str(workspace), "external")
    assert "failed to provision self_worktree" in detail and reason in detail
    assert "fixture-password" not in detail and "fixture-user" not in detail and "\x1b" not in detail
    if failure == "nonzero":
        assert "***REDACTED***" in detail and detail.endswith("…") and len(detail) < 850


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
    constraint = {"mode": "acting_subagent", "surface": surface,
                  "write_root": str(source), "base_sha": base}
    art, manifest = publish_capture(drive, {"id": "child-git", "parent_task_id": "parent", "root_task_id": "parent",
                                            "delegation_role": "subagent", "task_constraint": constraint,
                                            "workspace_root": str(source)})
    assert manifest["status"] == "ready_with_changes"
    assert manifest["file_output_changes"] and bool(manifest["patch_size"]) is mixed
    if not mixed:
        (art / "workspace.patch").unlink(missing_ok=True)
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
