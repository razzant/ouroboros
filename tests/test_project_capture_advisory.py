"""A PEM observation stays visible while Cyber allows the requested file result."""
import subprocess

import pytest

from ouroboros.project_sources import attach_snapshot_init
from ouroboros.subagent_worktrees import provision_execution_snapshot
from ouroboros.workspace_patch_capture import pem_private_key_reason, write_workspace_patch_artifacts


CONTENT = b"-----BEGIN PRIVATE KEY-----\nExample fixture, not a credential\n-----END PRIVATE KEY-----\n"


@pytest.mark.parametrize("mode", ["advanced", "cyber_pro"])
def test_a_private_copy_holds_the_pem_file_and_the_capture_keeps_the_original_finding(tmp_path, monkeypatch, mode):
    """A private copy is the source's current tree in every mode (no copy-in veto); the
    project capture still reports the PEM observation the way its mode decides."""
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: mode)
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "notes.txt").write_bytes(CONTENT)
    assert pem_private_key_reason(root, "notes.txt")
    handle = provision_execution_snapshot(target_root=root, task_id="task", snapshot_id="snapshot",
                                          worktree_root=tmp_path / "copies", data_dir=tmp_path / "data")
    from pathlib import Path
    assert (Path(handle.path) / "notes.txt").read_bytes() == CONTENT
    assert not hasattr(handle, "capture_warnings") and handle.excluded_untracked == ()
    _, manifest = write_workspace_patch_artifacts(root, tmp_path / "artifacts", task={})
    assert manifest["status"] == ("ready_with_changes" if mode == "cyber_pro" else "ready_no_changes")
    if mode == "cyber_pro":
        assert any(row.get("advisory") and row.get("path") == "notes.txt" for row in manifest["diagnostics"])


@pytest.mark.parametrize("mode", ["advanced", "cyber_pro"])
def test_explicit_attach_snapshot_reports_advisory_without_changing_bytes(tmp_path, monkeypatch, mode):
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: mode)
    root = tmp_path / "project"
    root.mkdir()
    (root / "notes.txt").write_bytes(CONTENT)
    warnings = []
    error, skipped = attach_snapshot_init(root, warnings=warnings)
    assert not error and (root / "notes.txt").read_bytes() == CONTENT
    captured = subprocess.run(["git", "cat-file", "-e", "HEAD:notes.txt"], cwd=root, capture_output=True)
    assert (captured.returncode == 0) is (mode == "cyber_pro")
    assert bool(warnings) is (mode == "cyber_pro")
    assert ("notes.txt" in skipped) is (mode == "advanced")


@pytest.mark.parametrize("mode", ["advanced", "cyber_pro"])
def test_cooperative_checkpoint_keeps_advisory_pem_fact(tmp_path, monkeypatch, mode):
    from ouroboros import coop_checkpoint
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: mode)
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "--allow-empty", "-qm", "seed"], cwd=root, check=True)
    (root / "notes.txt").write_bytes(CONTENT)
    monkeypatch.setattr(coop_checkpoint, "_task_tree_coop_roots", lambda *a: [root])
    report = coop_checkpoint.checkpoint_commit_coop_roots(tmp_path / "data", "task")[0]
    assert bool(report["committed"]) is (mode == "cyber_pro"), report
    assert bool(report.get("capture_warnings")) is (mode == "cyber_pro")
    assert (root / "notes.txt").read_bytes() == CONTENT
