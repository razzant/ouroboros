"""File-only and mixed delegated results cross the actual capture/disposition seam.

A binary or large result rides the B..C patch as a binary diff (no separate file
outputs); the copy's checkout goes once the capture is saved, and the result
commit C keeps the run's bytes until apply or reject."""

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
