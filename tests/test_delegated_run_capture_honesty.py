"""A capture claims only what it captured, and a startup GC fails closed.

Split verbatim out of ``tests/test_delegated_run_isolation.py`` by theme. This module
owns the failed manifest capture that must be disclosed rather than smoothed over, the
startup garbage collection that refuses to delete on an unreadable custody view, and the
split-drive read that must still find the capture.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.subagent_worktrees import (
    find_execution_snapshot,
    provision_execution_snapshot,
)

from tests._delegated_run_isolation_shared import (
    _HealthEnv,
    _isolated_entry,
    _nanny_ctx,
    _seed_target,
)


def _failed_manifest_capture(root, out_dir, *, task=None):
    """A ``write_workspace_patch_artifacts`` stand-in whose MANIFEST itself reports
    failure — the real function writes exactly this shape when its internal diff
    errors are RECORDED rather than raised (headless: ``if errors: status=failed``)."""
    manifest = {"schema_version": 1, "status": "failed", "sha256": "", "diffstat": "",
                "patch_size": 0,
                "errors": [{"type": "git_error", "message": "diff exploded"}]}
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "workspace_patch.json").write_text(json.dumps(manifest), encoding="utf-8")
    return [], manifest


class TestCaptureHonesty:
    """C1-R3: ``patch_captured`` MEANS "a usable patch artifact exists".

    A manifest whose own status is failed must not mint the PATCH_CAPTURED row
    (the idempotent early return would then serve the failed manifest forever),
    a reject must never release the snapshot over a non-usable capture (that
    destroys the child's only copy with nothing captured), and an exception
    ESCAPING the capture core at disposition is the same typed refusal — never
    a raw traceback out of the tool."""

    def _settled_run(self, tmp_path, monkeypatch, *, snapshot_id, run_id):
        """A settled mutating run with real durable rows (no daemon involved)."""
        target = _seed_target(tmp_path)
        ctx = _nanny_ctx(tmp_path, target, monkeypatch)
        handle = provision_execution_snapshot(
            target_root=target, task_id="t-nanny", snapshot_id=snapshot_id)
        (pathlib.Path(handle.path) / "tracked.txt").write_text(
            "one\ntwo\nCHILD-EDIT\n", encoding="utf-8")
        drive = custody.custody_root(ctx)
        entry = custody.RunCustody(
            run_id=run_id, task_id="t-nanny", route_id="some-route",
            snapshot_id=handle.snapshot_id, execution_root=handle.path,
            baseline_sha=handle.baseline_sha, target_root=str(target),
            authority_source="external_workspace_root")
        assert custody.record_started(drive, entry)
        custody.emit(drive, custody.SETTLED, {"run_id": run_id, "task_id": "t-nanny"})
        custody._CUSTODY.clear()
        return target, ctx, handle, drive

    def test_failed_status_manifest_never_mints_patch_captured(self, tmp_path, monkeypatch):
        # (a) The core wrote PATCH_CAPTURED unconditionally after
        # write_workspace_patch_artifacts returned — including over a manifest
        # whose own status is "failed". The row must stay uncaptured, both
        # dispositions must refuse typed, and the snapshot must persist.
        from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive
        from ouroboros.tools.subagent_integration import _integrate_delegated_patch

        target, ctx, handle, drive = self._settled_run(
            tmp_path, monkeypatch, snapshot_id="snapHon", run_id="run-hon")
        monkeypatch.setattr("ouroboros.headless.write_workspace_patch_artifacts",
                            _failed_manifest_capture)
        entry = custody.replay(drive)["run-hon"]
        block = capture_terminal_patch_for_drive(drive, entry)
        assert block["status"] == "failed"
        assert entry.patch_captured is False
        custody._CUSTODY.clear()
        assert custody.replay(drive)["run-hon"].patch_captured is False
        events = (drive / "logs" / "events.jsonl").read_text(encoding="utf-8")
        assert custody.PATCH_CAPTURED not in events
        # Both dispositions are the typed refusal; nothing is disposed.
        for decision in ("apply", "reject"):
            out = _integrate_delegated_patch(ctx, "run-hon", decision, "")
            assert "INTEGRATE_DELEGATED_CAPTURE_FAILED" in out, (decision, out)
            custody._CUSTODY.clear()
        events = (drive / "logs" / "events.jsonl").read_text(encoding="utf-8")
        assert custody.PATCH_DISPOSED not in events
        replayed = custody.replay(drive)["run-hon"]
        assert replayed.patch_disposed == "" and replayed.patch_captured is False
        assert find_execution_snapshot("snapHon") is not None
        assert pathlib.Path(handle.path).exists()
        # The shared tree was never touched.
        assert (target / "tracked.txt").read_text(encoding="utf-8") == "one\ntwo\n"
        # The health line keys on the honest flag: preserved, never "captured".
        from ouroboros.context_health import build_health_invariants

        surface = build_health_invariants(_HealthEnv(drive))
        assert "DELEGATED PATCH AWAITS DISPOSITION" in surface
        assert "changes captured" not in surface
        custody._CUSTODY.clear()

    def test_raising_capture_core_at_disposition_is_typed_not_a_traceback(self, tmp_path, monkeypatch):
        # (b) An exception ESCAPING the core (its internal try covers only the
        # diff itself — mkdir/custody failures propagate) used to leave the tool
        # as a raw RuntimeError. Both decisions must answer the typed refusal.
        from ouroboros.tools.subagent_integration import _integrate_delegated_patch

        target, ctx, handle, drive = self._settled_run(
            tmp_path, monkeypatch, snapshot_id="snapRaise", run_id="run-raise")

        def _exploding_core(*_a, **_kw):
            raise OSError("cap_dir mkdir blew up")

        monkeypatch.setattr(
            "ouroboros.tools.delegate_integration.capture_terminal_patch_for_drive",
            _exploding_core)
        for decision in ("apply", "reject"):
            out = _integrate_delegated_patch(ctx, "run-raise", decision, "")
            assert "INTEGRATE_DELEGATED_CAPTURE_FAILED" in out, (decision, out)
            assert "OSError" in out
            custody._CUSTODY.clear()
        replayed = custody.replay(drive)["run-raise"]
        assert replayed.patch_disposed == ""
        assert find_execution_snapshot("snapRaise") is not None
        assert pathlib.Path(handle.path).exists()
        custody._CUSTODY.clear()

    def test_reject_over_a_pre_fix_failed_capture_row_refuses_and_preserves(self, tmp_path, monkeypatch):
        # (c) A row written by pre-R3 code: PATCH_CAPTURED durable although the
        # manifest on disk says failed. patch_captured=True must not be trusted
        # over the manifest's own status — the reject used to release the
        # snapshot (the child's only copy) with nothing captured.
        from ouroboros.tools.subagent_integration import _integrate_delegated_patch

        target, ctx, handle, drive = self._settled_run(
            tmp_path, monkeypatch, snapshot_id="snapOld", run_id="run-old")
        entry = custody.replay(drive)["run-old"]
        cap_dir = custody.delegated_capture_dir(drive, "t-nanny", "snapOld")
        _failed_manifest_capture(handle.path, cap_dir)
        assert custody.record_patch_captured(drive, entry, status="failed")
        custody._CUSTODY.clear()
        assert custody.replay(drive)["run-old"].patch_captured is True  # poisoned row
        # The honest path re-captures instead of trusting the row; the diff
        # machinery is still broken, so the retry also yields a failed manifest.
        monkeypatch.setattr("ouroboros.headless.write_workspace_patch_artifacts",
                            _failed_manifest_capture)
        out = _integrate_delegated_patch(ctx, "run-old", "reject", "discard it")
        assert "INTEGRATE_DELEGATED_CAPTURE_FAILED" in out, out
        custody._CUSTODY.clear()
        assert custody.replay(drive)["run-old"].patch_disposed == ""
        assert find_execution_snapshot("snapOld") is not None
        assert pathlib.Path(handle.path).exists()
        custody._CUSTODY.clear()

    def test_reject_branch_itself_requires_a_ready_manifest(self, tmp_path, monkeypatch):
        # (c, belt) Even when the capture-at-disposition seam answers "usable",
        # the reject branch re-checks the manifest before releasing the
        # snapshot — the one decision that destroys the only copy.
        import ouroboros.tools.subagent_integration as si

        target, ctx, handle, drive = self._settled_run(
            tmp_path, monkeypatch, snapshot_id="snapBelt", run_id="run-belt")
        cap_dir = custody.delegated_capture_dir(drive, "t-nanny", "snapBelt")
        _failed_manifest_capture(handle.path, cap_dir)
        monkeypatch.setattr(si, "_capture_at_disposition", lambda *a, **k: "")
        out = si._integrate_delegated_patch(ctx, "run-belt", "reject", "")
        assert "INTEGRATE_DELEGATED_CAPTURE_FAILED" in out, out
        custody._CUSTODY.clear()
        assert custody.replay(drive)["run-belt"].patch_disposed == ""
        assert pathlib.Path(handle.path).exists()
        custody._CUSTODY.clear()

    def test_ready_no_changes_reject_still_releases_the_snapshot(self, tmp_path, monkeypatch):
        # (d) regression pin: rejecting a READY_NO_CHANGES capture is legitimate
        # (nothing to lose) and must keep releasing the snapshot.
        from ouroboros.tools.delegate import _capture_terminal_patch
        from ouroboros.tools.subagent_integration import _integrate_delegated_patch

        target = _seed_target(tmp_path)
        ctx = _nanny_ctx(tmp_path, target, monkeypatch)
        handle = provision_execution_snapshot(
            target_root=target, task_id="t-nanny", snapshot_id="snapNC")
        custody._CUSTODY.clear()
        entry = _isolated_entry(ctx, target, handle, run_id="run-nc")
        capture = _capture_terminal_patch(ctx, entry)
        assert capture["status"] == "ready_no_changes"
        assert entry.patch_captured is True  # ready_no_changes IS a usable capture
        out = _integrate_delegated_patch(ctx, "run-nc", "reject", "nothing to keep")
        assert "🚫 Rejected" in out, out
        assert entry.patch_disposed == "rejected"
        assert find_execution_snapshot("snapNC") is None
        custody._CUSTODY.clear()


class TestStartupGCFailClosed:
    """CR1-1: the startup GC must not destroy open snapshots when the custody
    log is unreadable. `_iter_rows` swallows OSError (right for the fail-soft
    readers), so an unreadable log replayed as "no open runs", the keep-set
    went empty, and `prune_execution_snapshots` deleted live, never-captured
    work. GC deletes only over PROVEN settled && patch_disposed; UNKNOWN
    custody state skips the destructive prune and says so loudly."""

    def _server_gc(self, tmp_path, monkeypatch):
        data, snaps = tmp_path / "data", tmp_path / "snaps"
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
        monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(snaps))
        import server as srv

        from ouroboros import server_maintenance
        # The prune reads its drive root from its owner module.
        monkeypatch.setattr(server_maintenance, "DATA_DIR", data)
        return srv, data, snaps

    @pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0,
                        reason="POSIX permission-bit semantics; skipped on Windows and under root")
    def test_unreadable_custody_log_skips_the_prune_and_discloses(self, tmp_path, monkeypatch):
        srv, data, snaps = self._server_gc(tmp_path, monkeypatch)
        target = _seed_target(tmp_path)
        handle = provision_execution_snapshot(
            target_root=target, task_id="t-gc", snapshot_id="gc-open",
            worktree_root=snaps, data_dir=data)
        entry = custody.RunCustody(run_id="run-gc", task_id="t-gc",
                                   snapshot_id="gc-open", execution_root=handle.path)
        assert custody.record_started(data, entry)
        custody._CUSTODY.clear()
        assert "gc-open" in custody.open_snapshot_ids(data)

        events = data / "logs" / "events.jsonl"
        # Write-only: the log EXISTS but cannot be READ — replay would answer {}.
        events.chmod(0o200)
        try:
            assert custody.custody_log_unreadable(data)
            srv._prune_delegated_snapshots()
        finally:
            events.chmod(0o644)
        # The open snapshot SURVIVED, registry row included.
        assert find_execution_snapshot("gc-open", data_dir=data) is not None
        assert pathlib.Path(handle.path).exists()
        # And the skip is a loud durable row, not a silent no-op.
        rows = [json.loads(line) for line in
                events.read_text(encoding="utf-8").splitlines()
                if '"delegated_snapshot_prune_skipped"' in line]
        assert rows and rows[-1]["reason"] == "custody_log_unreadable"
        custody._CUSTODY.clear()

    @pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0,
                        reason="POSIX permission-bit semantics; skipped on Windows and under root")
    def test_unwritable_skip_row_escalates_to_error_and_still_skips(
            self, tmp_path, monkeypatch, caplog):
        # CR2-2: a COMPLETELY inaccessible custody log (mode 000) still skips
        # the prune (snapshot safe), but the promised durable
        # delegated_snapshot_prune_skipped row cannot land — that failure must
        # be an ERROR-level disclosure, not a silently ignored return value.
        srv, data, snaps = self._server_gc(tmp_path, monkeypatch)
        target = _seed_target(tmp_path)
        handle = provision_execution_snapshot(
            target_root=target, task_id="t-gc3", snapshot_id="gc-open3",
            worktree_root=snaps, data_dir=data)
        entry = custody.RunCustody(run_id="run-gc3", task_id="t-gc3",
                                   snapshot_id="gc-open3", execution_root=handle.path)
        assert custody.record_started(data, entry)
        custody._CUSTODY.clear()
        events = data / "logs" / "events.jsonl"
        events.chmod(0o000)  # unreadable AND unwritable
        try:
            with caplog.at_level(logging.ERROR, logger="server"):
                srv._prune_delegated_snapshots()
        finally:
            events.chmod(0o644)
        # The skip itself still protects the open snapshot.
        assert find_execution_snapshot("gc-open3", data_dir=data) is not None
        assert pathlib.Path(handle.path).exists()
        # And the unwritable durable row is escalated loudly.
        assert any(
            record.levelno == logging.ERROR
            and "delegated_snapshot_prune_skipped" in record.getMessage()
            for record in caplog.records), caplog.records
        custody._CUSTODY.clear()

    def test_readable_log_still_prunes_closed_snapshots(self, tmp_path, monkeypatch):
        srv, data, snaps = self._server_gc(tmp_path, monkeypatch)
        target = _seed_target(tmp_path)
        provision_execution_snapshot(
            target_root=target, task_id="t-gc2", snapshot_id="gc-done",
            worktree_root=snaps, data_dir=data)
        done = custody.RunCustody(run_id="run-done", task_id="t-gc2",
                                  snapshot_id="gc-done", execution_root="/x")
        custody.record_started(data, done)
        custody.emit(data, custody.SETTLED, {"run_id": "run-done", "task_id": "t-gc2"})
        custody.record_patch_disposed(data, done, disposition="applied")
        custody._CUSTODY.clear()
        srv._prune_delegated_snapshots()
        assert find_execution_snapshot("gc-done", data_dir=data) is None
        custody._CUSTODY.clear()


class TestSplitDriveCaptureRead:
    """CR1-2: the capture artifact must be READABLE by a split-drive nanny.

    Capture writes under the CANONICAL (budget) drive (`custody_root` — right
    for durability), but `artifact_store` resolves from the CHILD's drive_root,
    so the owning forked task got NOT_FOUND for its own patch/manifest and
    could only dispose blindly. Reads of the `delegated_runs/` prefix are now
    anchored to the canonical root for the owning task."""

    def _split_ctx(self, tmp_path, target, monkeypatch):
        ctx = _nanny_ctx(tmp_path, target, monkeypatch)
        canonical = tmp_path / "canonical"
        canonical.mkdir(exist_ok=True)
        # drive_root (child) differs from the canonical/budget root.
        ctx.task_metadata = {"budget_drive_root": str(canonical)}
        return ctx, canonical

    def test_capture_reads_through_the_tool_surface_across_drives(self, tmp_path, monkeypatch):
        from ouroboros.tools.core_file_tools import _read_file
        from ouroboros.tools.delegate import _capture_terminal_patch

        target = _seed_target(tmp_path)
        ctx, canonical = self._split_ctx(tmp_path, target, monkeypatch)
        handle = provision_execution_snapshot(
            target_root=target, task_id="t-nanny", snapshot_id="snapSplit")
        (pathlib.Path(handle.path) / "tracked.txt").write_text(
            "one\ntwo\nCHILD-EDIT\n", encoding="utf-8")
        custody._CUSTODY.clear()
        entry = _isolated_entry(ctx, target, handle, run_id="run-split")
        block = _capture_terminal_patch(ctx, entry)
        assert block["status"] == "ready_with_changes", block
        # The capture landed on the CANONICAL drive, not the child drive.
        assert pathlib.Path(block["patch_artifact"]).is_relative_to(canonical)
        # The block hands a tool-surface read handle the owning ctx can use.
        read_handle = block["patch_read"]
        assert read_handle["root"] == "artifact_store"
        assert read_handle["path"].startswith("delegated_runs/")
        out = _read_file(ctx, read_handle["path"], root="artifact_store")
        assert "NOT_FOUND" not in out, out
        assert "CHILD-EDIT" in out
        manifest_out = _read_file(ctx, block["manifest_read"]["path"], root="artifact_store")
        assert "ready_with_changes" in manifest_out
        custody._CUSTODY.clear()

    def test_ordinary_single_drive_reads_are_unchanged(self, tmp_path, monkeypatch):
        from ouroboros.tools.core_file_tools import _read_file
        from ouroboros.tools.delegate import _capture_terminal_patch

        target = _seed_target(tmp_path)
        ctx = _nanny_ctx(tmp_path, target, monkeypatch)  # drive_root == canonical
        handle = provision_execution_snapshot(
            target_root=target, task_id="t-nanny", snapshot_id="snapOne")
        (pathlib.Path(handle.path) / "tracked.txt").write_text(
            "one\ntwo\nCHILD-EDIT\n", encoding="utf-8")
        custody._CUSTODY.clear()
        entry = _isolated_entry(ctx, target, handle, run_id="run-one")
        block = _capture_terminal_patch(ctx, entry)
        out = _read_file(ctx, block["patch_read"]["path"], root="artifact_store")
        assert "CHILD-EDIT" in out
        custody._CUSTODY.clear()
