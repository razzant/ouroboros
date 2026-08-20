"""C1 delegated-run isolation: private execution snapshots, capture and integration.

This module owns the snapshot a delegated run is provisioned with, its removal and
garbage collection, the terminal capture and the explicit integration of that capture,
the protected-path scope and sensitive veto around it, the legacy retry, and the durable
binding the integration leaves.

Phase C of the poltergeist sprint (owner 3=A: isolate ONLY delegated runs).

Orphan reconciliation and lazy capture, capture honesty with the fail-closed startup GC,
and the apply-intent ambiguity were split verbatim into
``tests/test_delegated_run_reconciliation_capture.py``,
``tests/test_delegated_run_capture_honesty.py`` and
``tests/test_delegated_run_apply_intent.py``; the repository, context and gateway
builders they share live in ``tests/_delegated_run_isolation_shared.py``.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.subagent_worktrees import (
    find_execution_snapshot,
    provision_execution_snapshot,
    prune_execution_snapshots,
    prune_orphans,
    remove_execution_snapshot,
)

from tests._delegated_run_isolation_shared import (
    _git,
    _isolated_entry,
    _nanny_ctx,
    _seed_target,
)


class TestProvision:
    def test_snapshot_captures_the_real_tree_and_vetoes_sensitive(self, tmp_path):
        target = _seed_target(tmp_path)
        snaps, data = tmp_path / "snaps", tmp_path / "data"
        before_status = _git(target, "status", "--porcelain").stdout
        handle = provision_execution_snapshot(
            target_root=target, task_id="t1", snapshot_id="snapA",
            worktree_root=snaps, data_dir=data)
        exec_root = pathlib.Path(handle.path)
        # The REAL current tree: tracked+staged+eligible untracked, at their
        # on-disk content (not HEAD's).
        assert (exec_root / "tracked.txt").read_text(encoding="utf-8") == "one\ntwo\n"
        assert (exec_root / "staged.txt").read_text(encoding="utf-8") == "staged\n"
        assert (exec_root / "untracked.txt").read_text(encoding="utf-8") == "loose\n"
        # The SAME sensitive veto patch capture applies: .env never rides.
        assert not (exec_root / ".env").exists()
        assert any(e.get("path") == ".env" for e in handle.excluded_untracked)
        # The target itself is untouched: same status, same HEAD, staged still staged.
        assert _git(target, "status", "--porcelain").stdout == before_status
        # The baseline is pinned by a protected ref and matches the handle.
        assert _git(target, "rev-parse", handle.baseline_ref).stdout.strip() == handle.baseline_sha
        assert handle.manifest_digest and handle.baseline_tree
        # Registered durably with its kind, BEFORE any start intent exists.
        entry = find_execution_snapshot("snapA", data_dir=data)
        assert entry is not None and entry["kind"] == "delegated_exec"
        assert entry["target_root"] == str(target.resolve())

    def test_unborn_target_snapshots_without_a_parent(self, tmp_path):
        target = tmp_path / "fresh"
        target.mkdir()
        _git(target, "init")
        (target / "only.txt").write_text("x\n", encoding="utf-8")
        handle = provision_execution_snapshot(
            target_root=target, task_id="t1", snapshot_id="snapU",
            worktree_root=tmp_path / "snaps", data_dir=tmp_path / "data")
        assert (pathlib.Path(handle.path) / "only.txt").exists()
        assert handle.target_head == ""  # unborn: baseline commit has no parent

    def test_non_git_target_is_refused(self, tmp_path):
        plain = tmp_path / "plain"
        plain.mkdir()
        with pytest.raises(ValueError, match="not a git working tree"):
            provision_execution_snapshot(
                target_root=plain, task_id="t1", snapshot_id="s",
                worktree_root=tmp_path / "snaps", data_dir=tmp_path / "data")


class TestRemoveAndGC:
    def test_remove_tears_down_worktree_ref_and_registry(self, tmp_path):
        target = _seed_target(tmp_path)
        snaps, data = tmp_path / "snaps", tmp_path / "data"
        handle = provision_execution_snapshot(
            target_root=target, task_id="t1", snapshot_id="snapR",
            worktree_root=snaps, data_dir=data)
        assert remove_execution_snapshot("snapR", worktree_root=snaps, data_dir=data)
        assert not pathlib.Path(handle.path).exists()
        assert find_execution_snapshot("snapR", data_dir=data) is None
        assert _git(target, "rev-parse", handle.baseline_ref, check=False).returncode != 0

    def test_gc_keeps_open_snapshots_and_removes_closed_ones(self, tmp_path):
        target = _seed_target(tmp_path)
        snaps, data = tmp_path / "snaps", tmp_path / "data"
        provision_execution_snapshot(target_root=target, task_id="t1",
                                     snapshot_id="open1", worktree_root=snaps, data_dir=data)
        provision_execution_snapshot(target_root=target, task_id="t1",
                                     snapshot_id="closed1", worktree_root=snaps, data_dir=data)
        report = prune_execution_snapshots({"open1"}, worktree_root=snaps, data_dir=data)
        assert report["kept"] == ["open1"] and report["removed"] == ["closed1"]
        assert find_execution_snapshot("open1", data_dir=data) is not None
        assert find_execution_snapshot("closed1", data_dir=data) is None

    def test_age_prune_never_eats_delegated_snapshots(self, tmp_path):
        # prune_orphans (retention/path heuristics) must skip delegated rows:
        # their lifecycle is custody-owned, and this loop cannot delete their ref.
        target = _seed_target(tmp_path)
        snaps, data = tmp_path / "snaps", tmp_path / "data"
        provision_execution_snapshot(target_root=target, task_id="t1",
                                     snapshot_id="aged", worktree_root=snaps, data_dir=data)
        report = prune_orphans(worktree_root=snaps, data_dir=data, retention_days=0)
        assert find_execution_snapshot("aged", data_dir=data) is not None, report


class TestCaptureAndIntegrate:
    def _provisioned(self, tmp_path, monkeypatch):
        target = _seed_target(tmp_path)
        ctx = _nanny_ctx(tmp_path, target, monkeypatch)
        handle = provision_execution_snapshot(
            target_root=target, task_id="t-nanny", snapshot_id="snapX")
        custody._CUSTODY.clear()
        return target, ctx, handle

    def test_capture_then_explicit_apply_stages_into_the_target(self, tmp_path, monkeypatch):
        from ouroboros.tools.delegate import _capture_terminal_patch
        from ouroboros.tools.subagent_integration import _integrate_delegated_patch

        target, ctx, handle = self._provisioned(tmp_path, monkeypatch)
        exec_root = pathlib.Path(handle.path)
        # The "harness" edits the SNAPSHOT only.
        (exec_root / "newfile.py").write_text("print('hi')\n", encoding="utf-8")
        (exec_root / "tracked.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
        entry = _isolated_entry(ctx, target, handle)
        capture = _capture_terminal_patch(ctx, entry)
        assert capture["status"] == "ready_with_changes", capture
        assert capture["authority_target_root"] == str(target)
        assert capture["baseline_id"] == handle.baseline_sha
        patch_path = pathlib.Path(capture["patch_artifact"])
        assert patch_path.exists()
        assert entry.patch_captured is True
        # NOTHING landed in the target yet.
        assert not (target / "newfile.py").exists()
        # Idempotent: a re-wait re-reads the capture, not a re-diff.
        again = _capture_terminal_patch(ctx, entry)
        assert again["patch_artifact"] == capture["patch_artifact"]

        out = _integrate_delegated_patch(ctx, "run-1", "apply", "looks good")
        assert "✅ Integrated" in out, out
        assert (target / "newfile.py").read_text(encoding="utf-8") == "print('hi')\n"
        assert (target / "tracked.txt").read_text(encoding="utf-8") == "one\ntwo\nthree\n"
        staged = _git(target, "diff", "--cached", "--name-only").stdout
        assert "newfile.py" in staged
        assert entry.patch_disposed == "applied"
        # Disposition released the snapshot (worktree + registry + ref).
        assert find_execution_snapshot("snapX") is None
        assert not exec_root.exists()
        custody._CUSTODY.clear()

    def test_reject_discards_without_touching_the_target(self, tmp_path, monkeypatch):
        from ouroboros.tools.delegate import _capture_terminal_patch
        from ouroboros.tools.subagent_integration import _integrate_delegated_patch

        target, ctx, handle = self._provisioned(tmp_path, monkeypatch)
        (pathlib.Path(handle.path) / "junk.py").write_text("bad\n", encoding="utf-8")
        entry = _isolated_entry(ctx, target, handle)
        _capture_terminal_patch(ctx, entry)
        out = _integrate_delegated_patch(ctx, "run-1", "reject", "not wanted")
        assert "🚫 Rejected" in out, out
        assert not (target / "junk.py").exists()
        assert entry.patch_disposed == "rejected"
        assert find_execution_snapshot("snapX") is None
        custody._CUSTODY.clear()

    def test_conflict_preserves_snapshot_and_patch_for_the_nanny(self, tmp_path, monkeypatch):
        from ouroboros.tools.delegate import _capture_terminal_patch
        from ouroboros.tools.subagent_integration import _integrate_delegated_patch

        target, ctx, handle = self._provisioned(tmp_path, monkeypatch)
        exec_root = pathlib.Path(handle.path)
        (exec_root / "tracked.txt").write_text("SNAPSHOT-EDIT\n", encoding="utf-8")
        entry = _isolated_entry(ctx, target, handle)
        _capture_terminal_patch(ctx, entry)
        # The target moved DIFFERENTLY on the same line after the snapshot.
        (target / "tracked.txt").write_text("TARGET-EDIT\n", encoding="utf-8")
        out = _integrate_delegated_patch(ctx, "run-1", "apply", "")
        assert "INTEGRATE_CONFLICT" in out, out
        assert "YOU own this conflict" in out
        # Conflict material persists: no disposition, snapshot + patch intact.
        assert entry.patch_disposed == ""
        assert find_execution_snapshot("snapX") is not None
        assert exec_root.exists()
        custody._CUSTODY.clear()

    def test_no_changes_run_disposes_cleanly(self, tmp_path, monkeypatch):
        from ouroboros.tools.delegate import _capture_terminal_patch
        from ouroboros.tools.subagent_integration import _integrate_delegated_patch

        target, ctx, handle = self._provisioned(tmp_path, monkeypatch)
        entry = _isolated_entry(ctx, target, handle)
        capture = _capture_terminal_patch(ctx, entry)
        assert capture["status"] == "ready_no_changes"
        out = _integrate_delegated_patch(ctx, "run-1", "apply", "")
        assert "changed NOTHING" in out, out
        assert entry.patch_disposed == "applied"
        assert find_execution_snapshot("snapX") is None
        custody._CUSTODY.clear()

    def test_unsettled_run_cannot_be_integrated(self, tmp_path, monkeypatch):
        from ouroboros.tools.subagent_integration import _integrate_delegated_patch

        target, ctx, handle = self._provisioned(tmp_path, monkeypatch)
        _isolated_entry(ctx, target, handle, settled=False)
        out = _integrate_delegated_patch(ctx, "run-1", "apply", "")
        assert "INTEGRATE_DELEGATED_NOT_TERMINAL" in out
        custody._CUSTODY.clear()

    def test_deleting_an_untracked_file_applies_and_stages_cleanly(self, tmp_path, monkeypatch):
        # F2: the baseline carries eligible UNTRACKED files as tree entries, so a
        # run deleting one produces a deletion the target cannot stage (nothing on
        # disk, nothing in the index). Naming it in the pathspec made `git add`
        # exit non-zero AFTER a successful apply, and that was reported as
        # "the tree moved" — inviting a retry over an already-mutated tree.
        from ouroboros.tools.delegate import _capture_terminal_patch
        from ouroboros.tools.subagent_integration import _integrate_delegated_patch

        target, ctx, handle = self._provisioned(tmp_path, monkeypatch)
        (pathlib.Path(handle.path) / "untracked.txt").unlink()
        entry = _isolated_entry(ctx, target, handle)
        assert _capture_terminal_patch(ctx, entry)["status"] == "ready_with_changes"
        out = _integrate_delegated_patch(ctx, "run-1", "apply", "drop the scratch file")
        assert "✅ Integrated" in out, out
        assert not (target / "untracked.txt").exists()
        assert entry.patch_disposed == "applied"
        custody._CUSTODY.clear()

    def test_quoted_and_renamed_paths_survive_the_pathspec(self, tmp_path, monkeypatch):
        # F2c: `git apply --numstat -z` is the only reader that does not munge
        # pathnames; the old tab-split + `diff --git` regex corrupted a quoted or
        # renamed path, and the corrupted name then reached the protected-path
        # gate and the staging step.
        from ouroboros.tools.delegate import _capture_terminal_patch
        from ouroboros.tools.subagent_integration import (
            _integrate_delegated_patch, _patch_touched_paths,
        )

        target, ctx, handle = self._provisioned(tmp_path, monkeypatch)
        exec_root = pathlib.Path(handle.path)
        # NTFS forbids '"' in filenames (Errno 22), so Windows exercises the
        # same git-quoting path with a legal odd name: spaces + an apostrophe
        # still force quoted pathspecs through the capture/apply cycle.
        odd = "we'ird na me.txt" if os.name == "nt" else 'we"ird na me.txt'
        (exec_root / odd).write_text("odd\n", encoding="utf-8")
        # A pure rename of a tracked file (content preserved -> git reports it as
        # a rename, the empty-path + two-fields shape of --numstat -z).
        _git(exec_root, "mv", "staged.txt", "moved.txt")
        entry = _isolated_entry(ctx, target, handle)
        capture = _capture_terminal_patch(ctx, entry)
        assert capture["status"] == "ready_with_changes", capture
        touched, err = _patch_touched_paths(pathlib.Path(capture["patch_artifact"]), target)
        assert err == ""
        assert odd in touched
        assert {"staged.txt", "moved.txt"} <= touched
        out = _integrate_delegated_patch(ctx, "run-1", "apply", "")
        assert "✅ Integrated" in out, out
        assert (target / odd).read_text(encoding="utf-8") == "odd\n"
        assert (target / "moved.txt").exists() and not (target / "staged.txt").exists()
        staged = _git(target, "diff", "--cached", "--name-only").stdout
        assert "moved.txt" in staged
        custody._CUSTODY.clear()

    def test_drift_at_an_offset_is_a_typed_conflict_not_a_shifted_apply(self, tmp_path, monkeypatch):
        # F4: a plain `git apply` RELOCATES hunks by offset, so a target that moved
        # since the snapshot still takes the patch — silently, at a shifted
        # position. Drift must be proven against the baseline instead.
        from ouroboros.tools.delegate import _capture_terminal_patch
        from ouroboros.tools.subagent_integration import _integrate_delegated_patch

        target = _seed_target(tmp_path)
        body = [f"line{i}\n" for i in range(1, 21)]
        (target / "big.txt").write_text("".join(body), encoding="utf-8")
        ctx = _nanny_ctx(tmp_path, target, monkeypatch)
        handle = provision_execution_snapshot(
            target_root=target, task_id="t-nanny", snapshot_id="snapX")
        custody._CUSTODY.clear()
        edited = list(body)
        edited[9] = "line10-EDITED\n"
        (pathlib.Path(handle.path) / "big.txt").write_text("".join(edited), encoding="utf-8")
        entry = _isolated_entry(ctx, target, handle)
        capture = _capture_terminal_patch(ctx, entry)
        # The target gains a PREPENDED line: the hunk still applies, just offset.
        (target / "big.txt").write_text("line0\n" + "".join(body), encoding="utf-8")
        assert _git(target, "apply", "--check", capture["patch_artifact"],
                    check=False).returncode == 0, "the patch must still apply — that is the class"
        out = _integrate_delegated_patch(ctx, "run-1", "apply", "")
        assert "INTEGRATE_CONFLICT" in out and "big.txt" in out, out
        assert "YOU own this conflict" in out
        # Nothing applied, nothing disposed, material preserved.
        assert (target / "big.txt").read_text(encoding="utf-8").startswith("line0\nline1\n")
        assert "line10-EDITED" not in (target / "big.txt").read_text(encoding="utf-8")
        assert entry.patch_disposed == ""
        assert find_execution_snapshot("snapX") is not None
        custody._CUSTODY.clear()

    def test_unwritten_disposition_keeps_the_snapshot_and_forbids_retry(self, tmp_path, monkeypatch):
        # F1: releasing the snapshot on an UNWRITTEN disposition row loses the only
        # durable record that the patch was handled — after a restart (no
        # in-process flag) the same patch could be applied a second time.
        # (CR1-3: only the DISPOSITION write fails here — a log broken before the
        # apply now refuses to mutate at the apply-intent row instead.)
        from ouroboros.tools.delegate import _capture_terminal_patch
        from ouroboros.tools.subagent_integration import _integrate_delegated_patch

        target, ctx, handle = self._provisioned(tmp_path, monkeypatch)
        (pathlib.Path(handle.path) / "newfile.py").write_text("x = 1\n", encoding="utf-8")
        entry = _isolated_entry(ctx, target, handle)
        _capture_terminal_patch(ctx, entry)
        real_emit = custody.emit
        monkeypatch.setattr(
            custody, "emit",
            lambda drive, kind, payload: (
                False if kind == custody.PATCH_DISPOSED else real_emit(drive, kind, payload)))
        out = _integrate_delegated_patch(ctx, "run-1", "apply", "")
        assert "INTEGRATE_DISPOSITION_UNWRITTEN" in out, out
        assert "Do NOT call integrate_delegated_patch again" in out
        # The apply DID happen and is not denied; the snapshot survives for it.
        assert (target / "newfile.py").exists()
        assert find_execution_snapshot("snapX") is not None
        assert pathlib.Path(handle.path).exists()
        custody._CUSTODY.clear()

    def test_unwritten_disposition_on_reject_changes_nothing(self, tmp_path, monkeypatch):
        from ouroboros.tools.delegate import _capture_terminal_patch
        from ouroboros.tools.subagent_integration import _integrate_delegated_patch

        target, ctx, handle = self._provisioned(tmp_path, monkeypatch)
        (pathlib.Path(handle.path) / "junk.py").write_text("bad\n", encoding="utf-8")
        entry = _isolated_entry(ctx, target, handle)
        _capture_terminal_patch(ctx, entry)
        monkeypatch.setattr(custody, "emit", lambda *a, **k: False)
        out = _integrate_delegated_patch(ctx, "run-1", "reject", "no")
        assert "INTEGRATE_DISPOSITION_UNWRITTEN" in out, out
        assert "only after a restart" in out
        assert not (target / "junk.py").exists()
        assert find_execution_snapshot("snapX") is not None
        custody._CUSTODY.clear()


class TestProtectedPathScope:
    """F3: the Ouroboros protected-path policy governs the OUROBOROS body only."""

    def _ready(self, tmp_path, monkeypatch, rel_path):
        from ouroboros.tools.delegate import _capture_terminal_patch

        target = _seed_target(tmp_path)
        ctx = _nanny_ctx(tmp_path, target, monkeypatch)
        handle = provision_execution_snapshot(
            target_root=target, task_id="t-nanny", snapshot_id="snapP")
        custody._CUSTODY.clear()
        path = pathlib.Path(handle.path) / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("name: ci\n", encoding="utf-8")
        entry = _isolated_entry(ctx, target, handle)
        assert _capture_terminal_patch(ctx, entry)["status"] == "ready_with_changes"
        return target, ctx, handle

    def test_foreign_workspace_ci_yml_integrates_in_advanced_mode(self, tmp_path, monkeypatch):
        import ouroboros.tools.subagent_integration as si

        target, ctx, _ = self._ready(tmp_path, monkeypatch, ".github/workflows/ci.yml")
        monkeypatch.setattr(si, "get_runtime_mode", lambda: "advanced")
        out = si._integrate_delegated_patch(ctx, "run-1", "apply", "")
        assert "✅ Integrated" in out, out
        assert (target / ".github" / "workflows" / "ci.yml").exists()
        custody._CUSTODY.clear()

    def test_system_repo_target_still_refuses(self, tmp_path, monkeypatch):
        import ouroboros.tools.subagent_integration as si

        target, ctx, _ = self._ready(tmp_path, monkeypatch, ".github/workflows/ci.yml")
        # The SAME patch, but the task's own surface says this tree is a checkout
        # of the Ouroboros body: the protected-path policy applies again.
        ctx.task_constraint = {"mode": "acting_subagent", "surface": "self_worktree",
                               "write_root": str(target)}
        monkeypatch.setattr(si, "get_runtime_mode", lambda: "advanced")
        out = si._integrate_delegated_patch(ctx, "run-1", "apply", "")
        assert "✅ Integrated" not in out
        assert ".github/workflows/ci.yml" in out
        assert not (target / ".github" / "workflows" / "ci.yml").exists()
        custody._CUSTODY.clear()


class TestSensitiveVeto:
    def test_vetoed_content_is_never_hashed_into_the_shared_odb(self, tmp_path):
        # F6: `git add -A` writes a blob for EVERY untracked file — including
        # `.env` — into the TARGET's object database, which the execution worktree
        # shares. Removing the index entry afterwards does not unwrite the object,
        # so the secret stayed readable by hash from inside the run's root.
        target = _seed_target(tmp_path)
        secret_sha = _git(target, "hash-object", ".env").stdout.strip()
        assert secret_sha
        handle = provision_execution_snapshot(
            target_root=target, task_id="t1", snapshot_id="snapV",
            worktree_root=tmp_path / "snaps", data_dir=tmp_path / "data")
        exec_root = pathlib.Path(handle.path)
        assert not (exec_root / ".env").exists()
        for root in (target, exec_root):
            assert _git(root, "cat-file", "-e", f"{secret_sha}^{{blob}}",
                        check=False).returncode != 0, f"vetoed blob is reachable from {root}"
        # The eligible tree is still complete.
        assert (exec_root / "untracked.txt").read_text(encoding="utf-8") == "loose\n"
        assert (exec_root / "staged.txt").read_text(encoding="utf-8") == "staged\n"
        assert (exec_root / "tracked.txt").read_text(encoding="utf-8") == "one\ntwo\n"


class TestLegacyRetry:
    def test_pre_isolation_mutating_retry_is_refused(self, tmp_path, monkeypatch):
        # F5: a stored PRE-C1 row has no snapshot binding and its recorded body's
        # scope.root IS the live tree — replaying it would write straight into the
        # shared tree in the in-place regime C1 retired.
        from ouroboros.tools.delegate import _delegate_start

        target = _seed_target(tmp_path)
        ctx = _nanny_ctx(tmp_path, target, monkeypatch)
        drive = custody.custody_root(ctx)
        assert custody.record_start_requested(
            drive, run_id="", task_id="t-nanny", idempotency_key="k",
            invocation_id="inv-legacy", request={
                "prompt": "do the thing", "access": "workspace_write", "mode": "agent",
                "scope": {"kind": "project", "root": str(target)},
                "execution": {"isolation": "live", "delegated": True},
                "primaryHarness": "some-route",
            }, project_id="p", project_owned=False, route="some-route")
        out = _delegate_start(ctx, "do the thing", retry_of="inv-legacy")
        assert "retry_binding_absent" in out, out
        custody._CUSTODY.clear()


class TestDurableBinding:
    def test_binding_rides_start_rows_and_replays(self, tmp_path):
        drive = tmp_path / "drive"
        entry = custody.RunCustody(
            run_id="run-9", task_id="t-n", route_id="r",
            snapshot_id="snap9", execution_root="/x/exec", baseline_sha="abc123",
            target_root="/x/target", authority_source="acting_constraint",
        )
        assert custody.record_started(drive, entry)
        custody._CUSTODY.clear()
        replayed = custody.replay(drive)["run-9"]
        assert replayed.snapshot_id == "snap9"
        assert replayed.execution_root == "/x/exec"
        assert replayed.baseline_sha == "abc123"
        assert replayed.target_root == "/x/target"
        assert replayed.authority_source == "acting_constraint"
        custody._CUSTODY.clear()

    def test_invocation_record_carries_the_binding_for_retry(self, tmp_path):
        drive = tmp_path / "drive"
        assert custody.record_start_requested(
            drive, run_id="", task_id="t-n", idempotency_key="k",
            invocation_id="inv-1", request={"prompt": "p"},
            project_id="prj", project_owned=True, route="r",
            snapshot_id="snap1", execution_root="/x/exec",
            baseline_sha="b1", target_root="/x/target",
            authority_source="external_workspace_root")
        record = custody.invocation_record(drive, "inv-1")
        assert record["snapshot_id"] == "snap1"
        assert record["execution_root"] == "/x/exec"
        assert record["baseline_sha"] == "b1"
        assert record["target_root"] == "/x/target"
        assert record["state"] == "pending"

    def test_open_snapshot_ids_keeps_undisposed_and_pending(self, tmp_path):
        drive = tmp_path / "drive"
        # Settled AND disposed -> closed.
        done = custody.RunCustody(run_id="r-done", task_id="t", snapshot_id="s-done",
                                  execution_root="/e1", settled=True)
        custody.record_started(drive, done)
        custody.emit(drive, custody.SETTLED, {"run_id": "r-done", "task_id": "t"})
        custody.record_patch_disposed(drive, done, disposition="applied")
        # Settled but NOT disposed -> open (conflict material persists).
        undisposed = custody.RunCustody(run_id="r-open", task_id="t", snapshot_id="s-open",
                                        execution_root="/e2", settled=True)
        custody.record_started(drive, undisposed)
        custody.emit(drive, custody.SETTLED, {"run_id": "r-open", "task_id": "t"})
        # Pending invocation naming a snapshot -> open.
        custody.record_start_requested(
            drive, run_id="", task_id="t", invocation_id="inv-p",
            request={"prompt": "p"}, snapshot_id="s-pending")
        custody._CUSTODY.clear()
        open_ids = custody.open_snapshot_ids(drive)
        assert "s-open" in open_ids
        assert "s-pending" in open_ids
        assert "s-done" not in open_ids
        custody._CUSTODY.clear()
