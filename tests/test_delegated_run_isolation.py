"""C1 delegated-run isolation: private execution snapshots, terminal capture,
explicit integration, durable binding, and the custody-cross-checked GC.

Phase C of the poltergeist sprint (owner 3=A: isolate ONLY delegated runs).
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import subprocess

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.subagent_worktrees import (
    find_execution_snapshot,
    provision_execution_snapshot,
    prune_execution_snapshots,
    prune_orphans,
    remove_execution_snapshot,
)


def _git(cwd, *args, check=True):
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=check,
    )


def _seed_target(tmp_path: pathlib.Path) -> pathlib.Path:
    """A target tree with every capture class: tracked, staged, unstaged,
    untracked-eligible, and untracked-sensitive."""
    target = tmp_path / "target"
    target.mkdir()
    _git(target, "init")
    (target / "tracked.txt").write_text("one\n", encoding="utf-8")
    _git(target, "add", "-A")
    _git(target, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "seed")
    (target / "tracked.txt").write_text("one\ntwo\n", encoding="utf-8")   # unstaged mod
    (target / "staged.txt").write_text("staged\n", encoding="utf-8")
    _git(target, "add", "staged.txt")                                     # staged add
    (target / "untracked.txt").write_text("loose\n", encoding="utf-8")    # eligible
    (target / ".env").write_text("SECRET=1\n", encoding="utf-8")          # sensitive
    return target


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


def _nanny_ctx(tmp_path, target, monkeypatch):
    """A nanny ToolContext whose active root IS the target external workspace,
    with the module-default snapshot/registry roots pinned inside the test tmp."""
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "snaps"))
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    drive = tmp_path / "drive"
    drive.mkdir(exist_ok=True)
    ctx = ToolContext(repo_dir=repo, drive_root=drive)
    ctx.workspace_root = str(target)
    ctx.workspace_mode = "external"
    ctx.task_id = "t-nanny"
    ctx.task_metadata = {}
    return ctx


def _isolated_entry(ctx, target, handle, *, run_id="run-1", settled=True):
    entry = custody.RunCustody(
        run_id=run_id, task_id="t-nanny", route_id="some-route",
        snapshot_id=handle.snapshot_id, execution_root=handle.path,
        baseline_sha=handle.baseline_sha, target_root=str(target),
        authority_source="external_workspace_root", settled=settled,
    )
    custody._CUSTODY[entry.run_id] = entry
    return entry


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


class _TerminalSweepGateway:
    """A daemon for the orphan sweep: recovery re-POSTs bind a run; every asked
    run is already terminal-succeeded; controls are accepted."""

    def __init__(self, run_id="run-rec", state="succeeded"):
        self.run_id, self.state = run_id, state

    def handshake(self, **_kw):
        return {"compatible": True}

    def start_run(self, request, *, idempotency_key=""):
        return {"runId": self.run_id}

    def get_run(self, rid, **_kw):
        return {"lastSeq": 2, "summary": {"state": self.state, "spendUsd": 0.0,
                                          "model": "m", "effectiveAccess": "workspace_write"}}

    def cancel_run(self, rid, reason=""):
        return {"accepted": True, "status": "ok"}

    def remove_project(self, pid):
        return {}

    def close(self):
        pass


def _binding_request_row(task_id, invocation_id, handle):
    """The exact START_REQUESTED payload delegate.py records for a mutating start."""
    body = {"prompt": "do work", "access": "workspace_write", "mode": "agent",
            "primaryHarness": "some-route", "model": "", "effort": "", "maxSeconds": 600,
            "execution": {"isolation": "live", "delegated": True},
            "scope": {"kind": "project", "root": handle.path}}
    return dict(
        run_id="", task_id=task_id, idempotency_key=f"k-{invocation_id}",
        invocation_id=invocation_id, max_seconds=600, request=body,
        project_id=f"prj-{invocation_id}", project_owned=True, route="some-route",
        root_task_id="", parent_task_id="",
        snapshot_id=handle.snapshot_id, execution_root=handle.path,
        baseline_sha=handle.baseline_sha, target_root=handle.target_root,
        authority_source="acting_constraint")


class TestOrphanReconciliation:
    """The two C1 release blockers: orphan recovery must not lose the snapshot
    binding (and the GC must not eat an undispositioned patch), and terminal
    reconciliation must capture the stranded diff — while NEVER applying it."""

    def _stranded(self, tmp_path, *, snapshot_id, task_id, started=True):
        """A crash-shaped mutating run: snapshot provisioned, child edited it,
        owner worker died. ``started=False`` stops one row earlier (accepted
        POST, no STARTED row — the pending-invocation class)."""
        target = _seed_target(tmp_path)
        data = tmp_path / "data"
        handle = provision_execution_snapshot(
            target_root=target, task_id=task_id, snapshot_id=snapshot_id,
            worktree_root=tmp_path / "snaps", data_dir=data)
        exec_root = pathlib.Path(handle.path)
        (exec_root / "tracked.txt").write_text("one\ntwo\nCHILD-EDIT\n", encoding="utf-8")
        assert custody.record_start_requested(
            data, **_binding_request_row(task_id, snapshot_id, handle))
        if started:
            entry = custody.RunCustody(
                run_id=f"run-{snapshot_id}", task_id=task_id, route_id="some-route",
                project_id=f"prj-{snapshot_id}", project_owned=True, ledger_root=str(data),
                idempotency_key=f"k-{snapshot_id}", invocation_id=snapshot_id,
                snapshot_id=handle.snapshot_id, execution_root=handle.path,
                baseline_sha=handle.baseline_sha, target_root=handle.target_root,
                authority_source="acting_constraint")
            assert custody.record_started(data, entry)
        custody._CUSTODY.clear()
        return target, data, handle

    def test_pending_invocation_recovery_preserves_the_binding_and_the_gc_keeps_it(self, tmp_path):
        # Blocker 1: the worker died between the accepted POST and record_started.
        # Recovery used to rebuild custody WITHOUT the C1 binding, so the recovered
        # run replayed bindingless, open_snapshot_ids went empty the moment the
        # invocation stopped being pending, and the startup GC deleted the snapshot
        # holding the child's ONLY copy of its work.
        target, data, handle = self._stranded(
            tmp_path, snapshot_id="inv-lost", task_id="t-dead", started=False)
        assert custody.open_snapshot_ids(data) == {"inv-lost"}

        outcomes = custody.reconcile_orphaned_runs(
            data, set(), gateway_factory=lambda: _TerminalSweepGateway("run-rec"))
        custody._CUSTODY.clear()
        assert [o["action"] for o in outcomes] == ["settle_attempted"]
        assert outcomes[0]["settled"] is True

        replayed = custody.replay(data)["run-rec"]
        assert replayed.snapshot_id == "inv-lost"
        assert replayed.execution_root == handle.path
        assert replayed.baseline_sha == handle.baseline_sha
        assert replayed.target_root == handle.target_root
        assert replayed.authority_source == "acting_constraint"
        # Undisposed -> still OPEN to custody, so the startup GC keeps everything.
        assert "inv-lost" in custody.open_snapshot_ids(data)
        report = prune_execution_snapshots(custody.open_snapshot_ids(data),
                                           worktree_root=tmp_path / "snaps", data_dir=data)
        assert report["kept"] == ["inv-lost"] and report["removed"] == []
        assert find_execution_snapshot("inv-lost", data_dir=data) is not None
        assert pathlib.Path(handle.path).exists()
        # And the work is no longer only-in-the-snapshot: the sweep captured it.
        patch = custody.delegated_capture_dir(data, "t-dead", "inv-lost") / "workspace.patch"
        assert "CHILD-EDIT" in patch.read_text(encoding="utf-8")
        custody._CUSTODY.clear()

    def test_reconcile_captures_the_stranded_patch_and_never_applies_it(self, tmp_path):
        # Blocker 2 + the no-auto-apply pin: a run reaching terminal through the
        # sweep (owner gone) got no patch capture at all — the work stayed
        # stranded in the snapshot with no apply/reject material. The sweep now
        # captures into the SAME durable artifact the nanny path uses, records
        # the pending disposition, and touches NOTHING in the shared tree.
        target, data, handle = self._stranded(
            tmp_path, snapshot_id="inv-orphan", task_id="t-dead2")
        before_status = _git(target, "status", "--porcelain").stdout

        outcomes = custody.reconcile_orphaned_runs(
            data, set(), gateway_factory=lambda: _TerminalSweepGateway("run-inv-orphan"))
        custody._CUSTODY.clear()
        row = outcomes[0]
        assert row["action"] == "settle_attempted" and row["settled"] is True
        assert row["patch_capture"] == "ready_with_changes"
        assert row["patch_disposition"] == "pending"
        cap_dir = custody.delegated_capture_dir(data, "t-dead2", "inv-orphan")
        assert pathlib.Path(row["patch_artifact"]) == cap_dir / "workspace.patch"
        assert "CHILD-EDIT" in (cap_dir / "workspace.patch").read_text(encoding="utf-8")
        assert (cap_dir / "workspace_patch.json").exists()
        # Durable, replayed: captured yes, disposed NO — the decision is an owner's.
        replayed = custody.replay(data)["run-inv-orphan"]
        assert replayed.patch_captured is True
        assert replayed.patch_disposed == ""
        # NO AUTO-APPLY: the shared tree is byte-identical to before the sweep.
        assert (target / "tracked.txt").read_text(encoding="utf-8") == "one\ntwo\n"
        assert _git(target, "status", "--porcelain").stdout == before_status
        # The typed disclosure is durable on the RECONCILED row, not only returned.
        events = (data / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        reconciled = [json.loads(l) for l in events if '"delegate_run_reconciled"' in l]
        assert reconciled[-1]["patch_disposition"] == "pending"
        custody._CUSTODY.clear()

    def test_a_still_live_run_is_not_captured_and_stays_open(self, tmp_path):
        # A cancel that is merely REQUESTED leaves the run live and its snapshot
        # still being written: capturing there would ship a torn diff. Nothing is
        # captured, nothing disposed, and the snapshot stays custody-open.
        target, data, handle = self._stranded(
            tmp_path, snapshot_id="inv-live", task_id="t-dead3")
        outcomes = custody.reconcile_orphaned_runs(
            data, set(),
            gateway_factory=lambda: _TerminalSweepGateway("run-inv-live", state="running"))
        custody._CUSTODY.clear()
        assert outcomes[0]["action"] == "cancelled"
        assert outcomes[0]["outcome"] == custody.CANCEL_REQUESTED
        assert "patch_capture" not in outcomes[0]
        replayed = custody.replay(data)["run-inv-live"]
        assert replayed.patch_captured is False
        assert "inv-live" in custody.open_snapshot_ids(data)
        assert not (custody.delegated_capture_dir(data, "t-dead3", "inv-live")
                    / "workspace.patch").exists()
        custody._CUSTODY.clear()

    def test_gc_requires_a_closed_run_and_a_recorded_disposition(self, tmp_path):
        # The GC predicate, end to end against real snapshots: settled+disposed is
        # the ONLY disposable combination; settled-but-undisposed (the orphan
        # shape) is preserved.
        target, data, handle = self._stranded(
            tmp_path, snapshot_id="inv-und", task_id="t-und")
        done = provision_execution_snapshot(
            target_root=target, task_id="t-done", snapshot_id="inv-done",
            worktree_root=tmp_path / "snaps", data_dir=data)
        entry = custody.RunCustody(run_id="run-done", task_id="t-done",
                                   snapshot_id="inv-done", execution_root=done.path)
        custody.record_started(data, entry)
        custody.emit(data, custody.SETTLED, {"run_id": "run-done", "task_id": "t-done"})
        custody.record_patch_disposed(data, entry, disposition="applied")
        custody.reconcile_orphaned_runs(
            data, {"t-done"}, gateway_factory=lambda: _TerminalSweepGateway("run-inv-und"))
        custody._CUSTODY.clear()

        report = prune_execution_snapshots(custody.open_snapshot_ids(data),
                                           worktree_root=tmp_path / "snaps", data_dir=data)
        assert report["removed"] == ["inv-done"]
        assert report["kept"] == ["inv-und"]
        assert find_execution_snapshot("inv-und", data_dir=data) is not None
        assert find_execution_snapshot("inv-done", data_dir=data) is None
        custody._CUSTODY.clear()

    def test_undisposed_patch_is_disclosed_on_the_health_surface_until_disposed(self, tmp_path):
        # "Preserved but invisible" is how an orphan's work sits on disk forever:
        # the pending disposition is a visible obligation (typed projection + the
        # health-invariant row) and self-clears when the disposition row lands.
        target, data, handle = self._stranded(
            tmp_path, snapshot_id="inv-vis", task_id="t-vis")
        custody.reconcile_orphaned_runs(
            data, set(), gateway_factory=lambda: _TerminalSweepGateway("run-inv-vis"))
        custody._CUSTODY.clear()

        pending = custody.undisposed_patches(data)
        assert [run.run_id for run in pending] == ["run-inv-vis"]

        from ouroboros.context_health import build_health_invariants

        class _Env:
            drive_root = data

            def drive_path(self, rel=""):
                return data / rel

            def repo_path(self, rel=""):
                return data / "repo" / rel

        surface = build_health_invariants(_Env())
        assert "DELEGATED PATCH AWAITS DISPOSITION" in surface
        assert "run-inv-vis" in surface and "integrate_delegated_patch" in surface
        # A PROVEN-terminal sweep captured eagerly, so the receipt is honest.
        assert "changes captured" in surface

        entry = custody.replay(data)["run-inv-vis"]
        custody.record_patch_disposed(data, entry, disposition="rejected")
        custody._CUSTODY.clear()
        assert custody.undisposed_patches(data) == []
        assert "DELEGATED PATCH AWAITS DISPOSITION" not in build_health_invariants(_Env())
        custody._CUSTODY.clear()


class _AbsentGateway:
    """A daemon that answers 404 for every run — the reconcile 'absent' branch.

    Across the D30 owned-daemon provisioning boundary this answer can come from a
    DIFFERENT daemon than the one that accepted the run, whose child may still be
    alive and writing to the snapshot."""

    def handshake(self, **_kw):
        return {"compatible": True}

    def get_run(self, rid, **_kw):
        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        raise ClaudexorUnavailable("not_found", "no such run", status_code=404)

    def cancel_run(self, rid, reason=""):
        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        raise ClaudexorUnavailable("not_found", "no such run", status_code=404)

    def remove_project(self, pid):
        return {}

    def close(self):
        pass


class _HealthEnv:
    """The minimal env build_health_invariants needs, rooted at one data dir."""

    def __init__(self, data: pathlib.Path):
        self.drive_root = data
        self._data = data

    def drive_path(self, rel=""):
        return self._data / rel

    def repo_path(self, rel=""):
        return self._data / "repo" / rel


class TestLazyCaptureAtDisposition:
    """C1-R2: where terminal truth is ABSENT, capture is lazy and disposition is
    the retry point. An absent run's state is unknowable from here (the child may
    still be writing across the D30 provisioning boundary), so an eager capture
    there would freeze a potentially incomplete patch — and the idempotent
    early return would make the incompleteness permanent."""

    def _stranded_absent(self, tmp_path, *, snapshot_id, task_id):
        """A stranded mutating run whose daemon answers 404 (state unknowable)."""
        target = _seed_target(tmp_path)
        data = tmp_path / "data"
        handle = provision_execution_snapshot(
            target_root=target, task_id=task_id, snapshot_id=snapshot_id,
            worktree_root=tmp_path / "snaps", data_dir=data)
        exec_root = pathlib.Path(handle.path)
        (exec_root / "tracked.txt").write_text("one\ntwo\nCHILD-EDIT\n", encoding="utf-8")
        assert custody.record_start_requested(
            data, **_binding_request_row(task_id, snapshot_id, handle))
        entry = custody.RunCustody(
            run_id=f"run-{snapshot_id}", task_id=task_id, route_id="some-route",
            project_id=f"prj-{snapshot_id}", project_owned=True, ledger_root=str(data),
            idempotency_key=f"k-{snapshot_id}", invocation_id=snapshot_id,
            snapshot_id=handle.snapshot_id, execution_root=handle.path,
            baseline_sha=handle.baseline_sha, target_root=handle.target_root,
            authority_source="acting_constraint")
        assert custody.record_started(data, entry)
        custody._CUSTODY.clear()
        return target, data, handle

    def test_absent_reconcile_does_not_capture_and_health_says_preserved(self, tmp_path):
        # (a) The absent branch closes custody WITHOUT freezing a patch over
        # unknowable state: the snapshot persists, the obligation surfaces, and
        # the health line says "preserved ... captured at disposition" — never
        # a "captured" receipt for a capture that did not happen.
        target, data, handle = self._stranded_absent(
            tmp_path, snapshot_id="inv-abs", task_id="t-abs")
        outcomes = custody.reconcile_orphaned_runs(
            data, set(), gateway_factory=lambda: _AbsentGateway())
        custody._CUSTODY.clear()
        assert [o["action"] for o in outcomes] == ["absent"]
        assert "patch_capture" not in outcomes[0]
        replayed = custody.replay(data)["run-inv-abs"]
        assert replayed.settled is True
        assert replayed.patch_captured is False
        assert not (custody.delegated_capture_dir(data, "t-abs", "inv-abs")
                    / "workspace.patch").exists()
        # The snapshot stays custody-open (undisposed), so the GC keeps it.
        assert "inv-abs" in custody.open_snapshot_ids(data)
        assert find_execution_snapshot("inv-abs", data_dir=data) is not None
        assert pathlib.Path(handle.path).exists()
        # The obligation is visible, with truthful state-dependent wording.
        assert [r.run_id for r in custody.undisposed_patches(data)] == ["run-inv-abs"]
        from ouroboros.context_health import build_health_invariants

        surface = build_health_invariants(_HealthEnv(data))
        assert "DELEGATED PATCH AWAITS DISPOSITION" in surface
        assert "preserved" in surface and "at disposition" in surface
        assert "changes captured" not in surface
        custody._CUSTODY.clear()

    def test_post_absent_write_reaches_the_patch_at_disposition(self, tmp_path, monkeypatch):
        # (b) The verifier's repro, inverted: a write landing AFTER the 404 must
        # reach the patch, because capture now happens at disposition — the
        # honest latest-possible capture point — not at the 404.
        from ouroboros.tools.subagent_integration import _integrate_delegated_patch

        target = _seed_target(tmp_path)
        ctx = _nanny_ctx(tmp_path, target, monkeypatch)
        handle = provision_execution_snapshot(
            target_root=target, task_id="t-nanny", snapshot_id="snapLazy")
        exec_root = pathlib.Path(handle.path)
        (exec_root / "tracked.txt").write_text("one\ntwo\nEARLY-EDIT\n", encoding="utf-8")
        drive = custody.custody_root(ctx)
        entry = custody.RunCustody(
            run_id="run-lazy", task_id="t-nanny", route_id="some-route",
            snapshot_id=handle.snapshot_id, execution_root=handle.path,
            baseline_sha=handle.baseline_sha, target_root=str(target),
            authority_source="external_workspace_root")
        assert custody.record_started(drive, entry)
        custody._CUSTODY.clear()
        outcomes = custody.reconcile_orphaned_runs(
            drive, set(), gateway_factory=lambda: _AbsentGateway())
        assert [o["action"] for o in outcomes] == ["absent"]
        # The still-alive child (other daemon's process) writes AFTER the 404.
        (exec_root / "tracked.txt").write_text(
            "one\ntwo\nEARLY-EDIT\nLATE-WRITE\n", encoding="utf-8")
        custody._CUSTODY.clear()
        out = _integrate_delegated_patch(ctx, "run-lazy", "apply", "collect the orphan")
        assert "✅ Integrated" in out, out
        applied = (target / "tracked.txt").read_text(encoding="utf-8")
        assert "LATE-WRITE" in applied and "EARLY-EDIT" in applied
        custody._CUSTODY.clear()

    def test_capture_failure_at_disposition_is_typed_and_keeps_the_obligation(self, tmp_path, monkeypatch):
        # (c) A capture that fails at disposition is a typed refusal for BOTH
        # decisions — never a silent reject/apply over nothing. No disposition
        # is recorded, so the obligation stays open and the snapshot persists.
        from ouroboros.tools.subagent_integration import _integrate_delegated_patch

        target = _seed_target(tmp_path)
        ctx = _nanny_ctx(tmp_path, target, monkeypatch)
        handle = provision_execution_snapshot(
            target_root=target, task_id="t-nanny", snapshot_id="snapFail")
        (pathlib.Path(handle.path) / "tracked.txt").write_text(
            "one\ntwo\nCHILD-EDIT\n", encoding="utf-8")
        drive = custody.custody_root(ctx)
        entry = custody.RunCustody(
            run_id="run-fail", task_id="t-nanny", route_id="some-route",
            snapshot_id=handle.snapshot_id, execution_root=handle.path,
            baseline_sha=handle.baseline_sha, target_root=str(target),
            authority_source="external_workspace_root")
        assert custody.record_started(drive, entry)
        custody._CUSTODY.clear()

        def _broken_capture(*_a, **_kw):
            raise RuntimeError("diff machinery broke")

        monkeypatch.setattr(
            "ouroboros.headless.write_workspace_patch_artifacts", _broken_capture)
        custody.reconcile_orphaned_runs(
            drive, set(), gateway_factory=lambda: _AbsentGateway())
        custody._CUSTODY.clear()
        for decision in ("apply", "reject"):
            out = _integrate_delegated_patch(ctx, "run-fail", decision, "")
            assert "INTEGRATE_DELEGATED_CAPTURE_FAILED" in out, (decision, out)
            custody._CUSTODY.clear()
        replayed = custody.replay(drive)["run-fail"]
        assert replayed.patch_disposed == ""
        assert [r.run_id for r in custody.undisposed_patches(drive)] == ["run-fail"]
        assert find_execution_snapshot("snapFail") is not None
        assert pathlib.Path(handle.path).exists()
        # The shared tree was never touched.
        assert (target / "tracked.txt").read_text(encoding="utf-8") == "one\ntwo\n"
        custody._CUSTODY.clear()

    def test_cancel_verified_terminal_still_captures_eagerly(self, tmp_path):
        # (d) regression pin: where a TERMINAL RECEIPT proves the run is over —
        # here a cancel verified terminal by the read-back — the sweep still
        # captures eagerly, exactly as before. (The is_terminal branch is pinned
        # by test_reconcile_captures_the_stranded_patch_and_never_applies_it.)
        target, data, handle = self._stranded_absent(
            tmp_path, snapshot_id="inv-can", task_id="t-can")

        class _CancelTerminalGateway(_TerminalSweepGateway):
            def __init__(self):
                super().__init__("run-inv-can")
                self.reads = 0

            def get_run(self, rid, **_kw):
                self.reads += 1
                state = "running" if self.reads == 1 else "cancelled"
                return {"lastSeq": 2, "summary": {"state": state, "spendUsd": 0.0,
                                                  "model": "m"}}

        outcomes = custody.reconcile_orphaned_runs(
            data, set(), gateway_factory=_CancelTerminalGateway)
        custody._CUSTODY.clear()
        row = outcomes[0]
        assert row["action"] == "cancelled"
        assert row["outcome"] == custody.CANCEL_CONFIRMED
        assert row["patch_capture"] == "ready_with_changes"
        assert row["patch_disposition"] == "pending"
        cap = custody.delegated_capture_dir(data, "t-can", "inv-can") / "workspace.patch"
        assert "CHILD-EDIT" in cap.read_text(encoding="utf-8")
        assert custody.replay(data)["run-inv-can"].patch_captured is True
        custody._CUSTODY.clear()


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
        from ouroboros.tools.core import _read_file
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
        from ouroboros.tools.core import _read_file
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


class TestApplyIntentAmbiguity:
    """CR1-3: crash replay must never record a false rejection. The apply
    persists a durable intent row BEFORE mutating the target; a run whose
    intent has neither a resolution nor a disposition replays as AMBIGUOUS
    (the tree may carry the patch), and both decisions refuse typed instead
    of pretending "not applied"."""

    def _settled_run(self, tmp_path, monkeypatch, *, snapshot_id, run_id):
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

    def test_crash_between_apply_and_disposition_refuses_false_rejection(self, tmp_path, monkeypatch):
        # The EXACT reproduced sequence: apply succeeds, the disposition row
        # fails to land, the process dies. After a restart the run reads as
        # undisposed — a reject then claimed "not applied", recorded
        # `rejected` and deleted the snapshot while the tree stayed modified
        # and staged. Now: typed ambiguity refusal, nothing disposed.
        from ouroboros.tools.subagent_integration import _integrate_delegated_patch

        target, ctx, handle, drive = self._settled_run(
            tmp_path, monkeypatch, snapshot_id="snapAmb", run_id="run-amb")
        real_emit = custody.emit
        monkeypatch.setattr(
            custody, "emit",
            lambda d, kind, payload: (
                False if kind == custody.PATCH_DISPOSED else real_emit(d, kind, payload)))
        out = _integrate_delegated_patch(ctx, "run-amb", "apply", "")
        assert "INTEGRATE_DISPOSITION_UNWRITTEN" in out, out
        assert (target / "tracked.txt").read_text(encoding="utf-8").endswith("CHILD-EDIT\n")
        # The RESTART: the in-process memo is gone, the event log heals.
        monkeypatch.setattr(custody, "emit", real_emit)
        custody._CUSTODY.clear()
        replayed = custody.replay(drive)["run-amb"]
        assert replayed.patch_apply_pending is True
        assert replayed.patch_disposed == ""
        for decision in ("reject", "apply"):
            out2 = _integrate_delegated_patch(ctx, "run-amb", decision, "discard it")
            assert "INTEGRATE_DELEGATED_APPLY_AMBIGUOUS" in out2, (decision, out2)
            custody._CUSTODY.clear()
        # No false rejection was recorded; material persists.
        assert custody.replay(drive)["run-amb"].patch_disposed == ""
        assert find_execution_snapshot("snapAmb") is not None
        assert pathlib.Path(handle.path).exists()
        # The tree honestly still carries the applied, staged patch.
        assert "CHILD-EDIT" in (target / "tracked.txt").read_text(encoding="utf-8")
        staged = _git(target, "diff", "--cached", "--name-only").stdout
        assert "tracked.txt" in staged
        custody._CUSTODY.clear()

    def test_conflict_resolution_row_keeps_the_retry_open_across_restart(self, tmp_path, monkeypatch):
        # A refused apply (proven drift — nothing mutated) must NOT wedge the
        # run into the ambiguity refusal: the durable resolution row clears the
        # intent, so the nanny's reconcile-and-retry flow survives a restart.
        from ouroboros.tools.subagent_integration import _integrate_delegated_patch

        target, ctx, handle, drive = self._settled_run(
            tmp_path, monkeypatch, snapshot_id="snapRetry", run_id="run-retry")
        # The target moved differently on the same line after the snapshot.
        (target / "tracked.txt").write_text("TARGET-EDIT\n", encoding="utf-8")
        out = _integrate_delegated_patch(ctx, "run-retry", "apply", "")
        assert "INTEGRATE_CONFLICT" in out, out
        custody._CUSTODY.clear()
        replayed = custody.replay(drive)["run-retry"]
        assert replayed.patch_apply_pending is False  # resolved: tree unmutated
        # The nanny reconciles the tree back, restarts, retries: apply works.
        (target / "tracked.txt").write_text("one\ntwo\n", encoding="utf-8")
        out2 = _integrate_delegated_patch(ctx, "run-retry", "apply", "")
        assert "✅ Integrated" in out2, out2
        custody._CUSTODY.clear()

    def test_unlanded_intent_row_refuses_to_mutate(self, tmp_path, monkeypatch):
        # Owed-before-sent: an apply whose intent row cannot land must not
        # touch the tree at all — a crash mid-apply would otherwise leave a
        # mutated tree nothing durable accounts for.
        from ouroboros.tools.subagent_integration import _integrate_delegated_patch

        target, ctx, handle, drive = self._settled_run(
            tmp_path, monkeypatch, snapshot_id="snapNoInt", run_id="run-noint")
        real_emit = custody.emit
        monkeypatch.setattr(
            custody, "emit",
            lambda d, kind, payload: (
                False if kind == custody.PATCH_APPLY_STARTED else real_emit(d, kind, payload)))
        out = _integrate_delegated_patch(ctx, "run-noint", "apply", "")
        assert "INTEGRATE_INTENT_UNWRITTEN" in out, out
        assert (target / "tracked.txt").read_text(encoding="utf-8") == "one\ntwo\n"
        custody._CUSTODY.clear()
        assert custody.replay(drive)["run-noint"].patch_disposed == ""
        assert find_execution_snapshot("snapNoInt") is not None
        custody._CUSTODY.clear()


class TestAmbiguityAcknowledgment:
    """CR2-1: the AMBIGUOUS state must have an owner exit, not be a permanent
    dead-end. `acknowledge_ambiguous=true` durably resolves the stale intent
    as owner-acknowledged and re-runs the NORMAL disposition guards — apply
    re-proves baseline drift (honest refusal over a tree that already carries
    the patch; clean apply over a clean tree), reject re-runs the
    ready-manifest guard and releases the snapshot while the captured patch
    artifact is retained. Without the flag, the refusal stands. CR2-3: a
    verdict-write failure in the reverted branch must not strand the intent."""

    _settled_run = TestApplyIntentAmbiguity._settled_run

    def test_acknowledgment_exits_the_crash_before_apply_wedge(self, tmp_path, monkeypatch):
        # Crash BEFORE the apply ran: the durable intent row landed, the tree
        # is provably clean. Pre-fix, every later apply/reject refused forever.
        from ouroboros.tools.subagent_integration import _integrate_delegated_patch

        target, ctx, handle, drive = self._settled_run(
            tmp_path, monkeypatch, snapshot_id="snapAck", run_id="run-ack")
        entry = custody.replay(drive)["run-ack"]
        assert custody.record_patch_apply_started(drive, entry, target_root=str(target))
        custody._CUSTODY.clear()
        # Without the flag the typed refusal stands — and now names the exit.
        out = _integrate_delegated_patch(ctx, "run-ack", "apply", "")
        assert "INTEGRATE_DELEGATED_APPLY_AMBIGUOUS" in out, out
        assert "acknowledge_ambiguous" in out, out
        custody._CUSTODY.clear()
        # With the flag: stale intent resolved durably, NORMAL apply succeeds.
        out2 = _integrate_delegated_patch(
            ctx, "run-ack", "apply", "inspected the tree", acknowledge_ambiguous=True)
        assert "✅ Integrated" in out2, out2
        assert "CHILD-EDIT" in (target / "tracked.txt").read_text(encoding="utf-8")
        custody._CUSTODY.clear()
        replayed = custody.replay(drive)["run-ack"]
        assert replayed.patch_disposed == "applied"
        assert replayed.patch_apply_pending is False
        rows = [json.loads(line) for line in
                custody.event_log_path(drive).read_text(encoding="utf-8").splitlines()
                if '"delegate_run_patch_apply_resolved"' in line]
        assert any(row.get("reason") == "owner_acknowledged"
                   and row.get("run_id") == "run-ack" for row in rows), rows
        custody._CUSTODY.clear()

    def test_acknowledged_crash_after_apply_gets_honest_drift_then_reject_releases(
            self, tmp_path, monkeypatch):
        # Crash AFTER the apply: the tree already carries the staged patch.
        from ouroboros.tools.subagent_integration import _integrate_delegated_patch

        target, ctx, handle, drive = self._settled_run(
            tmp_path, monkeypatch, snapshot_id="snapAck2", run_id="run-ack2")
        real_emit = custody.emit
        monkeypatch.setattr(
            custody, "emit",
            lambda d, kind, payload: (
                False if kind == custody.PATCH_DISPOSED else real_emit(d, kind, payload)))
        out = _integrate_delegated_patch(ctx, "run-ack2", "apply", "")
        assert "INTEGRATE_DISPOSITION_UNWRITTEN" in out, out
        monkeypatch.setattr(custody, "emit", real_emit)
        custody._CUSTODY.clear()
        assert custody.replay(drive)["run-ack2"].patch_apply_pending is True
        # Acknowledged apply re-runs the drift guard, which honestly refuses:
        # the tree diverged from the baseline (it carries the crashed apply).
        out2 = _integrate_delegated_patch(
            ctx, "run-ack2", "apply", "", acknowledge_ambiguous=True)
        assert "INTEGRATE_CONFLICT" in out2, out2
        custody._CUSTODY.clear()
        replayed = custody.replay(drive)["run-ack2"]
        assert replayed.patch_apply_pending is False  # resolved, no wedge
        assert replayed.patch_disposed == ""
        # Acknowledged reject releases the snapshot; the patch artifact and
        # the tree's applied changes survive (no work is destroyed).
        cap_dir = custody.delegated_capture_dir(drive, "t-nanny", "snapAck2")
        patch_path = cap_dir / "workspace.patch"
        assert patch_path.exists()
        out3 = _integrate_delegated_patch(
            ctx, "run-ack2", "reject", "keeping the tree as-is", acknowledge_ambiguous=True)
        assert "🚫 Rejected" in out3, out3
        custody._CUSTODY.clear()
        assert custody.replay(drive)["run-ack2"].patch_disposed == "rejected"
        assert find_execution_snapshot("snapAck2") is None
        assert patch_path.exists()
        assert "CHILD-EDIT" in (target / "tracked.txt").read_text(encoding="utf-8")
        custody._CUSTODY.clear()

    def test_flag_without_pending_intent_is_a_no_op(self, tmp_path, monkeypatch):
        # CR2-1 (4): acknowledge_ambiguous over a run with NO pending intent
        # behaves exactly like the plain call — no error, no spurious row.
        from ouroboros.tools.subagent_integration import _integrate_delegated_patch

        target, ctx, handle, drive = self._settled_run(
            tmp_path, monkeypatch, snapshot_id="snapNop", run_id="run-nop")
        out = _integrate_delegated_patch(
            ctx, "run-nop", "apply", "", acknowledge_ambiguous=True)
        assert "✅ Integrated" in out, out
        rows = [json.loads(line) for line in
                custody.event_log_path(drive).read_text(encoding="utf-8").splitlines()
                if '"delegate_run_patch_apply_resolved"' in line]
        assert not any(row.get("reason") == "owner_acknowledged" for row in rows), rows
        custody._CUSTODY.clear()

    def test_verdict_write_failure_after_revert_does_not_strand_the_intent(
            self, tmp_path, monkeypatch):
        # CR2-3: in the cleanly-reverted staging-failure branch the tree is
        # provably back to pre-apply; a _write_verdict raise (artifact-dir
        # mkdir failure) must not leave the durable intent pending — that was
        # a second entrance into the AMBIGUOUS wedge.
        from ouroboros.tools import subagent_integration as si

        target, ctx, handle, drive = self._settled_run(
            tmp_path, monkeypatch, snapshot_id="snapVw", run_id="run-vw")

        class _Proc:
            returncode = 0
            stdout = ""
            stderr = ""

        def _boom(*args, **kwargs):
            raise OSError("artifact dir mkdir failed")

        with monkeypatch.context() as patched:
            patched.setattr(si, "_locked_apply", lambda *a, **k: {
                "proc": _Proc(), "drifted": [], "drift_error": "",
                "staging_failure": "index locked", "reverted": True,
                "lock_error": ""})
            patched.setattr(si, "_write_verdict", _boom)
            with pytest.raises(OSError):  # the verdict-write failure stays loud
                si._integrate_delegated_patch(ctx, "run-vw", "apply", "")
        custody._CUSTODY.clear()
        replayed = custody.replay(drive)["run-vw"]
        assert replayed.patch_apply_pending is False  # resolved BEFORE the verdict
        assert replayed.patch_disposed == ""
        # The retry lane is open: a fresh, unpatched apply succeeds normally.
        out = si._integrate_delegated_patch(ctx, "run-vw", "apply", "")
        assert "✅ Integrated" in out, out
        custody._CUSTODY.clear()


class TestRootMutationAuthority:
    def test_external_workspace_root_derives_the_mutating_shape(self, tmp_path):
        # B5 (owner 2=A): the ROOT of an external-workspace task holds no acting
        # constraint — its authority derives from its own validated workspace.
        from ouroboros.tools.delegate import _derive_authority, _mutation_authority
        from ouroboros.tools.registry import ToolContext

        target = _seed_target(tmp_path)
        repo = tmp_path / "repo"
        repo.mkdir()
        ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "drive")
        ctx.workspace_root = str(target)
        ctx.workspace_mode = "external"
        ctx.task_metadata = {}
        authority = _derive_authority(ctx)
        assert authority.access == "workspace_write"
        assert authority.isolation == "live" and authority.delegated is True
        record, err = _mutation_authority(ctx, authority)
        assert err == "", err
        assert record["source"] == "external_workspace_root"
        assert record["capture_mode"] == "delegated_snapshot"
        assert pathlib.Path(record["target_root"]).resolve() == target.resolve()

    def test_root_workspace_divergence_is_a_typed_refusal(self, tmp_path):
        from ouroboros.subagents import delegated_run_shape
        from ouroboros.tools.delegate import _mutation_authority
        from ouroboros.tools.registry import ToolContext

        repo = tmp_path / "repo"
        repo.mkdir()
        ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "drive")
        ctx.workspace_root = None
        ctx.workspace_mode = ""
        ctx.task_metadata = {}
        record, err = _mutation_authority(ctx, delegated_run_shape(True))
        assert record == {} and "workspace_not_active" in err
