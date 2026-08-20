"""An ambiguous apply intent is acknowledged, not guessed — and who may mutate the root.

Split verbatim out of ``tests/test_delegated_run_isolation.py`` by theme. This module
owns the ambiguity the apply intent must surface, the acknowledgment that resolves it,
and the authority a run needs before it may mutate the root.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.subagent_worktrees import (
    find_execution_snapshot,
    provision_execution_snapshot,
)

from tests._delegated_run_isolation_shared import (
    _git,
    _nanny_ctx,
    _seed_target,
)


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
