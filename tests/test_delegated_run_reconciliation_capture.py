"""Reconciling an orphaned delegated run, and capturing its work at disposition.

Split verbatim out of ``tests/test_delegated_run_isolation.py`` by theme. This module
owns the orphan sweep across custody rows and snapshots, the binding request row it
reads, the run the daemon calls absent, and the lazy capture that only happens once the
disposition is known.
"""

from __future__ import annotations

import json
import pathlib


from ouroboros import delegate_custody as custody
from ouroboros.subagent_worktrees import (
    find_execution_snapshot,
    provision_execution_snapshot,
    prune_execution_snapshots,
)

from tests._delegated_run_isolation_shared import (
    _HealthEnv,
    _TerminalSweepGateway,
    _binding_request_row,
    _git,
    _nanny_ctx,
    _seed_target,
)


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
        assert [o["action"] for o in outcomes] == ["settled"]

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
        assert row["action"] == "settled" and row["settled"] is True
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
