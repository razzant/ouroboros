"""Orphan disposition across NESTED host-minted trees (В7-A) and the orphan
capture read.

Both existing target-mismatch tests use DISJOINT roots, so the containment
change would otherwise land with no failing test. The incident: a swarm fanned
into ``<project>/contributions/<track>`` clones, its children died, and the
recovery root -- whose active root was the PARENT project -- got
``INTEGRATE_DELEGATED_TARGET_MISMATCH`` for a target that was a strict
descendant of its own root, while the capture it was authorized to dispose
read back as "outside selected root=artifact_store".
"""

from __future__ import annotations

import pathlib

from ouroboros import delegate_custody as custody
from ouroboros.subagent_worktrees import provision_execution_snapshot
from ouroboros.task_results import STATUS_FAILED, write_task_result
from ouroboros.tools.delegate import _capture_terminal_patch
from ouroboros.tools.subagent_integration import _integrate_delegated_patch
from tests.test_delegated_run_isolation import _git, _isolated_entry, _nanny_ctx, _seed_target


def _nested_clone(tmp_path, monkeypatch, *, under_projects_root: bool):
    """A parent tree with a nested git clone, optionally inside the host-minted
    subagent-projects root, plus a nanny ctx whose active root IS the clone."""
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(projects_root))
    parent_root = (projects_root if under_projects_root else tmp_path / "attached") / "P"
    parent_root.mkdir(parents=True)
    target = _seed_target(parent_root)  # <parent>/target: a real git tree
    ctx = _nanny_ctx(tmp_path, target, monkeypatch)
    return parent_root, target, ctx


def _captured_orphan(tmp_path, monkeypatch, *, under_projects_root=True, terminal=True):
    parent_root, target, ctx = _nested_clone(
        tmp_path, monkeypatch, under_projects_root=under_projects_root)
    handle = provision_execution_snapshot(
        target_root=target, task_id="t-nanny", snapshot_id="snapNested")
    custody._CUSTODY.clear()
    (pathlib.Path(handle.path) / "newfile.py").write_text("print('hi')\n", encoding="utf-8")
    entry = _isolated_entry(ctx, target, handle)
    assert _capture_terminal_patch(ctx, entry)["status"] == "ready_with_changes"
    if terminal:
        write_task_result(custody.custody_root(ctx), "t-nanny", STATUS_FAILED)
    return parent_root, target, ctx, entry


def _captured_flat(tmp_path, monkeypatch, *, terminal=True):
    """One owner with a DURABLY recorded, settled, captured run (the production
    shape the read resolver replays, not just the in-process memo)."""
    target = _seed_target(tmp_path)
    ctx = _nanny_ctx(tmp_path, target, monkeypatch)
    handle = provision_execution_snapshot(
        target_root=target, task_id="t-nanny", snapshot_id="snapRead")
    (pathlib.Path(handle.path) / "tracked.txt").write_text(
        "one\ntwo\nCHILD-EDIT\n", encoding="utf-8")
    drive = custody.custody_root(ctx)
    entry = custody.RunCustody(
        run_id="run-read", task_id="t-nanny", route_id="some-route",
        snapshot_id=handle.snapshot_id, execution_root=handle.path,
        baseline_sha=handle.baseline_sha, target_root=str(target),
        authority_source="external_workspace_root")
    assert custody.record_started(drive, entry)
    custody.emit(drive, custody.SETTLED, {"run_id": "run-read", "task_id": "t-nanny"})
    custody._CUSTODY.clear()
    entry = custody.replay(drive)["run-read"]
    block = _capture_terminal_patch(ctx, entry)
    assert block["status"] == "ready_with_changes", block
    if terminal:
        write_task_result(drive, "t-nanny", STATUS_FAILED)
    custody._CUSTODY.clear()
    return target, ctx, entry, block


def _disposer(tmp_path, monkeypatch, parent_root, target, *, task_id="t-second"):
    disposer = _nanny_ctx(tmp_path, target, monkeypatch)
    disposer.task_id = task_id
    disposer.workspace_root = str(parent_root)
    return disposer


class TestNestedOrphanApply:
    def test_orphan_applies_from_the_parent_of_a_host_minted_clone(self, tmp_path, monkeypatch):
        parent_root, target, _ctx, entry = _captured_orphan(tmp_path, monkeypatch)
        disposer = _disposer(tmp_path, monkeypatch, parent_root, target)

        out = _integrate_delegated_patch(disposer, "run-1", "apply", "adopted the dead child")
        assert "✅ Integrated" in out, out
        assert "orphan of terminal task t-nanny" in out, out
        assert (target / "newfile.py").read_text(encoding="utf-8") == "print('hi')\n"
        assert "newfile.py" in _git(target, "diff", "--cached", "--name-only").stdout
        assert entry.patch_disposed == "applied"
        rows = [row for row in custody._iter_rows(
            custody.event_log_path(custody.custody_root(disposer)))
            if str(row.get("type") or "") == custody.PATCH_DISPOSED]
        assert [row["disposed_by_task_id"] for row in rows] == ["t-second"], rows
        custody._CUSTODY.clear()

    def test_nested_target_outside_the_projects_root_stays_a_mismatch(self, tmp_path, monkeypatch):
        """Nesting alone is not authority: outside the configured projects/worktrees
        roots the same geometry keeps refusing, whoever created the folder."""
        parent_root, target, _ctx, entry = _captured_orphan(
            tmp_path, monkeypatch, under_projects_root=False)
        disposer = _disposer(tmp_path, monkeypatch, parent_root, target)

        out = _integrate_delegated_patch(disposer, "run-1", "apply", "")
        assert "INTEGRATE_DELEGATED_TARGET_MISMATCH" in out, out
        assert "inside the configured projects/worktrees roots" in out, out
        assert not (target / "newfile.py").exists()
        assert entry.patch_disposed == ""
        custody._CUSTODY.clear()

    def test_a_live_owner_keeps_exact_equality_from_its_own_nested_root(self, tmp_path, monkeypatch):
        """The relaxation is the ORPHAN rule, not a looser gate: with the owner
        task still live the same nested geometry is refused for the owner too."""
        parent_root, target, _ctx, entry = _captured_orphan(
            tmp_path, monkeypatch, terminal=False)
        owner = _disposer(tmp_path, monkeypatch, parent_root, target, task_id="t-nanny")

        out = _integrate_delegated_patch(owner, "run-1", "apply", "")
        assert "INTEGRATE_DELEGATED_TARGET_MISMATCH" in out, out
        assert not (target / "newfile.py").exists()
        assert entry.patch_disposed == ""
        custody._CUSTODY.clear()

    def test_a_live_owner_still_refuses_a_foreign_disposer(self, tmp_path, monkeypatch):
        parent_root, target, _ctx, entry = _captured_orphan(
            tmp_path, monkeypatch, terminal=False)
        disposer = _disposer(tmp_path, monkeypatch, parent_root, target)

        out = _integrate_delegated_patch(disposer, "run-1", "apply", "")
        assert "INTEGRATE_DELEGATED_NOT_OWNED" in out, out
        assert not (target / "newfile.py").exists()
        assert entry.patch_disposed == ""
        custody._CUSTODY.clear()


class TestOrphanCaptureRead:
    """The actor the orphan rule lets APPLY a foreign capture may also READ it.

    `artifact_store` is pinned to the CALLER's task id, so the sanctioned
    recovery root got `outside selected root=artifact_store` twice on the very
    patch it was authorized to dispose, and then escalated a manual file-attach
    question to the owner's phone.
    """

    def _reader(self, tmp_path, monkeypatch, target, *, task_id="t-second"):
        reader = _nanny_ctx(tmp_path, target, monkeypatch)
        reader.task_id = task_id
        return reader

    def test_terminal_owner_capture_reads_by_absolute_path(self, tmp_path, monkeypatch):
        from ouroboros.tools.core import _read_file

        target, _ctx, _entry, block = _captured_flat(tmp_path, monkeypatch)
        reader = self._reader(tmp_path, monkeypatch, target)
        patch = _read_file(reader, str(block["patch_artifact"]), root="artifact_store")
        assert "outside selected root=artifact_store" not in patch, patch
        assert "CHILD-EDIT" in patch
        manifest = _read_file(reader, str(block["manifest_artifact"]), root="artifact_store")
        assert "ready_with_changes" in manifest, manifest
        custody._CUSTODY.clear()

    def test_live_owner_capture_stays_outside_the_readers_artifact_store(self, tmp_path, monkeypatch):
        from ouroboros.tools.core import _read_file

        target, _ctx, _entry, block = _captured_flat(tmp_path, monkeypatch, terminal=False)
        reader = self._reader(tmp_path, monkeypatch, target)
        out = _read_file(reader, str(block["patch_artifact"]), root="artifact_store")
        assert "outside selected root=artifact_store" in out, out
        custody._CUSTODY.clear()

    def test_a_subagent_profile_never_reaches_a_foreign_capture(self, tmp_path, monkeypatch):
        from ouroboros.tools.core import _read_file

        target, _ctx, _entry, block = _captured_flat(tmp_path, monkeypatch)
        reader = self._reader(tmp_path, monkeypatch, target)
        reader.task_constraint = {"mode": "local_readonly_subagent"}
        out = _read_file(reader, str(block["patch_artifact"]), root="artifact_store")
        assert "outside selected root=artifact_store" in out, out
        custody._CUSTODY.clear()

    def test_a_foreign_path_outside_the_capture_prefix_stays_refused(self, tmp_path, monkeypatch):
        from ouroboros.tools.core import _read_file

        target, ctx, _entry, _block = _captured_flat(tmp_path, monkeypatch)
        from ouroboros.artifacts import task_artifact_dir_path
        from ouroboros.tool_access import canonical_data_root

        sibling = task_artifact_dir_path(canonical_data_root(ctx), "t-nanny", create=True) / "notes.txt"
        sibling.write_text("not a capture\n", encoding="utf-8")
        reader = self._reader(tmp_path, monkeypatch, target)
        out = _read_file(reader, str(sibling), root="artifact_store")
        assert "outside selected root=artifact_store" in out, out
        custody._CUSTODY.clear()

    def test_a_shared_snapshot_costs_zero_extra_reads(self, tmp_path, monkeypatch):
        """R19: the resolver reads the SHARED custody snapshot when it has one,
        and reads the owner task's rows exactly once (never twice, never every
        row) when it does not."""
        from ouroboros import delegate_custody_current as current
        from ouroboros.delegate_shared import orphan_capture_read_target
        from ouroboros.delegate_terminal import custody_audit_snapshot

        target, ctx, _entry, block = _captured_flat(tmp_path, monkeypatch)
        reader = self._reader(tmp_path, monkeypatch, target)
        drive = custody.custody_root(ctx)
        snapshot = custody_audit_snapshot(drive)

        real_replay, real_runs, calls, replays = custody.replay, current.runs, [], []

        def _counting_runs(*args, **kwargs):
            calls.append(args[1:])
            return real_runs(*args, **kwargs)

        monkeypatch.setattr(current, "runs", _counting_runs)
        monkeypatch.setattr(custody, "replay", lambda *a, **k: replays.append(1) or real_replay(*a, **k))
        custody._CUSTODY.clear()
        assert orphan_capture_read_target(
            reader, block["patch_artifact"], snapshot=snapshot) is not None
        assert calls == [], f"a shared snapshot must not read: {len(calls)}"

        custody._CUSTODY.clear()
        assert orphan_capture_read_target(reader, block["patch_artifact"]) is not None
        assert calls == [("task_id=?", ("t-nanny",))], f"one owner read without a snapshot, got {calls}"

        custody._CUSTODY.clear()
        assert orphan_capture_read_target(reader, target / "tracked.txt") is None
        assert len(calls) == 1, "a non-capture path must not read at all"
        assert replays == [], "no path replays every row"
        custody._CUSTODY.clear()


class TestCaptureUserFilesHint:
    def test_a_capture_path_names_the_route_that_exists(self, tmp_path, monkeypatch):
        from ouroboros.tool_access import user_files_path_block_reason

        target, ctx, _entry, block = _captured_flat(tmp_path, monkeypatch)
        reason = user_files_path_block_reason(
            ctx, pathlib.Path(block["patch_artifact"]), operation="read")
        assert "delegated-run capture owned by task t-nanny" in reason, reason
        assert "integrate_delegated_patch(run_id=...)" in reason
        assert "root=active_workspace" not in reason
        custody._CUSTODY.clear()

    def test_a_non_capture_control_path_keeps_the_generic_text(self, tmp_path, monkeypatch):
        from ouroboros.tool_access import user_files_path_block_reason

        target, ctx, _entry, _block = _captured_flat(tmp_path, monkeypatch)
        reason = user_files_path_block_reason(
            ctx, pathlib.Path(ctx.drive_root) / "memory" / "identity.md", operation="read")
        assert "root=active_workspace" in reason, reason
        assert "delegated-run capture" not in reason
        custody._CUSTODY.clear()
