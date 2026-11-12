"""R1: deleted inputs, independent clean edits, and source-bound write authority."""
from pathlib import Path

import pytest

from ouroboros import subagent_worktrees as trees
from ouroboros.workspace_copies import same_directory, source_is_system_repo, copy_apply_refusal
from ouroboros.tools.subagent_integration import _integrate_subagent_patch
from tests.test_isolated_project_copies import case as copy_case, admit, capture, parent_context, git, repo


def late_native_case(case, *, original=False, run_id="late-run"):
    from ouroboros.headless import finalize_task_artifacts
    from ouroboros.task_results import write_task_result

    folder, task = admit(case, "native", "parent")
    if original:
        (folder / "original.txt").write_bytes(b"native original edit\n")
        (folder / "original.bin").write_bytes(b"native\0binary\n")
    data = case[2]
    write_task_result(data, "native", "failed", **{k: v for k, v in task.items() if k != "id"})
    finalize_task_artifacts(data, task)
    return folder, task, late_run(case, folder, run_id)


def late_run(case, folder, run_id, path="late.txt", *, task_id="native", parent_id="parent", explicit_selection=False):
    from ouroboros import delegate_custody as custody
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive
    data = case[2]
    handle = trees.provision_execution_snapshot(target_root=folder, task_id=task_id, snapshot_id=run_id)
    (Path(handle.path) / path).write_bytes(b"late delegated edit\n")
    if explicit_selection:
        git(handle.path, "add", "-f", path)  # explicit selection by this fixture's original leaf author
    entry = custody.RunCustody(run_id=run_id, task_id=task_id, route_id="fake",
                              root_task_id="parent", parent_task_id=parent_id, snapshot_id=handle.snapshot_id,
                              execution_root=handle.path, baseline_sha=handle.baseline_sha,
                              target_root=str(folder), authority_source="self_worktree")
    assert custody.record_started(data, entry)
    assert custody.emit(data, custody.SETTLED, {"run_id": run_id, "task_id": task_id})
    entry = custody.replay(data)[run_id]
    assert capture_terminal_patch_for_drive(data, entry)["status"] == "ready_with_changes"
    return entry


def running_native_own_run(case, run_id, *, child_drive=False):
    """A RUNNING native child with its own direct edit and its own captured delegated run (#2790); a forked
    child gets the scheduler's drive (``prepare_task_drive``), the layout startup recovery addresses."""
    from ouroboros import headless
    from ouroboros.task_results import write_task_result

    folder, task = admit(case, "native", "parent")
    if child_drive:
        drive = str(headless.prepare_task_drive(case[2], "native", "forked"))
        task.update(drive_root=drive, child_drive_root=drive, budget_drive_root=str(case[2]))
    write_task_result(case[2], "native", "running", **{k: v for k, v in task.items() if k != "id"})
    (folder / "own.txt").write_bytes(b"child direct edit\n")
    return folder, task, late_run(case, folder, run_id)


def own_applied(case, run_id, *, child_drive=False):
    """``running_native_own_run`` after the child applied its own run itself (confirmed: effect=applied)."""
    from ouroboros.tools import subagent_integration as si

    folder, task, _ = running_native_own_run(case, run_id, child_drive=child_drive)
    child = parent_context(case, task_id="native", source=folder, task=task)
    if child_drive:  # a forked child works on its own drive; custody and its capture debt stay canonical
        child.drive_root, child.budget_drive_root = Path(task["child_drive_root"]), str(case[2])
    out = si._integrate_delegated_patch(child, run_id)
    assert "✅ Integrated" in out, out
    return folder, task


def finish_native(case, task, status="completed"):
    """The child's terminal row lands in its own drive; the worker then runs terminal file preparation."""
    from ouroboros import headless
    from ouroboros.task_results import write_task_result

    write_task_result(Path(task.get("child_drive_root") or case[2]), "native", status, result="child finished")
    return headless.prepare_terminal_task_files(case[2], task)


@pytest.mark.parametrize("decision", ["apply", "reject"])
def test_disposition_lock_failure_names_os_cause_without_an_effect(case, monkeypatch, decision):
    from ouroboros import delegate_custody as custody
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch
    from tests._shared import full_disk_at

    run_id = f"own-lock-{decision}"
    folder, task, entry = running_native_own_run(case, run_id)
    before = custody.output_disposition(entry)
    copy_present = Path(entry.execution_root).exists()  # capture may already have released the execution copy
    full_disk_at(monkeypatch, "disposition")
    result = _integrate_delegated_patch(parent_context(case, task_id="native", source=folder, task=task),
                                      run_id, decision=decision)
    assert "errno 28" in result and "nothing was applied or rejected" in result, result
    assert "another disposition" not in result
    assert custody.output_disposition(custody.replay(case[2])[run_id]) == before
    assert not (folder / "late.txt").exists()
    assert Path(entry.execution_root).exists() == copy_present


@pytest.mark.parametrize("resolution", ["refused", "unknown", "wrong_target"])
def test_pending_in_place_invocation_resolves_only_from_its_addressed_fate(case, resolution):
    from ouroboros import delegate_custody as custody, headless
    from ouroboros.task_results import load_task_result, write_task_result

    folder, task = admit(case, "native", "parent")
    (folder / "original.txt").write_bytes(b"native work before start\n")
    data = case[2]
    write_task_result(data, "native", "completed", **{k: v for k, v in task.items() if k != "id"})
    request = {"task_id": "native", "invocation_id": "pending-start", "request": {"prompt": "work"},
               "execution_root": str(folder), "target_root": str(folder),
               "resource_ref": {"workspace_kind": "git", "strategy": "in_place"}}
    assert custody.emit(data, custody.START_REQUESTED, request)
    headless.finalize_task_artifacts(data, task)
    assert load_task_result(data, "native")["native_capture_pending"]
    if resolution in {"wrong_owner", "wrong_target"}:
        # A pending source must still match the addressed request before a no-run refusal can release it.
        row = load_task_result(data, "native")
        pending = row["native_capture_pending"]
        if resolution == "wrong_owner":
            pending["invocation:pending-start"]["invocation_id"] = "foreign-start"
            assert custody.emit(data, custody.START_REQUESTED, {**request, "task_id": "foreign", "invocation_id": "foreign-start"})
            assert custody.emit(data, custody.START_FAILED, {"invocation_id": "foreign-start", "definite": True})
        else:
            pending["invocation:pending-start"]["target_root"] = str(case[1])
        write_task_result(data, "native", "completed", native_capture_pending=pending)
    assert custody.emit(data, custody.START_FAILED, {"task_id": "native", "invocation_id": "pending-start",
                                                   "definite": resolution != "unknown"})
    if resolution == "started":
        entry = custody.RunCustody(run_id="bound-run", task_id="native", route_id="fake", invocation_id="pending-start",
                                  execution_root=str(folder), target_root=str(folder),
                                  resource_ref=request["resource_ref"])
        assert custody.record_started(data, entry)
        out = _integrate_subagent_patch(parent_context(case), task_id="native")
        assert "in-place writer live or unknown" in out, out
        assert custody.emit(data, custody.TERMINAL_OBSERVED, {"run_id": entry.run_id, "task_id": "native",
                                                           "state": "succeeded", "observation_seq": 1,
                                                           "source_ref": {"sha256": "a" * 64}})
    out = _integrate_subagent_patch(parent_context(case), task_id="native")
    if resolution in {"unknown", "wrong_owner", "wrong_target"}:
        assert "unavailable" in out and "refreshed" not in out, out
        assert load_task_result(data, "native")["native_capture_pending"]
    else:
        assert "refreshed" in out, out
        assert not load_task_result(data, "native")["native_capture_pending"]
        assert "✅ Integrated" in _integrate_subagent_patch(parent_context(case), task_id="native")
        assert (case[1] / "original.txt").read_bytes() == b"native work before start\n"


@pytest.mark.parametrize("publication_fails", [True, False])
def test_retirement_rechecks_capture_after_waiting_for_native_custody(case, monkeypatch, publication_fails):
    from contextlib import contextmanager
    from ouroboros import headless, owner_keep
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    folder, task, entry = late_native_case(case)
    gate, publisher = owner_keep.gate, headless.write_workspace_patch_artifacts
    entered, inside = [], []

    @contextmanager
    def late_apply_before_gate(root, task_id, **kwargs):
        if task_id == "native" and not entered:
            entered.append(task_id)
            with monkeypatch.context() as publish:
                if publication_fails:
                    publish.setattr(headless, "write_workspace_patch_artifacts",
                                    lambda *a, **kw: (_ for _ in ()).throw(OSError("publication unavailable")))
                out = _integrate_delegated_patch(parent_context(case, task_id="orphan-disposer"), entry.run_id)
                assert "✅ Integrated" in out, out
            assert (folder / "late.txt").is_file()
        with gate(root, task_id, **kwargs) as why:
            inside.append(task_id)
            try:
                yield why
            finally:
                inside.pop()

    def live(task_id):
        assert not inside, "queue liveness cannot be probed under task custody"
        return False

    monkeypatch.setattr(owner_keep, "gate", late_apply_before_gate)
    out = trees.prune_orphans(worktree_root=case[3], data_dir=case[2], retention_days=0, live=live, only={"native"})
    assert entered == ["native"]
    assert out["removed"] == (0 if publication_fails else 1), out
    assert folder.exists() == publication_fails
    if publication_fails:
        assert load_task_result(case[2], "native")["native_capture_pending"]
        assert trees.registered_checkout(folder, case[2])
        assert "refreshed" in headless.refresh_native_capture(case[2], "native")
        assert headless.write_workspace_patch_artifacts is publisher
        assert trees.prune_orphans(worktree_root=case[3], data_dir=case[2], retention_days=0,
                                   live=live, only={"native"})["removed"] == 1


@pytest.mark.parametrize("decision", ["apply", "reject"])
@pytest.mark.parametrize("refresh", [True, False])
def test_native_decision_binds_consumed_capture_during_orphan_apply(case, monkeypatch, decision, refresh):
    import json
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.delegate_custody_current import marks
    from ouroboros.task_results import write_task_result
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools import subagent_integration as integration
    from ouroboros.tools.join_ledger import _child_result_sha256, _current_child_result_disposition

    folder, task, entry = late_native_case(case, original=True)
    write_task_result(case[2], "native", "failed", delegation_role="subagent", root_task_id="parent")
    before = _child_result_sha256(load_effective_task_result(case[2], "native"))
    locate = integration._locate_child_patch

    def locate_then_refresh(ctx, tid):
        located = locate(ctx, tid)
        if refresh:
            disposer = parent_context(case, task_id="orphan-disposer")
            assert "✅ Integrated" in integration._integrate_delegated_patch(disposer, entry.run_id)
        return located

    monkeypatch.setattr(integration, "_locate_child_patch", locate_then_refresh)
    out = integration._integrate_subagent_patch(parent_context(case), task_id="native", decision=decision,
                                               reason="consume exactly the observed capture")
    assert ("✅ Integrated" if decision == "apply" else "Rejected") in out, out
    current = load_effective_task_result(case[2], "native")
    assert (_child_result_sha256(current) != before) == refresh
    assert not (case[1] / "late.txt").exists()
    assert (case[1] / "original.txt").exists() == (decision == "apply")
    disposition = _current_child_result_disposition(current)
    assert disposition == ("" if refresh else "integrated" if decision == "apply" else "irrelevant"), (out, disposition)
    if refresh:
        assert "CHILD_RESULT_STALE" in out, out
    verdict = task_artifact_dir_path(case[2], "parent") / "subagent_patch_verdict_native.json"
    assert json.loads(verdict.read_bytes())["child_result_sha256"] == before
    assert marks(case[2], "parent")["patch_verdicts"][-1]["child_result_sha256"] == before


def assert_current_consumers(case, task, expected_status, *, expected_path="", old_bytes=()):
    import json
    from types import SimpleNamespace
    from ouroboros import cli, copy_capture_proof, headless, owner_stop_receipt
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.control_task_results import _get_task_result
    from ouroboros.tools.subagent_integration import _compare_subagent_patches
    from tests.test_owner_stop_retained_work import _client
    ctx = parent_context(case)
    data = case[2]
    row = load_task_result(data, "native")
    source = headless.current_workspace_manifest(data, "native", row)
    before = source.read_bytes()
    view = json.loads(_get_task_result(ctx, "native", view="evidence"))
    assert view["artifact_facts"]["status"] == expected_status, view
    comparison = _compare_subagent_patches(ctx, ["native"])
    assert f"patch status: {expected_status}" in comparison, comparison
    with _client(data) as client:
        response = client.get("/api/tasks/native")
        assert response.status_code == 200, response.text
        public = response.json()
        assert public["artifact_status"] == public["artifact_bundle"]["status"] == expected_status
        assert public["status"] == row["status"]
        def get_bytes(path):
            answer = client.get(path)
            assert answer.status_code == 200, answer.text
            return answer.content
        if expected_status == "failed":
            with pytest.raises(cli.PatchCLIError):
                cli._patch_from_result(SimpleNamespace(get_bytes=get_bytes), "native", public, strict=False)
        else:
            assert expected_path in cli._patch_from_result(SimpleNamespace(get_bytes=get_bytes), "native", public, strict=True)
            assert expected_path in comparison
        for path, content in dict(old_bytes).items():
            relpath = Path(path).relative_to(headless.task_artifacts_dir(data, "native")).as_posix()
            archived = client.get(f"/api/tasks/native/artifacts/{Path(path).name}", params={"relpath": relpath})
            assert archived.status_code == 200 and archived.content == content
    receipt = owner_stop_receipt.capture_receipt(data, "parent", {"native": row["status"]})
    reader = receipt[owner_stop_receipt.FIELD]["children"][0]["result_read"]
    assert expected_status in _get_task_result(ctx, **reader["arguments"])
    assert source.read_bytes() == before
    proof = copy_capture_proof.plain_copy_verdict({"path": task["workspace_root"], "task_id": "native",
                                                 "base_sha": task["task_constraint"]["base_sha"]}, data)
    assert proof["remove"] == (expected_status != "failed"), proof


@pytest.mark.parametrize("original", [False, True])
def test_late_delegated_apply_refreshes_native_result_and_keeps_old_artifacts(case, monkeypatch, original):
    import json
    import os
    import shutil
    import tempfile
    from hashlib import sha256
    from ouroboros import headless
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.join_ledger import _child_result_sha256
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch, _locate_child_patch
    from ouroboros.workspace_file_outputs import file_output_changes
    from ouroboros.tools.patch_verdict import retained_verdicts

    folder, task, entry = late_native_case(case, original=original)
    before = load_task_result(case[2], "native")
    old_bytes = {row["path"]: Path(row["path"]).read_bytes() for row in before["artifacts"]}
    ctx = parent_context(case)
    old_decisions = []
    if original:
        assert "Rejected" in _integrate_subagent_patch(ctx, task_id="native", decision="reject", reason="old subject only")
        old_decisions = list(retained_verdicts(case[2], "parent", "native"))
        assert old_decisions and old_decisions[0]["outcome"] == "rejected"
    out = _integrate_delegated_patch(ctx, entry.run_id)
    assert "✅ Integrated" in out, out
    patch, manifest, result = _locate_child_patch(ctx, "native")
    assert manifest["status"] == "ready_with_changes"
    assert "late.txt" in manifest["untracked_included"] + manifest["tracked_changed"]
    assert result["status"] == "failed"
    assert result["artifact_status"] == result["artifact_bundle"]["status"] == manifest["status"]
    assert manifest["base_head"] == task["task_constraint"]["base_sha"]
    assert result["metadata"]["workspace_copy"] == task["workspace_copy"]
    assert _child_result_sha256(result) != _child_result_sha256(before)
    for path, content in old_bytes.items():
        assert Path(path).read_bytes() == content
    old_manifest = next(Path(row["path"]) for row in before["artifacts"] if row["kind"] == "workspace_patch_manifest")
    file_output_changes(json.loads(old_manifest.read_bytes()), old_manifest.parent)
    with monkeypatch.context() as reads:
        reads.setattr(headless, "write_workspace_patch_artifacts", lambda *a, **kw: pytest.fail("GET captured the tree"))
        assert_current_consumers(case, task, "ready_with_changes", expected_path="late.txt", old_bytes=old_bytes)
    assert "✅ Integrated" in _integrate_subagent_patch(ctx, task_id="native")
    assert (case[1] / "late.txt").read_bytes() == b"late delegated edit\n"
    if original:
        assert (case[1] / "original.txt").read_bytes() == b"native original edit\n"
        assert (case[1] / "original.bin").read_bytes() == b"native\0binary\n"
        assert all(old in list(retained_verdicts(case[2], "parent", "native")) for old in old_decisions)
    if evidence := os.environ.get("OUROBOROS_TEST_EVIDENCE_ROOT"):
        kept = Path(tempfile.mkdtemp(prefix=f"native-capture-{original}-", dir=evidence))
        base = headless.task_artifacts_dir(case[2], "native")
        shutil.copytree(base, kept / "artifacts")
        (kept / "validation.json").write_text(json.dumps({
            "original_native_base": task["task_constraint"]["base_sha"], "current_manifest_base": manifest["base_head"],
            "original_file_baseline_unchanged": result["metadata"]["workspace_copy"] == task["workspace_copy"],
            "current_manifest": str(headless.current_workspace_manifest(case[2], "native", result).relative_to(base)),
            "old_artifacts": {str(Path(path).relative_to(base)): sha256(data).hexdigest() for path, data in old_bytes.items()},
            "old_decisions": old_decisions, "consumer_checks": ["get_task_result", "HTTP", "CLI", "compare", "Stop", "copy_capture_proof", "native apply"],
        }, indent=2))


def test_late_publication_failure_is_repaired_for_inspection_without_reapplying(case, monkeypatch):
    from ouroboros import headless
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch
    folder, task, entry = late_native_case(case)
    publisher = headless.write_workspace_patch_artifacts
    monkeypatch.setattr(headless, "write_workspace_patch_artifacts",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("publication unavailable")))
    out = _integrate_delegated_patch(parent_context(case), entry.run_id)
    assert "✅ Integrated" in out and "publication is still owed" in out, out
    row = load_task_result(case[2], "native")
    assert row["status"] == "failed" and row["artifact_status"] == row["artifact_bundle"]["status"] == "failed"
    assert row["native_capture_pending"][entry.run_id]["effect"] == "applied"
    assert_current_consumers(case, task, "failed")
    monkeypatch.setattr(headless, "write_workspace_patch_artifacts", publisher)
    out = _integrate_subagent_patch(parent_context(case), task_id="native")
    assert "INTEGRATE_NATIVE_CAPTURE" in out and "refreshed" in out, out
    assert not (case[1] / "late.txt").exists()
    assert (folder / "late.txt").read_bytes() == b"late delegated edit\n"
    assert not load_task_result(case[2], "native")["native_capture_pending"]
    out = _integrate_subagent_patch(parent_context(case), task_id="native")
    assert "✅ Integrated" in out and (case[1] / "late.txt").is_file(), out


def test_late_capture_refresh_requires_current_read_authority(case, monkeypatch):
    from ouroboros.task_results import load_task_result
    from ouroboros.tools import subagent_integration as si
    folder, task, entry = late_native_case(case)
    reader = si._shared_read_refusal
    monkeypatch.setattr(si, "_shared_read_refusal", lambda *a, **kw: "native copy read denied")
    out = si._integrate_delegated_patch(parent_context(case), entry.run_id)
    assert "✅ Integrated" in out and "native copy read denied" in out, out
    assert load_task_result(case[2], "native")["artifact_status"] == "failed"
    assert "INTEGRATE_TARGET_FORBIDDEN" in _integrate_subagent_patch(parent_context(case), task_id="native")
    monkeypatch.setattr(si, "_shared_read_refusal", reader)
    assert "refreshed" in _integrate_subagent_patch(parent_context(case), task_id="native")
    assert not (case[1] / "late.txt").exists() and (folder / "late.txt").exists()


def test_run_target_mismatch_never_borrows_native_publication_authority(case, monkeypatch):
    from ouroboros import headless
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch
    folder, task, entry = late_native_case(case)
    other, _ = admit(case, "other", "parent")
    foreign = late_run(case, other, "wrong-target", "foreign.txt")
    before = load_task_result(case[2], "native")["artifacts"]
    monkeypatch.setattr(headless, "refresh_native_capture", lambda *a, **kw: pytest.fail("unrelated writer refreshed native copy"))
    out = _integrate_delegated_patch(parent_context(case), foreign.run_id)
    assert "✅ Integrated" in out and "promotion refused" in out, out
    assert (other / "foreign.txt").exists() and not (folder / "foreign.txt").exists()
    assert load_task_result(case[2], "native")["artifacts"] == before


def test_multiple_late_applies_and_stale_copyback_preserve_current_version(case):
    from ouroboros import headless
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch, _locate_child_patch
    folder, task, first = late_native_case(case, original=True)
    stale = load_task_result(case[2], "native")
    ctx = parent_context(case)
    assert "✅ Integrated" in _integrate_delegated_patch(ctx, first.run_id)
    once = load_task_result(case[2], "native")
    old_bytes = {row["path"]: Path(row["path"]).read_bytes() for row in once["artifacts"]}
    second = late_run(case, folder, "second-run", "second.txt")
    assert "✅ Integrated" in _integrate_delegated_patch(ctx, second.run_id)
    patch, manifest, current = _locate_child_patch(ctx, "native")
    assert {"original.txt", "late.txt", "second.txt"} <= set(manifest["untracked_included"] + manifest["tracked_changed"])
    replica = case[2].parent / "replica"
    stale["artifacts"] = [{k: v for k, v in row.items() if k != "immutable"} for row in stale["artifacts"]]
    write_task_result(replica, "native", stale["status"], **{k: v for k, v in stale.items() if k not in {"task_id", "status"}})
    headless.copy_child_task_result(case[2], {**task, "child_drive_root": str(replica)})
    after_patch, after_manifest, after = _locate_child_patch(ctx, "native")
    assert after_patch == patch and after_manifest == manifest
    assert after["artifact_bundle"]["status"] == "ready_with_changes" and not after["native_capture_pending"]
    assert all(row["immutable"] for row in after["artifacts"] if row["kind"] == "workspace_capture_history" and row.get("sha256"))
    assert_current_consumers(case, task, "ready_with_changes", expected_path="second.txt", old_bytes=old_bytes)
    for path, content in old_bytes.items():
        assert Path(path).read_bytes() == content
    assert "✅ Integrated" in _integrate_subagent_patch(ctx, task_id="native")
    assert (case[1] / "second.txt").read_bytes() == (case[1] / "late.txt").read_bytes()


def test_late_manifest_write_failure_blocks_apply_and_keeps_publication_addressable(case, monkeypatch):
    from ouroboros import headless, delegate_custody_current as current
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch, _locate_child_patch
    folder, task, entry = late_native_case(case)
    writer = headless.atomic_write_json
    def unavailable(path, *args, **kwargs):
        if path.name == "workspace_patch.json":
            raise OSError("manifest store unavailable")
        return writer(path, *args, **kwargs)
    monkeypatch.setattr(headless, "atomic_write_json", unavailable)
    out = _integrate_delegated_patch(parent_context(case), entry.run_id)
    assert "INTEGRATE_NATIVE_CAPTURE_UNAVAILABLE" in out and not (folder / "late.txt").exists(), out
    assert not current.run(case[2], entry.run_id).patch_apply_pending
    row = load_task_result(case[2], "native")
    assert row["native_capture_pending"][entry.run_id]["effect"] == "unmutated"
    assert row["artifact_status"] == row["artifact_bundle"]["status"] == "failed"
    assert _locate_child_patch(parent_context(case), "native")[1]["status"] == "failed"
    monkeypatch.setattr(headless, "atomic_write_json", writer)
    out = _integrate_subagent_patch(parent_context(case), task_id="native")
    assert "refreshed" in out and not (folder / "late.txt").exists(), out
    row = load_task_result(case[2], "native")
    assert row["artifact_status"] == row["artifact_bundle"]["status"] == "ready_no_changes"
    assert "✅ Integrated" in _integrate_delegated_patch(parent_context(case), entry.run_id)
    assert "✅ Integrated" in _integrate_subagent_patch(parent_context(case), task_id="native")


def test_late_apply_drift_refusal_refreshes_only_the_confirmed_unchanged_apply(case):
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch, _locate_child_patch
    folder, task, entry = late_native_case(case)
    (folder / "late.txt").write_bytes(b"native owner changed the target first\n")
    out = _integrate_delegated_patch(parent_context(case), entry.run_id)
    assert "INTEGRATE_CONFLICT" in out and "refreshed" in out, out
    row = load_task_result(case[2], "native")
    assert not row["native_capture_pending"] and row["artifact_status"] == "ready_with_changes"
    patch, _, _ = _locate_child_patch(parent_context(case), "native")
    assert b"native owner changed the target first" in patch.read_bytes()
    assert b"late delegated edit" not in patch.read_bytes()


def test_received_paths_union_on_native_owner_survives_stale_task_capture(case):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from ouroboros import headless
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.workspace_file_outputs import file_output_changes
    from ouroboros.tools.subagent_integration import _locate_child_patch
    (case[1] / ".gitignore").write_text("*.ignored\n")
    git(case[1], "add", ".gitignore"); git(case[1], "commit", "-qm", "ignore outputs")
    folder, task = admit(case, "native", "parent")
    write_task_result(case[2], "native", "completed", **{k: v for k, v in task.items() if k != "id"})
    for name in ("one.ignored", "two.ignored"):
        (folder / name).write_bytes(b"delivered unstaged\n")
    gate = Barrier(2)
    def record(name):
        gate.wait()
        headless.record_native_received_files(case[2], "native", folder, [name], {"run_id": name})
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(record, ["one.ignored", "two.ignored"]))
    headless.record_native_received_files(case[2], "parent", folder, ["wrong.ignored"], {"run_id": "wrong"})
    assert load_task_result(case[2], "parent") is None
    assert set(load_task_result(case[2], "native")["received_file_outputs"]) == {"one.ignored", "two.ignored"}
    headless.finalize_task_artifacts(case[2], task)  # the admitted task predates both receipts
    patch, manifest, row = _locate_child_patch(parent_context(case), "native")
    assert {item["path"] for item in file_output_changes(manifest, patch.parent)} == {"one.ignored", "two.ignored"}
    assert row["workspace_copy"] == task["workspace_copy"] and manifest["patch_size"] == 0
    assert git(folder, "ls-files", "one.ignored", "two.ignored") == b""


def test_ignored_delivery_crosses_two_native_captures_without_staging(case):
    from ouroboros.headless import finalize_task_artifacts
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools.subagent_integration import _locate_child_patch, _integrate_delegated_patch
    (case[1] / ".gitignore").write_text("*.ignored\n")
    git(case[1], "add", ".gitignore"); git(case[1], "commit", "-qm", "ignore outputs")
    outer, outer_task = admit(case, "outer", "parent")
    middle, middle_task = admit(case, "middle", "outer", outer)
    leaf, leaf_task = admit(case, "leaf", "middle", middle)
    for task in (outer_task, middle_task, leaf_task):
        write_task_result(case[2], task["id"], "completed", **{k: v for k, v in task.items() if k != "id"})
    finalize_task_artifacts(case[2], leaf_task)
    run = late_run(case, leaf, "ignored-run", "delivered.ignored", task_id="leaf", parent_id="middle", explicit_selection=True)
    out = _integrate_delegated_patch(parent_context(case, task_id="leaf", source=leaf, task=leaf_task), run.run_id)
    assert "✅ Integrated" in out and git(leaf, "ls-files", "delivered.ignored") == b"", out
    for child, target, task in (("leaf", middle, middle_task), ("middle", outer, outer_task)):
        ctx = parent_context(case, task_id=task["id"], source=target, task=task)
        out = _integrate_subagent_patch(ctx, task_id=child)
        assert "✅ Integrated" in out, out
        assert git(target, "ls-files", "delivered.ignored") == b""
        assert "delivered.ignored" in load_task_result(case[2], task["id"])["received_file_outputs"]
        finalize_task_artifacts(case[2], task)
        _, manifest, _ = _locate_child_patch(parent_context(case), task["id"])
        assert manifest["file_output_changes"][0]["path"] == "delivered.ignored" and manifest["patch_size"] == 0
    assert "✅ Integrated" in _integrate_subagent_patch(parent_context(case), task_id="outer")
    assert (case[1] / "delivered.ignored").read_bytes() == b"late delegated edit\n"
    assert git(case[1], "ls-files", "delivered.ignored") == b""


@pytest.mark.parametrize("after_effect", [False, True])
def test_late_apply_crash_keeps_unknown_even_when_tree_can_be_read(case, monkeypatch, after_effect):
    from ouroboros import delegate_custody_current as current
    from ouroboros.task_results import load_task_result
    from ouroboros.tools import subagent_integration as si
    folder, task, entry = late_native_case(case)
    apply = si._locked_apply
    def crash(*args, **kwargs):
        if after_effect:
            apply(*args, **kwargs)
        raise OSError("apply process disappeared")
    monkeypatch.setattr(si, "_locked_apply", crash)
    with pytest.raises(OSError, match="disappeared"):
        si._integrate_delegated_patch(parent_context(case), entry.run_id)
    row = load_task_result(case[2], "native")
    assert row["artifact_status"] == row["artifact_bundle"]["status"] == "failed"
    assert current.run(case[2], entry.run_id).patch_apply_pending
    assert (folder / "late.txt").exists() == after_effect
    from ouroboros.copy_capture_proof import plain_copy_verdict
    assert not plain_copy_verdict({"path": str(folder), "task_id": "native",
                                   "base_sha": task["task_constraint"]["base_sha"]}, case[2])["remove"]
    out = _integrate_subagent_patch(parent_context(case), task_id="native")
    assert "apply outcome unknown" in out and not (case[1] / "late.txt").exists(), out


@pytest.mark.parametrize("authority", ["cancelled", "readonly", "wrong_target"])
def test_late_capture_never_promotes_an_unrelated_or_cancelled_owner(case, authority):
    from ouroboros.headless import native_capture_owner
    from ouroboros.task_results import load_task_result
    from ouroboros.task_store import patch_document
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch
    folder, task, entry = late_native_case(case)
    before = load_task_result(case[2], "native")
    if authority == "cancelled":
        patch_document(case[2], "native", lambda row: {**row, "status": "cancelled"})
    else:
        constraint = {**task["task_constraint"], **({"mode": "local_readonly_subagent"} if authority == "readonly"
                      else {"write_root": str(case[1])})}
        patch_document(case[2], "native", lambda row: {**row, "task_constraint": constraint})
        assert not native_capture_owner(case[2], "native", folder)
    out = _integrate_delegated_patch(parent_context(case), entry.run_id)
    assert "✅ Integrated" in out, out  # the separate run's existing orphan authority
    row = load_task_result(case[2], "native")
    if authority == "cancelled":
        assert "promotion refused" in out and row["artifact_status"] == "failed"
        assert "INTEGRATE_NATIVE_CANCELLED" in _integrate_subagent_patch(parent_context(case), task_id="native")
    else:
        assert row["artifacts"] == before["artifacts"]


@pytest.mark.parametrize("child_drive", [False, True], ids=["canonical", "forked_drive"])
def test_running_native_own_apply_is_current_at_its_terminal(case, monkeypatch, child_drive):
    """#2790: a RUNNING native child applies its own delegated run. Its terminal step pays the recorded
    capture debt with a fresh version beside the in-flight gap, so the parent's first integrate applies;
    no step repeats the run's apply."""
    import json
    from ouroboros import delegate_custody_current as current, headless
    from ouroboros.task_results import load_task_result
    from ouroboros.tools import subagent_integration as si

    run_id = f"own-{'forked' if child_drive else 'canonical'}-run"
    folder, task = own_applied(case, run_id, child_drive=child_drive)
    row = load_task_result(case[2], "native")
    assert row["status"] == "running" and row["artifact_status"] == "pending"  # owed to the live owner's terminal step
    assert row["native_capture_pending"][run_id]["effect"] == "applied"
    gap = {item["path"]: Path(item["path"]).read_bytes() for item in row["artifacts"]}
    applies, apply = [], si._locked_apply
    monkeypatch.setattr(si, "_locked_apply", lambda *a, **kw: applies.append(a[1]) or apply(*a, **kw))
    report = finish_native(case, task)
    assert not report["error"] and applies == [], report
    row = load_task_result(case[2], "native")
    assert row["artifact_status"] == "ready_with_changes" and not row["native_capture_pending"], row
    assert headless.terminal_task_files_ready(case[2], task, row)
    assert any(item["kind"] == "memory_export" for item in row["artifacts"]) == child_drive
    fresh = headless.current_workspace_manifest(case[2], "native", row)
    assert str(fresh) not in gap and any(item["kind"] == "workspace_capture_history" for item in row["artifacts"])
    manifest = json.loads(fresh.read_bytes())
    assert manifest["native_capture_sources"][run_id]["effect"] == "applied"
    assert {"own.txt", "late.txt"} <= set(manifest["untracked_included"] + manifest["tracked_changed"])
    assert current.run(case[2], run_id).patch_disposed == "applied"
    with monkeypatch.context() as reads:
        reads.setattr(headless, "write_workspace_patch_artifacts", lambda *a, **kw: pytest.fail("GET captured the tree"))
        assert_current_consumers(case, task, "ready_with_changes", expected_path="late.txt", old_bytes=gap)
    out = _integrate_subagent_patch(parent_context(case), task_id="native")
    assert "✅ Integrated" in out and applies == [case[1]], out
    for name, body in (("own.txt", b"child direct edit\n"), ("late.txt", b"late delegated edit\n")):
        assert (case[1] / name).read_bytes() == (folder / name).read_bytes() == body


@pytest.mark.parametrize("control", ["ambiguous", "refused", "partial", "cancelled"])
def test_running_native_apply_controls_keep_their_outcome_at_terminal(case, monkeypatch, control):
    """#2790 controls: the terminal step pays only debt it can settle. A crash after the effect or a partial
    staging nobody could undo stays an honest failed capture; a refused apply captures the unmutated tree; a
    cancelled owner is never promoted."""
    from ouroboros.task_results import load_task_result
    from ouroboros.tools import subagent_integration as si, subagent_integration_delegated as delegated

    run_id = f"own-{control}-run"
    folder, task, _ = running_native_own_run(case, run_id)
    child = parent_context(case, task_id="native", source=folder, task=task)
    with monkeypatch.context() as patch:
        if control == "ambiguous":
            apply = si._locked_apply

            def crash(*args, **kwargs):
                apply(*args, **kwargs)
                raise OSError("apply process disappeared")

            patch.setattr(si, "_locked_apply", crash)
            with pytest.raises(OSError, match="disappeared"):
                si._integrate_delegated_patch(child, run_id)
        else:
            if control == "refused":
                (folder / "late.txt").write_bytes(b"conflicting local edit\n")
            elif control == "partial":
                patch.setattr(delegated, "stage_applied", lambda *a, **kw: ("index write failed", []))
                patch.setattr(delegated, "undo_applied", lambda *a, **kw: (False, "rollback incomplete"))
            out = si._integrate_delegated_patch(child, run_id)
            expected = {"refused": "INTEGRATE_CONFLICT", "partial": "INTEGRATE_APPLIED_UNSTAGED"}
            assert expected.get(control, "✅ Integrated") in out, out
    assert not finish_native(case, task, "cancelled" if control == "cancelled" else "completed")["error"]
    row = load_task_result(case[2], "native")
    if control == "refused":
        assert row["artifact_status"] == "ready_with_changes" and not row["native_capture_pending"], row
        assert_current_consumers(case, task, "ready_with_changes", expected_path="late.txt")
        assert "✅ Integrated" in _integrate_subagent_patch(parent_context(case), task_id="native")
        assert (case[1] / "late.txt").read_bytes() == b"conflicting local edit\n"
        return
    # A cancelled owner's capture stays owed and is never paid; the others are captured as honest failures.
    assert row["artifact_status"] == ("pending" if control == "cancelled" else "failed"), row
    assert row["native_capture_pending"][run_id], row
    if control != "cancelled":
        assert_current_consumers(case, task, "failed")
    out = _integrate_subagent_patch(parent_context(case), task_id="native")
    expected = {"ambiguous": "apply outcome unknown", "partial": "apply confirmation unavailable",
                "cancelled": "INTEGRATE_NATIVE_CANCELLED"}
    assert expected[control] in out and not (case[1] / "late.txt").exists(), out


@pytest.mark.parametrize("consumer", ["startup", "startup_forked", "reaper", "parent_wait"])
def test_running_native_own_apply_stays_owed_until_a_terminal_consumer_captures_it(case, monkeypatch, consumer):
    """#2790: the terminal row of a native child that applied its own run lands, but its worker's terminal step does
    not capture it (it met a busy publisher, then died; a forked worker died after copy-back; or the step is still
    to come). Every terminal consumer sees the capture owed: readers keep the child running, startup recovery or the
    reaper captures it once without re-applying, the old gap stays addressable, and the parent's first integrate
    applies."""
    import json
    from types import SimpleNamespace
    from ouroboros import headless, obligations
    from ouroboros.startup_task_files import recover_terminal_task_files
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools import subagent_integration as si
    from supervisor.task_reaper import _load_post_kill_terminal_result
    from tests.test_workspace_capture_recovery import _busy_custody, _consumers

    run_id, forked = f"own-{consumer}-run", consumer == "startup_forked"
    folder, task = own_applied(case, run_id, child_drive=forked)
    obligations.drive_started(case[2], task)  # the pool handoff
    gap = {item["path"]: Path(item["path"]).read_bytes() for item in load_task_result(case[2], "native")["artifacts"]}
    captures, applies, capture, apply = [], [], headless.write_workspace_patch_artifacts, si._locked_apply
    monkeypatch.setattr(headless, "write_workspace_patch_artifacts",
                        lambda *a, **kw: captures.append(a[0]) or capture(*a, **kw))
    monkeypatch.setattr(si, "_locked_apply", lambda *a, **kw: applies.append(a[1]) or apply(*a, **kw))
    write_task_result(Path(task.get("child_drive_root") or case[2]), "native", "completed", result="child finished")
    if forked:
        headless.copy_child_task_result(case[2], task)  # its preparation's first step; then the worker died
    elif consumer != "parent_wait":
        monkeypatch.setattr(headless, "PUBLICATION_LOCK_SEC", 0)
        with _busy_custody(case[2], "native"):
            assert headless.prepare_terminal_task_files(case[2], task)["error"].startswith("CustodyBusy")
    row = load_task_result(case[2], "native")
    assert row["status"] == "completed" and row["artifact_status"] == "pending" and not captures, row
    assert not headless.terminal_task_files_ready(case[2], task, row)
    assert "native" in obligations.members(case[2], "pending_drives")
    if consumer == "parent_wait":
        _consumers(case[2], "native", "running")  # the parent's wait, list and detail stay read-only
        assert not headless.prepare_terminal_task_files(case[2], task)["error"]  # the worker's step after post-work
    elif consumer == "reaper":
        for _ in range(2):  # the second load finds the published capture and prepares nothing
            assert _load_post_kill_terminal_result(SimpleNamespace(DRIVE_ROOT=case[2]), task, "native")[0] \
                == "completed"
    else:
        reports = [recover_terminal_task_files(case[2], set()) for _ in range(2)]
        assert [report["recovered"] for report in reports] == [["native"], []], reports
        assert not any(report["unresolved"] or report["errors"] for report in reports), reports
    row = load_task_result(case[2], "native")
    assert row["artifact_status"] == "ready_with_changes" and not row["native_capture_pending"], row
    assert len(captures) == 1 and applies == [], (captures, applies)
    assert "native" not in obligations.members(case[2], "pending_drives")
    assert any(item["kind"] == "memory_export" for item in row["artifacts"]) == forked
    manifest = json.loads(headless.current_workspace_manifest(case[2], "native", row).read_bytes())
    assert manifest["native_capture_sources"][run_id]["effect"] == "applied"
    assert {"own.txt", "late.txt"} <= set(manifest["untracked_included"] + manifest["tracked_changed"])
    assert all(Path(path).read_bytes() == body for path, body in gap.items())
    if consumer == "parent_wait":
        _consumers(case[2], "native", "completed")
    out = _integrate_subagent_patch(parent_context(case), task_id="native")
    assert "✅ Integrated" in out and applies == [case[1]], out
    for name, body in (("own.txt", b"child direct edit\n"), ("late.txt", b"late delegated edit\n")):
        assert (case[1] / name).read_bytes() == body


@pytest.mark.parametrize("control", ["capture_failure", "ambiguous", "cancelled"])
def test_owed_native_capture_recovery_keeps_honest_outcomes_without_churn(case, monkeypatch, control):
    """#2790 recovery controls: a genuine capture failure of the terminal step stays terminal and is never retried by
    recovery; an apply of unknown outcome is settled once as an honest failed capture and never applied; a cancelled
    owner is neither captured nor promoted. A repeated recovery pass finalizes nothing more."""
    from ouroboros import headless, obligations
    from ouroboros.startup_task_files import recover_terminal_task_files
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools import subagent_integration as si

    run_id = f"own-{control}-recovery-run"
    if control == "ambiguous":
        folder, task, _ = running_native_own_run(case, run_id)
        apply = si._locked_apply

        def crash(*args, **kwargs):
            apply(*args, **kwargs)
            raise OSError("apply process disappeared")

        with monkeypatch.context() as patch:
            patch.setattr(si, "_locked_apply", crash)
            with pytest.raises(OSError, match="disappeared"):
                si._integrate_delegated_patch(parent_context(case, task_id="native", source=folder, task=task), run_id)
    else:
        folder, task = own_applied(case, run_id)
    obligations.drive_started(case[2], task)
    finalized, finalize = [], headless._finalize_task_artifacts_locked
    monkeypatch.setattr(headless, "_finalize_task_artifacts_locked",
                        lambda *a, **kw: finalized.append(a[2]) or finalize(*a, **kw))
    write_task_result(case[2], "native", "cancelled" if control == "cancelled" else "completed",
                      result="child finished")
    if control == "capture_failure":
        monkeypatch.setattr(headless, "write_workspace_patch_artifacts",
                            lambda *a, **kw: (_ for _ in ()).throw(OSError("capture unavailable")))
        assert not headless.prepare_terminal_task_files(case[2], task)["error"]  # the worker's step fails honestly
    reports = [recover_terminal_task_files(case[2], set()) for _ in range(2)]
    recovered, status = {"capture_failure": ([], "failed"), "ambiguous": (["native"], "failed"),
                         "cancelled": ([], "pending")}[control]
    assert [report["recovered"] for report in reports] == [recovered, []], reports
    assert finalized == (["native"] if control != "cancelled" else []), finalized
    row = load_task_result(case[2], "native")
    assert row["artifact_status"] == status and row["native_capture_pending"][run_id], row
    assert headless.terminal_task_files_ready(case[2], task, row)
    assert "native" not in obligations.members(case[2], "pending_drives")
    out = _integrate_subagent_patch(parent_context(case), task_id="native")
    expected = {"capture_failure": "publication is still owed", "ambiguous": "apply outcome unknown",
                "cancelled": "INTEGRATE_NATIVE_CANCELLED"}
    assert expected[control] in out and not (case[1] / "late.txt").exists(), out


@pytest.mark.parametrize("publication_fails", [False, True])
def test_in_place_writer_live_then_terminal_updates_native_without_a_fictitious_apply(case, monkeypatch, publication_fails):
    from ouroboros import delegate_custody as custody, delegate_custody_current as current, headless
    from ouroboros.delegate_collection import collect_terminal
    from ouroboros.task_results import load_task_result, write_task_result
    from tests.test_delegate_body_collection import terminal
    folder, task = admit(case, "native", "parent")
    memory = case[2].parent / "child-memory"
    (memory / "memory").mkdir(parents=True)
    (memory / "memory" / "notes.md").write_text("native memory retained\n")
    task["child_drive_root"] = str(memory)
    write_task_result(case[2], "native", "completed", **{k: v for k, v in task.items() if k != "id"})
    write_task_result(memory, "native", "completed", result="native ended")
    headless.copy_child_task_result(case[2], task)
    entry = custody.RunCustody(run_id="in-place", task_id="native", route_id="fake", invocation_id="ip",
                              root_task_id="parent", parent_task_id="parent", execution_root=str(folder),
                              target_root=str(folder), resource_ref={"workspace_kind": "git", "strategy": "in_place"})
    assert custody.record_started(case[2], entry)
    (folder / "in-place.txt").write_bytes(b"still writing\n")
    headless.finalize_task_artifacts(case[2], task)
    row = load_task_result(case[2], "native")
    assert row["status"] == "completed" and row["artifact_status"] == "failed"
    exported = next(Path(item["path"]) for item in row["artifacts"] if item["kind"] == "memory_export")
    export_bytes = exported.read_bytes()
    assert b"native memory retained" in export_bytes
    assert headless.terminal_task_files_ready(case[2], task, row)
    assert_current_consumers(case, task, "failed")
    out = _integrate_subagent_patch(parent_context(case), task_id="native")
    assert "in-place writer live or unknown" in out and not (case[1] / "in-place.txt").exists(), out
    (folder / "in-place.txt").write_bytes(b"finished\n")
    publisher = headless.write_workspace_patch_artifacts
    if publication_fails:
        monkeypatch.setattr(headless, "write_workspace_patch_artifacts",
                            lambda *a, **kw: (_ for _ in ()).throw(OSError("capture unavailable")))
    result = collect_terminal(case[2], entry.run_id, detail=terminal(entry.run_id))
    assert result["workspace_capture"]["status"] == "in_place", result
    fresh = current.run(case[2], entry.run_id)
    assert not fresh.patch_apply_pending and not fresh.patch_disposed and not fresh.patch_captured
    if publication_fails:
        row = load_task_result(case[2], "native")
        assert row["artifact_status"] == "failed" and row["native_capture_pending"]
        monkeypatch.setattr(headless, "write_workspace_patch_artifacts", publisher)
        out = _integrate_subagent_patch(parent_context(case), task_id="native")
        assert "refreshed" in out and not (case[1] / "in-place.txt").exists(), out
    row = load_task_result(case[2], "native")
    assert row["status"] == "completed" and row["artifact_status"] == "ready_with_changes"
    assert exported.read_bytes() == export_bytes
    current_manifest = headless.current_workspace_manifest(case[2], "native", row)
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive
    capture_terminal_patch_for_drive(case[2], fresh)
    assert headless.current_workspace_manifest(case[2], "native", load_task_result(case[2], "native")) == current_manifest
    out = _integrate_subagent_patch(parent_context(case), task_id="native")
    assert "✅ Integrated" in out and (case[1] / "in-place.txt").read_bytes() == b"finished\n", out


def test_cancelled_native_in_place_result_becomes_historical_without_promotion(case):
    from ouroboros import delegate_custody as custody, headless
    from ouroboros.delegate_collection import collect_terminal
    from ouroboros.task_results import load_task_result, write_task_result
    from tests.test_delegate_body_collection import terminal
    folder, task = admit(case, "native", "parent")
    write_task_result(case[2], "native", "running", **{k: v for k, v in task.items() if k != "id"})
    headless.finalize_task_artifacts(case[2], task)
    before = {row["path"]: Path(row["path"]).read_bytes() for row in load_task_result(case[2], "native")["artifacts"]}
    entry = custody.RunCustody(run_id="cancelled-in-place", task_id="native", route_id="fake", invocation_id="ip",
                              root_task_id="parent", parent_task_id="parent", execution_root=str(folder),
                              target_root=str(folder), resource_ref={"workspace_kind": "git", "strategy": "in_place"})
    assert custody.record_started(case[2], entry)
    write_task_result(case[2], "native", "cancelled")
    assert load_task_result(case[2], "native")["status"] == "cancelled"
    (folder / "in-place.txt").write_bytes(b"retained late work\n")
    result = collect_terminal(case[2], entry.run_id, detail=terminal(entry.run_id))
    assert result["workspace_capture"]["status"] == "in_place"
    assert "cancelled" in result["workspace_capture"]["native_capture"]
    row = load_task_result(case[2], "native")
    assert row["status"] == "cancelled" and row["artifact_status"] == row["artifact_bundle"]["status"] == "failed"
    for path, data in before.items():
        assert Path(path).read_bytes() == data
    assert "INTEGRATE_NATIVE_CANCELLED" in _integrate_subagent_patch(parent_context(case), task_id="native")


@pytest.mark.parametrize("terminal_first", [False, True])
@pytest.mark.parametrize("publish_while_waiting", [False, True])
def test_revised_in_place_output_keeps_retired_native_capture(case, monkeypatch, terminal_first, publish_while_waiting):
    import json
    from contextlib import contextmanager
    from ouroboros import delegate_custody as custody, delegate_custody_current as current, headless
    from ouroboros.delegate_collection import collect_terminal
    from ouroboros.task_results import load_task_result, write_task_result
    from tests.test_delegate_body_collection import terminal

    folder, task = admit(case, "native", "parent")
    data = case[2]
    write_task_result(data, "native", "running", **{k: v for k, v in task.items() if k != "id"})
    entry = custody.RunCustody(run_id="in-place", task_id="native", route_id="fake", invocation_id="ip",
                              execution_root=str(folder), target_root=str(folder),
                              resource_ref={"workspace_kind": "git", "strategy": "in_place"})
    assert custody.record_started(data, entry)
    (folder / "in-place.txt").write_bytes(b"finished\n")
    if terminal_first:
        collect_terminal(data, entry.run_id, detail=terminal(entry.run_id))
    write_task_result(data, "native", "completed")
    if not terminal_first or not publish_while_waiting:
        headless.finalize_task_artifacts(data, task)
    if not terminal_first:
        assert load_task_result(data, "native")["native_capture_pending"]
        if not publish_while_waiting:
            collect_terminal(data, entry.run_id, detail=terminal(entry.run_id))
    saved = {}

    def retain_then_retire():
        before = load_task_result(data, "native")
        assert before["artifact_status"] == "ready_with_changes" and not before.get("native_capture_pending")
        path = headless.current_workspace_manifest(data, "native", before)
        saved.update(path=path, manifest=json.loads(path.read_bytes()),
                     bytes={row["path"]: Path(row["path"]).read_bytes() for row in before["artifacts"]},
                     source=current.run(data, entry.run_id).terminal_source)
        assert_current_consumers(case, task, "ready_with_changes", expected_path="in-place.txt")
        assert trees.prune_orphans(worktree_root=case[3], data_dir=data, retention_days=0,
                                  live=lambda _tid: False, only={"native"})["removed"] == 1
        assert not folder.exists()

    if publish_while_waiting:
        acquire = headless.native_capture_custody

        @contextmanager
        def publish_before_acquisition(root, writer):
            # Another finalizer wins custody after the collector's initial read.
            # Exercise both a formerly pending capture and an unpublished manifest.
            headless.finalize_task_artifacts(data, task)
            retain_then_retire()
            with acquire(root, writer) as owner:
                yield owner

        monkeypatch.setattr(headless, "native_capture_custody", publish_before_acquisition)
    else:
        retain_then_retire()
    old_source = current.run(data, entry.run_id).terminal_source
    out = collect_terminal(data, entry.run_id, detail=terminal(entry.run_id, text="revised answer", seq=8))
    fresh = current.run(data, entry.run_id)
    assert fresh.terminal_source != old_source and fresh.settled
    after = load_task_result(data, "native")
    assert after["artifact_status"] == "ready_with_changes" and not after.get("native_capture_pending"), out
    assert headless.current_workspace_manifest(data, "native", after) == saved["path"]
    assert all(Path(path).read_bytes() == body for path, body in saved["bytes"].items())
    manifest = saved["manifest"]
    witness = manifest["native_capture_sources"][entry.run_id]
    assert (witness["run_id"], witness["invocation_id"], witness["target_root"], witness["effect"]) == (
        entry.run_id, entry.invocation_id, str(folder), "in_place")
    assert witness["terminal_source"] == saved["source"]
    assert manifest["base_head"] == task["task_constraint"]["base_sha"]
    assert not fresh.patch_captured and not fresh.patch_disposed and not fresh.patch_apply_pending
    out = _integrate_subagent_patch(parent_context(case), task_id="native")
    assert "✅ Integrated" in out and (case[1] / "in-place.txt").read_bytes() == b"finished\n", out


@pytest.mark.parametrize("decision", ["apply", "reject"])
def test_ordinary_decision_then_late_capture_is_undecided_for_continue(case, monkeypatch, decision):
    import json
    from ouroboros.delegate_custody_current import marks
    from ouroboros.task_results import write_task_result
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.join_ledger import _child_result_sha256, _current_child_result_disposition
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch
    from tests.test_continue_successor_integration import interrupted_root, continue_from

    folder, task, entry = late_native_case(case, original=True)
    write_task_result(case[2], "native", "failed", delegation_role="subagent", root_task_id="parent")
    consumed = _child_result_sha256(load_effective_task_result(case[2], "native"))
    out = _integrate_subagent_patch(parent_context(case), task_id="native", decision=decision, reason="old capture")
    assert ("✅ Integrated" if decision == "apply" else "Rejected") in out, out
    out = _integrate_delegated_patch(parent_context(case, task_id="orphan-disposer"), entry.run_id)
    assert "✅ Integrated" in out, out
    current = load_effective_task_result(case[2], "native")
    assert _child_result_sha256(current) != consumed and not _current_child_result_disposition(current)
    interrupted_root(case[2], "parent")
    successor = continue_from(case[2], "parent")
    monkeypatch.setattr("ouroboros.task_status.task_has_live_queue_ownership", lambda *_a: False)
    out = _integrate_subagent_patch(parent_context(case, task_id=successor), task_id="native",
                                    decision="reject", reason="new capture")
    assert "Rejected" in out and "owner-Continue successor" in out, out
    from ouroboros.artifacts import task_artifact_dir_path
    receipt = json.loads((task_artifact_dir_path(case[2], "parent") / "subagent_patch_verdict_native.json").read_bytes())
    assert receipt["child_result_sha256"] == consumed
    assert marks(case[2], "parent")["patch_verdicts"][-1]["child_result_sha256"] == consumed


@pytest.mark.parametrize("writer", ["live", "unknown", "other_invocation"])
def test_covered_in_place_writer_does_not_hide_new_or_unknown_work(case, writer):
    import json
    from dataclasses import replace
    from ouroboros import delegate_custody as custody, headless
    from ouroboros.delegate_collection import collect_terminal
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive
    from tests.test_delegate_body_collection import terminal

    folder, task = admit(case, "native", "parent")
    data = case[2]
    write_task_result(data, "native", "completed", **{k: v for k, v in task.items() if k != "id"})
    first = custody.RunCustody(run_id="first", task_id="native", route_id="fake", invocation_id="first-start",
                              execution_root=str(folder), target_root=str(folder),
                              resource_ref={"workspace_kind": "git", "strategy": "in_place"})
    assert custody.record_started(data, first)
    (folder / "first.txt").write_text("first writer\n")
    collect_terminal(data, first.run_id, detail=terminal(first.run_id))
    old = headless.current_workspace_manifest(data, "native", load_task_result(data, "native"))
    old_bytes = old.read_bytes()
    if writer == "other_invocation":
        # A different invocation cannot borrow the recorded run's coverage.
        entry = replace(custody.replay(data)[first.run_id], invocation_id="another-start")
        out = capture_terminal_patch_for_drive(data, entry)
        assert "writer identity unavailable" in out["native_capture"], out
    else:
        entry = replace(first, run_id="second", invocation_id="second-start")
        assert custody.record_started(data, entry)
        (folder / "second.txt").write_text("new writer\n")
        if writer == "terminal":
            collect_terminal(data, entry.run_id, detail=terminal(entry.run_id))
        else:
            if writer == "unknown":
                assert custody.emit(data, custody.SETTLED, {"run_id": entry.run_id, "task_id": "native"})
            out = capture_terminal_patch_for_drive(data, custody.replay(data)[entry.run_id])
            assert "in-place writer live or unknown" in out["native_capture"], out
    row = load_task_result(data, "native")
    assert old.read_bytes() == old_bytes
    if writer != "terminal":
        assert row["artifact_status"] == "failed" and row["native_capture_pending"]
    else:
        manifest = json.loads(headless.current_workspace_manifest(data, "native", row).read_bytes())
        assert row["artifact_status"] == "ready_with_changes" and not row["native_capture_pending"]
        assert set(manifest["native_capture_sources"]) == {first.run_id, entry.run_id}
        assert manifest["base_head"] == task["task_constraint"]["base_sha"]
        out = _integrate_subagent_patch(parent_context(case), task_id="native")
        assert "✅ Integrated" in out, out
        assert (case[1] / "first.txt").read_text() == "first writer\n"
        assert (case[1] / "second.txt").read_text() == "new writer\n"


@pytest.fixture
def case(tmp_path, monkeypatch):
    return copy_case.__wrapped__(tmp_path, monkeypatch)


@pytest.mark.parametrize(("change", "owner"), [("delete", "task"), ("rename", "run"), ("recreated", "task")])
def test_staged_deleted_inputs_leave_current_snapshot_and_source_unchanged(case, owner, change):
    """The copy is the source's CURRENT tree: a staged delete or rename is in it, and a file
    recreated after ``git rm`` is an ordinary untracked input of the snapshot."""
    source = case[1]
    name = "a.txt"
    if change == "rename":
        git(source, "mv", name, "renamed.txt")
    else:
        git(source, "rm", name)
        if change == "recreated":
            (source / name).write_bytes(b"recreated after the staged delete\n")
    head, index = git(source, "rev-parse", "HEAD"), (source / ".git/index").read_bytes()
    status = git(source, "status", "--porcelain")
    args = {"task_id": "copy", "worktree_root": case[3], "data_dir": case[2]}
    handle = (trees.provision_worktree(repo_dir=source, **args) if owner == "task" else
              trees.provision_execution_snapshot(target_root=source, snapshot_id="run-copy", **args))
    output = Path(handle.path)
    if change == "recreated":
        assert (output / name).read_bytes() == b"recreated after the staged delete\n"
        assert git(output, "ls-files", name) == name.encode() + b"\n"
    else:
        assert not (output / name).exists()
        assert git(output, "ls-files", name) == b""
    if change == "rename":
        assert (output / "renamed.txt").read_bytes() == b"base\n"
    assert git(source, "rev-parse", "HEAD") == head
    assert (source / ".git/index").read_bytes() == index
    assert git(source, "status", "--porcelain") == status


def test_case_aliases_and_common_git_directory_are_physical_identity(case):
    system, external, _, _ = case
    alias = system.with_name(system.name.upper())
    if not alias.exists() or not alias.samefile(system):
        pytest.skip("filesystem does not support a samefile case alias")
    assert same_directory(alias, system)
    assert source_is_system_repo(alias, system)
    child, _ = admit(case, "body", "parent", system)
    assert source_is_system_repo(child, alias)
    assert not source_is_system_repo(external, alias)
    assert not same_directory(system.parent / "missing", system)


def seed_lines(source):
    (source / "a.txt").write_text("".join(f"line {i}\n" for i in range(30)), encoding="utf-8")
    git(source, "add", "a.txt"); git(source, "commit", "-qm", "lines")


def edit_line(folder, old, new):
    path = folder / "a.txt"
    path.write_text(path.read_text(encoding="utf-8").replace(old + "\n", new + "\n"), encoding="utf-8")


@pytest.mark.parametrize("commit_first", [False, True])
def test_clean_copies_synthesize_disjoint_hunks_on_same_file(case, commit_first):
    source = case[1]; seed_lines(source)
    a, ta = admit(case, "a", "parent"); b, tb = admit(case, "b", "parent")
    assert ta["workspace_copy"]["source_index_clean"]
    assert ta["workspace_copy"]["baseline_sha"] == ta["workspace_copy"]["target_head"]
    edit_line(a, "line 2", "A"); edit_line(b, "line 25", "B")
    capture(case, a, ta); capture(case, b, tb)
    ctx = parent_context(case)
    assert "✅ Integrated" in _integrate_subagent_patch(ctx, task_id="a")
    if commit_first:
        git(source, "commit", "-qm", "first contribution")
    out = _integrate_subagent_patch(ctx, task_id="b")
    assert "✅ Integrated" in out, out
    assert "A\n" in (source / "a.txt").read_text(encoding="utf-8")
    assert "B\n" in (source / "a.txt").read_text(encoding="utf-8")


def test_clean_copies_real_conflict_retains_both_results(case):
    seed_lines(case[1])
    a, ta = admit(case, "a", "parent"); b, tb = admit(case, "b", "parent")
    edit_line(a, "line 2", "A"); edit_line(b, "line 2", "B")
    capture(case, a, ta); artifacts, _ = capture(case, b, tb)
    ctx = parent_context(case)
    assert "✅ Integrated" in _integrate_subagent_patch(ctx, task_id="a")
    out = _integrate_subagent_patch(ctx, task_id="b")
    assert "INTEGRATE_CONFLICT" in out, out
    assert b.exists() and (artifacts / "workspace.patch").exists()
    assert git(case[1], "ls-files", "--unmerged")


def test_index_dirty_worktree_matches_head_keeps_current_tree_apply(case):
    source = case[1]
    (source / "a.txt").write_bytes(b"staged edit\n"); git(source, "add", "a.txt")
    (source / "a.txt").write_bytes(b"base\n")
    a, ta = admit(case, "a", "parent")
    assert ta["workspace_copy"]["baseline_sha"] == ta["workspace_copy"]["target_head"]
    assert ta["workspace_copy"]["source_index_clean"] is False
    (a / "a.txt").write_bytes(b"base\nchild\n"); capture(case, a, ta)
    out = _integrate_subagent_patch(parent_context(case), task_id="a")
    assert "✅ Integrated" in out, out
    assert (source / "a.txt").read_bytes() == b"base\nchild\n"


@pytest.mark.parametrize(("mode", "parent_focus"), [
    ("light", "main"),
    ("advanced", "other"),
    ("pro", "room"),
    ("cyber_pro", "main"),
])
def test_parent_returns_named_source_under_existing_write_rights(case, monkeypatch, parent_focus, mode):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
    # Production shape: Ouroboros data is under its own parent, not HOME/data
    # where the existing user-files policy correctly protects all of HOME.
    runtime = case[2].parent / "runtime-state"
    case[2].rename(runtime)
    case = (case[0], case[1], runtime, case[3])
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(runtime))
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(case[1].parent))
    child, task = admit(case, "child", "parent")
    (child / "BIBLE.md").write_bytes(b"foreign project docs\n"); capture(case, child, task)
    ctx = parent_context(case)
    if parent_focus in {"main", "room"}:
        ctx.workspace_root = None; ctx.workspace_mode = ""; ctx.is_direct_chat = True
        if parent_focus == "room":
            ctx.task_metadata = {"_project_room_dir": str(case[1])}
    else:
        ctx.workspace_root = repo(case[1].parent / "other", {"untouched": b"other"})
    out = _integrate_subagent_patch(ctx, task_id="child")
    assert "✅ Integrated" in out, out
    assert (case[1] / "BIBLE.md").read_bytes() == b"foreign project docs\n"


@pytest.mark.parametrize("restriction", ["readonly", "acting", "owner_store"])
def test_named_source_never_mints_parent_write_authority(case, monkeypatch, restriction):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.presence_authority import presence_ceiling_payload
    from tests.frozen_presence_authority import build_presence_capability_ceiling
    from tests.frozen_presence_capabilities import PresenceResourceTarget
    from tests.test_presence_authority import _resolution

    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(case[1].parent))
    child, task = admit(case, "child", "parent")
    path = ".ssh/id_rsa" if restriction == "owner_store" else "a.txt"
    (child / path).parent.mkdir(parents=True, exist_ok=True)
    (child / path).write_bytes(b"child output\n")
    if restriction == "owner_store":
        runtime = case[2].parent / "runtime-state"
        case[2].rename(runtime)
        case = (case[0], case[1], runtime, case[3])
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(runtime))
        ctx = parent_context(case); ctx.workspace_root = None; ctx.workspace_mode = ""
        # The folder is writable via user_files, but that does not authorize
        # its control files. Test the per-file rule after positive root binding.
        assert copy_apply_refusal(ctx, case[1], ["a.txt"]) == ""
        assert "VCS control directory" in copy_apply_refusal(ctx, case[1], [".git/config"])
        return
    capture(case, child, task)
    ctx = parent_context(case, source=case[0])
    if restriction in {"readonly", "acting"}:
        ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent" if restriction == "readonly" else "acting_subagent",
            surface="" if restriction == "readonly" else "external_workspace", write_root=str(case[0]))
    else:
        ceiling = build_presence_capability_ceiling(skill_name="fixture", skill_content_hash="c"*64,
            state_fingerprint="d"*64, resolution=_resolution(PresenceResourceTarget("active_workspace", ("write",), ".")))
        ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(ceiling)}
    out = _integrate_subagent_patch(ctx, task_id="child")
    assert "✅ Integrated" not in out and ("FORBIDDEN" in out or "PRESENCE_RESOURCE_BLOCKED" in out), out
    assert (case[1] / "a.txt").read_bytes() == b"base\n"


def test_actual_body_policy_survives_false_child_label(case, monkeypatch):
    child, task = admit(case, "child", "parent", case[0])
    task["workspace_copy"]["source_is_system_repo"] = False
    (child / "system.txt").write_bytes(b"body change\n"); capture(case, child, task)
    ctx = parent_context(case)  # parent is focused on the foreign project
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    out = _integrate_subagent_patch(ctx, task_id="child")
    assert "LIGHT_MODE_BLOCKED" in out, out
    assert (case[0] / "system.txt").read_bytes() == b"own body\n"


def test_apply_rechecks_presence_after_lock(case, monkeypatch):
    from ouroboros.tools import git as git_tools
    from ouroboros.presence_authority import presence_ceiling_payload
    from tests.frozen_presence_authority import build_presence_capability_ceiling
    from tests.test_presence_authority import _resolution
    child, task = admit(case, "child", "parent")
    (child / "a.txt").write_bytes(b"child\n"); capture(case, child, task)
    ctx = parent_context(case)
    lock = git_tools._acquire_git_lock
    def acquire(*args, **kwargs):
        held = lock(*args, **kwargs)
        ceiling = build_presence_capability_ceiling(skill_name="fixture", skill_content_hash="c"*64,
            state_fingerprint="d"*64, resolution=_resolution())
        ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(ceiling)}
        return held
    monkeypatch.setattr(git_tools, "_acquire_git_lock", acquire)
    out = _integrate_subagent_patch(ctx, task_id="child")
    assert "PRESENCE_RESOURCE_BLOCKED" in out and "✅ Integrated" not in out
    assert (case[1] / "a.txt").read_bytes() == b"base\n"


def test_real_schedule_supervisor_stamps_copy_metadata(case, monkeypatch):
    from ouroboros.tools.control_scheduling import _schedule_task
    from supervisor.events_schedule_task import _handle_schedule_task
    from ouroboros.task_results import load_task_result
    from tests.test_nested_rights_depth import _fake_ctx
    from tests.test_task_status_flow import _configure_test_subagent
    _configure_test_subagent(monkeypatch)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(case[1].parent))
    ctx = parent_context(case)
    (case[1] / "a.txt").write_bytes(b"uncommitted source\n")
    out = _schedule_task(ctx, subagent_id="api-scout", objective="Change source", expected_output="patch",
                         workspace_root=str(case[1]), write_surface="self_worktree")
    assert "queued" in out, out
    evt = ctx.pending_events[-1]
    enqueued = []
    supervisor = _fake_ctx(case[2], enqueued); supervisor.REPO_DIR = case[0]
    _handle_schedule_task(evt, supervisor)
    [task] = enqueued
    binding = task["metadata"]["workspace_copy"]
    assert binding["source_root"] == str(case[1])
    assert binding["execution_root"] == task["workspace_root"]
    assert binding == load_task_result(case[2], task["id"])["workspace_copy"]
    assert task["task_contract"]["workspace"]["root"] == task["workspace_root"]
    assert (Path(task["workspace_root"]) / "a.txt").read_bytes() == b"uncommitted source\n"
