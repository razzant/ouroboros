"""Retry result consumers retain each materializer's admission and durable debt."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import copy
import json
from pathlib import Path
import threading

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_delegate_retry_consumers import (
    bind_gateway, call, durable_run, queue_snapshot, retry_context,
    isolated_retry_state as isolated_retry_state,  # re-export the autouse pytest fixture
)
from tests.test_delegated_directory import DirectoryEngine
from tests._store_shared import result_signature


pytestmark = pytest.mark.serial


def directory_result(tmp_path, monkeypatch, *, chain=("retry-a", "retry-b"),
                     owner="retry-a", strategy="copy", lost_apply=False, captured=True):
    ctx = retry_context(tmp_path, monkeypatch, chain=chain)
    # The authority a real writable folder start records for this root's external
    # workspace. #1609: the first-wins authority source is what separates a writable
    # run from an evidence-only investigation, and a source-less direct run (no
    # snapshot, no engine copy) reads as one with nothing to integrate.
    entry = durable_run(ctx, owner=owner, run_id="directory-run", access="workspace_write",
                        authority_source="external_workspace_root",
                        resource_ref={"workspace_kind": "directory", "strategy": strategy,
                                      "scopePaths": ["."]})
    engine = DirectoryEngine(Path(ctx.workspace_root), strategy, lost_apply=lost_apply)
    bind_gateway(monkeypatch, engine)
    from ouroboros import claudexor_daemon

    @contextmanager
    def owned():
        yield engine

    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", owned)
    if captured:  # the body's capture, before decisions (#1969); False keeps capture-at-disposition
        from ouroboros.delegate_directory import capture_directory_result

        capture_directory_result(custody.custody_root(ctx), entry, engine)
        custody._CUSTODY.clear()
    return ctx, entry, engine


@pytest.mark.parametrize("owner", ["retry-a", "retry-b"])
def test_directory_disposition_clears_current_inherited_debt_only(tmp_path, monkeypatch, owner):
    from ouroboros.task_status import effective_task_result

    chain = ("retry-a", "retry-b", "retry-c")
    ctx, entry, engine = directory_result(tmp_path, monkeypatch, chain=chain, owner=owner)
    drive = custody.custody_root(ctx)
    original = {}
    for task_id in chain:
        row = load_task_result(drive, task_id)
        write_task_result(drive, task_id, row["status"],
                          reason_code="idle_timeout", result="historical answer",
                          accounted_usd=1.25, delegated_runs_started=7,
                          delegated_runs_settled=0, review_verdict="blockers",
                          delegated_runs_unreconciled=[entry.run_id, "patch:" + entry.run_id,
                                                       "patch:another-run"],
                          delegate_terminal_reconciliation={
                              "audit_status": "ok", "task_id": task_id,
                              "open_run_ids": [entry.run_id],
                              "undisposed_patch_run_ids": [entry.run_id, "another-run"],
                              "unreconciled": [entry.run_id, "patch:" + entry.run_id,
                                                "patch:another-run"],
                          })
        original[task_id] = load_task_result(drive, task_id)
    result = call(ctx, "integrate_delegated_patch", run_id=entry.run_id, decision="apply")
    assert result.text.startswith("{"), result.text
    assert json.loads(result.text)["status"] == "applied", result.text
    assert len(engine.applies) == 1
    for task_id in chain:
        row = load_task_result(drive, task_id)
        assert entry.run_id not in row["delegated_runs_unreconciled"]
        assert "patch:" + entry.run_id not in row["delegated_runs_unreconciled"]
        assert "patch:another-run" in row["delegated_runs_unreconciled"]
        assert entry.run_id not in row["delegate_terminal_reconciliation"]["open_run_ids"]
        assert entry.run_id not in row["delegate_terminal_reconciliation"]["undisposed_patch_run_ids"]
        for key in ("status", "reason_code", "result", "accounted_usd", "delegated_runs_started",
                    "delegated_runs_settled", "review_verdict"):
            assert row[key] == original[task_id][key]
    effective = effective_task_result(drive, load_task_result(drive, chain[0]))
    assert effective["delegated_runs_unreconciled"] == ["patch:another-run"]
    rows = custody.custody_rows(drive)
    disposed = [row for row in rows if row["type"] == custody.PATCH_DISPOSED]
    assert len(disposed) == 1 and disposed[0]["task_id"] == owner
    assert disposed[0]["disposed_by_task_id"] == "retry-c"


def test_concurrent_registered_directory_disposition_materializes_once(tmp_path, monkeypatch):
    ctx, entry, engine = directory_result(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    apply = engine.apply_run

    def hold_apply(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return apply(*args, **kwargs)

    monkeypatch.setattr(engine, "apply_run", hold_apply)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(call, copy(ctx), "integrate_delegated_patch", run_id=entry.run_id)
        assert entered.wait(5)
        second = pool.submit(call, copy(ctx), "integrate_delegated_patch", run_id=entry.run_id)
        release.set()
        results = [first.result(timeout=10).text, second.result(timeout=10).text]
    assert sum('"status": "applied"' in result for result in results) == 1, results
    assert any("ALREADY_DISPOSED" in result for result in results), results
    assert len(engine.applies) == 1


def test_directory_retry_keeps_unknown_intent_and_original_path_selection(tmp_path, monkeypatch):
    ctx, entry, engine = directory_result(tmp_path, monkeypatch, lost_apply=True)
    first = call(ctx, "integrate_delegated_patch", run_id=entry.run_id, paths=["out.bin"])
    assert "APPLY_UNCONFIRMED" in first.text
    blocked = call(ctx, "integrate_delegated_patch", run_id=entry.run_id, decision="reject")
    assert "APPLY_AMBIGUOUS" in blocked.text
    changed = call(ctx, "integrate_delegated_patch", run_id=entry.run_id,
                   acknowledge_ambiguous=True, paths=["other.bin"])
    assert "APPLY_AMBIGUOUS" in changed.text
    finished = call(ctx, "integrate_delegated_patch", run_id=entry.run_id,
                    acknowledge_ambiguous=True, paths=["out.bin"])
    assert json.loads(finished.text)["status"] == "applied", finished.text
    assert len(engine.applies) == 2 and engine.applies[0] == engine.applies[1]


@pytest.mark.parametrize("decision", ["apply", "reject"])
def test_stale_starter_cannot_dispose_after_host_retry(tmp_path, monkeypatch, decision):
    ctx, entry, engine = directory_result(tmp_path, monkeypatch)
    stale = copy(ctx)
    stale.task_id = entry.task_id
    result = call(stale, "integrate_delegated_patch", run_id=entry.run_id, decision=decision)
    assert "NOT_OWNED" in result.text
    assert engine.applies == engine.decisions == []


def test_successor_directory_apply_keeps_exact_target(tmp_path, monkeypatch):
    ctx, entry, engine = directory_result(tmp_path, monkeypatch)
    ctx.workspace_root = tmp_path / "another-project"
    ctx.workspace_root.mkdir()
    result = call(ctx, "integrate_delegated_patch", run_id=entry.run_id)
    assert "TARGET_MISMATCH" in result.text
    assert engine.applies == []


def captured_result(tmp_path, monkeypatch, kind):
    from ouroboros.subagent_worktrees import provision_execution_snapshot, provision_payload_snapshot
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive
    from tests.test_delegated_run_isolation import _seed_target
    from tests.test_delegated_skill_payload import _seed_skill

    ctx = retry_context(tmp_path, monkeypatch, chain=("retry-a", "retry-b", "retry-c"), split=True,
                        fixed_ownership_clock=True)
    data = custody.custody_root(ctx)
    if kind == "git":
        seed = tmp_path / "git-project"
        seed.mkdir()
        target = _seed_target(seed)
        handle = provision_execution_snapshot(target_root=target, task_id="retry-a", snapshot_id="retry-snapshot")
        authority, resource_ref = "external_workspace_root", {}
        changed_path, body = "tracked.txt", "one\ntwo\nCHILD-EDIT\n"
    else:
        target = _seed_skill(data)
        handle = provision_payload_snapshot(target_root=target, task_id="retry-a", snapshot_id="retry-snapshot")
        authority = "skill_payload"
        resource_ref = {"root": "skill_payload", "source": "external", "skill_name": "alpha",
                        "target_root": str(target.resolve()), "payload_hash": handle.payload_hash}
        changed_path, body = "notes.txt", "COMPLETE CHILD RESULT\n"
    ctx.workspace_root = target
    entry = durable_run(ctx, execution_root=handle.path, snapshot_id=handle.snapshot_id,
                        baseline_sha=handle.baseline_sha, authority_source=authority,
                        resource_ref=resource_ref, access="workspace_write")
    (Path(handle.path) / changed_path).write_text(body, encoding="utf-8")
    block = capture_terminal_patch_for_drive(data, entry)
    assert block["status"] == "ready_with_changes", block
    if kind == "payload":
        ctx.workspace_root, ctx.workspace_mode = None, ""
    return ctx, entry, target, changed_path, body


@pytest.mark.parametrize("kind", ["git", "payload"])
@pytest.mark.parametrize("decision", ["apply", "reject"])
def test_registered_successor_materializes_or_rejects_exact_capture(tmp_path, monkeypatch, kind, decision):
    from tests.test_delegated_run_isolation import _git

    ctx, entry, target, changed_path, body = captured_result(tmp_path, monkeypatch, kind)
    before = (target / changed_path).read_text()
    result = call(ctx, "integrate_delegated_patch", run_id=entry.run_id, decision=decision)
    assert result.status == "ok", result.text
    assert (target / changed_path).read_text() == (body if decision == "apply" else before)
    if kind == "git" and decision == "apply":
        assert changed_path in _git(target, "diff", "--cached", "--name-only").stdout
    if kind == "payload":
        assert not (target / ".git").exists()
    # Deliberately keep a stale in-process object after the durable disposition.
    custody._CUSTODY[entry.run_id] = entry
    repeated = call(ctx, "integrate_delegated_patch", run_id=entry.run_id, decision=decision)
    assert "ALREADY_DISPOSED" in repeated.text, repeated.text
    disposed = [row for row in custody.custody_rows(custody.custody_root(ctx))
                if row["type"] == custody.PATCH_DISPOSED]
    assert len(disposed) == 1 and disposed[0]["disposed_by_task_id"] == "retry-c"


@pytest.mark.parametrize("kind", ["git", "payload"])
def test_successor_materializer_keeps_target_drift_and_reject_exit(tmp_path, monkeypatch, kind):
    ctx, entry, target, changed_path, _ = captured_result(tmp_path, monkeypatch, kind)
    (target / changed_path).write_text("OWNER'S NEWER EDIT\n")
    result = call(ctx, "integrate_delegated_patch", run_id=entry.run_id)
    assert "CONFLICT" in result.text or "DRIFT" in result.text, result.text
    assert (target / changed_path).read_text() == "OWNER'S NEWER EDIT\n"
    assert not custody.replay(custody.custody_root(ctx))[entry.run_id].patch_disposed
    rejected = call(ctx, "integrate_delegated_patch", run_id=entry.run_id, decision="reject")
    assert rejected.status == "ok", rejected.text


@pytest.mark.parametrize("kind", ["git", "payload"])
def test_successor_keeps_ambiguous_intent_until_explicit_acknowledgment(tmp_path, monkeypatch, kind):
    ctx, entry, target, changed_path, _ = captured_result(tmp_path, monkeypatch, kind)
    assert custody.record_patch_apply_started(custody.custody_root(ctx), entry, target_root=str(target))
    before = (target / changed_path).read_bytes()
    blocked = call(ctx, "integrate_delegated_patch", run_id=entry.run_id, decision="reject")
    assert "APPLY_AMBIGUOUS" in blocked.text
    rejected = call(ctx, "integrate_delegated_patch", run_id=entry.run_id,
                    decision="reject", acknowledge_ambiguous=True)
    assert rejected.status == "ok", rejected.text
    assert (target / changed_path).read_bytes() == before


@pytest.mark.parametrize("kind", ["git", "payload", "directory"])
def test_registered_readonly_retry_never_gets_apply(tmp_path, monkeypatch, kind):
    if kind == "directory":
        ctx, entry, engine = directory_result(tmp_path, monkeypatch)
        before = list(engine.applies)
    else:
        ctx, entry, target, changed_path, _ = captured_result(tmp_path, monkeypatch, kind)
        before = (target / changed_path).read_bytes()
    ctx.task_constraint = {"mode": "local_readonly_subagent"}
    result = call(ctx, "integrate_delegated_patch", run_id=entry.run_id)
    assert result.status != "ok", result.text
    if kind == "directory":
        assert engine.applies == before
    else:
        assert (target / changed_path).read_bytes() == before


def test_successor_payload_rebind_refuses_moved_target_but_can_reject(tmp_path, monkeypatch):
    ctx, entry, target, _, _ = captured_result(tmp_path, monkeypatch, "payload")
    target.rename(target.with_name("renamed-alpha"))
    result = call(ctx, "integrate_delegated_patch", run_id=entry.run_id)
    assert "payload_target_unresolved" in result.text, result.text
    rejected = call(ctx, "integrate_delegated_patch", run_id=entry.run_id, decision="reject")
    assert rejected.status == "ok", rejected.text


def test_directory_partial_apply_retains_remaining_capture_then_discards(tmp_path, monkeypatch):
    ctx, entry, engine = directory_result(tmp_path, monkeypatch)
    get_run = engine.get_run

    def partial(run_id):
        detail = get_run(run_id)
        detail["summary"]["result"]["applyState"] = "partially_applied"
        return detail

    monkeypatch.setattr(engine, "get_run", partial)
    result = call(ctx, "integrate_delegated_patch", run_id=entry.run_id, paths=["out.bin"])
    assert json.loads(result.text)["status"] == "partially_applied", result.text
    current = custody.replay(custody.custody_root(ctx))[entry.run_id]
    assert current.patch_disposed == "" and not current.patch_apply_pending
    assert (custody.delegated_capture_dir(custody.custody_root(ctx), entry.task_id, entry.run_id)
            / "workspace_patch.json").exists()
    rejected = call(ctx, "integrate_delegated_patch", run_id=entry.run_id, decision="reject")
    assert json.loads(rejected.text)["status"] == "rejected", rejected.text
    assert (Path(ctx.workspace_root) / "out.bin").read_bytes() == engine.body


def supersede(ctx):
    drive = custody.custody_root(ctx)
    write_task_result(drive, ctx.task_id, "interrupted", superseded_by="retry-d", retry_task_id="retry-d")
    write_task_result(drive, "retry-d", "running", root_task_id="retry-a", parent_task_id="",
                      delegation_role="root", original_task_id=ctx.task_id, timeout_retry_from=ctx.task_id,
                      supersedes_task_id=ctx.task_id)
    queue_snapshot(ctx, ["retry-d"])


@pytest.mark.parametrize("decision", ["apply", "reject"])
def test_directory_superseded_during_capture_never_submits_effect(tmp_path, monkeypatch, decision):
    ctx, entry, engine = directory_result(tmp_path, monkeypatch, captured=False)
    get_run = engine.get_run

    def superseding_capture(run_id):
        detail = get_run(run_id)
        supersede(ctx)
        return detail

    monkeypatch.setattr(engine, "get_run", superseding_capture)
    result = call(ctx, "integrate_delegated_patch", run_id=entry.run_id, decision=decision)
    assert "NOT_OWNED" in result.text, result.text
    assert engine.applies == engine.decisions == []
    assert not custody.replay(custody.custody_root(ctx))[entry.run_id].patch_disposed


@pytest.mark.parametrize("kind", ["git", "payload"])
def test_superseded_during_materializer_preparation_never_applies(tmp_path, monkeypatch, kind):
    ctx, entry, target, changed_path, _ = captured_result(tmp_path, monkeypatch, kind)
    before = (target / changed_path).read_bytes()
    if kind == "git":
        from ouroboros.tools import subagent_integration as si

        baseline = si._baseline_drifted_paths

        def superseding_baseline(*args, **kwargs):
            result = baseline(*args, **kwargs)
            supersede(ctx)
            return result

        monkeypatch.setattr(si, "_baseline_drifted_paths", superseding_baseline)
    else:
        from ouroboros.tools import delegate_integration as di

        payload_hash = di.payload_content_hash

        def superseding_hash(*args, **kwargs):
            result = payload_hash(*args, **kwargs)
            supersede(ctx)
            return result

        monkeypatch.setattr(di, "payload_content_hash", superseding_hash)
    result = call(ctx, "integrate_delegated_patch", run_id=entry.run_id)
    assert "NOT_OWNED" in result.text, result.text
    assert (target / changed_path).read_bytes() == before
    current = custody.replay(custody.custody_root(ctx))[entry.run_id]
    assert not current.patch_disposed and not current.patch_apply_pending


def test_disposition_retry_heals_failed_debt_write_without_reapplying(tmp_path, monkeypatch):
    from ouroboros import task_results

    ctx, entry, engine = directory_result(tmp_path, monkeypatch)
    drive = custody.custody_root(ctx)
    write_task_result(drive, entry.task_id, "interrupted",
                      delegated_runs_unreconciled=["patch:" + entry.run_id])
    with monkeypatch.context() as patch:
        patch.setattr(task_results, "write_task_result", lambda *args, **kwargs: {})
        applied = call(ctx, "integrate_delegated_patch", run_id=entry.run_id)
    assert json.loads(applied.text)["status"] == "applied", applied.text
    assert load_task_result(drive, entry.task_id)["delegated_runs_unreconciled"] == ["patch:" + entry.run_id]
    repeated = call(ctx, "integrate_delegated_patch", run_id=entry.run_id)
    assert "ALREADY_DISPOSED" in repeated.text
    assert load_task_result(drive, entry.task_id)["delegated_runs_unreconciled"] == []
    assert len(engine.applies) == 1
    retained = result_signature(drive, entry.task_id)
    call(ctx, "integrate_delegated_patch", run_id=entry.run_id)
    assert result_signature(drive, entry.task_id) == retained, "already-current disclosure must not churn"


def test_unwritten_disposition_never_clears_current_debt(tmp_path, monkeypatch):
    ctx, entry, engine = directory_result(tmp_path, monkeypatch)
    drive = custody.custody_root(ctx)
    write_task_result(drive, entry.task_id, "interrupted",
                      delegated_runs_unreconciled=["patch:" + entry.run_id])
    emit = custody.emit

    def unwritten(root, kind, payload):
        return False if kind == custody.PATCH_DISPOSED else emit(root, kind, payload)

    monkeypatch.setattr(custody, "emit", unwritten)
    result = call(ctx, "integrate_delegated_patch", run_id=entry.run_id)
    assert "DISPOSITION_UNWRITTEN" in result.text
    assert len(engine.applies) == 1
    assert load_task_result(drive, entry.task_id)["delegated_runs_unreconciled"] == ["patch:" + entry.run_id]
    assert custody.replay(drive)[entry.run_id].patch_apply_pending


def test_direct_directory_reject_does_not_claim_undo(tmp_path, monkeypatch):
    ctx, entry, engine = directory_result(tmp_path, monkeypatch, strategy="direct")
    (Path(ctx.workspace_root) / "out.bin").write_bytes(engine.body)
    rejected = call(ctx, "integrate_delegated_patch", run_id=entry.run_id, decision="reject")
    assert "ALREADY_APPLIED" in rejected.text
    assert (Path(ctx.workspace_root) / "out.bin").read_bytes() == engine.body
    applied = call(ctx, "integrate_delegated_patch", run_id=entry.run_id)
    assert json.loads(applied.text)["status"] == "applied", applied.text
    assert engine.applies == engine.decisions == []


@pytest.mark.parametrize("kind", ["git", "payload", "directory"])
def test_superseded_while_recording_intent_never_materializes(tmp_path, monkeypatch, kind):
    if kind == "directory":
        ctx, entry, engine = directory_result(tmp_path, monkeypatch)
    else:
        ctx, entry, target, changed_path, _ = captured_result(tmp_path, monkeypatch, kind)
        before = (target / changed_path).read_bytes()
    record = custody.record_patch_apply_started

    def superseding_intent(*args, **kwargs):
        result = record(*args, **kwargs)
        supersede(ctx)
        return result

    monkeypatch.setattr(custody, "record_patch_apply_started", superseding_intent)
    result = call(ctx, "integrate_delegated_patch", run_id=entry.run_id)
    assert "NOT_OWNED" in result.text, result.text
    current = custody.replay(custody.custody_root(ctx))[entry.run_id]
    assert not current.patch_disposed and not current.patch_apply_pending
    if kind == "directory":
        assert engine.applies == []
    else:
        assert (target / changed_path).read_bytes() == before


def test_stale_retry_cannot_fall_back_to_terminal_owner_orphan_authority(tmp_path, monkeypatch):
    ctx, entry, engine = directory_result(tmp_path, monkeypatch,
                                         chain=("retry-a", "retry-b", "retry-c"))
    write_task_result(custody.custody_root(ctx), "retry-a", "failed")
    ctx.task_id = "retry-b"
    result = call(ctx, "integrate_delegated_patch", run_id=entry.run_id, decision="reject")
    assert "NOT_OWNED" in result.text, result.text
    assert engine.applies == engine.decisions == []


def test_stale_directory_replay_preserves_previous_unknown_intent(tmp_path, monkeypatch):
    ctx, entry, engine = directory_result(tmp_path, monkeypatch, lost_apply=True)
    first = call(ctx, "integrate_delegated_patch", run_id=entry.run_id, paths=["out.bin"])
    assert "APPLY_UNCONFIRMED" in first.text
    drive = custody.custody_root(ctx)
    original_key = custody.replay(drive)[entry.run_id].patch_apply_key
    record = custody.record_patch_apply_started

    def superseding_intent(*args, **kwargs):
        result = record(*args, **kwargs)
        supersede(ctx)
        return result

    monkeypatch.setattr(custody, "record_patch_apply_started", superseding_intent)
    replayed = call(ctx, "integrate_delegated_patch", run_id=entry.run_id,
                    acknowledge_ambiguous=True, paths=["out.bin"])
    assert "NOT_OWNED" in replayed.text, replayed.text
    current = custody.replay(drive)[entry.run_id]
    assert current.patch_apply_pending and current.patch_apply_key == original_key
    assert not current.patch_disposed and len(engine.applies) == 1


@pytest.mark.parametrize("strategy,decision", [("copy", "apply"), ("copy", "reject"), ("direct", "apply")])
def test_uncaptured_directory_retry_keeps_inspection_and_readable_manifest(tmp_path, monkeypatch, strategy, decision):
    """The successor can follow the advertised reader, not merely the absolute fallback."""
    from ouroboros.delegate_collection import capture_projection
    from ouroboros.delegate_custody_current import run

    ctx, entry, engine = directory_result(tmp_path, monkeypatch, captured=False, strategy=strategy)
    drive = custody.custody_root(ctx)
    first = call(ctx, "integrate_delegated_patch", run_id=entry.run_id, decision=decision)
    if (strategy, decision) == ("copy", "apply"):
        assert "INTEGRATE_DELEGATED_INSPECT_CAPTURE" in first.text
        assert not engine.applies and not run(drive, entry.run_id).patch_disposed
        block = json.loads(first.text.split("\n", 1)[1])
        reread = capture_projection(drive, run(drive, entry.run_id), reader_task_id=ctx.task_id)
        for capture in (block, reread):
            read = call(ctx, "read_file", **capture["manifest_read"])
            assert read.status == "ok", read.text
            assert "file_outputs" in read.text
        second = call(ctx, "integrate_delegated_patch", run_id=entry.run_id, decision="apply")
        assert json.loads(second.text)["status"] == "applied", second.text
        assert len(engine.applies) == 1
    else:
        assert "INTEGRATE_DELEGATED_INSPECT_CAPTURE" not in first.text, first.text
        assert run(drive, entry.run_id).patch_disposed == ("rejected" if decision == "reject" else "applied")
