"""Retry result consumers retain each materializer's admission and durable debt."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import copy
import hashlib
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
    (Path(handle.path) / changed_path).write_text(body, encoding="utf-8", newline="\n")  # LF: the tests compare bytes
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


@pytest.mark.parametrize("pin", ["readable", "unreadable", "manifest_unreadable"])
def test_unreceipted_ready_git_capture_is_published_never_captured_over(tmp_path, monkeypatch, pin):
    from ouroboros.delegate_custody_current import run
    from ouroboros.subagent_worktrees import provision_execution_snapshot
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive
    from tests.test_delegated_run_isolation import _seed_target

    ctx = retry_context(tmp_path, monkeypatch, chain=("retry-a", "retry-b", "retry-c"), split=True,
                        fixed_ownership_clock=True)
    data = custody.custody_root(ctx)
    (tmp_path / "git-project").mkdir()
    target = _seed_target(tmp_path / "git-project")
    handle = provision_execution_snapshot(target_root=target, task_id="retry-a", snapshot_id="retry-snapshot")
    ctx.workspace_root = target
    entry = durable_run(ctx, execution_root=handle.path, snapshot_id=handle.snapshot_id, baseline_sha=handle.baseline_sha,
                        authority_source="external_workspace_root", resource_ref={}, access="workspace_write")
    (Path(handle.path) / "tracked.txt").write_text("one\ntwo\nCHILD-EDIT\n", encoding="utf-8")
    emit = custody.emit
    with monkeypatch.context() as unavailable:  # the capture's own receipt row does not land
        unavailable.setattr(custody, "emit", lambda root, event, row, **kw:
                            event != custody.PATCH_CAPTURED and emit(root, event, row, **kw))
        first = capture_terminal_patch_for_drive(data, entry)
    assert (first["status"], first["retry"]) == ("failed", "transient"), first
    manifest, patch = Path(first["manifest_artifact"]), Path(first["patch_artifact"])
    declared = manifest.read_bytes(), patch.read_bytes()
    custody._CUSTODY.clear()
    with monkeypatch.context() as reopened:
        reopened.setattr("ouroboros.tools.delegate_integration._write_result_commit_artifacts",
                         lambda *_a, **_k: pytest.fail("a later capture wrote over bytes already shown"))
        if pin == "unreadable":
            reopened.setattr("ouroboros.subagent_worktrees.find_execution_snapshot",
                             lambda *_a, **_k: (_ for _ in ()).throw(OSError("snapshot record unavailable")))
        if pin == "manifest_unreadable":  # the saved manifest exists, but this read of it fails
            read = Path.read_bytes
            reopened.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(OSError("read failed"))
                             if self == manifest else read(self))
        second = capture_terminal_patch_for_drive(data, run(data, entry.run_id))
    assert (manifest.read_bytes(), patch.read_bytes()) == declared
    rows = [row for row in custody.custody_rows(data) if row["type"] == custody.PATCH_CAPTURED]
    if pin == "readable":
        assert second["status"] == "ready_with_changes" and second["sha256"] == first["sha256"], second
        receipt, = rows
        assert receipt["manifest_sha256"] == hashlib.sha256(declared[0]).hexdigest()
        assert run(data, entry.run_id).patch_manifest_sha == receipt["manifest_sha256"]
    elif pin == "unreadable":
        assert (second["status"], second["retry"]) == ("failed", "transient"), second
        assert "neither published nor captured over" in second["note"] and not rows
        assert second["patch_read"] and second["manifest_read"]  # the kept files stay readable
    else:
        assert (second["status"], second["retry"]) == ("failed", "transient"), second
        assert "nothing is captured over it" in second["note"] and not rows


@pytest.mark.parametrize("case", ["older_prepared", "older_unprepared", "modern_unpinned", "record_unreadable"])
def test_declared_capture_proof_accepts_an_older_copy_only_from_its_own_prepared_result(tmp_path, monkeypatch, case):
    from ouroboros import delegate_capture_job as job
    from ouroboros.tools.delegate_integration import _declared_capture_gap

    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "workspace.patch").write_bytes(b"--- a/r.txt\n+++ b/r.txt\n@@ -1 +1 @@\n-old\n+new\n")
    manifest = {"status": "ready_with_changes", "patch_size": 44,  # an older capture names no result commit
                "sha256": hashlib.sha256((capture / "workspace.patch").read_bytes()).hexdigest()}
    raw = json.dumps(manifest).encode("utf-8")
    entry = custody.RunCustody(run_id="older-run", task_id="t", route_id="r", model="m", snapshot_id="older-copy",
                               execution_root=str(tmp_path / "copy"), baseline_sha="b" * 40)
    if case != "older_unprepared":  # this run's own bound child result names exactly these manifest bytes
        entry.capture_job = {"state": "ready", "binding": job.identity(entry),
                             "manifest_sha256": hashlib.sha256(raw).hexdigest()}

    def record(_snapshot_id):
        if case == "record_unreadable":
            raise OSError("snapshot record unavailable")
        return {} if case == "modern_unpinned" else {"file_baseline": {}}

    monkeypatch.setattr("ouroboros.subagent_worktrees.find_execution_snapshot", record)
    gap, retry = _declared_capture_gap(entry, capture, raw, manifest)
    if case == "older_prepared":
        assert (gap, retry) == ("", "")
    else:  # no older-copy record, no own prepared result, or an unreadable record: no publication
        assert gap and retry == ("transient" if case == "record_unreadable" else ""), (gap, retry)


def test_git_apply_keeps_its_typed_effect_when_native_and_verdict_file_writes_fail(tmp_path, monkeypatch):
    from ouroboros.delegate_custody_current import run
    from ouroboros.tools import subagent_integration as integration

    ctx, entry, target, changed_path, expected = captured_result(tmp_path, monkeypatch, "git")
    receipts, unavailable, applies, apply = [], [True], [], integration._locked_apply

    def native(_root, current, outcome):  # this test's own synthetic native receipt write
        if unavailable[0]:
            raise OSError("native journal unavailable")
        receipts.append((current.patch_apply_key, outcome["proc"].returncode, outcome["unindexed"]))

    monkeypatch.setattr("ouroboros.headless.record_native_capture_effect", native)
    monkeypatch.setattr(integration, "_locked_apply", lambda *a, **kw: applies.append(a[1]) or apply(*a, **kw))
    monkeypatch.setattr("ouroboros.tools.patch_verdict.task_artifact_dir_path",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("artifact folder unavailable")))
    first = call(ctx, "integrate_delegated_patch", run_id=entry.run_id, reason="apply")
    assert (first.status, first.code) == ("blocked", "INTEGRATION_BLOCKED") and (target / changed_path).read_text() == expected, first.text
    facts = {key: first.meta[key] for key in ("target_files_changed", "files_changed", "unindexed",
                                              "disposition_recorded", "native_capture_effect")}
    assert facts == {"target_files_changed": True, "files_changed": 1, "unindexed": [], "disposition_recorded": False,
                     "native_capture_effect": "unknown"}, first.meta
    assert Path(first.meta["target_root"]).resolve() == target.resolve()
    assert "records this apply without changing a file" in first.text, first.text
    key = run(custody.custody_root(ctx), entry.run_id).patch_apply_key
    for healthy in (False, True):  # a repeated native failure keeps the intent; a later write only appends
        custody._CUSTODY.clear()
        unavailable[0] = not healthy
        again = call(copy(ctx), "integrate_delegated_patch", run_id=entry.run_id)
        assert again.meta.get("target_files_changed") is not True and "ALREADY_DISPOSED" not in again.text, again.text
        assert len(applies) == 1 and (target / changed_path).read_text() == expected
        current = run(custody.custody_root(ctx), entry.run_id)
        assert (again.status == "ok") == healthy and current.patch_apply_key == key, again.text
        assert (current.patch_disposed, current.patch_apply_pending) == (("applied", False) if healthy else ("", True))
    assert receipts == [(key, 0, [])]
    rows = [row for row in custody.custody_rows(custody.custody_root(ctx)) if row["type"] == custody.PATCH_DISPOSED]
    assert len(rows) == 1 and rows[0]["reason"] == "apply"


@pytest.mark.parametrize("refresh", ["native_owner", "owner_read"])
def test_git_effect_survives_a_failed_refresh_after_its_unwritten_disposition(tmp_path, monkeypatch, refresh):
    from contextlib import contextmanager as owned
    from ouroboros.delegate_custody_current import run
    from ouroboros.loop_tool_execution import _typed_result_metadata
    from ouroboros.outcomes import reviewable_effect_projection
    from ouroboros.tools import subagent_integration as integration
    from tests.test_delegated_run_isolation import _git

    ctx, entry, target, changed_path, expected = captured_result(tmp_path, monkeypatch, "git")
    drive, rid, index = custody.custody_root(ctx), entry.run_id, target / ".git" / "index"
    apply, emit, applied = integration._locked_apply, custody.emit, []
    monkeypatch.setattr(integration, "_locked_apply", lambda *a, **kw: applied.append(a[1]) or apply(*a, **kw))
    with monkeypatch.context() as down:  # the custody store refuses the disposition, then the refresh fails
        down.setattr(custody, "emit", lambda root, event, row, **kw: event != custody.PATCH_DISPOSED
                     and emit(root, event, row, **kw))
        unavailable = lambda *_a, **_k: (_ for _ in ()).throw(OSError("task result store unavailable"))
        if refresh == "native_owner":  # this run's copy is a registered native write root
            def native_owner(*_a):
                yield {"id": entry.task_id}

            down.setattr("ouroboros.headless.native_capture_custody", owned(native_owner))
            down.setattr("ouroboros.headless.refresh_native_capture", unavailable)
        else:
            down.setattr(integration, "load_task_result", unavailable)
        first = call(ctx, "integrate_delegated_patch", run_id=rid, reason="apply")
    assert (first.status, first.code) == ("blocked", "INTEGRATION_BLOCKED"), first.text
    assert "INTEGRATE_DISPOSITION_UNWRITTEN" in first.text and "records this apply without changing a file" in first.text
    assert ((f"Native capture publication is still owed: task {entry.task_id}'s capture was not refreshed"
             if refresh == "native_owner" else f"Whether task {entry.task_id}'s native capture needs a refresh is unknown")
            in first.text), first.text
    assert "task result store unavailable" in first.text, first.text
    facts = {key: first.meta[key] for key in ("target_files_changed", "files_changed", "unindexed", "disposition_recorded")}
    assert facts == {"target_files_changed": True, "files_changed": 1, "unindexed": [], "disposition_recorded": False}
    typed = _typed_result_metadata("integrate_delegated_patch", first.text, True, first)
    assert len(reviewable_effect_projection({"tool_calls": [{"tool": "integrate_delegated_patch", "is_error": True,
                                                             "result": first.text, **typed}]}, observed_only=True)) == 1
    assert (target / changed_path).read_text() == expected and _git(target, "show", f":{changed_path}").stdout == expected
    before = hashlib.sha256(index.read_bytes()).hexdigest()
    custody._CUSTODY.clear()  # the store recovers; an ordinary retry records the apply once, changing no file
    second = call(copy(ctx), "integrate_delegated_patch", run_id=rid)
    assert second.status == "ok" and second.meta["target_files_changed"] is False, second.text
    assert len(applied) == 1 and hashlib.sha256(index.read_bytes()).hexdigest() == before
    receipt, = [row for row in custody.custody_rows(drive) if row["type"] == custody.PATCH_DISPOSED]
    assert receipt["reason"] == "apply" and run(drive, rid).patch_disposed == "applied"


def legacy_result(tmp_path, monkeypatch, exclusions=()):
    """A copy recorded before result commits, with a copied file input, whose capture row did not land."""
    from ouroboros.subagent_worktrees import _update_record, provision_execution_snapshot
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive
    from ouroboros.workspace_file_outputs import _side
    from tests.test_delegated_run_isolation import _seed_target

    ctx = retry_context(tmp_path, monkeypatch, chain=("retry-a", "retry-b", "retry-c"), split=True,
                        fixed_ownership_clock=True)
    data = custody.custody_root(ctx)
    (tmp_path / "git-project").mkdir()
    target = _seed_target(tmp_path / "git-project")
    handle = provision_execution_snapshot(target_root=target, task_id="retry-a", snapshot_id="retry-snapshot")
    copy_root = Path(handle.path)
    for root in (target, copy_root):  # an input copied beside Git, as such copies received it
        (root / "input.dat").write_bytes(b"input before\n")
    _update_record(handle.path, lambda row: {**row, "file_baseline": {"input.dat": _side(copy_root / "input.dat")}},
                   op="test")
    ctx.workspace_root = target
    entry = durable_run(ctx, execution_root=handle.path, snapshot_id=handle.snapshot_id, baseline_sha=handle.baseline_sha,
                        authority_source="external_workspace_root", resource_ref={}, access="workspace_write")
    (copy_root / "tracked.txt").write_text("one\ntwo\nCHILD-EDIT\n", encoding="utf-8")
    (copy_root / "input.dat").write_bytes(b"input after\n")
    for kind in exclusions:  # inert files the current writer excludes by policy
        if kind == "evidence":
            evidence = copy_root / ".review-evidence"
            evidence.mkdir(parents=True, exist_ok=True)
            (evidence / "probe-evidence.txt").write_bytes(b"inert evidence\n")
        elif kind == "lockfile":
            (copy_root / "package-lock.json").write_bytes(b'{"name": "demo", "lockfileVersion": 3}\n')
    emit = custody.emit
    with monkeypatch.context() as unavailable:  # the capture saves its files; its receipt row does not land
        unavailable.setattr(custody, "emit", lambda root, event, row, **kw:
                            event != custody.PATCH_CAPTURED and emit(root, event, row, **kw))
        first = capture_terminal_patch_for_drive(data, entry)
    assert (first["status"], first["retry"]) == ("failed", "transient"), first
    manifest = json.loads(Path(first["manifest_artifact"]).read_bytes())
    assert "result_commit" not in manifest and [row["path"] for row in manifest["file_output_changes"]] == ["input.dat"]
    custody._CUSTODY.clear()
    return ctx, entry, target, copy_root, Path(first["manifest_artifact"]).parent


_RULES_REFUSED = ("rules_unknown", "rules_missing", "rules_711_exclusion_changed", "rules_711_copy_changed",
                  "rules_711_path_drift_apply", "rules_711_path_drift_reject")
# 7.11's writer (rules version 3) spelled these two exclusion reasons differently.
_RULES_711_RENAMES = {
    "named evidence, see the host's retention record": "named evidence, retained separately by its host",
    "incidental lockfile: no matching changed manifest found": "incidental lockfile without sibling manifest change",
}
_LEGACY_RENAME_CASES = {
    "rules_711_evidence_apply": ("evidence",),
    "rules_711_evidence_reject": ("evidence",),
    "rules_711_lockfile_apply": ("lockfile",),
    "rules_711_lockfile_reject": ("lockfile",),
    "rules_711_path_drift_apply": ("evidence", "lockfile"),
    "rules_711_path_drift_reject": ("evidence", "lockfile"),
}


@pytest.mark.parametrize("case", ["apply", "reject", "copy_changed", "copy_gone", "member_mismatch",
                                  "diffstat_only_apply", "diffstat_only_reject", "rules_711_apply", "rules_711_reject",
                                  "rules_711_evidence_apply", "rules_711_evidence_reject",
                                  "rules_711_lockfile_apply", "rules_711_lockfile_reject",
                                  *_RULES_REFUSED])
def test_saved_legacy_capture_is_decided_only_on_its_own_copy_captured_again(tmp_path, monkeypatch, case):
    import shutil
    from ouroboros import workspace_patch_capture
    from ouroboros.delegate_custody_current import run
    from tests.test_delegated_run_isolation import _git

    ctx, entry, target, copy_root, cap_dir = legacy_result(
        tmp_path, monkeypatch, exclusions=_LEGACY_RENAME_CASES.get(case, ()))
    drive, rid = custody.custody_root(ctx), entry.run_id
    restyled: list[str] = []
    if case.startswith("diffstat_only"):  # only --stat text re-renders; the patch bytes come from another command
        stat = workspace_patch_capture._git_stdout

        def reformat(cmd, cwd, **kw):
            out = stat(cmd, cwd, **kw)
            if "--stat" in cmd:
                out = "".join(f"  {line}\n" for line in out.splitlines())
                restyled.append(out)
            return out

        monkeypatch.setattr(workspace_patch_capture, "_git_stdout", reformat)
    if case.startswith("rules_"):  # 7.11's writer stamped rules version 3 on these same fields; or another label
        manifest_path = cap_dir / "workspace_patch.json"
        manifest = json.loads(manifest_path.read_bytes())
        assert manifest["exclude_rules_version"] == 4
        manifest["exclude_rules_version"] = 2 if case == "rules_unknown" else 3
        if case == "rules_missing":
            del manifest["exclude_rules_version"]
        if case == "rules_711_exclusion_changed":  # the label matches 7.11's, but not what was held out
            manifest["untracked_excluded"] = [*manifest["untracked_excluded"], {"path": "out.log", "reason": "log file"}]
            manifest["counts"]["untracked_excluded"] += 1
        if case in _LEGACY_RENAME_CASES:  # 7.11's writer spelled these two reasons differently
            manifest["untracked_excluded"] = [
                {**row, "reason": _RULES_711_RENAMES.get(row.get("reason"), row.get("reason"))}
                for row in manifest["untracked_excluded"]]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if case.startswith("rules_711_path_drift"):
        # the same number of excluded rows, but a different path: the refusal is about the path
        evidence = copy_root / ".review-evidence"
        (evidence / "probe-evidence.txt").rename(evidence / "probe-evidence-2.txt")
    if case in ("copy_changed", "rules_711_copy_changed"):  # the copy no longer holds what the saved bytes name
        (copy_root / "tracked.txt").write_text("one\ntwo\nANOTHER EDIT\n", encoding="utf-8")
    elif case == "copy_gone":
        shutil.rmtree(copy_root)
    elif case == "member_mismatch":  # an archive whose own record agrees, but whose member is not the named bytes
        import zipfile
        manifest_path = cap_dir / "workspace_patch.json"
        manifest = json.loads(manifest_path.read_bytes())
        record, = [row for row in manifest["file_outputs"] if str(row["name"]).endswith(".zip")]
        with zipfile.ZipFile(cap_dir / record["name"], "w") as zipped:
            zipped.writestr("input.dat", b"other bytes\n")
        record.update(size=(cap_dir / record["name"]).stat().st_size,
                      sha256=hashlib.sha256((cap_dir / record["name"]).read_bytes()).hexdigest())
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    saved = {path.name: path.read_bytes() for path in cap_dir.iterdir() if path.is_file()}
    before = {name: (target / name).read_bytes() for name in ("tracked.txt", "input.dat")}
    receipts = lambda: [row for row in custody.custody_rows(drive) if row["type"] == custody.PATCH_CAPTURED]
    first = call(ctx, "integrate_delegated_patch", run_id=rid,
                 decision="reject" if case.endswith("reject") else "apply")
    current = {name: (target / name).read_bytes() for name in before}
    if case in ("copy_changed", "copy_gone", "member_mismatch", *_RULES_REFUSED):
        refused = [first, call(copy(ctx), "integrate_delegated_patch", run_id=rid, decision="reject")]
        for result in refused:
            assert "INTEGRATE_DELEGATED_CAPTURE_FAILED" in result.text and "LOST" not in result.text, result.text
            assert {"copy_gone": "no longer present whole", "member_mismatch": "retained file result differs"}.get(
                case, "does not yield these bytes") in result.text, result.text
        assert not receipts() and not run(drive, rid).patch_disposed and current == before
    elif case.endswith("reject"):
        assert first.status == "ok" and "Rejected" in first.text, first.text
        assert run(drive, rid).patch_disposed == "rejected" and current == before
        if case in _LEGACY_RENAME_CASES:  # the proven legacy capture is recorded once, as on apply
            receipt, = receipts()
            assert receipt["manifest_sha256"] == hashlib.sha256(saved["workspace_patch.json"]).hexdigest()
    else:
        assert "INTEGRATE_DELEGATED_INSPECT_CAPTURE" in first.text and current == before, first.text
        receipt, = receipts()
        assert receipt["manifest_sha256"] == hashlib.sha256(saved["workspace_patch.json"]).hexdigest()
        second = call(copy(ctx), "integrate_delegated_patch", run_id=rid)
        assert second.status == "ok" and second.meta["target_files_changed"] is True, second.text
        assert (target / "tracked.txt").read_text() == "one\ntwo\nCHILD-EDIT\n"
        assert (target / "input.dat").read_bytes() == b"input after\n"
        assert "tracked.txt" in _git(target, "diff", "--cached", "--name-only").stdout
    if case.startswith("diffstat_only"):  # the stat-only drift was real, and it was not a reason to refuse
        assert restyled and restyled[-1] != json.loads(saved["workspace_patch.json"])["diffstat"]
    assert {name: (cap_dir / name).read_bytes() for name in saved} == saved  # advertised bytes are never replaced


def test_legacy_exclusion_rows_tolerate_only_a_confirmed_rename_of_a_string_reason():
    """A saved exclusion label is tolerated only as one of the confirmed strings: a non-string one is
    an ordinary mismatch (never an exception), and an unknown one is not a rename either."""
    from ouroboros.workspace_patch_rules import NAMED_EVIDENCE_EXCLUSION, legacy_manifest_mismatch

    current = {"path": ".review-evidence/x", "reason": NAMED_EVIDENCE_EXCLUSION}
    fresh = {"exclude_rules_version": 4, "untracked_excluded": [current]}
    renamed = {**current, "reason": _RULES_711_RENAMES[NAMED_EVIDENCE_EXCLUSION]}
    assert legacy_manifest_mismatch(fresh, {"exclude_rules_version": 3, "untracked_excluded": [renamed]}) is False
    for label in (None, 7, [renamed["reason"]], {renamed["reason"]: True}):
        saved = {"exclude_rules_version": 3, "untracked_excluded": [{**renamed, "reason": label}]}
        assert legacy_manifest_mismatch(fresh, saved) is True, label
    assert legacy_manifest_mismatch(  # a confirmed old label is tolerated only with exactly its new spelling
        {"exclude_rules_version": 4, "untracked_excluded": [{**current, "reason": "named evidence"}]},
        {"exclude_rules_version": 3, "untracked_excluded": [renamed]}) is True
    assert legacy_manifest_mismatch(  # an unknown string label is no rename either, even with no reason at all
        {"exclude_rules_version": 4, "untracked_excluded": [{"path": "out.log"}]},
        {"exclude_rules_version": 3, "untracked_excluded": [{"path": "out.log", "reason": "log file"}]}) is True


def test_legacy_manifest_mismatch_tolerates_only_the_confirmed_rules_transition():
    """The stamped version is tolerated only inside the confirmed 3 → 4 pair: another pair stays a
    mismatch even with the very same rows, and even with the rows that pair would have tolerated."""
    from ouroboros.workspace_patch_rules import NAMED_EVIDENCE_EXCLUSION, legacy_manifest_mismatch

    current = {"path": ".review-evidence/x", "reason": NAMED_EVIDENCE_EXCLUSION}
    renamed = {**current, "reason": _RULES_711_RENAMES[NAMED_EVIDENCE_EXCLUSION]}
    assert legacy_manifest_mismatch({"exclude_rules_version": 4, "untracked_excluded": [current]},
                                    {"exclude_rules_version": 3, "untracked_excluded": [renamed]}) is False
    for (saved_version, fresh_version), saved_rows in (((3, 5), [current]), ((3, 5), [renamed]),
                                                       ((2, 4), [renamed])):
        assert legacy_manifest_mismatch({"exclude_rules_version": fresh_version, "untracked_excluded": [current]},
                                        {"exclude_rules_version": saved_version,
                                         "untracked_excluded": saved_rows}) is True, (saved_version, fresh_version)


def test_staged_postimage_reads_the_head_basis_and_only_the_touched_index_entries(tmp_path, monkeypatch):
    from ouroboros import workspace_preflight
    from ouroboros.tools.patch_staging import index_entries, staged_postimage
    from tests.test_delegated_run_isolation import _git

    repo = tmp_path / "new-repo"
    (repo / "dir").mkdir(parents=True)
    _git(repo, "init", "--quiet")
    for name in ("a.txt", "dir/b.txt", "dir/c.txt"):
        (repo / name).write_text(f"{name}\n", encoding="utf-8")
    _git(repo, "add", "--", "a.txt", "dir/b.txt", "dir/c.txt")
    assert set(index_entries(repo, ["a.txt", "dir"])) == {"a.txt"}  # a folder's name lists none of its entries
    assert set(index_entries(repo, ["dir/b.txt", "missing.txt"])) == {"dir/b.txt"}
    unborn = staged_postimage(repo, ["a.txt"])  # a new repository's branch is positively unborn
    assert len(unborn) == 64 and staged_postimage(repo, ["a.txt"]) == unborn
    _git(repo, "symbolic-ref", "HEAD", "refs/heads/another-unborn")  # still unborn, another branch
    other = staged_postimage(repo, ["a.txt"])
    assert other not in ("", unborn)  # an unborn HEAD is identified by the branch it names
    _git(repo, "update-index", "--skip-worktree", "a.txt")
    assert staged_postimage(repo, ["a.txt"]) not in ("", other)  # an index flag is part of the state
    monkeypatch.setattr(workspace_preflight, "git_head_basis", lambda _root: ("", "unknown"))
    assert staged_postimage(repo, ["a.txt"]) == ""  # an unreadable HEAD is no proof


def test_index_entries_keeps_nfd_paths_and_flags_under_precomposeunicode(tmp_path):
    import subprocess
    from ouroboros.tools.patch_staging import index_entries, stage_applied, staged_postimage

    repo = tmp_path / "nfd-repo"
    repo.mkdir()

    def git(*args: str) -> bytes:
        return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, check=True).stdout

    git("init", "--quiet")
    git("config", "core.precomposeunicode", "true")
    tracked, intent = "cafe\u0301.txt", "resume\u0301.txt"
    (repo / tracked).write_text("v1\n", encoding="utf-8", newline="\n")
    git("-c", "core.precomposeunicode=false", "add", "--", tracked)
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "--quiet", "-m", "seed")
    (repo / intent).write_text("draft\n", encoding="utf-8", newline="\n")
    git("-c", "core.precomposeunicode=false", "add", "-N", "--", intent)
    raw_index = git("ls-files", "-z").split(b"\0")
    assert tracked.encode("utf-8") in raw_index and intent.encode("utf-8") in raw_index

    before = index_entries(repo, [tracked, intent])
    assert set(before) == {tracked, intent}
    assert len(before[tracked]) == 1 and before[tracked][0].startswith("100644 ")
    assert "+intent-to-add" in before[intent]

    (repo / tracked).write_text("v2\n", encoding="utf-8", newline="\n")
    assert stage_applied(repo, [tracked], before) == ("", [])
    after_plain = index_entries(repo, [tracked])
    assert set(after_plain) == {tracked} and after_plain[tracked][0] != before[tracked][0]
    assert git("cat-file", "blob", after_plain[tracked][0].split()[1]) == b"v2\n"
    post_plain = staged_postimage(repo, [tracked])
    assert len(post_plain) == 64

    git("-c", "core.precomposeunicode=false", "update-index", "--assume-unchanged", "--", tracked)
    git("-c", "core.precomposeunicode=false", "update-index", "--skip-worktree", "--", tracked)
    flagged = index_entries(repo, [tracked, intent])
    assert {"+assume-unchanged", "+skip-worktree"}.issubset(flagged[tracked])
    assert "+intent-to-add" in flagged[intent]
    post_flagged = staged_postimage(repo, [tracked])
    assert len(post_flagged) == 64 and post_flagged != post_plain

    (repo / tracked).write_text("v3\n", encoding="utf-8", newline="\n")
    assert stage_applied(repo, [tracked], flagged) == ("", [])
    after_flagged = index_entries(repo, [tracked])
    assert set(after_flagged) == {tracked} and after_flagged[tracked][0] != flagged[tracked][0]
    assert {"+assume-unchanged", "+skip-worktree"}.issubset(after_flagged[tracked])
    assert git("cat-file", "blob", after_flagged[tracked][0].split()[1]) == b"v3\n"
    post_after = staged_postimage(repo, [tracked])
    assert len(post_after) == 64 and post_after != post_flagged
    assert git("config", "--get", "core.precomposeunicode").strip() == b"true"


_RECORDED = ("unchanged", "successor", "verdict_folder_unavailable", "native_unwritten", "busy")
_AMBIGUOUS = ("index_drift", "head_moved", "other_apply_key", "verdict_unwritten", "postimage_unreadable")


@pytest.mark.parametrize("state", [*_RECORDED, *_AMBIGUOUS, "unadmitted"])
def test_git_disposition_write_failure_preserves_effect_and_never_applies_twice(tmp_path, monkeypatch, state):
    from ouroboros.delegate_custody_current import run
    from ouroboros.loop_tool_execution import _typed_result_metadata
    from ouroboros.outcomes import reviewable_effect_projection
    from ouroboros.subagent_worktrees import find_execution_snapshot
    from ouroboros.tools import subagent_integration as integration
    from tests.test_delegate_retry_consumers import queue_snapshot
    from tests.test_delegated_run_isolation import _git

    ctx, entry, target, changed_path, expected = captured_result(tmp_path, monkeypatch, "git")
    drive, rid = custody.custody_root(ctx), entry.run_id
    cap_dir = custody.delegated_capture_dir(drive, entry.task_id, custody.capture_key(entry))
    pinned = find_execution_snapshot(entry.snapshot_id)["result_ref"]
    index = target / ".git" / "index"
    reason = "original apply: " + " ".join(f"kept whole {n}" for n in range(60))  # longer than a row's 600
    apply, emit, applied, attempted = integration._locked_apply, custody.emit, [], []
    monkeypatch.setattr(integration, "_locked_apply", lambda *a, **kw: applied.append(a[1]) or apply(*a, **kw))
    if state == "native_unwritten":  # the native copy's journal refuses every write of this test
        monkeypatch.setattr("ouroboros.headless.record_native_capture_effect",
                            lambda *_a, **_k: (_ for _ in ()).throw(OSError("native journal unavailable")))

    def fail(root, event, row, **kw):
        if event == custody.PATCH_DISPOSED:
            attempted.append(row)
            return False
        if state == "verdict_unwritten" and event == "delegate_run_patch_verdict":
            return False
        return emit(root, event, row, **kw)

    def effects(result, error):
        facts = _typed_result_metadata("integrate_delegated_patch", result.text, error, result)
        return reviewable_effect_projection({"tool_calls": [{"tool": "integrate_delegated_patch",
                                                "is_error": error, "result": result.text, **facts}]}, observed_only=True)

    with monkeypatch.context() as unavailable:
        unavailable.setattr(custody, "emit", fail)
        if state == "postimage_unreadable":
            unavailable.setattr("ouroboros.tools.subagent_integration_delegated.staged_postimage", lambda *_: "")
        if state == "verdict_folder_unavailable":  # the optional verdict file's folder cannot be prepared
            unavailable.setattr("ouroboros.tools.patch_verdict.task_artifact_dir_path",
                                lambda *_a, **_k: (_ for _ in ()).throw(OSError("artifact folder unavailable")))
        first = call(ctx, "integrate_delegated_patch", run_id=rid, reason=reason)
    # A proven apply whose native receipt failed withholds its disposition instead of closing it unreceipted.
    assert len(attempted) == (state != "native_unwritten") and len(applied) == 1
    # The published code is the INTEGRATE_ family's answer: only the receipt is missing,
    # and the failure must keep counting in the verification ledger (not TOOL_REPORTED_FAILURE).
    assert (first.status, first.code) == ("blocked", "INTEGRATION_BLOCKED")
    assert "INTEGRATE_DISPOSITION_UNWRITTEN" in first.text and "staging completed" in first.text
    # The transfer stays a typed effect: where, how many files and which were delivered unstaged.
    assert first.meta["target_files_changed"] is True and first.meta["disposition_recorded"] is False
    assert Path(first.meta["target_root"]).resolve() == target.resolve()
    assert (first.meta["files_changed"], first.meta["unindexed"]) == (1, [])
    assert len(effects(first, True)) == 1
    assert ("native copy's capture could not record" in first.text) == (state == "native_unwritten")
    assert (target / changed_path).read_bytes() == expected.encode()
    assert _git(target, "show", f":{changed_path}").stdout == expected
    custody._CUSTODY.clear()  # later call uses a fresh registry and reconstructs from committed custody
    pending = run(drive, rid)
    assert pending.patch_apply_pending and pending.patch_apply_key and not pending.patch_disposed
    assert pending.patch_apply_by == ctx.task_id
    has_proof = state not in {"verdict_unwritten", "postimage_unreadable"}
    assert ("records this apply without changing a file" in first.text) == has_proof, first.text
    assert ("refuses as ambiguous instead of applying again" in first.text) == (not has_proof), first.text
    assert "acknowledge_ambiguous" not in first.text
    caller = copy(ctx)
    if state == "index_drift":
        _git(target, "reset", "--quiet", "HEAD", "--", changed_path)
        assert _git(target, "show", f":{changed_path}").stdout != expected
    elif state == "head_moved":  # HEAD alone moves: same tree, same index and files
        moved = _git(target, "-c", "user.email=t@t", "-c", "user.name=t", "commit-tree", "HEAD^{tree}", "-p", "HEAD",
                     "-m", "moved").stdout.strip()
        _git(target, "update-ref", "HEAD", moved)
    elif state == "other_apply_key":  # an intent that names no author
        assert custody.record_patch_apply_started(drive, pending, target_root=str(target), apply_idempotency_key="other")
    elif state == "successor":  # the host confirms retry-d as retry-c's successor
        fields = {"root_task_id": "retry-a", "parent_task_id": "", "delegation_role": "root"}
        write_task_result(drive, "retry-c", "interrupted", reason_code="idle_timeout", supersedes_task_id="retry-b",
                          original_task_id="retry-b", timeout_retry_from="retry-b", superseded_by="retry-d",
                          retry_task_id="retry-d", **fields)
        write_task_result(drive, "retry-d", "running", supersedes_task_id="retry-c", original_task_id="retry-c",
                          timeout_retry_from="retry-c", **fields)
        caller.task_id = "retry-d"
        queue_snapshot(caller, ["retry-d"])
    elif state == "unadmitted":
        caller.task_id = "unrelated-task"
    before = hashlib.sha256(index.read_bytes()).hexdigest()
    if state == "unchanged":
        with monkeypatch.context() as unavailable:
            unavailable.setattr(custody, "emit", fail)
            unrecorded = call(copy(caller), "integrate_delegated_patch", run_id=rid)
        assert (unrecorded.status, unrecorded.code) == ("blocked", "INTEGRATION_BLOCKED")
        assert "INTEGRATE_DISPOSITION_UNWRITTEN" in unrecorded.text
        assert unrecorded.meta["target_files_changed"] is False and not effects(unrecorded, True)
        assert hashlib.sha256(index.read_bytes()).hexdigest() == before
    if state == "busy":  # a busy target is retried later; the intent stays and nothing changes
        with monkeypatch.context() as held:
            held.setattr("ouroboros.tools.git._acquire_git_lock",
                         lambda *_a, **_k: (_ for _ in ()).throw(TimeoutError("git lock held")))
            later = call(copy(caller), "integrate_delegated_patch", run_id=rid)
        assert "INTEGRATE_NOT_STARTED" in later.text and "Retry later" in later.text, later.text
        assert "acknowledge_ambiguous" not in later.text and run(drive, rid).patch_apply_pending
    if state == "native_unwritten":  # while the native write still fails, nothing is disposed or applied
        held = call(copy(caller), "integrate_delegated_patch", run_id=rid)
        assert (held.status, held.code) == ("blocked", "INTEGRATION_BLOCKED")
        assert "INTEGRATE_DISPOSITION_UNWRITTEN" in held.text, held.text
        assert "native copy's capture could not record" in held.text and held.meta["target_files_changed"] is False
        assert run(drive, rid).patch_apply_pending and not run(drive, rid).patch_disposed
        monkeypatch.setattr("ouroboros.headless.record_native_capture_effect", lambda *_a, **_k: None)
    second = call(copy(caller), "integrate_delegated_patch", run_id=rid)
    assert len(applied) == 1 and hashlib.sha256(index.read_bytes()).hexdigest() == before
    assert (target / changed_path).read_bytes() == expected.encode()
    rows = [row for row in custody.custody_rows(drive) if row["type"] == custody.PATCH_DISPOSED]
    final = run(drive, rid)
    if state in _RECORDED:
        assert second.status == "ok", second.text
        assert second.meta["target_files_changed"] is False and effects(second, False) == []
        assert "native copy's capture could not record" not in second.text
        receipt, = rows
        assert receipt["reason"] == reason and receipt["apply_idempotency_key"] == pending.patch_apply_key
        assert receipt["disposed_by_task_id"] == caller.task_id
        assert final.patch_disposed == "applied" and not final.patch_apply_pending
    else:
        assert second.status == "blocked", second.text
        assert ("INTEGRATE_DELEGATED_NOT_OWNED" if state == "unadmitted"
                else "INTEGRATE_DELEGATED_APPLY_AMBIGUOUS") in second.text, second.text
        assert second.meta.get("target_files_changed") is not True and effects(second, True) == []
        assert not rows and final.patch_apply_pending and not final.patch_disposed
        assert (cap_dir / "workspace.patch").is_file()
        assert _git(target, "rev-parse", "--verify", pinned).stdout.strip()


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
