"""The maintenance child prepares real evidence; the server alone accepts its result."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ouroboros import delegate_capture_job as job, delegate_collection as body, delegate_custody as custody
from ouroboros import delegate_custody_current as current, maintenance
from ouroboros.subagent_worktrees import find_execution_snapshot
from tests.test_delegate_body_collection import terminal
from tests.test_recovered_capture import prepared


def _queued(tmp_path, monkeypatch):
    root = tmp_path / "data"
    entry, area, _ = prepared(tmp_path, root, "ordinary")
    requests = []
    monkeypatch.setattr(maintenance.child, "request", lambda: requests.append(True))
    result = body.collect_terminal(root, entry.run_id, detail=terminal(entry.run_id), background=True)
    queued = current.run(root, entry.run_id)
    # The capture is owed durably and the child is woken; a first money publication may wake the
    # same child in the same call (cold metrics), so the wake itself is not exclusive.
    assert result["workspace_capture"]["status"] == "pending" and queued.capture_job.get("state") == "pending"
    assert requests
    return root, queued, area


def _child(root, *, fail_named=False):
    code = '''
import errno, json, os, pathlib, sys
from ouroboros import delegate_custody as c, delegate_collection as body, maintenance
def forbidden(*a, **kw):
    raise AssertionError("a child tried to make a server ownership decision")
c.record_patch_captured = forbidden
c.record_patch_disposed = forbidden
body.record_terminal_money = forbidden
if sys.argv[2] == "fail":
    from ouroboros import delegate_named_evidence
    def locked(*a, **kw):
        raise OSError(errno.EBUSY, "a native capture handle is busy")
    delegate_named_evidence.retain_named_evidence = locked
result = maintenance.run_job(pathlib.Path(sys.argv[1]), "delegate_capture")
print(json.dumps({"pid": os.getpid(), "result": result}))
'''
    result = subprocess.run([sys.executable, "-c", code, str(root), "fail" if fail_named else ""],
                            capture_output=True, text=True, timeout=60, check=True)
    return json.loads(result.stdout)


def test_real_child_prepares_bytes_and_server_settles_and_publishes(tmp_path, monkeypatch):
    root, entry, area = _queued(tmp_path, monkeypatch)
    assert entry.settled and entry.output_complete and not entry.patch_captured
    assert list(maintenance.JOBS).index(job.NAME) < list(maintenance.JOBS).index("dialogue_import")
    report = _child(root)
    assert report["pid"] != os.getpid() and report["result"] == {"state": "done", "items": 1}
    prepared_entry = current.run(root, entry.run_id)
    assert prepared_entry.capture_job["state"] == "ready"
    assert not prepared_entry.patch_captured and not prepared_entry.named_evidence and area.is_dir()
    completed = body.collect_terminal(root, entry.run_id, background=True)
    final = current.run(root, entry.run_id)
    assert completed["workspace_capture"]["named_evidence"]["status"] == "retained"
    assert final.patch_captured and not final.patch_disposed and not current.collection_pending(final)
    assert _child(root)["result"] is None
    assert current.run(root, entry.run_id).collection_attempt == final.collection_attempt


def test_capture_wake_follows_server_publication_and_lock_release(tmp_path, monkeypatch):
    root = tmp_path / "data"
    entry, area, _ = prepared(tmp_path, root, "ordinary")
    wakes = []
    def request():
        fresh = current.run(root, entry.run_id)
        if fresh.capture_job.get("state") == "pending":
            wakes.append((custody.disposition_lock_path(root, entry).exists(), bool(fresh.collection_attempt)))
    monkeypatch.setattr(maintenance.child, "request", request)
    body.collect_terminal(root, entry.run_id, detail=terminal(entry.run_id), background=True)
    assert wakes == [(False, True)]
    assert _child(root)["result"] == {"state": "done", "items": 1}
    assert area.is_dir()  # a child's prepared result never releases the copy
    wakes.clear()
    body.collect_terminal(root, entry.run_id, background=True)
    # The server accepted it: capture row and named evidence saved, so the checkout went (the record keeps C).
    assert not wakes and not area.exists()
    assert find_execution_snapshot(entry.snapshot_id)["result_sha"]


def test_explicit_git_inspection_uses_the_ready_child_result_before_collection(tmp_path, monkeypatch):
    from ouroboros.delegate_start_instructions import EVIDENCE_DIRNAME
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch
    from tests.test_delegated_run_isolation import _git

    root, entry, area = _queued(tmp_path, monkeypatch)
    rid, target = entry.run_id, Path(entry.target_root)
    repo = tmp_path / "host-repo"
    repo.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=root, task_id=entry.task_id,
                      workspace_root=target, workspace_mode="external")
    (area / "check.py").unlink()
    (area / "plain-note.txt").write_bytes(b"Plain capture product.\n")
    evidence = area / EVIDENCE_DIRNAME
    (evidence / "result.bin").unlink()
    (evidence / "notes.txt").write_bytes(b"Plain named evidence.\n")
    assert job.run(root, lambda: False)["items"] == 1
    ready = current.run(root, rid)
    child = body._read_source(root, ready, ready.capture_job["source"])
    path = Path(child["manifest_artifact"])
    saved = path.read_bytes()
    assert ready.capture_job["state"] == "ready" and not ready.patch_captured

    inspected = _integrate_delegated_patch(ctx, rid)
    head, _, described = inspected.partition("\n")
    assert "INTEGRATE_DELEGATED_INSPECT_CAPTURE" in head, inspected
    assert not (target / "plain-note.txt").exists()
    # The inspection shows the receipt this call just published, not the row it read before publishing.
    shown, published = json.loads(described), current.run(root, rid)
    assert published.patch_captured and published.patch_manifest_sha == ready.capture_job["manifest_sha256"]
    assert (shown["status"], shown["sha256"], shown["manifest_artifact"]) == (
        "ready_with_changes", child["sha256"], str(path)), shown
    assert "retry" not in shown and "disposition" not in shown and "NOT APPLIED" in shown["note"]
    assert shown["named_evidence"]["status"] == "retained"
    assert shown["named_evidence"]["evidence_dir"] == child["named_evidence"]["evidence_dir"]
    assert [row["path"] for row in shown["named_evidence"]["files"]] == [f"{EVIDENCE_DIRNAME}/notes.txt"]
    assert shown == json.loads(json.dumps(body.capture_projection(root, published, reader_task_id=entry.task_id)))
    collected = body.collect_terminal(root, rid, background=True)
    fresh = current.run(root, rid)
    assert (path.read_bytes() == saved, collected["status"]) == (True, "collected"), collected
    assert fresh.capture_job["id"] == ready.capture_job["id"] and fresh.capture_job["state"] == "consumed"
    assert fresh.patch_manifest_sha == ready.capture_job["manifest_sha256"] and not fresh.patch_disposed
    named = collected["workspace_capture"]["named_evidence"]
    assert named["evidence_dir"] == child["named_evidence"]["evidence_dir"]

    applied = _integrate_delegated_patch(ctx, rid)
    assert "✅ Integrated" in applied, applied
    assert _git(target, "show", ":plain-note.txt").stdout == "Plain capture product.\n"
    assert path.read_bytes() == saved and not (target / EVIDENCE_DIRNAME).exists()
    rows = custody.custody_rows(root)
    assert sum(row["type"] == custody.PATCH_CAPTURED for row in rows) == 1
    assert sum(row["type"] == custody.PATCH_DISPOSED for row in rows) == 1


@pytest.mark.parametrize("lost", ["patch_receipt", "consume_receipt"])
def test_unwritten_capture_publication_rejoins_prepared_bytes(tmp_path, monkeypatch, lost):
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.delegate_capture_attempt import collection_due
    from tests.test_delegate_retry_consumers import call, retry_context

    ctx = retry_context(tmp_path, monkeypatch, chain=("investigation", "retry-b"), split=True,
                        fixed_ownership_clock=True)
    root = custody.custody_root(ctx)
    entry, area, _ = prepared(tmp_path, root, "investigation")
    (area / "plain-note.txt").write_bytes(b"Plain retained output.\n")
    wakes = []
    monkeypatch.setattr(maintenance.child, "request", lambda: wakes.append(True))
    body.collect_terminal(root, entry.run_id, detail=terminal(entry.run_id), background=True, generation="one")
    assert job.run(root, lambda: False)["items"] == 1
    waiting = current.run(root, entry.run_id)
    emit = custody.emit

    def fail(root, event, facts, **kwargs):
        if (lost == "patch_receipt" and event == custody.PATCH_CAPTURED) or (lost == "consume_receipt"
                and event == custody.CAPTURE_JOB_CHANGED and facts["outcome"]["state"] == "consumed"):
            return False
        return emit(root, event, facts, **kwargs)

    with monkeypatch.context() as unavailable:
        unavailable.setattr(custody, "emit", fail)
        unavailable.setattr("ouroboros.claudexor_daemon.read_owned_gateway",
                            lambda: pytest.fail("retained output or ready capture opened the engine"))
        wakes.clear()
        result = body.collect_terminal(root, entry.run_id, background=True, generation="one")
        failed = current.run(root, entry.run_id)
        assert result["status"] == "pending" and failed.output_complete and failed.settled
        assert result["workspace_capture"]["status"] == (
            "failed" if lost == "patch_receipt" else "ready_with_changes")
        assert failed.capture_job == waiting.capture_job and not failed.patch_disposed and area.is_dir()
        assert failed.patch_captured == (lost == "consume_receipt") and not wakes
        assert not collection_due(failed, generation="one") and collection_due(failed, generation="two")
        response = call(ctx, "delegate_wait", run_id=entry.run_id)
        delivery = json.loads(response.text)
        assert delivery["primary_output"]["text"] == "complete answer"
        assert delivery["collection"]["status"] == "pending"
        block = delivery["workspace_capture"]
        assert block["status"] == result["workspace_capture"]["status"]
        assert Path(block["patch_read"]["path"]).is_absolute()
        read = call(ctx, "read_file", **block["patch_read"])
        assert read.status == "ok" and "+Plain retained output." in read.text
        saved_patch = Path(block["patch_read"]["path"]).read_bytes()
        assert b"+Plain retained output.\n" in saved_patch

    custody._CUSTODY.clear()
    if lost == "patch_receipt":
        final = body.collect_terminal(root, entry.run_id, background=True, generation="two")
        assert final["status"] == "collected"
    else:
        final = call(ctx, "delegate_wait", run_id=entry.run_id)
        assert json.loads(final.text)["collection"]["status"] == "retained"
    settled = current.run(root, entry.run_id)
    assert settled.patch_disposed == custody.EVIDENCE_RETAINED
    assert settled.capture_job["id"] == waiting.capture_job["id"] and settled.capture_job["state"] == "consumed"
    # A successor gets its own stage address, but the complete bytes never change.
    assert settled.output_sha == waiting.output_sha and not area.exists() and not wakes
    assert read_actor_source_bytes(root, settled.output_source["task_id"], settled.output_source) == (
        read_actor_source_bytes(root, waiting.output_source["task_id"], waiting.output_source))
    rows = [row for row in custody.custody_rows(root) if row["type"] == custody.PATCH_CAPTURED]
    assert len(rows) == 1 and rows[0]["sha256"] == block["sha256"]
    assert Path(block["patch_read"]["path"]).read_bytes() == saved_patch


def test_transient_waits_for_another_child_generation_without_spawning_it(tmp_path, monkeypatch):
    root, entry, area = _queued(tmp_path, monkeypatch)
    _child(root, fail_named=True)
    body.collect_terminal(root, entry.run_id, background=True)
    failed = current.run(root, entry.run_id)
    assert failed.named_evidence["status"] == "capture_failed" and failed.settled
    assert failed.capture_job["retry"] == "transient" and current.collection_pending(failed)
    assert area.is_dir()  # a failed retention keeps the copy for the retry
    assert not job.due(root, {}, {}, False), "a transient alone cannot spawn another child"
    assert _child(root)["result"] == {"state": "done", "items": 1}
    prepared = current.run(root, entry.run_id)
    assert prepared.capture_job["id"] != failed.capture_job["id"]
    body.collect_terminal(root, entry.run_id, background=True)
    final = current.run(root, entry.run_id)
    assert final.named_evidence["status"] == "retained" and not current.collection_pending(final)
    assert _child(root)["result"] is None and not area.exists()


def test_continuation_between_child_return_and_server_acceptance_keeps_its_snapshot(tmp_path, monkeypatch):
    root, entry, area = _queued(tmp_path, monkeypatch)
    _child(root)
    successor = custody.RunCustody(run_id="continued", task_id=entry.task_id, invocation_id="next-invocation",
        snapshot_id=entry.snapshot_id, execution_root=entry.execution_root, baseline_sha=entry.baseline_sha,
        target_root=entry.target_root, authority_source=entry.authority_source, mode="agent",
        continuation_of=entry.run_id, capture_id="next-invocation")
    assert custody.record_started(root, successor)
    body.collect_terminal(root, entry.run_id, background=True)
    stale = current.run(root, entry.run_id)
    assert stale.superseded_by == successor.run_id and not stale.patch_captured
    assert not stale.named_evidence and area.is_dir()
    assert stale.capture_job["state"] == "ready", "a superseded result was never accepted"
    assert not current.run(root, successor.run_id).settled


def test_a_child_contending_with_the_disposition_owner_does_not_capture(tmp_path, monkeypatch):
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    root, entry, area = _queued(tmp_path, monkeypatch)
    path = custody.disposition_lock_path(root, entry)
    fd = acquire_exclusive_file_lock(path, timeout_sec=0, owner_aware_stale=True)
    assert fd is not None
    try:
        assert _child(root)["result"] == {"state": "pending", "items": 0}
        assert current.run(root, entry.run_id).capture_job["state"] == "pending"
        assert not (path.parent / "workspace_patch.json").exists()
    finally:
        release_exclusive_file_lock(path, fd)
    assert _child(root)["result"] == {"state": "done", "items": 1} and area.is_dir()


def test_first_child_after_owner_exit_reclaims_and_captures(tmp_path, monkeypatch):
    root, entry, area = _queued(tmp_path, monkeypatch)
    path = custody.disposition_lock_path(root, entry)
    code = """
import os, pathlib, sys
from ouroboros.platform_layer import acquire_exclusive_file_lock
assert acquire_exclusive_file_lock(pathlib.Path(sys.argv[1]), owner_aware_stale=True) is not None
os._exit(0)  # Leave the name behind, exactly as a stopped capture child does.
"""
    subprocess.run([sys.executable, "-c", code, str(path)], check=True, timeout=30)
    assert path.is_file()
    assert _child(root)["result"] == {"state": "done", "items": 1}
    assert current.run(root, entry.run_id).capture_job["state"] == "ready" and area.is_dir()
