"""The maintenance child prepares real evidence; the server alone accepts its result."""
from __future__ import annotations

import json
import os
import subprocess
import sys

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
