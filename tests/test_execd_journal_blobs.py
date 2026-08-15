# tests/test_execd_journal_blobs.py — the target's DURABLE STORAGE layer.
#
# Split out of test_execd_state.py, and the seam is the subject rather than the line
# count: everything here is about what the target REMEMBERS across a restart — the
# operation journal (task-bound, idempotent, fail-closed on a failed start write) and
# the content-addressed blob store that backs it (GC that may never reclaim a blob an
# unacknowledged journal row still references, pins visible across store instances,
# corruption detected on read). The custody, lease and protocol tests stayed behind;
# they are about what the target is DOING, not what it has kept.

import hashlib
import pathlib
import subprocess
import time
from typing import Any

import pytest

import ouroboros.execd_state as state_module
from ouroboros.execd import ExecdService
from ouroboros.execd_state import (
    CASBlobStore,
    ExecdError,
    OperationJournal,
    initialize_continuity_host_id,
)
from ouroboros.workspace_native import (
    MANDATORY_REMOTE_NATIVE_OPERATIONS,
)


def _capability_manifest() -> dict[str, Any]:
    return {
        "manifest_sha256": "a" * 64,
        "native_operations": sorted(MANDATORY_REMOTE_NATIVE_OPERATIONS),
    }


def _git_workspace(path: pathlib.Path) -> pathlib.Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "execd-tests@example.invalid"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Execd Tests"],
        cwd=path,
        check=True,
    )
    (path / "README.md").write_text("remote-only\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=path, check=True)
    return path


def _service(
    tmp_path: pathlib.Path,
    *,
    generation: str = "generation-a",
    connection_id: str = "connection-a",
    project_id: str = "project-a",
) -> ExecdService:
    workspace = tmp_path / "workspace"
    if not workspace.exists():
        _git_workspace(workspace)
    initialize_continuity_host_id(tmp_path / "state")
    return ExecdService(
        tmp_path / "state",
        workspace,
        connection_id=connection_id,
        project_id=project_id,
        server_generation=generation,
        release_id="test-release",
        artifact_sha256="f" * 64,
        capability_manifest=_capability_manifest(),
    )




def _journal(tmp_path: pathlib.Path) -> OperationJournal:
    blobs = CASBlobStore(tmp_path / "blobs")
    return OperationJournal(
        tmp_path / "operations",
        connection_id="connection-a",
        workspace_id="workspace-a",
        spool=CASBlobStore(tmp_path / "spool"),
        blobs=blobs,
    )


def _begin(
    journal: OperationJournal,
    *,
    task_id: str = "task-a",
    operation_id: str = "operation-a",
    request_hash: str = "b" * 64,
) -> tuple[str, dict[str, Any] | None]:
    return journal.begin(
        task_id=task_id,
        operation_id=operation_id,
        request_hash=request_hash,
        binding={"task_id": task_id, "operation_id": operation_id},
    )


def test_journal_is_task_bound_even_when_operation_id_and_hash_match(tmp_path):
    journal = _journal(tmp_path)
    assert _begin(journal) == ("started", None)

    with pytest.raises(ExecdError):
        _begin(journal, task_id="task-b")

    reopened = _journal(tmp_path)
    with pytest.raises(ExecdError):
        reopened.reconcile("task-b", "operation-a", "b" * 64)

    with pytest.raises(ExecdError):
        reopened.acknowledge("task-b", "operation-a", "b" * 64)


def test_journal_duplicate_completion_and_unknown_started_are_reconciled(tmp_path):
    journal = _journal(tmp_path)
    _begin(journal)

    assert journal.reconcile("task-a", "operation-a", "b" * 64) == {
        "completion": "unknown"
    }
    result = {"completion": "completed", "answer": "ok"}
    journal.complete(
        task_id="task-a",
        operation_id="operation-a",
        request_hash="b" * 64,
        result=result,
    )

    assert _begin(journal) == ("completed", result)
    assert journal.reconcile("task-a", "operation-a", "b" * 64) == {
        "completion": "completed",
        "result": result,
        "result_unavailable": False,
    }
    with pytest.raises(ExecdError) as conflict:
        _begin(journal, request_hash="c" * 64)
    assert conflict.value.code == "operation_id_conflict"


def test_journal_start_write_failure_is_fail_closed(tmp_path, monkeypatch):
    journal = _journal(tmp_path)
    original = state_module.durable_json

    def fail_start(path, payload):
        if payload.get("state") == "started":
            raise OSError("disk full")
        return original(path, payload)

    monkeypatch.setattr(state_module, "durable_json", fail_start)
    with pytest.raises(ExecdError) as failure:
        _begin(journal)

    assert failure.value.code == "journal_start_failed"
    assert journal.reconcile("task-a", "operation-a", "b" * 64) == {
        "completion": "not_started"
    }


def test_missing_spooled_result_is_unavailable_and_never_reruns(tmp_path, monkeypatch):
    monkeypatch.setattr(state_module, "MAX_RESULT_BYTES", 32)
    journal = _journal(tmp_path)
    _begin(journal)
    journal.complete(
        task_id="task-a",
        operation_id="operation-a",
        request_hash="b" * 64,
        result={"completion": "completed", "payload": "x" * 1000},
    )
    record = journal.list_records()[0]
    spool_path = journal.spool.path_for(record["result_blob_id"])
    spool_path.unlink()

    reconciled = journal.reconcile("task-a", "operation-a", "b" * 64)
    assert reconciled == {
        "completion": "completed",
        "result": None,
        "result_unavailable": True,
    }
    assert _begin(journal) == ("completed", None)


def test_ack_prunes_only_old_acknowledged_records(tmp_path, monkeypatch):
    monkeypatch.setattr(state_module, "MAX_RETAINED_ACKED_OPERATIONS", 2)
    monkeypatch.setattr(state_module, "ACKED_BLOB_EXPORT_GRACE_MS", 0)
    journal = _journal(tmp_path)
    for index in range(4):
        operation_id = f"operation-{index}"
        request_hash = hashlib.sha256(operation_id.encode()).hexdigest()
        _begin(
            journal,
            operation_id=operation_id,
            request_hash=request_hash,
        )
        journal.complete(
            task_id="task-a",
            operation_id=operation_id,
            request_hash=request_hash,
            result={"completion": "completed", "index": index},
        )
        journal.acknowledge("task-a", operation_id, request_hash)
        time.sleep(0.002)

    rows = journal.list_records()
    assert len(rows) == 2
    assert {row["operation_id"] for row in rows} == {
        "operation-2",
        "operation-3",
    }


def test_journal_live_capacity_ignores_bounded_acknowledged_rows(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(state_module, "MAX_LIVE_OPERATIONS", 1)
    monkeypatch.setattr(state_module, "MAX_TOTAL_OPERATION_RECORDS", 3)
    journal = _journal(tmp_path)
    _begin(journal, operation_id="operation-acked", request_hash="3" * 64)
    journal.complete(
        task_id="task-a",
        operation_id="operation-acked",
        request_hash="3" * 64,
        result={"completion": "completed"},
    )
    journal.acknowledge("task-a", "operation-acked", "3" * 64)

    assert _begin(
        journal,
        operation_id="operation-live",
        request_hash="4" * 64,
    ) == ("started", None)
    with pytest.raises(ExecdError, match="capacity"):
        _begin(
            journal,
            operation_id="operation-blocked",
            request_hash="5" * 64,
        )


def test_blob_gc_preserves_every_unacknowledged_journal_reference(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(state_module, "MAX_RESULT_BYTES", 32)
    journal = _journal(tmp_path)
    assert journal.blobs is not None
    input_blob = journal.blobs.put(b"prepared input")
    output_blob = journal.blobs.put(b"unfetched output")
    request_hash = "d" * 64
    journal.begin(
        task_id="task-live",
        operation_id="operation-live",
        request_hash=request_hash,
        binding={"blob_hashes": {"upload": input_blob}},
    )
    journal.complete(
        task_id="task-live",
        operation_id="operation-live",
        request_hash=request_hash,
        result={
            "completion": "completed",
            "payload": "x" * 1000,
            "output_blobs": {output_blob: output_blob},
        },
    )
    live = journal.list_records()[0]
    spooled_result = str(live["result_blob_id"])

    _begin(
        journal,
        task_id="task-acked",
        operation_id="operation-acked",
        request_hash="e" * 64,
    )
    journal.complete(
        task_id="task-acked",
        operation_id="operation-acked",
        request_hash="e" * 64,
        result={"completion": "completed"},
    )
    monkeypatch.setattr(state_module, "MAX_CAS_STORE_BLOBS", 0)
    monkeypatch.setattr(state_module, "MAX_CAS_STORE_BYTES", 0)
    monkeypatch.setattr(state_module, "CAS_ORPHAN_RETENTION_SECONDS", 0)
    journal.acknowledge("task-acked", "operation-acked", "e" * 64)

    assert journal.blobs.path_for(input_blob).exists()
    assert journal.blobs.path_for(output_blob).exists()
    assert journal.spool.path_for(spooled_result).exists()
    with pytest.raises(ExecdError) as full:
        journal.blobs.put(b"new blob")
    assert full.value.code == "blob_capacity_exhausted"


def test_blob_gc_reclaims_acked_result_and_staged_orphans(tmp_path, monkeypatch):
    monkeypatch.setattr(state_module, "MAX_RESULT_BYTES", 32)
    journal = _journal(tmp_path)
    assert journal.blobs is not None
    output_blob = journal.blobs.put(b"already imported output")
    staged_blob = journal.blobs.put(b"staged upload")
    journal.blobs.pin(staged_blob)
    request_hash = "f" * 64
    journal.begin(
        task_id="task-acked",
        operation_id="operation-acked",
        request_hash=request_hash,
        binding={},
    )
    journal.complete(
        task_id="task-acked",
        operation_id="operation-acked",
        request_hash=request_hash,
        result={
            "completion": "completed",
            "payload": "x" * 1000,
            "output_blobs": {output_blob: output_blob},
        },
    )
    spooled_result = str(journal.list_records()[0]["result_blob_id"])
    monkeypatch.setattr(state_module, "MAX_CAS_STORE_BLOBS", 0)
    monkeypatch.setattr(state_module, "MAX_CAS_STORE_BYTES", 0)
    monkeypatch.setattr(state_module, "CAS_ORPHAN_RETENTION_SECONDS", 0)
    monkeypatch.setattr(state_module, "ACKED_BLOB_EXPORT_GRACE_MS", 0)
    monkeypatch.setattr(
        state_module,
        "MAX_RETAINED_ACKED_OPERATION_AGE_MS",
        0,
    )

    journal.acknowledge("task-acked", "operation-acked", request_hash)

    assert journal.list_records() == []
    assert not journal.blobs.path_for(output_blob).exists()
    assert not journal.spool.path_for(spooled_result).exists()
    assert journal.blobs.path_for(staged_blob).exists()
    journal.blobs.unpin(staged_blob)
    journal.blobs.collect_garbage(set())
    assert not journal.blobs.path_for(staged_blob).exists()


def test_blob_gc_keeps_recent_acked_exports_until_bounded_grace_expires(
    tmp_path,
    monkeypatch,
):
    journal = _journal(tmp_path)
    assert journal.blobs is not None
    output_blob = journal.blobs.put(b"pending Home export")
    request_hash = "1" * 64
    journal.begin(
        task_id="task-export",
        operation_id="operation-export",
        request_hash=request_hash,
        binding={},
    )
    journal.complete(
        task_id="task-export",
        operation_id="operation-export",
        request_hash=request_hash,
        result={"output_blobs": {output_blob: output_blob}},
    )
    monkeypatch.setattr(state_module, "MAX_CAS_STORE_BLOBS", 0)
    monkeypatch.setattr(state_module, "MAX_CAS_STORE_BYTES", 0)
    monkeypatch.setattr(state_module, "CAS_ORPHAN_RETENTION_SECONDS", 0)
    journal.acknowledge("task-export", "operation-export", request_hash)
    assert len(journal.list_records()) == 1
    assert journal.blobs.path_for(output_blob).exists()

    monkeypatch.setattr(state_module, "ACKED_BLOB_EXPORT_GRACE_MS", 0)
    _begin(
        journal,
        task_id="task-trigger",
        operation_id="operation-trigger",
        request_hash="2" * 64,
    )
    journal.complete(
        task_id="task-trigger",
        operation_id="operation-trigger",
        request_hash="2" * 64,
        result={"completion": "completed"},
    )
    journal.acknowledge("task-trigger", "operation-trigger", "2" * 64)
    assert not journal.blobs.path_for(output_blob).exists()


def test_cas_reserves_the_full_snapshot_transaction_above_4096_blobs(tmp_path):
    store = CASBlobStore(tmp_path / "blobs")
    for index in range(4100):
        payload = index.to_bytes(4, "big")
        store.path_for(hashlib.sha256(payload).hexdigest()).write_bytes(payload)

    collected = store.collect_garbage(set())

    assert state_module.MAX_CAS_ATOMIC_BLOB_RESERVE == max(
        state_module.MAX_SNAPSHOT_FILES + 1,
        state_module.MAX_ATTACHMENT_COUNT,
    )
    assert state_module.MAX_CAS_STORE_BLOBS >= 32_768
    assert collected["removed_count"] == 0
    assert store.put(b"next snapshot blob")


def test_project_scoped_transport_state_prevents_cross_process_gc(tmp_path):
    first = _service(tmp_path, project_id="project-a")
    second = _service(tmp_path, project_id="project-b")
    digest = first.cas.put(b"project-a staged blob")
    first.cas.pin(digest)

    assert first.cas.root != second.cas.root
    assert first.spool.root != second.spool.root
    assert first.journal.root != second.journal.root
    assert first.custody.state_path != second.custody.state_path
    second.cas.collect_garbage(set())
    assert first.cas.path_for(digest).exists()


def test_cas_persistent_pin_is_visible_to_another_store_instance(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "shared-cas"
    owner = CASBlobStore(root)
    collector = CASBlobStore(root)
    digest = owner.put(b"cross-process staged blob")
    owner.pin(digest)
    monkeypatch.setattr(state_module, "MAX_CAS_STORE_BLOBS", 0)
    monkeypatch.setattr(state_module, "MAX_CAS_STORE_BYTES", 0)
    monkeypatch.setattr(state_module, "CAS_ORPHAN_RETENTION_SECONDS", 0)

    collector.collect_garbage(set())
    assert owner.path_for(digest).exists()

    owner.unpin(digest)
    collector.collect_garbage(set())
    assert not owner.path_for(digest).exists()


def test_cas_rejects_wrong_hash_and_detects_corruption(tmp_path):
    store = CASBlobStore(tmp_path / "blobs")
    payload = b"bounded remote blob"
    digest = store.put(payload)
    assert store.read(digest, max_bytes=len(payload)) == payload

    with pytest.raises(ExecdError) as mismatch:
        store.put(payload, expected_sha256="0" * 64)
    assert mismatch.value.code == "blob_hash_mismatch"

    store.path_for(digest).write_bytes(b"corrupt")
    with pytest.raises(ExecdError) as corrupt:
        store.read(digest, max_bytes=1024)
    assert corrupt.value.code == "blob_store_corrupt"
