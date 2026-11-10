"""Continuity Phase 3C: child forensic refs survive headless-drive GC."""

from __future__ import annotations

from ouroboros.headless import retry_child_task_refs

import json
import pathlib


from ouroboros.headless import (
    copy_child_task_result,
    prepare_task_drive,
    prune_headless_task_drives,
)
from ouroboros.observability import persist_call, read_blob_ref, write_blob
from ouroboros.task_results import STATUS_COMPLETED, write_task_result


def _child(tmp_path: pathlib.Path, task_id: str) -> tuple[pathlib.Path, pathlib.Path]:
    parent = tmp_path / "data"
    parent.mkdir()
    child = prepare_task_drive(parent, task_id, "empty")
    assert child is not None
    return parent, child


def _future_now() -> float:
    return 4_000_000_000.0


def _manifest(ref: dict) -> dict:
    return json.loads(pathlib.Path(ref["path"]).read_text(encoding="utf-8"))


def test_copyback_promotes_service_full_log_refs_in_durable_evidence_and_tool_payload(tmp_path):
    task_id = "phase3c-service"
    parent, child = _child(tmp_path, task_id)
    log_ref = write_blob(child, "READY\nfull service log\n", kind="txt")
    tool_trace = persist_call(
        child,
        task_id=task_id,
        call_id="tool_service_logs",
        call_type="tool_call",
        payload={
            "tool": "service_logs",
            "result": json.dumps({"tail": "READY", "full_log_ref": log_ref}),
        },
    )
    write_task_result(
        child,
        task_id,
        STATUS_COMPLETED,
        result="done",
        artifact_status="ready",
        trace_refs={"tool_call_refs": [{"manifest_ref": tool_trace["manifest_ref"]}]},
        verification_ledger={
            "entries": [{
                "kind": "runtime_event",
                "services": [{"log_finalization": {"full_log_ref": log_ref}}],
            }],
        },
    )

    copied = copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})
    copied = retry_child_task_refs(parent, child, task_id)

    assert copied is not None
    evidence_ref = copied["verification_ledger"]["entries"][0]["services"][0][
        "log_finalization"
    ]["full_log_ref"]
    assert read_blob_ref(parent, evidence_ref, expected_kind="txt") == "READY\nfull service log\n"
    tool_manifest = _manifest(copied["trace_refs"]["tool_call_refs"][0]["manifest_ref"])
    tool_payload = read_blob_ref(parent, tool_manifest["full_payload_ref"])
    nested_ref = json.loads(tool_payload["result"])["full_log_ref"]
    assert read_blob_ref(parent, nested_ref, expected_kind="txt") == "READY\nfull service log\n"

    prune_headless_task_drives(parent, retention_days=0, now=_future_now(), live=lambda _task: False)
    assert not child.exists()
    assert read_blob_ref(parent, evidence_ref, expected_kind="txt").endswith("service log\n")
    assert read_blob_ref(parent, nested_ref, expected_kind="txt").startswith("READY")


def test_copyback_source_handle_promotion_survives_a_lost_write_race(tmp_path, monkeypatch):
    """The Windows interleaving of the concurrent copy-back (CI 33579445704): two
    copiers both miss the destination handle, the winner's identical bytes land,
    and the loser's own os.replace over that destination is refused as a sharing
    violation (CPython opens files without FILE_SHARE_DELETE). The promotion's
    postcondition is the VERIFIED handle at the destination, not authorship of the
    write, so the loser must publish the same complete custody projection — not an
    incomplete promotion with a pending ref."""
    from ouroboros import artifacts as artifacts_module
    from ouroboros.artifacts import store_actor_source_bytes

    task_id = "phase3c-lost-write-race"
    parent, child = _child(tmp_path, task_id)
    source_ref = store_actor_source_bytes(
        child, task_id, category="tool_results", source_id="tool",
        data=b"actor promised source", extension="txt",
    )
    write_task_result(
        child, task_id, STATUS_COMPLETED, result="done", artifact_status="ready",
        review_evidence={"exact_source_ref": source_ref},
    )
    real_store = artifacts_module.store_actor_source_bytes

    def _losing_store(*args, **kwargs):
        real_store(*args, **kwargs)  # the concurrent winner's identical copy lands
        raise PermissionError("[WinError 32] the file is in use by another process")

    monkeypatch.setattr(artifacts_module, "store_actor_source_bytes", _losing_store)

    copied = copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})
    copied = retry_child_task_refs(parent, child, task_id)

    promotion = copied["child_ref_promotion"]
    assert promotion["status"] == "complete"
    assert promotion["promoted_source_handle_count"] == 1
    assert promotion["pending_refs"] == []
    assert copied["review_evidence"]["exact_source_ref"] == source_ref
    promoted = (
        parent / "task_results" / "artifacts" / task_id / source_ref["path"]
    )
    assert promoted.read_bytes() == b"actor promised source"
