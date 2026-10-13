"""Registered root sources preserve exact bytes, custody and retry read authority."""
from pathlib import Path

import pytest

from ouroboros import artifacts, headless, source_retention, task_custody
from ouroboros.task_results import write_task_result
from ouroboros.tools.registry import ToolRegistry


def test_retry_read_file_resolves_registered_internal_wait_source(tmp_path):
    owner, successor = "prior-wait", "retry-wait"
    ref = artifacts.store_actor_source_bytes(tmp_path, owner, category="tool_results", source_id="wait-handoff",
                                            data=b'{"result":"LAST_PRIVATE_FACT"}', extension="json", register=True)
    write_task_result(tmp_path, owner, "completed", result="old", retry_task_id=successor)
    write_task_result(tmp_path, successor, "running", timeout_retry_from=owner, original_task_id=owner)
    registry = ToolRegistry(repo_dir=Path(__file__).resolve().parents[1], drive_root=tmp_path)
    registry._ctx.task_id = successor
    assert "LAST_PRIVATE_FACT" in registry.execute("read_file", ref["read"]["arguments"])
    path = artifacts.task_artifact_dir_path(tmp_path, owner) / ref["path"]
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError):
        source_retention.retained_actor_source_read_target(tmp_path, successor, ref["path"])


@pytest.mark.parametrize("fault", ["ordinary_artifact", "unregistered", "foreign", "broken_retry"])
def test_root_source_alias_does_not_widen_retention_authority(tmp_path, fault):
    owner, successor = "prior-wait", "retry-wait"
    raw = b"not an implicit file grant"
    ref = artifacts.store_actor_source_bytes(tmp_path, owner, category="tool_results", source_id="wait-handoff",
                                            data=raw, extension="json", register=True)
    write_task_result(tmp_path, owner, "completed", result="old", retry_task_id=successor if fault != "broken_retry" else "other")
    write_task_result(tmp_path, successor, "running", timeout_retry_from=owner if fault != "foreign" else None,
                      original_task_id=owner)
    if fault == "unregistered":
        (artifacts.task_artifact_dir_path(tmp_path, owner) / ".artifact_manifest.json").unlink()
    elif fault == "ordinary_artifact":
        from ouroboros.utils import atomic_write_json, read_json_dict
        manifest = artifacts.task_artifact_dir_path(tmp_path, owner) / ".artifact_manifest.json"
        document = read_json_dict(manifest)
        document["artifacts"][ref["path"]]["kind"] = "user_file"
        atomic_write_json(manifest, document)
    assert source_retention.retained_actor_source_read_target(tmp_path, successor, ref["path"]) is None


def test_registered_root_source_captured_in_result_survives_history_retention(tmp_path):
    parent, task = tmp_path / "canonical", "wait-capture"
    child = headless.prepare_task_drive(parent, task, "empty")
    full = b'{"result":"LAST_PRIVATE_FACT"}'
    ref = artifacts.store_actor_source_bytes(child, task, category="tool_results", source_id="wait-handoff",
                                            data=full, extension="json", register=True)
    write_task_result(child, task, "completed", result="done", artifact_status="ready",
                      review_evidence={"source_refs": [ref]})
    headless.copy_child_task_result(parent, {"id": task, "drive_root": str(child)})
    row = headless.retry_child_task_refs(parent, child, task)
    assert row["child_ref_promotion"]["status"] == "complete", row["child_ref_promotion"]
    assert artifacts.read_actor_source_bytes(parent, task, ref) == full
    assert task_custody.settle_child_drive(parent, task, child, live=lambda _: False)["status"] == "removed"
    assert artifacts.read_actor_source_bytes(parent, task, ref) == full
    assert not artifacts.collect_task_artifact_records(parent, task)
