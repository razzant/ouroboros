"""Terminal file adoption releases only the child drive's rebuildable cache."""

import pytest

from ouroboros import headless, observability
from ouroboros.task_results import write_task_result
from tests._store_shared import result_signature


def inventory(root):
    path = root / "state" / "code_intel" / "repo" / "inventory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"cache": "derived"}', encoding="utf-8")
    return path


@pytest.mark.parametrize("status", ["completed", "failed", "rejected_duplicate"])
def test_terminal_adoption_drops_cache_before_child_drive_retention(tmp_path, status):
    parent, task_id = tmp_path / "canonical", "terminal-cache"
    child = headless.prepare_task_drive(parent, task_id, "empty")
    child_cache, canonical_cache = inventory(child), inventory(parent)
    write_task_result(child, task_id, status, result="retained answer", artifact_status="ready")
    original = result_signature(child, task_id)
    call = observability.persist_call(child, task_id=task_id, call_id="evidence", call_type="llm_response",
                                      payload={"answer": "retained answer"})
    task = {"id": task_id, "drive_root": str(child)}
    for _ in range(2):
        report = headless.prepare_terminal_task_files(parent, task)
        assert report["error"] == "" and report["result"]["status"] == status
        assert not child_cache.parent.parent.exists()
        assert child.is_dir() and result_signature(child, task_id) == original
        assert canonical_cache.read_text(encoding="utf-8") == '{"cache": "derived"}'
        manifest = observability.read_call_manifest_ref(child, call["manifest_ref"], task_id=task_id)
        assert observability.read_blob_ref(child, manifest["full_payload_ref"])["answer"] == "retained answer"


@pytest.mark.parametrize("status", [None, "running", "waiting_for_owner"])
def test_unsettled_child_keeps_its_inventory(tmp_path, status):
    parent, task_id = tmp_path / "canonical", "active-cache"
    child = headless.prepare_task_drive(parent, task_id, "empty")
    cache = inventory(child)
    write_task_result(parent, task_id, "completed", result="early canonical checkpoint")
    if status:
        write_task_result(child, task_id, status, result="still working")
    report = headless.prepare_terminal_task_files(parent, {"id": task_id, "drive_root": str(child)})
    assert report["error"] and report["terminal_source_present"] is False
    assert cache.exists()


@pytest.mark.parametrize("alias", [False, True])
def test_terminal_preparation_keeps_canonical_inventory(tmp_path, alias):
    root, task_id = tmp_path / "canonical", "same-root-cache"
    cache = inventory(root)
    drive = root
    if alias:
        drive = tmp_path / "alias"
        try:
            drive.symlink_to(root, target_is_directory=True)
        except OSError:
            pytest.skip("directory symlinks unavailable")
    write_task_result(root, task_id, "completed", result="done", artifact_status="ready")
    report = headless.prepare_terminal_task_files(root, {"id": task_id, "drive_root": str(drive)})
    assert not report["error"] and cache.is_file()
