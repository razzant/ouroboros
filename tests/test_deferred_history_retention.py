"""Completion and recovery do not wait for history; custody still prevents loss."""

import json

import pytest

from ouroboros import headless, observability
from ouroboros.history_retention import retention_summary
from ouroboros.task_custody import settle_child_drive
from ouroboros.task_results import load_task_result, write_task_result


@pytest.mark.parametrize("layout", ["direct", "headless"])
def test_completion_adopts_answer_without_history_and_keeps_source(tmp_path, monkeypatch, layout):
    parent = tmp_path / "parent"
    child = parent / (headless.TASK_DRIVES_DIR if layout == "direct" else headless.HEADLESS_TASKS_DIR) / "finished"
    if layout == "headless":
        child /= "data"
    child.mkdir(parents=True)
    ref = observability.persist_call(child, task_id="finished", call_id="call", call_type="llm_response",
                                     payload={"answer": "exact"})["manifest_ref"]
    write_task_result(parent, "finished", "running", result="")
    write_task_result(child, "finished", "completed", result="The answer", artifact_status="ready",
                      trace_refs={"llm_call_refs": [{"response_ref": ref}]})
    task = {"id": "finished", "drive_root": str(child)}
    with monkeypatch.context() as blocked:
        blocked.setattr(observability, "promote_child_task_refs",
                        lambda *a, **k: pytest.fail("history walked on completion/startup"))
        result = headless.copy_child_task_result(parent, task)
        assert result["result"] == "The answer"
        assert headless.terminal_task_files_ready(parent, task, result)
        assert not headless.prepare_terminal_task_files(parent, task)["error"]
    assert retention_summary(result)["status"] == "pending"
    assert retention_summary(result)["problem_count"] == 0
    assert result["trace_refs"]["llm_call_refs"][0]["response_ref"] == ref
    assert settle_child_drive(parent, "finished", child, live=lambda _: False)["reason"] == "child_refs_pending"
    assert child.exists()
    assert any(json.loads(line)["type"] == "history_retention"
               for line in (parent / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines())


def test_adoption_copies_output_files_before_marking_answer_ready(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    child = parent / "task_drives" / "outputs"
    output = child / "task_results" / "artifacts" / "outputs" / "answer.txt"
    output.parent.mkdir(parents=True)
    output.write_text("delivered bytes", encoding="utf-8")
    write_task_result(child, "outputs", "completed", result="see file", artifact_status="ready",
                      artifacts=[{"path": str(output), "name": "answer.txt", "kind": "file"}])
    monkeypatch.setattr(observability, "promote_child_task_refs",
                        lambda *a, **k: pytest.fail("history walked while saving outputs"))
    copied = headless.copy_child_task_result(parent, {"id": "outputs", "drive_root": str(child)})
    assert copied["status"] == "completed"
    from pathlib import Path
    saved = Path(copied["artifacts"][0]["path"])
    assert saved.is_relative_to(parent / "task_results" / "artifacts" / "outputs")
    assert saved.read_text(encoding="utf-8") == "delivered bytes"


def test_retention_problem_is_distinct_from_normal_background_work():
    normal = {"schema_version": 1, "status": "incomplete", "pending_refs": [
        {"kind": "history_retention_deferred", "path": "/retained"}], "unavailable_refs": []}
    assert retention_summary({"child_ref_promotion": normal})["status"] == "pending"
    failed = {**normal, "unavailable_refs": [{"reason": "disk full"}]}
    assert retention_summary({"child_ref_promotion": failed})["status"] == "problem"
    assert retention_summary({"child_ref_promotion": {**normal, "status": "complete", "pending_refs": []}})["status"] == "complete"


def test_startup_does_not_retry_history_of_already_adopted_answer(tmp_path, monkeypatch):
    from ouroboros.server_maintenance import _recover_terminal_task_files

    parent = tmp_path / "parent"
    child = parent / "task_drives" / "recovered"
    child.mkdir(parents=True)
    write_task_result(child, "recovered", "completed", result="saved", artifact_status="ready")
    monkeypatch.setattr(observability, "promote_child_task_refs",
                        lambda *a, **k: pytest.fail("history walked on restart"))
    from ouroboros.startup_migrations import prepare_startup_state
    prepare_startup_state(parent)
    first = _recover_terminal_task_files(parent, set())
    assert first["recovered"] == ["recovered"]
    assert not first["errors"]
    assert load_task_result(parent, "recovered")["result"] == "saved"
    second = _recover_terminal_task_files(parent, set())
    assert not second["errors"]
    assert load_task_result(parent, "recovered")["child_ref_promotion"]["status"] == "incomplete"


@pytest.mark.parametrize("prior", ["cancelled", "old_complete", "late_call"])
def test_gc_owes_unreferenced_physical_calls_including_old_and_cancelled_rows(tmp_path, prior):
    parent = tmp_path / "parent"
    child = parent / "task_drives" / "history"
    child.mkdir(parents=True)
    write_task_result(parent, "history", "cancelled" if prior == "cancelled" else "completed",
                      result="saved", artifact_status="ready", headless_child_drive_root=str(child),
                      child_ref_promotion={"schema_version": 1, "status": "complete", "pending_refs": []})
    if prior == "late_call":
        write_task_result(child, "history", "completed", result="saved", artifact_status="ready")
        headless.copy_child_task_result(parent, {"id": "history", "drive_root": str(child)})
        headless.retry_child_task_refs(parent, child, "history")
    # Real physical call omitted from the task-result references, e.g. compaction/postwork.
    trace = observability.persist_call(child, task_id="history", call_id="unlisted",
                                       call_type="context_compaction_map", payload={"source": "exact"})
    outcome = settle_child_drive(parent, "history", child, live=lambda _: False)
    assert outcome["status"] == "retained" and outcome["reason"] == "call_inventory_pending"
    pending = load_task_result(parent, "history")
    assert pending["child_ref_promotion"]["status"] == "incomplete"
    headless.retry_child_task_refs(parent, child, "history")
    assert settle_child_drive(parent, "history", child, live=lambda _: False)["status"] == "removed"
    assert not child.exists()
    manifest = observability.read_call_manifest_ref(parent, trace["manifest_ref"], task_id="history")
    assert observability.read_blob_ref(parent, manifest["full_payload_ref"]) == {"source": "exact"}


def test_public_exact_review_reader_works_while_history_is_deferred(tmp_path):
    from pathlib import Path
    from ouroboros.artifacts import store_actor_source_bytes, task_artifact_dir_path
    from ouroboros.task_finalization import review_source_reader
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    parent, task_id = tmp_path / "parent", "source-author"
    child = headless.prepare_task_drive(parent, task_id, "empty")
    raw = json.dumps({"authority": "host_root", "request": {
        "surface": "task_acceptance", "task_id": task_id, "subject": "exact original"}}).encode("utf-8")
    ref = store_actor_source_bytes(child, task_id, category="context_checkpoints", source_id="acceptance",
                                   data=raw, extension="json")
    write_task_result(child, task_id, "completed", result="saved", artifact_status="ready",
                      review_projection={"panels": [{"surface": "task_acceptance", "authority": "host_root",
                                                      "applied_source_ref": ref}]})
    headless.copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})
    canonical = task_artifact_dir_path(parent, task_id, create=False) / ref["path"]
    assert not canonical.exists()
    registry = ToolRegistry(repo_dir=Path.cwd(), drive_root=parent)
    registry.set_context(ToolContext(repo_dir=Path.cwd(), drive_root=parent, task_id="next-owner",
                                    task_metadata={"budget_drive_root": str(parent)}))
    write_task_result(parent, "next-owner", "running", root_task_id="next-owner")
    selector = review_source_reader(task_id, ref)
    result = registry.execute_result(selector["tool"], selector["arguments"])
    assert result.status == "ok"
    before = json.loads(result.text)["review_source"]
    assert before.get("status") != "unavailable" and before["source_ref"]["sha256"] == ref["sha256"]
    assert not canonical.exists(), "a read must not perform deferred history placement"
    headless.retry_child_task_refs(parent, child, task_id)
    assert settle_child_drive(parent, task_id, child, live=lambda _: False)["status"] == "removed"
    after = registry.execute_result(selector["tool"], selector["arguments"])
    assert json.loads(after.text)["review_source"] == before


def test_gc_rechecks_call_inventory_after_preparation(tmp_path, monkeypatch):
    from ouroboros import task_custody

    parent, task_id = tmp_path / "parent", "late-inventory"
    child = headless.prepare_task_drive(parent, task_id, "empty")
    write_task_result(child, task_id, "completed", result="saved", artifact_status="ready")
    observability.persist_call(child, task_id=task_id, call_id="first", call_type="tool_call", payload={"data": "first"})
    headless.copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})
    headless.retry_child_task_refs(parent, child, task_id)
    original = task_custody._child_store_plan
    late = []
    def write_during_preparation(*args, **kwargs):
        plan = original(*args, **kwargs)
        late.append(observability.persist_call(child, task_id=task_id, call_id="late", call_type="tool_call",
                                              payload={"data": "late physical result"})["manifest_ref"])
        return plan
    with monkeypatch.context() as writing:
        writing.setattr(task_custody, "_child_store_plan", write_during_preparation)
        outcome = settle_child_drive(parent, task_id, child, live=lambda _: False)
    assert outcome["status"] == "retained" and outcome["reason"] == "call_inventory_changed"
    assert child.exists()
    assert settle_child_drive(parent, task_id, child, live=lambda _: False)["reason"] == "call_inventory_pending"
    headless.retry_child_task_refs(parent, child, task_id)
    assert settle_child_drive(parent, task_id, child, live=lambda _: False)["status"] == "removed"
    stored = observability.read_call_manifest_ref(parent, late[0], task_id=task_id)
    assert observability.read_blob_ref(parent, stored["full_payload_ref"]) == {"data": "late physical result"}
