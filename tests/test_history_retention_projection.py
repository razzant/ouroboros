"""History placement is public detail, independent of task result and lifecycle."""
import copy

import pytest

from ouroboros.outcomes import public_task_result


@pytest.mark.parametrize(("pending", "unavailable", "status", "expected"), [
    ([{"kind": "history_retention_deferred"}], [], "incomplete", "pending"),
    ([{"kind": "context_checkpoint"}], [], "incomplete", "problem"),
    ([], [{"kind": "call_manifest"}], "complete", "problem"),
])
def test_public_history_summary_preserves_result_and_durable_source(pending, unavailable, status, expected):
    stored = {"task_id": "history", "status": "completed", "result": "The answer remains available.",
              "child_ref_promotion": {"schema_version": 1, "status": status,
                                      "pending_refs": pending, "unavailable_refs": unavailable}}
    before = copy.deepcopy(stored)
    public = public_task_result(stored)
    assert public["history_retention"]["status"] == expected
    assert public["status"] == stored["status"] and public["result"] == stored["result"]
    assert public["outcome_axes"]["lifecycle"]["status"] == "completed"
    assert stored == before
    assert public_task_result(public)["history_retention"] == public["history_retention"]


def test_old_task_without_retention_has_no_invented_status():
    assert "history_retention" not in public_task_result({"status": "completed"})


def test_problem_reasons_are_grouped_without_losing_raw_failure_facts():
    from ouroboros.history_retention import retention_diagnostics

    pending = [{"kind": "history_retention_deferred", "reason": "background_history_retention"},
               {"kind": "task_source", "path": "source-a", "reason": "OSError: disk full"}]
    unavailable = [{"kind": "blob", "path": path, "sha256": path, "reason": "source_missing"}
                   for path in ("source-b", "source-c")]
    stored = {"child_ref_promotion": {"schema_version": 1, "status": "incomplete",
                                      "pending_refs": pending, "unavailable_refs": unavailable}}
    public = public_task_result(stored)
    assert public["history_retention"]["problem_reasons"] == [
        {"reason": "OSError: disk full", "count": 1}, {"reason": "source_missing", "count": 2}]
    assert retention_diagnostics(stored) == {"pending_refs": pending[1:], "unavailable_refs": unavailable}
    assert public["child_ref_promotion"] == stored["child_ref_promotion"]


def test_cold_history_uses_current_retention_and_clears_stale_problem(tmp_path, monkeypatch):
    from ouroboros.gateway.history import _annotate_terminal_task_truth, _copy_task_summary_metadata
    from ouroboros.terminal_projection import _project_row

    stored = {"status": "completed", "child_ref_promotion": {"schema_version": 1, "status": "incomplete",
              "pending_refs": [{"kind": "context_checkpoint"}], "unavailable_refs": []}}
    row = _project_row("history", stored, {}, {"chat_id": 1})
    assert row["history_retention"]["status"] == "problem"
    message = {"task_id": "history", "system_type": "task_summary", "role": "system"}
    _copy_task_summary_metadata(message, row)
    assert message["history_retention"]["status"] == "problem"
    # The history endpoint already owns this current result. No new load or scan is needed.
    monkeypatch.setattr("ouroboros.task_status.load_effective_task_result",
                        lambda *a, **kw: pytest.fail("retention must use the existing cached result"))
    _annotate_terminal_task_truth([message], tmp_path, {"history": stored})
    assert message["history_retention"]["status"] == "problem"
    stored["child_ref_promotion"].update(status="complete", pending_refs=[])
    _annotate_terminal_task_truth([message], tmp_path, {"history": stored})
    assert message["history_retention"]["status"] == "complete"
    assert row["history_retention"]["status"] == "problem", "historical snapshot is not rewritten"
