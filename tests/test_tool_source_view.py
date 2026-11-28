"""Complete tool sources precede the fitting view, including parallel results."""

import json

from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.context_fit import project_tool_result_batch


def _fit(messages, tools):
    size = len(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False))
    return {"accepted": size <= 2200, "serialized_chars": size, "strict_bound_proven": False}


def test_escaped_parallel_sources_survive_before_bounded_projection(tmp_path):
    originals = ['α\n"\t' * 2000, 'Independent reviewer finding. ' * 2000]
    rows = [{"tool_call_id": str(i), "fn_name": name, "result": body,
             "result_meta": {"knowledge_source_complete": True}}
            for i, (name, body) in enumerate(zip(["knowledge_read", "commit_reviewed"], originals))]
    projected, receipt = project_tool_result_batch(rows, [], [], drive_root=tmp_path, task_id="task",
                                                   fit_candidate=_fit)
    assert receipt["status"] == "projected"
    assert [r["tool_call_id"] for r in projected] == ["0", "1"]
    assert [r["result"] for r in rows] == originals
    for row, original in zip(projected, originals):
        assert read_actor_source_bytes(tmp_path, "task", row["result_source_ref"]).decode() == original
        start, end = row["result_source_view"]["delivered_range"]
        assert start == 0 and row["result"].startswith(original[:end])
        assert row["result_partial"] and not row["result_meta"]["knowledge_source_complete"]
    assert receipt["fit"]["serialized_chars"] <= 2200


def test_fitting_full_batch_is_byte_identical_and_needs_no_source_write(tmp_path):
    rows = [{"tool_call_id": "one", "result": "Complete small result"}]
    projected, receipt = project_tool_result_batch(rows, [], [], drive_root=tmp_path, task_id="task",
                                                   fit_candidate=_fit)
    assert projected == rows and receipt["status"] == "complete"
    assert list(tmp_path.iterdir()) == []


def test_unfit_minimum_keeps_all_sources_and_does_not_claim_a_fitting_send(tmp_path):
    rows = [{"tool_call_id": str(i), "result": "content" * 1000} for i in range(3)]
    projected, receipt = project_tool_result_batch(rows, [], [], drive_root=tmp_path, task_id="task",
                                                   fit_candidate=lambda m,t: {"accepted": False})
    assert receipt["status"] == "minimum_view_unfit"
    assert len(projected) == 3
    for row, original in zip(projected, rows):
        assert read_actor_source_bytes(tmp_path, "task", row["result_source_ref"]).decode() == original["result"]
