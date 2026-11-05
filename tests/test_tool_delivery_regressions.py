"""A source address and a chosen view survive every batch delivery policy."""


from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.tool_result_delivery import project_tool_result_batch


def test_unknown_window_keeps_an_explicit_head_tail_request(tmp_path):
    original = "HEAD " + "middle " * 5000 + " TAIL"
    rows, receipt = project_tool_result_batch(
        [{"tool_call_id": "run", "result": original, "fn_name": "run_command",
          "tool_args": {"view_head_chars": 20, "view_tail_chars": 20}}], [], [],
        drive_root=tmp_path, task_id="t", policy="measured_frame",
        fit_candidate=lambda messages, tools: {"accepted": True, "capacity_total_tokens": None})
    assert receipt["status"] == "capacity_unknown"
    assert rows[0]["result_partial"]
    assert rows[0]["result_source_view"]["shown_ranges"] == [[0, 20], [len(original) - 20, len(original)]]
    assert read_actor_source_bytes(tmp_path, "t", rows[0]["result_source_ref"]).decode() == original
