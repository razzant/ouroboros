"""Exact generic call evidence survives metrics, authored summary and history."""
import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from ouroboros.agent_task_pipeline import emit_task_results
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.post_task_synthesis import _run_task_summary, task_tool_metrics
from ouroboros.task_results import load_task_result
from ouroboros.utils import append_jsonl
from supervisor.events_worker_reports import _handle_task_metrics


def _history(root):
    response = asyncio.run(make_chat_history_endpoint(root)(
        SimpleNamespace(query_params={"chat_id": "1"})))
    return json.loads(response.body)["messages"]


@pytest.mark.parametrize("names, failed, expected", [
    (["promote_chat_to_task"], [], {"promote_chat_to_task": 1}),
    (["read_file", "promote_chat_to_task"], [], {"read_file": 1, "promote_chat_to_task": 1}),
    (["steer_task", "steer_task", "route_to_project"], [1], {"steer_task": 2, "route_to_project": 1}),
    ([" read_file ", "new_extension_tool"], [], {"read_file": 1, "new_extension_tool": 1}),
    ([], [], {}),
])
def test_actual_metrics_summary_and_history_keep_complete_tool_census(tmp_path, monkeypatch, names, failed, expected):
    calls = [{"tool": name, "tool_call_id": f"call-{index}", "args": {},
              "result": "Recorded result", "is_error": index in failed,
              "status": "error" if index in failed else "ok"}
             for index, name in enumerate(names)]
    trace = {"tool_calls": calls, "reasoning_notes": []}
    task = {"id": "native-metrics", "chat_id": 1, "type": "task", "text": "Owner objective",
            "_is_direct_chat": True, "_skip_post_task_synthesis": True}
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    # Only the authored summary's model response is substituted. All writers,
    # metric forwarding, stored result and history projection execute normally.
    model_calls = []

    def summary_response(*args, **kwargs):
        model_calls.append(kwargs)
        return {"content": "Exact authored summary."}, {}

    monkeypatch.setattr("ouroboros.llm_observability.chat_observed", summary_response)
    usage = {"rounds": 2 if names else 0}
    pending = []
    emit_task_results(env, None, None, pending, task, "Done.", usage, trace, time.time(), logs)
    metric = next(row for row in pending if row["type"] == "task_metrics")
    evaluation = next(json.loads(line) for line in (logs / "events.jsonl").read_text().splitlines()
                      if json.loads(line).get("type") == "task_eval")
    forwarded = []
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, PENDING=[], append_jsonl=append_jsonl,
                          bridge=SimpleNamespace(push_log=forwarded.append))
    _handle_task_metrics(metric, ctx)
    stored_before = load_task_result(tmp_path, task["id"])
    _run_task_summary(env, None, task, {**usage, "outcome_axes": metric["outcome_axes"],
                                      "reason_code": metric["reason_code"]}, trace, logs)
    authored = next(json.loads(line) for line in (logs / "chat.jsonl").read_text().splitlines()
                    if json.loads(line).get("summary_kind") == "authored_root_summary")
    replay = next(row for row in _history(tmp_path) if row["system_type"] == "task_summary")
    addressing = sum(1 for name in names if name.strip() in ("promote_chat_to_task", "route_to_project", "steer_task"))
    for row in (metric, evaluation, forwarded[0], authored, replay):
        assert row["tool_calls"] == len(names)
        assert row["tool_errors"] == len(failed)
        assert row["routing_tool_calls"] == addressing
        assert row["tool_call_counts"] == expected
        assert row["outcome_axes"]["execution"] == metric["outcome_axes"]["execution"]
    assert len(model_calls) == bool(names)
    if names:
        assert "Owner objective" in model_calls[0]["messages"][0]["content"]
        assert authored["text"] == "Exact authored summary."
    stored_after = load_task_result(tmp_path, task["id"])
    assert stored_after["result"] == stored_before["result"] == "Done."
    for key in ("status", "outcome_axes", "accounted_upper_bound_usd", "cost_final"):
        assert stored_after[key] == stored_before[key]
    assert replay["text"] == authored["text"]


@pytest.mark.parametrize("trace, expected_total, expected_errors", [
    ({"tool_calls": [], "loop_evidence_unavailable": True}, None, None),
    ({"tool_calls": [{"tool": "steer_task", "is_error": True}], "loop_evidence_unavailable": True}, None, None),
    ({"tool_calls": [], "recovered_post_task_synthesis": True}, 0, 0),
    ({}, 0, 0),
    ({"tool_calls": [{"tool": "promote_chat_to_task"}, {}]}, 2, 0),
    ({"tool_calls": [{"tool": "promote_chat_to_task"}, {"tool": " "}]}, 2, 0),
    ({"tool_calls": [{"tool": "promote_chat_to_task"}, {"tool": None, "is_error": True}]}, 2, 1),
    ({"tool_calls": [{"tool": "promote_chat_to_task"}, {"tool": 12}]}, 2, 0),
    ({"tool_calls": [{"tool": "promote_chat_to_task"}, "malformed"]}, 2, 0),
])
def test_incomplete_trace_never_claims_empty_or_partial_census(trace, expected_total, expected_errors):
    # Keep the existing aggregate semantics; no partial dictionary can hide an
    # unrecognized row by classifying its missing name as addressing/non-work.
    metrics = task_tool_metrics(trace)
    expected_routing = None if expected_total is None else sum(
        1 for call in trace.get("tool_calls") or [] if isinstance(call, dict) and call.get("tool") == "promote_chat_to_task")
    assert metrics == {"tool_calls": expected_total, "tool_errors": expected_errors,
                       "routing_tool_calls": expected_routing, "tool_call_counts": None}


def test_unknown_and_legacy_evidence_keep_absence_distinct_from_zero(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.llm_observability.chat_observed",
                        lambda *a, **k: pytest.fail("unavailable trace buys no summary call"))
    task = {"id": "unknown", "chat_id": 1, "text": "Original objective"}
    _run_task_summary(SimpleNamespace(drive_root=tmp_path), None, task,
                      {"loop_evidence_unavailable": True},
                      {"loop_evidence_unavailable": True, "tool_calls": []}, tmp_path / "logs")
    [unknown] = _history(tmp_path)
    assert all(unknown[key] is None for key in ("tool_calls", "tool_errors", "routing_tool_calls", "tool_call_counts"))
    append_jsonl(tmp_path / "logs/chat.jsonl", {"type": "task_summary", "task_id": "legacy",
        "direction": "system", "chat_id": 1, "text": "Legacy summary", "tool_calls": 1, "rounds": 2})
    legacy = next(row for row in _history(tmp_path) if row["task_id"] == "legacy")
    assert "tool_call_counts" not in legacy and "tool_errors" not in legacy and "routing_tool_calls" not in legacy
    wire = []
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, PENDING=[], append_jsonl=append_jsonl,
                          bridge=SimpleNamespace(push_log=wire.append))
    _handle_task_metrics({"task_id": "legacy", "tool_calls": 1}, ctx)
    assert "tool_call_counts" not in wire[0] and "routing_tool_calls" not in wire[0]
    _handle_task_metrics({"task_id": "unknown", **task_tool_metrics({"loop_evidence_unavailable": True})}, ctx)
    assert all(wire[1][key] is None for key in ("tool_calls", "tool_errors", "routing_tool_calls", "tool_call_counts"))
