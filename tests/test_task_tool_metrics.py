"""Exact generic call evidence survives metrics, the free facts row and history."""

from tests import _dialogue as dialogue_test
import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_store
from ouroboros.agent_task_pipeline import emit_task_results
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.post_task_synthesis import _record_task_facts, build_trace_summary, task_tool_metrics
from ouroboros.task_results import load_task_result
from ouroboros.utils import append_jsonl
from supervisor.events_worker_reports import _handle_task_metrics


def _history(root):
    response = asyncio.run(make_chat_history_endpoint(root)(
        SimpleNamespace(query_params={"chat_id": "1"})))
    return json.loads(response.body)["messages"]


@pytest.mark.parametrize("names, failed, expected", [
    (["start_task"], [], {"start_task": 1}),
    (["presence_finish", "read_file", "finish_task"], [2], {"presence_finish": 1, "read_file": 1, "finish_task": 1}),
    ([], [], {}),
])
def test_actual_metrics_summary_and_history_keep_complete_tool_census(tmp_path, monkeypatch, names, failed, expected):
    calls = [{"tool": name, "tool_call_id": f"call-{index}", "args": {},
              "result": "Recorded result", "is_error": index in failed,
              "status": "error" if index in failed else "ok",
              "completion_control": name in {"finish_task", "presence_finish"}}
             for index, name in enumerate(names)]
    trace = {"tool_calls": calls, "reasoning_notes": []}
    task = {"id": "native-metrics", "chat_id": 1, "type": "task", "text": "Owner objective",
            "_is_direct_chat": True, "_skip_post_task_synthesis": True}
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    # The facts row buys no model call. All writers, metric forwarding,
    # stored result and history projection execute normally.
    model_calls = []
    monkeypatch.setattr("ouroboros.llm_observability.chat_observed",
                        lambda *args, **kwargs: model_calls.append(kwargs))
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
    _record_task_facts(env, task, {**usage, "outcome_axes": metric["outcome_axes"],
                                   "reason_code": metric["reason_code"]}, trace, logs)
    facts = next(json.loads(line) for line in dialogue_test.log_text(logs / "chat.jsonl").splitlines()
                 if json.loads(line).get("summary_kind") == "host_task_facts")
    replay = next(row for row in _history(tmp_path) if row["system_type"] == "task_summary")
    addressing = sum(1 for name in names if name.strip() in ("start_task", "move_to_project", "tell_task"))
    for row in (metric, evaluation, forwarded[0], facts, replay):
        assert row["tool_calls"] == len(names)
        assert row["tool_errors"] == len(failed)
        assert row["routing_tool_calls"] == addressing
        assert row["completion_tool_calls"] == sum(call["completion_control"] and not call["is_error"] for call in calls)
        assert row["tool_call_counts"] == expected
        assert row["outcome_axes"]["execution"] == metric["outcome_axes"]["execution"]
    assert model_calls == [] and facts["text"] == ""
    stored_after = load_task_result(tmp_path, task["id"])
    assert stored_after["result"] == stored_before["result"] == "Done."
    for key in ("status", "outcome_axes", "accounted_upper_bound_usd", "cost_final"):
        assert stored_after[key] == stored_before[key]
    assert replay["text"] == facts["text"]


@pytest.mark.parametrize("trace, expected_total, expected_errors", [
    ({"tool_calls": [], "loop_evidence_unavailable": True}, None, None),
    ({"tool_calls": [], "recovered_post_task_synthesis": True}, 0, 0),
    ({"tool_calls": [{"tool": "start_task"}, {}]}, 2, 0),
])
def test_incomplete_trace_never_claims_empty_or_partial_census(trace, expected_total, expected_errors):
    # Keep the existing aggregate semantics; no partial dictionary can hide an
    # unrecognized row by classifying its missing name as addressing/non-work.
    metrics = task_tool_metrics(trace)
    expected_routing = None if expected_total is None else sum(
        1 for call in trace.get("tool_calls") or [] if isinstance(call, dict) and call.get("tool") == "start_task")
    assert metrics == {"tool_calls": expected_total, "tool_errors": expected_errors,
                       "routing_tool_calls": expected_routing, "completion_tool_calls": None if expected_total is None else 0, "tool_call_counts": None}


def test_unknown_and_legacy_evidence_keep_absence_distinct_from_zero(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.llm_observability.chat_observed",
                        lambda *a, **k: pytest.fail("unavailable trace buys no summary call"))
    task = {"id": "unknown", "chat_id": 1, "text": "Original objective"}
    _record_task_facts(SimpleNamespace(drive_root=tmp_path), task,
                       {"loop_evidence_unavailable": True},
                       {"loop_evidence_unavailable": True, "tool_calls": []}, tmp_path / "logs")
    [unknown] = _history(tmp_path)
    assert all(unknown[key] is None for key in ("tool_calls", "tool_errors", "routing_tool_calls", "completion_tool_calls", "tool_call_counts"))
    dialogue_store.append(tmp_path, "chat", {"type": "task_summary", "task_id": "legacy",
        "direction": "system", "chat_id": 1, "text": "Legacy summary", "tool_calls": 1, "rounds": 2})
    legacy = next(row for row in _history(tmp_path) if row["task_id"] == "legacy")
    assert all(key not in legacy for key in ("tool_call_counts", "tool_errors", "routing_tool_calls", "completion_tool_calls"))
    wire = []
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, PENDING=[], append_jsonl=append_jsonl,
                          bridge=SimpleNamespace(push_log=wire.append))
    _handle_task_metrics({"task_id": "legacy", "tool_calls": 1}, ctx)
    assert all(key not in wire[0] for key in ("tool_call_counts", "routing_tool_calls", "completion_tool_calls"))
    _handle_task_metrics({"task_id": "unknown", **task_tool_metrics({"loop_evidence_unavailable": True})}, ctx)
    assert all(wire[1][key] is None for key in ("tool_calls", "tool_errors", "routing_tool_calls", "completion_tool_calls", "tool_call_counts"))


def test_completion_metrics_require_the_host_fact_and_success():
    calls = [{"tool": "finish_task"}, {"tool": "task_acceptance_review", "completion_control": True},
             {"tool": "finish_task", "completion_control": True, "is_error": True}]
    metrics = task_tool_metrics({"tool_calls": calls})
    assert (metrics["tool_calls"], metrics["tool_errors"], metrics["completion_tool_calls"]) == (3, 1, 1)


def test_host_refused_step_is_no_tool_error_but_an_independent_failure_stays_one():
    """The envelope types a host refusal as is_error; the host's own narrow mark — status
    "blocked" beside tool_result_meta.refused_step, the same predicate the durable-row
    replay reads (tool_call_log.is_plain_step_refusal) — says a further STEP was refused, not that
    the call failed. A failure beside a refusal keeps its own recorded status and counts."""
    refused = {"refused_step": True, "control_reason": "cancelled"}
    calls = [
        # a clean Stop: the extension's own action succeeded, the host refused the next one
        {"tool": "ext_stopper_send", "is_error": True, "status": "blocked",
         "tool_result_code": "OWNER_OPERATION_REFUSED", "tool_result_meta": refused},
        # independent failures the host ALSO refused a further step: real errors
        {"tool": "ext_stopper_send", "is_error": True, "status": "error", "tool_result_code": "EXTENSION_ERROR",
         "tool_result_meta": refused},
        {"tool": "run_command", "is_error": True, "status": "timeout",
         "tool_result_meta": {**refused, "control_reason": "task_cancelled"}},
        # an ordinary failure and a guard's plain blocked (no host mark) stay errors
        {"tool": "read_file", "is_error": True, "status": "error"},
        {"tool": "write_file", "is_error": True, "status": "blocked",
         "tool_result_code": "ACCESS_BLOCKED", "tool_result_meta": {}},
        {"tool": "edit_text", "is_error": False, "status": "ok"},
    ]
    metrics = task_tool_metrics({"tool_calls": calls})
    assert metrics["tool_calls"] == 6
    assert metrics["tool_errors"] == 4
    lines = build_trace_summary({"tool_calls": calls}).splitlines()
    assert lines[2].endswith("[status=error] → ERROR → FURTHER STEP REFUSED")
    assert lines[3].endswith("[status=timeout] → ERROR → FURTHER STEP REFUSED")
    assert lines[4].endswith("[status=error] → ERROR")


def test_a_clean_stop_reads_policy_denied_in_the_trace_summary():
    """The trace side needs no separate compact refusal fact: the classifier already buckets
    the refused "blocked" completion as a recorded policy denial, never an unresolved error."""
    trace = {"tool_calls": [{"tool": "ext_stopper_send", "is_error": True, "status": "blocked",
                             "tool_result_meta": {"refused_step": True, "control_reason": "cancelled"}}]}
    assert task_tool_metrics(trace)["tool_errors"] == 0
    summary = build_trace_summary(trace).splitlines()
    assert summary[1].endswith("[status=blocked] → FURTHER STEP REFUSED")
    assert summary[0] == "## Tool trace (1 calls, 0 errors, 1 policy-denied)"
