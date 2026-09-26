"""Native task facts rows preserve unknown counters in history."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.cost_projection import carry_cost_meta


def test_unknown_task_facts_counts_remain_readable_in_history(tmp_path, monkeypatch):
    from ouroboros.post_task_synthesis import _record_task_facts

    monkeypatch.setattr("ouroboros.llm_observability.chat_observed",
                        lambda *_a, **_k: pytest.fail("unknown evidence must not buy a model call"))
    _record_task_facts(SimpleNamespace(drive_root=tmp_path),
                       {"id": "uncaptured-summary", "chat_id": 1, "text": "Inspect current work"},
                       {"loop_evidence_unavailable": True},
                       {"loop_evidence_unavailable": True, "tool_calls": []}, tmp_path / "logs")
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"chat_id": "1"})))
    [row] = json.loads(response.body)["messages"]
    assert row["text"] == ""  # unknown stays a typed null below, never prose
    assert row["tool_calls"] is None and row["rounds"] is None


@pytest.mark.parametrize("stream", ["chat", "progress"])
def test_legacy_transient_history_does_not_invent_execution_facts(tmp_path, stream):
    """Retained historical rows remain readable after their producer is removed."""
    logs = tmp_path / "logs"
    logs.mkdir()
    entry = {
        "ts": "2026-09-06T00:00:00Z", "chat_id": 1, "direction": "out",
        "task_id": "old-transient", "ephemeral_decision": True,
        "text": "Original transient text", "content": "Original transient text",
    }
    (logs / f"{stream}.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"chat_id": "1"})))
    [row] = json.loads(response.body)["messages"]
    assert row["text"] == entry["text"]
    assert "outcome_axes" not in row
    assert "reason_code" not in row
    assert carry_cost_meta(row) == {}


@pytest.mark.parametrize("captured", [False, True])
def test_direct_exception_counts_keep_unknown_separate_from_recorded_zero(tmp_path, monkeypatch, captured):
    from ouroboros.agent import _task_exception_terminal
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros.task_results import load_task_result
    from supervisor import state

    logs = tmp_path / "logs"
    logs.mkdir()
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    task = {"id": "direct-exception", "type": "task", "chat_id": 1,
            "text": "Inspect current work", "_is_direct_chat": True, "_skip_post_task_synthesis": True}
    error = RuntimeError("context preparation failed")
    if captured:
        error._ouroboros_loop_usage = {"rounds": 0}
        error._ouroboros_loop_trace = {"tool_calls": [], "reasoning_notes": []}
    text, usage, trace = _task_exception_terminal(env, task, error, logs)
    monkeypatch.setattr(state, "reconstruct_task_cost", lambda *_a, **_k: {
        "accounted_upper_bound_usd": None, "cost_final": False, "cost_accounting_status": "unavailable"})
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async",
                        lambda *_a, **_k: pytest.fail("stopped post-task phase must not run"))
    pending = []
    pipeline.emit_task_results(env, None, None, pending, task, text, usage, trace,
                               start_time=0.0, drive_logs=logs)
    metrics = next(row for row in pending if row["type"] == "task_metrics")
    evaluation = next(json.loads(line) for line in (logs / "events.jsonl").read_text().splitlines()
                      if json.loads(line).get("type") == "task_eval")
    for row in (metrics, evaluation):
        assert row["tool_calls"] == (0 if captured else None)
        assert row["outcome_axes"]["execution"]["status"] == "infra_failed"
        assert row["reason_code"] == "task_exception"
    stored = load_task_result(tmp_path, task["id"])
    assert stored["status"] == "failed" and stored["result"] == text
    assert stored["loop_outcome"]["usage"]["total_rounds"] == (0 if captured else None)
