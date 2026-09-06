"""Existing observation records carry their actual task, call and start facts."""

import json
import queue
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from ouroboros.agent import OuroborosAgent
from ouroboros.agent_task_pipeline import _store_task_result
from ouroboros.task_results import load_task_result, write_task_result


def test_known_start_survives_finalization_and_legacy_ts_is_not_relabelled(tmp_path):
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    task = {"id": "known", "type": "task", "text": "work", "queued_at": 1700000000.0}
    write_task_result(tmp_path, "known", "scheduled", ts="2020-01-01T00:00:00+00:00")
    host = SimpleNamespace(env=env, _task_started_ts=1700000005.0)
    OuroborosAgent._persist_running_record(host, task)
    started = load_task_result(tmp_path, "known")["started_at"]
    assert started == datetime.fromtimestamp(host._task_started_ts, timezone.utc).isoformat()
    OuroborosAgent._persist_running_record(host, task)  # preflight amendment is the same start
    _store_task_result(env, task, "done", {"rounds": 1, "cost": 0}, {"tool_calls": []})
    stored = load_task_result(tmp_path, "known")
    assert stored["started_at"] == started and stored["queued_at"] == task["queued_at"]
    assert stored["ts"] != started
    write_task_result(tmp_path, "legacy", "running", ts="2020-01-01T00:00:00+00:00")
    _store_task_result(env, {"id": "legacy", "type": "task"}, "done", {}, {"tool_calls": []})
    assert "started_at" not in load_task_result(tmp_path, "legacy")


def test_real_llm_round_and_usage_join_across_worker_and_supervisor_logs(tmp_path):
    from ouroboros.loop_llm_call import call_llm_with_retry
    from supervisor.events_budget import _handle_llm_usage

    worker, canonical = tmp_path / "worker", tmp_path / "canonical"
    logs = worker / "logs"
    logs.mkdir(parents=True)
    events = queue.Queue()
    class LLM:
        def chat(self, **kwargs):
            return ({"content": "done", "tool_calls": [], "finish_reason": "stop"},
                    {"provider": "openai-compatible", "resolved_model": "local-model", "cost": 0.0,
                     "prompt_tokens": 12, "completion_tokens": 3})
    message, cost = call_llm_with_retry(LLM(), [{"role": "user", "content": "do work"}],
                                       "openai-compatible::local-model", None, "medium", 1,
                                       logs, "call-task", 7, events, {})
    assert message["content"] == "done" and cost == 0
    queued = list(events.queue)
    usage_event = next(row for row in queued if row.get("type") == "llm_usage")
    ctx = SimpleNamespace(DRIVE_ROOT=canonical, RUNNING={}, update_budget_from_usage=lambda usage: None,
                          bridge=SimpleNamespace(push_log=lambda event: None))
    _handle_llm_usage(usage_event, ctx)
    local = [json.loads(line) for line in (logs / "events.jsonl").read_text().splitlines()]
    global_rows = [json.loads(line) for line in (canonical / "logs" / "events.jsonl").read_text().splitlines()]
    round_row = next(row for row in local if row["type"] == "llm_round")
    saved = next(row for row in global_rows if row["type"] == "llm_usage")
    for key in ("llm_call_id", "execution_id", "round_id", "round"):
        assert saved[key] == round_row[key]
    assert saved["round"] == 7
    assert saved["cost"] == 0 and saved["cost_known"] is True
    assert not any(row["type"] == "llm_round" for row in global_rows)


def test_intrinsic_checkpoint_discloses_the_same_binding_ceiling_as_text(monkeypatch):
    from ouroboros import task_pacing as pacing

    monkeypatch.setattr(pacing, "get_pacing_interval_sec", lambda: 1)
    ceiling = pacing.CostCeiling(state="active", ceiling_usd=80, root_cap_usd=100, basis="root_cap")
    now = datetime.now(timezone.utc)
    note = pacing.build_intrinsic_pacing_note(SimpleNamespace(_cost_ceiling=ceiling),
        created=now - timedelta(seconds=60), now=now, round_idx=4, accumulated_usage={"cost": 10},
        tree_cost_provider=lambda: {"accounted_usd": 20, "root_limit_usd": 100})
    assert note.checkpoint["cost_ceiling"] == pacing.cost_ceiling_disclosure(ceiling)
    assert note.checkpoint["tree_cap_usd"] == 100
    assert "$80.00 in-task cost ceiling" in note.text


def test_terminal_delegation_rows_keep_existing_root_and_parent(tmp_path, monkeypatch):
    from ouroboros import delegate_custody as custody

    monkeypatch.setattr(custody, "_CUSTODY", {})
    row = custody.RunCustody(run_id="correlated", task_id="child", root_task_id="root", parent_task_id="parent",
                             route_id="route", ledger_root=str(tmp_path))
    custody.record_started(tmp_path, row)
    custody.settle_run(tmp_path, SimpleNamespace(), row,
                       {"summary": {"state": "succeeded", "model": "test-model", "spendUsd": 0, "spendEstimated": False}})
    custody.record_patch_disposed(tmp_path, row, disposition="rejected")
    row.output_artifact, row.output_complete = "recorded-output.txt", True
    assert custody.record_settled_unread(tmp_path, row)
    records = [json.loads(line) for line in custody.event_log_path(tmp_path).read_text().splitlines()]
    wanted = {custody.LEDGER_RECORDED, custody.SETTLED, custody.PATCH_DISPOSED, custody.SETTLED_UNREAD}
    found = [record for record in records if record["type"] in wanted]
    assert {record["type"] for record in found} == wanted
    assert all(record["root_task_id"] == "root" and record["parent_task_id"] == "parent" for record in found)


def test_tool_error_manifest_has_typed_code_and_redacted_reason(tmp_path):
    from ouroboros.loop_tool_execution import _execute_single_tool
    from ouroboros.tools.tool_result import ToolResult

    tools = SimpleNamespace(CODE_TOOLS=set(), _ctx=SimpleNamespace(task_metadata={}),
                            execute_result=lambda *_a: ToolResult(status="error", code="EXECUTOR_ERROR", text="Database initialization failed."))
    logs = tmp_path / "logs"
    logs.mkdir()
    result = _execute_single_tool(tools, {"id": "tc1", "function": {"name": "probe", "arguments": "{}"}}, logs, "task")
    manifest = json.loads(Path(result["trace_ref"]["manifest_ref"]["path"]).read_text())
    assert manifest["tool_code"] == "EXECUTOR_ERROR"
    assert manifest["error_preview"] == "Database initialization failed."
    assert manifest["semantic_ok"] is False
