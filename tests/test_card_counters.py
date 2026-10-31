"""Card rounds are accepted loop replies, independent of monetary physical attempts."""
import asyncio
import json
import queue
from types import SimpleNamespace

import pytest

from ouroboros import agent_task_pipeline as pipeline
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.loop_llm_call import call_llm_with_retry
from ouroboros.utils import append_jsonl
from supervisor.events_worker_reports import _handle_log_event, _handle_task_metrics
from supervisor.log_addressing import TurnEventQueue
from tests.test_task_correlation_facts import _Clock, _TimedLLM


@pytest.mark.parametrize("direct", [False, True])
def test_round_producer_transport_storage_and_cold_history(tmp_path, monkeypatch, direct):
    from ouroboros import loop_llm_call

    logs = tmp_path / "logs"
    logs.mkdir()
    clock, usage, queue_sink, pushed = _Clock(), {"rounds": 2}, queue.Queue(), []
    monkeypatch.setattr(loop_llm_call, "time", clock)
    events = TurnEventQueue(queue_sink, "counter-task", 7, direct_chat=direct, telemetry_only=not direct)
    # Two physical replies, one usable response. The loop's index is deliberately different.
    llm = _TimedLLM(clock, [({"content": "", "tool_calls": []}, {}),
                           ({"content": "Checked the source.", "tool_calls": []}, {})])
    message, _ = call_llm_with_retry(llm, [{"role": "user", "content": "check"}], "fixture/model",
                                     None, "medium", 2, logs, "counter-task", 18, events, usage)
    assert message["content"] == "Checked the source." and usage["rounds"] == 3
    raw = next(evt for evt in queue_sink.queue if evt.get("data", {}).get("type") == "llm_round_finished")
    # JSON serialization and the real supervisor handler preserve the addressed fact even after departure.
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, bridge=SimpleNamespace(push_log=pushed.append),
                          append_jsonl=lambda *_: None)
    _handle_log_event(json.loads(json.dumps(raw)), ctx)
    assert pushed[-1]["rounds"] == 3 and pushed[-1]["chat_id"] == 7
    assert pushed[-1]["round"] == 18
    assert bool(pushed[-1].get("_is_direct_chat")) is direct
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_task_cost_fields", lambda *a: {"total_rounds": 99})
    task = {"id": "counter-task", "type": "task", "chat_id": 7, "text": "check",
            "_is_direct_chat": direct, "_skip_post_task_synthesis": True}
    pending = []
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    pipeline.emit_task_results(env, None, None, pending, task, message["content"], usage,
                               {"tool_calls": [], "reasoning_notes": []}, 0.0, logs)
    metrics = next(evt for evt in pending if evt["type"] == "task_metrics")
    _handle_task_metrics(json.loads(json.dumps(metrics)), ctx)
    assert pushed[-1]["rounds"] == 3 and pushed[-1]["tool_calls"] == 0
    stored = pipeline.load_task_result(tmp_path, "counter-task")
    assert stored["total_rounds"] == 99
    assert stored["loop_outcome"]["usage"]["total_rounds"] == 3
    # History uses real stored results and the producer's saved task facts; no hand-built usage path.
    request = SimpleNamespace(query_params={"limit": "20", "chat_id": "7"})
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(request))
    rows = json.loads(response.body)["messages"]
    summaries = [row for row in rows if row.get("system_type") == "task_summary"]
    assert summaries and all(row["rounds"] == 3 for row in summaries)


@pytest.mark.parametrize("saved_usage", [{}, {"total_rounds": None}, {"total_rounds": 0}, {"total_rounds": 3}],
                         ids=["missing", "unknown", "zero", "accepted"])
@pytest.mark.parametrize("retained", [False, True])
def test_history_rounds_override_stale_physical_summary_only_when_known(tmp_path, saved_usage, retained):
    from ouroboros.project_dialogue import append_canonical_task_summary
    from tests._store_shared import read_result, seed_result

    seed_result(tmp_path, "old", {"status": "completed", "total_rounds": 99,
        "loop_outcome": {"usage": {**saved_usage, "extra": "U" * 100_000}, "extra": "L" * 100_000}})
    full = read_result(tmp_path, "old")
    append_canonical_task_summary(tmp_path, retained=True, row={
        "task_id": "old", "chat_id": 7, "direction": "system", "type": "task_summary",
        "summary_id": "task-facts:old", "text": "", "tool_calls": 1, "ts": "2026-10-01T00:00:00Z",
        **({"rounds": 99} if retained else {}),
    })
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"chat_id": "7"})))
    assert response.status_code == 200
    rows = [row for row in json.loads(response.body)["messages"] if row.get("system_type") == "task_summary"]
    assert rows and all(row["outcome_final"] for row in rows)
    expected = saved_usage.get("total_rounds")
    for row in rows:
        if expected is not None:
            assert row["rounds"] == expected
        elif retained:
            assert row["rounds"] == 99, "unknown canonical counts do not erase retained facts"
        else:
            assert "rounds" not in row, "physical task.total_rounds is never a fallback"
    assert read_result(tmp_path, "old") == full


@pytest.mark.parametrize("source, expected", [("no_response", 0), ("unavailable", None)])
def test_failed_loop_rounds_match_live_saved_facts_and_history(tmp_path, monkeypatch, source, expected):
    from ouroboros import loop_llm_call
    from ouroboros.agent import _task_exception_terminal
    from ouroboros.loop import run_llm_loop
    from ouroboros.tools.registry import ToolRegistry
    from tests import _dialogue

    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    task = {"id": "failed-counter", "type": "task", "chat_id": 7, "text": "check",
            "_is_direct_chat": True, "_skip_post_task_synthesis": True}
    logs = tmp_path / "logs"
    logs.mkdir()
    if source == "no_response":
        monkeypatch.setattr(loop_llm_call, "time", _Clock())

        class EmptyLLM:
            calls = 0

            def default_model(self):
                return "fixture/model"

            def chat(self, **_kwargs):
                self.calls += 1
                return {"content": "", "tool_calls": []}, {"response_finish_reason": "stop"}

        llm = EmptyLLM()
        text, usage, trace = run_llm_loop(
            messages=[{"role": "user", "content": "check"}],
            tools=ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), llm=llm,
            drive_logs=logs, emit_progress=lambda *_a, **_k: None, incoming_messages=queue.Queue(),
            task_id=task["id"], drive_root=tmp_path,
        )
        assert llm.calls > 0 and "rounds" not in usage
        assert usage["execution_status"] == "infra_failed"
    else:
        text, usage, trace = _task_exception_terminal(env, task, RuntimeError("no retained loop snapshot"), logs)
        assert usage["loop_evidence_unavailable"] and trace["loop_evidence_unavailable"]

    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    pending, pushed = [], []
    pipeline.emit_task_results(env, None, None, pending, task, text, usage, trace, 0.0, logs)
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, bridge=SimpleNamespace(push_log=pushed.append),
                          append_jsonl=append_jsonl)
    _handle_task_metrics(json.loads(json.dumps(next(evt for evt in pending if evt["type"] == "task_metrics"))), ctx)
    stored = pipeline.load_task_result(tmp_path, task["id"])
    facts = next(row for row in _dialogue.chat_rows(tmp_path) if row.get("summary_kind") == "host_task_facts")
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"chat_id": "7"})))
    assert response.status_code == 200
    summaries = [row for row in json.loads(response.body)["messages"] if row.get("system_type") == "task_summary"]
    assert summaries and stored["status"] == "failed"
    assert {"live": pushed[-1]["rounds"], "saved": stored["loop_outcome"]["usage"]["total_rounds"],
            "facts": facts["rounds"], "history": summaries[0].get("rounds")} == dict.fromkeys(
                ("live", "saved", "facts", "history"), expected)


@pytest.mark.parametrize("lane", ["ordinary", "split", "split-project"])
@pytest.mark.parametrize("skip_post", [False, True], ids=["post", "skip-post"])
@pytest.mark.parametrize("usage, expected", [
    ({"rounds": 3}, 3), ({"rounds": 0}, 0), ({}, 0),
    ({"rounds": 17, "loop_evidence_unavailable": True}, None),
], ids=["accepted", "zero", "no-response", "unknown"])
def test_emitted_rounds_survive_split_dispatch_and_result_pruning(
    tmp_path, monkeypatch, lane, skip_post, usage, expected,
):
    from ouroboros import llm, post_task_evolution
    from tests import _dialogue
    from tests._store_shared import delete_result

    canonical = tmp_path / "canonical"
    actor = canonical if lane == "ordinary" else tmp_path / "actor"
    for root in {canonical, actor}:
        (root / "logs").mkdir(parents=True)
    task = {"id": "root-counter", "root_task_id": "root-counter", "type": "task",
            "chat_id": 7, "text": "Check the result", "drive_root": str(actor),
            "budget_drive_root": str(canonical), "_is_direct_chat": lane == "ordinary",
            "_skip_post_task_synthesis": skip_post}
    if lane == "split-project":
        task["project_id"] = "counter-project"
    trace = {"tool_calls": [], "reasoning_notes": [],
             "loop_evidence_unavailable": bool(usage.get("loop_evidence_unavailable"))}
    post_roots = []
    # Keep real dispatch, snapshots, checkpoint and facts writes. Only paid work is replaced.
    monkeypatch.setattr(llm, "LLMClient", SimpleNamespace)
    monkeypatch.setattr(pipeline, "_run_memory_fallback_draft",
                        lambda env, *_args: post_roots.append(env.drive_root))
    monkeypatch.setattr(post_task_evolution, "maybe_promote", lambda *_args: None)
    monkeypatch.setattr(pipeline, "_task_cost_fields", lambda *_args: {"total_rounds": 99})
    events = []
    pipeline.emit_task_results(
        SimpleNamespace(drive_root=actor, repo_dir=tmp_path), None, None, events, task,
        "Checked the result.", dict(usage), trace, 0.0, actor / "logs",
    )
    assert post_roots == ([] if skip_post else [canonical if lane == "split" else actor])
    metrics = next(event for event in events if event["type"] == "task_metrics")
    assert metrics["total_rounds"] == 99, "ledger attempts differ from accepted loop replies"
    stored = pipeline.load_task_result(actor, task["id"])
    if actor != canonical:
        assert "loop_outcome" not in (pipeline.load_task_result(canonical, task["id"]) or {}), (
            "the facts must survive before the actor result is copied back")
    [facts] = [row for row in _dialogue.chat_rows(canonical)
               if row.get("summary_kind") == "host_task_facts" and row.get("task_id") == task["id"]]

    # Remove both result copies, then use the real cold-history consumer of the retained facts.
    for root in {canonical, actor}:
        delete_result(root, task["id"])
        assert pipeline.load_task_result(root, task["id"]) is None
    response = asyncio.run(make_chat_history_endpoint(canonical)(
        SimpleNamespace(query_params={"chat_id": "7"})))
    assert response.status_code == 200
    summaries = [row for row in json.loads(response.body)["messages"]
                 if row.get("system_type") == "task_summary" and row.get("task_id") == task["id"]
                 and "rounds" in row]
    assert summaries, "pruned results still have retained counter facts in history"
    assert {"live": metrics["rounds"], "saved": stored["loop_outcome"]["usage"]["total_rounds"],
            "facts": facts["rounds"], "history": summaries[0]["rounds"]} == dict.fromkeys(
                ("live", "saved", "facts", "history"), expected)
    assert all(row["rounds"] == expected for row in summaries)


@pytest.mark.parametrize("snapshot, expected", [
    ({}, None), ({"loop_outcome": {"usage": {"total_rounds": None}}}, None),
    ({"loop_outcome": {"usage": {"total_rounds": 0}}}, 0),
    ({"loop_outcome": {"usage": {"total_rounds": 3}}}, 3),
], ids=["missing", "unknown", "zero", "accepted"])
def test_recovery_facts_use_retained_loop_rounds(tmp_path, monkeypatch, snapshot, expected):
    from tests import _dialogue

    pipeline.write_task_result(tmp_path, "recovered", pipeline.STATUS_COMPLETED, chat_id=7,
        root_task_id="recovered", objective="finish recovery", total_rounds=99, **snapshot,
        root_phase_checkpoint={"phase": "task_acceptance", "status": "pass", "post_task_synthesis": "pending_once"})

    def free_facts(env, task, usage, trace, logs, **_kwargs):
        pipeline._record_task_facts(env, task, usage, trace, logs)
        pipeline._set_root_post_task_checkpoint(env, task, "completed")

    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", free_facts)
    assert pipeline.recover_pending_root_post_task_synthesis(tmp_path, tmp_path) == 1
    facts = next(row for row in _dialogue.chat_rows(tmp_path) if row.get("summary_kind") == "host_task_facts")
    assert facts["rounds"] == expected
    assert pipeline.recover_pending_root_post_task_synthesis(tmp_path, tmp_path) == 0
