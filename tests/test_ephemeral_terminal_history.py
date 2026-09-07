"""An ephemeral turn retains its terminal facts in chat without a task record."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from ouroboros import agent_task_pipeline as pipeline
from ouroboros.cost_projection import carry_cost_meta
from ouroboros.gateway.history import make_chat_history_endpoint
from supervisor import message_bus, state
from supervisor.events import _handle_send_message


@pytest.mark.parametrize("stream", ["chat", "progress"])
def test_unstamped_ephemeral_history_does_not_invent_execution_facts(tmp_path, stream):
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
    assert row["ephemeral_decision"] is True
    assert "outcome_axes" not in row
    assert "reason_code" not in row
    assert carry_cost_meta(row) == {}


@pytest.mark.parametrize("execution,reason", [
    ("ok", "final_message"), ("degraded", "tool_failure"),
    ("infra_failed", "provider_unavailable"), ("failed", "owner_requested_finalization"),
])
@pytest.mark.parametrize("cost", [
    {"accounted_upper_bound_usd": 0.75, "cost_final": True, "cost_accounting_status": "available"},
    {"accounted_upper_bound_usd": 0.75, "cost_final": False, "unresolved_upper_bound_usd": 0.5,
     "cost_accounting_status": "available"},
    {"accounted_upper_bound_usd": None, "cost_final": False, "unknown_unmetered": 1,
     "cost_accounting_status": "available"},
    {"accounted_upper_bound_usd": None, "cost_final": False, "cost_accounting_status": "unavailable",
     "cost_accounting_error": "ledger_unavailable"},
])
def test_ephemeral_terminal_facts_survive_real_chat_persistence_and_history(
    tmp_path, monkeypatch, execution, reason, cost,
):
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(state, "reconstruct_task_cost", lambda *_a, **_k: dict(cost))
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 1, "session_id": "test"})
    bridge = message_bus.LocalChatBridge({})
    live = []
    bridge._broadcast_fn = live.append
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    pending = []
    task = {"id": "ephemeral-history", "type": "task", "chat_id": 1, "text": "Inspect the task",
            "_is_direct_chat": True, "_ephemeral_turn": True}
    text = "The original answer stays intact."
    pipeline.emit_task_results(
        SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path), None, None,
        pending, task, text, {"execution_status": execution, "reason_code": reason},
        {"tool_calls": [], "reasoning_notes": [],
         "delivery_candidate": {"degraded": execution == "degraded", "degraded_reason": reason}},
        start_time=0.0, drive_logs=logs,
    )
    final = next(row for row in pending if row["type"] == "send_message")
    terminal = next(row for row in pending if row["type"] == "task_done")
    assert terminal["outcome_axes"]["execution"]["status"] == execution
    assert not (tmp_path / "task_results" / "ephemeral-history.json").exists()
    context = SimpleNamespace(DRIVE_ROOT=tmp_path, send_with_budget=message_bus.send_with_budget,
                              append_jsonl=lambda *_a, **_k: None)
    _handle_send_message(final, context)
    stored = json.loads((logs / "chat.jsonl").read_text())
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"chat_id": "1"})))
    [replayed] = json.loads(response.body)["messages"]
    [frame] = [row for row in live if row.get("type") == "chat"]
    for row in (final["progress_meta"], stored, frame, replayed):
        assert row["ephemeral_decision"] is True
        assert row["task_terminal_status"] == "completed"
        assert row["outcome_axes"] == terminal["outcome_axes"]
        assert row["reason_code"] == reason
        assert carry_cost_meta(row) == carry_cost_meta(cost)
        assert "cancelable" not in row and "task_id_pending" not in row
    assert stored["text"] == replayed["text"] == frame["content"] == text
    assert not (tmp_path / "task_results" / "ephemeral-history.json").exists()
