"""The origin label of a consciousness wake-up survives every hop (P1).

``metadata.initiator = "consciousness"`` rides the turn's live frames (progress,
heartbeat, terminal), the chat.jsonl row, the authored summary row and the
persisted result, and history replay returns it on each — so the block's meta
line and the final bubble are labelled live and after a reload alike.
"""

from __future__ import annotations

import asyncio
import json
import queue
from types import SimpleNamespace

TS = "2026-09-16T12:00:00Z"
WAKE_META = {
    "initiator": "consciousness", "usage_category": "consciousness", "wake_reason": "heartbeat",
    "consciousness_autonomy": "act", "model_role": "consciousness",
}


# --- the origin label: live frames, chat.jsonl rows, replay ----------------------


def test_progress_heartbeat_and_final_frames_carry_the_initiator(monkeypatch, tmp_path):
    import ouroboros.agent_task_pipeline as pipeline
    from ouroboros import agent as agent_module

    agent = object.__new__(agent_module.OuroborosAgent)
    agent._current_task_metadata = dict(WAKE_META)
    agent._current_task_id, agent._current_chat_id = "w1", 1
    agent._last_progress_ts = 0.0
    agent.tools = SimpleNamespace(_ctx=None)
    events = queue.Queue()
    agent._event_queue = events
    assert agent._subagent_progress_meta("progress") == {"initiator": "consciousness"}
    agent._emit_progress("thinking about the day")
    assert events.get_nowait()["progress_meta"]["initiator"] == "consciousness"
    agent._emit_task_heartbeat("w1", "running")
    heartbeat = events.get_nowait()
    assert heartbeat["type"] == "task_heartbeat" and heartbeat["initiator"] == "consciousness"
    # An owner's turn carries no origin label. Its meta holds only the voice the
    # worker stamps on every frame, so the wake label cannot leak onto it.
    agent._current_task_metadata = {"client_message_id": "c-1"}
    assert agent._subagent_progress_meta("progress") == {}
    agent._emit_progress("plain")
    assert events.get_nowait()["progress_meta"] == {"narration": False}

    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    pending: list = []
    task = {"id": "w2", "type": "task", "chat_id": 1, "text": "wake", "_is_direct_chat": True, "metadata": dict(WAKE_META)}
    pipeline.emit_task_results(
        SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path), None, None,
        pending, task, "A thought for the owner.", {"rounds": 1},
        {"tool_calls": [], "reasoning_notes": []}, 0.0, tmp_path / "logs",
    )
    final = next(event for event in pending if event["type"] == "send_message")
    assert final["progress_meta"]["initiator"] == "consciousness"
    assert pipeline.load_task_result(tmp_path, "w2")["metadata"]["initiator"] == "consciousness"


def test_initiator_meta_reads_the_record_or_its_metadata():
    from ouroboros.subagent_messages import initiator_meta, subagent_message_meta

    assert initiator_meta({"metadata": {"initiator": "consciousness"}}) == {"initiator": "consciousness"}
    assert initiator_meta({"initiator": "consciousness"}) == {"initiator": "consciousness"}
    assert initiator_meta({"metadata": {"initiator": " "}}) == {} and initiator_meta(None) == {}
    # The subagent identity projection is unchanged: a wake is not a child.
    assert subagent_message_meta({"metadata": {"initiator": "consciousness"}}, task_id="w") == {}


def test_the_initiator_survives_the_chat_row_the_summary_row_and_replay(tmp_path):
    import ouroboros.agent_task_pipeline as pipeline
    from ouroboros.gateway.history import (
        _PROGRESS_META_FIELDS,
        _annotate_terminal_task_truth,
        _copy_task_summary_metadata,
        make_chat_history_endpoint,
    )
    from ouroboros.utils import append_jsonl
    from supervisor.message_bus import log_chat

    (tmp_path / "logs").mkdir(parents=True)
    (tmp_path / "state").mkdir(parents=True)
    labelled = log_chat("out", 1, 0, "A thought.", ts="2026-09-16T12:00:02Z", task_id="w1",
                        message_meta={"initiator": "consciousness", "task_terminal_status": "completed"},
                        drive_root=tmp_path, require_write=True)
    assert labelled["initiator"] == "consciousness" and labelled["task_terminal_status"] == "completed"
    plain = log_chat("out", 1, 0, "Hi.", ts="2026-09-16T12:00:03Z", task_id="o1",
                     message_meta={"task_terminal_status": "completed"}, drive_root=tmp_path, require_write=True)
    assert "initiator" not in plain
    append_jsonl(tmp_path / "logs" / "progress.jsonl", {
        "ts": "2026-09-16T12:00:01Z", "type": "send_message", "task_id": "w1", "is_progress": True,
        "direction": "out", "chat_id": 1, "user_id": 0, "text": "💬 thinking", "content": "💬 thinking",
        "format": "", "initiator": "consciousness",
    })
    pipeline._record_task_facts(
        env=None,
        task={"id": "w1", "type": "task", "text": "wake", "chat_id": 1, "_is_direct_chat": True, "metadata": dict(WAKE_META)},
        usage={"rounds": 1, "cost": 0.0}, llm_trace={"tool_calls": [], "reasoning_notes": []},
        drive_logs=tmp_path / "logs",
    )
    rows = [json.loads(line) for line in (tmp_path / "logs" / "chat.jsonl").read_text(encoding="utf-8").splitlines()]
    assert next(row for row in rows if row.get("type") == "task_summary")["initiator"] == "consciousness"
    assert "initiator" in _PROGRESS_META_FIELDS

    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"limit": "20"})))
    payload = json.loads(response.body.decode("utf-8"))["messages"]
    by_key = {(m.get("task_id"), bool(m.get("is_progress")), str(m.get("system_type") or "")): m for m in payload}
    assert by_key[("w1", False, "")]["initiator"] == "consciousness", "the final bubble keeps its label on reload"
    assert by_key[("w1", True, "")]["initiator"] == "consciousness", "the replayed progress row keeps the label"
    assert by_key[("w1", False, "task_summary")]["initiator"] == "consciousness", "the summary row keeps the label"
    assert "initiator" not in by_key[("o1", False, "")]

    # Terminal truth from the persisted result metadata, on the summary row or the lone progress row.
    progress = {"task_id": "t", "is_progress": True, "text": "working", "ts": TS}
    summary = {"task_id": "t", "role": "system", "system_type": "task_summary", "ts": "2026-09-16T12:00:05Z"}
    stored = {"t": {"status": "completed", "_is_direct_chat": True, "metadata": {"initiator": "consciousness"}}}
    _annotate_terminal_task_truth([progress, summary], tmp_path, stored)
    assert summary["initiator"] == "consciousness" and "initiator" not in progress
    alone = {"task_id": "t", "is_progress": True, "text": "working", "ts": TS}
    _annotate_terminal_task_truth([alone], tmp_path, stored)
    assert alone["initiator"] == "consciousness"
    managed = [{"task_id": "m", "is_progress": True, "text": "working", "ts": TS}]
    _annotate_terminal_task_truth(managed, tmp_path, {"m": {"status": "completed"}})
    assert "initiator" not in managed[0]
    rec: dict = {}
    _copy_task_summary_metadata(rec, {"type": "task_summary", "initiator": "consciousness"})
    assert rec["initiator"] == "consciousness"
    bare: dict = {}
    _copy_task_summary_metadata(bare, {"type": "task_summary"})
    assert "initiator" not in bare
