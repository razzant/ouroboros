"""The origin label of a consciousness wake-up survives every hop (P1).

``metadata.initiator = "consciousness"`` rides the turn's live frames (progress,
heartbeat, terminal), the stored chat row, the authored summary row and the
persisted result, and history replay returns it on each — so the block's meta
line and the final bubble are labelled live and after a reload alike.
"""

from __future__ import annotations

from tests import _dialogue as dialogue_test

import asyncio
import json
import queue
from types import SimpleNamespace

TS = "2026-09-16T12:00:00Z"
WAKE_META = {
    "initiator": "consciousness", "usage_category": "consciousness", "wake_reason": "heartbeat",
    "consciousness_autonomy": "act", "model_role": "consciousness",
}


# --- the origin label: live frames, stored chat rows, replay ----------------------


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
    from ouroboros import dialogue_store
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
    dialogue_store.append(tmp_path, "progress", {
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
    rows = dialogue_test.log_rows(tmp_path / "logs" / "chat.jsonl")
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


# --- where a place's work came from and why: the same carriers ------------------------


FOCUS_META = {
    "initiator": "consciousness", "wake_cause": "focus", "focus_allowance": {"status": "available"},
    "focus": {"place_id": "pl-1", "matched_wakes": ["my_thread"],
              "actor": {"platform_actor_id": "731", "display_name": "qa-contributor", "owner_confirmed": False, "is_bot": False}},
    "presence": {"place": {"label": "Ouroboros"}, "event": {"provider": "github"}},
}


def test_place_origin_is_the_admissions_focus_by_value_on_the_record_or_its_metadata():
    import copy
    from ouroboros.subagent_messages import initiator_meta

    expected = {"place_id": "pl-1", "place_label": "Ouroboros", "provider": "github",
                "actor": {"id": "731", "label": "qa-contributor"}, "wakes": ["my_thread"]}
    assert initiator_meta({"metadata": FOCUS_META})["place_origin"] == expected
    assert initiator_meta(FOCUS_META)["place_origin"] == expected  # the metadata itself, as frame producers pass it
    owner = copy.deepcopy(FOCUS_META)
    owner["focus"]["actor"].update(owner_confirmed=True, is_bot=True)
    assert initiator_meta(owner)["place_origin"]["actor"] == {"id": "731", "label": "qa-contributor", "owner": True, "bot": True}
    unnamed = copy.deepcopy(FOCUS_META)
    unnamed["focus"].update(actor={"platform_actor_id": "", "display_name": ""}, matched_wakes=[], routed=True)
    expected_routed = {"place_id": "pl-1", "place_label": "Ouroboros", "provider": "github", "wakes": [], "routed": True}
    assert initiator_meta(unnamed)["place_origin"] == expected_routed  # several works read its conversation
    del unnamed["focus"]["routed"]
    unnamed["focus_allowance"]["route"] = True  # a focus minted before ``focus.routed``: its own allowance says it
    assert initiator_meta(unnamed)["place_origin"] == expected_routed
    started = {"initiator": "consciousness", "consciousness_cause": "started", "focus": FOCUS_META["focus"]}
    assert initiator_meta(started)["place_origin"] == {"place_id": "pl-1", "actor": {"id": "731", "label": "qa-contributor"},
                                                       "wakes": ["my_thread"]}, "a root it started, carrier not copied"
    assert "place_origin" not in initiator_meta({"metadata": {"initiator": "consciousness"}})
    assert "place_origin" not in initiator_meta({"metadata": {"focus": {"matched_wakes": ["mention"]}}})


def test_place_origin_rides_the_summary_row_the_progress_row_and_terminal_truth(tmp_path):
    from ouroboros.gateway.history import (_PROGRESS_META_FIELDS, _annotate_terminal_task_truth,
                                           _copy_task_summary_metadata)
    from ouroboros.subagent_messages import place_origin

    origin = place_origin(FOCUS_META)
    assert "place_origin" in _PROGRESS_META_FIELDS
    rec: dict = {}
    _copy_task_summary_metadata(rec, {"type": "task_summary", "place_origin": origin})
    assert rec["place_origin"] == origin
    summary = {"task_id": "f", "role": "system", "system_type": "task_summary", "ts": TS}
    _annotate_terminal_task_truth([summary], tmp_path, {"f": {"status": "completed", "metadata": FOCUS_META}})
    assert summary["place_origin"] == origin and summary["initiator"] == "consciousness"
