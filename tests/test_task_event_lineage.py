"""Telemetry is self-contained: lineage and chat ride the frame, stamped by the producer.

With the bus serving its control lane first, a task's ``task_done`` can be
handled before that task's earlier ``llm_usage`` and ``log_event`` frames, so
the supervisor's RUNNING row is gone when they arrive. The pooled worker now
creates one endpoint per task at task start (``log_addressing.task_event_queue``,
the direct lane's ``TurnEventQueue`` shape) and hands it down: every telemetry
frame carries the task's chat and lineage by value, a late callback keeps the
facts of the task it belongs to while a warm worker runs the next one, and the
supervisor's RUNNING fill is a fallback, never the source. Both directions of
each guard are pinned.
"""
from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from supervisor.log_addressing import (
    TASK_LINEAGE_FIELDS,
    stamp_recorded_row,
    task_event_queue,
    task_lineage,
)

LINEAGE = {
    "root_task_id": "root-1", "parent_task_id": "parent-1", "delegation_role": "subagent",
    "task_group_id": "group-1", "requested_model_lane": "auto", "effective_model_lane": "sonnet",
}


def _task(task_id="child-1", chat_id=7, **extra):
    return {"id": task_id, "type": "task", "chat_id": chat_id, "text": "work", **LINEAGE, **extra}


def _capture():
    captured = []
    return captured, SimpleNamespace(put=captured.append, put_nowait=captured.append)


def test_the_pooled_endpoint_stamps_lineage_and_chat_on_the_telemetry_lane_only():
    captured, inner = _capture()
    endpoint = task_event_queue(inner, _task())
    assert endpoint.task_id == "child-1" and task_lineage(_task()) == LINEAGE

    endpoint.put_nowait({"type": "llm_usage", "task_id": "child-1", "usage": {"prompt_tokens": 1}})
    endpoint.put({"type": "log_event", "data": {"type": "tool_call_started", "task_id": "child-1", "tool": "read_file"}})
    usage, frame = captured
    for key in TASK_LINEAGE_FIELDS:
        assert usage[key] == LINEAGE[key] and frame["data"][key] == LINEAGE[key]
    assert usage["chat_id"] == 7 and frame["data"]["chat_id"] == 7
    # A pooled task is not a direct turn: no lane fact, no Stop marker, no origin label.
    assert "_is_direct_chat" not in frame["data"] and "cancelable" not in frame["data"]

    # Explicit producer values win over the task's own facts.
    endpoint.put({"type": "llm_usage", "task_id": "child-1", "chat_id": 0, "effective_model_lane": "opus"})
    assert captured[-1]["chat_id"] == 0 and captured[-1]["effective_model_lane"] == "opus"
    assert captured[-1]["root_task_id"] == "root-1"

    # Control events keep their producer shape (they are handled while RUNNING exists).
    done = {"type": "task_done", "task_id": "child-1", "status": "completed"}
    assert endpoint.stamp(done) == {"type": "task_done", "task_id": "child-1", "status": "completed"}
    heartbeat = endpoint.stamp({"type": "task_heartbeat", "task_id": "child-1", "phase": "running"})
    assert "_carrier_rss" in heartbeat, "the worker's own size rides every task's beat (memory admission)"
    heartbeat.pop("_carrier_rss")
    assert heartbeat == {"type": "task_heartbeat", "task_id": "child-1", "phase": "running"}
    send = {"type": "send_message", "task_id": "child-1", "text": "answer"}
    assert "root_task_id" not in endpoint.stamp(send) and "chat_id" not in send

    # Another task's frame passes through untouched.
    other = {"type": "llm_usage", "task_id": "other", "usage": {}}
    assert endpoint.stamp(other) == {"type": "llm_usage", "task_id": "other", "usage": {}}


def test_absent_chat_stamps_nothing_while_the_hidden_partition_is_kept():
    captured, inner = _capture()
    task_event_queue(inner, _task(chat_id=None)).put({"type": "llm_usage", "task_id": "child-1"})
    assert "chat_id" not in captured[-1] and captured[-1]["root_task_id"] == "root-1"
    task_event_queue(inner, _task(chat_id=0)).put({"type": "llm_usage", "task_id": "child-1"})
    assert captured[-1]["chat_id"] == 0
    task_event_queue(inner, _task(chat_id="")).put({"type": "llm_usage", "task_id": "child-1"})
    assert "chat_id" not in captured[-1]


def test_a_direct_turn_resumed_on_a_pooled_worker_keeps_its_lane_fact_by_value():
    captured, inner = _capture()
    task_event_queue(inner, _task(_is_direct_chat=True)).put(
        {"type": "log_event", "data": {"type": "llm_round_finished", "task_id": "child-1"}})
    assert captured[-1]["data"]["_is_direct_chat"] is True
    assert "cancelable" not in captured[-1]["data"], "the Stop marker is the direct lane's own stamp"


def test_the_dispatch_resolution_seam_refreshes_the_effective_lane_by_value():
    captured, inner = _capture()
    endpoint = task_event_queue(inner, _task(effective_model_lane=""))
    endpoint.put({"type": "llm_usage", "task_id": "child-1"})
    assert "effective_model_lane" not in captured[-1]
    resolved = {"type": "task_dispatch_resolved", "task_id": "child-1",
                "resolution": {"effective_model_lane": "opus", "model": "anthropic/claude-opus-5.5"}}
    endpoint.put(resolved)
    assert captured[-1] is resolved and "chat_id" not in resolved, "the control event itself is not stamped"
    endpoint.put({"type": "llm_usage", "task_id": "child-1"})
    assert captured[-1]["effective_model_lane"] == "opus"


def test_a_warm_workers_next_task_does_not_relabel_the_previous_tasks_late_frames():
    captured, inner = _capture()
    first = task_event_queue(inner, _task("task-1", chat_id=1, root_task_id="root-1"))
    second = task_event_queue(inner, _task("task-2", chat_id=2, root_task_id="root-2", parent_task_id="parent-2"))
    # A callback that outlived task-1 still holds task-1's endpoint.
    first.put({"type": "llm_usage", "task_id": "task-1"})
    second.put({"type": "log_event", "data": {"type": "tool_call_finished", "task_id": "task-2"}})
    late, current = captured
    assert late["chat_id"] == 1 and late["root_task_id"] == "root-1"
    assert current["data"]["chat_id"] == 2 and current["data"]["root_task_id"] == "root-2"
    # Neither endpoint touches the other task's frame.
    stray = {"type": "llm_usage", "task_id": "task-1"}
    assert second.stamp(stray) == {"type": "llm_usage", "task_id": "task-1"}
    # The worker's own appended rows (sink-forwarded) are stamped by THEIR task id.
    row = stamp_recorded_row({"type": "llm_api_error", "task_id": "task-1", "error": "boom"})
    assert row["chat_id"] == 1 and row["root_task_id"] == "root-1"
    assert stamp_recorded_row({"type": "llm_api_error", "task_id": "task-2"})["chat_id"] == 2
    unknown = {"type": "llm_api_error", "task_id": "never-ran"}
    assert stamp_recorded_row(unknown) == {"type": "llm_api_error", "task_id": "never-ran"}


@pytest.mark.parametrize("metadata,named", [({}, True),
    ({"initiator": "consciousness", "wake_cause": "own_clock"}, False),
    ({"initiator": "consciousness", "wake_cause": "focus"}, False),
    ({"initiator": "consciousness", "usage_category": "consciousness_task"}, True)])
def test_the_conversation_carrier_endpoint_keeps_its_turn_stamps(monkeypatch, tmp_path, metadata, named):
    """The carrier names working Main turns, while a wake supplies its own title at finish."""
    from ouroboros import project_naming
    from ouroboros.task_carrier import EXECUTOR_WORKER, LANE_CONVERSATION
    from supervisor.worker_process import run_worker_task

    (tmp_path / "logs").mkdir()
    captured, inner = _capture()
    fired = []
    monkeypatch.setattr(project_naming, "spawn_turn_namer", lambda *a, **kw: fired.append(1))
    agent = SimpleNamespace()

    def handle_task(task):
        turn = agent._event_queue
        turn.put({"type": "task_done", "task_id": task["id"]})
        assert captured[-1]["chat_id"] == 42 and captured[-1]["_is_direct_chat"] is True
        assert captured[-1].get("initiator", "") == metadata.get("initiator", "")
        turn.put({"type": "log_event", "data": {"type": "tool_call_started", "task_id": task["id"], "tool": "read_file"}})
        return []

    agent.handle_task = handle_task
    # The admitted seed as the lane dispatches it: the carrier facts and the first-step facts.
    task = {"id": "turn-1", "chat_id": 42, "text": "Work", "metadata": dict(metadata), "_is_direct_chat": True,
            "_execution_owner": {"kind": "direct", "lane": LANE_CONVERSATION, "executor": EXECUTOR_WORKER},
            "_conversation_prepare": {"quiet": False, "reply_expected": True}}
    run_worker_task(agent, task, wid=0, out_q=inner, repo_dir=str(tmp_path), drive_root=str(tmp_path))
    assert captured[-1]["data"]["cancelable"] is True and fired == ([1] if named else [])
    assert "root_task_id" not in captured[-1]["data"], "a turn without lineage stamps none"


def test_a_pooled_task_publishes_its_start_phases_until_its_first_call(tmp_path):
    """#2862: a pooled root's endpoint feeds the same start-phase producer as a carried turn (``TurnPhaseFrames``):
    one execution, its attempt, forward only, nothing after its body ended and no direct-turn receipt."""
    from supervisor.worker_process import run_worker_task

    (tmp_path / "logs").mkdir()
    captured, inner = _capture()
    agent = SimpleNamespace()
    held = {}

    def handle_task(task):
        turn = held["endpoint"] = agent._event_queue
        tid = task["id"]
        turn.put({"type": "log_event", "data": {"type": "context_building_started", "task_id": tid}})
        turn.put({"type": "log_event", "data": {"type": "context_building_finished", "task_id": tid}})
        turn.put({"type": "main_llm_call_state", "task_id": tid, "llm_call_id": "c1", "phase": "started"})
        turn.put({"type": "main_llm_call_state", "task_id": tid, "llm_call_id": "c1", "phase": "observed",
                  "model_activity": {"phase": "awaiting_response"}})
        turn.put({"type": "log_event", "data": {"type": "context_building_started", "task_id": tid}})  # a later round
        return []

    agent.handle_task = handle_task
    task = {"id": "pooled-root", "chat_id": 42, "text": "Work", "_attempt": 2,
            "_execution_owner": {"kind": "pooled", "lane": "project"}}
    run_worker_task(agent, task, wid=3, out_q=inner, repo_dir=str(tmp_path), drive_root=str(tmp_path))
    held["endpoint"].put({"type": "log_event", "data": {"type": "context_building_started", "task_id": "pooled-root"}})
    phases = [frame for frame in captured if frame.get("type") == "direct_turn_phase"]
    # Its first frame leaves at the start (the slot's row says Preparing and is bound to this execution at once).
    assert [frame["phase"] for frame in phases] == ["started", "context_building", "context_built", "requested",
                                                    "dispatched"]
    assert {(frame["task_id"], frame["task_attempt"], frame["worker_id"]) for frame in phases} == {("pooled-root", 2, 3)}
    assert len({frame["execution_id"] for frame in phases}) == 1
    assert not any("direct_turn" in frame for frame in captured), "a pooled execution leaves no direct-turn receipt"


def test_the_worker_body_installs_the_endpoint_around_handle_task_and_the_sink_stamps_by_task():
    from supervisor.worker_process import run_worker_task, worker_main

    body = inspect.getsource(run_worker_task)
    assert "task_queue = task_event_queue(out_q, task" in body
    assert "runner._event_queue = task_queue" in body and "runner._event_queue = pool_queue" in body
    assert "task_queue.put(stamp_finalization_enqueue(e2))" in body
    assert "stamp_recorded_row(dict(obj))" in inspect.getsource(worker_main)


@pytest.fixture
def host(tmp_path):
    (tmp_path / "logs").mkdir()
    frames = []
    return SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING={}, frames=frames,
        bridge=SimpleNamespace(push_log=frames.append),
        append_jsonl=__import__("ouroboros.utils", fromlist=["append_jsonl"]).append_jsonl,
    )


def test_a_usage_row_handled_after_task_done_keeps_its_lineage_and_chat(host):
    """RUNNING is empty (task_done already popped the row): the durable row still
    carries the producer-stamped attribution; the RUNNING fill is only a fallback."""
    from supervisor.events_budget import _handle_llm_usage

    captured, inner = _capture()
    task_event_queue(inner, _task()).put({"type": "llm_usage", "task_id": "child-1", "model": "m",
                                          "usage": {"prompt_tokens": 3, "cost": 0.01}})
    _handle_llm_usage(captured[-1], host)
    row = json.loads((host.DRIVE_ROOT / "logs" / "events.jsonl").read_text(encoding="utf-8"))
    assert {key: row[key] for key in TASK_LINEAGE_FIELDS} == LINEAGE
    assert row["chat_id"] == 7 and host.frames[-1]["chat_id"] == 7
    # The fallback still fills a gap an older emitter left, from RUNNING, when it is there.
    host.RUNNING["old-1"] = {"task": {"id": "old-1", "root_task_id": "root-old", "delegation_role": "subagent"}}
    _handle_llm_usage({"type": "llm_usage", "task_id": "old-1", "usage": {"prompt_tokens": 1}}, host)
    rows = [json.loads(line) for line in (host.DRIVE_ROOT / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["root_task_id"] == "root-old" and rows[-1]["delegation_role"] == "subagent"


def test_a_real_checkpoint_envelope_handled_after_task_done_routes_to_the_right_chat(host):
    """The checkpoint producer emits through the task endpoint; the supervisor's
    log handler addresses the frame from what it carries: the task chat without a
    Project, the Project thread (resolved from the carried lineage) with one."""
    from ouroboros.loop_messages import _emit_checkpoint_event
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from supervisor.events_worker_reports import _handle_log_event

    captured, inner = _capture()
    _emit_checkpoint_event(task_event_queue(inner, _task()), "child-1", host.DRIVE_ROOT / "logs", {
        "checkpoint_kind": "usage_lock_wait", "owner_visible": True, "phase": "entered",
        "role": "system", "system_type": "task_checkpoint", "content": "Waiting for accounting access",
    })
    envelope = captured[-1]
    assert envelope["type"] == "log_event" and envelope["data"]["root_task_id"] == "root-1"
    assert envelope["data"]["chat_id"] == 7
    _handle_log_event(envelope, host)
    frame = host.frames[-1]
    assert frame["type"] == "task_checkpoint" and frame["chat_id"] == 7
    assert frame["parent_task_id"] == "parent-1" and frame["root_task_id"] == "root-1"

    project = create_project(host.DRIVE_ROOT, "proj", name="Project")
    bind_task_to_project(host.DRIVE_ROOT, "root-1", "proj", origin={"absent": "system"})
    _handle_log_event(envelope, host)
    assert host.frames[-1]["chat_id"] == int(project["chat_id"]), "the binding resolves from the carried lineage"
    assert host.RUNNING == {}
