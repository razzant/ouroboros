"""Host facts behind the conversation activity block (WP-G).

``_is_direct_chat`` (a direct conversation turn vs a managed/Swarm root) is a
host fact for routing, the census kind, Stop custody and the header pill; the
chat block's chrome follows the work it holds, never this fact. The block
represents an addressing-only turn by the
typed routing action, never by a client tool-name list. Live frames learn the
fact from the activity census and the rebuilt ``task_done``; replay learns it
from the terminal truth annotation and the free host facts row. These tests
pin each producer of that one fact.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import ouroboros.agent_task_pipeline as pipeline
from ouroboros.gateway.history import (
    _annotate_terminal_task_truth,
    _copy_task_summary_metadata,
    make_chat_history_endpoint,
)


def test_terminal_truth_carries_the_direct_turn_fact(tmp_path):
    progress = {"task_id": "t", "is_progress": True, "text": "working", "ts": "2026-09-15T00:00:00Z"}
    summary = {"task_id": "t", "role": "system", "system_type": "task_summary", "ts": "2026-09-15T00:00:05Z"}
    direct = {"t": {"status": "completed", "_is_direct_chat": True}}
    # The summary row is the truth anchor when present; the latest progress
    # row carries the fact only when no summary row is in the window.
    _annotate_terminal_task_truth([progress, summary], tmp_path, direct)
    assert summary["_is_direct_chat"] is True and "_is_direct_chat" not in progress
    alone = {"task_id": "t", "is_progress": True, "text": "working", "ts": "2026-09-15T00:00:00Z"}
    _annotate_terminal_task_truth([alone], tmp_path, direct)
    assert alone["_is_direct_chat"] is True
    managed = [{"task_id": "m", "is_progress": True, "text": "working", "ts": "2026-09-15T00:00:00Z"}]
    _annotate_terminal_task_truth(managed, tmp_path, {"m": {"status": "completed"}})
    assert managed[0]["_is_direct_chat"] is False


def test_summary_row_copies_direct_fact_and_typed_routing_action():
    rec: dict = {}
    _copy_task_summary_metadata(rec, {
        "type": "task_summary", "_is_direct_chat": True, "typed_routing_action": "promote_chat_to_task",
    })
    assert rec["_is_direct_chat"] is True
    assert rec["addressing_only"] == "promote_chat_to_task"
    plain: dict = {}
    _copy_task_summary_metadata(plain, {"type": "task_summary"})
    assert "_is_direct_chat" not in plain and "addressing_only" not in plain


def test_facts_row_writes_the_facts_and_history_replays_them(tmp_path):
    drive_logs = tmp_path / "logs"
    drive_logs.mkdir(parents=True)
    pipeline._record_task_facts(
        env=None,
        task={"id": "direct-1", "type": "task", "text": "hi", "chat_id": 1, "_is_direct_chat": True},
        usage={"rounds": 3, "cost": 0.0, "typed_routing_action": "promote_chat_to_task"},
        llm_trace={"tool_calls": [{"tool": "promote_chat_to_task"}], "reasoning_notes": []},
        drive_logs=drive_logs,
    )
    row = json.loads((drive_logs / "chat.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["summary_kind"] == "host_task_facts" and row["text"] == ""
    assert row["_is_direct_chat"] is True
    assert row["typed_routing_action"] == "promote_chat_to_task"
    # No task_results file: the history summary row falls back to the row's copy.
    (drive_logs / "progress.jsonl").write_text("", encoding="utf-8")
    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"limit": "10"})))
    payload = json.loads(response.body.decode("utf-8"))["messages"]
    summary = next(item for item in payload if item.get("system_type") == "task_summary")
    assert summary["_is_direct_chat"] is True
    assert summary["addressing_only"] == "promote_chat_to_task"
    assert summary["tool_calls"] == 1 and summary["routing_tool_calls"] == 1 and summary["rounds"] == 3


def _dispatch_task_done(tmp_path, *, evt_flag, stored_flag):
    from ouroboros.headless import prepare_terminal_task_files
    from ouroboros.task_results import write_task_result
    from supervisor.events import _handle_task_done

    pushed: list = []

    class Bridge:
        @staticmethod
        def push_log(data):
            pushed.append(json.loads(json.dumps(data)))

    class Ctx:
        DRIVE_ROOT = tmp_path
        RUNNING: dict = {}
        WORKERS: dict = {}
        PENDING: list = []
        bridge = Bridge()

        @staticmethod
        def persist_queue_snapshot(reason=""):
            pass

        @staticmethod
        def send_with_budget(chat_id, text, **kw):
            pass

        @staticmethod
        def load_state():
            return {}

        @staticmethod
        def save_state(st):
            pass

        @staticmethod
        def append_jsonl(path, data):
            from ouroboros.utils import append_jsonl
            append_jsonl(path, data)

        @staticmethod
        def sort_pending():
            pass

    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    task = {"id": "turn-1", "type": "task", "chat_id": 1, "_is_direct_chat": True}
    fields = {"result": "done", "chat_id": 1}
    if stored_flag:
        fields["_is_direct_chat"] = True
    write_task_result(tmp_path, task["id"], "completed", **fields)
    assert not prepare_terminal_task_files(tmp_path, task)["error"]
    evt = {"type": "task_done", "task_id": task["id"], "task_type": "task", "worker_id": 0,
           "chat_id": 1, "ts": "2026-09-15T00:00:00Z", "_files_prepared_attempt": 1}
    if evt_flag:
        evt["_is_direct_chat"] = True
    _handle_task_done(evt, Ctx())
    done = [row for row in pushed if row.get("type") == "task_done"]
    assert len(done) == 1
    return done[0]


def test_rebuilt_task_done_carries_the_direct_turn_fact_from_frame_or_result(tmp_path):
    assert _dispatch_task_done(tmp_path / "a", evt_flag=True, stored_flag=False)["_is_direct_chat"] is True
    # A reaper-delivered terminal carries no worker frame flag: the result answers.
    assert _dispatch_task_done(tmp_path / "b", evt_flag=False, stored_flag=True)["_is_direct_chat"] is True
    assert _dispatch_task_done(tmp_path / "c", evt_flag=False, stored_flag=False)["_is_direct_chat"] is False


def test_census_keeps_the_direct_kind_for_a_post_task_wait(tmp_path, monkeypatch):
    from ouroboros import post_task_checkpoint
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from supervisor import queue as queue_mod

    monkeypatch.setattr(queue_mod, "PENDING", [])
    monkeypatch.setattr(queue_mod, "RUNNING", {})
    monkeypatch.setattr(queue_mod, "BUDGET_ROOT_FENCES", {})

    def owner(task_id, **task):
        return SimpleNamespace(
            task_id=task_id, task={"id": task_id, "chat_id": 1, **task}, attempt=1, closed=False,
            snapshot=lambda: {"model_waits": {"w": {"wait_id": "w", "state": "waiting"}}},
        )

    monkeypatch.setattr(post_task_checkpoint, "post_task_model_waits",
                        lambda _root: [owner("direct-wait", _is_direct_chat=True), owner("managed-wait")])
    rows = {row["activity_id"]: row for row in _chat_activities_snapshot_safe(tmp_path, {}, direct_turns=[])}
    assert rows["direct-wait"]["kind"] == "direct_chat"
    assert rows["direct-wait"]["phase"] == "finalizing"
    assert rows["managed-wait"]["kind"] == "managed_task"


# --- The receipt row (WP-G2, owner decision 11.09 = 2A) ------------------------
#
# A turn that only addressed work draws no block: the annotation on the owner's
# message is the receipt. The host stamps the addressing calls on the live
# tool-call frames (`routing_action`) and counts them in the task metrics
# (`routing_tool_calls`), from the ONE routing-verb table control_events owns,
# so the client never keeps a list of tool names.


def test_the_routing_verb_table_is_the_one_owner_of_the_family():
    from ouroboros.tool_capabilities import ROUTING_VERBS, routing_action_for_tool
    from ouroboros.tools.control_events import _emit_control_event

    assert set(ROUTING_VERBS) == {"promote_chat_to_task", "route_to_project", "steer_task", "ensure_project_scope"}
    assert routing_action_for_tool(" steer_task ") == "steer_task"
    assert routing_action_for_tool("read_file") == "" and routing_action_for_tool(None) == ""
    # The typed action on task_done is keyed by the events those same tools emit.
    for tool, events in ROUTING_VERBS.items():
        for event_type in events:
            ctx = SimpleNamespace(event_queue=None, pending_events=[], drive_root=".")
            assert _emit_control_event(ctx, {"type": event_type, "routed_from_main": tool == "route_to_project"}) == "deferred"
            assert ctx._typed_routing_action_emitted == (tool if event_type == "promote_chat_to_task" else event_type)
    ctx = SimpleNamespace(event_queue=None, pending_events=[], drive_root=".")
    _emit_control_event(ctx, {"type": "cancel_task"})
    assert not hasattr(ctx, "_typed_routing_action_emitted")


def test_live_tool_call_frames_carry_the_routing_action_stamp(tmp_path):
    from ouroboros.loop_tool_execution import _execute_with_timeout
    from ouroboros.tools.tool_result import ToolResult

    drive_logs = tmp_path / "logs"
    drive_logs.mkdir()
    live: list = []
    tools = SimpleNamespace(
        CODE_TOOLS=set(),
        _ctx=SimpleNamespace(event_queue=SimpleNamespace(put_nowait=live.append)),
        execute_result=lambda _name, _args: ToolResult(status="ok", code="OK", text="OK"),
    )
    for index, tool in enumerate(("promote_chat_to_task", "read_file")):
        _execute_with_timeout(
            tools, {"id": f"call-{index}", "function": {"name": tool, "arguments": "{}"}},
            drive_logs, timeout_sec=5, task_id="turn-1",
        )
    frames = [event.get("data") or {} for event in live]
    by_call = {(row["tool_call_id"], row["type"]): row for row in frames
               if row.get("type") in {"tool_call_started", "tool_call_finished"}}
    assert by_call[("call-0", "tool_call_started")]["routing_action"] == "promote_chat_to_task"
    assert by_call[("call-0", "tool_call_finished")]["routing_action"] == "promote_chat_to_task"
    assert "routing_action" not in by_call[("call-1", "tool_call_started")]
    assert "routing_action" not in by_call[("call-1", "tool_call_finished")]


def test_task_metrics_count_the_addressing_calls_from_the_same_table(tmp_path):
    from ouroboros.post_task_synthesis import task_tool_metrics
    from ouroboros.utils import append_jsonl
    from supervisor.events_worker_reports import _handle_task_metrics

    metrics = task_tool_metrics({"tool_calls": [
        {"tool": "promote_chat_to_task"}, {"tool": "read_file"}, {"tool": "steer_task", "is_error": True},
    ]})
    assert (metrics["tool_calls"], metrics["tool_errors"], metrics["routing_tool_calls"]) == (3, 1, 2)
    assert task_tool_metrics({"tool_calls": [], "loop_evidence_unavailable": True})["routing_tool_calls"] is None
    wire: list = []
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, PENDING=[], append_jsonl=append_jsonl,
                          bridge=SimpleNamespace(push_log=wire.append))
    (tmp_path / "logs").mkdir()
    _handle_task_metrics({"task_id": "turn-1", **metrics}, ctx)
    assert wire[0]["routing_tool_calls"] == 2
    rec: dict = {}
    _copy_task_summary_metadata(rec, {"type": "task_summary", "tool_calls": 1, "routing_tool_calls": 1})
    assert rec["routing_tool_calls"] == 1
