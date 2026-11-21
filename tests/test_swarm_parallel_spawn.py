"""WS1+WS4: parallel schedule_subagent emission gate + parent-side thread safety."""
from __future__ import annotations

import threading
import types

from ouroboros.tool_capabilities import (
    PARALLEL_SAFE_ENQUEUE_TOOLS,
    READ_ONLY_PARALLEL_TOOLS,
    call_segments,
    legacy_pool_response,
)
from ouroboros.tools.control import _emit_control_event, _record_scheduled_subagent


def _call(name):
    return {"function": {"name": name}}


# --- WS1: parallel gate ---
def test_schedule_subagent_parallel_safe_not_readonly():
    assert "schedule_subagent" in PARALLEL_SAFE_ENQUEUE_TOOLS
    assert "schedule_subagent" not in READ_ONLY_PARALLEL_TOOLS


def test_burst_of_spawns_runs_parallel():
    """A burst of spawns is a legacy response: one pool over every call, as before."""
    assert legacy_pool_response(["schedule_subagent"] * 4) is True


def test_mixed_reads_and_spawns_runs_parallel():
    assert legacy_pool_response(["read_file", "schedule_subagent", "search_code"]) is True


def test_any_non_parallel_tool_forces_sequential():
    """Outside the legacy pool a spawn is no inert read: it runs alone, and so does the write after it."""
    assert legacy_pool_response(["schedule_subagent", "write_file"]) is False
    assert call_segments(["schedule_subagent", "write_file"]) == [[0], [1]]
    # A read in the same group would not see the child it schedules (a task-state or events-log read).
    assert call_segments(["schedule_subagent", "tree_read", "write_file"]) == [[0], [1], [2]]
    assert call_segments(["schedule_subagent", "recent_tasks", "write_file"]) == [[0], [1], [2]]


def test_the_legacy_pool_reads_frozen_history_not_the_live_read_set(monkeypatch):
    """Old+old is the legacy pool with its rows; old+new and new+new run in segments, where an escaped exception ends
    the response as one call at a time did; naming a read in the live set later changes neither."""
    from ouroboros import tool_capabilities
    monkeypatch.setattr(tool_capabilities, "READ_ONLY_PARALLEL_TOOLS", READ_ONLY_PARALLEL_TOOLS | {"knowledge_read"})
    assert legacy_pool_response(["read_file", "search_code"]) is True
    assert legacy_pool_response(["read_file", "knowledge_read"]) is False
    assert legacy_pool_response(["tree_read", "knowledge_read"]) is False
    assert call_segments(["read_file", "knowledge_read"]) == call_segments(["tree_read", "knowledge_read"]) == [[0, 1]]


def test_single_call_is_not_parallel():
    assert legacy_pool_response(["schedule_subagent"]) is False
    assert call_segments(["schedule_subagent"]) == [[0]]


# --- WS4: parent-side thread safety (no lost updates under concurrency) ---
def _fake_ctx():
    # event_queue=None forces the pending_events fallback in _emit_control_event.
    return types.SimpleNamespace(event_queue=None, pending_events=[], _last_scheduled_subagents=[])


def _run_concurrently(fn, n=200):
    start = threading.Event()

    def worker(i):
        start.wait()
        fn(i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join()


def test_concurrent_record_scheduled_subagent_no_lost_updates():
    ctx = _fake_ctx()
    _run_concurrently(lambda i: _record_scheduled_subagent(ctx, {"task_ids": [f"t{i}"]}))
    assert len(ctx._last_scheduled_subagents) == 200
    assert {r["task_ids"][0] for r in ctx._last_scheduled_subagents} == {f"t{i}" for i in range(200)}


def test_concurrent_emit_control_event_fallback_no_lost_updates():
    ctx = _fake_ctx()
    _run_concurrently(lambda i: _emit_control_event(ctx, {"type": "x", "i": i}))
    assert len(ctx.pending_events) == 200
