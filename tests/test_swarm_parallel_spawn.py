"""WS1+WS4: parallel schedule_subagent emission gate + parent-side thread safety."""
from __future__ import annotations

import threading
import types

from ouroboros.loop_tool_execution import escaped_errors_become_rows
from ouroboros.tool_capabilities import (
    PARALLEL_SAFE_ENQUEUE_TOOLS,
    READ_ONLY_PARALLEL_TOOLS,
    call_segments,
)
from ouroboros.tools.control import _emit_control_event, _record_scheduled_subagent


def _call(name):
    return {"function": {"name": name}}


# --- WS1: parallel gate ---
def test_schedule_subagent_parallel_safe_not_readonly():
    assert "schedule_subagent" in PARALLEL_SAFE_ENQUEUE_TOOLS
    assert "schedule_subagent" not in READ_ONLY_PARALLEL_TOOLS


def test_burst_of_spawns_runs_parallel():
    assert call_segments(["schedule_subagent"] * 4) == [[0, 1, 2, 3]]
    assert escaped_errors_become_rows([_call("schedule_subagent")] * 4) is True


def test_mixed_reads_and_spawns_runs_parallel():
    assert call_segments(["read_file", "schedule_subagent", "search_code"]) == [[0, 1, 2]]
    assert escaped_errors_become_rows(
        [_call("read_file"), _call("schedule_subagent"), _call("search_code")]
    ) is True


def test_any_non_parallel_tool_forces_sequential():
    """A call outside the class runs alone; the spawn before it is a group of one, the single-call path."""
    assert call_segments(["schedule_subagent", "write_file"]) == [[0], [1]]
    assert escaped_errors_become_rows(
        [_call("schedule_subagent"), _call("write_file")]
    ) is False


def test_the_exception_rule_reads_frozen_history_not_the_live_read_set(monkeypatch):
    """Old+old keeps the pool's row; old+new and new+new end on the exception, as sequential responses did; naming a
    read in the live set later changes neither."""
    from ouroboros import tool_capabilities
    monkeypatch.setattr(tool_capabilities, "READ_ONLY_PARALLEL_TOOLS", READ_ONLY_PARALLEL_TOOLS | {"knowledge_read"})
    assert escaped_errors_become_rows([_call("read_file"), _call("search_code")]) is True
    assert escaped_errors_become_rows([_call("read_file"), _call("knowledge_read")]) is False
    assert escaped_errors_become_rows([_call("tree_read"), _call("knowledge_read")]) is False
    assert call_segments(["read_file", "knowledge_read"]) == call_segments(["tree_read", "knowledge_read"]) == [[0, 1]]


def test_single_call_is_not_parallel():
    assert call_segments(["schedule_subagent"]) == [[0]]
    assert escaped_errors_become_rows([_call("schedule_subagent")]) is False


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
