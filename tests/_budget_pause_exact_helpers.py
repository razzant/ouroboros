"""Shared fixtures of the exact budget-pause suite (#1196): the installed queue,
the loop-side pause context, the fast HOLD stubs, one recorded pause and the
supervisor-side context and parked row (``test_budget_pause_exact``,
``test_budget_pause_exact_resume``, ``test_budget_pause_holds``,
``test_budget_pause_safety``, ``test_no_tool_budget_parity``)."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest



def _install_queue(tmp_path, monkeypatch):
    from supervisor import queue, state, workers

    state.init(tmp_path, total_budget_limit=10.0)
    queue.init(tmp_path)
    workers.DRIVE_ROOT = tmp_path
    queue.DRIVE_ROOT = tmp_path
    workers.PENDING[:] = []
    workers.RUNNING.clear()
    monkeypatch.setattr(workers, "WORKERS", {})
    queue.BUDGET_ROOT_FENCES.clear()
    queue.init_queue_refs(workers.PENDING, workers.RUNNING, workers.QUEUE_SEQ_COUNTER_REF)
    monkeypatch.setattr(workers, "load_state", lambda: {"owner_chat_id": 0})
    monkeypatch.setattr(queue, "load_state", lambda: {"owner_chat_id": 0}, raising=False)
    return queue, state, workers


def _running_row(root, task_id):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    write_task_result(root, task_id, STATUS_RUNNING, result="running")


def _loop_ctx(root, task_id="pause-task", *, direct=False, attempt=1):
    ctx = SimpleNamespace(
        task_id=task_id, task_attempt=attempt, drive_root=root, budget_drive_root=root,
        is_direct_chat=direct, owner_wait_callback=(lambda *_a, **_k: "owner_input"),
        task_started_at=time.time() - 100.0, root_task_id=task_id,
        _cost_ceiling=None, model_wait_context=None, context_fit_plan=None,
        _owner_directives=[], _delivery_candidate=None, _accumulated_usage={},
        task_metadata={}, active_model="m", active_effort="high", active_use_local=False,
        active_context_mode="max", event_queue=None,
    )
    messages = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_a", "type": "function", "function": {"name": "x", "arguments": "{}"}},
            {"id": "call_b", "type": "function", "function": {"name": "y", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_a", "content": "done a"},
    ]
    limit_ctx = SimpleNamespace(
        tools=SimpleNamespace(_ctx=ctx), accumulated_usage={"cost": 1.25, "execution_status": "failed",
                                                              "reason_code": "budget_exhausted"},
        messages=messages, llm_trace={"tool_calls": [1]}, owner_msg_seen={"m1"}, round_idx=4,
        tool_schemas=[{"type": "function", "function": {"name": "x"}}],
    )
    return ctx, limit_ctx


def _controls(*reasons):
    """A ``_hold_control_reason`` stub: these reasons in order, then silence."""
    remaining = list(reasons)
    return lambda _ctx: remaining.pop(0) if remaining else ""


def _fast_hold(monkeypatch, budget_pause):
    """Poll fast and leave the task's controls quiet unless a test says otherwise."""
    monkeypatch.setattr(budget_pause, "_HOLD_POLL_SEC", 0.01)
    monkeypatch.setattr(budget_pause, "_hold_control_reason", lambda _ctx: "")


def _pause(tmp_path, monkeypatch, *, task_id="pause-task", rail=None, scope="global",
           root_task_id=None):
    from ouroboros import budget_pause

    rail = rail or budget_pause.RAIL_GLOBAL_EXHAUSTED
    _running_row(tmp_path, task_id)
    ctx, limit_ctx = _loop_ctx(tmp_path, task_id)
    ctx.root_task_id = root_task_id or task_id
    _fast_hold(monkeypatch, budget_pause)
    _mock_pause_observation(monkeypatch, budget_pause, [
        {"run_id": "run-1", "state": "stop_requested", "stop_outcome": "requested"}])
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.request_pause(limit_ctx, rail=rail, scope=scope, reason_text="money gone",
                                   root_task_id=root_task_id or task_id)
    return ctx, limit_ctx, raised.value.pause


def _mock_pause_observation(monkeypatch, budget_pause, runs):
    """Stub the LOOP-side custody observation only (the pause names its own
    ``reason``); the grant side keeps re-reading custody through the real body."""
    real_observe = budget_pause.observe_task_runs

    def observe(root, task_id, **kw):
        if kw.get("reason") != "budget_pause_uncovered_cost":
            return real_observe(root, task_id, **kw)
        return {"runs": list(runs), "observed_at": time.time(), "custody_read": "ok",
                "coverage_basis": "test"}

    monkeypatch.setattr(budget_pause, "observe_task_runs", observe)


def _quiet_external(monkeypatch, budget_pause):
    _mock_pause_observation(monkeypatch, budget_pause, [])


def _supervisor_ctx(tmp_path, workers, queue, persisted, pushed):
    return SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING, WORKERS=workers.WORKERS,
        sort_pending=lambda: None,
        persist_queue_snapshot=lambda reason="": (persisted.append(reason) or True),
        bridge=SimpleNamespace(push_log=lambda event: pushed.append(event)),
    )


def _parked(tmp_path, monkeypatch, *, task_id="pause-task", scope="global", root_task_id=None, extra=None):
    from ouroboros import budget_pause
    from supervisor import workers

    # ONE lineage: the durable row, its queue marker and the pending task row all
    # name the same root, or a descendant's resume cannot see its paused root.
    ctx, _limit, pause = _pause(tmp_path, monkeypatch, task_id=task_id, scope=scope,
                                root_task_id=root_task_id or task_id)
    budget_pause.end_dispatch_fence(task_id)
    row = budget_pause.budget_pause_row(tmp_path, task_id)
    assert row["root_task_id"] == (root_task_id or task_id)
    marker = budget_pause.exact_pause_marker(row, default_root=root_task_id or task_id)
    if scope == "root":
        from supervisor.events_budget import _set_root_budget_pause_locked

        fence = _set_root_budget_pause_locked(marker["root_task_id"], marker)
        marker["fence_id"] = fence["fence_id"]
    task = {"id": task_id, "type": "task", "chat_id": 0, "root_task_id": root_task_id or task_id,
            "_attempt": 1, "_budget_pause": marker, **(extra or {})}
    workers.PENDING.append(task)
    budget_pause.set_budget_pause(tmp_path, task_id, {**row, "state": budget_pause.STATE_PAUSED})
    return task, row
