"""Shared fixtures of the exact budget-pause suite (#1196): the installed queue,
the loop-side pause context, the fast HOLD stubs, one recorded pause and the
supervisor-side context and parked row (``test_budget_pause_exact``,
``test_budget_pause_exact_resume``, ``test_budget_pause_holds``,
``test_budget_pause_safety``, ``test_no_tool_budget_parity``)."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from tests.worker_pool_support import fixed_pool


def _install_fresh_supervisor_memory(monkeypatch):
    """The owner-Pause settlement pass's memory-only book, fresh per test.

    ``supervisor.owner_pause_control`` keeps ``_LOOK_OWED`` / ``_CHANGED`` / ``_RECOVERED`` /
    ``_AT_REST`` / ``_UNWRITTEN_STOPS`` as plain module globals its producers mutate in place
    (``mark_tree_changed``, ``request_settlement_look``). One left set by an earlier test makes
    the next tick's ``settlement_look_owed`` true, so a later test's maintenance tick starts the
    ``owner_pause_settlement`` duty thread out of turn (#3428's cross-test leak). Rebinding
    fresh objects lets monkeypatch teardown restore whatever preceded the test."""
    from supervisor import owner_pause_control

    # _LOOK_OWED keeps the module's boot value: a process's first look reads every fenced tree once.
    monkeypatch.setattr(owner_pause_control, "_LOOK_OWED", [True])
    monkeypatch.setattr(owner_pause_control, "_CHANGED", set())
    monkeypatch.setattr(owner_pause_control, "_RECOVERED", {})
    monkeypatch.setattr(owner_pause_control, "_DISCOVERY_OWED", set())
    monkeypatch.setattr(owner_pause_control, "_AT_REST", {})
    monkeypatch.setattr(owner_pause_control, "_UNWRITTEN_STOPS", {})
    # task_lifecycle's cancellation-fence registries are per-PROCESS but keyed by task id alone,
    # so a cancel of a fixture id (e.g. "root-1") in one test fences the same id in the next
    # (#3428's cross-file leak). Fresh objects per test; monkeypatch teardown restores the prior ones.
    from supervisor import task_lifecycle

    monkeypatch.setattr(task_lifecycle, "CANCELLED_ROOT_FENCES", {})
    monkeypatch.setattr(task_lifecycle, "_ACTIVE_CASCADE_FENCES", {})


def _install_queue(tmp_path, monkeypatch):
    from supervisor import queue, state, workers

    state.init(tmp_path, total_budget_limit=10.0)
    queue.init(tmp_path)
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    # The fixture owns these maps. In-place clearing leaked its last live task
    # into later settings/consumer tests after monkeypatch teardown.
    monkeypatch.setattr(workers, "PENDING", [])
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(workers, "WORKERS", {})
    fixed_pool(monkeypatch)  # the slots a test installs, never a new process
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr(queue, "PENDING", workers.PENDING)
    monkeypatch.setattr(queue, "RUNNING", workers.RUNNING)
    # queue.init keeps live acceptance seals: an earlier test's seal would enter this
    # test's snapshot and fence its root at restore.
    monkeypatch.setattr(queue, "ACCEPTANCE_FENCES", {})
    queue.BUDGET_ROOT_FENCES.clear()
    _install_fresh_supervisor_memory(monkeypatch)
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
        active_model=ctx.active_model, active_use_local=ctx.active_use_local,  # as _RoundLimitContext carries them
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


def _daily_wake_pause(tmp_path, monkeypatch, *, scope="global", source="allowance", cap=20.0):
    """An own wake parked at its $4 daily-allowance ceiling (``own-wake``): $11 spent by another wake, $1.25 by this one
    and $2.75 by its helper beside one unpriced call, so $7 of the day's $22 are left and the tree has spent $4."""
    from ouroboros import budget_pause, config, usage_accounting as ua
    from ouroboros.consciousness import OWN_WAKE_STATE_KEY
    from ouroboros.consciousness_wake import wake_task_metadata
    from ouroboros.task_results import write_task_result

    queue, state, _workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 100.0)
    state.save_state({"owner_chat_id": 1, "bg_consciousness_enabled": True,
                      OWN_WAKE_STATE_KEY: {"task_id": "own-wake", "wake_cause": "own_clock",
                                           "initiator": "consciousness", "finished": False}})
    monkeypatch.setattr(queue, "load_state", state.load_state)
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "22")
    metadata = wake_task_metadata("act", "test", root_cost_ceiling_usd=4.0, root_cost_ceiling_source=source)
    task, _ = _parked(tmp_path, monkeypatch, task_id="own-wake", scope=scope, extra={"metadata": metadata})
    write_task_result(tmp_path, "own-wake", "scheduled", metadata=metadata)
    row = budget_pause.budget_pause_row(tmp_path, "own-wake")
    row.update(rail=budget_pause.RAIL_GRACEFUL_CEILING, cost_ceiling={
        "state": "active", "ceiling_usd": 4.0, "root_cap_usd": cap, "basis": "producer_allowance"})
    budget_pause.set_budget_pause(tmp_path, "own-wake", row)
    task["_budget_pause"].update(rail=row["rail"])
    # $11 by another wake + $1.25 by this wake + $2.75 by its child, plus one unpriced call.
    for tid, rid, category, usd in (("other-wake", "other-wake", "consciousness", 11.0),
                                    ("own-wake", "own-wake", "consciousness", 1.25),
                                    ("wake-helper", "own-wake", "task", 2.75),
                                    ("wake-helper", "own-wake", "task", None)):
        attempt = ua.reserve_attempt(ua.AttemptRequest(model="test", provider="openai", drive_root=tmp_path,
            task_id=tid, root_task_id=rid, category=category, source="test", allow_live_fetch=False,
            reservation_usd=usd, force_unknown_reservation=usd is None, global_limit_usd=100.0))
        ua.mark_dispatched(attempt)
        ua.settle_attempt(attempt, {}, cost_usd=usd, cost_final=usd is not None)
    return queue, task, metadata


def _wake_http_app(tmp_path):
    """The registered API and clock read the fixture's state; no supervisor ticks or worker processes."""
    from starlette.applications import Starlette

    from ouroboros.consciousness import BackgroundConsciousness
    from ouroboros.consciousness_view import describe_consciousness
    from ouroboros.gateway.router import collect_routes

    clock = BackgroundConsciousness(tmp_path, tmp_path, lambda: 1)
    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path
    app.state.describe_consciousness = lambda: describe_consciousness(clock)
    return app


def _observe_allowance_reads(monkeypatch):
    """Record ``(allow_stale, queue lock held)`` without changing an input or result of the real decision."""
    from ouroboros import consciousness_allowance as allowance
    from supervisor import queue as q

    real, calls = allowance.allowance_resume_decision, []

    def decision(*args, **kwargs):
        calls.append((kwargs.get("allow_stale", False), q._queue_lock._is_owned()))
        return real(*args, **kwargs)

    monkeypatch.setattr(allowance, "allowance_resume_decision", decision)
    return calls


def _charge_elsewhere(tmp_path, usd):
    """An intervening charge by another wake in the real rolling ledger."""
    from ouroboros import usage_accounting as ua

    attempt = ua.reserve_attempt(ua.AttemptRequest(
        model="test", provider="openai", drive_root=tmp_path, task_id="other-wake", root_task_id="other-wake",
        category="consciousness", source="test", reservation_usd=usd, global_limit_usd=100.0, allow_live_fetch=False))
    ua.mark_dispatched(attempt)
    ua.settle_attempt(attempt, {}, cost_usd=usd, cost_final=True)


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
