"""Original execution clocks through real recovery, assignment and agent startup."""
from __future__ import annotations

import json
import queue as stdqueue
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import agent as agent_module, config, model_wait, working_checkpoint as wc
from ouroboros.tools.registry import ToolRegistry
from supervisor import queue, queue_timeouts, task_reaper, worker_assignment, workers
from tests._budget_pause_exact_helpers import _loop_ctx
from tests.test_project_hold_recovery import worker
from tests.test_restart_saved_work import _stop_and_boot
from tests.test_swarm_host_admission import host  # noqa: F401
from tests.test_working_recovery_admission import _running

pytestmark = pytest.mark.serial


def _recover(host, tmp_path, monkeypatch, *, new_id=False, legacy=False):  # noqa: F811 - the imported fixture
    original = _running(host, tmp_path, monkeypatch, "main")
    now = time.time()
    clock = SimpleNamespace(wall=now, mono=10000.0)
    fake_time = SimpleNamespace(time=lambda: clock.wall, monotonic=lambda: clock.mono, sleep=time.sleep)
    for module in (model_wait, worker_assignment, agent_module):
        monkeypatch.setattr(module, "time", fake_time)
    monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: 750.0)
    monkeypatch.setattr(queue, "get_task_abs_ceiling_sec", lambda: 750.0)
    original["deadline_at"] = "2099-01-01T00:00:00+00:00"
    workers.RUNNING["held"]["task"].update(deadline_at=original["deadline_at"])
    ctx, limit = _loop_ctx(host.root, "held", attempt=1)
    ctx.task_started_at = now - 1000.0
    ctx.model_wait_context = model_wait.TaskModelWait(
        task=original, drive_root=host.root, event_queue=None, worker_slot_held=True)
    ctx.model_wait_context.restore_continuation({
        "quota_clock": {"elapsed_sec": 120.0, "observed_at": now, "active": False},
        "budget_paused_sec": 180.0, "auto_continue": {"main": False},
    }, started_at=ctx.task_started_at)
    assert wc.save_round(limit, "post_batch")
    path = wc.checkpoint_path(host.root, "held", 1)
    saved = json.loads(path.read_bytes())
    if legacy:
        saved.pop("started_at", None)
        path.write_text(json.dumps(saved))
    else:
        assert saved["started_at"] == now - 1000.0
    if new_id:
        workers.RUNNING.clear()
        answer = task_reaper._enqueue_retry(queue, original, task_id="held", retry_task_id="held-retry",
                                           attempt=1, terminal_reason="idle_timeout", recon_fields={})
        assert answer[:2] == (True, 2), answer
    else:
        assert _stop_and_boot(queue, workers) == 1
        assert queue.resume_budget_paused_task("held")["ok"]
    [task] = host.pending
    assert task["_attempt"] == 2
    locator = task["_working_recovery"]
    if legacy:
        assert not any(key in locator for key in ("started_at", "model_wait_quota_clock", "budget_paused_sec"))
    else:
        assert locator["started_at"] == now - 1000.0
        assert locator["model_wait_quota_clock"]["elapsed_sec"] == 120.0
        assert locator["model_wait_quota_clock"]["active"] is False
        assert locator["budget_paused_sec"] == 180.0
    return task, clock


@pytest.mark.parametrize("new_id", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_assignment_keeps_cumulative_execution_but_resets_idle_only(host, tmp_path, monkeypatch, new_id, legacy):  # noqa: F811
    task, clock = _recover(host, tmp_path, monkeypatch, new_id=new_id, legacy=legacy)
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert [row["id"] for row in sent] == [task["id"]]
    running = workers.RUNNING[task["id"]]
    assert running["started_at"] == clock.wall - (0.0 if legacy else 1000.0)
    assert running["last_progress_at"] == running["last_heartbeat_at"] == clock.wall
    assert model_wait.execution_elapsed_seconds(running, clock.wall) == (0.0 if legacy else 700.0)
    assert running["budget_paused_sec"] == (0.0 if legacy else 180.0)
    assert task["deadline_at"] == "2099-01-01T00:00:00+00:00"

    # Read the actual RUNNING row through the supervisor's finite lifetime rail.
    # Restored exclusions spare the task at 700 s (raw wall time is 1000 s),
    # while preserving the original start makes it expire 51 s later.
    monkeypatch.setattr(queue, "FINALIZATION_GRACE_SEC", 30.0)
    requests = []
    monkeypatch.setattr(queue, "_request_finalization_grace",
                        lambda _root, tid, reason, **_kw: requests.append((tid, reason)) or "clock-grace")
    queue_timeouts._enforce_task_timeouts_locked(workers, clock.wall, 1, {})
    assert not requests
    queue_timeouts._enforce_task_timeouts_locked(workers, clock.wall + 51.0, 1, {})
    assert requests == ([] if legacy else [(task["id"], "absolute_ceiling")])
    assert not host.attempts


class _AtRuntime(BaseException):
    """Stop after actual clock restoration, before any cognition/provider work."""


def _agent_start(root, monkeypatch, task, clock, observe):
    agent = object.__new__(agent_module.OuroborosAgent)
    agent.__dict__.update(
        env=agent_module.Env(repo_dir=root, drive_root=root),
        tools=ToolRegistry(repo_dir=root, drive_root=root), memory=None, llm=None,
        _event_queue=None, _incoming_messages=stdqueue.Queue(),
        _owner_message_admission_lock=threading.RLock(), owner_wait_callback=lambda *_: None,
    )
    for name in ("_emit_live_log", "_emit_typing_start", "_emit_progress", "_capture_mutation_baseline",
                 "_start_task_heartbeat_loop"):
        monkeypatch.setattr(agent, name, lambda *_a, **_kw: None)
    monkeypatch.setattr(agent, "_run_delegate_preflight", lambda _logs, _task, dispatch: (dispatch, False))

    def runtime_messages(*, ctx, **_kwargs):
        assert agent._task_started_ts == ctx.task_started_at
        assert agent._last_progress_ts == clock.wall
        observe(ctx)
        raise _AtRuntime

    monkeypatch.setattr(agent_module, "build_llm_messages", runtime_messages)
    with model_wait.task_model_wait_scope(task=task, drive_root=root, event_queue=None, worker_slot_held=True):
        with pytest.raises(_AtRuntime):
            agent._handle_task_scoped(task)


@pytest.mark.parametrize("new_id", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_real_agent_start_restores_same_finite_clock_before_runtime(host, tmp_path, monkeypatch, new_id, legacy):  # noqa: F811
    task, clock = _recover(host, tmp_path, monkeypatch, new_id=new_id, legacy=legacy)
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    [dispatched] = sent

    def observe(ctx):
        waiter = ctx.model_wait_context
        assert ctx.task_started_at == clock.wall - (0.0 if legacy else 1000.0)
        assert waiter.executed_seconds() == (0.0 if legacy else 700.0)
        assert waiter.execution_window_remaining() == (750.0 if legacy else 50.0)
        assert waiter.budget_paused_sec == (0.0 if legacy else 180.0)
        assert waiter.paused_seconds() == (0.0 if legacy else 120.0)
        assert waiter.auto_continue == ({} if new_id else {"main": False})
        state = wc.load_recovery(ctx)
        assert state["model_wait"]["budget_paused_sec"] == (0.0 if legacy else 180.0)
        # The cold loop's cognition restore must not reintroduce old exclusions
        # when its next checkpoint or park serializes this context.
        from ouroboros.owner_wait import restore_continuation_state
        restore_continuation_state(SimpleNamespace(_ctx=ctx), state, [], {}, {}, set())
        assert ctx._budget_paused_sec == (0.0 if legacy else 180.0)
        assert waiter.control_reason() is None
        clock.wall += 51.0
        clock.mono += 51.0
        assert waiter.control_reason() == (None if legacy else "absolute_ceiling")
        # Calendar time remains its own hard axis, even with unused lifetime.
        monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: None)
        waiter.task["deadline_at"] = "2000-01-01T00:00:00+00:00"
        assert waiter.control_reason() == "deadline"

    _agent_start(host.root, monkeypatch, dispatched, clock, observe)
    assert not host.attempts


def test_source_less_start_keeps_existing_fresh_clock(host, tmp_path, monkeypatch):  # noqa: F811
    from tests.test_main_project_consistency import admit_main

    task = admit_main(host, tid="fresh")
    now = time.time()
    clock = SimpleNamespace(wall=now, mono=10000.0)
    monkeypatch.setattr(agent_module, "time", SimpleNamespace(time=lambda: clock.wall))
    monkeypatch.setattr(model_wait, "time", SimpleNamespace(time=lambda: clock.wall, monotonic=lambda: clock.mono))
    monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: 750.0)

    def observe(ctx):
        assert ctx.task_started_at == clock.wall
        assert ctx.model_wait_context.executed_seconds() == 0.0
        assert ctx.model_wait_context.execution_window_remaining() == 750.0

    _agent_start(host.root, monkeypatch, task, clock, observe)
    assert not host.attempts
