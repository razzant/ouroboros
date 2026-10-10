"""A second Pause survives worker death with the same finite execution clock."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from ouroboros import agent, budget_pause, config, model_wait, owner_wait
from ouroboros import working_checkpoint as wc
from ouroboros.task_results import write_task_result
from supervisor import worker_health
from tests._budget_pause_exact_helpers import _fast_hold, _install_queue, _loop_ctx, _quiet_external

pytestmark = pytest.mark.serial


class _WorkerDied(BaseException):
    """Fault after the durable pausing seed and before its continuation source."""


def _consume(root, monkeypatch, task):
    ctx, limit = _loop_ctx(root, task["id"])
    ctx.task_started_at = task["_budget_pause_resume"].get("started_at") or time.time()
    ctx.model_wait_context = model_wait.TaskModelWait(
        task=task, drive_root=root, event_queue=None, worker_slot_held=True)
    agent._restore_saved_clocks(ctx, task)
    state = budget_pause.load_budget_pause(ctx)
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_kw: (None, "max"))
    budget_pause.resume_paused_loop(
        limit.tools, state, limit.messages, limit.llm_trace, limit.accumulated_usage,
        limit.owner_msg_seen, budget_remaining_usd=5.0)
    task.pop("_budget_pause_resume")
    return ctx, limit


def _after_second_pause_crash(tmp_path, monkeypatch, *, seed_origin=True, legacy=""):
    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_kw: 5.0)
    monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: 500.0)
    clock = SimpleNamespace(wall=time.time(), mono=10000.0)
    monkeypatch.setattr(time, "time", lambda: clock.wall)
    monkeypatch.setattr(model_wait, "time", SimpleNamespace(
        time=lambda: clock.wall, monotonic=lambda: clock.mono))
    _fast_hold(monkeypatch, budget_pause)
    _quiet_external(monkeypatch, budget_pause)
    task = {"id": "twice-paused", "type": "task", "root_task_id": "twice-paused",
            "_attempt": 1, "budget_drive_root": str(tmp_path)}
    write_task_result(tmp_path, task["id"], "running")
    ctx, limit = _loop_ctx(tmp_path, task["id"])
    origin = ctx.task_started_at = clock.wall - 500.0
    ctx.model_wait_context = model_wait.TaskModelWait(
        task=task, drive_root=tmp_path, event_queue=None, worker_slot_held=True)
    ctx.model_wait_context.restore_continuation({
        "quota_clock": {"elapsed_sec": 120.0, "observed_at": clock.wall, "active": False},
        "budget_paused_sec": 0.0,
    }, started_at=origin)

    # First real Pause/Resume accrues 180 seconds on the existing grant carrier.
    with pytest.raises(budget_pause.BudgetPauseRequested):
        budget_pause.request_pause(limit, rail=budget_pause.RAIL_GLOBAL_EXHAUSTED,
                                   scope="global", reason_text="first pause")
    first = budget_pause.budget_pause_row(tmp_path, task["id"])
    budget_pause.set_budget_pause(tmp_path, task["id"], {**first, "state": budget_pause.STATE_PAUSED})
    task["_budget_pause"] = budget_pause.exact_pause_marker(first, default_root=task["id"])
    workers.PENDING.append(task)
    clock.wall += 180.0
    clock.mono += 180.0
    assert queue.resume_budget_paused_task(task["id"])["ok"]
    ctx, limit = _consume(tmp_path, monkeypatch, task)
    assert ctx._budget_paused_sec == ctx.model_wait_context.budget_paused_sec == 180.0
    workers.PENDING.remove(task)

    # The task works another 100 seconds: 480 executed, 20 remain of 500.
    clock.wall += 100.0
    clock.mono += 100.0
    assert ctx.model_wait_context.executed_seconds() == 480.0
    wc.save_round(limit, "post_batch")
    path = wc.checkpoint_path(tmp_path, task["id"], 1)
    if legacy:
        saved = json.loads(path.read_bytes())
        if legacy == "missing_clock":
            saved.pop("model_wait")
        elif legacy == "missing_origin":
            saved.pop("started_at")
        path.write_text(json.dumps(saved))

    def die_before_source(*_args, **_kwargs):
        raise _WorkerDied

    monkeypatch.setattr(budget_pause, "_exact_continuation_row", die_before_source)
    with pytest.raises(_WorkerDied):
        budget_pause.request_pause(limit, rail=budget_pause.RAIL_GLOBAL_EXHAUSTED,
                                   scope="global", reason_text="second pause")
    seed = budget_pause.budget_pause_row(tmp_path, task["id"])
    assert seed["pause_generation"] == 2 and seed["source_ref"] is None
    if not seed_origin or legacy == "missing_origin":
        seed["started_at"] = None
        budget_pause.set_budget_pause(tmp_path, task["id"], seed)
    worker = SimpleNamespace(busy_task_id=task["id"])
    meta = {"task": task, "worker_id": 0, "attempt": 1}
    workers.WORKERS[0] = worker
    workers.RUNNING[task["id"]] = meta
    monkeypatch.setattr(worker_health, "_dead_job_is_current", lambda _job: True)
    job = {"worker": worker, "task_id": task["id"], "task": task, "meta": meta,
           "worker_id": 0, "exitcode": 1, "drive_root": str(tmp_path)}
    assert worker_health._complete_exact_budget_pause_after_death(
        job, tmp_path, task, task["id"], 1) == (True, True)
    assert workers.RUNNING == {}
    [task] = workers.PENDING
    row = budget_pause.budget_pause_row(tmp_path, task["id"])
    assert row["pause_source"] == "worker_death_during_pausing"
    assert row["source_ref"] and row["state"] == budget_pause.STATE_PAUSED
    clock.wall += 200.0
    clock.mono += 200.0
    return queue, task, row, clock, origin


@pytest.mark.parametrize("seed_origin", [True, False])
def test_resume_after_source_less_second_pause_keeps_prior_exclusions(tmp_path, monkeypatch, seed_origin):
    queue, task, row, clock, origin = _after_second_pause_crash(
        tmp_path, monkeypatch, seed_origin=seed_origin)
    result = queue.resume_budget_paused_task(task["id"])
    assert result["ok"], result
    assert result["paused_duration_sec"] == 380.0
    assert row["started_at"] == task["_budget_pause_resume"]["started_at"] == origin
    assert row["paused_duration_sec"] == 180.0
    assert row["model_wait_quota_clock"]["elapsed_sec"] == 120.0
    ctx, _limit = _consume(tmp_path, monkeypatch, task)
    assert ctx._budget_paused_sec == ctx.model_wait_context.budget_paused_sec == 380.0
    assert ctx.model_wait_context.executed_seconds() == 480.0
    assert ctx.model_wait_context.execution_window_remaining() == 20.0
    clock.wall += 21.0
    clock.mono += 21.0
    assert ctx.model_wait_context.control_reason() == "absolute_ceiling"


def test_source_less_resume_still_refuses_genuinely_spent_lifetime(tmp_path, monkeypatch):
    queue, task, _row, _clock, _origin = _after_second_pause_crash(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: 480.0)
    result = queue.resume_budget_paused_task(task["id"])
    assert result == {"ok": False, "error": "lifetime_exhausted", "executed_sec": 480.0}
    assert "_budget_pause_resume" not in task


def test_source_less_resume_does_not_extend_calendar_deadline(tmp_path, monkeypatch):
    queue, task, _row, _clock, _origin = _after_second_pause_crash(tmp_path, monkeypatch)
    task["deadline_at"] = "2000-01-01T00:00:00+00:00"
    assert queue.resume_budget_paused_task(task["id"])["error"] == "deadline_passed"
    assert "_budget_pause_resume" not in task


def test_legacy_source_without_clock_does_not_invent_prior_pauses(tmp_path, monkeypatch):
    queue, task, row, _clock, origin = _after_second_pause_crash(
        tmp_path, monkeypatch, legacy="missing_clock")
    result = queue.resume_budget_paused_task(task["id"])
    assert result == {"ok": False, "error": "lifetime_exhausted", "executed_sec": 780.0}
    assert row["started_at"] == origin and row["model_wait_quota_clock"] == {}
    assert not row.get("paused_duration_sec")


def test_legacy_source_without_origin_does_not_deduct_old_exclusions_from_fresh_start(tmp_path, monkeypatch):
    queue, task, row, clock, _origin = _after_second_pause_crash(
        tmp_path, monkeypatch, legacy="missing_origin")
    result = queue.resume_budget_paused_task(task["id"])
    assert result["ok"], result
    assert not row.get("started_at")
    assert row["model_wait_quota_clock"] == {}
    assert not row.get("paused_duration_sec")
    ctx, _limit = _consume(tmp_path, monkeypatch, task)
    assert ctx.task_started_at == clock.wall
    assert ctx.model_wait_context.paused_seconds() == 0.0
    assert ctx.model_wait_context.budget_paused_sec == 200.0  # this observed Pause only
