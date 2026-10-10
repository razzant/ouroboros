"""Owner Batch4 (6B): the model's own warm/cold sleep, through its REAL consumers.

``await_messages`` (its default in-slot wait untouched), the owner-wait park
loops, the execution clocks, the supervisor park/grant, the cold park at the
round boundary, the sleep-wake policy with its vetoes, and the Restart/Panic
holds are driven directly.
"""

from __future__ import annotations

import datetime
import json
import time
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _fast_hold, _install_queue, _loop_ctx, _quiet_external


def _ctx(tmp_path, task_id="sleeper", *, root="sleeper"):
    from ouroboros.model_wait import TaskModelWait

    ctx = SimpleNamespace(task_id=task_id, root_task_id=root, drive_root=tmp_path, budget_drive_root=tmp_path,
                          task_attempt=1, _loop_mailbox_seen_ids=set(), task_metadata={},
                          owner_wait_callback=lambda *_a, **_k: "unknown", pending_events=[],
                          event_queue=SimpleNamespace(put=lambda _e: None), current_chat_id=0,
                          _budget_paused_sec=0.0)
    ctx.model_wait_context = TaskModelWait(task={"id": task_id, "root_task_id": root,
                                                 "budget_drive_root": str(tmp_path)},
                                           drive_root=tmp_path, event_queue=None, worker_slot_held=False)
    ctx.model_wait_context.tool_context = ctx
    return ctx


def _result(tmp_path, task_id, status="running", **fields):
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, task_id, status, chat_id=0, **fields)


def _mail(tmp_path, task_id, text, *, sender="", kind="task_message", msg_id=None):
    from ouroboros.owner_mailbox import write_owner_message

    assert write_owner_message(tmp_path, text, task_id, msg_id=msg_id or f"m-{time.time_ns()}", kind=kind)
    if sender:  # write_owner_message has no sender field: stamp it as the task-message writers do
        from ouroboros.owner_mailbox import _mailbox_path

        path = _mailbox_path(tmp_path, task_id)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]["source_task_id"] = sender
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_warm_is_default_and_in_slot_is_an_explicit_choice(tmp_path):
    from ouroboros.tools.control_task_results import _await_messages

    ctx = _ctx(tmp_path)
    assert "no live children or selected source" in _await_messages(ctx)
    _result(tmp_path, "live-child", parent_task_id="sleeper", root_task_id="sleeper", delegation_role="subagent")
    assert json.loads(_await_messages(ctx))["mode"] == "warm"
    out = json.loads(_await_messages(ctx, 1, mode="in_slot"))
    assert out["reason"] == "timeout" and out["slot"] == "held"
    refused = _await_messages(ctx, 1, mode="in_slot", senders=["x"])
    assert "give mode warm or cold" in refused


def test_selectors_are_validated_and_a_ready_source_answers_at_once(tmp_path):
    from ouroboros import model_sleep
    from ouroboros.tools.control_task_results import _await_messages

    _result(tmp_path, "sleeper")
    _result(tmp_path, "peer-a")
    _result(tmp_path, "child-1")
    ctx = _ctx(tmp_path)
    assert "task ghost is unknown" in _await_messages(ctx, mode="warm", tasks=["ghost"])
    assert "not delegated runs of this task" in _await_messages(ctx, mode="warm", runs=["run-x"])
    assert "not both" in _await_messages(ctx, mode="warm", wake_at="2099-01-01T00:00:00+00:00", wake_after_sec=5)
    chosen = model_sleep.selectors(ctx, senders=["peer-a"], wake_after_sec=60)
    assert chosen["any_mail"] is False and chosen["wake_at"]
    # Already-terminal selected task: answered at once, nothing armed.
    _result(tmp_path, "child-1", "completed")
    ready = json.loads(_await_messages(ctx, mode="warm", tasks=["child-1"]))
    assert ready == {"reason": "ready", "woke_by": "task:child-1:completed", "slept": False, "mode": "warm"}
    assert not getattr(ctx, "_model_sleep", None)
    # A passed wake time and pending owner words also answer at once.
    past = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=5)).isoformat()
    assert json.loads(_await_messages(ctx, mode="warm", wake_at=past))["woke_by"] == "timeout"
    _mail(tmp_path, "sleeper", "please stop and summarise", kind="owner_text")
    assert json.loads(_await_messages(ctx, mode="warm", senders=["peer-a"]))["woke_by"] == "owner_text"


def test_a_warm_sleep_wakes_only_on_its_selected_source_and_never_loses_one_landing_before_the_park(tmp_path):
    """Check → install → recheck: selected mail written after the tool's check but
    before the park wakes the very first poll; unselected mail never wakes it."""
    from ouroboros.owner_wait import checkpoint_owner_wait, direct_owner_wait
    from ouroboros.tools.control_task_results import _await_messages

    _result(tmp_path, "sleeper")
    _result(tmp_path, "peer-a")
    _result(tmp_path, "peer-b")
    ctx = _ctx(tmp_path)
    armed = json.loads(_await_messages(ctx, mode="warm", senders=["peer-a"]))
    assert armed["reason"] == "sleep_armed" and ctx._owner_wait_requested.startswith("sleep:")
    checkpoint = checkpoint_owner_wait(ctx, [{"role": "user", "content": "x"}], {}, {}, 1, [], set())
    assert checkpoint["reason"] == "sleep" and checkpoint["quiz_id"] == ""
    assert checkpoint["sleep"]["senders"] == ["peer-a"]
    _mail(tmp_path, "sleeper", "noise", sender="peer-b")          # unselected: stays unread, no wake
    _mail(tmp_path, "sleeper", "tests done", sender="peer-a")     # lands before the park
    assert direct_owner_wait(ctx, checkpoint) == "mail:peer-a"
    from ouroboros.owner_mailbox import OwnerMailboxPeek

    assert OwnerMailboxPeek().pending(tmp_path, "sleeper", set(), 1), "nothing was acknowledged by the wake"


def test_an_owner_pause_or_stop_control_is_never_filtered(tmp_path):
    from ouroboros import model_sleep
    from ouroboros.owner_mailbox import KIND_OWNER_PAUSE

    _result(tmp_path, "sleeper")
    _result(tmp_path, "peer-a")
    ctx = _ctx(tmp_path)
    chosen = model_sleep.selectors(ctx, senders=["peer-a"])
    assert model_sleep.wake_reason(ctx, chosen) == ""
    _mail(tmp_path, "sleeper", "owner_pause", kind=KIND_OWNER_PAUSE)
    assert model_sleep.wake_reason(ctx, chosen) == "control:owner_pause"


def test_sleep_is_excluded_from_execution_once_and_the_deadline_does_not_move(tmp_path):
    from ouroboros import model_sleep
    from ouroboros.model_wait import execution_elapsed_seconds

    ctx = _ctx(tmp_path)
    waiter = ctx.model_wait_context
    waiter.started_monotonic -= 150.0                       # 150 s of wall time: 100 s working...
    model_sleep.begin(ctx)
    waiter.sleep_started_monotonic -= 50.0                  # ...then the last 50 s asleep
    ctx._model_sleep_started -= 50.0
    assert 99.0 <= waiter.executed_seconds() <= 101.0
    slept = model_sleep.end(ctx)
    assert 49.0 <= slept <= 51.0 and 49.0 <= waiter.budget_paused_sec <= 51.0
    assert 99.0 <= waiter.executed_seconds() <= 101.0, "folded once, never subtracted twice"
    # The supervisor's row: live exclusion while parked, folded at the grant.
    now = time.time()
    meta = {"started_at": now - 300, "sleep_parked_at": now - 120}
    assert 179.0 <= execution_elapsed_seconds(meta, now) <= 181.0


def test_the_pooled_park_excludes_the_sleep_live_and_the_grant_folds_it(tmp_path, monkeypatch):
    from supervisor import worker_owner_wait
    from supervisor.worker_owner_wait import handle_owner_wait

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _result(tmp_path, "sleeper")
    proc = SimpleNamespace(pid=77, is_alive=lambda: True)
    commands: list = []
    worker = SimpleNamespace(wid=0, proc=proc, busy_task_id="sleeper", reaping=False, active_capacity=True,
                             in_q=SimpleNamespace(put=commands.append))
    workers.WORKERS[0] = worker
    meta = {"task": {"id": "sleeper"}, "worker_id": 0, "attempt": 1, "started_at": time.time() - 10}
    workers.RUNNING["sleeper"] = meta
    ctx = SimpleNamespace(RUNNING=workers.RUNNING, WORKERS=workers.WORKERS, DRIVE_ROOT=tmp_path)
    checkpoint = {"wait_id": "w1", "task_attempt": 1, "source_ref": {"x": 1}, "reason": "sleep",
                  "sleep": {"senders": ["peer-a"], "any_mail": False}}
    handle_owner_wait({"task_id": "sleeper", "wait_id": "w1", "task_attempt": 1, "worker_id": 0, "pid": 77,
                       "phase": "park", "checkpoint": checkpoint}, ctx)
    assert isinstance(meta.get("sleep_parked_at"), float) and worker.active_capacity is False
    meta["sleep_parked_at"] -= 30.0
    meta["owner_wait_resume_requested"] = True
    monkeypatch.setattr(worker_owner_wait, "_resume_allowed", lambda *_a: True)
    assert worker_owner_wait._grant_resume("sleeper", meta, worker) is True
    assert "sleep_parked_at" not in meta and 29.0 <= meta["budget_paused_sec"] <= 32.0


def test_a_cold_sleep_is_refused_over_its_own_live_writers(tmp_path, monkeypatch):
    from ouroboros import model_sleep
    from ouroboros.tools import services

    _result(tmp_path, "sleeper")
    ctx = _ctx(tmp_path)
    record = SimpleNamespace(task_id="sleeper", name="dev-server", proc=SimpleNamespace(poll=lambda: None))
    monkeypatch.setitem(services._SERVICES, "sleeper:dev-server", record)
    with pytest.raises(ValueError, match="service dev-server"):
        model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, wake_after_sec=60), model_sleep.MODE_COLD)
    services._SERVICES.pop("sleeper:dev-server", None)
    armed = model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, wake_after_sec=60), model_sleep.MODE_COLD)
    assert armed["reason"] == "sleep_armed" and ctx._model_sleep["mode"] == "cold"


def _cold_park(tmp_path, monkeypatch, workers, task_id="sleeper", **selected):
    from ouroboros import budget_pause, model_sleep
    from supervisor.events_budget import install_exact_budget_pause

    _result(tmp_path, task_id)
    ctx, limit_ctx = _loop_ctx(tmp_path, task_id)
    _fast_hold(monkeypatch, budget_pause)
    _quiet_external(monkeypatch, budget_pause)
    monkeypatch.setattr(model_sleep, "cold_blockers", lambda _ctx, **_kw: [])
    ctx._model_sleep = {"sleep_id": "s1", "mode": "cold", **model_sleep.selectors(ctx, **selected)}
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.enter_cold_sleep(limit_ctx)
    budget_pause.end_dispatch_fence(task_id)
    row = raised.value.pause
    workers.RUNNING[task_id] = {"task": {"id": task_id, "type": "task", "chat_id": 0, "root_task_id": task_id},
                                "worker_id": 0, "attempt": 1}
    sup = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                          WORKERS=workers.WORKERS, sort_pending=lambda: None,
                          persist_queue_snapshot=lambda reason="": True, bridge=None)
    install_exact_budget_pause(sup, task_id, budget_pause.exact_pause_marker(row)["checkpoint"])
    return row


def test_a_cold_sleep_parks_exactly_and_only_its_own_readiness_wakes_it(tmp_path, monkeypatch):
    from ouroboros.budget_pause import budget_pause_row
    from supervisor.sleep_wake import wake_ready_sleepers

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    _result(tmp_path, "peer-a")
    row = _cold_park(tmp_path, monkeypatch, workers, senders=["peer-a"])
    assert row["reason"] == "sleep" and row["sleep"]["senders"] == ["peer-a"] and "sleep_seen" in row
    parked = workers.PENDING[0]
    assert parked["_budget_pause"]["reason"] == "sleep" and parked["_budget_pause"]["scope"] == "task"
    assert wake_ready_sleepers(queue) == [], "nothing selected has happened yet"
    _mail(tmp_path, "sleeper", "unrelated", sender="someone-else")
    assert wake_ready_sleepers(queue) == []
    _mail(tmp_path, "sleeper", "done", sender="peer-a")
    outcome = wake_ready_sleepers(queue)
    assert outcome[0]["ok"] is True and outcome[0]["ready"]["reason"] == "mail:peer-a"
    handoff = workers.PENDING[0]["_budget_pause_resume"]
    assert handoff["selected_by"] == "sleep_wake"
    assert budget_pause_row(tmp_path, "sleeper")["sleep_ready"]["reason"] == "mail:peer-a"


def test_readiness_is_recorded_but_never_overrides_an_owner_pause_or_a_restart_hold(tmp_path, monkeypatch):
    from ouroboros import owner_pause
    from ouroboros.budget_pause import budget_pause_row
    from supervisor.sleep_wake import wake_ready_sleepers

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    past = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=1)).isoformat()
    _cold_park(tmp_path, monkeypatch, workers, wake_at=past)
    time.sleep(1.2)
    owner_pause.install_fence(tmp_path, "sleeper", request_id="p")
    vetoed = wake_ready_sleepers(queue)
    assert vetoed[0]["error"] == "sleep_wake_vetoed" and vetoed[0]["veto"] == "owner_pause"
    assert budget_pause_row(tmp_path, "sleeper")["sleep_ready"]["reason"] == "timeout", "readiness kept"
    owner_pause.release_fence(tmp_path, "sleeper", reason="test")
    # The owner's Restart now holds it: the kept readiness still does not wake it.
    workers.kill_workers(force=True, terminal_status="cancelled", result_reason="Owner restart",
                         hold_never_started=True)
    held = wake_ready_sleepers(queue)
    assert held[0]["veto"] == "owner_restart_hold"
    assert workers.PENDING[0]["_budget_pause"]["reason"] == "sleep", "the saved sleep itself is retained"


def test_a_panic_boot_holds_a_cold_sleeper_before_its_marker_is_consumed(tmp_path, monkeypatch):
    from supervisor.events_budget import HOLD_PANIC, budget_hold_fact

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _cold_park(tmp_path, monkeypatch, workers, wake_after_sec=1)
    queue.persist_queue_snapshot(reason="before_panic")
    workers.PENDING[:] = []
    (tmp_path / "state" / "panic_stop.flag").write_text("panic", encoding="utf-8")
    assert queue.restore_pending_from_snapshot() == 1
    assert budget_hold_fact(workers.PENDING[0])["reason"] == HOLD_PANIC
