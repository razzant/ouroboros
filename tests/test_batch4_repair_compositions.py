"""Whole-consumer regressions for Pause, Restart and model sleep custody."""
import asyncio
import time
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
from tests.test_restart_retention import _pool_events, _restart_door
from tests._usage_store_testing import ledger_rows

pytestmark = pytest.mark.serial


def _running(root, workers, task_id="root"):
    from ouroboros.task_results import write_task_result
    write_task_result(root, task_id, "running", root_task_id=task_id, chat_id=0)
    task = {"id": task_id, "root_task_id": task_id, "type": "task", "chat_id": 0, "_attempt": 1}
    workers.RUNNING[task_id] = {"task": task, "attempt": 1, "worker_id": 0}
    return task


def test_pause_during_registry_preparation_prevents_handler(tmp_path, monkeypatch):
    from ouroboros import safety
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.task_results import load_task_result
    from supervisor.owner_pause_control import request_owner_pause
    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    order = []
    def preparation(*_a, **_kw):
        assert request_owner_pause("root", request_id="before-handler")["ok"]
        order.append("paused")
        return True, ""
    monkeypatch.setattr(safety, "check_safety", preparation)
    registry.override_handler("knowledge_read", lambda *_a, **_kw: order.append("effect") or "done")
    result = registry.execute_result("knowledge_read", {"topic": "x"})
    assert order == ["paused"]
    assert result.code == "OWNER_PAUSE_NOT_STARTED"
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")


@pytest.mark.parametrize("asynchronous", [False, True])
def test_pause_at_last_model_preparation_prevents_send(tmp_path, monkeypatch, asynchronous):
    from ouroboros import usage_accounting as ua
    from supervisor.owner_pause_control import request_owner_pause
    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    order = []
    capture = ua._record_attempt_capture
    def preparation(reservation, request, status, **kw):
        result = capture(reservation, request, status, **kw)
        if not order:
            assert request_owner_pause("root", request_id="before-send")["ok"]
            order.append("paused")
        return result
    monkeypatch.setattr(ua, "_record_attempt_capture", preparation)
    request = ua.AttemptRequest(model="test", provider="local", drive_root=tmp_path,
                               task_id="root", root_task_id="root", reservation_usd=0)
    def send():
        order.append("sent")
        return {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    async def send_async():
        return send()
    with pytest.raises(Exception):
        if asynchronous:
            asyncio.run(ua.execute_physical_attempt_async(request, send_async))
        else:
            ua.execute_physical_attempt(request, send)
    assert order == ["paused"]
    with ua._locked(tmp_path):
        rows = list({row['attempt_id']: row for row in ledger_rows(tmp_path)}.values())
    assert len(rows) == 1 and rows[0]["state"] == "released"


@pytest.mark.parametrize("topic", ["absent-repair-note", "../invalid"])
def test_real_completed_read_releases_custody_for_continue_and_pause(tmp_path, monkeypatch, topic):
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.continuation_admission import conflicting_writers
    from supervisor.owner_pause_control import request_owner_pause, refresh_owner_pause_tree
    from ouroboros.owner_pause import read_fence
    q, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    result = registry.execute_result("knowledge_read", {"topic": topic, "scope": "global"})
    assert result.code in {"LEGACY_WARNING", "TOOL_ARG_ERROR"}
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")
    assert request_owner_pause("root", request_id="settle-read")["ok"]
    workers.RUNNING.clear()
    refresh_owner_pause_tree("root")
    assert read_fence(tmp_path, "root")["state"] == "paused"
    write_task_result(tmp_path, "root", "failed", reason_code="task_exception")
    assert not conflicting_writers(q, "root")


def test_unstarted_pause_restart_explicit_resume_preserves_identity(tmp_path, monkeypatch):
    from ouroboros.task_results import write_task_result
    from ouroboros.owner_pause import read_fence
    from supervisor.owner_pause_control import request_owner_pause
    from supervisor.events_budget import budget_hold_fact
    q, _, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    write_task_result(tmp_path, "root", "scheduled", root_task_id="root", chat_id=0)
    task = {"id": "root", "root_task_id": "root", "type": "task", "chat_id": 0,
            "_attempt": 1, "admitted_dispatch": "none", "_admission_owner_token": "same-token"}
    workers.PENDING.append(task)
    assert request_owner_pause("root", request_id="pause-unstarted")["ok"]
    _restart_door(tmp_path, monkeypatch, workers)
    # Owner 2026-10-08 (quiz d2f7532b): the owner's Restart names its never-started row in
    # its restart transaction (returned after the acknowledged exit) instead of holding it.
    import json

    from ouroboros import delegate_recovery as dr

    active = json.loads(dr._active_restart_transaction_path(tmp_path).read_text(encoding="utf-8"))
    assert dr._read_restart_transaction(tmp_path, active["transaction_id"])["queue_ids"] == ["root"]
    assert budget_hold_fact(task) is None
    result = q.resume_budget_paused_task("root")
    assert result["ok"], result
    assert workers.PENDING == [task] and task["_admission_owner_token"] == "same-token"
    assert read_fence(tmp_path, "root")["state"] == "released"
    assert not budget_hold_fact(task) and "root" not in q.BUDGET_ROOT_FENCES


def test_unknown_root_claim_blocks_real_cold_request(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.tool_result import ToolResult
    from ouroboros.model_sleep import request_sleep
    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    def body(ctx, **_kw):
        from ouroboros.tools.control_events import _emit_and_wait_for_routing
        ctx.event_queue = None
        _, receipt = _emit_and_wait_for_routing(ctx, {
            "type": "promote_chat_to_task", "task_id": "pending-root", "routing_token": "pending-token",
            "client_message_id": "pending-message", "objective": "work"})
        assert receipt["status"] == "unconfirmed"
        return ToolResult(status="ok", code="LEGACY_UNTYPED", text="remote accepted",
                          meta={"dynamic_provider": True, "operation_outcome": "completed"})
    registry.override_handler("knowledge_read", body)
    registry.execute_result("knowledge_read", {"topic": "x"})
    registry._ctx.task_attempt = 1
    with pytest.raises(ValueError, match="member_custody"):
        request_sleep(registry._ctx, {"senders": [], "tasks": [], "runs": [], "wake_at": ""}, "cold")


def test_warm_sleep_checkpoint_survives_restart_then_explicit_resume(tmp_path, monkeypatch):
    from ouroboros.owner_wait import checkpoint_owner_wait, set_owner_wait
    from ouroboros import model_sleep, server_restart, budget_pause
    from supervisor.restart_retention import pause_retention
    from ouroboros.task_results import load_task_result
    from supervisor.events_budget import budget_hold_fact
    q, _, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _running(tmp_path, workers)
    ctx, limit = _loop_ctx(tmp_path, "root")
    model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, wake_after_sec=3600), "warm")
    model_sleep.begin(ctx)
    checkpoint = checkpoint_owner_wait(ctx, limit.messages, {}, {}, 1, [], set())
    set_owner_wait(tmp_path, "root", {**checkpoint, "state": "waiting"})
    monkeypatch.setattr(server_restart, "DATA_DIR", tmp_path)
    assert pause_retention(tmp_path, "root", 1)
    assert "root" not in server_restart._owned_live_task_ids(SimpleNamespace(RUNNING=workers.RUNNING))
    _restart_door(tmp_path, monkeypatch, workers)
    assert load_task_result(tmp_path, "root")["status"] != "cancelled"
    task = workers.PENDING[0]
    assert budget_hold_fact(task)["reason"] == "owner_restart_hold"
    assert q.resume_budget_paused_task("root")["ok"]
    ctx.budget_pause_resume = task["_budget_pause_resume"]
    restored = budget_pause.load_budget_pause(ctx)
    assert restored["messages"] == limit.messages
    assert restored["_pause_row"]["sleep"]["sleep_id"] == checkpoint["sleep"]["sleep_id"]


@pytest.mark.parametrize("lost_event", [False, True, "warm_first"])
def test_cold_consumption_propagates_retry_time_to_all_clocks(tmp_path, monkeypatch, lost_event):
    from tests.test_model_sleep import _cold_park
    from tests.test_budget_pause_holds import _idle_worker
    from ouroboros import budget_pause, owner_wait
    from ouroboros.model_wait import TaskModelWait, execution_elapsed_seconds
    clock = [time.time()]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    monkeypatch.setattr(time, "monotonic", lambda: clock[0] - 1_000_000)
    q, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    _cold_park(tmp_path, monkeypatch, workers, wake_after_sec=3600)
    task = workers.PENDING[0]
    task["deadline_at"] = "2099-01-01T00:00:00+00:00"
    clock[0] += 1000
    assert q.resume_budget_paused_task("sleeper")["ok"]
    clock[0] += 120
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert len(sent) == 1
    ctx, _ = _loop_ctx(tmp_path, "sleeper")
    ctx.budget_pause_resume = sent[0]["_budget_pause_resume"]
    saved = budget_pause.load_budget_pause(ctx)
    ctx.task_started_at = ctx.budget_pause_resume["started_at"]
    waiter = TaskModelWait(task=sent[0], drive_root=tmp_path, event_queue=None, worker_slot_held=True)
    ctx.model_wait_context = waiter
    waiter.tool_context = ctx
    waiter.restore_continuation(saved.get("model_wait") or {}, started_at=ctx.task_started_at,
                                budget_paused_sec=ctx.budget_pause_resume["paused_duration_sec"])
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_k: (None, "max"))
    real_set = budget_pause.set_budget_pause
    attempts = []
    def fail_first(root, task_id, pause, **kw):
        if pause.get("state") == budget_pause.STATE_RESUMED:
            attempts.append(clock[0])
            if len(attempts) == 1:
                raise OSError("disk-full")
        return real_set(root, task_id, pause, **kw)
    monkeypatch.setattr(budget_pause, "set_budget_pause", fail_first)
    monkeypatch.setattr(time, "sleep", lambda _: clock.__setitem__(0, clock[0] + 30))
    events = []
    ctx.event_queue = SimpleNamespace(put=lambda event: events.append(event))
    budget_pause.resume_paused_loop(SimpleNamespace(_ctx=ctx), saved, [], {}, {}, set(), budget_remaining_usd=5)
    from supervisor.events_budget import _handle_budget_pause
    consumed_event = next(event for event in events if event.get("phase") == "consumed")
    for wrong in ({"pause_id": "other"}, {"grant_id": "other"}, {"task_attempt": 99}):
        _handle_budget_pause({**consumed_event, **wrong}, workers)
        assert workers.RUNNING["sleeper"]["sleep_parked_at"]
        assert workers.RUNNING["sleeper"]["budget_paused_sec"] == 1000
    if lost_event == "warm_first":
        pass  # the real warm park below must reconcile before installing its interval
    elif lost_event:
        from supervisor.worker_assignment import _tick_parked_work
        _tick_parked_work(q)
    else:
        for event in events:
            if event.get("type") == "budget_pause":
                _handle_budget_pause(event, workers)
                _handle_budget_pause(event, workers)  # duplicate folds nothing twice
    row = budget_pause.budget_pause_row(tmp_path, "sleeper")
    assert row["grant"]["consumed_at"] == attempts[-1]
    assert ctx._budget_paused_sec == waiter.budget_paused_sec == 1150
    if lost_event != "warm_first":
        assert workers.RUNNING["sleeper"]["budget_paused_sec"] == 1150
    assert waiter.executed_seconds() == execution_elapsed_seconds(workers.RUNNING["sleeper"], clock[0]) == 100
    assert sent[0]["deadline_at"] == "2099-01-01T00:00:00+00:00"

    # A second, warm sleep in the SAME attempt owns a new interval. Neither an
    # old notification nor the lost-event repair tick may consume that interval.
    from supervisor.worker_owner_wait import handle_owner_wait, _grant_resume
    from supervisor.worker_assignment import _tick_parked_work
    meta = workers.RUNNING["sleeper"]
    worker = workers.WORKERS[meta["worker_id"]]
    worker.proc = SimpleNamespace(pid=77, is_alive=lambda: True)
    worker.reaping = False
    checkpoint = {"wait_id": "warm-after-cold", "task_attempt": meta["attempt"],
                  "source_ref": {"x": 1}, "reason": "sleep", "sleep": {"senders": ["peer"]}}
    clock[0] += 20
    handle_owner_wait({"task_id": "sleeper", "wait_id": checkpoint["wait_id"],
        "task_attempt": meta["attempt"], "worker_id": worker.wid, "pid": 77,
        "phase": "park", "checkpoint": checkpoint}, workers)
    warm_started = clock[0]
    clock[0] += 40
    _tick_parked_work(q)
    _handle_budget_pause(consumed_event, workers)
    assert meta["sleep_parked_at"] == warm_started
    assert meta["budget_paused_sec"] == 1150
    assert execution_elapsed_seconds(meta, clock[0]) == 120
    assert _grant_resume("sleeper", meta, worker)
    assert meta["budget_paused_sec"] == 1190
    _handle_budget_pause(consumed_event, workers)
    _tick_parked_work(q)
    assert meta["budget_paused_sec"] == 1190 and "sleep_parked_at" not in meta
    assert execution_elapsed_seconds(meta, clock[0]) == 120
