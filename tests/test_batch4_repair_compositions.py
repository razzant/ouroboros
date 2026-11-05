"""Whole-consumer regressions for Pause, Restart and model sleep custody."""
from types import SimpleNamespace


from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
from tests.test_restart_retention import _pool_events, _restart_door



def _running(root, workers, task_id="root"):
    from ouroboros.task_results import write_task_result
    write_task_result(root, task_id, "running", root_task_id=task_id, chat_id=0)
    task = {"id": task_id, "root_task_id": task_id, "type": "task", "chat_id": 0, "_attempt": 1}
    workers.RUNNING[task_id] = {"task": task, "attempt": 1, "worker_id": 0}
    return task


def _next_boot_after_owner_restart(root, monkeypatch, queue, *, refusal="selection_owner_held"):
    import threading
    from ouroboros import delegate_recovery, server_restart
    from supervisor.worker_chat_lane import auto_resume_after_restart

    assert queue.resume_budget_paused_task("root")["error"] == refusal
    monkeypatch.setattr(server_restart, "_OWNER_STOP_BEGAN", threading.Event())
    monkeypatch.setattr(server_restart, "_owner_restart_requested", threading.Event())
    monkeypatch.setattr(delegate_recovery, "_PROCESS_GENERATION", "next-test-generation")
    auto_resume_after_restart()
    assert delegate_recovery.active_restart_transaction(root)["door_applied_at"]


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
    _next_boot_after_owner_restart(tmp_path, monkeypatch, q, refusal="restart_no_resume")
    assert q.resume_budget_paused_task("root")["ok"]
    ctx.budget_pause_resume = task["_budget_pause_resume"]
    restored = budget_pause.load_budget_pause(ctx)
    assert restored["messages"] == limit.messages
    assert restored["_pause_row"]["sleep"]["sleep_id"] == checkpoint["sleep"]["sleep_id"]
