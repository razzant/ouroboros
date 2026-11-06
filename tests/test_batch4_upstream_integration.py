"""Published Batch2 + Batch4: stop causes through the kill and recovery consumers."""
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
from tests.test_batch4_repair_compositions import _running
from tests.test_processing_transport import transport  # noqa: F401 - fake physical transport
from tests.test_restart_retention import _done_ids, _pool_events
from tests.test_review_operation_lifetime import env  # noqa: F401 - isolated operation fixture
from tests.test_send_clock import ticking  # noqa: F401 - deterministic clock samples



@pytest.mark.parametrize("door,expected", [
    ("crash", "saved_sleep_recovery"), ("restart", "owner_restart_hold"), ("panic", "panic_hold"),
])
@pytest.mark.parametrize("conversion_fails", [False, True])
def test_kill_saved_warm_sleep_keeps_cause_before_and_after_restore(tmp_path, monkeypatch, door, expected,
                                                                  conversion_fails):
    from ouroboros import budget_pause, model_sleep, owner_wait
    from ouroboros.task_results import load_task_result
    from supervisor import restart_retention
    from supervisor.events_budget import budget_hold_fact

    q, _, workers = _install_queue(tmp_path, monkeypatch)
    events = _pool_events(workers, monkeypatch)
    _running(tmp_path, workers)
    ctx, limit = _loop_ctx(tmp_path, "root")
    model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, wake_after_sec=1), "warm")
    checkpoint = owner_wait.checkpoint_owner_wait(ctx, limit.messages, {}, {}, 1, [], set())
    owner_wait.set_owner_wait(tmp_path, "root", {**checkpoint, "state": "waiting"})
    # A fake worker gives the real kill consumer a signal boundary to cross.
    signalled = []
    workers.WORKERS[0] = SimpleNamespace(wid=0, proc=SimpleNamespace(
        pid=None, is_alive=lambda: not signalled, terminate=lambda: signalled.append(True),
        join=lambda **_kw: None))
    read_checkpoint = owner_wait.saved_sleep_checkpoint
    def after_signal(*args, **kwargs):
        assert signalled, "Panic must signal before retention reads or writes"
        return read_checkpoint(*args, **kwargs)
    monkeypatch.setattr(owner_wait, "saved_sleep_checkpoint", after_signal)
    if door == "panic":
        from tests._work_stop_helpers import live_panic
        live_panic(tmp_path, monkeypatch)
        (tmp_path / "state" / "panic_stop.flag").write_text("panic")
    # Restart's explicit door must work even when its flag was not written.
    with monkeypatch.context() as failing:
        if conversion_fails:
            def cannot_publish(*_a, **_kw):
                raise OSError("conversion write failed at kill")
            failing.setattr(budget_pause, "set_budget_pause", cannot_publish)
        assert workers.kill_workers(force=True, archive_service_logs=False,
                                    reconcile_delegate_custody=False, reconcile_review_custody=False,
                                    hold_never_started=door == "restart",
                                    stop_source="owner_restart" if door == "restart" else "")
    assert not _done_ids(events) and not workers.RUNNING
    parked = workers.PENDING[0]
    assert budget_hold_fact(parked)["reason"] == expected
    assert bool(parked.get("_budget_pause")) is not conversion_fails
    assert restart_retention.held_ids(workers.PENDING) == (["root"] if door == "restart" else [])
    workers.PENDING.clear()
    assert q.restore_pending_from_snapshot() == 1
    parked = workers.PENDING[0]
    assert budget_hold_fact(parked)["reason"] == expected
    assert load_task_result(tmp_path, "root")["status"] != "cancelled"
    assert not parked.get("_budget_pause_resume"), "snapshot null is not a Resume grant"
    if door == "panic":
        (tmp_path / "state" / "panic_stop.flag").unlink()
    assert q.resume_budget_paused_task("root")["ok"]
    ctx.budget_pause_resume = parked["_budget_pause_resume"]
    assert budget_pause.load_budget_pause(ctx)["messages"] == limit.messages
