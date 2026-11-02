"""Final handoff mirrors and narrow sleep diagnostic/cause regressions."""
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
from tests.test_batch4_repair_compositions import _running
from tests.test_budget_pause_holds import _idle_worker
from tests._store_shared import unreadable_result

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("refusal", ["pause", "snapshot", "none", "newer", "same", "newer_scheduled"])
def test_real_assignment_mirror_obeys_final_handoff(tmp_path, monkeypatch, refusal):
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.owner_pause_control import request_owner_pause

    q, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 5)
    write_task_result(tmp_path, "root", "scheduled", root_task_id="root")
    task = {"id": "root", "root_task_id": "root", "type": "task", "chat_id": 0,
            "drive_root": str(tmp_path), "admitted_dispatch": "none", "_attempt": 1}
    workers.PENDING.append(task)
    observed = []
    class Queue:
        def append(self, candidate):
            observed.append(load_task_result(tmp_path, "root"))
    _idle_worker(workers, Queue())
    def prepare(_candidate):
        if refusal == "pause":
            assert request_owner_pause("root", request_id="at-final-gate")["ok"]
        if refusal in {"newer", "same", "newer_scheduled"}:
            write_task_result(tmp_path, "root", "scheduled" if refusal == "newer_scheduled" else "running",
                              task_attempt=1 if refusal == "same" else 2,
                              result="newer worker facts", description="newer description")
        return ""
    monkeypatch.setattr(workers, "_evolution_assignment_error", prepare)
    persist = q.persist_queue_snapshot
    monkeypatch.setattr(q, "persist_queue_snapshot", lambda **kw: False if
        refusal == "snapshot" and kw.get("reason") == "worker_launch_claimed" else persist(**kw))
    workers.assign_tasks()
    stored = load_task_result(tmp_path, "root")
    if refusal in {"pause", "snapshot"}:
        assert not observed and workers.PENDING == [task]
        assert stored["status"] == "scheduled"
    elif refusal in {"newer", "same", "newer_scheduled"}:
        assert observed[0]["task_attempt"] == (1 if refusal == "same" else 2)
        assert stored["result"] == "newer worker facts"
        assert stored["description"] == "newer description"
    else:
        assert len(observed) == 1 and observed[0]["status"] == "running"


def test_child_writer_sleep_hold_is_typed_and_retries_after_settlement(tmp_path, monkeypatch):
    from ouroboros import budget_pause, model_sleep
    from ouroboros.task_results import write_task_result

    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    write_task_result(tmp_path, "child", "scheduled", root_task_id="root")
    ctx, limit = _loop_ctx(tmp_path, "root")
    ctx._model_sleep = {"sleep_id": "s", "mode": "cold", **model_sleep.selectors(ctx, wake_after_sec=3600)}
    original, census, holds = model_sleep.cold_blockers, [], []
    def initial_census(source, **kwargs):
        blockers = original(source, **kwargs)
        census.append(blockers)
        if len(census) == 1:
            write_task_result(tmp_path, "child", "scheduled", launch_handoffs={"external": {"tool": "opaque"}})
        return blockers
    def settle(_seconds):
        holds.append(dict(limit.accumulated_usage["budget_pause_hold"]))
        assert len(holds) == 1, "settled writer must allow the existing retry"
        write_task_result(tmp_path, "child", "scheduled", launch_handoffs={})
    monkeypatch.setattr(model_sleep, "cold_blockers", initial_census)
    monkeypatch.setattr(budget_pause.time, "sleep", settle)
    monkeypatch.setattr(budget_pause, "_hold_control_reason", lambda _ctx: "")
    try:
        with pytest.raises(budget_pause.BudgetPauseRequested):
            budget_pause.enter_cold_sleep(limit)
    finally:
        budget_pause.end_dispatch_fence("root")
    assert holds[0]["hold_reason"] == "cold_sleep_writers_unsettled"
    assert "budget_pause_hold" not in limit.accumulated_usage
    assert budget_pause.budget_pause_row(tmp_path, "root")["source_ref"]


def test_malformed_member_sleep_authority_is_typed_and_not_owner_intent(tmp_path, monkeypatch):
    from ouroboros import owner_pause
    from ouroboros.task_results import write_task_result

    _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    write_task_result(tmp_path, "child", "running", root_task_id="root")
    unreadable_result(monkeypatch, "child")
    source = SimpleNamespace(task_id="child", root_task_id="root", budget_drive_root=tmp_path)
    with pytest.raises(owner_pause.OwnerPauseRefused, match="model_sleep_authority_unreadable"):
        with owner_pause.launch_admission(source):
            pytest.fail("unreadable authority must not dispatch")
    assert not owner_pause.read_fence(tmp_path, "root")
    from ouroboros.tools.registry import ToolRegistry
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id, registry._ctx.root_task_id = "child", "root"
    result = registry.execute_result("knowledge_list", {})
    assert result.meta["control_reason"] == "model_sleep_authority_unreadable"
    # Pause and sleep share one unreadable-authority refusal: the typed reason names the
    # sleep authority, and the text never attributes the refusal to an owner Pause.
    assert "authority could not be read" in result.text
    assert "(model_sleep_authority_unreadable)" in result.text and "the owner paused" not in result.text
    assert not owner_pause.read_fence(tmp_path, "root")


@pytest.mark.parametrize("cause,expected", [
    ("crash", "saved_sleep_recovery"), ("restart", "owner_restart_hold"), ("panic", "panic_hold"),
])
@pytest.mark.parametrize("conversion_fails", [False, True])
def test_warm_snapshot_recovery_uses_actual_stop_cause(tmp_path, monkeypatch, cause, expected, conversion_fails):
    from ouroboros import budget_pause, model_sleep, owner_wait
    from ouroboros.task_results import load_task_result
    from supervisor.events_budget import budget_hold_fact
    from supervisor.sleep_wake import wake_ready_sleepers

    q, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    ctx, limit = _loop_ctx(tmp_path, "root")
    model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, wake_after_sec=1), "warm")
    checkpoint = owner_wait.checkpoint_owner_wait(ctx, limit.messages, {}, {}, 1, [], set())
    owner_wait.set_owner_wait(tmp_path, "root", {**checkpoint, "state": "waiting"})
    assert q.persist_queue_snapshot(reason="before-crash")
    if cause != "crash":
        flag = tmp_path / "state" / ("owner_restart_no_resume.flag" if cause == "restart" else "panic_stop.flag")
        flag.write_text("owner_restart_no_resume" if cause == "restart" else "panic")
    workers.RUNNING.clear()
    with monkeypatch.context() as failing:
        if conversion_fails:
            def cannot_publish(*_a, **_kw):
                raise OSError("conversion write failed")
            failing.setattr(budget_pause, "set_budget_pause", cannot_publish)
        assert q.restore_pending_from_snapshot() == 1
    task = workers.PENDING[0]
    assert budget_hold_fact(task)["reason"] == expected
    assert load_task_result(tmp_path, "root")["status"] != "cancelled"
    from ouroboros import deadline_utils
    wake_at = deadline_utils.parse_deadline_ts(checkpoint["sleep"]["wake_at"])
    with monkeypatch.context() as ready:
        ready.setattr(deadline_utils, "utc_now", lambda: wake_at)
        outcomes = wake_ready_sleepers(q)
    if not conversion_fails:
        assert outcomes[0]["error"] == "sleep_wake_vetoed" and outcomes[0]["veto"] == expected
    assert "_budget_pause_resume" not in task
    if cause == "restart":
        from ouroboros.delegate_recovery import consume_owner_restart, pending_owner_restart
        consume_owner_restart(tmp_path, pending_owner_restart(tmp_path))
    elif cause == "panic":
        flag.unlink()
    assert q.resume_budget_paused_task("root")["ok"]
    ctx.budget_pause_resume = task["_budget_pause_resume"]
    assert budget_pause.load_budget_pause(ctx)["messages"] == limit.messages
