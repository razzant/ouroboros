"""Residual lifecycle compositions through owner controls and actual assignment."""
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx, _parked
from tests.test_g1_followup_policy import world, register, restore, row, ORIGIN, BINDING  # noqa: F401
from tests.worker_pool_support import affirm_ready



def _worker(workers):
    sent = []
    workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                         in_q=SimpleNamespace(put=lambda task: sent.append(dict(task)))))
    return sent


def test_a_cold_sleep_on_its_queued_child_parks_and_fences_only_its_own_launches(tmp_path, monkeypatch):
    """The final census reads the sleeper's own work: a queued child it waits for is no dependency hold,
    and the parked sleep fences that child's launches no longer; the sleeper's own late launch it still does."""
    from ouroboros import budget_pause, owner_pause
    from ouroboros.task_results import write_task_result
    from tests._budget_pause_exact_helpers import _fast_hold, _quiet_external

    _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    write_task_result(tmp_path, "child", "scheduled", root_task_id="root")
    ctx, limit = _loop_ctx(tmp_path, "root")
    ctx._model_sleep = {"sleep_id": "cold", "mode": "cold", "tasks": ["child"], "senders": []}
    _fast_hold(monkeypatch, budget_pause)
    _quiet_external(monkeypatch, budget_pause)
    try:
        with pytest.raises(budget_pause.BudgetPauseRequested) as parked:
            budget_pause.enter_cold_sleep(limit)
    finally:
        budget_pause.end_dispatch_fence("root")
    assert parked.value.pause["reason"] == "sleep" and parked.value.pause["sleep"]["tasks"] == ["child"]
    with owner_pause.launch_admission(SimpleNamespace(task_id="child", root_task_id="root", drive_root=tmp_path)):
        pass
    with pytest.raises(owner_pause.OwnerPauseRefused, match="model_sleep"):
        with owner_pause.launch_admission(SimpleNamespace(task_id="root", root_task_id="root", drive_root=tmp_path)):
            pass


def test_terminal_root_pause_requires_selected_schedule_restore(world):  # noqa: F811
    from ouroboros import owner_pause
    from ouroboros.task_results import write_task_result
    from supervisor import queue
    from supervisor.worker_assignment import _claim_worker_launch

    _, root, pending = world
    selected, sibling = register(world), register(world)
    fence, _ = owner_pause.install_fence(root, ORIGIN, request_id="before-terminal")
    write_task_result(root, ORIGIN, "completed", result="Finished author")
    queue.check_scheduled_tasks()
    assert pending == []
    held = row(root, selected['id'])['followup_hold']
    assert held['controls']['pause:' + ORIGIN] == fence['fence_id']
    assert restore(root, selected['id'], held['hold_id'])['ok']
    assert owner_pause.read_fence(root, ORIGIN)['state'] != 'released'
    queue.check_scheduled_tasks()
    assert len(pending) == 1 and pending[0]['metadata']['schedule_id'] == selected['id']
    assert row(root, sibling['id'])['followup_hold']
    assert pending[0]['metadata']['billing_group'] == BINDING
    sent = []
    assert _claim_worker_launch(queue, pending[0], SimpleNamespace(in_q=SimpleNamespace(put=sent.append)))
    assert len(sent) == 1


@pytest.mark.parametrize("root_sleep", [False, True])
def test_terminal_root_explicit_child_resume_stays_exact_and_leaves_sibling_held(tmp_path, monkeypatch, root_sleep):
    from ouroboros import budget_pause, owner_pause
    from ouroboros.task_results import write_task_result
    from supervisor.owner_pause_control import request_owner_pause
    from tests.test_batch4_repair_compositions import _running

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 5.0)
    _running(tmp_path, workers)
    if root_sleep:
        from tests.test_model_sleep import _cold_park

        workers.RUNNING.clear()
        _cold_park(tmp_path, monkeypatch, workers, task_id="root", wake_after_sec=3600)
    child, before = _parked(tmp_path, monkeypatch, task_id="child", root_task_id="root")
    sibling, _ = _parked(tmp_path, monkeypatch, task_id="sibling", root_task_id="root")
    write_task_result(tmp_path, "child", "paused", root_task_id="root", parent_task_id="root")
    write_task_result(tmp_path, "sibling", "paused", root_task_id="root", parent_task_id="root")
    assert request_owner_pause("root", request_id="owner")['ok']
    assert queue.resume_budget_paused_task("child")['error'] == 'root_still_paused'
    workers.RUNNING.clear()
    workers.PENDING[:] = [item for item in workers.PENDING if item['id'] != 'root']
    write_task_result(tmp_path, "root", "failed", reason_code="worker_crash_signal")
    assert queue.resume_budget_paused_task("child", selected_by="root")['error'] == 'root_still_paused'
    outcome = queue.resume_budget_paused_task("child")
    assert outcome['ok'], outcome
    sent = _worker(workers)
    workers.assign_tasks()
    assert [item['id'] for item in sent] == ['child']
    assert sibling in workers.PENDING and sibling.get('_budget_pause')
    ctx, limit = _loop_ctx(tmp_path, 'child')
    ctx.root_task_id = 'root'
    ctx.budget_pause_resume = sent[0]['_budget_pause_resume']
    saved = budget_pause.load_budget_pause(ctx)
    assert saved['_pause_row']['source_ref'] == before['source_ref']
    monkeypatch.setattr('ouroboros.owner_wait.restore_continuation_state', lambda *_a, **_kw: None)
    monkeypatch.setattr('ouroboros.owner_wait.rebind_restored_route', lambda *_a, **_kw: (None, 'max'))
    budget_pause.resume_paused_loop(limit.tools, saved, list(limit.messages), {}, {}, set(), budget_remaining_usd=5)
    assert owner_pause.fence_closed(owner_pause.read_fence(tmp_path, 'root'))
    with owner_pause.launch_admission(ctx):
        pass
    assert owner_pause.run_operation(ctx, lambda: "selected effect") == "selected effect"
    with pytest.raises(owner_pause.OwnerPauseRefused):
        with owner_pause.launch_admission(SimpleNamespace(task_id='sibling', root_task_id='root', drive_root=tmp_path)):
            pass
