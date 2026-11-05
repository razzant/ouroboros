"""Residual lifecycle compositions through owner controls and actual assignment."""
from contextlib import contextmanager
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


@pytest.mark.parametrize("checkpoint", ["predispatch", "budget", "sleep"])
def test_pause_overlay_then_selected_resume_reaches_assignment(tmp_path, monkeypatch, checkpoint):
    from ouroboros import owner_pause, budget_pause
    from ouroboros.task_results import write_task_result
    from supervisor.owner_pause_control import request_owner_pause
    from tests.test_model_sleep import _cold_park

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    sent = _worker(workers)
    if checkpoint == "predispatch":
        task = {"id": "root", "root_task_id": "root", "type": "task", "chat_id": 0,
                "admitted_dispatch": "none", "_attempt": 1}
        write_task_result(tmp_path, "root", "scheduled", root_task_id="root")
        workers.PENDING.append(task)
        monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 0.0)
        workers.assign_tasks()
        assert task["_budget_pause"]["scope"] == "global"
    elif checkpoint == "budget":
        task, _ = _parked(tmp_path, monkeypatch, task_id="root", scope="global")
    else:
        _cold_park(tmp_path, monkeypatch, workers, task_id="root", wake_after_sec=3600)
        task = workers.PENDING[0]
    before = budget_pause.budget_pause_row(tmp_path, "root")
    assert request_owner_pause("root", request_id="layered")['ok']
    fence = owner_pause.read_fence(tmp_path, "root")
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 5.0)
    workers.assign_tasks()
    assert sent == [], "money restoration alone never selects the work"
    resumed = queue.resume_budget_paused_task("root")
    assert resumed["ok"], resumed
    workers.assign_tasks()
    assert [item["id"] for item in sent] == ["root"]
    if checkpoint != "predispatch":
        after = budget_pause.budget_pause_row(tmp_path, "root")
        assert after['source_ref'] == before['source_ref'] and after['reason'] == before['reason']
        assert after['grant']['owner_pause_fence_id'] == fence['fence_id']


@pytest.mark.parametrize("selector", ["tasks", "senders"])
def test_selected_queued_child_refuses_cold_and_runs_under_warm(tmp_path, monkeypatch, selector):
    from ouroboros import model_sleep
    from ouroboros.task_results import write_task_result
    from supervisor.worker_assignment import _claim_worker_launch

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    write_task_result(tmp_path, "child", "scheduled", root_task_id="root", parent_task_id="root")
    ctx, _ = _loop_ctx(tmp_path, "root")
    chosen = model_sleep.selectors(ctx, **{selector: ["child"]})
    with pytest.raises(ValueError, match="queued_member child.*sleep warm"):
        model_sleep.request_sleep(ctx, chosen, "cold")
    assert not getattr(ctx, "_model_sleep", None)
    assert model_sleep.request_sleep(ctx, chosen, "warm")["reason"] == "sleep_armed"
    child = {"id": "child", "root_task_id": "root", "parent_task_id": "root", "type": "task",
             "admitted_dispatch": "none", "_attempt": 1, "chat_id": 0}
    workers.PENDING.append(child)
    sent = _worker(workers)
    assert _claim_worker_launch(queue, child, workers.WORKERS[0])
    assert [item["id"] for item in sent] == ["child"]


def test_cold_final_census_rejects_new_queued_dependency_without_leaving_a_fence(tmp_path, monkeypatch):
    from ouroboros import budget_pause, model_sleep, owner_pause
    from ouroboros.task_results import write_task_result

    _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    write_task_result(tmp_path, "child", "scheduled", root_task_id="root")
    ctx, limit = _loop_ctx(tmp_path, "root")
    ctx._model_sleep = {"sleep_id": "cold", "mode": "cold", "tasks": ["child"], "senders": []}
    original = model_sleep.cold_blockers
    checks = []
    def census(source, **kw):
        checks.append(True)
        return [] if len(checks) == 1 else original(source, **kw)
    monkeypatch.setattr(model_sleep, "cold_blockers", census)
    monkeypatch.setattr(budget_pause, "_hold_control_reason", lambda _ctx: "")
    budget_pause.enter_cold_sleep(limit)
    assert budget_pause.budget_pause_row(tmp_path, "root")["abandon_reason"] == "cold_sleep_dependency_blocked"
    assert ctx._model_sleep is None and ctx._budget_pausing is False
    assert "Sleep warm" in limit.messages[-1]["content"]
    with owner_pause.launch_admission(SimpleNamespace(task_id="child", root_task_id="root", drive_root=tmp_path)):
        pass


def test_immediate_launch_refusal_uses_existing_poll_before_retry(monkeypatch):
    import time
    from ouroboros import owner_pause, llm_attempt, loop_round_limits, model_wait, budget_pause

    events = []
    monkeypatch.setattr(llm_attempt, "require_physical_dispatch_window", lambda: events.append("controls"))
    monkeypatch.setattr(time, "sleep", lambda seconds: events.append(("sleep", seconds)))
    @contextmanager
    def gate(*_a, **_kw):
        if len(events) == 1:
            raise owner_pause.LaunchBusy("contention")
        yield
    monkeypatch.setattr(owner_pause, "launch_admission", gate)
    assert loop_round_limits._handle_model_wait_control(SimpleNamespace(),
        model_wait.ModelWaitInterrupted("owner_launch_authority_unavailable")) is None
    assert events == ["controls", ("sleep", budget_pause._HOLD_POLL_SEC), "controls"]


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


def test_live_root_resume_keeps_model_selection_of_owner_paused_child(tmp_path, monkeypatch):
    from ouroboros import budget_pause, owner_pause
    from ouroboros.task_results import write_task_result
    from supervisor.owner_pause_control import request_owner_pause
    from supervisor.events_budget import install_exact_budget_pause
    from tests.test_owner_pause import _owner_park
    from tests.test_batch4_repair_compositions import _running

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 5.0)
    _running(tmp_path, workers)
    write_task_result(tmp_path, "child", "running", root_task_id="root", parent_task_id="root")
    workers.RUNNING['child'] = {'task': {'id': 'child', 'root_task_id': 'root', 'parent_task_id': 'root',
                                         'type': 'task', 'chat_id': 0, '_attempt': 1},
                                'worker_id': 1, 'attempt': 1}
    assert request_owner_pause('root', request_id='tree')['ok']
    park_ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
        WORKERS=workers.WORKERS, sort_pending=lambda: None, persist_queue_snapshot=queue.persist_queue_snapshot,
        bridge=None)
    for task_id in ('child', 'root'):
        pause = _owner_park(tmp_path, monkeypatch, task_id, [], root='root')
        install_exact_budget_pause(park_ctx, task_id, budget_pause.exact_pause_marker(pause)['checkpoint'])
    assert queue.resume_budget_paused_task('root')['ok']
    sent = _worker(workers)
    workers.assign_tasks()
    assert [item['id'] for item in sent] == ['root']
    ctx, limit = _loop_ctx(tmp_path, 'root')
    ctx.budget_pause_resume = sent[0]['_budget_pause_resume']
    saved = budget_pause.load_budget_pause(ctx)
    monkeypatch.setattr('ouroboros.owner_wait.restore_continuation_state', lambda *_a, **_kw: None)
    monkeypatch.setattr('ouroboros.owner_wait.rebind_restored_route', lambda *_a, **_kw: (None, 'max'))
    budget_pause.resume_paused_loop(limit.tools, saved, list(limit.messages), {}, {}, set(), budget_remaining_usd=5)
    assert not owner_pause.fence_closed(owner_pause.read_fence(tmp_path, 'root'))
    outcome = queue.resume_budget_paused_task('child', selected_by='root')
    assert outcome['ok'], outcome
    child = next(item for item in workers.PENDING if item['id'] == 'child')
    from supervisor.worker_assignment import _claim_worker_launch
    assert _claim_worker_launch(queue, child, SimpleNamespace(in_q=SimpleNamespace(put=sent.append)))
    assert [item['id'] for item in sent] == ['root', 'child']
