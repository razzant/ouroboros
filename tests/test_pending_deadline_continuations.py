"""Real continuation producers must expire in queue without another worker start."""
from __future__ import annotations

import queue as stdqueue
from types import SimpleNamespace

import pytest

from ouroboros.task_results import load_task_result, write_task_result
from supervisor import queue, workers
from tests._budget_pause_exact_helpers import _parked
from tests._store_shared import unreadable_result
from tests.test_swarm_host_admission import host  # noqa: F401
from tests.test_owner_wait_restart import restart_case  # noqa: F401
from tests.worker_pool_support import affirm_ready



@pytest.fixture
def expiry(monkeypatch):
    from supervisor import queue_timeouts

    jobs, frames = stdqueue.Queue(), []
    monkeypatch.setattr(queue, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(queue, "_reap_queue", jobs)
    monkeypatch.setattr(queue_timeouts, "_pending_deadline_jobs", {})
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=frames.append))
    monkeypatch.setattr(queue, "load_state", lambda: {})
    monkeypatch.setattr(workers, "load_state", lambda: {})
    monkeypatch.setattr("supervisor.state.budget_remaining", lambda *_a, **_k: 5)
    monkeypatch.setattr("supervisor.worker_owner_wait.grant_warm_returns", lambda: None)

    def reap():
        while not jobs.empty():
            job = jobs.get_nowait()
            assert job["kind"] == "pending_deadline"
            queue_timeouts._reap_pending_deadline(job)
            jobs.task_done()

    def mark(task):
        task["deadline_at"] = "2000-01-01T00:00:00Z"
        queue.enforce_task_timeouts()
        assert task["_terminalization_retry"]["trigger"] == "deadline"
        assert jobs.qsize() == 1

    return SimpleNamespace(jobs=jobs, frames=frames, reap=reap, mark=mark)


def _exact(host, monkeypatch, *, recovered=False):  # noqa: F811
    from tests.test_project_hold_recovery import accepted, worker

    admitted = accepted(host, host.root.parent, tid="exact")
    if recovered:
        from ouroboros import budget_pause, working_checkpoint
        from supervisor.events_budget import install_exact_budget_pause
        from tests._budget_pause_exact_helpers import _loop_ctx, _fast_hold, _quiet_external
        from tests.test_retained_work_recovery import _saved

        _saved(host.root, admitted["id"], 1)
        admitted.update(_attempt=2, _working_recovery=working_checkpoint.prepare_recovery(
            host.root, admitted["id"], from_attempt=1, cause="worker_crash"))
        write_task_result(host.root, admitted["id"], "interrupted", task_attempt=1,
                          admitted_dispatch="possible", admitted_dispatch_attempt=1)
        events = workers.get_event_q()
        sent = worker(host, monkeypatch)
        monkeypatch.setattr(workers, "get_event_q", lambda: events)
        workers.assign_tasks()
        assert len(sent) == 1 and queue.RUNNING[admitted["id"]]["task"]["_working_recovery"]
        ctx, loop = _loop_ctx(host.root, admitted["id"], attempt=2)
        _fast_hold(monkeypatch, budget_pause)
        _quiet_external(monkeypatch, budget_pause)
        with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
            budget_pause.request_pause(loop, rail=budget_pause.RAIL_GLOBAL_EXHAUSTED,
                                       scope="global", reason_text="money gone", root_task_id=admitted["id"])
        budget_pause.end_dispatch_fence(admitted["id"])
        install_exact_budget_pause(workers, admitted["id"],
                                   budget_pause.exact_pause_marker(raised.value.pause)["checkpoint"])
        [task] = queue.PENDING
        assert task["_working_recovery"] == admitted["_working_recovery"]
    else:
        host.pending.remove(admitted)
        task, _ = _parked(host.root, monkeypatch, task_id="exact", extra={**admitted, "execution_timeout_sec": 900})
    assert queue.resume_budget_paused_task(task["id"])["ok"]
    events = workers.get_event_q()
    sent = worker(host, monkeypatch)
    monkeypatch.setattr(workers, "get_event_q", lambda: events)
    # A real assignment lookup outage attaches the Project verification hold.
    unknown = unreadable_result(monkeypatch, task["id"])
    workers.assign_tasks()
    unknown["on"] = False
    assert not sent
    workers.WORKERS.clear()
    assert task.get("_project_admission_restore_hold"), task
    return task


@pytest.mark.parametrize(("recovered", "entry"), [
    pytest.param(False, "tick", id='first-run-tick'),
    pytest.param(True, "assignment", id='working-recovery-assignment'),
])
def test_exact_resume_wait_expires_without_worker_dispatch(host, expiry, monkeypatch, entry, recovered):  # noqa: F811
    task = _exact(host, monkeypatch, recovered=recovered)
    sent = []
    workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                         in_q=SimpleNamespace(put=sent.append)))
    task["deadline_at"] = "2000-01-01T00:00:00Z"
    queue.enforce_task_timeouts() if entry == "tick" else workers.assign_tasks()
    workers.assign_tasks()  # reserve a marker made by Project revalidation
    expiry.reap()
    assert not sent and not queue.PENDING and not queue.RUNNING and not host.attempts
    assert load_task_result(host.root, task["id"])["reason_code"] == "deadline"
    assert [e["status"] for e in expiry.frames if e.get("type") == "task_done"] == ["failed"]


@pytest.mark.parametrize(("during", "change"), [
    pytest.param(False, "pause", id='before-reap-pause'),
    pytest.param(False, "completed", id='before-reap-completed'),
    pytest.param(False, "unreadable", id='before-reap-unreadable'),
    pytest.param(False, "selected_source", id='before-reap-selected_source'),
    pytest.param(True, "stop", id='during-custody-stop'),
    pytest.param(True, "live", id='during-custody-live'),
    pytest.param(True, "revoked", id='during-custody-revoked'),
    pytest.param(True, "selected_grant", id='during-custody-selected_grant'),
    pytest.param(True, "selected_attempt", id='during-custody-selected_attempt'),
])
def test_exact_grant_authority_is_rechecked_after_mark(host, expiry, monkeypatch, change, during):  # noqa: F811
    from supervisor.owner_pause_control import request_owner_pause
    from supervisor.queue_transitions import revoke_exact_budget_resume
    from ouroboros.cancel_intents import request_cancel

    task = _exact(host, monkeypatch, recovered=True)
    expiry.mark(task)

    def alter(*_a, **_k):
        if change == "pause":
            assert request_owner_pause(task["id"], request_id="after-mark")["ok"]
        elif change == "stop":
            request_cancel(host.root, task["id"], source="test", reason="Stop after marking")
        elif change == "live":
            queue.RUNNING[task["id"]] = {"task": task}
        elif change == "completed":
            write_task_result(host.root, task["id"], "completed", result="Saved answer")
        elif change == "revoked":
            revoke_exact_budget_resume(task, "budget_exhausted_before_dispatch")
        elif change == "unreadable":
            unreadable_result(monkeypatch, task["id"])
        elif change == "selected_grant":
            task["_budget_pause_resume"]["grant_id"] = "foreign-grant"
        elif change == "selected_source":
            row = load_task_result(host.root, task["id"])["budget_pause"]
            write_task_result(host.root, task["id"], "running", budget_pause={
                **row, "source_ref": {**row["source_ref"], "sha256": "0" * 64}})
        else:
            task["_attempt"] += 1

    if during:
        monkeypatch.setattr(workers, "_audit_delegate_terminal_custody", alter)
    else:
        alter()
    expiry.reap()
    assert not host.attempts
    if change == "completed":
        assert not queue.PENDING
        assert load_task_result(host.root, task["id"])["result"] == "Saved answer"
        assert [e["status"] for e in expiry.frames if e.get("type") == "task_done"] == ["completed"]
    else:
        assert queue.PENDING == [task] and not expiry.frames
        if change != "unreadable":
            assert load_task_result(host.root, task["id"]).get("reason_code") != "deadline"


def test_aborted_update_saved_successor_expires_without_dispatch(tmp_path, monkeypatch, expiry):
    from tests.test_update_abort_retention import _system, _assert_returned

    # The real respawn caches ``_system``'s fake multiprocessing context in ``workers._CTX``;
    # restore that global at teardown so later tests get the real context again.
    monkeypatch.setattr(workers, "_CTX", None)
    system = _system(tmp_path, monkeypatch)
    # _system rebinds queues, but leaves the deadline reaper and its reservation intact.
    system.saved()
    assert system.control._quiesce_repo_writers("auto_merge") == []
    system.control._respawn_workers_after_failed_update()
    task = _assert_returned(system, tmp_path)
    expiry.mark(task)
    workers.assign_tasks()
    expiry.reap()
    assert not queue.PENDING and not queue.RUNNING
    assert all(commands.empty() for commands in system.launches)
    assert load_task_result(tmp_path, task["id"])["reason_code"] == "deadline"


@pytest.mark.parametrize("authority", ["valid", "spent", "transaction_unreadable"])
def test_owner_wait_restoration_uses_its_existing_settlement_owner(restart_case, expiry, monkeypatch, authority):  # noqa: F811
    from ouroboros import owner_wait, delegate_recovery
    from tests.test_owner_wait_restart import first_cleanup, acknowledge

    case = restart_case
    task = first_cleanup(case)
    acknowledge(case, monkeypatch, "launcher")
    task["_project_admission_restore_hold"] = {"reason": "project_routing_fence_lookup_failed"}
    if authority == "spent":
        owner_wait.set_owner_wait(case.root, case.task_id, {**case.wait, "state": "resumed"}, case.wait["wait_id"])
    elif authority == "source_unreadable":
        monkeypatch.setattr(owner_wait, "read_actor_source_bytes", lambda *_a: (_ for _ in ()).throw(OSError("unreadable source")))
    elif authority == "transaction_unreadable":
        delegate_recovery._restart_transaction_path(case.root, case.transaction_id).write_text("{torn")
    task["deadline_at"] = "2000-01-01T00:00:00Z"
    workers.assign_tasks()
    queue.enforce_task_timeouts()
    expiry.reap()
    result = load_task_result(case.root, case.task_id)
    if authority == "valid":
        assert not queue.PENDING and result["reason_code"] == "deadline"
    elif authority == "spent":
        from ouroboros.cancel_intents import has_active_intent

        assert result["status"] == "cancelled" or has_active_intent(case.root, case.task_id, strict=True)
        assert result.get("reason_code") != "deadline"
    else:
        assert queue.PENDING == [task] and not task.get("_terminalization_retry")
        assert result["status"] == "running" and not expiry.frames


@pytest.mark.parametrize("change", ["none", "foreign_transaction", "unreadable_result"])
def test_assisted_resume_checks_current_transaction_before_expiry(host, expiry, monkeypatch, change):  # noqa: F811
    from ouroboros import projects_registry as registry
    from supervisor import update_merge
    from tests.test_project_authority_outages import _resolver, _acknowledged_resolver_restart

    monkeypatch.setattr(workers, "ensure_worker_pool_started", lambda **_kw: True)
    tx = _resolver(host.root)
    [task] = queue.PENDING
    assert queue.persist_queue_snapshot()
    _acknowledged_resolver_restart(host.root, queue.PENDING)
    write_task_result(host.root, task["id"], "scheduled", admitted_dispatch="possible")
    bindings = registry._bindings_path(host.root)
    bindings.parent.mkdir(parents=True, exist_ok=True)
    bindings.write_text("{torn")
    queue.PENDING.clear()
    assert queue.restore_pending_from_snapshot() == 1
    assert update_merge.enqueue_assisted_resolution_task(tx) == task["id"]
    [task] = queue.PENDING
    assert task["_managed_update_resume"] and task["_project_admission_restore_hold"]
    expiry.mark(task)
    if change == "foreign_transaction":
        update_merge.write_update_tx({**tx, "target_sha": "different"})
    elif change == "unreadable_result":
        unreadable_result(monkeypatch, task["id"])
    expiry.reap()
    assert not queue.RUNNING and not host.attempts
    if change == "none":
        assert not queue.PENDING
        assert load_task_result(host.root, task["id"])["reason_code"] == "deadline"
    else:
        assert queue.PENDING == [task] and not expiry.frames


def test_spent_owner_wait_after_mark_retains_custody_until_its_owner_settles(restart_case, expiry, monkeypatch):  # noqa: F811
    from ouroboros import owner_wait
    from tests.test_owner_wait_restart import first_cleanup, acknowledge

    case = restart_case
    task = first_cleanup(case)
    acknowledge(case, monkeypatch, "launcher")
    expiry.mark(task)
    # Use the existing writer to test stale-mark handling. This does not assert
    # a stopped predecessor can reach its warm-wait return after acknowledged exit.
    owner_wait.set_owner_wait(case.root, case.task_id, {**case.wait, "state": "resumed"}, case.wait["wait_id"])
    expiry.reap()
    assert queue.PENDING == [task] and not expiry.frames
    write_task_result(case.root, case.task_id, "completed", result="Settled by the wait owner")
    queue.enforce_task_timeouts()
    expiry.reap()
    assert not queue.PENDING
    assert load_task_result(case.root, case.task_id)["result"] == "Settled by the wait owner"


def test_owner_restart_never_started_hold_also_respects_calendar_expiry(host, expiry):  # noqa: F811
    from supervisor.events_budget import HOLD_OWNER_RESTART, budget_hold_fact
    from tests.test_restart_retention import RESTART_REASON

    task = queue.enqueue_task({"id": "restart-held", "type": "task", "chat_id": 1, "text": "Still queued"})
    queue.ensure_control_task_result(task["id"])
    workers.WORKERS.clear()
    assert workers.kill_workers(force=True, terminal_status="cancelled", result_reason=RESTART_REASON,
                                reconcile_delegate_custody=False, hold_never_started=True,
                                archive_service_logs=False)
    [task] = queue.PENDING
    assert budget_hold_fact(task)["reason"] == HOLD_OWNER_RESTART
    expiry.mark(task)
    expiry.reap()
    assert not queue.PENDING and not queue.RUNNING and not host.attempts
    assert load_task_result(host.root, task["id"])["reason_code"] == "deadline"
