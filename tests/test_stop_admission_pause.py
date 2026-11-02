"""Every process-ending stop pauses admission before it reads the queue; an abandoned one reopens it.

The lane-level behaviour (a committed launch reaches its worker while the stop waits, a waiting one
writes nothing, the bound) lives in tests/test_launch_outside_q.py; this module pins each door.
"""
from __future__ import annotations

import inspect
import threading
from types import SimpleNamespace

import pytest

from ouroboros import server_restart
from ouroboros.task_results import write_task_result
from supervisor import queue, schedule_occurrence, worker_assignment, workers
from supervisor import workspace_preparation as prep
from tests.test_g1_followup_policy import BINDING, DEADLINE, ORIGIN
from tests.test_launch_outside_q import hold_after
from tests.test_schedule_followup import _ctx
from tests.test_scheduled_receipt_contention import install_worker
from tests.test_workspace_preparation import _WAIT_SEC
from tests.test_workspace_preparation import env as env

pytestmark = pytest.mark.serial


@pytest.fixture
def generation(monkeypatch):
    monkeypatch.setattr(queue, "RETURNS_NAMED", {})
    prep._reset_for_tests()
    current = prep.start_generation(stop=lambda: False)
    yield current
    prep._reset_for_tests()


def test_pause_stops_launches_only_and_resume_reopens_them(generation):
    assert not generation.launches_closed()
    assert prep.pause_admission(1.0) is True and generation.launches_closed()
    assert not generation.closed()  # the generation itself goes on: a child's admission is queued as ever
    prep.resume_admission()
    assert not generation.launches_closed()


def test_a_stop_that_finishes_never_lifts_another_stops_pause(generation):
    """An Update that ends its writer stop reopens only its own pause, not an owner Restart's under way."""
    assert prep.pause_admission(1.0) and prep.pause_admission(1.0)
    prep.resume_admission()
    assert generation.launches_closed()
    prep.resume_admission()
    assert not generation.launches_closed()
    prep.resume_admission()  # a stray reopen never goes below zero
    assert prep.pause_admission(1.0) and generation.launches_closed()


def test_an_abandoned_restart_reopens_the_queue_and_the_admission_lane(generation):
    assert prep.pause_admission(1.0) and generation.launches_closed()
    queue.name_queue_returns([], "planned_restart")
    queue.reopen_queue_returns("planned_restart")
    assert not generation.launches_closed() and not queue.RETURNS_NAMED


def test_planned_restart_and_quit_teardown_pause_admission_first(monkeypatch):
    import server
    from ouroboros import server_restart

    calls = []
    monkeypatch.setattr(prep, "pause_admission", lambda timeout=None: calls.append("pause") or True)
    assert server_restart._begin_task_teardown(True) == server_restart._shutdown_task_cleanup_args(True)
    assert server_restart._begin_task_teardown(False) == server_restart._shutdown_task_cleanup_args(False)
    assert calls == ["pause", "pause"]
    # Both server doors begin their teardown through it, before they read the queue or retire the pool.
    assert "_begin_task_teardown(restart_requested=True)" in inspect.getsource(server_restart._perform_supervisor_restart)
    assert inspect.getsource(server).count("_begin_task_teardown(restart_requested)") == 1


@pytest.mark.parametrize("raises", [False, True])
def test_the_update_stops_its_writers_with_admission_paused(monkeypatch, raises):
    from ouroboros.gateway import control

    events = []
    monkeypatch.setattr(prep, "pause_admission", lambda timeout=None: events.append("pause") or True)
    monkeypatch.setattr(prep, "resume_admission", lambda: events.append("resume"))

    def stop(reason):
        events.append(("stop", reason, "paused" if events[-1] == "pause" else "open"))
        if raises:
            raise RuntimeError("writer stop failed")
        return ["active:writer"]

    monkeypatch.setattr(control, "_stop_repo_writers", stop)
    if raises:
        with pytest.raises(RuntimeError):
            control._quiesce_repo_writers("smart")
    else:
        assert control._quiesce_repo_writers("smart") == ["active:writer"]
    assert events == ["pause", ("stop", "smart", "paused"), "resume"]


# The owner's Restart through its real order (checkout, pause, flags, returns, pool retirement), adapted
# from the R5 reviewers' probes: only the checkout, OS termination and daemon stop are stand-ins.
def retiring_worker(env, monkeypatch):
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", env.data / "state" / "queue_snapshot.json")
    monkeypatch.setattr(queue, "RETURNS_NAMED", {})
    monkeypatch.setattr(worker_assignment, "_COMMITTED_UNSENT", {})
    monkeypatch.setattr(workers, "_LAST_WORKER_EXIT_CENSUS", None)
    worker, delivered = install_worker(env, monkeypatch)
    alive = [True]
    worker.proc.is_alive = lambda: alive[0]
    worker.proc.join = lambda **_kw: None
    monkeypatch.setattr(workers, "kill_worker_tree", lambda _pid: alive.__setitem__(0, False))
    monkeypatch.setattr(workers, "_kill_survivors", lambda: None)
    monkeypatch.setattr("supervisor.worker_process.close_worker_stop_channel", lambda _proc: None)
    monkeypatch.setattr("supervisor.worker_health.retire_confirmed_worker_consumers", lambda *_a: None)
    return worker, delivered


def scheduled_followup(env, monkeypatch, relation):
    from ouroboros.tools.followup import _handle_schedule_followup

    monkeypatch.setattr("ouroboros.config.get_bg_wakeup_min_sec", lambda: 0)
    write_task_result(env.data, ORIGIN, "completed", root_task_id=ORIGIN, billing_group=BINDING, deadline_at=DEADLINE)
    ctx = _ctx(env.data.parent)
    ctx.current_chat_id = 7
    ctx.task_metadata["resource_intent"] = {"kind": "system_repo"}
    result = _handle_schedule_followup(ctx, relation=relation, run_at="2000-01-01T00:00:00+00:00",
                                      objective="Finish the original work")
    assert result.startswith("FOLLOWUP_SCHEDULED"), result
    queue.check_scheduled_tasks()
    [row] = queue.PENDING
    assert schedule_occurrence.restore_verdict(row) == "replay"
    return row


def owner_restart_context(env, monkeypatch):
    monkeypatch.setattr(server_restart, "DATA_DIR", env.data)
    monkeypatch.setattr(server_restart, "_OWNER_STOP_BEGAN", threading.Event())
    monkeypatch.setattr(server_restart, "_LAST_RETURNING", set())
    monkeypatch.setattr(server_restart, "_LAST_RETURN_TX", {"id": ""})
    monkeypatch.setattr(server_restart, "_safe_restart_serialized", lambda *_a, **_kw: (True, ""))
    monkeypatch.setattr(server_restart, "_managed_update_pending_kwargs", lambda: {})
    monkeypatch.setattr(server_restart, "_stop_owned_daemon", lambda *_a: None)
    monkeypatch.setattr("ouroboros.owned_shutdown.begin_owned_stop", lambda *_a, **_kw: None)
    monkeypatch.setattr("ouroboros.owned_shutdown.stop_owned_work", lambda *_a: None)
    monkeypatch.setattr("ouroboros.delegate_custody.reconcile_orphaned_runs", lambda *_a, **_kw: [])
    monkeypatch.setattr("supervisor.active_activity.get_direct_activity_registry",
                        lambda: SimpleNamespace(snapshot=lambda: [], get=lambda _tid: None))
    monkeypatch.setattr(workers, "direct_chat_turns", lambda: [])
    exits = []
    monkeypatch.setattr(server_restart, "_request_restart_exit", lambda owner=False: exits.append(owner))

    def kill(**kwargs):
        return workers.kill_workers(**{**kwargs, "archive_service_logs": False, "reconcile_review_custody": False})

    return SimpleNamespace(PENDING=queue.PENDING, RUNNING=queue.RUNNING, kill_workers=kill,
                           safe_restart=lambda **_kw: (True, ""), consciousness=None), exits


@pytest.mark.parametrize("relation", ["independent", "related"])
def test_owner_restart_hands_a_committed_followup_over_before_its_own_restriction(env, monkeypatch, relation):
    """A follow-up a task scheduled is between its committed marks and its put when the owner presses
    Restart: the Restart pauses admission before it writes its follow-up restriction, so the launch reaches
    its worker instead of being refused by that restriction and restored as possibly dispatched."""
    _worker, delivered = retiring_worker(env, monkeypatch)
    row = scheduled_followup(env, monkeypatch, relation)
    ctx, exits = owner_restart_context(env, monkeypatch)
    reached, proceed = hold_after(monkeypatch, worker_assignment, "_commit_launch", row["id"])
    real_pause = prep.pause_admission
    observed = {}

    def pause(timeout=None):
        from ouroboros.delegate_recovery import active_restart_transaction
        observed["flag_before_pause"] = bool(active_restart_transaction(env.data))
        proceed.set()
        observed["drained"] = real_pause(timeout)
        observed["delivered"] = [item["id"] for item in delivered if "id" in item]
        return observed["drained"]

    monkeypatch.setattr(prep, "pause_admission", pause)
    try:
        workers.assign_tasks()
        assert reached.wait(_WAIT_SEC) and worker_assignment._unsent(row) is not None and not delivered
        assert server_restart._perform_owner_restart(ctx) == (True, "") and exits == [True]
    finally:
        proceed.set()
        assert env.gen.wait_idle(_WAIT_SEC)
    assert observed == {"flag_before_pause": False, "drained": True, "delivered": [row["id"]]}
