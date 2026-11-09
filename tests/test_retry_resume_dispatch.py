"""An unreadable retry lineage holds one dispatch, without spending the owner's Resume.

The real Pause, park, Resume and assign_tasks consumer run. The injected fault is
only the dispatch's own lineage read (inside budget_resume_dispatch_allowed, or the
chooser's read of it ahead of that decision, ``dispatch_choice._actor_reads``), like
the reproduced one-read failure; every other lineage read succeeds. A full pool
chooses nothing and reads nothing, so it neither spends nor reads the grant.
"""
import inspect

import pytest

from ouroboros import budget_pause, cancel_intents
from supervisor import queue, workers
from supervisor.owner_pause_control import request_owner_pause
from tests.test_execution_clock_resume_handover import clock  # noqa: F401
from tests.test_project_hold_recovery import worker
from tests.test_retry_pause_generations import _park, _running_actor
from tests.test_retry_pause_identity import _fresh_plan, reaper  # noqa: F401
from tests.test_swarm_host_admission import host  # noqa: F401

DISPATCH_READS = {"budget_resume_dispatch_allowed", "_actor_reads"}



@pytest.mark.parametrize(("busy", "fault_count", "actor"), [
    pytest.param(False, 1, "retry", id='assignment-1-retry'),
    pytest.param(True, 2, "retry_chain", id='busy-pool-2-retry_chain'),
])
def test_unreadable_lineage_preserves_grant_until_a_readable_dispatch(
        host, monkeypatch, clock, reaper, actor, fault_count, busy):  # noqa: F811
    running = _running_actor(host, monkeypatch, clock, reaper, actor)
    assert request_owner_pause(running["id"], request_id="pause")["ok"]
    _park(host, running, _fresh_plan(monkeypatch))
    assert queue.resume_budget_paused_task(running["id"])["ok"]
    before = budget_pause.budget_pause_row(host.root, running["id"])
    [pending] = [task for task in host.pending if task["id"] == running["id"]]
    handoff = dict(pending["_budget_pause_resume"])
    real, calls = cancel_intents._validated_single_cancel_target, []
    unavailable, forecasts = True, []

    def unreadable_dispatch(*args, **kwargs):
        if DISPATCH_READS & {frame.function for frame in inspect.stack()}:
            calls.append(args[1])
            if unavailable:
                raise OSError("one targeted transient lineage-read refusal")
        return real(*args, **kwargs)

    monkeypatch.setattr(cancel_intents, "_validated_single_cancel_target", unreadable_dispatch)
    sent = worker(host, monkeypatch)
    # Exercise the actual residual-demand callback without creating native workers.
    monkeypatch.setattr("supervisor.worker_pool_lifecycle.reconcile_worker_capacity",
                        lambda demand, _reading=None: forecasts.append(demand()))
    if busy:  # a full pool: no pass, no read, no demand, the grant untouched; then the slot frees
        workers.WORKERS[0].busy_task_id = "another-task"
        workers.assign_tasks()
        assert not calls and not sent and forecasts[-1] == 0 and pending["_budget_pause_resume"] == handoff
        assert budget_pause.budget_pause_row(host.root, running["id"])["grant"] == before["grant"]
        workers.WORKERS[0].busy_task_id = None
    for attempt in range(fault_count):
        reads_before = len(calls)
        workers.assign_tasks()
        after = budget_pause.budget_pause_row(host.root, running["id"])
        facts = {"targeted_reads": len(calls), "sent": len(sent), "state": after["state"],
                 "same_grant": after["grant"] == before["grant"],
                 "pending": [task["id"] for task in host.pending]}
        print("unreadable dispatch witness:", facts)
        assert not sent and pending in host.pending, facts
        assert after["state"] == budget_pause.STATE_RESUME_GRANTED and after["grant"] == before["grant"], facts
        assert pending["_budget_pause_resume"] == handoff and len(calls) > reads_before
        assert forecasts[-1] == 0, "unknown selection never becomes capacity demand"
    unavailable = False
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [running["id"]]
    assert sent[0]["_budget_pause_resume"] == handoff
    assert running["id"] in workers.RUNNING and pending not in host.pending
    assert budget_pause.budget_pause_row(host.root, running["id"])["grant"] == before["grant"]


@pytest.mark.parametrize("fault", ["lineage", "grant_row"])
@pytest.mark.parametrize("after_readable", ["unchanged", "new_generation"])
@pytest.mark.parametrize("busy", [False, True], ids=["assignment", "busy-pool"])
def test_child_dispatch_preserves_unknown_but_revokes_proven_stale_authority(
        tmp_path, monkeypatch, fault, after_readable, busy):
    from tests.test_budget_pause_lineage import CHILD, ROOT, _resumed_root_over_parked_child

    q, pool, sent, child, _ctx, root_grant = _resumed_root_over_parked_child(
        tmp_path, monkeypatch, old_writer=False)
    assert q.resume_budget_paused_task(CHILD)["ok"]
    before = budget_pause.budget_pause_row(tmp_path, CHILD)
    handoff = dict(child["_budget_pause_resume"])
    module, name = ((cancel_intents, "_validated_single_cancel_target") if fault == "lineage"
                    else (budget_pause, "budget_pause_row"))
    real, calls = getattr(module, name), []
    forecasts = []
    monkeypatch.setattr("supervisor.worker_pool_lifecycle.reconcile_worker_capacity",
                        lambda demand, _reading=None: forecasts.append(demand()))
    def unreadable(*args, **kwargs):
        if args[1] == ROOT and DISPATCH_READS & {f.function for f in inspect.stack()}:
            calls.append(args[1])
            raise OSError("current root authority unreadable")
        return real(*args, **kwargs)

    with monkeypatch.context() as unavailable:
        unavailable.setattr(module, name, unreadable)
        if busy:  # a full pool reads nothing, the unreadable authority included; then the slot frees
            pool.WORKERS[1].busy_task_id = "another-task"
            pool.assign_tasks()
            assert not calls and not sent and forecasts[-1] == 0 and child["_budget_pause_resume"] == handoff
            pool.WORKERS[1].busy_task_id = None
        pool.assign_tasks()
    assert calls and set(calls) == {ROOT} and not sent and child in pool.PENDING
    assert budget_pause.budget_pause_row(tmp_path, CHILD) == before
    assert child["_budget_pause_resume"] == handoff
    assert forecasts[-1] == 0
    if after_readable == "new_generation":
        row = budget_pause.budget_pause_row(tmp_path, ROOT)
        budget_pause.set_budget_pause(tmp_path, ROOT, {
            **row, "resume_generation": 2,
            "grant": {**root_grant, "grant_id": "later-root-grant", "generation": 2}})
    pool.assign_tasks()
    if after_readable == "unchanged":
        assert [task["id"] for task in sent] == [CHILD]
        assert sent[0]["_budget_pause_resume"] == handoff
    else:
        assert not sent and child in pool.PENDING and "_budget_pause_resume" not in child
        revoked = budget_pause.budget_pause_row(tmp_path, CHILD)
        assert revoked["state"] == budget_pause.STATE_PAUSED
        assert revoked["grant"]["revoke_reason"] == "root_resume_generation_stale"
        assert forecasts[-1] == 0
