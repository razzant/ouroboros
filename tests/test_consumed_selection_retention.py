"""Bounded consumer falsifier; no live process, service or provider is used.

Exercise the actual consumed-grant retirement callers. The sole injected failure
is the existing reciprocal retry reader *during* the projection's identity read;
restore/source saving and later child selection are real. This is not a complete
application boot or a product verdict.
"""
import copy

import pytest

from ouroboros import budget_pause as bp, cancel_intents, owner_pause
from supervisor import events_budget as eb, queue, workers
from supervisor.budget_resume import revoke_exact_budget_resume
from supervisor.restart_retention import recover_saved_running
from supervisor.owner_pause_control import request_owner_pause
from tests.test_dir02_consumed_grant_retry import (
    CHILD, _child, _row, _select_child_as,
)
from tests.test_execution_clock_resume_handover import _supervisor_events
from tests.test_execution_clock_resume_handover import clock as clock  # pytest fixture export
from tests.test_retry_pause_generations import _park, _running_actor
from tests.test_retry_pause_identity import _assign, _fresh_plan, _loop_once
from tests.test_retry_pause_identity import reaper as reaper  # pytest fixture export
from tests.test_swarm_host_admission import host as host  # pytest fixture export

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("transition", ["saved_restart", "owner_wait"])
@pytest.mark.parametrize("read_fails", [False, True, "persistent"],
                         ids=["readable-control", "temporary-read-failure", "persistent-read-failure"])
def test_consumed_retry_selection_survives_projection_read_failure(
        host, monkeypatch, clock, reaper, transition, read_fails):
    running = _running_actor(host, monkeypatch, clock, reaper, "retry")
    actor = running["id"]
    assert actor != "held"
    plan = _fresh_plan(monkeypatch)
    _child(parent=actor)
    pause = request_owner_pause(actor, request_id="projection-fault")
    assert pause["ok"], pause
    _park(host, running, plan)
    resumed = queue.resume_budget_paused_task(actor)
    assert resumed["ok"] and resumed["exact_continuation"], resumed
    [run] = _assign(host, monkeypatch)
    assert run["id"] == actor
    _loop_once(host, monkeypatch, run, plan=plan, events=_supervisor_events(host))
    exact = bp.budget_pause_row(host.root, actor)
    assert exact["state"] == bp.STATE_RESUMED and exact["grant"]["consumed_at"]
    grant_id = exact["grant"]["grant_id"]
    assert eb.live_root_resume_grant(queue, "held", host.root)["grant_id"] == grant_id
    assert eb.budget_hold_fact(_row(host, CHILD)), "child is not selected implicitly"

    # The normal root classifier still runs; only its underlying read fails.
    classify = owner_pause.root_actor
    probes = []
    def temporary_identity_failure(root, logical_id, task_id, **kwargs):
        probes.append((logical_id, task_id))
        with monkeypatch.context() as fault:
            if read_fails:
                def unreadable(*args, **kw):
                    raise OSError("bounded retry-chain read unavailable")
                fault.setattr(cancel_intents, "_validated_single_cancel_target", unreadable)
            return classify(root, logical_id, task_id, **kwargs)

    meta = workers.RUNNING[actor]
    with monkeypatch.context() as during_retirement:
        during_retirement.setattr(owner_pause, "root_actor", temporary_identity_failure)
        if transition == "saved_restart":
            recovered = recover_saved_running(
                [copy.deepcopy(meta)], host.root,
                transaction={"return_ids": [actor]}, cause="restart")
            assert len(recovered) == 1, recovered
            successor = recovered[0]
        else:
            successor = copy.deepcopy(meta["task"])
            successor["_owner_wait_resume"] = {
                "state": "waiting", "wait_id": "kept-wait",
                "restart_transaction_id": "kept-transaction"}
            assert revoke_exact_budget_resume(successor, "restart_before_dispatch") is False
    assert not probes, "retaining a proved consumed selection needs no new role read"
    with queue._queue_lock:
        workers.RUNNING.pop(actor)
        host.pending.insert(0, successor)
    assert bp.budget_pause_row(host.root, actor)["grant"] == exact["grant"], "no grant reissued"
    assert eb.budget_hold_fact(_row(host, CHILD)), "still no implicit fanout"
    if read_fails == "persistent":
        before = copy.deepcopy(_row(host, CHILD))
        with monkeypatch.context() as still_unreadable:
            def refuse(*args, **kwargs):
                raise OSError("retry identity remains unavailable at dispatch")
            still_unreadable.setattr(cancel_intents, "_validated_single_cancel_target", refuse)
            assert eb.budget_resume_dispatch_allowed(queue, successor) is None
            refused = _select_child_as(host, actor)
            assert not refused["ok"], refused
            assert _row(host, CHILD) == before
    outcome = _select_child_as(host, actor)
    assert outcome["ok"], outcome
    assert _row(host, CHILD)[eb.SELECTION_KEY]["root_grant_id"] == grant_id
    assert eb.budget_resume_dispatch_allowed(queue, _row(host, CHILD)) is True


@pytest.mark.parametrize("transition", ["saved_restart", "owner_wait"])
@pytest.mark.parametrize("by_model", [False, True], ids=["owner-unbound-member", "model-bound-member"])
@pytest.mark.parametrize("root_change", ["none", "newer_root_generation", "root_held_again"])
def test_consumed_member_handoff_retires_and_the_member_continues(
        tmp_path, monkeypatch, transition, by_model, root_change):
    """A member never lends a root grant, so its spent handoff becomes no dispatch carrier;
    a later root generation neither re-holds nor strands its own continuation."""
    from types import SimpleNamespace

    from ouroboros import owner_wait
    from tests._budget_pause_exact_helpers import _loop_ctx, _parked
    from tests.test_budget_pause_holds import _idle_worker
    from tests.test_saved_root_resume_grant import _saved_tree

    q, workers, rows = _saved_tree(tmp_path, monkeypatch)
    if by_model:
        assert q.resume_budget_paused_task("saved-root")["ok"]
    root_grant = eb.live_root_resume_grant(q, "saved-root", tmp_path)
    workers.PENDING.remove(rows["first"])
    member, _pause = _parked(tmp_path, monkeypatch, task_id="first", root_task_id="saved-root",
                            extra={"parent_task_id": "saved-root", "delegation_role": "subagent", "depth": 1})
    granted = q.resume_budget_paused_task("first", selected_by="saved-root" if by_model else "")
    assert granted["ok"], granted
    ctx, _limit = _loop_ctx(tmp_path, "first")
    ctx.root_task_id = "saved-root"
    ctx.task_metadata = {"parent_task_id": "saved-root", "delegation_role": "subagent", "depth": 1}
    ctx.budget_pause_resume = member["_budget_pause_resume"]
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *a, **k: (None, "max"))
    bp.resume_paused_loop(SimpleNamespace(_ctx=ctx), bp.load_budget_pause(ctx), [], {}, {}, set(),
                          budget_remaining_usd=5.0)
    exact = bp.budget_pause_row(tmp_path, "first")
    assert exact["state"] == bp.STATE_RESUMED and exact["grant"]["consumed_at"]
    assert exact["grant"]["root_grant_id"] == (root_grant["grant_id"] if by_model else "")
    if root_change == "newer_root_generation":
        if not by_model:  # This owner choice had no lender yet; publish the root's actual first grant.
            assert q.resume_budget_paused_task("saved-root")["ok"]
        rows["saved-root"][eb.SELECTION_KEY]["root_grant_id"] = "newer-root-resume"
    elif root_change == "root_held_again":
        eb.hold_budget_row(rows["saved-root"], reason=eb.HOLD_SAVED_WORK)
    index = next(i for i, row in enumerate(workers.PENDING) if row is member)
    if transition == "saved_restart":
        workers.PENDING.pop(index)
        recovered = recover_saved_running([{"task": member, "attempt": 1}], tmp_path,
                                           transaction={"return_ids": ["first"]}, cause="restart")
        assert len(recovered) == 1
        successor = recovered[0]
        workers.PENDING.insert(0, successor)
    else:
        successor = copy.deepcopy(member)
        successor["_owner_wait_resume"] = {"state": "waiting", "wait_id": "member-wait",
                                           "restart_transaction_id": "confirmed"}
        workers.PENDING[index] = successor
        assert revoke_exact_budget_resume(successor, "restart_before_dispatch") is False
    assert "_budget_pause_resume" not in successor
    assert successor.get(eb.SELECTION_KEY) is None, "a member's spent handoff is no dispatch carrier"
    assert eb.BUDGET_HOLD_KEY not in successor, "retirement cannot manufacture a stop"
    assert bp.budget_pause_row(tmp_path, "first")["grant"] == exact["grant"], "nothing reissued or revoked"
    assert eb.budget_resume_dispatch_allowed(q, successor) is True
    sent = []
    _idle_worker(workers, sent)
    for _ in range(3):
        workers.WORKERS[0].busy_task_id = None
        workers.assign_tasks()
    dispatched = [task["id"] for task in sent]
    assert dispatched.count("first") == 1, dispatched
    assert "second" not in dispatched and eb.budget_hold_fact(rows["second"]), "an unchanged sibling stays held"


def test_budget_exact_after_warm_keeps_newer_selection_on_return(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ouroboros import owner_wait
    from supervisor import continuation_admission
    from supervisor.budget_resume import resume_warm_owner_pause_root
    from tests._budget_pause_exact_helpers import _loop_ctx, _parked
    from tests.test_budget_pause_holds import _idle_worker
    from tests.test_saved_root_resume_grant import _saved_tree

    q, workers, rows = _saved_tree(tmp_path, monkeypatch)
    assert q.resume_budget_paused_task("saved-root")["ok"]
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    live = workers.RUNNING["saved-root"]["task"]
    fence, _ = owner_pause.install_fence(tmp_path, "saved-root", request_id="warm-before-budget")
    eb._set_root_budget_pause_locked("saved-root", {**fence, "cause": "owner_pause"})
    workers.RUNNING["saved-root"]["owner_wait"] = {"state": "waiting", "reason": "owner_pause"}
    monkeypatch.setattr(continuation_admission, "action_writers", lambda *a, **k: [])
    warm = resume_warm_owner_pause_root("saved-root")
    assert warm["ok"] and eb.live_root_resume_grant(q, "saved-root", tmp_path)["grant_id"] == warm["grant_id"]
    # Physical parking is a fixture; the budget grant and loop consumption run.
    workers.RUNNING.pop("saved-root")
    exact_task, _ = _parked(tmp_path, monkeypatch, task_id="saved-root")
    exact_task[eb.SELECTION_KEY] = copy.deepcopy(live[eb.SELECTION_KEY])
    newer = q.resume_budget_paused_task("saved-root")
    assert newer["ok"] and newer["exact_continuation"], newer
    ctx, _ = _loop_ctx(tmp_path, "saved-root")
    ctx.budget_pause_resume = exact_task["_budget_pause_resume"]
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *a, **k: (None, "max"))
    bp.resume_paused_loop(SimpleNamespace(_ctx=ctx), bp.load_budget_pause(ctx), [], {}, {}, set(),
                          budget_remaining_usd=5.0)
    before = bp.budget_pause_row(tmp_path, "saved-root")
    assert before["grant"]["consumed_at"] and not before["grant"].get("owner_pause_fence_id")
    assert owner_pause.read_fence(tmp_path, "saved-root")["resume_grant"]["grant_id"] == warm["grant_id"]
    returned = recover_saved_running([{"task": exact_task, "attempt": 1}], tmp_path,
                                     transaction={"return_ids": ["saved-root"]}, cause="restart")
    assert len(returned) == 1
    workers.PENDING.remove(exact_task)
    workers.PENDING.insert(0, returned[0])
    assert eb.live_root_resume_grant(q, "saved-root", tmp_path)["grant_id"] == newer["grant_id"]
    assert q.resume_budget_paused_task("first", selected_by="saved-root")["ok"]
    assert eb.budget_resume_dispatch_allowed(q, rows["first"])
    assert eb.budget_hold_fact(rows["second"])
    assert bp.budget_pause_row(tmp_path, "saved-root")["grant"] == before["grant"]
