# tests/test_dir02_consumed_grant_retry.py
import json
import queue as stdqueue
from types import SimpleNamespace

import pytest

from ouroboros import budget_pause as bp, owner_pause, working_checkpoint as wc
from supervisor import events_budget as eb, queue, worker_health, workers
from supervisor.owner_pause_control import request_owner_pause
from tests._budget_pause_exact_helpers import _loop_ctx
from tests.test_execution_clock_resume_handover import _supervisor_events
from tests.test_execution_clock_resume_handover import clock as clock  # pytest fixture export
from tests.test_retry_pause_generations import _park, _running_actor
from tests.test_retry_pause_identity import (
    _assign, _fresh_plan, _hung_first_attempt, _loop_once,
)
from tests.test_retry_pause_identity import reaper as reaper  # pytest fixture export
from tests.test_resume_handover import _stop_boot
from tests.test_swarm_host_admission import host as host  # pytest fixture export
from tests.test_worker_crash_retry import _make_worker

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("return_first", [False, True],
                         ids=["same-attempt-working", "later-attempt-retry"])
def test_consumed_g1_selection_after_recovery(
        host, monkeypatch, clock, reaper, return_first):
    plan = _fresh_plan(monkeypatch)
    first = _hung_first_attempt(host, monkeypatch, clock)
    child_id = "held-child"
    child = queue.enqueue_task({
        "id": child_id, "type": "task", "chat_id": 0, "text": "child work",
        "root_task_id": "held", "parent_task_id": "held", "depth": 1,
        "delegation_role": "subagent",
    })
    assert not child.get("_admission_blocked"), child
    pause = request_owner_pause("held", request_id="g1-pause")
    assert pause["ok"], pause
    _park(host, first, plan)
    g1 = queue.resume_budget_paused_task("held")
    assert g1["ok"] and g1["exact_continuation"], g1
    [resumed] = _assign(host, monkeypatch)
    assert resumed["id"] == "held"
    assert _loop_once(host, monkeypatch, resumed, plan=plan,
                      events=_supervisor_events(host))["kind"] == "pause"
    exact = bp.budget_pause_row(host.root, "held")
    assert exact["state"] == bp.STATE_RESUMED
    assert exact["grant"]["consumed_at"] and not exact["grant"].get("revoked_at")
    assert exact["grant"]["grant_id"] == g1["grant_id"]
    assert eb.live_root_resume_grant(queue, "held", host.root)["grant_id"] == g1["grant_id"]
    assert eb.budget_hold_fact(next(t for t in host.pending if t["id"] == child_id))

    # Silence beside a held descendant manufactures no reap job.
    assert queue._has_pending_descendant("held")
    clock[0] += max(float(queue.get_task_idle_timeout_sec()),
                    float(queue.get_per_call_timeout_ceiling_sec()) + 120.0) + 1
    queue.enforce_task_timeouts()
    assert "held" in workers.RUNNING
    assert not workers.RUNNING["held"].get("finalization_requested_at")
    assert reaper.jobs.empty()

    if return_first:
        # Real acknowledged Restart/stop/boot; no new Pause or Resume.
        assert _stop_boot(monkeypatch, returning=True) == 2
        returned = next(t for t in host.pending if t["id"] == "held")
        assert returned["_attempt"] == exact["task_attempt"] + 1
        assert "_budget_pause_resume" not in returned
        assert eb.budget_hold_fact(returned) is None
        [running] = _assign(host, monkeypatch)
        assert running["id"] == "held"
        seen = _loop_once(host, monkeypatch, running, plan=plan)
        assert seen["kind"] == "working"
        # A subsequent real checkpoint gives the crash current saved cognition.
        ctx, limit = _loop_ctx(host.root, "held", attempt=running["_attempt"])
        ctx.active_model, ctx.context_fit_plan = plan.model, plan
        limit.messages = seen["messages"]
        wc.save_round(limit, "post_batch")

    meta = workers.RUNNING["held"]
    dead_attempt = meta["attempt"]
    assert (dead_attempt != exact["task_attempt"]) is return_first
    saved = json.loads(wc.checkpoint_path(host.root, "held", dead_attempt).read_text())
    assert saved["working"]["boundary"] == ("post_batch" if return_first else "resumed")
    workers.WORKERS[0] = _make_worker(busy_task_id="held", exitcode=1)
    jobs = stdqueue.Queue()
    monkeypatch.setattr(queue, "_reap_queue", jobs)
    monkeypatch.setattr(queue, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(workers, "_LAST_SPAWN_TIME", 0)
    monkeypatch.setattr(workers, "CRASH_TS", [])
    monkeypatch.setattr(workers, "_reconcile_confirmed_dead_review_owner", lambda *_a: None)
    monkeypatch.setattr(workers, "respawn_worker", lambda *_a, **_k: None)
    monkeypatch.setattr(workers, "reconstruct_task_cost", lambda *_a, **_k: {
        "cost_accounting_status": "available", "cost_final": True})
    monkeypatch.setattr("ouroboros.headless.prepare_terminal_task_files",
                        lambda *_a, **_k: {"terminal_source_present": False})
    workers.ensure_workers_healthy()
    job = jobs.get_nowait()
    assert job["kind"] == "confirmed_dead_worker" and job["meta"] is meta
    worker_health.recover_confirmed_dead_worker(job)

    retry = next(t for t in host.pending if t["id"] == "held")
    assert retry["_attempt"] == dead_attempt + 1
    assert retry["_working_recovery"]["cause"] == "worker_crash"
    assert "_budget_pause_resume" not in retry
    assert bp.budget_pause_row(host.root, "held")["grant"] == exact["grant"]
    assert owner_pause.read_fence(host.root, "held")["fence_id"] == pause["fence_id"]
    assert not owner_pause.fence_closed(owner_pause.read_fence(host.root, "held"))
    [running] = _assign(host, monkeypatch)
    assert running["id"] == "held"
    assert _loop_once(host, monkeypatch, running, plan=plan)["kind"] == "working"

    pushed = []
    sctx = SimpleNamespace(
        DRIVE_ROOT=host.root, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
        WORKERS=workers.WORKERS, sort_pending=queue.sort_pending,
        persist_queue_snapshot=queue.persist_queue_snapshot,
        bridge=SimpleNamespace(push_log=pushed.append))
    eb._handle_budget_resume_child(
        {"task_id": child_id, "requested_by": "held", "reason": "continue saved tree"}, sctx)
    assert pushed and pushed[-1]["type"] == "budget_resume_child_outcome"
    print("G1 -> acknowledged return -> crash retry -> child:", pushed[-1])
    assert pushed[-1]["ok"], pushed[-1]
    selected = next(t for t in host.pending if t["id"] == child_id)
    assert selected[eb.SELECTION_KEY]["root_grant_id"] == g1["grant_id"]
    assert eb.budget_resume_dispatch_allowed(queue, selected) is True
    assert [t["id"] for t in _assign(host, monkeypatch)] == [child_id]

# --- branches after a consumed exact root grant ---

CHILD = "held-child"


def _child(*, parent="held"):
    child = queue.enqueue_task({
        "id": CHILD, "type": "task", "chat_id": 0, "text": "child work",
        "root_task_id": "held", "parent_task_id": parent, "depth": 1,
        "delegation_role": "subagent",
    })
    assert not child.get("_admission_blocked"), child


def _row(host, task_id):
    return next(t for t in host.pending if t["id"] == task_id)


def _select_child_as(host, requester):
    pushed = []
    sctx = SimpleNamespace(
        DRIVE_ROOT=host.root, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
        WORKERS=workers.WORKERS, sort_pending=queue.sort_pending,
        persist_queue_snapshot=queue.persist_queue_snapshot,
        bridge=SimpleNamespace(push_log=pushed.append))
    eb._handle_budget_resume_child(
        {"task_id": CHILD, "requested_by": requester, "reason": "dir02 branch"}, sctx)
    assert pushed and pushed[-1]["type"] == "budget_resume_child_outcome", pushed
    return pushed[-1]


def _consumed_g1_tree(host, monkeypatch, clock, plan):
    """The real prefix of test_consumed_g1_selection_after_recovery."""
    first = _hung_first_attempt(host, monkeypatch, clock)
    _child()
    pause = request_owner_pause("held", request_id="g1-pause")
    assert pause["ok"], pause
    _park(host, first, plan)
    g1 = queue.resume_budget_paused_task("held")
    assert g1["ok"] and g1["exact_continuation"], g1
    [resumed] = _assign(host, monkeypatch)
    assert resumed["id"] == "held"
    assert _loop_once(host, monkeypatch, resumed, plan=plan,
                      events=_supervisor_events(host))["kind"] == "pause"
    exact = bp.budget_pause_row(host.root, "held")
    assert exact["state"] == bp.STATE_RESUMED and exact["grant"]["consumed_at"]
    assert exact["grant"]["grant_id"] == g1["grant_id"]
    assert eb.budget_hold_fact(_row(host, CHILD))["root_grant_id"] == g1["grant_id"]
    return pause, g1, exact


def test_holding_stop_lends_no_g1_and_owner_resume_rebinds_the_child(host, monkeypatch, clock, reaper):
    plan = _fresh_plan(monkeypatch)
    _pause, g1, exact = _consumed_g1_tree(host, monkeypatch, clock, plan)
    _stop_boot(monkeypatch, returning=False)  # an ordinary holding stop: nothing returns
    root = _row(host, "held")
    hold = eb.budget_hold_fact(root)
    assert hold and hold["reason"] == eb.HOLD_SAVED_WORK and not hold.get("root_grant_id"), hold
    assert "_budget_pause_resume" not in root
    assert eb.live_root_resume_grant(queue, "held", host.root) == {}
    assert bp.budget_pause_row(host.root, "held")["grant"] == exact["grant"], "consumed history is kept"
    assert eb.budget_hold_fact(_row(host, CHILD))["root_grant_id"] == g1["grant_id"]
    refused = _select_child_as(host, "held")
    assert not refused["ok"] and refused["error"] == "root_resume_grant_missing", refused

    assert queue.resume_budget_paused_task("held")["ok"]  # the owner's newer root Resume
    g2 = eb.live_root_resume_grant(queue, "held", host.root)
    assert g2 and g2["grant_id"] != g1["grant_id"]
    selected = _select_child_as(host, "held")
    assert selected["ok"], selected  # current J by source: root_resume_generation_stale (P3)
    child = _row(host, CHILD)
    assert child[eb.SELECTION_KEY]["root_grant_id"] == g2["grant_id"]
    assert eb.budget_resume_dispatch_allowed(queue, child) is True


def test_newer_owner_pause_defeats_projected_g1(host, monkeypatch, clock, reaper):
    plan = _fresh_plan(monkeypatch)
    pause, g1, _exact = _consumed_g1_tree(host, monkeypatch, clock, plan)
    _stop_boot(monkeypatch, returning=True)
    projected = _row(host, "held")[eb.SELECTION_KEY]
    assert projected["grant_id"] == g1["grant_id"], projected
    [running] = _assign(host, monkeypatch)
    assert running["id"] == "held"
    assert _loop_once(host, monkeypatch, running, plan=plan)["kind"] == "working"
    newer = request_owner_pause("held", request_id="newer-than-g1")
    assert newer["ok"] and newer["fence_id"] != pause["fence_id"], newer
    assert eb.live_root_resume_grant(queue, "held", host.root) == {}
    assert not _select_child_as(host, "held")["ok"]
    _park(host, running, plan)
    g3 = queue.resume_budget_paused_task("held")
    assert g3["ok"] and g3["exact_continuation"] and g3["grant_id"] != g1["grant_id"], g3
    # Exact Resume reopens the durable owner fence only at actual consumption.
    assert eb.live_root_resume_grant(queue, "held", host.root) == {}
    [resumed] = _assign(host, monkeypatch)
    assert resumed["id"] == "held"
    _loop_once(host, monkeypatch, resumed, plan=plan, events=_supervisor_events(host))
    assert eb.live_root_resume_grant(queue, "held", host.root)["grant_id"] == g3["grant_id"]
    outcome = _select_child_as(host, "held")
    assert outcome["ok"], outcome
    assert _row(host, CHILD)[eb.SELECTION_KEY]["root_grant_id"] == g3["grant_id"]


def test_consumed_exact_grant_beats_an_older_saved_hold_after_return(host, monkeypatch, clock, reaper):
    plan = _fresh_plan(monkeypatch)
    _hung_first_attempt(host, monkeypatch, clock)
    _stop_boot(monkeypatch, returning=False)  # holding stop
    assert queue.resume_budget_paused_task("held")["ok"]  # owner Resume: G2 on the root's hold
    g2 = eb.live_root_resume_grant(queue, "held", host.root)
    assert g2
    [running] = _assign(host, monkeypatch)
    assert running["id"] == "held"
    _child()
    pause = request_owner_pause("held", request_id="g5-pause")
    assert pause["ok"], pause
    _park(host, running, plan)
    g5 = queue.resume_budget_paused_task("held")
    assert g5["ok"] and g5["exact_continuation"], g5
    [resumed] = _assign(host, monkeypatch)
    assert resumed["id"] == "held"
    _loop_once(host, monkeypatch, resumed, plan=plan, events=_supervisor_events(host))
    exact = bp.budget_pause_row(host.root, "held")
    assert exact["state"] == bp.STATE_RESUMED and exact["grant"]["grant_id"] == g5["grant_id"]
    live = workers.RUNNING["held"]["task"]
    assert eb.SELECTION_KEY not in live, "the new exact handoff supersedes the older selection"
    assert eb.live_root_resume_grant(queue, "held", host.root)["grant_id"] == g5["grant_id"]
    assert eb.budget_hold_fact(_row(host, CHILD))["root_grant_id"] == g5["grant_id"]

    _stop_boot(monkeypatch, returning=True)  # acknowledged automatic return
    returned = _row(host, "held")
    assert returned[eb.SELECTION_KEY]["grant_id"] == g5["grant_id"]
    assert eb.live_root_resume_grant(queue, "held", host.root)["grant_id"] == g5["grant_id"]
    outcome = _select_child_as(host, "held")
    assert outcome["ok"], outcome


def test_owner_wait_return_retirement_keeps_consumed_g1_selection(host, monkeypatch, clock, reaper):
    """Site-level: the preserve_running successor shape (workers.py) plus the restore's
    revocation call (queue_snapshot.py); the planned-restart transport is not driven."""
    from supervisor.budget_resume import revoke_exact_budget_resume

    plan = _fresh_plan(monkeypatch)
    _pause, g1, exact = _consumed_g1_tree(host, monkeypatch, clock, plan)
    with queue._queue_lock:
        meta = workers.RUNNING.pop("held")
        successor = dict(meta["task"])
        successor["_owner_wait_resume"] = {"state": "waiting", "wait_id": "dir02-wait",
                                           "restart_transaction_id": "dir02-tx"}
        workers.PENDING.insert(0, successor)
    assert successor["_budget_pause_resume"]["grant_id"] == g1["grant_id"]
    assert revoke_exact_budget_resume(successor, "restart_before_dispatch") is False
    assert "_budget_pause_resume" not in successor and successor.get("_owner_wait_resume")
    assert bp.budget_pause_row(host.root, "held")["grant"] == exact["grant"]
    assert eb.live_root_resume_grant(queue, "held", host.root).get("grant_id") == g1["grant_id"]  # J: {} (P2)
    outcome = _select_child_as(host, "held")
    assert outcome["ok"], outcome


def test_retry_actor_return_projects_on_its_physical_row(host, monkeypatch, clock, reaper):
    running = _running_actor(host, monkeypatch, clock, reaper, "retry")
    plan = _fresh_plan(monkeypatch)
    assert running["id"] != "held"
    _child(parent=running["id"])
    pause = request_owner_pause(running["id"], request_id="retry-g1")
    assert pause["ok"] and pause["root_task_id"] == "held", pause
    _park(host, running, plan)
    g1 = queue.resume_budget_paused_task(running["id"])
    assert g1["ok"] and g1["exact_continuation"], g1
    [resumed] = _assign(host, monkeypatch)
    assert resumed["id"] == running["id"]
    _loop_once(host, monkeypatch, resumed, plan=plan, events=_supervisor_events(host))
    assert eb.live_root_resume_grant(queue, "held", host.root)["grant_id"] == g1["grant_id"]
    _stop_boot(monkeypatch, returning=True)
    hold = _row(host, running["id"])[eb.SELECTION_KEY]
    assert (hold["root_task_id"], hold["grant_id"]) == ("held", g1["grant_id"]), hold
    assert eb.live_root_resume_grant(queue, "held", host.root)["grant_id"] == g1["grant_id"]
    child = _row(host, CHILD)
    assert eb.budget_hold_fact(child), "return did not select this child"
    outcome = _select_child_as(host, running["id"])
    assert outcome["ok"], outcome
    assert eb.budget_resume_dispatch_allowed(queue, child) is True


@pytest.mark.parametrize("actor", ["root", "retry", "retry_chain"])
def test_holding_stop_owner_resume_mints_grant_on_current_root_actor(host, monkeypatch, clock, reaper, actor):
    running = _running_actor(host, monkeypatch, clock, reaper, actor)
    plan = _fresh_plan(monkeypatch)
    ctx, limit = _loop_ctx(host.root, running["id"], attempt=running["_attempt"])
    ctx.root_task_id = "held"
    ctx.task_metadata = {key: running[key] for key in (
        "root_task_id", "original_task_id", "timeout_retry_from") if running.get(key)}
    ctx.active_model, ctx.context_fit_plan = plan.model, plan
    wc.save_round(limit, "post_batch")
    _child(parent=running["id"])
    _stop_boot(monkeypatch, returning=False)
    root = _row(host, running["id"])
    assert eb.budget_hold_fact(root)["reason"] == eb.HOLD_SAVED_WORK
    assert owner_pause.root_actor(host.root, "held", running["id"])
    assert eb.live_root_resume_grant(queue, "held", host.root) == {}
    resumed = queue.resume_budget_paused_task(running["id"])
    assert resumed["ok"], resumed
    grant = eb.live_root_resume_grant(queue, "held", host.root)
    assert grant and grant["grant_id"], "owner Resume must grant child-selection eligibility to the physical root"
    child = _row(host, CHILD)
    assert eb.budget_hold_fact(child), "root Resume never selects the child"
    assert _select_child_as(host, running["id"])["ok"]
    assert eb.budget_resume_dispatch_allowed(queue, child) is True


@pytest.mark.parametrize("fail_after_observation", [False, True])
def test_unknown_retry_identity_cannot_mint_or_release_a_saved_hold(
        host, monkeypatch, clock, reaper, fail_after_observation):
    import copy
    from ouroboros import task_results

    running = _running_actor(host, monkeypatch, clock, reaper, "retry")
    plan = _fresh_plan(monkeypatch)
    ctx, limit = _loop_ctx(host.root, running["id"], attempt=running["_attempt"])
    ctx.root_task_id = "held"
    ctx.active_model, ctx.context_fit_plan = plan.model, plan
    wc.save_round(limit, "post_batch")
    _stop_boot(monkeypatch, returning=False)
    root = _row(host, running["id"])
    before = copy.deepcopy(root)
    observation = eb.observe_held_budget_selection(queue, running["id"]) if fail_after_observation else None
    if observation is not None:
        assert observation["authority"]["root_actor"] is True
    load = task_results.load_task_result

    def unreadable(drive, task_id, *args, **kwargs):
        if task_id == "held":
            raise OSError("logical root lineage temporarily unreadable")
        return load(drive, task_id, *args, **kwargs)

    monkeypatch.setattr(task_results, "load_task_result", unreadable)
    if observation is None:
        observation = eb.observe_held_budget_selection(queue, running["id"])
    with queue._queue_lock:
        result = eb.select_held_budget_row(queue, root, eb.budget_hold_fact(root),
                                          selected_by="", observation=observation)
    assert result == {"ok": False, "error": "selection_authority_unavailable" if fail_after_observation
                      else "selection_observation_unavailable"}
    assert root == before, "failed identity reads preserve the original hold and source"
