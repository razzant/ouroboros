"""Off-lock pause observers never write over a newer Resume, revocation or pause.

The owner-Pause settlement re-check (``owner_pause_control``) and the cold-sleep
readiness pass (``sleep_wake``) read the durable pause row, observe custody or
readiness OUTSIDE the queue lock, then record their observation. A real owner
Resume (a grant), a grant then its revocation (back to a paused state), or a
newer pause may land during that observation; the write compares the pause id,
state AND grant it read (a warm park's re-read: its wait id and state), and a
lost comparison is a benign stale observation that the next pass redoes. Driven through the real Resume/revoke/assignment
seams, never by rewriting the row by hand, except where a test names a newer
writer explicitly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _fast_hold, _install_queue, _loop_ctx, _mock_pause_observation
from tests.test_owner_wait_pool import pool  # noqa: F401

RUNNING_RUN = [{"run_id": "run-1", "state": "running", "stop_outcome": ""}]


def _quiet_custody(root, task_id, **kw):
    return {"custody_read": "ok", "runs": [], "observed_at": 0.0, "coverage_basis": "test"}


def _owner_park(tmp_path, monkeypatch, task_id, root):
    from ouroboros import budget_pause

    _ctx, limit_ctx = _loop_ctx(tmp_path, task_id)
    _ctx.root_task_id = root
    _fast_hold(monkeypatch, budget_pause)
    _mock_pause_observation(monkeypatch, budget_pause, RUNNING_RUN)
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.enter_owner_pause(limit_ctx)
    budget_pause.end_dispatch_fence(task_id)
    return raised.value.pause


def _sup(tmp_path, queue, workers):
    return SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                           WORKERS=workers.WORKERS, sort_pending=lambda: None,
                           persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None)


def _owner_paused_solo(tmp_path, monkeypatch):
    """A root parked by the owner's Pause while the run it sent still ran."""
    from ouroboros import budget_pause, owner_pause
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from supervisor.events_budget import install_exact_budget_pause
    from supervisor.owner_pause_control import request_owner_pause

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    write_task_result(tmp_path, "solo", STATUS_RUNNING, chat_id=0)
    workers.RUNNING["solo"] = {"task": {"id": "solo", "type": "task", "chat_id": 0, "root_task_id": "solo"},
                               "worker_id": 0, "attempt": 1}
    assert request_owner_pause("solo", request_id="p")["ok"]
    row = _owner_park(tmp_path, monkeypatch, "solo", "solo")
    install_exact_budget_pause(_sup(tmp_path, queue, workers), "solo",
                               budget_pause.exact_pause_marker(row)["checkpoint"])
    parked = budget_pause.budget_pause_row(tmp_path, "solo")
    assert parked["state"] == budget_pause.STATE_PAUSING
    assert parked["settlement"] == owner_pause.SETTLEMENT_EXTERNAL_RUNNING
    return queue, workers


def _competing(tmp_path, queue, kind, task_id):
    """The newer authority that lands while an observer is off the lock."""
    from ouroboros import budget_pause
    from supervisor.budget_resume import revoke_exact_budget_resume

    if kind in {"resume", "grant_then_revoke"}:
        granted = queue.resume_budget_paused_task(task_id)
        assert granted["ok"] is True, granted
        if kind == "grant_then_revoke":
            with queue._queue_lock:
                task = next(item for item in queue.PENDING if item.get("id") == task_id)
                assert revoke_exact_budget_resume(task, "competing_test_revoke") is True
        return granted
    if kind == "newer_pause":
        # A newer pause of the same task, as its own park writer would record it.
        row = budget_pause.budget_pause_row(tmp_path, task_id)
        budget_pause.set_budget_pause(tmp_path, task_id, {**row, "pause_id": "pause-newer"},
                                      expected_pause_id=str(row["pause_id"]))
        return {}
    assert kind == "none"
    return {}


@pytest.mark.parametrize("kind", ["none", "resume", "grant_then_revoke", "newer_pause"])
def test_owner_pause_settlement_never_overwrites_a_newer_resume_or_pause(tmp_path, monkeypatch, kind):
    from ouroboros import budget_pause, owner_pause
    from supervisor import owner_pause_control

    queue, workers = _owner_paused_solo(tmp_path, monkeypatch)
    before = budget_pause.budget_pause_row(tmp_path, "solo")
    landed: dict = {}

    def observe(root, task_id, **kw):
        if kw.get("reason") == "owner_pause_settlement_check" and "done" not in landed:
            landed["done"] = True
            landed["granted"] = _competing(tmp_path, queue, kind, task_id)
        return _quiet_custody(root, task_id, **kw)

    monkeypatch.setattr(budget_pause, "observe_task_runs", observe)
    monkeypatch.setattr(owner_pause_control, "_LAST_SETTLE_CHECK", {})
    settled = owner_pause_control.settle_requested_owner_pauses(queue, now=100.0)
    row = budget_pause.budget_pause_row(tmp_path, "solo")
    if kind == "none":
        # The fix is not vacuous: an uncontested observation still settles the tree.
        assert settled == ["solo"]
        assert row["settlement"] == owner_pause.SETTLEMENT_SETTLED and row["state"] == before["state"]
        assert owner_pause.read_fence(tmp_path, "solo")["state"] == owner_pause.FENCE_PAUSED
        return
    assert "settlement_observed_at" not in row, "the stale observation wrote nothing"
    if kind == "newer_pause":
        assert row["pause_id"] == "pause-newer"
        assert row["settlement"] == owner_pause.SETTLEMENT_EXTERNAL_RUNNING
        return
    grant_id = landed["granted"]["grant_id"]
    assert row["grant"]["grant_id"] == grant_id and row["resume_generation"] == 1
    if kind == "grant_then_revoke":
        assert row["state"] == budget_pause.STATE_PAUSED and row["grant"]["revoked_at"]
        assert owner_pause.read_fence(tmp_path, "solo")["state"] != owner_pause.FENCE_RELEASED
        return
    # The owner's Resume survives: the queue carrier still matches the durable
    # grant, and real assignment hands exactly that grant to a worker once.
    assert row["state"] == budget_pause.STATE_RESUME_GRANTED and not row["grant"].get("revoked_at")
    assert workers.PENDING[0]["_budget_pause_resume"]["grant_id"] == grant_id
    sent = []
    workers.WORKERS[0] = SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                         in_q=SimpleNamespace(put=lambda task: sent.append(dict(task))))
    workers.assign_tasks()
    assert [(task["id"], task["_budget_pause_resume"]["grant_id"]) for task in sent] == [("solo", grant_id)]


def test_one_superseded_member_neither_blocks_its_sibling_nor_the_tree_census(tmp_path, monkeypatch):
    from ouroboros import budget_pause, owner_pause
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from supervisor import owner_pause_control
    from supervisor.events_budget import install_exact_budget_pause
    from supervisor.owner_pause_control import request_owner_pause

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root-1", STATUS_RUNNING, chat_id=0)
    workers.RUNNING["root-1"] = {"task": {"id": "root-1", "type": "task", "chat_id": 0, "root_task_id": "root-1"},
                                 "worker_id": 0, "attempt": 1}
    for index, child in enumerate(("child-a", "child-b"), start=1):
        write_task_result(tmp_path, child, STATUS_RUNNING, chat_id=0)
        workers.RUNNING[child] = {"task": {"id": child, "type": "task", "chat_id": 0, "root_task_id": "root-1",
                                           "parent_task_id": "root-1", "delegation_role": "subagent"},
                                  "worker_id": index, "attempt": 1}
    assert request_owner_pause("root-1", request_id="p")["ok"]
    sup = _sup(tmp_path, queue, workers)
    for member in ("child-a", "child-b", "root-1"):
        row = _owner_park(tmp_path, monkeypatch, member, "root-1")
        install_exact_budget_pause(sup, member, budget_pause.exact_pause_marker(row)["checkpoint"])
    assert owner_pause.read_fence(tmp_path, "root-1")["state"] == owner_pause.FENCE_REQUESTED

    competing = {"grant_id": "g-newer", "granted_at": "later"}

    def observe(root, task_id, **kw):
        if kw.get("reason") == "owner_pause_settlement_check" and task_id == "child-a" and "seen" not in competing:
            competing["seen"] = True
            # A newer grant written and revoked meanwhile, back to the same state:
            # only the grant identity tells this observation is stale.
            current = budget_pause.budget_pause_row(tmp_path, "child-a")
            budget_pause.set_budget_pause(
                tmp_path, "child-a", {**current, "grant": {**competing, "revoked_at": "later"},
                                      "resume_generation": 1},
                expected_pause_id=current["pause_id"], expected_state=current["state"], expected_grant_id="")
        return _quiet_custody(root, task_id, **kw)

    census: list = []
    real_refresh = owner_pause_control.refresh_owner_pause_tree
    monkeypatch.setattr(owner_pause_control, "refresh_owner_pause_tree",
                        lambda root_id: census.append(root_id) or real_refresh(root_id))
    monkeypatch.setattr(budget_pause, "observe_task_runs", observe)
    monkeypatch.setattr(owner_pause_control, "_LAST_SETTLE_CHECK", {})
    assert owner_pause_control.settle_requested_owner_pauses(queue, now=100.0) == []
    a, b = (budget_pause.budget_pause_row(tmp_path, member) for member in ("child-a", "child-b"))
    assert a["grant"]["grant_id"] == "g-newer" and a["resume_generation"] == 1
    assert a["settlement"] == owner_pause.SETTLEMENT_EXTERNAL_RUNNING, "the lost write was dropped, not retried blind"
    assert b["settlement"] == owner_pause.SETTLEMENT_SETTLED, "the sibling still settled in the same pass"
    assert census == ["root-1"], "the tree's current-authority census still ran"
    assert owner_pause.read_fence(tmp_path, "root-1")["state"] == owner_pause.FENCE_REQUESTED

    # The next pass re-observes the member under its current identity and settles the tree.
    assert owner_pause_control.settle_requested_owner_pauses(queue, now=110.0) == ["root-1"]
    a = budget_pause.budget_pause_row(tmp_path, "child-a")
    assert a["settlement"] == owner_pause.SETTLEMENT_SETTLED and a["grant"]["grant_id"] == "g-newer"
    assert owner_pause.read_fence(tmp_path, "root-1")["state"] == owner_pause.FENCE_PAUSED


def _cold_sleeper(tmp_path, monkeypatch):
    from tests.test_model_sleep import _cold_park, _result

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    _result(tmp_path, "peer-a")
    _cold_park(tmp_path, monkeypatch, workers, senders=["peer-a"])
    return queue, workers


@pytest.mark.parametrize("kind", ["none", "resume", "grant_then_revoke", "newer_pause"])
def test_sleep_readiness_never_overwrites_a_newer_resume_or_pause(tmp_path, monkeypatch, kind):
    from ouroboros import budget_pause, model_sleep
    from supervisor.sleep_wake import wake_ready_sleepers

    queue, workers = _cold_sleeper(tmp_path, monkeypatch)
    landed: dict = {}

    def ready(ctx, sleep):
        if "done" not in landed:
            landed["done"] = True
            landed["granted"] = _competing(tmp_path, queue, kind, "sleeper")
        return "mail:peer-a"

    monkeypatch.setattr(model_sleep, "wake_reason", ready)
    outcomes = wake_ready_sleepers(queue)
    row = budget_pause.budget_pause_row(tmp_path, "sleeper")
    if kind == "none":
        assert outcomes[0]["ok"] is True and row["sleep_ready"]["reason"] == "mail:peer-a"
        assert row["grant"]["selected_by"] == "sleep_wake" and row["resume_generation"] == 1
        return
    assert outcomes == [], "a superseded observation grants nothing in this pass"
    assert "sleep_ready" not in row
    if kind == "newer_pause":
        assert row["pause_id"] == "pause-newer" and row["state"] == budget_pause.STATE_PAUSED
        return
    grant_id = landed["granted"]["grant_id"]
    assert row["grant"]["grant_id"] == grant_id and row["resume_generation"] == 1
    if kind == "resume":
        assert row["state"] == budget_pause.STATE_RESUME_GRANTED and row["grant"]["selected_by"] == "owner"
        assert workers.PENDING[0]["_budget_pause_resume"]["grant_id"] == grant_id
        return
    # Grant then revoke returned the SAME paused state; the revocation and its
    # generation survive, and the next pass wakes it under a HIGHER generation.
    assert row["state"] == budget_pause.STATE_PAUSED and row["grant"]["revoked_at"]
    again = wake_ready_sleepers(queue)
    assert again[0]["ok"] is True and again[0]["grant_generation"] == 2
    row = budget_pause.budget_pause_row(tmp_path, "sleeper")
    assert row["grant"]["grant_id"] != grant_id and row["resume_generation"] == 2


def _direct_warm_author(tmp_path, monkeypatch):
    """A direct root the owner Paused while the run it sent still ran: the actor parks warm."""
    from ouroboros import delegate_custody as dc
    from ouroboros import owner_wait
    from ouroboros.gateways import claudexor as gw
    from ouroboros.model_wait import TaskModelWait
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from supervisor.owner_pause_control import request_owner_pause
    from tests.test_direct_chat_turn_owner_control import _live_chat_agent

    queue, state, _workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    write_task_result(tmp_path, "author", STATUS_RUNNING, chat_id=0)
    _live_chat_agent(monkeypatch, task_id="author")
    dc.record_started(tmp_path, dc.RunCustody(
        run_id="run-own", task_id="author", route_id="r", model="m", project_id="p",
        project_owned=False, root_task_id="author", ledger_root=str(tmp_path)))
    remote = {"state": "running"}

    class Daemon:
        def handshake(self, **_kw):
            return {"compatible": True}

        def cancel_run(self, run_id, reason=""):
            return {"accepted": True, "status": "accepted"}

        def get_run(self, run_id, **_kw):
            return {"lastSeq": 3, "summary": {"state": remote["state"], "spendUsd": 0.0}}

        def close(self):
            return None

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: Daemon())
    assert request_owner_pause("author", request_id="press")["state"] == "requested"
    ctx, limit = _loop_ctx(tmp_path, "author", direct=True)
    ctx.pending_events, ctx.current_chat_id = [], 0
    waiter = TaskModelWait(task={"id": "author", "budget_drive_root": str(tmp_path)},
                           drive_root=tmp_path, event_queue=None, worker_slot_held=False)
    waiter.tool_context, ctx.model_wait_context = ctx, waiter
    ctx.owner_wait_callback = owner_wait.direct_owner_wait
    return queue, remote, ctx, limit


@pytest.mark.parametrize("resume_lands", [False, True])
def test_warm_park_reread_never_restores_a_resumed_direct_stack(tmp_path, monkeypatch, resume_lands):
    """The settle tick re-reads a warm direct park's runs off-lock while the actor thread
    takes the owner's real Resume, writes ``resumed`` and saves newer working state. The
    stale observation must not put ``waiting`` back: Restart would then retain the old
    park's source over the newer cognition. Uncontested, the waiting park records it."""
    import threading
    import time

    from ouroboros import budget_pause, owner_pause, owner_wait, working_checkpoint
    from ouroboros.external_runs import EXTERNAL_STOP_CONFIRMED, EXTERNAL_STOP_REQUESTED
    from ouroboros.task_results import load_task_result
    from supervisor import owner_pause_control
    from supervisor.budget_resume import resume_warm_owner_pause_root
    from supervisor.restart_retention import RETAIN_OWNER_PAUSE, pause_retention

    queue, remote, ctx, limit = _direct_warm_author(tmp_path, monkeypatch)
    actor, real_observe, real_sleep = threading.current_thread(), budget_pause.observe_task_runs, time.sleep
    observing, proceed, box = threading.Event(), threading.Event(), {}

    def observe(root, task_id, **kw):  # the tick's custody read, held while the actor moves on
        observed = real_observe(root, task_id, **kw)
        if kw.get("reason") == "owner_pause_settlement_check":
            observing.set()
            assert proceed.wait(10)
        return observed

    def tick():
        box["settled"] = owner_pause_control.settle_requested_owner_pauses(queue, now=100.0)

    def poll(seconds):  # the parked direct actor's poll; other threads sleep for real
        if threading.current_thread() is not actor or "tick" in box:
            return real_sleep(min(seconds, 0.01))
        box["parked"] = load_task_result(tmp_path, "author")["owner_wait"]
        remote["state"] = "cancelled"  # the run obeys the stop the park requested
        box["tick"] = threading.Thread(target=tick, daemon=True)
        box["tick"].start()
        assert observing.wait(10)
        if not resume_lands:
            proceed.set()
            box["tick"].join(10)
            box["recorded"] = load_task_result(tmp_path, "author")["owner_wait"]
            box["retention"] = pause_retention(tmp_path, "author", 1)
        box["resume"] = resume_warm_owner_pause_root("author")
        return None

    monkeypatch.setattr(budget_pause, "observe_task_runs", observe)
    monkeypatch.setattr(owner_pause_control, "_LAST_SETTLE_CHECK", {})
    monkeypatch.setattr(owner_wait.time, "sleep", poll)
    try:
        cause = owner_wait.park_owner_pause_warm(limit, ctx, fence=owner_pause.read_fence(tmp_path, "author"),
                                                 detached=[])
        assert box["resume"]["ok"] and cause == "control:owner_resume", (box, cause)
        # The resumed stack works on: its newer cognition lands in the rolling checkpoint.
        limit.messages.append({"role": "tool", "tool_call_id": "call_b", "content": "done b after resume"})
        limit.round_idx = 5
        assert working_checkpoint.save_round(limit, "post_batch")
    finally:
        proceed.set()
        if "tick" in box:
            box["tick"].join(10)
    parked = box["parked"]
    assert parked["state"] == "waiting" and parked["reason"] == "owner_pause"
    assert [run["state"] for run in parked["owner_pause"]["external_runs"]["runs"]] == [EXTERNAL_STOP_REQUESTED]
    row = load_task_result(tmp_path, "author")["owner_wait"]
    if not resume_lands:
        # Not vacuous: the valid waiting park records the fresh facts, the tree settles,
        # and that park is Restart's retained source until the owner's Resume.
        recorded = box["recorded"]
        assert recorded["state"] == "waiting" and recorded["wait_id"] == parked["wait_id"]
        assert [run["state"] for run in recorded["owner_pause"]["external_runs"]["runs"]] == [
            EXTERNAL_STOP_CONFIRMED]
        assert box["settled"] == ["author"] and box["retention"] == RETAIN_OWNER_PAUSE
        assert row["state"] == "resumed" and row["resume_reason"] == "control:owner_resume"
        return
    handoff = working_checkpoint.prepare_recovery(tmp_path, "author", from_attempt=1, cause="restart")
    consumers = {"state": row["state"], "warm_paused": owner_pause_control._warm_paused_direct_turn(tmp_path, "author"),
                 "restart_retains": pause_retention(tmp_path, "author", 1), "recovers": handoff.get("source_kind")}
    assert consumers == {"state": "resumed", "warm_paused": False, "restart_retains": "",
                         "recovers": working_checkpoint.SOURCE_WORKING}, consumers
    assert row["wait_id"] == parked["wait_id"] and row["resume_reason"] == "control:owner_resume"
    assert row["owner_pause"]["external_runs"] == parked["owner_pause"]["external_runs"], "the stale read wrote nothing"
    recovered = working_checkpoint.load_recovery(ctx, handoff)
    assert recovered["messages"][-1]["content"] == "done b after resume" and recovered["round_idx"] == 5


@pytest.mark.parametrize("grant_lands", [False, True])
def test_warm_park_reread_never_rewrites_a_granted_pool_mirror(pool, monkeypatch, grant_lands):  # noqa: F811
    """The pooled twin. Today the grant and this re-read share the supervisor loop thread;
    the same identity+state comparison keeps the durable row and its RUNNING mirror on the
    real grant should they interleave, and never drops a live mirror's own fields."""
    from ouroboros import budget_pause
    from ouroboros.task_results import load_task_result
    from supervisor import owner_pause_control, queue, worker_owner_wait, workers
    from supervisor.restart_retention import RETAIN_OWNER_PAUSE, pause_retention

    requested = {"custody_read": "ok", "runs": [{"run_id": "run-own", "state": "stop_requested"}]}
    confirmed = {"custody_read": "ok", "runs": [{"run_id": "run-own", "state": "stop_confirmed"}]}
    wait = {**pool.wait, "quiz_id": "", "reason": "owner_pause", "owner_pause": {"external_runs": requested}}
    worker_owner_wait.handle_owner_wait({**pool.event, "checkpoint": wait}, workers)
    assert pool.original.in_q.get_nowait()["phase"] == "parked"
    with queue._queue_lock:
        snapshot = {**pool.meta, "owner_wait": dict(pool.meta["owner_wait"])}  # the tick's RUNNING copy
    assert owner_pause_control.warm_paused_member(snapshot)

    def observe(root, task_id, **kw):
        # The stack's Resume request lands meanwhile; with ``grant_lands`` the pool grants it.
        worker_owner_wait.handle_owner_wait(
            {**pool.event, "phase": "resume", "resume_reason": "control:owner_resume"}, workers)
        if grant_lands:
            worker_owner_wait.maintain_owner_wait_capacity()
            assert pool.original.in_q.get_nowait()["phase"] == "resume_granted"
        return confirmed

    monkeypatch.setattr(budget_pause, "observe_task_runs", observe)
    owner_pause_control._reread_warm_member(queue, "owner", snapshot)
    durable, live = load_task_result(pool.root, "owner")["owner_wait"], pool.meta["owner_wait"]
    if not grant_lands:
        assert durable["state"] == live["state"] == "waiting"
        assert durable["owner_pause"]["external_runs"] == live["owner_pause"]["external_runs"] == confirmed
        assert live["resume_reason"] == "control:owner_resume", "the mirror's pending wake survives"
        assert pause_retention(pool.root, "owner", 3) == RETAIN_OWNER_PAUSE
        return
    consumers = {"durable": durable["state"], "mirror": live["state"],
                 "warm_paused": owner_pause_control.warm_paused_member(pool.meta),
                 "restart_retains": pause_retention(pool.root, "owner", 3)}
    assert consumers == {"durable": "resumed", "mirror": "resumed", "warm_paused": False,
                         "restart_retains": ""}, consumers
    assert durable["owner_pause"]["external_runs"] == live["owner_pause"]["external_runs"] == requested


@pytest.mark.parametrize("landed", ["newer_park", "terminal"])
def test_warm_park_reread_drops_only_a_superseded_row(pool, monkeypatch, landed):  # noqa: F811
    """A newer park is a benign stale observation; a genuine refusal still reaches the tick's warning."""
    from ouroboros import budget_pause
    from ouroboros.owner_wait import OwnerWaitSuperseded, set_owner_wait
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor import owner_pause_control, queue, worker_owner_wait, workers

    requested = {"custody_read": "ok", "runs": [{"run_id": "run-own", "state": "stop_requested"}]}
    wait = {**pool.wait, "quiz_id": "", "reason": "owner_pause", "owner_pause": {"external_runs": requested}}
    worker_owner_wait.handle_owner_wait({**pool.event, "checkpoint": wait}, workers)
    assert pool.original.in_q.get_nowait()["phase"] == "parked"
    snapshot = {**pool.meta, "owner_wait": dict(pool.meta["owner_wait"])}

    def observe(root, task_id, **kw):
        if landed == "newer_park":
            set_owner_wait(pool.root, "owner", {**snapshot["owner_wait"], "wait_id": "newer"}, expected_wait_id="wait")
        else:
            write_task_result(pool.root, "owner", "failed", result="ended meanwhile")
        return {"custody_read": "ok", "runs": [{"run_id": "run-own", "state": "stop_confirmed"}]}

    monkeypatch.setattr(budget_pause, "observe_task_runs", observe)
    if landed == "terminal":
        with pytest.raises(ValueError, match="terminal") as raised:
            owner_pause_control._reread_warm_member(queue, "owner", snapshot)
        assert not isinstance(raised.value, OwnerWaitSuperseded)
    else:
        owner_pause_control._reread_warm_member(queue, "owner", snapshot)
        durable = load_task_result(pool.root, "owner")["owner_wait"]
        assert durable["wait_id"] == "newer" and durable["owner_pause"]["external_runs"] == requested
    assert pool.meta["owner_wait"]["owner_pause"]["external_runs"] == requested, "the mirror took nothing"
