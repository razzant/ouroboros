"""Owner Batch4 (5A): the owner's Pause of a whole tree, through its REAL consumers.

The accept step (``supervisor/owner_pause_control.py``), every launch family's
handoff (a real ``ToolRegistry`` tool body, the model send's
``require_physical_dispatch_window``, a pre-dispatch model wait), the member's
safe boundary, the supervisor park, the Resume and a Restart are driven
directly — never the fence predicate alone.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _fast_hold, _install_queue, _loop_ctx, _mock_pause_observation


def _live_tree(tmp_path, workers):
    """A RUNNING root, its RUNNING child and a never-started PENDING child."""
    from ouroboros.task_results import STATUS_RUNNING, STATUS_SCHEDULED, write_task_result

    write_task_result(tmp_path, "root-1", STATUS_RUNNING, chat_id=0)
    write_task_result(tmp_path, "child-run", STATUS_RUNNING, chat_id=0)
    write_task_result(tmp_path, "child-new", STATUS_SCHEDULED, chat_id=0)
    workers.RUNNING["root-1"] = {"task": {"id": "root-1", "type": "task", "chat_id": 0,
                                          "root_task_id": "root-1"}, "worker_id": 0, "attempt": 1}
    workers.RUNNING["child-run"] = {"task": {
        "id": "child-run", "type": "task", "chat_id": 0, "root_task_id": "root-1",
        "parent_task_id": "root-1", "delegation_role": "subagent"}, "worker_id": 1, "attempt": 1}
    workers.PENDING.append({"id": "child-new", "type": "task", "chat_id": 0, "_attempt": 1,
                            "root_task_id": "root-1", "parent_task_id": "root-1",
                            "delegation_role": "subagent"})


def _mail_kinds(root, task_id):
    from ouroboros.owner_mailbox import _mailbox_path

    path = _mailbox_path(root, task_id)
    return [json.loads(line)["kind"] for line in path.read_text().splitlines()] if path.exists() else []


def test_accept_writes_the_durable_fence_then_latches_the_queue_and_wakes_live_members(tmp_path, monkeypatch):
    from ouroboros.owner_pause import FENCE_REQUESTED, read_fence
    from supervisor.owner_pause_control import request_owner_pause

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _live_tree(tmp_path, workers)

    ack = request_owner_pause("root-1", request_id="press-1")

    assert ack["ok"] is True and ack["state"] == FENCE_REQUESTED and ack["duplicate"] is False
    fence = read_fence(tmp_path, "root-1")
    assert fence["state"] == FENCE_REQUESTED and fence["fence_id"] == ack["fence_id"]
    latch = queue.BUDGET_ROOT_FENCES["root-1"]
    assert latch["cause"] == "owner_pause" and latch["fence_id"] == ack["fence_id"]
    snap = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())
    assert snap["budget_root_fences"][0]["cause"] == "owner_pause"
    assert sorted(ack["members"]) == ["child-run", "root-1"]
    assert _mail_kinds(tmp_path, "root-1") == ["owner_pause"] == _mail_kinds(tmp_path, "child-run")
    assert _mail_kinds(tmp_path, "child-new") == [], "a never-started member is fenced, not woken"

    # A descendant a delegate event drains NOW is refused, typed as the owner's Pause.
    admitted = queue.enqueue_task({"id": "child-late", "type": "task", "chat_id": 0,
                                   "root_task_id": "root-1", "parent_task_id": "root-1"})
    assert admitted["_admission_blocked"] == "root_owner_paused"
    # The same press retried is the same Pause.
    again = request_owner_pause("root-1", request_id="press-1")
    assert again["ok"] is True and again["duplicate"] is True and again["fence_id"] == ack["fence_id"]


def test_accept_refuses_a_child_a_stopping_root_and_an_unknown_task_without_effects(tmp_path, monkeypatch):
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.owner_pause import read_fence
    from supervisor.owner_pause_control import request_owner_pause

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _live_tree(tmp_path, workers)
    assert request_owner_pause("child-run", request_id="r")["error"] == "not_a_root_task"
    assert request_owner_pause("nobody", request_id="r")["error"] == "task_not_live"
    request_cancel(tmp_path, "root-1", reason="owner stop", source="owner", requested_by="owner")
    assert request_owner_pause("root-1", request_id="r")["error"] == "cancel_pending"
    assert read_fence(tmp_path, "root-1") == {} and "root-1" not in queue.BUDGET_ROOT_FENCES


def test_every_launch_family_refuses_after_the_fence_as_not_started(tmp_path, monkeypatch):
    """Tool body, model send and pre-dispatch wait read the SAME durable fence
    at their handoff; before it they pass, after it nothing starts. A sent
    model operation's result wait is never interrupted (control_reason)."""
    from ouroboros import owner_pause
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted, require_physical_dispatch_window
    from ouroboros.loop_tool_execution import _execute_single_tool
    from ouroboros.model_wait import ModelWaitInterrupted, TaskModelWait, propagate_model_control
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.usage_accounting import UsageScope, usage_scope
    from supervisor.owner_pause_control import request_owner_pause

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _live_tree(tmp_path, workers)
    repo = tmp_path / "repo"
    repo.mkdir()
    (tmp_path / "logs").mkdir(exist_ok=True)
    tools = ToolRegistry(repo_dir=repo, drive_root=tmp_path)
    tools._ctx.task_id, tools._ctx.root_task_id = "child-run", "root-1"
    tools._ctx.budget_drive_root = tmp_path
    ran: list = []
    tools.override_handler("knowledge_read", lambda *_a, **_k: ran.append("body") or "ok")
    call = {"id": "call-1", "function": {"name": "knowledge_read", "arguments": json.dumps({"topic": "x"})}}
    wait = TaskModelWait(task={"id": "child-run", "root_task_id": "root-1", "budget_drive_root": str(tmp_path)},
                         drive_root=tmp_path, event_queue=None, worker_slot_held=True)
    scope = UsageScope(drive_root=tmp_path, task_id="child-run", root_task_id="root-1")

    first = _execute_single_tool(tools, call, tmp_path / "logs", "child-run")
    assert ran == ["body"] and "NOT STARTED" not in str(first.get("result"))
    with usage_scope(scope):
        require_physical_dispatch_window()
    assert wait.pre_dispatch_pause() is None

    assert request_owner_pause("root-1", request_id="press")["ok"] is True

    refused = _execute_single_tool(tools, {**call, "id": "call-2"}, tmp_path / "logs", "child-run")
    assert ran == ["body"], "the tool body never started after the fence"
    assert "NOT STARTED" in str(refused["result"])
    assert refused["result_meta"]["tool_result_code"] == "OWNER_PAUSE_NOT_STARTED"
    with usage_scope(scope), pytest.raises(PhysicalDispatchInterrupted) as raised:
        require_physical_dispatch_window()
    assert raised.value.control_reason == owner_pause.RAIL_OWNER_PAUSE
    with pytest.raises(ModelWaitInterrupted) as translated:
        propagate_model_control(raised.value)
    assert translated.value.control_reason == owner_pause.RAIL_OWNER_PAUSE
    assert wait.pre_dispatch_pause() == owner_pause.RAIL_OWNER_PAUSE
    # A SENT operation's result wait polls control_reason: never an owner pause.
    assert wait.control_reason() is None
    # A task outside the tree is untouched by this fence.
    with usage_scope(UsageScope(drive_root=tmp_path, task_id="other", root_task_id="other")):
        require_physical_dispatch_window()


def test_the_boundary_saves_an_exact_owner_pause_buying_no_round_and_observing_runs(tmp_path, monkeypatch):
    """No model call is made (the limit context has no LLM); the member's own
    delegated run is stop-requested under the owner Pause's policy (owner
    2026-10-07 fork 1 = A; its reviewers' runs are spared by source), and while
    that stop is unconfirmed the saved pause is marked unsettled rather than
    claimed as a clean Paused."""
    from ouroboros import budget_pause, owner_pause
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "solo", STATUS_RUNNING)
    owner_pause.install_fence(tmp_path, "solo", request_id="p")
    ctx, limit_ctx = _loop_ctx(tmp_path, "solo")
    _fast_hold(monkeypatch, budget_pause)
    seen: list = []
    real = budget_pause.observe_task_runs

    def observe(root, task_id, **kw):
        seen.append(kw)
        if kw.get("reason") != "budget_pause_uncovered_cost":
            return real(root, task_id, **kw)
        return {"runs": [{"run_id": "run-9", "state": "running", "stop_outcome": ""}],
                "observed_at": 0.0, "custody_read": "ok", "coverage_basis": "test"}

    monkeypatch.setattr(budget_pause, "observe_task_runs", observe)
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.enter_owner_pause(limit_ctx)
    budget_pause.end_dispatch_fence("solo")
    row = raised.value.pause
    assert row["reason"] == "owner" and row["rail"] == owner_pause.RAIL_OWNER_PAUSE
    assert row["settlement"] == owner_pause.SETTLEMENT_EXTERNAL_RUNNING
    assert row["owner_fence_id"] == owner_pause.read_fence(tmp_path, "solo")["fence_id"]
    assert seen and all(kw.get("stop_policy") == budget_pause.STOP_POLICY_TASK_OWNED for kw in seen)
    assert limit_ctx.accumulated_usage["reason_code"] == "owner_paused"
    marker = budget_pause.exact_pause_marker(row)
    assert marker["reason"] == "owner" and marker["settlement"] == owner_pause.SETTLEMENT_EXTERNAL_RUNNING


def _owner_park(tmp_path, monkeypatch, task_id, runs, root="root-1"):
    from ouroboros import budget_pause

    ctx, limit_ctx = _loop_ctx(tmp_path, task_id)
    ctx.root_task_id = root
    _fast_hold(monkeypatch, budget_pause)
    _mock_pause_observation(monkeypatch, budget_pause, runs)
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.enter_owner_pause(limit_ctx)
    budget_pause.end_dispatch_fence(task_id)
    return raised.value.pause


def test_the_tree_turns_paused_only_when_no_member_runs_and_sent_work_settled(tmp_path, monkeypatch):
    from ouroboros import budget_pause, owner_pause
    from supervisor.events_budget import install_exact_budget_pause
    from supervisor.owner_pause_control import request_owner_pause

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _live_tree(tmp_path, workers)
    workers.PENDING[:] = []
    assert request_owner_pause("root-1", request_id="p")["ok"]
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                          WORKERS=workers.WORKERS, sort_pending=lambda: None,
                          persist_queue_snapshot=lambda reason="": True, bridge=None)

    child = _owner_park(tmp_path, monkeypatch, "child-run",
                        [{"run_id": "run-1", "state": "running", "stop_outcome": ""}])
    install_exact_budget_pause(ctx, "child-run", budget_pause.exact_pause_marker(child)["checkpoint"])
    assert budget_pause.budget_pause_row(tmp_path, "child-run")["state"] == budget_pause.STATE_PAUSING
    assert owner_pause.read_fence(tmp_path, "root-1")["state"] == owner_pause.FENCE_REQUESTED

    root = _owner_park(tmp_path, monkeypatch, "root-1", [])
    install_exact_budget_pause(ctx, "root-1", budget_pause.exact_pause_marker(root)["checkpoint"])
    assert budget_pause.budget_pause_row(tmp_path, "root-1")["state"] == budget_pause.STATE_PAUSED
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    phase = lambda: next(row["phase"] for row in _chat_activities_snapshot_safe(tmp_path)
                         if row["activity_id"] == "root-1")
    assert phase() == "budget_pausing"
    # Nothing runs any more, but the child's sent run still does: not a clean Paused.
    assert owner_pause.read_fence(tmp_path, "root-1")["state"] == owner_pause.FENCE_REQUESTED

    # With every member parked, only the assignment tick's re-check can notice the
    # member's stopped run ending; it re-reads the stop the park issued, never a new one.
    from supervisor import owner_pause_control
    from supervisor.owner_pause_control import settle_requested_owner_pauses

    monkeypatch.setattr(owner_pause_control, "_LAST_SETTLE_CHECK", {})
    observed: list = []

    def custody(_root, task_id, *, runs, **kwargs):
        observed.append((task_id, kwargs.get("stop_policy"), "prior" in kwargs))
        return {"custody_read": "ok", "runs": runs}

    monkeypatch.setattr(budget_pause, "observe_task_runs",
                        lambda root, task_id, **kw: custody(root, task_id, runs=[{"run_id": "run-1"}], **kw))
    assert settle_requested_owner_pauses(queue, now=100.0) == []
    assert observed == [("child-run", budget_pause.STOP_POLICY_TASK_OWNED, True)]
    monkeypatch.setattr(budget_pause, "observe_task_runs",
                        lambda root, task_id, **kw: custody(root, task_id, runs=[], **kw))
    assert settle_requested_owner_pauses(queue, now=101.0) == [], "throttled per root"
    assert settle_requested_owner_pauses(queue, now=110.0) == ["root-1"]
    assert budget_pause.budget_pause_row(tmp_path, "child-run")["settlement"] == owner_pause.SETTLEMENT_SETTLED
    assert owner_pause.read_fence(tmp_path, "root-1")["state"] == owner_pause.FENCE_PAUSED
    assert phase() == "budget_paused"
    # The latch kept its cause through the members' root-scoped parks.
    assert queue.BUDGET_ROOT_FENCES["root-1"]["cause"] == "owner_pause"


def test_resume_observes_without_stop_and_the_fence_reopens_where_the_root_starts(tmp_path, monkeypatch):
    """The Resume re-reads custody WITHOUT a stop request; the fence stays closed
    after the grant and reopens only when the resumed root consumes it. A grant
    revoked before it ran (a restart) leaves the tree fenced."""
    from ouroboros import budget_pause, owner_pause
    from supervisor.events_budget import install_exact_budget_pause

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    write_task_result(tmp_path, "solo", STATUS_RUNNING, chat_id=0)
    workers.RUNNING["solo"] = {"task": {"id": "solo", "type": "task", "chat_id": 0, "root_task_id": "solo"},
                               "worker_id": 0, "attempt": 1}
    from supervisor.owner_pause_control import request_owner_pause

    assert request_owner_pause("solo", request_id="p")["ok"]
    row = _owner_park(tmp_path, monkeypatch, "solo", [], root="solo")
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                          WORKERS=workers.WORKERS, sort_pending=lambda: None,
                          persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None)
    install_exact_budget_pause(ctx, "solo", budget_pause.exact_pause_marker(row)["checkpoint"])
    assert owner_pause.read_fence(tmp_path, "solo")["state"] == owner_pause.FENCE_PAUSED
    stops: list = []
    real = budget_pause.observe_task_runs
    monkeypatch.setattr(budget_pause, "observe_task_runs",
                        lambda root, task_id, **kw: stops.append(kw.get("request_stop")) or real(root, task_id, **kw))

    granted = queue.resume_budget_paused_task("solo")
    assert granted["ok"] is True and stops and all(stop is False for stop in stops)
    assert owner_pause.read_fence(tmp_path, "solo")["state"] == owner_pause.FENCE_PAUSED, \
        "the grant alone reopens nothing"

    # Actual assignment must hand the root its grant through the still-closed
    # fence; only consumption reopens ordinary tools and sends.
    sent = []
    workers.WORKERS[0] = SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                         in_q=SimpleNamespace(put=lambda task: sent.append(dict(task))))
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["solo"]
    assert owner_pause.read_fence(tmp_path, "solo")["state"] == owner_pause.FENCE_PAUSED
    handoff = sent[0]["_budget_pause_resume"]
    loop_ctx, limit_ctx = _loop_ctx(tmp_path, "solo")
    loop_ctx.budget_pause_resume = handoff
    state_blob = budget_pause.load_budget_pause(loop_ctx)
    monkeypatch.setattr("ouroboros.owner_wait.rebind_restored_route", lambda *_a, **_k: (None, "max"))
    monkeypatch.setattr("ouroboros.owner_wait.restore_continuation_state", lambda *_a, **_k: None)
    budget_pause.resume_paused_loop(limit_ctx.tools, state_blob, list(limit_ctx.messages), {}, {}, set(),
                                    budget_remaining_usd=5.0)
    assert owner_pause.read_fence(tmp_path, "solo")["state"] == owner_pause.FENCE_RELEASED
    assert owner_pause.member_fence(loop_ctx) == {}


def test_an_owner_paused_tree_survives_restart_as_itself(tmp_path, monkeypatch):
    """A Restart keeps the saved member and the latch's cause: the next boot's
    model sends are gated by the owner fence, never refused as a MONEY latch
    (which would turn the owner's Pause into a budget pause)."""
    from ouroboros import owner_pause
    from ouroboros.usage_accounting import AttemptRequest, UsageScope, reserve_attempt, usage_scope
    from supervisor.owner_pause_control import request_owner_pause

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _live_tree(tmp_path, workers)
    workers.PENDING[:] = []
    assert request_owner_pause("root-1", request_id="p")["ok"]
    _owner_park(tmp_path, monkeypatch, "child-run", [])   # checkpoint stored, park event not handled
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=lambda _e: None), raising=False)

    workers.kill_workers(force=True, terminal_status="cancelled", result_reason="Owner restart",
                         hold_never_started=True)

    assert [row["id"] for row in workers.PENDING] == ["child-run"]
    assert workers.PENDING[0]["_budget_pause"]["reason"] == "owner"
    workers.PENDING[:] = []
    queue.BUDGET_ROOT_FENCES.clear()
    assert queue.restore_pending_from_snapshot() >= 1
    assert queue.BUDGET_ROOT_FENCES["root-1"]["cause"] == "owner_pause"
    assert owner_pause.read_fence(tmp_path, "root-1")["state"] in owner_pause.CLOSED_FENCE_STATES
    monkeypatch.setattr("ouroboros.usage_accounting._reservation_cost", lambda _r: 0.01)
    with usage_scope(UsageScope(drive_root=tmp_path, task_id="child-run", root_task_id="root-1")):
        reservation = reserve_attempt(AttemptRequest(model="m", provider="p", task_id="child-run"))
    assert reservation.attempt_id, "an owner latch is not a money refusal at reservation"


def test_resume_of_a_never_started_owner_paused_root_reopens_both_fences(tmp_path, monkeypatch):
    from ouroboros import owner_pause
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result
    from supervisor.owner_pause_control import request_owner_pause

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "queued-root", STATUS_SCHEDULED, chat_id=0)
    workers.PENDING.append({"id": "queued-root", "type": "task", "chat_id": 0, "_attempt": 1,
                            "root_task_id": "queued-root", "admitted_dispatch": "none"})
    assert request_owner_pause("queued-root", request_id="p")["ok"]
    from supervisor.queue_transitions import budget_pause_fact

    assert budget_pause_fact(workers.PENDING[0]) is not None, "fenced from dispatch"
    outcome = queue.resume_budget_paused_task("queued-root")
    assert outcome["ok"] is True and outcome["owner_pause_released"] is True
    assert "queued-root" not in queue.BUDGET_ROOT_FENCES
    assert owner_pause.read_fence(tmp_path, "queued-root")["state"] == owner_pause.FENCE_RELEASED
    assert budget_pause_fact(workers.PENDING[0]) is None


def test_the_wake_control_is_never_dialogue_and_wakes_a_warm_owner_wait(tmp_path):
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import KIND_OWNER_PAUSE, write_owner_message
    from ouroboros.owner_wait import classify_wake
    import queue as stdqueue

    write_owner_message(tmp_path, "owner_pause", "t-1", msg_id="owner_pause:f", kind=KIND_OWNER_PAUSE)
    messages: list = []
    controls = _drain_incoming_messages(messages, stdqueue.Queue(), tmp_path, "t-1", None, set())
    from ouroboros.owner_mailbox import acknowledged_task_message_ids
    assert controls == {} and messages == []
    assert acknowledged_task_message_ids(tmp_path, "t-1") == {"owner_pause:f"}
    assert classify_wake([{"kind": KIND_OWNER_PAUSE, "msg_id": "owner_pause:f"}], "q") == "control:owner_pause"


def test_the_pause_endpoint_is_text_free_and_typed(tmp_path, monkeypatch):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.task_pause import api_task_pause

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _live_tree(tmp_path, workers)
    client = TestClient(Starlette(routes=[Route("/api/tasks/{task_id}/pause", api_task_pause, methods=["POST"])]))
    assert client.post("/api/tasks/root-1/pause", json={"request_id": "r", "text": "x"}).status_code == 400
    assert client.post("/api/tasks/root-1/pause", json={}).status_code == 400
    ok = client.post("/api/tasks/root-1/pause", json={"request_id": "r"})
    assert ok.status_code == 200 and ok.json()["state"] == "requested"
    child = client.post("/api/tasks/child-run/pause", json={"request_id": "r"})
    assert child.status_code == 409 and child.json()["reason_code"] == "not_a_root_task"
    assert client.post("/api/tasks/ghost/pause", json={"request_id": "r"}).status_code == 404


def test_a_direct_roots_warm_park_settles_when_its_requested_stop_ends_and_resume_proceeds(tmp_path, monkeypatch):
    """R1: a direct actor parks WARM outside ``RUNNING``. Its own run's stop answered
    ``requested``; when that run ends later, the settle tick reaches the park through
    the direct registry and re-reads (never re-issues) the stop it issued, so the tree
    turns Paused and the owner's Resume is no longer refused."""
    import threading
    import time

    from ouroboros import delegate_custody as dc
    from ouroboros import owner_pause, owner_wait
    from ouroboros.external_runs import EXTERNAL_STOP_CONFIRMED, EXTERNAL_STOP_REQUESTED
    from ouroboros.gateways import claudexor as gw
    from ouroboros.model_wait import TaskModelWait
    from ouroboros.task_results import STATUS_RUNNING, load_task_result, write_task_result
    from supervisor import owner_pause_control
    from supervisor.budget_resume import resume_warm_owner_pause_root
    from supervisor.owner_pause_control import (
        refresh_owner_pause_tree,
        request_owner_pause,
        settle_requested_owner_pauses,
    )
    from tests.test_direct_chat_turn_owner_control import _live_chat_agent

    queue, state, _workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    write_task_result(tmp_path, "author", STATUS_RUNNING, chat_id=0)
    _live_chat_agent(monkeypatch, task_id="author")
    dc.record_started(tmp_path, dc.RunCustody(
        run_id="run-own", task_id="author", route_id="r", model="m", project_id="p",
        project_owned=False, root_task_id="author", ledger_root=str(tmp_path)))
    remote = {"state": "running", "cancels": []}

    class Daemon:
        def handshake(self, **_kw):
            return {"compatible": True}

        def cancel_run(self, run_id, reason=""):
            remote["cancels"].append(run_id)
            return {"accepted": True, "status": "accepted"}

        def get_run(self, run_id, **_kw):
            return {"lastSeq": 3, "summary": {"state": remote["state"], "spendUsd": 0.0}}

        def close(self):
            return None

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: Daemon())
    ack = request_owner_pause("author", request_id="press")
    assert ack["ok"] and ack["state"] == "requested", ack

    ctx, limit = _loop_ctx(tmp_path, "author", direct=True)
    ctx.pending_events, ctx.current_chat_id = [], 0
    waiter = TaskModelWait(task={"id": "author", "budget_drive_root": str(tmp_path)},
                           drive_root=tmp_path, event_queue=None, worker_slot_held=False)
    waiter.tool_context, ctx.model_wait_context = ctx, waiter
    ctx.owner_wait_callback = owner_wait.direct_owner_wait
    real_sleep = time.sleep
    monkeypatch.setattr(owner_wait.time, "sleep", lambda seconds: real_sleep(min(seconds, 0.02)))
    woke: dict = {}
    parked = threading.Thread(target=lambda: woke.update(cause=owner_wait.park_owner_pause_warm(
        limit, ctx, fence=owner_pause.read_fence(tmp_path, "author"), detached=["review-op"])), daemon=True)
    parked.start()
    try:
        deadline = time.monotonic() + 10
        while (load_task_result(tmp_path, "author").get("owner_wait") or {}).get("state") != "waiting":
            assert time.monotonic() < deadline and parked.is_alive(), woke
            real_sleep(0.02)
        wait = load_task_result(tmp_path, "author")["owner_wait"]
        assert [run["state"] for run in wait["owner_pause"]["external_runs"]["runs"]] == [EXTERNAL_STOP_REQUESTED]
        assert remote["cancels"] == ["run-own"]
        # The stopped run has not ended yet: still Pausing, and Resume waits for it.
        assert refresh_owner_pause_tree("author") == owner_pause.FENCE_REQUESTED
        assert resume_warm_owner_pause_root("author")["error"] == "owner_pause_effects_unsettled"

        remote["state"] = "cancelled"  # the run obeys the requested stop later
        monkeypatch.setattr(owner_pause_control, "_LAST_SETTLE_CHECK", {})
        assert settle_requested_owner_pauses(queue, now=100.0) == ["author"]
        assert remote["cancels"] == ["run-own"], "the issued stop is re-read, never repeated"
        wait = load_task_result(tmp_path, "author")["owner_wait"]
        assert [run["state"] for run in wait["owner_pause"]["external_runs"]["runs"]] == [EXTERNAL_STOP_CONFIRMED]
        assert owner_pause.read_fence(tmp_path, "author")["warm_members"] == ["author"]

        resumed = resume_warm_owner_pause_root("author")
        assert resumed["ok"] and resumed["warm"], resumed
        parked.join(10)
        assert woke.get("cause") == "control:owner_resume"
    finally:
        owner_pause.release_fence(tmp_path, "author", reason="test_cleanup")
        parked.join(10)
