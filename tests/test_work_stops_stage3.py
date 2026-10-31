"""Stage3 consumer scenarios: answered trees, exact Resume and shared projection."""

from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_owner_pause import _live_tree, _owner_park
from tests.test_budget_pause_holds import _idle_worker


@pytest.mark.parametrize("mind", [False, True])
def test_answered_tree_pause_and_explicit_children_selection(tmp_path, monkeypatch, mind):
    from contextlib import nullcontext
    from ouroboros import budget_pause, owner_pause
    from ouroboros.control_actor import mind_control
    from ouroboros.task_results import write_task_result, load_task_result
    from supervisor import owner_pause_control as control
    from supervisor.events_budget import install_exact_budget_pause
    from tests.test_work_projection_pause_producers import _works

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    _live_tree(tmp_path, workers)
    workers.RUNNING.pop("root-1")
    write_task_result(tmp_path, "root-1", "completed", result="Delivered answer")
    [work] = _works(tmp_path)
    assert {"action": "pause", "target": "root-1"} in work["controls"]
    with mind_control("controller", "Wait for review") if mind else nullcontext():
        assert control.request_owner_pause("root-1", request_id="late-tree")["ok"]
    assert owner_pause.read_fence(tmp_path, "root-1")["state"] == "requested"
    assert queue.resume_budget_paused_task("root-1")["error"] == "owner_pause_effects_unsettled"
    saved = _owner_park(tmp_path, monkeypatch, "child-run", [])
    install_exact_budget_pause(workers, "child-run", budget_pause.exact_pause_marker(saved)["checkpoint"])
    [work] = _works(tmp_path)
    assert work["activity"] == "paused"
    assert {"action": "resume", "target": "root-1"} in work["controls"]
    with mind_control("controller", "Continue the helpers") if mind else nullcontext():
        resumed = queue.resume_budget_paused_task("root-1")
    assert resumed["ok"] and resumed["tree_only"]
    from typing import get_type_hints
    from ouroboros.gateway.client_contracts import TaskResumeResponse
    from ouroboros.gateway.schema import validate_ingress
    assert set(resumed) <= set(get_type_hints(TaskResumeResponse))
    assert not validate_ingress(resumed, TaskResumeResponse)
    assert load_task_result(tmp_path, "root-1")["status"] == "completed"
    [work] = _works(tmp_path)
    assert sorted(c["target"] for c in work["controls"] if c["action"] == "resume") == ["child-new", "child-run"]
    assert all(c["scope"] == "single" for c in work["controls"] if c["action"] == "resume")
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert not sent
    assert queue.resume_budget_paused_task("child-run", selected_by="root-1")["ok"]
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["child-run"]
    assert [t["id"] for t in workers.PENDING] == ["child-new"]
    assert control.request_owner_pause("root-1", request_id="again")["ok"]
    assert queue.resume_budget_paused_task("child-new", selected_by="root-1")["ok"] is False


@pytest.mark.parametrize("door", ["owner", "child", "stopped"])
def test_terminal_tree_resume_keeps_warm_children_parked_until_each_is_selected(tmp_path, monkeypatch, door):
    from ouroboros import owner_pause, owner_wait
    from ouroboros.task_results import write_task_result
    from supervisor.owner_pause_control import request_owner_pause

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    _live_tree(tmp_path, workers)
    workers.RUNNING.pop("root-1")
    write_task_result(tmp_path, "root-1", "completed")
    assert request_owner_pause("root-1", request_id="warm")["ok"]
    workers.RUNNING["child-run"]["owner_wait"] = {"state": "waiting", "reason": "owner_pause"}
    ctx = SimpleNamespace(task_id="child-run", root_task_id="root-1", budget_drive_root=tmp_path, task_attempt=1)
    if door == "stopped":
        from ouroboros.cancel_intents import request_cancel
        request_cancel(tmp_path, "root-1", reason="owner stop", source="http", requested_by="owner",
                       requested_stop_policy="immediate", allow_settled_target=True)
        assert queue.resume_budget_paused_task("root-1")["error"] == "cancel_intent_active"
        assert owner_pause.fence_closed(owner_pause.read_fence(tmp_path, "root-1"))
        assert owner_wait.consume_warm_resume(ctx) is False
        return
    assert queue.resume_budget_paused_task("root-1")["ok"]
    assert owner_pause.member_fence(ctx)["awaiting_selection"] is True
    assert owner_wait.consume_warm_resume(ctx) is False
    from tests.test_work_projection_pause_producers import _works
    [work] = _works(tmp_path)
    assert work["members"]["states"]["child-run"]["activity"] == "paused"
    assert {"action": "resume", "target": "child-run", "scope": "single"} in work["controls"]
    if door == "child":
        from tests._budget_pause_exact_helpers import _supervisor_ctx
        from supervisor.events_budget import _handle_budget_resume_child
        outcomes = []
        _handle_budget_resume_child({"task_id": "child-run", "requested_by": "root-1"},
                                   _supervisor_ctx(tmp_path, workers, queue, [], outcomes))
        [selected] = outcomes
    else:
        selected = queue.resume_budget_paused_task("child-run")
    assert selected["ok"], selected
    assert not owner_pause.member_fence(ctx)
    assert owner_wait.consume_warm_resume(ctx) is True
    assert owner_pause.read_fence(tmp_path, "root-1")["selected_members"]["child-run"]["consumed_at"]
    assert all(t["id"] != "root-1" for t in workers.PENDING)


@pytest.mark.parametrize("live", [False, True])
def test_pausing_is_execution_and_refusal_does_not_hide_resume(live):
    from ouroboros.gateway.work_status import classify_member, controls_for
    row = {"activity_id": "r", "phase": "budget_pausing", "pause_cause": "owner"}
    raw = {"running": live, "pause_by": {"actor": "ouroboros", "text": "before update"},
           "refusal": {"reason": "source_missing"}}
    member = classify_member(row, raw)
    assert member["activity"] == ("working" if live else "paused")
    assert member["transition"] == ("pausing" if live else "none")
    assert ("resume" in [c["action"] for c in controls_for(member, row)]) is not live
    assert member["stop"]["release"] == "resume"
    assert member["reasons"][0]["text"] == "Ouroboros paused it"


@pytest.mark.parametrize("kind", ["tool", "model"])
@pytest.mark.parametrize("ended", [False, True])
@pytest.mark.parametrize("mind", [False, True])
def test_addressed_resume_repairs_ended_recipients_and_names_live_blockers(tmp_path, monkeypatch, kind, ended, mind):
    import os
    from contextlib import nullcontext
    from ouroboros import budget_pause, platform_layer, model_wait, usage_accounting as ua
    from ouroboros.control_actor import mind_control
    from ouroboros.task_results import write_task_result, load_task_result
    from supervisor.owner_pause_control import request_owner_pause
    from supervisor.events_budget import install_exact_budget_pause
    from tests.test_local_custody_recovery import _dead_process_identity
    from tests.test_work_projection_pause_producers import _works

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    _live_tree(tmp_path, workers)
    workers.RUNNING.pop("child-run")
    workers.PENDING.clear()
    write_task_result(tmp_path, "child-run", "cancelled", root_task_id="root-1", parent_task_id="root-1")
    pid, birth = _dead_process_identity() if ended else (os.getpid(), platform_layer.process_start_time(os.getpid()))
    from ouroboros.tool_custody import owner_proof
    proof = {} if ended else owner_proof()  # a live owner is this image: every row it writes carries its session
    if kind == "tool":
        write_task_result(tmp_path, "child-run", "cancelled", launch_handoffs={"op": {
            "tool": "delegate_wait", "task_id": "child-run", "root_task_id": "root-1", "state": "claimed",
            "local_owner": {"pid": pid, "process_birth": birth, "task_attempt": 1},
            **({"owner_proof": proof} if proof else {})}})
    else:
        monkeypatch.setenv("TOTAL_BUDGET", "1000")
        receiver = model_wait.TaskModelWait(task={"id": "child-run", "_attempt": 1}, drive_root=tmp_path,
                                            event_queue=None, worker_slot_held=True)
        monkeypatch.setattr(receiver, "bind_answer_consumer", lambda: {
            "local_answer_consumer_id": receiver.answer_consumer_id, "local_answer_owner_pid": pid,
            "local_answer_owner_birth": birth, "local_answer_task_attempt": 1,
            **({"local_answer_owner_proof": proof} if proof else {})})
        with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="child-run", root_task_id="root-1")):
            with model_wait.operation_wait_scope(receiver):
                attempt = ua.reserve_attempt(ua.AttemptRequest(model="m", provider="t", reservation_usd=.2))
                ua.mark_dispatched(attempt, local_answer_owner_pid=os.getpid())
                ua.mark_unresolved(attempt, "receipt_unknown")
    assert request_owner_pause("root-1", request_id="repair")["ok"]
    saved = _owner_park(tmp_path, monkeypatch, "root-1", [])
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                          WORKERS=workers.WORKERS, sort_pending=queue.sort_pending,
                          persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None)
    install_exact_budget_pause(ctx, "root-1", budget_pause.exact_pause_marker(saved)["checkpoint"])
    if kind == "model":
        [work] = _works(tmp_path)
        assert work["activity"] == "paused" and {"action": "resume", "target": "root-1"} in work["controls"]
        assert any(t["code"] == "model_handoff" and t["state"] == "unknown" for t in work["tails"])
    before_money = ua.read_usage_records(tmp_path, final_only=True)
    with mind_control("controller", "Continue saved work") if mind else nullcontext():
        result = queue.resume_budget_paused_task("root-1")
    assert result["ok"] is ended, result
    held = load_task_result(tmp_path, "child-run")
    if ended:
        assert held.get("retired_tool_invocations" if kind == "tool" else "retired_model_consumers")
        if kind == "tool":
            assert held["retired_tool_invocations"]["op"]["replay_authorized"] is False
    else:
        assert result["error"] == "owner_pause_effects_unsettled"
        assert any(b["kind"] == ("tool_handoff" if kind == "tool" else "model_handoff") for b in result["blockers"])
        assert ("call of task child-run is in a live worker" if kind == "tool"
                else "a model answer for task child-run is awaited by a live worker") in result["message"]
        assert "acceptable_unknown" not in result, "a live call is never offered to go on past"
    assert ua.read_usage_records(tmp_path, final_only=True) == before_money


def test_scope_precedence_keeps_the_members_release_rule_for_owner_and_mind():
    from ouroboros.work_stop import current_stop, stop_presentation
    from ouroboros.gateway.work_status import classify_member
    from ouroboros.peer_roster import _queue_hold_facts

    task = {"id": "r", "_budget_pause_hold": {"reason": "panic_hold", "release": "owner"}}
    fence = {"state": "paused", "by": {"actor": "owner", "at": "T0"}}
    stop = current_stop(task, {"owner": fence})
    assert stop["scope"] == "tree" and stop["member_stop"]["cause"] == "panic"
    view = stop_presentation(task, {"owner": fence})
    member = classify_member({"activity_id": "r", "phase": "budget_paused"},
                             {**task, "fences": {"owner": fence}})
    [mind] = _queue_hold_facts(task, {"owner_pause": fence}, [])
    assert view["release"] == member["release"] == mind["release"] == "owner"
    assert view["reason"]["text"] == member["reasons"][0]["text"] == mind["text"] == "You paused it"
    assert stop_presentation(task, {"owner": {"state": "released"}})["reason"]["text"] == "Stopped by Panic"
    installation = {"cause": "app_stop", "release": "resume", "door": "owner_restart"}
    combined = current_stop(task, {"installation": installation, "owner": fence})
    assert combined["scope"] == "installation" and combined["member_stop"] == stop
