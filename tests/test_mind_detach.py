"""Same work, cold continuation, unchanged authority and actual attempt ownership."""

import json
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _fast_hold, _install_queue, _loop_ctx, _quiet_external


def _arm(tmp_path, monkeypatch):
    from ouroboros import model_detach
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.control_mind import _continue_in_background

    monkeypatch.setattr(model_detach, "pool_available", lambda: {"available": True})
    write_task_result(tmp_path, "head", "running", _is_direct_chat=True)
    ctx, limit = _loop_ctx(tmp_path, "head", direct=True)
    # A completed batch is the cold-sleep checkpoint boundary.
    limit.messages.append({"role": "tool", "tool_call_id": "call_b", "content": "done b"})
    result = json.loads(_continue_in_background(ctx, "keep working"))
    return ctx, limit, result


@pytest.mark.parametrize("door", ["ordinary", "owner_restart", "panic"])
def test_ready_detach_arms_then_parks_same_id_and_sleep_grant_is_single_use(tmp_path, monkeypatch, door):
    from ouroboros import budget_pause
    from ouroboros.task_results import load_task_result
    from supervisor.direct_pause import park_direct_pause
    from supervisor.sleep_wake import wake_ready_sleepers

    q, state, workers = _install_queue(tmp_path, monkeypatch)
    from supervisor import message_bus
    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(push_log=lambda *_a: None))
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    ctx, limit, result = _arm(tmp_path, monkeypatch)
    assert result["ok"] and result["status"] == "detach_armed"
    assert "browser" in result["note"] and "REPL" in result["note"]
    assert ctx._model_sleep["intent"] == "continue_in_background"
    _fast_hold(monkeypatch, budget_pause)
    _quiet_external(monkeypatch, budget_pause)
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.enter_cold_sleep(limit)
    pause = raised.value.pause
    assert pause["resume_point"]["unanswered_tool_call_ids"] == []
    task = {"id": "head", "type": "task", "chat_id": 0, "_is_direct_chat": True,
            "_attempt": 1, "tool_profile": "operator_control"}
    assert park_direct_pause(task, budget_pause.pause_event(task, pause))
    assert len(workers.PENDING) == 1 and workers.PENDING[0]["id"] == "head"
    parked = workers.PENDING[0]
    assert parked["tool_profile"] == "operator_control" and parked["_is_direct_chat"]
    assert parked["detached_by"] == "mind" and parked["detached_at"] > 0
    durable = load_task_result(tmp_path, "head")
    assert durable["metadata"]["detached_at"] == parked["detached_at"]
    assert wake_ready_sleepers(q)[0]["ok"]
    grant = budget_pause.budget_pause_row(tmp_path, "head")["grant"]
    assert grant["selected_by"] == "sleep_wake"
    assert wake_ready_sleepers(q) == []
    assert budget_pause.budget_pause_row(tmp_path, "head")["grant"] == grant
    # An ordinary restart (update, crash recovery) restores the detached sleep, which wakes
    # itself; the owner's Restart or a Panic retains it for an explicit Resume.
    flag = {"owner_restart": "owner_restart_no_resume.flag", "panic": "panic_stop.flag"}.get(door)
    if flag:
        (tmp_path / "state" / flag).write_text("panic" if door == "panic" else "owner_restart", encoding="utf-8")
    assert q.persist_queue_snapshot(reason="detached-before-restart")
    workers.PENDING[:] = []
    assert q.restore_pending_from_snapshot() == 1
    restored = workers.PENDING[0]
    assert restored["id"] == "head" and restored["_is_direct_chat"]
    assert restored["detached_at"] == durable["detached_at"]
    assert restored["tool_profile"] == "operator_control"
    woke = [outcome for outcome in wake_ready_sleepers(q) if outcome.get("ok")]
    if not flag:
        regrant = budget_pause.budget_pause_row(tmp_path, "head")["grant"]
        assert woke and regrant["selected_by"] == "sleep_wake"
        assert regrant["grant_id"] != grant["grant_id"], "the pre-restart grant stays single-use"
        return
    assert not woke
    if door == "owner_restart":
        from ouroboros.delegate_recovery import consume_owner_restart, pending_owner_restart
        consume_owner_restart(tmp_path, pending_owner_restart(tmp_path))
    else:
        (tmp_path / "state" / flag).unlink()
    assert q.resume_budget_paused_task("head")["ok"], "explicit Resume can release the retained sleep"


@pytest.mark.parametrize("kind", ["service", "delegated_run", "running_member", "member_custody", "custody_unreadable"])
def test_detach_refuses_blocker_list_without_arming(tmp_path, monkeypatch, kind):
    from ouroboros import model_sleep

    blockers = [{"kind": kind, "detail": "still owned"}]
    monkeypatch.setattr(model_sleep, "cold_blockers", lambda *_a, **_k: blockers)
    ctx, _limit, result = _arm(tmp_path, monkeypatch)
    assert result == {"ok": False, "error": "detach_blocked", "blockers": blockers}
    assert not getattr(ctx, "_model_sleep", None)


def test_detach_does_not_accept_transfer_eligible_runs(tmp_path, monkeypatch):
    from ouroboros import budget_pause, model_sleep, sleep_custody
    from tests.test_model_sleep import _ctx

    monkeypatch.setattr(budget_pause, "observe_task_runs", lambda *_a, **_k: {
        "custody_read": "ok", "runs": [{"run_id": "r", "invocation_id": "i"}]})
    monkeypatch.setattr(sleep_custody, "transferable_pairs", lambda *_a: ({("r", "i")}, False))
    ctx = _ctx(tmp_path)
    assert not model_sleep.cold_blockers(ctx, allow_transfer=True)
    assert model_sleep.cold_blockers(ctx, chosen={"intent": "continue_in_background"}, allow_transfer=True) == [
        {"kind": "delegated_run", "run_id": "r", "invocation_id": "i"}]


def test_unreadable_custody_root_is_an_explicit_detach_blocker(tmp_path, monkeypatch):
    from ouroboros import delegate_custody, model_sleep
    from tests.test_model_sleep import _ctx

    ctx = _ctx(tmp_path)
    assert model_sleep.cold_blockers(ctx) == []

    def unavailable(_ctx):
        raise OSError("custody root unavailable")

    monkeypatch.setattr(delegate_custody, "custody_root", unavailable)
    blockers = model_sleep.cold_blockers(ctx, chosen={"intent": "continue_in_background"})
    assert blockers[0] == {"kind": "custody_unreadable", "detail": "custody root unavailable"}


def test_detach_bypasses_selector_readiness_but_keeps_ending_controls(tmp_path, monkeypatch):
    from ouroboros import model_sleep

    monkeypatch.setattr(model_sleep, "wake_reason", lambda *_a: "control:finalize_now")
    ctx, _limit, result = _arm(tmp_path, monkeypatch)
    assert result["error"] == "detach_control_pending"
    assert not getattr(ctx, "_model_sleep", None)
    monkeypatch.setattr(model_sleep, "wake_reason", lambda *_a: "timeout")
    ctx, _limit, result = _arm(tmp_path, monkeypatch)
    assert result["status"] == "detach_armed" and ctx._model_sleep


def test_pooled_noop_and_pool_failure_keeps_head(tmp_path, monkeypatch):
    from ouroboros import model_detach

    ctx, _limit = _loop_ctx(tmp_path, "head", direct=True)
    ctx.task_metadata = {"_execution_owner": {"kind": "pooled"}}
    assert model_detach.arm_detach(ctx, "")["status"] == "already_in_background"
    ctx.task_metadata = {}
    monkeypatch.setattr(model_detach, "pool_available", lambda: {"available": False, "disabled_reason": "crash_storm"})
    assert model_detach.arm_detach(ctx, "")["error"] == "worker_pool_unavailable"
    assert not getattr(ctx, "_model_sleep", None)


def test_boundary_rechecks_pool_and_custody(tmp_path, monkeypatch):
    from ouroboros import budget_pause, model_detach, model_sleep

    ctx, limit, result = _arm(tmp_path, monkeypatch)
    assert result["ok"]
    monkeypatch.setattr(model_detach, "pool_available", lambda: {"available": False})
    budget_pause.enter_cold_sleep(limit)
    assert ctx._model_sleep is None and "worker_pool_unavailable" in limit.messages[-1]["content"]
    monkeypatch.setattr(model_detach, "pool_available", lambda: {"available": True})
    assert model_detach.arm_detach(ctx, "")["ok"]
    monkeypatch.setattr(model_sleep, "cold_blockers", lambda *_a, **_k: [{"kind": "service", "name": "late"}])
    budget_pause.enter_cold_sleep(limit)
    assert ctx._model_sleep is None and "service late" in limit.messages[-1]["content"]


def test_frozen_profile_survives_carrier_change_but_not_subagent_ceiling():
    from ouroboros.tool_access_roots import active_tool_profile, freeze_tool_profile

    task = {"_is_direct_chat": True}
    assert freeze_tool_profile(task) == "operator_control"
    ctx = SimpleNamespace(task_metadata={"tool_profile": task["tool_profile"]}, is_direct_chat=False)
    assert active_tool_profile(ctx) == "operator_control"
    ctx.task_metadata["delegation_role"] = "subagent"
    assert active_tool_profile(ctx) == "local_readonly_subagent"
    task = {"workspace_root": "/tmp/a", "workspace_mode": "external"}
    assert freeze_tool_profile(task) == "external_workspace_task"
    task.pop("workspace_root")
    assert freeze_tool_profile(task) == "external_workspace_task"


def test_owner_question_stays_warm_in_either_tool_order(tmp_path, monkeypatch):
    from ouroboros import model_detach, owner_wait

    ctx, limit, result = _arm(tmp_path, monkeypatch)
    assert result["ok"]
    ctx._owner_wait_requested = "quiz-created-after-detach"

    class WarmCheckpoint(Exception):
        pass

    def warm(ctx, *_a, **_k):
        assert ctx._model_sleep is None
        assert ctx._owner_wait_requested == "quiz-created-after-detach"
        raise WarmCheckpoint

    monkeypatch.setattr(owner_wait, "checkpoint_owner_wait", warm)
    with pytest.raises(WarmCheckpoint):
        owner_wait.wait_after_tools(ctx, limit.messages, {}, {}, 1, [], set())
    assert model_detach.arm_detach(ctx, "")["error"] == "owner_question_pending"


def test_raise_after_saved_pause_parks_before_unregister_without_error_terminal(tmp_path, monkeypatch):
    from ouroboros import budget_pause
    from supervisor import message_bus, worker_chat_lane

    _q, _state, workers = _install_queue(tmp_path, monkeypatch)
    ctx, limit, _result = _arm(tmp_path, monkeypatch)
    _fast_hold(monkeypatch, budget_pause)
    _quiet_external(monkeypatch, budget_pause)
    task = {"id": "head", "type": "task", "chat_id": 7, "text": "work",
            "project_id": "fixture", "_is_direct_chat": True, "_attempt": 1}
    emitted, released = [], []
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=emitted.append))
    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(push_log=lambda _e: None))
    monkeypatch.setattr(worker_chat_lane, "_report_direct_chat_error", lambda *_a: pytest.fail("pause became terminal"))

    def fail(_task):
        try:
            budget_pause.enter_cold_sleep(limit)
        except budget_pause.BudgetPauseRequested:
            raise OSError("after saved pause")

    def unregister(task_id):
        assert workers.PENDING[0]["id"] == task_id
        released.append(task_id)

    assert worker_chat_lane._execute_chat_task({"task": task, "agent": SimpleNamespace(handle_task=fail),
        "chat_id": 7, "registry": SimpleNamespace(unregister=unregister)})
    assert released == ["head"] and not emitted
    assert not budget_pause.dispatch_fenced("head")


@pytest.mark.parametrize("unreadable", [False, True])
def test_failed_pause_source_stays_visible_without_dispatch_authority(tmp_path, monkeypatch, unreadable):
    from ouroboros import budget_pause, working_checkpoint
    from ouroboros.work_stop import stop_presentation
    from supervisor.direct_pause import recover_direct_pause
    from supervisor.events_budget import HOLD_UNSAVED_PAUSE, budget_hold_fact

    _q, _state, workers = _install_queue(tmp_path, monkeypatch)
    task = {"id": "head", "type": "task", "chat_id": 0, "_attempt": 1, "_is_direct_chat": True}
    assert not recover_direct_pause(task)
    assert not workers.PENDING

    def read(*_a):
        if unreadable:
            raise OSError("pause store unavailable")
        return {"state": "pausing", "task_attempt": 1}

    monkeypatch.setattr(budget_pause, "budget_pause_row", read)
    monkeypatch.setattr(working_checkpoint, "complete_pause_from_working", lambda *_a: None)
    assert recover_direct_pause(task)
    assert len(workers.PENDING) == 1
    held = workers.PENDING[0]
    assert held["id"] == "head" and held["_is_direct_chat"]
    # Unreadable evidence after the body ran is an unknown, never work "saved when the app stopped".
    assert budget_hold_fact(held)["reason"] == ("authority" if unreadable else HOLD_UNSAVED_PAUSE)
    if unreadable:
        assert stop_presentation(held)["reason"]["text"] == "After a technical problem"
    assert budget_hold_fact(held) and "_resume_selection" not in held
    assert not held.get("_budget_pause_resume")
    assert recover_direct_pause(task) and len(workers.PENDING) == 1
