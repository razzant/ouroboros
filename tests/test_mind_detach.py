"""Same work, cold continuation, unchanged authority and actual attempt ownership.

A direct turn is a conversation-lane RUNNING row carried by the pool: its exact pause is parked by the
supervisor from the body's event (the one pooled path), never by the body's own process."""

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
    from supervisor.sleep_wake import wake_ready_sleepers
    from tests.conversation_carrier_support import lane_row, park_lane_pause

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
    slot = lane_row(workers, task)  # the turn's RUNNING row, as the lane admitted and dispatched it
    park_lane_pause(budget_pause.pause_event(task, pause), worker_id=slot.wid)  # the body's event, parked by the supervisor
    budget_pause.end_dispatch_fence("head")  # the carrier's process-local fence ends with its body
    assert slot.busy_task_id is None and "head" not in workers.RUNNING
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


@pytest.mark.parametrize("kind", ["service", "delegated_run", "custody_unreadable"])
def test_detach_refuses_blocker_list_without_arming(tmp_path, monkeypatch, kind):
    from ouroboros import model_sleep

    blockers = [{"kind": kind, "detail": "still owned"}]
    monkeypatch.setattr(model_sleep, "cold_blockers", lambda *_a, **_k: blockers)
    ctx, _limit, result = _arm(tmp_path, monkeypatch)
    assert result == {"ok": False, "error": "detach_blocked", "blockers": blockers}
    assert not getattr(ctx, "_model_sleep", None)


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


def test_the_final_census_withdraws_a_handoff_awake_where_it_holds_a_sleep(tmp_path, monkeypatch):
    """Work first found by the final census (``budget_pause._exact_continuation_row``): a background handoff
    abandons its seed and stays awake; an ordinary cold sleep holds there until that work settles."""
    from ouroboros import budget_pause, model_sleep

    _install_queue(tmp_path, monkeypatch)
    ctx, limit, result = _arm(tmp_path, monkeypatch)
    assert result["ok"]
    _fast_hold(monkeypatch, budget_pause)
    _quiet_external(monkeypatch, budget_pause)
    censuses = []  # the round boundary's census finds nothing; the final one finds a late service
    monkeypatch.setattr(model_sleep, "cold_blockers", lambda *_a, **_k: [] if censuses.append(1) or len(censuses) == 1
                        else [{"kind": "service", "name": "late"}])
    budget_pause.enter_cold_sleep(limit)
    assert budget_pause.budget_pause_row(tmp_path, "head")["abandon_reason"] == "cold_detach_blocked"
    assert ctx._model_sleep is None and ctx._budget_pausing is False
    notice = limit.messages[-1]["content"]
    assert "background handoff was not taken" in notice and "'service'" in notice and "'late'" in notice
    ctx._model_sleep = {"sleep_id": "s1", "mode": "cold"}
    with pytest.raises(budget_pause.ColdSleepWritersUnsettled, match="'late'"):
        budget_pause._exact_continuation_row(limit, ctx, pause_id="p2", rail=budget_pause.RAIL_MODEL_SLEEP,
                                             scope="task", reason_text="", root_task_id="head", local={})


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


def test_raise_after_saved_pause_is_no_error_terminal_and_death_returns_to_the_pause(tmp_path, monkeypatch):
    """The carrier's body raises after its pause was saved: no exception terminal is written and nothing
    but the start phase it reported is emitted; the saved pause of this attempt is what the crash recovery
    parks again (#1196, F4)."""
    from ouroboros import budget_pause
    from ouroboros.task_results import load_task_result
    from supervisor import message_bus, worker_process

    _q, _state, workers = _install_queue(tmp_path, monkeypatch)
    ctx, limit, _result = _arm(tmp_path, monkeypatch)
    _fast_hold(monkeypatch, budget_pause)
    _quiet_external(monkeypatch, budget_pause)
    task = {"id": "head", "type": "task", "chat_id": 7, "text": "work",
            "project_id": "fixture", "_is_direct_chat": True, "_attempt": 1}
    emitted = []
    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(push_log=lambda _e: None))
    monkeypatch.setattr("ouroboros.agent._task_exception_terminal", lambda *_a, **_k: pytest.fail("pause became terminal"))

    def fail(_task):
        try:
            budget_pause.enter_cold_sleep(limit)
        except budget_pause.BudgetPauseRequested:
            raise OSError("after saved pause")

    agent = SimpleNamespace(handle_task=fail, env=SimpleNamespace(drive_root=tmp_path, drive_path=lambda name: tmp_path / name))
    try:
        with pytest.raises(OSError, match="after saved pause"):
            worker_process.run_worker_task(agent, task, wid=0, out_q=SimpleNamespace(put=emitted.append),
                                           repo_dir=str(tmp_path), drive_root=str(tmp_path))
    finally:
        budget_pause.end_dispatch_fence("head")  # the dying carrier's process-local fence goes with it
    assert [row["type"] for row in emitted] == ["direct_turn_phase"]
    assert budget_pause.death_returns_to_pause(budget_pause.budget_pause_row(tmp_path, "head"), 1)
    assert (load_task_result(tmp_path, "head") or {}).get("status") != "failed"


# ``direct_pause.recover_direct_pause`` (the inline actor's own post-raise pause recovery: an unsaved or
# unreadable pause held visibly with no dispatch authority) is retired with the in-process lane. A carrier
# that raises after an incomplete pause dies with it, and its evidence is the pool's crash recovery's
# (``worker_health._complete_exact_budget_pause_after_death``: the same attempt's working state completes
# the pause, otherwise the attempt is terminal and never replayed), pinned by that recovery's own tests.
