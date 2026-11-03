"""Saved warm parks retain custody across restart selection and legacy handoffs.

Real checkpoint, worker-stop, snapshot, Resume and loop-restore owners; no live
provider or application process. Legacy carriers model the pre-fix stop output.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import budget_pause, delegate_recovery, owner_pause, owner_wait
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.task_results import STATUS_RUNNING, STATUS_SCHEDULED, load_task_result, write_task_result
from supervisor import restart_retention
from supervisor.events_budget import budget_hold_fact
from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
from tests.test_restart_retention import _pool_events

pytestmark = pytest.mark.serial


def _park(root, workers, reason, *, direct=False, attempt=1):
    tid = "saved-warm"
    write_task_result(root, tid, STATUS_RUNNING, chat_id=0, task_attempt=attempt,
                      root_task_id=tid, _is_direct_chat=direct,
                      origin_message_text="Keep the completed effect; continue the analysis",
                      origin_message_ref={"chat_id": 0, "client_message_id": "original"},
                      billing_group={"billing_group_id": tid, "billing_group_limit_usd": 20.0,
                          "billing_group_limit_source": "initial_task_admission",
                          "billing_group_limit_revision": "original"})
    ctx, limit = _loop_ctx(root, tid, direct=direct, attempt=attempt)
    from tests.test_context_fit_v664 import _plan
    # Use the real loop's governance + initial-input shape, matching its fit plan.
    limit.messages[:1] = _plan().messages_for("max")
    extra = {}
    if reason == "owner_pause":
        fence, _ = owner_pause.install_fence(root, tid, request_id="pause-once")
        extra["pause"] = {"fence_id": fence["fence_id"], "generation": 1, "root_task_id": tid,
                          "detached_reviews": [{"operation_id": "paid-review", "state": "in_flight"}],
                          "external_runs": {"runs": [], "custody_read": "ok"}}
    elif reason == "sleep":
        ctx._model_sleep = {"sleep_id": "sleep-once", "mode": "warm",
                            "wake_at": "2099-01-01T00:00:00+00:00", "senders": []}
    elif reason == "review":
        extra["review_binding"] = "paid-review"
    # call_a completed; call_b was sent without a result. Neither may replay.
    wait = owner_wait.checkpoint_owner_wait(ctx, limit.messages, {}, limit.accumulated_usage,
                                            4, [], {"seen-owner"}, **extra)
    owner_wait.set_owner_wait(root, tid, {**wait, "state": "waiting"})
    task = {"id": tid, "type": "task", "chat_id": 0, "root_task_id": tid, "_attempt": attempt}
    workers.RUNNING[tid] = {"task": task, "worker_id": 0, "attempt": attempt}
    return ctx, limit, wait, task


def _handoff(root, workers, wait, *, legacy=False, include_sleep=True, ack=False):
    tid = "saved-warm"
    if legacy:
        # Exact carrier the old common selector wrote, before any stop mutation.
        workers.RUNNING[tid]["task"]["_owner_wait_resume"] = {
            **wait, "state": "waiting", "restart_transaction_id": "restart-once"}
        selected = {tid}
    else:
        selected = owner_wait.prepare_owner_wait_handoffs(root, workers.RUNNING, "restart-once",
                                                          include_sleep=include_sleep)
    restart_retention.prepare_restart_returns(root, workers.RUNNING, workers.PENDING,
                                               transaction_id="restart-once", owner_wait_ids=selected)
    if ack:
        transaction = delegate_recovery._read_restart_transaction(root, "restart-once")
        delegate_recovery._write_restart_transaction(root, {
            **transaction, "status": "normal_exit_acknowledged", "exit_code": 42})
    return selected


def _stop_boot(queue, workers, selected):
    workers.kill_workers(force=True, terminal_status="cancelled", result_reason="Server shutdown.",
                         stop_source="server_shutdown", retain_saved_work=True, preserve_pending=True,
                         preserve_running_task_ids=selected)
    workers.PENDING.clear()
    workers.RUNNING.clear()
    queue.restore_pending_from_snapshot()
    return next(task for task in workers.PENDING if task["id"] == "saved-warm")


@pytest.mark.parametrize("reason", ["owner_pause", "sleep", "owner", "review"])
@pytest.mark.parametrize("ack", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_current_and_legacy_handoffs_keep_source_and_applicable_hold(tmp_path, monkeypatch, reason, ack, legacy):
    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _, limit, wait, _ = _park(tmp_path, workers, reason)
    original = read_actor_source_bytes(tmp_path, "saved-warm", wait["source_ref"])
    selected = _handoff(tmp_path, workers, wait, legacy=legacy, ack=ack)
    if not legacy:
        assert selected == (set() if reason == "owner_pause" else {"saved-warm"})
    task = _stop_boot(queue, workers, selected)
    assert read_actor_source_bytes(tmp_path, "saved-warm", wait["source_ref"]) == original
    current = load_task_result(tmp_path, "saved-warm")
    if reason == "owner_pause" or reason == "sleep" and not ack:
        pause = current["budget_pause"]
        assert pause["state"] == "paused" and pause["exact_continuation"]
        assert not task.get("_owner_wait_resume")
        assert task["_attempt"] == 1
        source = pause["source_ref"]
        if reason == "sleep":
            assert budget_hold_fact(task)["reason"] == "saved_sleep_recovery"
        else:
            assert owner_pause.fence_closed(owner_pause.read_fence(tmp_path, "saved-warm"))
    elif ack:
        assert task["_owner_wait_resume"]["wait_id"] == wait["wait_id"]
        assert budget_hold_fact(task) is None
        source = wait["source_ref"]
    else:
        assert budget_hold_fact(task)["reason"] == "saved_work_hold"
        source = task["_working_recovery"]["source_ref"]
    assert json.loads(read_actor_source_bytes(tmp_path, "saved-warm", source))["messages"] == limit.messages


@pytest.mark.parametrize("project_hold", [False, True])
def test_legacy_pause_under_project_hold_is_not_cancelled(tmp_path, monkeypatch, project_hold):
    from supervisor.task_admission import revalidate_project_holds
    from ouroboros.cancel_intents import has_active_intent

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _, _, wait, task = _park(tmp_path, workers, "owner_pause")
    if project_hold:
        task["_project_admission_restore_hold"] = {"reason": "project_routing_fence_lookup_failed"}
    selected = _handoff(tmp_path, workers, wait, legacy=True)
    restored = _stop_boot(queue, workers, selected)
    revalidate_project_holds()
    assert restored["_budget_pause"]["exact_continuation"]
    assert not has_active_intent(tmp_path, "saved-warm", strict=True)


@pytest.mark.parametrize("prior_pause", [False, True])
def test_direct_warm_owner_pause_retains_current_attempt(tmp_path, monkeypatch, prior_pause):
    from ouroboros.cancel_intents import has_active_intent

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    _, limit, wait, _ = _park(tmp_path, workers, "owner_pause", direct=True, attempt=2)
    if prior_pause:
        write_task_result(tmp_path, "saved-warm", STATUS_RUNNING, budget_pause={
            "state": "resumed", "task_attempt": 1, "is_direct_chat": True,
            "source_ref": wait["source_ref"], "pause_id": "old-consumed"})
    workers.RUNNING.clear()
    monkeypatch.setattr(queue, "PRIOR_DIRECT_ROOTS", {"task_ids": ["saved-warm"]})
    queue.persist_queue_snapshot(reason="before-crash")
    queue.restore_pending_from_snapshot()
    task = next(row for row in workers.PENDING if row["id"] == "saved-warm")
    pause = load_task_result(tmp_path, "saved-warm")["budget_pause"]
    assert task["_attempt"] == pause["task_attempt"] == 2
    assert pause["rail"] == "owner_pause" and pause["state"] == "paused"
    assert json.loads(read_actor_source_bytes(tmp_path, "saved-warm", pause["source_ref"]))["messages"] == limit.messages
    assert not has_active_intent(tmp_path, "saved-warm", strict=True)


@pytest.mark.parametrize("reason", ["owner_pause", "sleep"])
def test_retained_warm_pause_resumes_without_replaying_effects(tmp_path, monkeypatch, reason):
    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    ctx, limit, wait, _ = _park(tmp_path, workers, reason)
    selected = _handoff(tmp_path, workers, wait, legacy=True)
    task = _stop_boot(queue, workers, selected)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    before_group = load_task_result(tmp_path, "saved-warm")["billing_group"]
    outcome = queue.resume_budget_paused_task("saved-warm")
    assert outcome["ok"], outcome
    ctx.budget_pause_resume = task["_budget_pause_resume"]
    saved = budget_pause.load_budget_pause(ctx)
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_k: (None, "max"))
    messages, trace, usage, seen = [], {}, {}, set()
    model, effort, _, mode, round_idx, _ = budget_pause.resume_paused_loop(
        SimpleNamespace(_ctx=ctx), saved, messages, trace, usage, seen, budget_remaining_usd=5.0)
    assert (model, effort, mode, round_idx) == ("m", "high", "max", 4)
    assert messages[:len(limit.messages)] == limit.messages
    unknown = [m for m in messages if m.get("tool_call_id") == "call_b" and m.get("role") == "tool"]
    assert len(unknown) == 1 and "UNKNOWN" in unknown[0]["content"]
    assert usage["cost"] == 1.25 and "seen-owner" in seen
    assert load_task_result(tmp_path, "saved-warm")["billing_group"] == before_group
    assert budget_pause.budget_pause_row(tmp_path, "saved-warm")["grant"]["consumed_at"]
    with pytest.raises(ValueError):
        budget_pause.load_budget_pause(ctx, task["_budget_pause_resume"])


@pytest.mark.parametrize("ack", [False, True])
@pytest.mark.parametrize("reason", ["owner_pause", "sleep"])
def test_a_resumed_owner_pause_returns_at_the_next_stop_from_its_newer_work(tmp_path, monkeypatch, ack, reason):
    """19.10.2026: an owner Pause landed while the task's own review ran, the update's stop
    retained that warm wait into an exact pause, and Resume continued the SAME attempt. The
    next update's stop promised to return the task, but restore read the old ``retained``
    wait as a live park, refused the newer working state and cancelled the task. The Resume
    ended that park: the task returns under its own id from the work it did after it."""
    from ouroboros import working_checkpoint
    from ouroboros.cancel_intents import has_active_intent
    from tests.test_restart_retention import _supervisor_rows

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    ctx, limit, wait, _ = _park(tmp_path, workers, reason)
    task = _stop_boot(queue, workers, _handoff(tmp_path, workers, wait, legacy=True))
    assert load_task_result(tmp_path, "saved-warm")["owner_wait"]["state"] == "retained"
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    assert queue.resume_budget_paused_task("saved-warm")["ok"]
    ctx.budget_pause_resume = task["_budget_pause_resume"]
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_k: (None, "max"))
    messages, trace, usage, seen = [], {}, {}, set()
    budget_pause.resume_paused_loop(SimpleNamespace(_ctx=ctx), budget_pause.load_budget_pause(ctx),
                                    messages, trace, usage, seen, budget_remaining_usd=5.0)
    assert budget_pause.budget_pause_row(tmp_path, "saved-warm")["state"] == budget_pause.STATE_RESUMED
    # The dispatched attempt works on and saves its newer state; then the next update stops it.
    workers.PENDING.remove(task)
    workers.RUNNING["saved-warm"] = {"task": task, "worker_id": 0, "attempt": 1}
    limit.messages = messages + [{"role": "user", "content": "[owner] work after the Resume"}]
    working_checkpoint.save_round(limit, "ready")
    returns = restart_retention.prepare_restart_returns(tmp_path, workers.RUNNING, workers.PENDING,
                                                        transaction_id="restart-twice")
    assert returns == {"saved-warm"}
    if ack:
        transaction = delegate_recovery._read_restart_transaction(tmp_path, "restart-twice")
        delegate_recovery._write_restart_transaction(tmp_path, {
            **transaction, "status": "normal_exit_acknowledged", "exit_code": 42})
    returned = _stop_boot(queue, workers, set())
    assert not has_active_intent(tmp_path, "saved-warm", strict=True)
    assert load_task_result(tmp_path, "saved-warm")["status"] == STATUS_SCHEDULED, "queued again, not cancelled"
    assert returned["_attempt"] == 2
    source = returned["_working_recovery"]
    assert source["source_kind"] == working_checkpoint.SOURCE_WORKING
    frozen = json.loads(read_actor_source_bytes(tmp_path, "saved-warm", source["source_ref"]))
    assert frozen["messages"][-1]["content"] == "[owner] work after the Resume"
    hold = budget_hold_fact(returned)
    assert hold is None if ack else hold["reason"] == "saved_work_hold"
    assert not [row for row in _supervisor_rows(tmp_path) if row.get("type") == "restart_return_refused"]


def test_a_promised_return_that_cannot_come_back_says_why(tmp_path, monkeypatch):
    """The other direction: a returned id with nothing saved keeps the ordinary interrupted
    path, and the refusal is a typed supervisor fact, not silence."""
    from tests.test_restart_retention import _supervisor_rows

    _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "nothing-saved", STATUS_RUNNING, chat_id=0)
    row = {"task": {"id": "nothing-saved", "type": "task", "chat_id": 0, "_attempt": 1}, "attempt": 1}
    assert restart_retention.recover_saved_running(
        [row], tmp_path, transaction={"return_ids": ["nothing-saved"]}, cause="app_stop") == []
    [refused] = [r for r in _supervisor_rows(tmp_path) if r.get("type") == "restart_return_refused"]
    assert refused["refused"] == {"nothing-saved": "no_usable_recovery_source"} and refused["cause"] == "app_stop"
    assert restart_retention.recover_saved_running([row], tmp_path, transaction={}, cause="app_stop") == []
    assert len([r for r in _supervisor_rows(tmp_path) if r.get("type") == "restart_return_refused"]) == 1, \
        "a row nothing promised to return is the ordinary path, not a refusal"


@pytest.mark.parametrize("reason", ["orphaned_running_after_worker_restart", "interrupted_retry_lost"])
def test_historical_orphan_continue_uses_known_cause_and_keeps_vetoes(tmp_path, monkeypatch, reason):
    from ouroboros.owner_continue import continuation_eligibility
    from supervisor.continuation_admission import admit_continuation
    from tests.test_owner_continue import _interrupted, NONCE

    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, reason_code=reason)
    assert continuation_eligibility(load_task_result(tmp_path, "pred-1"), "pred-1")["cause"] == reason
    result = admit_continuation("pred-1", action_nonce=NONCE)
    assert result["ok"], result
    assert admit_continuation("pred-1", action_nonce=NONCE)["successor_task_id"] == result["successor_task_id"]
    assert len(workers.PENDING) == 1
    original = load_task_result(tmp_path, "pred-1")
    for source in ("owner", "panic"):
        blocked = continuation_eligibility({**original, "cancel_origin": {"source": source}}, "pred-1")
        assert not blocked["eligible"] and blocked["refusal"] == "stopped_by_owner"
    assert not continuation_eligibility({**original, "deadline_at": "2000-01-01T00:00:00Z"}, "pred-1")["eligible"]


@pytest.mark.parametrize("door", ["planned", "update", "manual"])
@pytest.mark.parametrize("reason", ["owner_pause", "sleep"])
@pytest.mark.parametrize("ack", [False, True])
def test_real_stop_door_selection_and_boot(tmp_path, monkeypatch, door, reason, ack):
    import threading
    import server
    from ouroboros import server_restart
    from ouroboros.gateway import control
    from supervisor import git_ops

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _park(tmp_path, workers, reason)
    monkeypatch.setattr(workers, "direct_chat_turns", lambda: [])
    monkeypatch.setattr(server_restart, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server_restart, "_LAST_RETURN_TX", {"id": ""})
    monkeypatch.setattr(server_restart, "_OWNER_STOP_BEGAN", threading.Event())
    monkeypatch.setattr(server_restart, "_owner_restart_requested", threading.Event())
    monkeypatch.setattr(server_restart, "_LAST_RETURNING", set())
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                          kill_workers=workers.kill_workers, load_state=lambda: {},
                          update_state=lambda mutate: mutate({}), safe_restart=lambda **_k: (True, ""),
                          persist_queue_snapshot=queue.persist_queue_snapshot)
    if door == "planned":
        monkeypatch.setattr(server, "_safe_restart_serialized", lambda *_a, **_k: (True, ""))
        monkeypatch.setattr(server, "_request_restart_exit", lambda: None)
        monkeypatch.setattr(server, "_planned_delegate_restart_transaction_id", "")
        server._perform_supervisor_restart(ctx)
    elif door == "update":
        monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path)
        monkeypatch.setattr(restart_retention, "_update_returns", {})
        monkeypatch.setattr(workers, "close_repo_writer_admission", lambda *_a: None)
        monkeypatch.setattr(workers, "drain_repo_writers", lambda: [])
        monkeypatch.setattr(workers, "open_repo_writer_admission", lambda: None)
        monkeypatch.setattr("ouroboros.tools.services.kill_all_services", lambda *_a, **_k: [])
        monkeypatch.setattr("ouroboros.process_custody.quiesce_custodied_services", lambda *_a: (True, []))
        assert control._quiesce_repo_writers("managed_update") == []
    else:
        transaction = delegate_recovery.prepare_owner_restart(tmp_path)
        server_restart._LAST_RETURN_TX["id"] = transaction["transaction_id"]
        monkeypatch.setattr(server_restart, "_stop_owned_daemon", lambda *_a: None)
        monkeypatch.setattr("ouroboros.delegate_custody.reconcile_orphaned_runs", lambda *_a, **_k: None)
        monkeypatch.setattr("ouroboros.owned_shutdown.stop_owned_work", lambda *_a: None)
        server_restart._stop_owned_work(ctx)
    active = json.loads(delegate_recovery._active_restart_transaction_path(tmp_path).read_text())
    tx = delegate_recovery._read_restart_transaction(tmp_path, active["transaction_id"])
    if ack:
        delegate_recovery._write_restart_transaction(tmp_path, {**tx, "status": "normal_exit_acknowledged"})
    workers.PENDING.clear()
    workers.RUNNING.clear()
    monkeypatch.setattr(server_restart, "_OWNER_STOP_BEGAN", threading.Event())
    monkeypatch.setattr(delegate_recovery, "_PROCESS_GENERATION", "next-boot")
    queue.restore_pending_from_snapshot()
    [task] = workers.PENDING
    assert task["id"] == "saved-warm"
    if reason == "sleep" and ack and door != "manual":
        assert task["_owner_wait_resume"]["reason"] == "sleep" and budget_hold_fact(task) is None
    else:
        assert task["_budget_pause"]["exact_continuation"]
        if reason == "sleep":
            assert budget_hold_fact(task)
        else:
            assert owner_pause.fence_closed(owner_pause.read_fence(tmp_path, "saved-warm"))


def test_first_resumed_model_boundary_sees_tool_results_unknown_and_review_custody(tmp_path, monkeypatch):
    import copy
    from ouroboros import loop, agent
    from tests.test_working_checkpoint import _registry
    from tests.test_loop_transport_wait import _loop_kwargs

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _, limit, wait, _ = _park(tmp_path, workers, "owner_pause")
    selected = _handoff(tmp_path, workers, wait, legacy=True)
    task = _stop_boot(queue, workers, selected)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    assert queue.resume_budget_paused_task("saved-warm")["ok"]
    registry = _registry(tmp_path, monkeypatch, "saved-warm", 1)
    registry._ctx.model_wait_context = None
    agent._restore_saved_clocks(registry._ctx, task)
    observed = []

    class ModelBoundary(BaseException):
        pass

    def first_model(ctx, *_a, **_k):
        observed.append(copy.deepcopy(ctx.messages))
        raise ModelBoundary()

    monkeypatch.setattr(loop, "_dispatch_round_model", first_model)
    monkeypatch.setattr(registry, "execute_result", lambda *_a, **_k: pytest.fail("old tool replayed"))
    args = _loop_kwargs(tmp_path, registry, [])
    args["task_id"] = "saved-warm"
    with pytest.raises(ModelBoundary):
        loop.run_llm_loop(**args)
    [messages] = observed
    # Context rebind adds a text-block cache marker to the initial user input;
    # it must not change that input or any saved tool call/result.
    start = messages.index(limit.messages[0])
    content = messages[start + 1]["content"]
    assert content == limit.messages[1]["content"] or (
        isinstance(content, list) and len(content) == 1
        and content[0]["type"] == "text" and content[0]["text"] == limit.messages[1]["content"])
    assert messages[start + 2:start + len(limit.messages)] == limit.messages[2:]
    unknown = [m for m in messages if m.get("role") == "tool" and m.get("tool_call_id") == "call_b"]
    assert len(unknown) == 1 and "UNKNOWN" in unknown[0]["content"]
    notices = "\n".join(str(m.get("content") or "") for m in messages)
    assert "paid-review" in notices and "not current liveness or a verdict" in notices
    assert "not permission to dispatch it again" in notices
    from ouroboros.llm_messages import _MessageShapingMixin
    wire = _MessageShapingMixin._normalize_system_message_placement(messages)
    calls = next(i for i, m in enumerate(wire) if m.get("tool_calls"))
    assert [m.get("tool_call_id") for m in wire[calls + 1:calls + 3]] == ["call_a", "call_b"]
    assert budget_pause.budget_pause_row(tmp_path, "saved-warm")["grant"]["consumed_at"]


@pytest.mark.parametrize("reason,ack", [("owner_pause", False), ("owner_pause", True), ("sleep", False)])
def test_failed_legacy_conversion_keeps_project_custody_and_recovers(tmp_path, monkeypatch, reason, ack):
    from supervisor.task_admission import revalidate_project_holds
    from ouroboros.cancel_intents import has_active_intent

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _, _, wait, task = _park(tmp_path, workers, reason)
    task["_project_admission_restore_hold"] = {"reason": "project_routing_fence_lookup_failed"}
    selected = _handoff(tmp_path, workers, wait, legacy=True, ack=ack)
    name = "retain_owner_pause_park" if reason == "owner_pause" else "retain_sleep_checkpoint"
    with monkeypatch.context() as broken:
        def cannot_publish(*_a, **_k):
            raise OSError("controlled conversion write failure")
        broken.setattr(restart_retention, name, cannot_publish)
        restored = _stop_boot(queue, workers, selected)
        assert budget_hold_fact(restored)
        revalidate_project_holds()
        assert not has_active_intent(tmp_path, "saved-warm", strict=True)
    revalidate_project_holds()
    assert restored["_budget_pause"]["exact_continuation"]
    assert not restored.get("_owner_wait_resume")
    assert not has_active_intent(tmp_path, "saved-warm", strict=True)
    snapshot = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())
    [durable] = [row for row in snapshot["pending"] if row["id"] == "saved-warm"]
    assert durable["task"]["_budget_pause"] == restored["_budget_pause"]
    assert not durable["task"].get("_owner_wait_resume")
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    outcome = queue.resume_budget_paused_task("saved-warm")
    assert outcome["ok"], outcome


@pytest.mark.parametrize("reason", ["owner_pause", "sleep"])
def test_update_abort_preserves_pause_and_returns_existing_sleep(tmp_path, monkeypatch, reason):
    from ouroboros.gateway import control
    from supervisor import git_ops

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _park(tmp_path, workers, reason)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(restart_retention, "_update_returns", {})
    monkeypatch.setattr(workers, "direct_chat_turns", lambda: [])
    monkeypatch.setattr(workers, "close_repo_writer_admission", lambda *_a: None)
    monkeypatch.setattr(workers, "drain_repo_writers", lambda: [])
    monkeypatch.setattr("ouroboros.tools.services.kill_all_services", lambda *_a, **_k: [])
    monkeypatch.setattr("ouroboros.process_custody.quiesce_custodied_services", lambda *_a: (True, []))
    assert control._quiesce_repo_writers("managed_update") == []
    assert restart_retention.recover_aborted_update(tmp_path, workers.RUNNING, workers.PENDING)
    [task] = workers.PENDING
    if reason == "owner_pause":
        assert task["_budget_pause"]["exact_continuation"]
        assert owner_pause.fence_closed(owner_pause.read_fence(tmp_path, "saved-warm"))
    else:
        assert task["_owner_wait_resume"]["reason"] == "sleep"
        assert budget_hold_fact(task) is None


def test_acknowledged_sleep_returns_to_its_selected_wake_source(tmp_path, monkeypatch):
    from ouroboros import model_sleep, deadline_utils
    from datetime import datetime, timezone

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    ctx, limit, wait, _ = _park(tmp_path, workers, "sleep")
    selected = _handoff(tmp_path, workers, wait, ack=True)
    task = _stop_boot(queue, workers, selected)
    ctx.owner_wait_resume = task["_owner_wait_resume"]
    saved = owner_wait.load_owner_wait(ctx)
    wakes = []

    def wake_on_original_source(context, checkpoint):
        chosen = checkpoint["sleep"]
        assert chosen["wake_at"] == "2099-01-01T00:00:00+00:00"
        assert model_sleep.wake_reason(context, chosen) == ""
        monkeypatch.setattr(deadline_utils, "utc_now", lambda: datetime(2099, 1, 2, tzinfo=timezone.utc))
        wake = model_sleep.wake_reason(context, chosen)
        wakes.append(wake)
        return wake

    ctx.owner_wait_callback = wake_on_original_source
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_k: (None, "max"))
    messages, trace, usage, seen = [], {}, {}, set()
    result = owner_wait.resume_native_loop(SimpleNamespace(_ctx=ctx), saved, messages, trace, usage, seen)
    assert wakes == ["timeout"] and result[4] == 4
    assert messages[:len(limit.messages)] == limit.messages
    assert usage["cost"] == 1.25 and "seen-owner" in seen


@pytest.mark.parametrize("reason", ["owner_pause", "sleep"])
def test_panic_overrides_acknowledged_warm_return(tmp_path, monkeypatch, reason):
    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _, _, wait, _ = _park(tmp_path, workers, reason)
    selected = _handoff(tmp_path, workers, wait, legacy=True, ack=True)
    (tmp_path / "state" / "panic_stop.flag").write_text("panic", encoding="utf-8")
    task = _stop_boot(queue, workers, selected)
    assert not task.get("_owner_wait_resume")
    assert task["_budget_pause"]["exact_continuation"]
    if reason == "sleep":
        from supervisor.events_budget import HOLD_PANIC
        assert budget_hold_fact(task)["reason"] == HOLD_PANIC
    else:
        assert owner_pause.fence_closed(owner_pause.read_fence(tmp_path, "saved-warm"))


def test_unreadable_unrelated_row_does_not_remove_manual_restart_returns(tmp_path, monkeypatch):
    from ouroboros import server_restart

    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _park(tmp_path, workers, "owner")
    workers.RUNNING["unreadable"] = {"task": {"id": "unreadable", "_attempt": 1}, "attempt": 1}
    read = owner_wait.load_task_result

    def fault(root, tid, **kwargs):
        if tid == "unreadable":
            raise ValueError("controlled unreadable unrelated result")
        return read(root, tid, **kwargs)

    monkeypatch.setattr(owner_wait, "load_task_result", fault)
    monkeypatch.setattr(server_restart, "DATA_DIR", tmp_path)
    transaction = delegate_recovery.prepare_owner_restart(tmp_path)
    monkeypatch.setattr(server_restart, "_LAST_RETURN_TX", {"id": transaction["transaction_id"]})
    monkeypatch.setattr(server_restart, "_LAST_RETURNING", set())
    monkeypatch.setattr(workers, "direct_chat_turns", lambda: [])
    selected, _ = server_restart._prepare_owner_restart_returns(SimpleNamespace(
        RUNNING=workers.RUNNING, PENDING=workers.PENDING))
    assert selected == {"saved-warm"}


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("ack", [False, True])
def test_conversion_failure_without_project_hold_recovers_on_resume(tmp_path, monkeypatch, legacy, ack):
    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _, _, wait, _ = _park(tmp_path, workers, "owner_pause")
    selected = _handoff(tmp_path, workers, wait, legacy=legacy, ack=ack)
    with monkeypatch.context() as broken:
        def cannot_publish(*_a, **_k):
            raise OSError("controlled non-Project conversion failure")
        broken.setattr(restart_retention, "retain_owner_pause_park", cannot_publish)
        restored = _stop_boot(queue, workers, selected)
        assert not restored.get("_budget_pause")
        assert not restored.get("_project_admission_restore_hold")
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    outcome = queue.resume_budget_paused_task("saved-warm")
    assert outcome["ok"], outcome
    restored = next(t for t in workers.PENDING if t["id"] == "saved-warm")
    assert restored["_budget_pause_resume"]["grant_id"]
    assert not restored.get("_owner_wait_resume")
    snapshot = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())
    [durable] = [row for row in snapshot["pending"] if row["id"] == "saved-warm"]
    assert durable["task"]["_budget_pause_resume"] == restored["_budget_pause_resume"]
    assert not durable["task"].get("_owner_wait_resume")


def test_a_closed_wait_with_a_retention_hold_still_reaches_shutdown_custody(tmp_path, monkeypatch):
    from supervisor.task_admission import revalidate_project_holds
    from ouroboros.cancel_intents import has_active_intent

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _, _, wait, task = _park(tmp_path, workers, "owner_pause")
    task["_project_admission_restore_hold"] = {"reason": "project_routing_fence_lookup_failed"}
    selected = _handoff(tmp_path, workers, wait, legacy=True)
    with monkeypatch.context() as broken:
        def fail(*_a, **_k):
            raise OSError("controlled conversion failure")
        broken.setattr(restart_retention, "retain_owner_pause_park", fail)
        restored = _stop_boot(queue, workers, selected)
    assert budget_hold_fact(restored)
    owner_wait.set_owner_wait(tmp_path, "saved-warm", {**wait, "state": "resumed"})
    revalidate_project_holds()
    assert has_active_intent(tmp_path, "saved-warm", strict=True)


@pytest.mark.parametrize("ack", [False, True])
@pytest.mark.parametrize("project_hold", [False, True])
@pytest.mark.parametrize("reason", ["owner", "owner_pause"])
def test_unreadable_wait_does_not_abort_other_pending_restore(tmp_path, monkeypatch, ack, project_hold, reason):
    from ouroboros import task_results
    from ouroboros.work_stop import continuation_refusal
    from ouroboros.cancel_intents import has_active_intent

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _, _, wait, task = _park(tmp_path, workers, reason)
    if project_hold:
        task["_project_admission_restore_hold"] = {"reason": "project_routing_fence_lookup_failed"}
    from tests.test_restart_retention import _queued
    from ouroboros.task_results import STATUS_SCHEDULED
    write_task_result(tmp_path, "bystander", STATUS_SCHEDULED, chat_id=0)
    workers.PENDING.append(_queued("bystander", root_task_id="bystander"))
    selected = _handoff(tmp_path, workers, wait, legacy=True, ack=ack)
    workers.kill_workers(force=True, terminal_status="cancelled", result_reason="Server shutdown.",
                         stop_source="server_shutdown", retain_saved_work=True, preserve_pending=True,
                         preserve_running_task_ids=selected)
    workers.RUNNING.clear()
    workers.PENDING.clear()
    original = task_results.load_task_result

    def unreadable(root, tid, **kwargs):
        if tid == "saved-warm":
            raise ValueError("controlled unreadable wait result")
        return original(root, tid, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(task_results, "load_task_result", unreadable)
        fault.setattr(owner_wait, "load_task_result", unreadable)
        queue.restore_pending_from_snapshot()
    restored = {t["id"]: t for t in workers.PENDING}
    assert {"saved-warm", "bystander"} <= restored.keys()
    assert restored["saved-warm"]["_owner_wait_resume"]["wait_id"] == wait["wait_id"]
    assert continuation_refusal(restored["saved-warm"]) and not restored["saved-warm"].get("_resume_selection")
    assert not restored["saved-warm"].get("_terminalization_retry")
    assert not has_active_intent(tmp_path, "saved-warm", strict=True)
    if reason == "owner_pause":
        from supervisor import state
        monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
        assert queue.resume_budget_paused_task("saved-warm")["ok"]
        row = next(t for t in workers.PENDING if t["id"] == "saved-warm")
        assert row["_budget_pause_resume"]["grant_id"] and not row.get("_terminalization_retry")
