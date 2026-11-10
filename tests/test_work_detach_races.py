"""Same-id direct handoff at durable prefixes, with real boot and Pause owners.

The direct turn is an inline executor's (a registered actor of this process with its RUNNING lane row on
a pseudo-slot); its exact pause is parked by the pooled path, the supervisor's ``budget_pause`` handler
over the row, as every carried turn's is."""

import copy
import json
from threading import RLock
from types import SimpleNamespace

import pytest

from ouroboros import budget_pause as bp
from ouroboros.task_results import load_task_result, write_task_result
from tests._budget_pause_exact_helpers import _loop_ctx
from tests.test_project_hold_recovery import worker
from tests.test_swarm_host_admission import host  # noqa: F401
from tests.test_work_intent_races import Crash
from ouroboros.task_carrier import EXECUTOR_INLINE_DEGRADED
from tests.conversation_carrier_support import lane_row, park_lane_pause

INLINE_WID = 97  # the inline executor's pseudo-slot



def direct(host, monkeypatch):  # noqa: F811
    from ouroboros import working_checkpoint
    from supervisor import active_activity, queue, workers

    sent = worker(host, monkeypatch)
    monkeypatch.setattr(host.ctx.bridge, "push_log", lambda _event: None, raising=False)
    registry = active_activity.DirectActivityRegistry()
    monkeypatch.setattr(active_activity, "_DIRECT_ACTIVITY_REGISTRY", registry)
    ctx, limit = _loop_ctx(host.root, "head", direct=True)
    ctx.current_chat_id = 1
    ctx.task_metadata = {"tool_profile": "operator_control"}
    limit.messages.append({"role": "tool", "tool_call_id": "call_b", "content": "done b"})
    task = {"id": "head", "type": "task", "root_task_id": "head", "chat_id": 1, "text": "work",
            "_is_direct_chat": True, "_attempt": 1, "tool_profile": "operator_control"}
    write_task_result(host.root, "head", "running", _is_direct_chat=True, chat_id=1,
                      tool_profile="operator_control", task_attempt=1)
    actor = SimpleNamespace(_owner_message_admission_lock=RLock(), _busy=True,
        _accepting_owner_messages=True, _current_task_id="head", _current_chat_id=1,
        _current_task_metadata=ctx.task_metadata, _current_task_text="work", tools=limit.tools)
    lane_row(workers, task, wid=INLINE_WID, executor=EXECUTOR_INLINE_DEGRADED, pid=None)
    registry.register("head", 1, actor=actor)
    assert working_checkpoint.save_round(limit, "pre_effect")
    assert queue.persist_queue_snapshot(reason="direct-before-detach")
    return ctx, limit, task, actor, registry, sent


def arm(ctx):
    from ouroboros.tools.control_mind import _continue_in_background

    result = json.loads(_continue_in_background(ctx, "continue the same work"))
    assert result["status"] == "detach_armed", result
    assert "browser" in result["note"] and "REPL" in result["note"]


def park(limit, task):
    """The body's pause event, parked by the supervisor over the lane row; the pseudo-slot leaves with the body."""
    from supervisor import workers

    with pytest.raises(bp.BudgetPauseRequested) as stopped:
        bp.enter_cold_sleep(limit)
    park_lane_pause(bp.pause_event(task, stopped.value.pause))
    assert "head" not in workers.RUNNING and [row["id"] for row in workers.PENDING] == ["head"]
    workers.WORKERS.pop(INLINE_WID, None)


@pytest.mark.parametrize("cut", ["arm", "seed", "wake_grant"])
def test_detach_crash_prefix_restores_the_same_work_and_runs_it_once(host, monkeypatch, cut):  # noqa: F811
    from ouroboros import owner_wait, working_checkpoint
    from supervisor import queue, workers
    from supervisor.events_budget import budget_hold_fact
    from supervisor.sleep_wake import wake_ready_sleepers

    ctx, limit, task, actor, registry, sent = direct(host, monkeypatch)
    arm(ctx)
    if cut != "arm":
        assert working_checkpoint.save_round(limit, "post_batch")
    if cut in {"seed", "source", "pause_row"}:
        with monkeypatch.context() as faults:
            if cut == "source":
                real_source = owner_wait.store_continuation_source

                def save_source(*args, **kwargs):
                    real_source(*args, **kwargs)
                    raise Crash

                faults.setattr(owner_wait, "store_continuation_source", save_source)
            else:
                real_write = bp.set_budget_pause

                def write(root, tid, row, **kwargs):
                    result = real_write(root, tid, row, **kwargs)
                    if bool(row.get("source_ref")) == (cut == "pause_row"):
                        raise Crash
                    return result

                faults.setattr(bp, "set_budget_pause", write)
            with pytest.raises(Crash):
                bp.enter_cold_sleep(limit)
    elif cut in {"park", "unregister", "wake_grant"}:
        with pytest.raises(bp.BudgetPauseRequested) as stopped:
            bp.enter_cold_sleep(limit)
        # The body returns its pause event; the supervisor parks the row (pooled custody) and persists
        # the snapshot under the same lock. "park": the process dies right after that park, the inline
        # actor still registered; "unregister": after the actor left too; "wake_grant": after the grant.
        park_lane_pause(bp.pause_event(task, stopped.value.pause))
        assert registry.get("head") is not None and [row["id"] for row in host.pending] == ["head"]
        if cut != "park":
            registry.unregister("head")
        if cut == "wake_grant":
            assert wake_ready_sleepers(queue)[0]["ok"]
            assert bp.budget_pause_row(host.root, "head")["grant"]["selected_by"] == "sleep_wake"
            assert queue.persist_queue_snapshot(reason="grant-before-crash")

    # Only volatile process state is lost. No shutdown handler repairs the prefix.
    registry.unregister("head")
    workers.WORKERS.pop(INLINE_WID, None)
    host.pending.clear()
    host.running.clear()
    bp.end_dispatch_fence("head")
    queue.init(host.root)
    assert queue.restore_pending_from_snapshot() == 1
    [restored] = host.pending
    assert restored["id"] == "head" and restored["tool_profile"] == "operator_control"
    # An ordinary crash restores a saved detach as the same cold sleep, which wakes by its
    # own readiness. An earlier prefix waits for explicit Resume: held working state, or
    # the pause seed boot completed from it (no sleep selectors to wake by).
    wakes_itself = cut in {"pause_row", "park", "unregister", "wake_grant"}
    assert restored["_is_direct_chat"] and bool(budget_hold_fact(restored)) is (cut in {"arm", "checkpoint"})
    if cut in {"arm", "checkpoint"}:
        assert restored["_working_recovery"]
    else:
        assert restored["_budget_pause"]["reason"] == "sleep"
    from ouroboros.artifacts import read_actor_source_bytes

    saved = restored.get("_working_recovery") or bp.budget_pause_row(host.root, "head")
    cognition = json.loads(read_actor_source_bytes(host.root, "head", saved["source_ref"]))
    assert cognition["messages"] == limit.messages and cognition["round_idx"] == limit.round_idx
    assert "continued_by" not in load_task_result(host.root, "head")
    if cut in {"pause_row", "park", "unregister", "wake_grant"}:
        assert restored["detached_by"] == "mind" and restored["detached_at"] == ctx._model_sleep["detached_at"]
    for _ in range(2):
        woke = any(result.get("ok") for result in wake_ready_sleepers(queue))
        workers.assign_tasks()
        assert not woke or wakes_itself
    if wakes_itself:  # no Resume needed, and exactly one dispatch
        assert [row["id"] for row in sent] == ["head"] and list(host.running) == ["head"]
        return
    assert not sent and not host.running and len(host.pending) == 1
    resumed = queue.resume_budget_paused_task("head")
    assert resumed["ok"], resumed
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["head"] and list(host.running) == ["head"]


@pytest.mark.parametrize("blocker", ["no_checkpoint", "owner_stop"])
def test_seed_cannot_invent_saved_work_or_override_stop(host, monkeypatch, blocker):  # noqa: F811
    from ouroboros import working_checkpoint
    from ouroboros.cancel_intents import has_active_intent, request_cancel
    from supervisor import queue, workers

    ctx, limit, _task, _actor, registry, sent = direct(host, monkeypatch)
    arm(ctx)
    real_write = bp.set_budget_pause

    def write(*args, **kwargs):
        real_write(*args, **kwargs)
        raise Crash

    with monkeypatch.context() as faults:
        faults.setattr(bp, "set_budget_pause", write)
        with pytest.raises(Crash):
            bp.enter_cold_sleep(limit)
    if blocker == "no_checkpoint":
        working_checkpoint.discard(host.root, "head")
    else:
        request_cancel(host.root, "head", reason="owner_stop", source="http_single")
    registry.unregister("head")
    workers.WORKERS.pop(INLINE_WID, None)
    host.running.clear()
    bp.end_dispatch_fence("head")
    queue.init(host.root)
    assert queue.restore_pending_from_snapshot() == 0
    workers.assign_tasks()
    assert not sent and not host.pending and not host.running
    assert not bp.budget_pause_row(host.root, "head").get("source_ref")
    assert has_active_intent(host.root, "head", strict=True)


@pytest.mark.parametrize("pause_at", ["none", "after_grant", "launch"])
def test_owner_pause_vetoes_detach_wake_and_last_launch_gate(host, monkeypatch, pause_at):  # noqa: F811
    from ouroboros import project_placement
    from ouroboros.owner_pause import read_fence
    from supervisor import queue, sleep_wake, workers
    from supervisor.owner_pause_control import request_owner_pause

    ctx, limit, task, _actor, registry, sent = direct(host, monkeypatch)
    arm(ctx)
    park(limit, task)
    registry.unregister("head")
    fence = None

    def pause():
        nonlocal fence
        result = request_owner_pause("head", request_id="owner-pause")
        assert result["ok"], result
        fence = copy.deepcopy(read_fence(host.root, "head"))

    if pause_at == "readiness":
        real = sleep_wake._readiness

        def readiness(*args):
            reason = real(*args)
            assert reason
            pause()
            return reason

        monkeypatch.setattr(sleep_wake, "_readiness", readiness)
    outcomes = sleep_wake.wake_ready_sleepers(queue)
    assert any(row.get("ok") for row in outcomes) == (pause_at != "readiness"), outcomes
    if pause_at == "after_grant":
        pause()
    if pause_at == "launch":
        real = project_placement.readmit_placement

        def readmit(*args, **kwargs):
            allowed = real(*args, **kwargs)
            assert allowed
            pause()
            return allowed

        monkeypatch.setattr(project_placement, "readmit_placement", readmit)
    workers.assign_tasks()
    assert len(sent) == (1 if pause_at == "none" else 0)
    if fence:
        assert read_fence(host.root, "head") == fence
        assert not host.running and [row["id"] for row in host.pending] == ["head"]
        if pause_at == "launch":
            monkeypatch.setattr(project_placement, "readmit_placement", real)
            # The next chooser repairs the wake grant superseded after selection.
            workers.assign_tasks()
            assert not sent
        resumed = queue.resume_budget_paused_task("head")
        assert resumed["ok"], resumed
        workers.assign_tasks()
        assert [row["id"] for row in sent] == ["head"]
