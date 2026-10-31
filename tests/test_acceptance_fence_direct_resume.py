"""#1183: a direct turn's exact Pause continues on a pooled worker with its own fence.

The direct actor applies its admission fence (stored under the acceptance fence's names)
in-process, so the row names no pooled owner. A fence still open across a Pause is a
pre-fold state — the retired acceptance gate held it open while a child ran — that a paused
continuation still carries: its exact park binds that row; only the consumed Resume of THAT
pause on the current pooled attempt rebinds it. The fence begin, owner/budget Pause,
direct-lane park, Resume grant, assignment, loop consumption, pooled event/ack and the seal
at an author stop are real; model, process and paid transports are synthetic.
"""
import copy
import os
import queue as stdqueue
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _fast_hold, _install_queue, _loop_ctx, _quiet_external, _supervisor_ctx
from tests.test_acceptance_fence_transport import _pooled_agent, _Supervisor
from tests.worker_pool_support import affirm_ready

pytestmark = pytest.mark.serial

TASK = "direct-cold"


def _bind(ctx, actor):
    ctx.begin_acceptance_fence = actor._begin_acceptance_fence
    ctx.inspect_acceptance_fence = actor._inspect_acceptance_fence
    ctx.end_acceptance_fence = actor._end_acceptance_fence


def _direct_paused(tmp_path, monkeypatch, *, park="inline", rail="budget", change=""):
    """A direct turn's fence is open (as a pre-fold acceptance gate left it); the root then pauses exactly."""
    from ouroboros import budget_pause, loop
    from ouroboros.task_results import write_task_result
    from supervisor import message_bus, worker_chat_lane
    from supervisor.events_budget import _handle_budget_pause

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(queue, "ACCEPTANCE_FENCES", {})
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    direct = _pooled_agent(tmp_path, None, TASK)
    direct._fence_in_process = lambda: True  # a direct turn runs in the queue's process
    ctx, limit = _loop_ctx(tmp_path, TASK, direct=True)
    _bind(ctx, direct)
    write_task_result(tmp_path, TASK, "running", root_task_id=TASK, _is_direct_chat=True)
    trace = {"tool_calls": []}
    # The fence's own begin (the one the seal at an author stop takes) opens the row the
    # retired acceptance gate kept open across the pause.
    assert loop._begin_task_acceptance_fence(ctx, TASK)[0]
    row = queue.ACCEPTANCE_FENCES[TASK]
    assert "pooled_owner" not in row
    row["owner_message_generation"] = 3
    if change == "sealed":
        row["status"] = "sealed"
    elif change == "foreign_task":
        row["task_id"] = "another-task"
    elif change == "pooled_owner":  # a row some pooled attempt began is not this direct turn's
        row["pooled_owner"] = {"task_attempt": 1, "pid": 4242}
    prior = copy.deepcopy(row)
    _fast_hold(monkeypatch, budget_pause)
    _quiet_external(monkeypatch, budget_pause)
    registry = SimpleNamespace(unregister=lambda _id: None)
    if rail == "owner":  # the owner's Pause control finds the live direct turn and closes its tree
        from supervisor.active_activity import get_direct_activity_registry
        from supervisor.owner_pause_control import request_owner_pause

        registry = get_direct_activity_registry()  # conftest gives each test a fresh one
        registry.register(TASK, 7, actor=SimpleNamespace(_busy=True, _current_task_id=TASK,
                                                         _accepting_owner_messages=True, _current_chat_id=7))
        assert request_owner_pause(TASK, request_id="direct-owner-pause")["ok"]
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        if rail == "owner":
            budget_pause.enter_owner_pause(limit)  # saves the owner-rail exact pause and raises
        else:
            budget_pause.request_pause(limit, rail=budget_pause.RAIL_GLOBAL_EXHAUSTED, scope="global",
                                       reason_text="budget exhausted")
    task = {"id": TASK, "type": "task", "chat_id": 7, "text": "do it", "project_id": "fixture",
            "_is_direct_chat": True, "root_task_id": TASK, "_attempt": 1}
    event = budget_pause.pause_event(task, raised.value.pause)
    if park == "inline":  # the direct lane parks its own turn in-process
        events = []
        monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=events.append))
        monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(push_log=lambda _evt: None))
        assert worker_chat_lane._execute_chat_task({
            "task": task, "agent": SimpleNamespace(handle_task=lambda _task: [event]), "chat_id": 7,
            "registry": registry})
        assert not events
    else:  # the inline park failed: the same event takes the supervisor's ordinary path
        budget_pause.end_dispatch_fence(TASK)
        _handle_budget_pause(event, _supervisor_ctx(tmp_path, workers, queue, [], []))
    assert workers.PENDING[0]["_is_direct_chat"] and not workers.RUNNING
    assert {key: value for key, value in queue.ACCEPTANCE_FENCES[TASK].items()
            if key != "paused_continuation"} == prior, "Pause keeps its fence intact"
    return queue, workers, prior, trace


def _pooled_resume(tmp_path, monkeypatch, queue, workers, *, consume=True):
    """Owner Resume, assignment to a pooled worker, and that worker's real loop restoration."""
    from ouroboros import budget_pause, owner_wait

    assert queue.resume_budget_paused_task(TASK)["ok"]
    sent = []
    workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                         proc=SimpleNamespace(pid=os.getpid(), is_alive=lambda: True),
                                         in_q=SimpleNamespace(put=lambda row: sent.append(dict(row)))))
    workers.assign_tasks()
    assert len(sent) == 1 and TASK in workers.RUNNING
    ctx, _limit = _loop_ctx(tmp_path, TASK, direct=True)
    ctx.budget_pause_resume = sent[0]["_budget_pause_resume"]
    if consume:
        monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_k: (None, "max"))
        budget_pause.resume_paused_loop(SimpleNamespace(_ctx=ctx), budget_pause.load_budget_pause(ctx),
                                        [], {}, {}, set(), budget_remaining_usd=5.0)
        assert budget_pause.budget_pause_row(tmp_path, TASK)["grant"]["consumed_at"]
    events = stdqueue.Queue()
    _Supervisor(events, tmp_path)
    pooled = _pooled_agent(tmp_path, events, TASK)
    pooled.tools._ctx = ctx
    _bind(ctx, pooled)
    return ctx


@pytest.mark.parametrize("park,rail", [("inline", "budget"), ("event", "budget"), ("inline", "owner")])
def test_direct_exact_pause_reacquires_its_fence_on_pooled_resume(tmp_path, monkeypatch, park, rail):
    from ouroboros import budget_pause, loop
    from ouroboros.loop_delivery import _seal_admission_before_delivery

    queue, workers, prior, trace = _direct_paused(tmp_path, monkeypatch, park=park, rail=rail)
    owner = rail == "owner"
    assert budget_pause.budget_pause_row(tmp_path, TASK)["rail"] == (
        budget_pause.RAIL_OWNER_PAUSE if owner else budget_pause.RAIL_GLOBAL_EXHAUSTED)
    assert owner == ((queue.BUDGET_ROOT_FENCES.get(TASK) or {}).get("cause") == "owner_pause")
    assert queue.ACCEPTANCE_FENCES[TASK]["paused_continuation"]["direct_owner"] == {"task_attempt": 1}
    ctx = _pooled_resume(tmp_path, monkeypatch, queue, workers)
    opened, token = loop._begin_task_acceptance_fence(ctx, TASK)
    assert opened, opened
    row = queue.ACCEPTANCE_FENCES[TASK]
    assert token != prior["token"] and row["pooled_owner"] == {"task_attempt": 1, "pid": os.getpid()}
    assert row["owner_message_generation"] == 3 and "paused_continuation" not in row
    # The ended direct actor's token owns nothing, and no direct begin takes the row back.
    for action, outcome in [("inspect", ""), ("end", "revision"), ("end", "terminal")]:
        assert queue.transition_acceptance_fence(action=action, token=prior["token"], outcome=outcome)["row_absent"]
    for stale in (prior["token"], "f" * 32):
        assert not queue.transition_acceptance_fence(action="begin", token=stale, task_id=TASK, root_task_id=TASK)["ok"]
    controls = []
    monkeypatch.setattr(loop, "_arm_delivery_control", lambda *_a, **kw: controls.append(kw["control"]))
    assert _seal_admission_before_delivery(SimpleNamespace(_ctx=ctx), SimpleNamespace(task_id=TASK), trace)
    assert controls == [] and queue.ACCEPTANCE_FENCES[TASK]["status"] == "sealed"


@pytest.mark.parametrize("change", ["unconsumed", "revoked", "missing_binding", "attempt",
                                    "sealed", "foreign_task", "pooled_owner"])
def test_direct_handoff_needs_its_own_park_and_consumed_grant(tmp_path, monkeypatch, change):
    from ouroboros import budget_pause, loop

    queue, workers, _prior, _trace = _direct_paused(tmp_path, monkeypatch, change=change)
    row = queue.ACCEPTANCE_FENCES[TASK]
    assert ("paused_continuation" in row) == (change not in {"sealed", "foreign_task", "pooled_owner"})
    if change == "missing_binding":  # e.g. a direct row parked before this binding existed
        row.pop("paused_continuation")
    elif change == "attempt":
        row["paused_continuation"]["direct_owner"]["task_attempt"] = 2
    ctx = _pooled_resume(tmp_path, monkeypatch, queue, workers, consume=change != "unconsumed")
    if change == "revoked":
        current = budget_pause.budget_pause_row(tmp_path, TASK)
        current["grant"]["revoked_at"] = "revoked"
        budget_pause.set_budget_pause(tmp_path, TASK, current)
    before = copy.deepcopy(queue.ACCEPTANCE_FENCES)
    assert not loop._begin_task_acceptance_fence(ctx, TASK)[0]
    assert queue.ACCEPTANCE_FENCES == before
