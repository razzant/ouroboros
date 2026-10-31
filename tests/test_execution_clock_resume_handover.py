"""#939 x #1716: one execution clock through saved work, an exact owner Pause, its
capacity-delayed Resume, the Resume handover, an application stop and the next boot.

The real producers run: API ingress, Main admission and assignment, the application
stop teardown and boot restore, the owner's Resume doors, the real worker start
(``agent.handle_task`` up to its clock restore), the agent's pause handler and exact
park, the loop's consumption of the grant and its handover, and the supervisor's
consumption event. The wall/monotonic clock is faked and advances only where the
scenario says that work, queue or Pause time passes, so every spent time below is
an absolute oracle, never two clocks agreeing with each other (they could agree on
a wrong queue charge).
"""
from __future__ import annotations

import copy
import json
import queue as queue_lib
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import agent, budget_pause, loop, owner_wait, working_checkpoint as wc
from ouroboros.model_slots import MODEL_ACCOUNTS_KEY
from ouroboros.model_wait import TaskModelWait, execution_elapsed_seconds
from supervisor import queue, workers
from supervisor.events import dispatch_event
from supervisor.events_budget import budget_hold_fact, install_exact_budget_pause
from supervisor.owner_pause_control import request_owner_pause
from tests._budget_pause_exact_helpers import _loop_ctx
from tests._execution_clock_seams import agent_start
from tests.test_context_fit_v664 import _plan
from tests.test_loop_transport_wait import _loop_kwargs
from tests.test_resume_handover import MARK, ProcessLost, _account_route, _dispatch_one, _has_mark, _stop_boot
from tests.test_swarm_host_admission import host  # noqa: F401
from tests.test_working_checkpoint import _registry

pytestmark = pytest.mark.serial

# Seconds of each interval of the scenario.
WORK_1, HELD_1, WORK_2, PAUSED, CAPACITY, WORK_RESUMED, HELD_2 = 60, 600, 20, 900, 600, 10, 600


@pytest.fixture
def clock(monkeypatch):
    now = [time.time()]
    monkeypatch.setattr(time, "time", lambda: now[0])
    monkeypatch.setattr(time, "monotonic", lambda: now[0] - 1_000_000)
    return now


def _expected(budget, *intervals):
    """Spent execution: work always; held/queued saved work only without an explicit budget."""
    return sum(seconds for seconds, kind in intervals if kind == "work" or kind == "held" and not budget)


def _admit(host, budget):  # noqa: F811
    """The public API ingress (``/api/tasks``) with or without the new execution budget."""
    from starlette.requests import Request

    from ouroboros.gateway.tasks import _create_task_from_body

    request = Request({"type": "http", "app": SimpleNamespace(state=SimpleNamespace(
        drive_root=host.root, repo_dir=workers.REPO_DIR))})
    body = {"task_id": "held", "description": "Ordinary Main work"}
    if budget is not None:
        body["execution_timeout_sec"] = budget
    response = _create_task_from_body(request, body)
    assert response.status_code == 200, response.body
    [row] = [row for row in host.pending if row["id"] == "held"]
    return row


def _worker_start(host, task, monkeypatch):  # noqa: F811
    """The real worker start of ``task`` (origin decision and saved clocks); returns its wait owner."""
    with monkeypatch.context() as patched:
        return agent_start(task, host.root, patched)


def _metadata(task):
    """The execution fields the real agent projects into its tool context (``task_metadata``)."""
    return {key: task[key] for key in ("execution_timeout_sec", "_execution_consumed_sec")
            if task.get(key) not in (None, "")}


def _attempt_ctx(host, task, monkeypatch, *, attempt):  # noqa: F811
    """A loop context carrying the clock its real worker start restored, plus the loop's own
    restore of the saved working state (``ctx._budget_paused_sec``, read by a later pause row)."""
    waiter = _worker_start(host, task, monkeypatch)
    started = waiter.tool_context
    ctx, limit = _loop_ctx(host.root, "held", attempt=attempt)
    ctx.model_wait_context, ctx.task_metadata = waiter, started.task_metadata
    ctx.task_started_at, ctx.working_recovery = started.task_started_at, started.working_recovery
    if ctx.working_recovery:
        saved = loop._load_working_recovery(ctx, [])
        owner_wait.restore_continuation_state(SimpleNamespace(_ctx=ctx), saved, limit.messages, {}, {}, set())
    return ctx, limit, waiter


def _save_first_attempt(host, task, monkeypatch, clock):  # noqa: F811
    ctx, limit, waiter = _attempt_ctx(host, task, monkeypatch, attempt=1)
    clock[0] += WORK_1
    wc.save_round(limit, "post_batch")
    monkeypatch.setattr(workers, "WORKERS", {})  # the physical worker is doubled; its stop is real
    return waiter


def _pause_through_agent(host, task, ctx, limit, *, plan, round_idx=29, request_id="clock-pause"):  # noqa: F811
    """``round_idx`` meets the owner's Pause; the real agent handler parks it exactly."""
    assert request_owner_pause("held", request_id=request_id)["ok"]
    ctx.active_model, ctx.context_fit_plan = plan.model, plan
    limit.round_idx, limit.budget_tail = round_idx, "tool"
    limit.messages = [{"role": "user", "content": "do it"}, {"role": "assistant", "content": MARK, "tool_calls": [
        {"id": "call_a", "type": "function", "function": {"name": "x", "arguments": "{}"}},
        {"id": "call_b", "type": "function", "function": {"name": "y", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_a", "content": "done a"}]
    wc.save_round(limit, "post_batch")
    owner = object.__new__(agent.OuroborosAgent)
    owner.env = agent.Env(repo_dir=host.root, drive_root=host.root)
    owner.tools = SimpleNamespace(_ctx=ctx)
    owner._owner_message_admission_lock = threading.RLock()
    owner._incoming_messages = queue_lib.Queue()
    owner._event_queue = None
    owner._emit_live_log = lambda *a, **k: None
    owner._start_task_heartbeat_loop = lambda *a: None

    def prepare(*_args):
        budget_pause.enter_owner_pause(limit)
        pytest.fail("the owner pause did not park")

    owner._prepare_task_context = prepare
    events = owner._handle_task_scoped(task)
    budget_pause.end_dispatch_fence("held")
    assert any(event.get("type") == "budget_pause" for event in events)
    pause = budget_pause.budget_pause_row(host.root, "held")
    _park(host, pause)
    return pause


def _park(host, pause):  # noqa: F811
    """The supervisor side of an exact park (``events_budget.install_exact_budget_pause``)."""
    install_exact_budget_pause(SimpleNamespace(
        DRIVE_ROOT=host.root, RUNNING=workers.RUNNING, PENDING=workers.PENDING, WORKERS=workers.WORKERS,
        sort_pending=queue.sort_pending, persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None),
        "held", budget_pause.exact_pause_marker(pause)["checkpoint"])


def _supervisor_events(host):  # noqa: F811
    """The worker event channel: its budget-pause events reach the supervisor's real dispatcher."""
    ctx = SimpleNamespace(RUNNING=workers.RUNNING, DRIVE_ROOT=host.root, append_jsonl=lambda *_a, **_k: None)
    delivered = []

    def put(event):
        if event.get("type") == "budget_pause":
            delivered.append(dict(event))
            dispatch_event(dict(event), ctx)

    return SimpleNamespace(put=put, delivered=delivered)


def _loop_once(host, monkeypatch, task, *, plan, events=None, cut=None, expect=ProcessLost):  # noqa: F811
    """The real loop of one dispatched attempt until it continued (the process is lost there)."""
    registry = _registry(host.root, monkeypatch, task["id"], task["_attempt"])
    ctx = registry._ctx
    ctx.context_fit_plan = plan  # the new worker's initial plan (Main)
    monkeypatch.setattr("ouroboros.context._context_fit_route", _account_route)
    ctx.task_metadata = _metadata(task)
    ctx.task_started_at = agent._settle_start_origin(task)
    ctx.model_wait_context = TaskModelWait(task=task, drive_root=host.root, event_queue=None, worker_slot_held=False)
    agent._restore_saved_clocks(ctx, task)
    real_resume, seen = loop._resume_continuation, {"ctx": ctx}

    def resumed(*args, **kwargs):
        real_resume(*args, **kwargs)
        saved, pause, working = args[1]
        state = saved or pause or working
        seen.update(kind="pause" if pause else "working" if working else "none", round_idx=state.get("round_idx"),
                    task_attempt=state.get("task_attempt"), boundary=(state.get("working") or {}).get("boundary"),
                    messages=list(args[2]))
        raise ProcessLost("continued; nothing ordinary saved since")

    with monkeypatch.context() as fault:
        fault.setattr(loop, "_resume_continuation", resumed)
        fault.setattr(loop, "_dispatch_round_model", lambda *a, **k: pytest.fail("model dispatch before the cut"))
        if cut is not None:
            cut(fault, ctx)
        kwargs = _loop_kwargs(host.root, registry, [])
        kwargs["task_id"], kwargs["event_queue"] = task["id"], events
        with pytest.raises(expect) as raised:
            loop.run_llm_loop(**kwargs)
    seen["raised"] = raised.value
    return seen


def _fresh_plan(monkeypatch):
    """Main and fallback share one model on different accounts; every new worker starts on Main."""
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "account-a", "fallback": ["account-b"]}))
    return replace(_plan(), model="claudexor::codex=route-fixture-model")


def _spent(task_id="held"):
    meta = workers.RUNNING[task_id]
    return execution_elapsed_seconds(meta, time.time())


def _working_then_paused(host, monkeypatch, clock, budget, plan):  # noqa: F811
    """Saved attempt 1 → application stop → boot hold → Resume → attempt 2 works → owner Pause."""
    admitted = _admit(host, budget)
    assert admitted.get("execution_timeout_sec") == budget
    first = _dispatch_one(host, monkeypatch)
    assert first["_attempt"] == 1 and first.get("execution_timeout_sec") == budget
    _save_first_attempt(host, first, monkeypatch, clock)
    assert _stop_boot(monkeypatch) == 1
    [held] = host.pending
    assert held["_attempt"] == 2 and budget_hold_fact(held)
    clock[0] += HELD_1  # held for the owner's Resume, then waiting for a worker
    assert queue.resume_budget_paused_task("held")["ok"]
    second = _dispatch_one(host, monkeypatch)
    assert second["_working_recovery"]["from_attempt"] == 1
    assert _spent() == pytest.approx(_expected(budget, (WORK_1, "work"), (HELD_1, "held")))
    ctx, limit, waiter = _attempt_ctx(host, second, monkeypatch, attempt=2)
    assert waiter.executed_seconds() == pytest.approx(_spent())
    clock[0] += WORK_2
    pause = _pause_through_agent(host, second, ctx, limit, plan=plan)
    assert pause["started_at"] == pytest.approx(second["_working_recovery"]["started_at"])
    return pause




def _granted_after_capacity(host, monkeypatch, clock, pause):  # noqa: F811
    """The owner's Resume of the exact pause, then a capacity wait before a worker takes it."""
    clock[0] += PAUSED
    granted = queue.resume_budget_paused_task("held")
    assert granted["ok"] and granted["exact_continuation"], granted
    clock[0] += CAPACITY
    resumed = _dispatch_one(host, monkeypatch)
    assert resumed["_budget_pause_resume"]["pause_id"] == pause["pause_id"]
    return resumed


def _returned_after_stop(host, monkeypatch, clock):  # noqa: F811
    """Application stop, boot, the held row's owner Resume after HELD_2; the attempt it returns as."""
    assert _stop_boot(monkeypatch) == 1
    [row] = host.pending
    returned = copy.deepcopy(row)  # as the boot restored it, before Resume and assignment consume it
    assert budget_hold_fact(returned), "an application stop returns saved work only through Resume"
    clock[0] += HELD_2
    assert queue.resume_budget_paused_task("held")["ok"]
    return returned, _dispatch_one(host, monkeypatch)


# Absolute oracles: 60 s of attempt 1, 20 s of attempt 2 before the Pause, 10 s after its
# Resume. With an explicit budget only that work counts. Without one (absent or 0) the
# legacy clock is unchanged: the Pause (900 s) is excluded, but held saved work (600 s
# twice) and the granted Resume's capacity wait (600 s) are charged as before.
BEFORE_RESUME = {True: 80, False: 1280}
AT_STOP = {True: 90, False: 1290}
RETURNED = {True: 90, False: 1890}


@pytest.mark.parametrize("budget", [3600, 0, None], ids=["execution-budget", "zero-budget", "no-budget"])
@pytest.mark.parametrize("notified", [True, False], ids=["consumption-event", "event-lost-with-process"])
def test_spent_time_crosses_pause_capacity_resume_handover_stop_and_boot(
    host, monkeypatch, clock, budget, notified,  # noqa: F811
):
    bounded = bool(budget)
    fresh = _fresh_plan(monkeypatch)
    pause = _working_then_paused(host, monkeypatch, clock, budget, replace(fresh, model_role="fallback:0"))
    assert pause["paused_duration_sec"] == pytest.approx(HELD_1 if bounded else 0)
    resumed = _granted_after_capacity(host, monkeypatch, clock, pause)
    assert resumed["_attempt"] == 2
    assert _spent() == pytest.approx(BEFORE_RESUME[bounded]), "the Pause never counts; capacity only legacy"
    assert bool(resumed["_budget_pause_resume"].get("sleep_exclusion_since")) is bounded
    events = _supervisor_events(host) if notified else None
    seen = _loop_once(host, monkeypatch, resumed, plan=fresh, events=events)
    assert (seen["kind"], seen["round_idx"], _has_mark(seen["messages"])) == ("pause", 29, True)
    consumed = budget_pause.budget_pause_row(host.root, "held")
    excluded = PAUSED + (HELD_1 + CAPACITY if bounded else 0)
    assert consumed["state"] == budget_pause.STATE_RESUMED
    assert consumed["paused_duration_sec"] == pytest.approx(excluded), "grant to consumption folded once, if at all"
    handover = wc.read_working(host.root, "held", 2)[1]
    assert handover["working"]["boundary"] == wc.RESUMED and handover["context_model_role"] == "fallback:0"
    assert handover["started_at"] == pytest.approx(pause["started_at"])
    assert handover["model_wait"]["budget_paused_sec"] == pytest.approx(excluded), "folded BEFORE the handover"
    waiter = seen["ctx"].model_wait_context
    assert waiter.executed_seconds() == pytest.approx(BEFORE_RESUME[bounded])
    if not bounded:  # no live exclusion was granted, so there is nothing to notify
        assert "sleep_parked_at" not in workers.RUNNING["held"] and not (events and events.delivered)
    elif notified:
        assert [event["phase"] for event in events.delivered] == ["consumed"]
        assert "sleep_parked_at" not in workers.RUNNING["held"]
        assert workers.RUNNING["held"]["budget_paused_sec"] == pytest.approx(excluded)
    else:
        assert workers.RUNNING["held"].get("sleep_parked_at"), "the notification never reached the supervisor"
    clock[0] += WORK_RESUMED  # new work after the consumption; the process ends before any ordinary save
    assert waiter.executed_seconds() == pytest.approx(AT_STOP[bounded])
    row, final = _returned_after_stop(host, monkeypatch, clock)
    assert row["_attempt"] == 3 and row["_working_recovery"]["boundary"] == wc.RESUMED
    assert row.get("execution_timeout_sec") == budget
    if bounded:
        assert row["_execution_handoff_sec"] == pytest.approx(AT_STOP[bounded]), "the stop's own total, never less"
    else:
        assert "_execution_handoff_sec" not in row
    assert final.get("execution_timeout_sec") == budget and not final.get("_execution_consumed_sec")
    assert "_execution_handoff_sec" not in final
    assert _spent() == pytest.approx(RETURNED[bounded])
    assert _worker_start(host, final, monkeypatch).executed_seconds() == pytest.approx(RETURNED[bounded])
    restored = _loop_once(host, monkeypatch, final, plan=fresh)
    assert (restored["kind"], restored["task_attempt"], restored["round_idx"], restored["boundary"],
            _has_mark(restored["messages"])) == ("working", 2, 29, wc.RESUMED, True), "the latest cognition"
    ctx = restored["ctx"]
    assert ctx.model_wait_context.executed_seconds() == pytest.approx(RETURNED[bounded])
    assert ctx._budget_paused_sec == pytest.approx(ctx.model_wait_context.budget_paused_sec)
    plan = ctx.context_fit_plan
    assert (plan.model_role, plan.model_route["credentialProfileId"]) == ("fallback:0", "account-b")
    assert not host.attempts


def _lost_before_the_handover(fault, _ctx):
    def lose(*_args, **_kwargs):
        raise ProcessLost("before the handover write")

    fault.setattr(wc, "save_resumed", lose)


def _revoked_after_the_handover(fault, ctx):
    """A revocation (the real writer, as a new root budget fence issues it) lands between the
    handover write and the consumption's compare-and-set."""
    from supervisor.budget_resume import revoke_exact_budget_resume

    real = wc.save_resumed

    def write_then_revoke(*args, **kwargs):
        real(*args, **kwargs)
        waiter = ctx.model_wait_context
        assert waiter.sleep_started_monotonic is None, "the fold is installed for the handover write"
        assert revoke_exact_budget_resume(dict(workers.RUNNING["held"]["task"]), "new_root_budget_fence") is False

    fault.setattr(wc, "save_resumed", write_then_revoke)


@pytest.mark.parametrize("cut", ["lost_before_handover", "refused_at_cas"])
def test_a_failed_handover_consumes_no_time_and_the_next_resume_folds_once(
    host, monkeypatch, clock, cut,  # noqa: F811
):
    from ouroboros.task_results import load_task_result

    fresh = _fresh_plan(monkeypatch)
    pause = _working_then_paused(host, monkeypatch, clock, 3600, fresh)
    resumed = _granted_after_capacity(host, monkeypatch, clock, pause)
    lost = cut == "lost_before_handover"
    seen = _loop_once(host, monkeypatch, resumed, plan=fresh, events=_supervisor_events(host),
                      cut=_lost_before_the_handover if lost else _revoked_after_the_handover,
                      expect=ProcessLost if lost else budget_pause.BudgetPauseRequested)
    durable = budget_pause.budget_pause_row(host.root, "held")
    assert not durable["grant"].get("consumed_at"), "nothing consumed"
    assert durable["paused_duration_sec"] == pytest.approx(HELD_1), "the pause row's own carrier is unchanged"
    if lost:
        assert durable["state"] == budget_pause.STATE_RESUME_GRANTED
    else:
        assert durable["state"] == budget_pause.STATE_PAUSED and durable["grant"]["revoked_at"]
        waiter = seen["ctx"].model_wait_context
        assert waiter.sleep_started_monotonic is not None, "the live Pause exclusion was restored"
        assert waiter.budget_paused_sec == pytest.approx(HELD_1 + PAUSED), "the grant's carrier, unfolded"
        assert waiter.executed_seconds() == pytest.approx(BEFORE_RESUME[True])
        # The refused worker retires its handover under the live pause (the agent's handler, #1716).
        agent._retire_working_checkpoint(seen["ctx"], {"id": "held", "_attempt": 2})
        assert not wc.checkpoint_path(host.root, "held", 2).exists()
    clock[0] += WORK_RESUMED  # wall time while nothing ran: still the same Pause
    assert _stop_boot(monkeypatch) == 1
    [row] = host.pending
    assert row["_attempt"] == 2 and row["_budget_pause"]["reason"] == "owner", "the live pause owns recovery"
    resumed = _granted_after_capacity(host, monkeypatch, clock, pause)
    assert _spent() == pytest.approx(BEFORE_RESUME[True])
    events = _supervisor_events(host)
    seen = _loop_once(host, monkeypatch, resumed, plan=fresh, events=events)
    assert (seen["kind"], seen["round_idx"], _has_mark(seen["messages"])) == ("pause", 29, True)
    consumed = budget_pause.budget_pause_row(host.root, "held")
    assert consumed["state"] == budget_pause.STATE_RESUMED
    assert consumed["paused_duration_sec"] == pytest.approx(
        consumed["resumed_at"] - pause["paused_at"] + HELD_1), "every second since the Pause, folded once"
    assert seen["ctx"].model_wait_context.executed_seconds() == pytest.approx(BEFORE_RESUME[True])
    assert _spent() == pytest.approx(BEFORE_RESUME[True])
    assert load_task_result(host.root, "held").get("status") not in {"failed", "cancelled", "completed"}


def test_a_second_pause_after_the_handover_folds_only_its_own_interval_through_stop_and_boot(
    host, monkeypatch, clock,  # noqa: F811
):
    fresh = _fresh_plan(monkeypatch)
    first = _working_then_paused(host, monkeypatch, clock, 3600, fresh)
    resumed = _granted_after_capacity(host, monkeypatch, clock, first)
    seen = _loop_once(host, monkeypatch, resumed, plan=fresh, events=_supervisor_events(host))
    folded = budget_pause.budget_pause_row(host.root, "held")["paused_duration_sec"]
    assert folded == pytest.approx(HELD_1 + PAUSED + CAPACITY)
    clock[0] += WORK_RESUMED
    ctx = seen["ctx"]
    _, limit = _loop_ctx(host.root, "held", attempt=2)
    limit.tools = SimpleNamespace(_ctx=ctx)
    second = _pause_through_agent(host, resumed, ctx, limit, plan=fresh, round_idx=31, request_id="clock-pause-2")
    assert second["pause_id"] != first["pause_id"]
    assert second["paused_duration_sec"] == pytest.approx(folded), "the cumulative carrier, nothing re-added"
    assert second["started_at"] == pytest.approx(first["started_at"])
    again = _granted_after_capacity(host, monkeypatch, clock, second)
    assert _spent() == pytest.approx(AT_STOP[True])
    seen = _loop_once(host, monkeypatch, again, plan=fresh, events=_supervisor_events(host))
    assert (seen["kind"], seen["round_idx"]) == ("pause", 31)
    consumed = budget_pause.budget_pause_row(host.root, "held")
    assert consumed["paused_duration_sec"] == pytest.approx(folded + PAUSED + CAPACITY), "only its own interval"
    assert seen["ctx"].model_wait_context.executed_seconds() == pytest.approx(AT_STOP[True])
    clock[0] += WORK_RESUMED
    row, final = _returned_after_stop(host, monkeypatch, clock)
    assert row["_working_recovery"]["boundary"] == wc.RESUMED
    assert row["_execution_handoff_sec"] == pytest.approx(AT_STOP[True] + WORK_RESUMED)
    assert _spent() == pytest.approx(AT_STOP[True] + WORK_RESUMED)
    restored = _loop_once(host, monkeypatch, final, plan=fresh)
    assert (restored["kind"], restored["round_idx"], restored["boundary"]) == ("working", 31, wc.RESUMED)
    assert restored["ctx"].model_wait_context.executed_seconds() == pytest.approx(AT_STOP[True] + WORK_RESUMED)


@pytest.mark.parametrize("change", ["stop", "completed", "owner_restart"])
def test_current_authority_still_decides_a_returned_handover(host, monkeypatch, clock, change):  # noqa: F811
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.task_results import load_task_result, write_task_result
    from tests.test_project_hold_recovery import worker

    fresh = _fresh_plan(monkeypatch)
    pause = _working_then_paused(host, monkeypatch, clock, 3600, fresh)
    resumed = _granted_after_capacity(host, monkeypatch, clock, pause)
    _loop_once(host, monkeypatch, resumed, plan=fresh)
    assert _stop_boot(monkeypatch) == 1
    [row] = host.pending
    assert row["_working_recovery"]["boundary"] == wc.RESUMED and row["_execution_handoff_sec"] == pytest.approx(80)
    if change == "stop":
        request_cancel(host.root, "held", reason="owner stop", source="owner", requested_by="owner")
    elif change == "completed":
        write_task_result(host.root, "held", "completed", result="Saved answer")
    else:
        from tests._work_stop_helpers import live_restart
        live_restart(host.root, monkeypatch)
    clock[0] += HELD_2
    answer = queue.resume_budget_paused_task("held")
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    workers.assign_tasks()
    assert not answer["ok"] and not sent and "held" not in workers.RUNNING, answer
    if change == "completed":
        assert load_task_result(host.root, "held")["result"] == "Saved answer"
    assert not host.attempts
