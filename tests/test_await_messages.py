"""await_messages: hold the worker slot until an unread mailbox entry exists.

The tool delivers nothing and takes no lease of its own. Its window is the request
and, under a deadline or execution budget, the emit window: no ceiling of its own (owner 30 Oct, B).
What keeps the supervisor's idle rail off a waiting task is the executor's typed
in-flight tool lease (kind ``tool``), NOT the window, and the close of that lease
is the progress stamp the next model round starts from (a completed tool call is
the task's own work, like a completed model round or a narration line). The
lifecycle tests drive the real enforcer through that lease instead of comparing
constants, and check that the hard rails (deadline, absolute ceiling) stay untouched.
"""

from __future__ import annotations

import datetime as _dt
import inspect
import json
import queue
import time as _time
import types
from types import SimpleNamespace

import pytest

from ouroboros.owner_mailbox import (
    KIND_FINALIZE_NOW,
    PROVENANCE_PEER_TASK,
    acknowledged_task_message_ids,
    write_owner_message,
    write_task_message,
)
from ouroboros.tools import control_task_results as control_mod
from ouroboros.tools.control_task_results import _AWAIT_MESSAGES_POLL_SEC, _await_messages


class _FakeClock:
    """Deterministic monotonic clock: sleeping advances time, nothing else does."""

    def __init__(self, on_sleep=None):
        self.now = 1000.0
        self.sleeps: list[float] = []
        self._on_sleep = on_sleep

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(float(seconds))
        self.now += float(seconds)
        if self._on_sleep is not None:
            self._on_sleep(len(self.sleeps))


def _ctx(tmp_path):
    return SimpleNamespace(drive_root=tmp_path, task_id="waiter", task_attempt=1,
                           _loop_mailbox_seen_ids=set(), event_queue=queue.Queue())


def _install_clock(monkeypatch, clock):
    monkeypatch.setattr(control_mod, "time", clock)


def test_pending_mail_returns_immediately_without_delivering_or_acknowledging(tmp_path, monkeypatch):
    clock = _FakeClock()
    _install_clock(monkeypatch, clock)
    ctx = _ctx(tmp_path)
    assert write_task_message(tmp_path, "my next turn", "waiter", source_task_id="sib-1",
                              provenance=PROVENANCE_PEER_TASK, relation="sibling", msg_id="turn-1")

    out = json.loads(_await_messages(ctx, 600))

    assert out == {
        "reason": "owner_mailbox_pending", "pending": True, "elapsed_sec": 0.0,
        "requested_sec": 600, "window_sec": 600, "window_bound": "requested",
        "slot": "held", "note": out["note"],
    }
    assert "next round top" in out["note"] and "holds the worker slot" in out["note"]
    assert clock.sleeps == []
    # Nothing delivered, nothing acknowledged, the loop's seen-set untouched.
    assert acknowledged_task_message_ids(tmp_path, "waiter", attempt_key=1) == set()
    assert ctx._loop_mailbox_seen_ids == set()
    frames = [ctx.event_queue.get_nowait() for _ in range(ctx.event_queue.qsize())]
    # Only the typed in-slot wait lease and the task's re-announcement (DESIGN §9 typing): no delivery.
    assert [(frame["type"], frame.get("kind"), frame.get("phase")) for frame in frames] == [
        ("cognitive_operation", "in_slot_wait", "started"), ("cognitive_operation", "in_slot_wait", "finished"),
        ("typing_start", None, "thinking")]


def test_an_empty_mailbox_waits_the_whole_requested_window(tmp_path, monkeypatch):
    clock = _FakeClock()
    _install_clock(monkeypatch, clock)

    out = json.loads(_await_messages(_ctx(tmp_path), 5000))

    assert out["reason"] == "timeout" and out["pending"] is False
    assert (out["requested_sec"], out["window_sec"], out["window_bound"]) == (5000, 5000, "requested")
    assert out["elapsed_sec"] == 5000.0 and out["slot"] == "held"
    assert max(clock.sleeps) <= _AWAIT_MESSAGES_POLL_SEC and sum(clock.sleeps) == 5000.0
    assert "cache_horizon" not in out  # no applied horizon recorded on this ctx


def test_a_message_arriving_mid_wait_ends_it_early(tmp_path, monkeypatch):
    def arrive(slices):
        if slices == 3:
            write_owner_message(tmp_path, "Owner: go with option B.", "waiter", msg_id="late")

    clock = _FakeClock(on_sleep=arrive)
    _install_clock(monkeypatch, clock)

    out = json.loads(_await_messages(_ctx(tmp_path), 900))

    assert out["reason"] == "owner_mailbox_pending" and out["pending"] is True
    assert out["elapsed_sec"] == 3 * _AWAIT_MESSAGES_POLL_SEC
    assert acknowledged_task_message_ids(tmp_path, "waiter", attempt_key=1) == set()


def test_an_owner_stop_control_ends_the_wait_like_any_message(tmp_path, monkeypatch):
    """Stop is a mailbox control (finalize_now); the peek wakes for it, the round
    top drains and acts on it — the wait never swallows an owner stop."""
    def stop(slices):
        if slices == 2:
            write_owner_message(tmp_path, "owner_stop", "waiter", msg_id="stop-1", kind=KIND_FINALIZE_NOW)

    clock = _FakeClock(on_sleep=stop)
    _install_clock(monkeypatch, clock)

    out = json.loads(_await_messages(_ctx(tmp_path), 900))

    assert out["reason"] == "owner_mailbox_pending" and out["pending"] is True
    assert out["elapsed_sec"] == 2 * _AWAIT_MESSAGES_POLL_SEC


@pytest.mark.parametrize("state", ["stop-now", "unreadable"])
def test_an_in_process_turn_reads_its_open_intent_and_keeps_an_unreadable_one_unknown(tmp_path, monkeypatch, state):
    """R4: an inline Presence turn's Stop now writes no mail, so the poll reads the turn's open intent: an immediate
    one ends the wait at once; an unreadable projection stops nothing and is recorded as unknown, never as no Stop."""
    clock = _FakeClock()
    _install_clock(monkeypatch, clock)
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "cancel_intents.json").write_text(json.dumps(
        {"intents": {"waiter": {"stop_policy": "immediate", "request_id": "r1"}}}) if state == "stop-now" else "{not json",
        encoding="utf-8")
    ctx = _ctx(tmp_path)
    ctx.model_wait_context = SimpleNamespace(worker_slot_held=False, canonical_root=tmp_path, task_id="waiter")

    out = json.loads(_await_messages(ctx, 6))

    if state == "stop-now":
        assert (out["reason"], out["pending"], out["elapsed_sec"]) == ("owner_stop", False, 0)
        assert "owner_stop_state" not in out
    else:
        assert (out["reason"], out["owner_stop_state"], out["elapsed_sec"]) == ("timeout", "unknown", 6)


@pytest.mark.parametrize("requested, window, bound", [
    (0, 1, "minimum"), (-5, 1, "minimum"), (1, 1, "requested"),
    (1800, 1800, "requested"), (1801, 1801, "requested"), (7200, 7200, "requested"),
])
def test_the_window_is_the_request_with_a_one_second_floor(tmp_path, monkeypatch, requested, window, bound):
    clock = _FakeClock()
    _install_clock(monkeypatch, clock)

    out = json.loads(_await_messages(_ctx(tmp_path), requested))

    assert (out["requested_sec"], out["window_sec"], out["window_bound"]) == (requested, window, bound)
    assert out["elapsed_sec"] == float(window) and out["reason"] == "timeout"


def test_a_deadline_bounds_the_window_inside_the_executors_emit_window(tmp_path, monkeypatch):
    """Under a task deadline the tool stops one second inside the emit window the
    executor's deadline clamp gives its kill timer, so it returns typed
    (``deadline``) instead of being timed out mid-sleep."""
    import ouroboros.task_pacing as pacing
    from ouroboros.deadline_utils import utc_now
    from ouroboros.loop_tool_execution import _deadline_clamped_timeout

    clock = _FakeClock()
    _install_clock(monkeypatch, clock)
    monkeypatch.setattr(pacing, "effective_finalization_reserve_sec", lambda ctx: 60.0)
    ctx = _ctx(tmp_path)
    ctx.task_metadata = {"deadline_at": (utc_now() + _dt.timedelta(seconds=400)).isoformat()}
    outer = _deadline_clamped_timeout(SimpleNamespace(_ctx=ctx), "await_messages", None)

    out = json.loads(_await_messages(ctx, 7200))

    assert out["window_bound"] == "deadline" and out["reason"] == "deadline" and out["pending"] is False
    assert outer - 2 <= out["window_sec"] <= outer - 1 < outer <= 340  # emit window ≈ 400 − 60
    assert out["elapsed_sec"] == float(out["window_sec"])


def test_a_spent_emit_window_peeks_once_and_returns_without_sleeping(tmp_path, monkeypatch):
    import ouroboros.task_pacing as pacing
    from ouroboros.deadline_utils import utc_now

    clock = _FakeClock()
    _install_clock(monkeypatch, clock)
    monkeypatch.setattr(pacing, "effective_finalization_reserve_sec", lambda ctx: 60.0)
    ctx = _ctx(tmp_path)
    ctx.task_metadata = {"deadline_at": (utc_now() + _dt.timedelta(seconds=30)).isoformat()}

    out = json.loads(_await_messages(ctx, 600))
    assert (out["window_sec"], out["window_bound"], out["reason"]) == (0, "deadline", "deadline")
    assert out["elapsed_sec"] == 0.0 and clock.sleeps == []

    # The one peek still reports a message that is already there.
    write_owner_message(tmp_path, "answer", "waiter", msg_id="a1")
    out = json.loads(_await_messages(ctx, 600))
    assert out["reason"] == "owner_mailbox_pending" and out["pending"] is True and clock.sleeps == []


def test_under_an_execution_budget_the_real_executor_gets_the_typed_result_not_a_timeout(tmp_path, monkeypatch):
    """Through the real dispatch, in real time: the executor's kill timer for this windowless entry is the
    task's explicit execution budget less the reserve (``deadline_clamped_timeout_sec``, #939), and the real
    handler ends its wait one second inside it with its typed ``deadline`` result, never a TOOL_TIMEOUT
    while the wait keeps sleeping in an abandoned thread."""
    import ouroboros.loop_tool_execution as lte
    import ouroboros.task_pacing as pacing
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    monkeypatch.setenv("OUROBOROS_TASK_ABS_CEILING_SEC", "unlimited")
    monkeypatch.setattr(pacing, "effective_finalization_reserve_sec", lambda ctx: 1.0)
    monkeypatch.setattr(lte, "load_settings", lambda: {})
    (tmp_path / "work").mkdir()
    (tmp_path / "logs").mkdir()
    ctx = ToolContext(repo_dir=tmp_path / "work", drive_root=tmp_path, task_id="waiter")
    ctx.task_attempt, ctx.task_metadata, ctx.task_started_at = 1, {"execution_timeout_sec": 5}, _time.time()
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=tmp_path)
    registry.set_context(ctx)
    args = {"timeout_sec": 6}
    outer = lte._get_tool_timeout(registry, "await_messages", args)
    assert outer is not None and 2 <= outer <= 4, outer
    call = {"id": "call-await", "function": {"name": "await_messages", "arguments": json.dumps(args)}}

    start = _time.monotonic()
    result = lte._execute_with_timeout(registry, call, tmp_path / "logs", outer, task_id="waiter")
    took = _time.monotonic() - start

    assert result["tool_result"].code != "TOOL_TIMEOUT", result["result"]
    out = json.loads(result["result"])
    assert (out["reason"], out["window_bound"], out["pending"]) == ("deadline", "deadline", False)
    assert 1 <= out["window_sec"] <= outer - 1 and took < outer


def test_the_applied_cache_horizon_is_reported_from_the_recorded_fact(tmp_path, monkeypatch):
    clock = _FakeClock()
    _install_clock(monkeypatch, clock)
    ctx = _ctx(tmp_path)
    ctx._accumulated_usage = {"_last_prompt_cache_ttl": "5m"}

    out = json.loads(_await_messages(ctx, 1800))

    assert out["reason"] == "timeout"
    assert "5m" in out["cache_horizon"] and "300s" in out["cache_horizon"]


def test_a_non_integer_window_is_an_argument_error(tmp_path):
    out = _await_messages(_ctx(tmp_path), "soon")
    assert out.startswith("⚠️ TOOL_ARG_ERROR (await_messages)")


def _supervisor_over_one_stale_task(monkeypatch, tmp_path, *, now: float, stale_sec: float):
    """The real enforcer, lease handler and executor frames over ONE RUNNING row whose
    last progress stamp is ``stale_sec`` old — deeper than the configured silence
    window (60 s here) — so a supervisor tick that finds no reprieve dates silence
    attention (#1376); only a hard rail reaps into ``jobs``."""
    from supervisor import queue as squeue, workers

    task_id = "waiter"
    meta = {"task": {"id": task_id, "type": "task", "chat_id": 0}, "started_at": now - 5000.0,
            "last_heartbeat_at": now, "last_progress_at": now - stale_sec,
            "attempt": 1, "worker_id": -1}
    running = {task_id: meta}
    squeue.init_queue_refs([], running, {"value": 0})
    monkeypatch.setattr(squeue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(squeue, "FINALIZATION_GRACE_SEC", 0.0)
    monkeypatch.setattr(squeue, "get_task_idle_timeout_sec", lambda: 60.0)
    monkeypatch.setattr(squeue, "get_task_abs_ceiling_sec", lambda: 10_000_000.0)
    monkeypatch.setattr(squeue, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(squeue, "persist_queue_snapshot", lambda reason="": True)
    jobs: list[dict] = []
    monkeypatch.setattr(squeue, "_reap_queue", types.SimpleNamespace(put=jobs.append))
    monkeypatch.setattr(workers, "WORKERS", {})
    assert stale_sec > 60.0
    events: queue.Queue = queue.Queue()
    tools = SimpleNamespace(_ctx=SimpleNamespace(event_queue=events, task_id=task_id, task_attempt=1,
                                                 task_metadata={}))
    return SimpleNamespace(
        task_id=task_id, meta=meta, running=running, jobs=jobs, tools=tools, events=events,
        supervisor_ctx=SimpleNamespace(RUNNING=running),
        tick=lambda at: squeue._enforce_task_timeouts_locked(workers, at, 0, {}),
    )


def _executor_frame(sim, frame: dict) -> dict:
    """Emit one of the executor's own live-log frames for an await_messages call and
    hand the typed lease fact it carries to the supervisor's handler."""
    from supervisor.cognitive_operations import _handle_cognitive_operation
    from ouroboros.loop_tool_execution import _emit_live_log

    _emit_live_log(sim.tools, {"task_id": sim.task_id, "tool": "await_messages", "tool_call_id": "call-1", **frame})
    drained = []
    while True:
        try:
            drained.append(sim.events.get_nowait())
        except queue.Empty:
            break
    [fact] = [e for e in drained if e.get("type") == "cognitive_operation"]
    _handle_cognitive_operation(fact, sim.supervisor_ctx)
    return fact


def test_the_idle_rail_spares_a_waiting_task_through_the_executors_tool_lease_and_its_close_is_progress(tmp_path, monkeypatch):
    """The real enforcer, driven through the real executor frames.

    ``last_progress_at`` is stamped by completed model rounds and narration, so a
    round that spent 1500s in earlier tools and then waits the 1800s it asked for sits
    3300s past its last stamp — far beyond the configured silence window.
    The task survives while the executor's ``tool_call_started`` cognitive lease is
    open; and when ``tool_call_finished`` closes it, that close IS a progress stamp
    (a completed tool call is the task's own work), so the supervisor tick between
    the finished lease and the next model call spares the task and its next
    addressed turn can start. Without a new model round the ordinary idle rail then
    dates silence attention one full window after the stamp, never a reap (#1376) —
    the rail is intact, only re-based.
    """
    now = _time.time()
    waited = 1800.0
    sim = _supervisor_over_one_stale_task(monkeypatch, tmp_path, now=now, stale_sec=1500.0 + waited)

    started = _executor_frame(sim, {"type": "tool_call_started", "timeout_sec": None,
                                    "args": {"timeout_sec": 1800}})
    assert (started["kind"], started["phase"], started["operation_id"]) == ("tool", "started", "call-1")
    assert sim.meta["active_operation_leases"]["call-1"]["kind"] == "tool"
    sim.tick(now)
    assert sim.jobs == [] and sim.task_id in sim.running and "finalization_requested_at" not in sim.meta
    assert sim.meta["last_progress_at"] == now - (1500.0 + waited)  # an OPEN lease is not progress

    finished = _executor_frame(sim, {"type": "tool_call_finished", "duration_sec": waited, "is_error": False})
    assert finished["phase"] == "finished" and "active_operation_leases" not in sim.meta
    assert now <= sim.meta["last_progress_at"] <= _time.time()  # the close stamped progress

    # The supervisor tick between the finished lease and the next LLM call: spared —
    # the next addressed model turn is permitted.
    sim.tick(now + 1.0)
    assert sim.jobs == [] and sim.task_id in sim.running and "finalization_requested_at" not in sim.meta

    # The ordinary idle rail is intact: one full window after the stamp with no model
    # round since, the task's silence is dated for attention and it keeps running.
    rail = 60.0  # the configured window itself (owner 30 Oct, A), no command ceiling beneath it
    stamp = sim.meta["last_progress_at"]
    sim.tick(stamp + rail - 1.0)
    assert sim.jobs == [] and sim.task_id in sim.running and "silence_attention" not in sim.meta
    sim.tick(stamp + rail + 1.0)
    assert sim.jobs == [] and sim.task_id in sim.running and "finalization_requested_at" not in sim.meta
    assert sim.meta["silence_attention"]["state"] == "pending_attention"
    assert sim.meta["silence_attention"]["last_progress_at"] == stamp


@pytest.mark.parametrize("hard_rail", ["absolute_ceiling", "deadline"])
def test_a_closed_tool_lease_stamps_progress_but_extends_no_hard_rail(tmp_path, monkeypatch, hard_rail):
    """The stamp spares only the idle rail: with progress fresh from the finished
    wait, a spent absolute ceiling or an explicit deadline still finalizes."""
    from ouroboros.deadline_utils import utc_now
    from supervisor import queue as squeue

    now = _time.time()
    waited = 1800.0
    sim = _supervisor_over_one_stale_task(monkeypatch, tmp_path, now=now, stale_sec=1500.0 + waited)
    _executor_frame(sim, {"type": "tool_call_started", "timeout_sec": None, "args": {"timeout_sec": 1800}})
    _executor_frame(sim, {"type": "tool_call_finished", "duration_sec": waited, "is_error": False})
    assert sim.meta["last_progress_at"] >= now
    if hard_rail == "absolute_ceiling":
        monkeypatch.setattr(squeue, "get_task_abs_ceiling_sec", lambda: 4000.0)  # started 5000s ago
    else:
        sim.meta["task"]["deadline_at"] = (utc_now() - _dt.timedelta(seconds=1)).isoformat()

    sim.tick(now + 1.0)

    assert [job["terminal_reason"] for job in sim.jobs] == [hard_rail]
    assert sim.task_id not in sim.running


def test_registered_on_every_contract_surface_and_takes_no_lease_of_its_own():
    from ouroboros.loop_tool_execution import _DEADLINE_CLAMPED_TOOLS, _PER_CALL_TIMEOUT_TOOLS
    from ouroboros.safety import POLICY_SKIP, TOOL_POLICY
    from ouroboros.tool_capabilities import (
        ACTING_SUBAGENT_TOOL_NAMES,
        CORE_TOOL_NAMES,
        LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
        UNTRUNCATED_TOOL_RESULTS,
    )
    from ouroboros.tools import control

    entry = next(e for e in control.get_tools() if e.name == "await_messages")
    assert entry.handler is control_mod._await_messages
    # Owner Batch4 6B: the same tool also carries the model's own warm/cold sleep;
    # ``timeout_sec`` still drives the default in-slot wait (nothing is required).
    assert "required" not in entry.schema["parameters"]
    assert set(entry.schema["parameters"]["properties"]) == {
        "timeout_sec", "mode", "senders", "tasks", "runs", "services", "wake_at", "wake_after_sec"}
    assert entry.schema["parameters"]["properties"]["mode"]["enum"] == ["in_slot", "warm", "cold"]
    # No outer window of its own: the request, the executor's deadline clamp and Stop (its mail or the turn's intent) bound it.
    assert entry.timeout_sec is None
    for surface in (CORE_TOOL_NAMES, LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
                    ACTING_SUBAGENT_TOOL_NAMES, UNTRUNCATED_TOOL_RESULTS, _DEADLINE_CLAMPED_TOOLS):
        assert "await_messages" in surface
    assert TOOL_POLICY["await_messages"] == POLICY_SKIP
    assert "await_messages" not in _PER_CALL_TIMEOUT_TOOLS  # its own window bounds it, not a per-call override
    # No lease and no slot lending of its own: the body names no lease and emits no
    # supervisor event — the executor's per-call tool lease is the one that spares it.
    body = inspect.getsource(control_mod._await_messages).split('"""', 2)[2]  # past the docstring
    assert "lease" not in body and "emit" not in body and "event_queue" not in body
    description = entry.schema["description"]
    assert "holds your worker slot" in description and "delivers nothing" in description
    assert "never reaped" not in description and "in-flight tool lease" in description


def test_the_schema_states_the_attention_lease_bound_the_enforcer_applies():
    """R4: without a deadline or execution budget the tool lease spares silence attention for IDLE_LEASE_MAX_SEC
    from its start, while a longer wait keeps running; the schema says so instead of "while it runs"."""
    from ouroboros.config import IDLE_LEASE_MAX_SEC
    from supervisor.cognitive_operations import _handle_cognitive_operation

    meta = {"task": {"id": "w"}, "attempt": 1}
    _handle_cognitive_operation({"task_id": "w", "task_attempt": 1, "operation_id": "call-1", "phase": "started",
                                 "kind": "tool"}, SimpleNamespace(RUNNING={"w": meta}))
    lease = meta["active_operation_leases"]["call-1"]["until_ts"] - _time.time()
    assert IDLE_LEASE_MAX_SEC - 60 < lease <= IDLE_LEASE_MAX_SEC
    description = control_mod.await_messages_entry().schema["description"]
    assert f"at most {IDLE_LEASE_MAX_SEC // 3600} hours" in description and "keeps running" in description


def test_unknown_or_replayed_completion_cannot_refresh_progress():
    from supervisor.cognitive_operations import _handle_cognitive_operation
    meta = {"task": {"id": "w"}, "attempt": 1, "last_progress_at": 10.0}
    ctx = SimpleNamespace(RUNNING={"w": meta})
    event = {"task_id": "w", "task_attempt": 1, "operation_id": "unknown",
             "phase": "finished", "kind": "tool"}
    _handle_cognitive_operation(event, ctx)
    assert meta["last_progress_at"] == 10.0
    meta["active_operation_leases"] = {"unknown": {"kind": "tool", "task_attempt": 1}}
    _handle_cognitive_operation(event, ctx)
    fresh = meta["last_progress_at"]
    assert fresh > 10.0
    _handle_cognitive_operation(event, ctx)
    assert meta["last_progress_at"] == fresh
