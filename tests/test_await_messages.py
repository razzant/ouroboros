"""await_messages: hold the worker slot until an unread mailbox entry exists.

The tool delivers nothing and takes no lease of its own. Its window is bounded by
the request, the per-call timeout ceiling and, under a deadline, the emit window.
What keeps the supervisor's idle rail off a waiting task is the executor's typed
in-flight tool lease (kind ``tool``), NOT the ceiling, and the close of that lease
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

import ouroboros.config as config_mod
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
    monkeypatch.setattr(config_mod, "get_per_call_timeout_ceiling_sec", lambda: 1800)


def test_pending_mail_returns_immediately_without_delivering_or_acknowledging(tmp_path, monkeypatch):
    clock = _FakeClock()
    _install_clock(monkeypatch, clock)
    ctx = _ctx(tmp_path)
    assert write_task_message(tmp_path, "my next turn", "waiter", source_task_id="sib-1",
                              provenance=PROVENANCE_PEER_TASK, relation="sibling", msg_id="turn-1", attention_kind="question")

    out = json.loads(_await_messages(ctx, 600, mode="in_slot"))

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
    assert ctx.event_queue.empty()


def test_empty_mailbox_times_out_after_the_ceiling_bounded_window(tmp_path, monkeypatch):
    clock = _FakeClock()
    _install_clock(monkeypatch, clock)

    out = json.loads(_await_messages(_ctx(tmp_path), 5000, mode="in_slot"))

    assert out["reason"] == "timeout" and out["pending"] is False
    assert (out["requested_sec"], out["window_sec"], out["window_bound"]) == (5000, 1800, "ceiling")
    assert out["elapsed_sec"] == 1800.0 and out["slot"] == "held"
    assert max(clock.sleeps) <= _AWAIT_MESSAGES_POLL_SEC and sum(clock.sleeps) == 1800.0
    assert "cache_horizon" not in out  # no applied horizon recorded on this ctx


def test_a_message_arriving_mid_wait_ends_it_early(tmp_path, monkeypatch):
    def arrive(slices):
        if slices == 3:
            write_owner_message(tmp_path, "Owner: go with option B.", "waiter", msg_id="late")

    clock = _FakeClock(on_sleep=arrive)
    _install_clock(monkeypatch, clock)

    out = json.loads(_await_messages(_ctx(tmp_path), 900, mode="in_slot"))

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

    out = json.loads(_await_messages(_ctx(tmp_path), 900, mode="in_slot"))

    assert out["reason"] == "owner_mailbox_pending" and out["pending"] is True
    assert out["elapsed_sec"] == 2 * _AWAIT_MESSAGES_POLL_SEC


@pytest.mark.parametrize("requested, window, bound", [
    (0, 1, "minimum"), (-5, 1, "minimum"), (1, 1, "requested"),
    (1800, 1800, "requested"), (1801, 1800, "ceiling"),
])
def test_the_window_is_bounded_by_the_request_and_the_per_call_ceiling(tmp_path, monkeypatch, requested, window, bound):
    clock = _FakeClock()
    _install_clock(monkeypatch, clock)

    out = json.loads(_await_messages(_ctx(tmp_path), requested, mode="in_slot"))

    if requested == 0:
        assert out == {"reason": "snapshot", "slept": False, "mode": "in_slot"}
        assert clock.sleeps == []
        return
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
    outer = _deadline_clamped_timeout(SimpleNamespace(_ctx=ctx), "await_messages", 1860)

    out = json.loads(_await_messages(ctx, 1800, mode="in_slot"))

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

    out = json.loads(_await_messages(ctx, 600, mode="in_slot"))
    assert (out["window_sec"], out["window_bound"], out["reason"]) == (0, "deadline", "deadline")
    assert out["elapsed_sec"] == 0.0 and clock.sleeps == []

    # The one peek still reports a message that is already there.
    write_owner_message(tmp_path, "answer", "waiter", msg_id="a1")
    out = json.loads(_await_messages(ctx, 600, mode="in_slot"))
    assert out["reason"] == "owner_mailbox_pending" and out["pending"] is True and clock.sleeps == []


def test_the_applied_cache_horizon_is_reported_from_the_recorded_fact(tmp_path, monkeypatch):
    clock = _FakeClock()
    _install_clock(monkeypatch, clock)
    ctx = _ctx(tmp_path)
    ctx._accumulated_usage = {"_last_prompt_cache_ttl": "5m"}

    out = json.loads(_await_messages(ctx, 1800, mode="in_slot"))

    assert out["reason"] == "timeout"
    assert "5m" in out["cache_horizon"] and "300s" in out["cache_horizon"]


def test_a_non_integer_window_is_an_argument_error(tmp_path):
    out = _await_messages(_ctx(tmp_path), "soon", mode="in_slot")
    assert out.startswith("⚠️ TOOL_ARG_ERROR (await_messages)")


def _supervisor_over_one_stale_task(monkeypatch, tmp_path, *, now: float, ceiling: float, stale_sec: float):
    """The real enforcer, lease handler and executor frames over ONE RUNNING row whose
    last progress stamp is ``stale_sec`` old — deeper than the rail
    ``max(idle, ceiling + 120)`` — so a supervisor tick that finds no reprieve reaps
    into ``jobs``."""
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
    monkeypatch.setattr(squeue, "get_per_call_timeout_ceiling_sec", lambda: ceiling)
    monkeypatch.setattr(squeue, "get_task_abs_ceiling_sec", lambda: 10_000_000.0)
    monkeypatch.setattr(squeue, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(squeue, "persist_queue_snapshot", lambda reason="": True)
    jobs: list[dict] = []
    monkeypatch.setattr(squeue, "_reap_queue", types.SimpleNamespace(put=jobs.append))
    monkeypatch.setattr(workers, "WORKERS", {})
    assert stale_sec > max(60.0, ceiling + 120.0)
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
    round that spent 1500s in earlier tools and then waits the full ceiling sits
    3300s past its last stamp — far beyond the rail ``max(idle, ceiling + 120)``.
    The task survives while the executor's ``tool_call_started`` cognitive lease is
    open; and when ``tool_call_finished`` closes it, that close IS a progress stamp
    (a completed tool call is the task's own work), so the supervisor tick between
    the finished lease and the next model call spares the task and its next
    addressed turn can start. Without a new model round the ordinary idle rail then
    reaps, one full window after the stamp — the rail is intact, only re-based.
    """
    now = _time.time()
    ceiling = 1800.0
    sim = _supervisor_over_one_stale_task(monkeypatch, tmp_path, now=now, ceiling=ceiling,
                                          stale_sec=1500.0 + ceiling)

    started = _executor_frame(sim, {"type": "tool_call_started", "timeout_sec": int(ceiling) + 60,
                                    "args": {"timeout_sec": 1800}})
    assert (started["kind"], started["phase"], started["operation_id"]) == ("tool", "started", "call-1")
    assert sim.meta["active_operation_leases"]["call-1"]["kind"] == "tool"
    sim.tick(now)
    assert sim.jobs == [] and sim.task_id in sim.running and "finalization_requested_at" not in sim.meta
    assert sim.meta["last_progress_at"] == now - (1500.0 + ceiling)  # an OPEN lease is not progress

    finished = _executor_frame(sim, {"type": "tool_call_finished", "duration_sec": ceiling, "is_error": False})
    assert finished["phase"] == "finished" and "active_operation_leases" not in sim.meta
    assert now <= sim.meta["last_progress_at"] <= _time.time()  # the close stamped progress

    # The supervisor tick between the finished lease and the next LLM call: spared —
    # the next addressed model turn is permitted.
    sim.tick(now + 1.0)
    assert sim.jobs == [] and sim.task_id in sim.running and "finalization_requested_at" not in sim.meta

    # The ordinary idle rail is intact: one full window after the stamp with no model
    # round since, the task is reaped as idle.
    rail = max(60.0, ceiling + 120.0)
    sim.tick(sim.meta["last_progress_at"] + rail - 1.0)
    assert sim.jobs == [] and sim.task_id in sim.running
    sim.tick(sim.meta["last_progress_at"] + rail + 1.0)
    assert [job["terminal_reason"] for job in sim.jobs] == ["idle_timeout"]
    assert sim.jobs[0]["task_id"] == sim.task_id and sim.task_id not in sim.running


@pytest.mark.parametrize("hard_rail", ["absolute_ceiling", "deadline"])
def test_a_closed_tool_lease_stamps_progress_but_extends_no_hard_rail(tmp_path, monkeypatch, hard_rail):
    """The stamp spares only the idle rail: with progress fresh from the finished
    wait, a spent absolute ceiling or an explicit deadline still finalizes."""
    from ouroboros.deadline_utils import utc_now
    from supervisor import queue as squeue

    now = _time.time()
    ceiling = 1800.0
    sim = _supervisor_over_one_stale_task(monkeypatch, tmp_path, now=now, ceiling=ceiling,
                                          stale_sec=1500.0 + ceiling)
    _executor_frame(sim, {"type": "tool_call_started", "timeout_sec": int(ceiling) + 60,
                          "args": {"timeout_sec": 1800}})
    _executor_frame(sim, {"type": "tool_call_finished", "duration_sec": ceiling, "is_error": False})
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
    # The registered kill timeout sits above the largest window the tool can choose.
    assert entry.timeout_sec == int(config_mod.get_per_call_timeout_ceiling_sec()) + 60
    for surface in (CORE_TOOL_NAMES, LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
                    ACTING_SUBAGENT_TOOL_NAMES, UNTRUNCATED_TOOL_RESULTS, _DEADLINE_CLAMPED_TOOLS):
        assert "await_messages" in surface
    assert TOOL_POLICY["await_messages"] == POLICY_SKIP
    assert "await_messages" not in _PER_CALL_TIMEOUT_TOOLS  # the ceiling is the clamp, not a per-call override
    # No lease and no slot lending of its own: the body names no lease and emits no
    # supervisor event — the executor's per-call tool lease is the one that spares it.
    body = inspect.getsource(control_mod._await_messages).split('"""', 2)[2]  # past the docstring
    assert "lease" not in body and "emit" not in body and "event_queue" not in body
    description = entry.schema["description"]
    assert "Default warm" in description and "in_slot holds capacity" in description
    assert "never reaped" not in description and "informational mail" in description


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
