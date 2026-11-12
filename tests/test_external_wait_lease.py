"""The idle rail's leases: what spares a silent task, and what never does.

A typed cognitive operation in flight (``cognitive_operation``) or a live model call
(``main_llm_call_state``) is legitimate silence: each spares ONLY the idle rail, and the
explicit deadline, the absolute ceiling, budget fences and cancel all cut through it.
Since #1376 that rail records dated silence attention, never a grace window or stop. No
background thread holds anything (a lease is a number). The external-wait lease a
windowed ``delegate_wait`` granted went with that window: ``delegate_wait`` is a sleep on
runs and holds no slot to spare.
"""

import datetime as dt
import queue as stdqueue
import threading
import time
import types

import pytest


# -- the enforcer spares ONLY the idle rail -------------------------------------


def _enforcer(monkeypatch, tmp_path, running, *, abs_ceiling=10_000_000):
    from supervisor import queue as queue_mod

    monkeypatch.setattr(queue_mod, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue_mod, "RUNNING", running)
    monkeypatch.setattr(queue_mod, "PENDING", [])
    monkeypatch.setattr(queue_mod, "get_task_idle_timeout_sec", lambda: 900)
    monkeypatch.setattr(queue_mod, "get_task_abs_ceiling_sec", lambda: abs_ceiling)
    monkeypatch.setattr(queue_mod, "FINALIZATION_GRACE_SEC", 300)
    monkeypatch.setattr(queue_mod, "_request_finalization_grace",
                        lambda *_a, **_k: "msg-1")

    def _run(now):
        queue_mod._enforce_task_timeouts_locked(
            types.SimpleNamespace(WORKERS={}), now, 7, {})

    return _run


def test_an_operation_lease_never_spares_deadline_or_ceiling(monkeypatch, tmp_path):
    """A live operation lease spares the idle rail and nothing else: the absolute ceiling and an explicit deadline
    cut through it."""
    lease = {"active_operation_leases": {"llm-1": {"until_ts": 10_000_000.0, "kind": "llm", "task_attempt": 1}}}
    idle = {"task": {"id": "t0", "chat_id": 7}, "attempt": 1, "started_at": 1000.0, "last_progress_at": 1000.0,
            "worker_id": 0, **lease}
    _enforcer(monkeypatch, tmp_path, {"t0": idle})(2500.0)  # idle > 960s, leased
    assert "finalization_requested_at" not in idle and "silence_attention" not in idle

    meta = {"task": {"id": "t1", "chat_id": 7}, "attempt": 1, "started_at": 1000.0, "last_progress_at": 1000.0,
            "worker_id": 0, **{key: {**value} for key, value in lease.items()}}
    _enforcer(monkeypatch, tmp_path, {"t1": meta}, abs_ceiling=1000)(2500.0)
    assert meta.get("finalization_requested_at") == 2500.0
    assert meta.get("finalization_reason") == "absolute_ceiling"

    deadline = dt.datetime.fromtimestamp(2000.0, tz=dt.timezone.utc).isoformat()
    meta2 = {"task": {"id": "t2", "chat_id": 7, "deadline_at": deadline}, "attempt": 1, "started_at": 1000.0,
             "last_progress_at": 2400.0, "worker_id": 0, **{key: {**value} for key, value in lease.items()}}
    _enforcer(monkeypatch, tmp_path, {"t2": meta2})(2500.0)
    assert meta2.get("finalization_requested_at") == 2500.0
    assert meta2.get("finalization_reason") == "deadline"


def test_no_task_lifetime_never_ends_a_progressing_task_and_silence_still_gets_attention(monkeypatch, tmp_path):
    """#1196: an unlimited lifetime is no ceiling at all — a task a year old keeps running
    while it progresses — and once it stops progressing the idle rail still dates the
    silence (#1376: attention, never a stop)."""
    year = 365 * 86400.0
    meta = {"task": {"id": "t1", "chat_id": 7}, "started_at": 1000.0,
            "last_progress_at": 1000.0 + year, "worker_id": 0}
    _enforcer(monkeypatch, tmp_path, {"t1": meta}, abs_ceiling=None)(1010.0 + year)
    assert "finalization_requested_at" not in meta and "silence_attention" not in meta
    meta["last_progress_at"] = 1000.0
    _enforcer(monkeypatch, tmp_path, {"t1": meta}, abs_ceiling=None)(1010.0 + year)
    assert meta["silence_attention"]["state"] == "pending_attention"
    assert meta["silence_attention"]["last_progress_at"] == 1000.0
    assert "finalization_requested_at" not in meta


def test_an_operation_lease_stays_finite_without_a_task_lifetime(monkeypatch, tmp_path):
    """#1196: without a task lifetime the finite idle lease bounds the idle reprieve from its
    start fact, so a lost terminal can never spare the rail forever. The renamed constant is
    attention only: the operation it describes gets no timer from it (1A)."""
    from ouroboros.config import IDLE_LEASE_MAX_SEC
    from supervisor import events as events_mod

    monkeypatch.setattr("ouroboros.config.get_task_abs_ceiling_sec", lambda: None)
    monkeypatch.setattr(events_mod.time, "time", lambda: 1100.0)
    meta = {"task": {"id": "t5", "chat_id": 7}, "attempt": 1, "started_at": 1000.0,
            "last_progress_at": 1000.0, "worker_id": 0}
    events_mod._handle_cognitive_operation(
        {"type": "cognitive_operation", "task_id": "t5", "operation_id": "llm-1",
         "phase": "started", "kind": "llm", "task_attempt": 1},
        types.SimpleNamespace(RUNNING={"t5": meta}),
    )
    assert meta["active_operation_leases"]["llm-1"]["until_ts"] == 1100.0 + IDLE_LEASE_MAX_SEC == 1100.0 + 21600


def test_parallel_cognitive_operations_are_independent_and_attempt_bound(
    monkeypatch, tmp_path,
):
    from supervisor import events as events_mod

    monkeypatch.setattr("ouroboros.config.get_task_abs_ceiling_sec", lambda: 10_000)
    monkeypatch.setattr(events_mod.time, "time", lambda: 1100.0)

    meta = {
        "task": {"id": "t3", "chat_id": 7},
        "attempt": 2,
        "started_at": 1000.0,
        "last_progress_at": 1000.0,
        "worker_id": 0,
    }
    ctx = types.SimpleNamespace(RUNNING={"t3": meta})

    for operation_id in ("llm-1", "review-1"):
        events_mod._handle_cognitive_operation(
            {
                "type": "cognitive_operation", "task_id": "t3",
                "operation_id": operation_id, "phase": "started",
                "kind": "llm", "task_attempt": 2, "lease_until": 3000.0,
            },
            ctx,
        )
    assert set(meta["active_operation_leases"]) == {"llm-1", "review-1"}

    # A late event from attempt 1 cannot close either operation of attempt 2.
    events_mod._handle_cognitive_operation(
        {
            "type": "cognitive_operation", "task_id": "t3",
            "operation_id": "llm-1", "phase": "finished",
            "task_attempt": 1,
        },
        ctx,
    )
    assert set(meta["active_operation_leases"]) == {"llm-1", "review-1"}

    # One operation may settle while the other still spares the idle rail.
    events_mod._handle_cognitive_operation(
        {
            "type": "cognitive_operation", "task_id": "t3",
            "operation_id": "llm-1", "phase": "finished",
            "task_attempt": 2,
        },
        ctx,
    )
    assert set(meta["active_operation_leases"]) == {"review-1"}
    _enforcer(monkeypatch, tmp_path, {"t3": meta})(2500.0)
    assert "finalization_requested_at" not in meta and "silence_attention" not in meta

    events_mod._handle_cognitive_operation(
        {
            "type": "cognitive_operation", "task_id": "t3",
            "operation_id": "review-1", "phase": "failed",
            "task_attempt": 2,
        },
        ctx,
    )
    assert "active_operation_leases" not in meta
    _enforcer(monkeypatch, tmp_path, {"t3": meta})(2500.0)
    assert meta["silence_attention"]["state"] == "pending_attention"
    assert meta["silence_attention"]["task_attempt"] == 2
    assert "finalization_requested_at" not in meta


def test_cognitive_operation_event_reaches_idle_enforcer_via_dispatch(
    monkeypatch, tmp_path,
):
    from supervisor import events as events_mod

    monkeypatch.setattr(events_mod.time, "time", lambda: 1100.0)
    meta = {
        "task": {"id": "t4", "chat_id": 7},
        "attempt": 1,
        "started_at": 1000.0,
        "last_progress_at": 1000.0,
        "worker_id": 0,
    }
    ctx = types.SimpleNamespace(
        RUNNING={"t4": meta},
        DRIVE_ROOT=tmp_path,
        append_jsonl=lambda *_args, **_kwargs: None,
    )
    events_mod.dispatch_event(
        {
            "type": "cognitive_operation", "task_id": "t4",
            "operation_id": "llm-live", "phase": "started", "kind": "llm",
            "task_attempt": 1, "lease_until": 3000.0,
        },
        ctx,
    )
    _enforcer(monkeypatch, tmp_path, {"t4": meta})(2500.0)
    assert "finalization_requested_at" not in meta


def test_cognitive_operation_terminal_event_matches_execution_identity(
    monkeypatch,
):
    from supervisor import events as events_mod

    monkeypatch.setattr(events_mod.time, "time", lambda: 1100.0)
    meta = {
        "task": {"id": "t5", "chat_id": 7}, "attempt": 1,
        "started_at": 1000.0, "last_progress_at": 1000.0, "worker_id": 0,
    }
    ctx = types.SimpleNamespace(RUNNING={"t5": meta})
    for execution_id, round_id in (("exec-1", "round-1"), ("exec-2", "round-2")):
        events_mod._handle_cognitive_operation(
            {
                "type": "cognitive_operation", "task_id": "t5",
                "operation_id": "call-1", "phase": "started", "kind": "tool",
                "task_attempt": 1, "lease_until": 3000.0,
                "execution_id": execution_id, "round_id": round_id,
            },
            ctx,
        )
    events_mod._handle_cognitive_operation(
        {
            "type": "cognitive_operation", "task_id": "t5",
            "operation_id": "call-1", "phase": "finished", "kind": "tool",
            "task_attempt": 1, "execution_id": "exec-1", "round_id": "round-1",
        },
        ctx,
    )
    assert "call-1" in meta["active_operation_leases"]
    # A legacy/partial terminal event cannot close a row whose correlation
    # identity is known; it remains leased until its real settlement or ceiling.
    events_mod._handle_cognitive_operation(
        {
            "type": "cognitive_operation", "task_id": "t5",
            "operation_id": "call-1", "phase": "finished", "kind": "tool",
            "task_attempt": 1,
        },
        ctx,
    )
    assert "call-1" in meta["active_operation_leases"]


def test_timed_out_tool_closes_its_cognitive_lease_when_worker_settles(tmp_path, monkeypatch):
    from ouroboros import loop_tool_execution as loop_tools

    events = stdqueue.Queue()
    ctx = types.SimpleNamespace(
        event_queue=events, task_id="tool-task", task_attempt=1,
        task_metadata={}, _current_llm_call_meta={"execution_id": "exec-1", "round_id": "round-1"},
    )
    tools = types.SimpleNamespace(_ctx=ctx, CODE_TOOLS=set())

    # The tool OUTLIVES the timeout deterministically: it blocks on an event the test
    # releases only after the timeout has answered. A sleep(0.05) against a 0.01 s timeout
    # is a 40 ms race that a loaded macos-latest runner lost (rc.14 dispatch: the wait was
    # scheduled late, the tool had already finished, is_error came back False).
    release = threading.Event()

    def slow_tool(*_args):
        release.wait(10)
        return {
            "tool_call_id": "tool-1", "fn_name": "read_file", "result": "ok",
            "is_error": False, "args_for_log": {}, "is_code_tool": False,
            "result_meta": {},
        }

    monkeypatch.setattr(loop_tools, "_execute_single_tool", slow_tool)
    result = loop_tools._execute_with_timeout(
        tools,
        {"id": "tool-1", "function": {"name": "read_file", "arguments": "{}"}},
        tmp_path / "logs",
        timeout_sec=0.01,
        task_id="tool-task",
    )
    release.set()   # let the worker settle now that the timeout has answered
    assert result["is_error"] is True
    # The worker settles on its own schedule (a loaded macOS runner took more
    # than the old fixed 0.1 s): drain until the terminal row lands, bounded.
    rows = []
    terminal = []
    deadline = time.monotonic() + 10.0
    while not terminal and time.monotonic() < deadline:
        while not events.empty():
            rows.append(events.get_nowait())
        terminal = [
            row for row in rows
            if row.get("type") == "cognitive_operation"
            and row.get("task_id") == "tool-task"
            and row.get("phase") == "finished"
        ]
        if not terminal:
            time.sleep(0.02)
    assert terminal, rows
    started = [row for row in rows if row.get("type") == "cognitive_operation"
               and row.get("phase") == "started" and row.get("task_id") == "tool-task"]
    # The lease follows the host's frozen invocation, not a provider-reused call ID.
    assert len(started) == len(terminal) == 1
    assert terminal[0]["operation_id"] == started[0]["operation_id"]


def _llm_call_event(phase, **overrides):
    return {
        "type": "main_llm_call_state",
        "task_id": "t3",
        "task_attempt": 2,
        "llm_call_id": "call-1",
        "execution_id": "exec-1",
        "round_id": "exec-1:round:4",
        "call_attempt": 1,
        "phase": phase,
        **overrides,
    }


def test_main_llm_call_state_is_attempt_and_call_identity_bound():
    from supervisor import events as events_mod

    meta = {
        "task": {"id": "t3", "_attempt": 2},
        "attempt": 2,
        "started_at": 1000.0,
    }
    ctx = types.SimpleNamespace(RUNNING={"t3": meta})

    events_mod.dispatch_event(_llm_call_event("started", task_attempt=None), ctx)
    events_mod.dispatch_event(_llm_call_event("started", task_attempt=1), ctx)
    assert "active_llm_call" not in meta

    events_mod.dispatch_event(_llm_call_event("started"), ctx)
    assert meta["active_llm_call"]["llm_call_id"] == "call-1"

    # The next retry supersedes the settled call identity. A late terminal from
    # the prior call, or a partial terminal, cannot clear the newer in-flight row.
    events_mod.dispatch_event(_llm_call_event(
        "started", llm_call_id="call-2", call_attempt=2,
    ), ctx)
    events_mod.dispatch_event(_llm_call_event("failed"), ctx)
    events_mod.dispatch_event(_llm_call_event(
        "finished", llm_call_id="call-2", call_attempt=2, round_id="",
    ), ctx)
    assert meta["active_llm_call"]["llm_call_id"] == "call-2"

    events_mod.dispatch_event(_llm_call_event(
        "finished", llm_call_id="call-2", call_attempt=2,
    ), ctx)
    assert "active_llm_call" not in meta


def test_active_llm_call_spares_idle_without_elapsed_expiry(monkeypatch, tmp_path):
    meta = {
        "task": {"id": "t4", "chat_id": 7, "_attempt": 2},
        "attempt": 2,
        "started_at": 1000.0,
        "last_progress_at": 1000.0,
        "worker_id": 0,
        "active_llm_call": {
            "task_attempt": 2,
            "llm_call_id": "call-live",
            "execution_id": "exec-live",
            "round_id": "exec-live:round:1",
            "call_attempt": 1,
            "started_at": 1100.0,
        },
    }
    # The call has been silent longer than the current 2700s transport default.
    # Elapsed time is not semantic stall evidence; only hard task axes cut through.
    _enforcer(monkeypatch, tmp_path, {"t4": meta}, abs_ceiling=10_000)(5000.0)
    assert "finalization_requested_at" not in meta and "silence_attention" not in meta

    meta.pop("active_llm_call")
    _enforcer(monkeypatch, tmp_path, {"t4": meta}, abs_ceiling=10_000)(5000.0)
    assert meta["silence_attention"]["state"] == "pending_attention"
    assert meta["silence_attention"]["observed_at"] == 5000.0
    assert "finalization_requested_at" not in meta


@pytest.mark.parametrize("kind, phase, stamped", [
    ("tool", "finished", True), ("review", "finished", True), ("llm", "finished", False),
])
def test_only_a_closed_tool_or_review_lease_is_fresh_progress(kind, phase, stamped, monkeypatch):
    """A completed tool call and a review seat that physically ended, with an answer or an error, are the
    task's own work: the confirmed lease closes INTO a progress stamp, so no tick reads the task as lease-less
    and stale (a deep self-review reports nothing else before it ends; tests/test_deep_review_lease.py drives
    that branch). A tool's ``failed`` stamps nothing, a model call is stamped by its own usage row and a wait
    mark is no work; a terminal with no confirmed lease behind it never stamps."""
    from supervisor import events as events_mod

    monkeypatch.setattr(events_mod.time, "time", lambda: 5000.0)
    meta = {"task": {"id": "t9", "chat_id": 7}, "attempt": 1, "started_at": 1000.0,
            "last_progress_at": 1000.0, "worker_id": 0}
    ctx = types.SimpleNamespace(RUNNING={"t9": meta})

    def emit(step):
        events_mod._handle_cognitive_operation({"type": "cognitive_operation", "task_id": "t9", "operation_id": "op-1",
                                                "phase": step, "kind": kind, "task_attempt": 1}, ctx)

    emit("started")
    emit(phase)
    assert "active_operation_leases" not in meta
    closed_at = 5000.0 if stamped else 1000.0
    assert meta["last_progress_at"] == closed_at
    monkeypatch.setattr(events_mod.time, "time", lambda: 6000.0)
    emit(phase)  # replayed: the lease it would close is gone
    assert meta["last_progress_at"] == closed_at


def test_stale_attempt_llm_call_does_not_spare_idle(monkeypatch, tmp_path):
    meta = {
        "task": {"id": "t5", "chat_id": 7, "_attempt": 2},
        "attempt": 2,
        "started_at": 1000.0,
        "last_progress_at": 1000.0,
        "worker_id": 0,
        "active_llm_call": {"task_attempt": 1, "llm_call_id": "stale"},
    }
    _enforcer(monkeypatch, tmp_path, {"t5": meta})(2500.0)
    assert meta["silence_attention"]["state"] == "pending_attention"
    assert meta["silence_attention"]["task_attempt"] == 2
    assert "finalization_requested_at" not in meta


@pytest.mark.parametrize("hard_axis", ["deadline", "absolute_ceiling"])
def test_active_llm_call_never_spares_hard_task_axes(
    hard_axis, monkeypatch, tmp_path,
):
    task = {"id": "t6", "chat_id": 7, "_attempt": 1}
    abs_ceiling = 10_000
    if hard_axis == "deadline":
        task["deadline_at"] = dt.datetime.fromtimestamp(
            2000.0, tz=dt.timezone.utc,
        ).isoformat()
    else:
        abs_ceiling = 1000
    meta = {
        "task": task,
        "attempt": 1,
        "started_at": 1000.0,
        "last_progress_at": 1000.0,
        "worker_id": 0,
        "active_llm_call": {
            "task_attempt": 1,
            "llm_call_id": "call-hard-axis",
            "execution_id": "exec-hard-axis",
            "round_id": "exec-hard-axis:round:1",
            "call_attempt": 1,
        },
    }
    _enforcer(monkeypatch, tmp_path, {"t6": meta}, abs_ceiling=abs_ceiling)(2500.0)
    assert meta["finalization_requested_at"] == 2500.0
    assert meta["finalization_reason"] == hard_axis
