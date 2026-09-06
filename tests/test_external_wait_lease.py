"""Poltergeist phase B (B3): the typed external-wait lease.

A nanny holding a bounded ``delegate_wait`` window over a live delegated run is
legitimately silent: the run streams into the daemon, not into the supervisor's
``last_progress_at``. The lease is typed metadata with a hard expiry — granted
per window, bounded by min(task deadline, the run's own maxSeconds horizon, an
absolute ceiling), released when the wait returns — and it spares ONLY the idle
rail: the explicit deadline, the absolute ceiling, budget fences and cancel all
cut through it. No background thread holds anything (a thread-pool timeout does
not kill threads; a lease is a number).
"""

import datetime as dt
import json
import queue as stdqueue
import threading
import time
import types

import pytest

from ouroboros.delegate_progress import (
    DELEGATE_WAIT_LEASE_GRACE_SEC,
    EXTERNAL_WAIT_LEASE_CEILING_SEC,
)


@pytest.fixture(autouse=True)
def _owned_gateway_uses_each_test_transport(monkeypatch):
    from ouroboros import claudexor_daemon
    from ouroboros.gateways import claudexor as gateway_module

    monkeypatch.setattr(
        claudexor_daemon,
        "ensure_owned_gateway",
        lambda: gateway_module.ClaudexorGateway(),
    )


# -- the bound ------------------------------------------------------------------


def _ctx(meta=None):
    return types.SimpleNamespace(task_metadata=meta or {}, task_id="t", event_queue=None)


def test_lease_until_is_the_window_plus_grace_by_default():
    from ouroboros.tools.delegate import _external_wait_lease_until

    now = time.time()
    until = _external_wait_lease_until(_ctx(), 600, None, 0)
    assert until == pytest.approx(now + 600 + DELEGATE_WAIT_LEASE_GRACE_SEC, abs=2.0)


def test_lease_until_is_clamped_by_the_task_deadline():
    from ouroboros.tools.delegate import _external_wait_lease_until

    deadline = dt.datetime.now(tz=dt.timezone.utc) + dt.timedelta(seconds=60)
    until = _external_wait_lease_until(
        _ctx({"deadline_at": deadline.isoformat()}), 1800, None, 0)
    assert until == pytest.approx(deadline.timestamp(), abs=2.0)


def test_lease_until_is_clamped_by_the_runs_own_max_seconds_horizon():
    from ouroboros.tools.delegate import _external_wait_lease_until

    started = dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(seconds=100)
    until = _external_wait_lease_until(_ctx(), 1800, started, 150)
    assert until == pytest.approx(
        started.timestamp() + 150 + DELEGATE_WAIT_LEASE_GRACE_SEC, abs=2.0)


def test_lease_until_never_exceeds_the_absolute_ceiling():
    from ouroboros.tools.delegate import _external_wait_lease_until

    now = time.time()
    until = _external_wait_lease_until(_ctx(), 10_000_000, None, 0)
    assert until <= now + EXTERNAL_WAIT_LEASE_CEILING_SEC + 2.0


# -- the wait grants and releases ----------------------------------------------


def test_delegate_wait_grants_the_lease_and_releases_it(tmp_path, monkeypatch):
    import ouroboros.tools.delegate as delegate
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.gateways import claudexor as gw
    from ouroboros.tools.registry import ToolContext

    class _Stub:
        engine_version = "3.3.6"

        def handshake(self, **_kw): return {}
        def get_run(self, rid, *, timeout_sec=None):
            return {"lastSeq": 1, "summary": {
                "state": "running", "effectiveAccess": "readonly",
            }}
        def close(self): pass

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    repo = tmp_path / "repo"
    repo.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path,
                      task_constraint=TaskConstraint(mode="local_readonly_subagent"))
    ctx.task_id = "t-nanny"
    ctx.event_queue = stdqueue.Queue()
    delegate._CUSTODY.clear()
    delegate._CUSTODY["run-1"] = delegate._RunCustody(
        task_id="t-nanny", route_id="some-route", model="m",
        project_id="prj", project_owned=False,
    )
    out = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1, since_seq=1))
    delegate._CUSTODY.clear()
    assert out["status"] in ("progress", "no_progress"), out

    events = []
    while True:
        try:
            events.append(ctx.event_queue.get_nowait())
        except stdqueue.Empty:
            break
    leases = [e for e in events if e.get("type") == "external_wait_lease"]
    assert len(leases) == 2, events
    grant, release = leases
    assert grant["task_id"] == "t-nanny" and grant["run_id"] == "run-1"
    assert grant["until_ts"] > time.time()
    assert grant["until_ts"] <= time.time() + 1 + DELEGATE_WAIT_LEASE_GRACE_SEC + 2.0
    assert release["until_ts"] == 0.0
    # F5b lease identity: the grant minted an id and the release names the SAME
    # one, so the supervisor can match them.
    assert grant["lease_id"] and grant["lease_id"] == release["lease_id"]


# -- the supervisor handler -----------------------------------------------------


def test_handler_sets_clamps_and_clears_the_lease():
    from supervisor import events as events_mod

    meta = {"task": {"id": "t1"}}
    ctx = types.SimpleNamespace(RUNNING={"t1": meta})
    now = time.time()

    events_mod._handle_external_wait_lease(
        {"type": "external_wait_lease", "task_id": "t1", "run_id": "r1",
         "until_ts": now + 300}, ctx)
    assert meta["external_wait_lease_until"] == pytest.approx(now + 300, abs=2.0)
    assert meta["external_wait_lease_run_id"] == "r1"

    # A malformed/oversized worker value is re-clamped supervisor-side.
    events_mod._handle_external_wait_lease(
        {"type": "external_wait_lease", "task_id": "t1", "run_id": "r1",
         "until_ts": now + 10_000_000}, ctx)
    assert meta["external_wait_lease_until"] <= now + EXTERNAL_WAIT_LEASE_CEILING_SEC + 2.0

    # Release drops it immediately.
    events_mod._handle_external_wait_lease(
        {"type": "external_wait_lease", "task_id": "t1", "run_id": "r1",
         "until_ts": 0.0}, ctx)
    assert "external_wait_lease_until" not in meta
    assert "external_wait_lease_run_id" not in meta

    # An unknown task is a no-op, never a resurrection.
    events_mod._handle_external_wait_lease(
        {"type": "external_wait_lease", "task_id": "gone", "until_ts": now + 60}, ctx)
    assert set(ctx.RUNNING) == {"t1"}


def test_a_stale_release_cannot_blank_a_newer_grant():
    """F5b lease identity, direction 1: an abandoned, executor-killed wait
    thread's LATE release (its own lease_id) arrives after the task's next wait
    granted a NEW lease — the release must be refused, or the newer hold loses
    its idle-rail protection mid-window."""
    from supervisor import events as events_mod

    meta = {"task": {"id": "t1"}}
    ctx = types.SimpleNamespace(RUNNING={"t1": meta})
    now = time.time()

    events_mod._handle_external_wait_lease(
        {"type": "external_wait_lease", "task_id": "t1", "run_id": "r1",
         "until_ts": now + 300, "lease_id": "old"}, ctx)
    events_mod._handle_external_wait_lease(
        {"type": "external_wait_lease", "task_id": "t1", "run_id": "r1",
         "until_ts": now + 600, "lease_id": "new"}, ctx)
    assert meta["external_wait_lease_id"] == "new"

    # The abandoned thread's stale release names "old": refused.
    events_mod._handle_external_wait_lease(
        {"type": "external_wait_lease", "task_id": "t1", "run_id": "r1",
         "until_ts": 0.0, "lease_id": "old"}, ctx)
    assert meta.get("external_wait_lease_id") == "new"
    assert meta.get("external_wait_lease_until") == pytest.approx(now + 600, abs=2.0)


def test_a_matching_release_clears_the_grant():
    """F5b lease identity, direction 2: the release that NAMES the stored grant
    clears it (and a legacy id-less release keeps the old unconditional
    behaviour)."""
    from supervisor import events as events_mod

    meta = {"task": {"id": "t1"}}
    ctx = types.SimpleNamespace(RUNNING={"t1": meta})
    now = time.time()

    events_mod._handle_external_wait_lease(
        {"type": "external_wait_lease", "task_id": "t1", "run_id": "r1",
         "until_ts": now + 600, "lease_id": "new"}, ctx)
    events_mod._handle_external_wait_lease(
        {"type": "external_wait_lease", "task_id": "t1", "run_id": "r1",
         "until_ts": 0.0, "lease_id": "new"}, ctx)
    assert "external_wait_lease_until" not in meta
    assert "external_wait_lease_id" not in meta

    # Legacy emitter: a release with NO id still clears an id-carrying grant.
    events_mod._handle_external_wait_lease(
        {"type": "external_wait_lease", "task_id": "t1", "run_id": "r1",
         "until_ts": now + 600, "lease_id": "solo"}, ctx)
    events_mod._handle_external_wait_lease(
        {"type": "external_wait_lease", "task_id": "t1", "run_id": "r1",
         "until_ts": 0.0}, ctx)
    assert "external_wait_lease_until" not in meta


# -- the enforcer spares ONLY the idle rail -------------------------------------


def _enforcer(monkeypatch, tmp_path, running, *, abs_ceiling=10_000_000):
    from supervisor import queue as queue_mod

    monkeypatch.setattr(queue_mod, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue_mod, "RUNNING", running)
    monkeypatch.setattr(queue_mod, "PENDING", [])
    monkeypatch.setattr(queue_mod, "get_task_idle_timeout_sec", lambda: 900)
    monkeypatch.setattr(queue_mod, "get_per_call_timeout_ceiling_sec", lambda: 60)
    monkeypatch.setattr(queue_mod, "get_task_abs_ceiling_sec", lambda: abs_ceiling)
    monkeypatch.setattr(queue_mod, "FINALIZATION_GRACE_SEC", 300)
    monkeypatch.setattr(queue_mod, "_request_finalization_grace",
                        lambda *_a, **_k: "msg-1")

    def _run(now):
        queue_mod._enforce_task_timeouts_locked(
            types.SimpleNamespace(WORKERS={}), now, 7, {})

    return _run


def test_a_live_lease_spares_the_idle_rail(monkeypatch, tmp_path):
    meta = {
        "task": {"id": "t1", "chat_id": 7},
        "started_at": 1000.0, "last_progress_at": 1000.0, "worker_id": 0,
        "external_wait_lease_until": 3000.0,
    }
    _enforcer(monkeypatch, tmp_path, {"t1": meta})(2500.0)  # idle > 960s, leased
    assert "finalization_requested_at" not in meta

    # The SAME idleness with the lease expired opens the grace episode.
    meta["external_wait_lease_until"] = 2400.0
    _enforcer(monkeypatch, tmp_path, {"t1": meta})(2500.0)
    assert meta["finalization_requested_at"] == 2500.0


def test_the_lease_never_spares_deadline_or_ceiling(monkeypatch, tmp_path):
    # Absolute ceiling cuts through a live lease.
    meta = {
        "task": {"id": "t1", "chat_id": 7},
        "started_at": 1000.0, "last_progress_at": 1000.0, "worker_id": 0,
        "external_wait_lease_until": 10_000_000.0,
    }
    _enforcer(monkeypatch, tmp_path, {"t1": meta}, abs_ceiling=1000)(2500.0)
    assert meta.get("finalization_requested_at") == 2500.0
    assert meta.get("finalization_reason") == "absolute_ceiling"

    # An explicit task deadline cuts through it too.
    deadline = dt.datetime.fromtimestamp(2000.0, tz=dt.timezone.utc).isoformat()
    meta2 = {
        "task": {"id": "t2", "chat_id": 7, "deadline_at": deadline},
        "started_at": 1000.0, "last_progress_at": 2400.0, "worker_id": 0,
        "external_wait_lease_until": 10_000_000.0,
    }
    _enforcer(monkeypatch, tmp_path, {"t2": meta2})(2500.0)
    assert meta2.get("finalization_requested_at") == 2500.0
    assert meta2.get("finalization_reason") == "deadline"


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
    assert "finalization_requested_at" not in meta

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
    assert meta.get("finalization_requested_at") == 2500.0


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
            and row.get("operation_id") == "tool-1"
            and row.get("phase") == "finished"
        ]
        if not terminal:
            time.sleep(0.02)
    assert terminal, rows


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
    assert "finalization_requested_at" not in meta

    meta.pop("active_llm_call")
    _enforcer(monkeypatch, tmp_path, {"t4": meta}, abs_ceiling=10_000)(5000.0)
    assert meta["finalization_requested_at"] == 5000.0
    assert meta["finalization_reason"] == "idle_timeout"


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
    assert meta["finalization_reason"] == "idle_timeout"


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
