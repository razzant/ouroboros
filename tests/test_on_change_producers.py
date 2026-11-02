"""Cost law L11, producers: the high-rate control facts are spoken only when they change.

On 01.11 ``external_wait_lease`` (788/min), ``cognitive_operation`` (308/min) and
``main_llm_call_state`` (158/min) were most of the control lane; every one is a bus round
trip, a dispatch and a frame on the server. Their producers now say a fact once per change:

* a supervising hold leases the idle rail ONCE for the whole hold (``HoldLease``), renewed
  only when less than the teardown grace remains, released once — its three-second
  observation beats lease nothing of their own (they used to grant, re-grant and release);
* a tool call's lifecycle is one ``started`` and one ``finished`` fact: the settlement row
  closes the lease, and the later ``tool_call_finished`` frame states no change;
* a model operation's display observation is forwarded only when it differs from the last.

Each guard is pinned both ways: the fact still arrives on a real change.
"""
from __future__ import annotations

import json
import queue as stdqueue
import time
from types import SimpleNamespace

from ouroboros import delegate_progress, delegate_supervision
from tests._delegated_transport_shared import _delegating_ctx


def _drain(q) -> list:
    events = []
    while True:
        try:
            events.append(q.get_nowait())
        except stdqueue.Empty:
            return events


def _leases(events) -> list:
    return [event for event in events if event.get("type") == "external_wait_lease"]


def test_a_supervised_hold_speaks_one_lease_however_many_beats(tmp_path, monkeypatch):
    ctx = _delegating_ctx(tmp_path, acting=False)
    ctx.event_queue = stdqueue.Queue()
    monkeypatch.setattr(delegate_supervision.time, "sleep", lambda _sec: None)
    beats = iter([json.dumps({"status": "no_progress", "run_id": "run-existing"})] * 25
                 + [json.dumps({"status": "terminal", "run_id": "run-existing", "state": "succeeded"})])
    result = json.loads(delegate_supervision.supervised_wait(
        ctx, "run-existing", wait_once=lambda *_a, **_k: next(beats)).text)
    assert result["status"] == "terminal"
    grant, release = _leases(_drain(ctx.event_queue))  # 26 beats, two lease facts
    assert grant["until_ts"] > time.time() and grant["run_id"] == "run-existing"
    assert release["until_ts"] == 0.0 and release["lease_id"] == grant["lease_id"]


def test_an_observation_beat_leases_nothing_while_a_model_wait_grants_and_releases(tmp_path, monkeypatch):
    from ouroboros import delegate_observe
    from ouroboros.gateways import claudexor as gw
    from ouroboros.tools import delegate

    class _Stub:
        engine_version = "5.0.1"

        def handshake(self, **_kw):
            return {}

        def get_run(self, rid, *, timeout_sec=None):
            return {"lastSeq": 1, "summary": {"state": "running", "effectiveAccess": "readonly"}}

        def close(self):
            pass

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    monkeypatch.setattr(delegate_observe, "step", lambda *a, **k: SimpleNamespace(
        payload=json.dumps({"status": "no_progress", "run_id": "run-1"}), detail={}))
    ctx = _delegating_ctx(tmp_path, acting=False)
    ctx.event_queue = stdqueue.Queue()
    delegate._CUSTODY.clear()
    delegate._CUSTODY["run-1"] = delegate._RunCustody(task_id=ctx.task_id, route_id="r", model="m",
                                                       project_id="prj", project_owned=False)
    try:
        for _ in range(3):
            assert json.loads(delegate._delegate_wait(ctx, "run-1", 3, 1, observation_only=True))["status"] \
                == "no_progress"
        assert _leases(_drain(ctx.event_queue)) == [], "a beat of a supervising hold leases nothing"
        delegate._delegate_wait(ctx, "run-1", wait_sec=1, since_seq=1)
        grant, release = _leases(_drain(ctx.event_queue))
        assert grant["until_ts"] > 0 and release["until_ts"] == 0.0 and grant["lease_id"] == release["lease_id"]
    finally:
        delegate._CUSTODY.clear()


def test_the_hold_lease_never_lapses_between_beats_and_speaks_only_a_later_expiry(monkeypatch):
    clock = [10_000.0]
    monkeypatch.setattr(delegate_progress.time, "time", lambda: clock[0])
    from ouroboros import delegate_custody

    monkeypatch.setattr(delegate_custody, "run_timing", lambda *_a: ("", 0))
    monkeypatch.setattr(delegate_custody, "custody_root", lambda _ctx: "/nowhere")
    ctx = SimpleNamespace(event_queue=stdqueue.Queue(), task_id="t", task_metadata={})
    lease = delegate_progress.HoldLease(ctx)
    spoken = []
    for phase in ([3.0] * 100, [63.0] * 10):  # ordinary 3 s beats, then beats as slow as a read bound
        for step in phase:
            lease.renew("run-1")
            clock[0] += step
            assert lease._until > clock[0], "the lease lapsed before the next beat"
        spoken.append(len(_leases(_drain(ctx.event_queue))))
    lease.release()
    released = _leases(_drain(ctx.event_queue))
    # 300 s of 3 s beats: the grant and one renewal a minute (the window, 60 s, before the grace) —
    # 5 facts where the per-beat lease spoke 300; a beat that nearly spends the read bound renews
    # at most once itself.
    assert spoken == [5, 10] and [event["until_ts"] for event in released] == [0.0]
    assert lease._until == 0.0

    # A lease bounded by the task's deadline cannot move later: one grant, then silence.
    import datetime as dt

    deadline = dt.datetime.fromtimestamp(clock[0] + 30, tz=dt.timezone.utc).isoformat()
    monkeypatch.setattr(delegate_progress.time, "time", lambda: clock[0])
    ctx = SimpleNamespace(event_queue=stdqueue.Queue(), task_id="t", task_metadata={"deadline_at": deadline})
    monkeypatch.setattr("ouroboros.deadline_utils.window_within_deadline", lambda _ctx, requested: 1)
    lease = delegate_progress.HoldLease(ctx)
    for _ in range(5):
        lease.renew("run-1")
        clock[0] += 3.0
    assert len(_leases(_drain(ctx.event_queue))) == 1


def test_a_tool_call_speaks_one_started_and_one_finished_lifecycle_fact():
    from ouroboros import loop_tool_execution as lte
    from ouroboros.tool_call_log import CALL_SETTLED

    q = stdqueue.Queue()
    tools = SimpleNamespace(_ctx=SimpleNamespace(event_queue=q, task_id="t", task_attempt=1, task_metadata={}))
    base = {"task_id": "t", "tool": "repo_read", "invocation_id": "inv-1", "tool_call_id": "call-1"}
    for kind in ("tool_call_started", CALL_SETTLED, "tool_call_finished"):
        lte._emit_live_log(tools, {**base, "type": kind})
    events = _drain(q)
    lifecycle = [(event["operation_id"], event["phase"]) for event in events if event["type"] == "cognitive_operation"]
    assert lifecycle == [("inv-1", "started"), ("inv-1", "finished")]
    assert [event["data"]["type"] for event in events if event["type"] == "log_event"] == [
        "tool_call_started", CALL_SETTLED, "tool_call_finished"], "every live frame still flows"


def test_a_model_operation_observation_is_forwarded_only_when_it_changes():
    from ouroboros.llm_claudexor import _ModelInvocation

    rows = []
    invocation = _ModelInvocation({}, {}, {"model_operation_observer": rows.append})
    invocation.operation_id, invocation.invocation_id = "op", "attempt"
    invocation.detail = {"state": "running", "dispatch": {"state": "started"}}
    for _ in range(5):
        invocation.observe_operation()
    assert [row["model_activity"]["phase"] for row in rows] == ["awaiting_response"]
    invocation.detail = {"state": "running", "dispatch": {"state": "started"}, "response": {"state": "ready"}}
    invocation.observe_operation()
    invocation.observe_operation()
    assert [row["model_activity"]["phase"] for row in rows] == ["awaiting_response", "response_ready"]
