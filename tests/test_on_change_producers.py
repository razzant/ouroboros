"""Cost law L11, producers: the high-rate control facts are spoken only when they change.

On 01.11 ``external_wait_lease`` (788/min), ``cognitive_operation`` (308/min) and
``main_llm_call_state`` (158/min) were most of the control lane; every one is a bus round
trip, a dispatch and a frame on the server. Their producers now say a fact once per change:

* a ``delegate_wait`` is a sleep: it holds no slot, so it leases nothing for the idle rail to
  spare, and the one read its wake makes leases nothing of its own (the three-second beats
  that used to grant, re-grant and release are gone with the tick);
* a tool call's lifecycle is one ``started`` and one ``finished`` fact: the settlement row
  closes the lease, and the later ``tool_call_finished`` frame states no change;
* a model operation's display observation is forwarded only when it differs from the last.

Each guard is pinned both ways: the fact still arrives on a real change.
"""
from __future__ import annotations

import json
import queue as stdqueue
from types import SimpleNamespace

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


def test_a_wake_of_a_sleep_on_runs_speaks_no_lease(tmp_path, monkeypatch):
    from tests._run_wait_shared import held_run, wake_now

    ctx = _delegating_ctx(tmp_path, acting=False)
    ctx.event_queue = stdqueue.Queue()
    held_run(tmp_path, "run-existing", ctx.task_id, ended="succeeded")
    result = json.loads(wake_now(ctx, "run-existing", wait_once=lambda *_a, **_k: json.dumps(
        {"status": "terminal", "run_id": "run-existing", "state": "succeeded"})).text)
    assert result["status"] == "terminal"
    assert _leases(_drain(ctx.event_queue)) == [], "a sleep holds no slot: nothing to lease"


def test_a_wake_read_leases_nothing_while_a_model_wait_grants_and_releases(tmp_path, monkeypatch):
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
    ctx = _delegating_ctx(tmp_path, acting=False)
    ctx.event_queue = stdqueue.Queue()
    delegate._CUSTODY.clear()
    delegate._CUSTODY["run-1"] = delegate._RunCustody(task_id=ctx.task_id, route_id="r", model="m",
                                                       project_id="prj", project_owned=False)
    try:
        for _ in range(3):
            assert json.loads(delegate._delegate_wait(ctx, "run-1", 3, 1, observation_only=True))["state"] \
                == "running"
        assert _leases(_drain(ctx.event_queue)) == [], "a wake's read leases nothing"
        delegate._delegate_wait(ctx, "run-1", wait_sec=1, since_seq=1)
        grant, release = _leases(_drain(ctx.event_queue))
        assert grant["until_ts"] > 0 and release["until_ts"] == 0.0 and grant["lease_id"] == release["lease_id"]
    finally:
        delegate._CUSTODY.clear()


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
