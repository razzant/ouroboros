"""Cost law L11, producers: the high-rate control facts are spoken only when they change.

On 01.11 ``external_wait_lease`` (788/min), ``cognitive_operation`` (308/min) and
``main_llm_call_state`` (158/min) were most of the control lane; every one is a bus round
trip, a dispatch and a frame on the server. Their producers now say a fact once per change:

* a ``delegate_wait`` is a sleep: it holds no slot, so nothing leases the idle rail for it; the
  ``external_wait_lease`` fact went with the windowed read that was its only speaker;
* a tool call's lifecycle is one ``started`` and one ``finished`` fact: the settlement row
  closes the lease, and the later ``tool_call_finished`` frame states no change;
* a model operation's display observation is forwarded only when it differs from the last.

Each guard is pinned both ways: the fact still arrives on a real change.
"""
from __future__ import annotations

import queue as stdqueue
from types import SimpleNamespace


def _drain(q) -> list:
    events = []
    while True:
        try:
            events.append(q.get_nowait())
        except stdqueue.Empty:
            return events


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
