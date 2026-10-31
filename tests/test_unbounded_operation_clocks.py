"""1A/R3: an operation with no true bound gets no invented clock; a true bound still binds.

A packet review row is ONE request, which keeps its socket bound. A retrieving row (an agent
session, a native API episode) sends many, and only an explicit window, the owner deadline
less the grace, or the task's finite execution window bounds it. Absence reaches every
consumer as absence — no ``maxSeconds`` on the wire, no execution-deadline scope, no host tool
timeout — and each such wait keeps a positive Stop path. Finite bounds keep their numbers.
"""

from __future__ import annotations

import math
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tests._review_session_route_shared import _owned_gateway_uses_each_test_transport as __owned_transport
from tests._review_session_route_shared import fake_route as __fake_route

_owned_gateway_uses_each_test_transport = __owned_transport
fake_route = __fake_route

from tests._review_session_route_shared import FakeGateway, FakeLLM, _agent_request, _agent_slot  # noqa: E402


@pytest.fixture(autouse=True)
def _no_true_bound(monkeypatch):
    for key in ("OUROBOROS_TASK_ABS_CEILING_SEC", "OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC",
                "OUROBOROS_LLM_TRANSPORT_READ_TIMEOUT_SEC", "OUROBOROS_FINALIZATION_GRACE_SEC"):
        monkeypatch.delenv(key, raising=False)


def _iso_in(seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def _slot(route: str, native=None, **values):
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewSlot

    return ReviewSlot(slot_id=values.pop("slot_id", "seat-1"), model="cursor/test", route=ReviewRouteKind(route),
                      native_retrieval_override=native, **values)


@pytest.mark.parametrize("route,native,unbounded", [
    pytest.param("api_chat", None, 2700.0, id="packet-keeps-its-socket-bound"),
    pytest.param("api_chat", True, None, id="native-episode"),
    pytest.param("agent_session", None, None, id="agent-session"),
])
def test_the_delivery_fact_decides_a_slot_window(monkeypatch, route, native, unbounded):
    """R3: a native API episode used to take the 2700 s socket bound as its whole window,
    and a session the 6 h fallback; now only a true bound gives either one."""
    import ouroboros.review_custody as custody
    from ouroboros.review_substrate import ReviewRequest

    slot = _slot(route, native)
    request = ReviewRequest(surface="multi_model_review", goal="g", task_id="t-r3")
    assert custody._logical_timeout(slot, request, {}) == unbounded
    monkeypatch.setenv("OUROBOROS_TASK_ABS_CEILING_SEC", "3600")
    assert custody._logical_timeout(slot, request, {}) == (2700.0 if unbounded else 3600.0)
    soon = ReviewRequest(surface="multi_model_review", goal="g", task_id="t-r3", deadline_at=_iso_in(600))
    assert 0 < custody._logical_timeout(slot, soon, {}) <= 600 - 120  # the owner deadline less the grace


def _custody_wave(tmp_path, monkeypatch, slot, task_id):
    """The real coordinator over one slot whose worker, once sent, moves the slot clock past
    both retired windows (2700 s, 21 600 s) and holds; the drain re-reads that clock in short
    slices around the real ``drain_slice``. Returns the actor, what the worker saw, the slices."""
    import ouroboros.review_custody as custody
    from ouroboros.review_substrate import ReviewActorRecord, ReviewRequest
    from ouroboros.usage_accounting import UsageScope

    clock = {"now": 100.0}
    monkeypatch.setattr(custody, "monotonic_now", lambda _slot_id: clock["now"])
    real_slice, slices = custody.review_pause.drain_slice, []

    def drain_slice(operation, remaining):
        bounded = real_slice(operation, remaining)
        slices.append((remaining, bounded))
        return min(bounded, 0.05)

    monkeypatch.setattr(custody.review_pause, "drain_slice", drain_slice)
    entered, release, seen = threading.Event(), threading.Event(), {}

    def run_slot(_slot, _operation_id, _retry_state, deadline, _checkpoint):
        from ouroboros.model_wait import dispatch_deadline_remaining_sec

        seen.update(deadline=deadline, scoped=dispatch_deadline_remaining_sec())
        clock["now"] = 100.0 + 30_000.0
        entered.set()
        assert release.wait(10), "the test never released the review worker"
        return ReviewActorRecord(slot_id=slot.slot_id, model=slot.model, status="ok", raw_text="[]")

    def error_actor(_slot, error, operation_id="", operation_state="settled"):
        return ReviewActorRecord(slot_id=slot.slot_id, model=slot.model, status="error", error=error,
                                 operation_id=operation_id, operation_state=operation_state)

    def advance():
        assert entered.wait(10)
        time.sleep(0.5)  # about ten drain slices read the advanced clock first
        release.set()

    threading.Thread(target=advance, daemon=True).start()
    request = ReviewRequest(surface="multi_model_review", goal="g", task_id=task_id,
                            retry_key=f"commit_review:{task_id}")
    try:
        [actor] = custody.run_custodied_review_slots(
            request=request, slots=[slot], usage_ctx=SimpleNamespace(_review_paid_stamp=lambda: None),
            task_id=task_id, usage_meta={}, review_usage_scope=UsageScope(drive_root=tmp_path),
            run_slot=run_slot, error_actor=error_actor)
    finally:
        release.set()
    return actor, seen, slices


@pytest.mark.parametrize("route,native", [("api_chat", True), ("agent_session", None)],
                         ids=["native-episode", "agent-session"])
def test_an_unbounded_retrieving_slot_outlives_both_retired_clocks(tmp_path, monkeypatch, route, native):
    actor, seen, slices = _custody_wave(tmp_path, monkeypatch, _slot(route, native), f"t-free-{route}")
    assert (actor.status, actor.operation_state) == ("ok", "settled"), actor.error
    # Absence reaches the executor as absence: no deadline, and no execution-deadline scope.
    assert seen == {"deadline": None, "scoped": None}
    # The drain waited on an infinite remainder in finite slices (never Queue.get(timeout=inf)).
    assert any(math.isinf(remaining) for remaining, _ in slices)
    assert all(math.isfinite(bounded) for _, bounded in slices)


def test_a_finite_slot_window_still_expires_on_the_same_clock(tmp_path, monkeypatch):
    actor, seen, _slices = _custody_wave(tmp_path, monkeypatch, _slot("api_chat", True, timeout_sec=600), "t-finite")
    assert actor.operation_state == "in_flight" and actor.error.startswith("Timeout after 600s")
    assert seen["deadline"] == 700.0 and seen["scoped"] is not None


class _SlowGateway(FakeGateway):
    """A run that stays live for a few reads while the slot clock jumps past both retired windows."""

    reads_live = 3

    def get_run(self, run_id, **kwargs):
        if len(self.run_gets) < _SlowGateway.reads_live:
            self.run_gets.append(run_id)
            return {"summary": {"state": "running"}, "lastSeq": 1}
        return super().get_run(run_id, **kwargs)


def _review_session(tmp_path, monkeypatch, fake_route):
    import ouroboros.review_execution as execution
    from ouroboros.review_substrate import run_review_request

    monkeypatch.setattr("ouroboros.gateways.claudexor.ClaudexorGateway", _SlowGateway)
    monkeypatch.setattr(execution, "_SESSION_POLL_SEC", 0.01)
    real = time.monotonic
    # The session's own poll clock reads 30 000 s later once its first read is made, after
    # the poll took its deadline: past both retired windows (2700 s, 21 600 s).
    monkeypatch.setattr(execution, "time", SimpleNamespace(
        monotonic=lambda: real() + (30_000.0 if fake_route.instances and fake_route.instances[0].run_gets
                                    else 0.0), sleep=time.sleep))
    result = run_review_request(_agent_request(task_id="t-session"), slots=[_agent_slot(timeout_sec=None)],
                                drive_root=tmp_path, llm=FakeLLM())
    return result.actors[0], fake_route.instances[0]


def test_an_unbounded_session_is_posted_without_max_seconds_and_never_cancelled(tmp_path, monkeypatch, fake_route):
    from ouroboros import delegate_custody as custody

    actor, gateway = _review_session(tmp_path, monkeypatch, fake_route)
    assert actor["status"] == "ok", actor
    [body] = gateway.start_requests
    assert "maxSeconds" not in body  # the pinned engine reads omission as its run clock off
    assert gateway.cancels == []  # observed to its own terminal, never cut at 21 600 s
    record = custody.invocation_record(tmp_path, gateway.start_keys[0])
    assert record["max_seconds"] == 0 and "maxSeconds" not in record["request"]
    assert custody.run_timing(tmp_path, "run-1")[1] == 0  # no recorded cap, not an invented one


def test_a_finite_lifetime_still_reaches_the_wire_as_max_seconds(tmp_path, monkeypatch, fake_route):
    monkeypatch.setenv("OUROBOROS_TASK_ABS_CEILING_SEC", "900")
    _actor, gateway = _review_session(tmp_path, monkeypatch, fake_route)
    assert 1 <= gateway.start_requests[0]["maxSeconds"] <= 900
    # The jumped clock passes the 900 s window: the slot ends its run through the verified cancel.
    assert ("run-1", "review_slot_timeout") in gateway.cancels


MARGIN = 30.0


def _durable_rows(tmp_path, monkeypatch, *, body, max_seconds, started_ago=100.0):
    from ouroboros import delegate_custody as custody

    started = (datetime.now(timezone.utc) - timedelta(seconds=started_ago)).isoformat()
    row = {"run_id": "", "task_id": "t-rejoin", "invocation_id": "inv-1", "operation_id": "op-1",
           "surface": "plan_review", "slot_id": "seat-1", "max_seconds": max_seconds,
           "idempotency_key": "key-1"}
    if body is not None:
        row["request"] = body
    with monkeypatch.context() as stamped:
        stamped.setattr(custody, "utc_now_iso", lambda: started)
        assert custody.emit(tmp_path, custody.START_REQUESTED, row)
        assert custody.record_started(tmp_path, custody.RunCustody(
            run_id="run-1", task_id="t-rejoin", invocation_id="inv-1", route_id="claude", model="m",
            ledger_root=str(tmp_path)), shape={"max_seconds": max_seconds, "surface": "plan_review",
                                               "slot_id": "seat-1"})


@pytest.mark.parametrize("case", ["disabled", "finite", "recorded-cap-wins", "unknown", "unknown-no-facts"])
def test_a_rejoin_follows_the_clock_its_stored_body_sent(tmp_path, monkeypatch, case):
    """Key absence with no recorded cap = the run's clock is off (no window); a recorded cap
    keeps its remaining original window, whatever the body's shape (a continuation's body names
    only its predecessor); an unreadable body keeps the saved facts, else only the margin."""
    import ouroboros.review_custody as custody

    frozen = {"awaiting_since": (datetime.now(timezone.utc) - timedelta(seconds=100)).isoformat()}
    body = {"prompt": "p", "harnesses": ["claude"]}
    if case == "disabled":
        _durable_rows(tmp_path, monkeypatch, body=body, max_seconds=0)
        assert custody._rejoin_window(tmp_path, "inv-1", frozen, 600.0) is None
        assert custody._rejoin_window(tmp_path, "inv-1", frozen, None) is None
    elif case == "finite":
        _durable_rows(tmp_path, monkeypatch, body={**body, "maxSeconds": 600}, max_seconds=600)
        assert 490 <= custody._rejoin_window(tmp_path, "inv-1", frozen, None) <= 500
    elif case == "recorded-cap-wins":
        _durable_rows(tmp_path, monkeypatch, body={"continueFrom": "run-0"}, max_seconds=600)
        assert 490 <= custody._rejoin_window(tmp_path, "inv-1", frozen, None) <= 500
    elif case == "unknown":
        _durable_rows(tmp_path, monkeypatch, body=None, max_seconds=600)
        assert 490 <= custody._rejoin_window(tmp_path, "inv-1", frozen, None) <= 500
    else:
        _durable_rows(tmp_path, monkeypatch, body=None, max_seconds=0)
        assert custody._rejoin_window(tmp_path, "inv-1", frozen, None) == MARGIN
        assert 490 <= custody._rejoin_window(tmp_path, "inv-1", frozen, 600.0) <= 500  # today's rule


class _LiveCustody:
    cancelled: list = []

    @staticmethod
    def is_terminal(detail):
        return (detail.get("summary") or {}).get("state") == "succeeded"

    @staticmethod
    def summary_of(detail):
        return detail.get("summary") or {}

    @staticmethod
    def cancel_and_verify(_drive, _gateway, _entry, reason):
        _LiveCustody.cancelled.append(reason)
        return {"outcome": "confirmed", "state": "cancelled"}


def _live_poll(tmp_path, monkeypatch, *, source, seconds, stop_after=2, terminal_after=6):
    import ouroboros.review_execution as execution
    from ouroboros.gateways import claudexor_observation as observation

    reads = []

    class _Gateway:
        def get_run(self, run_id, **_kw):
            reads.append(run_id)
            state = "succeeded" if len(reads) >= terminal_after else "running"
            return {"summary": {"state": state}, "lastSeq": len(reads)}

    _LiveCustody.cancelled = []
    monkeypatch.setattr(execution, "_SESSION_POLL_SEC", 0.01)
    monkeypatch.setattr(observation, "wait_many_supported", lambda *_a, **_k: False)
    owner = SimpleNamespace(control_reason=lambda: "cancelled" if len(reads) >= stop_after else None)
    monkeypatch.setattr(execution, "current_model_wait", lambda: owner)
    entry = SimpleNamespace(run_id="run-s", source=source)
    return execution._poll_session_terminal(_Gateway(), _LiveCustody, tmp_path, entry, "run-s", seconds)


def test_an_unbounded_poll_of_its_own_run_ends_on_the_owners_stop(tmp_path, monkeypatch):
    """Deep self-review's session (no panel owns it) has no window: Stop ends the wait through
    the verified cancel instead of leaving an uninterruptible infinite poll."""
    from ouroboros.model_wait import ModelWaitInterrupted

    with pytest.raises(ModelWaitInterrupted) as stopped:
        _live_poll(tmp_path, monkeypatch, source="deep_self_review", seconds=float("inf"))
    assert stopped.value.control_reason == "cancelled"
    assert _LiveCustody.cancelled == ["owner_cancelled"]


@pytest.mark.parametrize("source,seconds", [("review_substrate:scope_review", float("inf")),
                                            ("deep_self_review", 600.0)], ids=["panel-run", "finite-window"])
def test_a_panel_run_and_a_finite_window_keep_their_existing_poll(tmp_path, monkeypatch, source, seconds):
    """#1006: a panel's run outlives its author's controls (custody cancels it only on an
    explicit cancellation); a finite window keeps its own clock and reads no control."""
    detail = _live_poll(tmp_path, monkeypatch, source=source, seconds=seconds)
    assert detail["summary"]["state"] == "succeeded" and _LiveCustody.cancelled == []


def _tools(entries, metadata=None, seen=None):
    from ouroboros.model_wait import dispatch_deadline_remaining_sec
    from ouroboros.tools.tool_result import ToolResult

    def execute_result(name, _args):
        if seen is not None:
            seen.append(dispatch_deadline_remaining_sec())
        time.sleep(0.3)  # longer than the finite entry below
        return ToolResult(status="ok", code="OK", text=f"{name} settled")

    return SimpleNamespace(get_timeout=entries.get, CODE_TOOLS=set(), execute_result=execute_result,
                           _ctx=SimpleNamespace(event_queue=SimpleNamespace(put_nowait=lambda _event: None),
                                                task_metadata=dict(metadata or {}), task_id="t-tool"))


def test_a_long_lived_tool_entry_has_no_host_timeout_and_only_a_deadline_narrows_it(monkeypatch):
    from ouroboros import loop_tool_execution as execution

    monkeypatch.setattr(execution, "load_settings", lambda: {"OUROBOROS_TOOL_TIMEOUT_SEC": 600})
    entries = {"review_change": None, "plan_task": None, "numeric": 30}
    tools = _tools(entries)
    assert execution._get_tool_timeout(tools, "review_change") is None  # Tool Timeout never re-caps it
    assert execution._get_tool_timeout(tools, "plan_task") is None
    assert execution._get_tool_timeout(tools, "numeric") == 600  # max(Tool Timeout, entry) is unchanged
    bounded = _tools(entries, {"deadline_at": _iso_in(1_000)})
    assert 870 <= execution._get_tool_timeout(bounded, "plan_task") <= 880  # the deadline less the grace
    assert execution._get_tool_timeout(bounded, "review_change") is None  # an unclamped tool keeps none


def test_the_loop_waits_for_a_long_lived_handler_without_scoping_a_deadline(tmp_path, monkeypatch):
    from ouroboros import loop_tool_execution as execution

    waits, seen = [], []
    real_wait = execution.future_result

    def wait(future, timeout):
        waits.append(timeout)
        return real_wait(future, timeout)

    monkeypatch.setattr(execution, "future_result", wait)
    tools = _tools({"review_change": None, "numeric": 0.1}, seen=seen)
    call = {"id": "call-1", "function": {"name": "review_change", "arguments": "{}"}}
    result = execution._execute_with_timeout(tools, call, tmp_path, None, task_id="t-tool")
    assert result["result"] == "review_change settled"
    assert waits == [None] and seen == [None]  # no host timeout and no execution-deadline scope
    call = {"id": "call-2", "function": {"name": "numeric", "arguments": "{}"}}
    timed_out = execution._execute_with_timeout(tools, call, tmp_path, 0.1, task_id="t-tool")
    until = time.monotonic() + 10
    while len(seen) < 2 and time.monotonic() < until:
        time.sleep(0.01)
    assert "TOOL_TIMEOUT" in timed_out["result"] and seen[1] is not None  # a finite entry is unchanged


def test_long_lived_envelopes_follow_the_task_lifetime(monkeypatch):
    from ouroboros.tools import delegate, plan_review, preflight_review, review_change

    for lifetime, envelope, wait in ((None, None, None), (3600, 3600 + 120, 3600 + 120)):
        for module in (plan_review, review_change):
            monkeypatch.setattr(module, "get_task_abs_ceiling_sec", lambda value=lifetime: value)
        monkeypatch.setattr("ouroboros.config.get_task_abs_ceiling_sec", lambda value=lifetime: value)
        monkeypatch.setattr(plan_review, "get_finalization_grace_sec", lambda: 120)
        monkeypatch.setattr(review_change, "get_finalization_grace_sec", lambda: 120)
        entries = {entry.name: entry for module in (plan_review, review_change, preflight_review, delegate)
                   for entry in module.get_tools()}
        for name in ("plan_task", "review_change", "preflight_review", "advisory_review"):
            assert entries[name].timeout_sec == envelope, name
        assert entries["delegate_wait"].timeout_sec == wait
        assert entries["delegate_start"].timeout_sec == 120  # an ordinary numeric entry is unchanged


def test_the_attention_lease_stays_finite_and_names_no_operation_timer():
    """The 21 600 s value survives only as the idle lease of a typed fact (attention, never a stop)."""
    import ouroboros.config as config
    import ouroboros.runtime_limits as limits

    assert config.IDLE_LEASE_MAX_SEC == limits.IDLE_LEASE_MAX_SEC == 21_600
    for retired in ("OPERATION_WINDOW_FALLBACK_SEC", "operation_window_sec", "get_plan_task_deadline_min_sec"):
        assert not hasattr(config, retired) and not hasattr(limits, retired), retired
