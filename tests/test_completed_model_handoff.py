"""Unbound sends that already raised retain unknown money, not a live writer.

Real sync/async accounting, store, Continue admission/selection, owner Pause
settlement and paused-task Resume run here. Provider outcomes are local
callbacks; no provider is called.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import model_wait, usage_accounting as ua, usage_store
from ouroboros.task_results import write_task_result
from supervisor.continuation_admission import action_writers, admit_continuation, conflicting_writers
from tests._budget_pause_exact_helpers import _install_queue
from tests.test_owner_continue import NONCE, _interrupted

pytestmark = pytest.mark.serial

# These waits synchronize events, not a latency promise. Store creation and
# admission can take seconds under load; keep one bounded deadlock escape.
_EVENT_TIMEOUT = 60


def _request(root):
    return ua.AttemptRequest(model="m", provider="test", reservation_usd=.2,
                             drive_root=root, task_id="root", root_task_id="root")


def _http_error():
    response = httpx.Response(400, request=httpx.Request("POST", "https://provider.invalid/model"),
                              json={"error": {"type": "invalid_request", "message": "rejected"}})
    return httpx.HTTPStatusError("request rejected", request=response.request, response=response)


def _row(root, attempt_id):
    with usage_store.read(root) as txn:
        return txn.attempt(attempt_id)


def _assert_http_failure(attempt):
    facts = attempt["physical_failure"]
    assert facts["stage"] == "raised_exception"
    assert type(facts["provider_status_code"]) is int and facts["provider_status_code"] == 400
    assert facts["transport_cause_type"] is None


def _handoffs(queue):
    return [row for row in conflicting_writers(queue, "root") if row["kind"] == "model_handoff"]


def _failed_send(root, *, asynchronous=False, owner=None, error=None, while_sending=None):
    reservations = []
    error = error or _http_error()

    def send():
        if while_sending:
            while_sending()
        raise error

    async def async_send():
        return send()

    def before(reservation):
        reservations.append(reservation)

    scope = ua.UsageScope(drive_root=root, task_id="root", root_task_id="root")
    with ua.usage_scope(scope), (model_wait.operation_wait_scope(owner) if owner else contextlib.nullcontext()):
        with pytest.raises(type(error)):
            if asynchronous:
                asyncio.run(ua.execute_physical_attempt_async(_request(root), async_send, before_dispatch=before))
            else:
                ua.execute_physical_attempt(_request(root), send, before_dispatch=before)
    return reservations[0]


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("bound", [False, True], ids=["unbound", "bound"])
def test_producer_to_continue_preserves_live_receiver_and_late_money(tmp_path, monkeypatch, asynchronous, bound):
    from supervisor.events_budget import observe_held_budget_selection

    queue, _, _ = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    owner = model_wait.TaskModelWait(task={"id": "root", "_attempt": 1}, drive_root=tmp_path,
                                    event_queue=None, worker_slot_held=True) if bound else None
    inflight = []
    try:
        reservation = _failed_send(tmp_path, asynchronous=asynchronous, owner=owner,
                                    while_sending=lambda: inflight.extend(_handoffs(queue)))
        assert len(inflight) == 1, "the real sender is still a writer before its error returns"
        before = _row(tmp_path, reservation.attempt_id)
        assert before["state"] == "unresolved" and before.get("cost_usd") is None
        assert before.get("cost_final") is not True
        _assert_http_failure(before)
        _interrupted(tmp_path, "root")
        money = ua.usage_projection(tmp_path)

        assert bool(_handoffs(queue)) is bound
        admitted = admit_continuation("root", action_nonce=NONCE)
        assert admitted["ok"] and admitted["held"] is bound, admitted
        assert _row(tmp_path, reservation.attempt_id) == before
        assert ua.usage_projection(tmp_path) == money
        if owner:
            held = observe_held_budget_selection(queue, admitted["successor_task_id"])
            assert [b["kind"] for b in held["blockers"]] == ["model_handoff"]
            owner.close()
            assert observe_held_budget_selection(queue, admitted["successor_task_id"])["blockers"] == []

        ua.settle_attempt(reservation, {"prompt_tokens": 1}, cost_usd=.03, cost_final=True)
        after = _row(tmp_path, reservation.attempt_id)
        assert after["settle_reason"] == "late_receipt" and after["cost_usd"] == .03
        assert after["cost_final"] is True and after["physical_failure"] == before["physical_failure"]
    finally:
        if owner:
            owner.close()


def _paused_root(tmp_path, monkeypatch):
    """A running root under its owner's Pause, parked exactly: (queue, install)."""
    from ouroboros import budget_pause
    from supervisor.events_budget import install_exact_budget_pause
    from supervisor.owner_pause_control import request_owner_pause
    from tests.test_owner_pause import _owner_park

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_kw: 5.0)
    write_task_result(tmp_path, "root", "running", root_task_id="root", chat_id=0)
    workers.RUNNING["root"] = {"task": {"id": "root", "type": "task", "chat_id": 0,
                                        "root_task_id": "root"}, "worker_id": 0, "attempt": 1}

    def install():
        assert request_owner_pause("root", request_id="pause")["ok"]
        pause = _owner_park(tmp_path, monkeypatch, "root", [], root="root")
        context = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                                  WORKERS=workers.WORKERS, sort_pending=lambda: None,
                                  persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None)
        install_exact_budget_pause(context, "root", budget_pause.exact_pause_marker(pause)["checkpoint"])
    return queue, install


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("terminal_http", [False, True], ids=["transport-unknown", "http-finished"])
def test_actual_paused_task_resume_observes_completion(tmp_path, monkeypatch, asynchronous, terminal_http):
    from ouroboros import owner_pause

    queue, install = _paused_root(tmp_path, monkeypatch)
    reservation = _failed_send(tmp_path, asynchronous=asynchronous,
                                error=_http_error() if terminal_http else httpx.ReadTimeout("outcome unknown"))
    before = _row(tmp_path, reservation.attempt_id)
    assert before["state"] == "unresolved" and before["physical_failure"]["stage"] == "raised_exception"
    install()

    # The send returned by raising either way: only the remote outcome differs,
    # and it stays unresolved money, never a writer the Pause waits for.
    fence_before = owner_pause.read_fence(tmp_path, "root")
    assert fence_before["state"] == owner_pause.FENCE_PAUSED
    outcome = queue.resume_budget_paused_task("root")
    assert outcome["ok"], outcome
    assert _row(tmp_path, reservation.attempt_id) == before
    assert owner_pause.read_fence(tmp_path, "root") == fence_before


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_observed_http_response_is_held_until_unbound_send_exits(tmp_path, monkeypatch, asynchronous):
    from supervisor.events_budget import observe_held_budget_selection

    queue, _, _ = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    observed, release = threading.Event(), threading.Event()
    reservations, failures = [], []
    error = _http_error()  # A response exists; the physical callback has not exited.

    def receiving():
        observed.set()
        assert release.wait(_EVENT_TIMEOUT)

    def caller():
        try:
            reservations.append(_failed_send(tmp_path, asynchronous=asynchronous,
                                               error=error, while_sending=receiving))
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=caller)
    thread.start()
    try:
        assert observed.wait(_EVENT_TIMEOUT)
        with usage_store.read(tmp_path) as txn:
            attempt, = txn.open_attempts(root_task_id="root")
        assert attempt["state"] == "dispatched" and "physical_failure" not in attempt
        _interrupted(tmp_path, "root")
        admitted = admit_continuation("root", action_nonce=NONCE)
        assert admitted["ok"] and admitted["held"], admitted
        selection = observe_held_budget_selection(queue, admitted["successor_task_id"])
        assert [b["kind"] for b in selection["blockers"]] == ["model_handoff"]
    finally:
        release.set()
        thread.join(_EVENT_TIMEOUT)
    assert not thread.is_alive() and not failures, failures
    completed = _row(tmp_path, reservations[0].attempt_id)
    _assert_http_failure(completed)
    assert completed["state"] == "unresolved" and completed.get("cost_usd") is None
    assert observe_held_budget_selection(queue, admitted["successor_task_id"])["blockers"] == []
    resumed = queue.resume_budget_paused_task(admitted["successor_task_id"])
    assert resumed["ok"], resumed


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_failed_completion_evidence_write_keeps_continue_held(tmp_path, monkeypatch, asynchronous):
    queue, _, _ = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "running", root_task_id="root")

    def failed_write(*_a, **_kw):
        raise OSError("evidence unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(usage_store.Txn, "record_evidence", failed_write)
        reservation = _failed_send(tmp_path, asynchronous=asynchronous)
    before = _row(tmp_path, reservation.attempt_id)
    assert before["state"] == "unresolved" and "physical_failure" not in before
    assert before.get("cost_usd") is None
    _interrupted(tmp_path, "root")
    admitted = admit_continuation("root", action_nonce=NONCE)
    assert admitted["ok"] and admitted["held"], admitted
    assert _row(tmp_path, reservation.attempt_id) == before


def test_abandoned_receiver_stays_dispatched_until_sender_finishes(tmp_path, monkeypatch):
    from ouroboros import owner_pause
    from ouroboros._usage_wait import receiver_abandonable

    queue, _, _ = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    entered, release, returned, settled = (threading.Event() for _ in range(4))
    reservations, errors = [], []

    def send():
        entered.set()
        assert release.wait(_EVENT_TIMEOUT)
        raise _http_error()

    def caller():
        try:
            with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="root", root_task_id="root")):
                with receiver_abandonable():
                    ua.execute_physical_attempt(_request(tmp_path), send,
                        before_dispatch=lambda r: reservations.append(r),
                        late_owner=lambda *_: settled.set())
        except BaseException as exc:
            errors.append(exc)
        finally:
            returned.set()

    thread = threading.Thread(target=caller)
    thread.start()
    try:
        assert entered.wait(_EVENT_TIMEOUT)
        owner_pause.install_fence(tmp_path, "root", request_id="pause")
        assert returned.wait(_EVENT_TIMEOUT) and getattr(errors[0], "receiver_abandoned", False)
        before = _row(tmp_path, reservations[0].attempt_id)
        assert before["state"] == "dispatched" and "physical_failure" not in before
        assert len(_handoffs(queue)) == 1
    finally:
        release.set()
        thread.join(_EVENT_TIMEOUT)
    assert settled.wait(_EVENT_TIMEOUT) and not thread.is_alive()
    after = _row(tmp_path, reservations[0].attempt_id)
    assert after["state"] == "unresolved" and after.get("cost_usd") is None
    _assert_http_failure(after)
    assert _handoffs(queue) == []


def _legacy(root, *, pid, facts=None, state="unresolved", **binding):
    reservation = ua.reserve_attempt(_request(root))
    ua.mark_dispatched(reservation, local_answer_owner_pid=pid)
    with usage_store.hold(root) as txn:
        current = txn.attempt(reservation.attempt_id)
        txn.record_evidence(reservation.attempt_id, {"physical_failure": facts or {
            "stage": "raised_exception", "provider_status_code": 400, "transport_cause_type": None}},
            expected_revision=current["revision"])
    if state == "unresolved":
        ua._transition(reservation, state, reason="retained failure", **binding)
    return reservation


# The 2026-10-20 voice probes' retained facts, verbatim: a realtime setup refusal or the
# probe's own deadline, recorded by the sender itself — no HTTP status, no transport cause.
_PROBE_FACTS = {exc: {"stage": "raised_exception", "exception_type": exc, "provider_code": None,
                      "provider_status_code": None, "transport_cause_type": None}
                for exc in ("TimeoutError", "RealtimeSetupError")}
_HTTP_FACTS = {"stage": "raised_exception", "provider_status_code": 400, "transport_cause_type": None}


@pytest.mark.parametrize("facts", [_HTTP_FACTS, _PROBE_FACTS["TimeoutError"]], ids=["http-400", "timeout"])
@pytest.mark.parametrize("pid_state", ["dead", "live", "missing", "inaccessible"])
def test_historic_raised_send_uses_only_addressed_open_rows_without_pid_gate(tmp_path, monkeypatch, pid_state, facts):
    from ouroboros import platform_layer

    queue, _, _ = _install_queue(tmp_path, monkeypatch)
    pid = os.getpid() if pid_state == "live" else None
    if pid_state == "dead":
        # A real reaped PID, with no birth/consumer/task-attempt: the incident's shape.
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait(10)
        pid = child.pid
    if pid_state == "inaccessible":
        pid = 424242
        monkeypatch.setattr(platform_layer, "pid_provably_gone", lambda _pid: False)
    reservation = _legacy(tmp_path, pid=pid, facts=dict(facts))
    _interrupted(tmp_path, "root")
    before = _row(tmp_path, reservation.attempt_id)
    assert before["state"] == "unresolved" and before["physical_failure"] == facts
    assert not any(before.get(k) for k in ("local_answer_consumer_id", "local_answer_owner_birth",
                                         "local_answer_task_attempt"))
    read_attempts = usage_store.Txn.attempt_rows  # the attempts read every reader of this flow takes
    open_reads = []
    with monkeypatch.context() as patch:
        def addressed_open_only(txn, where="1", params=()):
            if (where, tuple(params)) != (f"{usage_store._OPEN_PREDICATE} AND root_task_id=?", ("root",)):
                pytest.fail("completed request classification must read only this root's open attempts")
            open_reads.append((where, tuple(params)))
            return read_attempts(txn, where, params)
        patch.setattr(usage_store.Txn, "attempt_rows", addressed_open_only)
        assert action_writers(queue, "root") == []
        admitted = admit_continuation("root", action_nonce=NONCE)
        assert admitted["ok"] and not admitted["held"], admitted
    assert open_reads
    assert _row(tmp_path, reservation.attempt_id) == before


@pytest.mark.parametrize("case", ["unknown-status", "success-status", "transport-cause", "missing-cause"])
def test_remote_facts_of_an_unbound_raised_send_do_not_hold_continue(tmp_path, monkeypatch, case):
    """Status and transport cause describe the remote exchange, never a local receiver."""
    queue, _, _ = _install_queue(tmp_path, monkeypatch)
    facts = dict(_HTTP_FACTS)
    if case in ("unknown-status", "success-status"):
        facts["provider_status_code"] = None if case == "unknown-status" else 200
    if case == "transport-cause":
        facts["transport_cause_type"] = "ReadError"
    if case == "missing-cause":
        facts.pop("transport_cause_type")
    reservation = _legacy(tmp_path, pid=os.getpid(), facts=facts)
    _interrupted(tmp_path, "root")
    before = _row(tmp_path, reservation.attempt_id)
    admitted = admit_continuation("root", action_nonce=NONCE)
    assert admitted["ok"] and not admitted["held"], admitted
    assert action_writers(queue, "root") == []
    assert _row(tmp_path, reservation.attempt_id) == before


@pytest.mark.parametrize("case", ["body-error", "no-stage", "failure-not-dict", "bound", "partial-birth",
                                   "partial-attempt", "still-dispatched"])
def test_bound_dispatched_or_unproven_failures_keep_continue_held(tmp_path, monkeypatch, case):
    queue, _, _ = _install_queue(tmp_path, monkeypatch)
    facts = dict(_PROBE_FACTS["TimeoutError"])
    if case == "body-error":
        facts["stage"] = "response_body_error"  # a received body, not a raised send: outside this classifier
    if case == "no-stage":
        facts.pop("stage")
    if case == "failure-not-dict":
        facts = "raised_exception"
    binding = {"bound": {"local_answer_consumer_id": "receiver"},
               "partial-birth": {"local_answer_owner_birth": "100.boot"},
               "partial-attempt": {"local_answer_task_attempt": 0}}.get(case, {})
    reservation = _legacy(tmp_path, pid=os.getpid(), facts=facts,
                           state="dispatched" if case == "still-dispatched" else "unresolved", **binding)
    _interrupted(tmp_path, "root")
    before = _row(tmp_path, reservation.attempt_id)
    admitted = admit_continuation("root", action_nonce=NONCE)
    assert admitted["ok"] and admitted["held"], admitted
    assert [b["kind"] for b in action_writers(queue, "root")] == ["model_handoff"]
    assert _row(tmp_path, reservation.attempt_id) == before


class RealtimeSetupError(RuntimeError):
    """A realtime provider's setup refusal: neither an HTTP response nor a transport cause."""


@pytest.mark.parametrize("failure", ["TimeoutError", "RealtimeSetupError"])
def test_probe_shaped_send_holds_the_pause_only_while_it_runs(tmp_path, monkeypatch, failure):
    """A standalone realtime probe attributed to a paused root, with no receiver bound.

    While its send runs, the owner Pause stays requested and Resume names the
    attempt; once the send raised (its own ``wait_for`` deadline, or a setup
    refusal), the same Pause settles and Resume proceeds. The row keeps its
    unresolved money and still accepts a late receipt.
    """
    from ouroboros import owner_pause
    from supervisor.owner_pause_control import refresh_owner_pause_tree

    queue, install = _paused_root(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    reservations, errors = [], []

    async def session():
        if failure == "RealtimeSetupError":
            raise RealtimeSetupError('provider_error: {"code":"string_above_max_length"}')
        await asyncio.Event().wait()  # never answers: the probe's own deadline ends it

    async def send():
        entered.set()
        await asyncio.to_thread(release.wait, _EVENT_TIMEOUT)
        return await asyncio.wait_for(session(), timeout=0.05)

    def probe():
        request = ua.AttemptRequest(model="openai::gpt-realtime-2.1", provider="openai_realtime",
                                    reservation_usd=1.0, drive_root=tmp_path, task_id="root",
                                    root_task_id="root", source="voice.protocol_probe")
        scope = ua.UsageScope(drive_root=tmp_path, task_id="root", root_task_id="root",
                              source="voice.protocol_probe")
        try:
            with ua.usage_scope(scope):
                asyncio.run(ua.execute_physical_attempt_async(request, send, before_dispatch=reservations.append))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=probe)
    thread.start()
    try:
        assert entered.wait(_EVENT_TIMEOUT)
        install()
        assert owner_pause.read_fence(tmp_path, "root")["state"] == owner_pause.FENCE_REQUESTED
        refused = queue.resume_budget_paused_task("root")
        assert refused["error"] == "owner_pause_effects_unsettled", refused
        assert refused["blockers"] == [{"kind": "model_handoff", "attempt_id": reservations[0].attempt_id,
                                        "review_owned": False, "task_id": "root", "reason": "no_local_receiver"}]
        assert _row(tmp_path, reservations[0].attempt_id)["state"] == "dispatched"
    finally:
        release.set()
        thread.join(_EVENT_TIMEOUT)
    assert not thread.is_alive() and [type(e).__name__ for e in errors] == [failure], errors
    ended = _row(tmp_path, reservations[0].attempt_id)
    assert ended["state"] == "unresolved" and ended.get("cost_usd") is None and ended.get("cost_final") is not True
    assert {k: ended["physical_failure"].get(k) for k in _PROBE_FACTS[failure]} == _PROBE_FACTS[failure]
    assert ended["local_answer_owner_pid"] == os.getpid()
    assert not any(ended.get(k) for k in ("local_answer_consumer_id", "local_answer_owner_birth"))
    assert ended.get("local_answer_task_attempt") is None
    money = ua.usage_projection(tmp_path)

    assert refresh_owner_pause_tree("root") == owner_pause.FENCE_PAUSED
    resumed = queue.resume_budget_paused_task("root")
    assert resumed["ok"], resumed
    assert _row(tmp_path, reservations[0].attempt_id) == ended
    assert ua.usage_projection(tmp_path) == money

    ua.settle_attempt(reservations[0], {"prompt_tokens": 1}, cost_usd=.04, cost_final=True)
    late = _row(tmp_path, reservations[0].attempt_id)
    assert late["settle_reason"] == "late_receipt" and late["cost_usd"] == .04
    assert late["physical_failure"] == ended["physical_failure"]


@pytest.mark.parametrize("failure", ["create-timeout", "stream-read-error", "stream-without-completion"])
def test_web_search_read_that_ended_leaves_the_same_fact(tmp_path, monkeypatch, failure):
    """OpenAI Responses search keeps its own attempt lifecycle; an ended read is not a writer.

    While the request or stream runs, its attempt holds like any send; after
    the read raised or the stream closed without ``response.completed``, the
    row keeps unresolved money with the retained fact, and the census frees it.
    """
    import httpx

    from ouroboros import net_transport
    from ouroboros.tools import search

    queue, _, _ = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    calls, inflight = [], []

    def stream():
        inflight.extend(_handoffs(queue))
        yield SimpleNamespace(type="response.output_text.delta", delta="partial")
        if failure == "stream-read-error":
            raise httpx.ReadError("connection lost mid-stream")

    def create(**_kwargs):
        calls.append(1)
        if failure == "create-timeout":
            inflight.extend(_handoffs(queue))
            raise TimeoutError("read timed out after dispatch")
        return stream()

    for name in ("ddgs", "openai"):  # no real backend or SDK client can be reached from here
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setattr(search, "_resolve_openai_client_settings",
                        lambda: ("synthetic", "https://invalid.example/v1", "openai", "synthetic"))
    monkeypatch.setattr(net_transport, "web_search_openai_client",
                        lambda **_kw: SimpleNamespace(responses=SimpleNamespace(create=create)))
    ctx = SimpleNamespace(task_id="root", pending_events=[],
                          task_metadata={"root_task_id": "root", "budget_drive_root": str(tmp_path)})

    result = json.loads(search._web_search_openai(ctx, "synthetic query"))  # the leg itself, no routing

    assert result.get("reason_code") == "provider_outcome_unknown" and calls == [1], result
    assert len(inflight) == 1, "the running request or stream is a writer"
    with usage_store.read(tmp_path) as txn:
        row, = txn.open_attempts(root_task_id="root")
    assert row["state"] == "unresolved" and row.get("cost_usd") is None
    expected = {"create-timeout": "TimeoutError", "stream-read-error": "ReadError",
                "stream-without-completion": "IncompleteProviderStream"}[failure]
    assert row["physical_failure"]["stage"] == "raised_exception"
    assert row["physical_failure"]["exception_type"] == expected
    assert _handoffs(queue) == []
    assert _row(tmp_path, row["attempt_id"]) == row
