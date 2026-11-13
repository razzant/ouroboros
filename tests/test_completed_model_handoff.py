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


@pytest.mark.parametrize(("bound", "asynchronous"), [
    pytest.param(False, False, id='unbound-sync'),
    pytest.param(True, True, id='bound-async'),
])
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


@pytest.mark.parametrize(("terminal_http", "asynchronous"), [
    pytest.param(False, False, id='transport-unknown-sync'),
    pytest.param(True, True, id='http-finished-async'),
])
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


@pytest.mark.parametrize(("pid_state", "facts"), [
    pytest.param("dead", _HTTP_FACTS, id='dead-http-400'),
    pytest.param("live", _PROBE_FACTS["TimeoutError"], id='live-timeout'),
    pytest.param("missing", _HTTP_FACTS, id='missing-http-400'),
    pytest.param("inaccessible", _PROBE_FACTS["TimeoutError"], id='inaccessible-timeout'),
])
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


@pytest.mark.parametrize("case", ["unknown-status", "transport-cause", "missing-cause"])
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


@pytest.mark.parametrize("case", ["body-error", "partial-attempt", "still-dispatched"])
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


@pytest.mark.parametrize("failure", ["create-timeout", "stream-read-error", "stream-without-completion", None])
def test_web_search_read_that_ended_leaves_the_same_fact(tmp_path, monkeypatch, failure):
    """OpenAI Responses search keeps its own attempt lifecycle; an ended read is not a writer.

    While the request or stream runs, its attempt holds like any send; after
    the read raised or the stream closed without ``response.completed``, the
    row keeps unresolved money with the retained fact, and the census frees it;
    a completed stream settles. Inside its task's model wait (the tool's copied
    context) the leg records its own process as the sender and binds no receiver.
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
        if failure is None:
            yield SimpleNamespace(type="response.completed", response={
                "usage": {"input_tokens": 3, "output_tokens": 2}, "output": []})

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

    with model_wait.task_model_wait_scope(task={"id": "root", "_attempt": 1}, drive_root=tmp_path,
                                          event_queue=None, worker_slot_held=True) as owner:
        result = json.loads(search._web_search_openai(ctx, "synthetic query"))  # the leg itself, no routing
        assert owner.answer_consumer_bound is False, "the task's own receiver is not this leg's"

    assert [b["reason"] for b in inflight] == ["no_local_receiver"], "the running request is a writer, no more"
    assert calls == [1], result
    if failure is None:
        assert result["answer"] == "partial", result
        with usage_store.read(tmp_path) as txn:
            assert txn.open_attempts(root_task_id="root") == []
        return
    assert result.get("reason_code") == "provider_outcome_unknown", result
    with usage_store.read(tmp_path) as txn:
        row, = txn.open_attempts(root_task_id="root")
    assert row["local_answer_owner_pid"] == os.getpid() == row["local_answer_owner_proof"]["pid"]
    assert not any(row.get(k) for k in ("local_answer_consumer_id", "local_answer_owner_birth"))
    assert row.get("local_answer_task_attempt") is None
    assert row["state"] == "unresolved" and row.get("cost_usd") is None
    expected = {"create-timeout": "TimeoutError", "stream-read-error": "ReadError",
                "stream-without-completion": "IncompleteProviderStream"}[failure]
    assert row["physical_failure"]["stage"] == "raised_exception"
    assert row["physical_failure"]["exception_type"] == expected
    assert _handoffs(queue) == []
    assert _row(tmp_path, row["attempt_id"]) == row


# ---- an unbound send whose sending process ended (no receiver, outcome unknown) ----

_BINDING = {"endpoint": "https://openrouter.invalid/api/v1", "credential_sha256": "0" * 64}
_ENDED_SENDER = {"direct": """
import os, sys
from ouroboros import usage_accounting as ua
root = sys.argv[1]
scope = ua.UsageScope(drive_root=root, task_id="root", root_task_id="root")
request = ua.AttemptRequest(model="m", provider="openrouter", reservation_usd=.2, drive_root=root, task_id="root",
                            root_task_id="root", provider_receipt_binding=%r)
with ua.usage_scope(scope):  # the process ends inside its send: dispatched, transport unknown, no receiver
    ua.execute_physical_attempt(request, lambda: os._exit(0))
""" % (_BINDING,), "web_search": """
import os, sys
from types import SimpleNamespace
from ouroboros import model_wait, net_transport
from ouroboros.tools import search
root = sys.argv[1]
for name in ("ddgs", "openai"):  # no real backend or SDK client can be reached from here
    sys.modules[name] = None

def stream():  # the process ends mid-stream, after a delta: dispatched, transport unknown
    yield SimpleNamespace(type="response.output_text.delta", delta="partial")
    os._exit(0)

search._resolve_openai_client_settings = lambda: ("synthetic", "https://invalid.example/v1", "openai", "synthetic")
net_transport.web_search_openai_client = lambda **_kw: SimpleNamespace(
    responses=SimpleNamespace(create=lambda **_k: stream()))
ctx = SimpleNamespace(task_id="root", pending_events=[],
                      task_metadata={"root_task_id": "root", "budget_drive_root": root})
with model_wait.task_model_wait_scope(task={"id": "root", "_attempt": 1}, drive_root=root, event_queue=None,
                                      worker_slot_held=True):  # the tool runs in its task's copied context
    search._web_search_openai(ctx, "synthetic query")
"""}
_SENDER = 4_123_457  # a synthetic recorded sender: only the controlled probe below answers for it
_KEY = "unbound-attempt:"  # the stored retirement key of one unbound attempt (a persisted format)


def _ended_sender(root, producer="direct"):
    """The process-boundary witness: a real producer in a child that ends mid-send, reaped here."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, "-c", _ENDED_SENDER[producer], str(root)], cwd=repo,
                   env={**os.environ, "PYTHONPATH": repo}, check=True, timeout=_EVENT_TIMEOUT)
    with usage_store.read(root) as txn:
        row, = txn.open_attempts(root_task_id="root")
    assert row["state"] == "dispatched" and row["transport_outcome"] == "unknown" and "physical_failure" not in row
    from ouroboros.platform_layer import pid_provably_gone
    # A Windows venv redirector can have a different pid from the writing interpreter.
    assert type(row.get("local_answer_owner_pid")) is int and row["local_answer_owner_pid"] > 0
    assert pid_provably_gone(row["local_answer_owner_pid"])
    assert not any(row.get(k) for k in ("local_answer_consumer_id", "local_answer_owner_birth"))
    assert row.get("local_answer_task_attempt") is None
    return row


def _probe(monkeypatch, gone=(), kernel=None):
    """The public OS probe, controlled: a pid in ``gone`` is positively absent (ESRCH), any other is not
    provably gone (alive or EPERM); ``kernel`` (when given) answers each pid's live validated birth."""
    from ouroboros import local_custody_repair, platform_layer, process_identity

    gone = set(gone)
    monkeypatch.setattr(platform_layer, "pid_provably_gone", lambda pid: pid in gone)
    monkeypatch.setattr(local_custody_repair, "_exited_unreaped", lambda _pid: False)
    if kernel is not None:
        monkeypatch.setattr(process_identity, "process_kernel_birth", lambda pid: kernel.get(pid))
    return gone


def _sent(root, *, pid, task_id="root", root_task_id="root", **binding):
    """A dispatched send by the producer's own call; another pid records no proof (the incident's row)."""
    reservation = ua.reserve_attempt(ua.AttemptRequest(model="m", provider="test", reservation_usd=.2,
                                                       drive_root=root, task_id=task_id, root_task_id=root_task_id))
    ua.mark_dispatched(reservation, local_answer_owner_pid=pid)
    if binding:
        ua._transition(reservation, "unresolved", reason="outcome unknown", **binding)
    return reservation.attempt_id


def _retired(root, task_id="root"):
    from ouroboros.task_results import load_task_result
    return (load_task_result(root, task_id) or {}).get("retired_model_consumers") or {}


@pytest.mark.parametrize("producer", ["direct", "web_search"])
def test_ended_unbound_sender_frees_continue_once_and_keeps_unknown_money(tmp_path, monkeypatch, producer):
    """The web search leg runs inside its task's model wait, yet records only its own sender."""
    from ouroboros import platform_layer

    queue, _, _ = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    row = _ended_sender(tmp_path, producer)
    attempt_id, money = row["attempt_id"], ua.usage_projection(tmp_path)
    _interrupted(tmp_path, "root")
    assert [b["reason"] for b in _handoffs(queue)] == ["no_local_receiver"]
    assert _retired(tmp_path) == {}, "the passive census writes nothing"

    admitted = admit_continuation("root", action_nonce=NONCE)
    assert admitted["ok"] and not admitted["held"], admitted
    assert row["local_answer_owner_proof"]["pid"] == row["local_answer_owner_pid"], "the sender's own proof"
    retired = _retired(tmp_path)
    assert retired == {_KEY + attempt_id: {"receiver": "unbound", "attempt_id": attempt_id,
                                           "owner_pid": row["local_answer_owner_pid"],
                                           "retired_at": retired[_KEY + attempt_id]["retired_at"]}}
    assert _row(tmp_path, attempt_id) == row and ua.usage_projection(tmp_path) == money
    monkeypatch.setattr(platform_layer, "pid_provably_gone", lambda _pid: False)  # the pid is reused
    assert _handoffs(queue) == [], "a reused pid revives no blocker"

    if producer == "direct":  # an exact provider price fact
        late = ua.apply_provider_price_receipt(tmp_path, attempt_id, {
            "attempt_id": attempt_id, "provider": "openrouter", "cost_usd": .05,
            "evidence_ref": "synthetic:generation", "binding": dict(_BINDING)})
        assert late["status"] == "applied", late
    else:  # this route has no price binding: its late usage settles the same exact attempt
        ua.settle_attempt(ua.AttemptReservation(attempt_id, tmp_path, row["model"], row["provider"], None),
                          {"prompt_tokens": 1}, cost_usd=.05, cost_final=True)
    settled = _row(tmp_path, attempt_id)
    assert settled["state"] == "settled" and settled["cost_usd"] == .05 and settled["cost_final"] is True
    assert settled.get("settle_reason") == ("late_receipt" if producer == "direct" else None)
    assert _retired(tmp_path) == retired


def test_paused_root_resume_records_the_ended_sender_and_goes_on(tmp_path, monkeypatch):
    """The incident's door: an owner-paused root whose only open custody is an ended unbound send."""
    queue, install = _paused_root(tmp_path, monkeypatch)
    gone = _probe(monkeypatch)
    attempt_id = _sent(tmp_path, pid=_SENDER)
    install()
    before, money = _row(tmp_path, attempt_id), ua.usage_projection(tmp_path)
    refused = queue.resume_budget_paused_task("root")  # the sender is not provably gone: held, as before
    assert refused["error"] == "owner_pause_effects_unsettled", refused
    assert [b["reason"] for b in refused["blockers"]] == ["no_local_receiver"] and _retired(tmp_path) == {}
    gone.add(_SENDER)
    outcome = queue.resume_budget_paused_task("root")
    assert outcome["ok"], outcome
    assert set(_retired(tmp_path)) == {_KEY + attempt_id}
    assert _row(tmp_path, attempt_id) == before and ua.usage_projection(tmp_path) == money


@pytest.mark.parametrize("case", ["ended", "earlier-image", "own-session", "own-empty-session", "own-foreign-proof",
                                  "not-provably-gone", "probe-error", "no-pid", "nonpositive-pid",
                                  "partial-birth", "partial-attempt", "partial-consumer"])
def test_only_positive_absence_of_a_fully_unbound_sender_is_retired(tmp_path, monkeypatch, case):
    from ouroboros import platform_layer, tool_custody

    queue, _, _ = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    _probe(monkeypatch, () if case == "not-provably-gone" else {_SENDER})
    pid = {"no-pid": 0, "nonpositive-pid": -5}.get(case, _SENDER)
    if case.startswith("own") or case == "earlier-image":
        pid, proof = os.getpid(), {"earlier-image": {"pid": os.getpid(), "session": "an-earlier-image"},
                                   "own-empty-session": {"pid": os.getpid(), "session": ""},
                                   "own-foreign-proof": {"pid": _SENDER, "session": "an-earlier-image"}}.get(case)
        if proof is not None:  # else the producer's own proof: this image's current session
            monkeypatch.setattr(tool_custody, "owner_proof", lambda: dict(proof, kernel_birth=""))
    binding = {"partial-birth": {"local_answer_owner_birth": "100.boot"},
               "partial-attempt": {"local_answer_task_attempt": 1},
               "partial-consumer": {"local_answer_consumer_id": "receiver"}}.get(case, {})
    attempt_id = _sent(tmp_path, pid=pid, **binding)
    if case == "probe-error":
        def unreadable(_pid):
            raise OSError("probe failed")
        monkeypatch.setattr(platform_layer, "pid_provably_gone", unreadable)
    _interrupted(tmp_path, "root")
    before = _row(tmp_path, attempt_id)
    admitted = admit_continuation("root", action_nonce=NONCE)
    freed = case in ("ended", "earlier-image")
    assert admitted["ok"] and admitted["held"] is not freed, admitted
    assert bool(_retired(tmp_path)) is freed and _row(tmp_path, attempt_id) == before
    assert [b["attempt_id"] for b in _handoffs(queue)] == ([] if freed else [attempt_id])


def test_retirement_is_per_attempt_per_tree_and_a_failed_or_taskless_write_stays_held(tmp_path, monkeypatch, caplog):
    """A send whose task has no lifecycle row (a provider test's ``system:`` id) is skipped in the read: no
    write is tried or warned on each boot, no row is made, a raised read is no absence, a later row is met."""
    from ouroboros import task_store
    from ouroboros.local_custody_repair import reconcile_open_local_custody, repair_ended_local_custody
    from ouroboros.store import StoreBusy, Txn
    from ouroboros.task_results import load_task_result

    queue, _, _ = _install_queue(tmp_path, monkeypatch)
    for root_id in ("root", "other"):
        write_task_result(tmp_path, root_id, "running", root_task_id=root_id)
    _probe(monkeypatch, {_SENDER, _SENDER + 1, _SENDER + 2})
    ended, live = _sent(tmp_path, pid=_SENDER), _sent(tmp_path, pid=os.getpid())
    elsewhere = _sent(tmp_path, pid=_SENDER + 1, task_id="other", root_task_id="other")
    taskless = _sent(tmp_path, pid=_SENDER + 2, task_id="ghost")
    attempted = []
    with monkeypatch.context() as patch:
        def refused(_root, task_id, *_a, **_kw):
            attempted.append(task_id)
            raise OSError("store refused")
        patch.setattr(task_store, "put_handoffs", refused)
        assert repair_ended_local_custody(tmp_path, ["root", "ghost"], root_task_id="root") == []
    assert attempted == ["root"], "no write is tried for a task without a lifecycle row"
    assert _retired(tmp_path) == {}
    assert sorted(b["attempt_id"] for b in _handoffs(queue)) == sorted([ended, live, taskless])
    repaired = repair_ended_local_custody(tmp_path, ["root", "ghost"], root_task_id="root")
    assert [(r["kind"], r["task_id"]) for r in repaired] == [("receiver_owner_absent", "root")]
    assert set(_retired(tmp_path)) == {_KEY + ended}, "the sibling and the taskless send stay held"
    assert sorted(b["attempt_id"] for b in _handoffs(queue)) == sorted([live, taskless])
    assert _retired(tmp_path, "other") == {} and [b["attempt_id"] for b in conflicting_writers(queue, "other")
                                                  if b["kind"] == "model_handoff"] == [elsewhere]
    first = _retired(tmp_path)
    assert repair_ended_local_custody(tmp_path, ["root"], root_task_id="root") == [] and _retired(tmp_path) == first

    def swept():  # the boot's sweep reads every open row, of any tree; what it warned
        caplog.clear()
        with caplog.at_level("WARNING", logger="ouroboros.local_custody_repair"):
            done = [(r["task_id"], r["receiver"]) for r in reconcile_open_local_custody(tmp_path)]
        return done, [r.getMessage() for r in caplog.records if r.name == "ouroboros.local_custody_repair"]
    assert swept() == ([("other", "unbound")], []) and load_task_result(tmp_path, "ghost") is None
    write_task_result(tmp_path, "ghost", "running", root_task_id="root")  # its row appears later
    with monkeypatch.context() as patch:
        def busy(txn, task_id, *columns, row=Txn.row):
            if task_id == "ghost":
                raise StoreBusy("store busy")
            return row(txn, task_id, *columns)
        patch.setattr(Txn, "row", busy)
        assert swept() == ([], ["Retired receivers unreadable; receivers stay held"]), "a raised read is no absence"
    assert swept() == ([("ghost", "unbound")], []) and set(_retired(tmp_path, "ghost")) == {_KEY + taskless}


@pytest.mark.parametrize("refused", ["unbound", "bound"])
def test_a_refused_write_holds_only_its_own_receivers(tmp_path, monkeypatch, refused):
    """One task's ended bound receiver and ended unbound sender are written apart: a store refusal of
    one leaves exactly its rows held and unreported; the other lands and reports itself, once."""
    from ouroboros import task_store
    from ouroboros.local_custody_repair import repair_ended_local_custody

    queue, _, _ = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    _probe(monkeypatch, {_SENDER, _SENDER + 1})
    unbound = _sent(tmp_path, pid=_SENDER)
    bound = _sent(tmp_path, pid=_SENDER + 1, local_answer_consumer_id="receiver", local_answer_task_attempt=1,
                  local_answer_owner_birth="100.boot")
    keys = {"unbound": _KEY + unbound, "bound": "receiver"}
    rows, money = {a: _row(tmp_path, a) for a in (unbound, bound)}, ua.usage_projection(tmp_path)
    with monkeypatch.context() as patch:
        def refusing(*_a, **_kw):
            raise OSError("store refused")
        # the unbound row is written here directly; the bound one through its owner, ``retire_model_consumers``
        patch.setattr(task_store if refused == "unbound" else model_wait, "put_handoffs", refusing)
        repaired = repair_ended_local_custody(tmp_path, ["root"], root_task_id="root")
    landed = "bound" if refused == "unbound" else "unbound"
    assert [(r["kind"], r["receiver"]) for r in repaired] == [("receiver_owner_absent", landed)]
    assert set(_retired(tmp_path)) == {keys[landed]}
    assert [b["attempt_id"] for b in _handoffs(queue)] == [{"unbound": unbound, "bound": bound}[refused]]
    again = repair_ended_local_custody(tmp_path, ["root"], root_task_id="root")
    assert [(r["receiver"], r["pid"]) for r in again] == [
        (refused, _SENDER if refused == "unbound" else _SENDER + 1)], "the held one alone, later"
    assert set(_retired(tmp_path)) == set(keys.values()) and _handoffs(queue) == []
    assert {a: _row(tmp_path, a) for a in rows} == rows and ua.usage_projection(tmp_path) == money


def test_unbound_owner_proof_matrix(monkeypatch):
    from ouroboros.local_custody_repair import unbound_owner_ended
    from ouroboros.tool_custody import owner_proof

    me, live, gone, current = os.getpid(), _SENDER, _SENDER + 1, owner_proof()
    birth, other = "darwin-start:1794470000.000001", "darwin-start:1794470001.000001"
    _probe(monkeypatch, {gone}, kernel={live: birth})
    assert not any(unbound_owner_ended(pid, {}) for pid in (0, -5, None, str(live), True))
    assert not unbound_owner_ended(me, current) and not unbound_owner_ended(me, {})
    assert not unbound_owner_ended(me, {"pid": me, "session": ""})
    assert unbound_owner_ended(me, {"pid": me, "session": "an-earlier-image"})
    assert not unbound_owner_ended(me, {"pid": me + 1, "session": "an-earlier-image"})
    assert unbound_owner_ended(gone, {}) and not unbound_owner_ended(live, {})
    assert not unbound_owner_ended(live, {"pid": live, "kernel_birth": birth})
    assert not unbound_owner_ended(live, {"pid": live, "kernel_birth": ""})
    assert not unbound_owner_ended(live, {"pid": live, "kernel_birth": "Thu Nov 12 10:00:00 2026"}), "ps lstart"
    assert unbound_owner_ended(live, {"pid": live, "kernel_birth": other}), "the pid is another process now"
    assert not unbound_owner_ended(live, {"pid": live + 7, "kernel_birth": other}), "another pid's proof"
    _probe(monkeypatch, {gone}, kernel={})
    assert not unbound_owner_ended(live, {"pid": live, "kernel_birth": other}), "an unreadable birth proves nothing"


@pytest.mark.parametrize("origin", [{"source": "http_single", "reason": "Stop now"}, {"source": "panic"}],
                         ids=["owner-stop", "panic"])
def test_owner_stop_or_panic_of_the_work_still_refuses_continue(tmp_path, monkeypatch, origin):
    """Recording the ended sender (the boot's sweep) is custody only: it admits no successor."""
    from ouroboros.local_custody_repair import reconcile_open_local_custody
    from ouroboros.task_results import load_task_result

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    _probe(monkeypatch, {_SENDER})
    attempt_id = _sent(tmp_path, pid=_SENDER)
    _interrupted(tmp_path, "root", origin=origin)
    stopped, row, money = load_task_result(tmp_path, "root"), _row(tmp_path, attempt_id), ua.usage_projection(tmp_path)
    assert [r["kind"] for r in reconcile_open_local_custody(tmp_path)] == ["receiver_owner_absent"]
    assert reconcile_open_local_custody(tmp_path) == [], "written once"
    assert set(_retired(tmp_path)) == {_KEY + attempt_id} and _handoffs(queue) == []
    assert admit_continuation("root", action_nonce=NONCE) == {
        "ok": False, "error": "stopped_by_owner", "cause": origin["source"]}
    after = load_task_result(tmp_path, "root")
    assert "continued_by" not in after and workers.PENDING == []
    assert [after.get(k) for k in ("status", "cancel_origin")] == [stopped.get(k) for k in ("status", "cancel_origin")]
    assert _row(tmp_path, attempt_id) == row and ua.usage_projection(tmp_path) == money


@pytest.mark.parametrize("panic", [False, True], ids=["writer-wait", "panic-hold"])
def test_a_panic_hold_on_a_held_continue_outlives_its_retired_sender(tmp_path, monkeypatch, panic):
    """Without a Panic the held selection records the ended sender and starts the Continue once. A Panic
    replaces that wait (the restore's own rule): its sender's recorded end starts nothing, the mind's
    Resume is refused, and only the owner's Resume releases it."""
    from ouroboros.control_actor import mind_control
    from ouroboros.local_custody_repair import reconcile_open_local_custody
    from ouroboros.work_stop import hold_fact
    from supervisor.continuation_admission import recheck_held_continuations
    from supervisor.events_budget import HOLD_PANIC
    from supervisor.restart_retention import retained_pending

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    gone = _probe(monkeypatch)
    attempt_id = _sent(tmp_path, pid=_SENDER)
    _interrupted(tmp_path, "root")
    admitted = admit_continuation("root", action_nonce=NONCE)
    assert admitted["ok"] and admitted["held"], admitted
    successor = admitted["successor_task_id"]

    def held():
        return hold_fact(next(t for t in workers.PENDING if t["id"] == successor))
    assert held()["reason"] == "continuation_writer_unsettled"
    if panic:
        assert retained_pending(next(t for t in workers.PENDING if t["id"] == successor), sleep_hold_reason=HOLD_PANIC)
        assert held()["reason"] == HOLD_PANIC and held()["release"] == "owner"
    gone.add(_SENDER)
    if not panic:
        assert recheck_held_continuations() == [successor] and held() is None
        assert set(_retired(tmp_path)) == {_KEY + attempt_id}
        return
    assert [r["kind"] for r in reconcile_open_local_custody(tmp_path)] == ["receiver_owner_absent"]
    assert set(_retired(tmp_path)) == {_KEY + attempt_id} and _handoffs(queue) == []
    assert recheck_held_continuations() == [] and held()["reason"] == HOLD_PANIC
    with mind_control("controller", "its earlier sender ended"):
        refused = queue.resume_budget_paused_task(successor)
    assert refused["ok"] is False and refused["error"] == "panic_requires_owner_resume", refused
    assert held()["reason"] == HOLD_PANIC
    resumed = queue.resume_budget_paused_task(successor)  # the owner's own Resume: the required path
    assert resumed["ok"] and not resumed.get("held") and held() is None, resumed
