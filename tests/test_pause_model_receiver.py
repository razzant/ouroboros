"""Owner Pause interrupts the author's LOCAL wait for an already-sent model answer.

Owner 2026-10-08 (quiz 9312a119, option 1): the receiver stops waiting and the
loop pauses at its previous ready point; the physical sender still owns its
reservation and settles whatever arrives later — the late answer is retained as
evidence and never adopted, its tools never run, its money is never invented.
Driven through the real ledger, the real executor handoff, the real fence and
the real ``TaskModelWait`` receiver identity.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import nullcontext
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import model_wait, owner_pause
from ouroboros import usage_accounting as ua
from ouroboros._usage_wait import receiver_abandonable
from ouroboros.task_results import load_task_result, write_task_result
from tests._usage_store_testing import attempt_rows_in_start_order
from tests.test_llm_claudexor import setup as subscription_setup  # noqa: F401 - the imported fixture



@pytest.fixture
def scope(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    monkeypatch.setattr("ouroboros._usage_wait.ABANDON_POLL_SEC", 0.05)
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    return ua.UsageScope(drive_root=tmp_path, task_id="root", root_task_id="root")


def _request():
    return ua.AttemptRequest(model="m", provider="test", reservation_usd=.2)


def _extract(_response):
    return {"prompt_tokens": 3, "completion_tokens": 2}, .1, True


def _call(tmp_path, scope, send, outcome, *, abandonable=True, owner=None):
    """The author's own round call on a worker thread, under its real wait owner."""
    owner = owner or model_wait.TaskModelWait(task={"id": "root", "_attempt": 1}, drive_root=tmp_path,
                                              event_queue=None, worker_slot_held=True)

    def run():
        try:
            with ua.usage_scope(scope), model_wait.operation_wait_scope(owner):
                if abandonable:
                    with receiver_abandonable():
                        outcome["result"] = ua.execute_physical_attempt(_request(), send, extractor=_extract)
                else:
                    outcome["result"] = ua.execute_physical_attempt(_request(), send, extractor=_extract)
        except BaseException as exc:  # noqa: BLE001 - the test inspects the exact interruption
            outcome["error"] = exc
        finally:
            outcome["returned_at"] = time.monotonic()

    thread = threading.Thread(target=run)
    thread.start()
    return thread, owner


def _until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_pause_abandons_the_sent_wait_at_once_and_the_sender_settles_and_retains_the_late_answer(tmp_path, scope):
    sent, release = threading.Event(), threading.Event()
    answer = {"choices": [{"message": {"content": "late", "tool_calls": [{"id": "never"}]}}]}

    def send():
        sent.set()
        assert release.wait(60)
        return answer

    outcome: dict = {}
    thread, owner = _call(tmp_path, scope, send, outcome)
    try:
        assert sent.wait(30)
        consumer = owner.answer_consumer_id
        owner_pause.install_fence(tmp_path, "root", request_id="pause")
        # The local wait ends on the Pause while the provider has not answered yet.
        assert _until(lambda: "returned_at" in outcome)
        error = outcome["error"]
        assert isinstance(error, model_wait.ModelWaitInterrupted)
        assert error.control_reason == "owner_pause" and error.receiver_abandoned is True
        row = attempt_rows_in_start_order(tmp_path)[0]
        assert row["state"] == "dispatched", "an abandoned wait claims nothing about the send"
        assert error.ledger_attempt_ids == [row["attempt_id"]]
        assert error.physical_attempt_capture.state == "dispatched"
        # The exact receiver is retired at once: the in-flight attempt is no writer any more.
        retired = load_task_result(tmp_path, "root")["retired_model_consumers"]
        assert retired[consumer]["task_attempt"] == 1 and owner.answer_consumer_id != consumer
    finally:
        release.set()
        thread.join(5)
    # The sender still owned its reservation: it settles the late answer exactly once.
    assert _until(lambda: attempt_rows_in_start_order(tmp_path)[0]["state"] == "settled")
    settled = attempt_rows_in_start_order(tmp_path)[0]
    assert settled["cost_usd"] == pytest.approx(.1)
    import json

    from ouroboros.observability import call_manifest_path

    path = call_manifest_path(tmp_path, "root", f"physical_{settled['attempt_id']}_response")
    assert _until(path.exists)  # retention follows the accounting settlement on the sender thread
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["control_reason"] == "owner_pause_abandoned", "the late paid answer is retained as evidence"
    assert "result" not in outcome, "the late answer never reached the caller"


def test_a_late_provider_error_is_settled_by_the_sender_as_unresolved(tmp_path, scope):
    sent, release = threading.Event(), threading.Event()

    def send():
        sent.set()
        assert release.wait(10)
        raise RuntimeError("provider reset after the receiver left")

    outcome: dict = {}
    thread, _owner = _call(tmp_path, scope, send, outcome)
    try:
        assert sent.wait(5)
        owner_pause.install_fence(tmp_path, "root", request_id="pause")
        assert _until(lambda: "returned_at" in outcome)
        assert getattr(outcome["error"], "receiver_abandoned", False)
    finally:
        release.set()
        thread.join(5)
    assert _until(lambda: attempt_rows_in_start_order(tmp_path)[0]["state"] == "unresolved")


def test_an_answer_that_arrives_before_the_receiver_notices_is_taken_not_abandoned(tmp_path, scope, monkeypatch):
    monkeypatch.setattr("ouroboros._usage_wait.ABANDON_POLL_SEC", 5.0)
    sent, release = threading.Event(), threading.Event()

    def send():
        sent.set()
        assert release.wait(10)
        return {"answer": True}

    outcome: dict = {}
    thread, _owner = _call(tmp_path, scope, send, outcome)
    try:
        assert sent.wait(5)
        owner_pause.install_fence(tmp_path, "root", request_id="pause")
    finally:
        release.set()  # the answer is handed over inside the same wait slice
        thread.join(5)
    assert outcome.get("result") == {"answer": True} and "error" not in outcome
    assert attempt_rows_in_start_order(tmp_path)[0]["state"] == "settled"


def test_only_the_main_round_wait_is_abandonable(tmp_path, scope):
    """A tool's or a reviewer's own model call keeps its wait (no receiver_abandonable scope)."""
    sent, release = threading.Event(), threading.Event()

    def send():
        sent.set()
        assert release.wait(10)
        return {"answer": "kept"}

    outcome: dict = {}
    thread, _owner = _call(tmp_path, scope, send, outcome, abandonable=False)
    try:
        assert sent.wait(5)
        owner_pause.install_fence(tmp_path, "root", request_id="pause")
        time.sleep(0.3)
        assert "returned_at" not in outcome
    finally:
        release.set()
        thread.join(5)
    assert outcome["result"] == {"answer": "kept"}


def test_the_abandoned_attempt_no_longer_holds_the_pause_census_but_money_stays_open(tmp_path, scope, monkeypatch):
    from tests._budget_pause_exact_helpers import _install_queue
    from supervisor.continuation_admission import conflicting_writers

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    sent, release = threading.Event(), threading.Event()

    def send():
        sent.set()
        assert release.wait(10)
        return {"late": True}

    outcome: dict = {}
    thread, _owner = _call(tmp_path, scope, send, outcome)
    try:
        assert sent.wait(5)
        before = [b for b in conflicting_writers(queue, "root") if b["kind"] == "model_handoff"]
        assert before and before[0]["review_owned"] is False
        owner_pause.install_fence(tmp_path, "root", request_id="pause")
        assert _until(lambda: "returned_at" in outcome)
        assert [b for b in conflicting_writers(queue, "root") if b["kind"] == "model_handoff"] == []
        assert ua.read_usage_records(tmp_path)[-1]["state"] == "dispatched", "unknown money is not zero"
    finally:
        release.set()
        thread.join(5)


def test_claudexor_sender_acknowledges_and_closes_after_its_receiver_left():
    from ouroboros.llm_claudexor import _ModelInvocation

    events = []

    class Gateway:
        def acknowledge_model_result(self, operation_id, sha):
            events.append(("ack", operation_id, sha))
            return {"response": {"state": "acknowledged"}}

        def close(self):
            events.append(("close",))

    invocation = _ModelInvocation.__new__(_ModelInvocation)
    invocation.gateway, invocation.io_lock = Gateway(), threading.Lock()
    invocation.operation_id, invocation.response_ref = "op-1", {"sha256": "abc"}
    invocation.retained, invocation.retention_error = {"manifest_ref": {"path": "x"}}, ""
    from ouroboros._usage_wait import close_unless_abandoned, late_transport_custody
    from ouroboros.model_wait import ModelWaitInterrupted

    late_transport_custody(invocation)({"outcome": "completed"}, None)
    assert events == [("ack", "op-1", "abc"), ("close",)]
    events.clear()
    invocation.gateway = Gateway()
    late_transport_custody(invocation)(None, RuntimeError("late failure"))
    assert events == [("close",)], "a failed late send acknowledges nothing it never received"
    events.clear()
    invocation.gateway = Gateway()
    abandoned = ModelWaitInterrupted("owner_pause")
    abandoned.receiver_abandoned = True
    with pytest.raises(ModelWaitInterrupted):
        try:  # the receiver's own abandonment: its sender still owns the gateway
            raise abandoned
        finally:
            close_unless_abandoned(invocation)
    assert events == []
    close_unless_abandoned(invocation)  # any other exit closes as before
    assert events == [("close",)]


@pytest.mark.parametrize(("timeout_type", "abandonable"), [
    pytest.param(TimeoutError, False, id='TimeoutError0-False'),
    pytest.param(FutureTimeout, True, id='TimeoutError1-True'),
])
def test_provider_timeout_is_an_error_not_a_receiver_poll(tmp_path, scope, monkeypatch, abandonable, timeout_type):
    failure = timeout_type("provider timed out")
    def send():
        raise failure
    def unexpected_tick(_reservation):
        pytest.fail("a completed provider TimeoutError was treated as a polling tick")
    monkeypatch.setattr("ouroboros._usage_wait._receiver_paused", unexpected_tick)
    with ua.usage_scope(scope), receiver_abandonable() if abandonable else nullcontext():
        with pytest.raises(timeout_type) as caught:
            ua.execute_physical_attempt(_request(), send, extractor=_extract)
    assert caught.value is failure
    row, = attempt_rows_in_start_order(tmp_path)
    assert row["state"] == "unresolved" and row.get("cost_usd") is None


@pytest.mark.parametrize("ending", ["answer", "timeout", "cancel"])
def test_completion_at_poll_boundary_wins_over_pause(monkeypatch, ending):
    from concurrent.futures import CancelledError, Future
    from ouroboros._usage_wait import model_send

    future = Future()
    failure = TimeoutError("completed at the wait boundary")
    def complete_at_boundary(*_args, **_kwargs):
        if ending == "cancel":
            future.cancel()
        elif ending == "timeout":
            future.set_exception(failure)
        else:
            future.set_result("completed")
        return SimpleNamespace(done=set())  # readiness changed after the wait tick
    monkeypatch.setattr("concurrent.futures.wait", complete_at_boundary)
    monkeypatch.setattr(owner_pause, "submit_model", lambda *_: future)
    monkeypatch.setattr("ouroboros._usage_wait._receiver_paused", lambda _: True)
    with receiver_abandonable():
        if ending == "answer":
            assert model_send(None, None, settle_late=lambda **_: pytest.fail("late settlement")) == "completed"
        else:
            with pytest.raises(CancelledError if ending == "cancel" else TimeoutError) as caught:
                model_send(None, None, settle_late=lambda **_: pytest.fail("late settlement"))
            if ending == "timeout":
                assert caught.value is failure


@pytest.mark.parametrize("late_error", [False, True])
def test_async_pause_leaves_sender_accounting_and_retires_receiver(tmp_path, scope, late_error):
    from ouroboros.observability import read_call_payload
    async def run():
        sent, release = asyncio.Event(), asyncio.Event()
        answer = {"content": "late async", "tool_calls": [{"id": "never"}]}
        async def send():
            sent.set()
            await release.wait()
            if late_error:
                raise TimeoutError("late provider timeout")
            return answer
        owner = model_wait.TaskModelWait(task={"id": "root", "_attempt": 1}, drive_root=tmp_path,
                                        event_queue=None, worker_slot_held=True)
        with ua.usage_scope(scope), model_wait.operation_wait_scope(owner), receiver_abandonable():
            receiver = asyncio.create_task(ua.execute_physical_attempt_async(_request(), send, extractor=_extract))
            await asyncio.wait_for(sent.wait(), 5)
            consumer = owner.answer_consumer_id
            owner_pause.install_fence(tmp_path, "root", request_id="pause")
            try:
                with pytest.raises(model_wait.ModelWaitInterrupted) as caught:
                    await asyncio.wait_for(receiver, 5)
                error = caught.value
                assert error.receiver_abandoned and error.physical_attempt_capture.state == "dispatched"
                assert attempt_rows_in_start_order(tmp_path)[0]["state"] == "dispatched"
                assert consumer in load_task_result(tmp_path, "root")["retired_model_consumers"]
            finally:
                release.set()
            await asyncio.wait_for(error.model_sender_future, 5)
            row, = attempt_rows_in_start_order(tmp_path)
            assert row["state"] == ("unresolved" if late_error else "settled")
            if not late_error:
                manifest, payload, _ = read_call_payload(tmp_path, task_id="root",
                    call_id=f"physical_{row['attempt_id']}_response")
                assert manifest["control_reason"] == "owner_pause_abandoned"
                assert payload["response"] == answer and row["cost_usd"] == pytest.approx(.1)
    asyncio.run(run())


@pytest.mark.parametrize("abandonable", [False, True])
def test_async_provider_timeout_propagates_after_accounting(tmp_path, scope, monkeypatch, abandonable):
    failure = TimeoutError("async provider timed out")
    async def send():
        raise failure
    async def run():
        with ua.usage_scope(scope), receiver_abandonable() if abandonable else nullcontext():
            with pytest.raises(TimeoutError) as caught:
                await asyncio.wait_for(ua.execute_physical_attempt_async(_request(), send), 5)
        assert caught.value is failure
        assert caught.value.physical_attempt_capture.state == "unresolved"
    asyncio.run(run())
    row, = attempt_rows_in_start_order(tmp_path)
    assert row["state"] == "unresolved" and row.get("cost_usd") is None


def test_async_cancellation_still_joins_received_answer_during_abandonable_poll(tmp_path, scope, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = ua.settle_attempt
    answer = {"content": "received before cancellation"}
    def settle(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)
    monkeypatch.setattr(ua, "settle_attempt", settle)
    async def send():
        return answer
    async def run():
        received = {}
        async def call():
            try:
                return await ua.execute_physical_attempt_async(_request(), send, extractor=_extract)
            except asyncio.CancelledError as error:
                received["error"] = error  # inspect before Task synthesizes its cancellation
                raise
        with ua.usage_scope(scope), receiver_abandonable():
            receiver = asyncio.create_task(call())
            assert await asyncio.to_thread(entered.wait, 5)
            receiver.cancel()
            await asyncio.sleep(.02)
            assert not receiver.done(), "caller cancellation must join received-answer accounting"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await receiver
            assert received["error"].response == answer
            assert received["error"].response_manifest_ref
            assert received["error"].physical_attempt_capture.state == "settled"
    try:
        asyncio.run(run())
    finally:
        release.set()
    row, = attempt_rows_in_start_order(tmp_path)
    assert row["state"] == "settled" and row["cost_usd"] == pytest.approx(.1)


@pytest.mark.parametrize(("retention_fails", "asynchronous"), [(False, True), (True, False)])
def test_subscription_chat_pause_preserves_late_receipt_without_adopting_it(
        subscription_setup, monkeypatch, asynchronous, retention_fails):  # noqa: F811 - the imported fixture
    from ouroboros import llm_claudexor
    from tests.test_llm_claudexor import MODEL, retained, result

    root, gateway, client = subscription_setup
    monkeypatch.setattr("ouroboros._usage_wait.ABANDON_POLL_SEC", .01)
    write_task_result(root, "task-one", "running", root_task_id="task-one")
    sent, release = threading.Event(), threading.Event()
    original = gateway.get_model_result
    def receive(*args, **kwargs):
        sent.set()
        assert release.wait(60)
        return original(*args, **kwargs)
    monkeypatch.setattr(gateway, "get_model_result", receive)
    monkeypatch.setattr(llm_claudexor._ModelInvocation, "finish", lambda *_: pytest.fail("late answer adopted"))
    persist = llm_claudexor.persist_call
    def retain(*args, **kwargs):
        if retention_fails and kwargs.get("call_type") == "llm_claudexor_response":
            raise OSError("exact wire retention unavailable")
        return persist(*args, **kwargs)
    monkeypatch.setattr(llm_claudexor, "persist_call", retain)
    async def run():
        with receiver_abandonable():
            call = client.chat_async([], MODEL) if asynchronous else asyncio.to_thread(client.chat, [], MODEL)
            receiver = asyncio.create_task(call)
            assert await asyncio.to_thread(sent.wait, 30)
            owner_pause.install_fence(root, "task-one", request_id="pause")
            try:
                with pytest.raises(model_wait.ModelWaitInterrupted) as caught:
                    await asyncio.wait_for(receiver, 5)
                assert caught.value.receiver_abandoned
                assert gateway.closed == 0 and gateway.acks == []
                assert attempt_rows_in_start_order(root)[0]["state"] == "dispatched"
            finally:
                release.set()
            sender = caught.value.model_sender_future
            await asyncio.wait_for(sender if asynchronous else asyncio.wrap_future(sender), 5)
    try:
        asyncio.run(run())
    finally:
        release.set()
    row, = attempt_rows_in_start_order(root)
    assert row["state"] == "settled" and row["cost_usd"] is None
    assert len(gateway.creates) == gateway.closed == 1
    assert len(gateway.acks) == (0 if retention_fails else 1)
    assert gateway.cancels == []
    if retention_fails:
        from ouroboros.observability import read_call_payload
        _manifest, payload, _ = read_call_payload(root, task_id="task-one",
            call_id=f"physical_{row['attempt_id']}_response")
        assert payload["response"] == result(), "generic receipt remains, exact engine bytes stay unacknowledged"
    else:
        assert retained(root) == result()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_temporary_api_client_stays_with_its_abandoned_sender(tmp_path, scope, monkeypatch, asynchronous):
    from ouroboros.llm import LLMClient
    from ouroboros.observability import read_call_payload
    from tests.test_physical_candidate_capture import _Response, _target

    client, response = LLMClient(api_key="unused"), _Response(text="late API answer")
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    def send(**_kwargs):
        entered.set()
        assert release.wait(60)
        assert not closed.is_set(), "caller closed the sender's transport"
        return response
    async def send_async(**kwargs):
        return await asyncio.to_thread(send, **kwargs)
    async def close_async():
        closed.set()
    transport = SimpleNamespace(close=closed.set, aclose=close_async)
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=send_async if asynchronous else send)))
    monkeypatch.setattr(client, "_resolve_remote_target", lambda *_: _target())
    monkeypatch.setattr(client, "_make_no_proxy_async_client" if asynchronous else "_make_no_proxy_client",
                        lambda *_args, **_kwargs: (sdk, transport))
    monkeypatch.setattr(client, "_build_remote_kwargs", lambda *_args, **_kwargs: {"model": "gpt-5.2", "messages": []})
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *_args, **_kwargs: .2)
    monkeypatch.setattr(client, "_normalize_remote_response", lambda *_args, **_kwargs: pytest.fail("late answer adopted"))
    async def run():
        with ua.usage_scope(scope), receiver_abandonable():
            call = (client.chat_async([], "m", no_proxy=True) if asynchronous
                    else asyncio.to_thread(client.chat, [], "m", no_proxy=True))
            receiver = asyncio.create_task(call)
            assert await asyncio.to_thread(entered.wait, 30)
            owner_pause.install_fence(tmp_path, "root", request_id="pause")
            try:
                with pytest.raises(model_wait.ModelWaitInterrupted) as caught:
                    await asyncio.wait_for(receiver, 5)
                assert not closed.is_set()
            finally:
                release.set()
            sender = caught.value.model_sender_future
            await asyncio.wait_for(sender if asynchronous else asyncio.wrap_future(sender), 5)
            assert await asyncio.to_thread(closed.wait, 5)
    try:
        asyncio.run(run())
    finally:
        release.set()
    row, = attempt_rows_in_start_order(tmp_path)
    assert row["state"] == "settled" and row["cost_usd"] == 0
    manifest, payload, _ = read_call_payload(tmp_path, task_id="root", call_id=f"physical_{row['attempt_id']}_response")
    assert manifest["control_reason"] == "owner_pause_abandoned"
    assert payload["response"] == response.model_dump()


def _census_bench(tmp_path, monkeypatch, direct):
    """A root work's real activity record (pooled RUNNING or the direct registry) and the public census read of it,
    fed exactly as the worker's event queue feeds them."""
    from copy import deepcopy

    from ouroboros.gateway import state as gateway_state
    from supervisor import events, queue as task_queue
    from supervisor.active_activity import get_direct_activity_registry

    write_task_result(tmp_path, "root", "running", root_task_id="root", chat_id=1, task_attempt=1,
                      _is_direct_chat=direct)
    meta = {"task": {"id": "root", "root_task_id": "root", "chat_id": 1, "_attempt": 1}, "attempt": 1, "started_at": 1.0}
    running = {} if direct else {"root": meta}
    for name, value in (("RUNNING", running), ("PENDING", []), ("BUDGET_ROOT_FENCES", {})):
        monkeypatch.setattr(task_queue, name, value)
    registry = get_direct_activity_registry()
    registry.clear()
    owner = model_wait.TaskModelWait(task={"id": "root", "_attempt": 1}, drive_root=tmp_path, event_queue=None,
                                     worker_slot_held=not direct)
    if direct:
        registry.register("root", 1, actor=SimpleNamespace(tools=SimpleNamespace(
            _ctx=SimpleNamespace(model_wait_context=owner))))

    class EventQueue:
        def put(self, event):
            if event.get("type") == "main_llm_call_state":
                events._handle_main_llm_call_state(deepcopy(event), SimpleNamespace(RUNNING=running))
        put_nowait = put

    return SimpleNamespace(meta=meta, registry=registry, owner=owner, queue=EventQueue(),
                           census=lambda: gateway_state._chat_activities_snapshot_safe(tmp_path)[0].get("context_attempt"))


def _low_context(**changes):
    from dataclasses import replace

    from ouroboros.physical_attempt_context import PhysicalAttemptContext

    low = PhysicalAttemptContext(profile="task_local_low", rendered_mode="low", measurement_basis="cold_estimate",
                                 route_fp="", round_id="exec:round:1", target_total_tokens=None,
                                 capacity_total_tokens=None, context_target_miss=False, automatic_pass_used=False,
                                 preferred_mode="max", mode_cause="memory")
    return replace(low, **changes)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("neighbour", ["alone", "after-genuine-send", "beside-newer-send"])
@pytest.mark.parametrize("direct", [False, True], ids=["pooled", "direct"])
def test_a_late_proof_of_no_start_retracts_the_paused_send_in_the_public_census(
        tmp_path, scope, monkeypatch, direct, neighbour, asynchronous):
    """Pause leaves the sent Low request dispatched and its call closed; its sender later proves it never started.
    The real ledger releases it, the sender publishes that release through the bound observer (``llm_observability``),
    and the activity record (pooled RUNNING or the direct registry) retracts exactly that retained identity after its
    call closed: an earlier genuine send stays its one previous, a newer send stays current and only loses the
    disproven previous, the next attempt not proven unsent taking its place; a forged release of another call or
    task attempt changes nothing."""
    from dataclasses import replace

    from ouroboros.llm_observability import emit_main_call_state
    from ouroboros.physical_attempt_context import observe_dispatch
    from ouroboros.transport_custody import ProviderNotDispatched

    bench = _census_bench(tmp_path, monkeypatch, direct)
    meta, registry, owner, queue_, census = bench.meta, bench.registry, bench.owner, bench.queue, bench.census
    paused = ("root", 1, "call-x", "exec", "exec:round:1", 1)
    low = _low_context()
    request = ua.AttemptRequest(model="m", provider="test", reservation_usd=.2, physical_context=low)
    sent, release, outcome = threading.Event(), threading.Event(), {}

    def refused():
        raise ProviderNotDispatched("the transport proves generation never started")

    def send():
        sent.set()
        assert release.wait(10)
        refused()

    async def send_async():
        sent.set()
        await asyncio.to_thread(release.wait, 10)
        refused()

    closed = threading.Event()

    def close(exc):  # the loop's own closing frame of the paused call, as ``loop_llm_call`` emits it
        outcome["error"] = exc
        emit_main_call_state(queue_, paused, "failed", capture=ua.physical_attempt_capture_from_exception(exc))
        closed.set()

    async def attempt_async():
        try:
            await ua.execute_physical_attempt_async(request, send_async, extractor=_extract)
        except BaseException as exc:  # noqa: BLE001 - the test inspects the exact interruption
            close(exc)
            if (sender := getattr(exc, "model_sender_future", None)) is not None:
                await asyncio.wait_for(sender, 10)  # the abandoned sender settles before this loop ends

    def run():
        emit_main_call_state(queue_, paused, "started")
        try:
            with ua.usage_scope(scope), model_wait.operation_wait_scope(owner), receiver_abandonable(), \
                    observe_dispatch(lambda capture: emit_main_call_state(queue_, paused, "dispatched", capture=capture)):
                if asynchronous:
                    asyncio.run(attempt_async())
                else:
                    ua.execute_physical_attempt(request, send, extractor=_extract)
        except BaseException as exc:  # noqa: BLE001 - the test inspects the exact interruption
            close(exc)

    if neighbour in {"after-genuine-send", "beside-newer-send"}:  # an earlier call of this task attempt sent P in Max and was answered
        prior_id = ("root", 1, "call-w", "exec", "exec:round:0", 1)
        p = SimpleNamespace(state="dispatched", attempt_id="attempt-p", physical_context=replace(
            low, profile="owner_max", rendered_mode="max", mode_cause="", round_id="exec:round:0"))
        emit_main_call_state(queue_, prior_id, "started")
        emit_main_call_state(queue_, prior_id, "dispatched", capture=p)
        emit_main_call_state(queue_, prior_id, "finished", capture=SimpleNamespace(**{**vars(p), "state": "settled"}))
        assert census()["ledger_state"] == "settled"
    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert sent.wait(10)
        owner_pause.install_fence(tmp_path, "root", request_id="pause")
        assert closed.wait(10)
        assert getattr(outcome.get("error"), "receiver_abandoned", False)
        held = census()
        assert (held["ledger_state"], held["applied"], held["llm_call_id"]) == ("dispatched", "low", "call-x")
        if neighbour == "after-genuine-send":
            assert (held["previous"]["attempt_id"], held["previous"]["applied"]) == ("attempt-p", "max")
        record = registry.get("root") if direct else None
        assert (record.active_llm_call if direct else meta.get("active_llm_call")) is None, "its call is closed"
        expected = {**held, "ledger_state": "released"}  # the same send, its time and its one previous kept
        if neighbour == "beside-newer-send":  # B is provisional; A is later released after B is released
            newer_id = ("root", 1, "call-y", "exec", "exec:round:2", 1)
            emit_main_call_state(queue_, newer_id, "started")
            b = SimpleNamespace(state="dispatched", attempt_id="attempt-b", physical_context=replace(
                low, profile="owner_max", rendered_mode="max", mode_cause="", round_id="exec:round:2"))
            emit_main_call_state(queue_, newer_id, "dispatched", capture=b)
            assert census()["previous"]["attempt_id"] == held["attempt_id"], "the next attempt not proven unsent is A"
            b_held = census()
            b_release = {**b_held, "ledger_state": "released"}
            queue_.put({"type": "main_llm_call_state", "task_id": "root", "task_attempt": 1,
                        "llm_call_id": "call-y", "execution_id": "exec", "round_id": "exec:round:2",
                        "call_attempt": 1, "phase": "dispatched", "context_attempt": b_release})
            after_b_release = census()
            assert after_b_release["ledger_state"] == "released"
            assert after_b_release["previous"]["attempt_id"] == held["attempt_id"], "A is still not proven unsent"
            # A's late proof arrives after B was released: it retracts A alone and P takes its place.
            queue_.put({"type": "main_llm_call_state", "task_id": "root", "task_attempt": 1,
                        "llm_call_id": "call-x", "execution_id": "exec", "round_id": "exec:round:1",
                        "call_attempt": 1, "phase": "dispatched",
                        "context_attempt": {**held, "ledger_state": "released"}})
            after_a_release = census()
            assert after_a_release == {**after_b_release, "previous": {
                key: value for key, value in held["previous"].items() if key != "previous"}}
            assert after_a_release["previous"]["attempt_id"] == "attempt-p"
            expected = after_a_release
        # Forged releases of the same attempt under another call or task attempt change nothing.
        for forged in ({"call_attempt": 2}, {"llm_call_id": "call-z"}, {"task_attempt": 2}):
            frame = {"type": "main_llm_call_state", "task_id": "root", "task_attempt": 1, "llm_call_id": "call-x",
                     "execution_id": "exec", "round_id": "exec:round:1", "call_attempt": 1, "phase": "dispatched",
                     "context_attempt": {**held, "ledger_state": "released"}, **forged}
            before = census()
            queue_.put(frame)
            assert census() == before, forged
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive()
    assert _until(lambda: attempt_rows_in_start_order(tmp_path)[0]["state"] == "released")
    assert len(attempt_rows_in_start_order(tmp_path)) == 1, "one physical send, released once"
    assert _until(lambda: census() == expected), (census(), expected)
    registry.clear()


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("ending", ["a-settled", "a-unresolved", "a-unresolved-released", "a-released", "b-settled", "a-settled-first", "a-released-first"])
@pytest.mark.parametrize("direct", [False, True], ids=["pooled", "direct"])
def test_the_newest_request_not_proven_unsent_stands_in_the_census(tmp_path, scope, monkeypatch, direct, ending, asynchronous):
    """P settled in Max, A dispatched in Low and abandoned by Pause, B dispatched in Max and then released: the capsule's
    fact is the newest request not proven unsent, whichever way A's real sender ends and in whichever order the
    proofs arrive. A's outcome arrives through the real ledger and its own sender's observer, after its call closed."""
    from ouroboros.llm_observability import emit_main_call_state
    from ouroboros.physical_attempt_context import observe_dispatch
    from ouroboros.transport_custody import ProviderNotDispatched

    bench = _census_bench(tmp_path, monkeypatch, direct)
    census, queue_ = bench.census, bench.queue
    late = {"a-settled": "settled", "a-settled-first": "settled", "b-settled": "settled", "a-unresolved": "unresolved", "a-unresolved-released": "unresolved",
            "a-released": "released", "a-released-first": "released"}[ending]
    a_call = ("root", 1, "call-a", "exec", "exec:round:1", 1)
    sent, release, outcome, closed = threading.Event(), threading.Event(), {}, threading.Event()

    def conclude():
        if late == "released":
            raise ProviderNotDispatched("the transport proves generation never started")
        if late == "unresolved":
            raise RuntimeError("provider reset after the receiver left")
        return {"late": "answer"}

    def send():
        sent.set()
        assert release.wait(10)
        return conclude()

    async def send_async():
        sent.set()
        await asyncio.to_thread(release.wait, 10)
        return conclude()

    def close(exc):
        outcome["error"] = exc
        emit_main_call_state(queue_, a_call, "failed", capture=ua.physical_attempt_capture_from_exception(exc))
        closed.set()

    request = ua.AttemptRequest(model="m", provider="test", reservation_usd=.2, physical_context=_low_context())

    def prepared(reservation):
        outcome["reservation"] = reservation

    async def attempt_async():
        try:
            await ua.execute_physical_attempt_async(request, send_async, extractor=_extract, before_dispatch=prepared)
        except BaseException as exc:  # noqa: BLE001 - the test inspects the exact interruption
            close(exc)
            if (sender := getattr(exc, "model_sender_future", None)) is not None:
                await asyncio.wait_for(sender, 10)

    def run():
        emit_main_call_state(queue_, a_call, "started")
        try:
            with ua.usage_scope(scope), model_wait.operation_wait_scope(bench.owner), receiver_abandonable(), \
                    observe_dispatch(lambda capture: emit_main_call_state(queue_, a_call, "dispatched", capture=capture)):
                if asynchronous:
                    asyncio.run(attempt_async())
                else:
                    ua.execute_physical_attempt(request, send, extractor=_extract, before_dispatch=prepared)
        except BaseException as exc:  # noqa: BLE001 - the test inspects the exact interruption
            close(exc)

    p_call, b_call = ("root", 1, "call-p", "exec", "exec:round:0", 1), ("root", 1, "call-b", "exec", "exec:round:2", 1)
    p = SimpleNamespace(state="dispatched", attempt_id="attempt-p", physical_context=_low_context(
        profile="owner_max", rendered_mode="max", mode_cause="", round_id="exec:round:0"))
    b = SimpleNamespace(state="dispatched", attempt_id="attempt-b", physical_context=_low_context(
        profile="owner_max", rendered_mode="max", mode_cause="", round_id="exec:round:2"))
    emit_main_call_state(queue_, p_call, "started")
    emit_main_call_state(queue_, p_call, "dispatched", capture=p)
    emit_main_call_state(queue_, p_call, "finished", capture=SimpleNamespace(**{**vars(p), "state": "settled"}))
    thread = threading.Thread(target=run)
    thread.start()

    def a_state():
        return attempt_rows_in_start_order(tmp_path)[0]["state"]

    def finish_a():
        release.set()
        assert _until(lambda: a_state() == late), a_state()

    try:
        assert sent.wait(10)
        owner_pause.install_fence(tmp_path, "root", request_id="pause")
        assert closed.wait(10) and getattr(outcome.get("error"), "receiver_abandoned", False)
        held_a = census()
        assert (held_a["ledger_state"], held_a["applied"], held_a["previous"]["attempt_id"]) == ("dispatched", "low", "attempt-p")
        emit_main_call_state(queue_, b_call, "started")
        emit_main_call_state(queue_, b_call, "dispatched", capture=b)
        held_b = census()
        assert (held_b["attempt_id"], held_b["ledger_state"]) == ("attempt-b", "dispatched")
        if ending.endswith("-first"):
            finish_a()
        if ending == "b-settled":
            emit_main_call_state(queue_, b_call, "finished", capture=SimpleNamespace(**{**vars(b), "state": "settled"}))
        else:
            emit_main_call_state(queue_, b_call, "dispatched", capture=SimpleNamespace(**{**vars(b), "state": "released"}))
        if not ending.endswith("-first"):
            finish_a()

        def settled_view():
            fact = census()
            if ending == "b-settled":
                return (fact["attempt_id"], fact["ledger_state"], "previous" in fact) == ("attempt-b", "settled", False)
            if fact["ledger_state"] != "released" or fact["attempt_id"] != "attempt-b":
                return False
            previous = fact.get("previous") or {}
            if late == "released":
                return (previous.get("attempt_id"), previous.get("applied")) == ("attempt-p", "max")
            return (previous.get("attempt_id"), previous.get("ledger_state"), previous.get("applied")) == (
                held_a["attempt_id"], late, "low")

        assert _until(settled_view), census()
        fact = census()
        assert "previous" not in fact.get("previous", {}), "one level, never a chain"
        assert held_b["previous"]["attempt_id"] == held_a["attempt_id"], "dispatched attempts do not prune each other"
        if ending == "a-unresolved-released":
            # The real ledger permits a later unsent proof to replace uncertainty. An unresolved
            # capture is not a pruning barrier: after releasing A the genuine P must still be there.
            with ua.usage_scope(scope), observe_dispatch(
                    lambda capture: emit_main_call_state(queue_, a_call, "dispatched", capture=capture)):
                ua.release_attempt(outcome["reservation"], "before_dispatch_failed:late_proof", proven_unsent=True)
                ua._record_attempt_capture(outcome["reservation"], request, "released")
            assert a_state() == "released"
            assert census().get("previous", {}).get("attempt_id") == "attempt-p", census()
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive()
    bench.registry.clear()


def test_the_attempt_fold_keeps_only_what_can_still_be_the_answer():
    """The private newest-first projection behind the public fact: only a sender-settled capture ends what lies behind it;
    dispatched/unresolved attempts keep their fallback, a release is terminal and states only move forward."""
    from ouroboros.physical_attempt_context import attempt_view, fold_attempt

    def frame(attempt, state, call="c", **changes):
        return {"attempt_id": attempt, "ledger_state": state, "applied": "max", "observed_at": f"t-{attempt}", "task_attempt": 1,
                "llm_call_id": call, "execution_id": "e", "round_id": "r", "call_attempt": 1, **changes}

    held = fold_attempt([], frame("p", "settled", "c0"), active=True)
    for attempt, call in (("a", "c1"), ("b", "c2")):
        held = fold_attempt(held, frame(attempt, "dispatched", call), active=True)
    assert [e["attempt_id"] for e in held] == ["b", "a", "p"]
    # Dispatched neighbours stay; the public fact names the next one not proven unsent.
    assert attempt_view(held)["previous"]["attempt_id"] == "a"
    unresolved = fold_attempt(held, frame("a", "unresolved", "c1"), active=False)
    assert [e["attempt_id"] for e in unresolved] == ["b", "a", "p"]
    released_unknown = fold_attempt(unresolved, frame("a", "released", "c1"), active=False)
    assert attempt_view(released_unknown)["previous"]["attempt_id"] == "p"
    # A confirmed newest ends the fallback behind it.
    settled = fold_attempt(held, frame("b", "settled", "c2"), active=True)
    assert [e["attempt_id"] for e in settled] == ["b"] and "previous" not in attempt_view(settled)
    # Unknown older attempts and other calls cannot insert or change anything.
    assert fold_attempt(held, frame("late", "settled", "c1"), active=False) is None
    assert fold_attempt(held, frame("a", "settled", "c-other"), active=False) is None
    # A stale state never moves one backward; a settled send is never retracted; a release is terminal.
    a_settled = fold_attempt(held, frame("a", "settled", "c1"), active=False)
    assert fold_attempt(a_settled, frame("a", "dispatched", "c1"), active=False) is None
    assert fold_attempt(a_settled, frame("a", "released", "c1"), active=False) is None
    released = fold_attempt(held, frame("a", "released", "c1"), active=False)
    assert [e["attempt_id"] for e in released] == ["b", "p"], "a disproven middle attempt is not kept for another call"
    assert attempt_view(released)["previous"]["attempt_id"] == "p"
    # Beside its own call, a released attempt stays so a stale frame of that call cannot revive it.
    first = fold_attempt([], frame("x", "dispatched"), active=True)
    first = fold_attempt(first, frame("x", "released"), active=True)
    repaired = fold_attempt(first, frame("y", "dispatched"), active=True)
    assert [e["ledger_state"] for e in repaired] == ["dispatched", "released"]
    assert fold_attempt(repaired, frame("x", "dispatched"), active=True) is None
    # The time of a request is its first observation, and another task attempt's record is not inherited.
    live = fold_attempt([], frame("x", "dispatched"), active=True)
    assert fold_attempt(live, frame("x", "settled", observed_at="later"), active=True)[0]["observed_at"] == "t-x"
    assert [e["attempt_id"] for e in fold_attempt(released, frame("n", "dispatched", task_attempt=2), active=True)] == ["n"]
    # More than two overlaps, every release order: no cap may silently erase a possible fallback.
    from itertools import permutations

    for order in permutations("abc"):
        entries = fold_attempt([], frame("p", "settled", "c0"), active=True)
        for attempt in "abc":
            entries = fold_attempt(entries, frame(attempt, "dispatched", attempt), active=True)
        remaining = list("cba")
        for attempt in order:
            entries = fold_attempt(entries, frame(attempt, "released", attempt), active=False)
            remaining.remove(attempt)
            fact = attempt_view(entries)
            shown = fact.get("previous") if fact["ledger_state"] == "released" else fact
            assert shown["attempt_id"] == (remaining[0] if remaining else "p")
