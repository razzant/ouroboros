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

pytestmark = pytest.mark.serial


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
        assert release.wait(10)
        return answer

    outcome: dict = {}
    thread, owner = _call(tmp_path, scope, send, outcome)
    try:
        assert sent.wait(5)
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


@pytest.mark.parametrize("abandonable", [False, True])
@pytest.mark.parametrize("timeout_type", [TimeoutError, FutureTimeout])
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


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("retention_fails", [False, True])
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
        assert release.wait(10)
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
            assert await asyncio.to_thread(sent.wait, 5)
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
        assert release.wait(10)
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
            assert await asyncio.to_thread(entered.wait, 5)
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
