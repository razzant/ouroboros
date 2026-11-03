"""Cancellation after receipt keeps the response in CAS without resuming execution."""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from ouroboros import usage_accounting as ua
from ouroboros import usage_ledger as ledger
from ouroboros.observability import read_call_payload
from tests.test_physical_candidate_capture import (
    LLMClient, _Response, _rows, _scope, _target, data_root as data_root,
)
from tests.test_llm_claudexor import setup as setup

pytestmark = pytest.mark.serial
PHASE_HANG_SEC = 30  # only a hang bound: the 3 s it replaced timed cold reservation and fsync (#1687)


async def _reach(phase, task):
    """Await a product phase (settlement, retention) while watching its task: ordering, not
    latency. A task that ends first raises its own failure instead of a missed deadline."""
    deadline = time.monotonic() + PHASE_HANG_SEC
    while not phase.is_set():
        if task.done():
            task.result()
            pytest.fail(f"{task!r} ended before reaching its phase")
        assert time.monotonic() < deadline, "the phase was never reached"
        await asyncio.sleep(.01)


@pytest.mark.parametrize("terminal", ["settled", "unresolved", "dispatched"])
@pytest.mark.parametrize("deadline", [False, True])
def test_real_api_consumer_cancel_retains_response_capture_and_claim(data_root, monkeypatch, terminal, deadline):
    client, response = LLMClient(api_key="unused"), _Response(text="exact paid answer 🐍\r\n")
    entered, release, retaining, retained = [threading.Event() for _ in range(4)]
    original = ua.settle_attempt
    sends, captures = [], []
    from ouroboros import llm_observability
    original_retain = llm_observability.retain_cancelled_response

    def settle(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        if terminal != "settled":
            raise ledger.UsageLockUnavailable("controlled settlement refusal", reason="contention")
        return original(*args, **kwargs)

    def retain(*args):
        retaining.set()
        assert retained.wait(5)
        return original_retain(*args)

    monkeypatch.setattr(ua, "settle_attempt", settle)
    monkeypatch.setattr(llm_observability, "retain_cancelled_response", retain)
    if terminal == "dispatched":
        def refuse(*args):
            raise ledger.UsageLockUnavailable("controlled unresolved refusal", reason="contention")
        monkeypatch.setattr(ua, "mark_unresolved", refuse)

    async def send(**candidate):
        sends.append(candidate)
        return response

    async def operation():
        try:
            await client._create_chat_completion_with_retries_async(
                send, {"model": "gpt-5.2", "messages": [{"role": "user", "content": "once"}]}, _target())
            pytest.fail("cancelled owner continued to execute")
        except asyncio.CancelledError as exc:
            if not entered.is_set():
                raise  # cancelled before the response phase (a failed run's teardown): no capture yet
            capture = ua.last_physical_attempt_capture()
            assert capture is exc.physical_attempt_capture
            assert capture.state == terminal
            assert exc.response is response
            assert exc.response_manifest_ref
            assert not hasattr(exc, "response_retention_error")
            assert ua._PHYSICAL_LIMIT.get().used == 1
            captures.append(capture)
            raise

    async def run():
        with ua.usage_scope(_scope(data_root, "task-async")), ua.physical_attempt_limit(1):
            task = asyncio.create_task(operation())
            try:
                await _reach(entered, task)
                waiter = asyncio.create_task(asyncio.wait_for(task, .01)) if deadline else None
                if not deadline:
                    task.cancel()
                await asyncio.sleep(.03)
                task.cancel()  # Joining accounting survives repeated cancellation.
                assert not task.done()
                release.set()
                await _reach(retaining, task)
                task.cancel()  # Retention itself is joined, not abandoned.
                await asyncio.sleep(.01)
                assert not task.done()
                retained.set()
                with pytest.raises(asyncio.TimeoutError if deadline else asyncio.CancelledError):
                    await (waiter or task)
            finally:  # open the gates before loop shutdown cancels and joins a still-running task
                release.set()
                retained.set()
    asyncio.run(run())
    assert len(sends) == len(captures) == 1
    capture = captures[0]
    manifest, payload, _ = read_call_payload(data_root, task_id="task-async",
                                             call_id=f"physical_{capture.attempt_id}_response")
    assert manifest["control_reason"] == "caller_cancelled"
    assert manifest["full_payload_redacted"] is False
    assert payload["response"] == response.model_dump()
    assert payload["physical_attempt_capture"]["state"] == terminal
    assert _rows(data_root)[-1]["state"] == terminal
    expected = 0.0 if terminal == "settled" else .01
    assert ua.usage_projection(data_root)["accounted_usd"] == expected


def test_subscription_consumer_cancel_keeps_wire_receipt_and_stops_before_finish(setup, monkeypatch):
    from tests.test_llm_claudexor import MODEL, ledger as subscription_rows, retained, result
    from ouroboros import llm_claudexor

    root, gateway, client = setup
    entered, release = threading.Event(), threading.Event()
    original = ua.settle_attempt
    def settle(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(ua, "settle_attempt", settle)
    monkeypatch.setattr(llm_claudexor._ModelInvocation, "finish", lambda *a: pytest.fail("execution after cancellation"))
    async def receive():
        try:
            return await client.chat_async([], MODEL)
        except asyncio.CancelledError as error:
            if not entered.is_set():
                raise  # cancelled before the response phase (a failed run's teardown): no capture yet
            assert error.physical_attempt_capture.state == "settled"
            assert ua.last_physical_attempt_capture() is error.physical_attempt_capture
            assert error.response_manifest_ref
            raise
    async def run():
        operation = asyncio.create_task(receive())
        try:
            await _reach(entered, operation)
            operation.cancel()
            await asyncio.sleep(.01)
            operation.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await operation
        finally:  # open the gate before loop shutdown cancels and joins a still-running task
            release.set()
    asyncio.run(run())
    assert len(gateway.creates) == 1
    assert retained(root) == result()
    assert subscription_rows(root)[-1]["state"] == "settled"
    assert gateway.closed == 1
