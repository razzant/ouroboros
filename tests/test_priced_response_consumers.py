"""Real stream accounting must not decide whether its response can be consumed."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import loop, loop_llm_call, owner_mailbox
from ouroboros.llm import LLMClient
from ouroboros.model_wait import ModelWaitInterrupted
from ouroboros.transport_custody import (
    attempt_custody_event_fields, is_retryable_transport_death, outcome_unknown_on_chain,
)
from tests._usage_store_testing import ledger_rows
from tests import test_openrouter_transport_evidence as evidence_fixtures
from tests.test_openrouter_transport_evidence import Wire, frame, run_driver, target

from tests.test_transport_death_retry import _loop_kwargs, _no_chain, _presence_turn

isolated = evidence_fixtures.isolated


class StreamConsumer:
    """Only the SDK socket is replaced; retries use real physical sends and receipts."""

    def __init__(self, cost, *, asynchronous=False, failure=httpx.ReadError, recover=False, stop=None):
        self.cost, self.asynchronous, self.failure = cost, asynchronous, failure
        self.recover, self.stop, self.calls = recover, stop, 0

    def default_model(self):
        return "vendor/fixture"

    def chat(self, **kwargs):
        self.calls += 1
        complete = self.recover and self.calls > 1
        wire = Wire([frame(generation=f"gen-priced-{self.calls}", usage={"cost": self.cost},
                           finish="stop" if complete else None)],
                    read_error=None if complete else self.failure("socket response incomplete"))
        try:
            response = run_driver(wire, asynchronous=self.asynchronous)
        except Exception:
            if self.stop:
                self.stop()
            raise
        return LLMClient()._normalize_remote_response(response.model_dump(), target())


def presence(root, monkeypatch, consumer):
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    monkeypatch.setattr(loop, "_run_cross_model_fallback_chain", _no_chain)
    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline",
                        lambda *a, **k: not k.get("wake_check", lambda: False)())
    kwargs = _presence_turn(_loop_kwargs(root, consumer, []))
    kwargs["tools"]._ctx.task_id = "t-death"
    kwargs["tools"]._ctx.task_attempt = 1
    return loop.run_llm_loop(**kwargs)


@pytest.mark.parametrize("cost", [0.0, 1.25])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("recover", [False, True])
def test_real_inline_presence_repeats_priced_incomplete_stream_with_existing_bound(
        isolated, monkeypatch, cost, asynchronous, recover):
    consumer = StreamConsumer(cost, asynchronous=asynchronous, recover=recover)
    text, usage, trace = presence(isolated, monkeypatch, consumer)
    assert consumer.calls == (2 if recover else 3)
    rows = ledger_rows(isolated)
    assert len(rows) == consumer.calls
    assert len({row["attempt_id"] for row in rows}) == consumer.calls
    assert all(row["state"] == "settled" and row["cost_usd"] == cost and row["cost_final"] for row in rows)
    events = [json.loads(line) for line in (isolated / "events.jsonl").read_text().splitlines()]
    failures = [row for row in events if row.get("type") == "llm_api_error"]
    assert [row["retry_same_request"] for row in failures] == ([True] if recover else [True, True, False])
    assert all(row["attempt_custody_state"] == "settled" and row["stream_incomplete"] for row in failures)
    if recover:
        assert text == "answer"
        assert loop_llm_call.TRANSPORT_DEATHS_KEY not in usage
    else:
        assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"
        assert trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"


@pytest.mark.parametrize("cost", [0.0, 1.25])
@pytest.mark.parametrize("stop", [False, True])
def test_real_presence_priced_timeout_or_owner_stop_sends_no_repeat(isolated, monkeypatch, cost, stop):
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN

    def request_stop():
        assert owner_mailbox.write_owner_message(isolated, REASON_OWNER_STOPPED_DIRECT_TURN, "t-death",
                                                msg_id="priced-stop", kind=owner_mailbox.KIND_FINALIZE_NOW)

    consumer = StreamConsumer(cost, failure=httpx.ReadError if stop else httpx.ReadTimeout,
                              stop=request_stop if stop else None)
    _text, usage, trace = presence(isolated, monkeypatch, consumer)
    assert consumer.calls == 1
    assert len(ledger_rows(isolated)) == 1 and ledger_rows(isolated)[0]["cost_usd"] == cost
    assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"
    assert usage["_last_llm_retry_same_request"] is False
    assert trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"


@pytest.mark.parametrize("cost", [0.0, 1.25])
def test_presence_repeat_deadline_still_excludes_priced_stream(isolated, monkeypatch, cost):
    consumer, usage = StreamConsumer(cost), {}
    monkeypatch.setattr(loop_llm_call, "_prepare_main_messages", lambda messages, **kw: messages)
    message, _ = loop_llm_call.call_llm_with_retry(
        consumer, [{"role": "user", "content": "hello"}], "vendor/fixture", [], "low", 3,
        isolated, "money-task", 1, None, usage, "presence", transport_death_retries=2,
        deadline_ts=time.time() + 32, transport_reserve_sec=30)
    assert message is None and consumer.calls == 1
    assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"
    assert usage["_last_llm_retry_same_request"] is False
    assert loop_llm_call.TRANSPORT_DEATHS_KEY not in usage
    assert ledger_rows(isolated)[0]["cost_usd"] == cost


@pytest.mark.parametrize("cost", [0.0, 1.25])
def test_priced_revoked_finalize_control_retains_actual_no_resend_rail(isolated, cost):
    from ouroboros.loop_round_limits import _RoundLimitContext, _handle_model_wait_control

    failure = httpx.ReadError("response lost after priced usage")
    with pytest.raises(httpx.ReadError):
        run_driver(Wire([frame(usage={"cost": cost})], read_error=failure))
    error = ModelWaitInterrupted("finalize_requested", cause=failure)
    assert error.physical_attempt_capture.state == "settled"
    assert owner_mailbox.write_owner_message(isolated, "Wrap up", "money-task",
                                            msg_id="revoked-priced", kind=owner_mailbox.KIND_FINALIZE_NOW)
    assert owner_mailbox.revoke_owner_control(isolated, "money-task", "revoked-priced")
    consumer = StreamConsumer(cost)
    ctx = _RoundLimitContext(messages=[], llm=consumer, active_model="vendor/fixture", active_effort="low",
        max_retries=2, drive_logs=isolated, task_id="money-task", round_idx=1, event_queue=None,
        accumulated_usage={}, task_type="task", active_use_local=False, max_rounds=None, drive_root=isolated)
    result = _handle_model_wait_control(ctx, error)
    assert result is not None and consumer.calls == 0
    assert result[2]["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"
    assert ctx.accumulated_usage["ledger_attempt_ids"] == [failure.physical_attempt_capture.attempt_id]
    assert len(ledger_rows(isolated)) == 1 and ledger_rows(isolated)[0]["cost_usd"] == cost


@pytest.mark.parametrize("cost", [None, 0.0, 1.25])
def test_search_unknown_response_guard_is_independent_of_received_price(isolated, cost):
    from ouroboros.tools.search import _provider_outcome_is_unknown

    failure = httpx.ReadError("response lost")
    with pytest.raises(httpx.ReadError):
        run_driver(Wire([frame(usage={"cost": cost})], read_error=failure))
    assert _provider_outcome_is_unknown(failure)


@pytest.mark.parametrize("provider,loopback,error_type,retryable", [
    ("openrouter", False, httpx.ReadError, True),
    ("local", False, httpx.ReadError, False),
    ("openai-compatible", True, httpx.ReadError, False),
    ("openrouter", False, httpx.ReadTimeout, False),
    ("openrouter", False, httpx.WriteTimeout, False),
    ("openrouter", False, httpx.ConnectTimeout, False),
])
def test_priced_incomplete_chain_keeps_locality_and_timeout_exclusions(provider, loopback, error_type, retryable):
    from ouroboros.usage_accounting import PhysicalAttemptCapture

    inner = error_type("socket failed")
    inner.stream_incomplete = True
    inner.physical_attempt_capture = PhysicalAttemptCapture(
        attempt_id="priced", model="fixture", provider=provider, state="settled",
        candidate_measurement_kind="opaque", route_is_loopback=loopback)
    outer = RuntimeError("wrapped")
    outer.__cause__ = inner
    assert outcome_unknown_on_chain(outer)
    assert is_retryable_transport_death(outer) is retryable
    assert attempt_custody_event_fields(outer)["stream_incomplete"] is True
    del inner.stream_incomplete
    assert not outcome_unknown_on_chain(outer)
    assert not is_retryable_transport_death(outer)


@pytest.mark.parametrize("cost", [None, 0.0, 1.25])
def test_review_preserves_response_custody_after_actual_price_settlement(isolated, cost):
    from ouroboros.review_custody import (
        _ReviewAttemptHistory, _attach_worker_exception_facts, _review_exception_projection,
        _worker_exception_operation_state, finalize_review_actor, retryable_review_exception,
    )

    failure = httpx.ReadError("response lost after priced usage")
    with pytest.raises(httpx.ReadError):
        run_driver(Wire([frame(usage={"cost": cost})], read_error=failure))
    history = _ReviewAttemptHistory()
    history.observe(failure)
    custody, state, status, operation, code = _review_exception_projection(failure, {}, history, {})
    assert state == ("unresolved" if cost is None else "settled") and status is None
    assert operation == "custody_lost" and code == "provider_outcome_unknown"
    assert custody["stream_incomplete"] is True
    assert not retryable_review_exception(failure, None, attempt_history=history)
    assert _worker_exception_operation_state(failure, {}) == "settled"
    actor = {}
    _attach_worker_exception_facts(actor, failure)
    assert actor["usage"]["stream_incomplete"] is True
    assert actor["usage"]["physical_attempt_state"] == state
    worker_actor = SimpleNamespace(**actor, status="error", operation_state="settled")
    finalize_review_actor(worker_actor, operation_id="review-priced")
    assert worker_actor.operation_state == "custody_lost"
    assert worker_actor.failure_code == "provider_outcome_unknown"


def test_terminal_local_rejection_keeps_its_existing_provider_error_policy():
    from ouroboros.llm_stream import RejectedProviderStream
    from ouroboros.review_custody import _ReviewAttemptHistory, _attach_worker_exception_facts, finalize_review_actor
    from ouroboros.usage_accounting import PhysicalAttemptCapture

    failure = RejectedProviderStream("complete but unusable")
    failure.physical_attempt_capture = PhysicalAttemptCapture(
        attempt_id="rejected", model="fixture", provider="openrouter", state="settled",
        candidate_measurement_kind="opaque")
    history = _ReviewAttemptHistory()
    history.observe(failure)
    assert not history.unknown_outcome_seen
    assert not outcome_unknown_on_chain(failure)
    assert not outcome_unknown_on_chain(ModelWaitInterrupted("finalize_requested", cause=failure))
    assert attempt_custody_event_fields(failure)["stream_incomplete"] is True
    assert loop_llm_call.classify_llm_exception(failure).kind == "provider_error"
    actor = SimpleNamespace(status="error", operation_state="settled")
    _attach_worker_exception_facts(actor, failure)
    finalize_review_actor(actor, operation_id="review-rejected")
    assert actor.operation_state == "settled"
