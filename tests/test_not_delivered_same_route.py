"""The engine's not-delivered receipt repeats on its own route before configured routes are tried.

The receipt proves the complete request never reached the provider and generation never started ($0, released
custody). It takes the ordinary same-request bound that already repeats an overloaded route (``max_retries``
attempts, pauses 4 s then 8 s) before the configured-route walk and the free wait; never a cooldown, never a
second burst inside an open wait episode. Typed Claudexor daemon-hop prepare failures share this bound.
Direct-API connects, undated pool refusals, local slot refusals and held paid repeats keep one attempt.
"""

from __future__ import annotations

import json
import time

import httpx
import pytest

import ouroboros.loop as loop_mod
import ouroboros.loop_transport as loop_transport
from ouroboros import fallback_cooldown
from ouroboros import usage_accounting as ua
from ouroboros.llm_claudexor import ClaudexorModelNotDispatched
from ouroboros.model_concurrency import ModelSlotTimeout
from ouroboros.loop import run_llm_loop
from ouroboros.loop_llm_call import (RETRY_WALL_EXHAUSTED_KEY, TRANSPORT_DEATHS_KEY, TRANSPORT_WAIT_REDIAL_KEY,
                                     call_llm_with_retry, classify_llm_exception)
from ouroboros.tools.registry import ToolRegistry
from tests.test_loop_transport_wait import _counting_chain, _loop_kwargs, _read_network_wait_events, _typed_transport_exc


def _receipt(code: str = "transport_not_delivered", basis: str = "incomplete_upload", *,
             operation_id: str = "op-nd", provider: str = "claudexor") -> ClaudexorModelNotDispatched:
    """The engine's terminal not-started receipt as the transport raises it, with its released capture."""
    context = ({"stage": "fetch", "errorCode": "EPIPE", "generationStarted": False,
                "requestDelivery": {"state": "not_delivered", "basis": basis, "bodyBytes": 4_910_776,
                                    "handedOffBytes": 397_312, "responseReceived": False}}
               if code == "transport_not_delivered" else {})
    error = ClaudexorModelNotDispatched({"code": code, "message": "Controlled engine receipt.", "retryable": True,
                                         "context": context}, model_role="main", operation_id=operation_id)
    error.physical_attempt_capture = ua.PhysicalAttemptCapture(
        attempt_id="pa-nd", model="claudexor::codex=gpt-6-astra", provider=provider, state="released",
        candidate_measurement_kind="opaque")
    return error


def _prepare_failure(code="daemon_unreachable"):
    return _receipt(code=code, operation_id="")


def _untyped_daemon_failure():
    error = _typed_transport_exc(provider="claudexor")
    error.code, error.operation_id = "daemon_unreachable", ""
    return error


class _ScriptedLLM:
    """Fails ``fail_times`` sends with ``failure()``, then answers."""

    def __init__(self, fail_times: int, failure=_receipt):
        self.fail_times, self.failure, self.calls = fail_times, failure, 0

    def default_model(self):
        return "test-model"

    def chat(self, **_kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.failure()
        return {"role": "assistant", "content": "done"}, {"prompt_tokens": 1, "completion_tokens": 1}


@pytest.fixture
def slept(monkeypatch):
    """The same-request backoff's sleeps (``loop_llm_call`` sleeps through the shared ``time`` module)."""
    sleeps: list = []
    monkeypatch.setattr(time, "sleep", lambda sec: sleeps.append(sec))
    return sleeps


def _call(llm, tmp_path, usage, *, round_idx: int = 1):
    return call_llm_with_retry(llm, [{"role": "user", "content": "hi"}], "test-model", None, "low", 3,
                               tmp_path, "t-nd", round_idx, None, usage)


def test_receipt_classifies_as_the_released_transport_kind():
    classified = classify_llm_exception(_receipt())
    assert (classified.kind, classified.retry_same_request, classified.provider_code) == (
        "transport_unavailable", True, "transport_not_delivered")


@pytest.mark.parametrize("basis", ["incomplete_upload", "connect_failure"])
def test_not_delivered_repeats_on_its_route_within_the_overload_bound(tmp_path, slept, basis):
    llm = _ScriptedLLM(fail_times=99, failure=lambda: _receipt(basis=basis))
    usage: dict = {}
    msg, _cost = _call(llm, tmp_path, usage)
    assert msg is None
    assert llm.calls == 3 and slept == [4.0, 8.0]  # one attempt and two repeats, as overload is repeated
    assert usage["_last_llm_error_kind"] == "transport_unavailable"
    assert usage[RETRY_WALL_EXHAUSTED_KEY] is True  # now true: the route's own bound was spent
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    errors = [row for row in rows if row.get("type") == "llm_api_error"]
    assert [row["attempt"] for row in errors] == [1, 2, 3]
    assert all(row["retry_same_request"] and row["attempt_custody_state"] == "released" for row in errors)


@pytest.mark.parametrize("code", ["daemon_unreachable", "daemon_not_answering", "daemon_descriptor_changed"])
def test_typed_daemon_prepare_failure_repeats_within_same_route_bound(tmp_path, slept, code):
    llm = _ScriptedLLM(fail_times=99, failure=lambda: _prepare_failure(code))
    usage = {}
    _call(llm, tmp_path, usage)
    assert llm.calls == 3 and slept == [4.0, 8.0]
    assert usage["_last_llm_failure"]["operation_id"] == ""
    assert usage[RETRY_WALL_EXHAUSTED_KEY] is True


@pytest.mark.parametrize("failure", [
    lambda: _receipt("daemon_unreachable"),  # an accepted operation is not prepare
    lambda: _prepare_failure("arbitrary_daemon_error"),
    _untyped_daemon_failure,  # code, provider and empty operation alone are insufficient
])
def test_daemon_burst_requires_every_typed_fact(tmp_path, slept, failure):
    llm = _ScriptedLLM(fail_times=99, failure=failure)
    usage = {}
    _call(llm, tmp_path, usage)
    assert usage["_last_llm_error_kind"] == "transport_unavailable"
    assert llm.calls == 1 and slept == []


def test_local_slot_timeout_keeps_one_attempt(tmp_path, slept):
    def failure():
        error = ModelSlotTimeout("Capacity wait expired")
        error.physical_attempt_capture = _receipt().physical_attempt_capture
        return error

    llm = _ScriptedLLM(fail_times=99, failure=failure)
    usage = {}
    _call(llm, tmp_path, usage)
    assert usage["_last_llm_error_kind"] == "transport_unavailable"
    assert usage["_last_llm_failure"]["provider_code"] == "model_capacity_unavailable"
    assert llm.calls == 1 and slept == []


def test_a_repeat_that_answers_ends_the_burst(tmp_path, slept):
    llm = _ScriptedLLM(fail_times=1)
    msg, _cost = _call(llm, tmp_path, {})
    assert msg == {"role": "assistant", "content": "done"}
    assert llm.calls == 2 and slept == [4.0]


def test_the_transient_knob_does_not_widen_the_burst(tmp_path, slept, monkeypatch):
    monkeypatch.setenv("OUROBOROS_TRANSIENT_RETRY_MAX", "12")
    llm = _ScriptedLLM(fail_times=99)
    _call(llm, tmp_path, {})
    assert llm.calls == 3 and slept == [4.0, 8.0]


@pytest.mark.parametrize(("failure", "held"), [
    pytest.param(_receipt, "episode_redial", id='_receipt-episode_redial'),
    pytest.param(_prepare_failure, "paid_repeat_record", id='_prepare_failure-paid_repeat_record'),
])
def test_an_open_wait_or_a_held_repeat_keeps_one_attempt(tmp_path, slept, held, failure):
    usage: dict = {"execution_id": "ex"}
    if held == "episode_redial":
        usage[TRANSPORT_WAIT_REDIAL_KEY] = True
    else:
        usage[TRANSPORT_DEATHS_KEY] = {"round_id": "ex:round:1", "count": 1, "backoff_sec": 4.0}
    llm = _ScriptedLLM(fail_times=99, failure=failure)
    _call(llm, tmp_path, usage)
    assert llm.calls == 1 and slept == []
    assert RETRY_WALL_EXHAUSTED_KEY not in usage


@pytest.mark.parametrize("failure", [
    lambda: _receipt(code="credential_pool_exhausted"),  # an undated engine verdict: configured routes first
    _typed_transport_exc,  # a host connect failure: configured routes first (#1409 case 4)
])
def test_other_released_failures_keep_one_attempt(tmp_path, slept, failure):
    llm = _ScriptedLLM(fail_times=99, failure=failure)
    usage: dict = {}
    _call(llm, tmp_path, usage)
    assert usage["_last_llm_error_kind"] == "transport_unavailable"
    assert llm.calls == 1 and slept == []


@pytest.fixture
def loop_env(monkeypatch, slept):
    episode_sleeps: list = []
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep",
                        lambda sec, _wake: (episode_sleeps.append(sec), False)[1])
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    fallback_cooldown.reset_for_tests()
    yield episode_sleeps
    fallback_cooldown.reset_for_tests()


@pytest.mark.parametrize("failure", [_receipt, _prepare_failure])
def test_a_recovered_repeat_never_dials_the_configured_route(tmp_path, monkeypatch, loop_env, slept, failure):
    chain, chain_calls = _counting_chain(answer_on=1)
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", chain)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "other/model")
    llm = _ScriptedLLM(fail_times=2, failure=failure)
    result, _usage, _trace = run_llm_loop(
        **_loop_kwargs(tmp_path, ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), [], llm=llm))
    assert result == "done"
    assert llm.calls == 3 and loop_env == [4.0, 8.0] and slept == []  # the repeats' waits wake for owner words
    assert chain_calls["n"] == 0  # without the repeat the first receipt went straight to the paid route
    assert _read_network_wait_events(tmp_path) == []


@pytest.mark.parametrize("failure", [_receipt, _prepare_failure])
def test_a_spent_burst_walks_the_configured_route(tmp_path, monkeypatch, loop_env, slept, failure):
    # No cooldown follows: `transport_unavailable` stays outside `_COOLDOWN_ERROR_KINDS` (test_loop_transport_wait).
    chain, chain_calls = _counting_chain(answer_on=1)
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", chain)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "other/model")
    llm = _ScriptedLLM(fail_times=99, failure=failure)
    result, _usage, _trace = run_llm_loop(
        **_loop_kwargs(tmp_path, ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), [], llm=llm))
    assert result == "fallback-ok"
    assert llm.calls == 3 and chain_calls["n"] == 1


def test_an_episode_redial_is_one_attempt_after_the_burst(tmp_path, loop_env, slept):
    llm = _ScriptedLLM(fail_times=4)
    result, _usage, _trace = run_llm_loop(
        **_loop_kwargs(tmp_path, ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), [], llm=llm))
    assert result == "done"
    # The round's burst (three sends, waits of 4 s and 8 s), then the free wait: each redial is one send.
    assert slept == [] and loop_env[:2] == [4.0, 8.0]
    assert llm.calls == 5 and len(loop_env) == 4
    assert [row["phase"] for row in _read_network_wait_events(tmp_path)] == ["entered", "waiting", "waiting", "recovered"]


def test_a_host_connect_failure_keeps_configured_routes_before_any_repeat(tmp_path, monkeypatch, loop_env, slept):
    chain, chain_calls = _counting_chain(answer_on=1)
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", chain)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "other/model")
    llm = _ScriptedLLM(fail_times=99, failure=lambda: _typed_transport_exc(exc_cls=httpx.ConnectError))
    result, _usage, _trace = run_llm_loop(
        **_loop_kwargs(tmp_path, ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), [], llm=llm))
    assert result == "fallback-ok"
    assert llm.calls == 1 and chain_calls["n"] == 1 and slept == []


@pytest.mark.parametrize("words", [True, False])
def test_owner_words_during_a_granted_repeat_ride_the_next_send(tmp_path, monkeypatch, loop_env, slept, words):
    """#2996 joins a Main follow-up to the running turn. Words that arrive while its request fails undelivered end the
    repeat's wait: the round start drains them and the next send on the same route carries them. Without them the
    granted repeat re-sends the request as built, after the same 4 s."""
    from ouroboros.owner_mailbox import write_owner_message

    waits: list = []
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda sec, wake: (waits.append(sec), wake())[1])
    chain, chain_calls = _counting_chain(answer_on=1)
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", chain)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "other/model")
    sent: list = []

    class _Recording(_ScriptedLLM):
        def chat(self, **kwargs):
            sent.append(json.dumps(kwargs["messages"], ensure_ascii=False))
            if words and self.calls == 0:  # the owner writes while the first request is failing
                assert write_owner_message(tmp_path, "Any reply yet?", "t-wait", msg_id="m-follow-up")
            return super().chat(**kwargs)

    llm = _Recording(fail_times=1)
    result, _usage, _trace = run_llm_loop(
        **_loop_kwargs(tmp_path, ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), [], llm=llm))
    assert result == "done" and llm.calls == 2 and chain_calls["n"] == 0  # same route, no configured route
    assert waits == [4.0] and slept == []  # the repeat's wait, woken at once only when words are waiting
    assert "Any reply yet?" not in sent[0]
    assert ("Any reply yet?" in sent[1]) is words
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["reason_code"] for row in rows if row.get("type") == "llm_not_dispatched"] == (
        ["owner_words_waiting"] if words else [])


def test_unread_input_a_yield_already_met_wakes_no_second_repeat(tmp_path, monkeypatch, slept):
    """A row the round-start drain leaves unread (a host-carried row under a sealed acceptance fence) must not turn
    every granted repeat into an immediate rebuild; new words still end the wait."""
    from ouroboros.model_wait import OWNER_WORDS_WAITING, ModelWaitInterrupted

    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda _sec, wake: wake())
    llm, usage = _ScriptedLLM(fail_times=99), {}

    def call(unread):
        return call_llm_with_retry(llm, [{"role": "user", "content": "hi"}], "test-model", None, "low", 3,
                                   tmp_path, "t-nd", 1, None, usage, owner_input_check=lambda: unread)

    with pytest.raises(ModelWaitInterrupted) as yielded:
        call(["stuck"])
    assert yielded.value.control_reason == OWNER_WORDS_WAITING and llm.calls == 1
    assert call(["stuck"]) == (None, 0.0) and llm.calls == 4  # the same unread row: an ordinary burst
    with pytest.raises(ModelWaitInterrupted):
        call(["new", "stuck"])  # new words still end the wait
    assert llm.calls == 5 and call(["stuck"]) == (None, 0.0) and llm.calls == 8  # the met stuck row alone: a burst


@pytest.mark.parametrize("unread", [["m-1"], []])
def test_waiting_words_yield_even_when_the_deadline_refuses_the_repeat(tmp_path, monkeypatch, slept, unread):
    """Out of time for the 4 s repeat: waiting words still yield, so the round start reads them before its deadline rail
    refuses another send; without them the repeat stops as deadline-exhausted, as before."""
    from ouroboros.model_wait import ModelWaitInterrupted

    llm, usage = _ScriptedLLM(fail_times=99), {}
    try:
        call_llm_with_retry(llm, [{"role": "user", "content": "hi"}], "test-model", None, "low", 3, tmp_path, "t-nd", 1,
                            None, usage, deadline_ts=time.time() + 5, owner_input_check=lambda: unread)
        yielded = False
    except ModelWaitInterrupted:
        yielded = True
    assert (yielded, llm.calls, slept) == (bool(unread), 1, [])
    assert (RETRY_WALL_EXHAUSTED_KEY in usage) is not bool(unread)
