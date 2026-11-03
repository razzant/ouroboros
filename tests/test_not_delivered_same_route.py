"""The engine's not-delivered receipt repeats on its own route before configured routes are tried.

The receipt proves the complete request never reached the provider and generation never started ($0, released
custody). It takes the ordinary same-request bound that already repeats an overloaded route (``max_retries``
attempts, pauses 4 s then 8 s) before the configured-route walk and the free wait; never a cooldown, never a
second burst inside an open wait episode. Every other released failure keeps its one attempt: a host connect
failure (configured routes first, #1409 case 4), an engine verdict (an undated pool), a held paid repeat.
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
from ouroboros.loop import run_llm_loop
from ouroboros.loop_llm_call import (RETRY_WALL_EXHAUSTED_KEY, TRANSPORT_DEATHS_KEY, TRANSPORT_WAIT_REDIAL_KEY,
                                     call_llm_with_retry, classify_llm_exception)
from ouroboros.tools.registry import ToolRegistry
from tests.test_loop_transport_wait import _counting_chain, _loop_kwargs, _read_network_wait_events, _typed_transport_exc


def _receipt(code: str = "transport_not_delivered", basis: str = "incomplete_upload") -> ClaudexorModelNotDispatched:
    """The engine's terminal not-started receipt as the transport raises it, with its released capture."""
    context = ({"stage": "fetch", "errorCode": "EPIPE", "generationStarted": False,
                "requestDelivery": {"state": "not_delivered", "basis": basis, "bodyBytes": 4_910_776,
                                    "handedOffBytes": 397_312, "responseReceived": False}}
               if code == "transport_not_delivered" else {})
    error = ClaudexorModelNotDispatched({"code": code, "message": "Controlled engine receipt.", "retryable": True,
                                         "context": context}, model_role="main")
    error.physical_attempt_capture = ua.PhysicalAttemptCapture(
        attempt_id="pa-nd", model="claudexor::codex=gpt-6-astra", provider="claudexor", state="released",
        candidate_measurement_kind="opaque")
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


@pytest.mark.parametrize("held", ["episode_redial", "paid_repeat_record"])
def test_an_open_wait_or_a_held_repeat_keeps_one_attempt(tmp_path, slept, held):
    usage: dict = {"execution_id": "ex"}
    if held == "episode_redial":
        usage[TRANSPORT_WAIT_REDIAL_KEY] = True
    else:
        usage[TRANSPORT_DEATHS_KEY] = {"round_id": "ex:round:1", "count": 1, "backoff_sec": 4.0}
    llm = _ScriptedLLM(fail_times=99)
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


def test_a_recovered_repeat_never_dials_the_configured_route(tmp_path, monkeypatch, loop_env, slept):
    chain, chain_calls = _counting_chain(answer_on=1)
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", chain)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "other/model")
    llm = _ScriptedLLM(fail_times=2)
    result, _usage, _trace = run_llm_loop(
        **_loop_kwargs(tmp_path, ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), [], llm=llm))
    assert result == "done"
    assert llm.calls == 3 and slept == [4.0, 8.0]
    assert chain_calls["n"] == 0  # without the repeat the first receipt went straight to the paid route
    assert _read_network_wait_events(tmp_path) == []


def test_a_spent_burst_walks_the_configured_route(tmp_path, monkeypatch, loop_env, slept):
    # No cooldown follows: `transport_unavailable` stays outside `_COOLDOWN_ERROR_KINDS` (test_loop_transport_wait).
    chain, chain_calls = _counting_chain(answer_on=1)
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", chain)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "other/model")
    llm = _ScriptedLLM(fail_times=99)
    result, _usage, _trace = run_llm_loop(
        **_loop_kwargs(tmp_path, ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), [], llm=llm))
    assert result == "fallback-ok"
    assert llm.calls == 3 and chain_calls["n"] == 1


def test_an_episode_redial_is_one_attempt_after_the_burst(tmp_path, loop_env, slept):
    llm = _ScriptedLLM(fail_times=4)
    result, _usage, _trace = run_llm_loop(
        **_loop_kwargs(tmp_path, ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), [], llm=llm))
    assert result == "done"
    # The round's burst (three sends, 4 s and 8 s), then the free wait: each redial is one send.
    assert slept == [4.0, 8.0]
    assert llm.calls == 5 and len(loop_env) == 2
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
