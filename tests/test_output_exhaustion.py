"""Output exhaustion: an EMPTY reply that ended on its output limit before any text or tool call.

The finish reason is read once, by presence of the key (``_usage_response.response_finish_reason``):
usage first (the OpenAI-compatible, Claudexor, local and GigaChat lanes write it there), then the
message (native Anthropic). An empty reply without a body error that ended on ``length`` /
``max_tokens`` is output exhaustion: an ``llm_empty_response`` event carrying the real finish,
kind ``llm_output_exhausted``, ONE send (the same request would end the same way), no fallback
cooldown, no response-cache bypass, no spent retry wall, never ``infra_failed``. A genuinely
absent finish keeps the glitch path; body errors and overflow keep precedence.

The incident replies are rebuilt from their recorded numbers only (no private payload): usage
``response_finish_reason="length"``, every completion token a reasoning token, an empty message
with no finish key of its own.
"""

from __future__ import annotations

import copy
import json
import queue

import httpx
import pytest

import ouroboros.loop as loop_mod
import ouroboros.loop_llm_call as call_mod
import ouroboros.loop_transport as loop_transport
from ouroboros import usage_accounting as ua
from ouroboros.loop import _output_exhausted_notice, run_llm_loop
from ouroboros.tools.registry import ToolRegistry
from ouroboros._usage_response import output_exhaustion_facts, reported_reasoning_tokens, response_finish_reason
from ouroboros.loop_llm_call import (
    _COOLDOWN_ERROR_KINDS,
    _TRANSIENT_RETRY_KINDS,
    RETRY_WALL_EXHAUSTED_KEY,
    _classify_empty_response,
    call_llm_with_retry,
    provider_no_call_source,
    transient_retry_max,
)

MESSAGES = [{"role": "user", "content": "work"}]
OK_RESPONSE = ({"role": "assistant", "content": "done"}, {"prompt_tokens": 1, "completion_tokens": 1})
# The six recorded incident replies: completion tokens, all of them reasoning.
INCIDENT_COMPLETIONS = (216, 188, 160, 132, 105, 77)


def _incident_reply(ordinal: int):
    tokens = INCIDENT_COMPLETIONS[ordinal]
    return (
        {"role": "assistant", "content": "", "tool_calls": [], "reasoning": "r" * 400},
        {"prompt_tokens": 70_900 + 41 * ordinal, "completion_tokens": tokens,
         "completion_tokens_details": {"audio_tokens": 0, "image_tokens": 0, "reasoning_tokens": tokens},
         "response_finish_reason": "length", "response_provider": "upstream-a",
         "provider": "openrouter", "resolved_model": "openai/gpt-5.5", "cost": 0.002, "cost_final": True},
    )


def _receipt(sent: int, provider: str = "openrouter") -> ua.PhysicalAttemptCapture:
    """The settled physical receipt of the attempt that produced the reply (its sent cap)."""
    return ua.PhysicalAttemptCapture(attempt_id=f"pa-{sent}", model="test-model", provider=provider,
                                     state="settled", candidate_measurement_kind="opaque",
                                     max_completion_tokens=sent)


class _ScriptedLLM:
    """chat() returns the scripted replies in order (then OK), adopting a receipt when given one."""

    def __init__(self, *script, receipts=()):
        self.script, self.receipts, self.requests = list(script), list(receipts), []

    @property
    def calls(self):
        return len(self.requests)

    def default_model(self):
        return "test-model"

    def chat(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        if self.receipts:
            ua.adopt_physical_attempt_capture(self.receipts.pop(0))
        step = self.script.pop(0) if self.script else OK_RESPONSE
        if callable(step):
            raise step()
        return copy.deepcopy(step)


@pytest.fixture(autouse=True)
def _fresh_receipt():
    """No earlier test's physical receipt may speak for these fake sends."""
    previous = ua.last_physical_attempt_capture()
    ua.adopt_physical_attempt_capture(None)
    yield
    ua.adopt_physical_attempt_capture(previous)


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(call_mod, "_sleep_within_deadline", lambda sec, _dl, **_kw: (sleeps.append(sec), True)[1])
    return sleeps


def _events(drive_logs, kind):
    path = drive_logs / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    return [row for row in rows if row.get("type") == kind]


def _call(llm, drive_logs, usage, model="openai/gpt-5.5"):
    return call_llm_with_retry(llm, MESSAGES, model, None, "high", 3, drive_logs, "t-exhaust", 1, None, usage, "task")


# ------------------------------------------------------------------ the shared reader (B1)

def test_finish_reader_reads_by_presence_and_keeps_an_explicit_null():
    assert response_finish_reason({"response_finish_reason": "length"}, {"finish_reason": "stop"}) == (True, "length")
    assert response_finish_reason({}, {"stop_reason": "max_tokens"}) == (True, "max_tokens")
    # An explicit null is a fact of its own: a lower field never replaces it.
    assert response_finish_reason({"response_finish_reason": None}, {"stop_reason": "max_tokens"}) == (True, None)
    assert response_finish_reason({}, {"finish_reason": None, "stop_reason": "end_turn"}) == (True, None)
    assert response_finish_reason({}, {}) == (False, None)
    assert response_finish_reason(None, None) == (False, None)


def test_reasoning_and_allowance_facts_name_only_what_the_attempt_recorded():
    assert reported_reasoning_tokens(_incident_reply(0)[1]) == 216
    assert reported_reasoning_tokens({"output_tokens_details": {"reasoning_tokens": 9}}) == 9
    assert reported_reasoning_tokens({"reasoning_tokens": 0}) == 0  # a reported zero is a fact
    assert reported_reasoning_tokens({"reasoning_tokens": None, "completion_tokens": 50}) is None
    assert reported_reasoning_tokens({"reasoning_tokens": True}) is None
    assert output_exhaustion_facts(_incident_reply(0)[1], 216) == {"sent_allowance_tokens": 216, "reasoning_tokens": 216}
    assert output_exhaustion_facts({"stop_reason_only": True}, None) == {"sent_allowance_tokens": None, "reasoning_tokens": None}
    # Claudexor sends no output cap: its receipt carries a reservation, which is no allowance.
    claudexor = {"reasoning_tokens": 30_000, "claudexor": {"output_cap_applied": False, "output_reserve_tokens": 65_536}}
    assert output_exhaustion_facts(claudexor, 65_536) == {"sent_allowance_tokens": None, "reasoning_tokens": 30_000}


# ------------------------------------------------------------------ classification (T9)

@pytest.mark.parametrize(("usage", "msg"), [
    (_incident_reply(0)[1], _incident_reply(0)[0]),                               # usage `length` (OpenAI family)
    ({"prompt_tokens": 9, "completion_tokens": 4096}, {"content": "", "stop_reason": "max_tokens"}),  # Anthropic
])
def test_empty_reply_on_the_output_limit_is_exhaustion(usage, msg):
    assert _classify_empty_response(usage, msg) == ("llm_empty_response", False, True)


def test_body_errors_and_overflow_keep_precedence_over_the_output_limit():
    length = {"response_finish_reason": "length"}
    assert _classify_empty_response({**length, "provider_error": {"kind": "rate_limit", "code": 429}}, {}) == (
        "llm_empty_response", False, False)  # a transient body error still retries, as itself
    assert _classify_empty_response({**length, "provider_error": {"kind": "provider_error", "code": 401}}, {}) == (
        "provider_body_error", False, True)
    assert _classify_empty_response({"response_finish_reason": "model_context_window_exceeded"}, {}) == (
        "remote_context_overflow", False, True)
    # A missing or explicit-null finish stays the glitch; a plain "stop" is an ordinary empty reply.
    assert _classify_empty_response({}, {}) == ("provider_incomplete_response", True, False)
    assert _classify_empty_response({"response_finish_reason": None}, {"stop_reason": "max_tokens"}) == (
        "provider_incomplete_response", True, False)
    assert _classify_empty_response({"response_finish_reason": "stop"}, {}) == ("llm_empty_response", False, False)


def test_incident_reply_is_sent_once_and_stamped_as_exhaustion(tmp_path, no_sleep):
    """The six recorded replies are no longer six same-request attempts: the first one ends
    the call, as output exhaustion — no cooldown kind, no bypass, no spent wall, not infra."""
    llm = _ScriptedLLM(*[_incident_reply(index) for index in range(6)], receipts=[_receipt(216)])
    usage = {}
    msg, cost = _call(llm, tmp_path, usage)

    assert msg is None and cost == pytest.approx(0.002)
    assert llm.calls == 1 and no_sleep == []
    assert [call["bypass_response_cache"] for call in llm.requests] == [False]
    rows = _events(tmp_path, "llm_empty_response")
    assert [(row["finish_reason"], row["attempt"]) for row in rows] == [("length", 1)]
    assert _events(tmp_path, "provider_incomplete_response") == []
    assert _events(tmp_path, "llm_retry_deadline_exhausted") == []
    kind = usage["_last_llm_error_kind"]
    assert kind == "llm_output_exhausted"
    assert kind not in _TRANSIENT_RETRY_KINDS and kind not in _COOLDOWN_ERROR_KINDS
    assert usage["execution_status"] == "failed" and usage["reason_code"] == "llm_empty_response"
    assert usage["_last_llm_call_meta"]["failure_code"] == "llm_output_exhausted"
    assert RETRY_WALL_EXHAUSTED_KEY not in usage
    assert provider_no_call_source(usage, False) == ("", False)  # a forced call stays possible
    assert usage["_last_llm_output_exhausted"] == {"sent_allowance_tokens": 216, "reasoning_tokens": 216}
    assert "output limit" in usage["_last_llm_error"]


def test_anthropic_max_tokens_exhaustion_reports_no_invented_reasoning(tmp_path, no_sleep):
    reply = ({"role": "assistant", "content": "", "stop_reason": "max_tokens"},
             {"prompt_tokens": 9, "completion_tokens": 4096, "provider": "anthropic"})
    llm = _ScriptedLLM(reply, reply, receipts=[_receipt(4096, "anthropic")])
    usage = {}
    assert _call(llm, tmp_path, usage, model="anthropic::claude-sonnet")[0] is None
    assert llm.calls == 1
    assert [row["finish_reason"] for row in _events(tmp_path, "llm_empty_response")] == ["max_tokens"]
    assert usage["_last_llm_error_kind"] == "llm_output_exhausted"
    assert usage["_last_llm_output_exhausted"] == {"sent_allowance_tokens": 4096, "reasoning_tokens": None}


def test_absent_finish_keeps_the_glitch_path_unchanged(tmp_path, no_sleep):
    """Quiet side: no finish fact anywhere is still the transient glitch — every attempt of the
    budget, a fresh-response request on each retry, the cooldown kind and a spent wall."""
    budget = transient_retry_max(3)
    glitch = ({"role": "assistant", "content": "", "tool_calls": []}, {"prompt_tokens": 5, "completion_tokens": 0})
    llm = _ScriptedLLM(*([glitch] * budget))
    usage = {}
    assert _call(llm, tmp_path, usage)[0] is None

    assert llm.calls == budget and len(no_sleep) == budget - 1
    assert [call["bypass_response_cache"] for call in llm.requests] == [False] + [True] * (budget - 1)
    assert len(_events(tmp_path, "provider_incomplete_response")) == budget
    assert _events(tmp_path, "llm_empty_response") == []
    assert usage["_last_llm_error_kind"] == "provider_incomplete_response"
    assert usage["_last_llm_error_kind"] in _COOLDOWN_ERROR_KINDS
    assert usage["execution_status"] == "infra_failed"
    assert usage[RETRY_WALL_EXHAUSTED_KEY] is True
    assert "_last_llm_output_exhausted" not in usage


def test_empty_stop_reply_is_retried_as_an_ordinary_empty_response(tmp_path, no_sleep):
    """The reader fix alone: a usage "stop" is no null glitch (no cooldown, no bypass, no wall),
    and it is not exhaustion either, so its retries are unchanged."""
    empty_stop = ({"role": "assistant", "content": ""}, {"prompt_tokens": 5, "response_finish_reason": "stop"})
    llm = _ScriptedLLM(empty_stop, OK_RESPONSE)
    usage = {}
    msg, _cost = _call(llm, tmp_path, usage)

    assert msg == OK_RESPONSE[0] and llm.calls == 2
    assert [call["bypass_response_cache"] for call in llm.requests] == [False, False]
    assert [row["finish_reason"] for row in _events(tmp_path, "llm_empty_response")] == ["stop"]
    assert "_last_llm_error_kind" not in usage and "_last_llm_output_exhausted" not in usage


# ------------------------------------------------------------------ the next ordinary round (T10)

HOST_FACT = "[SYSTEM NOTICE]\nThe provider ended your previous reply on its length limit"


def _custody_failure(cause_cls, state):
    try:
        raise RuntimeError("Connection error.") from cause_cls("socket failure")
    except RuntimeError as exc:
        exc.physical_attempt_capture = ua.PhysicalAttemptCapture(
            attempt_id=f"pa-{state}", model="test-model", provider="openrouter", state=state,
            candidate_measurement_kind="opaque")
        return exc


def _death():  # a dispatched request whose socket died: its outcome stays unknown
    return _custody_failure(httpx.ReadError, "unresolved")


def _released_connect():  # a $0 connect failure before dispatch: no transport
    return _custody_failure(httpx.ConnectError, "released")


def _no_chain(**_kwargs):
    raise AssertionError("output exhaustion must not walk the configured routes")


def _quiet_chain(**kwargs):  # the configured routes answer nothing
    return None, kwargs["active_model"], kwargs["active_use_local"], kwargs["context_fit_plan"], kwargs["active_context_mode"]


def _loop_kwargs(tmp_path, llm, notes):
    return dict(messages=[{"role": "user", "content": "go"}], tools=ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path),
                llm=llm, drive_logs=tmp_path, emit_progress=lambda text, **_meta: notes.append(text),
                incoming_messages=queue.Queue(), task_id="t-exhaust-loop", drive_root=tmp_path)


def _host_facts(messages):
    return [row for row in messages if row.get("role") == "user" and str(row.get("content") or "").startswith(HOST_FACT)]


@pytest.fixture
def loop_env(monkeypatch):
    # Configured routes exist, so entering the round recovery would reach the chain.
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", _no_chain)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "other/model")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)


def test_exhaustion_adds_one_host_fact_and_the_next_ordinary_round_answers(tmp_path, loop_env, no_sleep):
    llm = _ScriptedLLM(_incident_reply(0), OK_RESPONSE, receipts=[_receipt(216)])
    result, usage, trace = run_llm_loop(**_loop_kwargs(tmp_path, llm, []))

    assert result == "done" and llm.calls == 2  # no same-request retry, no route walk, no forced call
    first, second = (request["messages"] for request in llm.requests)
    assert _host_facts(first) == []
    assert [row["content"] for row in _host_facts(second)] == [
        _output_exhausted_notice({"sent_allowance_tokens": 216, "reasoning_tokens": 216})]
    assert "allowance sent was 216 tokens" in _host_facts(second)[0]["content"]
    assert second[:len(first)] == first  # appended: the exhausted request stays a prefix of the next
    assert [(row["round"], row["finish_reason"]) for row in _events(tmp_path, "llm_empty_response")] == [(1, "length")]
    assert [row["round"] for row in _events(tmp_path, "llm_round")] == [2]  # the next ORDINARY round answered
    assert "forced_finalization" not in trace
    # The usable reply cleared every stamp the exhausted one left.
    for key in ("_last_llm_error_kind", "_last_llm_error", "_last_llm_output_exhausted",
                "execution_status", "reason_code", RETRY_WALL_EXHAUSTED_KEY):
        assert key not in usage


def test_repeated_exhaustion_runs_to_the_existing_round_limit(tmp_path, loop_env, no_sleep, monkeypatch):
    """No separate counter: every exhausted round leaves one fact and the loop goes on until an
    existing limit (here the round limit, whose own forced call reads every fact) ends it."""
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "3")
    llm = _ScriptedLLM(*[_incident_reply(index) for index in range(6)])
    result, usage, trace = run_llm_loop(**_loop_kwargs(tmp_path, llm, []))

    assert llm.calls == 4  # rounds 1-3, then the round-limit rail's forced call
    assert [len(_host_facts(request["messages"])) for request in llm.requests] == [0, 1, 2, 3]
    assert [row["round"] for row in _events(tmp_path, "llm_empty_response")] == [1, 2, 3, 4]
    assert _events(tmp_path, "provider_incomplete_response") == []
    assert usage["reason_code"] == "round_limit" and usage.get("execution_status") != "infra_failed"
    assert trace["forced_finalization"]["reason_code"] == "round_limit"
    assert result.startswith("⚠️ Task exceeded MAX_ROUNDS (3)")


def test_exhaustion_over_an_unresolved_attempt_starts_no_new_generation(tmp_path, loop_env, no_sleep):
    """The fence: a round still holding an unresolved attempt (a granted transport-death repeat)
    may start no new generation, so an exhausted repeat keeps the no-resend terminal."""
    llm = _ScriptedLLM(_death, _incident_reply(0), OK_RESPONSE)
    kwargs = _loop_kwargs(tmp_path, llm, [])
    kwargs["task_type"] = "presence"  # inline Presence alone keeps the bounded paid repeat
    kwargs["tools"]._ctx.is_direct_chat = True
    kwargs["tools"]._ctx.current_task_type = "presence"
    _result, usage, trace = run_llm_loop(**kwargs)

    assert llm.calls == 2  # the death and its one repeat; no round over the unresolved attempt
    assert usage["_last_llm_error_kind"] == "llm_output_exhausted"
    assert trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"
    assert all(_host_facts(request["messages"]) == [] for request in llm.requests)


def test_exhaustion_ends_an_active_transport_wait_episode(tmp_path, loop_env, no_sleep, monkeypatch):
    """The provider answered, so the outage episode ends at the exhausted reply (a later round
    must not inherit a stale episode, which would reshape its deadline and Stop handling)."""
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", _quiet_chain)
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda _sec, _wake: False)
    llm = _ScriptedLLM(_released_connect, _incident_reply(0), OK_RESPONSE)
    result, _usage, _trace = run_llm_loop(**_loop_kwargs(tmp_path, llm, []))

    assert result == "done" and llm.calls == 3
    waits = [(row["phase"], row.get("detail")) for row in _events(tmp_path, "network_wait")]
    assert waits[0][0] == "entered"
    assert waits[-1] == ("ended", "error_kind_changed:llm_output_exhausted")
    assert len(_host_facts(llm.requests[-1]["messages"])) == 1


def test_an_owner_terminal_names_output_exhaustion_not_a_provider_failure():
    """A terminal composed after an exhausted reply that did NOT take the next-round branch
    (an unresolved attempt fences it, or a configured-route candidate exhausted) names the
    length limit; any other kind keeps its own words and an unknown one stays generic."""
    exhausted = loop_transport.provider_failure_hint(
        {"_last_llm_error": "x", "_last_llm_error_kind": "llm_output_exhausted"})
    assert "reached its length limit before any visible output" in exhausted
    assert "a provider failure" not in exhausted
    empty = loop_transport.provider_failure_hint({"_last_llm_error": "x", "_last_llm_error_kind": "llm_empty_response"})
    assert "classified the failure as an empty response." in empty
    unknown = loop_transport.provider_failure_hint({"_last_llm_error": "x", "_last_llm_error_kind": "not_a_kind"})
    assert "classified the failure as a provider failure." in unknown


def _walk_loop(tmp_path, monkeypatch, *script, fallbacks="other/model", reachable=None):
    """The real loop with the real configured-route walk (a route switch keeps the captured core)."""
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", fallbacks)
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda _sec, _wake: False)
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable", lambda *_a, **_kw: reachable or {})
    from dataclasses import replace
    from tests.test_context_fit_v664 import _plan

    def rebind(plan_, tools, _messages, *, model, use_local, **kwargs):
        rebound = replace(plan_, model=model, route_fp=f"route-{model}")
        tools._ctx.context_fit_plan = rebound
        return rebound, "max"

    monkeypatch.setattr(loop_mod, "_rebind_context_fit_plan", rebind)
    llm = _ScriptedLLM(*script)
    kwargs = _loop_kwargs(tmp_path, llm, [])
    kwargs["tools"]._ctx.context_fit_plan = _plan(preferred="max", window=1_000_000, known=True)
    return (llm, *run_llm_loop(**kwargs))


def test_a_configured_candidate_that_exhausts_reaches_the_next_round_not_a_redial(tmp_path, no_sleep, monkeypatch):
    """The real configured-route walk: the primary fails before dispatch, the fallback candidate
    answers but spends its reply allowance. That candidate's kind survives the walk (it is no
    outage), so the next ORDINARY round carries the host fact; the exhausted request is not
    redialed inside the same round under the primary's wait."""
    llm, result, usage, _trace = _walk_loop(tmp_path, monkeypatch, _released_connect, _incident_reply(0), OK_RESPONSE)

    assert result == "done" and llm.calls == 3
    primary, fallback, next_round = llm.requests
    assert fallback["model"] == "other/model" and _host_facts(fallback["messages"]) == []
    assert len(_host_facts(next_round["messages"])) == 1  # the fact reached the next round
    assert [row.get("phase") for row in _events(tmp_path, "network_wait")] in ([], ["entered", "ended"])
    assert len(_events(tmp_path, "llm_round")) == 1  # one usable round; the exhausted one consumed its own


def test_an_exhausted_candidate_ends_the_walk_before_the_next_configured_route(tmp_path, no_sleep, monkeypatch):
    """With two configured routes, the first one's exhaustion is the round's outcome: the second
    route is not dialed with the same transcript, and the next ordinary round carries the fact."""
    llm, result, _usage, _trace = _walk_loop(tmp_path, monkeypatch, _released_connect, _incident_reply(0), OK_RESPONSE,
                                             fallbacks="other/model,third/model")
    assert result == "done" and llm.calls == 3
    primary, fallback, next_round = llm.requests
    assert fallback["model"] == "other/model" and next_round["model"] == primary["model"]
    assert len(_host_facts(next_round["messages"])) == 1


@pytest.mark.parametrize("script,reachable", [
    ((_death, _incident_reply(0), OK_RESPONSE), {"kind": "upstream_http", "status_code": 200}),  # no episode yet
    ((_death, _released_connect, _incident_reply(0), OK_RESPONSE), None),  # inside the unknown outcome's episode
])
def test_an_exhausted_candidate_after_an_unknown_primary_keeps_its_wait_and_its_fact(
        tmp_path, no_sleep, monkeypatch, script, reachable):
    """An outstanding unknown outcome owns the round's wait, whether its episode starts now or is
    already open: the exhausted candidate neither ends the task nor drops that custody, and its
    host fact reaches the continuation the wait authorizes (one fact, not a repeated request)."""
    llm, result, usage, _trace = _walk_loop(tmp_path, monkeypatch, *script, reachable=reachable)
    assert result == "done" and llm.calls == len(script)
    assert len(_host_facts(llm.requests[-1]["messages"])) == 1
    assert llm.requests[-1]["messages"] != llm.requests[-2]["messages"]
    assert "entered" in [row.get("phase") for row in _events(tmp_path, "network_wait")]
    assert "_pending_transport_outcome" not in usage  # the usable reply closed the unknown record
