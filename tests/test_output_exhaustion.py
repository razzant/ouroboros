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

import json

import httpx
import pytest

import ouroboros.loop as loop_mod
import ouroboros.loop_llm_call as call_mod
import ouroboros.loop_transport as loop_transport
from ouroboros import loop_output_exhaustion as recovery, usage_accounting as ua
from ouroboros.loop import _output_exhausted_notice, run_llm_loop
from ouroboros.primary_route_observation import ROUND_REFUSALS_KEY, refusal_words
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
from ouroboros.llm_claudexor import ClaudexorContentRefused
from ouroboros.outcomes import _refusal_final_answer, derive_loop_outcome, selected_final_answer
from tests._model_replies import MIXED_REFUSAL_PARTS, MIXED_REFUSAL_TEXT, finish
from tests._scripted_model_loop import (
    OK_RESPONSE,
    ScriptedLLM as _ScriptedLLM,
    loop_kwargs as _loop_kwargs,
    walk_loop as _walk_loop,
)
from tests._shared import saved_settings

MESSAGES = [{"role": "user", "content": "work"}]
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
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
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


@pytest.mark.parametrize(("usage", "msg"), [
    ({"prompt_tokens": 5, "response_finish_reason": "content_filter"}, {"role": "assistant", "content": ""}),  # OpenAI family
    ({"prompt_tokens": 5}, {"role": "assistant", "content": "", "stop_reason": "refusal"}),  # Anthropic
])
def test_a_provider_content_refusal_is_typed_sent_once_and_carries_its_lineage(tmp_path, no_sleep, usage, msg):
    """A provider's refusal is a route fact, not a transient empty reply: one send, its own event
    and kind, no cooldown, no spent wall, not infra; a transient body error still retries as itself."""
    assert _classify_empty_response(usage, msg) == ("provider_content_refusal", False, True)
    assert _classify_empty_response({**usage, "provider_error": {"kind": "rate_limit", "code": 429}}, msg) == (
        "llm_empty_response", False, False)
    llm, totals = _ScriptedLLM((msg, usage), (msg, usage)), {}
    with ua.usage_scope(ua.UsageScope(task_id="t-exhaust", root_task_id="root", parent_task_id="parent")):
        assert _call(llm, tmp_path, totals)[0] is None

    assert llm.calls == 1 and no_sleep == []
    [row] = _events(tmp_path, "provider_content_refusal")
    assert (row["finish_reason"], row["root_task_id"], row["parent_task_id"]) == (
        usage.get("response_finish_reason") or msg.get("stop_reason"), "root", "parent")
    assert _events(tmp_path, "llm_empty_response") == _events(tmp_path, "provider_incomplete_response") == []
    assert totals["_last_llm_error_kind"] == "content_filter" == totals["_last_llm_call_meta"]["failure_code"]
    assert "content_filter" not in _TRANSIENT_RETRY_KINDS | _COOLDOWN_ERROR_KINDS
    assert (totals["execution_status"], totals["reason_code"]) == ("failed", "provider_content_refusal")
    assert RETRY_WALL_EXHAUSTED_KEY not in totals and "_last_llm_output_exhausted" not in totals
    assert loop_transport._FAILURE_KIND_WORDS["content_filter"] != loop_transport._FAILURE_KIND_WORDS[
        "provider_policy_refusal"]


REFUSAL_WORDS = "the provider's content filter stopping this request (no content returned)"


def test_a_content_refusal_speaks_in_its_own_words_and_its_round_keeps_them(tmp_path, no_sleep, monkeypatch):
    """The kind's one sentence is about the wire and this request, never a verdict on a model or the work. The
    recorded wire's ``delta.refusal`` becomes the owner-facing provider sentence, its event's and its round's, and
    the feed row quotes it; with no usage chunk its money stays unknown. A refusal that carried no sentence stamps
    none, so no earlier words speak for it."""
    import ouroboros.pricing as pricing
    from tests.test_llm_wire_corpus import WIRE, _assemble, _normalized

    assert loop_transport._FAILURE_KIND_WORDS["content_filter"] == REFUSAL_WORDS
    monkeypatch.setattr(pricing, "estimate_cost_optional", lambda *_a, **_kw: None)  # no price lookup
    wire = WIRE / "openrouter/claude-fable-5/refusal_stream_v2.sse"
    body, receipt = _assemble(wire, 0)
    msg, usage = _normalized(wire, {**body, "_stream_receipt": receipt})
    sentence, totals = " ".join(msg["refusal"].split()), {}
    assert sentence.startswith("This request was blocked") and "prompt_tokens" not in usage
    assert _call(_ScriptedLLM((msg, usage)), tmp_path, totals)[0] is None

    assert totals["_last_llm_provider_message"] == sentence  # before: none, stamped from the blank body error
    [row] = _events(tmp_path, "provider_content_refusal")
    assert row["provider_message"] == sentence and "refusal_category" not in row
    [kept] = totals[ROUND_REFUSALS_KEY]
    assert (kept["round"], kept["provider_message"], kept["response_provider"], kept["cost"]) == (
        1, sentence, usage["response_provider"], None)
    assert loop_transport.fallback_reason_text(totals, "openai/gpt-5.5") == (
        f' after {REFUSAL_WORDS} on openai/gpt-5.5. The provider said: "{sentence}"')
    bare = ({"role": "assistant", "content": ""}, {"prompt_tokens": 5, "response_finish_reason": "content_filter"})
    assert _call(_ScriptedLLM(bare), tmp_path, totals)[0] is None
    assert "_last_llm_provider_message" not in totals and "provider_message" not in totals[ROUND_REFUSALS_KEY][-1]
    assert loop_transport.fallback_reason_text(totals, "openai/gpt-5.5") == f" after {REFUSAL_WORDS} on openai/gpt-5.5."


def test_a_native_refusal_keeps_its_category_and_explanation(monkeypatch):
    """The native Messages lane keeps ``stop_details`` as a call fact the refusal's words read; a null category is
    left out (the row then says it was not given), and a reply without ``stop_details`` carries no words."""
    from ouroboros.llm import LLMClient

    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    client = LLMClient()
    target = client._resolve_remote_target("anthropic::claude-sonnet-4-6")
    for category, named in (("cyber", {"refusal_category": "cyber"}), (None, {})):
        msg, usage = client._normalize_anthropic_response({
            "content": [], "stop_reason": "refusal", "usage": {"input_tokens": 9, "output_tokens": 0},
            "stop_details": {"type": "refusal", "category": category, "explanation": "Flagged  as cyber."}}, target)
        assert _classify_empty_response(usage, msg) == ("provider_content_refusal", False, True)
        assert refusal_words(usage, msg) == {"provider_message": "Flagged as cyber.", **named}
    msg, usage = client._normalize_anthropic_response(
        {"content": [], "stop_reason": "refusal", "usage": {"input_tokens": 9, "output_tokens": 0}}, target)
    assert "stop_details" not in usage and refusal_words(usage, msg) == {}


@pytest.mark.parametrize("refused", [True, False])
def test_a_round_ending_on_a_content_refusal_names_it_and_never_an_outage(tmp_path, no_sleep, monkeypatch, refused):
    """No configured route to ask: the refusal's one feed row states it with its provider's words (no "Fallback",
    nothing "trying"), and the forced final's prompt and the terminal notice name it in the kind's words with its
    route, adding no act. A round ended by another failure is told in its provider's own sentence, never as an
    outage the host did not establish, and gets no such row."""
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "")
    failed = (({"role": "assistant", "content": "", "refusal": "Blocked here."},
               {"prompt_tokens": 5, "response_finish_reason": "content_filter"}) if refused else
              ({"role": "assistant", "content": ""},
               {"prompt_tokens": 5, "provider_error": {"kind": "provider_error", "code": 400, "message": "Rejected."}}))
    llm, notes = _ScriptedLLM(failed, ({"role": "assistant", "content": "summary"}, {"prompt_tokens": 1})), []
    _result, usage, _trace = run_llm_loop(**_loop_kwargs(tmp_path, llm, notes))

    model = llm.requests[0]["model"]
    [prompt] = [row["content"] for row in llm.requests[-1]["messages"] if "[PROVIDER_UNAVAILABLE]" in str(row.get("content"))]
    rows = [note for note in notes if "No usable reply" in note]
    met = f" This round's failed asks: {REFUSAL_WORDS} on {model} once."
    assert llm.calls == 2 and ROUND_REFUSALS_KEY not in usage
    assert not any("Fallback" in note or "trying" in note for note in notes)
    if refused:
        assert f"[PROVIDER_UNAVAILABLE] No route returned a usable response.{met} The task is being INTERRUPTED" in prompt
        assert "outage" not in prompt and "outage" not in usage["terminal_provider_notice"]
        assert usage["terminal_provider_notice"].endswith(met)
        assert rows == [f'⚠️ No usable reply after {REFUSAL_WORDS} on {model}. The provider said: "Blocked here."']
    else:
        assert ('No route returned a usable response. The provider said: "Rejected." The task is being INTERRUPTED, not'
                in prompt and "outage" not in prompt and "failed asks" not in prompt and rows == [])


BODY_ERROR = ({"role": "assistant", "content": ""},
              {"prompt_tokens": 5, "provider_error": {"kind": "provider_error", "code": 400, "message": "Rejected."}})
REFUSAL = ({"role": "assistant", "content": "", "refusal": "Blocked here."},
           {"prompt_tokens": 5, "response_finish_reason": "content_filter"})
FORCED_ERROR = ({"role": "assistant", "content": ""},
                {"prompt_tokens": 5, "provider_error": {"kind": "provider_error", "code": 400, "message": "Forced call failed."}})


@pytest.mark.parametrize("forced", ["refused", "answers", "fails"])
def test_a_forced_call_stopped_by_the_filter_leaves_the_owner_the_notice_that_names_it(
        tmp_path, no_sleep, monkeypatch, forced):
    """The host's words for a round nobody answered are composed before the forced final's own call. When that call
    meets the content filter, the owner reads the notice recomputed with it, and the published answer is that same
    text: one text that names the refusal, never the words written before it. A forced call that answers keeps its
    own words; one that fails in a round with no content refusal leaves the owner the words that named the round's
    own failure, as before."""
    import hashlib

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "")
    summary = ({"role": "assistant", "content": "summary"}, {"prompt_tokens": 1})
    llm = _ScriptedLLM(BODY_ERROR, {"refused": REFUSAL, "answers": summary, "fails": FORCED_ERROR}[forced])
    result, usage, trace = run_llm_loop(**_loop_kwargs(tmp_path, llm, []))

    model, notice = llm.requests[0]["model"], usage["terminal_provider_notice"]
    assert llm.calls == 2 and ROUND_REFUSALS_KEY not in usage
    if forced == "answers":
        assert result == "summary" and REFUSAL_WORDS not in notice
        return
    if forced == "fails":  # before: the owner's text was replaced by the notice quoting only the forced call
        assert '"Rejected."' in result and "Forced call failed." not in result and "Forced call failed." in notice
        assert usage["terminal_origin"] == "host_notice" and REFUSAL_WORDS not in result + notice
        assert trace["delivery_candidate"]["content_sha256"] == hashlib.sha256(result.encode("utf-8")).hexdigest()
        return
    words = loop_transport._failure_words(next(row["failure_code"] for row in usage["llm_call_refs"]))
    assert notice.endswith(f" This round's failed asks: {words} on {model} once; {REFUSAL_WORDS} on {model} once.")
    assert result == notice and usage["terminal_origin"] == "host_notice"  # before: the words without the refusal
    sha = hashlib.sha256(notice.encode("utf-8")).hexdigest()
    assert trace["delivery_candidate"]["content_sha256"] == trace["forced_finalization"]["candidate_sha256"] == sha


def test_a_later_rounds_end_names_its_own_failures_never_an_earlier_rounds_refusal(tmp_path, no_sleep, monkeypatch):
    """Round 1: the primary's content refusal, and the adopted spare's reply spends its length limit (no answer).
    Round 2: the spare answers with a tool call. Round 3: the spare, the walk's primary and the forced final's call
    return a body error. Round 3's forced-final prompt and terminal text are round 3's own: its body error's
    sentence, no outage claimed. Round 1's refusal is never told as this round's (before: it was, and round 3's errors
    were left out); its unanswered record is dropped at round 2's request, which carries no row of it."""
    tools = ({"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {
        "name": "chat_history", "arguments": "{}"}}]}, {"prompt_tokens": 1})
    llm, result, usage, _trace = _walk_loop(tmp_path, monkeypatch, REFUSAL, _incident_reply(0), tools, BODY_ERROR,
                                            BODY_ERROR, BODY_ERROR)

    primary = llm.requests[0]["model"]
    assert [request["model"] for request in llm.requests[:5]] == [primary] + ["other/model"] * 3 + [primary]
    assert llm.calls == 6 and [row["round"] for row in usage["llm_call_refs"]
                               if row.get("failure_code") == "content_filter"] == [1]
    [prompt] = [row["content"] for row in llm.requests[-1]["messages"] if "[PROVIDER_UNAVAILABLE]" in str(row.get("content"))]
    notice = usage["terminal_provider_notice"]
    assert 'The provider said: "Rejected."' in prompt and "outage" not in prompt and "Rejected." in notice and result == notice
    for text in (prompt, notice):
        assert REFUSAL_WORDS not in text and "failed asks" not in text and "Blocked here." not in text
    assert not any("[ROUTE FACTS] This round's asks" in str(row.get("content")) for request in llm.requests
                   for row in request["messages"])
    assert ROUND_REFUSALS_KEY not in usage


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


def test_repeated_exhaustion_ends_without_purchasing_a_recap(tmp_path, loop_env, no_sleep, monkeypatch):
    """The owner chose finite recovery, not repeated ordinary rounds. No eligible source
    view or configured reserve here: two empty replies hand back the saved work."""
    saved_settings(monkeypatch, tmp_path, OUROBOROS_MODEL_FALLBACKS="")  # a saved blank: no chain
    llm = _ScriptedLLM(*[_incident_reply(index) for index in range(6)])
    result, usage, trace = run_llm_loop(**_loop_kwargs(tmp_path, llm, []))

    assert llm.calls == 2
    assert [len(_host_facts(request["messages"])) for request in llm.requests] == [0, 1]
    assert [row["round"] for row in _events(tmp_path, "llm_empty_response")] == [1, 2]
    assert _events(tmp_path, "provider_incomplete_response") == []
    assert usage["reason_code"] == "output_recovery_exhausted" and usage["execution_status"] == "infra_failed"
    assert trace["forced_finalization"]["source"] == "output_exhaustion_no_recap"
    assert "Work is incomplete" in result


def test_exhaustion_over_an_unresolved_attempt_starts_no_new_generation(tmp_path, loop_env, no_sleep):
    """The fence: a round still holding an unresolved attempt (a granted transport-death repeat)
    may start no new generation, so an exhausted repeat keeps the no-resend terminal."""
    from tests.test_transport_wait_repeat_interaction import _held_repeat
    llm = _ScriptedLLM(_incident_reply(0), OK_RESPONSE)
    kwargs = _loop_kwargs(tmp_path, llm, [])
    kwargs["task_type"] = "presence"
    kwargs["tools"]._ctx.is_direct_chat = True
    kwargs["tools"]._ctx.current_task_type = "presence"
    _held_repeat(kwargs)
    _result, usage, trace = run_llm_loop(**kwargs)

    assert llm.calls == 1  # the exhausted response; no round over the retained unknown attempt
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
    assert fallback["model"] == next_round["model"] == "other/model"  # informed chance stays on the answering route
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


@pytest.mark.parametrize("cooling", [True, False])
def test_a_refusal_whose_configured_routes_all_cool_gets_its_own_row_once(tmp_path, no_sleep, monkeypatch, cooling):
    """The real walk with its one configured route cooling asks nobody: the refusal's own feed row, once, with no
    "Fallback" claim. When that route is asked, the walk's own rows name the refusal as before and none is added."""
    from ouroboros import fallback_cooldown

    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_a: cooling)
    refusal = ({"role": "assistant", "content": "", "refusal": "Blocked here."},
               {"prompt_tokens": 5, "response_finish_reason": "content_filter"})
    tools = ({"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {
        "name": "chat_history", "arguments": "{}"}}]}, {"prompt_tokens": 1})
    done = (finish("done"), {"prompt_tokens": 1})
    notes = []
    llm, result, _usage, _trace = _walk_loop(tmp_path, monkeypatch, *((refusal, done) if cooling else (refusal, tools, done)),
                                             notes=notes)
    primary, rows = llm.requests[0]["model"], [note for note in notes if note.startswith("⚠️ No usable reply")]
    assert result == "done"
    if cooling:
        assert [request["model"] for request in llm.requests] == [primary, primary]  # the forced final, nobody else
        assert rows == [f'⚠️ No usable reply after {REFUSAL_WORDS} on {primary}. The provider said: "Blocked here."']
        assert not any("Fallback" in note or "trying" in note for note in notes)
    else:
        assert rows == [] and llm.requests[1]["model"] == "other/model"
        assert any(note.startswith("⚡ Fallback: trying other/model") and "Blocked here." in note for note in notes)


# ------------------------------------------------------------------ text a typed finish cut (#1472)

CUT_TEXT = "The probe finished: both files were read and the first count is 12, while the second"


def _finished(source, finish, text=CUT_TEXT):
    """A usable reply whose finish arrives where its lane writes it: usage (OpenAI family, Claudexor, local,
    GigaChat) or the native Anthropic message's ``stop_reason``; ``none`` carries no finish fact."""
    msg, usage = {"role": "assistant", "content": text}, {"prompt_tokens": 9, "completion_tokens": 4096, "cost": 0.0}
    if source == "usage":
        usage["response_finish_reason"] = finish
    elif source == "message":
        msg["stop_reason"] = finish
    return msg, usage


@pytest.mark.parametrize("shape", ["task", "main"])
@pytest.mark.parametrize(("source", "finish"), [("usage", "length"), ("usage", "tool_calls"), ("usage", "function_call"),
                                                  ("message", "max_tokens"), ("message", "tool_use")])
def test_text_a_typed_finish_cut_gets_one_informed_turn_on_every_lane(tmp_path, loop_env, no_sleep, source, finish, shape):
    """Each lane's typed finish makes callless text no final: the next ORDINARY round on the same route reads
    the kept words and one host fact, and its answer is the final; no route is walked (``loop_env``). ``main``
    is only the loop's Main shape (``is_direct_chat``): Main admission and delivery are not run here."""
    llm = _ScriptedLLM(_finished(source, finish), OK_RESPONSE)
    kwargs = _loop_kwargs(tmp_path, llm, [])
    if shape == "main":
        kwargs["tools"]._ctx.is_direct_chat, kwargs["tools"]._ctx.current_chat_id = True, 1
    result, usage, trace = run_llm_loop(**kwargs)
    assert result == "done" and llm.calls == 2
    first, second = (request["messages"] for request in llm.requests)
    assert second[:len(first)] == first  # appended: the cut request stays a prefix of the next
    kept = second.index({"role": "assistant", "content": CUT_TEXT})
    assert f"(finish_reason={finish})" in second[kept + 1]["content"] and second[kept + 1]["role"] == "user"
    assert [row["round"] for row in _events(tmp_path, "llm_round")] == [1, 2]
    assert "forced_finalization" not in trace and recovery.UNFINISHED_KEY not in usage


@pytest.mark.parametrize(("source", "finish"), [("usage", "stop"), ("message", "end_turn"), ("usage", None),
                                                  ("none", None), ("usage", "content_filter")])
def test_other_finishes_leave_text_the_plain_final(tmp_path, loop_env, no_sleep, source, finish):
    """Negative control: only the output-limit and tool-call finishes mark a cut. Stop, end_turn, a null or absent
    finish and any other typed finish leave the text the round's final, as before: one call, no host fact."""
    llm = _ScriptedLLM(_finished(source, finish))
    result, usage, trace = run_llm_loop(**_loop_kwargs(tmp_path, llm, []))
    assert (result, llm.calls) == (CUT_TEXT, 1)
    [event] = _events(tmp_path, "llm_round")
    assert ("finish_reason" in event) == (source != "none")
    assert event.get("finish_reason") == finish
    assert recovery.UNFINISHED_KEY not in usage and "forced_finalization" not in trace


@pytest.mark.parametrize("finish,expected", [("content_filter", {"finish_reason": "content_filter"}),
                                            ("private diagnostic text", {"finish_reason_origin": "discarded"})])
def test_accepted_tool_calls_keep_only_a_finish_label_in_the_round(tmp_path, no_sleep, finish, expected):
    from unittest.mock import Mock
    from ouroboros.llm import LLMClient

    calls = [{"id": "c", "type": "function", "function": {"name": "inspect", "arguments": "{}"}}]
    message = {"role": "assistant", "content": "", "tool_calls": calls}
    body = {"choices": [{"message": message, "finish_reason": finish}], "usage": {"cost": 0.0}}
    client, llm = LLMClient(api_key="k"), Mock()
    llm.chat.side_effect = lambda **_: client._normalize_remote_response(body, {"provider": "openrouter"},
                                                                        skip_cost_fetch=True)
    accepted, _ = _call(llm, tmp_path, {})
    assert accepted["tool_calls"] == calls and llm.chat.call_count == 1
    [row] = _events(tmp_path, "llm_round")
    assert {key: row[key] for key in ("finish_reason", "finish_reason_origin") if key in row} == expected


# ------------------------------------------------------------------ provider-stated usability (Responses terminals)

COMMENTARY = {"response_finish_reason": "commentary", "response_status": "completed"}


def _reply(text, facts, **message):
    return {"role": "assistant", "content": text, **message}, {"prompt_tokens": 9, "completion_tokens": 7, "cost": 0.0,
                                                               **facts}


def test_completed_commentary_continues_ordinary_rounds_until_the_final(tmp_path, loop_env, no_sleep):
    """Healthy intermediate progress is no cut and no episode: each commentary is kept as written (a progress
    update, no host notice, no cap) and the next send is an ordinary round; two in a row reach the final."""
    llm, notes = _ScriptedLLM(_reply("Reading the files first.", COMMENTARY), _reply("Still comparing.", COMMENTARY),
                              OK_RESPONSE), []
    result, usage, trace = run_llm_loop(**_loop_kwargs(tmp_path, llm, notes))
    assert result == "done" and llm.calls == 3
    third = llm.requests[2]["messages"]
    kept = [row["content"] for row in third if row.get("role") == "assistant"]
    assert kept == ["Reading the files first.", "Still comparing."]
    assert not [row for row in third if row.get("role") == "user" and "[SYSTEM NOTICE]" in str(row.get("content"))]
    assert llm.requests[1]["messages"][:len(llm.requests[0]["messages"])] == llm.requests[0]["messages"]
    assert {"Reading the files first.", "Still comparing."} <= {str(note).strip() for note in notes}
    assert recovery.UNFINISHED_KEY not in usage and "forced_finalization" not in trace
    assert [row["round"] for row in _events(tmp_path, "llm_round")] == [1, 2, 3]



def test_an_owner_stop_after_commentary_ends_the_turn_with_no_further_call(tmp_path, loop_env, no_sleep):
    """Commentary continues as an ordinary round, so the round top's existing controls still decide first: the
    owner's Stop arriving after a commentary ends the turn at that boundary with zero further model calls and no
    tool — the commentary stays a kept progress update, never the delivered answer, and no cut episode opens."""
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN

    class _StopAfterCommentary(_ScriptedLLM):
        def chat(self, **kwargs):
            reply = super().chat(**kwargs)
            if self.calls == 1:  # the owner presses Stop while the commentary round is answering
                assert write_owner_message(tmp_path, REASON_OWNER_STOPPED_DIRECT_TURN, "t-exhaust-loop",
                                           kind=KIND_FINALIZE_NOW)
            return reply

    from ouroboros.outcomes import derive_loop_outcome

    words = "Reading the files first.\nFINAL ANSWER: draft"  # a marker inside progress is no answer
    llm, notes = _StopAfterCommentary(_reply(words, COMMENTARY), OK_RESPONSE), []
    kwargs = _loop_kwargs(tmp_path, llm, notes)
    result, usage, trace = run_llm_loop(**kwargs)
    assert llm.calls == 1 and "Reading the files first." not in str(result)
    assert usage["reason_code"] == REASON_OWNER_REQUESTED_FINALIZATION and kwargs["tools"]._ctx._skip_post_task_synthesis
    assert {"role": "assistant", "content": words} in kwargs["tools"]._ctx.messages
    assert words in {str(note).strip() for note in notes}
    assert "best_valid_final_answer" not in trace and derive_loop_outcome(result, usage, trace)["final_answer"] == ""
    assert recovery.UNFINISHED_KEY not in usage and recovery.KEY not in usage
    assert [row["round"] for row in _events(tmp_path, "llm_round")] == [1]  # the stopped boundary bought no round


@pytest.mark.parametrize(("facts", "latched"), [
    (COMMENTARY, ""),
    ({"response_finish_reason": "max_output_tokens", "response_status": "incomplete"}, ""),
    ({"response_finish_reason": "length"}, ""),
    ({"response_finish_reason": "stop", "response_status": "completed"}, "draft"),  # an ordinary final still latches
])
def test_only_a_reply_eligible_as_a_final_latches_its_marker(tmp_path, loop_env, no_sleep, facts, latched):
    """The FINAL ANSWER latch is what a deadline, a budget end or a Stop publishes when the last text names none
    (``outcomes.selected_final_answer``, ``loop_budget``): a commentary or a cut keeps its words as progress and
    latches nothing, so a later marker-less final publishes no draft; a completed final latches as before."""
    from ouroboros.outcomes import derive_loop_outcome

    llm = _ScriptedLLM(_reply("Checked the inputs.\nFINAL ANSWER: draft", facts), OK_RESPONSE)
    result, usage, trace = run_llm_loop(**_loop_kwargs(tmp_path, llm, []))
    assert (trace.get("best_valid_final_answer") or "") == latched
    if latched:
        assert llm.calls == 1 and derive_loop_outcome(result, usage, trace)["final_answer"] == latched
    else:
        assert (result, llm.calls) == ("done", 2) and derive_loop_outcome(result, usage, trace)["final_answer"] == ""


@pytest.mark.parametrize(("boundary", "reason"), [("deadline", "deadline_local"), ("budget", "budget_exhausted")])
def test_a_commentary_marker_is_no_answer_at_a_deadline_or_a_budget_end(tmp_path, loop_env, no_sleep, monkeypatch,
                                                                       boundary, reason):
    """The rails that publish the latch with no further send: the task deadline passes while a completed commentary
    carrying a FINAL ANSWER line answers (``_forced_final_answer``'s elapsed window), or money refuses the next
    dispatch before it is sent (``_handle_budget_exceeded``, whose latched fallback once returned the draft). Either
    ends the work on that one provider send, and neither its result, the selected answer nor the latch holds it."""
    import datetime

    import ouroboros.usage_accounting as accounting
    from ouroboros.outcomes import derive_loop_outcome, selected_final_answer

    words = "Reading the files first.\nFINAL ANSWER: draft"
    kwargs = _loop_kwargs(tmp_path, None, [])
    tool_ctx = kwargs["tools"]._ctx

    class _Boundary(_ScriptedLLM):
        refused = 0

        def chat(self, **chat_kwargs):
            if self.calls and boundary == "budget":  # admission refuses the next dispatch: nothing is sent
                self.refused += 1
                raise accounting.BudgetExceeded("root budget closed", limit_scope="root", root_task_id="t-exhaust-loop")
            reply = super().chat(**chat_kwargs)
            if boundary == "deadline":  # the deadline passes while this commentary round is answering
                passed = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5)
                tool_ctx.task_metadata = {**tool_ctx.task_metadata, "deadline_at": passed.isoformat()}
            return reply

    # The ledger's own fact that the commentary was a physical send (the scripted client writes none), as the
    # budget rail's existing tests state it; with none, a refused first dispatch is replay-safe and raises instead.
    monkeypatch.setattr(accounting, "usage_breakdown", lambda *_a, **_k: {"physical_calls": 1, "integrity_degraded": False})
    llm = kwargs["llm"] = _Boundary(_reply(words, COMMENTARY), OK_RESPONSE)
    result, usage, trace = run_llm_loop(**kwargs)
    assert llm.calls == 1 and llm.refused == (boundary == "budget") and usage["reason_code"] == reason
    assert {"role": "assistant", "content": words} in tool_ctx.messages  # kept as progress, never delivered
    assert "draft" not in str(result) and "best_valid_final_answer" not in trace
    assert selected_final_answer(result, trace) == "" and derive_loop_outcome(result, usage, trace)["final_answer"] == ""


@pytest.mark.parametrize("reason", ["max_messages", "steered", None])
def test_an_unsuitable_incomplete_terminal_with_text_or_a_call_is_never_a_result(tmp_path, loop_env, no_sleep, reason):
    """``status=incomplete`` for a reason other than the output limit makes neither its text (valid JSON here) nor
    its call a result: it reads as a reply without usable output, so the round's ordinary retry asks again and
    nothing of it is delivered, kept or executed."""
    unusable = _reply('{"verdict": "PASS"}', {"response_finish_reason": reason, "response_status": "incomplete"},
                      tool_calls=[{"id": "c1", "type": "function", "function": {"name": "shell", "arguments": "{}"}}])
    llm = _ScriptedLLM(unusable, OK_RESPONSE)
    result, _usage, _trace = run_llm_loop(**_loop_kwargs(tmp_path, llm, []))
    assert result == "done" and llm.calls == 2
    assert all(row.get("role") not in {"assistant", "tool"} for row in llm.requests[1]["messages"])
    events = _events(tmp_path, "llm_empty_response") + _events(tmp_path, "provider_incomplete_response")
    assert [row["finish_reason"] for row in events] == [reason]


def test_an_incomplete_content_filter_with_text_is_the_providers_refusal(tmp_path, no_sleep, monkeypatch):
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "")
    cut = _reply('{"verdict": "PASS"}', {"response_finish_reason": "content_filter", "response_status": "incomplete"})
    llm = _ScriptedLLM(cut, ({"role": "assistant", "content": "summary"}, {"prompt_tokens": 1}))
    result, usage, _trace = run_llm_loop(**_loop_kwargs(tmp_path, llm, []))
    assert "PASS" not in str(result) and "PASS" not in json.dumps(llm.requests[-1]["messages"])
    assert [row["finish_reason"] for row in _events(tmp_path, "provider_content_refusal")] == ["content_filter"]
    assert usage["terminal_provider_notice"]  # the round ended on the provider's refusal, said as such


@pytest.mark.parametrize("preamble", [False, True])
def test_a_completed_responses_refusal_is_the_providers_refusal_sent_once(tmp_path, no_sleep, preamble):
    """A completed Response carrying the model's typed refusal, alone or after a commentary preamble, reaches the
    round as the provider's content refusal through the real adapter: one send, its own event in its own words,
    never a retried empty reply nor intermediate progress."""
    from tests.test_openai_responses import REFUSAL, _body, _message, _normalize

    msg, usage = _normalize(_body(*([_message("Let me look first.", "commentary")] if preamble else []), REFUSAL))
    llm, totals = _ScriptedLLM((msg, usage), OK_RESPONSE), {}
    assert _call(llm, tmp_path, totals)[0] is None
    assert llm.calls == 1 and no_sleep == [] and _events(tmp_path, "llm_empty_response") == []
    [row] = _events(tmp_path, "provider_content_refusal")
    assert (row["finish_reason"], row["provider_message"]) == ("refusal", REFUSAL["content"][0]["refusal"])
    assert totals["_last_llm_error_kind"] == "content_filter" and totals["reason_code"] == "provider_content_refusal"


def test_an_output_limit_incomplete_keeps_the_existing_cut_recovery(tmp_path, loop_env, no_sleep):
    cut = _reply(CUT_TEXT, {"response_finish_reason": "max_output_tokens", "response_status": "incomplete"})
    llm = _ScriptedLLM(cut, OK_RESPONSE)
    result, usage, _trace = run_llm_loop(**_loop_kwargs(tmp_path, llm, []))
    assert result == "done" and llm.calls == 2
    second = llm.requests[1]["messages"]
    kept = second.index({"role": "assistant", "content": CUT_TEXT})
    assert "(finish_reason=max_output_tokens)" in second[kept + 1]["content"]


def test_forced_finals_read_intermediate_and_unusable_terminals_as_incomplete():
    from ouroboros.loop_llm_call import forced_response_is_incomplete

    assert forced_response_is_incomplete({"finish_reason_present": True, "finish_reason": "commentary"})
    assert forced_response_is_incomplete({"finish_reason_present": True, "finish_reason": "steered",
                                          "response_status": "incomplete"})
    assert not forced_response_is_incomplete({"finish_reason_present": True, "finish_reason": "stop",
                                              "response_status": "completed"})
    assert not forced_response_is_incomplete({"finish_reason_present": True, "finish_reason": "max_output_tokens",
                                              "response_status": "incomplete"}, completion_request=True)


@pytest.mark.parametrize(("parts", "visible"), [
    (MIXED_REFUSAL_PARTS, MIXED_REFUSAL_TEXT),
    ([{"type": "refusal", "refusal": ""}], ""),
    ([{"type": "refusal", "refusal": " \n"}], " \n"),
    ([{"type": "text", "text": "  "}, {"type": "refusal", "refusal": ""}], "  "),
])
def test_completed_refusal_is_a_literal_reply_even_with_configured_alternates(tmp_path, monkeypatch, parts, visible):
    msg = {"role": "assistant", "content": parts}
    llm, text, usage, trace = _walk_loop(tmp_path, monkeypatch, (msg, {"provider": "claudexor"}))
    assert text == visible and llm.calls == 1
    assert usage.get("execution_status") != "failed"
    assert usage["terminal_origin"] == "model_final"
    assert derive_loop_outcome(text, usage, trace)["reason_code"] != "empty_final_text"
    assert selected_final_answer(text, trace) == "" and _refusal_final_answer(text, trace) == ""  # hash-bound


@pytest.mark.parametrize("calls", ["none", "unrun", "refused_finish", "own_finish"])
def test_a_forced_blank_refusal_beside_an_unrun_call_keeps_the_incomplete_rail(tmp_path, monkeypatch, calls):
    """A blank completed refusal with no call answers; beside any offered call the rail never runs, or a finish_task
    whose request is refused, it stays incomplete (no answer is recorded, nothing executes); the reply's one
    adopted finish_task still selects as before."""
    tool_calls = {"none": [], "own_finish": finish("Two salads.")["tool_calls"],
                  "refused_finish": finish("Two salads.", action=[])["tool_calls"],  # offered, its request refused
                  "unrun": [{"id": "c1", "type": "function", "function": {"name": "shell", "arguments": "{}"}}]}[calls]
    reply = ({"role": "assistant", "content": [{"type": "refusal", "refusal": ""}], "tool_calls": tool_calls}, {})
    llm, text, usage, trace = _walk_loop(tmp_path, monkeypatch, BODY_ERROR, reply, fallbacks="")
    assert llm.calls == 2 and not trace.get("tool_calls")
    if calls == "own_finish":
        assert text == "Two salads." and trace["forced_completion_request"]["status"] == "selected"
    else:
        answered = calls == "none"
        if calls == "refused_finish":
            assert trace["forced_completion_request"]["status"] == "refused"
        assert (_refusal_final_answer(text, trace) is not None) is answered
        assert (usage.get("terminal_origin") == "model_final") is answered


@pytest.mark.parametrize("words", ["", " \n"])
def test_a_forced_refusal_keeps_its_exact_blank_words_bound_without_another_send(tmp_path, monkeypatch, words):
    reply = ({"role": "assistant", "content": [{"type": "refusal", "refusal": words}]}, {})
    llm, text, usage, trace = _walk_loop(tmp_path, monkeypatch, BODY_ERROR, reply, fallbacks="")
    assert llm.calls == 2 and text == words and usage["terminal_origin"] == "model_final"
    assert _refusal_final_answer(text, trace) == "" and selected_final_answer(text, trace) == ""


# Two ordinary runs around a refusal: the first holds a whole marker, the refusal's own words end with another.
SPLIT_MARKER = [{"type": "text", "text": "Menu:\nFINAL ANSWER: two salads\n"},
                {"type": "refusal", "refusal": "FINAL ANSWER: not this"}, {"type": "text", "text": "\nEnjoy."}]
GLUED_MARKER = [{"type": "text", "text": "FINAL ANS"}, {"type": "refusal", "refusal": ""},
                {"type": "text", "text": "WER: picnic"}]


@pytest.mark.parametrize("forced", [False, True])
@pytest.mark.parametrize(("parts", "answer"), [(SPLIT_MARKER, "two salads"), (GLUED_MARKER, "")])
def test_a_final_answer_comes_only_from_one_ordinary_run_on_either_exit(tmp_path, monkeypatch, forced, parts, answer):
    """The visible words glue the runs (and the refusal's words) together; the selection never reads them so."""
    reply = ({"role": "assistant", "content": parts}, {})
    script = (BODY_ERROR, reply) if forced else (reply,)
    llm, text, _usage, trace = _walk_loop(tmp_path, monkeypatch, *script, fallbacks="")
    assert llm.calls == len(script) and selected_final_answer(text, trace) == answer


@pytest.mark.parametrize(("parts", "visible"), [
    (MIXED_REFUSAL_PARTS, MIXED_REFUSAL_TEXT),
    (GLUED_MARKER, "FINAL ANSWER: picnic"),
    (SPLIT_MARKER, "Menu:\nFINAL ANSWER: two salads\nFINAL ANSWER: not this\nEnjoy."),
])
def test_later_provider_failure_keeps_multipart_tool_history_only_as_salvage(tmp_path, monkeypatch, parts, visible):
    def read_round(content, call_id):
        call = {"id": call_id, "type": "function", "function": {"name": "chat_history", "arguments": "{}"}}
        return {"role": "assistant", "content": content, "tool_calls": [call]}, {"response_finish_reason": "tool_calls"}

    script = (read_round("Earlier narration.", "old-read"), read_round(parts, "new-read"), BODY_ERROR, FORCED_ERROR)
    llm, text, usage, trace = _walk_loop(tmp_path, monkeypatch, *script, fallbacks="")
    assert llm.calls == len(script)  # existing provider-failure grace call, no new one
    assert [row["tool_call_id"] for row in trace["tool_calls"]] == ["old-read", "new-read"]
    assert any(row.get("tool_call_id") == "new-read" for row in llm.requests[2]["messages"])
    assert text == visible and usage["terminal_origin"] == "host_salvage"
    assert usage["reason_code"] == "provider_unavailable" and usage["execution_status"] == "infra_failed"
    assert selected_final_answer(text, trace) == ""  # no glued/refusal marker; the old ordinary marker preceded a tool
    assert "forced_completion_request" not in trace and "task_completion" not in trace


@pytest.mark.parametrize("parts", [
    [{"type": "text", "text": '{"action": "finish", '}, {"type": "refusal", "refusal": ""},
     {"type": "text", "text": '"answer": "picnic"}'}],  # a control body glued across a refusal
    [{"type": "text", "text": "Menu."}, {"type": "refusal", "refusal": '{"action": "finish", "answer": "picnic"}'}],
])
def test_a_forced_refusal_turns_neither_glued_nor_refused_words_into_a_request(tmp_path, monkeypatch, parts):
    reply = ({"role": "assistant", "content": parts}, {})
    llm, text, _usage, trace = _walk_loop(tmp_path, monkeypatch, BODY_ERROR, reply, fallbacks="")
    assert llm.calls == 2 and "forced_completion_request" not in trace and text != "picnic"
    assert selected_final_answer(text, trace) == ""


@pytest.mark.parametrize("words", ["", " \n"])
def test_handle_task_stores_blank_refusal_words_exact_with_the_hosts_notice_beside(tmp_path, monkeypatch, words):
    from ouroboros import loop
    from ouroboros.agent import _resolve_intentional_empty
    from ouroboros.task_finalization import EMPTY_REFUSAL_NOTICE
    from ouroboros.task_results import load_task_result
    from tests._agent_entry_helpers import body_agent

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    agent = body_agent(tmp_path, monkeypatch, [{"role": "user", "content": "The picnic menu, please."}])
    models = []

    def generate(_llm, _messages, model, *_args, **_kwargs):  # the generation seam: one received refusal
        models.append(model)
        return {"role": "assistant", "content": [{"type": "refusal", "refusal": words}]}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", generate)
    events = agent.handle_task({"id": "blank-refusal", "type": "task", "chat_id": 1, "_is_direct_chat": True,
                                "text": "The picnic menu, please.", "_skip_post_task_synthesis": True})
    stored = load_task_result(tmp_path, "blank-refusal")
    assert len(models) == 1  # no resend and no other model
    assert stored["result"] == words and stored["terminal_origin"] == "model_final"
    assert stored["terminal_host_notice"] == EMPTY_REFUSAL_NOTICE  # the host's words beside, never inside, the answer
    assert stored["loop_outcome"]["final_answer"] == "" and stored["loop_outcome"]["reason_code"] != "empty_final_text"
    finals = [event for event in events if event.get("type") == "send_message"]
    assert finals and all(EMPTY_REFUSAL_NOTICE not in e["text"] and "empty response" not in e["text"] for e in finals)
    # A trace that binds other bytes legalizes no ordinary empty answer.
    usage = {}
    other = {"delivery_candidate": {"refusal_final_answer": "", "content_sha256": "0" * 64}}
    assert _resolve_intentional_empty(None, usage, words, other)[0] is True and "terminal_host_notice" not in usage


def test_a_forced_mixed_refusal_records_its_words_never_its_unrun_calls(tmp_path, monkeypatch):
    calls = [{"id": "c1", "type": "function", "function": {"name": "shell", "arguments": '{"cmd": "ls"}'}}]
    reply = ({"role": "assistant", "content": MIXED_REFUSAL_PARTS, "tool_calls": calls,
              "nativeContinuation": {"route": {"source": "codex"}}}, {})
    seen = {}
    llm, text, _usage, trace = _walk_loop(tmp_path, monkeypatch, BODY_ERROR, reply, fallbacks="", seen=seen)
    assert llm.calls == 2 and selected_final_answer(text, trace) == ""  # the refusal's words select nothing
    transcript = seen["tools"]._ctx.messages
    rows = [row for row in transcript if row.get("content") == MIXED_REFUSAL_PARTS]
    assert rows and all(row["role"] == "assistant" and "tool_calls" not in row and "nativeContinuation" not in row
                        for row in rows)  # its words in order; no call reads as made
    assert not trace.get("tool_calls") and all(row.get("role") != "tool" for row in transcript)


def test_a_forced_refusals_own_finish_call_keeps_its_path_and_reads_as_unrun(tmp_path, monkeypatch):
    reply = ({"role": "assistant", "content": MIXED_REFUSAL_PARTS, "tool_calls": finish("Two salads.")["tool_calls"]}, {})
    seen = {}
    llm, text, _usage, trace = _walk_loop(tmp_path, monkeypatch, BODY_ERROR, reply, fallbacks="", seen=seen)
    assert llm.calls == 2 and text == "Two salads."  # the reply's own control action selects, as before
    assert trace["forced_completion_request"]["status"] == "selected"
    rows = [row for row in seen["tools"]._ctx.messages if row.get("content") == MIXED_REFUSAL_PARTS]
    assert rows and all("tool_calls" not in row for row in rows) and not trace.get("tool_calls")


@pytest.mark.parametrize("failed", [False, True])
def test_forced_call_returns_the_refusal_without_another_send(tmp_path, monkeypatch, failed):
    error = ClaudexorContentRefused({"code": "provider_content_refused", "message": ClaudexorContentRefused.notice})
    reply = (lambda: error) if failed else ({"role": "assistant", "content": MIXED_REFUSAL_PARTS}, {})
    llm, text, usage, trace = _walk_loop(tmp_path, monkeypatch, BODY_ERROR, reply, fallbacks="")
    assert llm.calls == 2  # initial ordinary failure, then this one forced call
    assert all(request["accepts_typed_refusals"] for request in llm.requests)
    assert text == (error.notice if failed else MIXED_REFUSAL_TEXT)
    assert selected_final_answer(text, trace) == ""
    if failed:
        assert usage["reason_code"] == "provider_content_refused"
