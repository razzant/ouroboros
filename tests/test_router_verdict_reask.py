"""OpenRouter's own final verdict is re-asked once without fallbacks.

The recorded shape (BYOK incident receipts, 2026-10-22): an HTTP-200 stream whose only
frame is OpenRouter's ``402 payment_required`` (a credit check on paid shared capacity),
with ``openrouter_metadata.attempts`` listing twelve Anthropic attempts that each answered
400; the same request with ``provider.only=["anthropic"]`` and ``allow_fallbacks=False``
answered the upstream's own ``400 prompt is too long: 1067828 tokens > 1000000 maximum``. A $0
probe (2026-10-26, an oversized prompt to a 200k-window model through BYOK keys) showed that
``allow_fallbacks=False`` alone lets the router pick one endpoint itself (Amazon Bedrock, paid
shared capacity) and that ``only`` accepts the receipt's provider names ("Amazon Bedrock",
"Google"). Nothing here touches the network: the SDK create call is a scripted fake, and
accounting runs on a temp root.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import openai
import pytest

from ouroboros import usage_accounting as ua
from ouroboros.llm import LLMClient
from ouroboros.llm_attempt import single_physical_send
from ouroboros.llm_routing import _resolve_or_provider
from ouroboros.llm_stream import AssembledResponse, ProviderStreamError
from ouroboros.loop_llm_call import classify_llm_exception
from ouroboros.openrouter_attribution import OPENROUTER_APP_HEADERS
from tests._usage_store_testing import ledger_rows

MODEL = "anthropic/claude-opus-5.5"
OVERFLOW = "prompt is too long: 1067828 tokens > 1000000 maximum"
CREDITS = ("This request's maximum cost exceeds your available credits. "
           "Add credits, or lower max_tokens or prompt size.")


def receipt(*statuses, providers=("Anthropic",)):
    """The router receipt as the incident recorded it (provider display names, statuses only)."""
    return {"requested": MODEL, "strategy": "direct", "summary": f"available=16, attempts={len(statuses)}",
            "attempt": len(statuses), "is_byok": False,
            "attempts": [{"provider": providers[index % len(providers)],
                          "model": "anthropic/claude-opus-5.5-20260921", "status": s}
                         for index, s in enumerate(statuses)]}


def error_frame(code, message, metadata):
    frame = {"id": f"gen-{code}", "model": "unknown", "provider": "unknown",
             "error": {"code": code, "message": message,
                       "metadata": {"error_type": "payment_required" if code == 402 else "invalid_request"}}}
    if metadata is not None:
        frame["openrouter_metadata"] = metadata
    return frame


MASKED = error_frame(402, CREDITS, receipt(*[400] * 12))
UPSTREAM = error_frame(400, OVERFLOW, receipt(*[400] * 12))
ANSWER = {"id": "gen-answer", "choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": "stop"}],
          "usage": {"cost": 0.5, "prompt_tokens": 10, "completion_tokens": 2}}


class Wire:
    """An SSE body that yields the given frames, like the SDK's raw stream response."""

    def __init__(self, *frames):
        self.chunks = [f if isinstance(f, bytes) else ("data: " + json.dumps(f) + "\n\n").encode() for f in frames]
        self.headers = {}

    def iter_bytes(self):
        yield from self.chunks

    async def aiter_bytes(self):
        for chunk in self.chunks:
            yield chunk

    def close(self):
        pass

    async def aclose(self):
        pass


def http_error(status, body):
    """A real SDK status error: ``body`` keeps only ``error``; the read response keeps all."""
    response = httpx.Response(status, json=body, request=httpx.Request(
        "POST", "https://openrouter.ai/api/v1/chat/completions"))
    error_class = openai.BadRequestError if status == 400 else openai.APIStatusError
    return error_class(f"Error code: {status} - {json.dumps(body)}", response=response, body=body.get("error", body))


def target(provider="openrouter"):
    return {"provider": provider, "resolved_model": MODEL, "usage_model": MODEL,
            "base_url": "https://openrouter.ai/api/v1", "api_key": "fixture-key",
            "supports_openrouter_extensions": provider == "openrouter", "supports_generation_cost": True}


def request_kwargs(*, stream=True, routing=None):
    kwargs = {"model": MODEL, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 100,
              "extra_body": {"reasoning": {"effort": "xhigh", "exclude": False},
                             "session_id": "ouroboros-session-fixture",
                             "provider": dict(routing or {"require_parameters": True})}}
    if stream:
        kwargs.update(stream=True, stream_options={"include_usage": True})
    return kwargs


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    import ouroboros.pricing as pricing
    import ouroboros.request_wire_contract as wire

    monkeypatch.setattr(wire, "canonical_wire_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(pricing, "estimate_cost_optional", lambda *a, **k: None)
    monkeypatch.setattr(ua, "_reservation_cost", lambda request: 2.0)
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="reask-task", root_task_id="reask-task")):
        yield tmp_path


def drive(script, *, asynchronous=False, provider="openrouter", kwargs=None):
    """Run one ladder call over scripted replies; return (result or raised error, sends)."""
    script, sends = list(script), []

    def reply(candidate):
        sends.append(candidate)
        item = script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def create_async(**candidate):
        return reply(candidate)

    client = LLMClient()
    payload = kwargs or request_kwargs()
    try:
        if asynchronous:
            result = asyncio.run(client._create_chat_completion_with_retries_async(
                create_async, payload, target(provider)))
        else:
            result = client._create_chat_completion_with_retries(
                lambda **candidate: reply(candidate), payload, target(provider))
    except Exception as exc:  # noqa: BLE001 - the raised error is the outcome under test
        result = exc
    assert not script, "every scripted reply was consumed"
    return result, sends


def routing(send):
    return send["extra_body"]["provider"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_masked_402_is_reasked_once_and_the_upstream_overflow_classifies(isolated, asynchronous):
    outcome, sends = drive([Wire(MASKED), Wire(UPSTREAM)], asynchronous=asynchronous)

    assert len(sends) == 2
    assert "allow_fallbacks" not in routing(sends[0])
    # The re-ask forbids fallbacks and stays on the attempted providers; the rest of the routing stays.
    assert routing(sends[1]) == {"require_parameters": True, "allow_fallbacks": False, "only": ["Anthropic"]}
    assert sends[1]["extra_body"]["session_id"] == sends[0]["extra_body"]["session_id"]
    assert sends[1]["messages"] == sends[0]["messages"]
    assert isinstance(outcome, ProviderStreamError) and outcome.status_code == 400
    classification = classify_llm_exception(outcome)
    assert classification.kind == "context_overflow"
    assert classification.retry_same_request is False
    # Each send is its own physical attempt in the ledger.
    assert len(ledger_rows(isolated)) == 2


@pytest.mark.parametrize("attempts,only", [
    (receipt(400, 400, 400, providers=("Anthropic", "Amazon Bedrock", "Anthropic")), ["Anthropic", "Amazon Bedrock"]),
    ({"attempts": [{"status": 400}, {"provider": " ", "status": 400}]}, None),  # no names: the router's own choice
], ids=["named-providers-in-order", "unnamed"])
def test_the_reask_stays_on_the_attempted_providers(isolated, attempts, only):
    _outcome, sends = drive([Wire(error_frame(402, CREDITS, attempts)), Wire(UPSTREAM)])

    assert len(sends) == 2 and routing(sends[1])["allow_fallbacks"] is False
    assert routing(sends[1]).get("only") == only


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_successful_reask_is_the_calls_answer(isolated, asynchronous):
    outcome, sends = drive([Wire(MASKED), Wire(ANSWER, b"data: [DONE]\n\n")], asynchronous=asynchronous)

    assert len(sends) == 2 and routing(sends[1])["allow_fallbacks"] is False
    assert outcome.model_dump()["choices"][0]["message"]["content"] == "answer"
    rows = ledger_rows(isolated)
    assert len(rows) == 2
    assert rows[1]["state"] == "settled" and rows[1]["cost_usd"] == 0.5 and rows[1]["cost_final"] is True


@pytest.mark.parametrize("metadata", [
    receipt(400, 402),  # an upstream itself answered 402: its own billing verdict
    None,  # no receipt at all
    {"requested": MODEL, "attempt": 0, "summary": "available=16"},  # refused before any upstream
], ids=["402-among-attempts", "no-receipt", "no-attempts"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_genuine_402_stays_quota_without_reask(isolated, asynchronous, metadata):
    outcome, sends = drive([Wire(error_frame(402, CREDITS, metadata))], asynchronous=asynchronous)

    assert len(sends) == 1
    assert isinstance(outcome, ProviderStreamError)
    assert classify_llm_exception(outcome).kind == "quota_exhausted"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_final_verdict_without_an_http_status_is_not_compared(isolated, asynchronous):
    statusless = {**MASKED, "error": {"message": CREDITS, "type": "overloaded_error"}}
    outcome, sends = drive([Wire(statusless)], asynchronous=asynchronous)

    assert len(sends) == 1 and outcome.stream_rejected is True


def test_typed_refusals_are_never_reasked():
    from ouroboros.llm_attempt import ProcessingNotStarted, ProviderPolicyRefusal

    masked = http_error(402, {key: MASKED[key] for key in ("error", "openrouter_metadata")})
    refusal = ProviderPolicyRefusal("refused before dispatch")
    refusal.status_code, refusal.response = masked.status_code, masked.response
    assert LLMClient._router_reask_kwargs(target(), request_kwargs(), failure=masked) is not None
    for typed in (ProcessingNotStarted(masked, reason="capacity"), refusal):
        assert LLMClient._router_reask_kwargs(target(), request_kwargs(), failure=typed) is None


@pytest.mark.parametrize("asynchronous", [False, True])
def test_no_reask_when_fallbacks_are_already_forbidden(isolated, asynchronous):
    kwargs = request_kwargs(routing={"require_parameters": True, "allow_fallbacks": False})
    outcome, sends = drive([Wire(MASKED)], asynchronous=asynchronous, kwargs=kwargs)

    assert len(sends) == 1
    assert classify_llm_exception(outcome).kind == "quota_exhausted"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_no_reask_on_a_non_openrouter_route(isolated, asynchronous):
    outcome, sends = drive([Wire(MASKED)], asynchronous=asynchronous, provider="openai-compatible")

    assert len(sends) == 1 and isinstance(outcome, ProviderStreamError)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_no_reask_after_model_output_arrived(isolated, asynchronous):
    partial = {"id": "gen-402", "choices": [{"index": 0, "delta": {"content": "partial"}}]}
    outcome, sends = drive([Wire(partial, MASKED)], asynchronous=asynchronous)

    assert len(sends) == 1
    assert outcome.stream_receipt["model_output_received"] is True


@pytest.mark.parametrize("asynchronous", [False, True])
def test_no_reask_inside_a_single_physical_send_scope(isolated, asynchronous):
    with single_physical_send(lambda reservation: None):
        outcome, sends = drive([Wire(MASKED)], asynchronous=asynchronous)

    assert len(sends) == 1
    assert classify_llm_exception(outcome).kind == "quota_exhausted"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_an_http_status_verdict_reads_the_receipt_from_the_response(isolated, asynchronous):
    masked = {key: MASKED[key] for key in ("error", "openrouter_metadata")}
    upstream = {key: UPSTREAM[key] for key in ("error", "openrouter_metadata")}
    outcome, sends = drive([http_error(402, masked), http_error(400, upstream)],
                           asynchronous=asynchronous, kwargs=request_kwargs(stream=False))

    assert len(sends) == 2 and routing(sends[1])["allow_fallbacks"] is False
    assert isinstance(outcome, openai.BadRequestError)
    assert classify_llm_exception(outcome).kind == "context_overflow"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_an_http_200_body_verdict_is_reasked_and_answered(isolated, asynchronous):
    masked = AssembledResponse({**{key: MASKED[key] for key in ("id", "error", "openrouter_metadata")},
                                "choices": None})
    answer = AssembledResponse({"id": "gen-answer", "usage": {"cost": 0.5, "prompt_tokens": 10, "completion_tokens": 2},
                                "choices": [{"message": {"role": "assistant", "content": "answer"},
                                             "finish_reason": "stop"}]})
    outcome, sends = drive([masked, answer], asynchronous=asynchronous, kwargs=request_kwargs(stream=False))

    assert len(sends) == 2 and routing(sends[1])["allow_fallbacks"] is False
    assert outcome.model_dump()["choices"][0]["message"]["content"] == "answer"


def test_stream_error_retains_the_router_receipt():
    error = ProviderStreamError(MASKED)
    assert error.openrouter_metadata == MASKED["openrouter_metadata"]
    assert error.openrouter_metadata is not MASKED["openrouter_metadata"]
    assert ProviderStreamError({"error": {"code": 402, "message": CREDITS}}).openrouter_metadata is None


def test_openrouter_targets_ask_for_the_routing_receipt():
    client = LLMClient()
    headers = client._resolve_remote_target(MODEL)["default_headers"]
    assert headers["X-OpenRouter-Metadata"] == "enabled"
    assert {key: headers[key] for key in OPENROUTER_APP_HEADERS} == OPENROUTER_APP_HEADERS
    assert "X-OpenRouter-Metadata" not in client._resolve_remote_target(
        "anthropic::claude-opus-5.5")["default_headers"]


@pytest.fixture
def global_routing(monkeypatch):
    """The global ``OUROBOROS_OR_PROVIDER`` seen by the routing resolver, other keys untouched."""
    import ouroboros.llm_routing as llm_routing

    real = llm_routing.runtime_setting
    value = json.dumps({"data_collection": "deny", "allow_fallbacks": True})
    monkeypatch.setattr(llm_routing, "runtime_setting", lambda key, *args, **kwargs: (
        value if key == "OUROBOROS_OR_PROVIDER" else real(key, *args, **kwargs)))
    return json.loads(value)


def provider_block(monkeypatch, model, routing_value, messages=None):
    client = LLMClient()
    monkeypatch.setattr(client, "_get_supported_parameters", lambda _model_id: None)
    route = {**target(), "resolved_model": model, "usage_model": model}
    if routing_value is not None:
        route["provider_routing"] = routing_value
    kwargs = client._build_remote_kwargs(
        route, messages or [{"role": "user", "content": "hello"}], "medium", 64, "auto", None, None)
    return kwargs["extra_body"].get("provider", {})


@pytest.mark.parametrize("routing_value,expected", [
    ("repro", {"data_collection": "deny", "allow_fallbacks": False}),
    ({"only": ["anthropic"]}, {"data_collection": "deny", "allow_fallbacks": True, "only": ["anthropic"]}),
    ('{"order": ["anthropic"], "allow_fallbacks": false}',
     {"data_collection": "deny", "allow_fallbacks": False, "order": ["anthropic"]}),
    (None, {"data_collection": "deny", "allow_fallbacks": True}),
], ids=["preset", "object", "json-text", "built-in"])
def test_connection_routing_merges_over_the_global_setting(monkeypatch, global_routing, routing_value, expected):
    assert provider_block(monkeypatch, MODEL, routing_value) == {"require_parameters": True, **expected}
    # The global value itself, and so every other connection's requests, are unchanged.
    assert _resolve_or_provider() == global_routing


def test_connection_routing_never_lifts_the_sealed_reasoning_pin(monkeypatch, global_routing):
    sealed = [{"role": "user", "content": "inspect"},
              {"role": "assistant", "content": "thinking",
               "reasoning_details": [{"type": "reasoning.encrypted", "data": "blob"}]}]
    block = provider_block(monkeypatch, "deepseek/deepseek-v4-flash-0731", {"allow_fallbacks": True}, sealed)
    assert block["allow_fallbacks"] is False
