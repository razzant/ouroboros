"""The owner's effort on every direct Chat Completions route, read off the HTTP body.

The real OpenAI SDK serializes each request and a ``httpx.MockTransport`` peer reads
the JSON it would put on the wire, so these checks cover the shared builder through
its actual consumers: sync, async, streaming and no-proxy sends. Nothing here calls a
vendor. The refusal bodies below are SYNTHETIC — constructed for these tests, not
recorded provider responses — and prove only the host's own adaptation rule: a
rejection naming the field adapts that exact route, a nonspecific 400 adapts nothing.
Carriage is not evidence that a provider applied the effort; the reported effort
stays unknown until a provider reports one.
"""

import asyncio
import json

import httpx
import openai
import pytest

from ouroboros.llm import LLMClient
from tests.test_transport_b_stream_deadlines import chunk, completion, sse
from tests.test_transport_b_stream_deadlines import isolated as isolated

ROUTES = {
    "openai-compatible": ("openai-compatible::glm-5", {
        "OPENAI_COMPATIBLE_API_KEY": "compatible-fixture-key",
        "OPENAI_COMPATIBLE_BASE_URL": "https://vllm.invalid/v1"}),
    "minimax": ("minimax::MiniMax-M3.1", {
        "MINIMAX_API_KEY": "minimax-fixture-key", "MINIMAX_REGION": "global_en"}),
    "cloudru": ("cloudru::zai-org/GLM-4.7", {
        "CLOUDRU_FOUNDATION_MODELS_API_KEY": "cloudru-fixture-key",
        "CLOUDRU_FOUNDATION_MODELS_BASE_URL": "https://cloudru.invalid/v1"}),
    # Two models on one local Ollama-shaped endpoint: each is its own exact route.
    "ollama": ("openai-compatible::llama3.2:latest", {
        "OPENAI_COMPATIBLE_API_KEY": "ollama", "OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:11434/v1"}),
    "ollama-thinking": ("openai-compatible::qwen3:latest", {
        "OPENAI_COMPATIBLE_API_KEY": "ollama", "OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:11434/v1"}),
}
MESSAGES = [{"role": "user", "content": "hello"}]
UNSUPPORTED_FIELD = {"error": {  # SYNTHETIC: names the field as unsupported
    "type": "invalid_request_error", "param": "reasoning_effort",
    "message": "Unrecognized request argument supplied: reasoning_effort"}}
NONSPECIFIC = {"error": {"type": "invalid_request_error", "message": "Bad request"}}  # SYNTHETIC
# The envelope Ollama's /v1/chat/completions builds for a model without the thinking capability
# (server/routes.go ChatHandler, re-encoded by middleware/openai.go BaseWriter.writeError, read at
# ollama/ollama@8718c2bd2363c3f9f422d82bbad322bc2b791d60 on 2026-10-14); served here by the mock.
NO_THINKING = {"error": {"message": '"llama3.2:latest" does not support thinking',
                         "type": "invalid_request_error", "param": None, "code": None}}


def _refusal(message, status=400):
    return httpx.Response(status, json={"error": {"message": message, "type": "invalid_request_error"}})


def _answer(payload, script):
    step = script.pop(0) if script else None
    if isinstance(step, httpx.Response):
        return step
    if payload.get("stream"):
        wire = sse(chunk({"role": "assistant", "content": "done"}, "stop", usage=completion()["usage"]))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=wire)
    return httpx.Response(200, json=completion())


@pytest.fixture
def route(monkeypatch):
    """One provider route whose HTTP peer records each JSON body it receives."""
    opened = []

    def make(provider, *, asynchronous=False, no_proxy=False, script=None):
        model, env = ROUTES[provider]
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        client, sent, script = LLMClient(), [], list(script or [])

        def respond(request):
            payload = json.loads(request.content)
            sent.append(payload)
            return _answer(payload, script)

        transport = httpx.MockTransport(respond)
        http = (httpx.AsyncClient if asynchronous else httpx.Client)(transport=transport)
        remote = (openai.AsyncOpenAI if asynchronous else openai.OpenAI)(
            api_key="fixture-not-live", base_url="https://route.invalid/v1", max_retries=0, http_client=http)
        opened.append(remote)
        if no_proxy:
            name = "_make_no_proxy_async_client" if asynchronous else "_make_no_proxy_client"
            monkeypatch.setattr(client, name, lambda *_a, **_kw: (remote, http))
        else:
            monkeypatch.setattr(client, "_get_remote_client", lambda *_a, **_kw: remote)
            monkeypatch.setattr(client, "_get_async_remote_client", lambda *_a, **_kw: remote)
        return client, model, sent

    yield make
    for remote in opened:
        if isinstance(remote, openai.AsyncOpenAI):
            asyncio.run(remote.close())
        else:
            remote.close()


def _chat(client, model, effort, *, asynchronous=False, **kwargs):
    call = dict(messages=list(MESSAGES), model=model, reasoning_effort=effort, max_tokens=64, **kwargs)
    return asyncio.run(client.chat_async(**call)) if asynchronous else client.chat(**call)


@pytest.mark.parametrize("provider", sorted(ROUTES))
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("delivery", ["json", "stream", "no_proxy"])
def test_the_standard_field_reaches_the_wire_on_every_compatible_route(
        isolated, route, provider, asynchronous, delivery):
    client, model, sent = route(provider, asynchronous=asynchronous, no_proxy=delivery == "no_proxy")
    msg, usage = _chat(client, model, "xhigh", asynchronous=asynchronous,
                       stream=delivery == "stream", no_proxy=delivery == "no_proxy")
    assert msg["content"] == "done"
    assert len(sent) == 1 and sent[0]["reasoning_effort"] == "xhigh"
    assert bool(sent[0].get("stream")) is (delivery == "stream")
    # Requested and sent are host facts; what the provider applied stays unknown.
    assert usage["effort"] == {
        "requested": "xhigh", "sent": {"reasoning_effort": "xhigh"}, "sent_state": "explicit",
        "sent_source": "host_candidate", "reported": None, "report_source": None,
    }


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_a_refusal_naming_the_field_adapts_only_that_route(isolated, route, asynchronous):
    refused = httpx.Response(400, json=UNSUPPORTED_FIELD)
    client, model, sent = route("cloudru", asynchronous=asynchronous, script=[refused])
    msg, usage = _chat(client, model, "high", asynchronous=asynchronous)
    assert msg["content"] == "done"
    assert [("reasoning_effort" in body) for body in sent] == [True, False]
    assert usage["effort"]["requested"] == "high"
    assert usage["effort"]["sent"] == {} and usage["effort"]["sent_state"] == "omitted"
    assert usage["effort"]["reported"] is None
    # Another route keeps its own evidence: one endpoint's refusal is not a global "no".
    other, other_model, other_sent = route("openai-compatible", asynchronous=asynchronous)
    _chat(other, other_model, "high", asynchronous=asynchronous)
    assert other_sent[0]["reasoning_effort"] == "high"


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_a_nonspecific_400_neither_drops_effort_nor_resends(isolated, route, asynchronous):
    client, model, sent = route("minimax", asynchronous=asynchronous,
                                script=[httpx.Response(400, json=NONSPECIFIC)])
    with pytest.raises(Exception):
        _chat(client, model, "high", asynchronous=asynchronous)
    assert len(sent) == 1 and sent[0]["reasoning_effort"] == "high"
    retry_client, _model, retry_sent = route("minimax", asynchronous=asynchronous)
    _chat(retry_client, model, "high", asynchronous=asynchronous)
    assert retry_sent[0]["reasoning_effort"] == "high"  # nothing was learned from it


@pytest.mark.parametrize(("stream", "asynchronous"), [
    pytest.param(False, False, id='json-sync'),
    pytest.param(True, True, id='stream-async'),
])
def test_a_model_without_thinking_is_resent_without_the_field_and_that_route_remembers(
        isolated, route, asynchronous, stream):
    client, model, sent = route("ollama", asynchronous=asynchronous, script=[httpx.Response(400, json=NO_THINKING)])
    msg, usage = _chat(client, model, "medium", asynchronous=asynchronous, stream=stream)
    assert msg["content"] == "done"
    assert [("reasoning_effort" in body) for body in sent] == [True, False]
    assert usage["effort"]["requested"] == "medium" and usage["effort"]["sent_state"] == "omitted"
    # Only that success taught the exact route: its next call starts without the field ...
    again, _model, again_sent = route("ollama", asynchronous=asynchronous)
    _chat(again, model, "medium", asynchronous=asynchronous, stream=stream)
    assert len(again_sent) == 1 and "reasoning_effort" not in again_sent[0]
    # ... while another model on the same server keeps the owner's effort.
    other, other_model, other_sent = route("ollama-thinking", asynchronous=asynchronous)
    _chat(other, other_model, "medium", asynchronous=asynchronous, stream=stream)
    assert other_sent[0]["reasoning_effort"] == "medium"


def test_the_learned_omission_expires_with_its_evidence(isolated, route, monkeypatch):
    import datetime

    from ouroboros import request_wire_contract as wire

    client, model, _sent = route("ollama", script=[httpx.Response(400, json=NO_THINKING)])
    _chat(client, model, "medium")
    later = wire.utc_now() + datetime.timedelta(seconds=wire.REQUEST_WIRE_TTL_SEC + 60)
    monkeypatch.setattr(wire, "utc_now", lambda: later)
    again, _model, again_sent = route("ollama")
    _chat(again, model, "medium")
    assert again_sent[0]["reasoning_effort"] == "medium"


def test_a_fallback_that_does_not_succeed_teaches_nothing(isolated, route):
    client, model, sent = route("ollama", script=[httpx.Response(400, json=NO_THINKING),
                                                  httpx.Response(400, json=NONSPECIFIC)])
    with pytest.raises(Exception):
        _chat(client, model, "medium")
    assert [("reasoning_effort" in body) for body in sent] == [True, False]
    again, _model, again_sent = route("ollama")
    _chat(again, model, "medium")
    assert again_sent[0]["reasoning_effort"] == "medium"


@pytest.mark.parametrize("refusal", [
    _refusal("thinking budget exceeds max_tokens"),  # names thinking, refuses no capability
    _refusal('"llama3.2:latest" does not support tools'),  # refuses another capability
    _refusal('"llama3.2:latest" does not support thinking', status=403),  # not a compatibility status
], ids=["thinking_wording", "other_capability", "non_compatibility_status"])
def test_thinking_wording_alone_adapts_nothing(isolated, route, refusal):
    client, model, sent = route("ollama", script=[refusal])
    with pytest.raises(Exception):
        _chat(client, model, "medium")
    assert len(sent) == 1 and sent[0]["reasoning_effort"] == "medium"
