"""A keyless OpenAI-compatible endpoint is reached with no credential by every real SDK consumer.

Setup documents the dedicated endpoint's blank key as "no auth" (``provider_models.compatible_endpoint``),
while openai 2.x refuses a blank key before any HTTP: construction raises "Missing credentials" and each
request "Could not resolve authentication method". The oracle here is the endpoint's own record of every
request it received: its path and whether an ``Authorization`` header arrived.
"""
from __future__ import annotations

import asyncio
import json
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import openai
import pytest

import ouroboros.usage_accounting as ua
from ouroboros.llm import LLMClient
from ouroboros.net_transport import openai_client_class

pytestmark = pytest.mark.serial

MODEL = "openai-compatible::stub-model"
MESSAGES = [{"role": "user", "content": "Reply OK"}]
_AMBIENT = ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY", "OPENAI_CUSTOM_HEADERS", "OPENAI_BASE_URL",
            "OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_BASE_URL")


class _Endpoint:
    """A loopback OpenAI-compatible server recording each request's path and Authorization header."""

    def __init__(self):
        self.requests: list[dict] = []
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - stdlib callback
                endpoint._record(self)
                endpoint._send(self, {"object": "list", "data": [{"id": "stub-model", "object": "model"}]})

            def do_POST(self):  # noqa: N802
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                endpoint._record(self)
                streaming = payload.get("stream") is True
                message = {"role": "assistant", "content": "OK"}
                answer = {"id": "keyless-stub", "object": "chat.completion", "created": 0, "model": "stub-model",
                          "choices": [{"index": 0, "finish_reason": "stop",
                                       ("delta" if streaming else "message"): message}],
                          "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
                endpoint._send(self, answer, streaming=streaming)

            def log_message(self, *_args):
                return

        class LoopbackServer(ThreadingHTTPServer):
            def server_bind(self):  # no reverse DNS for a loopback fixture
                socketserver.TCPServer.server_bind(self)
                self.server_name, self.server_port = self.server_address[:2]

        self.server = LoopbackServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, name="keyless-endpoint", daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def _record(self, handler) -> None:
        self.requests.append({"method": handler.command, "path": handler.path,
                              "authorization": handler.headers.get("Authorization")})

    @staticmethod
    def _send(handler, answer: dict, *, streaming: bool = False) -> None:
        if streaming:
            body, kind = f"data: {json.dumps(answer)}\n\ndata: [DONE]\n\n".encode(), "text/event-stream"
        else:
            body, kind = json.dumps(answer).encode(), "application/json"
        handler.send_response(200)
        handler.send_header("Content-Type", kind)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def completions(self) -> list[dict]:
        return [r for r in self.requests if r["method"] == "POST" and r["path"] == "/v1/chat/completions"]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)


@pytest.fixture(autouse=True)
def _no_ambient_sdk_configuration(monkeypatch):
    """The SDK's own environment (an admin key passes its credential check) is set per test only."""
    for name in _AMBIENT:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    import ouroboros.pricing as pricing
    import ouroboros.request_wire_contract as wire

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setattr(wire, "canonical_wire_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(pricing, "estimate_cost_optional", lambda *a, **k: None)
    monkeypatch.setattr(ua, "_reservation_cost", lambda _request: 0.01)
    monkeypatch.setattr(LLMClient, "_get_supported_parameters", lambda *a, **k: None)
    served = _Endpoint()
    try:
        with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="keyless-task", root_task_id="keyless-task")):
            yield served
    finally:
        served.close()
    assert not served.thread.is_alive()


def _chat(client, **options):
    try:
        return client.chat(messages=MESSAGES, model=MODEL, max_tokens=16, timeout=20, **options)[0]["content"]
    finally:
        for sdk in client._remote_clients.values():
            sdk.close()


def _chat_async(client, **options):
    async def call_and_close():
        try:
            return (await client.chat_async(messages=MESSAGES, model=MODEL, max_tokens=16, timeout=20,
                                            **options))[0]["content"]
        finally:
            for sdk in client._async_remote_clients.values():
                await sdk.close()

    return asyncio.run(call_and_close())


CONSUMERS = {
    "chat": lambda client, url: _chat(client),
    "chat-no-proxy": lambda client, url: _chat(client, no_proxy=True),
    "chat-stream": lambda client, url: _chat(client, stream=True),
    "chat-async": lambda client, url: _chat_async(client),
    "chat-async-no-proxy": lambda client, url: _chat_async(client, no_proxy=True),
    "chat-async-stream": lambda client, url: _chat_async(client, stream=True),
    # Settings' provider test: a request-local mapping, a one-shot client and ``with_options``.
    "provider-test": lambda client, url: client.probe_provider_readiness(
        MODEL, settings={"OPENAI_COMPATIBLE_BASE_URL": url}, timeout=20)["ok"],
    # The context-window probe: the cached client and ``with_options``.
}


@pytest.mark.parametrize("consumer", list(CONSUMERS))
def test_every_sdk_consumer_reaches_the_keyless_endpoint_without_a_credential(endpoint, monkeypatch, consumer):
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", endpoint.base_url)
    result = CONSUMERS[consumer](LLMClient(), endpoint.base_url)
    assert result in ("OK", True), result
    assert endpoint.completions(), endpoint.requests
    assert [r["authorization"] for r in endpoint.requests] == [None] * len(endpoint.requests)


@pytest.mark.parametrize("header_name", ["Authorization", "authorization", "AUTHORIZATION"])
def test_no_ambient_sdk_credential_reaches_the_keyless_endpoint(endpoint, monkeypatch, header_name):
    """The SDK reads OPENAI_API_KEY, OPENAI_ADMIN_KEY and OPENAI_CUSTOM_HEADERS itself; none of them
    may turn the owner's keyless endpoint into a recipient of some other credential."""
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", endpoint.base_url)
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai-key")
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "ambient-admin-key")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", f"{header_name}: Bearer ambient-custom")
    client = LLMClient()
    target = client._resolve_remote_target(MODEL)
    assert target["keyless"] is True and target["api_key"] == ""
    sdk = client._get_remote_client(target)
    # The ambient header is live in the client's defaults; the request still carries no credential.
    assert sdk.default_headers[header_name] == "Bearer ambient-custom" and sdk.api_key == ""
    assert _chat(client) == "OK" and _chat_async(LLMClient()) == "OK"
    assert len(endpoint.completions()) == 2
    assert [r["authorization"] for r in endpoint.requests] == [None] * len(endpoint.requests)


@pytest.mark.parametrize("header_name", ["Authorization", "authorization", "AUTHORIZATION"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_keyless_request_headers_cannot_restore_authorization(endpoint, header_name, asynchronous):
    target = {"keyless": True, "api_key": ""}
    sdk = openai_client_class(target, async_client=asynchronous)(base_url=endpoint.base_url, max_retries=0)
    kwargs = {"model": "stub-model", "messages": MESSAGES,
              "extra_headers": {header_name: "Bearer synthetic-request"}}
    if asynchronous:
        async def run():
            try:
                await sdk.chat.completions.create(**kwargs)
            finally:
                await sdk.close()
        asyncio.run(run())
    else:
        try:
            sdk.chat.completions.create(**kwargs)
        finally:
            sdk.close()
    assert endpoint.completions() and all(r["authorization"] is None for r in endpoint.requests)


def test_a_key_on_the_route_is_always_sent(endpoint, monkeypatch):
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", endpoint.base_url)
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "compat-key")
    assert _chat(LLMClient()) == "OK" and _chat_async(LLMClient(), no_proxy=True) == "OK"
    assert LLMClient().probe_provider_readiness(
        MODEL, settings={"OPENAI_COMPATIBLE_BASE_URL": endpoint.base_url,
                         "OPENAI_COMPATIBLE_API_KEY": "compat-key"}, timeout=20)["ok"] is True
    assert [r["authorization"] for r in endpoint.completions()] == ["Bearer compat-key"] * 3


def test_a_route_that_needs_its_key_still_refuses_before_any_http(endpoint, monkeypatch):
    # The legacy pair travels only with OPENAI_API_KEY: without it the lane is unconfigured, not keyless.
    monkeypatch.setenv("OPENAI_BASE_URL", endpoint.base_url)
    client = LLMClient()
    target = client._resolve_remote_target(MODEL)
    assert target["base_url"] == endpoint.base_url and target["keyless"] is False
    with pytest.raises(openai.OpenAIError, match="Missing credentials"):
        client._get_remote_client(target)
    with pytest.raises(Exception, match="Missing credentials"):
        LLMClient().chat(messages=MESSAGES, model=MODEL, max_tokens=16, timeout=20, wait_for_resources=False)
    assert endpoint.requests == []


def test_only_the_dedicated_endpoint_saved_without_a_key_is_keyless():
    client = LLMClient()
    url = "http://127.0.0.1:9/v1"

    def keyless(settings, model=MODEL):
        return client._resolve_remote_target(model, settings=settings).get("keyless")

    assert keyless({"OPENAI_COMPATIBLE_BASE_URL": url}) is True
    assert keyless({"OPENAI_COMPATIBLE_BASE_URL": url, "OPENAI_COMPATIBLE_API_KEY": "k"}) is False
    assert keyless({"OPENAI_BASE_URL": url, "OPENAI_API_KEY": "k"}) is False
    assert keyless({"OPENAI_BASE_URL": url}) is False and keyless({}) is False
    # A key the dedicated endpoint lacks is never borrowed from the legacy pair.
    assert keyless({"OPENAI_COMPATIBLE_BASE_URL": url, "OPENAI_API_KEY": "k"}) is True
    # No other route is ever keyless, whatever its key: those keep the SDK's own refusal.
    for model in ("openrouter::vendor/model", "vendor/model", "openai::m", "anthropic::m", "minimax::m",
                  "cloudru::m", "gigachat::m", "deepseek::m", "zai::m", "claudexor::source=model"):
        assert not keyless({}, model), model


@pytest.mark.parametrize("model", ["openrouter::vendor/model", "openai::m", "minimax::m", "cloudru::m",
                                   "deepseek::m", "zai::m"])
def test_keyed_sdk_routes_without_their_key_keep_the_sdk_refusal(model):
    target = LLMClient()._resolve_remote_target(model, settings={})
    assert target["api_key"] == ""
    assert openai_client_class(target) is openai.OpenAI
    assert openai_client_class(target, async_client=True) is openai.AsyncOpenAI
    with pytest.raises(openai.OpenAIError, match="Missing credentials"):
        LLMClient._new_remote_client(target)


def test_keyless_and_unconfigured_routes_never_share_a_cached_client():
    client = LLMClient()
    keyless = client._resolve_remote_target(MODEL, settings={"OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:9/v1"})
    sdk = client._get_remote_client(keyless)
    assert type(sdk).__name__ == "KeylessOpenAI" and isinstance(sdk, openai.OpenAI)
    assert client._get_remote_client(dict(keyless)) is sdk
    try:
        # Same provider, address and blank key, but a route that needs its key: never the keyless client.
        with pytest.raises(openai.OpenAIError, match="Missing credentials"):
            client._get_remote_client({**keyless, "keyless": False})
        # ``with_options`` rebuilds through the keyless form, never back into the SDK's refusal.
        assert type(sdk.with_options(timeout=5)).__name__ == "KeylessOpenAI"
    finally:
        sdk.close()
