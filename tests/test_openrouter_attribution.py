from __future__ import annotations

import urllib.request

import pytest

from ouroboros.openrouter_attribution import OPENROUTER_APP_HEADERS
from scripts.run_external_review import _probe_model_for_key


def test_external_review_probe_uses_canonical_app_attribution(monkeypatch):
    import httpx

    captured = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "pong"}}]}

    def post(_url, **kwargs):
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr(httpx, "post", post)

    assert _probe_model_for_key("secret", "openai/gpt-5.6-sol")[0] is True
    assert {
        key: captured["headers"][key] for key in OPENROUTER_APP_HEADERS
    } == OPENROUTER_APP_HEADERS


def test_gaia_judge_uses_canonical_app_attribution(monkeypatch):
    import devtools.benchmarks.gaia.audit_leakage as audit

    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"{\\"verdict\\":\\"clean\\",\\"rationale\\":\\"ok\\"}"}}]}'

    def urlopen(req, timeout=0):
        captured["headers"] = req.headers
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    result = audit._judge("sample", "answer", [], "openai/gpt-5.6-luna", "secret")
    assert result["verdict"] == "clean"
    actual_headers = {key.lower(): value for key, value in captured["headers"].items()}
    assert all(
        actual_headers.get(key.lower()) == value
        for key, value in OPENROUTER_APP_HEADERS.items()
    )


# --------------------------------------------------------------------------- OpenAI-compatible endpoint on OpenRouter
# Every case below goes through the public client (``LLMClient.chat``/``chat_async``/
# ``probe_provider_readiness``) and the real openai SDK; only the socket is a recording
# ``httpx`` peer. Keys are dummies; nothing reaches a provider.

_OR = "https://openrouter.ai/api/v1"
_MESSAGES = [{"role": "user", "content": "hi"}]
_COMPLETION = {"id": "c", "object": "chat.completion", "created": 0, "model": "vendor/model",
               "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "ok"}}],
               "usage": {"prompt_tokens": 3, "completion_tokens": 1}}


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    import ouroboros.pricing as pricing
    import ouroboros.request_wire_contract as wire
    from ouroboros import usage_accounting as ua
    from ouroboros.llm import LLMClient

    monkeypatch.setattr(wire, "canonical_wire_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(pricing, "estimate_cost_optional", lambda *a, **k: None)
    monkeypatch.setattr(ua, "_reservation_cost", lambda request: 1.0)
    monkeypatch.setattr(LLMClient, "_get_supported_parameters", lambda *a, **k: None)
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="attr-task", root_task_id="attr-task")):
        yield tmp_path


def _configure(monkeypatch, base_url, connection=""):
    """The default compatible endpoint, or a named compatible connection, at ``base_url``."""
    import json

    from ouroboros.provider_models import CONNECTIONS_KEY

    if connection:
        monkeypatch.setenv(CONNECTIONS_KEY, json.dumps([{"kind": "openai-compatible", "name": connection,
                                                         "api_key": "dummy-key", "base_url": base_url}]))
        return f"openai-compatible@{connection}::vendor/model"
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", base_url)
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "dummy-key")
    return "openai-compatible::vendor/model"


def _peer(sent, *, redirect_from="", location="https://elsewhere.example/v1/chat/completions"):
    import httpx

    def handler(request):
        sent.append(request)
        if request.url.host == redirect_from and request.url.path.endswith("/chat/completions"):
            return httpx.Response(307, headers={"Location": location})
        return httpx.Response(200, json=_COMPLETION)

    return handler


def _keepalive_wire(monkeypatch, handler):
    """Production clients (keepalive and no-proxy) over a recording transport."""
    import httpx

    from ouroboros import net_transport

    monkeypatch.setattr(net_transport, "env_proxies_configured", lambda: False)
    monkeypatch.setattr(net_transport, "remote_httpx_transport", lambda *_a, **_kw: httpx.MockTransport(handler))


def _socket_wire(monkeypatch, handler):
    """Whatever client the SDK or httpx builds itself: only the connection layer records."""
    import httpx

    async def handle_async(_transport, request):
        await request.aread()
        return handler(request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda _transport, request: handler(request))
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_async)


def _chat(model, delivery, client=None):
    import asyncio

    from ouroboros.llm import LLMClient

    client = client or LLMClient()
    call = dict(messages=list(_MESSAGES), model=model, max_tokens=16, no_proxy=delivery.startswith("no_proxy"))
    if delivery.endswith("async"):
        return client, asyncio.run(client.chat_async(**call))
    return client, client.chat(**call)


def _identity(request):
    return {key: request.headers.get(key) for key in OPENROUTER_APP_HEADERS}


_ABSENT = dict.fromkeys(OPENROUTER_APP_HEADERS)
_DELIVERIES = ["sync", "async", "no_proxy_sync", "no_proxy_async"]


_LOCAL = "http://localhost:11434/v1"
_EDGE_ENDPOINTS = [
    ("https://OpenRouter.ai:443/api/v1/", True),
    ("http://openrouter.ai/api/v1", False),
    ("https://openrouter.ai:8443/api/v1", False),
    ("https://user:secret@openrouter.ai/api/v1", False),
    ("https://openrouter.ai.example.com/api/v1", False),
    ("https://proxy.example.com/openrouter.ai/api/v1", False),
]


# The endpoint is judged in the target before any delivery: OpenRouter and one ordinary
# endpoint cross every delivery; each edge URL qualifies the predicate once.
@pytest.mark.parametrize("base_url,attributed,delivery", [
    *[(url, attributed, delivery) for url, attributed in [(_OR, True), (_LOCAL, False)] for delivery in _DELIVERIES],
    *[(url, attributed, "sync") for url, attributed in _EDGE_ENDPOINTS],
])
def test_compatible_endpoint_sends_app_identity_only_to_openrouter(
        isolated, monkeypatch, base_url, attributed, delivery):
    sent = []
    _keepalive_wire(monkeypatch, _peer(sent))
    model = _configure(monkeypatch, base_url)
    client, (msg, _usage) = _chat(model, delivery)
    client, (again, _usage) = _chat(model, delivery, client)  # the cached client serves the repeat
    assert msg["content"] == again["content"] == "ok" and len(sent) == 2
    for request in sent:
        assert _identity(request) == (OPENROUTER_APP_HEADERS if attributed else _ABSENT)
        assert "x-openrouter-metadata" not in request.headers  # identity only; the lane keeps its routing
    cached = client._async_remote_clients if delivery == "async" else client._remote_clients
    assert len(cached) == (1 if delivery in ("sync", "async") else 0)


@pytest.mark.parametrize("delivery", _DELIVERIES)
@pytest.mark.parametrize("base_url,attributed", [(_OR, True), ("https://compat.example/v1", False)])
def test_named_compatible_connection_follows_the_same_rule(isolated, monkeypatch, base_url, attributed, delivery):
    sent = []
    _keepalive_wire(monkeypatch, _peer(sent))
    _chat(_configure(monkeypatch, base_url, connection="orc"), delivery)
    assert len(sent) == 1 and _identity(sent[0]) == (OPENROUTER_APP_HEADERS if attributed else _ABSENT)


@pytest.mark.parametrize("delivery", _DELIVERIES)
@pytest.mark.parametrize("location", ["https://elsewhere.example/v1/chat/completions",
                                      "https://openrouter.ai/api/v1/other/chat/completions"],
                         ids=["cross_origin", "same_origin"])
def test_attributed_compatible_endpoint_follows_no_redirect(isolated, monkeypatch, location, delivery):
    sent = []
    _keepalive_wire(monkeypatch, _peer(sent, redirect_from="openrouter.ai", location=location))
    with pytest.raises(Exception):
        _chat(_configure(monkeypatch, _OR), delivery)
    assert sent and all(str(request.url) == _OR + "/chat/completions" for request in sent)


def test_without_the_refusal_the_redirect_would_carry_the_identity(isolated, monkeypatch):
    """Negative control: the attributed target with redirects followed hands its identity elsewhere."""
    from ouroboros.llm import LLMClient

    sent = []
    _keepalive_wire(monkeypatch, _peer(sent, redirect_from="openrouter.ai"))
    _configure(monkeypatch, _OR)
    target = LLMClient()._resolve_remote_target("openai-compatible::vendor/model")
    assert target["follow_redirects"] is False
    unguarded = {key: value for key, value in target.items() if key != "follow_redirects"}
    LLMClient()._new_remote_client(unguarded).chat.completions.create(model="vendor/model", messages=_MESSAGES)
    assert [request.url.host for request in sent] == ["openrouter.ai", "elsewhere.example"]
    assert _identity(sent[1]) == OPENROUTER_APP_HEADERS


@pytest.mark.parametrize("delivery", ["sync", "async"])
def test_unattributed_compatible_endpoint_keeps_following_redirects(isolated, monkeypatch, delivery):
    sent = []
    _keepalive_wire(monkeypatch, _peer(sent, redirect_from="compat.example"))
    _, (msg, _usage) = _chat(_configure(monkeypatch, "https://compat.example/v1"), delivery)
    assert msg["content"] == "ok"
    assert [request.url.host for request in sent] == ["compat.example", "elsewhere.example"]
    assert all(_identity(request) == _ABSENT for request in sent)


@pytest.mark.parametrize("delivery", ["sync", "async"])
@pytest.mark.parametrize("failure", ["missing_default_client", "rejected_options"])
def test_an_sdk_without_a_redirect_refusing_client_sends_unlabelled(isolated, monkeypatch, caplog, failure, delivery):
    """The SDK's own client carries the first request on the same route, key and endpoint; only the
    app identity is left off. No retry, no other route, one diagnostic per client construction."""
    import logging

    import openai

    def rejected(**_kwargs):
        raise TypeError("SDK client does not accept the requested options")

    name = "DefaultAsyncHttpxClient" if delivery == "async" else "DefaultHttpxClient"
    monkeypatch.setattr(openai, name, None if failure == "missing_default_client" else rejected)
    sent = []
    _socket_wire(monkeypatch, _peer(sent))
    model = _configure(monkeypatch, _OR)
    with caplog.at_level(logging.WARNING, logger="ouroboros.llm_routing"):
        client, (msg, _usage) = _chat(model, delivery)
        _chat(model, delivery, client)
    assert msg["content"] == "ok" and len(sent) == 2
    for request in sent:
        assert str(request.url) == _OR + "/chat/completions"
        assert request.headers["authorization"] == "Bearer dummy-key" and _identity(request) == _ABSENT
    notes = [record.getMessage() for record in caplog.records if record.name == "ouroboros.llm_routing"]
    assert len(notes) == 1 and "attribution omitted" in notes[0]
    assert "openrouter.ai" not in notes[0] and "dummy-key" not in notes[0]
    # The shared header owner is untouched: the next resolution still asks for the identity.
    assert client._resolve_remote_target(model)["default_headers"] == OPENROUTER_APP_HEADERS


@pytest.mark.parametrize("failure", ["missing_default_client", "rejected_options"])
def test_routes_without_the_refusal_keep_their_headers_on_that_sdk(isolated, monkeypatch, failure):
    import openai

    def rejected(**_kwargs):
        raise TypeError("SDK client does not accept the requested options")

    monkeypatch.setattr(openai, "DefaultHttpxClient", None if failure == "missing_default_client" else rejected)
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-key")
    sent = []
    _socket_wire(monkeypatch, _peer(sent))
    _chat("openai/gpt-5.6-sol", "sync")  # the named OpenRouter route keeps its transport unchanged
    assert len(sent) == 1 and _identity(sent[0]) == OPENROUTER_APP_HEADERS


@pytest.mark.parametrize("bundle", [False, True], ids=["proxy", "proxy_and_extra_ca"])
@pytest.mark.parametrize("delivery", ["sync", "async"])
def test_proxied_install_still_refuses_redirects_with_identity(isolated, monkeypatch, bundle, delivery):
    """A proxy-routed install keeps the SDK Default client's proxy mounts, still holding the refusal."""
    import certifi

    from ouroboros import net_transport

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    if bundle:
        monkeypatch.setattr(net_transport, "extra_ca_bundle", lambda: certifi.where())
    sent = []
    _socket_wire(monkeypatch, _peer(sent, redirect_from="openrouter.ai"))
    assert net_transport.env_proxies_configured()
    with pytest.raises(Exception):
        _chat(_configure(monkeypatch, _OR), delivery)
    assert len(sent) == 1 and str(sent[0].url) == _OR + "/chat/completions"
    assert _identity(sent[0]) == OPENROUTER_APP_HEADERS


def test_named_openrouter_route_wire_carries_categories(isolated, monkeypatch):
    sent = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-key")
    _keepalive_wire(monkeypatch, _peer(sent))
    for delivery in _DELIVERIES:
        _chat("openai/gpt-5.6-sol", delivery)
    assert len(sent) == 4 and {request.url.host for request in sent} == {"openrouter.ai"}
    for request in sent:
        assert _identity(request) == {"HTTP-Referer": "https://ouroboros-agent.ai/", "X-OpenRouter-Title": "Ouroboros",
                                      "X-OpenRouter-Categories": "personal-agent,programming-app"}


@pytest.mark.parametrize("base_url,attributed", [(_OR, True), ("https://compat.example/v1", False)])
def test_provider_key_probe_uses_the_same_compatible_client(monkeypatch, base_url, attributed):
    from ouroboros.llm import LLMClient

    sent = []
    _keepalive_wire(monkeypatch, _peer(sent, redirect_from="openrouter.ai"))
    settings = {"OPENAI_COMPATIBLE_BASE_URL": base_url, "OPENAI_COMPATIBLE_API_KEY": "dummy-key"}
    result = LLMClient().probe_provider_readiness("openai-compatible::vendor/model", settings=settings, timeout=5)
    assert result["ok"] is not attributed  # the attributed probe sees the 307 instead of following it
    assert sent[0].headers["authorization"] == "Bearer dummy-key"
    assert _identity(sent[0]) == (OPENROUTER_APP_HEADERS if attributed else _ABSENT)
    assert len(sent) == 1
