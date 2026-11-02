"""Provider Test sends a kept saved secret only to the endpoint it was saved for.

A key typed in the request may test any address; an omitted or masked one is the
saved key, and it travels only to the saved scheme, host, port and path, with no
weaker certificate check. The refusal happens before catalog discovery and before
the probe, so nothing leaves the process. All keys here are synthetic.
"""

from __future__ import annotations

import asyncio
import copy
import json

import pytest

import ouroboros.gateway.models as provider_api
from ouroboros.llm import LLMClient
from ouroboros.provider_models import CONNECTIONS_KEY
from tests._socket_seal import block_network_except_socketpair

SAVED = "synthetic-saved-key-0001"
TYPED = "synthetic-typed-key-0002"
MOVED = "masked_key_endpoint_changed"


class _Request:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


@pytest.fixture
def sends(monkeypatch):
    """Every outbound attempt as ``(stage, provider, base_url, secrets)``; nothing reaches a network."""
    calls = []

    async def catalog(_client, _provider_id, _label, api_key, base_url, *, record_image_input=False):
        calls.append(("catalog", "openai-compatible", base_url, (api_key,)))
        return [provider_api._build_model_catalog_entry(_provider_id, _label, "listed-model", "listed-model")]

    def probe(self, model, *, settings, timeout=20.0):
        target = self._resolve_remote_target(model, settings=settings)
        secrets = tuple(value for value in (target.get("api_key"), target.get("password")) if value)
        calls.append(("probe", target["provider"], target["base_url"], secrets))
        return {"ok": True, "status_code": 200, "exception_type": ""}

    monkeypatch.setattr(provider_api, "_fetch_openai_compatible_model_catalog", catalog)
    monkeypatch.setattr(LLMClient, "probe_provider_readiness", probe)
    return calls


def _test(monkeypatch, saved, payload):
    monkeypatch.setattr(provider_api, "load_settings", lambda: copy.deepcopy(saved))
    response = asyncio.run(provider_api.api_provider_test(_Request(payload)))
    return response.status_code, json.loads(response.body.decode("utf-8"))


def _refused(status, body, sends):
    assert status == 400, body
    assert body["code"] == MOVED and "Enter its key again" in body["error"]
    assert SAVED not in json.dumps(body)
    assert sends == []


COMPATIBLE = {"OPENAI_COMPATIBLE_API_KEY": SAVED, "OPENAI_COMPATIBLE_BASE_URL": "https://saved.example/v1"}


@pytest.mark.parametrize("draft_url", [
    "https://evil.example/v1",           # another host
    "http://saved.example/v1",           # TLS downgrade
    "https://saved.example:8443/v1",     # another port
    "https://saved.example/v2",          # another path on the same host
    "https://user@saved.example/v1",     # added userinfo
    "https://saved.example/v1?route=x",  # added query
    # Transports append after a query and a catalog path lands in a fragment, so even an
    # empty one moves the request off /v1/models and /v1/chat/completions.
    "https://saved.example/v1?", "https://saved.example/v1/?", "https://saved.example/v1#models",
])
@pytest.mark.parametrize("key_override", [None, "***set***", "syntheti...", "***"])
def test_builtin_compatible_kept_key_never_reaches_another_endpoint(monkeypatch, sends, draft_url, key_override):
    overrides = {"OPENAI_COMPATIBLE_BASE_URL": draft_url}
    if key_override is not None:
        overrides["OPENAI_COMPATIBLE_API_KEY"] = key_override
    _refused(*_test(monkeypatch, COMPATIBLE, {"provider_id": "openai-compatible", "overrides": overrides}), sends)


@pytest.mark.parametrize("draft_url", [
    "https://saved.example/v1", "https://saved.example/v1/", " https://SAVED.example/v1 ",
    "HTTPS://saved.example:443/v1", "https://saved.example/v1/models/..",  # httpx resolves dot segments
])
def test_builtin_compatible_same_endpoint_spelling_keeps_the_saved_key(monkeypatch, sends, draft_url):
    status, body = _test(monkeypatch, COMPATIBLE, {
        "provider_id": "openai-compatible", "overrides": {"OPENAI_COMPATIBLE_BASE_URL": draft_url}})
    assert (status, body) == (200, {"ok": True})
    assert {call[3] for call in sends} == {(SAVED,)} and [call[0] for call in sends] == ["catalog", "probe"]


@pytest.mark.parametrize("saved_url,draft_url,same", [
    ("https://saved.example/v1?tenant=a", "https://saved.example/v1?tenant=a", True),
    ("https://saved.example/v1?tenant=a", "https://saved.example/v1/?tenant=a", False),
    ("https://saved.example/v1/?tenant=a", "https://saved.example/v1?tenant=a", False),
    ("https://saved.example/v1#x", "https://saved.example/v1", False),
])
def test_a_query_or_fragment_endpoint_is_the_same_only_spelled_the_same(monkeypatch, sends, saved_url, draft_url, same):
    status, body = _test(monkeypatch, {**COMPATIBLE, "OPENAI_COMPATIBLE_BASE_URL": saved_url}, {
        "provider_id": "openai-compatible", "overrides": {"OPENAI_COMPATIBLE_BASE_URL": draft_url}})
    if same:
        assert (status, body) == (200, {"ok": True}) and sends[-1][3] == (SAVED,)
    else:
        _refused(status, body, sends)


def test_typed_key_tests_a_new_endpoint(monkeypatch, sends):
    status, body = _test(monkeypatch, COMPATIBLE, {"provider_id": "openai-compatible", "overrides": {
        "OPENAI_COMPATIBLE_BASE_URL": "https://new.example/v1", "OPENAI_COMPATIBLE_API_KEY": TYPED}})
    assert (status, body) == (200, {"ok": True})
    assert sends == [("catalog", "openai-compatible", "https://new.example/v1", (TYPED,)),
                     ("probe", "openai-compatible", "https://new.example/v1", (TYPED,))]


def test_cleared_key_tests_a_new_endpoint_without_any_key(monkeypatch, sends):
    status, body = _test(monkeypatch, COMPATIBLE, {"provider_id": "openai-compatible", "overrides": {
        "OPENAI_COMPATIBLE_BASE_URL": "https://new.example/v1", "OPENAI_COMPATIBLE_API_KEY": ""}})
    assert (status, body) == (200, {"ok": True})
    assert {call[3] for call in sends} == {("",), ()}
    assert all(call[2] == "https://new.example/v1" for call in sends)


LEGACY = {"OPENAI_API_KEY": SAVED, "OPENAI_BASE_URL": "https://legacy.example/v1"}


def test_legacy_pair_key_never_follows_a_new_dedicated_url(monkeypatch, sends):
    _refused(*_test(monkeypatch, LEGACY, {"provider_id": "openai-compatible",
                                          "overrides": {"OPENAI_COMPATIBLE_BASE_URL": "https://evil.example/v1"}}), sends)


def test_cleared_dedicated_key_never_resurrects_the_legacy_key(monkeypatch, sends):
    status, _body = _test(monkeypatch, LEGACY, {"provider_id": "openai-compatible", "overrides": {
        "OPENAI_COMPATIBLE_BASE_URL": "https://new.example/v1", "OPENAI_COMPATIBLE_API_KEY": ""}})
    assert status == 200
    assert sends and all(SAVED not in call[3] for call in sends)


@pytest.mark.parametrize("provider_id,key", [
    ("openai-compatible", "OPENAI_BASE_URL"), ("openai-compatible", "OPENAI_API_KEY"),
    ("openrouter", "ANTHROPIC_API_KEY"), ("cloudru", "OPENAI_COMPATIBLE_BASE_URL"),
])
def test_builtin_override_keys_are_the_tested_kinds_own(monkeypatch, sends, provider_id, key):
    status, body = _test(monkeypatch, {**LEGACY, **COMPATIBLE},
                         {"provider_id": provider_id, "overrides": {key: "https://evil.example/v1"}})
    assert status == 400 and "unsupported override keys" in body["error"]
    assert sends == []


def _named(**row):
    return {CONNECTIONS_KEY: [{"kind": "openai-compatible", "name": "lab", "enabled": False,
                               "api_key": SAVED, "base_url": "https://lab.example/v1", **row}]}


@pytest.mark.parametrize("draft_url", ["https://evil.example/v1", "https://lab.example/v1?"])
@pytest.mark.parametrize("key_override", [None, "***set***", "syntheti..."])
def test_named_kept_key_never_reaches_another_endpoint(monkeypatch, sends, key_override, draft_url):
    overrides = {"base_url": draft_url}
    if key_override is not None:
        overrides["api_key"] = key_override
    _refused(*_test(monkeypatch, _named(), {
        "provider_id": "openai-compatible", "connection": "lab", "overrides": overrides}), sends)


def test_named_saved_endpoint_and_typed_key_still_test(monkeypatch, sends):
    payload = {"provider_id": "openai-compatible", "connection": "lab"}
    assert _test(monkeypatch, _named(), {**payload, "overrides": {"base_url": "https://lab.example/v1/"}})[0] == 200
    assert _test(monkeypatch, _named(), {**payload, "overrides": {
        "base_url": "https://new.example/v1", "api_key": TYPED}})[0] == 200
    assert [(call[2], call[3]) for call in sends if call[0] == "probe"] == [
        ("https://lab.example/v1/", (SAVED,)), ("https://new.example/v1", (TYPED,))]


def test_unsaved_named_draft_is_tested_from_the_draft_alone(monkeypatch, sends):
    status, _body = _test(monkeypatch, _named(), {"provider_id": "openai-compatible", "connection": "fresh",
                                                  "overrides": {"base_url": "https://new.example/v1"}})
    assert status == 200 and {call[3] for call in sends} == {("",), ()}


CLOUDRU = {"CLOUDRU_FOUNDATION_MODELS_API_KEY": SAVED, "CLOUDRU_FOUNDATION_MODELS_BASE_URL": ""}


def test_cloudru_effective_default_endpoint_is_the_saved_one(monkeypatch, sends):
    status, _body = _test(monkeypatch, CLOUDRU, {"provider_id": "cloudru", "overrides": {
        "CLOUDRU_FOUNDATION_MODELS_BASE_URL": "https://foundation-models.api.cloud.ru/v1/"}})
    assert status == 200 and sends[-1][3] == (SAVED,)
    sends.clear()
    _refused(*_test(monkeypatch, CLOUDRU, {"provider_id": "cloudru", "overrides": {
        "CLOUDRU_FOUNDATION_MODELS_BASE_URL": "https://evil.example/v1"}}), sends)


GIGACHAT = {"GIGACHAT_CREDENTIALS": SAVED, "GIGACHAT_BASE_URL": "https://giga.example/api/v1",
            "GIGACHAT_VERIFY_SSL_CERTS": ""}


@pytest.mark.parametrize("overrides", [
    {"GIGACHAT_BASE_URL": "https://evil.example/api/v1"},
    {"GIGACHAT_BASE_URL": "https://evil.example/api/v1", "GIGACHAT_CREDENTIALS": "***set***"},
    # Typing a password does not release the kept authorization key.
    {"GIGACHAT_BASE_URL": "https://evil.example/api/v1", "GIGACHAT_USER": "u", "GIGACHAT_PASSWORD": TYPED},
])
def test_gigachat_kept_credentials_never_reach_another_endpoint(monkeypatch, sends, overrides):
    _refused(*_test(monkeypatch, GIGACHAT, {"provider_id": "gigachat", "overrides": overrides}), sends)


def test_gigachat_kept_credentials_need_the_certificate_check_they_were_saved_with(monkeypatch, sends):
    status, body = _test(monkeypatch, GIGACHAT, {"provider_id": "gigachat",
                                                 "overrides": {"GIGACHAT_VERIFY_SSL_CERTS": "false"}})
    assert status == 400 and body["code"] == MOVED and "Certificate checking" in body["error"]
    assert sends == []
    # Turning the check back on, or typing the key, is fine.
    relaxed = {**GIGACHAT, "GIGACHAT_VERIFY_SSL_CERTS": "false"}
    assert _test(monkeypatch, relaxed, {"provider_id": "gigachat",
                                        "overrides": {"GIGACHAT_VERIFY_SSL_CERTS": "true"}})[0] == 200
    assert _test(monkeypatch, GIGACHAT, {"provider_id": "gigachat", "overrides": {
        "GIGACHAT_VERIFY_SSL_CERTS": "false", "GIGACHAT_CREDENTIALS": TYPED}})[0] == 200
    assert [call[3] for call in sends] == [(SAVED,), (TYPED,)]


def test_gigachat_typed_password_with_cleared_credentials_tests_a_new_endpoint(monkeypatch, sends):
    status, _body = _test(monkeypatch, {**GIGACHAT, "GIGACHAT_USER": "u", "GIGACHAT_PASSWORD": SAVED}, {
        "provider_id": "gigachat", "overrides": {
            "GIGACHAT_BASE_URL": "https://new.example/api/v1", "GIGACHAT_CREDENTIALS": "",
            "GIGACHAT_PASSWORD": TYPED}})
    assert status == 200 and sends == [("probe", "gigachat", "https://new.example/api/v1", (TYPED,))]


PASSWORD_AUTH = {"GIGACHAT_CREDENTIALS": "", "GIGACHAT_USER": "u", "GIGACHAT_PASSWORD": SAVED,
                 "GIGACHAT_BASE_URL": "https://giga.example/api/v1", "GIGACHAT_VERIFY_SSL_CERTS": ""}


@pytest.mark.parametrize("overrides", [
    {"GIGACHAT_BASE_URL": "https://evil.example/api/v1"},
    {"GIGACHAT_BASE_URL": "https://giga.example/api/v1?"},
])
def test_gigachat_kept_password_never_reaches_another_endpoint(monkeypatch, sends, overrides):
    _refused(*_test(monkeypatch, PASSWORD_AUTH, {"provider_id": "gigachat", "overrides": overrides}), sends)


def test_gigachat_kept_password_needs_the_certificate_check_it_was_saved_with(monkeypatch, sends):
    status, body = _test(monkeypatch, PASSWORD_AUTH, {"provider_id": "gigachat",
                                                      "overrides": {"GIGACHAT_VERIFY_SSL_CERTS": "false"}})
    assert status == 400 and body["code"] == MOVED and sends == []


@pytest.mark.parametrize("overrides", [
    {"GIGACHAT_BASE_URL": "https://new.example/api/v1", "GIGACHAT_CREDENTIALS": TYPED},
    {"GIGACHAT_VERIFY_SSL_CERTS": "false", "GIGACHAT_CREDENTIALS": TYPED},
])
def test_a_typed_gigachat_key_tests_while_an_unused_saved_password_stays(monkeypatch, sends, overrides):
    """A set authorization key is the one the library authenticates with; the saved password is not sent."""
    status, body = _test(monkeypatch, PASSWORD_AUTH, {"provider_id": "gigachat", "overrides": overrides})
    assert (status, body) == (200, {"ok": True}) and len(sends) == 1


def test_named_gigachat_kept_password_never_reaches_another_endpoint(monkeypatch, sends):
    saved = {CONNECTIONS_KEY: [{"kind": "gigachat", "name": "corp", "user": "u", "password": SAVED,
                                "base_url": "https://giga.example/api/v1"}]}
    _refused(*_test(monkeypatch, saved, {"provider_id": "gigachat", "connection": "corp",
                                         "overrides": {"base_url": "https://evil.example/api/v1"}}), sends)


@pytest.mark.parametrize("provider_id,saved,overrides", [
    ("minimax", {"MINIMAX_API_KEY": SAVED, "MINIMAX_REGION": "global_en"}, {"MINIMAX_REGION": "cn_zh"}),
    ("zai", {"ZAI_API_KEY": SAVED, "ZAI_PLAN": "payg"}, {"ZAI_PLAN": "coding"}),
])
def test_vendor_region_and_plan_choices_keep_the_saved_key(monkeypatch, sends, provider_id, saved, overrides):
    """A closed vendor choice is the same vendor, not an address the caller picks."""
    status, body = _test(monkeypatch, saved, {"provider_id": provider_id, "overrides": overrides})
    assert (status, body) == (200, {"ok": True}) and sends[-1][3] == (SAVED,)


# --- Built requests -------------------------------------------------------------------------
# The real catalog read, OpenAI SDK and GigaChat library build every request; only the physical
# httpx transports answer in memory and the accounting wrapper is bypassed. A socket tripwire
# fails any real connection.

@pytest.fixture
def wire(monkeypatch):
    import socket

    import httpx

    import ouroboros.llm_probe as llm_probe

    requests = []

    def reply(request):
        requests.append({"url": str(request.url), "raw_path": request.url.raw_path.decode(),
                         "authorization": request.headers.get("authorization") or "",
                         "body": request.content.decode("utf-8", "replace")})
        if request.method == "GET":
            return httpx.Response(200, request=request, json={"data": [{"id": "listed-model"}]})
        if request.url.path.endswith("/oauth"):
            return httpx.Response(200, request=request, json={"access_token": "issued-token", "expires_at": 9999999999999})
        if request.url.raw_path.endswith(b"/token"):
            return httpx.Response(200, request=request, json={"tok": "issued-token", "exp": 9999999999999})
        return httpx.Response(200, request=request, json={
            "id": "x", "object": "chat.completion", "created": 1, "model": "listed-model",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    async def reply_async(_transport, request):
        await request.aread()
        return reply(request)

    def refuse(*_args, **_kwargs):
        raise AssertionError("a provider test request tried to reach the network")

    monkeypatch.setattr(socket, "socketpair", getattr(socket, "_fallback_socketpair", socket.socketpair))  # Windows loop path, on every OS
    block_network_except_socketpair(monkeypatch, refuse)  # Windows builds asyncio's self-pipe through connect
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda _transport, request: reply(request))
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", reply_async)
    monkeypatch.setattr(llm_probe, "accounted_one_shot", lambda _target, candidate, send, **_kw: send(candidate))
    for key in ("GIGACHAT_CREDENTIALS", "GIGACHAT_USER", "GIGACHAT_PASSWORD", "GIGACHAT_ACCESS_TOKEN",
                "GIGACHAT_BASE_URL", "GIGACHAT_AUTH_URL"):
        monkeypatch.delenv(key, raising=False)
    return requests


def _carried(requests, secret):
    """The paths of built requests whose Authorization (Basic decoded too) or body holds ``secret``."""
    import base64

    def basic(header):
        try:
            return base64.b64decode(header.partition("Basic ")[2], validate=True).decode("utf-8", "replace")
        except ValueError:
            return ""

    return [r["raw_path"] for r in requests
            if secret in r["authorization"] or secret in r["body"] or secret in basic(r["authorization"])]


@pytest.mark.parametrize("payload", [
    {"provider_id": "openai-compatible", "overrides": {"OPENAI_COMPATIBLE_BASE_URL": "https://saved.example/v1?"}},
    {"provider_id": "openai-compatible", "overrides": {"OPENAI_COMPATIBLE_BASE_URL": "https://saved.example/v1#x"}},
])
def test_built_requests_never_carry_a_kept_key_off_the_saved_paths(monkeypatch, wire, payload):
    status, body = _test(monkeypatch, COMPATIBLE, payload)
    assert status == 400 and body["code"] == MOVED and wire == []
    # Control: the saved address spelled with a trailing slash builds exactly the saved paths.
    status, body = _test(monkeypatch, COMPATIBLE, {"provider_id": "openai-compatible", "overrides": {
        "OPENAI_COMPATIBLE_BASE_URL": "https://saved.example/v1/"}})
    assert (status, body) == (200, {"ok": True})
    assert _carried(wire, SAVED) == ["/v1/models", "/v1/chat/completions"]


def test_built_requests_of_a_named_connection_stay_on_its_saved_paths(monkeypatch, wire):
    payload = {"provider_id": "openai-compatible", "connection": "lab"}
    status, body = _test(monkeypatch, _named(), {**payload, "overrides": {"base_url": "https://lab.example/v1?"}})
    assert status == 400 and body["code"] == MOVED and wire == []
    assert _test(monkeypatch, _named(), payload) == (200, {"ok": True})
    assert _carried(wire, SAVED) == ["/v1/models", "/v1/chat/completions"]


def test_built_gigachat_token_request_keeps_a_saved_password_on_its_address(monkeypatch, wire):
    status, body = _test(monkeypatch, PASSWORD_AUTH, {"provider_id": "gigachat", "overrides": {
        "GIGACHAT_BASE_URL": "https://giga.example/api/v1?"}})
    assert status == 400 and body["code"] == MOVED and wire == []
    status, body = _test(monkeypatch, PASSWORD_AUTH, {"provider_id": "gigachat", "overrides": {
        "GIGACHAT_BASE_URL": "https://giga.example/api/v1/"}})
    assert (status, body) == (200, {"ok": True})
    assert _carried(wire, SAVED) == ["/api/v1/token"]


def test_built_gigachat_requests_with_a_typed_key_never_send_the_unused_saved_password(monkeypatch, wire):
    status, body = _test(monkeypatch, PASSWORD_AUTH, {"provider_id": "gigachat", "overrides": {
        "GIGACHAT_BASE_URL": "https://new.example/api/v1", "GIGACHAT_CREDENTIALS": TYPED}})
    assert (status, body) == (200, {"ok": True})
    assert _carried(wire, SAVED) == []
    assert len(_carried(wire, TYPED)) == 1 and wire[0]["url"].endswith("/oauth")
    assert wire[-1]["url"] == "https://new.example/api/v1/chat/completions"
    assert wire[-1]["authorization"] == "Bearer issued-token"
