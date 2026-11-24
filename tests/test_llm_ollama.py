"""Native Ollama contracts for llm_ollama."""
import asyncio
from types import SimpleNamespace
import httpx
import pytest
from tests._ollama import (
    TARGET, MESSAGES, completed, overflow_reply, OllamaHTTP,
    LlamaCppHTTP, chat, install_http as install_http,
)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_native_guards_are_present_on_every_send(install_http, asynchronous, stream):
    server = install_http(OllamaHTTP())
    message, usage = chat(asynchronous=asynchronous, stream=stream)
    assert message["content"] == "391"
    assert usage["prompt_tokens"] == 25
    assert usage["completion_tokens"] == 4
    assert server.posts
    for body in server.posts:
        assert body["truncate"] is False
        assert body["shift"] is False
        assert body["options"]["num_predict"] == 2048
        assert "think" not in body  # preserve the model's default unless explicitly disabled
        assert body["messages"] == MESSAGES


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("kind", ["json_schema", "json_object"])
def test_public_chat_carries_structured_output_to_native(install_http, monkeypatch, asynchronous, kind):
    from ouroboros.llm import LLMClient
    from tests._ollama import _compatible_target

    target = _compatible_target(monkeypatch, TARGET["base_url"])
    client = LLMClient()
    monkeypatch.setattr(client, "_resolve_remote_target", lambda model: target)
    schema = {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}
    response_format = {"type": kind}
    if kind == "json_schema":
        response_format["json_schema"] = {"name": "answer", "schema": schema, "strict": True}
    server = install_http(OllamaHTTP(completed(message={"role": "assistant", "content": '{"value":391}'})))
    kwargs = dict(messages=MESSAGES, model=TARGET["requested_model"], response_format=response_format, no_proxy=True)
    message, usage = asyncio.run(client.chat_async(**kwargs)) if asynchronous else client.chat(**kwargs)
    assert [body["format"] for body in server.posts] == [schema if kind == "json_schema" else "json"]
    assert message["content"] == '{"value":391}'
    assert usage["response_finish_reason"] == "stop" and "response_status" not in usage


def test_native_400_exposes_both_counts_to_existing_refusal_parser(install_http):
    from ouroboros.size_refusal import from_exception

    install_http(OllamaHTTP({"error": "request (20216 tokens) exceeds the available context size "
                          "(4096 tokens), try increasing it"}, status=400))
    with pytest.raises(Exception) as raised:
        chat()
    refusal = from_exception(raised.value)
    assert refusal is not None
    assert (refusal.requested, refusal.window, refusal.meaning) == (20_216, 4096, "total")


def test_unrelated_native_400_is_not_reclassified_as_context(install_http):
    from ouroboros.size_refusal import from_exception

    install_http(OllamaHTTP({"error": "Failed to initialize samplers: failed to parse grammar"}, status=400))
    with pytest.raises(Exception) as raised:
        chat()
    assert from_exception(raised.value) is None
    assert getattr(raised.value, "status_code", None) == 400


def test_cancellation_propagates_and_closes_native_stream(install_http):
    class CancelledStream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b'{"message":{"thinking":"unfinished"},"done":false}\n'
            raise asyncio.CancelledError()

        async def aclose(self):
            self.closed = True

    stream = CancelledStream()
    install_http(OllamaHTTP(lambda _: httpx.Response(200, stream=stream,
                               headers={"Content-Type": "application/x-ndjson"})))
    with pytest.raises(asyncio.CancelledError):
        chat(asynchronous=True, stream=True)
    assert stream.closed, "cancelled native requests must close their connection"


def test_discovery_does_not_probe_public_or_other_provider_routes(install_http):
    from ouroboros.llm_ollama import is_ollama

    server = install_http(OllamaHTTP())
    assert not is_ollama({**TARGET, "base_url": "https://api.example.com/v1"})
    assert not is_ollama({**TARGET, "provider": "openrouter"})
    assert server.gets == []


@pytest.mark.parametrize("asynchronous", [False, True])
def test_missing_credentials_precede_native_reads_and_send(install_http, asynchronous):
    from ouroboros import capability_evidence as ce
    from ouroboros.llm_attempt import ConnectionUnavailable

    target = {**TARGET, "api_key": "", "keyless": False}
    install_http(lambda request: pytest.fail(f"unsigned request to {request.url.path}"))
    ce.probe(ce.canonical_evidence_root(), provider=target["provider"], base_url=target["base_url"],
             model=target["requested_model"], target=target, force=True)
    with pytest.raises(ConnectionUnavailable):
        chat(asynchronous=asynchronous, target=target, window=80000)


def test_known_lan_ollama_recovery_reads_the_same_model_behind_its_route_header(install_http):
    from ouroboros.llm_ollama import is_ollama
    from ouroboros.llm_probe import upstream_transport_reachable

    target = {**TARGET, "base_url": "http://runner.lan/v1", "default_headers": {"X-Runner": "lab"}}
    seen = []
    server = OllamaHTTP()
    def handle(request):
        assert request.url.host == "runner.lan" and request.headers["X-Runner"] == "lab"
        seen.append((request.method, request.url.path))
        if request.url.path == "/api/show":
            import json
            assert json.loads(request.content) == {"model": target["resolved_model"]}
            return httpx.Response(200, json={"model_info": {"general.architecture": "text"},
                                             "capabilities": ["completion"]})
        return server(request)

    install_http(handle)
    assert is_ollama(target)
    seen.clear()
    llm = SimpleNamespace(_resolve_remote_target=lambda model: target, _no_proxy_timeout=httpx.Timeout)
    result = upstream_transport_reachable(llm, target["requested_model"], timeout=2)
    assert result["program"] == "ollama" and result["model"] == target["resolved_model"]
    assert seen == [("GET", "/api/version"), ("POST", "/api/show")]


def test_wrong_version_shape_does_not_claim_ollama(install_http):
    from ouroboros.llm_ollama import is_ollama

    install_http(lambda request: httpx.Response(200, json={"models": []}))
    assert not is_ollama({**TARGET, "base_url": "http://127.0.0.1:11438/v1"})


def test_configured_window_is_sent_but_unknown_window_is_not_invented(install_http, monkeypatch):
    from ouroboros import llm_routing as policy

    server = install_http(OllamaHTTP())
    monkeypatch.setattr(policy, "configured_window", lambda target: 80_000)
    chat()
    assert server.posts[-1]["options"]["num_ctx"] == 80_000
    monkeypatch.setattr(policy, "configured_window", lambda target: 0)
    chat()
    assert "num_ctx" not in server.posts[-1]["options"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_identified_ollama_with_unavailable_ps_never_falls_back_to_v1(install_http, asynchronous):
    from ouroboros.llm_ollama import is_ollama, is_ollama_async, OllamaIntegrityError

    asked = []

    def handler(request):
        asked.append(request.url.path)
        return httpx.Response(200, json={"version": "0.40.16"}) if request.url.path == "/api/version" else httpx.Response(503)

    install_http(handler)
    target = {**TARGET, "base_url": "http://127.0.0.1:11439/v1"}
    for _ in range(2):
        with pytest.raises(OllamaIntegrityError) as raised:
            if asynchronous:
                asyncio.run(is_ollama_async(target))
            else:
                is_ollama(target)
        assert raised.value.code == "ollama_route_unsettled"
    # The identity is saved only after every read agreed; a withheld discovery
    # is asked again and never turns into a /v1 send.
    assert asked == ["/api/version", "/api/ps"] * 2


def test_server_padding_not_used_if_post_response_ps_fails(install_http, monkeypatch):
    server = OllamaHTTP(completed(prompt_eval_count=80_042, eval_count=1, done_reason="length"))

    def handler(request):
        if request.url.path == "/api/ps" and server.posts:
            return httpx.Response(503)
        return server(request)

    install_http(handler)
    monkeypatch.setattr("ouroboros.llm_routing.configured_window", lambda target: 80_000)
    _, usage = chat()
    assert usage["ollama_context"]["exceeded_by"] == 43
    assert server.posts[0]["options"]["num_ctx"] == 80_000


@pytest.mark.parametrize("effort,sent", [("none", {"think": False}), ("minimal", {}), ("high", {})])
def test_physical_attempt_records_requested_effort_and_actual_native_allowance(monkeypatch, effort, sent):
    from ouroboros.llm_ollama import _prepare
    from ouroboros import llm_attempt

    # Model an allowance adjusted by the established Nano budget at the final send.
    monkeypatch.setattr(llm_attempt, "bound_reply_allowance", lambda target, payload: {**payload, "max_tokens": 1472})
    wire, attempt = _prepare(dict(TARGET), MESSAGES, None, effort, 4096, "auto", None, False, None)
    assert wire["options"]["num_predict"] == 1472
    assert attempt.max_completion_tokens == 1472
    assert attempt.effort["requested"] == effort
    assert attempt.effort["sent"] == sent
    assert attempt.effort["sent_state"] == ("explicit" if sent else "omitted")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("explicit,expected", [(None, 733.25), (19.5, 19.5)])
def test_native_dispatch_obeys_shared_transport_timeout(install_http, monkeypatch, asynchronous, explicit, expected):
    from ouroboros import config

    monkeypatch.setattr(config, "get_llm_transport_read_timeout_sec", lambda: 733.25)
    read_timeouts = []
    server = OllamaHTTP()

    def handler(request):
        if request.url.path == "/api/chat":
            read_timeouts.append(request.extensions["timeout"]["read"])
        return server(request)

    install_http(handler)
    chat(asynchronous=asynchronous, timeout=explicit)
    assert read_timeouts == [expected]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_padded_refusal_reads_nominal_loaded_window_before_propagating(install_http, monkeypatch, asynchronous):
    from ouroboros import llm_ollama
    from ouroboros.size_refusal import from_exception

    server = install_http(OllamaHTTP({"error": "request (90068 tokens) exceeds the available context size "
                                    "(80128 tokens), try increasing it"}, window=80_000, status=400))
    recorded = []
    record_window = llm_ollama.record_window
    monkeypatch.setattr(llm_ollama, "record_window", lambda target, window, **kw: (
        recorded.append(window) or record_window(target, window, **kw)))
    with pytest.raises(httpx.HTTPStatusError) as raised:
        chat(asynchronous=asynchronous)
    assert from_exception(raised.value).window == 80_128
    assert from_exception(raised.value).token_window == 80_000
    assert recorded == [80_000], "physical padding must not erase the nominal serving capacity witness"
    assert server.gets == ["/api/version", "/api/ps", "/api/ps"]


@pytest.mark.parametrize("preflight_window", [0, 80_000])
@pytest.mark.parametrize("requested", [0, 120_000], ids=["auto", "explicit"])
def test_refusal_without_post_ps_keeps_only_observed_nominal_window(install_http, preflight_window, requested):
    from ouroboros.size_refusal import from_exception

    server = OllamaHTTP({"error": "request (90068 tokens) exceeds the available context size "
                                  "(80128 tokens), try increasing it"}, window=preflight_window, status=400)
    def handle(request):
        if request.url.path == "/api/ps" and server.posts:
            return httpx.Response(503)
        return server(request)
    install_http(handle)
    with pytest.raises(httpx.HTTPStatusError) as caught:
        chat(window=requested)
    refusal = from_exception(caught.value)
    assert (refusal.window, refusal.token_window) == (80_128, 0 if requested else preflight_window)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_ndjson_size_error_reaches_existing_refusal_parser(install_http, asynchronous):
    from ouroboros.size_refusal import from_exception

    install_http(OllamaHTTP([{"error": "request (90068 tokens) exceeds the available context size "
                                      "(80128 tokens), try increasing it"}]))
    with pytest.raises(Exception) as raised:
        chat(asynchronous=asynchronous, stream=True)
    refusal = from_exception(raised.value)
    assert refusal is not None
    assert (refusal.requested, refusal.window, refusal.token_window) == (90_068, 80_128, 80_000)


def test_fake_ollama_negative_control_unguarded_20216_tokens_silently_loses_history(install_http):
    """The same fake overflow branch used by the real-loop test has both server outcomes."""
    server = install_http(OllamaHTTP(lambda body: overflow_reply(body, 20_216, 4096), window=4096))
    request = {"model": "mock-model", "messages": [{"role": "user", "content": "START_CANARY"}], "stream": False}
    with httpx.Client() as client:
        lossy = client.post("http://127.0.0.1:11434/api/chat", json=request)
        guarded = client.post("http://127.0.0.1:11434/api/chat", json={**request, "truncate": False, "shift": False})
    assert lossy.status_code == 200
    assert lossy.json()["prompt_eval_count"] == 2048 < 20_216
    assert lossy.json()["message"]["content"] == "MISSING_START_CANARY"
    assert guarded.status_code == 400
    from ouroboros.size_refusal import parse
    from ouroboros.llm_ollama_wire import native_error_body
    assert parse(guarded.status_code, guarded.json()) is None  # generic prose adds no recognition
    refusal = parse(guarded.status_code, native_error_body(guarded.json()))
    assert (refusal.requested, refusal.window) == (20_216, 4096)
    assert len(server.posts) == 2


@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_identity_survives_restart_and_unrelated_window_evidence(tmp_path, monkeypatch, install_http, asynchronous):
    from ouroboros import capability_evidence as ce, llm_ollama

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    target = {**TARGET, "base_url": "http://127.0.0.1:11451/v1"}
    server = OllamaHTTP()
    asked = []
    offline = False

    def handler(request):
        asked.append(request.url.path)
        if offline:
            raise httpx.ConnectTimeout("The identified server is temporarily unreachable", request=request)
        return server(request)

    install_http(handler)

    def discover():
        if asynchronous:
            return asyncio.run(llm_ollama.is_ollama_async(target))
        return llm_ollama.is_ollama(target)

    assert discover()
    assert asked == ["/api/version", "/api/ps", "/api/show"]
    # A fresh process has no in-memory identity. Generic window evidence may now
    # refer to owner acknowledgement, stale probes, or another metadata reader.
    with llm_ollama._identity_lock:
        llm_ollama._rejected.clear()
    monkeypatch.setattr(ce, "probe", lambda *args, **kwargs: SimpleNamespace(detail="owner acknowledged capacity"))
    offline = True
    assert discover(), "durable protocol identity must not depend on mutable window evidence"
    assert asked == ["/api/version", "/api/ps", "/api/show"], "a known native route must not re-probe then fall back to /v1"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_identity_persistence_failure_never_authorizes_v1_fallback(tmp_path, monkeypatch, install_http, asynchronous):
    from ouroboros import capability_evidence as ce, llm_ollama

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(ce, "_store_evidence", lambda *args, **kwargs: False)
    server = install_http(OllamaHTTP())
    target = {**TARGET, "base_url": "http://127.0.0.1:11452/v1"}
    with pytest.raises(llm_ollama.OllamaIntegrityError):
        if asynchronous:
            asyncio.run(llm_ollama.is_ollama_async(target))
        else:
            llm_ollama.is_ollama(target)
    assert server.gets == ["/api/version", "/api/ps"]
    assert len(server.shows) == 1
    assert server.posts == []


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status,error", [
    (200, "Unexpected endpoint or method. (GET /api/version)"),
    (200, {"code": 404, "message": "File Not Found", "type": "not_found_error"}),
    (404, "not found"), (501, "not implemented"),
])
def test_passive_non_ollama_lookup_reuses_the_negative_verdict(install_http, monkeypatch, asynchronous, status, error):
    from ouroboros import llm_ollama

    server = LlamaCppHTTP()
    def handler(request):
        response = server(request)
        if request.url.path == "/api/version":
            return httpx.Response(status, json={"error": error})
        return response
    install_http(handler)
    target = {**TARGET, "base_url": "http://127.0.0.1:11462/v1"}

    def discover():
        return asyncio.run(llm_ollama.is_ollama_async(target)) if asynchronous else llm_ollama.is_ollama(target)

    assert not discover()
    assert server.paths == [("GET", "/api/version")]
    for _ in range(5):
        assert not discover()
    assert server.paths == [("GET", "/api/version")], "a cached verdict asks nothing"
    clock = llm_ollama.time.monotonic() + llm_ollama._REJECTION_TTL_SEC + 1
    monkeypatch.setattr(llm_ollama.time, "monotonic", lambda: clock)
    assert not discover()
    assert server.paths == [("GET", "/api/version")] * 2, "the verdict lasts ten minutes"
    llm_ollama.forget_rejection(target)
    assert not discover()
    assert server.paths == [("GET", "/api/version")] * 3, "a route error asks again"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("version_reply", ["ollama", "unreachable"])
def test_native_404_keeps_identity_when_ollama_answers_or_address_is_unreachable(
        install_http, asynchronous, version_reply):
    from ouroboros import capability_evidence as ce, llm_ollama

    target = {**TARGET, "base_url": "http://127.0.0.1:11465/v1"}
    install_http(OllamaHTTP())
    assert llm_ollama.is_ollama(target)
    missing = OllamaHTTP(httpx.Response(404, json={"error": "model 'mock-model' not found"}))

    def handler(request):
        if missing.posts and version_reply == "unreachable" and request.url.path == "/api/version":
            raise httpx.ConnectError("no route to host", request=request)
        return missing(request)

    install_http(handler)
    with pytest.raises(httpx.HTTPStatusError) as raised:
        chat(target=dict(target), asynchronous=asynchronous)
    assert raised.value.response.status_code == 404
    assert "not found" in raised.value.response.text, "Ollama's own 404 is shown as it is"
    assert llm_ollama.is_ollama(target)
    identity = llm_ollama._identity(target)
    assert ce._read(ce.canonical_evidence_root())["native_protocols"][identity]["protocol"] == "ollama"


def test_single_label_ollama_host_is_discovered_but_public_one_is_not(install_http):
    from ouroboros.llm_ollama import is_ollama

    server = install_http(OllamaHTTP())
    assert not is_ollama({**TARGET, "base_url": "https://ollama.example.com/v1"})
    assert server.gets == []
    assert is_ollama({**TARGET, "base_url": "http://ollama:11434/v1"})
    assert server.gets[:1] == ["/api/version"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_identity_needs_version_ps_and_show_and_is_saved_after_all_of_them(tmp_path, install_http, asynchronous):
    from ouroboros import capability_evidence as ce, llm_ollama

    order = []
    server = OllamaHTTP()

    def handler(request):
        saved = bool(ce._read(ce.canonical_evidence_root()).get("native_protocols"))
        order.append((request.url.path, saved))
        return server(request)

    install_http(handler)
    target = {**TARGET, "base_url": "http://127.0.0.1:11466/v1"}
    found = asyncio.run(llm_ollama.is_ollama_async(target)) if asynchronous else llm_ollama.is_ollama(target)
    assert found
    assert order == [("/api/version", False), ("/api/ps", False), ("/api/show", False)]
    assert server.shows == [{"model": "mock-model"}]
    assert ce._read(ce.canonical_evidence_root())["native_protocols"]


@pytest.mark.parametrize("collision", ["same-facts", "changed-digest", "write-failed"])
def test_interleaved_native_refresh_reuses_only_matching_saved_facts(install_http, monkeypatch, collision):
    from datetime import datetime, timedelta, timezone
    from ouroboros import capability_evidence as ce, llm_ollama, utils
    from ouroboros.server_window import read_ollama_window

    now = datetime(2000, 1, 3, tzinfo=timezone.utc)
    prior_at = (now - timedelta(seconds=ce._CONFIRMED_TTL_SEC + 1)).isoformat()
    peer_at, own_at = (now - timedelta(seconds=1)).isoformat(), now.isoformat()
    shown_at = prior_at
    monkeypatch.setattr(utils, "utc_now_iso", lambda *args, **kwargs: shown_at)
    monkeypatch.setattr(ce, "utc_now", lambda: now)
    server = install_http(OllamaHTTP())
    target = dict(TARGET)
    assert llm_ollama.is_ollama(target)
    peer, prior = dict(target), target["ollama_record"]
    server.shows.clear()

    def handle(request):
        nonlocal shown_at
        response = server(request)
        if request.url.path == "/api/ps" and shown_at == peer_at and collision == "changed-digest":
            body = response.json()
            body["models"][0]["digest"] = "sha256:next-model"
            return httpx.Response(200, json=body)
        if request.url.path == "/api/show" and len(server.shows) == 1:
            # The first sender has taken its prior snapshot and read its facts.
            # Let the peer publish before the first sender receives /show.
            if collision != "write-failed":
                shown_at = peer_at
                assert read_ollama_window(peer, 80_000) == 80_000
            shown_at = own_at
        return response

    install_http(handle)
    if collision == "write-failed":
        monkeypatch.setattr(ce, "_save", lambda *_: False)
    if collision == "same-facts":
        assert chat(target=target, window=80_000)[0]["content"] == "391"
        [sent] = server.posts
        assert sent["truncate"] is sent["shift"] is False
        assert target["ollama_record"] == peer["ollama_record"]
    else:
        with pytest.raises(llm_ollama.OllamaRouteUnsettledError, match="request not sent"):
            chat(target=target, window=80_000)
        assert not server.posts and target["ollama_record"] == prior
    held = ce._read(ce.canonical_evidence_root())["native_protocols"][llm_ollama._identity(target)]
    assert held == peer["ollama_record"]
    assert held["shown_at"] == (prior_at if collision == "write-failed" else peer_at)
    assert held["digest"] == ("sha256:next-model" if collision == "changed-digest" else prior["digest"])


@pytest.mark.parametrize("show,found", [
    (httpx.Response(404, json={"error": "model 'mock-model' not found, try pulling it first"}), True),
    (httpx.Response(200, json={"models": []}), False),
    (httpx.Response(405, text="Method Not Allowed"), False),
    (httpx.Response(404, json={"error": "File Not Found"}), False),
])
def test_show_signal_accepts_an_unpulled_model_but_rejects_a_mimic(install_http, show, found):
    from ouroboros import capability_evidence as ce, llm_ollama

    server = OllamaHTTP(loaded=False)  # nothing loaded yet: /api/ps lists no model

    def handler(request):
        return show if request.url.path == "/api/show" else server(request)

    install_http(handler)
    target = {**TARGET, "base_url": "http://127.0.0.1:11467/v1"}
    assert llm_ollama.is_ollama(target) is found
    assert bool(ce._read(ce.canonical_evidence_root()).get("native_protocols")) is found


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", [408, 429, 503, "malformed", httpx.ReadTimeout, httpx.ConnectError, httpx.ProxyError, "lan-down"])
def test_first_version_unknown_withholds_every_real_send(install_http, monkeypatch, asynchronous, failure):
    """A busy, unreachable or cut-short read withholds the send; a refused one (401/403) states nothing and
    belongs to ``test_llm_ollama_gateway``."""
    from tests._ollama import _compatible_target
    from ouroboros.llm import LLMClient
    from ouroboros import llm_ollama
    from ouroboros.loop_llm_call import classify_llm_exception
    host = "runner.lan" if failure == "lan-down" else "127.0.0.1"
    target = _compatible_target(monkeypatch, f"http://fixture-user:fixture-pass@{host}:11434/v1?key=fixture-value")
    monkeypatch.setattr(LLMClient, "_resolve_remote_target", lambda self, model: dict(target))
    paths = []
    def handler(request):
        paths.append(request.url.path)
        if isinstance(failure, int):
            return httpx.Response(failure, json={"error": {"message": "busy"}})
        if failure == "malformed":
            return httpx.Response(200, content=b'{"version":', headers={"content-type": "application/problem+json"})
        error = httpx.ConnectError if failure == "lan-down" else failure
        raise error("unreachable", request=request)
    install_http(handler)
    client = LLMClient()
    for _ in range(2):
        with pytest.raises(llm_ollama.OllamaRouteUnsettledError) as caught:
            if asynchronous:
                asyncio.run(client.chat_async(MESSAGES, model=TARGET["requested_model"]))
            else:
                client.chat(MESSAGES, model=TARGET["requested_model"])
        from ouroboros.transport_custody import is_pre_dispatch_transport_failure
        verdict = classify_llm_exception(caught.value)
        assert is_pre_dispatch_transport_failure(caught.value)
        assert verdict.retry_same_request
        expected = "transport_unavailable" if failure == "lan-down" else "provider_transient"
        assert verdict.kind == ("provider_error" if isinstance(failure, type) else expected)
        assert "fixture-" not in str(caught.value)
        assert f"{host}:11434" in str(caught.value) and "not sent" in str(caught.value)
        assert llm_ollama._identity(target) not in llm_ollama._rejected
        from ouroboros.review_custody import _ReviewAttemptHistory, _review_exception_projection
        history = _ReviewAttemptHistory()
        history.observe(caught.value)
        assert _review_exception_projection(caught.value, {}, history, None)[3] == "not_dispatched"
    assert paths == ["/api/version"] * 2


@pytest.mark.parametrize("asynchronous", [False, True])
def test_expired_negative_does_not_authorize_chat_on_a_busy_address(install_http, monkeypatch, asynchronous):
    from tests._ollama import _compatible_target
    from ouroboros.llm import LLMClient
    from ouroboros import llm_ollama
    target = _compatible_target(monkeypatch, TARGET["base_url"])
    monkeypatch.setattr(LLMClient, "_resolve_remote_target", lambda self, model: dict(target))
    server = LlamaCppHTTP()
    busy = False
    versions = []
    def handler(request):
        if request.url.path == "/api/version":
            versions.append(busy)
            if busy:
                return httpx.Response(503)
        return server(request)
    install_http(handler)
    assert not llm_ollama.is_ollama(target)
    busy = True
    clock = llm_ollama.time.monotonic() + 601
    monkeypatch.setattr(llm_ollama.time, "monotonic", lambda: clock)
    for _ in range(2):
        client = LLMClient()
        with pytest.raises(llm_ollama.OllamaRouteUnsettledError):
            if asynchronous:
                asyncio.run(client.chat_async(MESSAGES, model=TARGET["requested_model"]))
            else:
                client.chat(MESSAGES, model=TARGET["requested_model"])
    assert server.paths == [("GET", "/api/version")]
    assert versions == [False, True, True]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [429, 503])
def test_native_404_and_busy_version_retain_protocol(install_http, asynchronous, status):
    from ouroboros import llm_ollama
    target = dict(TARGET)
    server = install_http(OllamaHTTP(httpx.Response(404, json={"error": "missing model"})))
    assert llm_ollama.is_ollama(target)
    def handler(request):
        if server.posts and request.url.path == "/api/version":
            return httpx.Response(status)
        return server(request)
    install_http(handler)
    with pytest.raises(httpx.HTTPStatusError) as caught:
        chat(target=target, asynchronous=asynchronous)
    assert caught.value.response.status_code == 404
    assert len(server.posts) == 1 and llm_ollama.is_ollama(target)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_malformed_ndjson_has_typed_physical_failure(install_http, asynchronous):
    from ouroboros.llm_ollama_wire import OllamaIntegrityError
    from ouroboros.usage_accounting import last_physical_attempt_capture
    server = install_http(OllamaHTTP(lambda _: httpx.Response(200, content=b'{broken\n')))
    with pytest.raises(OllamaIntegrityError, match="invalid NDJSON"):
        chat(asynchronous=asynchronous, stream=True)
    assert len(server.posts) == 1
    capture = last_physical_attempt_capture()
    assert capture is not None


def test_other_workers_native_discovery_supersedes_local_negative(install_http):
    from ouroboros import capability_evidence as ce, llm_ollama
    target = dict(TARGET)
    install_http(LlamaCppHTTP())
    assert not llm_ollama.is_ollama(target)
    server = install_http(OllamaHTTP())
    identity = llm_ollama._identity(target)
    record = {"protocol": "ollama", "version": "0.40.16"}
    assert ce._store_evidence(ce.canonical_evidence_root(), "native_protocols", identity, record)
    assert llm_ollama.is_ollama(target) and target["ollama_record"] == record
    assert not server.gets
    replacement = {"protocol": "openai-compatible", "replaced": "ollama"}
    assert ce._store_evidence(ce.canonical_evidence_root(), "native_protocols", identity, replacement)
    compatible = install_http(LlamaCppHTTP())
    assert not llm_ollama.is_ollama(target) and "ollama_record" not in target
    assert compatible.paths == [("GET", "/api/version")] and "truncat" in target["route_notice"]


@pytest.mark.parametrize("version,supported", [("0.12.5", False), ("0.12.6", True)])
def test_history_guard_requires_both_native_request_flags(install_http, monkeypatch, version, supported):
    from tests._ollama import _compatible_target
    from ouroboros.llm import LLMClient
    from ouroboros.llm_ollama import OllamaRouteUnsettledError
    server = OllamaHTTP()
    def handle(request):
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": version})
        return server(request)
    install_http(handle)
    model = _compatible_target(monkeypatch, TARGET["base_url"])["capability_route"]["model"]
    if supported:
        LLMClient().chat(MESSAGES, model=model)
        [sent] = server.posts
        assert sent["truncate"] is sent["shift"] is False
    else:
        with pytest.raises(OllamaRouteUnsettledError, match="0.12.6") as caught:
            LLMClient().chat(MESSAGES, model=model)
        assert not caught.value.retryable and not server.posts


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("previous", ["native", "compatible"])
def test_old_replacement_withholds_generation_even_after_recent_success(install_http, monkeypatch, asynchronous, previous):
    from tests._ollama import _compatible_target
    from ouroboros.llm import LLMClient
    from ouroboros.llm_ollama import OllamaRouteUnsettledError

    model = _compatible_target(monkeypatch, TARGET["base_url"])["capability_route"]["model"]
    modern, compatible = OllamaHTTP(), LlamaCppHTTP()
    initial = True
    sends = []

    def handle(request):
        if not initial and request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.12.5"})
        if request.url.path in ("/api/chat", "/v1/chat/completions"):
            sends.append(request.url.path)
        return (compatible if initial and previous == "compatible" else modern)(request)

    install_http(handle)
    client = LLMClient()
    def send():
        return asyncio.run(client.chat_async(MESSAGES, model=model)) if asynchronous else client.chat(MESSAGES, model=model)

    assert send()[0]["content"] == ("391" if previous == "native" else "from llama.cpp")
    initial = False
    with pytest.raises(OllamaRouteUnsettledError, match="Upgrade to 0.12.6") as caught:
        send()
    assert not caught.value.retryable and sends == ["/api/chat" if previous == "native" else "/v1/chat/completions"]


@pytest.mark.parametrize("asynchronous,previous", [(True, "compatible"), (False, "native"),
                                                   (True, "native"), (False, "compatible")])
def test_slot_wait_rechecks_the_program_before_the_only_post(install_http, monkeypatch, tmp_path, asynchronous, previous):
    from ouroboros import model_concurrency as slots, usage_accounting as ua
    from ouroboros.llm import LLMClient
    from ouroboros.llm_ollama import OllamaUpgradeRequired
    from tests._ollama import _compatible_target
    from tests._usage_store_testing import attempt_rows_in_start_order

    target = _compatible_target(monkeypatch, "http://127.1:11434/v1")
    server, compatible = OllamaHTTP(), LlamaCppHTTP()
    queued = []
    def handle(request):
        if queued and request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.12.6-rc.1"})
        return (compatible if previous == "compatible" and not queued else server)(request)
    install_http(handle)

    take = slots._Waiter.step
    def wait_once(waiter):
        if not queued:
            queued.append(waiter)
            return False
        assert queued == [waiter], "checking the admitted candidate must not reserve another slot"
        return take(waiter)
    monkeypatch.setattr(slots._Waiter, "step", wait_once)
    monkeypatch.setattr(slots, "_queue_root", lambda: tmp_path / "slots")
    monkeypatch.setattr(slots, "_cap", lambda local: 1)
    client, model = LLMClient(), target["capability_route"]["model"]
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="queue", root_task_id="queue")):
        with pytest.raises(OllamaUpgradeRequired, match="Upgrade to 0.12.6"):
            if asynchronous:
                asyncio.run(client.chat_async(MESSAGES, model=model))
            else:
                client.chat(MESSAGES, model=model)
    assert queued and not server.posts and ("POST", "/v1/chat/completions") not in compatible.paths
    [prepared] = attempt_rows_in_start_order(tmp_path)
    assert prepared["state"] == "released"


@pytest.mark.parametrize("ps", ["unsupported", "empty", "contradictory", "named"])
def test_passive_lan_replacement_requires_consistent_metadata(install_http, ps):
    """Only another program's own document retires a saved native route. Paths that state nothing leave it saved,
    for its next send to hold; contradictory metadata is no observation of the server at all."""
    from ouroboros import capability_evidence as ce
    from ouroboros.llm_ollama import is_ollama
    from ouroboros.llm_probe import upstream_transport_reachable

    target = {**TARGET, "base_url": "http://runner.lan/v1", "default_headers": {"X-Runner": "lab"}}
    install_http(OllamaHTTP())
    assert is_ollama(target)
    seen = []
    def replace(request):
        seen.append((request.method, request.url.path))
        if request.method == "HEAD":
            assert "authorization" not in request.headers and "X-Runner" not in request.headers
            return httpx.Response(200)
        assert request.method == "GET" and request.headers["X-Runner"] == "lab"
        if request.url.path == "/api/ps" and ps in ("empty", "contradictory"):
            return httpx.Response(200, json={} if ps == "empty" else {"models": []})
        if request.url.path == "/props" and ps == "named":
            return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 8192}, "total_slots": 1})
        return httpx.Response(200, json={"error": f"Unexpected endpoint or method. (GET {request.url.path})"})
    install_http(replace)
    llm = SimpleNamespace(_resolve_remote_target=lambda model: target, _no_proxy_timeout=httpx.Timeout)
    result = upstream_transport_reachable(llm, target["requested_model"], timeout=2)
    expected = [("GET", "/api/version"), ("GET", "/api/ps")]
    if ps == "contradictory":
        assert result == {}
    else:
        assert result["kind"] == "upstream_http"
        expected += [("GET", "/props"), *([] if ps == "named" else [("GET", "/api/v0/models")]), ("HEAD", "/v1")]
    assert seen == expected
    [saved] = ce._read(ce.canonical_evidence_root())["native_protocols"].values()
    assert (saved.get("protocol") == "ollama") is (ps != "named") and ("ollama_record" in target) is (ps != "named")


@pytest.mark.parametrize("ollama", [True, False])
def test_old_version_on_an_address_without_ps_needs_program_evidence(install_http, monkeypatch, ollama):
    from ouroboros.llm import LLMClient
    from ouroboros.llm_ollama import OllamaUpgradeRequired
    from tests._ollama import _compatible_target

    target = _compatible_target(monkeypatch, TARGET["base_url"])
    compatible, native = LlamaCppHTTP(), OllamaHTTP()
    def handle(request):
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.1.2"})
        if ollama and request.url.path == "/api/show":
            return native(request)
        return compatible(request)
    install_http(handle)
    client = LLMClient()
    if ollama:
        with pytest.raises(OllamaUpgradeRequired, match="0.1.2.*Upgrade"):
            client.chat(MESSAGES, model=target["capability_route"]["model"])
        assert not any(path.endswith("completions") for _, path in compatible.paths)
    else:
        assert client.chat(MESSAGES, model=target["capability_route"]["model"])[0]["content"] == "from llama.cpp"
