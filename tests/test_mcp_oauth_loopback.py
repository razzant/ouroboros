"""A remote MCP server's sign-in against a real loopback authorization server (``ouroboros/mcp_oauth.py``).

One FastMCP app is also its own OAuth server through the SDK's provider interface: dynamic
registration, PKCE, refresh tokens that rotate (a reused one is refused, as for public clients),
revocation and an ``iss`` in every return. The real SDK transport lists and calls through
``StoredBearer``; ``begin``, the provider's redirect and ``complete`` run as the owner's browser and
the builtin return page would.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import socket
import threading
import time
import urllib.parse

import httpx
import pytest

from ouroboros import config, mcp_client, mcp_oauth
from tests.test_mcp_headers_loopback import loopback  # noqa: F401 -- the SSE fixture


class _Provider:
    """The SDK provider interface, in memory, auto-approving (no login page)."""

    def __init__(self, base: str) -> None:
        self.base, self.clients, self.codes, self.access, self.refresh = base, {}, {}, {}, {}
        self.counts = {"register": 0, "refresh_ok": 0, "refresh_refused": 0, "revoke": 0}

    async def get_client(self, client_id):
        return self.clients.get(client_id)

    async def register_client(self, client_info):
        self.counts["register"] += 1
        self.clients[client_info.client_id] = client_info

    async def authorize(self, client, params):
        from mcp.server.auth.provider import AuthorizationCode, construct_redirect_uri

        code = secrets.token_urlsafe(16)
        self.codes[code] = AuthorizationCode(
            code=code, scopes=params.scopes or ["mcp"], expires_at=time.time() + 300, client_id=client.client_id,
            code_challenge=params.code_challenge, redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly, resource=params.resource)
        return construct_redirect_uri(str(params.redirect_uri), code=code, state=params.state, iss=self.base + "/")

    async def load_authorization_code(self, client, code):
        return self.codes.get(code)

    def _issue(self, client_id, scopes):
        from mcp.server.auth.provider import AccessToken, RefreshToken
        from mcp.shared.auth import OAuthToken

        access, refresh = secrets.token_urlsafe(16), secrets.token_urlsafe(16)
        self.access[access] = AccessToken(token=access, client_id=client_id, scopes=scopes,
                                          expires_at=int(time.time()) + 3600)
        self.refresh[refresh] = RefreshToken(token=refresh, client_id=client_id, scopes=scopes)
        return OAuthToken(access_token=access, token_type="Bearer", expires_in=3600, refresh_token=refresh,
                          scope=" ".join(scopes))

    async def exchange_authorization_code(self, client, code):
        self.codes.pop(code.code, None)
        return self._issue(client.client_id, code.scopes)

    async def load_refresh_token(self, client, refresh_token):
        found = self.refresh.get(refresh_token)
        if found is None:
            self.counts["refresh_refused"] += 1
        return found

    async def exchange_refresh_token(self, client, refresh_token, scopes):
        from mcp.server.auth.provider import TokenError

        if self.refresh.pop(refresh_token.token, None) is None:  # rotation: each refresh token works once
            self.counts["refresh_refused"] += 1
            raise TokenError("invalid_grant", "refresh token already used")
        self.counts["refresh_ok"] += 1
        return self._issue(client.client_id, scopes or refresh_token.scopes)

    async def load_access_token(self, token):
        return self.access.get(token)

    async def revoke_token(self, token):
        self.counts["revoke"] += 1
        self.access.pop(token.token, None)
        self.refresh.pop(token.token, None)


def _serve_authorizing_server(running):
    import uvicorn
    from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
    from mcp.server.fastmcp import FastMCP

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    provider = _Provider(base)
    server = FastMCP("signin", host="127.0.0.1", port=port, log_level="WARNING", auth_server_provider=provider,
                     auth=AuthSettings(issuer_url=base, resource_server_url=base + "/mcp", required_scopes=["mcp"],
                                       client_registration_options=ClientRegistrationOptions(
                                           enabled=True, valid_scopes=["mcp"], default_scopes=["mcp"]),
                                       revocation_options=RevocationOptions(enabled=True)))

    @server.tool()
    def echo(text: str) -> str:
        return f"echo:{text}"

    uv = uvicorn.Server(uvicorn.Config(server.streamable_http_app(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=uv.run, name="mcp-oauth-loopback", daemon=True)
    thread.start()
    running.append((uv, thread))
    deadline = time.monotonic() + 10
    while not uv.started:
        assert thread.is_alive() and time.monotonic() < deadline, "loopback authorization server did not start"
        time.sleep(0.02)
    return provider, base + "/mcp"


@pytest.fixture
def signin(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mcp_client.MCPManager, "refresh_all_background", lambda *args, **kwargs: None)
    running = []
    mcp_client.reset_manager_for_tests()

    def start():
        provider, url = _serve_authorizing_server(running)
        settings = {"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": 20,
                    "MCP_SERVERS": [{"id": "docs", "enabled": True, "transport": "streamable_http", "url": url}]}
        manager = mcp_client.get_manager()
        manager.reconfigure(settings)
        return provider, manager, manager.server_config("docs"), settings

    try:
        yield start
    finally:
        mcp_client.reset_manager_for_tests()
        for server, thread in running:
            server.should_exit = True
            thread.join(10)


def _provider_return(cfg) -> str:
    """``begin``, then the owner's browser at the provider: the address it is sent back to."""
    started = asyncio.run(mcp_oauth.begin(cfg))
    answer = httpx.get(started["url"], follow_redirects=False, timeout=10)
    assert answer.status_code in (302, 303, 307), answer.text
    assert answer.headers["location"].startswith(started["redirect_uri"])
    return answer.headers["location"]


def _complete(cfg, address: str) -> dict:
    return asyncio.run(mcp_oauth.complete(cfg, **mcp_oauth.parse_return(address)))


def _file(cfg) -> dict:
    return json.loads(mcp_oauth._path(cfg.id).read_text(encoding="utf-8"))


def _sign_in_row(manager) -> dict:
    return next(row for row in manager.status_payload()["servers"] if row["id"] == "docs").get("sign_in") or {}


def test_an_unsigned_server_refuses_at_once_and_the_return_page_route_signs_it_in(signin, monkeypatch):
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from ouroboros.gateway import mcp as gateway_mcp

    provider, manager, cfg, settings = signin()
    began = time.monotonic()
    listed = manager.refresh_server("docs")
    assert listed["ok"] is False and listed["code"] == "MCP_AUTH_REQUIRED", listed
    assert time.monotonic() - began < 5  # a typed refusal, not a wait for a browser no one sees
    assert _sign_in_row(manager) == {"state": "sign_in_required", "reason": "not_signed_in",
                                     "since": _sign_in_row(manager)["since"], "issuer": ""}

    returned = _provider_return(cfg)
    monkeypatch.setattr(gateway_mcp, "load_settings", lambda: settings)
    client = TestClient(Starlette(routes=gateway_mcp.mcp_routes()))
    body = {"action": "complete", **mcp_oauth.parse_return(returned)}  # what the builtin return page posts
    answer = client.post("/api/mcp/sign-in", json=body).json()
    assert answer["ok"] is True and answer["server"] == "docs" and answer["listing"]["tool_count"] == 1, answer
    assert client.post("/api/mcp/sign-in", json=body).json()["reason"] == "sign_in_superseded"  # no replay

    called = manager._call_tool_result("mcp_docs__echo", {"text": "hi"})
    assert called.status == "ok" and "echo:hi" in called.text
    row, stored = _sign_in_row(manager), _file(cfg)
    assert row["state"] == "signed_in" and row["issuer"] == stored["issuer"] and provider.counts["register"] == 1
    assert stored["tokens"]["access_token"] not in json.dumps(manager.status_payload())
    assert os.stat(mcp_oauth._path(cfg.id)).st_mode & 0o777 == 0o600 or os.name == "nt"


def test_refresh_needs_no_sign_in_happens_once_for_concurrent_callers_and_a_revoked_grant_refuses(signin):
    provider, manager, cfg, _settings = signin()
    _complete(cfg, _provider_return(cfg))
    assert manager.refresh_server("docs")["ok"] is True
    path = mcp_oauth._path(cfg.id)

    # The recorded expiry passed: two callers (threads, or processes on the same lock file) at once.
    doc = _file(cfg)
    doc["tokens"]["expires_at"] = time.time() - 1
    path.write_text(json.dumps(doc), encoding="utf-8")
    tokens = []
    callers = [threading.Thread(target=lambda: tokens.append(asyncio.run(mcp_oauth.current_token(cfg))))
               for _ in range(2)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(30)
    assert len(tokens) == 2 and len(set(tokens)) == 1
    assert provider.counts["refresh_ok"] == 1 and provider.counts["refresh_refused"] == 0  # rotation survived

    # The server forgot the access token (a 401 challenge): one refresh, one retry, the call succeeds.
    provider.access.clear()
    called = manager._call_tool_result("mcp_docs__echo", {"text": "again"})
    assert called.status == "ok" and "echo:again" in called.text, called.text
    assert provider.counts["refresh_ok"] == 2

    # Revoked at the provider: refused at once, the dead grant is not kept, the card says why.
    provider.access.clear()
    provider.refresh.clear()
    began = time.monotonic()
    refused = manager._call_tool_result("mcp_docs__echo", {"text": "gone"})
    assert refused.status == "unavailable" and refused.code == "MCP_AUTH_REQUIRED", refused.text
    assert refused.meta["mcp_call_not_sent"] is True and refused.meta["sign_in_reason"] == "refresh_rejected"
    assert time.monotonic() - began < 5
    assert "tokens" not in _file(cfg) and _sign_in_row(manager)["reason"] == "refresh_rejected"


def test_returns_that_do_not_match_the_pending_sign_in_are_refused(signin):
    provider, manager, cfg, _settings = signin()
    older = _provider_return(cfg)
    newer = _provider_return(cfg)  # a later begin replaces the earlier one, its client reused
    assert provider.counts["register"] == 1

    def refusal(address, **changes):
        fields = {**mcp_oauth.parse_return(address), **changes}
        with pytest.raises(mcp_oauth.MCPAuthRequired) as caught:
            asyncio.run(mcp_oauth.complete(cfg, **{key: value for key, value in fields.items() if value is not None}))
        return caught.value.reason

    assert refusal(older) == "sign_in_superseded"
    assert refusal(newer, iss="https://elsewhere.example/") == "issuer_mismatch"  # RFC 9207
    doc = _file(cfg)
    doc["pending"]["metadata"]["authorization_response_iss_parameter_supported"] = True
    mcp_oauth._path(cfg.id).write_text(json.dumps(doc), encoding="utf-8")
    query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(newer).query))
    with pytest.raises(mcp_oauth.MCPAuthRequired) as caught:  # promised an iss, sent none
        asyncio.run(mcp_oauth.complete(cfg, state=query["state"], code=query["code"]))
    assert caught.value.reason == "issuer_missing"
    assert "tokens" not in _file(cfg)  # refusals changed nothing
    assert _complete(cfg, newer)["ok"] is True and _sign_in_row(manager)["state"] == "signed_in"


def test_the_owners_own_app_signs_in_with_its_secret_and_sign_out_revokes(signin):
    from mcp.shared.auth import OAuthClientInformationFull

    provider, manager, cfg, _settings = signin()
    provider.clients["owner-app"] = OAuthClientInformationFull(  # registered by the owner at the provider
        client_id="owner-app", client_secret="owner-app-secret-0001", redirect_uris=[mcp_oauth.redirect_uri("")],
        token_endpoint_auth_method="client_secret_post", grant_types=["authorization_code", "refresh_token"],
        response_types=["code"], scope="mcp")
    started = asyncio.run(mcp_oauth.begin(cfg, client_id="owner-app", client_secret="owner-app-secret-0001"))
    answer = httpx.get(started["url"], follow_redirects=False, timeout=10)
    assert _complete(cfg, answer.headers["location"])["ok"] is True and provider.counts["register"] == 0
    stored = _file(cfg)
    assert stored["client"]["source"] == "owner" and "owner-app-secret-0001" in mcp_oauth.secret_values(cfg.id)

    out = asyncio.run(mcp_oauth.sign_out(cfg))
    assert out == {"ok": True, "server": "docs", "revoked": True} and provider.counts["revoke"] == 2
    assert stored["tokens"]["refresh_token"] not in provider.refresh
    assert stored["tokens"]["access_token"] not in provider.access
    assert "tokens" not in _file(cfg) and _file(cfg)["client"]["client_id"] == "owner-app"
    assert _sign_in_row(manager)["reason"] == "signed_out"


def test_the_minds_sign_in_hands_the_new_listing_to_the_server_process(signin, monkeypatch):
    import queue
    import types

    from ouroboros.tools.mcp_sign_in import _mcp_sign_in
    from supervisor.events_worker_reports import _handle_mcp_signed_in

    provider, manager, cfg, settings = signin()
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    events = queue.Queue()
    ctx = types.SimpleNamespace(event_queue=events)
    began = json.loads(_mcp_sign_in(ctx, action="begin", server="docs"))
    assert began["server"] == "docs" and began["url"].startswith(provider.base) and "note" in began
    location = httpx.get(began["url"], follow_redirects=False, timeout=10).headers["location"]
    done = json.loads(_mcp_sign_in(ctx, action="complete", returned_url=location))
    assert done["ok"] is True and done["catalog_refresh"] == "queued_for_host"
    handed = events.get_nowait()
    assert handed == {"type": "mcp_signed_in", "server_id": "docs"}

    _handle_mcp_signed_in(handed, None)  # the server process lists that server on its own thread
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not any(
            row["id"] == "docs" and row.get("tool_count") for row in manager.status_payload()["servers"]):
        time.sleep(0.05)
    assert next(row for row in manager.status_payload()["servers"] if row["id"] == "docs")["tool_count"] == 1
    refused = _mcp_sign_in(ctx, action="complete", returned_url="https://elsewhere.example/?state=x")
    assert "TOOL_ARG_ERROR" in refused  # no code: not a provider's return
    out = json.loads(_mcp_sign_in(ctx, action="sign_out", server="docs"))
    assert out["ok"] is True and _sign_in_row(manager)["reason"] == "signed_out"


def test_a_bare_401_without_a_bearer_challenge_stays_an_ordinary_error(tmp_path, monkeypatch):
    from tests.test_mcp_headers_loopback import start_asgi

    async def bare_401(scope, receive, send):
        if scope["type"] == "http":
            await send({"type": "http.response.start", "status": 401, "headers": []})
            await send({"type": "http.response.body", "body": b"no"})

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    running = []
    mcp_client.reset_manager_for_tests()
    try:
        port = start_asgi(bare_401, running)
        manager = mcp_client.get_manager()
        manager.reconfigure({"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": 10, "MCP_SERVERS": [
            {"id": "plain", "enabled": True, "url": f"http://127.0.0.1:{port}/mcp"}]})
        outcome = manager.refresh_server("plain")
        assert outcome["ok"] is False and outcome.get("code") != "MCP_AUTH_REQUIRED"
        assert not mcp_oauth._path("plain").exists() and "sign_in" not in json.dumps(manager.status_payload())
    finally:
        mcp_client.reset_manager_for_tests()
        for server, thread in running:
            server.should_exit = True
            thread.join(10)


# -- review findings (MCP_RV1), each against its producer ------------------------------------------------------


def _mock_as(monkeypatch, tmp_path, *, issuer="https://auth.example", token_scope=None, prm_status=200):
    """An authorization server in memory: resource metadata (or not), metadata, registration and tokens."""
    import types

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    cfg = mcp_client.normalize_server_config({"id": "rv1", "enabled": True, "url": "https://mcp.example/mcp"})
    base = issuer.rstrip("/")
    metadata = {"issuer": issuer, "authorization_endpoint": base + "/authorize", "token_endpoint": base + "/token",
                "registration_endpoint": base + "/register", "code_challenge_methods_supported": ["S256"],
                "authorization_response_iss_parameter_supported": True}
    seen = types.SimpleNamespace(paths=[])

    def answer(request):
        seen.paths.append(request.url.path)
        if "oauth-protected-resource" in request.url.path:
            if prm_status != 200:
                return httpx.Response(prm_status)
            return httpx.Response(200, json={"resource": cfg.url, "authorization_servers": [issuer],
                                             "scopes_supported": ["files:read"]})
        if "oauth-authorization-server" in request.url.path:
            return httpx.Response(200, json=metadata)
        if request.url.path == "/register":
            return httpx.Response(201, json={**json.loads(request.content), "client_id": "synthetic-client"})
        if request.url.path == "/token":
            body = {"access_token": "synthetic-access", "refresh_token": "synthetic-refresh", "expires_in": 3600,
                    "token_type": "Bearer", **({"scope": token_scope} if token_scope is not None else {})}
            return httpx.Response(200, json=body)
        raise AssertionError(str(request.url))

    monkeypatch.setattr(mcp_oauth, "_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(answer)))
    return cfg, metadata, seen


def _state_of(started) -> str:
    return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(started["url"]).query))["state"]


def _bearer_call(cfg, resource):
    async def call():
        async with httpx.AsyncClient(auth=mcp_oauth.StoredBearer(cfg), transport=httpx.MockTransport(resource)) as client:
            with pytest.raises(mcp_oauth.MCPAuthRequired) as caught:
                await client.get(cfg.url)
            return caught.value

    return asyncio.run(call())


def test_an_issuer_published_without_a_trailing_slash_is_compared_as_published(monkeypatch, tmp_path):
    cfg, _metadata, _seen = _mock_as(monkeypatch, tmp_path)  # "https://auth.example", as the RFC 9207 iss sends it
    started = asyncio.run(mcp_oauth.begin(cfg))
    assert asyncio.run(mcp_oauth.complete(cfg, state=_state_of(started), code="c", iss="https://auth.example"))["ok"]
    assert _file(cfg)["issuer"] == "https://auth.example"


def test_without_resource_metadata_the_issuer_must_be_the_servers_origin(monkeypatch, tmp_path):
    cfg, metadata, _seen = _mock_as(monkeypatch, tmp_path, issuer="https://mcp.example", prm_status=404)
    assert asyncio.run(mcp_oauth.begin(cfg))["url"].startswith("https://mcp.example/authorize")  # legacy, honest
    metadata["issuer"] = "https://elsewhere.example"
    with pytest.raises(mcp_oauth.MCPAuthRequired) as caught:
        asyncio.run(mcp_oauth.begin(cfg))
    assert caught.value.reason == "issuer_mismatch"


def test_a_busy_resource_server_stops_discovery_instead_of_trying_another_issuer(monkeypatch, tmp_path):
    cfg, _metadata, seen = _mock_as(monkeypatch, tmp_path, prm_status=503)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(mcp_oauth.begin(cfg))
    assert not any("oauth-authorization-server" in path for path in seen.paths)


def test_a_draft_test_of_another_address_never_replaces_the_saved_sign_in(signin):
    _provider, manager, cfg, settings = signin()
    _complete(cfg, _provider_return(cfg))
    before = _file(cfg)
    draft = {**settings["MCP_SERVERS"][0], "url": cfg.url + "?edited-draft=1"}
    assert manager.test_server(draft)["ok"] is False  # the draft asked for sign-in ...
    assert _file(cfg) == before  # ... and the saved server keeps its grant
    assert manager.test_server(settings["MCP_SERVERS"][0])["ok"] is True  # the saved address still uses it


def test_an_sse_post_refused_for_sign_in_ends_the_session_at_once(loopback, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mcp_client.MCPManager, "refresh_all_background", lambda *args, **kwargs: None)

    class RefusePosts:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http" and scope["method"] == "POST":
                await send({"type": "http.response.start", "status": 401, "headers": [(b"www-authenticate", b"Bearer")]})
                await send({"type": "http.response.body", "body": b"no grant"})
                return
            await self.app(scope, receive, send)

    _app, url = loopback("sse", {}, wrap=RefusePosts)
    manager = mcp_client.get_manager()
    manager.reconfigure({"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": 10, "MCP_SERVERS": [
        {"id": "sseprobe", "enabled": True, "transport": "sse", "url": url}]})
    began = time.monotonic()
    result = manager.refresh_server("sseprobe")
    assert result.get("code") == "MCP_AUTH_REQUIRED", result
    assert time.monotonic() - began < 5  # not the 10 s tool timeout


def test_insufficient_scope_keeps_the_grant_and_shows_the_step_up(monkeypatch, tmp_path):
    cfg, _metadata, _seen = _mock_as(monkeypatch, tmp_path)
    mcp_oauth._save(cfg, {"tokens": {"access_token": "synthetic-access", "scope": "files:read"}})
    refused = _bearer_call(cfg, lambda request: httpx.Response(403, headers={
        "WWW-Authenticate": 'Bearer error="insufficient_scope", scope="files:write"'}))
    assert refused.reason == "insufficient_scope"
    row = mcp_oauth.status(cfg)
    assert (row["state"], row["reason"], row["required_scope"]) == ("signed_in", "insufficient_scope", "files:read files:write")


def test_a_403_after_a_refreshed_retry_is_still_the_typed_step_up(monkeypatch, tmp_path):
    cfg, _metadata, _seen = _mock_as(monkeypatch, tmp_path)
    mcp_oauth._save(cfg, {"tokens": {"access_token": "old-access", "refresh_token": "synthetic-refresh"},
                          "metadata": {"token_endpoint": "https://auth.example/token"}, "client": {"client_id": "c"}})
    answers = [httpx.Response(401, headers={"WWW-Authenticate": "Bearer"}),
               httpx.Response(403, headers={"WWW-Authenticate": 'Bearer error="insufficient_scope", scope="write"'})]
    assert _bearer_call(cfg, lambda request: answers.pop(0)).reason == "insufficient_scope"
    assert mcp_oauth._load(cfg)["required_scope"] == "write"


def test_two_requests_of_one_event_loop_share_one_refresh_without_blocking_it(monkeypatch, tmp_path):
    cfg, _metadata, _seen = _mock_as(monkeypatch, tmp_path)
    mcp_oauth._save(cfg, {"tokens": {"access_token": "old-access", "expires_at": 1, "refresh_token": "synthetic-refresh"},
                          "metadata": {"token_endpoint": "https://auth.example/token"}, "client": {"client_id": "c"}})
    monkeypatch.setattr(mcp_oauth, "MCP_OAUTH_LOCK_WAIT_SEC", 0.25)  # only the failure wait is shortened
    refreshes = []

    async def answer(request):
        refreshes.append(request.url.path)
        await asyncio.sleep(0.05)
        return httpx.Response(200, json={"access_token": "new-access", "expires_in": 3600})

    monkeypatch.setattr(mcp_oauth, "_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(answer)))

    async def both():
        return await asyncio.gather(mcp_oauth.current_token(cfg), mcp_oauth.current_token(cfg), return_exceptions=True)

    assert asyncio.run(both()) == ["new-access", "new-access"] and refreshes == ["/token"]


def test_a_scope_the_token_endpoint_does_not_repeat_survives_a_step_up(monkeypatch, tmp_path):
    cfg, _metadata, _seen = _mock_as(monkeypatch, tmp_path)  # token answers omit scope (RFC 6749 §5.1: unchanged)
    started = asyncio.run(mcp_oauth.begin(cfg))
    asyncio.run(mcp_oauth.complete(cfg, state=_state_of(started), code="c", iss="https://auth.example"))
    assert _file(cfg)["tokens"]["scope"] == "files:read"
    _bearer_call(cfg, lambda request: httpx.Response(403, headers={
        "WWW-Authenticate": 'Bearer error="insufficient_scope", scope="files:write"'}))
    again = asyncio.run(mcp_oauth.begin(cfg))
    asked = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(again["url"]).query))["scope"]
    assert set(asked.split()) == {"files:read", "files:write"}


def test_a_malformed_refresh_keeps_the_grant_and_a_refused_one_asks_for_sign_in(monkeypatch, tmp_path):
    cfg, _metadata, _seen = _mock_as(monkeypatch, tmp_path)
    grant = {"tokens": {"access_token": "old-access", "expires_at": 1, "refresh_token": "synthetic-refresh"},
             "metadata": {"token_endpoint": "https://auth.example/token"}, "client": {"client_id": "c"}}
    for error, kept in (("invalid_request", True), ("invalid_grant", False)):
        mcp_oauth._save(cfg, grant)
        monkeypatch.setattr(mcp_oauth, "_client", lambda error=error: httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(400, json={"error": error}))))
        with pytest.raises(httpx.HTTPStatusError if kept else mcp_oauth.MCPAuthRequired):
            asyncio.run(mcp_oauth.current_token(cfg))
        assert ("tokens" in _file(cfg)) is kept, error


def test_parallel_streams_of_a_real_session_survive_one_rejected_access_token(signin, monkeypatch):
    provider, manager, cfg, _settings = signin()
    _complete(cfg, _provider_return(cfg))
    old_access = _file(cfg)["tokens"]["access_token"]
    load, refresh = provider.load_access_token, provider.exchange_refresh_token
    old_seen, both_rejected = 0, None

    async def reject_after_initialize(token):  # the GET stream and the POST are both refused together
        nonlocal old_seen, both_rejected
        if token == old_access:
            old_seen += 1
            if old_seen > 1:
                if both_rejected is None:
                    both_rejected = asyncio.Event()
                else:
                    both_rejected.set()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(both_rejected.wait(), 2)
                return None
        return await load(token)

    async def slow_refresh(client, refresh_token, scopes):
        await asyncio.sleep(0.1)
        return await refresh(client, refresh_token, scopes)

    monkeypatch.setattr(provider, "load_access_token", reject_after_initialize)
    monkeypatch.setattr(provider, "exchange_refresh_token", slow_refresh)
    monkeypatch.setattr(mcp_oauth, "MCP_OAUTH_LOCK_WAIT_SEC", 0.25)
    assert manager.refresh_server("docs")["ok"] is True


def test_the_sign_in_tool_honours_the_tasks_network_contract(tmp_path):
    from ouroboros.tools import registry_guards
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    repo, drive = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    drive.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id="oauth-gate",
                      task_metadata={"task_contract": {"allowed_resources": {"network": False}}})
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry.set_context(ctx)
    args = {"action": "begin", "server": "docs"}
    assert registry.get_schema_by_name("remote_mcp_sign_in") is None
    assert getattr(registry_guards._capability_resource_guard_result(ctx, "remote_mcp_sign_in", args), "code",
                   None) == "RESOURCE_CONSTRAINT_BLOCKED"
    ctx.task_metadata["task_contract"]["allowed_resources"]["network"] = True
    assert registry.get_schema_by_name("remote_mcp_sign_in") is not None
    assert registry_guards._capability_resource_guard_result(ctx, "remote_mcp_sign_in", args) is None


# -- review findings (MCP_RV2) ------------------------------------------------------------------------------


def test_a_revoked_registration_is_not_reused_from_an_unfinished_sign_in(signin, monkeypatch):
    provider, _manager, cfg, _settings = signin()
    _complete(cfg, _provider_return(cfg))
    original = _file(cfg)
    asyncio.run(mcp_oauth.begin(cfg))  # a step-up begun and left pending, with the same registration
    provider.clients.clear()  # the provider revokes the registration

    async def invalid_client(http, endpoint, client, data):
        return httpx.Response(401, json={"error": "invalid_client"}, request=httpx.Request("POST", endpoint))

    with monkeypatch.context() as stub:
        stub.setattr(mcp_oauth, "_token_request", invalid_client)
        with pytest.raises(mcp_oauth.MCPAuthRequired):
            asyncio.run(mcp_oauth.current_token(cfg, rejected=original["tokens"]["access_token"]))
    assert not _file(cfg).get("client") and not _file(cfg).get("pending") and not _file(cfg).get("tokens")
    restarted = asyncio.run(mcp_oauth.begin(cfg))
    client_id = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(restarted["url"]).query))["client_id"]
    assert client_id in provider.clients and provider.counts["register"] == 2  # registered again
    answer = httpx.get(restarted["url"], follow_redirects=False, timeout=10)
    assert _complete(cfg, answer.headers["location"])["ok"] is True


def test_invalid_client_keeps_a_pending_sign_in_of_another_registration(monkeypatch, tmp_path):
    cfg, _metadata, _seen = _mock_as(monkeypatch, tmp_path)
    pending = {"nonce": "n", "client": {"client_id": "replacement-client", "issuer": "https://auth.example"}}
    mcp_oauth._save(cfg, {"tokens": {"access_token": "a", "refresh_token": "r", "expires_at": 1}, "pending": pending,
                          "metadata": {"token_endpoint": "https://auth.example/token"},
                          "client": {"client_id": "old-client", "issuer": "https://auth.example"}})
    monkeypatch.setattr(mcp_oauth, "_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(401, json={"error": "invalid_client"}))))
    with pytest.raises(mcp_oauth.MCPAuthRequired):
        asyncio.run(mcp_oauth.current_token(cfg))
    assert _file(cfg)["pending"] == pending and not _file(cfg).get("client")


@pytest.mark.parametrize(("status", "error"), [(400, None), (401, None), (401, "invalid_client")])
def test_refresh_refusals_without_a_body_ask_for_sign_in(monkeypatch, tmp_path, status, error):
    cfg, _metadata, _seen = _mock_as(monkeypatch, tmp_path)
    mcp_oauth._save(cfg, {"tokens": {"access_token": "a", "refresh_token": "r", "expires_at": 1},
                          "metadata": {"token_endpoint": "https://auth.example/token"}, "client": {"client_id": "c"}})
    monkeypatch.setattr(mcp_oauth, "_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(status, **({"json": {"error": error}} if error else {})))))
    with pytest.raises(mcp_oauth.MCPAuthRequired) as caught:
        asyncio.run(mcp_oauth.current_token(cfg))
    assert caught.value.reason == "refresh_rejected" and not _file(cfg).get("tokens")
    assert bool(_file(cfg).get("client")) is (error != "invalid_client")


def test_saving_a_new_address_while_an_old_refresh_finishes_allows_the_new_sign_in(signin, monkeypatch):
    _provider, manager, cfg, settings = signin()
    _complete(cfg, _provider_return(cfg))
    old_access = _file(cfg)["tokens"]["access_token"]
    newer = {**settings, "MCP_SERVERS": [{**settings["MCP_SERVERS"][0], "url": cfg.url + "?saved-new-address=1"}]}
    real_token_request = mcp_oauth._token_request

    async def interleave():
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed(http, endpoint, client, data):
            entered.set()
            await release.wait()
            return await real_token_request(http, endpoint, client, data)

        with monkeypatch.context() as stub:
            stub.setattr(mcp_oauth, "_token_request", delayed)
            refresh = asyncio.create_task(mcp_oauth.current_token(cfg, rejected=old_access))
            await entered.wait()
            manager.configuration_changed(newer, reason="settings")  # Save prunes the old address's file
            release.set()
            await refresh  # ... and the old refresh writes it back

    asyncio.run(interleave())
    new_cfg = manager.server_config("docs")
    assert _complete(new_cfg, _provider_return(new_cfg))["ok"] is True


def test_a_contended_lock_does_not_keep_its_finished_event_loop(monkeypatch, tmp_path):
    import gc
    import weakref

    cfg, _metadata, _seen = _mock_as(monkeypatch, tmp_path)

    async def contend():
        loop_ref, ready = weakref.ref(asyncio.get_running_loop()), asyncio.Event()

        async def holder():
            async with mcp_oauth._locked(cfg):
                ready.set()
                await asyncio.sleep(0.02)

        async def waiter():
            await ready.wait()
            async with mcp_oauth._locked(cfg):
                pass

        await asyncio.gather(holder(), waiter())
        return loop_ref

    loop_ref = asyncio.run(contend())
    gc.collect()
    assert loop_ref() is None


# -- review findings (MCP_RV3) ------------------------------------------------------------------------------


@pytest.mark.parametrize("replacement", ["issuer", "secret"])
def test_invalid_client_of_the_old_grant_keeps_a_pending_sign_in_with_new_credentials(monkeypatch, tmp_path,
                                                                                         replacement):
    issuer = "https://auth.example"
    cfg, _metadata, _seen = _mock_as(monkeypatch, tmp_path, issuer=issuer)
    first = asyncio.run(mcp_oauth.begin(cfg, client_id="owner-app", client_secret="old-secret"))
    assert asyncio.run(mcp_oauth.complete(cfg, state=_state_of(first), code="first", iss=issuer))["ok"]
    old_access = _file(cfg)["tokens"]["access_token"]
    new_issuer = "https://replacement.example" if replacement == "issuer" else issuer
    new_secret = "new-secret" if replacement == "secret" else "old-secret"
    _mock_as(monkeypatch, tmp_path, issuer=new_issuer)  # the same MCP address, newly published metadata
    started = asyncio.run(mcp_oauth.begin(cfg, client_id="owner-app", client_secret=new_secret))

    async def refused_old_client(http, endpoint, client, data):
        assert client["client_secret"] == "old-secret"
        return httpx.Response(401, json={"error": "invalid_client"}, request=httpx.Request("POST", endpoint))

    with monkeypatch.context() as stub:
        stub.setattr(mcp_oauth, "_token_request", refused_old_client)
        with pytest.raises(mcp_oauth.MCPAuthRequired):
            asyncio.run(mcp_oauth.current_token(cfg, rejected=old_access))
    assert asyncio.run(mcp_oauth.complete(cfg, state=_state_of(started), code="new", iss=new_issuer))["ok"]


def test_a_new_pending_sign_in_survives_an_old_refresh_that_finishes_after_it(signin, monkeypatch):
    from ouroboros import store

    _provider, manager, cfg, settings = signin()
    _complete(cfg, _provider_return(cfg))
    old_access = _file(cfg)["tokens"]["access_token"]
    newer = {**settings, "MCP_SERVERS": [{**settings["MCP_SERVERS"][0], "url": cfg.url + "?saved=2"}]}
    real_token_request, real_save, real_name_lock = mcp_oauth._token_request, mcp_oauth._save, store.name_lock
    entered, release, failures = threading.Event(), threading.Event(), []
    main_thread = threading.current_thread()

    # The old refresh resumes when the new begin either writes its pending (the broken order) or finds the
    # lock busy (the fixed one); no scheduling sleep decides the order.
    def saved(new_cfg, doc):
        real_save(new_cfg, doc)
        if new_cfg.url == newer["MCP_SERVERS"][0]["url"]:
            release.set()

    @contextlib.contextmanager
    def observed_lock(root, filename, **kwargs):
        try:
            with real_name_lock(root, filename, **kwargs):
                yield
        except store.StoreBusy:
            if threading.current_thread() is main_thread and filename == f"mcp/oauth/server-{cfg.id}.lock":
                release.set()
            raise

    async def delayed(http, endpoint, client, data):
        if data["grant_type"] == "refresh_token":
            entered.set()
            assert await asyncio.to_thread(release.wait, 20)
        return await real_token_request(http, endpoint, client, data)

    def old_refresh():
        try:
            asyncio.run(mcp_oauth.current_token(cfg, rejected=old_access))
        except BaseException as exc:  # noqa: BLE001 -- reported below
            failures.append(exc)

    monkeypatch.setattr(mcp_oauth, "_token_request", delayed)
    monkeypatch.setattr(mcp_oauth, "_save", saved)
    monkeypatch.setattr(store, "name_lock", observed_lock)
    thread = threading.Thread(target=old_refresh, name="mcp-oauth-old-refresh")
    thread.start()
    try:
        assert entered.wait(10)
        manager.configuration_changed(newer, reason="settings")  # Save prunes the old address's file
        new_cfg = manager.server_config("docs")
        returned = _provider_return(new_cfg)
    finally:
        release.set()
        thread.join(20)
    assert not thread.is_alive() and not failures
    assert _complete(new_cfg, returned)["ok"] is True
