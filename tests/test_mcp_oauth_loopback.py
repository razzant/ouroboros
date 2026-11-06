"""A remote MCP server's sign-in against a real loopback authorization server (``ouroboros/mcp_oauth.py``).

One FastMCP app is also its own OAuth server through the SDK's provider interface: dynamic
registration, PKCE, refresh tokens that rotate (a reused one is refused, as for public clients),
revocation and an ``iss`` in every return. The real SDK transport lists and calls through
``StoredBearer``; ``begin``, the provider's redirect and ``complete`` run as the owner's browser and
the builtin return page would.
"""

from __future__ import annotations

import asyncio
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
