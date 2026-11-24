"""What remote MCP sign-in relies on beyond its loopback suite."""

from __future__ import annotations

import asyncio
import base64
import json
import urllib.parse

import httpx
import pytest

from ouroboros import mcp_oauth, oauth_engine

ISSUER = "https://issuer.example"
ENDPOINTS = {"authorization_endpoint": ISSUER + "/authorize", "token_endpoint": ISSUER + "/token",
             "revocation_endpoint": ISSUER + "/revoke", "authorization_response_iss_parameter_supported": True}
CLIENT = {"client_id": "desktop-client", "client_secret": None, "token_endpoint_auth_method": "none",
          "redirect_uri": "http://127.0.0.1:8765" + oauth_engine.RETURN_PATH, "issuer": ISSUER}


def _provider(monkeypatch, answer):
    seen = []

    def handler(request):
        seen.append(request)
        return answer(request)

    monkeypatch.setattr(oauth_engine, "http_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return seen


def test_the_mcp_return_rule_keeps_loopback_and_https_origins_and_falls_back_to_this_server(monkeypatch):
    monkeypatch.setattr(oauth_engine, "_server_port", lambda: 8765)
    back = oauth_engine.RETURN_PATH
    own = "http://127.0.0.1:8765" + back
    for base, expected in (("https://box.example/prefix", "https://box.example/prefix" + back),
                           ("http://localhost:8080", "http://localhost:8080" + back),
                           ("http://192.168.1.5:8765", own), ("", own)):
        assert oauth_engine.redirect_uri(base) == expected
    assert mcp_oauth.redirect_uri("") == own  # MCP keeps its rule


def test_an_mcp_state_is_unchanged_byte_for_byte_and_another_recipients_is_refused():
    mcp_state = oauth_engine.state("mcp", "docs", "n1")
    legacy = base64.urlsafe_b64encode(b'{"v":1,"mcp":"docs","nonce":"n1"}').decode("ascii").rstrip("=")
    assert mcp_state == legacy and mcp_oauth.server_of(mcp_state) == "docs"
    assert oauth_engine.envelope(mcp_state) == {"v": 1, "mcp": "docs", "nonce": "n1"}
    skills_own = base64.urlsafe_b64encode(json.dumps({"v": 1, "skill": "calendar", "nonce": "n3"}).encode()).decode()
    for kind in (None, "mcp"):
        with pytest.raises(ValueError):
            oauth_engine.envelope(skills_own, kind)
    with pytest.raises(ValueError, match="not an MCP sign-in return"):
        mcp_oauth.server_of(skills_own)
    for kind, name, nonce in (("skill", "docs", "n5"), ("mcp", "", "n5"), ("mcp", "docs", "")):
        with pytest.raises(ValueError):
            oauth_engine.state(kind, name, nonce)


def test_revoke_says_confirmed_not_confirmed_or_no_address(monkeypatch):
    tokens = {"access_token": "access-1", "refresh_token": "refresh-1"}
    seen = _provider(monkeypatch, lambda request: httpx.Response(200))
    assert asyncio.run(oauth_engine.revoke(ENDPOINTS["revocation_endpoint"], CLIENT, tokens)) is True
    assert [urllib.parse.parse_qs(request.content.decode())["token_type_hint"] for request in seen] == [
        ["refresh_token"], ["access_token"]]
    _provider(monkeypatch, lambda request: httpx.Response(
        400 if b"access-1" in request.content else 200, json={"error": "unsupported_token_type"}))
    assert asyncio.run(oauth_engine.revoke(ENDPOINTS["revocation_endpoint"], CLIENT, tokens)) is False

    def unreachable(request):
        raise httpx.ConnectError("no route", request=request)

    _provider(monkeypatch, unreachable)
    assert asyncio.run(oauth_engine.revoke(ENDPOINTS["revocation_endpoint"], CLIENT, tokens)) is False
    seen = _provider(monkeypatch, lambda request: httpx.Response(200))
    assert asyncio.run(oauth_engine.revoke(None, CLIENT, tokens)) is None
    assert asyncio.run(oauth_engine.revoke(ENDPOINTS["revocation_endpoint"], CLIENT, {})) is None and not seen


def test_a_declared_issuer_is_discovered_through_openid_configuration_and_another_is_refused(monkeypatch):
    document = {"issuer": ISSUER, "authorization_endpoint": ENDPOINTS["authorization_endpoint"],
                "token_endpoint": ENDPOINTS["token_endpoint"], "revocation_endpoint": ENDPOINTS["revocation_endpoint"],
                "userinfo_endpoint": ISSUER + "/userinfo", "response_types_supported": ["code"],
                "code_challenge_methods_supported": ["S256"]}
    seen = _provider(monkeypatch, lambda request: httpx.Response(200, json=document) if request.url.path == (
        "/.well-known/openid-configuration") else httpx.Response(404))

    async def discover(expected):
        async with oauth_engine.http_client() as http:
            return await oauth_engine.discover_issuer(ISSUER, ISSUER, expected, http)

    _metadata, endpoints, issuer = asyncio.run(discover(ISSUER))
    assert issuer == ISSUER and endpoints["token_endpoint"] == ENDPOINTS["token_endpoint"]
    assert [request.url.path for request in seen] == ["/.well-known/oauth-authorization-server",
                                                      "/.well-known/openid-configuration"]
    with pytest.raises(oauth_engine.SignInRequired) as caught:
        asyncio.run(discover("https://other.example"))
    assert (caught.value.code, caught.value.reason) == ("SIGN_IN_REQUIRED", "issuer_mismatch")
    mcp = mcp_oauth.MCPAuthRequired("docs", "issuer_mismatch")  # MCP's refusal is the engine's, with its own words
    assert isinstance(mcp, oauth_engine.SignInRequired) and mcp.code == "MCP_AUTH_REQUIRED"
    assert str(mcp).startswith("MCP_AUTH_REQUIRED: server 'docs' needs the owner's sign-in (issuer_mismatch)")
