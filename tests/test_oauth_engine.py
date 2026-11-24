"""The OAuth engine's parameters for its second consumer (``ouroboros/oauth_engine.py``).

Remote MCP servers exercise the whole engine against a real loopback authorization server
(``tests/test_mcp_oauth_loopback.py``); these tests pin what MCP does not use: the desktop and exact return
rules, the ``sign_in`` kind of the routing state, ``min_valid``, extra authorize parameters, a completion that an
identity callback files under another grant, discovery of a declared issuer and the three answers of
``revoke``. Each pins the parameter's working case and its refused one.
"""

from __future__ import annotations

import asyncio
import base64
import functools
import json
import time
import urllib.parse
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import config, mcp_oauth, oauth_engine, service_sign_in
from ouroboros import service_sign_in_declaration as declaration

ISSUER = "https://issuer.example"
ENDPOINTS = {"authorization_endpoint": ISSUER + "/authorize", "token_endpoint": ISSUER + "/token",
             "revocation_endpoint": ISSUER + "/revoke", "authorization_response_iss_parameter_supported": True}
CLIENT = {"client_id": "desktop-client", "client_secret": None, "token_endpoint_auth_method": "none",
          "redirect_uri": "http://127.0.0.1:8765" + oauth_engine.RETURN_PATH, "issuer": ISSUER}


def _grant(tmp_path, name: str) -> oauth_engine.Grant:
    return oauth_engine.Grant(path=tmp_path / "state" / "sign_in" / "grants" / "calendar" / "google" / f"{name}.json",
                              lock=f"sign_in/grants/calendar/google/{name}.lock",
                              binding={"skill": "calendar", "service": "google"})


def _provider(monkeypatch, answer):
    seen = []

    def handler(request):
        seen.append(request)
        return answer(request)

    monkeypatch.setattr(oauth_engine, "http_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return seen


def test_return_rules_keep_the_mcp_rule_and_add_the_desktop_and_exact_ones(monkeypatch):
    monkeypatch.setattr(oauth_engine, "_server_port", lambda: 8765)
    back = oauth_engine.RETURN_PATH
    own = "http://127.0.0.1:8765" + back
    for base, mcp_rule, desktop in (("https://box.example/prefix", "https://box.example/prefix" + back, own),
                                    ("http://localhost:8080", "http://localhost:8080" + back,
                                     "http://localhost:8080" + back),
                                    ("http://192.168.1.5:8765", own, own)):
        assert oauth_engine.redirect_uri(base, "loopback_or_https") == mcp_rule
        assert oauth_engine.redirect_uri(base, "loopback") == desktop  # an HTTPS tunnel never reaches a desktop client
    assert mcp_oauth.redirect_uri("") == own  # MCP keeps its rule
    registered = "https://box.example/onboarding/static/provider-return.html"
    assert oauth_engine.redirect_uri("http://localhost:8080", f"exact:{registered}") == registered
    assert oauth_engine.redirect_uri("", "exact:http://127.0.0.1:9000/back") == "http://127.0.0.1:9000/back"
    for refused in ("exact:http://192.168.1.5/onboarding/static/provider-return.html", "exact:https://", "desktop"):
        with pytest.raises(ValueError):
            oauth_engine.redirect_uri("https://box.example", refused)


def test_a_state_names_one_recipient_and_another_recipients_is_refused():
    mcp_state = oauth_engine.state("mcp", "docs", "n1")
    legacy = base64.urlsafe_b64encode(b'{"v":1,"mcp":"docs","nonce":"n1"}').decode("ascii").rstrip("=")
    assert mcp_state == legacy and mcp_oauth.server_of(mcp_state) == "docs"  # an MCP state is unchanged byte for byte
    sign_in = oauth_engine.state("sign_in", "calendar", "n2", svc="google")
    assert oauth_engine.envelope(sign_in, "sign_in") == {"v": 1, "sign_in": "calendar", "nonce": "n2", "svc": "google"}
    assert oauth_engine.envelope(sign_in)["sign_in"] == "calendar" and oauth_engine.envelope(mcp_state)["mcp"] == "docs"
    with pytest.raises(ValueError):
        oauth_engine.envelope(sign_in, "mcp")
    with pytest.raises(ValueError, match="not an MCP sign-in return"):
        mcp_oauth.server_of(sign_in)
    skills_own = base64.urlsafe_b64encode(json.dumps({"v": 1, "skill": "calendar", "nonce": "n3"}).encode()).decode()
    no_service = base64.urlsafe_b64encode(json.dumps({"v": 1, "sign_in": "calendar", "nonce": "n4"}).encode()).decode()
    for foreign, kind in ((skills_own, None), (skills_own, "sign_in"), (no_service, "sign_in"), (mcp_state, "skill")):
        with pytest.raises(ValueError):
            oauth_engine.envelope(foreign, kind)
    for kind, extra in (("sign_in", {}), ("skill", {}), ("mcp", {"v": "2"}), ("mcp", {"sign_in": "calendar"})):
        with pytest.raises(ValueError):
            oauth_engine.state(kind, "calendar", "n5", **extra)


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


def test_min_valid_refreshes_a_token_that_expires_sooner_and_stays_a_transport_bound(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    grant = _grant(tmp_path, "a-0001")
    oauth_engine.save(grant, {"tokens": {"access_token": "old-access", "refresh_token": "refresh-1",
                                         "expires_at": time.time() + 100},
                              "metadata": ENDPOINTS, "client": CLIENT})
    seen = _provider(monkeypatch, lambda request: httpx.Response(200, json={
        "access_token": "new-access", "expires_in": 3600, "token_type": "Bearer"}))
    assert asyncio.run(oauth_engine.current_token(grant)) == "old-access" and not seen
    assert asyncio.run(oauth_engine.current_token(grant, min_valid=60)) == "old-access" and not seen
    assert asyncio.run(oauth_engine.current_token(grant, min_valid=200)) == "new-access" and len(seen) == 1
    assert oauth_engine.load(grant)["tokens"]["refresh_token"] == "refresh-1"  # an omitted refresh token is unchanged
    for bound in (oauth_engine.OAUTH_MIN_VALID_MAX_SEC + 1, -1):
        with pytest.raises(ValueError):
            asyncio.run(oauth_engine.current_token(grant, min_valid=bound))
    assert len(seen) == 1


def _token_endpoint(request):
    return httpx.Response(200, json={"access_token": "account-access", "refresh_token": "account-refresh",
                                     "expires_in": 3600, "token_type": "Bearer", "scope": "openid email"})


def _begin(grant, **extra) -> dict:
    url = asyncio.run(oauth_engine.authorize(grant, ENDPOINTS, CLIENT, "openid email", extra,
                                             functools.partial(oauth_engine.state, "sign_in", "calendar", svc="google"),
                                             issuer=ISSUER))
    return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))


def test_extra_authorize_parameters_reach_the_provider_but_never_replace_the_engines(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    pending = _grant(tmp_path, "pending")
    asked = _begin(pending, login_hint="owner@example.com", access_type="offline")
    assert (asked["login_hint"], asked["access_type"], asked["scope"]) == ("owner@example.com", "offline", "openid email")
    assert oauth_engine.envelope(asked["state"], "sign_in")["svc"] == "google"
    stored = oauth_engine.load(pending)
    assert stored["pending"]["issuer"] == ISSUER and stored["skill"] == "calendar"  # the binding is stamped
    oauth_engine.save(pending, {})
    for owned in ("scope", "redirect_uri", "state"):
        with pytest.raises(ValueError):
            _begin(pending, **{owned: "elsewhere"})
    assert not oauth_engine.load(pending).get("pending")  # a refused start saved nothing


def test_an_identity_callback_files_the_sign_in_under_its_account(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    pending, account = _grant(tmp_path, "pending"), _grant(tmp_path, "a-0001")
    _provider(monkeypatch, _token_endpoint)
    calls = []

    async def identity(tokens, started):
        calls.append((tokens["access_token"], started["scope"]))
        return account, {"account": {"key": "a-0001", "subject": "sub-1", "label": "owner@example.com"}}

    asked = _begin(pending)
    saved = asyncio.run(oauth_engine.complete(pending, {"state": asked["state"], "code": "c1", "iss": ISSUER},
                                              identity=identity))
    filed = oauth_engine.load(account)
    assert calls == [("account-access", "openid email")]
    assert filed == {**saved, "v": 1, "skill": "calendar", "service": "google"} and filed["account"]["subject"] == "sub-1"
    assert filed["tokens"]["refresh_token"] == "account-refresh" and filed["sign_in"]["state"] == oauth_engine.SIGNED_IN
    assert not oauth_engine.load(pending).get("pending") and not oauth_engine.load(pending).get("tokens")

    async def refused(tokens, started):
        raise oauth_engine.SignInRequired("identity_unconfirmed")

    asked = _begin(pending)
    with pytest.raises(oauth_engine.SignInRequired):
        asyncio.run(oauth_engine.complete(pending, {"state": asked["state"], "code": "c2", "iss": ISSUER},
                                          identity=refused))
    assert oauth_engine.load(pending)["pending"] and oauth_engine.load(account) == filed  # nothing was saved

    asked = _begin(pending)  # without an identity the grant itself is signed in (as an MCP server's is)
    itself = asyncio.run(oauth_engine.complete(pending, {"state": asked["state"], "code": "c3", "iss": ISSUER}))
    assert itself == oauth_engine.load(pending) and itself["tokens"]["access_token"] == "account-access"
    assert "pending" not in itself and "account" not in itself


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


def test_consent_repair_keeps_provider_options_and_routes_a_service_return(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    endpoints = {**ENDPOINTS, "authorization_endpoint": ISSUER + "/authorize?view=one%20two&view=second&flag="}
    skill = "calendar_" + "x" * 55  # the longest canonical skill name (MAX_SKILL_NAME_LENGTH, 64)
    url = asyncio.run(oauth_engine.authorize(_grant(tmp_path, "pending"), endpoints, CLIENT, "openid", {"prompt": "consent"},
                                             functools.partial(oauth_engine.state, "sign_in", skill, svc="calendar"),
                                             issuer=ISSUER))
    raw_query = urllib.parse.urlsplit(url).query
    query = urllib.parse.parse_qs(raw_query, keep_blank_values=True)
    assert raw_query.startswith("view=one%20two&view=second&flag=&")
    assert query["scope"] == ["openid"] and query["prompt"] == ["consent"]
    fields = {"state": query["state"][0], "code": "test-code", "iss": ISSUER}
    assert oauth_engine.parse_return(CLIENT["redirect_uri"] + "?" + urllib.parse.urlencode(fields)) == fields
    recipient = oauth_engine.envelope(fields["state"], "sign_in")
    assert (recipient["v"], recipient["sign_in"], recipient["svc"]) == (1, skill, "calendar")
    # JSON version 1 is a number on both sides of the browser/Python boundary.
    numeric = base64.urlsafe_b64encode(json.dumps({**recipient, "v": 1.0}).encode()).decode()
    assert oauth_engine.envelope(numeric, "sign_in")["sign_in"] == skill


@pytest.mark.parametrize("since", ["2026-11-01T12:00:00Z", None], ids=["dated", "legacy"])
def test_consent_repair_keeps_an_mcp_requirement_until_the_provider_grants_it(monkeypatch, tmp_path, since):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    cfg = SimpleNamespace(id="docs", url=ISSUER + "/mcp", transport="streamable_http", carries_credentials=lambda: False)
    grant = mcp_oauth._grant(cfg)
    oauth_engine.save(grant, {"required_scope": "docs.read docs.write", "required_scope_since": since})

    async def run_step(scope):
        _provider(monkeypatch, lambda request: httpx.Response(200, json={
            "access_token": "mock-bearer", "token_type": "Bearer", "scope": scope}))
        url = await oauth_engine.authorize(grant, ENDPOINTS, CLIENT, "docs.read docs.write", {},
                                          functools.partial(oauth_engine.state, "mcp", cfg.id), issuer=ISSUER)
        state = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["state"][0]
        assert mcp_oauth.server_of(state) == cfg.id
        answer = await mcp_oauth.complete(cfg, state=state, code="test-code", iss=ISSUER)
        assert answer["scope"] == scope
        return mcp_oauth.status(cfg)

    partial = asyncio.run(run_step("docs.read"))
    assert partial["state"] == oauth_engine.SIGNED_IN
    assert partial["required_scope"] == "docs.read docs.write" and partial["required_scope_since"] == since
    done = asyncio.run(run_step("docs.read docs.write"))
    assert done["state"] == oauth_engine.SIGNED_IN and "required_scope" not in done
    assert "required_scope_since" not in oauth_engine.load(grant)


@pytest.mark.parametrize("legacy_since", [False, True], ids=["dated", "legacy"])
def test_consent_repair_keeps_each_service_accounts_own_requirement(monkeypatch, tmp_path, legacy_since):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    service = declaration.parse([{"id": "records", "label": "Records",
                                  "endpoints": {"authorization": ENDPOINTS["authorization_endpoint"],
                                                "token": ENDPOINTS["token_endpoint"], "identity": ISSUER + "/userinfo"},
                                  "scopes": ["openid", "records.read"],
                                  "optional_scopes": [{"scope": "records.write", "label": "Write"},
                                                      {"scope": "records.manage", "label": "Manage"}]}])[0]["records"]
    monkeypatch.setattr(service_sign_in, "_declared", lambda *_: service)
    skill = "work_records"
    first = service_sign_in.account_key(ISSUER, "personal")
    second = service_sign_in.account_key(ISSUER, "team")
    grant_a = service_sign_in._grant(skill, service.id, first)
    grant_b = service_sign_in._grant(skill, service.id, second)

    async def sign_in(account, subject, granted, optional=()):
        def provider(request):
            if request.url.path == "/userinfo":
                return httpx.Response(200, json={"sub": subject, "email": f"{subject}@example.test"})
            assert request.url.path == "/token"
            return httpx.Response(200, json={"access_token": f"mock-{subject}", "token_type": "Bearer", "scope": granted})

        _provider(monkeypatch, provider)
        base = "http://127.0.0.1:8765"
        begin = await service_sign_in.begin(skill, service.id, account=account, return_base=base, optional_scopes=optional,
                                            scope_words=["openid", "records.read", *optional],
                                            registration=oauth_engine.registration_ref(service_sign_in.clients()[0], base))
        state = urllib.parse.parse_qs(urllib.parse.urlsplit(begin["url"]).query)["state"][0]
        return begin, await service_sign_in.complete({"state": state, "code": "test-code"})

    async def calls():
        await service_sign_in.client_save(
            ISSUER, {"label": "Fixture", "client_id": "fixture-desktop"}, endpoints=dict(service.endpoints))
        _, personal = await sign_in("", "personal", "openid records.read")
        _, team = await sign_in("", "team", "openid records.read")
        assert (personal["state"], personal["account"]["key"]) == (oauth_engine.SIGNED_IN, first)
        assert (team["state"], team["account"]["key"]) == (oauth_engine.SIGNED_IN, second)
        first_since, second_since = "2026-01-06T12:00:00+00:00", "2026-01-07T12:00:00+00:00"
        for key, scope, since in ((first, "records.write", first_since), (second, "records.manage", second_since)):
            monkeypatch.setattr(oauth_engine, "utc_now_iso", lambda since=since: since)
            with pytest.raises(oauth_engine.SignInRequired) as caught:
                await service_sign_in.token(skill, service.id, key, required_scope=scope)
            assert caught.value.reason == "insufficient_scope"
        old_a, old_b = oauth_engine.load(grant_a), oauth_engine.load(grant_b)
        assert (old_a["required_scope_since"], old_b["required_scope_since"]) == (first_since, second_since)
        if legacy_since:
            oauth_engine.save(grant_b, {**old_b, "required_scope_since": None})
        monkeypatch.setattr(oauth_engine, "utc_now_iso", lambda: "2026-01-08T12:00:00+00:00")

        with pytest.raises(oauth_engine.SignInRequired) as caught:  # sent to repair one account, it names another saved one
            await sign_in(first, "team", "openid records.read", ("records.write",))
        assert caught.value.reason == "account_mismatch" and oauth_engine.load(grant_a) == old_a
        assert oauth_engine.load(grant_b)["required_scope_since"] == (None if legacy_since else second_since)
        begin, other = await sign_in(second, "team", "openid records.read", ("records.write",))  # chosen: its own row
        assert begin["scope"] == "openid records.read records.write" and other["account"]["key"] == second
        assert other["state"] == oauth_engine.SIGNED_IN and other["reason"] == "insufficient_scope"
        assert other["scope"] == "openid records.read" and other["missing_required"] == []
        _, guest = await sign_in(first, "guest", "openid records.read")  # another person: a new row of its own
        assert guest["account"]["key"] == service_sign_in.account_key(ISSUER, "guest")
        rows = {row["key"]: row for row in service_sign_in.accounts(skill, service)}
        assert rows[first]["scope"] == rows[second]["scope"] == ["openid", "records.read"]
        assert rows[second]["not_allowed"] == ["records.write"]
        assert oauth_engine.load(grant_a) == old_a
        actual_b = oauth_engine.load(grant_b)
        assert actual_b["required_scope"] == old_b["required_scope"] == "openid records.read records.manage"
        assert actual_b["required_scope_since"] == (None if legacy_since else second_since)

        _, done = await sign_in(second, "team", "openid records.read records.manage", ("records.manage",))
        assert done["account"]["key"] == second
        assert not {"required_scope", "required_scope_since"} & oauth_engine.load(grant_b).keys()
        assert oauth_engine.load(grant_a) == old_a
        assert (await service_sign_in.token(skill, service.id, second, required_scope="records.manage"))["scope"] == done["scope"]

    asyncio.run(calls())


def test_consent_repair_recovers_an_ordinary_filing_interruption_without_another_exchange(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    pending, account = _grant(tmp_path, "pending"), _grant(tmp_path, "a-0001")
    calls = _provider(monkeypatch, _token_endpoint)
    asked = _begin(pending)
    returned = {"state": asked["state"], "code": "test-code", "iss": ISSUER}
    save = oauth_engine.save

    async def identity(_tokens, _pending):
        return account, {"account": {"key": "a-0001", "label": "fixture@example.test"}}

    def interrupted(grant, doc):
        if grant == pending and "pending" not in doc:
            assert oauth_engine.load(account)["tokens"]["access_token"] == "account-access"
            raise OSError("synthetic storage interruption")
        return save(grant, doc)

    monkeypatch.setattr(oauth_engine, "save", interrupted)
    with pytest.raises(OSError, match="synthetic storage interruption"):
        asyncio.run(oauth_engine.complete(pending, returned, identity=identity))
    assert oauth_engine.load(pending)["pending"]
    assert "account-access" in oauth_engine.secret_values(pending.path)
    monkeypatch.setattr(oauth_engine, "save", save)
    tracked = oauth_engine.load(account)
    save(account, {**tracked, "tokens": {**tracked["tokens"], "access_token": "rotated-in-normal-use"}})
    saved = asyncio.run(oauth_engine.complete(pending, returned, identity=identity))
    assert saved["tokens"]["access_token"] == "rotated-in-normal-use" and len(calls) == 1
    assert "pending" not in oauth_engine.load(pending) and "account-access" not in oauth_engine.secret_values(pending.path)


@pytest.mark.parametrize("can_refresh", [False, True], ids=["stored", "refreshed"])
def test_consent_repair_keeps_short_tokens_for_requests_they_can_serve(monkeypatch, tmp_path, can_refresh):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(oauth_engine, "time", SimpleNamespace(time=lambda: 100.0, monotonic=time.monotonic))
    grant = _grant(tmp_path, "short")
    oauth_engine.save(grant, {"metadata": ENDPOINTS, "client": CLIENT, "tokens": {
        "access_token": "mock-old", "expires_at": 130.0, "scope": "openid records.read",
        "refresh_token": "mock-refresh" if can_refresh else None}})
    calls = _provider(monkeypatch, lambda request: httpx.Response(200, json={
        "access_token": "mock-new", "token_type": "Bearer", "expires_in": 30}))

    with pytest.raises(oauth_engine.SignInRequired) as caught:
        asyncio.run(oauth_engine.current_token(grant, min_valid=60))
    assert caught.value.reason == "min_valid_unmet"
    assert asyncio.run(oauth_engine.current_token(grant, min_valid=10)) == ("mock-new" if can_refresh else "mock-old")
    assert len(calls) == int(can_refresh)


def test_consent_repair_uses_actual_token_scope_after_refresh(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(oauth_engine, "time", SimpleNamespace(time=lambda: 100.0, monotonic=time.monotonic))
    grant = _grant(tmp_path, "narrower")
    oauth_engine.save(grant, {"metadata": ENDPOINTS, "client": CLIENT, "tokens": {
        "access_token": "mock-old", "expires_at": 90.0, "scope": "records.read", "refresh_token": "mock-refresh"}})
    calls = _provider(monkeypatch, lambda request: httpx.Response(200, json={
        "access_token": "mock-narrower", "token_type": "Bearer", "expires_in": 60, "scope": ""}))
    with pytest.raises(oauth_engine.SignInRequired) as caught:
        asyncio.run(oauth_engine.current_grant(grant, required_scope="records.read"))
    assert caught.value.reason == "insufficient_scope"
    doc = asyncio.run(oauth_engine.current_grant(grant))
    assert doc["tokens"]["scope"] == "" and doc["required_scope"] == "records.read"
    assert doc["required_scope_since"] and len(calls) == 1


def test_service_accepts_the_shown_scope_spellings_at_exchange_and_refresh(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    issuer, skill = "https://accounts.example.test", "calendar"
    alias = issuer + "/auth/userinfo.email"
    shown = ["openid", "email", alias]
    first_scope, renewed_scope = " ".join(shown), f"openid {alias}"
    service = declaration.parse([{"id": "fixture", "label": "Fixture", "issuer": issuer,
                                  "scopes": ["openid", "email"], "scope_spellings": {"email": [alias]}}])[0]["fixture"]
    monkeypatch.setattr(service_sign_in, "_declared", lambda *_: service)

    def provider(request):
        path = request.url.path
        if path == "/.well-known/openid-configuration":
            return httpx.Response(200, json={"issuer": issuer, "authorization_endpoint": issuer + "/authorize",
                "token_endpoint": issuer + "/token", "userinfo_endpoint": issuer + "/userinfo",
                "response_types_supported": ["code"], "code_challenge_methods_supported": ["S256"]})
        if path.startswith("/.well-known/"):
            return httpx.Response(404)
        if path == "/token":
            form = dict(urllib.parse.parse_qsl(request.content.decode()))
            refreshing = form["grant_type"] == "refresh_token"
            if refreshing:
                assert form["scope"] == first_scope
            return httpx.Response(200, json={"access_token": "fixture-new" if refreshing else "fixture-first",
                "refresh_token": "fixture-refresh", "token_type": "Bearer", "expires_in": 600,
                "scope": renewed_scope if refreshing else first_scope})
        assert path == "/userinfo"
        return httpx.Response(200, json={"sub": "fixture-id", "email": "person@example.test"})

    _provider(monkeypatch, provider)

    async def sign_in():
        client = await service_sign_in.client_save(issuer, {"label": "Example IT", "client_id": "fixture-it"})
        base = "http://localhost:8765"
        start = await service_sign_in.begin(skill, service.id, client=client["id"], scope_words=shown, return_base=base,
                                            registration=oauth_engine.registration_ref(service_sign_in.clients()[0], base))
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(start["url"]).query)
        assert query["scope"] == ["openid email"]
        pending = oauth_engine.load(service_sign_in._grant(skill, service.id, "pending"))["pending"]
        assert pending["attempt_id"] == start["attempt_id"] and pending["scope_words"] == shown
        done = await service_sign_in.complete({"state": query["state"][0], "code": "fixture-code", "iss": issuer},
                                              attempt_id=start["attempt_id"])
        assert done["state"] == oauth_engine.SIGNED_IN and not done["missing_required"]
        key = done["account"]["key"]
        first = await service_sign_in.token(skill, service.id, key, required_scope="email")
        later = replace(service, scope_spellings={"email": (alias, issuer + "/auth/email")})
        monkeypatch.setattr(service_sign_in, "_declared", lambda *_: later)
        renewed = await service_sign_in.token(skill, service.id, key, required_scope="email", rejected=first["access_token"])
        assert first["scope"] == done["scope"] == first_scope
        assert renewed["scope"] == renewed_scope and renewed["access_token"] == "fixture-new"
        filed = oauth_engine.load(service_sign_in._grant(skill, service.id, key))
        assert filed["scopes"]["requested"] == shown and filed["tokens"]["scope"] == renewed_scope
        assert filed["scope_spellings"] == {alias: "email"}

    asyncio.run(sign_in())
