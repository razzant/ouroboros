"""Remote MCP servers and their sign-in add the owner's Extra certificate bundle to httpx's own trust.

Throwaway authorities sign loopback HTTPS servers whose certificates name only ``127.0.0.1``: ``env`` stands for the
environment's trust (``SSL_CERT_FILE`` or an ``SSL_CERT_DIR``), ``extra`` for the owner's file. The real SDK
transports, the Settings Test route and the sign-in trust both while hostname checks stay on; an ``http`` address never
reads the file; skill sign-ins and providers keep their own trust. A file that cannot serve, or a certificate refused
before the code exchange, stops a sign-in step before anything reaches the provider, as a typed no-send the return page
and the tool can finish again from the same return; any uncertain exchange keeps its mark. A refresh the file cannot
serve ends an SSE call at once. Saving only the path relists the servers that read it, and nothing else.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import ssl
import time
import types

import httpx
import pytest

from ouroboros import config, mcp_client, mcp_oauth, net_transport, oauth_engine
from tests._mcp_loopback import _Provider, _Synthetic, start_asgi
from tests._throwaway_ca import file_in_capath, throwaway_ca, trusted_certificate

_PROXIES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


@pytest.fixture
def tls(tmp_path, monkeypatch):
    """Throwaway authorities (``tls.ca["env"]``, ``tls.ca["extra"]``, ``tls.ca["trusted"]``) and ``tls(app_for,
    signer)``: the app ``app_for(base)`` builds, over HTTPS with a ``127.0.0.1`` leaf of ``signer`` (plain HTTP for
    ``None``)."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    for name in ("OUROBOROS_EXTRA_CA_BUNDLE", "SSL_CERT_FILE", "SSL_CERT_DIR", *_PROXIES):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(mcp_client.MCPManager, "refresh_all_background", lambda *args, **kwargs: None)
    net_transport._merged_bundle_cache.clear()
    signers = {name: throwaway_ca(tmp_path / name, f"ouroboros {name} test ca") for name in ("env", "extra", "trusted")}
    running = []
    mcp_client.reset_manager_for_tests()

    def serve(app_for, signer="extra"):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        base = f"{'https' if signer else 'http'}://127.0.0.1:{port}"
        app = app_for(base)
        options = {} if signer is None else {"ssl_certfile": str(signers[signer][1]), "ssl_keyfile": str(signers[signer][2])}
        start_asgi(app[0] if isinstance(app, tuple) else app, running, f"mcp-tls-{signer}", port=port, **options)
        return app, base

    serve.ca = {name: files[0] for name, files in signers.items()}
    try:
        yield serve
    finally:
        mcp_client.reset_manager_for_tests()
        net_transport._merged_bundle_cache.clear()
        for server, thread in running:
            server.should_exit = True
            thread.join(10)


def _trust(monkeypatch, value) -> None:
    monkeypatch.setenv("OUROBOROS_EXTRA_CA_BUNDLE", str(value))


def _settings(*servers, timeout=20) -> dict:
    return {"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": timeout, "MCP_SERVERS": list(servers)}


def _settings_test(raw: dict, settings: dict) -> dict:
    """The Settings Test button: ``POST /api/mcp/test`` with an edited candidate (``gateway/mcp.py``)."""
    from unittest.mock import patch

    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway import mcp as gateway_mcp

    app = Starlette(routes=[Route("/api/mcp/test", endpoint=gateway_mcp.api_mcp_test, methods=["POST"])])
    with patch("ouroboros.gateway.mcp.load_settings", return_value=settings), TestClient(app) as client:
        return client.post("/api/mcp/test", json={"server": raw}).json()


def _probe(server_id: str, url: str, transport: str = "streamable_http") -> dict:
    raw = {"id": server_id, "enabled": True, "transport": transport, "url": url}
    return _settings_test(raw, _settings(raw))


def _refused_certificate(outcome: dict) -> bool:
    return outcome["ok"] is False and "CERTIFICATE_VERIFY_FAILED" in outcome["error"]


@pytest.mark.parametrize("transport", ["streamable_http", "sse"])
def test_test_listing_and_calls_trust_the_owner_ca_with_hostname_checks_on(tls, monkeypatch, transport):
    app, base = tls(lambda _base: _Synthetic("http" if transport == "streamable_http" else "sse", {}))
    url = base + ("/mcp" if transport == "streamable_http" else "/sse")
    raw = {"id": "bank", "enabled": True, "transport": transport, "url": url}

    assert _refused_certificate(_probe("bank", url, transport))  # negative control: certifi alone lacks it
    assert app.seen == []

    _trust(monkeypatch, tls.ca["extra"])
    tested = _probe("bank", url, transport)
    assert tested["ok"] is True and tested["tool_count"] == 1, tested
    manager = mcp_client.get_manager()
    manager.reconfigure(_settings(raw))
    assert manager.refresh_server("bank")["ok"] is True
    called = manager._call_tool_result("mcp_bank__echo", {"text": "hi"})
    assert called.status == "ok" and "echo:hi" in called.text, called.text

    # The same trusted authority, a name its certificate does not carry: hostname verification still refuses.
    mismatch = _probe("bank", url.replace("127.0.0.1", "localhost"), transport)
    assert _refused_certificate(mismatch) and "mismatch" in mismatch["error"].lower(), mismatch


@pytest.mark.parametrize("environment", ["SSL_CERT_FILE", "SSL_CERT_DIR"])
def test_the_owner_ca_joins_the_trust_the_environment_gives_httpx(tls, monkeypatch, tmp_path, environment):
    _env_app, env_base = tls(lambda _base: _Synthetic("http", {}), signer="env")
    _extra_app, extra_base = tls(lambda _base: _Synthetic("http", {}), signer="extra")
    env_url, extra_url = env_base + "/mcp", extra_base + "/mcp"
    value = tls.ca["env"] if environment == "SSL_CERT_FILE" else file_in_capath(tls.ca["env"], tmp_path / "capath").parent

    _trust(monkeypatch, tls.ca["extra"])  # without the environment's trust, httpx's base is certifi
    assert _probe("bank", extra_url)["ok"] is True and _refused_certificate(_probe("envsrv", env_url))

    monkeypatch.delenv("OUROBOROS_EXTRA_CA_BUNDLE")
    monkeypatch.setenv(environment, str(value))  # httpx's own default, unchanged without the setting
    assert _probe("envsrv", env_url)["ok"] is True and _refused_certificate(_probe("bank", extra_url))

    _trust(monkeypatch, tls.ca["extra"])  # both: the environment's authority keeps working beside the owner's
    assert _probe("envsrv", env_url)["ok"] is True and _probe("bank", extra_url)["ok"] is True
    servers = [{"id": name, "enabled": True, "url": url} for name, url in (("envsrv", env_url), ("bank", extra_url))]
    manager = mcp_client.get_manager()
    manager.reconfigure(_settings(*servers))
    for name in ("envsrv", "bank"):
        assert manager.refresh_server(name)["ok"] is True
        called = manager._call_tool_result(f"mcp_{name}__echo", {"text": "hi"})
        assert called.status == "ok" and "echo:hi" in called.text, called.text
    for url in (env_url, extra_url):
        mismatch = _probe("other", url.replace("127.0.0.1", "localhost"))
        assert _refused_certificate(mismatch) and "mismatch" in mismatch["error"].lower(), mismatch


@pytest.mark.parametrize("stated", ["trusted", "rejected"])
def test_a_mixed_bundle_keeps_each_root_as_openssl_reads_its_own_file(tls, monkeypatch, tmp_path, stated):
    """An ordinary root beside a ``TRUSTED CERTIFICATE`` the owner trusts for servers: both reach their servers, through
    MCP's context and the providers' merged bundle alike; a root the file rejects for servers stays refused."""
    _ordinary_app, ordinary = tls(lambda _base: _Synthetic("http", {}), signer="extra")
    _trusted_app, trusted = tls(lambda _base: _Synthetic("http", {}), signer="trusted")
    owner = tmp_path / "owner.pem"
    owner.write_bytes(b"\xef\xbb\xbf" + tls.ca["extra"].read_bytes() + "# корень брокера\n".encode()
                      + trusted_certificate(tls.ca["trusted"], reject=stated == "rejected"))
    # Paired control: Python's ``cadata`` reader, as the earlier loader fed it, keeps the ordinary root alone.
    as_text = re.sub(rb"[^\x00-\x7f]", b"?", owner.read_bytes()[3:]).decode("ascii")
    loaded = {}
    for how, source in (("cadata", {"cadata": as_text}), ("cafile", {"cafile": str(owner)})):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_verify_locations(**source)
        loaded[how] = context.cert_store_stats()["x509_ca"]
    assert loaded == {"cadata": 1, "cafile": 2}

    _trust(monkeypatch, owner)
    assert _probe("bank", ordinary + "/mcp")["ok"] is True
    broker = _probe("broker", trusted + "/mcp")
    if stated == "rejected":
        assert _refused_certificate(broker), broker
    else:
        assert broker["ok"] is True, broker
        mismatch = _probe("broker", trusted.replace("127.0.0.1", "localhost") + "/mcp")
        assert _refused_certificate(mismatch) and "mismatch" in mismatch["error"].lower(), mismatch
    context = net_transport.extra_ca_context()
    assert (context.verify_mode, context.check_hostname) == (ssl.CERT_REQUIRED, True)
    merged = net_transport.trust_ssl_context()
    assert (merged.verify_mode, merged.check_hostname) == (ssl.CERT_REQUIRED, True)
    for base, reaches in ((ordinary, True), (trusted, stated == "trusted")):
        with httpx.Client(verify=merged, trust_env=False, timeout=10) as client:
            if reaches:
                client.get(base + "/mcp")  # any answer: the handshake verified
            else:
                with pytest.raises(httpx.ConnectError, match="CERTIFICATE_VERIFY_FAILED"):
                    client.get(base + "/mcp")
    assert owner.read_bytes().count(b"TRUSTED CERTIFICATE") == 2  # the owner's file is read, never rewritten


def test_the_sign_in_client_adds_the_owner_ca_while_skill_sign_ins_and_providers_keep_their_trust(tls, monkeypatch):
    _env_app, env_base = tls(lambda _base: _Synthetic("http", {}), signer="env")
    _extra_app, extra_base = tls(lambda _base: _Synthetic("http", {}), signer="extra")
    monkeypatch.setenv("SSL_CERT_FILE", str(tls.ca["env"]))
    _trust(monkeypatch, tls.ca["extra"])

    async def reach(client: httpx.AsyncClient) -> list:
        async with client as http:
            answers = []
            for base in (env_base, extra_base):
                try:
                    answers.append((await http.get(base + "/mcp")).status_code)  # any answer: the TLS session held
                except httpx.ConnectError as exc:
                    answers.append("refused" if oauth_engine.certificate_refused(exc) else repr(exc))
            return answers

    mcp_side = asyncio.run(reach(oauth_engine.http_client(extra_ca=True)))
    assert all(isinstance(answer, int) for answer in mcp_side), mcp_side
    skill_side = asyncio.run(reach(oauth_engine.http_client()))  # skill sign-ins: httpx's own trust only
    assert isinstance(skill_side[0], int) and skill_side[1] == "refused", skill_side
    providers = asyncio.run(reach(httpx.AsyncClient(verify=net_transport.trust_ssl_context())))  # certifi + the file
    assert providers[0] == "refused" and isinstance(providers[1], int), providers

    cfg = types.SimpleNamespace(id="bank", transport="streamable_http", url=extra_base + "/mcp")
    plain = types.SimpleNamespace(id="local", transport="streamable_http", url="http://127.0.0.1:9/mcp")
    assert mcp_oauth._grant(cfg).extra_ca is True and mcp_oauth._grant(plain).extra_ca is False
    from ouroboros import service_sign_in

    assert service_sign_in._clients_file().extra_ca is False


def test_a_plain_http_address_never_reads_the_bundle(tls, monkeypatch, tmp_path):
    app, base = tls(lambda _base: _Synthetic("http", {}), signer=None)
    broken = tmp_path / "not-a-certificate.pem"
    broken.write_bytes(b"not a certificate\n")
    _trust(monkeypatch, broken)
    raw = {"id": "local", "enabled": True, "url": base + "/mcp"}
    assert _settings_test(raw, _settings(raw))["ok"] is True
    manager = mcp_client.get_manager()
    manager.reconfigure(_settings(raw))
    assert manager.refresh_server("local")["ok"] is True
    called = manager._call_tool_result("mcp_local__echo", {"text": "hi"})
    assert called.status == "ok" and "echo:hi" in called.text, called.text


def test_an_unusable_bundle_refuses_before_any_request(tls, monkeypatch, tmp_path):
    app, base = tls(lambda _base: _Synthetic("http", {}))
    raw = {"id": "bank", "enabled": True, "transport": "streamable_http", "url": base + "/mcp"}
    _trust(monkeypatch, tls.ca["extra"])
    manager = mcp_client.get_manager()
    manager.reconfigure(_settings(raw))
    assert manager.refresh_server("bank")["ok"] is True  # listed while the file was good
    served = len(app.seen)

    broken = tmp_path / "not-a-certificate.pem"
    broken.write_bytes(b"not a certificate\n")
    _trust(monkeypatch, broken)
    tested = _settings_test(raw, _settings(raw))
    assert tested["ok"] is False and "ExtraCaBundleError" in tested["error"], tested
    assert "OUROBOROS_EXTRA_CA_BUNDLE" in tested["error"] and str(broken) in tested["error"]
    called = manager._call_tool_result("mcp_bank__echo", {"text": "hi"})
    assert called.status != "ok" and "ExtraCaBundleError" in called.text, called.text
    assert called.meta.get("mcp_call_not_sent") is True, called.meta
    assert len(app.seen) == served  # nothing reached the server


def test_a_file_replaced_under_its_path_serves_the_next_connection(tls, monkeypatch, tmp_path):
    _app, base = tls(lambda _base: _Synthetic("http", {}))
    owner = tmp_path / "owner.pem"
    owner.write_bytes(tls.ca["env"].read_bytes())  # readable, but another authority
    _trust(monkeypatch, owner)
    assert _refused_certificate(_probe("bank", base + "/mcp"))
    staged = tmp_path / "staged.pem"
    staged.write_bytes(tls.ca["extra"].read_bytes())
    os.replace(staged, owner)
    assert _probe("bank", base + "/mcp")["ok"] is True


class _Recorded:
    """An ASGI app that records each HTTP request's method and path before passing it on."""

    def __init__(self, app):
        self.app, self.seen = app, []

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            self.seen.append((scope["method"], scope["path"]))
        await self.app(scope, receive, send)


def _authorizing_app(base):
    from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
    from mcp.server.fastmcp import FastMCP

    provider = _Provider(base)
    server = FastMCP("bank", host="127.0.0.1", log_level="WARNING", auth_server_provider=provider,
                     auth=AuthSettings(issuer_url=base, resource_server_url=base + "/mcp", required_scopes=["mcp"],
                                       client_registration_options=ClientRegistrationOptions(
                                           enabled=True, valid_scopes=["mcp"], default_scopes=["mcp"]),
                                       revocation_options=RevocationOptions(enabled=True)))

    @server.tool()
    def echo(text: str) -> str:
        return f"echo:{text}"

    return _Recorded(server.streamable_http_app()), provider


def _owners_browser(started: dict, ca_pem) -> str:
    """The owner's browser at the provider: the return address it is sent back to."""
    answer = httpx.get(started["url"], follow_redirects=False, timeout=10, verify=ssl.create_default_context(cafile=str(ca_pem)))
    return answer.headers["location"]


def test_sign_in_refresh_and_revocation_reach_the_authorization_server_through_the_bundle(tls, monkeypatch):
    (app, provider), base = tls(_authorizing_app)
    manager = mcp_client.get_manager()
    manager.reconfigure(_settings({"id": "bank", "enabled": True, "url": base + "/mcp"}))
    cfg = manager.server_config("bank")

    with pytest.raises(httpx.ConnectError, match="CERTIFICATE_VERIFY_FAILED"):  # negative control: discovery
        asyncio.run(mcp_oauth.begin(cfg))
    assert provider.counts["register"] == 0

    _trust(monkeypatch, tls.ca["extra"])
    listed = manager.refresh_server("bank")  # TLS now succeeds: the server's own 401 asks for sign-in
    assert listed["ok"] is False and listed["code"] == "MCP_AUTH_REQUIRED", listed
    started = asyncio.run(mcp_oauth.begin(cfg))  # resource and issuer metadata, dynamic registration
    assert provider.counts["register"] == 1
    returned = mcp_oauth.parse_return(_owners_browser(started, tls.ca["extra"]))
    assert asyncio.run(mcp_oauth.complete(cfg, **returned))["ok"] is True
    assert provider.codes == {} and [seen for seen in app.seen if seen[1] == "/token"] == [("POST", "/token")]
    assert manager.refresh_server("bank")["ok"] is True
    called = manager._call_tool_result("mcp_bank__echo", {"text": "hi"})
    assert called.status == "ok" and "echo:hi" in called.text, called.text

    answers = []  # every token-endpoint and revocation POST: (grant or token kind, HTTP status, body)
    real_token_request = oauth_engine.token_request

    async def recorded(http, endpoint, client, data):
        response = await real_token_request(http, endpoint, client, data)
        answers.append((data.get("grant_type") or data.get("token_type_hint"), response.status_code, response.text))
        return response

    monkeypatch.setattr(oauth_engine, "token_request", recorded)
    path = mcp_oauth._path(cfg.id)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["tokens"]["expires_at"] = time.time() - 1
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert asyncio.run(mcp_oauth.current_token(cfg)) != doc["tokens"]["access_token"]
    assert provider.counts["refresh_ok"] == 1 and answers[-1][:2] == ("refresh_token", 200)

    del answers[:]
    out = asyncio.run(mcp_oauth.sign_out(cfg))
    # Both revocation requests are answered over the verified connection (a TLS refusal answers nothing). The
    # SDK's loopback endpoint then requires a secret this public, self-registered client does not have.
    assert out["revoked"] is False and [(kind, status) for kind, status, _text in answers] == [
        ("refresh_token", 400), ("access_token", 400)], (out, answers)
    assert all("client_secret" in text for _kind, _status, text in answers)


def test_a_return_stopped_before_the_exchange_is_typed_and_finished_again_from_the_same_return(tls, monkeypatch, tmp_path):
    from unittest.mock import patch

    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from ouroboros.gateway import mcp as gateway_mcp
    from ouroboros.tools.mcp_sign_in import _mcp_sign_in
    from ouroboros.tools.tool_result import _TOOL_RESULT_ATTR

    (app, provider), base = tls(_authorizing_app)
    settings = _settings({"id": "bank", "enabled": True, "url": base + "/mcp"})
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    manager = mcp_client.get_manager()
    manager.reconfigure(settings)
    cfg = manager.server_config("bank")
    _trust(monkeypatch, tls.ca["extra"])
    location = _owners_browser(asyncio.run(mcp_oauth.begin(cfg)), tls.ca["extra"])
    pending = lambda: json.loads(mcp_oauth._path("bank").read_text(encoding="utf-8"))["pending"]  # noqa: E731
    token_posts = lambda: [seen for seen in app.seen if seen[1] == "/token"]  # noqa: E731
    broken = tmp_path / "broken.pem"
    broken.write_bytes(b"not a certificate\n")

    with patch("ouroboros.gateway.mcp.load_settings", return_value=settings), TestClient(
            Starlette(routes=gateway_mcp.mcp_routes())) as client:
        def finish():
            return client.post("/api/mcp/sign-in", json={"action": "complete", "returned_url": location})

        _trust(monkeypatch, broken)  # an unusable file: refused before the client exists
        stopped = finish()
        assert stopped.status_code == 200 and stopped.json()["reason"] == "not_sent", stopped.json()
        assert "Extra certificate bundle" in stopped.json()["error"] and str(broken) not in stopped.json()["error"]
        assert str(broken) in stopped.json()["detail"]  # the path stays beside the words, for this computer only
        ctx = types.SimpleNamespace(event_queue=None, **{_TOOL_RESULT_ATTR: None})
        text = _mcp_sign_in(ctx, action="complete", returned_url=location)
        typed = getattr(ctx, _TOOL_RESULT_ATTR)
        assert text.startswith("⚠️ MCP_UNAVAILABLE: Sign-in stopped before anything changed at the provider"), text
        assert f"Detail: OUROBOROS_EXTRA_CA_BUNDLE holds no loadable PEM certificate: {broken}" in text, text
        assert "new tasks; this task keeps the settings it started with" in text, text  # where a saved fix applies
        assert typed.meta["operation_outcome"] == "completed_no_effect" and typed.meta["sign_in_reason"] == "not_sent"

        _trust(monkeypatch, tls.ca["env"])  # a readable file of another authority: the token endpoint is refused
        refused = finish()
        assert refused.status_code == 200 and refused.json()["reason"] == "not_sent", refused.json()
        assert "an authority this computer doesn’t trust" in refused.json()["error"], refused.json()
        assert "Extra certificate bundle" in refused.json()["error"] and "127.0.0.1" not in refused.json()["error"]
        assert "exchange" not in pending() and token_posts() == [] and len(provider.codes) == 1

        _trust(monkeypatch, tls.ca["extra"])
        done = finish()
        assert done.status_code == 200 and done.json()["ok"] is True and done.json()["listing"]["ok"] is True, done.json()
    assert token_posts() == [("POST", "/token")] and provider.codes == {}  # the one code went out once


@pytest.mark.parametrize("cause", ["bundle", "issuer"])
@pytest.mark.parametrize("stale", [True, False])
def test_a_refresh_that_never_left_ends_an_sse_call_at_once_and_is_unsent_only_when_proven(tls, monkeypatch, stale,
                                                                                           cause):
    """``bundle``: the file cannot serve the refresh; ``issuer``: the token endpoint's certificate is refused (the
    MCP server itself is trusted). Either way at both refresh entries (a stale token, a Bearer challenge)."""
    armed, posts = [], []

    class ChallengeTheCall:  # the call's POST (after initialize and initialized) meets the server's Bearer challenge
        def __init__(self, inner):
            self.inner = inner

        async def __call__(self, scope, receive, send):
            if armed and scope["type"] == "http" and scope["method"] == "POST":
                posts.append(scope["path"])
                if len(posts) == 3:
                    await send({"type": "http.response.start", "status": 401,
                                "headers": [(b"www-authenticate", b'Bearer error="invalid_token"')]})
                    await send({"type": "http.response.body", "body": b"revoked"})
                    return
            await self.inner(scope, receive, send)

    inner = _Synthetic("sse", {})
    _app, base = tls(lambda _base: ChallengeTheCall(inner))
    _trust(monkeypatch, tls.ca["extra"])
    manager = mcp_client.get_manager()
    manager.reconfigure(_settings({"id": "bank", "enabled": True, "transport": "sse", "url": base + "/sse"}, timeout=10))
    cfg = manager.server_config("bank")
    tokens = {"access_token": "synthetic-access", "refresh_token": "synthetic-refresh", "expires_at": time.time() + 3600}
    _issuer, issuer_base = tls(lambda _base: _Synthetic("http", {}), signer="env")  # an authority nobody here trusts
    token_endpoint = (base if cause == "bundle" else issuer_base) + "/token"
    grant = {"tokens": tokens, "metadata": {"token_endpoint": token_endpoint}, "client": {"client_id": "synthetic"}}
    mcp_oauth._save(cfg, grant)
    assert manager.refresh_server("bank")["ok"] is True

    real, reads = net_transport.extra_ca_context, []

    def unusable_after_the_connection_read_it():
        reads.append(1)
        if len(reads) > 1:
            raise net_transport.ExtraCaBundleError("OUROBOROS_EXTRA_CA_BUNDLE is not readable: synthetic")
        return real()

    if cause == "bundle":
        monkeypatch.setattr(net_transport, "extra_ca_context", unusable_after_the_connection_read_it)
    if stale:
        mcp_oauth._save(cfg, {**grant, "tokens": {**tokens, "expires_at": time.time() - 1}})
    served, armed[:] = len(inner.seen), [True]
    began = time.monotonic()
    called = manager._call_tool_result("mcp_bank__echo", {"text": "hi"})
    assert time.monotonic() - began < 5, "the session ends at once, not at its 10 s tool timeout"
    named = "ExtraCaBundleError" if cause == "bundle" else "CERTIFICATE_VERIFY_FAILED"
    assert called.status != "ok" and named in called.text, called.text
    # Stale: the first request refreshes, so nothing left. Otherwise initialize got past sign-in: never claimed unsent.
    assert called.meta.get("mcp_call_not_sent") is stale, called.meta
    assert (len(inner.seen) == served) is stale and len(reads) == (2 if cause == "bundle" else 0)
    assert _issuer.seen == []  # the refused handshake carried no request


def test_saving_only_the_bundle_path_relists_the_servers_that_read_it(monkeypatch, tmp_path):
    """A disabled row that reads the path is no reason to list anything; an enabled one that does gets a full pass."""
    from ouroboros.mcp_discovery import settings_fingerprint

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    listed = []

    async def list_tools(cfg, _timeout):  # the manager's test hook: no transport, no process
        listed.append(cfg.id)
        return [{"name": "echo", "description": "", "input_schema": {"type": "object"}}]

    servers = [{"id": "bank", "enabled": True, "url": "https://127.0.0.1:9/mcp"},  # HTTP by default, over TLS
               {"id": "feed", "enabled": True, "transport": "sse", "url": "https://127.0.0.1:9/sse"},
               {"id": "plain", "enabled": True, "url": "http://127.0.0.1:9/mcp"},
               {"id": "off", "enabled": False, "url": "https://127.0.0.1:9/mcp"},
               {"id": "local", "enabled": True, "transport": "stdio", "command": "synthetic-never-started"},
               {"id": "reader", "enabled": False, "transport": "stdio", "command": "synthetic-never-started",
                "env_from_settings": {"SSL_EXTRA": "OUROBOROS_EXTRA_CA_BUNDLE"}}]
    before = {**_settings(*servers), "OUROBOROS_EXTRA_CA_BUNDLE": ""}
    saved = {**before, "OUROBOROS_EXTRA_CA_BUNDLE": " /owner/extra.pem "}
    # The Settings hook's comparison sees the path; the connections it decides do not change with it.
    assert settings_fingerprint(saved) != settings_fingerprint(before)
    assert settings_fingerprint(saved, values=False) == settings_fingerprint(before, values=False)
    # No extra missing-skill pass is owed: the disabled row is not a consumer of the raw setting.
    assert settings_fingerprint(saved, path="") == settings_fingerprint(before)
    unread = {**before, "MCP_SERVERS": servers[:-1]}  # the TLS path is read normalized; a process gets the literal
    assert settings_fingerprint({**unread, "OUROBOROS_EXTRA_CA_BUNDLE": None}) == settings_fingerprint(unread)
    assert settings_fingerprint({**unread, "OUROBOROS_EXTRA_CA_BUNDLE": " "}) == settings_fingerprint(unread)
    mcp_client.reset_manager_for_tests()
    try:
        manager = mcp_client.get_manager()
        manager._async_list_tools = list_tools
        manager.configuration_changed(before, reason="startup")
        assert manager.discovery.wait(20) and sorted(listed) == ["bank", "feed", "local", "plain"]
        del listed[:]
        manager.configuration_changed(saved, reason="settings")
        assert manager.discovery.wait(20) and listed == ["bank", "feed"]  # no stdio start, no plain HTTP, none off
        del listed[:]
        manager.configuration_changed(saved, reason="settings")
        assert manager.discovery.wait(20) and listed == []
        edited = {**saved, "MCP_SERVERS": [*servers[:-2], {**servers[-2], "args": ["--synthetic"]}, servers[-1]]}
        manager.configuration_changed(edited, reason="settings")  # a server edit keeps the full pass
        assert manager.discovery.wait(20) and sorted(listed) == ["bank", "feed", "local", "plain"]
        del listed[:]
        reading = {**edited, "MCP_SERVERS": [*edited["MCP_SERVERS"][:-1], {**servers[-1], "enabled": True}]}
        manager.configuration_changed(reading, reason="settings")
        assert manager.discovery.wait(20) and sorted(listed) == ["bank", "feed", "local", "plain", "reader"]
        del listed[:]
        manager.configuration_changed({**reading, "OUROBOROS_EXTRA_CA_BUNDLE": "/owner/other.pem"}, reason="settings")
        assert manager.discovery.wait(20) and sorted(listed) == ["bank", "feed", "local", "plain", "reader"]
    finally:
        mcp_client.reset_manager_for_tests()


def test_the_sdk_factory_runs_unchanged_without_a_bundle_and_its_construction_is_kept_with_one(monkeypatch, tmp_path):
    from mcp.shared import _httpx_utils as sdk

    from ouroboros.mcp_headers import _CredentialOriginGuard

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.delenv("OUROBOROS_EXTRA_CA_BUNDLE", raising=False)
    calls = []
    factory = sdk.create_mcp_http_client
    monkeypatch.setattr(sdk, "create_mcp_http_client", lambda **kwargs: calls.append(kwargs) or factory(**kwargs))
    cfg = types.SimpleNamespace(carries_credentials=lambda: False, url="https://127.0.0.1/mcp", id="x",
                                transport="streamable_http")
    shapes = [{}, {"headers": {"X-Synthetic": "1"}, "timeout": httpx.Timeout(5.0),
                   "auth": httpx.BasicAuth("synthetic", "synthetic")}]
    for shape in shapes:
        asyncio.run(_CredentialOriginGuard(cfg).client(**shape).aclose())
    assert calls == [{"headers": None, "timeout": None, "auth": None},
                     {"headers": shapes[1]["headers"], "timeout": shapes[1]["timeout"], "auth": shapes[1]["auth"]}]

    ca_pem, _cert, _key = throwaway_ca(tmp_path / "ca")
    monkeypatch.setenv("OUROBOROS_EXTRA_CA_BUNDLE", str(ca_pem))
    guard = _CredentialOriginGuard(cfg)
    assert guard.trust.verify_mode == ssl.CERT_REQUIRED and guard.trust.check_hostname is True
    assert _CredentialOriginGuard(types.SimpleNamespace(**{**vars(cfg), "url": "http://127.0.0.1/mcp"})).trust is None
    for shape in shapes:
        ours, theirs = guard.client(**shape), factory(**shape)
        for name in ("timeout", "follow_redirects", "trust_env", "max_redirects", "auth", "base_url"):
            assert repr(getattr(ours, name)) == repr(getattr(theirs, name)), name
        assert dict(ours.headers) == dict(theirs.headers) and ours.event_hooks == theirs.event_hooks
        assert ours._transport._pool._ssl_context is guard.trust
        asyncio.run(ours.aclose())
        asyncio.run(theirs.aclose())
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")  # never contacted: the client is only built
    proxied = guard.client()  # the environment's proxy stays mounted, and the server is verified through it
    tunnels = [mounted for pattern, mounted in proxied._mounts.items() if mounted is not None and pattern.scheme == "https"]
    assert tunnels and all(mounted._pool._ssl_context is guard.trust for mounted in tunnels), proxied._mounts
    asyncio.run(proxied.aclose())


def test_an_unusable_bundle_names_the_setting_at_sign_in_and_sign_out_still_ends_locally(monkeypatch, tmp_path):
    from unittest.mock import patch

    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from ouroboros.gateway import mcp as gateway_mcp

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(mcp_client.MCPManager, "refresh_all_background", lambda *args, **kwargs: None)
    broken = tmp_path / "broken.pem"
    broken.write_bytes(b"not a certificate\n")
    _trust(monkeypatch, broken)
    url = "https://127.0.0.1:9/mcp"  # never contacted: every step below refuses before sending
    settings = _settings({"id": "bank", "enabled": True, "url": url})
    sent = []

    async def no_post(*args, **kwargs):
        sent.append(args)
        raise AssertionError("no token or revocation request may leave")

    monkeypatch.setattr(oauth_engine, "token_request", no_post)
    mcp_client.reset_manager_for_tests()
    try:
        app = Starlette(routes=gateway_mcp.mcp_routes())
        with patch("ouroboros.gateway.mcp.load_settings", return_value=settings), TestClient(app) as client:
            began = client.post("/api/mcp/sign-in", json={"action": "begin", "server_id": "bank"})
            cfg = mcp_client.get_manager().server_config("bank")
            mcp_oauth._save(cfg, {"url": url, "metadata": {"revocation_endpoint": "https://127.0.0.1:9/revoke"},
                                  "client": {"client_id": "synthetic"},
                                  "tokens": {"access_token": "synthetic-access", "refresh_token": "synthetic-refresh"}})
            out = client.post("/api/mcp/sign-in", json={"action": "sign_out", "server_id": "bank"}).json()
    finally:
        mcp_client.reset_manager_for_tests()
    assert began.status_code == 200 and began.json()["reason"] == "not_sent", began.json()
    assert "Extra certificate bundle" in began.json()["error"] and str(broken) not in began.json()["error"]
    assert str(broken) in began.json()["detail"]
    assert out == {"ok": True, "server": "bank", "revoked": False}, out
    stored = json.loads(mcp_oauth._path("bank").read_text(encoding="utf-8"))
    assert "tokens" not in stored and stored["sign_in"]["reason"] == "signed_out"
    assert sent == []


def test_a_certificate_refused_at_begin_is_a_typed_no_change_whose_words_name_only_its_own_cure(tls, monkeypatch):
    from unittest.mock import patch

    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from ouroboros.gateway import mcp as gateway_mcp
    from ouroboros.tools.mcp_sign_in import _mcp_sign_in
    from ouroboros.tools.tool_result import _TOOL_RESULT_ATTR

    (app, provider), base = tls(_authorizing_app)
    settings = _settings({"id": "bank", "enabled": True, "url": base + "/mcp"},
                         {"id": "byname", "enabled": True, "url": base.replace("127.0.0.1", "localhost") + "/mcp"})
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    mcp_client.get_manager().reconfigure(settings)
    with patch("ouroboros.gateway.mcp.load_settings", return_value=settings), TestClient(
            Starlette(routes=gateway_mcp.mcp_routes())) as client:
        def begin(server_id):
            return client.post("/api/mcp/sign-in", json={"action": "begin", "server_id": server_id})

        untrusted = begin("bank")  # certifi alone: the first discovery request meets an unknown authority
        assert untrusted.status_code == 200 and untrusted.json()["reason"] == "not_sent", untrusted.json()
        words = untrusted.json()["error"]
        assert "an authority this computer doesn’t trust" in words and "Extra certificate bundle" in words, words
        assert "CERTIFICATE_VERIFY_FAILED" not in words and "127.0.0.1" not in words  # fixed words; the rest is detail
        assert untrusted.json()["detail"].startswith("connecting to 127.0.0.1: [SSL: CERTIFICATE_VERIFY_FAILED]")
        ctx = types.SimpleNamespace(event_queue=None, **{_TOOL_RESULT_ATTR: None})
        text = _mcp_sign_in(ctx, action="begin", server="bank")
        typed = getattr(ctx, _TOOL_RESULT_ATTR)
        assert typed.meta["operation_outcome"] == "completed_no_effect" and typed.meta["sign_in_reason"] == "not_sent"
        assert "Detail:" in text and "CERTIFICATE_VERIFY_FAILED" in text, text

        _trust(monkeypatch, tls.ca["extra"])  # a trusted authority, a name its certificate does not carry
        mismatch = begin("byname")
        assert mismatch.status_code == 200 and mismatch.json()["reason"] == "not_sent", mismatch.json()
        assert "didn’t pass its checks" in mismatch.json()["error"] and "Hostname mismatch" in mismatch.json()["detail"]
        assert "Extra certificate bundle" not in mismatch.json()["error"]  # a bundle cures no wrong name
    assert app.seen == [] and provider.counts["register"] == 0
    assert not mcp_oauth._path("bank").exists() and not mcp_oauth._path("byname").exists()  # nothing pending


def test_a_plain_http_row_whose_issuer_is_https_keeps_default_trust_and_is_never_told_to_add_the_bundle(tls, monkeypatch):
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    (issuer_app, provider), issuer = tls(_authorizing_app)  # HTTPS, signed by the owner's authority

    def plain(base):
        async def resource(_request):
            return JSONResponse({"resource": base + "/mcp", "authorization_servers": [issuer]})

        return Starlette(routes=[Route(path, resource) for path in (
            "/.well-known/oauth-protected-resource/mcp", "/.well-known/oauth-protected-resource")])

    _app, base = tls(plain, signer=None)
    _trust(monkeypatch, tls.ca["extra"])  # the file would trust the issuer, but this row's sign-in never reads it
    manager = mcp_client.get_manager()
    manager.reconfigure(_settings({"id": "docs", "enabled": True, "url": base + "/mcp"}))
    with pytest.raises(oauth_engine.CertificateRefused) as refused:
        asyncio.run(mcp_oauth.begin(manager.server_config("docs")))
    assert refused.value.bundle is False and issuer_app.seen == [] and provider.counts["register"] == 0
    words, detail = mcp_oauth.not_sent(refused.value)
    assert "address is http" in words and "add its certificate" not in words and "127.0.0.1" not in words, words
    assert detail.startswith("connecting to 127.0.0.1: [SSL: CERTIFICATE_VERIFY_FAILED]"), detail


@pytest.mark.parametrize("proxy", ["", "http://proxy.synthetic.test:3128", "https://proxy.synthetic.test:8443"])
def test_certificate_words_offer_the_bundle_only_where_it_can_cure_and_never_name_who_presented_it(monkeypatch, proxy):
    """Default TLS at the HTTPS proxy and additive trust at the origin are separate, whichever route is used."""
    import urllib.request

    monkeypatch.setattr(urllib.request, "getproxies", lambda: {"https": proxy} if proxy else {})
    request = httpx.Request("POST", "https://issuer.synthetic.test/token")

    def refused(code, *, bundle):
        failure = ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
        failure.verify_code, failure.verify_message = code, "synthetic"
        return oauth_engine.CertificateRefused(httpx.ConnectError("x", request=request), failure, bundle=bundle)

    for code, bundle in ((20, True), (20, False), (62, True)):  # unknown issuer (with and without the file), wrong name
        words, detail = mcp_oauth.not_sent(refused(code, bundle=bundle))
        assert "issuer.synthetic.test" not in words and detail.startswith("connecting to issuer.synthetic.test: [SSL:")
        offered = "add its certificate under Extra certificate bundle" in words
        assert offered is (code == 20 and bundle), words
        assert ("address is http" in words) is (code == 20 and not bundle), words
        assert ("HTTPS proxy" in words) is (code == 20 and bundle), words


def test_a_mistyped_address_never_stops_the_other_rows_or_the_status(monkeypatch, tmp_path):
    from unittest.mock import patch

    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from ouroboros.gateway import mcp as gateway_mcp

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.delenv("OUROBOROS_EXTRA_CA_BUNDLE", raising=False)
    listed = []

    async def list_tools(cfg, _timeout):
        listed.append(cfg.id)
        return [{"name": "echo", "description": "", "input_schema": {"type": "object"}}]

    settings = _settings({"id": "typo", "enabled": True, "url": "https://127.0.0.999/mcp"},  # admitted, never valid
                         {"id": "bank", "enabled": True, "url": "https://127.0.0.1:9/mcp"})
    mcp_client.reset_manager_for_tests()
    try:
        manager = mcp_client.get_manager()
        manager._async_list_tools = list_tools
        manager.configuration_changed(settings, reason="startup")
        assert manager.discovery.wait(20) and sorted(listed) == ["bank", "typo"]
        with patch("ouroboros.gateway.mcp.load_settings", return_value=settings), TestClient(
                Starlette(routes=gateway_mcp.mcp_routes())) as client:
            status = client.get("/api/mcp/status")
        assert status.status_code == 200 and {row["id"] for row in status.json()["servers"]} >= {"typo", "bank"}
    finally:
        mcp_client.reset_manager_for_tests()


@pytest.mark.parametrize("kind", ["der", "home"])
def test_a_bundle_that_is_not_a_readable_pem_stops_sign_in_typed_not_as_a_crash(monkeypatch, tmp_path, kind):
    from unittest.mock import patch

    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from ouroboros.gateway import mcp as gateway_mcp

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(mcp_client.MCPManager, "refresh_all_background", lambda *args, **kwargs: None)
    ca_pem, _cert, _key = throwaway_ca(tmp_path / "ca")
    der = tmp_path / "root.cer"
    der.write_bytes(ssl.PEM_cert_to_DER_cert(ca_pem.read_text()))
    _trust(monkeypatch, der if kind == "der" else "~synthetic-nobody-e22094a0/root.pem")
    settings = _settings({"id": "bank", "enabled": True, "url": "https://127.0.0.1:9/mcp"})  # never contacted
    mcp_client.reset_manager_for_tests()
    try:
        with patch("ouroboros.gateway.mcp.load_settings", return_value=settings), TestClient(
                Starlette(routes=gateway_mcp.mcp_routes())) as client:
            began = client.post("/api/mcp/sign-in", json={"action": "begin", "server_id": "bank"})
    finally:
        mcp_client.reset_manager_for_tests()
    assert began.status_code == 200 and began.json()["reason"] == "not_sent", began.json()
    expected = "binary (DER) certificate: convert it to PEM" if kind == "der" else "is not readable"
    assert expected in began.json()["detail"] and "Extra certificate bundle" in began.json()["error"], began.json()
