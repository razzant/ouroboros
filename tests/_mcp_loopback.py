"""Real loopback MCP servers over HTTP and SSE, shared by the tests that need a real answer.

``loopback(transport, required, wrap=None)`` serves one synthetic FastMCP app (one ``echo`` tool) behind a guard
that records and checks request headers, on a free loopback port (``start_asgi``), and resets the process
manager around the test. ``_Provider`` is an in-memory authorization server for the SDK's auth routes.
"""

from __future__ import annotations

import secrets
import socket
import threading
import time

import pytest

from ouroboros import mcp_client


class _Synthetic:
    """One FastMCP app behind a guard that records and checks request headers."""

    def __init__(self, transport, required):
        from mcp.server.fastmcp import FastMCP

        server = FastMCP("synthetic", host="127.0.0.1", log_level="WARNING")

        @server.tool()
        def echo(text: str) -> str:
            return f"echo:{text}"

        self.app = server.streamable_http_app() if transport == "http" else server.sse_app()
        self.required, self.seen = required, []

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = {key.decode().lower(): value.decode() for key, value in scope["headers"]}
            self.seen.append(headers)
            if any(headers.get(name.lower()) != value for name, value in self.required.items()):
                await send({"type": "http.response.start", "status": 401, "headers": []})
                await send({"type": "http.response.body", "body": b"missing synthetic header"})
                return
        await self.app(scope, receive, send)


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


def start_asgi(app, running, name="mcp-loopback", port=0, **options):
    """Serve one ASGI app on a loopback port (a free one by default); ``running`` collects it for shutdown.
    ``options`` go to ``uvicorn.Config`` (``ssl_certfile``/``ssl_keyfile`` serve HTTPS)."""
    import uvicorn

    if not port:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", **options))
    thread = threading.Thread(target=server.run, name=name, daemon=True)
    thread.start()
    running.append((server, thread))
    deadline = time.monotonic() + 10
    while not server.started:
        assert thread.is_alive() and time.monotonic() < deadline, "synthetic MCP server did not start"
        time.sleep(0.02)
    return port


@pytest.fixture
def loopback():
    running = []

    def serve(transport, required, wrap=None):
        app = _Synthetic(transport, required)
        port = start_asgi(wrap(app) if wrap else app, running, f"mcp-loopback-{transport}")
        path = "/mcp" if transport == "http" else "/sse"
        return app, f"http://127.0.0.1:{port}{path}"

    mcp_client.reset_manager_for_tests()
    try:
        yield serve
    finally:
        mcp_client.reset_manager_for_tests()
        for server, thread in running:
            server.should_exit = True
            thread.join(10)
