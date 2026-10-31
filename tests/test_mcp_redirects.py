"""Saved MCP credentials never follow an automatic redirect to another origin.

The real SDK transports (Streamable HTTP and SSE) talk to synthetic loopback
servers. A raw socket recorder stands at every redirect target, so "stopped
before it was sent" is observed at the socket rather than inferred from the
returned error. Same-origin redirects and servers without saved credentials
keep the SDK's redirect behavior.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

from ouroboros import mcp_client
from tests.test_mcp_headers_loopback import loopback  # noqa: F401

pytestmark = pytest.mark.serial  # binds real loopback ports

KEY = "synthetic-redirect-key-0001"
TOKEN = "Bearer synthetic-redirect-token-0002"


class Recorder:
    """A raw listener: records each connection's first bytes, then answers ``reply`` and closes."""

    def __init__(self, reply=b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.sock.settimeout(0.2)
        self.port = self.sock.getsockname()[1]
        self.reply, self.connections, self.closed = reply, [], False
        self.thread = threading.Thread(target=self._accept, name="mcp-redirect-recorder", daemon=True)
        self.thread.start()

    def _accept(self):
        while not self.closed:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                continue
            with conn:
                conn.settimeout(2)
                try:
                    data = conn.recv(65536)
                except OSError:
                    data = b""
                self.connections.append(data)
                reply = self.reply(self) if callable(self.reply) else self.reply
                try:
                    conn.sendall(reply)
                except OSError:
                    pass

    def url(self, path="/mcp", scheme="http"):
        return f"{scheme}://127.0.0.1:{self.port}{path}"

    def close(self):
        self.closed = True
        self.thread.join(5)
        self.sock.close()


@pytest.fixture
def recorder():
    made = []

    def make(*args, **kwargs):
        made.append(Recorder(*args, **kwargs))
        return made[-1]

    yield make
    for item in made:
        item.close()


class Redirect:
    """Answers matching requests with 307 to ``location`` and passes the rest to the MCP app."""

    def __init__(self, app, location, *, method="POST", path=None):
        self.app, self.location, self.method, self.path, self.redirected = app, location, method, path, 0

    async def __call__(self, scope, receive, send):
        if (scope["type"] == "http" and scope["method"] == self.method
                and (self.path is None or scope["path"].startswith(self.path))):
            self.redirected += 1
            await send({"type": "http.response.start", "status": 307,
                        "headers": [(b"location", self.location(scope).encode())]})
            await send({"type": "http.response.body", "body": b""})
            return
        await self.app(scope, receive, send)


def _refresh(server, timeout=20):
    manager = mcp_client.get_manager()
    manager.reconfigure({"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": timeout, "MCP_SERVERS": [server]})
    started = time.monotonic()
    outcome = manager.refresh_server(server["id"])
    return outcome, time.monotonic() - started


def _refused(outcome, source, target):
    assert outcome["ok"] is False, outcome
    error = outcome["error"]
    assert error.startswith("MCPRedirectRefused: MCP_REDIRECT_REFUSED"), error
    assert f"at {source} redirected a request to {target}" in error, error
    assert "?" not in error and "/mcp" not in error and KEY not in error and TOKEN not in error, error


@pytest.mark.parametrize("credentials", [{"headers": {"X-Api-Key": KEY}},
                                         {"auth_header": "Authorization", "auth_token": TOKEN}])
def test_streamable_http_cross_origin_redirect_is_stopped_before_it_is_sent(loopback, recorder, credentials):  # noqa: F811
    target = recorder()
    _app, url = loopback("http", {}, wrap=lambda app: Redirect(app, lambda scope: target.url("/mcp?token=q")))
    outcome, _ = _refresh({"id": "moved", "enabled": True, "url": url, **credentials})
    _refused(outcome, url.rsplit("/", 1)[0], f"http://127.0.0.1:{target.port}")
    # The legacy Authorization pair is stopped too: the client would strip it and follow.
    assert target.connections == []


def test_http_to_https_on_the_same_host_and_port_is_a_different_origin(recorder):
    def upgrade(rec):
        return (f"HTTP/1.1 307 Temporary Redirect\r\nLocation: https://127.0.0.1:{rec.port}/mcp\r\n"
                "Content-Length: 0\r\nConnection: close\r\n\r\n").encode()

    server = recorder(reply=upgrade)
    outcome, _ = _refresh({"id": "upgrade", "enabled": True, "url": server.url(),
                           "headers": {"X-Api-Key": KEY}})
    _refused(outcome, f"http://127.0.0.1:{server.port}", f"https://127.0.0.1:{server.port}")
    assert len(server.connections) == 1 and server.connections[0].startswith(b"POST /mcp HTTP/1.1")
    assert KEY.encode() in server.connections[0]  # the configured origin received it; no TLS attempt followed


def test_sse_stream_redirect_to_another_origin_is_stopped(loopback, recorder):  # noqa: F811
    target = recorder()
    _app, url = loopback("sse", {}, wrap=lambda app: Redirect(app, lambda scope: target.url("/sse"), method="GET"))
    outcome, _ = _refresh({"id": "stream", "enabled": True, "transport": "sse", "url": url,
                           "headers": {"X-Api-Key": KEY}})
    _refused(outcome, url.rsplit("/", 1)[0], f"http://127.0.0.1:{target.port}")
    assert target.connections == []


def test_sse_message_post_redirect_to_another_origin_is_stopped_without_waiting_for_the_timeout(loopback, recorder):  # noqa: F811
    target = recorder()
    _app, url = loopback("sse", {}, wrap=lambda app: Redirect(
        app, lambda scope: target.url("/messages/"), path="/messages"))
    outcome, elapsed = _refresh({"id": "posts", "enabled": True, "transport": "sse", "url": url,
                                 "headers": {"X-Api-Key": KEY}}, timeout=30)
    _refused(outcome, url.rsplit("/", 1)[0], f"http://127.0.0.1:{target.port}")
    assert target.connections == [] and elapsed < 15, elapsed


@pytest.mark.parametrize("transport", ["http", "sse"])
def test_same_origin_redirects_keep_working_with_saved_headers(loopback, transport):  # noqa: F811
    method, old, new = ("POST", "/old", "/mcp") if transport == "http" else ("GET", "/old", "/sse")
    app, url = loopback(transport, {"X-Api-Key": KEY}, wrap=lambda app: Redirect(
        app, lambda scope: new, method=method, path=old))
    moved = url.rsplit("/", 1)[0] + old
    outcome, _ = _refresh({"id": "same", "enabled": True, "transport": "sse" if transport == "sse" else "streamable_http",
                           "url": moved, "headers": {"X-Api-Key": KEY}})
    assert outcome["ok"] and [tool["name"] for tool in outcome["tools"]] == ["echo"], outcome
    assert app.seen and all(headers.get("x-api-key") == KEY for headers in app.seen)


def test_a_server_without_saved_credentials_keeps_following_cross_origin_redirects(loopback):  # noqa: F811
    _final, final_url = loopback("http", {})
    _app, url = loopback("http", {}, wrap=lambda app: Redirect(app, lambda scope: final_url))
    outcome, _ = _refresh({"id": "open", "enabled": True, "url": url})
    assert outcome["ok"] and [tool["name"] for tool in outcome["tools"]] == ["echo"], outcome


def test_the_guard_is_per_connection_and_leaves_no_hook_behind(loopback):  # noqa: F811
    seen = []
    original = mcp_client._CredentialOriginGuard.client

    def spy(self, *args, **kwargs):
        client = original(self, *args, **kwargs)
        seen.append(client)
        return client

    app, url = loopback("http", {"X-Api-Key": KEY})
    mcp_client._CredentialOriginGuard.client = spy
    try:
        for _ in range(2):
            outcome, _ = _refresh({"id": "twice", "enabled": True, "url": url, "headers": {"X-Api-Key": KEY}})
            assert outcome["ok"], outcome
    finally:
        mcp_client._CredentialOriginGuard.client = original
    assert len(seen) == 2 and seen[0] is not seen[1]
    assert all(client.is_closed and len(client.event_hooks["request"]) == 1 for client in seen)


@pytest.fixture(autouse=True)
def _private_mcp_catalog_root(tmp_path, monkeypatch):
    from ouroboros import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
