"""Real loopback MCP servers over HTTP and SSE, shared by the tests that need a real answer.

``loopback(transport, required, wrap=None)`` serves one synthetic FastMCP app (one ``echo`` tool) behind a guard
that records and checks request headers, on a free loopback port (``start_asgi``), and resets the process
manager around the test.
"""

from __future__ import annotations

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


def start_asgi(app, running, name="mcp-loopback"):
    """Serve one ASGI app on a free loopback port; ``running`` collects it for shutdown."""
    import uvicorn

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
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
