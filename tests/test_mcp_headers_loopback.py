"""Saved literal headers reach real loopback MCP servers over HTTP and SSE.

A synthetic FastMCP server per transport refuses any request without the exact
headers. The configuration goes through the real Settings POST/GET handlers
(repeated masked saves included), the saved document configures the process
MCPManager, and the real SDK transports initialize, list and call a tool.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

from ouroboros import mcp_client
from tests.test_settings_secret_mask import settings_client  # noqa: F401

pytestmark = pytest.mark.serial  # binds real loopback ports

BEARER = "Bearer synthetic-loopback-0001"
BASIC = "Basic c3ludGhldGljOmxvb3BiYWNr"
CUSTOM = "synthetic-loopback-key-0002"


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


@pytest.fixture
def loopback():
    import uvicorn

    running = []

    def serve(transport, required):
        app = _Synthetic(transport, required)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        thread = threading.Thread(target=server.run, name=f"mcp-loopback-{transport}", daemon=True)
        thread.start()
        running.append((server, thread))
        deadline = time.monotonic() + 10
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline, "synthetic MCP server did not start"
            time.sleep(0.02)
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


def test_saved_headers_initialize_list_and_call_over_http_and_sse(loopback, settings_client, monkeypatch):  # noqa: F811
    monkeypatch.setattr(mcp_client, "refresh_all_background", lambda **kwargs: None)
    client, on_disk, saved = settings_client
    http_app, http_url = loopback("http", {"Authorization": BEARER, "X-Api-Key": CUSTOM})
    sse_app, sse_url = loopback("sse", {"Authorization": BASIC, "X-Api-Key": CUSTOM})
    servers = [
        {"id": "http_docs", "enabled": True, "transport": "streamable_http", "url": http_url,
         "headers": {"Authorization": BEARER, "X-Api-Key": CUSTOM, "X-Empty": ""}},
        # The legacy pair keeps working beside a distinct literal header.
        {"id": "sse_docs", "enabled": True, "transport": "sse", "url": sse_url,
         "auth_header": "Authorization", "auth_token": BASIC, "headers": {"X-Api-Key": CUSTOM}},
    ]
    assert client.post("/api/settings", json={"MCP_ENABLED": True, "MCP_SERVERS": servers}).status_code == 200
    for _ in range(2):  # the masked document Settings serves saves back unchanged
        served = client.get("/api/settings")
        assert BEARER not in served.text and CUSTOM not in served.text and BASIC not in served.text
        response = client.post("/api/settings", json={"MCP_SERVERS": served.json()["MCP_SERVERS"]})
        assert response.status_code == 200, response.text
    assert [entry["headers"] for entry in saved["MCP_SERVERS"]] == [entry["headers"] for entry in servers]

    manager = mcp_client.get_manager()
    manager.reconfigure(dict(on_disk))
    for server_id in ("http_docs", "sse_docs"):
        listed = manager.refresh_server(server_id)
        assert listed["ok"] and [tool["name"] for tool in listed["tools"]] == ["echo"], listed
        called = manager._call_tool_result(f"mcp_{server_id}__echo", {"text": server_id})
        assert called.status == "ok" and f"echo:{server_id}" in called.text
    for app in (http_app, sse_app):
        assert len(app.seen) >= 3  # initialize, list and call each reached the guard
        assert all("x-empty" not in headers for headers in app.seen)
    assert {headers["authorization"] for headers in http_app.seen} == {BEARER}
    assert {headers["authorization"] for headers in sse_app.seen} == {BASIC}


def test_a_rejected_header_reports_the_failure_without_its_value(loopback):
    _app, url = loopback("http", {"X-Api-Key": CUSTOM})
    wrong = "synthetic-wrong-key-0003"
    manager = mcp_client.get_manager()
    manager.reconfigure({"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": 10, "MCP_SERVERS": [
        {"id": "denied", "enabled": True, "url": url, "headers": {"X-Api-Key": wrong}}]})
    outcome = manager.refresh_server("denied")
    assert outcome["ok"] is False and wrong not in outcome["error"]
    assert wrong not in str(manager.status_payload())
