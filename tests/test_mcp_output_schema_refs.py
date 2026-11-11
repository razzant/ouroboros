"""A server-declared tool output schema never makes the client fetch a URL (CVE-2026-116686).

A synthetic loopback MCP server lists one tool whose ``outputSchema`` names a
``$ref`` on a raw socket recorder, then answers a call with structured content.
The real SDK client validates that result during the call; the recorder shows
whether the validation fetched the reference. An in-document ``$ref`` keeps
working.
"""

from __future__ import annotations

import pytest

from ouroboros import mcp_client
from tests.test_mcp_headers_loopback import loopback  # noqa: F401
from tests.test_mcp_redirects import recorder  # noqa: F401


def _server(output_schema):
    """A FastMCP app whose single tool is listed with ``output_schema`` and returns ``{"n": 1}``."""
    from mcp import types
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("schema-refs", host="127.0.0.1", log_level="WARNING")
    lowlevel = server._mcp_server

    @lowlevel.list_tools()
    async def list_tools():
        return [types.Tool(name="count", inputSchema={"type": "object"}, outputSchema=output_schema)]

    @lowlevel.call_tool()
    async def call_tool(name, arguments):
        # A ready CallToolResult skips the server's own output validation, so
        # only the client can touch the reference.
        return types.CallToolResult(content=[types.TextContent(type="text", text="1")], structuredContent={"n": 1})

    return server.streamable_http_app()


@pytest.fixture(autouse=True)
def _private_mcp_catalog_root(tmp_path, monkeypatch):
    from ouroboros import config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)


def _call(serve, output_schema):
    _app, url = serve("http", {}, wrap=lambda _app: _server(output_schema))
    manager = mcp_client.get_manager()
    manager.reconfigure({"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": 20, "MCP_SERVERS": [
        {"id": "refs", "enabled": True, "url": url}]})
    listed = manager.refresh_server("refs")
    assert listed["ok"] and [tool["name"] for tool in listed["tools"]] == ["count"], listed
    return manager._call_tool_result("mcp_refs__count", {})


def test_an_external_ref_in_a_tool_output_schema_is_not_fetched(loopback, recorder):  # noqa: F811
    body = b'{"type": "object"}'
    target = recorder(reply=b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                           + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
    called = _call(loopback, {"$ref": target.url("/schema.json")})
    assert target.connections == [], target.connections
    assert called.status != "ok" and "Invalid schema for tool count" in called.text, (called.status, called.text)


def test_an_in_document_ref_still_validates_the_result(loopback):  # noqa: F811
    called = _call(loopback, {"$defs": {"count": {"type": "object", "properties": {"n": {"type": "integer"}},
                                                 "required": ["n"]}},
                             "$ref": "#/$defs/count"})
    assert called.status == "ok", (called.status, called.text)
