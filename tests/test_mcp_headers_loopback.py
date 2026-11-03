"""Saved literal headers reach real loopback MCP servers over HTTP and SSE.

A synthetic FastMCP server per transport refuses any request without the exact
headers. The configuration goes through the real Settings POST/GET handlers
(repeated masked saves included), the saved document configures the process
MCPManager, and the real SDK transports initialize, list and call a tool.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Annotated

import pytest
from pydantic import BaseModel, Field

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


def test_saved_headers_initialize_list_and_call_over_http_and_sse(loopback, settings_client, monkeypatch):  # noqa: F811
    monkeypatch.setattr(mcp_client.MCPManager, "refresh_all_background", lambda *args, **kwargs: None)
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


ORDINARY = {"X-Tenant": "7", "Accept-Language": "en", "X-Api-Version": "2", "X-Release": "2023-06-01"}
SUMMARY = "Found 17 items; open the 7th one. lang en, api 2, release 2023-06-01."
DESCRIPTION = "Search documents; returns at most 7 entries in English (en), API 2 of 2023-06-01."


class _Window(BaseModel):
    since: str = Field(description="Release such as 2023-06-01; at most 7 days, en dates.")


class _Ordinary(_Synthetic):
    """A server whose ordinary text and data contain the same characters as the configured header values."""

    def __init__(self, transport, required):
        from mcp.server.fastmcp import FastMCP

        server = FastMCP("ordinary", host="127.0.0.1", log_level="WARNING")

        @server.tool(description=DESCRIPTION)
        def search(query: Annotated[str, Field(description="Words; open the 7th result.")], window: _Window) -> dict:
            return {"summary": f"{SUMMARY} {query}", "count": 7, "release": window.since}

        self.app = server.streamable_http_app() if transport == "http" else server.sse_app()
        self.required, self.seen = required, []


@pytest.mark.parametrize("transport", ["http", "sse"])
def test_header_values_are_forwarded_configuration_not_filters_over_server_content(loopback, transport):
    """Short and long ordinary header values leave results, descriptions, nested schemas and addresses intact.

    Settings masks every value and the origin guard limits where values go; text the
    server returns is not searched for them. So an echoed custom credential reaches the
    result (an accepted residual, not a goal), while the legacy single token keeps its
    own redaction.
    """
    import json

    headers = {**ORDINARY, "X-Api-Key": CUSTOM}
    app = _Ordinary(transport, headers)
    _unused, url = loopback(transport, {}, wrap=lambda _echo: app)
    kind = "sse" if transport == "sse" else "streamable_http"
    servers = [{"id": "ordinary", "enabled": True, "transport": kind, "url": url, "headers": headers},
               {"id": "legacy", "enabled": True, "transport": kind, "url": url, "headers": ORDINARY,
                "auth_header": "X-Api-Key", "auth_token": CUSTOM}]
    manager = mcp_client.get_manager()
    manager.reconfigure({"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": 10, "MCP_SERVERS": servers})
    assert manager.refresh_server("ordinary")["ok"] and manager.refresh_server("legacy")["ok"]
    tool = next(item for item in manager.list_tools_for_registry() if item["server_id"] == "ordinary")
    assert tool["raw_description"] == DESCRIPTION and DESCRIPTION in tool["description"]
    schema = json.dumps(tool["schema"])
    assert "Words; open the 7th result." in schema and "Release such as 2023-06-01; at most 7 days, en dates." in schema
    status = {row["id"]: row for row in manager.status_payload()["servers"]}["ordinary"]
    assert status["url"] == url and status["tools"][0]["description"] == DESCRIPTION
    assert manager.test_server(servers[0])["tools"][0]["description"] == DESCRIPTION

    def call(server_id):
        name = next(item["name"] for item in manager.list_tools_for_registry() if item["server_id"] == server_id)
        text = manager.call_tool(name, {"query": "invalid key " + CUSTOM, "window": {"since": "2023-06-01"}})
        return text.split("\n\n", 1)[1]

    body = call("ordinary")
    structured = json.loads(body.rsplit("structuredContent: ", 1)[-1])
    assert structured == {"summary": f"{SUMMARY} invalid key {CUSTOM}", "count": 7, "release": "2023-06-01"}
    assert SUMMARY in body and CUSTOM in body  # the server's own echo is not rewritten
    legacy = call("legacy")
    assert CUSTOM not in legacy and SUMMARY in legacy and "<redacted:mcp-auth-token>" in legacy
    assert {row["x-api-key"] for row in app.seen} == {CUSTOM} and {row["x-tenant"] for row in app.seen} == {"7"}


class _Echoing:
    """Answers the first MCP message with invalid JSON that starts with a received header value."""

    def __init__(self, app, transport):
        self.app, self.transport = app, transport

    async def __call__(self, scope, receive, send):
        headers = {key.decode().lower(): value.decode() for key, value in scope.get("headers", [])}
        if scope["type"] == "http" and scope["method"] == "POST" and self.transport == "http":
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": (headers["x-api-key"] + " is not JSON").encode()})
            return
        if scope["type"] == "http" and scope["method"] == "GET" and self.transport == "sse":
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"text/event-stream")]})
            body = (f"event: endpoint\ndata: /messages/?session_id=fixture\n\n"
                    f"event: message\ndata: {headers['x-api-key']} is not JSON\n\n")
            await send({"type": "http.response.body", "body": body.encode(), "more_body": True})
            while (await receive())["type"] != "http.disconnect":
                pass
            return
        if scope["type"] == "http" and scope["method"] == "POST":
            await send({"type": "http.response.start", "status": 202, "headers": []})
            await send({"type": "http.response.body", "body": b""})
            return
        await self.app(scope, receive, send)


@pytest.mark.parametrize("transport, sdk_line", [("http", "Error parsing JSON response"),
                                                  ("sse", "Error parsing server message")])
@pytest.mark.parametrize("redacted", [True, False])
def test_sdk_parse_tracebacks_in_process_logs_never_carry_a_saved_token(loopback, monkeypatch, transport, sdk_line,
                                                                        redacted):
    """The SDK logs the raw invalid body with its traceback before Ouroboros sees an error.

    The real formatted log text is captured; the echo is the legacy token (a
    redaction key, unlike a header-map value). Disabling the projection
    (negative control) shows the witness would see the echoed value.
    """
    import contextlib
    import io
    import logging

    if not redacted:
        monkeypatch.setattr(mcp_client, "_redacted_sdk_logs", lambda cfg, name: contextlib.nullcontext())
    _app, url = loopback(transport, {}, wrap=lambda app: _Echoing(app, transport))
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        manager = mcp_client.get_manager()
        manager.reconfigure({"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": 3, "MCP_SERVERS": [
            {"id": "echo", "enabled": True, "transport": "sse" if transport == "sse" else "streamable_http",
             "url": url, "auth_header": "X-Api-Key", "auth_token": CUSTOM}]})
        outcome = manager.refresh_server("echo")
    finally:
        root.removeHandler(handler)
    logged = stream.getvalue()
    assert outcome["ok"] is False and CUSTOM not in outcome["error"]
    assert sdk_line in logged and "json_invalid" in logged, logged
    if redacted:
        assert CUSTOM not in logged and "<redacted:mcp-auth-token> is not JSON" in logged, logged
    else:
        assert CUSTOM in logged  # negative control: the unprojected SDK record carries the value


class _NotifyingSSE(_Echoing):
    """Streams a log notification whose invalid level echoes a received header; the session warns on the root logger."""

    async def __call__(self, scope, receive, send):
        headers = {key.decode().lower(): value.decode() for key, value in scope.get("headers", [])}
        if scope["type"] == "http" and scope["method"] == "GET":
            note = {"jsonrpc": "2.0", "method": "notifications/message",
                    "params": {"level": headers["x-api-key"], "data": "synthetic"}}
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"text/event-stream")]})
            body = f"event: endpoint\ndata: /messages/?session_id=fixture\n\nevent: message\ndata: {json.dumps(note)}\n\n"
            await send({"type": "http.response.body", "body": body.encode(), "more_body": True})
            while (await receive())["type"] != "http.disconnect":
                pass
            return
        await super().__call__(scope, receive, send)


@pytest.mark.parametrize("redacted", [True, False])
def test_sdk_session_warnings_on_the_root_logger_never_carry_a_saved_token(loopback, monkeypatch, redacted):
    """The SDK session logs an invalid notification on the root logger with its raw parameters.

    The echo is the legacy token. Only records made inside this connection are
    projected; a root record made after it keeps its text, and disabling the
    projection shows the raw echo.
    """
    import contextlib
    import io
    import logging

    if not redacted:
        monkeypatch.setattr(mcp_client, "_redacted_sdk_logs", lambda cfg, name: contextlib.nullcontext())
    _app, url = loopback("sse", {}, wrap=lambda app: _NotifyingSSE(app, "sse"))
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        manager = mcp_client.get_manager()
        manager.reconfigure({"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": 3, "MCP_SERVERS": [
            {"id": "notes", "enabled": True, "transport": "sse", "url": url,
             "auth_header": "X-Api-Key", "auth_token": CUSTOM}]})
        outcome = manager.refresh_server("notes")
        logging.warning("after the connection: %s", CUSTOM)
    finally:
        root.removeHandler(handler)
    session, _, after = stream.getvalue().partition("WARNING root after the connection")
    assert outcome["ok"] is False and CUSTOM not in outcome["error"]
    assert "WARNING root Failed to validate notification" in session, session
    assert after.strip() == ": " + CUSTOM  # outside the connection the root logger is not projected
    if redacted:
        assert CUSTOM not in session and "<redacted:mcp-auth-token>" in session, session
    else:
        assert CUSTOM in session  # negative control: the unprojected session record carries the value


class _Recording:
    def __init__(self, app):
        self.app, self.seen = app, []

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            self.seen.append((scope["method"], [(key.decode().lower(), value.decode())
                                                for key, value in scope["headers"]]))
        await self.app(scope, receive, send)


def test_effective_wire_precedence_of_protocol_headers_is_transport_specific(loopback):
    """Literal means sent as written unless the transport sets that name itself.

    Streamable HTTP sets Accept and Content-Type on every request, and
    Mcp-Protocol-Version once negotiated. The SSE helper sets Accept and
    Cache-Control on its stream request; SSE message POSTs send a configured
    Accept or Content-Type as written, and an MCP server may refuse the latter.
    Every other configured header arrives exactly as written.
    """
    configured = {"Accept": "text/plain", "Cache-Control": "max-age=7", "Mcp-Protocol-Version": "1999-01-01",
                  "Authorization": "Bearer  two-spaces", "X-Api-Key": CUSTOM}
    recorders = []

    def recorded(app):
        recorders.append(_Recording(app))
        return recorders[-1]

    _http, http_url = loopback("http", {}, wrap=recorded)
    _sse, sse_url = loopback("sse", {}, wrap=recorded)
    _typed, typed_url = loopback("sse", {}, wrap=recorded)
    http_rec, sse_rec, typed_rec = recorders
    manager = mcp_client.get_manager()
    manager.reconfigure({"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": 4, "MCP_SERVERS": [
        {"id": "h", "enabled": True, "url": http_url, "headers": {**configured, "Content-Type": "text/plain"}},
        {"id": "s", "enabled": True, "transport": "sse", "url": sse_url, "headers": configured},
        {"id": "t", "enabled": True, "transport": "sse", "url": typed_url, "headers": {"Content-Type": "text/plain"}}]})
    assert manager.refresh_server("h")["ok"] and manager.refresh_server("s")["ok"]
    assert manager.refresh_server("t")["ok"] is False  # the server refused the literal Content-Type

    def values(rec, method):
        rows = [dict(headers) for seen_method, headers in rec.seen if seen_method == method]
        assert rows, (method, rec.seen)
        return rows

    posts = values(http_rec, "POST")
    for request in posts:
        assert request["accept"] == "application/json, text/event-stream"
        assert request["content-type"] == "application/json"
        assert request["cache-control"] == "max-age=7"
        assert request["authorization"] == "Bearer  two-spaces" and request["x-api-key"] == CUSTOM
    assert posts[0]["mcp-protocol-version"] == "1999-01-01"  # before negotiation, the configured value
    assert all(request["mcp-protocol-version"] != "1999-01-01" for request in posts[1:])  # then the SDK's
    for request in values(sse_rec, "GET"):
        assert request["accept"] == "text/event-stream" and request["cache-control"] == "no-store"
        assert request["authorization"] == "Bearer  two-spaces" and request["x-api-key"] == CUSTOM
    for request in values(sse_rec, "POST"):
        assert request["accept"] == "text/plain" and request["content-type"] == "application/json"
        assert request["cache-control"] == "max-age=7" and request["x-api-key"] == CUSTOM
    assert {request["content-type"] for request in values(typed_rec, "POST")} == {"text/plain"}


def test_cyber_url_userinfo_replaces_a_configured_authorization_header(loopback, monkeypatch):
    """With userinfo in the URL (Cyber only), the HTTP client sends Basic auth in Authorization."""
    import base64

    monkeypatch.setattr(mcp_client, "mode_has_unrestricted_agency", lambda _mode: True)
    basic = "Basic " + base64.b64encode(b"user:synthetic-userinfo").decode()
    app, url = loopback("http", {"Authorization": basic, "X-Api-Key": CUSTOM})
    url = url.replace("http://", "http://user:synthetic-userinfo@")
    manager = mcp_client.get_manager()
    manager.reconfigure({"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": 10, "MCP_SERVERS": [
        {"id": "u", "enabled": True, "url": url, "headers": {"Authorization": BEARER, "X-Api-Key": CUSTOM}}]})
    assert manager.refresh_server("u")["ok"]
    assert {headers["authorization"] for headers in app.seen} == {basic}


@pytest.fixture(autouse=True)
def _private_mcp_catalog_root(tmp_path, monkeypatch):
    from ouroboros import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
