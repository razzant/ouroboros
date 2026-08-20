"""HTTP surface tests for ouroboros.gateway.mcp.

Uses a narrow Starlette TestClient app around the real endpoint callables.
The MCP manager is wired with a fake transport so no live MCP server is
needed, and the full server lifespan is deliberately not started.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import mcp_client
from ouroboros.tools.tool_result import ToolResult


@pytest.fixture(autouse=True)
def _isolate_manager():
    mcp_client.reset_manager_for_tests()
    yield
    mcp_client.reset_manager_for_tests()


class _FakeTransport:
    def __init__(self, response):
        self.response = response
        self.list_calls = []
        self.call_calls = []

    async def list_tools(self, cfg, timeout):
        self.list_calls.append(
            (cfg.id, cfg.url, cfg.auth_token, timeout, cfg.command, list(cfg.args))
        )
        return list(self.response)

    async def call_tool(self, cfg, name, arguments, timeout):
        self.call_calls.append((cfg.id, name, dict(arguments or {}), timeout))
        return ToolResult(
            status="ok",
            code="OK",
            text=f"echo({cfg.id}/{name})",
            meta={"mcp_is_error": False},
        )


def _wire_singleton(transport):
    mgr = mcp_client.get_manager()
    mgr._async_list_tools = transport.list_tools
    mgr._async_call_tool = transport.call_tool


def _good_server(**overrides) -> dict:
    base = {
        "id": "demo",
        "name": "Demo",
        "enabled": True,
        "transport": "streamable_http",
        "url": "https://e.example/mcp",
        "auth_header": "Authorization",
        "auth_token": "Bearer secret-1234",
        "allowed_tools": [],
    }
    base.update(overrides)
    return base


def _make_client(tmp_path, monkeypatch):
    import server as srv
    from ouroboros.gateway import mcp as mcp_api

    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    patches = [
        patch.object(srv, "_start_supervisor_if_needed", lambda *_a, **_k: None),
        patch.object(srv, "_apply_settings_to_env", lambda *_a, **_k: None),
        patch.object(srv, "apply_runtime_provider_defaults", lambda s: (s, False, [])),
        patch("ouroboros.server_auth.get_configured_network_password", return_value=""),
    ]
    for p in patches:
        p.start()
    app = Starlette(routes=[
        Route("/api/settings", endpoint=srv.api_settings_get, methods=["GET"]),
        Route("/api/settings", endpoint=srv.api_settings_post, methods=["POST"]),
        Route("/api/mcp/status", endpoint=mcp_api.api_mcp_status, methods=["GET"]),
        Route("/api/mcp/refresh", endpoint=mcp_api.api_mcp_refresh, methods=["POST"]),
        Route("/api/mcp/test", endpoint=mcp_api.api_mcp_test, methods=["POST"]),
    ])
    app.state.drive_root = drive_root
    app.state.repo_dir = tmp_path / "repo"
    return TestClient(app), patches


def _stop(patches):
    for p in patches:
        try:
            p.stop()
        except RuntimeError:
            pass


def test_status_endpoint_returns_redacted_payload(tmp_path, monkeypatch):
    fake = _FakeTransport([
        {"name": "ping", "description": "Ping", "input_schema": {"type": "object", "properties": {}}},
    ])
    _wire_singleton(fake)
    mcp_client.reconfigure_from_settings({
        "MCP_ENABLED": True,
        "MCP_TOOL_TIMEOUT_SEC": 30,
        "MCP_SERVERS": [_good_server()],
    })
    mcp_client.get_manager().refresh_server("demo")

    client, patches = _make_client(tmp_path, monkeypatch)
    try:
        # Patch load_settings used by mcp_api._ensure_configured to avoid
        # touching the real on-disk settings.json (and accidentally
        # clobbering the fake transport's wiring).
        with patch("ouroboros.gateway.mcp.load_settings", return_value={
            "MCP_ENABLED": True,
            "MCP_TOOL_TIMEOUT_SEC": 30,
            "MCP_SERVERS": [_good_server()],
        }):
            # Re-wire the fake transport AFTER the load_settings patch so
            # ``_ensure_configured`` keeps the manager state consistent.
            _wire_singleton(fake)
            resp = client.get("/api/mcp/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["sdk_available"] in (True, False)  # depends on host
        body = data["servers"]
        assert body and body[0]["id"] == "demo"
        assert body[0]["auth_configured"] is True
        assert "auth_token" not in body[0]  # never leaked
        assert "secret-1234" not in resp.text
    finally:
        _stop(patches)


def test_test_endpoint_with_inline_server(tmp_path, monkeypatch):
    fake = _FakeTransport([
        {"name": "hello", "description": "Say hi", "input_schema": {}},
    ])
    _wire_singleton(fake)
    mcp_client.reconfigure_from_settings({
        "MCP_ENABLED": True,
        "MCP_TOOL_TIMEOUT_SEC": 10,
        "MCP_SERVERS": [],
    })
    client, patches = _make_client(tmp_path, monkeypatch)
    try:
        with patch("ouroboros.gateway.mcp.load_settings", return_value={
            "MCP_ENABLED": True,
            "MCP_TOOL_TIMEOUT_SEC": 10,
            "MCP_SERVERS": [],
        }):
            _wire_singleton(fake)
            resp = client.post(
                "/api/mcp/test",
                json={"server": _good_server(id="probe")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["tool_count"] == 1
        assert data["tools"][0]["name"] == "hello"
    finally:
        _stop(patches)


def test_test_endpoint_accepts_inline_stdio_server(tmp_path, monkeypatch):
    fake = _FakeTransport([
        {"name": "hello", "description": "Say hi", "input_schema": {}},
    ])
    _wire_singleton(fake)
    mcp_client.reconfigure_from_settings({
        "MCP_ENABLED": True,
        "MCP_TOOL_TIMEOUT_SEC": 10,
        "MCP_SERVERS": [],
    })
    client, patches = _make_client(tmp_path, monkeypatch)
    try:
        with patch("ouroboros.gateway.mcp.load_settings", return_value={
            "MCP_ENABLED": True,
            "MCP_TOOL_TIMEOUT_SEC": 10,
            "MCP_SERVERS": [],
        }):
            _wire_singleton(fake)
            resp = client.post(
                "/api/mcp/test",
                json={"server": {
                    "id": "local",
                    "transport": "stdio",
                    "command": "python3",
                    "args": ["server.py", "value with spaces"],
                }},
            )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert fake.list_calls[-1][1:3] == ("", "")
        assert fake.list_calls[-1][4:] == (
            "python3",
            ["server.py", "value with spaces"],
        )
    finally:
        _stop(patches)


def test_test_endpoint_rehydrates_masked_inline_candidate_by_server_id(tmp_path, monkeypatch):
    """Saved-server Test flow must use edited fields plus persisted token.

    A saved card arrives in the UI with a masked auth token. If the user
    edits URL/transport/header and clicks Test, the frontend sends both
    ``server_id`` and the edited inline ``server`` payload. The backend
    must rehydrate the masked token from saved settings while preserving
    the edited URL (otherwise Test either runs unauthenticated or checks
    stale config).
    """
    fake = _FakeTransport([
        {"name": "hello", "description": "Say hi", "input_schema": {}},
    ])
    _wire_singleton(fake)
    persisted = _good_server(id="demo", url="https://saved.example/mcp")
    persisted["auth_token"] = "Bearer persisted-secret"
    mcp_client.reconfigure_from_settings({
        "MCP_ENABLED": True,
        "MCP_TOOL_TIMEOUT_SEC": 10,
        "MCP_SERVERS": [persisted],
    })
    client, patches = _make_client(tmp_path, monkeypatch)
    try:
        with patch("ouroboros.gateway.mcp.load_settings", return_value={
            "MCP_ENABLED": True,
            "MCP_TOOL_TIMEOUT_SEC": 10,
            "MCP_SERVERS": [persisted],
        }):
            _wire_singleton(fake)
            edited = dict(persisted)
            edited["url"] = "https://edited.example/mcp"
            edited["auth_token"] = "Bearer p..."
            resp = client.post(
                "/api/mcp/test",
                json={"server_id": "demo", "server": edited},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert fake.list_calls, "fake transport was not called"
        # It must test the edited candidate URL while using the persisted
        # real token rehydrated from saved settings.
        assert fake.list_calls[-1][1] == "https://edited.example/mcp"
        assert fake.list_calls[-1][2] == "Bearer persisted-secret"
        assert "persisted-secret" not in resp.text
    finally:
        _stop(patches)


def test_test_endpoint_accepts_friendly_server_id(tmp_path, monkeypatch):
    """Single-server test lookup canonicalizes ``server_id`` before matching."""
    fake = _FakeTransport([
        {"name": "hello", "description": "Say hi", "input_schema": {}},
    ])
    _wire_singleton(fake)
    persisted = _good_server(id="GitHub Server!", url="https://saved.example/mcp")
    persisted["auth_token"] = "Bearer persisted-secret"
    mcp_client.reconfigure_from_settings({
        "MCP_ENABLED": True,
        "MCP_TOOL_TIMEOUT_SEC": 10,
        "MCP_SERVERS": [persisted],
    })
    client, patches = _make_client(tmp_path, monkeypatch)
    try:
        with patch("ouroboros.gateway.mcp.load_settings", return_value={
            "MCP_ENABLED": True,
            "MCP_TOOL_TIMEOUT_SEC": 10,
            "MCP_SERVERS": [persisted],
        }):
            _wire_singleton(fake)
            resp = client.post("/api/mcp/test", json={"server_id": "GitHub Server!"})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert fake.list_calls[-1][0] == "github_server"
    finally:
        _stop(patches)


def test_test_endpoint_rejects_invalid_url(tmp_path, monkeypatch):
    client, patches = _make_client(tmp_path, monkeypatch)
    try:
        with patch("ouroboros.gateway.mcp.load_settings", return_value={
            "MCP_ENABLED": True,
            "MCP_TOOL_TIMEOUT_SEC": 10,
            "MCP_SERVERS": [],
        }):
            resp = client.post(
                "/api/mcp/test",
                json={"server": {"id": "x", "url": "ftp://nope"}},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is False
        assert "Invalid" in data["error"]
    finally:
        _stop(patches)


def test_refresh_endpoint_targets_single_server(tmp_path, monkeypatch):
    fake = _FakeTransport([
        {"name": "ping", "description": "", "input_schema": {"type": "object", "properties": {}}},
    ])
    _wire_singleton(fake)
    mcp_client.reconfigure_from_settings({
        "MCP_ENABLED": True,
        "MCP_TOOL_TIMEOUT_SEC": 10,
        "MCP_SERVERS": [_good_server()],
    })
    client, patches = _make_client(tmp_path, monkeypatch)
    try:
        with patch("ouroboros.gateway.mcp.load_settings", return_value={
            "MCP_ENABLED": True,
            "MCP_TOOL_TIMEOUT_SEC": 10,
            "MCP_SERVERS": [_good_server()],
        }):
            _wire_singleton(fake)
            resp = client.post("/api/mcp/refresh", json={"server_id": "demo"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["server_id"] == "demo"
        assert data["ok"] is True
        assert data["tool_count"] == 1
    finally:
        _stop(patches)


def test_settings_post_rehydrates_masked_auth_token(tmp_path, monkeypatch):
    """End-to-end HTTP test: real token must survive a masked-token round-trip.

    The UI flow is:
      1. GET /api/settings  -> server returns auth_token masked as ``"abcd..."``.
      2. User edits other fields (name, transport, allowed_tools, ...).
      3. POST /api/settings -> the masked token must be replaced by the
         on-disk real token before persistence; otherwise the next save
         silently overwrites the working credential with the literal
         mask string.

    This is the test critic finding #4 (real bug surface). The mask is
    written by ``server._mask_mcp_servers_payload`` and the rehydration
    is performed by ``server._rehydrate_mcp_servers_payload``; both
    paths must hold for this round-trip to succeed.
    """
    import server as srv

    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    tmp_path / "settings.json"

    # Seed: a real auth token already saved on disk.
    real_token = "Bearer plaintext-live-token-DO-NOT-LEAK"
    on_disk = {
        "MCP_ENABLED": True,
        "MCP_TOOL_TIMEOUT_SEC": 60,
        "MCP_SERVERS": [
            {
                "id": "demo",
                "name": "Demo",
                "enabled": True,
                "transport": "streamable_http",
                "url": "https://example.com/mcp",
                "auth_header": "Authorization",
                "auth_token": real_token,
                "allowed_tools": [],
            }
        ],
    }

    saved_payload: Dict[str, Any] = {}

    def fake_load_settings():
        # Always return a fresh copy so the merge logic does not mutate
        # the shared seed dict in place.
        import copy
        return copy.deepcopy(on_disk)

    def fake_save_settings(settings, *_, **__):
        saved_payload.clear()
        saved_payload.update(settings)
        # Mirror what ``save_settings`` would do: persist for subsequent
        # ``load_settings`` calls in the same test by mutating the seed.
        on_disk.update(settings)

    patches = [
        patch.object(srv, "_start_supervisor_if_needed", lambda *_a, **_k: None),
        patch.object(srv, "_apply_settings_to_env", lambda *_a, **_k: None),
        patch.object(srv, "apply_runtime_provider_defaults", lambda s: (dict(s), False, [])),
        patch.object(srv, "load_settings", side_effect=fake_load_settings),
        patch.object(srv, "save_settings", side_effect=fake_save_settings),
        patch.object(srv._gateway_settings, "_owner_read_settings_raw", side_effect=fake_load_settings),
        patch.object(srv._gateway_settings, "_owner_write_settings", side_effect=fake_save_settings),
        patch("ouroboros.server_auth.get_configured_network_password", return_value=""),
    ]
    for p in patches:
        p.start()
    try:
        app = Starlette(routes=[
            Route("/api/settings", endpoint=srv.api_settings_get, methods=["GET"]),
            Route("/api/settings", endpoint=srv.api_settings_post, methods=["POST"]),
        ])
        app.state.drive_root = drive_root
        app.state.repo_dir = tmp_path / "repo"
        with TestClient(app) as client:
            # Step 1: GET — verify the wire surface masks the token.
            resp = client.get("/api/settings")
            assert resp.status_code == 200
            wire_servers = resp.json().get("MCP_SERVERS") or []
            assert wire_servers and wire_servers[0]["auth_configured"] is True
            wire_token = wire_servers[0]["auth_token"]
            assert wire_token != real_token  # must be masked
            assert real_token not in resp.text

            # Step 2 + 3: POST the wire shape back unchanged (user edited
            # `name` / `allowed_tools` but did NOT re-type the token).
            edited = dict(wire_servers[0])
            edited["name"] = "Demo Renamed"
            edited["allowed_tools"] = ["search"]
            payload = {
                "MCP_ENABLED": True,
                "MCP_TOOL_TIMEOUT_SEC": 90,
                "MCP_SERVERS": [edited],
            }
            resp = client.post("/api/settings", json=payload)
            assert resp.status_code == 200, resp.text

            # Persisted dict must contain the REAL token, not the mask.
            persisted_servers = saved_payload.get("MCP_SERVERS") or []
            assert persisted_servers, "MCP_SERVERS not persisted"
            persisted = persisted_servers[0]
            assert persisted["auth_token"] == real_token, (
                f"Rehydration regression: expected real token, got {persisted['auth_token']!r}"
            )
            # And the cosmetic edits flowed through.
            assert persisted["name"] == "Demo Renamed"
            assert persisted["allowed_tools"] == ["search"]
            assert saved_payload["MCP_TOOL_TIMEOUT_SEC"] == 90
    finally:
        _stop(patches)


def test_settings_post_canonicalizes_mcp_server_ids(tmp_path, monkeypatch):
    """Friendly server ids should be canonicalized before persistence.

    This keeps settings.json, /api/settings, /api/mcp/status, and single
    server actions from disagreeing about the id for e.g. "GitHub Server!".
    """
    import server as srv

    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    saved_payload: Dict[str, Any] = {}

    def fake_load_settings():
        return {
            "MCP_ENABLED": True,
            "MCP_TOOL_TIMEOUT_SEC": 60,
            "MCP_SERVERS": [
                {
                    "id": "GitHub Server!",
                    "name": "GitHub",
                    "enabled": True,
                    "transport": "streamable_http",
                    "url": "https://example.com/mcp",
                    "auth_header": "Authorization",
                    "auth_token": "Bearer real-token",
                    "allowed_tools": [],
                }
            ],
        }

    def fake_save_settings(settings, *_, **__):
        saved_payload.clear()
        saved_payload.update(settings)

    patches = [
        patch.object(srv, "_start_supervisor_if_needed", lambda *_a, **_k: None),
        patch.object(srv, "_apply_settings_to_env", lambda *_a, **_k: None),
        patch.object(srv, "apply_runtime_provider_defaults", lambda s: (dict(s), False, [])),
        patch.object(srv, "load_settings", side_effect=fake_load_settings),
        patch.object(srv, "save_settings", side_effect=fake_save_settings),
        patch.object(srv._gateway_settings, "_owner_read_settings_raw", side_effect=fake_load_settings),
        patch.object(srv._gateway_settings, "_owner_write_settings", side_effect=fake_save_settings),
        patch("ouroboros.server_auth.get_configured_network_password", return_value=""),
    ]
    for p in patches:
        p.start()
    try:
        app = Starlette(routes=[
            Route("/api/settings", endpoint=srv.api_settings_get, methods=["GET"]),
            Route("/api/settings", endpoint=srv.api_settings_post, methods=["POST"]),
        ])
        app.state.drive_root = drive_root
        app.state.repo_dir = tmp_path / "repo"
        with TestClient(app) as client:
            get_resp = client.get("/api/settings")
            assert get_resp.status_code == 200
            assert (get_resp.json().get("MCP_SERVERS") or [])[0]["id"] == "github_server"

            post_resp = client.post(
                "/api/settings",
                json={
                    "MCP_ENABLED": True,
                    "MCP_TOOL_TIMEOUT_SEC": 60,
                    "MCP_SERVERS": [
                        {
                            "id": "GitHub Server!",
                            "name": "GitHub",
                            "enabled": True,
                            "transport": "streamable_http",
                            "url": "https://example.com/mcp",
                            "auth_header": "Authorization",
                            "auth_token": "Bearer r...",
                            "auth_configured": True,
                            "allowed_tools": [],
                        }
                    ],
                },
            )
            assert post_resp.status_code == 200, post_resp.text
            persisted = (saved_payload.get("MCP_SERVERS") or [])[0]
            assert persisted["id"] == "github_server"
            assert persisted["auth_token"] == "Bearer real-token"
    finally:
        _stop(patches)


def test_settings_get_masks_mcp_auth_token(tmp_path, monkeypatch):
    """The full /api/settings GET path must mask MCP auth tokens.

    This is independent of the manager — the mask is applied in
    ``server.api_settings_get``.
    """
    import server as srv

    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    fake_settings = {
        "MCP_ENABLED": True,
        "MCP_TOOL_TIMEOUT_SEC": 30,
        "MCP_SERVERS": [_good_server(auth_token="Bearer plain-text-token-XYZ")],
    }
    patches = [
        patch.object(srv, "_start_supervisor_if_needed", lambda *_a, **_k: None),
        patch.object(srv, "_apply_settings_to_env", lambda *_a, **_k: None),
        patch.object(srv, "apply_runtime_provider_defaults", lambda s: (dict(s), False, [])),
        patch.object(srv, "load_settings", return_value=fake_settings),
        patch("ouroboros.server_auth.get_configured_network_password", return_value=""),
    ]
    for p in patches:
        p.start()
    try:
        app = Starlette(routes=[
            Route("/api/settings", endpoint=srv.api_settings_get, methods=["GET"]),
        ])
        app.state.drive_root = drive_root
        app.state.repo_dir = tmp_path / "repo"
        with TestClient(app) as client:
            resp = client.get("/api/settings")
            assert resp.status_code == 200
            assert "plain-text-token-XYZ" not in resp.text
            data = resp.json()
            servers = data.get("MCP_SERVERS") or []
            assert servers and servers[0]["auth_configured"] is True
            assert servers[0]["auth_token"] != "Bearer plain-text-token-XYZ"
    finally:
        _stop(patches)
