"""Settings round-trip tests for MCP — masking, rehydration, and env propagation."""

from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def _isolate_settings(tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.json"
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(settings_path))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data_dir))
    # Reload the config module so it picks up the patched paths.
    import importlib
    import ouroboros.config as cfg

    importlib.reload(cfg)
    yield cfg
    # Restore by reloading once more without the override on teardown so
    # downstream tests in the same process see the real settings path.
    monkeypatch.delenv("OUROBOROS_SETTINGS_PATH", raising=False)
    monkeypatch.delenv("OUROBOROS_DATA_DIR", raising=False)
    importlib.reload(cfg)


def test_settings_defaults_include_mcp_keys(_isolate_settings):
    cfg = _isolate_settings
    assert "MCP_ENABLED" in cfg.SETTINGS_DEFAULTS
    assert "MCP_SERVERS" in cfg.SETTINGS_DEFAULTS
    assert "MCP_TOOL_TIMEOUT_SEC" in cfg.SETTINGS_DEFAULTS
    assert cfg.SETTINGS_DEFAULTS["MCP_ENABLED"] is False
    assert cfg.SETTINGS_DEFAULTS["MCP_SERVERS"] == []
    assert cfg.SETTINGS_DEFAULTS["MCP_TOOL_TIMEOUT_SEC"] == 60


def test_settings_round_trip_preserves_servers_list(_isolate_settings):
    cfg = _isolate_settings
    servers = [
        {
            "id": "github",
            "name": "GitHub",
            "enabled": True,
            "transport": "streamable_http",
            "url": "https://example.com/mcp",
            "auth_header": "Authorization",
            "auth_token": "Bearer secret",
            "allowed_tools": ["search_repos"],
        }
    ]
    payload = dict(cfg.SETTINGS_DEFAULTS)
    payload["MCP_ENABLED"] = True
    payload["MCP_SERVERS"] = servers
    payload["MCP_TOOL_TIMEOUT_SEC"] = 90
    cfg.save_settings(payload, allow_elevation=True)

    loaded = cfg.load_settings()
    assert loaded["MCP_ENABLED"] is True
    assert loaded["MCP_TOOL_TIMEOUT_SEC"] == 90
    assert isinstance(loaded["MCP_SERVERS"], list)
    assert loaded["MCP_SERVERS"][0]["id"] == "github"
    assert loaded["MCP_SERVERS"][0]["auth_token"] == "Bearer secret"


def test_settings_round_trip_preserves_stdio_command_and_exact_args(_isolate_settings):
    cfg = _isolate_settings
    server = {
        "id": "filesystem",
        "name": "Filesystem",
        "enabled": True,
        "transport": "stdio",
        "url": "",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path with spaces"],
        "auth_header": "Authorization",
        "auth_token": "",
        "allowed_tools": ["read_file"],
        "cwd": "/path with spaces/project",
        "env_from_settings": {"TOKEN": "CUSTOM_MCP_KEY"},
        "future_option": {"keep": True},
    }
    payload = dict(cfg.SETTINGS_DEFAULTS)
    payload["MCP_ENABLED"] = True
    payload["MCP_SERVERS"] = [server]
    cfg.save_settings(payload, allow_elevation=True)

    loaded = cfg.load_settings()["MCP_SERVERS"][0]
    assert loaded["transport"] == "stdio"
    assert loaded["command"] == "npx"
    assert loaded["cwd"] == server["cwd"]
    assert loaded["env_from_settings"] == server["env_from_settings"]
    assert loaded["future_option"] == {"keep": True}
    assert loaded["args"] == [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/path with spaces",
    ]


def test_get_mcp_servers_returns_list_of_dicts(_isolate_settings):
    cfg = _isolate_settings
    payload = dict(cfg.SETTINGS_DEFAULTS)
    payload["MCP_SERVERS"] = [{"id": "a", "url": "https://e.example/mcp", "enabled": True}]
    cfg.save_settings(payload, allow_elevation=True)
    servers = cfg.get_mcp_servers()
    assert isinstance(servers, list)
    assert servers and servers[0]["id"] == "a"


def test_get_mcp_tool_timeout_falls_back_to_default(_isolate_settings, monkeypatch):
    cfg = _isolate_settings
    monkeypatch.delenv("MCP_TOOL_TIMEOUT_SEC", raising=False)
    payload = dict(cfg.SETTINGS_DEFAULTS)
    payload["MCP_TOOL_TIMEOUT_SEC"] = 0  # invalid -> default
    cfg.save_settings(payload, allow_elevation=True)
    assert cfg.get_mcp_tool_timeout_sec() == cfg.SETTINGS_DEFAULTS["MCP_TOOL_TIMEOUT_SEC"]


def test_get_mcp_tool_timeout_respects_env(_isolate_settings, monkeypatch):
    cfg = _isolate_settings
    monkeypatch.setenv("MCP_TOOL_TIMEOUT_SEC", "120")
    assert cfg.get_mcp_tool_timeout_sec() == 120


def test_apply_settings_to_env_does_not_serialize_servers_list(_isolate_settings, monkeypatch):
    cfg = _isolate_settings
    monkeypatch.delenv("MCP_SERVERS", raising=False)
    cfg.apply_settings_to_env(
        {
            "MCP_ENABLED": True,
            "MCP_TOOL_TIMEOUT_SEC": 45,
            "MCP_SERVERS": [{"id": "x", "url": "https://e.example/mcp"}],
        }
    )
    import os
    assert os.environ.get("MCP_ENABLED") == "True"
    assert os.environ.get("MCP_TOOL_TIMEOUT_SEC") == "45"
    # MCP_SERVERS is intentionally excluded from env propagation —
    # it stays a list-of-dicts read via load_settings().
    assert "MCP_SERVERS" not in os.environ


def test_coerce_setting_value_handles_string_payload(_isolate_settings):
    cfg = _isolate_settings
    raw = json.dumps([{"id": "a", "url": "https://e.example/mcp"}])
    coerced = cfg._coerce_setting_value("MCP_SERVERS", raw)
    assert isinstance(coerced, list)
    assert coerced and coerced[0]["id"] == "a"


def test_coerce_setting_value_handles_garbage_payload(_isolate_settings):
    cfg = _isolate_settings
    assert cfg._coerce_setting_value("MCP_SERVERS", "not json") == []
    assert cfg._coerce_setting_value("MCP_SERVERS", None) == []
    assert cfg._coerce_setting_value("MCP_SERVERS", 123) == []
