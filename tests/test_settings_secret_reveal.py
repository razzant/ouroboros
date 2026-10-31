"""An explicit Show reads one real secret without changing Settings or passive GET.

Exercise the mounted gateway and real Settings loader with disposable synthetic
credentials. The UI's Show/Hide draft and save invariants live in browser tests.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from ouroboros.gateway.router import collect_routes
from ouroboros.secret_masking import MASKED_SECRET_SETTING_KEYS, mask_settings_secret
from ouroboros.server_auth import NetworkAuthGate

pytestmark = pytest.mark.serial


@pytest.fixture
def reveal_settings(tmp_path, monkeypatch):
    from ouroboros import config

    data = tmp_path / "data"
    data.mkdir()
    path = data / "settings.json"
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "SETTINGS_PATH", path)
    for key in config.SETTINGS_DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("OUROBOROS_SETTINGS_SHA256", raising=False)
    settings = {  # an install that already carries the consciousness level: loading runs no one-time copy
        "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE": "full",
        "ANTHROPIC_API_KEY": "sk-ant-api03-fixture-complete-tail",
        "CUSTOM_BRIDGE_TOKEN": "custom-fixture-kept-complete",
        "MCP_SERVERS": [
            {"id": "first-server", "auth_token": "Bearer first-synthetic-token"},
            {"id": "second-server", "auth_token": "Bearer second-synthetic-token"},
        ],
    }
    path.write_text(json.dumps(settings), encoding="utf-8")
    app = Starlette(routes=collect_routes(data_dir=data))
    app.state.drive_root = data
    app.state.repo_dir = tmp_path
    app.state.bind_host = "127.0.0.1"
    app.state.port_file = data / "server_port"
    app.state.default_port = 8765
    return SimpleNamespace(app=NetworkAuthGate(app), path=path, settings=settings, data=data)


@pytest.mark.parametrize("key", sorted(MASKED_SECRET_SETTING_KEYS) + ["CUSTOM_BRIDGE_TOKEN"])
def test_show_returns_exact_selected_value_and_keeps_passive_get_masked(reveal_settings, key):
    case = reveal_settings
    value = "synthetic selected secret with complete suffix..."
    case.settings[key] = value
    case.path.write_text(json.dumps(case.settings), encoding="utf-8")
    before = case.path.read_bytes()
    environment = dict(os.environ)
    with TestClient(case.app, client=("127.0.0.1", 50000)) as client:
        passive = client.get("/api/settings")
        assert passive.status_code == 200
        assert passive.json()[key] == mask_settings_secret(key, value)
        assert value not in passive.text
        shown = client.post("/api/settings/secret", json={"key": key})
        assert shown.status_code == 200
        assert shown.json() == {"value": value}
        assert shown.headers["cache-control"] == "no-store"
        assert client.get("/api/settings").json()[key] == mask_settings_secret(key, value)
    assert case.path.read_bytes() == before
    assert dict(os.environ) == environment
    assert not (case.data / "logs").exists()


@pytest.mark.parametrize("stored", [None, "", "sk-ant-a..."])
def test_reveal_uses_environment_fallback_without_persisting_it(reveal_settings, monkeypatch, stored):
    case = reveal_settings
    value = "sk-ant-api03-environment-fixture-tail"
    monkeypatch.setenv("ANTHROPIC_API_KEY", value)
    if stored is None:
        case.path.unlink()
        before = None
    else:
        case.settings["ANTHROPIC_API_KEY"] = stored
        case.path.write_text(json.dumps(case.settings), encoding="utf-8")
        before = case.path.read_bytes()
    with TestClient(case.app) as client:
        response = client.post("/api/settings/secret", json={"key": "ANTHROPIC_API_KEY"})
        assert response.status_code == 200
        assert response.json() == {"value": value}
    assert (case.path.read_bytes() if case.path.exists() else None) == before


def test_saved_key_wins_over_environment_and_absent_builtin_is_empty(reveal_settings, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-other-environment-key")
    with TestClient(reveal_settings.app) as client:
        response = client.post("/api/settings/secret", json={"key": "ANTHROPIC_API_KEY"})
        assert response.json() == {"value": reveal_settings.settings["ANTHROPIC_API_KEY"]}
        response = client.post("/api/settings/secret", json={"key": "OPENAI_API_KEY"})
        assert response.json() == {"value": ""}


@pytest.mark.parametrize("body", [
    None, [], "ANTHROPIC_API_KEY", {}, {"key": ""}, {"key": "  "}, {"key": 7},
    {"mcp_server_id": False}, {"mcp_server_id": ""},
    {"key": "ANTHROPIC_API_KEY", "mcp_server_id": "first-server"},
    {"key": "ANTHROPIC_API_KEY", "extra": True},
    {"connection": "openrouter@claude-credits"}, {"connection": "", "field": "api_key"},
    {"connection": "openrouter@claude-credits", "field": 7},
    {"connection": "openrouter@claude-credits", "field": "api_key", "key": "OPENROUTER_API_KEY"},
])
def test_invalid_selector_is_refused_without_value(reveal_settings, body):
    with TestClient(reveal_settings.app) as client:
        response = client.post("/api/settings/secret", json=body)
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_secret_selector"
    assert "value" not in response.json()
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("body", [
    {"key": "TOTAL_BUDGET"}, {"key": "MCP_SERVERS"}, {"key": "OUROBOROS_UNLISTED_SECRET"},
    {"key": "UNSAVED_CUSTOM_TOKEN"}, {"key": "../settings.json"},
    {"mcp_server_id": "missing-server"}, {"mcp_server_id": "!!!"},
])
def test_nonsecret_or_missing_identity_is_not_revealable(reveal_settings, monkeypatch, body):
    monkeypatch.setenv("UNSAVED_CUSTOM_TOKEN", "unrelated-process-secret-fixture")
    with TestClient(reveal_settings.app) as client:
        response = client.post("/api/settings/secret", json=body)
    assert response.status_code == 404
    assert response.json()["code"] == "secret_not_found"
    assert "value" not in response.json()
    assert "unrelated-process-secret-fixture" not in response.text
    assert response.headers["cache-control"] == "no-store"


_ADDED_KEY = "sk-or-v1-synthetic-added-key-complete-tail"


def _save_added_keys(case):
    case.settings["OUROBOROS_CONNECTIONS"] = [
        {"kind": "openrouter", "name": "claude-credits", "label": "Claude credits", "api_key": _ADDED_KEY},
        {"kind": "openai-compatible", "name": "local-vllm", "label": "Local vLLM",
         "base_url": "http://127.0.0.1:8000/v1", "api_key": ""},
    ]
    case.path.write_text(json.dumps(case.settings), encoding="utf-8")


def test_added_api_key_is_read_by_its_connection_and_secret_field(reveal_settings):
    case = reveal_settings
    _save_added_keys(case)
    before = case.path.read_bytes()
    with TestClient(case.app) as client:
        assert _ADDED_KEY not in client.get("/api/settings").text
        shown = client.post("/api/settings/secret", json={"connection": "openrouter@claude-credits", "field": "api_key"})
        assert shown.status_code == 200
        assert shown.json() == {"value": _ADDED_KEY}
        assert shown.headers["cache-control"] == "no-store"
        keyless = client.post("/api/settings/secret",
                              json={"connection": "openai-compatible@local-vllm", "field": "api_key"})
        assert keyless.json() == {"value": ""}
        assert _ADDED_KEY not in client.get("/api/settings").text
    assert case.path.read_bytes() == before


@pytest.mark.parametrize("body", [
    {"connection": "openrouter@missing", "field": "api_key"},
    {"connection": "openrouter", "field": "api_key"},  # the built-in key is read by its ``key``
    {"connection": "openrouter@default", "field": "api_key"},
    {"connection": "anthropic@claude-credits", "field": "api_key"},  # the name belongs to another kind
    {"connection": "openai-compatible@local-vllm", "field": "base_url"},  # not a secret field
    {"connection": "openrouter@claude-credits", "field": "label"},
    {"connection": "claudexor@claude-credits", "field": "api_key"},
    {"connection": "openrouter@Claude Credits", "field": "api_key"},
])
def test_added_api_key_selector_reads_only_a_saved_secret_field(reveal_settings, body):
    _save_added_keys(reveal_settings)
    with TestClient(reveal_settings.app) as client:
        response = client.post("/api/settings/secret", json=body)
    assert response.status_code == 404
    assert response.json()["code"] == "secret_not_found"
    assert _ADDED_KEY not in response.text
    assert response.headers["cache-control"] == "no-store"


def test_malformed_json_is_a_typed_refusal(reveal_settings):
    with TestClient(reveal_settings.app) as client:
        response = client.post("/api/settings/secret", content="{", headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_secret_selector"


@pytest.mark.parametrize("identity_field", ["id", "slug", "name"])
def test_mcp_reveal_uses_saved_canonical_identity_not_array_position(reveal_settings, identity_field):
    case = reveal_settings
    value = "Bearer complete-selected-mcp-token"
    case.settings["MCP_SERVERS"].insert(0, {identity_field: "Chosen server", "auth_token": value})
    case.path.write_text(json.dumps(case.settings), encoding="utf-8")
    with TestClient(case.app) as client:
        first = client.post("/api/settings/secret", json={"mcp_server_id": "chosen_server"})
        assert first.status_code == 200
        assert first.json() == {"value": value}
        case.settings["MCP_SERVERS"].reverse()
        case.path.write_text(json.dumps(case.settings), encoding="utf-8")
        before = case.path.read_bytes()
        second = client.post("/api/settings/secret", json={"mcp_server_id": "chosen_server"})
        assert second.json() == first.json()
        assert second.headers["cache-control"] == "no-store"
    assert case.path.read_bytes() == before


def test_mcp_duplicate_identity_refuses_even_when_tokens_are_equal(reveal_settings):
    case = reveal_settings
    case.settings["MCP_SERVERS"] = [
        {"id": "same-server", "auth_token": "same-synthetic-token"},
        {"name": "same_server", "auth_token": "same-synthetic-token"},
    ]
    case.path.write_text(json.dumps(case.settings), encoding="utf-8")
    with TestClient(case.app) as client:
        response = client.post("/api/settings/secret", json={"mcp_server_id": "same-server"})
    assert response.status_code == 409
    assert response.json()["code"] == "MCP_ID_AMBIGUOUS_SECRET"
    assert "same-synthetic-token" not in response.text
    assert response.headers["cache-control"] == "no-store"


def test_mcp_missing_token_is_empty_and_invalid_saved_rows_are_not_identities(reveal_settings):
    case = reveal_settings
    case.settings["MCP_SERVERS"] = [None, "not-a-server", {"auth_token": "unbound-token"}, {"id": "empty"}]
    case.path.write_text(json.dumps(case.settings), encoding="utf-8")
    with TestClient(case.app) as client:
        assert client.post("/api/settings/secret", json={"mcp_server_id": "!!!"}).status_code == 404
        response = client.post("/api/settings/secret", json={"mcp_server_id": "empty"})
        assert response.json() == {"value": ""}


def test_mounted_reveal_inherits_network_password_gate_and_loopback_access(reveal_settings, monkeypatch):
    from ouroboros import server_auth

    case = reveal_settings
    case.settings["OUROBOROS_NETWORK_PASSWORD"] = "synthetic-network-password"
    case.path.write_text(json.dumps(case.settings), encoding="utf-8")
    monkeypatch.setattr(server_auth, "_auth_secret_cache", b"synthetic-session-signing-key")
    selector = {"key": "ANTHROPIC_API_KEY"}
    expected = {"value": case.settings["ANTHROPIC_API_KEY"]}
    with TestClient(case.app, client=("192.0.2.10", 50000)) as remote:
        refused = remote.post("/api/settings/secret", json=selector)
        assert refused.status_code == 401
        assert case.settings["ANTHROPIC_API_KEY"] not in refused.text
        allowed = remote.post("/api/settings/secret", json=selector,
                              headers={"x-ouroboros-password": "synthetic-network-password"})
        assert allowed.status_code == 200 and allowed.json() == expected
        login = remote.post("/auth/login", json={"password": "synthetic-network-password"})
        assert login.status_code == 200
        assert remote.post("/api/settings/secret", json=selector).json() == expected
    with TestClient(case.app, client=("127.0.0.1", 50000)) as local:
        assert local.post("/api/settings/secret", json=selector).json() == expected
