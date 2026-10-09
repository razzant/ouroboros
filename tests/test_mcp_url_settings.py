"""Editable MCP URL secrets survive Settings and Test without new credentials."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.gateway.settings import _mask_mcp_servers_payload, _rehydrate_mcp_servers_payload
from ouroboros.secret_masking import mask_mcp_url, rehydrate_mcp_url
from tests.test_settings_secret_mask import settings_client  # noqa: F401

URL = "https://fixture-user:fixture-password@service.example:9443/mcp?route=one#fragment"


def server(**changes):
    return {"id": "Demo Server!", "name": "Demo", "transport": "streamable_http", "enabled": True,
            "url": URL, "auth_token": "Bearer fixture-header-token", **changes}


@pytest.mark.parametrize("url", [URL, "https://user-only@service.example/mcp",
                                  "https://u%40name:p%3Ass@[::1]:9443/a?b=#c", "HTTPS://u:p@EXAMPLE.test/mcp?"])
def test_mask_roundtrip_preserves_exact_original_url(url):
    masked = mask_mcp_url(url)
    assert "***@" in masked and masked != url
    assert rehydrate_mcp_url(masked, url) == url
    assert mask_mcp_url(masked) == masked


def test_masked_settings_roundtrip_changes_name_without_losing_url_or_header():
    current = [server()]
    before = copy.deepcopy(current)
    incoming = _mask_mcp_servers_payload(current)
    assert "fixture-user" not in str(incoming) and "fixture-password" not in str(incoming)
    incoming[0]["name"] = "Edited label"
    restored = _rehydrate_mcp_servers_payload(incoming, current)[0]
    assert restored["url"] == URL and restored["auth_token"] == current[0]["auth_token"]
    assert restored["name"] == "Edited label" and "auth_configured" not in restored
    assert current == before


@pytest.mark.parametrize("change", [
    {"url": "https://***@different.example:9443/mcp?route=one#fragment"},
    {"url": "https://***@service.example:9443/changed?route=one#fragment"},
    {"url": "http://***@service.example:9443/mcp?route=one#fragment"},
    {"url": "https://***@service.example:9443/mcp?route=two#fragment"},
    {"url": "https://***@service.example:9443/mcp?route=one#new"},
    {"id": "another-server"},
])
def test_changed_server_or_target_never_receives_saved_userinfo(change):
    masked = _mask_mcp_servers_payload([server()])[0]
    masked.update(change)
    restored = _rehydrate_mcp_servers_payload([masked], [server()])[0]
    assert "@" not in restored["url"]
    assert "fixture-user" not in restored["url"] and "fixture-password" not in restored["url"]
    assert "***" not in restored["url"]


@pytest.mark.parametrize("replacement", ["https://service.example:9443/mcp?route=one#fragment",
                                          "https://new-user:new-password@other.example/mcp", ""])
def test_explicit_removal_replacement_and_clear_are_kept(replacement):
    incoming = _mask_mcp_servers_payload([server()])[0]
    incoming["url"] = replacement
    restored = _rehydrate_mcp_servers_payload([incoming], [server()])[0]
    assert restored["url"] == replacement
    assert restored["auth_token"] == server()["auth_token"]


def test_mask_without_current_server_cannot_be_saved_as_credentials():
    incoming = _mask_mcp_servers_payload([server()])
    restored = _rehydrate_mcp_servers_payload(incoming, [])[0]
    assert restored["url"] == URL.split("@", 1)[-1].join(["https://", ""])
    assert "***" not in restored["url"]


def test_invalid_saved_authority_masks_secrets_and_can_still_roundtrip_or_be_replaced():
    malformed = "https://private-user:private-password@[broken/mcp"
    masked = mask_mcp_url(malformed)
    assert "private" not in masked
    assert rehydrate_mcp_url(masked, malformed) == malformed
    assert rehydrate_mcp_url("https://fixed.example/mcp", malformed) == "https://fixed.example/mcp"


def test_real_settings_get_post_uses_url_roundtrip(settings_client, monkeypatch):  # noqa: F811
    from ouroboros import mcp_client
    monkeypatch.setattr(mcp_client, "refresh_all_background", lambda **kwargs: None)
    client, on_disk, saved = settings_client
    on_disk.update(MCP_ENABLED=False, MCP_SERVERS=[server()])
    received = client.get("/api/settings")
    assert received.status_code == 200
    assert "fixture-user" not in received.text and "fixture-password" not in received.text
    candidate = received.json()["MCP_SERVERS"][0]
    candidate["name"] = "Changed title"
    response = client.post("/api/settings", json={"MCP_SERVERS": [candidate]})
    assert response.status_code == 200, response.text
    assert saved["MCP_SERVERS"][0]["url"] == URL
    assert saved["MCP_SERVERS"][0]["name"] == "Changed title"
    assert "fixture-user" not in response.text and "fixture-password" not in response.text


@pytest.fixture
def test_candidate_client(monkeypatch):
    from ouroboros.gateway import mcp
    current = server()
    seen = []
    def test(candidate, *, settings):
        seen.append(copy.deepcopy(candidate))
        return {"ok": True, "tool_count": 0}
    monkeypatch.setattr(mcp, "_ensure_configured", lambda: None)
    monkeypatch.setattr(mcp, "load_settings", lambda: {"MCP_SERVERS": [copy.deepcopy(current)]})
    monkeypatch.setattr(mcp, "get_manager", lambda: SimpleNamespace(test_server=test))
    with TestClient(Starlette(routes=[Route("/test", mcp.api_mcp_test, methods=["POST"])])) as client:
        yield client, current, seen


@pytest.mark.parametrize("change", [{}, {"name": "Edited"}, {"url": "https://***@different.example/mcp"},
                                    {"id": "other"}, {"url": "https://service.example/no-auth"}])
def test_saved_test_candidate_uses_same_scoped_url_rehydration(test_candidate_client, change):
    client, current, seen = test_candidate_client
    candidate = _mask_mcp_servers_payload([current])[0]
    candidate.update(change)
    response = client.post("/test", json={"server_id": current["id"], "server": candidate})
    assert response.status_code == 200 and response.json()["ok"]
    expected = _rehydrate_mcp_servers_payload([candidate], [current])[0]
    assert seen[-1]["url"] == expected["url"]
    # Token and URL masks use the same unique candidate-identity restoration.
    assert seen[-1]["auth_token"] == expected["auth_token"]
    assert "fixture-user" not in response.text and "fixture-password" not in response.text


def test_test_candidate_can_explicitly_omit_header_and_url_credentials(test_candidate_client):
    client, current, seen = test_candidate_client
    candidate = {"id": current["id"], "transport": "streamable_http", "url": "https://service.example/mcp"}
    assert client.post("/test", json={"server_id": current["id"], "server": candidate}).json()["ok"]
    assert "auth_token" not in seen[-1] and seen[-1]["url"] == candidate["url"]


def test_unsaved_test_candidate_cannot_recover_another_saved_url(test_candidate_client):
    client, current, seen = test_candidate_client
    candidate = _mask_mcp_servers_payload([current])[0]
    response = client.post("/test", json={"server": candidate})
    assert response.json()["ok"]
    assert "@" not in seen[-1]["url"]
    assert seen[-1]["auth_token"] == ""  # no saved identity: never send a display mask
