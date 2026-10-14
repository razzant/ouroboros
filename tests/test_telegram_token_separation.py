"""Host-owned token equality fence; neither secret crosses to the other skill."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from ouroboros.telegram_token_separation import (
    OWNER_KEY,
    PRESENCE_KEY,
    TokenSeparationError,
    api_bot_token_status,
    current_status,
    guard_settings_write,
    status,
)


def test_equal_token_refused_on_both_settings_rotation_directions():
    owner, second = "12345678:owner-private", "87654321:second-private"
    current = {OWNER_KEY: owner, PRESENCE_KEY: second}
    guard_settings_write(current)
    assert status(current) == "distinct"
    for mutation in ({**current, OWNER_KEY: second}, {**current, PRESENCE_KEY: owner}):
        with pytest.raises(TokenSeparationError, match="nothing was saved"):
            guard_settings_write(mutation)
    with pytest.raises(TokenSeparationError):
        guard_settings_write({PRESENCE_KEY: second})
    guard_settings_write({OWNER_KEY: "", PRESENCE_KEY: ""})


def test_current_check_requires_exact_running_candidate_after_rotation(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    owner, second = "12345678:owner-private", "87654321:second-private"
    path.write_text(json.dumps({OWNER_KEY: owner, PRESENCE_KEY: second}), encoding="utf-8")
    assert current_status(path, candidate=second) == "distinct"
    assert current_status(path, candidate="old-second") == "unavailable"
    assert current_status(path, candidate=owner) == "unavailable"
    path.write_text(json.dumps({OWNER_KEY: second, PRESENCE_KEY: second}), encoding="utf-8")
    assert current_status(path, candidate=second) == "collision"
    path.write_text("broken-json", encoding="utf-8")
    assert current_status(path, candidate=second) == "unavailable"


def test_host_check_discloses_only_status_and_refuses_foreign_skill(tmp_path, monkeypatch):
    from ouroboros.gateway import host_service

    owner, second = "12345678:owner-private", "87654321:second-private"
    (tmp_path / "settings.json").write_text(json.dumps({OWNER_KEY: owner, PRESENCE_KEY: second}), encoding="utf-8")

    async def authenticate(_ctx, _token, _permission):
        return "telegram-presence-bot", {"content_hash": "test"}

    async def read(_ctx, fn):
        return fn()

    monkeypatch.setattr(host_service, "_authenticated", authenticate)
    monkeypatch.setattr(host_service, "_bounded_host_read", read)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        host_service_context=SimpleNamespace(data_dir=tmp_path))),
        headers={"x-skill-token": "opaque-host", "x-bot-token": second})
    response = asyncio.run(api_bot_token_status(request))
    assert response.status_code == 200
    assert json.loads(response.body) == {"ok": True, "status": "distinct"}
    assert owner.encode() not in response.body and second.encode() not in response.body
    request.headers["x-bot-token"] = owner
    assert asyncio.run(api_bot_token_status(request)).status_code == 409

    async def foreign(_ctx, _token, _permission):
        return "telegram", {}

    monkeypatch.setattr(host_service, "_authenticated", foreign)
    assert asyncio.run(api_bot_token_status(request)).status_code == 403


def test_real_host_auth_and_token_rotation(tmp_path, monkeypatch):
    from starlette.testclient import TestClient

    from ouroboros.gateway.host_service import create_host_service_app
    from tests.test_host_service_api import _seed_token

    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", "")
    _seed_token(tmp_path, skill="telegram-presence-bot", token="host-secret",
                permissions=["presence"], manifest_permissions=["presence"])
    owner, second = "12345678:owner-private", "87654321:second-private"
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({OWNER_KEY: owner, PRESENCE_KEY: second}), encoding="utf-8")
    client = TestClient(create_host_service_app(tmp_path))
    url = "/telegram/presence-bot-token"
    headers = {"x-skill-token": "host-secret", "x-bot-token": second}
    response = client.get(url, headers=headers)
    assert response.status_code == 200 and response.json() == {"ok": True, "status": "distinct"}
    settings.write_text(json.dumps({OWNER_KEY: second, PRESENCE_KEY: second}), encoding="utf-8")
    assert client.get(url, headers=headers).json() == {"ok": False, "status": "collision"}
    settings.write_text(json.dumps({OWNER_KEY: owner, PRESENCE_KEY: "rotated"}), encoding="utf-8")
    assert client.get(url, headers=headers).json() == {"ok": False, "status": "unavailable"}
    assert client.get(url, headers={"x-skill-token": "wrong", "x-bot-token": second}).status_code == 403


def test_real_settings_writer_refuses_both_token_rotation_directions(tmp_path, monkeypatch):
    from ouroboros import config

    path = tmp_path / "settings.json"
    owner, second = "12345678:owner-private", "87654321:second-private"
    original = {"OUROBOROS_RUNTIME_MODE": "advanced", "OUROBOROS_CONTEXT_MODE": "max",
                "OUROBOROS_SAFETY_MODE": "full", OWNER_KEY: owner, PRESENCE_KEY: second}
    path.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "_guard_live_settings_write", lambda: None)
    monkeypatch.setattr(config, "get_runtime_mode", lambda: "advanced")
    for mutation in ({**original, OWNER_KEY: second}, {**original, PRESENCE_KEY: owner}):
        with pytest.raises(TokenSeparationError):
            config.save_settings(mutation)
        assert json.loads(path.read_text(encoding="utf-8")) == original
