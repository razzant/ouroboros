"""Owner context intent is independent of Main-route context-window evidence."""

from __future__ import annotations

import json
import os

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient


def test_owner_can_enable_max_without_main_route_capability_ack(tmp_path, monkeypatch):
    """Max is an owner projection preference; exact-route fitting happens at dispatch."""
    import ouroboros.capability_evidence as ce
    from ouroboros import config as cfg
    from ouroboros.gateway import settings as gateway_settings

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({
        "OUROBOROS_CONTEXT_MODE": "low",
        "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "false",
        "OUROBOROS_MODEL": "openai/gpt-4o-mini",
    }), encoding="utf-8")
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    os.environ["OUROBOROS_CONTEXT_MODE"] = "low"
    os.environ["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] = "false"

    def unexpected_probe(*_args, **_kwargs):
        raise AssertionError("owner Max must not perform a Main-route capability probe")

    monkeypatch.setattr(ce, "probe", unexpected_probe)
    monkeypatch.setattr(gateway_settings, "_has_running_agent_tasks", lambda: False)

    app = Starlette(routes=[
        Route(
            "/api/owner/context-mode",
            endpoint=gateway_settings.api_owner_context_mode,
            methods=["POST"],
        ),
    ])
    app.state.drive_root = tmp_path
    response = TestClient(app).post("/api/owner/context-mode", json={"mode": "max"})

    assert response.status_code == 200, response.text
    assert response.json() == {"ok": True, "context_mode": "max"}
    stored = json.loads(settings_path.read_text(encoding="utf-8"))
    assert stored["OUROBOROS_CONTEXT_MODE"] == "max"
    assert stored["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"
    assert os.environ["OUROBOROS_CONTEXT_MODE"] == "max"
    assert os.environ["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"


def test_bare_env_low_cannot_author_owner_low_while_busy(tmp_path, monkeypatch):
    """Sizing-only env Low must not bypass the owner-intent idle guard."""
    from ouroboros import config as cfg
    from ouroboros.gateway import settings as gateway_settings

    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    os.environ["OUROBOROS_CONTEXT_MODE"] = "low"
    os.environ.pop("OUROBOROS_CONTEXT_MODE_AUTO_LOW", None)
    monkeypatch.setattr(gateway_settings, "_has_running_agent_tasks", lambda: True)

    assert cfg.get_context_mode() == "low"
    assert cfg.get_owner_context_mode() == "max"

    app = Starlette(routes=[
        Route(
            "/api/owner/context-mode",
            endpoint=gateway_settings.api_owner_context_mode,
            methods=["POST"],
        ),
    ])
    app.state.drive_root = tmp_path
    response = TestClient(app).post("/api/owner/context-mode", json={"mode": "low"})

    assert response.status_code == 409, response.text
    assert not settings_path.exists()
    assert cfg.get_owner_context_mode() == "max"


def test_main_route_resolution_remains_available_to_dispatch_fit():
    """Removing settings-time gating must not remove context_fit's route resolver."""
    from ouroboros.gateway import settings as gateway_settings

    route = gateway_settings._active_main_route({
        "OUROBOROS_MODEL": "minimax::MiniMax-M3",
        "MINIMAX_REGION": "cn_zh",
    })

    assert route == {
        "provider": "minimax",
        "model": "minimax::MiniMax-M3",
        "base_url": "https://api.minimaxi.com/v1",
        "use_local": False,
    }
