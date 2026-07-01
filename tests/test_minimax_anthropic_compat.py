"""MiniMax Token Plan via Anthropic-compatible base URL."""

from __future__ import annotations

from pathlib import Path


def test_resolve_anthropic_minimax_base_url_from_env(monkeypatch):
    from ouroboros.llm import LLMClient

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-token")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", " https://api.minimax.io/anthropic/ ")

    target = LLMClient()._resolve_remote_target("anthropic::MiniMax-M3")

    assert target["provider"] == "anthropic"
    assert target["resolved_model"] == "MiniMax-M3"
    assert target["base_url"] == "https://api.minimax.io/anthropic/v1"
    assert target["auth_scheme"] == "bearer"


def test_anthropic_minimax_chat_uses_v1_messages_and_bearer_auth(monkeypatch):
    from ouroboros.llm import LLMClient

    captured = {}

    class _Response:
        status_code = 200
        reason = "OK"
        url = "https://api.minimax.io/anthropic/v1/messages"
        text = ""

        def json(self):
            return {
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "stop_reason": "end_turn",
            }

    def fake_post(url, *, headers, json, timeout):
        captured.update({"url": url, "headers": dict(headers), "json": dict(json), "timeout": timeout})
        return _Response()

    monkeypatch.setattr("requests.post", fake_post)
    client = LLMClient()
    target = {
        "provider": "anthropic",
        "resolved_model": "MiniMax-M3",
        "usage_model": "anthropic/MiniMax-M3",
        "api_key": "test-token",
        "base_url": "https://api.minimax.io/anthropic/v1",
        "auth_scheme": "bearer",
    }

    message, usage = client._chat_anthropic(
        target,
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        reasoning_effort="medium",
        max_tokens=8,
        tool_choice="auto",
    )

    assert captured["url"] == "https://api.minimax.io/anthropic/v1/messages"
    assert captured["headers"]["Authorization"] == "Bearer test-token"
    assert "x-api-key" not in captured["headers"]
    assert captured["json"]["model"] == "MiniMax-M3"
    assert message["content"] == "ok"
    assert usage["resolved_model"] == "anthropic/MiniMax-M3"


def test_official_anthropic_route_remains_v1_x_api_key(monkeypatch):
    from ouroboros.llm import LLMClient

    monkeypatch.setenv("ANTHROPIC_API_KEY", "official-token")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    target = LLMClient()._resolve_remote_target("anthropic::claude-opus-4-8")

    assert target["base_url"] == "https://api.anthropic.com/v1"
    assert target["auth_scheme"] == "x-api-key"


def test_active_main_route_fingerprints_anthropic_base_url():
    from ouroboros.gateway.settings import _active_main_route

    route = _active_main_route({
        "OUROBOROS_MODEL": "anthropic::MiniMax-M3",
        "ANTHROPIC_BASE_URL": "https://api.minimax.io/anthropic",
    })

    assert route["provider"] == "anthropic"
    assert route["base_url"] == "https://api.minimax.io/anthropic/v1"


def test_anthropic_base_url_is_non_secret_setting():
    from ouroboros.config import SETTINGS_DEFAULTS
    from ouroboros.gateway import settings as settings_api

    assert "ANTHROPIC_BASE_URL" in SETTINGS_DEFAULTS
    assert "ANTHROPIC_BASE_URL" not in settings_api._SECRET_SETTING_KEYS


def test_minimax_model_catalog_does_not_call_official_anthropic():
    from ouroboros.gateway.models import _provider_specs

    specs = dict(_provider_specs({
        "ANTHROPIC_API_KEY": "minimax-token",
        "ANTHROPIC_BASE_URL": "https://api.minimax.io/anthropic",
    }))

    assert "anthropic" in specs

    class _Client:
        async def get(self, *args, **kwargs):  # pragma: no cover - fail path
            raise AssertionError("MiniMax catalog must not query official Anthropic")

    import asyncio

    async def _load():
        return await specs["anthropic"](_Client())  # type: ignore[arg-type]

    items = asyncio.run(_load())
    assert items[0]["value"] == "anthropic::MiniMax-M3"
    assert items[0]["provider"] == "MiniMax Token Plan"


def test_apply_settings_to_env_derives_minimax_claude_tooling_env(monkeypatch):
    from ouroboros.config import SETTINGS_DEFAULTS, apply_settings_to_env

    for key in (
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
        "CLAUDE_CODE_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = dict(SETTINGS_DEFAULTS)
    settings.update({
        "ANTHROPIC_API_KEY": "minimax-token",
        "ANTHROPIC_BASE_URL": "https://api.minimax.io/anthropic",
    })

    apply_settings_to_env(settings)

    import os

    assert os.environ["ANTHROPIC_BASE_URL"] == "https://api.minimax.io/anthropic"
    assert os.environ["ANTHROPIC_AUTH_TOKEN"] == "minimax-token"
    assert os.environ["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "700000"
    assert os.environ["CLAUDE_CODE_MODEL"] == "MiniMax-M3[1m]"
    assert os.environ["ANTHROPIC_MODEL"] == "MiniMax-M3[1m]"
    assert os.environ["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "MiniMax-M3[1m]"
    assert os.environ["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "MiniMax-M3[1m]"
    assert os.environ["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "MiniMax-M3[1m]"


def test_minimax_claude_tooling_allows_explicit_compact_override(monkeypatch):
    from ouroboros.config import SETTINGS_DEFAULTS, apply_settings_to_env
    import os

    monkeypatch.delenv("CLAUDE_CODE_AUTO_COMPACT_WINDOW", raising=False)
    settings = dict(SETTINGS_DEFAULTS)
    settings.update({
        "ANTHROPIC_API_KEY": "minimax-token",
        "ANTHROPIC_BASE_URL": "https://api.minimaxi.com/anthropic",
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "1000000",
    })

    apply_settings_to_env(settings)

    assert os.environ["ANTHROPIC_BASE_URL"] == "https://api.minimaxi.com/anthropic"
    assert os.environ["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "1000000"


def test_settings_ui_documents_minimax_anthropic_values():
    root = Path(__file__).resolve().parents[1]
    settings_ui = (root / "web" / "modules" / "settings_ui.js").read_text(encoding="utf-8")
    settings_js = (root / "web" / "modules" / "settings.js").read_text(encoding="utf-8")

    assert "s-anthropic-base-url" in settings_ui
    assert "ANTHROPIC_BASE_URL" in settings_js
    assert "https://api.minimax.io/anthropic" in settings_ui
    assert "https://api.minimaxi.com/anthropic" in settings_ui
    assert "anthropic::MiniMax-M3" in settings_ui
    assert "MiniMax-M3[1m]" in settings_ui
    assert "700000" in settings_ui
    assert "1000000" in settings_ui
    assert "ANTHROPIC_AUTH_TOKEN" in settings_ui
