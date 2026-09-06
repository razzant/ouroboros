import asyncio
import inspect
import json

import httpx
import ouroboros.gateway.models as model_catalog_api


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_model_catalog_tags_provider_values(monkeypatch):
    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: {
        "OPENROUTER_API_KEY": "or-key",
        "OPENAI_API_KEY": "openai-key",
        "ANTHROPIC_API_KEY": "anthropic-key",
        "OPENAI_COMPATIBLE_API_KEY": "compat-key",
        "OPENAI_COMPATIBLE_BASE_URL": "https://compat.example/v1",
        "CLOUDRU_FOUNDATION_MODELS_API_KEY": "cloudru-key",
        "GIGACHAT_CREDENTIALS": "giga-creds",
        "MINIMAX_API_KEY": "minimax-key",
        "DEEPSEEK_API_KEY": "deepseek-key",
    })

    async def fake_openrouter(_client, _api_key):
        return [model_catalog_api._build_model_catalog_entry(
            "openrouter", "Anthropic", "anthropic/claude-sonnet-4-6", "Claude Sonnet 4.6",
            source="OpenRouter",
        )]

    async def fake_anthropic(_client, _api_key):
        return [model_catalog_api._build_model_catalog_entry(
            "anthropic", "Anthropic", "claude-sonnet-4-6", "Claude Sonnet 4.6"
        )]

    async def fake_compatible(_client, provider_id, provider_label, _api_key, _base_url):
        model_id = {
            "openai": "gpt-4.1",
            "openai-compatible": "compatible-pro",
            "cloudru": "cloudru-pro",
            "minimax": "MiniMax-M3",
            "deepseek": "deepseek-v4-pro",
        }[provider_id]
        return [model_catalog_api._build_model_catalog_entry(provider_id, provider_label, model_id, model_id)]

    async def fake_gigachat(_credentials, _scope, _base_url, _verify, _user="", _password=""):
        return [model_catalog_api._build_model_catalog_entry("gigachat", "GigaChat", "giga-pro", "giga-pro")]

    monkeypatch.setattr(model_catalog_api, "_fetch_openrouter_model_catalog", fake_openrouter)
    monkeypatch.setattr(model_catalog_api, "_fetch_anthropic_model_catalog", fake_anthropic)
    monkeypatch.setattr(model_catalog_api, "_fetch_openai_compatible_model_catalog", fake_compatible)
    monkeypatch.setattr(model_catalog_api, "_fetch_gigachat_model_catalog", fake_gigachat)

    response = asyncio.run(model_catalog_api.api_model_catalog(None))
    payload = json.loads(response.body.decode("utf-8"))
    items_by_value = {item["value"]: item for item in payload["items"]}
    values = set(items_by_value)

    assert "anthropic/claude-sonnet-4-6" in values
    assert "openai::gpt-4.1" in values
    assert "anthropic::claude-sonnet-4-6" in values
    assert "openai-compatible::compatible-pro" in values
    assert "cloudru::cloudru-pro" in values
    assert "gigachat::giga-pro" in values
    # MiniMax and DeepSeek ride the shared OpenAI-compatible live fetcher
    # (GET /v1/models on their fixed hosts), so the fake returns one model each.
    assert "minimax::MiniMax-M3" in values
    assert "deepseek::deepseek-v4-pro" in values
    assert payload["errors"] == []

    openrouter_item = items_by_value["anthropic/claude-sonnet-4-6"]
    direct_item = items_by_value["anthropic::claude-sonnet-4-6"]
    assert openrouter_item["source"] == "OpenRouter"
    assert openrouter_item["label"] == "OpenRouter · Claude Sonnet 4.6"
    assert direct_item["source"] == "Anthropic"
    assert direct_item["label"] == "Anthropic · Claude Sonnet 4.6"
    assert openrouter_item["label"] != direct_item["label"]
    assert all(item["label"] == f'{item["source"]} · {item["name"]}' for item in payload["items"])
    assert [
        item["value"] for item in payload["items"]
        if item["provider"] == "Anthropic" and item["name"] == "Claude Sonnet 4.6"
    ] == ["anthropic/claude-sonnet-4-6", "anthropic::claude-sonnet-4-6"]


def test_model_catalog_returns_errors_nonfatally(monkeypatch):
    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: {
        "OPENROUTER_API_KEY": "or-key",
        "ANTHROPIC_API_KEY": "anthropic-key",
        "OPENAI_COMPATIBLE_API_KEY": "compat-key",
        "OPENAI_COMPATIBLE_BASE_URL": "https://compat.example/v1",
    })

    async def fake_openrouter(_client, _api_key):
        return [model_catalog_api._build_model_catalog_entry(
            "openrouter", "Anthropic", "anthropic/claude-opus", "Claude Opus", source="OpenRouter"
        )]

    async def fake_anthropic(_client, _api_key):
        raise RuntimeError("anthropic failed")

    async def fake_compatible(_client, _provider_id, _provider_label, _api_key, _base_url):
        raise RuntimeError("catalog failed")

    monkeypatch.setattr(model_catalog_api, "_fetch_openrouter_model_catalog", fake_openrouter)
    monkeypatch.setattr(model_catalog_api, "_fetch_anthropic_model_catalog", fake_anthropic)
    monkeypatch.setattr(model_catalog_api, "_fetch_openai_compatible_model_catalog", fake_compatible)

    response = asyncio.run(model_catalog_api.api_model_catalog(None))
    payload = json.loads(response.body.decode("utf-8"))

    assert any(item["value"] == "anthropic/claude-opus" for item in payload["items"])
    assert [(error["provider_id"], error["error"], error["stage"]) for error in payload["errors"]] == [
        ("anthropic", "anthropic failed", "error"),
        ("openai-compatible", "catalog failed", "error"),
    ]
    assert all(isinstance(error.get("duration_ms"), int) for error in payload["errors"])


def test_model_catalog_registers_openai_compatible_base_url_without_key(monkeypatch):
    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: {
        "OPENAI_COMPATIBLE_BASE_URL": "https://compat.example/v1",
    })
    captured = {}

    async def fake_compatible(_client, provider_id, provider_label, api_key, base_url):
        captured.update({
            "provider_id": provider_id,
            "provider_label": provider_label,
            "api_key": api_key,
            "base_url": base_url,
        })
        return [model_catalog_api._build_model_catalog_entry(provider_id, provider_label, "local-model", "local-model")]

    monkeypatch.setattr(model_catalog_api, "_fetch_openai_compatible_model_catalog", fake_compatible)

    response = asyncio.run(model_catalog_api.api_model_catalog(None))
    payload = json.loads(response.body.decode("utf-8"))

    assert payload["items"][0]["value"] == "openai-compatible::local-model"
    assert captured == {
        "provider_id": "openai-compatible",
        "provider_label": "OpenAI Compatible",
        "api_key": "",
        "base_url": "https://compat.example/v1",
    }


def test_openai_compatible_model_fetch_omits_blank_bearer_header():
    captured = {}

    class Client:
        async def get(self, url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers")
            return _Response({"data": [{"id": "local-model"}]})

    models = asyncio.run(model_catalog_api._fetch_openai_compatible_model_catalog(
        Client(),
        "openai-compatible",
        "OpenAI Compatible",
        "",
        "https://compat.example/v1/",
    ))

    assert models[0]["value"] == "openai-compatible::local-model"
    assert captured["url"] == "https://compat.example/v1/models"
    assert captured["headers"] is None


def test_model_catalog_runs_provider_loaders_as_native_async(monkeypatch):
    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: {})
    calls = []

    async def _loader(_client):
        await asyncio.sleep(0)
        calls.append("provider")
        return [{"value": "provider::model", "label": "Provider Model"}]

    monkeypatch.setattr(model_catalog_api, "_provider_specs", lambda settings: [("provider", _loader)])

    response = asyncio.run(model_catalog_api.api_model_catalog(None))
    payload = json.loads(response.body.decode("utf-8"))

    assert payload["items"] == [{"value": "provider::model", "label": "Provider Model"}]
    assert payload["errors"] == []
    assert calls == ["provider"]


def test_model_catalog_classifies_httpx_error_stage(monkeypatch):
    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: {})

    async def _loader(_client):
        raise httpx.ConnectError("network down")

    monkeypatch.setattr(model_catalog_api, "_provider_specs", lambda settings: [("provider", _loader)])

    response = asyncio.run(model_catalog_api.api_model_catalog(None))
    payload = json.loads(response.body.decode("utf-8"))

    assert payload["items"] == []
    assert payload["errors"][0]["provider_id"] == "provider"
    assert payload["errors"][0]["stage"] == "connect"


def test_model_catalog_no_longer_uses_requests_or_to_thread():
    source = "\n".join(
        inspect.getsource(obj)
        for obj in (
            model_catalog_api.api_model_catalog,
            model_catalog_api._provider_specs,
            model_catalog_api._load_provider,
        )
    )
    assert "import requests" not in source
    assert "asyncio.to_thread" not in source
