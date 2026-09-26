import asyncio
from copy import deepcopy
import inspect
import json

import httpx
import pytest
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


def test_direct_api_catalog_loaders_keep_the_native_async_transport():
    # The owned Claudexor gateway is synchronous and is explicitly offloaded
    # at the endpoint boundary. Direct API fetchers must remain natively async.
    source = "\n".join(
        inspect.getsource(obj)
        for obj in (
            model_catalog_api._provider_specs,
            model_catalog_api._load_provider,
            model_catalog_api._fetch_openai_compatible_model_catalog,
            model_catalog_api._fetch_openrouter_model_catalog,
        )
    )
    assert "import requests" not in source
    assert "asyncio.to_thread" not in source


def _account_operation(path):
    return {"method": "GET", "path": path,
            "parameters": [{"name": "view", "location": "query", "enum": ["accounts"]}]}


@pytest.fixture
def account_engine(monkeypatch):
    from ouroboros import claudexor_daemon as owned

    problem = {"code": "subscription_window_exhausted", "message": "Window is spent", "context": {}}
    accounts = []
    for profile, window, effort, modes, availability in (
        ("a", 272000, "medium", ["standard"], "available"),
        ("b", 1000000, "high", ["standard", "fast"], "unavailable"),
    ):
        catalog = {"source": "codex", "credentialProfileId": profile, "accountFingerprint": "fixture-" + profile,
            "observedAt": "2026-09-12T00:00:00Z" if profile == "a" else None, "provenance": "fixture_per_account",
            "models": [{"id": "shared", "contextWindow": window, "maxContextWindow": window,
                "reasoningEfforts": [effort], "supportedOptions": ["processingPreference"],
                "processing": {"modes": modes, "eligible": None, "source": "fixture_per_account", "observedAt": None}},
                {"id": "only-" + profile}]}
        accounts.append({"credentialProfileId": profile, "availability": availability,
                         "problem": problem if profile == "b" else None, "catalog": catalog})
    accounts.append({"credentialProfileId": "c", "availability": "unknown", "catalog": None,
        "problem": {"code": "model_catalog_unavailable", "message": "Could not read account", "context": {}}})

    class Gateway:
        view = True
        operations_error = False
        calls = []
        sources = [{"id": "codex", "label": "Codex", "credentialHarness": "codex"}]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def operations(self):
            if self.operations_error:
                raise RuntimeError("operation catalog unavailable")
            return [_account_operation(path) for path in ("/v2/model-sources", "/v2/model-sources/:id/models")] if self.view else []

        def list_model_sources(self, **kwargs):
            assert kwargs == ({"view": "accounts"} if self.view and not self.operations_error else {})
            sources = deepcopy(self.sources)
            if kwargs:
                sources = [{**row, "processingPreferences": ["standard", "fast"], "accountCatalog": True} for row in sources]
            return {"sources": sources}

        def list_source_models(self, source, credential_profile_id=None, **kwargs):
            self.calls.append((source, credential_profile_id, kwargs))
            assert kwargs == ({"view": "accounts"} if self.view and not self.operations_error else {})
            if source == "broken":
                raise RuntimeError("source catalog unavailable")
            if not kwargs:
                return deepcopy(accounts[0]["catalog"])
            selected = [row for row in accounts if not credential_profile_id or row["credentialProfileId"] == credential_profile_id]
            return {"source": source, "accounts": deepcopy(selected), "partial": any(row["catalog"] is None for row in selected)}

    gateway = Gateway()
    monkeypatch.setattr(owned, "owned_daemon_provisioned", lambda: True)
    monkeypatch.setattr(owned, "read_owned_gateway", lambda: gateway)
    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: {})
    return gateway, accounts


def test_auto_catalog_keeps_every_account_and_independent_capabilities(account_engine):
    gateway, accounts = account_engine
    payload = json.loads(asyncio.run(model_catalog_api.api_model_catalog(None)).body)
    shared = [row for row in payload["items"] if row["id"] == "shared"]
    assert [(row["credential_profile_id"], row["context_window"], row["reasoning_efforts"], row["processing"]["modes"])
            for row in shared] == [("a", 272000, ["medium"], ["standard"]), ("b", 1000000, ["high"], ["standard", "fast"])]
    assert {row["id"] for row in payload["items"]} == {"shared", "only-a", "only-b"}
    assert shared[1]["availability"] == "unavailable" and shared[1]["observed_at"] is None
    assert shared[1]["problem"]["code"] == "subscription_window_exhausted"
    assert shared[1]["processing"]["eligible"] is None
    assert payload["account_catalogs"] == [{"source": "codex", "accounts": accounts, "partial": True}]
    assert payload["account_catalogs"][0]["accounts"][1]["problem"]["code"] == "subscription_window_exhausted"
    assert payload["partial"] is True
    assert payload["model_sources"][0]["processingPreferences"] == ["standard", "fast"]
    # A spent window on a READABLE account is account state, not a catalog read failure.
    assert [(row["credential_profile_id"], row["code"]) for row in payload["errors"]] == [
        ("c", "model_catalog_unavailable")]
    assert gateway.calls == [("codex", None, {"view": "accounts"})]


@pytest.mark.parametrize("profile", ["a", "b", "c"])
def test_pinned_account_catalog_does_not_include_sibling_models(account_engine, profile):
    gateway, _ = account_engine
    payload = model_catalog_api._subscription_model_catalog("codex", profile)
    assert {row["credential_profile_id"] for row in payload["items"]} == ({profile} if profile != "c" else set())
    assert [row["credentialProfileId"] for row in payload["account_catalogs"][0]["accounts"]] == [profile]
    assert gateway.calls == [("codex", profile, {"view": "accounts"})]


@pytest.mark.parametrize("operations_error", [False, True])
def test_legacy_or_unreadable_operation_catalog_keeps_strict_old_queries(account_engine, operations_error):
    gateway, _ = account_engine
    gateway.view = operations_error
    gateway.operations_error = operations_error
    payload = model_catalog_api._subscription_model_catalog("codex")
    assert payload["errors"] == [] and len(payload["items"]) == 2
    assert set(payload) == {"items", "errors", "model_sources"}
    assert gateway.calls == [("codex", None, {})]


def test_one_failed_source_preserves_other_account_catalogs(account_engine):
    gateway, _ = account_engine
    gateway.sources.insert(0, {"id": "broken"})
    payload = model_catalog_api._subscription_model_catalog()
    assert len(payload["items"]) == 4 and payload["partial"] is True
    assert payload["errors"][0]["source_id"] == "broken"
    assert payload["account_catalogs"][0]["source"] == "codex"


@pytest.mark.parametrize("change", [{"method": "POST"}, {"path": "/wrong"},
    {"parameters": [{"name": "view", "location": "header", "enum": ["accounts"]}]},
    {"parameters": [{"name": "view", "location": "query", "enum": ["selected"]}]}])
def test_account_view_requires_the_exact_declared_query(change):
    path = "/v2/model-sources/:id/models"
    assert not model_catalog_api.account_catalog_supported([{**_account_operation(path), **change}], path)


def test_model_catalog_reports_a_bad_extra_ca_bundle_instead_of_a_bare_500(monkeypatch):
    """An unreadable OUROBOROS_EXTRA_CA_BUNDLE is named in the catalog errors; the engine
    catalog still loads and the endpoint answers 200."""
    from ouroboros.net_transport import ExtraCaBundleError

    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: {"OPENAI_API_KEY": "openai-key"})
    monkeypatch.setattr(model_catalog_api, "_subscription_model_catalog",
                        lambda source_id, profile_id: {"items": [{"value": "engine-model"}], "errors": []})

    def _boom():
        raise ExtraCaBundleError("OUROBOROS_EXTRA_CA_BUNDLE is not readable: /nope.pem")

    monkeypatch.setattr(model_catalog_api, "verify_kwargs", _boom)
    response = asyncio.run(model_catalog_api.api_model_catalog(None))
    assert response.status_code == 200
    payload = json.loads(response.body.decode("utf-8"))
    assert [item["value"] for item in payload["items"]] == ["engine-model"]
    trust = [row for row in payload["errors"] if row.get("provider_id") == "extra_ca_bundle"]
    assert trust and "not readable" in trust[0]["error"] and trust[0]["stage"] == "trust"
