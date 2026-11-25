import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

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

    async def fake_openrouter(_client, _api_key, _cache_ttl=""):
        return [model_catalog_api._build_model_catalog_entry(
            "openrouter", "Anthropic", "anthropic/claude-sonnet-4-6", "Claude Sonnet 4.6",
            source="OpenRouter",
        )]

    async def fake_anthropic(_client, _api_key, **_kwargs):
        return [model_catalog_api._build_model_catalog_entry(
            "anthropic", "Anthropic", "claude-sonnet-4-6", "Claude Sonnet 4.6"
        )]

    async def fake_compatible(_client, provider_id, provider_label, _api_key, _base_url, **_kwargs):
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

    async def fake_openrouter(_client, _api_key, _cache_ttl=""):
        return [model_catalog_api._build_model_catalog_entry(
            "openrouter", "Anthropic", "anthropic/claude-opus", "Claude Opus", source="OpenRouter"
        )]

    async def fake_anthropic(_client, _api_key, **_kwargs):
        raise RuntimeError("anthropic failed")

    async def fake_compatible(_client, _provider_id, _provider_label, _api_key, _base_url, **_kwargs):
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

    async def fake_compatible(_client, provider_id, provider_label, api_key, base_url, **_kwargs):
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


@pytest.mark.parametrize("settings, expected", [
    ({"LOCAL_MODEL_SOURCE": "~/models/qwen3-coder-q4.gguf", "LOCAL_MODEL_CONTEXT_LENGTH": 32768},
     {"id": "qwen3-coder-q4.gguf", "context_window": 32768}),
    ({"LOCAL_MODEL_SOURCE": "bartowski/Qwen3-GGUF", "LOCAL_MODEL_FILENAME": "sub/qwen3-q8.gguf",
      "LOCAL_MODEL_CONTEXT_LENGTH": "auto"}, {"id": "qwen3-q8.gguf"}),
])
def test_a_configured_local_model_is_one_catalog_row_that_says_it_is_local(monkeypatch, settings, expected):
    """An install whose only model is local still has a row to choose (a local model has no route prefix)."""
    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: settings)

    payload = json.loads(asyncio.run(model_catalog_api.api_model_catalog(None)).body.decode("utf-8"))

    local = [item for item in payload["items"] if item.get("use_local")]
    assert len(local) == 1 and payload["items"] == local
    row = local[0]
    assert row["processing_transport"] == ["standard"]
    assert row["use_local"] is True and row["provider_id"] == "local" and row["id"] == expected["id"]
    assert row["label"] == f'Local · {expected["id"]}'
    assert row.get("context_window") == expected.get("context_window")
    # No API source claims the row: every picker that lists a source's models by its route prefix leaves it out.
    assert row["value"] == f'local::{expected["id"]}'
    # The published shape declares the field the row is told apart by.
    from ouroboros.gateway.management_contracts import ModelCatalogItem, ModelCatalogResponse
    from ouroboros.gateway.schema import validate_ingress

    assert "use_local" in ModelCatalogItem.__annotations__
    assert not validate_ingress(payload, ModelCatalogResponse)


def test_no_local_row_without_a_configured_local_model(monkeypatch):
    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: {"LOCAL_MODEL_SOURCE": "  ", "LOCAL_MODEL_PORT": 8766})

    payload = json.loads(asyncio.run(model_catalog_api.api_model_catalog(None)).body.decode("utf-8"))

    assert payload["items"] == []


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

    monkeypatch.setattr(model_catalog_api, "_provider_specs", lambda settings, **kwargs: [({"id": "provider", "label": "Provider"}, _loader)])

    response = asyncio.run(model_catalog_api.api_model_catalog(None))
    payload = json.loads(response.body.decode("utf-8"))

    assert payload["items"] == [{"value": "provider::model", "label": "Provider Model"}]
    assert payload["errors"] == []
    assert calls == ["provider"]


def test_model_catalog_classifies_httpx_error_stage(monkeypatch):
    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: {})

    async def _loader(_client):
        raise httpx.ConnectError("network down")

    monkeypatch.setattr(model_catalog_api, "_provider_specs", lambda settings, **kwargs: [({"id": "provider", "label": "Provider"}, _loader)])

    response = asyncio.run(model_catalog_api.api_model_catalog(None))
    payload = json.loads(response.body.decode("utf-8"))

    assert payload["items"] == []
    assert payload["errors"][0]["provider_id"] == "provider"
    assert payload["errors"][0]["stage"] == "connect"


def test_direct_api_catalog_loaders_keep_the_native_async_transport():
    # HTTP stays on the calling loop. Saved evidence persistence has its own
    # off-loop witness in test_saved_catalog_image_input.py.
    async def run():
        loop, calls = asyncio.get_running_loop(), []

        class Client:
            async def get(self, url, **kwargs):
                assert asyncio.get_running_loop() is loop
                calls.append(url)
                await asyncio.sleep(0)
                return _Response({"data": [{"id": "vendor/model"}]})

        for source, loader in model_catalog_api._provider_specs({
            "OPENROUTER_API_KEY": "fixture", "ANTHROPIC_API_KEY": "fixture", "DEEPSEEK_API_KEY": "fixture",
            "OPENAI_COMPATIBLE_BASE_URL": "https://compat.invalid/v1",
        }):
            items, error = await model_catalog_api._load_provider(Client(), source["id"], loader)
            assert items and not error
        assert len(calls) == 4

    asyncio.run(run())


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


def test_account_catalog_matches_the_published_response_shape(account_engine):
    from ouroboros.gateway.management_contracts import ModelCatalogResponse
    from ouroboros.gateway.schema import validate_ingress

    payload = json.loads(asyncio.run(model_catalog_api.api_model_catalog(None)).body)
    assert any(row['default_reasoning_effort'] is None for row in payload['items'])
    assert any(row['observed_at'] is None for row in payload['items'])
    assert not validate_ingress(payload, ModelCatalogResponse)


def test_agent_inventory_does_not_supply_an_absent_main_chat_transport(account_engine):
    gateway, _ = account_engine
    # Cursor login and agent inventory may succeed independently. Neither is a
    # raw-model source; borrowing them here would reproduce the false-ready bug.
    gateway.harness_model_catalog = lambda **_: {"models": [{"id": "cursor-test"}]}
    gateway.credential_profiles = lambda: {"profiles": [{
        "profile": {"harness_id": "cursor", "profile_id": "cursor-default"},
        "status": {"verification": "passed", "verification_source": "vendor"},
    }]}
    assert gateway.credential_profiles()["profiles"][0]["status"]["verification"] == "passed"
    assert gateway.harness_model_catalog()["models"]
    result = model_catalog_api._subscription_model_catalog("cursor", "cursor-default")
    assert result["items"] == []
    assert result["errors"][0]["code"] == "model_source_unavailable"
    assert gateway.calls == []  # no unrelated/default model request
    assert model_catalog_api._subscription_model_catalog("codex")["items"]


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
    {"parameters": [{"name": "view", "location": "query", "enum": ["selected"]}]}])
def test_account_view_requires_the_exact_declared_query(change):
    path = "/v2/model-sources/:id/models"
    assert not model_catalog_api.account_catalog_supported([{**_account_operation(path), **change}], path)


def test_profile_identity_compares_the_last_catalog_observation_and_current_store(account_engine, monkeypatch):
    from ouroboros import claudexor_daemon as owned

    gateway, accounts = account_engine
    gateway.store_identity = "first-store"
    monkeypatch.setattr(owned, "read_owned_gateway", lambda **_: gateway)
    model_catalog_api._subscription_model_catalog("codex", "a")
    assert model_catalog_api.observed_profile("codex", "a") == ("first-store", "fixture-a")
    accounts[0]["catalog"]["accountFingerprint"] = "new-login"
    model_catalog_api._subscription_model_catalog("codex", "a")
    calls = list(gateway.calls)
    assert model_catalog_api.observed_profile("codex", "a") == ("first-store", "new-login")
    gateway.store_identity = "other-store"
    assert model_catalog_api.observed_profile("codex", "a") is None
    assert gateway.calls == calls  # the write-side comparison never requests provider models
    accounts[0]["catalog"]["source"] = "another-source"
    model_catalog_api._subscription_model_catalog("codex", "a")
    assert model_catalog_api.observed_profile("codex", "a") is None


def test_model_catalog_reports_a_bad_extra_ca_bundle_instead_of_a_bare_500(monkeypatch):
    """An unreadable OUROBOROS_EXTRA_CA_BUNDLE is named in the catalog errors; the engine
    catalog still loads and the endpoint answers 200."""
    from ouroboros.net_transport import ExtraCaBundleError

    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: {"OPENAI_API_KEY": "openai-key"})
    monkeypatch.setattr(model_catalog_api, "_subscription_model_catalog",
                        lambda *_args: {"items": [{"value": "engine-model"}], "errors": []})

    def _boom():
        raise ExtraCaBundleError("OUROBOROS_EXTRA_CA_BUNDLE is not readable: /nope.pem")

    monkeypatch.setattr(model_catalog_api, "verify_kwargs", _boom)
    response = asyncio.run(model_catalog_api.api_model_catalog(None))
    assert response.status_code == 200
    payload = json.loads(response.body.decode("utf-8"))
    assert [item["value"] for item in payload["items"]] == ["engine-model"]
    trust = [row for row in payload["errors"] if row.get("provider_id") == "extra_ca_bundle"]
    assert trust and "not readable" in trust[0]["error"] and trust[0]["stage"] == "trust"


@pytest.mark.parametrize("program", [None, {"path": "/vendor/program", "version": "2.1.319"}, {"path": "/vendor/program", "version": None}])
def test_catalog_producer_survives_flattening_without_borrowing_a_version(account_engine, program):
    gateway, accounts = account_engine
    for account in accounts:
        if account["catalog"]:
            account["catalog"]["sourceProgram"] = program
    payload = model_catalog_api._subscription_model_catalog("codex")
    assert payload["items"]
    assert all("sourceProgram" not in item for item in payload["items"])
    assert all("credentialHarness" not in item for item in payload["items"])
    assert payload["model_sources"][0]["credentialHarness"] == "codex"
    accounts[0]["catalog"]["models"] = []
    payload = model_catalog_api._subscription_model_catalog("codex", "a")
    assert payload["items"] == []
    assert payload["account_catalogs"][0]["accounts"][0]["catalog"]["sourceProgram"] == program


# --- the catalog's published price on an OpenRouter row (T06) ---------------------------------

def _openrouter_items(records):
    class Client:
        async def get(self, url, **_kwargs):
            return _Response({"data": records})

    return asyncio.run(model_catalog_api._fetch_openrouter_model_catalog(Client(), "or-key"))


@pytest.mark.parametrize("pricing, expected", [
    ({"prompt": "0.000004", "completion": "0.00002"}, {"input": 4.0, "output": 20.0}),
    ({"prompt": "0.000000075", "completion": "0.0000003", "request": "0"}, {"input": 0.075, "output": 0.3}),
    # An explicit zero is a free model's price, not a missing one.
    ({"prompt": "0", "completion": "0"}, {"input": 0.0, "output": 0.0}),
    ({"prompt": "0.000002", "completion": "0.00001", "input_cache_read": "0.0000002", "overrides": [
        {"min_prompt_tokens": 200000, "prompt": "0.000004", "completion": "0.000015"}]},
     {"input": 2.0, "output": 10.0, "tiers": [{"min_prompt_tokens": 200000, "input": 4.0, "output": 15.0}]}),
])
def test_an_openrouter_row_carries_the_price_its_catalog_states(pricing, expected):
    [item] = _openrouter_items([{"id": "vendor/model", "name": "Vendor: Model", "pricing": pricing}])
    assert item["price"] == expected
    assert item["label"] == "OpenRouter · Vendor: Model", "the name stays the name"


@pytest.mark.parametrize("pricing", [
    None, {}, {"prompt": "0.000004"}, {"completion": "0.00002"}, {"prompt": "0.000004", "completion": None},
    {"prompt": "0.000004", "completion": "0.00002", "request": "0.005"},  # a per-request fee on every call
    {"prompt": "0.000004", "completion": "0.00002", "request": "n/a"},
    {"prompt": "-1", "completion": "-1"},  # the router's variable price
    {"prompt": "n/a", "completion": "0.00002"}, {"prompt": [1], "completion": "0.00002"},
    {"prompt": "nan", "completion": "0.00002"}, {"prompt": "0.002", "completion": "0.00002"},  # over $1000 per 1M
    {"prompt": "0.000004", "completion": "0.00002", "overrides": [{"min_prompt_tokens": 1000, "prompt": "-0.000001"}]},
    # A tier accounting cannot read or skips leaves the price partial: no price rather than the base alone.
    {"prompt": "0.000002", "completion": "0.00001", "overrides": [{"min_prompt_tokens": 200000, "prompt": "n/a", "completion": "0.000015"}]},
    {"prompt": "0.000002", "completion": "0.00001", "overrides": [{"min_prompt_tokens": 200000, "prompt": "inf"}]},
    {"prompt": "0.000002", "completion": "0.00001", "overrides": [{"min_prompt_tokens": 200000, "prompt": None}]},
    {"prompt": "0.000002", "completion": "0.00001", "overrides": [{"min_prompt_tokens": 200000, "prompt": "0.002"}]},
    {"prompt": "0.000002", "completion": "0.00001", "overrides": [
        {"min_prompt_tokens": 200000, "input_cache_read": "n/a", "prompt": "0.000004"}]},
    # A price that changes with the hour is not an input-length tier the picker can state.
    {"prompt": "0.00000015", "completion": "0.0000006", "overrides": [{"utc_start": 1500, "utc_end": 100, "prompt": "0.000000075"}]},
    # A charge four decimals per 1M would print as zero is never shown as free.
    {"prompt": "0.00000000001", "completion": "0.00000000001"},
    {"prompt": "0.00000000001", "completion": "0.000005"},
    # Reasoning billed at its own rate: the output figure is not the output price.
    {"prompt": "0.00000015", "completion": "0.0000006", "internal_reasoning": "0.0000035"},
    # Free by tokens, yet the call is charged for something else: "free" would be false.
    {"prompt": "0", "completion": "0", "image": "0.04"}, {"prompt": "0", "completion": "0", "web_search": "0.004"},
    {"prompt": "0", "completion": "0", "image": "0.04", "overrides": [{"min_prompt_tokens": 1000, "prompt": "0", "completion": "0"}]},
    # A flag is no price, and a tier with its own per-request fee is not a plain input-length tier.
    {"prompt": False, "completion": False},
    {"prompt": "0.000002", "completion": "0.00001", "overrides": [
        {"min_prompt_tokens": 1000, "prompt": "0.000004", "completion": "0.00002", "request": "0.01"}]},
    # A tier that also names an hour window is not a plain input-length tier.
    {"prompt": "0.000002", "completion": "0.00001", "overrides": [
        {"min_prompt_tokens": 1000, "utc_start": 1500, "utc_end": 100, "prompt": "0.000004"}]},
])
def test_an_openrouter_row_without_a_complete_token_price_has_no_price_field(pricing):
    record = {"id": "vendor/model", "name": "Vendor: Model", **({} if pricing is None else {"pricing": pricing})}
    [item] = _openrouter_items([record])
    assert "price" not in item, "absent, never null or zero"
    assert item["value"] == "vendor/model", "a price never decides whether a model is listed"


@pytest.mark.parametrize("meter", ["input_cache_write_1h", "input_cache_read", "image"])
def test_zero_token_rates_with_a_paid_override_do_not_advertise_free(meter):
    pricing = {"prompt": "0", "completion": "0", "overrides": [{"min_prompt_tokens": 100, meter: "0.000004"}]}
    [item] = _openrouter_items([{"id": "vendor/model", "pricing": pricing}])
    assert item["value"] == "vendor/model" and "price" not in item


@pytest.mark.parametrize("pricing, cache_ttl, expected", [
    # The record's own figure for the cache tier the install sends: 1h reads the 1-hour field, any other named tier the base.
    ({"input_cache_write": "0.000005", "input_cache_write_1h": "0.000008"}, "default", 5.0),
    ({"input_cache_write": "0.000005", "input_cache_write_1h": "0.000008"}, "5m", 5.0),
    ({"input_cache_write": "0.000005", "input_cache_write_1h": "0.000008"}, "1h", 8.0),
    ({"input_cache_write": "0"}, "5m", 0.0),
    # Never the other tier's figure, never one derived from the input price, and none when no tier is named.
    ({"input_cache_write": "0.000005"}, "1h", None),
    ({"input_cache_write_1h": "0.000008"}, "5m", None),
    ({}, "1h", None),
    ({"input_cache_write": "0.000005", "input_cache_write_1h": "0.000008"}, "", None),
    # A figure that cannot be read, or would print a charge as zero, is not stated; the price itself stays.
    ({"input_cache_write": "-0.000001"}, "5m", None), ({"input_cache_write": True}, "5m", None),
    ({"input_cache_write": "0.00000000001"}, "5m", None),
    # A price that changes with input length states no single cache-write figure.
    ({"input_cache_write": "0.000005", "overrides": [{"min_prompt_tokens": 200000, "prompt": "0.000008"}]}, "5m", None),
    ({"input_cache_write_1h": "0.000008", "overrides": [{"min_prompt_tokens": 200000, "input_cache_write": "0.000006"}]}, "1h", None),
])
def test_a_price_states_the_cache_write_figure_of_the_installs_cache_tier_or_none(pricing, cache_ttl, expected):
    from ouroboros.llm_pricing import openrouter_catalog_price

    price = openrouter_catalog_price({"id": "vendor/model", "pricing": {"prompt": "0.000004", "completion": "0.00002", **pricing}},
                                     cache_ttl)
    assert (price["input"], price["output"]) == (4.0, 20.0)
    assert price.get("cache_write") == expected
    assert ("cache_write" in price) is (expected is not None), "absent, never null"
    # The figure carries the tier it is of: whoever holds it later says it only while the settings still name that tier.
    assert price.get("cache_write_ttl") == (cache_ttl if expected is not None else None)


@pytest.mark.parametrize("stored, tier", [("1h", "1h"), ("5m", "5m"), ("1H", "1h"), ("", "1h"), ("weekly", "1h")])
def test_the_catalog_reads_the_cache_tier_as_the_wire_resolves_it(monkeypatch, stored, tier):
    """The figure shown is the tier's the install sends (`settings_scales.resolve_prompt_cache_ttl`: lower-cased, an
    unknown or absent value is the shipped default), never the raw text of the setting."""
    from ouroboros.config import SETTINGS_DEFAULTS

    assert SETTINGS_DEFAULTS["OUROBOROS_PROMPT_CACHE_TTL"] == "1h"
    seen = []
    monkeypatch.setattr(model_catalog_api, "_connection_loader", lambda kind, target, record, cache_ttl="": seen.append(cache_ttl))
    model_catalog_api._provider_specs({"OPENROUTER_API_KEY": "or-key", "OUROBOROS_PROMPT_CACHE_TTL": stored})
    assert seen == [tier]


@pytest.mark.parametrize("stored, figure, tier", [("5m", 5.0, "5m"), ("1h", 8.0, "1h"), ("default", 5.0, "default"), ("", 8.0, "1h")])
def test_the_settings_cache_tier_reaches_the_row_through_the_real_loader(stored, figure, tier):
    """From a settings document to a catalog row with nothing replaced but the network: the loader the catalog runs
    hands the resolved tier to the price, so the row's figure is that tier's and says so (`cache_write_ttl`)."""
    record = {"id": "vendor/model", "name": "Vendor: Model", "pricing": {
        "prompt": "0.000004", "completion": "0.00002", "input_cache_write": "0.000005", "input_cache_write_1h": "0.000008"}}

    class Client:
        async def get(self, _url, **_kwargs):
            return httpx.Response(200, json={"data": [record]}, request=httpx.Request("GET", _url))

    [(source, loader)] = model_catalog_api._provider_specs(
        {"OPENROUTER_API_KEY": "or-key", "OUROBOROS_PROMPT_CACHE_TTL": stored}, record_image_input=False)
    [row] = asyncio.run(loader(Client()))
    assert source["id"] == "openrouter"
    assert (row["price"]["cache_write"], row["price"]["cache_write_ttl"]) == (figure, tier)


def test_an_openrouter_row_shows_the_window_its_catalog_publishes():
    """Display only, beside the price (`context_window`): sizing reads its own evidence, never a catalog row."""
    rows = _openrouter_items([{"id": "vendor/wide", "context_length": 1000000}, {"id": "vendor/unsized"},
                              {"id": "vendor/zero", "context_length": 0}, {"id": "vendor/text", "context_length": "200000"},
                              {"id": "vendor/flag", "context_length": True}])
    assert [row.get("context_window") for row in rows] == [1000000, None, None, None, None]
    assert all("max_context_window" not in row for row in rows)


def _draft_catalog(monkeypatch, tmp_path, saved, draft, *, provider=None):
    """POST /api/onboarding/model-catalog over a saved settings file; the provider reads are recorded, not sent."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    import ouroboros.config as config
    import ouroboros.gateway.onboarding as gw_onboarding

    monkeypatch.setattr(config, "load_settings", lambda: dict(saved))
    def subscriptions(*_a):
        assert provider is None, "an API connection does not require the subscription engine"
        return {"items": [], "errors": [], "model_sources": []}
    monkeypatch.setattr(model_catalog_api, "_subscription_model_catalog", subscriptions)
    asked = []

    async def openrouter(_client, api_key, cache_ttl=""):
        asked.append(("openrouter", api_key))
        return [{**model_catalog_api._build_model_catalog_entry("openrouter", "Vendor", "vendor/model", "Model", source="OpenRouter"),
                 "price": {"input": 4.0, "output": 20.0}}]

    async def compatible(_client, provider_id, _label, api_key, base_url, **kwargs):
        asked.append((provider_id, api_key, base_url, bool(kwargs.get("record_image_input"))))
        return [model_catalog_api._build_model_catalog_entry(provider_id, "Compat", "served", "served")]

    monkeypatch.setattr(model_catalog_api, "_fetch_openrouter_model_catalog", openrouter)
    monkeypatch.setattr(model_catalog_api, "_fetch_openai_compatible_model_catalog", compatible)
    app = Starlette(routes=[Route("/api/onboarding/model-catalog", gw_onboarding.api_onboarding_model_catalog, methods=["POST"])])
    app.state.drive_root = tmp_path
    with TestClient(app) as client:
        response = client.post("/api/onboarding/model-catalog", params={"provider_id": provider} if provider else {}, json=draft)
    return response, asked


def test_exact_api_connection_preserves_its_price_and_leaves_other_sources_unread(monkeypatch):
    asked = []

    async def catalog(_client, key, _ttl):
        asked.append(key)
        return [{**model_catalog_api._build_model_catalog_entry("openrouter", "Vendor", "vendor/model", "Model"),
                 "price": {"input": 3}, "context_window": 200_000}]

    monkeypatch.setattr(model_catalog_api, "_fetch_openrouter_model_catalog", catalog)
    monkeypatch.setattr(model_catalog_api, "_subscription_model_catalog", lambda *_: pytest.fail("unrelated subscription read"))
    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: {
        "OPENROUTER_API_KEY": "builtin", "LOCAL_MODEL_SOURCE": "local/model",
        "OUROBOROS_CONNECTIONS": [{"kind": "openrouter", "name": "work", "api_key": "named", "enabled": True}]})

    def get(**query):
        return asyncio.run(model_catalog_api.api_model_catalog(SimpleNamespace(query_params=query)))

    payload = json.loads(get(provider_id="openrouter@work").body)
    [entry] = payload["items"]
    assert (entry["value"], entry["price"], entry["context_window"]) == ("openrouter@work::vendor/model", {"input": 3}, 200_000)
    assert payload["errors"] == [] and payload["model_sources"] == []
    assert json.loads(get(provider_id="local").body)["items"][0]["use_local"]
    missing = json.loads(get(provider_id="openrouter@removed").body)
    assert missing["items"] == [] and missing["errors"][0]["provider_id"] == "openrouter@removed"
    assert get(provider_id="", source_id="codex").status_code == 400 and asked == ["named"]


@pytest.mark.parametrize("engine_slow", [False, True])
def test_slow_connection_reaches_its_deadline_without_losing_healthy_catalogs(monkeypatch, engine_slow):
    from contextlib import asynccontextmanager
    from threading import Event

    api_started = Event()

    def engine(*_args):
        # If discovery ran ahead of the HTTP loaders, this wait would expire before fast starts.
        assert api_started.wait(2)
        if engine_slow:
            import time
            time.sleep(0.2)
        return {"model_sources": [{"id": "opaque"}], "items": [], "errors": []}

    async def fast(_client):
        api_started.set()
        return [{"value": "fast::model"}]

    async def slow(_client):
        await asyncio.Event().wait()

    @asynccontextmanager
    async def client(**_kwargs):
        yield None  # The loaders are synthetic; don't spend the 80 ms fixture deadline loading trust roots.

    monkeypatch.setattr(model_catalog_api.httpx, "AsyncClient", client)
    monkeypatch.setattr(model_catalog_api, "_subscription_model_catalog", engine)
    monkeypatch.setattr(model_catalog_api, "_CATALOG_HTTP_TIMEOUT_SEC", 0.08)
    sources = [{"id": "fast", "label": "Fast source", "processing_transport": ["standard"]},
               {"id": "slow@work", "label": "Slow source", "processing_transport": ["standard", "fast"]}]
    monkeypatch.setattr(model_catalog_api, "_provider_specs", lambda *_a, **_kw: list(zip(sources, (fast, slow))))
    payload = asyncio.run(asyncio.wait_for(model_catalog_api.catalog_payload({}), timeout=3))
    assert payload["items"] == [{"value": "fast::model"}]
    expected = {"slow@work", "claudexor"} if engine_slow else {"slow@work"}
    assert {err["provider_id"] for err in payload["errors"]} == expected
    assert all(err["stage"] == "timeout" and err["error"] for err in payload["errors"])
    assert payload["model_sources"] == ([] if engine_slow else [{"id": "opaque"}])
    assert payload["api_sources"] == sources  # Transport evidence is not lost with an inventory timeout.


@pytest.mark.parametrize("source_ids", [("slow", "quick"), ("quick", "slow")])
def test_batch_keeps_finished_subscription_and_closes_its_gateway_after_the_late_read(monkeypatch, source_ids):
    from threading import Event
    from ouroboros import claudexor_daemon

    release, quick, closed = Event(), Event(), Event()
    declarations = [{"id": source} for source in source_ids]

    class Gateway:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            closed.set()

        def operations(self):
            return []

        def list_model_sources(self):
            return {"sources": declarations}

        def list_source_models(self, source, **_kwargs):
            if source == "slow":
                assert release.wait(3)
            else:
                quick.set()
            assert not closed.is_set()
            return {"models": [{"id": "one"}]}

    monkeypatch.setattr(claudexor_daemon, "owned_daemon_provisioned", lambda: True)
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", Gateway)
    monkeypatch.setattr(model_catalog_api, "_CATALOG_HTTP_TIMEOUT_SEC", 0.35)

    async def check():
        try:
            payload = await model_catalog_api.catalog_payload({"LOCAL_MODEL_SOURCE": "local/model"})
            assert quick.is_set() and not closed.is_set()
            assert payload["model_sources"] == declarations
            assert {row["value"] for row in payload["items"]} == {"claudexor::quick=one", "local::model"}
            assert [(row["source_id"], row["stage"]) for row in payload["errors"]] == [("slow", "timeout")]
        finally:
            release.set()
        assert await asyncio.to_thread(closed.wait, 3)
        assert not any(row["value"] == "claudexor::slow=one" for row in payload["items"])

    asyncio.run(check())


def test_a_setup_draft_reads_the_catalog_of_its_own_access_before_anything_is_saved(monkeypatch, tmp_path):
    """The wizard's Models step: a key typed in the draft is in no saved file, so `GET /api/model-catalog` cannot
    list or price its models. The draft read asks the providers with the draft's access, while Main is still blank,
    and writes nothing; without a draft key it lists what the saved access gives."""
    response, asked = _draft_catalog(monkeypatch, tmp_path, {}, {"OPENROUTER_API_KEY": "sk-or-v1-typed-in-the-draft"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert asked == [("openrouter", "sk-or-v1-typed-in-the-draft")]
    assert [(item["value"], item["price"]) for item in body["items"]] == [("vendor/model", {"input": 4.0, "output": 20.0})]
    assert body["errors"] == []
    assert not (tmp_path / "settings.json").exists()
    # Select a named draft connection with another usable key present.
    draft = {"OPENROUTER_API_KEY": "sk-or-v1-other", "OUROBOROS_CONNECTIONS": [
        {"kind": "openrouter", "name": "work", "enabled": True, "api_key": "sk-or-v1-named"}]}
    selected, asked = _draft_catalog(monkeypatch, tmp_path, {}, draft, provider="openrouter@work")
    assert [item["value"] for item in selected.json()["items"]] == ["openrouter@work::vendor/model"]
    assert asked == [("openrouter", "sk-or-v1-named")]
    # The saved access alone: nothing typed, nothing asked.
    response, asked = _draft_catalog(monkeypatch, tmp_path, {}, {})
    assert (response.json()["items"], asked) == ([], [])
    assert _draft_catalog(monkeypatch, tmp_path, {}, {"OPENROUTER_API_KEY": ["not", "text"]})[0].status_code == 400


@pytest.mark.parametrize("provider", [None, "openai-compatible"])
def test_a_draft_read_sends_a_kept_key_only_where_it_was_saved_for(monkeypatch, tmp_path, provider):
    """Provider Test's rule (`_saved_secret_moved`): a draft that keeps a saved key (its field posts the marker back)
    and moves the address would send that key to a server its author chose now. The connection is not asked and its
    error row says why; a key typed in the draft may go to the draft's address. A draft read records no evidence."""
    from ouroboros.secret_masking import mask_settings_secret

    saved = {"OPENAI_COMPATIBLE_API_KEY": "sk-kept-compatible-key-123456", "OPENAI_COMPATIBLE_BASE_URL": "https://saved.example/v1"}
    marker = mask_settings_secret("OPENAI_COMPATIBLE_API_KEY", saved["OPENAI_COMPATIBLE_API_KEY"])
    moved = {"OPENAI_COMPATIBLE_API_KEY": marker, "OPENAI_COMPATIBLE_BASE_URL": "https://elsewhere.example/v1"}
    response, asked = _draft_catalog(monkeypatch, tmp_path, saved, moved, provider=provider)
    body = response.json()
    assert asked == [] and body["items"] == [] and body["api_sources"] == []
    assert [(row["provider_id"], row["stage"]) for row in body["errors"]] == [("openai-compatible", "kept_key")]
    # The same kept key at its saved address is asked, and so is a typed key at the new address.
    response, asked = _draft_catalog(monkeypatch, tmp_path, saved, {**moved, "OPENAI_COMPATIBLE_BASE_URL": "https://saved.example/v1"}, provider=provider)
    assert asked == [("openai-compatible", "sk-kept-compatible-key-123456", "https://saved.example/v1", False)]
    response, asked = _draft_catalog(monkeypatch, tmp_path, saved, {**moved, "OPENAI_COMPATIBLE_API_KEY": "sk-typed-now-for-elsewhere"}, provider=provider)
    assert asked == [("openai-compatible", "sk-typed-now-for-elsewhere", "https://elsewhere.example/v1", False)]
    assert response.json()["errors"] == []
    # The saved catalog's own read is the one that records what it learns.
    asked.clear()
    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: dict(saved))
    request = SimpleNamespace(query_params={"provider_id": provider}) if provider else None
    asyncio.run(model_catalog_api.api_model_catalog(request))
    assert asked == [("openai-compatible", "sk-kept-compatible-key-123456", "https://saved.example/v1", True)]


def test_a_draft_read_binds_a_kept_key_to_its_own_saved_row_or_pair(monkeypatch, tmp_path):
    """The binding is the kept key's own saved row or pair, whatever serves today (Provider Test prepares the same):
    a draft that only switches a saved connection on, or clears the dedicated compatible pair to use the saved legacy
    one, moved nothing and is asked; the same rows at another address are still refused."""
    from ouroboros.gateway.secret_rows import mask_connections

    row = {"kind": "openai-compatible", "name": "lab", "api_key": "sk-kept-lab-key-1234567890",
           "base_url": "https://lab.example/v1", "enabled": False}
    saved = {"OUROBOROS_CONNECTIONS": [row]}
    [masked] = mask_connections([row])
    response, asked = _draft_catalog(monkeypatch, tmp_path, saved, {"OUROBOROS_CONNECTIONS": [{**masked, "enabled": True}]})
    assert response.json()["errors"] == []
    assert asked == [("openai-compatible", "sk-kept-lab-key-1234567890", "https://lab.example/v1", False)]
    response, asked = _draft_catalog(monkeypatch, tmp_path, saved, {"OUROBOROS_CONNECTIONS": [
        {**masked, "enabled": True, "base_url": "https://elsewhere.example/v1"}]})
    assert asked == [] and [error["stage"] for error in response.json()["errors"]] == ["kept_key"]
    # A key typed into the row in this draft is its author's to send to the row's new address.
    response, asked = _draft_catalog(monkeypatch, tmp_path, saved, {"OUROBOROS_CONNECTIONS": [
        {**masked, "enabled": True, "base_url": "https://elsewhere.example/v1", "api_key": "sk-typed-for-elsewhere-123456"}]})
    assert asked == [("openai-compatible", "sk-typed-for-elsewhere-123456", "https://elsewhere.example/v1", False)]
    assert response.json()["errors"] == []
    # Both built-in compatible pairs saved; the draft clears the dedicated one and so reaches the legacy pair.
    saved = {"OPENAI_COMPATIBLE_API_KEY": "sk-kept-dedicated-123456", "OPENAI_COMPATIBLE_BASE_URL": "https://dedicated.example/v1",
             "OPENAI_API_KEY": "sk-kept-legacy-1234567890", "OPENAI_BASE_URL": "https://legacy.example/v1"}
    from ouroboros.secret_masking import mask_settings_secret

    kept = mask_settings_secret("OPENAI_API_KEY", saved["OPENAI_API_KEY"])  # the legacy field posts its marker back
    cleared = {"OPENAI_COMPATIBLE_API_KEY": "", "OPENAI_COMPATIBLE_BASE_URL": "", "OPENAI_API_KEY": kept}
    response, asked = _draft_catalog(monkeypatch, tmp_path, saved, cleared)
    assert response.json()["errors"] == [], response.json()["errors"]
    assert [call for call in asked if call[2] == "https://legacy.example/v1"] == [
        ("openai-compatible", "sk-kept-legacy-1234567890", "https://legacy.example/v1", False)], \
        "the kept legacy key goes to the pair the draft reaches, at its own saved address"
    assert "sk-kept-dedicated-123456" not in [call[1] for call in asked]
    # The legacy address is no draft field: a draft that names another one moves nothing, and the kept key still
    # goes only where it was saved for.
    response, moved = _draft_catalog(monkeypatch, tmp_path, saved, {**cleared, "OPENAI_BASE_URL": "https://elsewhere.example/v1"})
    assert moved == asked and "https://elsewhere.example/v1" not in [call[2] for call in moved]


def test_the_shown_price_is_the_catalog_row_fetch_openrouter_pricing_reads():
    from unittest.mock import MagicMock, patch
    from ouroboros.gateway.management_contracts import ModelPrice
    from ouroboros.gateway.schema import validate_ingress
    from ouroboros.llm_pricing import fetch_openrouter_pricing, openrouter_catalog_price

    record = {"id": "vendor/tiered", "pricing": {"prompt": "0.00000125", "completion": "0.00001", "overrides": [
        {"min_prompt_tokens": 200000, "prompt": "0.0000025", "completion": "0.000015"},
        {"min_prompt_tokens": 100000, "completion": "0.000012"}]}}
    response = MagicMock()
    response.json.return_value = {"data": [record]}
    with patch("requests.get", return_value=response):
        row = fetch_openrouter_pricing(timeout_sec=5.0)["vendor/tiered"]
    assert openrouter_catalog_price(record) == {"input": row[0], "output": row[3], "tiers": [
        {"min_prompt_tokens": minimum, "input": tier[0], "output": tier[3]} for minimum, tier in row.tiers]}
    [item] = _openrouter_items([record])
    assert item["price"] == {"input": 1.25, "output": 10.0, "tiers": [
        {"min_prompt_tokens": 100000, "input": 1.25, "output": 12.0},
        {"min_prompt_tokens": 200000, "input": 2.5, "output": 12.0},
    ]}  # the last supplied output price also wins at the longer tier
    assert not validate_ingress(item["price"], ModelPrice)


def test_a_direct_route_with_the_same_model_id_gets_no_openrouter_price(monkeypatch):
    from ouroboros.gateway.management_contracts import ModelCatalogItem, ModelCatalogResponse
    from ouroboros.gateway.schema import validate_ingress

    priced = {"prompt": "0.000004", "completion": "0.00002"}
    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: {"OPENROUTER_API_KEY": "or-key",
                                                                     "OPENAI_API_KEY": "openai-key"})

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, url, **_kwargs):
            # A direct server answering with the very same records, pricing field included, still prices nothing.
            return _Response({"data": [{"id": "openai/gpt-x", "name": "OpenAI: GPT X", "pricing": priced},
                                       *([] if "openrouter.ai" in url else [{"id": "gpt-x", "pricing": priced}])]})

    monkeypatch.setattr(model_catalog_api.httpx, "AsyncClient", Client)
    payload = json.loads(asyncio.run(model_catalog_api.api_model_catalog(None)).body.decode("utf-8"))

    prices = {item["value"]: item.get("price") for item in payload["items"]}
    assert prices == {"openai/gpt-x": {"input": 4.0, "output": 20.0}, "openai::openai/gpt-x": None, "openai::gpt-x": None}
    assert not any("price" in item for item in payload["items"] if item["value"].startswith("openai::"))
    # The published shape declares the field and its type.
    assert "price" in ModelCatalogItem.__annotations__
    assert not validate_ingress(payload, ModelCatalogResponse)
    damaged = {**payload, "items": [{**payload["items"][0], "price": {"input": "4", "output": 20.0}}]}
    assert validate_ingress(damaged, ModelCatalogResponse)


@pytest.mark.parametrize("second, expected", [
    ({"prompt": "0.000004", "completion": "0.00002"}, {"input": 4.0, "output": 20.0}),  # the records agree: one price
    ({"prompt": "0.000005", "completion": "0.000025"}, None),  # they disagree: no record's price is the model's
    (None, None),  # one record states none
])
def test_two_catalog_records_of_one_model_show_a_price_only_when_they_agree(monkeypatch, second, expected):
    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: {"OPENROUTER_API_KEY": "or-key"})
    first = {"id": "vendor/x", "name": "Vendor: X", "pricing": {"prompt": "0.000004", "completion": "0.00002"}}

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, url, **_kwargs):
            return _Response({"data": [first, {"id": "vendor/x", "name": "Vendor: X",
                                               **({} if second is None else {"pricing": second})}]})

    monkeypatch.setattr(model_catalog_api.httpx, "AsyncClient", Client)
    payload = json.loads(asyncio.run(model_catalog_api.api_model_catalog(None)).body.decode("utf-8"))
    [item] = [item for item in payload["items"] if item["value"] == "vendor/x"]
    assert item.get("price") == expected and item["name"] == "Vendor: X", "the model stays listed either way"


def test_two_records_of_one_model_disagreeing_on_its_price_condition_show_none(monkeypatch):
    from ouroboros import pricing_catalog

    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: {"DEEPSEEK_API_KEY": "synthetic-key"})
    said = iter(["off_peak_usd", ""])
    monkeypatch.setattr(pricing_catalog, "price_condition", lambda _model, **_kwargs: next(said, ""))

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, url, **_kwargs):
            return _Response({"data": [{"id": "deepseek-v4-pro"}, {"id": "deepseek-v4-pro"}] if "deepseek" in url else []})

    monkeypatch.setattr(model_catalog_api.httpx, "AsyncClient", Client)
    payload = json.loads(asyncio.run(model_catalog_api.api_model_catalog(None)).body.decode("utf-8"))
    [item] = [item for item in payload["items"] if item["value"] == "deepseek::deepseek-v4-pro"]
    assert "price_condition" not in item and "price" not in item, "no one record's condition is the model's"


@pytest.mark.parametrize("pricing, expected", [
    # Most reasoning models bill reasoning as output: the pair is the whole token price.
    ({"prompt": "0.000004", "completion": "0.00002", "internal_reasoning": "0.00002"}, {"input": 4.0, "output": 20.0}),
    ({"prompt": "0", "completion": "0", "internal_reasoning": "0", "image": "0"}, {"input": 0.0, "output": 0.0}),
])
def test_a_price_whose_other_fees_change_nothing_is_still_shown(pricing, expected):
    [item] = _openrouter_items([{"id": "vendor/model", "name": "Vendor: Model", "pricing": pricing}])
    assert item["price"] == expected


def test_a_model_listed_first_without_a_price_keeps_none_when_a_later_record_states_one(monkeypatch):
    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: {"OPENROUTER_API_KEY": "or-key"})

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, url, **_kwargs):
            return _Response({"data": [{"id": "vendor/x", "name": "Vendor: X"},
                                       {"id": "vendor/x", "name": "Vendor: X", "pricing": {"prompt": "0.000004", "completion": "0.00002"}}]})

    monkeypatch.setattr(model_catalog_api.httpx, "AsyncClient", Client)
    payload = json.loads(asyncio.run(model_catalog_api.api_model_catalog(None)).body.decode("utf-8"))
    [item] = [item for item in payload["items"] if item["value"] == "vendor/x"]
    assert "price" not in item


def known_spend_gaps(settings, subagents=None):
    """What the limits name as not counted: ``pricing.spend_coverage``'s first list."""
    from ouroboros.pricing import spend_coverage

    return spend_coverage(settings, subagents)[0]


def test_the_limits_name_what_they_cannot_count_and_nothing_where_every_route_has_a_price():
    """Dashboard › Costs says it under the limits: a call on a route without a known price adds nothing to known
    spend (`usage_accounting.paid_admission`), so the limits count without it."""
    from ouroboros.pricing import price_source

    routed = {"OUROBOROS_MODEL": "anthropic/claude-opus-5.5", "OUROBOROS_MODEL_LIGHT": "openai::gpt-6-mini",
              "OUROBOROS_MODEL_FALLBACKS": "openai::gpt-6", "OUROBOROS_WEBSEARCH_BACKEND": "ddgs"}
    # An empty Safety runs on Light's slot; Vision and Consciousness follow Main, which has a catalog price.
    assert known_spend_gaps(routed) == ["Light", "Safety Supervisor", "Fallback"]
    assert known_spend_gaps({**routed, "OUROBOROS_MODEL_SAFETY": "anthropic/claude-opus-5.5",
                             "OUROBOROS_MODEL_LIGHT": "", "OUROBOROS_MODEL_FALLBACKS": ""}) == []
    assert known_spend_gaps({"OUROBOROS_MODEL": "openai::gpt-6", "OUROBOROS_WEBSEARCH_BACKEND": "ddgs"}) == ["all model roles"]
    assert known_spend_gaps({"OUROBOROS_MODEL": "claudexor::codex=gpt-6"}) == [], "a subscription spends time, not money"
    # A slot with its own model keeps its own local flag; only an empty one runs on Main's (route_choice).
    assert known_spend_gaps({"OUROBOROS_MODEL": "qwen3.gguf", "USE_LOCAL_MAIN": "true", "OUROBOROS_MODEL_LIGHT": "openai::gpt-6-mini",
                             "USE_LOCAL_LIGHT": False, "OUROBOROS_WEBSEARCH_BACKEND": "ddgs"}) == ["Light", "Safety Supervisor"]
    assert known_spend_gaps({"OUROBOROS_MODEL": "openai::gpt-6", "USE_LOCAL_MAIN": "true",
                             "OUROBOROS_WEBSEARCH_BACKEND": "ddgs"}) == [], "a local model costs nothing"
    # Built-in web search by its paid legs: a direct key has no catalog, OpenRouter has one, ddgs is free.
    base = {"OUROBOROS_MODEL": "anthropic/claude-opus-5.5", "OPENROUTER_API_KEY": "k"}
    assert known_spend_gaps({**base, "OPENAI_API_KEY": "k", "OUROBOROS_WEBSEARCH_BACKEND": "openai"}) == ["Web search"]
    assert known_spend_gaps({**base, "OUROBOROS_WEBSEARCH_BACKEND": "openrouter"}) == []
    # Enabled API helpers on a route without a price; a switched-off one spends nothing.
    helper = {"route": {"kind": "api_model", "target_id": "deepseek::deepseek-v4"}}
    assert known_spend_gaps({**base, "OUROBOROS_WEBSEARCH_BACKEND": "ddgs"}, {"items": [helper]}) == ["Agents"]
    assert known_spend_gaps({**base, "OUROBOROS_WEBSEARCH_BACKEND": "ddgs"}, {"items": [{**helper, "enabled": False}]}) == []
    assert price_source("cloudru", {"OUROBOROS_RUB_USD_RATE": "0.012", "CLOUDRU_FOUNDATION_MODELS_API_KEY": "k"}) == "catalog"
    assert price_source("cloudru", {"OUROBOROS_RUB_USD_RATE": "0.012"}) == "unknown", "the catalog is read with the built-in key"
    assert price_source("cloudru", {"CLOUDRU_FOUNDATION_MODELS_API_KEY": "k"}) == "unknown", "roubles without a rate"


def test_the_spend_gaps_follow_the_runtime_lane_of_each_role(monkeypatch):
    """Each role's route by the one rule the runtime resolves it with (route_choice.following_route, asked over the
    document): a remote role on a direct key is named even beside a local Main, and a local one is not."""
    from ouroboros import search_routes

    local_main = {"OUROBOROS_MODEL": "qwen3-8b", "USE_LOCAL_MAIN": True, "OUROBOROS_WEBSEARCH_BACKEND": "ddgs"}
    assert known_spend_gaps({**local_main, "OUROBOROS_MODEL_VISION": "openai::gpt-6"}) == ["Vision"]
    assert known_spend_gaps({**local_main, "OUROBOROS_MODEL_SAFETY": "anthropic::claude-haiku-5-5"}) == ["Safety Supervisor"]
    assert known_spend_gaps({**local_main, "OUROBOROS_MODEL_VISION": "openai::gpt-oss-20b (local)"}) == [], (
        "a model stored with its local label runs locally, whatever its prefix")
    # An unchecked box is a row's resting state, stored as False: an empty Consciousness follows the local Main
    # (owner, 12.11) and spends nothing, as the wake itself runs.
    unchecked = {**local_main, "OUROBOROS_MODEL": "openai::gpt-6", "USE_LOCAL_CONSCIOUSNESS": False, "USE_LOCAL_LIGHT": False}
    assert known_spend_gaps(unchecked) == []
    # An empty Light follows its own box (owner, 15.11): checked under a cloud Main, Light and the Supervisor that
    # inherits it run locally, while Main and the Consciousness that follows it stay on the unpriced cloud route.
    boxed = {"OUROBOROS_MODEL": "openai::gpt-6", "USE_LOCAL_LIGHT": True, "OUROBOROS_WEBSEARCH_BACKEND": "ddgs",
             "OUROBOROS_MODEL_VISION": "anthropic/claude-opus-5.5"}
    assert known_spend_gaps(boxed) == ["Main", "Consciousness"]
    assert known_spend_gaps({**boxed, "USE_LOCAL_LIGHT": False}) == ["Main", "Light", "Safety Supervisor", "Consciousness"]
    routed = {"OUROBOROS_MODEL": "anthropic/claude-opus-5.5", "OUROBOROS_WEBSEARCH_BACKEND": "ddgs"}
    helper = {"route": {"kind": "api_model", "target_id": "deepseek::deepseek-v4"}}
    assert known_spend_gaps(routed, {"enabled": False, "items": [helper]}) == ["Agents"], (
        "a review names any enabled row with the catalog switched off")
    assert known_spend_gaps(routed, {"enabled": False, "items": [{**helper, "enabled": False}]}) == []
    # An empty Consciousness with its own box checked under a cloud Main wakes on the local runtime.
    priced = {"OUROBOROS_MODEL_LIGHT": "anthropic/claude-opus-5.5", "OUROBOROS_MODEL_VISION": "anthropic/claude-opus-5.5"}
    assert known_spend_gaps({**priced, "OUROBOROS_MODEL": "openai::gpt-6", "USE_LOCAL_CONSCIOUSNESS": True,
                             "OUROBOROS_WEBSEARCH_BACKEND": "ddgs"}) == ["Main"]
    # Vision has no box: an empty Vision runs Main's lane whatever a key of that name says (slot_lane_label).
    assert known_spend_gaps({**priced, "OUROBOROS_MODEL_VISION": "", "OUROBOROS_MODEL": "openai::gpt-6",
                             "USE_LOCAL_CONSCIOUSNESS": True, "USE_LOCAL_VISION": True,
                             "OUROBOROS_WEBSEARCH_BACKEND": "ddgs"}) == ["Main", "Vision"]
    # The Supervisor is named only while a Safety mode asks it: off in both, no check is sent (safety.py).
    asked = {"OUROBOROS_MODEL": "anthropic/claude-opus-5.5", "OUROBOROS_MODEL_SAFETY": "openai::gpt-6-mini",
             "OUROBOROS_WEBSEARCH_BACKEND": "ddgs"}
    off = {"OUROBOROS_SAFETY_MODE": "off", "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE": "off"}
    assert known_spend_gaps(asked) == ["Safety Supervisor"]
    assert known_spend_gaps({**asked, **off}) == []
    assert known_spend_gaps({**asked, **off, "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE": "light"}) == ["Safety Supervisor"]
    # Web search by every paid leg: a later one runs when the earlier fails, as a fallback model does.
    monkeypatch.setattr(search_routes, "_package_discoverable", lambda _name: True)
    two_keys = {"OUROBOROS_MODEL": "anthropic/claude-opus-5.5", "OPENROUTER_API_KEY": "k", "ANTHROPIC_API_KEY": "k"}
    assert [leg["source"] for leg in search_routes.resolve_web_search_route(settings=two_keys)["legs"]] == [
        "openrouter", "anthropic", "ddgs"]
    assert known_spend_gaps(two_keys) == ["Web search"], "the first leg has a catalog price, the second none"
    assert known_spend_gaps({**two_keys, "OUROBOROS_WEBSEARCH_BACKEND": "openrouter"}) == []
    rouble = {"OUROBOROS_RUB_USD_RATE": "0.012", "CLOUDRU_FOUNDATION_MODELS_API_KEY": "k", "OUROBOROS_WEBSEARCH_BACKEND": "ddgs"}
    assert known_spend_gaps({**rouble, "OUROBOROS_MODEL": "cloudru::GigaChat-2-Max"}) == []
    assert known_spend_gaps({**rouble, "OUROBOROS_MODEL": "cloudru@work::GigaChat-2-Max"}) == [], "one catalog for every connection"
    rouble.pop("CLOUDRU_FOUNDATION_MODELS_API_KEY")
    assert known_spend_gaps({**rouble, "OUROBOROS_MODEL": "cloudru@work::GigaChat-2-Max"}) == ["all model roles"], (
        "the catalog is read with the built-in key")


def test_a_catalog_kind_on_a_private_endpoint_is_named_as_not_counted():
    """A Cloud.ru route resolved to another base bills only what that endpoint reports
    (``llm_routing.provider_reported_cost_only``), as its sends are priced: the public catalog never counts it."""
    from ouroboros.provider_models import CONNECTIONS_KEY

    rouble = {"OUROBOROS_RUB_USD_RATE": "0.012", "CLOUDRU_FOUNDATION_MODELS_API_KEY": "k", "OUROBOROS_WEBSEARCH_BACKEND": "ddgs"}
    private = "https://private.invalid/v1"
    assert known_spend_gaps({**rouble, "OUROBOROS_MODEL": "cloudru::GigaChat-2-Max"}) == []
    assert known_spend_gaps({**rouble, "OUROBOROS_MODEL": "cloudru::GigaChat-2-Max",
                             "CLOUDRU_FOUNDATION_MODELS_BASE_URL": private}) == ["all model roles"]
    rows = [{"kind": "cloudru", "name": "work", "api_key": "row", "base_url": private},
            {"kind": "cloudru", "name": "public", "api_key": "row"}]
    assert known_spend_gaps({**rouble, CONNECTIONS_KEY: rows, "OUROBOROS_MODEL": "cloudru@work::GigaChat-2-Max"}) == [
        "all model roles"]
    assert known_spend_gaps({**rouble, CONNECTIONS_KEY: rows, "OUROBOROS_MODEL": "cloudru@public::GigaChat-2-Max"}) == []


# --- processing tiers: the transport each API row rides, the engine's own words, one route's endpoints ---------

def test_each_api_row_carries_its_own_connections_transport_tiers(monkeypatch):
    """The tiers come from the resolved target of the row's own connection, never from another key's or a guess:
    a transport that says nothing (None) leaves the field out; the local adapter offers ordinary service."""
    from ouroboros import llm_attempt
    from ouroboros.provider_models import CONNECTIONS_KEY

    seen = []

    def transport(target):
        seen.append((target["provider"], target.get("base_url"), target.get("api_surface")))
        return {"openrouter": ("standard", "economy", "fast", "ultrafast"),
                "openai": ("standard", "economy", "fast"), "local": ("standard",)}.get(target["provider"])

    monkeypatch.setattr(llm_attempt, "processing_transport_modes", transport, raising=False)
    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: {
        "OPENROUTER_API_KEY": "or-key", "ANTHROPIC_API_KEY": "an-key", "LOCAL_MODEL_SOURCE": "org/repo",
        CONNECTIONS_KEY: [{"kind": "openai", "name": "work", "api_key": "oa-key"}]})

    async def fake_openrouter(_client, _api_key, _cache_ttl=""):
        return [model_catalog_api._build_model_catalog_entry("openrouter", "OpenAI", "openai/gpt-x", "GPT X",
                                                             source="OpenRouter")]

    async def fake_anthropic(_client, _api_key, **_kwargs):
        return [model_catalog_api._build_model_catalog_entry("anthropic", "Anthropic", "claude-x", "Claude X")]

    async def fake_compatible(_client, provider_id, provider_label, _api_key, _base_url, **_kwargs):
        return [model_catalog_api._build_model_catalog_entry(provider_id, provider_label, "gpt-x", "gpt-x")]

    monkeypatch.setattr(model_catalog_api, "_fetch_openrouter_model_catalog", fake_openrouter)
    monkeypatch.setattr(model_catalog_api, "_fetch_anthropic_model_catalog", fake_anthropic)
    monkeypatch.setattr(model_catalog_api, "_fetch_openai_compatible_model_catalog", fake_compatible)
    payload = json.loads(asyncio.run(model_catalog_api.api_model_catalog(None)).body)
    rows = {item["value"]: item for item in payload["items"]}
    assert rows["openai/gpt-x"]["processing_transport"] == ["standard", "economy", "fast", "ultrafast"]
    assert rows["openai@work::gpt-x"]["processing_transport"] == ["standard", "economy", "fast"]
    assert "processing_transport" not in rows["anthropic::claude-x"]  # None: unknown, not "none supported"
    assert rows["local::repo"]["processing_transport"] == ["standard"]
    assert ("openrouter", "https://openrouter.ai/api/v1", None) in seen
    # The named OpenAI key's own resolved target: its endpoint and the API surface its row selects.
    assert any(provider == "openai" and base == "https://api.openai.com/v1" and surface
               for provider, base, surface in seen)


def test_named_api_source_keeps_its_transport_even_without_model_rows(monkeypatch):
    from ouroboros.gateway.management_contracts import ModelCatalogResponse
    from ouroboros.gateway.schema import validate_ingress
    from ouroboros.provider_models import CONNECTIONS_KEY

    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: {CONNECTIONS_KEY: [
        {"kind": "openai-compatible", "name": "lab", "label": "Sandbox", "base_url": "https://lab.example/v1"},
        {"kind": "openai-compatible", "name": "offline", "base_url": "https://offline.example/v1"},
        {"kind": "openai-compatible", "name": "disabled", "enabled": False, "base_url": "https://disabled.example/v1"},
    ]})
    monkeypatch.setattr(model_catalog_api, "_subscription_model_catalog",
                        lambda *_: {"items": [], "errors": [], "model_sources": []})

    async def empty(_client, _provider_id, _label, _key, base_url, **_kwargs):
        if base_url == "https://offline.example/v1":
            raise httpx.ConnectError("offline")
        assert base_url == "https://lab.example/v1"
        return []

    monkeypatch.setattr(model_catalog_api, "_fetch_openai_compatible_model_catalog", empty)
    payload = json.loads(asyncio.run(model_catalog_api.api_model_catalog(None)).body)
    assert not validate_ingress(payload, ModelCatalogResponse)
    assert payload["items"] == []
    assert [(error["provider_id"], error["stage"]) for error in payload["errors"]] == [("openai-compatible@offline", "connect")]
    assert payload["api_sources"] == [
        {"id": "openai-compatible@lab", "label": "OpenAI-compatible endpoint · Sandbox", "processing_transport": ["standard"]},
        {"id": "openai-compatible@offline", "label": "OpenAI-compatible endpoint · offline", "processing_transport": ["standard"]},
    ]


@pytest.mark.parametrize("store, expected", [("store-7", "store-7"), (" store-7 ", "store-7"), ("", None), (None, None)])
def test_the_engine_store_is_named_only_as_its_handshake_names_it(account_engine, store, expected):
    gateway, _ = account_engine
    gateway.store_identity = store
    payload = json.loads(asyncio.run(model_catalog_api.api_model_catalog(None)).body)
    assert payload.get("store_id") == expected
    from ouroboros.gateway.management_contracts import ModelCatalogResponse
    from ouroboros.gateway.schema import validate_ingress
    assert not validate_ingress(payload, ModelCatalogResponse)


def test_an_older_engine_paths_rows_keep_the_tiers_it_states(account_engine):
    gateway, accounts = account_engine
    gateway.view = False
    payload = model_catalog_api._subscription_model_catalog("codex")
    rows = {row["id"]: row for row in payload["items"]}
    assert rows["shared"]["processing"] == accounts[0]["catalog"]["models"][0]["processing"]
    assert "processing" not in rows["only-a"]  # nothing stated is nothing invented


# The endpoint records OpenRouter publishes for one model; only ``service_tier`` names a tier.
def _record(tag, prompt="0.000002", completion="0.000008", tier=None, **extra):
    record = {"tag": tag, "model_id": "vendor/tier-model", "status": 0,
              "pricing": {"prompt": prompt, "completion": completion, **extra}}
    if tier != "absent":
        record["service_tier"] = tier
    return record


@pytest.fixture
def public_endpoints(monkeypatch):
    from unittest.mock import MagicMock, patch
    import ouroboros.pricing as pricing

    def forget():
        for cache in (pricing._cached_pricing, pricing._pricing_fetched_at, pricing._pricing_retry_after,
                      pricing._pricing_fetch_in_progress):
            cache.clear()

    forget()
    response = MagicMock()
    response.raise_for_status.return_value = None
    records = [_record("vendor"), _record("vendor/flex", "0.000001", "0.000003", "flex"),
               _record("vendor/fast", "0.000004", "0.000015", "priority", request="0.01")]
    response.json.return_value = {"data": {"id": "vendor/tier-model", "endpoints": records}}
    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: {"OPENROUTER_API_KEY": "or-key"})
    with patch("requests.get", return_value=response) as request:
        yield records, request, response, forget
    forget()


def _processing(value, **query):
    from types import SimpleNamespace

    response = asyncio.run(model_catalog_api.api_model_catalog(
        SimpleNamespace(query_params={"processing_model": value, **query})))
    return response.status_code, json.loads(response.body)


def test_one_routes_tiers_come_from_its_endpoints_own_words(public_endpoints):
    from ouroboros.gateway.management_contracts import ModelProcessingResponse
    from ouroboros.gateway.schema import validate_ingress

    _records, request, _response, _forget = public_endpoints
    status, payload = _processing("vendor/tier-model")
    assert status == 200 and not validate_ingress(payload, ModelProcessingResponse)
    assert payload == {"value": "vendor/tier-model", "processing": {
        "modes": ["standard", "fast", "economy"], "complete": True, "endpoints": [
            {"tag": "vendor", "service_tier": None, "mode": "standard", "price": {"input": 2.0, "output": 8.0}},
            {"tag": "vendor/flex", "service_tier": "flex", "mode": "economy", "price": {"input": 1.0, "output": 3.0}},
            # A per-request fee: the tier stays, the token figure alone would be a false price.
            {"tag": "vendor/fast", "service_tier": "priority", "mode": "fast"}]}}
    request.assert_called_once_with("https://openrouter.ai/api/v1/models/vendor/tier-model/endpoints", timeout=5.0)


def test_only_the_documented_tier_words_are_tiers_and_others_keep_the_list_incomplete(public_endpoints):
    """null, ``flex``, ``priority`` and ``ultrafast`` are the endpoints API's words. Another string is kept as stated
    with no mode; a non-string is no tier and not kept (null would say Standard); an absent field says nothing; an
    Ultra endpoint with a fee is still Ultra, without a price."""
    records, _request, _response, _forget = public_endpoints
    records += [_record("vendor/turbo", tier="absent"), _record("vendor/next", tier="hyper"),
                _record("vendor/plain", tier="standard"), _record("vendor/odd", tier=5),
                _record("vendor/ultra", "0.00001", "0.00004", "ultrafast"),
                _record("vendor/ultra-fee", "0.00001", "0.00004", "ultrafast", request="0.02")]
    _status, payload = _processing("vendor/tier-model")
    processing = payload["processing"]
    assert processing["modes"] == ["standard", "fast", "economy", "ultrafast"] and processing["complete"] is False
    price = {"input": 2.0, "output": 8.0}
    assert processing["endpoints"][3:] == [
        {"tag": "vendor/turbo", "price": price}, {"tag": "vendor/next", "service_tier": "hyper", "price": price},
        {"tag": "vendor/plain", "service_tier": "standard", "price": price}, {"tag": "vendor/odd", "price": price},
        {"tag": "vendor/ultra", "service_tier": "ultrafast", "mode": "ultrafast", "price": {"input": 10.0, "output": 40.0}},
        {"tag": "vendor/ultra-fee", "service_tier": "ultrafast", "mode": "ultrafast"}]


@pytest.mark.parametrize("payload", [
    {"data": {"id": "vendor/tier-model", "endpoints": []}},
    {"data": {"id": "vendor/other", "endpoints": [_record("vendor")]}},
    {"data": {"id": "vendor/tier-model", "endpoints": [{"model_id": "vendor/tier-model"}]}},
    {"data": {}},
])
def test_an_empty_foreign_or_malformed_list_is_unknown_never_standard_only(public_endpoints, payload):
    _records, _request, response, _forget = public_endpoints
    response.json.return_value = payload
    _status, answer = _processing("vendor/tier-model")
    assert answer["processing"] == {"modes": [], "complete": False, "endpoints": []}


@pytest.mark.parametrize("unbound", [None, {"model_id": "vendor/tier-model"},
                                       {**_record("vendor/other"), "model_id": "vendor/foreign"}])
def test_partial_catalog_keeps_bound_endpoint_but_cannot_price_a_pool(public_endpoints, unbound):
    from ouroboros.pricing import estimate_cost_optional, get_pricing, wrapup_screen_bound

    records, request, _response, _forget = public_endpoints
    records[:] = [unbound, _record("vendor/flex", tier="flex")]
    model = "vendor/tier-model"
    expected = {"modes": ["economy"], "complete": False, "endpoints": [
        {"tag": "vendor/flex", "service_tier": "flex", "mode": "economy", "price": {"input": 2.0, "output": 8.0}}]}
    assert _processing(model)[1]["processing"] == expected
    for mode in ("standard", "economy", "fast", "ultrafast", "ordinary"):
        assert estimate_cost_optional(model, 1000, 100, processing_mode=mode, routing_pinned=False) is None
    for native in ("", "default", "flex", "priority", "ultrafast"):
        assert wrapup_screen_bound(model, native, 100, routing_pinned=False) is None
    assert get_pricing(model=model, allow_live_fetch=False) == {}  # no quote of a shared price either
    assert _processing(model)[1]["processing"] == expected  # the single cache retained both facts
    request.assert_called_once()


def test_a_disputed_tag_cannot_attest_a_tier_including_under_only_routing(public_endpoints, monkeypatch):
    records, request, _response, _forget = public_endpoints
    records[:] = [_record("vendor/disputed", tier="ultrafast"), _record("vendor/other", tier="priority"),
                  _record("vendor/disputed", tier="flex"), _record("vendor/disputed")]
    model = "vendor/tier-model"
    whole = _processing(model)[1]["processing"]
    assert whole == {"modes": ["fast"], "complete": False, "endpoints": [
        {"tag": "vendor/other", "service_tier": "priority", "mode": "fast", "price": {"input": 2.0, "output": 8.0}}]}
    for tags, expected in ((["vendor/disputed"], {"modes": [], "endpoints": [], "complete": False}),
                           (["vendor/other", "vendor/disputed"], whole)):
        monkeypatch.setattr(model_catalog_api, "load_settings", lambda: {
            "OPENROUTER_API_KEY": "or-key", "OUROBOROS_OR_PROVIDER": {"only": tags}})
        assert _processing(model)[1]["processing"] == expected
    request.assert_called_once()


def test_a_failed_read_is_unknown_never_standard_only(public_endpoints):
    import requests

    _records, request, _response, _forget = public_endpoints
    request.side_effect = requests.ConnectionError("offline")
    _status, answer = _processing("vendor/tier-model")
    assert answer["processing"] == {"modes": [], "complete": False, "endpoints": []}


@pytest.mark.parametrize("value, settings", [
    ("unknown::private/model", {"OPENROUTER_API_KEY": "or-key"}),  # an opaque id: nothing private leaves
    ("vendor/tier-model", {}),  # no configured OpenRouter connection
    ("openrouter@gone::vendor/tier-model", {"OPENROUTER_API_KEY": "or-key"}),
    ("openai-compatible::vendor/tier-model", {"OPENAI_COMPATIBLE_API_KEY": "k",
                                              "OPENAI_COMPATIBLE_BASE_URL": "https://openrouter.ai/api/v1"}),
    ("anthropic::claude-x", {"ANTHROPIC_API_KEY": "k"}),
    ("claudexor::codex=gpt-x", {"OPENROUTER_API_KEY": "or-key"}),
    ("vendor/tier-model (local)", {"OPENROUTER_API_KEY": "or-key"}),
])
def test_no_other_route_or_base_reads_or_shares_the_public_list(public_endpoints, monkeypatch, value, settings):
    _records, request, _response, _forget = public_endpoints
    _processing("vendor/tier-model")  # the public list is cached for the public route…
    request.reset_mock()
    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: settings)
    status, answer = _processing(value)  # …and never answers another route, base or unconfigured key
    assert status == 200 and answer == {"value": value, "processing": {"modes": [], "complete": False, "endpoints": []}}
    request.assert_not_called()


def test_spellings_of_one_public_route_share_one_read_and_a_pinned_connection_is_not_complete(public_endpoints,
                                                                                              monkeypatch):
    from ouroboros.provider_models import CONNECTIONS_KEY

    _records, request, _response, _forget = public_endpoints
    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: {"OPENROUTER_API_KEY": "or-key", CONNECTIONS_KEY: [
        {"kind": "openrouter", "name": "byok", "api_key": "row-or", "provider_routing": "repro"}]})
    plain = _processing("vendor/tier-model")[1]["processing"]
    assert _processing("openrouter::vendor/tier-model:nitro")[1]["processing"] == plain
    pinned = _processing("openrouter@byok::vendor/tier-model")[1]
    assert pinned["value"] == "openrouter@byok::vendor/tier-model"
    assert pinned["processing"] == {"modes": [], "endpoints": [], "complete": False}
    request.assert_called_once()


@pytest.mark.parametrize("value", ["", "   ", "unknown::private/model", "anthropic::claude-x", "vendor/tier-model (local)"])
def test_an_empty_or_non_public_value_reads_nothing_at_all(public_endpoints, monkeypatch, value):
    _records, request, _response, _forget = public_endpoints
    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: pytest.fail("no settings read for this value"))
    monkeypatch.setattr(model_catalog_api, "_subscription_model_catalog", lambda *_: pytest.fail("no catalog scan"))
    status, answer = _processing(value)
    assert status == 200 and answer == {"value": value, "processing": {"modes": [], "complete": False, "endpoints": []}}
    request.assert_not_called()


def test_global_provider_routing_in_these_settings_leaves_the_list_incomplete_and_ambient_is_not_borrowed(
        public_endpoints, monkeypatch):
    monkeypatch.setenv("OUROBOROS_OR_PROVIDER", '{"order": ["vendor"]}')  # ambient: not the settings asked about
    assert _processing("vendor/tier-model")[1]["processing"]["complete"] is True
    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: {
        "OPENROUTER_API_KEY": "or-key", "OUROBOROS_OR_PROVIDER": '{"order": ["vendor/fast"]}'})
    processing = _processing("vendor/tier-model")[1]["processing"]
    assert processing == {"modes": [], "endpoints": [], "complete": False}


def test_named_connections_only_list_overrides_the_global_provider_pin(public_endpoints, monkeypatch):
    from ouroboros.provider_models import CONNECTIONS_KEY

    _records, request, _response, _forget = public_endpoints
    settings = {"OPENROUTER_API_KEY": "fixture", "OUROBOROS_OR_PROVIDER": {"only": ["vendor/fast"]},
                CONNECTIONS_KEY: [{"kind": "openrouter", "name": "selected", "api_key": "fixture",
                                   "provider_routing": {"only": ["vendor/flex"]}}]}
    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: settings)
    for route, tier, tag in (("vendor/tier-model", "fast", "vendor/fast"),
                             ("openrouter@selected::vendor/tier-model", "economy", "vendor/flex")):
        processing = _processing(route)[1]["processing"]
        assert processing["modes"] == [tier] and not processing["complete"]
        assert [item["tag"] for item in processing["endpoints"]] == [tag]
    request.assert_called_once()


@pytest.mark.parametrize("query", [{"source_id": ""}, {"credential_profile_id": ""}, {"source_id": "codex"}])
def test_the_processing_read_takes_no_catalog_parameter_even_empty(public_endpoints, query):
    status, answer = _processing("vendor/tier-model", **query)
    assert status == 400 and "processing_model" in answer["error"]
