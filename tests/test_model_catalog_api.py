import asyncio
from copy import deepcopy
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

    async def fake_openrouter(_client, _api_key):
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

    monkeypatch.setattr(model_catalog_api, "_provider_specs", lambda settings, **kwargs: [("provider", _loader)])

    response = asyncio.run(model_catalog_api.api_model_catalog(None))
    payload = json.loads(response.body.decode("utf-8"))

    assert payload["items"] == [{"value": "provider::model", "label": "Provider Model"}]
    assert payload["errors"] == []
    assert calls == ["provider"]


def test_model_catalog_classifies_httpx_error_stage(monkeypatch):
    monkeypatch.setattr(model_catalog_api, "load_settings", lambda: {})

    async def _loader(_client):
        raise httpx.ConnectError("network down")

    monkeypatch.setattr(model_catalog_api, "_provider_specs", lambda settings, **kwargs: [("provider", _loader)])

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

        for provider, loader in model_catalog_api._provider_specs({
            "OPENROUTER_API_KEY": "fixture", "ANTHROPIC_API_KEY": "fixture", "DEEPSEEK_API_KEY": "fixture",
            "OPENAI_COMPATIBLE_BASE_URL": "https://compat.invalid/v1",
        }):
            _, items, error, _, _ = await model_catalog_api._load_provider(Client(), provider, loader)
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
    {"prompt": "0.000002", "completion": "0.00001", "overrides": [{"min_prompt_tokens": 200000, "prompt": "0.002"}]},
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


def test_the_shown_price_is_the_catalog_row_fetch_openrouter_pricing_reads():
    from unittest.mock import MagicMock, patch
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
    assert [tier["min_prompt_tokens"] for tier in openrouter_catalog_price(record)["tiers"]] == [100000, 200000]


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


def test_the_limits_name_what_they_cannot_count_and_nothing_where_every_route_has_a_price():
    """Dashboard › Costs says it under the limits: a call on a route without a known price adds nothing to known
    spend (`usage_accounting.paid_admission`), so the limits count without it."""
    from ouroboros.pricing import known_spend_gaps, price_source

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
    # Built-in web search by its first leg: a direct key has no catalog, OpenRouter has one, ddgs is free.
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


def test_the_spend_gaps_follow_the_runtime_lane_of_each_role():
    """Each role's route as the runtime resolves it (route_choice, safety_config, slot_lane_label): a remote role on a
    direct key is named even beside a local Main, and a local one is not."""
    from ouroboros.pricing import known_spend_gaps

    local_main = {"OUROBOROS_MODEL": "qwen3-8b", "USE_LOCAL_MAIN": True, "OUROBOROS_WEBSEARCH_BACKEND": "ddgs"}
    assert known_spend_gaps({**local_main, "OUROBOROS_MODEL_VISION": "openai::gpt-6"}) == ["Vision"]
    assert known_spend_gaps({**local_main, "OUROBOROS_MODEL_SAFETY": "anthropic::claude-haiku-5-5"}) == ["Safety Supervisor"]
    assert known_spend_gaps({**local_main, "OUROBOROS_MODEL_VISION": "openai::gpt-oss-20b (local)"}) == [], (
        "a model stored with its local label runs locally, whatever its prefix")
    # A stored False is a set flag: an empty Consciousness then wakes on Main's model remotely.
    assert known_spend_gaps({**local_main, "OUROBOROS_MODEL": "openai::gpt-6",
                             "USE_LOCAL_CONSCIOUSNESS": False}) == ["Consciousness"]
    # An empty Safety beside an empty Light runs Main's slot and flag, not Light's leftover one.
    assert known_spend_gaps({"OUROBOROS_MODEL": "openai::gpt-6", "USE_LOCAL_LIGHT": True, "OUROBOROS_WEBSEARCH_BACKEND": "ddgs",
                             "OUROBOROS_MODEL_VISION": "anthropic/claude-opus-5.5"}) == [
        "Main", "Light", "Safety Supervisor", "Consciousness"]
    routed = {"OUROBOROS_MODEL": "anthropic/claude-opus-5.5", "OUROBOROS_WEBSEARCH_BACKEND": "ddgs"}
    helper = {"route": {"kind": "api_model", "target_id": "deepseek::deepseek-v4"}}
    assert known_spend_gaps(routed, {"enabled": False, "items": [helper]}) == ["Agents"], (
        "a review names any enabled row with the catalog switched off")
    assert known_spend_gaps(routed, {"enabled": False, "items": [{**helper, "enabled": False}]}) == []
    # An empty Consciousness with its own flag set wakes on that flag alone; Light keeps Main's lane too.
    priced = {"OUROBOROS_MODEL_LIGHT": "anthropic/claude-opus-5.5", "OUROBOROS_MODEL_VISION": "anthropic/claude-opus-5.5"}
    assert known_spend_gaps({**priced, "OUROBOROS_MODEL": "openai::gpt-6", "USE_LOCAL_CONSCIOUSNESS": True,
                             "OUROBOROS_WEBSEARCH_BACKEND": "ddgs"}) == ["Main"]
    # An empty Vision runs Main's lane whatever its own flag says (slot_lane_label).
    assert known_spend_gaps({**priced, "OUROBOROS_MODEL_VISION": "", "OUROBOROS_MODEL": "openai::gpt-6",
                             "USE_LOCAL_CONSCIOUSNESS": True, "USE_LOCAL_VISION": True,
                             "OUROBOROS_WEBSEARCH_BACKEND": "ddgs"}) == ["Main", "Vision"]
    rouble = {"OUROBOROS_RUB_USD_RATE": "0.012", "CLOUDRU_FOUNDATION_MODELS_API_KEY": "k", "OUROBOROS_WEBSEARCH_BACKEND": "ddgs"}
    assert known_spend_gaps({**rouble, "OUROBOROS_MODEL": "cloudru::GigaChat-2-Max"}) == []
    assert known_spend_gaps({**rouble, "OUROBOROS_MODEL": "cloudru@work::GigaChat-2-Max"}) == [], "one catalog for every connection"
    rouble.pop("CLOUDRU_FOUNDATION_MODELS_API_KEY")
    assert known_spend_gaps({**rouble, "OUROBOROS_MODEL": "cloudru@work::GigaChat-2-Max"}) == ["all model roles"], (
        "the catalog is read with the built-in key")
