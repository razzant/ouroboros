"""Saved discovery -> durable evidence -> Auto send copies, with synthetic HTTP rows.

DeepSeek's optional top-level shape: https://api-docs.deepseek.com/api/list-models/
(read 2026-10-25). Fixtures prove conditional ingestion, not live account availability.
"""

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import capability_evidence as evidence, vision_routing as vision
from ouroboros.gateway import models
from ouroboros.llm import LLMClient
from ouroboros.provider_models import supports_vision


SETTINGS = {
    "openai-compatible": {"OPENAI_COMPATIBLE_BASE_URL": "https://compat.invalid/v1/",
                          "OPENAI_COMPATIBLE_API_KEY": "same-fixture-key"},
    "anthropic": {"ANTHROPIC_API_KEY": "same-fixture-key"},
    "deepseek": {"DEEPSEEK_API_KEY": "same-fixture-key"},
}


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    requests, writes = [], []
    state = SimpleNamespace(rows=[], requests=requests, writes=writes, root=tmp_path,
                            observed=datetime(2026, 10, 25, tzinfo=timezone.utc))
    original_client, original_store = httpx.AsyncClient, evidence._store_evidence
    monkeypatch.setattr(evidence, "canonical_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(vision, "utc_now_iso", lambda: state.observed.isoformat())
    monkeypatch.setattr(evidence, "utc_now", lambda: state.observed)
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "auto")
    monkeypatch.setattr(models, "_subscription_model_catalog", lambda *args: {"items": [], "errors": []})

    def store(*args, **kwargs):
        # The file RMW is off the event loop, including when all providers load concurrently.
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        writes.append((args[1], args[2]))
        return original_store(*args, **kwargs)

    def respond(request):
        requests.append(request)
        assert request.method == "GET" and request.url.path.endswith("/models")
        return httpx.Response(200, json={"data": deepcopy(state.rows)})

    monkeypatch.setattr(evidence, "_store_evidence", store)
    monkeypatch.setattr(models.httpx, "AsyncClient",
                        lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs))
    return state


def configure(monkeypatch, settings):
    monkeypatch.setattr(models, "load_settings", lambda: dict(settings))
    for key, value in settings.items():
        monkeypatch.setenv(key, value)


def discover():
    response = asyncio.run(models.api_model_catalog(None))
    assert response.status_code == 200
    payload = json.loads(response.body)
    assert payload["errors"] == []
    return payload["items"]


def message():
    return [{"role": "user", "content": [{"type": "image_url",
             "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1Pe"
                                  "AAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"},
             "_caption": "Existing caption"}]}]


def assert_auto(model, verdict):
    original = message()
    projected = vision.prepare_messages_for_send(original, routing=vision.VisionRoutingContext(model, object(), {}))
    if verdict is False:
        assert projected[0]["content"][0] == {"type": "text", "text": "[image caption: Existing caption]"}
        assert original[0]["content"][0]["type"] == "image_url"
    else:
        assert projected[0]["content"][0]["type"] == "image_url"
        assert projected[0]["content"][0]["image_url"] == original[0]["content"][0]["image_url"]
    assert supports_vision(model) is verdict


@pytest.mark.parametrize("provider", SETTINGS)
def test_saved_catalog_reaches_auto_without_an_extra_request(catalog, monkeypatch, provider):
    configure(monkeypatch, SETTINGS[provider])
    if provider == "anthropic":
        metadata = lambda supported: {"capabilities": {"image_input": {"supported": supported}}}
    elif provider == "openai-compatible":
        metadata = lambda supported: {"architecture": {"input_modalities": ["text", "image"] if supported else ["text"]}}
    else:
        metadata = lambda supported: {"input_modalities": ["text", "image"] if supported else ["text"]}
    catalog.rows = [{"id": "sees", **metadata(True)}, {"id": "blind", **metadata(False)}, {"id": "unknown"},
                    {"id": "malformed", **metadata(False), "input_modalities": "text"},
                    {"id": "conflict", **metadata(False), "supports_vision": True}]
    items = discover()
    assert all(set(item) == {"provider_id", "provider", "source", "id", "name", "value", "label"} for item in items)
    for model, verdict in (("blind", False), ("sees", True), ("unknown", None), ("malformed", None), ("conflict", None)):
        assert_auto(f"{provider}::{model}", verdict)
    assert len(catalog.requests) == 1 and len(catalog.writes) == 1
    fact = vision.route_image_input(f"{provider}::blind")
    assert fact.source and fact.observed_at == catalog.observed.isoformat()

    # The owner's explicit modes retain their meanings against a saved negative.
    for mode, expected_type in (("inline", "image_url"), ("caption", "text"), ("off", "text")):
        monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", mode)
        projected = vision.prepare_messages_for_send(message(), routing=vision.VisionRoutingContext(
            f"{provider}::blind", object(), {}))
        assert projected[0]["content"][0]["type"] == expected_type
        if mode == "off":
            assert "Existing caption" not in projected[0]["content"][0]["text"]
    assert len(catalog.requests) == 1 and len(catalog.writes) == 1


@pytest.mark.parametrize("provider", SETTINGS)
def test_every_observation_replaces_missing_facts_and_expires_at_24_hours(catalog, monkeypatch, provider):
    configure(monkeypatch, SETTINGS[provider])
    catalog.rows = [{"id": "removed", "input_modalities": ["text"]},
                    {"id": "omitted", "input_modalities": ["text"]},
                    {"id": "kept", "input_modalities": ["text"]}]
    discover()
    assert_auto(f"{provider}::removed", False)
    catalog.observed += timedelta(seconds=1)  # no write throttle, even inside the same second/minute
    catalog.rows = [{"id": "omitted"}, {"id": "kept", "input_modalities": ["text"]}]
    discover()
    assert len(catalog.requests) == 2 and len(catalog.writes) == 2
    assert_auto(f"{provider}::removed", None)
    assert_auto(f"{provider}::omitted", None)
    assert_auto(f"{provider}::kept", False)
    observed = catalog.observed
    monkeypatch.setattr(evidence, "utc_now", lambda: observed + timedelta(hours=24, seconds=1))
    assert_auto(f"{provider}::kept", None)
    assert len(catalog.requests) == 2 and len(catalog.writes) == 2


@pytest.mark.parametrize("field,expected", [
    (["text"], False), (["text", "image"], True), ([], None), (None, None),
    ("text", None), ([False], None), ([""], None), ({"image": True}, None),
])
def test_top_level_modalities_follow_existing_unknown_semantics(catalog, monkeypatch, field, expected):
    configure(monkeypatch, SETTINGS["deepseek"])
    catalog.rows = [{"id": "exact-model", "input_modalities": field}]
    discover()
    assert_auto("deepseek::exact-model", expected)


def test_record_binds_runtime_provider_endpoint_and_surface(catalog, monkeypatch):
    configure(monkeypatch, SETTINGS["openai-compatible"])
    catalog.rows = [{"id": "exact-model", "input_modalities": ["text"]}]
    discover()
    assert_auto("openai-compatible::exact-model", False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture")
    assert_auto("anthropic::exact-model", None)
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://elsewhere.invalid/v1")
    assert_auto("openai-compatible::exact-model", None)
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://compat.invalid/v1")
    # #1520: the statement belongs to the key that read it. Another key at the same endpoint is
    # unknown (pixels go) without a new request; returning to the key that read it restores it.
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "another-fixture-key")
    assert_auto("openai-compatible::exact-model", None)
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", SETTINGS["openai-compatible"]["OPENAI_COMPATIBLE_API_KEY"])
    assert_auto("openai-compatible::exact-model", False)
    assert len(catalog.requests) == 1
    assert "fixture-key" not in evidence._store_path(catalog.root).read_text(encoding="utf-8")


@pytest.mark.parametrize("draft,provider", [
    ("onboarding", "openai-compatible"),
    *(("provider-test", provider) for provider in SETTINGS),
])
def test_drafts_never_write_even_when_they_equal_saved_credentials(catalog, monkeypatch, draft, provider):
    settings = SETTINGS[provider]
    configure(monkeypatch, settings)
    catalog.rows = [{"id": "exact-model", "input_modalities": ["text"]}]
    discover()
    before = evidence._store_path(catalog.root).read_bytes()
    catalog.rows[0]["input_modalities"] = ["text", "image"]
    catalog.observed += timedelta(seconds=1)

    async def body():
        if draft == "onboarding":
            return {"baseUrl": settings["OPENAI_COMPATIBLE_BASE_URL"], "apiKey": settings["OPENAI_COMPATIBLE_API_KEY"]}
        return {"provider_id": provider, "overrides": settings}

    monkeypatch.setattr(LLMClient, "probe_provider_readiness", lambda *a, **k: {"ok": True})
    endpoint = models.api_openai_compatible_models if draft == "onboarding" else models.api_provider_test
    response = asyncio.run(endpoint(SimpleNamespace(json=body)))
    assert response.status_code == 200
    if draft == "provider-test":
        assert json.loads(response.body) == {"ok": True}
    # Anthropic/DeepSeek Test uses maintained defaults, without catalog discovery.
    assert len(catalog.requests) == (2 if provider == "openai-compatible" else 1)
    assert len(catalog.writes) == 1
    assert evidence._store_path(catalog.root).read_bytes() == before
    assert_auto(f"{provider}::exact-model", False)


def test_saved_discovery_does_not_add_unselected_providers_as_sources(catalog, monkeypatch):
    configure(monkeypatch, {"OPENAI_API_KEY": "fixture", "MINIMAX_API_KEY": "fixture",
                           "ZAI_API_KEY": "fixture", "CLOUDRU_FOUNDATION_MODELS_API_KEY": "fixture"})
    catalog.rows = [{"id": "exact-model", "input_modalities": ["text"]}]
    assert len(discover()) == 4
    assert len(catalog.requests) == 4 and not catalog.writes
    assert not evidence._store_path(catalog.root).exists()
