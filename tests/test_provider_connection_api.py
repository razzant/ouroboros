"""Connections in the Settings API, the model catalog and Provider Test (C5, C6).

GET masks row secrets and describes every connection in ``_meta.connections``; a
save validates the rows, rehydrates placeholders by (kind, name) and names the rows
that still point at a connection which cannot serve them. The catalog reads each
connection that can serve, so the same model on two keys is two entries; Provider
Test tests one named connection with that row's draft fields. Keys are fixtures.
"""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any, Dict

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

import ouroboros.gateway.models as models_api
from ouroboros.llm import LLMClient
from ouroboros.provider_models import CONNECTIONS_KEY

BYOK = {"kind": "openrouter", "name": "byok", "label": "Claude credits", "api_key": "sk-or-v1-named-0123456789"}


@pytest.fixture()
def settings_client(tmp_path, monkeypatch):
    import server as srv

    on_disk: Dict[str, Any] = {"OPENROUTER_API_KEY": "sk-or-v1-builtin-0123456789", CONNECTIONS_KEY: [dict(BYOK)]}
    saved: Dict[str, Any] = {}

    def load():
        return copy.deepcopy(on_disk)

    def save(settings, *_a, **_k):
        saved.clear()
        saved.update(settings)
        on_disk.update(settings)

    gateway = srv._gateway_settings
    defaults = lambda settings: (dict(settings), False, [])  # noqa: E731
    for module in (gateway, srv):
        monkeypatch.setattr(module, "load_settings", load)
        monkeypatch.setattr(module, "save_settings", save, raising=False)
        monkeypatch.setattr(module, "_apply_settings_to_env", lambda *a, **k: None)
        monkeypatch.setattr(module, "apply_runtime_provider_defaults", defaults)
    monkeypatch.setattr(gateway, "_owner_read_settings_raw", load)
    monkeypatch.setattr(gateway, "_owner_settings_document", load)
    monkeypatch.setattr(gateway, "_owner_write_settings", save)
    app = Starlette(routes=[Route("/api/settings", endpoint=srv.api_settings_get, methods=["GET"]),
                            Route("/api/settings", endpoint=srv.api_settings_post, methods=["POST"])])
    app.state.drive_root = tmp_path
    app.state.repo_dir = tmp_path
    with TestClient(app) as client:
        yield client, on_disk, saved


def test_get_masks_row_secrets_and_describes_every_connection(settings_client):
    client, _on_disk, _saved = settings_client
    body = client.get("/api/settings").json()
    assert BYOK["api_key"] not in json.dumps(body)
    assert body[CONNECTIONS_KEY] == [{**BYOK, "api_key": "sk-or-v1..."}]
    meta = {row["id"]: row for row in body["_meta"]["connections"]}
    assert meta["openrouter@byok"] == {
        "kind": "openrouter", "name": "byok", "label": "Claude credits", "builtin": False, "enabled": True,
        "configured": True, "source": "settings", "id": "openrouter@byok", "display": "OpenRouter · Claude credits"}
    assert meta["openrouter"]["display"] == "OpenRouter" and meta["openrouter"]["source"] == "settings"
    assert [row["id"] for row in body["_meta"]["connections"]][:2] == ["openrouter", "openrouter@byok"]


def test_reposting_the_masked_rows_keeps_the_saved_key(settings_client):
    client, on_disk, saved = settings_client
    rows = client.get("/api/settings").json()[CONNECTIONS_KEY]
    response = client.post("/api/settings", json={CONNECTIONS_KEY: rows + [
        {"kind": "anthropic", "name": "k2", "api_key": "sk-ant-typed-0123456789"}]})
    assert response.status_code == 200, response.text
    assert saved[CONNECTIONS_KEY] == [BYOK, {"kind": "anthropic", "name": "k2", "api_key": "sk-ant-typed-0123456789"}]
    assert "connection_references" not in response.json()


def test_a_malformed_row_is_refused_before_anything_is_saved(settings_client):
    client, on_disk, saved = settings_client
    response = client.post("/api/settings", json={CONNECTIONS_KEY: [{"kind": "openrouter", "name": "Bad Name"}]})
    assert response.status_code == 400 and "name must be" in response.json()["error"]
    assert saved == {} and on_disk[CONNECTIONS_KEY] == [BYOK]
    response = client.post("/api/settings", json={CONNECTIONS_KEY: [
        {"kind": "openrouter", "name": "default", "enabled": False, "api_key": "k"}]})
    assert response.status_code == 400 and "carries only kind, name and enabled" in response.json()["error"]


def test_the_save_names_rows_left_pointing_at_a_removed_connection(settings_client):
    client, on_disk, _saved = settings_client
    on_disk["OUROBOROS_MODEL"] = "openrouter@byok::anthropic/claude-opus-5.5"
    response = client.post("/api/settings", json={CONNECTIONS_KEY: []})
    assert response.status_code == 200, response.text
    assert response.json()["connection_references"] == [{
        "setting": "OUROBOROS_MODEL", "route": "openrouter@byok::anthropic/claude-opus-5.5",
        "connection": "openrouter@byok", "reason": "missing"}]


def _catalog(monkeypatch, settings):
    seen = []

    async def fake_openrouter(_client, api_key):
        seen.append(("openrouter", api_key))
        return [models_api._build_model_catalog_entry("openrouter", "Anthropic", "anthropic/claude-opus-5.5",
                                                      "Claude Opus 5.5", source="OpenRouter")]

    async def fake_compatible(_client, provider_id, label, api_key, base_url, **_kwargs):
        seen.append((provider_id, api_key, base_url))
        return [models_api._build_model_catalog_entry(provider_id, label, "llama-4", "llama-4")]

    monkeypatch.setattr(models_api, "load_settings", lambda: copy.deepcopy(settings))
    monkeypatch.setattr(models_api, "_subscription_model_catalog", lambda *_a: {"items": [], "errors": [], "model_sources": []})
    monkeypatch.setattr(models_api, "_fetch_openrouter_model_catalog", fake_openrouter)
    monkeypatch.setattr(models_api, "_fetch_openai_compatible_model_catalog", fake_compatible)
    response = asyncio.run(models_api.api_model_catalog(None))
    return json.loads(response.body.decode("utf-8")), seen


def test_the_catalog_lists_the_same_model_once_per_connection(monkeypatch):
    payload, seen = _catalog(monkeypatch, {
        "OPENROUTER_API_KEY": "builtin-key", CONNECTIONS_KEY: [
            dict(BYOK), {"kind": "openai-compatible", "name": "vllm", "base_url": "https://vllm.example/v1"},
            {"kind": "openrouter", "name": "off", "api_key": "off-key", "enabled": False}]})
    items = {item["value"]: item for item in payload["items"]}
    assert set(items) == {"anthropic/claude-opus-5.5", "openrouter@byok::anthropic/claude-opus-5.5",
                          "openai-compatible@vllm::llama-4"}
    named = items["openrouter@byok::anthropic/claude-opus-5.5"]
    assert {key: named[key] for key in ("provider", "provider_label", "provider_id", "label")} == {
        "provider": "openrouter@byok", "provider_label": "OpenRouter · Claude credits",
        "provider_id": "openrouter@byok", "label": "OpenRouter · Claude credits · Claude Opus 5.5"}
    assert items["anthropic/claude-opus-5.5"]["provider"] == "Anthropic"  # built-in items are unchanged
    assert sorted(seen) == sorted([("openrouter", "builtin-key"), ("openrouter", BYOK["api_key"]),
                                   ("openai-compatible", "", "https://vllm.example/v1")])  # its own /models
    from ouroboros.gateway.management_contracts import ModelCatalogResponse
    from ouroboros.gateway.schema import validate_ingress

    assert not validate_ingress(payload, ModelCatalogResponse)  # the published shape carries named items


def _provider_test(payload):
    class _Request:
        async def json(self):
            return payload

    response = asyncio.run(models_api.api_provider_test(_Request()))
    return response.status_code, json.loads(response.body.decode("utf-8"))


def test_provider_test_reaches_the_named_connection_with_its_draft(monkeypatch):
    probes = []

    def probe(self, model, *, settings, timeout=20.0):
        target = self._resolve_remote_target(model, settings=settings)
        probes.append((model, target["api_key"], target["connection"]))
        return {"ok": True, "status_code": 200, "exception_type": ""}

    saved = {"OPENROUTER_API_KEY": "builtin-key", "OUROBOROS_MODEL": "anthropic/claude-opus-5.5",
             CONNECTIONS_KEY: [{**BYOK, "enabled": False}]}
    monkeypatch.setattr(models_api, "load_settings", lambda: copy.deepcopy(saved))
    monkeypatch.setattr(LLMClient, "probe_provider_readiness", probe)
    # The saved, switched-off key: an explicit Test still reaches it; a placeholder keeps the saved value.
    assert _provider_test({"provider_id": "openrouter", "connection": "byok",
                           "overrides": {"api_key": "sk-or-v1..."}}) == (200, {"ok": True})
    # A draft row not saved yet is tested from its fields alone.
    assert _provider_test({"provider_id": "openrouter", "connection": "fresh",
                           "overrides": {"api_key": "draft-key"}}) == (200, {"ok": True})
    # Without a connection the built-in key is tested, as before.
    assert _provider_test({"provider_id": "openrouter"}) == (200, {"ok": True})
    assert probes == [("openrouter@byok::anthropic/claude-opus-5.5", BYOK["api_key"], "byok"),
                      ("openrouter@fresh::anthropic/claude-opus-5.5", "draft-key", "fresh"),
                      ("anthropic/claude-opus-5.5", "builtin-key", "")]
    assert saved[CONNECTIONS_KEY][0]["enabled"] is False  # nothing is saved


@pytest.mark.parametrize("payload,fragment", [
    ({"provider_id": "openrouter", "connection": "Bad Name"}, "connection must be"),
    ({"provider_id": "openrouter", "connection": "byok", "overrides": {"OPENROUTER_API_KEY": "k"}},
     "unsupported override keys"),
    ({"provider_id": "minimax", "connection": "cn", "overrides": {"api_key": "k", "region": "mars"}},
     "unknown MiniMax region"),
])
def test_provider_test_refuses_a_malformed_connection_request(monkeypatch, payload, fragment):
    monkeypatch.setattr(models_api, "load_settings", lambda: {})
    monkeypatch.setattr(LLMClient, "probe_provider_readiness", lambda *a, **k: pytest.fail("must not probe"))
    status, body = _provider_test(payload)
    assert status == 400 and fragment in body["error"]


def test_review_model_warning_reads_a_named_route_by_its_native_id(monkeypatch):
    from ouroboros.gateway import settings as gateway_settings

    monkeypatch.setattr(LLMClient, "openrouter_context_length", classmethod(lambda cls, *_a, **_k: 0))
    monkeypatch.setattr(LLMClient, "_CAPABILITIES_FETCH_OK", True)
    monkeypatch.setattr(LLMClient, "_CONTEXT_LENGTH_CACHE", {"anthropic/claude-opus-5.5": 1000000})
    assert gateway_settings._unrecognised_review_models(
        ["openrouter@byok::anthropic/claude-opus-5.5", "openrouter::anthropic/claude-opus-5.5"]) == []
    assert gateway_settings._unrecognised_review_models(["openrouter@byok::anthropic/claude-opus-5"]) == [
        "openrouter@byok::anthropic/claude-opus-5"]  # the twin: a truncated id is still named


def test_wizard_model_discovery_restores_only_the_selected_connection_secret(monkeypatch):
    from types import SimpleNamespace
    from ouroboros.gateway.secret_rows import mask_connections
    rows = [{"kind": "openai-compatible", "name": name, "base_url": "http://localhost:41001/v1",
             "api_key": f"fixture-{name}-key"} for name in ("alpha", "beta")]
    monkeypatch.setattr(models_api, "load_settings", lambda: {CONNECTIONS_KEY: rows})
    sent = []

    async def discover(client, provider, label, key, url):
        sent.append((key, url))
        return [{"value": "openai-compatible::same-model"}]

    monkeypatch.setattr(models_api, "_fetch_openai_compatible_model_catalog", discover)
    for name, key in (("beta", mask_connections(rows)[1]["api_key"]), ("alpha", ""), ("new", "x"), ("missing", "***set***")):
        async def body():
            return {"connection": name, "baseUrl": "http://localhost:41001/v1", "apiKey": key}
        response = asyncio.run(models_api.api_openai_compatible_models(SimpleNamespace(json=body)))
        assert response.status_code == 200
        assert json.loads(response.body) == {"models": ["same-model"]}
    assert sent == [("fixture-beta-key", "http://localhost:41001/v1"),
                    ("", "http://localhost:41001/v1"), ("x", "http://localhost:41001/v1"), ("", "http://localhost:41001/v1")]


@pytest.mark.parametrize("saved, key, expected", [
    ({"OPENAI_COMPATIBLE_API_KEY": "dedicated-key", "OPENAI_COMPATIBLE_BASE_URL": "https://saved.example/v1"}, "***set***", "dedicated-key"),
    ({"OPENAI_API_KEY": "legacy-key", "OPENAI_BASE_URL": "https://legacy.example/v1"}, "***set***", "legacy-key"),
    ({"OPENAI_COMPATIBLE_API_KEY": "saved-key", "OPENAI_COMPATIBLE_BASE_URL": "https://saved.example/v1"}, "", ""),
])
def test_wizard_discovery_resolves_the_builtin_secret_without_changing_settings(monkeypatch, saved, key, expected):
    from copy import deepcopy
    from types import SimpleNamespace

    # Discovery is explicit even when routing through Default is switched off.
    saved[CONNECTIONS_KEY] = [{"kind": "openai-compatible", "name": "default", "enabled": False}]
    before = deepcopy(saved)
    monkeypatch.setattr(models_api, "load_settings", lambda: dict(saved))
    sent = []

    async def discover(client, provider, label, api_key, url):
        sent.append((api_key, url))
        return [{"value": "openai-compatible::same-model"}]

    url = saved.get("OPENAI_COMPATIBLE_BASE_URL") or saved["OPENAI_BASE_URL"]

    async def body():
        return {"baseUrl": url, "apiKey": key}

    monkeypatch.setattr(models_api, "_fetch_openai_compatible_model_catalog", discover)
    response = asyncio.run(models_api.api_openai_compatible_models(SimpleNamespace(json=body)))
    assert response.status_code == 200 and json.loads(response.body) == {"models": ["same-model"]}
    assert sent == [(expected, url)]
    assert saved == before


@pytest.mark.parametrize("body", [
    {"baseUrl": "https://elsewhere.example/v1", "apiKey": "***set***"},
    {"baseUrl": "https://elsewhere.example/v1", "apiKey": "***set***", "connection": "beta"},
    {"baseUrl": "https://elsewhere.example/v1", "apiKey": "saved-be...", "connection": "beta"},
])
def test_wizard_discovery_never_sends_a_saved_key_to_another_endpoint(monkeypatch, body):
    from types import SimpleNamespace

    # Any page can post here on a loopback install without a password: a placeholder must not
    # carry the saved key to an address it was not saved with. A changed address needs its key again.
    monkeypatch.setattr(models_api, "load_settings", lambda: {
        "OPENAI_COMPATIBLE_API_KEY": "saved-builtin-key", "OPENAI_COMPATIBLE_BASE_URL": "https://saved.example/v1",
        CONNECTIONS_KEY: [{"kind": "openai-compatible", "name": "beta", "base_url": "https://beta.example/v1",
                           "api_key": "saved-beta-key"}]})
    sent = []

    async def discover(client, provider, label, api_key, url):
        sent.append((api_key, url))
        return []

    async def read():
        return body

    monkeypatch.setattr(models_api, "_fetch_openai_compatible_model_catalog", discover)
    response = asyncio.run(models_api.api_openai_compatible_models(SimpleNamespace(json=read)))
    assert response.status_code == 400 and json.loads(response.body)["code"] == "masked_key_endpoint_changed"
    assert sent == []
