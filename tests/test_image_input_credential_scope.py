"""#1520: a catalog's image-input statement belongs to the key that read it.

Saved discovery -> durable evidence -> resolver -> Auto send copy, with a synthetic endpoint
whose catalog depends on the key in the request. Another key at the same endpoint reads
unknown (pixels go) without another request; aliases of one key and endpoint share; the
OpenRouter catalog stays public while a refusal keeps its own key; keyless endpoints keep
working. Fixture keys only; no live provider.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import threading
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import capability_evidence as evidence, vision_routing as vision
from ouroboros.gateway import models
from tests.test_saved_catalog_image_input import assert_auto, configure, discover

KEY_A, KEY_B, KEY_C = "fixture-key-alpha", "fixture-key-beta", "fixture-key-gamma"
ENDPOINT = "https://compat.invalid/v1"
SEES = {"architecture": {"input_modalities": ["text", "image"]}}
BLIND = {"architecture": {"input_modalities": ["text"]}}


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    """Saved discovery against an endpoint whose answer depends on the request's key (or path, keyless)."""
    state = SimpleNamespace(answers={}, requests=[], recorded=[], hold=None, root=tmp_path,
                            observed=datetime(2026, 10, 27, tzinfo=timezone.utc), released=threading.Event())
    monkeypatch.setattr(evidence, "canonical_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(vision, "utc_now_iso", lambda: state.observed.isoformat())
    monkeypatch.setattr(evidence, "utc_now", lambda: state.observed)
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "auto")
    monkeypatch.setattr(models, "_subscription_model_catalog", lambda *args: {"items": [], "errors": []})
    original_client, original_record = httpx.AsyncClient, vision.record_catalog_image_input

    def record(*args, **kwargs):  # observes the completion order; the write itself is the real one
        original_record(*args, **kwargs)
        state.recorded.append(kwargs.get("api_key", ""))
        state.released.set()  # the held response may now complete: completion order is forced

    async def respond(request):
        key = request.headers.get("authorization", "").removeprefix("Bearer ")
        state.requests.append(key or request.url.path)
        if state.hold and key == state.hold[1]:  # this response waits until the other one is recorded
            await asyncio.to_thread(state.released.wait, 10)
        return httpx.Response(200, json={"data": deepcopy(state.answers.get(key or request.url.path, []))})

    monkeypatch.setattr(vision, "record_catalog_image_input", record)
    monkeypatch.setattr(models.httpx, "AsyncClient",
                        lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs))
    return state


def _rows(*rows):
    return json.dumps([{"kind": "openai-compatible", **row} for row in rows])


def _store_text(root):
    return evidence._store_path(root).read_text(encoding="utf-8")


@pytest.mark.parametrize("first", ["alpha", "beta"])
def test_two_keys_on_one_endpoint_keep_their_own_statements_in_either_completion_order(gateway, monkeypatch, first):
    configure(monkeypatch, {"OUROBOROS_CONNECTIONS": _rows({"name": "alpha", "api_key": KEY_A, "base_url": ENDPOINT},
                                                           {"name": "beta", "api_key": KEY_B, "base_url": ENDPOINT})})
    gateway.answers = {KEY_A: [{"id": "m", **BLIND}], KEY_B: [{"id": "m", **SEES}]}
    gateway.hold = (KEY_A, KEY_B) if first == "alpha" else (KEY_B, KEY_A)
    discover()
    assert len(gateway.requests) == 2
    assert_auto("openai-compatible@alpha::m", False)  # its own catalog says no: caption up front
    assert_auto("openai-compatible@beta::m", True)    # the other key's yes is not overwritten
    assert gateway.recorded == list(gateway.hold)     # in the forced completion order
    assert KEY_A not in _store_text(gateway.root) and KEY_B not in _store_text(gateway.root)
    assert len(gateway.requests) == 2


def test_a_rotated_key_reads_unknown_without_a_read_while_aliases_of_one_key_share(gateway, monkeypatch):
    settings = {"OPENAI_COMPATIBLE_BASE_URL": ENDPOINT, "OPENAI_COMPATIBLE_API_KEY": KEY_A,
                "OUROBOROS_CONNECTIONS": _rows({"name": "alias", "api_key": KEY_A, "base_url": ENDPOINT + "/"},
                                               {"name": "beta", "api_key": KEY_B, "base_url": ENDPOINT})}
    configure(monkeypatch, settings)
    gateway.answers = {KEY_A: [{"id": "m", **BLIND}], KEY_B: [{"id": "m", **SEES}]}
    discover()
    reads = len(gateway.requests)
    assert reads == 3
    assert_auto("openai-compatible::m", False)
    assert_auto("openai-compatible@alias::m", False)  # same key and endpoint, another name: one fact
    assert_auto("openai-compatible@beta::m", True)

    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", KEY_C)  # the owner replaces the built-in key
    assert_auto("openai-compatible::m", None)  # unknown, so pixels go; never another key's "no"
    assert_auto("openai-compatible@alias::m", False)
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", KEY_A)  # switching back finds its own fresh fact
    assert_auto("openai-compatible::m", False)
    assert len(gateway.requests) == reads

    gateway.observed += timedelta(hours=24, seconds=1)  # the 24-hour freshness is unchanged
    assert_auto("openai-compatible::m", None)
    assert_auto("openai-compatible@beta::m", None)
    assert not any(key in _store_text(gateway.root) for key in (KEY_A, KEY_B, KEY_C))


def test_openrouter_catalog_stays_public_while_its_refusals_keep_their_key(gateway, monkeypatch):
    import requests

    from ouroboros.llm import LLMClient
    from tests._shared import _PROVIDER_CATALOG_PRISTINE, _restore_provider_catalog

    configure(monkeypatch, {"OPENROUTER_API_KEY": KEY_A, "OUROBOROS_CONNECTIONS": json.dumps([
        {"kind": "openrouter", "name": "byok", "api_key": KEY_B, "provider_routing": "repro"}])})
    catalogue = {"data": [{"id": "vendor/blind", **BLIND}, {"id": "vendor/sees", **SEES}]}
    answer = SimpleNamespace(status_code=200, json=lambda: catalogue)
    sent = []
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: sent.append(kwargs.get("headers")) or answer)
    _restore_provider_catalog(LLMClient, _PROVIDER_CATALOG_PRISTINE)
    try:
        LLMClient._read_openrouter_catalogue()  # the real public catalog read and its producer
        assert sent == [None] and gateway.recorded == [""]
        for model in ("vendor/blind", "openrouter::vendor/blind", "openrouter@byok::vendor/blind"):
            assert_auto(model, False)
        assert_auto("openrouter@byok::vendor/sees", True)
        monkeypatch.setenv("OPENROUTER_API_KEY", KEY_C)
        assert_auto("vendor/blind", False)  # a public statement serves any key
    finally:
        _restore_provider_catalog(LLMClient, _PROVIDER_CATALOG_PRISTINE)

    # A refusal observes one authenticated request: another key or connection is another route.
    keyed_c = vision.image_route_key("vendor/blind")
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY_A)
    keyed_a, named = vision.image_route_key("vendor/blind"), vision.image_route_key("openrouter@byok::vendor/blind")
    assert all("|" in key for key in (keyed_a, keyed_c, named))
    assert len({keyed_a, keyed_c, named}) == 3
    target = LLMClient()._resolve_remote_target("vendor/blind")
    assert vision.image_route_scope(target) == vision.image_route_scope(dict(target, api_key=KEY_C))
    assert vision.image_route_scope(target, catalog=False) != vision.image_route_scope(
        dict(target, api_key=KEY_C), catalog=False)


def test_an_unparseable_endpoint_keeps_its_refusal_and_the_caption_choice_moves_on(gateway, monkeypatch):
    """``urlsplit`` cannot parse ``http://[::1``, so that route has no image scope. Its refusal key falls
    back to the model alone (``image_route_key``'s existing boundary) on record and lookup alike, and
    choosing a caption route goes on to another route instead of failing the whole projection."""
    broken, other = "openai-compatible::m", "openai-compatible@beta::m"
    configure(monkeypatch, {"OPENAI_COMPATIBLE_BASE_URL": "http://[::1", "OPENAI_COMPATIBLE_API_KEY": KEY_A,
                            "OUROBOROS_CONNECTIONS": _rows({"name": "beta", "api_key": KEY_B, "base_url": ENDPOINT})})
    with pytest.raises(ValueError):
        vision.image_route_scope({"provider": "openai-compatible", "base_url": "http://[::1", "api_key": KEY_A})
    assert vision.image_route_key(broken) == broken
    assert "|" in vision.image_route_key(other)  # the other route keeps its own scoped key
    usage, digest, facts = {}, "d" * 64, {"status": 400, "message": "image rejected"}
    vision.record_image_refusal(usage, broken, [digest], facts)
    assert list(usage[vision.REFUSED_IMAGES_KEY]) == [broken]
    refused = vision.refusal_check(usage, [digest])
    assert refused(broken) == vision.refusal_words(facts) and refused(other) == ""
    assert vision.choose_image_model([broken], [other], refused=refused) == (
        other, [(broken, vision.refusal_words(facts))])
    assert gateway.requests == []


def test_keyless_endpoints_work_and_path_case_names_another_endpoint(gateway, monkeypatch):
    upper, lower = "https://compat.invalid/Tenant/v1", "https://compat.invalid/tenant/v1"
    configure(monkeypatch, {"OUROBOROS_CONNECTIONS": _rows({"name": "upper", "base_url": upper},
                                                           {"name": "lower", "base_url": lower})})
    gateway.answers = {"/Tenant/v1/models": [{"id": "m", **BLIND}], "/tenant/v1/models": [{"id": "m", **SEES}]}
    discover()
    assert sorted(gateway.requests) == ["/Tenant/v1/models", "/tenant/v1/models"]
    assert_auto("openai-compatible@upper::m", False)
    assert_auto("openai-compatible@lower::m", True)
    scope = vision.image_route_scope
    assert scope({"provider": "openai-compatible", "base_url": "HTTPS://COMPAT.invalid/Tenant/v1/"}) == scope(
        {"provider": "openai-compatible", "base_url": upper})  # scheme and host fold, the path does not
    assert scope({"provider": "openai-compatible", "base_url": upper}) != scope(
        {"provider": "openai-compatible", "base_url": lower})


def test_a_legacy_accountless_record_no_longer_answers_a_keyed_route(gateway, monkeypatch):
    legacy = sha256(json.dumps({"provider": "openai-compatible", "endpoint": ENDPOINT, "surface": "chat.completions",
                                "routing": {}, "account": ""}, sort_keys=True).encode()).hexdigest()[:24]
    assert legacy == vision.image_route_scope({"provider": "openai-compatible", "base_url": ENDPOINT})
    evidence._store_evidence(gateway.root, vision.EVIDENCE_NAMESPACE, legacy, {
        "source": "a record written before #1520", "observed_at": gateway.observed.isoformat(),
        "models": {"m": False}})
    configure(monkeypatch, {"OPENAI_COMPATIBLE_BASE_URL": ENDPOINT, "OPENAI_COMPATIBLE_API_KEY": KEY_A})
    assert_auto("openai-compatible::m", None)  # keyed: unknown, pixels go, nothing is fetched
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "")
    assert_auto("openai-compatible::m", False)  # keyless identity is unchanged, so its record still holds
    assert gateway.requests == []


@pytest.mark.parametrize("explicit", [None, ""])
def test_the_metadata_probe_records_under_the_key_its_request_sent(gateway, monkeypatch, explicit):
    """Resolved before the request (``None``) or explicitly unauthenticated (``""``); a settings change
    while the response is in flight never rebinds the observation."""
    configure(monkeypatch, {"OPENAI_COMPATIBLE_BASE_URL": ENDPOINT, "OPENAI_COMPATIBLE_API_KEY": KEY_A})
    sent = []

    def get(url, headers=None, **_kwargs):
        sent.append(dict(headers or {}))
        sidecar = evidence._store_path(gateway.root).with_name("capability_evidence.json.lock")
        assert not sidecar.exists()  # the network read runs outside any store transaction
        monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", KEY_B)  # the owner saves another key meanwhile
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"data": [{"id": "m", **BLIND}]})

    monkeypatch.setattr(httpx, "get", get)
    evidence._openai_compatible_metadata_window("openai-compatible::m", ENDPOINT, True, explicit,
                                                provider="openai-compatible", evidence_root=gateway.root)
    assert sent == ([{"Authorization": f"Bearer {KEY_A}"}] if explicit is None else [{}])
    assert_auto("openai-compatible::m", None)  # KEY_B never read this catalog
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", KEY_A)
    assert_auto("openai-compatible::m", False if explicit is None else None)
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "")
    assert_auto("openai-compatible::m", None if explicit is None else False)
    assert gateway.recorded == [KEY_A if explicit is None else ""]
