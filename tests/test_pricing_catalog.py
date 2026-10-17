"""Catalog evidence, exact provider identity and nullable direct tariffs."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Event
from unittest.mock import Mock

import pytest

from ouroboros import pricing, pricing_catalog as catalog


@pytest.fixture
def catalog_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    catalog._cache.clear()
    catalog._retry_after.clear()
    catalog._fetching.clear()
    rows = {
        "gpt-realtime-2.1": {"litellm_provider": "openai", "input_cost_per_token": 2e-6,
            "output_cost_per_token": 8e-6, "input_cost_per_audio_token": 10e-6,
            "output_cost_per_audio_token": 20e-6, "input_cost_per_image_token": 5e-6,
            "cache_read_input_token_cost": 0.2e-6, "cache_read_input_audio_token_cost": 1e-6,
            "cache_read_input_image_token_cost": 0.5e-6},
        "gemini/gemini-3.8-live": {"litellm_provider": "gemini", "input_cost_per_token": 1e-6,
            "output_cost_per_token": 3e-6, "input_cost_per_audio_token": 5e-6,
            "output_cost_per_audio_token": 10e-6},
        "claude-example": {"litellm_provider": "anthropic", "input_cost_per_token": 3e-6,
            "output_cost_per_token": 15e-6, "cache_read_input_token_cost": 0.3e-6,
            "cache_creation_input_token_cost": 4e-6, "cache_creation_input_token_cost_above_1hr": 7e-6},
    }
    response = Mock()
    response.content = json.dumps(rows).encode()
    fetch = Mock(return_value=response)
    monkeypatch.setattr("requests.get", fetch)
    yield tmp_path, rows, response, fetch
    catalog._cache.clear()
    catalog._retry_after.clear()
    catalog._fetching.clear()



def test_a_quote_keeps_the_tariff_and_memory_keeps_no_raw_catalog_text(catalog_fixture):
    root, rows, response, _fetch = catalog_fixture
    entry = dict(rows["gpt-realtime-2.1"], max_tokens=32000, supports_audio_input=True, mode="chat",
                 search_context_cost_per_query={"search_context_size_low": 0.03})
    response.content = json.dumps({**rows, "gpt-realtime-2.1": entry}).encode()
    quote = catalog.get_catalog_quote("gpt-realtime-2.1", provider="openai")
    assert quote["entry"] == {key: value for key, value in entry.items()
                              if key not in {"max_tokens", "supports_audio_input", "mode"}}
    (snapshot,) = catalog._cache.values()
    assert "content" not in snapshot and snapshot["entries"]["gpt-realtime-2.1"] == entry
    assert json.loads(json.loads((root / catalog.CACHE_REL).read_text())["content"])["gpt-realtime-2.1"] == entry

def test_fetch_persists_timestamp_digest_and_exact_entry(catalog_fixture):
    root, rows, response, fetch = catalog_fixture
    quote = catalog.get_catalog_quote("openai::gpt-realtime-2.1", provider="openai")
    assert quote["entry"] == rows["gpt-realtime-2.1"]
    assert quote["url"] == catalog.CATALOG_URL
    assert quote["sha256"] == hashlib.sha256(response.content).hexdigest()
    assert datetime.fromisoformat(quote["fetched_at"]).tzinfo is not None
    cache = json.loads((root / catalog.CACHE_REL).read_text())
    assert cache["sha256"] == quote["sha256"]
    assert json.loads(cache["content"]) == rows
    fetch.assert_called_once_with(catalog.CATALOG_URL, timeout=5.0)
    quote["entry"]["input_cost_per_token"] = 42
    catalog._cache.clear()
    again = catalog.get_catalog_quote("gpt-realtime-2.1", provider="openai", allow_live_fetch=False)
    assert again["entry"] == rows["gpt-realtime-2.1"]
    assert fetch.call_count == 1


@pytest.mark.parametrize("provider,model,expected", [
    ("openai", "openai::gpt-realtime-2.1", "gpt-realtime-2.1"),
    ("openai", "openai/gpt-realtime-2.1", "gpt-realtime-2.1"),
    ("google", "google::gemini-3.8-live", "gemini/gemini-3.8-live"),
    ("google", "gemini/gemini-3.8-live", "gemini/gemini-3.8-live"),
    ("anthropic", "anthropic::claude-example", "claude-example"),
    ("openai", "anthropic::gpt-realtime-2.1", None),
    ("openai", "google/gpt-realtime-2.1", None),
    ("openai", "openai::gpt-realtime-2.1-preview", None),
    ("google", "google::gemini-3.8-live-new", None),
    ("openai-compatible", "gpt-realtime-2.1", None),
])
def test_exact_id_and_route_identity(catalog_fixture, provider, model, expected):
    quote = catalog.get_catalog_quote(model, provider=provider)
    assert (quote["catalog_id"] if quote else None) == expected


@pytest.mark.parametrize("identity", ["azure", "vertex_ai", "openrouter", None])
def test_entry_provider_must_match(catalog_fixture, identity):
    _, rows, response, _ = catalog_fixture
    rows["gpt-realtime-2.1"]["litellm_provider"] = identity
    response.content = json.dumps(rows).encode()
    assert catalog.get_catalog_quote("gpt-realtime-2.1", provider="openai") is None


def test_refresh_uses_openrouter_cadence_and_failed_refresh_discards_stale(catalog_fixture, monkeypatch):
    root, _, _, fetch = catalog_fixture
    monkeypatch.setenv("OUROBOROS_PRICING_TTL_SEC", "60")
    monkeypatch.setattr(catalog.time, "time", lambda: 1000.0)
    monkeypatch.setattr(catalog, "utc_now_iso", lambda: datetime.fromtimestamp(1000, timezone.utc).isoformat())
    first = catalog.get_catalog_quote("gpt-realtime-2.1", provider="openai")
    assert first is not None
    monkeypatch.setattr(catalog.time, "time", lambda: 1059.0)
    assert catalog.get_catalog_quote("gpt-realtime-2.1", provider="openai") == first
    monkeypatch.setattr(catalog.time, "time", lambda: 1061.0)
    assert catalog.get_catalog_quote("gpt-realtime-2.1", provider="openai", allow_live_fetch=False) is None
    fetch.side_effect = OSError("offline")
    assert catalog.get_catalog_quote("gpt-realtime-2.1", provider="openai") is None
    assert catalog.get_catalog_quote("gpt-realtime-2.1", provider="openai") is None
    assert fetch.call_count == 2
    assert (root / catalog.CACHE_REL).is_file()
    monkeypatch.setattr(catalog.time, "time", lambda: 1092.0)
    monkeypatch.setattr(catalog, "utc_now_iso", lambda: datetime.fromtimestamp(1092, timezone.utc).isoformat())
    fetch.side_effect = None
    assert catalog.get_catalog_quote("gpt-realtime-2.1", provider="openai")["fetched_at"] != first["fetched_at"]


@pytest.mark.parametrize("payload", [b"bad json", b"[]", b"{}", b'{"gpt-realtime-2.1": []}'])
def test_malformed_catalog_is_unknown(catalog_fixture, payload):
    _, _, response, _ = catalog_fixture
    response.content = payload
    assert catalog.get_catalog_quote("gpt-realtime-2.1", provider="openai") is None


@pytest.mark.parametrize("value", [True, -1, float("inf"), float("nan"), "0.002", None, {}])
def test_malformed_needed_rate_is_unknown(catalog_fixture, value):
    _, rows, response, _ = catalog_fixture
    rows["gpt-realtime-2.1"]["input_cost_per_token"] = value
    response.content = json.dumps(rows).encode()
    assert pricing.estimate_cost_optional("openai::gpt-realtime-2.1", 100, 10) is None


def test_corrupt_cache_is_not_used(catalog_fixture):
    root, _, _, fetch = catalog_fixture
    catalog.get_catalog_quote("gpt-realtime-2.1", provider="openai")
    path = root / catalog.CACHE_REL
    saved = json.loads(path.read_text())
    saved["sha256"] = "f" * 64
    path.write_text(json.dumps(saved))
    catalog._cache.clear()
    assert catalog.get_catalog_quote("gpt-realtime-2.1", provider="openai", allow_live_fetch=False) is None
    assert fetch.call_count == 1


def test_cold_concurrent_fetch_is_single_flight(catalog_fixture, monkeypatch):
    _, _, _, fetch = catalog_fixture
    entered, release = Event(), Event()
    original = catalog._fetch_catalog
    def blocking():
        entered.set()
        assert release.wait(5)
        return original()
    monkeypatch.setattr(catalog, "_fetch_catalog", blocking)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(catalog.get_catalog_quote, "gpt-realtime-2.1", provider="openai")
        try:
            assert entered.wait(5)
            assert catalog.get_catalog_quote("gpt-realtime-2.1", provider="openai") is None
        finally:
            release.set()
        assert pending.result() is not None
    assert fetch.call_count == 1


@pytest.mark.parametrize("model,expected", [("openai::gpt-realtime-2.1", .006),
    ("anthropic::claude-example", .0105), ("google::gemini-3.8-live", .0025)])
def test_other_direct_routes_are_priced(catalog_fixture, model, expected):
    assert pricing.estimate_cost_optional(model, 1000, 500) == pytest.approx(expected)


def test_direct_cache_prices_use_literal_rates_including_hour_tier(catalog_fixture):
    usage = {"cached_tokens": 100, "cache_write_tokens": 400, "prompt_cache_ttl": "1h",
             "cache_write_tokens_by_ttl": {"5m": 100, "1h": 300}}
    assert pricing.estimate_cost_optional("anthropic::claude-example", 1000, 100,
        cache_usage=usage) == pytest.approx(.00553)


def test_missing_hour_rate_does_not_inherit_multiplier(catalog_fixture):
    _, rows, response, _ = catalog_fixture
    del rows["claude-example"]["cache_creation_input_token_cost_above_1hr"]
    response.content = json.dumps(rows).encode()
    assert pricing.estimate_cost_optional("anthropic::claude-example", 1000, 100,
        cache_usage={"cache_write_tokens": 100, "prompt_cache_ttl": "1h"}) is None


def test_prompt_length_tiers_are_exact(catalog_fixture):
    _, rows, response, _ = catalog_fixture
    rows["claude-example"].update(input_cost_per_token_above_200k_tokens=6e-6,
                                 output_cost_per_token_above_200k_tokens=30e-6)
    response.content = json.dumps(rows).encode()
    assert pricing.estimate_cost_optional("anthropic::claude-example", 200_000, 100) == pytest.approx(.6015)
    assert pricing.estimate_cost_optional("anthropic::claude-example", 200_001, 100) == pytest.approx(1.203006)


def test_provider_mismatch_never_gets_a_direct_text_price(catalog_fixture):
    assert pricing.estimate_cost_optional("anthropic::gpt-realtime-2.1", 1000, 500, provider="openai") is None


@pytest.mark.parametrize("mode,known", [("standard", True), ("default", True), ("priority", False), ("flex", False)])
def test_direct_standard_receipts_use_catalog_but_other_tiers_do_not_inherit_it(catalog_fixture, mode, known):
    result = pricing.estimate_cost_optional("openai/gpt-realtime-2.1", 1000, 500, provider="openai", processing_mode=mode)
    assert (result == pytest.approx(.006)) if known else result is None
    if known:
        assert result.quote["catalog_id"] == "gpt-realtime-2.1"


@pytest.mark.parametrize("calls,tariff,mode,known", [(2, .01, "standard", True), (0, None, "standard", True),
    (2, None, "standard", False), (None, .01, "standard", False), (2, .01, "priority", False)])
def test_search_receipt_includes_each_server_call_or_remains_unknown(catalog_fixture, calls, tariff, mode, known):
    root, rows, response, _ = catalog_fixture
    rows["gpt-realtime-2.1"]["search_context_cost_per_query"] = {"search_context_size_low": tariff}
    response.content = json.dumps(rows).encode()
    catalog.get_catalog_quote("gpt-realtime-2.1", provider="openai")
    usage = {"input_tokens": 1000, "output_tokens": 500, "processing": {"observed": mode}}
    cost = catalog.search_response_cost("gpt-realtime-2.1", usage, calls=calls, context_size="low", drive_root=root)
    assert (cost == pytest.approx(.006 + calls * (tariff or 0))) if known else cost is None
    assert usage["cost_coverage"] == ("complete" if known else "partial")
    assert usage["cost_evidence"]["web_search"]["calls"] == calls



@pytest.mark.parametrize("usage,known", [({}, False), ({"input_tokens": 1000}, False),
    ({"input_tokens": 0, "output_tokens": 0}, True), ({"prompt_tokens": 1000, "completion_tokens": 500}, True)])
def test_a_search_receipt_without_its_token_counts_stays_unknown(catalog_fixture, usage, known):
    root = catalog_fixture[0]
    catalog.get_catalog_quote("gpt-realtime-2.1", provider="openai")
    cost = catalog.search_response_cost("gpt-realtime-2.1", usage, calls=0, context_size="low", drive_root=root)
    assert (cost is not None) is known and usage["cost_coverage"] == ("complete" if known else "partial")

def test_openrouter_still_uses_its_own_catalog(catalog_fixture, monkeypatch):
    _, _, _, fetch = catalog_fixture
    monkeypatch.setattr(pricing, "_fetch_live_rows", lambda *args: {"openai/gpt-realtime-2.1": (5, None, None, 10)})
    pricing._cached_pricing.clear()
    pricing._pricing_fetched_at.clear()
    pricing._pricing_retry_after.clear()
    assert pricing.estimate_cost_optional("openai/gpt-realtime-2.1", 1000, 500) == .01
    fetch.assert_not_called()


def test_direct_google_attribution_keeps_route_identity():
    assert pricing.infer_api_key_type("google::gemini-3.8-live") == "google"
    assert pricing.infer_provider_from_model("google::gemini-3.8-live") == "google"
    assert pricing.infer_api_key_type("google/gemini-3.8-live") == "openrouter"


def test_failed_cache_write_returns_unknown(catalog_fixture, monkeypatch):
    _, _, _, fetch = catalog_fixture
    monkeypatch.setattr(catalog, "atomic_write_json", Mock(side_effect=OSError("read-only data root")))
    assert catalog.get_catalog_quote("gpt-realtime-2.1", provider="openai") is None
    assert catalog.get_catalog_quote("gpt-realtime-2.1", provider="openai", allow_live_fetch=False) is None
    assert not catalog._cache
    fetch.assert_called_once()


def test_estimate_carries_the_same_quote_even_when_catalog_changes(catalog_fixture, monkeypatch):
    import copy
    first = catalog.get_catalog_quote("gpt-realtime-2.1", provider="openai")
    changed = copy.deepcopy(first)
    changed["entry"]["input_cost_per_token"] = 1
    changed["sha256"] = "f" * 64
    lookup = Mock(side_effect=[first, changed])
    monkeypatch.setattr(catalog, "get_catalog_quote", lookup)
    estimate = pricing.estimate_cost_optional("openai::gpt-realtime-2.1", 100, 10)
    assert estimate == pytest.approx(.00028)
    assert estimate.quote == first
    assert copy.deepcopy(estimate).quote == first
    assert json.loads(json.dumps({"cost": estimate}))["cost"] == pytest.approx(.00028)
    assert lookup.call_count == 1
