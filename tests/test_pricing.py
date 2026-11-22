"""Route-aware best-effort pricing tests."""

from __future__ import annotations

import queue
from unittest.mock import MagicMock, patch

import pytest

from ouroboros.llm import fetch_cloudru_pricing, fetch_openrouter_pricing
from ouroboros.pricing import (
    PricingSchedule,
    emit_llm_usage_event,
    estimate_cost_optional,
    get_pricing,
    infer_api_key_type,
    infer_model_category,
)


def _forget_prices():
    import ouroboros.pricing as pricing

    pricing._cached_pricing.clear()
    pricing._pricing_fetched_at.clear()
    pricing._pricing_retry_after.clear()
    pricing._pricing_fetch_in_progress.clear()


@pytest.fixture(autouse=True)
def _reset_pricing_cache():
    _forget_prices()
    yield
    _forget_prices()


def test_unknown_direct_model_cost_is_none_and_does_not_query_openrouter():
    with patch("ouroboros.llm.fetch_openrouter_pricing") as fetch:
        assert estimate_cost_optional(
            "openai::future-model", 1_000, 500, provider="openai",
        ) is None
    fetch.assert_not_called()


def test_provider_is_inferred_without_openrouter_fallback():
    with patch("ouroboros.llm.fetch_openrouter_pricing") as fetch:
        assert estimate_cost_optional(
            "openai::future-model", 1_000, 500, provider=None,
        ) is None
    fetch.assert_not_called()


def test_live_catalog_prices_exact_new_openrouter_model():
    with patch(
        "ouroboros.llm.fetch_openrouter_pricing",
        return_value={"openai/gpt-new": (2.0, 0.2, None, 8.0)},
    ) as fetch:
        cost = estimate_cost_optional(
            "openai/gpt-new", 1_000, 500, provider="openrouter",
        )
    assert cost == 0.006
    fetch.assert_called_once_with(timeout_sec=5.0)


def test_similar_model_name_does_not_inherit_prefix_price():
    with patch(
        "ouroboros.llm.fetch_openrouter_pricing",
        return_value={"openai/gpt-new": (2.0, 0.2, None, 8.0)},
    ):
        assert estimate_cost_optional(
            "openai/gpt-new:beta", 1_000, 500, provider="openrouter",
        ) is None


def test_failed_catalog_fetch_has_short_process_local_cooldown():
    with patch("ouroboros.llm.fetch_openrouter_pricing", return_value={}) as fetch:
        assert get_pricing(provider="openrouter") == {}
        assert get_pricing(provider="openrouter") == {}
    fetch.assert_called_once_with(timeout_sec=5.0)


def test_missing_cache_prices_are_not_invented():
    with patch(
        "ouroboros.llm.fetch_openrouter_pricing",
        return_value={"provider/model": (1.0, None, None, 3.0)},
    ):
        assert estimate_cost_optional(
            "provider/model", 1_000, 100, provider="openrouter",
        ) == pytest.approx(0.0013)
        assert estimate_cost_optional(
            "provider/model", 1_000, 100, cache_usage={"cached_tokens": 10},
            provider="openrouter", allow_live_fetch=False,
        ) is None
        assert estimate_cost_optional(
            "provider/model", 1_000, 100, cache_usage={"cache_write_tokens": 10},
            provider="openrouter", allow_live_fetch=False,
        ) is None


def test_cache_heavy_anthropic_usage_keeps_a_nonzero_fresh_input_component():
    """v6.77.0 accounting boundary: `regular_input = prompt_tokens - cached - cache_write`
    only yields the FRESH input when `prompt_tokens` is the OpenAI-semantics TOTAL input.
    Direct Anthropic used to report `input_tokens` alone (cache reads/writes excluded), so
    the subtraction clamped fresh input to 0 and the row understated the prompt."""
    with patch(
        "ouroboros.llm.fetch_openrouter_pricing",
        return_value={"anthropic/claude-x": (3.0, 0.3, 3.75, 15.0)},
    ):
        # Provider row: input_tokens=500, cache_read=9_000, cache_creation=500.
        pre_fix = estimate_cost_optional(
            "anthropic/claude-x", 500, 100,
            cache_usage={"cached_tokens": 9_000, "cache_write_tokens": 500},
            provider="openrouter",
        )
        post_fix = estimate_cost_optional(
            "anthropic/claude-x", 10_000, 100,
            cache_usage={"cached_tokens": 9_000, "cache_write_tokens": 500},
            provider="openrouter",
        )

    cached_and_write = (9_000 * 0.3 + 500 * 3.75) / 1_000_000
    completion = 100 * 15.0 / 1_000_000
    assert pre_fix == pytest.approx(cached_and_write + completion)  # regular_input clamped to 0
    assert post_fix == pytest.approx(500 * 3.0 / 1_000_000 + cached_and_write + completion)
    assert post_fix > pre_fix


def test_prompt_tier_starts_strictly_above_its_threshold(monkeypatch):
    from ouroboros.loop_messages import cached_tariff_text

    row = PricingSchedule((1.0, 0.1, None, 3.0), ((100_000, (2.0, 0.2, None, 5.0)),))
    monkeypatch.setattr("ouroboros.pricing.get_pricing", lambda **_: {"x/model": row})
    assert estimate_cost_optional("x/model", 100_000, 1_000) == pytest.approx(0.103)
    assert estimate_cost_optional("x/model", 100_001, 1_000) == pytest.approx(0.205002)
    assert "$1.00/M in" in cached_tariff_text("x/model", "openrouter", 100_000)
    assert "$2.00/M in" in cached_tariff_text("x/model", "openrouter", 100_001)


def test_openrouter_catalog_accepts_arbitrary_model_family():
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": [{
            "id": "mistralai/brand-new",
            "pricing": {"prompt": "0.000002", "completion": "0.000006"},
        }]
    }
    with patch("requests.get", return_value=response) as request:
        rows = fetch_openrouter_pricing(timeout_sec=5.0)
    assert rows["mistralai/brand-new"] == (2.0, None, None, 6.0)
    request.assert_called_once_with("https://openrouter.ai/api/v1/models", timeout=5.0)


def test_cloudru_requires_explicit_fx_rate(monkeypatch):
    monkeypatch.setenv("CLOUDRU_FOUNDATION_MODELS_API_KEY", "secret")
    monkeypatch.delenv("OUROBOROS_RUB_USD_RATE", raising=False)
    with patch("requests.get") as request:
        assert fetch_cloudru_pricing(timeout_sec=5.0) == {}
    request.assert_not_called()


def test_cloudru_catalog_uses_exact_model_and_explicit_fx(monkeypatch):
    monkeypatch.setenv("CLOUDRU_FOUNDATION_MODELS_API_KEY", "secret")
    monkeypatch.setenv("OUROBOROS_RUB_USD_RATE", "100")
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"data": [{
        "id": "vendor/new-model",
        "metadata": {
            "is_billable": True,
            "prompt_tokens_cost": 100,
            "generated_tokens_cost": 500,
            "cache_read_tokens_cost": None,
            "cache_write_tokens_cost": None,
        },
    }]}
    with patch("requests.get", return_value=response):
        rows = fetch_cloudru_pricing(timeout_sec=5.0)
    assert rows["cloudru/vendor/new-model"] == (1.0, None, None, 5.0)


@pytest.mark.parametrize("provider", ["openai", "gigachat", "anthropic"])
def test_routes_without_automatic_catalog_return_empty(provider):
    assert get_pricing(provider=provider) == {}


def test_nullable_usage_event_does_not_label_unknown_as_estimated():
    events = queue.Queue()
    emit_llm_usage_event(
        events,
        "task",
        "openai::future-model",
        {"prompt_tokens": 3, "completion_tokens": 2},
        None,
        provider="openai",
    )
    event = events.get_nowait()
    assert event["cost"] is None
    assert event["cost_estimated"] is False


def test_provider_reported_zero_cost_remains_known_zero():
    events = queue.Queue()
    emit_llm_usage_event(
        events, "task", "local/model", {"cost": 0}, 0.0, provider="local",
    )
    assert events.get_nowait()["cost"] == 0.0


def test_inference_helpers_keep_route_identity(monkeypatch):
    assert infer_api_key_type("openai::gpt-x") == "openai"
    assert infer_api_key_type("mistralai/model") == "openrouter"
    # Direct MiniMax uses the :: spelling; the slash form is a REAL OpenRouter
    # vendor namespace and must keep routing (and safety-key inference) through
    # the OpenRouter key, unlike cloudru/gigachat which exist nowhere on OpenRouter.
    assert infer_api_key_type("minimax::MiniMax-M3") == "minimax"
    assert infer_api_key_type("minimax/minimax-m2") == "openrouter"
    monkeypatch.setenv("OUROBOROS_MODEL", "mistralai/model")
    assert infer_model_category("mistralai/model") == "main"


_NO_TIER = object()  # the record carries no ``service_tier`` field at all


def _endpoint(tag, prompt="0.000002", completion="0.000008", *, service_tier=None, **pricing):
    return {"tag": tag, "model_id": "vendor/tier-model", "status": 0,
            "supported_parameters": ["temperature"],
            **({} if service_tier is _NO_TIER else {"service_tier": service_tier}),
            "pricing": {"prompt": prompt, "completion": completion, **pricing}}


@pytest.fixture
def endpoint_catalog():
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"data": {"id": "vendor/tier-model", "endpoints": [
        _endpoint("vendor", "0.000002", "0.000008"),
        _endpoint("vendor/flex", "0.000001", "0.000003", service_tier="flex"),
        _endpoint("vendor/fast", "0.000004", "0.000015", service_tier="priority"),
        _endpoint("other/region", "0.000005", "0.000006"),
    ]}}
    with patch("requests.get", return_value=response) as request:
        yield response.json.return_value["data"]["endpoints"], request


@pytest.mark.parametrize("mode,expected", [("default", 0.008), ("priority", 0.0115), ("flex", 0.0025)])
def test_processing_prices_the_complete_eligible_endpoint_pool(endpoint_catalog, mode, expected):
    rows, request = endpoint_catalog
    # Endpoint status and supported_parameters are not tariff eligibility:
    # unhealthy price rows remain covered; service_tier need not be in the list.
    rows[3]["status"] = -2
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode=mode) == pytest.approx(expected)
    request.assert_called_once_with("https://openrouter.ai/api/v1/models/vendor/tier-model/endpoints", timeout=5.0)


@pytest.mark.parametrize("prefix", ["", "openrouter::", "openrouter@team::"], ids=["bare", "builtin", "named"])
def test_a_dotted_router_id_reads_its_own_endpoints_not_its_shared_identity(endpoint_catalog, prefix):
    """``anthropic/claude-opus-5.5`` shares its identity with direct ``claude-opus-5-5``, yet its
    tariffs are OpenRouter's for the slug the route names."""
    rows, request = endpoint_catalog
    slug = "anthropic/claude-opus-5.5"
    request.return_value.json.return_value = {"data": {"id": slug, "endpoints": [{**r, "model_id": slug} for r in rows]}}
    assert estimate_cost_optional(prefix + slug, 1000, 500, processing_mode="fast") == pytest.approx(0.0115)
    request.assert_called_once_with(f"https://openrouter.ai/api/v1/models/{slug}/endpoints", timeout=5.0)


def test_priority_bound_includes_a_more_expensive_standard_fallback(endpoint_catalog):
    rows, _ = endpoint_catalog
    rows[3]["pricing"]["completion"] = "0.0001"
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="priority") == pytest.approx(0.055)
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="flex") == pytest.approx(0.0025)


def test_named_router_connection_keeps_its_identity_after_tiered_settlement(endpoint_catalog, tmp_path, monkeypatch):
    from ouroboros import usage_accounting as ua, usage_store

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(tmp_path / "settings.json"))
    model = "openrouter@team::vendor/tier-model"
    request = ua.AttemptRequest(model=model, provider="openrouter", drive_root=tmp_path,
                                processing_preference="fast", submitted_processing_mode="priority",
                                prompt_tokens_estimate=1000, max_completion_tokens=500)
    attempt = ua.reserve_attempt(request)
    ua.mark_dispatched(attempt)
    ua.settle_attempt(attempt, {"prompt_tokens": 1000, "completion_tokens": 500, "service_tier": "priority"})
    with usage_store.read(tmp_path) as txn:
        row = txn.attempt(attempt.attempt_id)
    assert row["model"] == model and row["provider"] == "openrouter"
    assert row["reservation_upper_bound_usd"] == pytest.approx(0.0115)
    assert row["cost_estimated"] and row["cost_usd"] == pytest.approx(0.0115)


def test_flex_without_a_flex_endpoint_routes_as_the_ordinary_route_does(endpoint_catalog):
    """No Flex endpoint: the call routes normally. Default routing may land on a Priority endpoint (an open-weight
    model's take part, and no catalogue field says which are open-weight), so it bounds it; an explicit Standard
    keeps Priority out."""
    rows, _ = endpoint_catalog
    rows[:] = [row for row in rows if row.get("service_tier") != "flex"]
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="flex") == pytest.approx(0.0115)
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="economy") == pytest.approx(0.0115)
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="default") == pytest.approx(0.008)


def test_flex_admits_what_either_reading_of_its_rule_could_admit(endpoint_catalog):
    """Flex is billed by a Flex endpoint or another no dearer than one by input AND by output. Whether the provider
    compares base rates or this prompt's input-length band, the bound covers both: an endpoint dearer at base but
    cheaper past its threshold is in the pool there, and its large cache-write rate decides the amount."""
    rows, _ = endpoint_catalog
    cheap_past_2000 = [{"min_prompt_tokens": 2000, "prompt": "0.0000005"}]
    rows[:] = [_endpoint("vendor/flex", "0.000001", "0.000003", service_tier="flex", input_cache_write="0.000001"),
               _endpoint("vendor", "0.000002", "0.000003", input_cache_write="0.00002", overrides=cheap_past_2000),
               _endpoint("vendor/fast", "0.000004", "0.000015", service_tier="priority")]
    writes = {"cache_write_tokens": 1000, "prompt_cache_ttl": "5m"}
    # Below the threshold the Standard endpoint is dearer by input under both readings: Flex alone.
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, cache_usage=writes,
                                  processing_mode="economy") == pytest.approx(0.001 + 0.0015)
    # Past it, its band is no dearer: its 20/1M cache write makes it the bound (0.001 + 0.02 + 0.0015).
    assert estimate_cost_optional("vendor/tier-model", 3000, 500, cache_usage=writes,
                                  processing_mode="economy") == pytest.approx(0.0225)
    # The other reading: no dearer at base, dearer in its band; still admitted, billed at its band's 5/1M input.
    rows[1]["pricing"].update(prompt="0.0000005", overrides=[{"min_prompt_tokens": 2000, "prompt": "0.000005"}])
    _forget_prices()
    assert estimate_cost_optional("vendor/tier-model", 3000, 500, cache_usage=writes,
                                  processing_mode="economy") == pytest.approx(0.01 + 0.02 + 0.0015)
    rows[1]["pricing"]["completion"] = "0.0000031"  # dearer by output under both readings: out of the pool
    _forget_prices()
    assert estimate_cost_optional("vendor/tier-model", 3000, 500, cache_usage=writes,
                                  processing_mode="economy") == pytest.approx(0.002 + 0.001 + 0.0015)


def test_an_explicit_ultrafast_endpoint_bounds_only_ultrafast_and_needs_its_price(endpoint_catalog):
    rows, _ = endpoint_catalog
    rows.append(_endpoint("vendor/ultra", "0.00001", "0.00004", service_tier="ultrafast"))
    assert get_pricing(provider="openrouter", model="vendor/tier-model")["vendor/ultra"].source["processing"] == "ultrafast"
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="ultrafast") == pytest.approx(0.03)
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="fast") == pytest.approx(0.0115)
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="standard") == pytest.approx(0.008)
    rows[-1]["pricing"].pop("completion")  # Ultra listed without a price: what Ultrafast may cost is unknown
    _forget_prices()
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="ultrafast") is None
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="fast") == pytest.approx(0.0115)
    # Without an Ultra endpoint, Ultrafast falls back to Priority, then Standard: a priced Standard can bound it.
    rows.pop()
    rows[2]["pricing"]["completion"] = "0.000001"
    _forget_prices()
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="ultrafast") == pytest.approx(0.008)


@pytest.mark.parametrize("word", ["default", "standard", "fast", "ultra-new"])
def test_only_the_documented_tier_words_name_a_tier(endpoint_catalog, word):
    """null, flex, priority and ultrafast are the endpoints API's words; any other string is kept as stated and read
    as no tier, so its price joins every pool rather than being taken for Standard."""
    rows, _ = endpoint_catalog
    rows.append(_endpoint("vendor/odd", "0.00001", "0.00004", service_tier=word))
    source = get_pricing(provider="openrouter", model="vendor/tier-model")["vendor/odd"].source
    assert source["service_tier"] == word and "processing" not in source
    for mode in ("standard", "fast", "economy", "ultrafast"):
        assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode=mode) == pytest.approx(0.03)
    rows[-1]["pricing"].pop("completion")
    _forget_prices()
    for mode in ("standard", "fast", "economy", "ultrafast"):
        assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode=mode) is None


@pytest.mark.parametrize("value", [5, True, ["priority"]])
def test_a_non_string_tier_is_no_tier_and_is_not_kept_as_one(endpoint_catalog, value):
    rows, _ = endpoint_catalog
    rows[0]["service_tier"] = value
    source = get_pricing(provider="openrouter", model="vendor/tier-model")["vendor"].source
    assert "service_tier" not in source and "processing" not in source  # never null: null would say Standard


def test_a_missing_tier_field_is_unknown_but_null_is_standard(endpoint_catalog):
    rows, _ = endpoint_catalog
    rows[:] = [_endpoint("vendor", service_tier=None),
               _endpoint("vendor/fast", "0.00001", "0.00004", service_tier=_NO_TIER)]
    endpoints = get_pricing(provider="openrouter", model="vendor/tier-model")
    assert endpoints["vendor"].source["service_tier"] is None and endpoints["vendor"].source["processing"] == "standard"
    assert "service_tier" not in endpoints["vendor/fast"].source and "processing" not in endpoints["vendor/fast"].source
    # The tag says "fast", the record says nothing: no tier is read from a name, and its price stays in the bound.
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="standard") == pytest.approx(0.03)


@pytest.mark.parametrize("tag", ["vendor/fast", "vendor/turbo", "prime/fast", "vendor/flex"])
def test_a_tag_name_is_never_a_tier(endpoint_catalog, tag):
    rows, _ = endpoint_catalog
    rows[:] = [_endpoint(tag, "0.00001", "0.00004")]  # service_tier null: the record's own word is Standard
    assert get_pricing(provider="openrouter", model="vendor/tier-model")[tag].source["processing"] == "standard"
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="fast") == pytest.approx(0.03)


@pytest.mark.parametrize("route", ["vendor/tier-model:nitro", "vendor/tier-model:floor",
                                   "openrouter::vendor/tier-model:nitro", "openrouter@team::vendor/tier-model"])
def test_pool_shortcuts_and_route_spellings_price_the_models_own_endpoints(endpoint_catalog, route):
    _, request = endpoint_catalog
    assert estimate_cost_optional(route, 1000, 500, processing_mode="priority") == pytest.approx(0.0115)
    request.assert_called_once_with("https://openrouter.ai/api/v1/models/vendor/tier-model/endpoints", timeout=5.0)


@pytest.mark.parametrize("route", ["unknown::private/model", "private/model::x", "~vendor/model",
                                   "vendor/model (local)", "openai-compatible::vendor/tier-model"])
def test_no_private_or_foreign_route_reaches_the_public_endpoint_list(endpoint_catalog, route):
    from ouroboros.llm_pricing import fetch_openrouter_endpoint_pricing, openrouter_endpoint_slug

    _, request = endpoint_catalog
    assert openrouter_endpoint_slug(route) == ""
    assert fetch_openrouter_endpoint_pricing(route) == {}
    assert estimate_cost_optional(route, 1000, 500, provider="openrouter", processing_mode="fast") is None
    request.assert_not_called()


def test_a_missing_potential_tariff_makes_the_bound_unknown(endpoint_catalog):
    rows, _ = endpoint_catalog
    rows[3]["pricing"].pop("completion")
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="priority") is None
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="default") is None
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="flex") == pytest.approx(0.0025)


@pytest.mark.parametrize("mode", ["unknown", "auto", "unsupported"])
def test_unknown_processing_never_uses_an_ordinary_model_price(endpoint_catalog, mode):
    _, request = endpoint_catalog
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode=mode) is None
    request.assert_not_called()


@pytest.mark.parametrize("provider", ["openai", "anthropic", "cloudru"])
def test_direct_processing_does_not_borrow_openrouter_tariffs(endpoint_catalog, provider):
    _, request = endpoint_catalog
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, provider=provider, processing_mode="priority") is None
    request.assert_not_called()


def test_endpoint_tiers_keep_literal_cache_prices_and_original_precision(endpoint_catalog):
    rows, _ = endpoint_catalog
    rows[:] = [_endpoint("vendor", input_cache_read="0.0000001", input_cache_write="0.000003",
        input_cache_write_1h="0.000007", discount=0.9,
        overrides=[{"min_prompt_tokens": 2000, "prompt": "0.000003", "input_cache_write_1h": "0.000009"}])]
    cache = {"cached_tokens": 100, "cache_write_tokens": 500, "prompt_cache_ttl": "1h",
             "cache_write_tokens_by_ttl": {"5m": 200, "1h": 300}}
    assert estimate_cost_optional("vendor/tier-model", 1000, 100, cache_usage=cache,
                                  processing_mode="default") == pytest.approx(0.00431)
    assert estimate_cost_optional("vendor/tier-model", 2000, 100, cache_usage=cache,
                                  processing_mode="default") == pytest.approx(0.00631)
    assert estimate_cost_optional("vendor/tier-model", 2001, 100, cache_usage=cache,
                                  processing_mode="default") == pytest.approx(0.008313)
    schedule = get_pricing(provider="openrouter", model="vendor/tier-model")["vendor"]
    assert schedule.cache_write_1h == 7 and schedule.tiers[0][1].cache_write_1h == 9
    assert schedule.source == {"provider": "openrouter", "model": "vendor/tier-model", "endpoint_tag": "vendor",
        "service_tier": None, "processing": "standard", "price": {"input": 2.0, "output": 8.0, "tiers": [
            {"min_prompt_tokens": 2000, "input": 3.0, "output": 8.0}]},
        "url": "https://openrouter.ai/api/v1/models/vendor/tier-model/endpoints", "status": 0}


def test_endpoint_rows_keep_their_declared_token_limits_beside_the_tariffs(endpoint_catalog):
    """The read that prices a model's endpoints also carries what each declares it accepts. ``max_prompt_tokens``
    caps the input and ``context_length`` is a total; an endpoint that declares neither adds nothing. The refusal
    path reads them through the same cache: one free fetch at a miss, none after, and a route without a catalogue
    declares nothing and fetches nothing."""
    from ouroboros.pricing import declared_token_limits

    rows, request = endpoint_catalog
    rows[0].update(context_length=1_000_000, max_prompt_tokens=983_616)
    rows[1].update(context_length=1_000_000, max_prompt_tokens=None)
    rows[2].update(context_length=262_144)
    limits = declared_token_limits("openrouter", "vendor/tier-model")
    rows[2].update(max_prompt_tokens=1_000_000)  # one number, a total on an earlier row and an input cap here
    monkeypatch_rows = {tag: PricingSchedule((1.0, None, None, 2.0), source={key: row[key] for key in (
        "context_length", "max_prompt_tokens") if row.get(key) is not None}) for tag, row in (("a", rows[0]), ("b", rows[2]))}
    with patch("ouroboros.pricing.get_pricing", lambda **_route: monkeypatch_rows):
        both = declared_token_limits("openrouter", "vendor/tier-model")
    assert both[1_000_000] == ("input", "the endpoint catalogue (b max_prompt_tokens)")  # the input reading sheds less
    assert both[262_144][0] == "total" and both[983_616][0] == "input"
    rows[2].update(max_prompt_tokens=None)
    assert limits == {983_616: ("input", "the endpoint catalogue (vendor max_prompt_tokens)"),
                      1_000_000: ("total", "the endpoint catalogue (vendor context_length)"),
                      262_144: ("total", "the endpoint catalogue (vendor/fast context_length)")}
    assert declared_token_limits("openrouter", "openrouter::vendor/tier-model") == limits  # the route's other spelling
    request.assert_called_once()
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="default") == pytest.approx(0.008)
    assert get_pricing(provider="openrouter", model="vendor/tier-model")["other/region"].source.keys() == {
        "provider", "model", "endpoint_tag", "service_tier", "processing", "price", "url", "status"}  # it declared none
    assert declared_token_limits("anthropic", "claude-fixture") == {} == declared_token_limits("openai-compatible", "vendor/tier-model")
    assert declared_token_limits("openrouter", "vendor/unlisted") == {}  # an endpoint list for another id certifies nothing
    assert request.call_count == 2


def test_missing_exact_hour_cache_rate_does_not_use_the_legacy_multiplier(endpoint_catalog):
    rows, _ = endpoint_catalog
    rows[:] = [_endpoint("vendor", input_cache_write="0.000003")]
    assert estimate_cost_optional("vendor/tier-model", 1000, 100, processing_mode="default",
                                  cache_usage={"cache_write_tokens": 500, "prompt_cache_ttl": "1h"}) is None
    assert estimate_cost_optional("vendor/tier-model", 1000, 100, processing_mode="default",
                                  cache_usage={"cached_tokens": 1}) is None


def test_tiny_endpoint_prices_are_not_rounded_into_free_work(endpoint_catalog):
    rows, _ = endpoint_catalog
    rows[:] = [_endpoint("vendor", "0.000000000123456789", "0.000000000234567891")]
    cost = estimate_cost_optional("vendor/tier-model", 1, 1, processing_mode="default")
    assert cost == pytest.approx(0.000000000358024680, rel=1e-12, abs=0)


@pytest.mark.parametrize("value", [True, "nan", "unknown"])
def test_invalid_endpoint_rates_remain_unknown(endpoint_catalog, value):
    rows, _ = endpoint_catalog
    rows[3]["pricing"]["prompt"] = value
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="default") is None


def test_endpoint_fetch_reuses_existing_cache_owner_without_fetching_all_models(endpoint_catalog):
    _, request = endpoint_catalog
    for mode in ("priority", "flex", "default"):
        assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode=mode) is not None
    request.assert_called_once()
    import ouroboros.pricing as pricing
    assert set(pricing._cached_pricing) == {("openrouter", "vendor/tier-model")}
    assert estimate_cost_optional("vendor/other-model", 1000, 500, processing_mode="priority",
                                  allow_live_fetch=False) is None


def test_expired_endpoint_prices_are_unknown_until_the_shared_cache_refreshes(endpoint_catalog, monkeypatch):
    import ouroboros.pricing as pricing
    rows, request = endpoint_catalog
    clock = [100.0]
    monkeypatch.setattr(pricing.time, "time", lambda: clock[0])
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="priority") is not None
    clock[0] += 21601
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="priority", allow_live_fetch=False) is None
    rows.clear()
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="priority") is None
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="priority") is None
    assert request.call_count == 2


def test_incomplete_endpoint_pool_retries_after_the_existing_short_cooldown(endpoint_catalog, monkeypatch):
    import ouroboros.pricing as pricing

    rows, request = endpoint_catalog
    clock = [100.0]
    monkeypatch.setattr(pricing.time, "time", lambda: clock[0])
    rows.append({"tag": "unbound"})
    model = "vendor/tier-model"
    assert estimate_cost_optional(model, 1000, 500, processing_mode="priority") is None
    rows.pop()
    clock[0] += 29
    assert estimate_cost_optional(model, 1000, 500, processing_mode="priority") is None
    clock[0] += 2
    assert estimate_cost_optional(model, 1000, 500, processing_mode="priority") == pytest.approx(0.0115)
    assert request.call_count == 2


def test_concurrent_endpoint_readers_share_one_inflight_fetch(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from ouroboros import llm_pricing

    entered, release = Event(), Event()
    calls = []
    schedule = PricingSchedule((2.0, None, None, 8.0), source={"service_tier": None, "processing": "standard"})
    def fetch(model, **kwargs):
        calls.append(model)
        entered.set()
        assert release.wait(2)
        return {"vendor": schedule}
    monkeypatch.setattr(llm_pricing, "fetch_openrouter_endpoint_pricing", fetch)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(estimate_cost_optional, "vendor/tier-model", 1000, 500, processing_mode="default")
        try:
            assert entered.wait(2)
            assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="default") is None
        finally:
            release.set()
        assert first.result() == pytest.approx(0.006)
    assert calls == ["vendor/tier-model"]


def test_empty_mode_uses_the_published_price_for_each_cache_tier():
    with patch("ouroboros.llm.fetch_openrouter_pricing", return_value={"vendor/model": (2.0, None, 3.0, 8.0)}):
        assert estimate_cost_optional("vendor/model", 1000, 100,
            cache_usage={"cache_write_tokens": 500, "prompt_cache_ttl": "1h"}) is None
        assert estimate_cost_optional("vendor/model", 1000, 100,
            cache_usage={"cache_write_tokens": 500, "prompt_cache_ttl": "5m"}) == pytest.approx(0.0033)


@pytest.mark.parametrize("charge", [
    {"request": "0.01"}, {"internal_reasoning": "0.00002"},
    {"web_search": "0.01"}, {"image": "0.02"}, {"future_billable_unit": "0.03"},
    {"overrides": [{"min_prompt_tokens": 2000, "image": "0.02"}]},
    {"overrides": [{"min_prompt_tokens": 2000, "request": "0.01"}]},
    {"overrides": [{"min_prompt_tokens": 2000, "internal_reasoning": "0.00002"}]},  # a tier's own reasoning rate
    {"request": "free?"}, {"request": "inf"},  # an unreadable or non-finite fee is no free one
    {"overrides": [{"min_prompt_tokens": 2000, "internal_reasoning": "n/a"}]},
])
def test_a_charge_token_rates_cannot_carry_keeps_the_bound_unknown(endpoint_catalog, charge):
    """A fee or reasoning at its own rate, on the base or any input-length tier, would make the token bound a small
    false price, and the endpoint shows no price. Its published rates still prove it outside Flex's pool."""
    rows, _ = endpoint_catalog
    rows[3]["pricing"].update(charge)
    source = get_pricing(provider="openrouter", model="vendor/tier-model")["other/region"].source
    assert source["unpriced"] is True and "price" not in source and source["processing"] == "standard"  # its tier stays
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="standard") is None
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="economy") == pytest.approx(0.0025)
    rows[3]["pricing"].update(prompt="0.0000005", completion="0.000003")  # now Flex may land on it: unknown
    _forget_prices()
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="economy") is None


@pytest.mark.parametrize("pricing", [
    {"prompt": "0", "completion": "0", "image": "0.01"},  # free by tokens, charged for something else
    {"prompt": "0.000005", "completion": "0.000006",
     "overrides": [{"min_prompt_tokens": 1, "utc_start": "22:00", "prompt": "0.0000001"}]},  # a time window
    {"prompt": "0.000005", "completion": "0.000006", "overrides": ["cheap at night"]},  # a tier that is no object
])
def test_a_schedule_token_rates_cannot_state_prices_nothing_and_shows_no_price(endpoint_catalog, pricing):
    rows, _ = endpoint_catalog
    rows[3]["pricing"] = pricing
    source = get_pricing(provider="openrouter", model="vendor/tier-model")["other/region"].source
    assert source["unpriced"] is True and "price" not in source
    for mode in ("standard", "economy"):  # its Flex eligibility cannot be ruled out either
        assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode=mode) is None


# --- one route's endpoint cache: its exact public slug, never another route's warm rows -------------------------

def test_a_warm_endpoint_list_answers_only_its_own_public_slug(endpoint_catalog):
    _, request = endpoint_catalog
    assert get_pricing(provider="openrouter", model="vendor/tier-model")
    for spelling in ("vendor/tier-model:nitro", "vendor/tier-model:floor", "openrouter::vendor/tier-model",
                     "openrouter@team::vendor/tier-model:nitro"):
        assert get_pricing(provider="openrouter", model=spelling, allow_live_fetch=False)  # one read, shared
    for other in ("Vendor/Tier-Model", "vendor/tier-model-2", "vendor/tier-model (local)", "openai::vendor/tier-model",
                  "unknown::vendor/tier-model", "anthropic::vendor/tier-model"):
        assert get_pricing(provider="openrouter", model=other, allow_live_fetch=False) == {}
    assert estimate_cost_optional("vendor/tier-model (local)", 1000, 500, provider="openrouter",
                                  processing_mode="fast") is None
    request.assert_called_once()


def test_two_similar_public_slugs_are_two_reads(endpoint_catalog):
    """``normalize_model_identity`` folds a dotted and a hyphenated Anthropic id together; their endpoint lists
    (and the dot's direct ``anthropic::`` twin) never share a cache entry."""
    rows, request = endpoint_catalog

    def answer(url, **_kwargs):
        slug = url.split("/models/", 1)[1].rsplit("/endpoints", 1)[0]
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"data": {"id": slug, "endpoints": [{**row, "model_id": slug} for row in rows]}}
        return response

    request.side_effect = answer
    dotted, hyphen = "anthropic/claude-opus-5.5", "anthropic/claude-opus-5-5"
    assert get_pricing(provider="openrouter", model=dotted)
    assert get_pricing(provider="openrouter", model=hyphen, allow_live_fetch=False) == {}
    assert get_pricing(provider="openrouter", model=hyphen)
    assert [call.args[0] for call in request.call_args_list] == [
        f"https://openrouter.ai/api/v1/models/{dotted}/endpoints", f"https://openrouter.ai/api/v1/models/{hyphen}/endpoints"]
    assert get_pricing(provider="anthropic", model="anthropic::claude-opus-5-5") == {}


def test_no_route_without_a_public_slug_reads_or_scans_for_a_tier(endpoint_catalog):
    from ouroboros.pricing import declared_token_limits

    _, request = endpoint_catalog
    with patch("ouroboros.llm.fetch_openrouter_pricing") as scan:
        for route in ("", ":nitro", "unknown::private/model", "vendor/model (local)"):
            for mode in ("fast", "ordinary"):
                assert estimate_cost_optional(route, 1000, 500, provider="openrouter", processing_mode=mode) is None
        assert declared_token_limits("openrouter", "") == {}
    scan.assert_not_called()
    request.assert_not_called()


# --- "ordinary": a public OpenRouter reservation that carries no native tier --------------------------------------

def test_ordinary_fetches_only_the_selected_models_endpoints_when_allowed(endpoint_catalog):
    _, request = endpoint_catalog
    with patch("ouroboros.llm.fetch_openrouter_pricing") as scan:
        assert estimate_cost_optional("vendor/tier-model", 1000, 500, allow_live_fetch=False,
                                      processing_mode="ordinary") is None
        request.assert_not_called()
        for route in ("vendor/tier-model", "vendor/tier-model:nitro"):
            assert estimate_cost_optional(route, 1000, 500, processing_mode="ordinary") == pytest.approx(0.0115)
    scan.assert_not_called()
    request.assert_called_once_with("https://openrouter.ai/api/v1/models/vendor/tier-model/endpoints", timeout=5.0)


def test_ordinary_bounds_every_tier_ultra_included_from_a_fresh_complete_list(endpoint_catalog):
    """A workspace default or preset the host cannot see may serve the call at any tier: the dearest endpoint bounds
    it, at this prompt's band and with every price the call needs, whatever sort shortcut the route names."""
    rows, request = endpoint_catalog
    rows.append(_endpoint("vendor/ultra", "0.0001", "0.0004", service_tier="ultrafast",
                          overrides=[{"min_prompt_tokens": 2000, "prompt": "0.0002"}]))
    assert get_pricing(provider="openrouter", model="vendor/tier-model")  # a priced call read it
    for route in ("vendor/tier-model", "vendor/tier-model:nitro", "vendor/tier-model:floor",
                  "openrouter::vendor/tier-model"):
        assert estimate_cost_optional(route, 1000, 500, processing_mode="ordinary") == pytest.approx(0.3)
    assert estimate_cost_optional("vendor/tier-model", 3000, 500, processing_mode="ordinary") == pytest.approx(0.8)
    # Ultra states no cache-read price: a call reading the cache has no bound.
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, cache_usage={"cached_tokens": 100},
                                  processing_mode="ordinary") is None
    # An explicit tier keeps its own pool: Standard pins Standard, a shortcut's admission included.
    assert estimate_cost_optional("vendor/tier-model:nitro", 1000, 500, processing_mode="default") == pytest.approx(0.008)
    assert estimate_cost_optional("vendor/tier-model:floor", 1000, 500, processing_mode="standard") == pytest.approx(0.008)
    for provider in ("openai", "anthropic", "cloudru", "openai-compatible"):
        assert estimate_cost_optional("vendor/tier-model", 1000, 500, provider=provider, processing_mode="ordinary") is None
    request.assert_called_once()


@pytest.mark.parametrize("change", ["absent tier", "unread tier", "fee", "no price"])
def test_ordinary_is_unknown_while_any_endpoint_is_unread_or_unpriced(endpoint_catalog, change):
    rows, _ = endpoint_catalog
    if change == "absent tier":
        rows[3].pop("service_tier")
    elif change == "unread tier":
        rows[3]["service_tier"] = "hyper"
    elif change == "fee":
        rows[3]["pricing"]["request"] = "0.01"
    else:
        rows[3]["pricing"].pop("completion")
    assert get_pricing(provider="openrouter", model="vendor/tier-model")
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="ordinary") is None


def test_ordinary_never_uses_an_expired_list(endpoint_catalog, monkeypatch):
    import ouroboros.pricing as pricing

    _, request = endpoint_catalog
    clock = [100.0]
    monkeypatch.setattr(pricing.time, "time", lambda: clock[0])
    assert get_pricing(provider="openrouter", model="vendor/tier-model")
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="ordinary") == pytest.approx(0.0115)
    clock[0] += 21601
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="ordinary", allow_live_fetch=False) is None
    request.assert_called_once()


def test_an_empty_mode_keeps_its_model_price_and_adds_no_endpoint_read(endpoint_catalog):
    """The historic reference figure, no bound on the serving endpoint: one model-level catalogue, no per-model read,
    and a shortcut id it does not list stays unpriced."""
    _, request = endpoint_catalog
    with patch("ouroboros.llm.fetch_openrouter_pricing", return_value={"vendor/tier-model": (1.0, None, None, 2.0)}):
        assert estimate_cost_optional("vendor/tier-model", 1000, 500) == pytest.approx(0.002)
        assert estimate_cost_optional("vendor/tier-model:nitro", 1000, 500) is None
    request.assert_not_called()


def test_provider_routing_makes_the_whole_list_the_bound(endpoint_catalog, monkeypatch):
    """An order or preset may admit any endpoint, Ultra included: with routing set only the whole list bounds a call."""
    from ouroboros import llm_routing

    rows, _ = endpoint_catalog
    rows.append(_endpoint("vendor/ultra", "0.00001", "0.00004", service_tier="ultrafast"))
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="priority") == pytest.approx(0.0115)
    monkeypatch.setattr(llm_routing, "_resolve_or_provider", lambda routing=None: {"order": ["vendor/ultra"]})
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="priority") == pytest.approx(0.03)
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="default") == pytest.approx(0.03)


def test_a_candidates_own_routing_fact_outranks_what_the_settings_say_now(endpoint_catalog, monkeypatch):
    """A reservation prices the routing its sealed candidate carries (``routing_pinned``), never a setting read later;
    without a candidate the settings decide. The host's filters (``require_parameters``/``allow_fallbacks``, as the
    ``resilience``/``repro`` presets) keep a tier pool, never a Flex one: dropping every Flex endpoint may route the
    call normally, so the whole list bounds it."""
    from ouroboros import llm_routing
    from ouroboros.pricing import wrapup_screen_bound

    rows, _ = endpoint_catalog
    rows.append(_endpoint("vendor/ultra", "0.00001", "0.00004", service_tier="ultrafast"))
    monkeypatch.setattr(llm_routing, "_resolve_or_provider", lambda routing=None: {"order": ["vendor/ultra"]})
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="priority",
                                  routing_pinned=False) == pytest.approx(0.0115)
    for narrowing in ({"allow_fallbacks": True}, {"allow_fallbacks": False, "require_parameters": True}):
        monkeypatch.setattr(llm_routing, "_resolve_or_provider", lambda routing=None, value=narrowing: dict(value))
        assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="priority") == pytest.approx(0.0115)
        assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="priority",
                                      routing_pinned=True) == pytest.approx(0.03)
        assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="flex") == pytest.approx(0.03)
    monkeypatch.setattr(llm_routing, "_resolve_or_provider", lambda routing=None: {})
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="flex") == pytest.approx(0.0025)
    # The last-fit screen reads the same fact: an output-only tariff, the dearest output of the pool it admits.
    for row in rows:
        row["pricing"]["prompt"] = "0"
    _forget_prices()
    assert wrapup_screen_bound("vendor/tier-model", "priority", 1000, routing_pinned=False) == pytest.approx(0.015)
    assert wrapup_screen_bound("vendor/tier-model", "priority", 1000, routing_pinned=True) == pytest.approx(0.04)


@pytest.fixture(params=["model", "endpoint"])
def load_openrouter_tariff(request, monkeypatch, endpoint_catalog):
    from ouroboros.llm_pricing import openrouter_pricing_rows

    def load(fields):
        mode = "" if request.param == "model" else "default"
        if mode:
            endpoints, _ = endpoint_catalog
            endpoints[:] = [_endpoint("vendor", **fields)]
            row = get_pricing(provider="openrouter", model="vendor/tier-model")["vendor"]
        else:
            rows = openrouter_pricing_rows([{"id": "vendor/tier-model", "pricing": fields}])
            monkeypatch.setattr("ouroboros.pricing.get_pricing", lambda **_: rows)
            row = rows["vendor/tier-model"]
        return row, mode

    return load


def test_catalog_rates_follow_provider_order_per_key_at_each_threshold(load_openrouter_tariff):
    from ouroboros.pricing import cost_class_weights

    row, mode = load_openrouter_tariff({
        "prompt": "0.000002", "completion": "0.000008", "input_cache_read": "0.0000005",
        "input_cache_write": "0.000003", "input_cache_write_1h": "0.00001100049", "overrides": [
            {"min_prompt_tokens": 3000, "prompt": "0.000004", "completion": "0.000010", "input_cache_write_1h": "0.000013"},
            {"min_prompt_tokens": 2000, "completion": "0.000009"},
            {"min_prompt_tokens": 2000, "input_cache_write": "0"},
            {"min_prompt_tokens": 4000, "input_cache_write": "0.000005"},
        ],
    })
    assert tuple(row) == (2, 0.5, 3, 8) and row.cache_write_1h == 11.00049
    assert [(minimum, tuple(tier), tier.cache_write_1h) for minimum, tier in row.tiers] == [
        (2000, (2, 0.5, 0, 9), 11.00049), (3000, (4, 0.5, 0, 9), 13), (4000, (4, 0.5, 5, 9), 13)]
    cache = {"cached_tokens": 100, "cache_write_tokens": 500, "prompt_cache_ttl": "1h",
             "cache_write_tokens_by_ttl": {"5m": 200, "1h": 300}}
    for prompt, expected in [(2000, 0.007550147), (2001, 0.007052147), (3001, 0.014454), (4001, 0.019454)]:
        assert estimate_cost_optional("vendor/tier-model", prompt, 100, cache_usage=cache,
                                      processing_mode=mode) == pytest.approx(expected, rel=1e-12, abs=0)
    weights = cost_class_weights(row, {**cache, "prompt_tokens": 4001, "completion_tokens": 100})
    assert weights == pytest.approx({"cache_write": 0.0049, "cache_read": 0.00005, "output": 0.0009,
                                     "uncached_input": 0.013604})


@pytest.mark.parametrize("base_hour, override", [({}, {}), ({"input_cache_write_1h": "0.000011"}, {"input_cache_write_1h": None})])
def test_a_missing_or_explicitly_unknown_hour_rate_is_not_a_short_rate(load_openrouter_tariff, base_hour, override):
    _, mode = load_openrouter_tariff({
        "prompt": "0.000002", "completion": "0.000008", "input_cache_write": "0.000003", **base_hour,
        "overrides": [{"min_prompt_tokens": 1000, "input_cache_write": "0.000005", **override}],
    })
    for lifetime, expected in [("1h", None), ("5m", pytest.approx(0.002305))]:
        value = estimate_cost_optional("vendor/tier-model", 1001, 0, processing_mode=mode,
                                       cache_usage={"cache_write_tokens": 101, "prompt_cache_ttl": lifetime})
        assert value == expected


def test_unresolved_utc_overrides_only_erase_rates_they_could_change(load_openrouter_tariff):
    row, mode = load_openrouter_tariff({
        "prompt": "0.000002", "completion": "0.000008", "input_cache_read": "0.000001",
        "input_cache_write_1h": "0.000011", "overrides": [
            {"min_prompt_tokens": 1000, "utc_start": 1500, "utc_end": 100, "input_cache_write_1h": "0.000007"},
            {"min_prompt_tokens": 2000, "input_cache_write_1h": "0"},
            {"utc_start": 1500, "utc_end": 100, "input_cache_read": "0.0000005"},
        ],
    })
    assert tuple(row) == (2, None, None, 8)  # time-only rule may change just the listed read rate
    cache = {"cache_write_tokens": 100, "prompt_cache_ttl": "1h"}
    for prompt, expected in [(1000, pytest.approx(0.0037)), (1001, None), (2001, pytest.approx(0.004602))]:
        assert estimate_cost_optional("vendor/tier-model", prompt, 100, cache_usage=cache, processing_mode=mode) == expected
    assert estimate_cost_optional("vendor/tier-model", 1000, 100, cache_usage={"cached_tokens": 100}, processing_mode=mode) is None
    assert estimate_cost_optional("vendor/tier-model", 1001, 100, processing_mode=mode) == pytest.approx(0.002802)


@pytest.mark.parametrize("short_rate,hour_rate,short_tokens,hour_tokens,expected", [
    ("0.000003", "0.000011", 40, 60, 0.00078),
    (None, "0.000011", 0, 100, 0.0011),
    (None, "0", 0, 100, 0.0),
    ("0.000003", None, 100, 0, 0.0003),
    ("0.000003", None, 0, 100, None),
    ("0.000003", None, 40, 60, None),
    (None, "0.000011", 40, 60, None),
    ("0.000003", "unreadable", 100, 0, 0.0003),
    ("0.000003", False, 0, 100, None),
])
def test_only_the_rates_for_reported_writes_are_required(monkeypatch, short_rate, hour_rate, short_tokens, hour_tokens, expected):
    from ouroboros.llm_pricing import openrouter_pricing_rows
    from ouroboros.pricing import cost_class_weights

    rows = openrouter_pricing_rows([{"id": "vendor/model", "pricing": {"prompt": "0.000002", "completion": "0.000008",
        "input_cache_write": short_rate, "input_cache_write_1h": hour_rate}}])
    monkeypatch.setattr("ouroboros.pricing.get_pricing", lambda **_: rows)
    usage = {"prompt_tokens": 100, "completion_tokens": 0, "cached_tokens": 0, "cache_write_tokens": 100,
             "prompt_cache_ttl": "default", "cache_write_tokens_by_ttl": {"5m": short_tokens, "1h": hour_tokens}}
    estimate = estimate_cost_optional("vendor/model", 100, 0, cache_usage=usage)
    weights = cost_class_weights(rows["vendor/model"], usage)
    if expected is None:
        assert estimate is None and weights is None
    else:
        assert estimate == pytest.approx(expected, rel=1e-12, abs=0)
        assert weights["cache_write"] == pytest.approx(expected, rel=1e-12, abs=0)


@pytest.mark.parametrize("ttl,split,hour_rate,expected", [
    ("1h", {"5m": 40}, None, None),
    ("1h", {"5m": 40}, 11, None),
    ("1h", {"1h": 60}, 11, None),
    ("1h", {"5m": True, "1h": 99}, 11, None),
    ("1h", {"5m": 100, "1h": -1}, 11, None),
    ("1h", {"2h": 100}, 11, None),
    ("1h", [], 11, None),
    ("1h", {"5m": 100}, None, 0.0003),
    ("1h", {"5m": 40, "1h": 60}, 11, 0.00078),
    ("1h", {}, 11, 0.0011),
    ("", {"1h": 60}, 11, None),
    ("", {"1h": 60}, None, None),
    ("", {}, 11, 0.0003),
    ("default", None, None, 0.0003),
])
def test_only_absent_bins_allow_the_requested_or_default_write_tier(monkeypatch, ttl, split, hour_rate, expected):
    rate = PricingSchedule((2, None, 3, 8), cache_write_1h=hour_rate)
    monkeypatch.setattr("ouroboros.pricing.get_pricing", lambda **_: {"vendor/model": rate})
    usage = {"cache_write_tokens": 100, "prompt_cache_ttl": ttl, "cache_write_tokens_by_ttl": split}
    estimate = estimate_cost_optional("vendor/model", 100, 0, cache_usage=usage)
    if expected is None:
        assert estimate is None
    else:
        assert estimate == pytest.approx(expected, rel=1e-12, abs=0)


@pytest.mark.parametrize("reported,expected", [
    pytest.param({}, 20e-6, id="absent"),
    pytest.param({"cache_creation_input_tokens": False}, None, id="boolean"),
    pytest.param({"cache_creation_input_tokens": -1}, None, id="negative"),
    pytest.param({"cache_creation_input_tokens": 0}, 20e-6, id="zero"),
])
def test_native_and_raw_estimates_distinguish_absent_and_malformed_writes(monkeypatch, reported, expected):
    from ouroboros.llm import LLMClient

    model = "vendor/model"
    monkeypatch.setattr("ouroboros.pricing.get_pricing", lambda **_: {model: PricingSchedule((1, 0, 3, 0))})
    amount = estimate_cost_optional(model, 20, 0, cache_usage=reported)
    body = {"content": [], "usage": {"input_tokens": 20, "output_tokens": 0, **reported}}
    _, reply = LLMClient(api_key="unused")._normalize_anthropic_response(
        body, {"provider": "anthropic", "usage_model": model, "base_url": "https://api.anthropic.com/v1"})
    for cost in (amount, reply["cost"]):
        assert cost == (None if expected is None else pytest.approx(expected, rel=1e-12, abs=0))


def test_model_rates_keep_precision_at_each_prompt_tier(monkeypatch):
    from ouroboros.llm_pricing import openrouter_catalog_price, openrouter_price_row

    rates = {"prompt": "0.000000000012", "input_cache_read": "0.000000000013",
             "input_cache_write": "0.000000000014", "completion": "0.000000000015"}
    override = {"min_prompt_tokens": 5, "prompt": "0.000000000022", "input_cache_read": "0.000000000023",
                "input_cache_write": "0.000000000024", "completion": "0.000000000025"}
    record = {"id": "vendor/model", "pricing": {**rates, "overrides": [override]}}
    row = openrouter_price_row(record)
    monkeypatch.setattr("ouroboros.pricing.get_pricing", lambda **_: {"vendor/model": row})
    usage = {"cached_tokens": 1, "cache_write_tokens": 1}
    assert estimate_cost_optional("vendor/model", 3, 1, cache_usage=usage) == pytest.approx(54e-12, rel=1e-12, abs=0)
    assert estimate_cost_optional("vendor/model", 5, 1, cache_usage=usage) == pytest.approx(78e-12, rel=1e-12, abs=0)
    assert estimate_cost_optional("vendor/model", 6, 1, cache_usage=usage) == pytest.approx(160e-12, rel=1e-12, abs=0)
    assert openrouter_catalog_price(record) is None  # these prices would print as zero in the picker


@pytest.mark.parametrize("override,completion,expected", [
    pytest.param({"completion": None}, 0, 0.0011, id="unknown-output-rate-explicit-zero-output"),
    pytest.param({"prompt": None}, 0, 0.0011, id="unknown-input-rate-no-uncached-input"),
    pytest.param({"completion": None}, 1, None, id="unknown-output-rate-reported-output"),
    pytest.param({"input_cache_write_1h": None}, 0, None, id="unknown-hour-rate-hourly-writes"),
])
def test_a_rate_is_required_only_for_a_class_the_call_reported(load_openrouter_tariff, override, completion, expected):
    """100 hourly writes at $11/M above a threshold whose override leaves one other rate unknown: the unknown
    rate prices nothing the call reported, so the estimate and the Costs shares are the writes alone."""
    from ouroboros.pricing import cost_class_weights

    row, mode = load_openrouter_tariff({
        "prompt": "0.000002", "completion": "0.000008", "input_cache_write": "0.000003",
        "input_cache_write_1h": "0.000011", "overrides": [{"min_prompt_tokens": 50, **override}]})
    usage = {"prompt_tokens": 100, "completion_tokens": completion, "cached_tokens": 0, "cache_write_tokens": 100,
             "prompt_cache_ttl": "1h", "cache_write_tokens_by_ttl": {"5m": 0, "1h": 100}}
    estimate = estimate_cost_optional("vendor/tier-model", 100, completion, cache_usage=usage, processing_mode=mode)
    weights = cost_class_weights(row, usage)
    if expected is None:
        assert estimate is None and weights is None
    else:
        assert estimate == pytest.approx(expected, rel=1e-12, abs=0)
        assert weights == pytest.approx({"cache_write": expected, "cache_read": 0, "output": 0, "uncached_input": 0})
    assert cost_class_weights(row, {**usage, "completion_tokens": None}) is None  # an absent count stays unknown
