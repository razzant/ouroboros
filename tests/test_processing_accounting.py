"""Processing qualifiers survive reservation, native observations and settlement."""

import copy

import pytest

from ouroboros import usage_accounting as ua
from ouroboros._usage_response import processing_receipt, usage_from_response
from tests.test_usage_accounting import data_root as _usage_data_root
from tests._usage_store_testing import ledger_rows

data_root = _usage_data_root


def rows(root):
    return ledger_rows(root)


def test_native_mode_reaches_pricing_before_send_and_observation_before_settlement(data_root, monkeypatch):
    priced = []

    def price(*args, **kwargs):
        priced.append(kwargs.get("processing_mode"))
        return {"priority": 2.0, "default": 1.0}.get(kwargs.get("processing_mode"))

    monkeypatch.setattr(ua, "estimate_cost_optional", price)
    request = ua.AttemptRequest("openai/model", "openai", drive_root=data_root,
                                prompt_tokens_estimate=10, max_completion_tokens=20,
                                processing_preference="fast", submitted_processing_mode="priority")

    def send():
        assert priced == ["priority"]
        assert rows(data_root)[0]["reservation_upper_bound_usd"] == 2.0
        return {"service_tier": "default", "usage": {"prompt_tokens": 10, "completion_tokens": 2}}

    ua.execute_physical_attempt(request, send)
    assert priced == ["priority", "default"]
    ledger = rows(data_root)
    assert all(row["processing_preference"] == "fast" for row in ledger)
    assert all(row["submitted_processing_mode"] == "priority" for row in ledger)
    assert ledger[-1]["cost_usd"] == 1.0 and not ledger[-1]["cost_final"]
    assert ledger[-1]["processing"]["observed"] == "standard"
    assert ua.last_physical_attempt_capture().submitted_processing_mode == "priority"


def test_missing_observation_never_borrows_requested_fast_or_ordinary_tariff(data_root, monkeypatch):
    priced = []
    monkeypatch.setattr(ua, "estimate_cost_optional",
                        lambda *args, **kwargs: priced.append(kwargs.get("processing_mode")))
    request = ua.AttemptRequest("openai/model", "openai", drive_root=data_root,
                                processing_preference="fast", submitted_processing_mode="priority")
    ua.execute_physical_attempt(request, lambda: {"usage": {"prompt_tokens": 10, "completion_tokens": 1}})
    assert priced == ["priority", "unknown"]
    assert rows(data_root)[-1]["cost_usd"] is None
    assert rows(data_root)[-1]["processing"]["observed"] == "unknown"


def test_anthropic_fast_cache_does_not_warm_standard_reservation(data_root, monkeypatch):
    from ouroboros._usage_cache_splits import last_task_cache_split

    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *args, **kwargs: 1.0)
    with ua.usage_scope(ua.UsageScope(drive_root=data_root, task_id="cache-task")):
        request = ua.AttemptRequest("anthropic/model", "anthropic", processing_preference="fast",
                                    submitted_processing_mode="fast")
        ua.execute_physical_attempt(request, lambda: {"usage": {
            "input_tokens": 10, "output_tokens": 2, "cache_read_input_tokens": 90, "speed": "fast"}})
        assert last_task_cache_split("cache-task", "anthropic/model", provider="anthropic", processing_mode="fast") == 90
        assert last_task_cache_split("cache-task", "anthropic/model", provider="anthropic", processing_mode="standard") is None


def test_anthropic_capacity_tier_is_not_speed_and_engine_mixed_stays_mixed():
    usage, _, _ = usage_from_response({"usage": {"service_tier": "standard", "speed": "fast"}}, provider="anthropic")
    assert usage["service_tier"] == "standard"
    assert processing_receipt("anthropic", usage)["observed"] == "fast"
    assert processing_receipt("anthropic", {"service_tier": "standard"}) is None
    receipt = {"requested": "fast", "submitted": "fast", "submittedNative": "fast",
               "observed": "mixed", "observedNative": ["fast", "default"],
               "reason": None, "source": "native_session"}
    assert processing_receipt("claudexor", {"processing": receipt}) == receipt


@pytest.mark.parametrize("reported", [False, True])
def test_sdk_default_null_is_distinct_from_a_reported_null(reported):
    from types import SimpleNamespace
    from ouroboros.llm_routing import _reply_dict

    reply = SimpleNamespace(model_fields_set={"service_tier"} if reported else set(),
                            model_dump=lambda: {"service_tier": None, "usage": {}})
    target = {"provider": "openai", "api_surface": "chat_completions"}
    for facts in (_reply_dict(reply, target), usage_from_response(reply)[0]):
        assert ("service_tier" in facts) is reported
        receipt = processing_receipt("openai", facts)
        if reported:
            assert receipt["observed"] == "unknown" and receipt["observedNative"] == []
            assert receipt["submitted"] is None and receipt["source"] == "provider_response"
        else:
            assert receipt is None


@pytest.mark.parametrize("attempts,expected", [
    ([{"usageCost": {"cashUsd": 0, "cashKnowledge": "unknown", "valuationUsd": 5}}], (None, False)),
    ([{"usageCost": {"cashUsd": 2, "cashKnowledge": "estimated"}}, {}], (2.0, True)),
    ([{}, {"usageCost": {"cashUsd": 2, "cashKnowledge": "exact"}}], (2.0, True)),
])
def test_session_actual_cost_components_preserve_unknown_and_cash_without_valuation(attempts, expected):
    from ouroboros.delegate_custody_usage import disclosed_spend

    assert disclosed_spend({"spendUsd": 99}, attempt_execution=attempts) == expected


def test_session_processing_and_components_are_retained_once_without_repricing(data_root):
    from ouroboros.delegate_custody_usage import disclosed_spend

    evidence = [{"attemptId": "a1", "harnessId": "claude",
                 "processing": {"requested": "fast", "submitted": "fast", "submittedNative": "fast",
                                "observed": "mixed", "observedNative": ["fast", "standard"],
                                "reason": None, "source": "native_telemetry"},
                 "processingCostBasis": {"nativeMode": "fast", "kind": "paid_credits", "source": "native_policy"},
                 "usageCost": {"cashUsd": 0, "valuationUsd": 5, "unknownUsd": 2,
                               "cashKnowledge": "unknown", "valuationKnowledge": "exact"}}]
    original = copy.deepcopy(evidence)
    spend, estimated = disclosed_spend({}, attempt_execution=evidence)
    ua.record_subscription_session("session", drive_root=data_root, route="claude", spend_usd=spend,
                                   spend_estimated=estimated, attempt_execution=evidence)
    before = ledger_rows(data_root)
    evidence[0]["usageCost"]["cashUsd"] = 90
    ua.record_subscription_session("session", drive_root=data_root, route="claude", spend_usd=90,
                                   attempt_execution=evidence)
    assert ledger_rows(data_root) == before
    assert rows(data_root)[-1]["attempt_execution"] == original
    assert rows(data_root)[-1]["cost_usd"] is None
    assert ua.usage_projection(data_root)["unknown_unmetered"] == 1



def test_review_wave_prices_each_captured_processing_choice_before_dispatch(data_root, monkeypatch):
    monkeypatch.setenv("OUROBOROS_PROCESSING_PREFERENCE", "fast")
    seen = []
    monkeypatch.setattr(ua, "_reservation_cost", lambda request: seen.append(
        (request.processing_preference, request.submitted_processing_mode)) or 0.1)
    result = ua.review_wave_admission(data_root, root_task_id="review", models=["openai::same"] * 3,
        prompt_chars=100, remaining_usd_override=1,
        processing_preferences=["standard", "economy", ""])
    assert seen == [("standard", "default"), ("economy", "flex"), ("", "")]
    assert result["fits"] and result["slot_bounds"] == [0.1, 0.1, 0.1]


def test_review_and_reserve_price_the_same_public_carrier_without_a_probe(data_root, monkeypatch):
    from ouroboros.llm import LLMClient
    from ouroboros.llm_attempt import _attempt_request, apply_processing_preference

    slug = "vendor/synthetic-tariff-scope"
    public = LLMClient()
    private = LLMClient(base_url="https://private.invalid/v1")
    calls = []
    monkeypatch.setenv("USE_LOCAL_MAIN", "0")

    def cached_quote(model, *args, processing_mode, allow_live_fetch, **kwargs):
        calls.append((model, processing_mode, allow_live_fetch))
        return {"ordinary": 4.0, "default": 1.0, "ultrafast": 3.0}[processing_mode]

    monkeypatch.setattr(ua, "estimate_cost_optional", cached_quote)
    monkeypatch.setattr("requests.get", lambda *a, **kw: pytest.fail("preflight made a network request"))
    options = dict(root_task_id="scope", models=[slug] * 3, prompt_chars=100, remaining_usd_override=1,
                   processing_preferences=["", "standard", "ultrafast"])
    wave = ua.review_wave_admission(data_root, **options)
    assert wave["slot_bounds"] == [4.0, 1.0, 3.0] and wave["fits"]
    assert calls == [(slug, mode, False) for mode in ("ordinary", "default", "ultrafast")]
    calls.clear()
    for preference, expected in zip(options["processing_preferences"], wave["slot_bounds"]):
        payload = {"model": slug}
        target = {**public.resolve_remote_target(slug), "processing_preference": preference}
        apply_processing_preference(target, payload)
        request = _attempt_request(target, payload)
        assert ua._reservation_cost(request) == expected
        assert request.submitted_processing_mode == payload.get("service_tier", "")
    assert calls == [(slug, "ordinary", False), (slug, "default", True), (slug, "ultrafast", True)]
    calls.clear()
    for resolver in (private.resolve_remote_target, lambda _: None):
        wave = ua.review_wave_admission(data_root, resolve_target_for_model=resolver, **options)
        assert wave["fits"] and wave["slot_bounds"] == [None] * 3
        assert wave["unpriced_slots"] == 3 and calls == []


@pytest.mark.parametrize("native", [None, "", "ordinary", "priority", "ultrafast"])
def test_explicit_native_override_cannot_select_the_omitted_pool(monkeypatch, native):
    from dataclasses import replace
    from ouroboros.llm import LLMClient
    from ouroboros.llm_attempt import _attempt_request, apply_processing_preference

    slug = "vendor/synthetic:variant"
    payload = {"model": slug, "extra_body": {"service_tier": native}}
    target = {**LLMClient().resolve_remote_target(slug), "processing_preference": "standard"}
    apply_processing_preference(target, payload)
    request = _attempt_request(target, payload)
    assert request.model == slug and not request.estimate_ordinary_pool and "service_tier" not in payload
    calls = []
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda model, *a, **kw: calls.append((model, kw["processing_mode"])) or 2.0)
    assert ua._reservation_cost(request) == (2.0 if native in ("priority", "ultrafast") else None)
    assert calls == ([(slug, native)] if native in ("priority", "ultrafast") else [])
    assert ua._reservation_cost(replace(request, provider_reported_cost_only=True, max_budget_usd=5.0)) == 5.0


def test_ordinary_pool_is_never_used_to_price_a_reported_tier():
    from ouroboros._usage_response import observed_processing_mode

    for usage in ({"service_tier": None}, {"service_tier": "ordinary"},
                  {"processing": {"observed": "standard", "observedNative": ["ordinary"]}}):
        assert observed_processing_mode("openrouter", usage) == "unknown"


def test_named_router_cache_writes_use_its_tariff_and_only_its_own_warm_split(monkeypatch):
    from dataclasses import replace
    from ouroboros import pricing

    model = "openrouter@first::anthropic/fixture-1.2"
    rows = {"endpoint": pricing.PricingSchedule((1, 0.5, 4, 2), cache_write_1h=8,
                                                source={"processing": "standard"})}
    calls = []
    monkeypatch.setattr(pricing, "get_pricing", lambda **kw: calls.append(kw) or rows)
    request = ua.AttemptRequest(model, "openrouter", task_id="fixture", prompt_tokens_estimate=1000,
                                prompt_cache_ttl="1h", estimate_ordinary_pool=True)
    ua._reset_task_cache_splits()
    try:
        ua.stash_task_cache_split("fixture", model, 1000, provider="openrouter", ttl_seconds=300)
        assert ua._reservation_cost(request) == pytest.approx(0.0005)
        other = "openrouter@second::anthropic/fixture-1.2"
        assert ua._reservation_cost(replace(request, model=other)) == pytest.approx(0.008)
        ua.stash_task_cache_split("fixture", "anthropic/fixture-1-2", 1000, provider="openrouter", ttl_seconds=300)
        public = "openrouter::anthropic/fixture-1.2"
        assert ua._reservation_cost(replace(request, model=public)) == pytest.approx(0.008)
        rows["endpoint"] = pricing.PricingSchedule((1, 0.5, 4, 2), source={"processing": "standard"})
        assert ua._reservation_cost(replace(request, model=other)) is None
        assert [call["model"] for call in calls] == [model, other, public, other]
    finally:
        ua._reset_task_cache_splits()


def test_private_router_cannot_report_a_free_error_by_reporting_only_zero_tokens(data_root):
    from ouroboros.llm import LLMClient
    from ouroboros.llm_attempt import _attempt_request
    from dataclasses import replace

    client = LLMClient(base_url="https://private.invalid/v1")
    target = client.resolve_remote_target("vendor/synthetic-tariff-scope")
    request = replace(_attempt_request(target, {"model": target["resolved_model"]}), drive_root=data_root)
    ua.execute_physical_attempt(request, lambda: {"error": {"message": "rejected"},
                                                  "usage": {"prompt_tokens": 0, "completion_tokens": 0}})
    [row] = rows(data_root)
    assert row["cost_usd"] is None and row["cost_final"] is False


@pytest.mark.parametrize("observed,modes,expected", [("unknown", [], "unknown"),
    ("mixed", ["fast", "default"], "unknown"), ("standard", ["default"], "default")])
def test_loop_and_helper_display_keep_the_observed_price_qualifier(monkeypatch, observed, modes, expected):
    from types import SimpleNamespace
    from ouroboros import loop_llm_call
    from ouroboros.tools import search

    calls = []
    def price(*args, **kwargs):
        calls.append(kwargs.get("processing_mode"))
        return None if kwargs.get("processing_mode") == "unknown" else 0.1
    monkeypatch.setattr("ouroboros.pricing.estimate_cost_optional", price)
    monkeypatch.setattr(ua, "estimate_cost_optional", price)
    usage = {"cost": None, "provider": "openrouter", "prompt_tokens": 10, "completion_tokens": 2,
             "processing": {"observed": observed, "observedNative": modes}}
    cost, _, _, _ = loop_llm_call._normalize_usage_cost(usage, model="openai/same", use_local=False)
    ctx = SimpleNamespace(pending_events=[], task_metadata={}, task_id="helper")
    # The helper display prices nothing: it states what its attempt settled, and settlement keeps the qualifier.
    attempt = SimpleNamespace(provider="openrouter", model="openai/same", processing_preference="",
                              submitted_processing_mode="", provider_reported_cost_only=False,
                              unpriced_add_on=False)
    settled = ua._settlement_fields(attempt, {**usage, "cost": None}, None, False)
    search._emit_simple_usage(ctx, provider="openrouter", model="openai/same", usage={**usage, "cost": None},
                              settled={"cost": settled["cost_usd"], "cost_estimated": settled["cost_estimated"]})
    assert calls == [expected, expected]
    assert ctx.pending_events[0]["cost"] == cost == (None if expected == "unknown" else 0.1)
