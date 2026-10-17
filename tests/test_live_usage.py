"""Live response receipts exercise the real ledger, scopes and budget limits."""

from dataclasses import FrozenInstanceError
from unittest.mock import Mock

import pytest

from ouroboros import live_usage as live, usage_accounting as accounting
from ouroboros.gateways.realtime_engine import USAGE_KEYS
from tests._usage_store_testing import ledger_rows
from tests.test_pricing_catalog import catalog_fixture  # noqa: F401


@pytest.fixture
def live_root(catalog_fixture, monkeypatch):  # noqa: F811
    root, _, _, _ = catalog_fixture
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    return root


def reserve(root, *, warm=True, **values):
    # The ledger is engine-agnostic; this catalog row is the fixture's one tariff pricing every modality.
    args = dict(engine="openai", model="gpt-realtime-2.1", call_id="call-1", response_key="response-1",
                task_id="call-task", rendered_mode="nano", drive_root=root)
    args.update(values)
    if warm:  # as the driver does on its own thread before the call's loop exists
        live.warm_quote(engine=args["engine"], model=args["model"], drive_root=root)
    return live.reserve_live_response(**args)


def settle(attempt, **usage):
    live.settle_live_response(attempt, usage=usage, raw_usage={}, status="completed")


def test_reserved_response_has_explicit_modality_estimate_and_mode(live_root, catalog_fixture):  # noqa: F811
    root = live_root
    with accounting.capture_attempt_ids() as ids:
        attempt = reserve(root)
    row, = ledger_rows(root)
    assert ids == [row["attempt_id"]]
    assert row["kind"] == "attempt" and row["state"] == "dispatched"
    assert row["reservation_upper_bound_usd"] == pytest.approx(1024 * (2 + 8 + 10 + 20 + 5) / 1e6)
    assert row["rendered_mode"] == "nano"
    assert row["cost_evidence"]["call_id"] == "call-1"
    assert row["cost_evidence"]["response_key"] == "response-1"
    assert row["cost_evidence"]["reservation_usage"] == live.PROVISIONAL_USAGE
    assert row["cost_evidence"]["tariff_quote"]["catalog_id"] == "gpt-realtime-2.1"
    with pytest.raises(FrozenInstanceError):
        attempt._quote = None


def test_all_modalities_and_cached_split_are_priced_once(live_root):
    attempt = reserve(live_root)
    usage = dict(text_input_tokens=100, audio_input_tokens=200, image_input_tokens=50,
                 cached_input_tokens=55, text_output_tokens=30, audio_output_tokens=60)
    raw = {"input_token_details": {"cached_tokens_details": {"text_tokens": 20, "audio_tokens": 30, "image_tokens": 5}},
           "future_usage_field": {"retained": True}}
    live.settle_live_response(attempt, usage=usage, raw_usage=raw, status="cancelled")
    row, = ledger_rows(live_root)
    expected = (80 * 2 + 170 * 10 + 45 * 5 + 20 * .2 + 30 * 1 + 5 * .5 + 30 * 8 + 60 * 20) / 1e6
    assert row["cost_usd"] == pytest.approx(expected)
    assert row["cost_final"] is False and row["cost_estimated"] is True
    assert row["cost_coverage"] == "complete"
    assert row["prompt_tokens"] == 100 and row["completion_tokens"] == 30
    assert row["cost_evidence"]["usage"] == usage
    assert row["cost_evidence"]["raw_usage"] == raw
    assert row["cost_evidence"]["status"] == "cancelled"
    assert row["price_source"].startswith("litellm@")
    # Exact duplicate receipts do not charge twice.
    live.settle_live_response(attempt, usage=usage, raw_usage=raw, status="cancelled")
    assert accounting.usage_projection(live_root)["settled_usd"] == .003562  # six-decimal display projection


def test_gemini_cached_modality_details(live_root, catalog_fixture):  # noqa: F811
    import json
    _, rows, response, _ = catalog_fixture
    rows["gemini/gemini-3.8-live"].update(cache_read_input_token_cost=.1e-6,
                                         cache_read_input_audio_token_cost=.5e-6)
    response.content = json.dumps(rows).encode()
    attempt = reserve(live_root, engine="google", model="gemini-3.8-live")
    raw = {"usageMetadata": {"cacheTokensDetails": [{"modality": "TEXT", "tokenCount": 10},
                                                    {"modality": "AUDIO", "tokenCount": 20}]}}
    live.settle_live_response(attempt, usage={"text_input_tokens": 100, "audio_input_tokens": 200,
        "cached_input_tokens": 30, "audio_output_tokens": 50}, raw_usage=raw, status="completed")
    row, = ledger_rows(live_root)
    assert row["cost_usd"] == pytest.approx((90 + 180 * 5 + 10 * .1 + 20 * .5 + 50 * 10) / 1e6)


@pytest.mark.parametrize("usage,raw", [
    ({"text_input_tokens": 10, "audio_input_tokens": 10, "cached_input_tokens": 5}, {}),
    ({"text_input_tokens": 10, "cached_input_tokens": 11}, {}),
    ({"audio_input_tokens": 10, "cached_input_tokens": 5}, {"input_token_details": {"cached_tokens_details": {"audio_tokens": 4}}}),
    ({"audio_input_tokens": -1}, {}), ({"audio_input_tokens": True}, {}),
    ({"audio_input_tokens": None}, {}), ({"audio_input_tokens": 1.2}, {}),
    ({"text_input_tokens": 10, "video_input_tokens": 1}, {}),
    ({"text_input_tokens": 10}, {"promptTokensDetails": [{"modality": "VIDEO", "tokenCount": 1}]}),
    ({"text_input_tokens": 10}, {"unpriced_components": ["google_search"]}),
    ({"text_input_tokens": 10}, {"toolUsePromptTokenCount": 5}),
    ({"text_input_tokens": 10}, {"input_token_details": {"audio_tokens": 5}}),
    ({"text_input_tokens": 10}, {"input_token_details": {"video_tokens": 5}}),
    ({"text_input_tokens": 10}, {"input_token_details": {"audio_tokens": "bad"}}),
    ({"text_input_tokens": 10}, {"promptTokensDetails": [{"modality": "AUDIO", "tokenCount": 5}]}),
    ({"text_input_tokens": 10}, {"promptTokensDetails": [{"modality": "AUDIO", "tokenCount": "bad"}]}),
    ({}, {}),
])
def test_unknown_or_inconsistent_modality_keeps_whole_cost_unknown(live_root, usage, raw):
    attempt = reserve(live_root)
    live.settle_live_response(attempt, usage=usage, raw_usage=raw, status="completed")
    row, = ledger_rows(live_root)
    assert row["cost_usd"] is None and row["cost_final"] is False
    assert row["cost_estimated"] is False and row["cost_coverage"] == "partial"
    assert row["cost_evidence"]["unpriced"]


@pytest.mark.parametrize("field,usage", [("input_cost_per_audio_token", {"audio_input_tokens": 10}),
    ("output_cost_per_audio_token", {"audio_output_tokens": 10}),
    ("input_cost_per_image_token", {"image_input_tokens": 10}),
    ("cache_read_input_audio_token_cost", {"audio_input_tokens": 10, "cached_input_tokens": 5})])
def test_missing_needed_rate_is_partial_but_unused_rates_are_not_required(live_root, catalog_fixture, field, usage):  # noqa: F811
    import json
    _, rows, response, _ = catalog_fixture
    del rows["gpt-realtime-2.1"][field]
    response.content = json.dumps(rows).encode()
    attempt = reserve(live_root)
    raw = {"input_token_details": {"cached_tokens_details": {"audio_tokens": 5}}} if usage.get("cached_input_tokens") else {}
    live.settle_live_response(attempt, usage={"text_input_tokens": 100, **usage}, raw_usage=raw, status="completed")
    first, = ledger_rows(live_root)
    assert first["cost_usd"] is None and first["cost_coverage"] == "partial"
    assert field in first["cost_evidence"]["unpriced"]
    text = reserve(live_root, response_key="text-only")
    settle(text, text_input_tokens=100)
    assert ledger_rows(live_root)[-1]["cost_usd"] == pytest.approx(.0002)


def test_single_modality_cache_can_be_allocated_without_guessing(live_root):
    attempt = reserve(live_root)
    settle(attempt, audio_input_tokens=10, cached_input_tokens=5)
    assert ledger_rows(live_root)[0]["cost_usd"] == pytest.approx((5 * 10 + 5) / 1e6)


def test_unknown_catalog_reservation_does_not_use_text_estimator(live_root, monkeypatch):
    estimator = Mock(side_effect=AssertionError("must not use the text estimator"))
    monkeypatch.setattr(accounting, "estimate_cost_optional", estimator)
    attempt = reserve(live_root, model="gpt-realtime-absent")
    assert ledger_rows(live_root)[0]["reservation_upper_bound_usd"] is None
    settle(attempt, text_input_tokens=100, audio_input_tokens=100)
    assert ledger_rows(live_root)[0]["cost_usd"] is None
    estimator.assert_not_called()


def test_original_quote_survives_catalog_refresh(live_root, monkeypatch):
    from ouroboros import pricing_catalog
    attempt = reserve(live_root)
    pricing_catalog._cache.clear()
    monkeypatch.setattr(pricing_catalog, "get_catalog_quote", Mock(side_effect=AssertionError("settlement fetch")))
    settle(attempt, audio_input_tokens=100)
    assert ledger_rows(live_root)[0]["cost_usd"] == pytest.approx(.001)


def test_response_inherits_scope_and_bound_physical_context(live_root):
    scope = accounting.UsageScope(drive_root=live_root, task_id="call-task", root_task_id="root-task",
        parent_task_id="parent-task", category="consciousness", global_limit_usd=20,
        root_limit_usd=10, billing_group_id="shared-work", billing_group_limit_usd=15,
        non_task_operation=True)
    context = accounting.PhysicalAttemptContext(profile="owner_nano", rendered_mode="nano",
        measurement_basis="fresh_route_usage", route_fp="route", round_id="round", target_total_tokens=100,
        capacity_total_tokens=200, context_target_miss=False, automatic_pass_used=False)
    with accounting.usage_scope(scope), accounting.bind_physical_attempt_context(context):
        attempt = reserve(live_root)
    # A late receipt retains the reservation's scope, even off the original thread.
    settle(attempt, audio_input_tokens=100)
    row, = ledger_rows(live_root)
    for key in ("task_id", "root_task_id", "parent_task_id", "category", "billing_group_id"):
        assert row[key] == getattr(scope, key)
    assert row["physical_context"]["rendered_mode"] == "nano"
    assert accounting.usage_projection(live_root, billing_group_id="shared-work")["settled_usd"] == pytest.approx(.001)


@pytest.mark.parametrize("axis", ["global", "root", "group"])
@pytest.mark.parametrize("unknown_model", [False, True])
def test_known_spend_refuses_next_reservation(live_root, axis, unknown_model):
    scope = accounting.UsageScope(drive_root=live_root, task_id="call-task", root_task_id="root-task",
        global_limit_usd=.001 if axis == "global" else 100, root_limit_usd=.001 if axis == "root" else 100,
        billing_group_id="work", billing_group_limit_usd=.001 if axis == "group" else 100,
        non_task_operation=True)
    with accounting.usage_scope(scope):
        first = reserve(live_root)
        settle(first, audio_input_tokens=100)
        with pytest.raises(accounting.BudgetExceeded) as caught:
            reserve(live_root, response_key="next", model="absent" if unknown_model else "gpt-realtime-2.1")
    assert caught.value.limit_scope == ("root" if axis == "group" else axis)
    if axis == "group":
        assert "whole-work budget" in str(caught.value)
    assert len(ledger_rows(live_root)) == 1


def test_abandonment_is_unresolved_and_late_receipt_settles_same_attempt(live_root):
    attempt = reserve(live_root)
    live.abandon_live_response(attempt, reason="socket closed without response receipt")
    row, = ledger_rows(live_root)
    assert row["state"] == "unresolved" and row.get("cost_usd") is None
    assert row["reservation_upper_bound_usd"] > 0
    assert row["cost_evidence"]["response_key"] == "response-1"
    settle(attempt, audio_input_tokens=100)
    row, = ledger_rows(live_root)
    assert row["state"] == "settled" and row["settle_reason"] == "late_receipt"
    assert row["cost_usd"] == pytest.approx(.001)


@pytest.mark.parametrize("modality", [
    {"audio_input_tokens": 1}, {"audio_output_tokens": 1}, {"image_input_tokens": 1},
    {"input_token_details": {"audio_tokens": 1}}, {"prompt_tokens_details": {"image_tokens": 2}},
    {"cached_audio_input_tokens": 1}, {"cached_image_input_tokens": 1},
    {"usageMetadata": {"promptTokensDetails": [{"modality": "AUDIO", "tokenCount": 3}]}},
])
def test_text_fallback_cannot_price_nontext_usage(live_root, monkeypatch, modality):
    estimator = Mock(return_value=.05)
    monkeypatch.setattr(accounting, "estimate_cost_optional", estimator)
    attempt = accounting.reserve_attempt(accounting.AttemptRequest(model="vendor/model", provider="openrouter",
        drive_root=live_root, reservation_usd=1))
    accounting.mark_dispatched(attempt)
    usage = {"prompt_tokens": 100, "completion_tokens": 10, "new_field": {"keep": 123}, **modality}
    accounting.settle_attempt(attempt, usage)
    row, = ledger_rows(live_root)
    assert row["cost_usd"] is None
    assert row["cost_evidence"]["usage"] == usage
    estimator.assert_not_called()


def test_text_fallback_still_works_when_nontext_counts_are_zero(live_root, monkeypatch):
    estimator = Mock(return_value=.05)
    monkeypatch.setattr(accounting, "estimate_cost_optional", estimator)
    attempt = accounting.reserve_attempt(accounting.AttemptRequest(model="vendor/model", provider="openrouter",
        drive_root=live_root, reservation_usd=1))
    accounting.mark_dispatched(attempt)
    accounting.settle_attempt(attempt, {"prompt_tokens": 100, "completion_tokens": 10, "audio_input_tokens": 0})
    assert ledger_rows(live_root)[0]["cost_usd"] == .05
    estimator.assert_called_once()


def test_provider_reported_price_still_wins_for_audio(live_root):
    attempt = accounting.reserve_attempt(accounting.AttemptRequest(model="vendor/model", provider="openrouter",
        drive_root=live_root, reservation_usd=1))
    accounting.mark_dispatched(attempt)
    accounting.settle_attempt(attempt, {"prompt_tokens": 100, "audio_input_tokens": 100}, cost_usd=.2, cost_final=True)
    row, = ledger_rows(live_root)
    assert row["cost_usd"] == .2 and row["cost_final"] is True


def test_matched_raw_modality_counts_and_measured_zero_are_priceable(live_root):
    first = reserve(live_root)
    live.settle_live_response(first, usage={"audio_input_tokens": 10},
        raw_usage={"input_token_details": {"audio_tokens": 10, "video_tokens": 0},
                   "promptTokensDetails": [{"modality": "AUDIO", "tokenCount": 10}]}, status="completed")
    assert ledger_rows(live_root)[0]["cost_usd"] == pytest.approx(.0001)
    # The provider reported zeros: a free response, priced zero.
    zero = reserve(live_root, response_key="zero")
    live.settle_live_response(zero, usage=dict.fromkeys(USAGE_KEYS, 0),
                              raw_usage={"input_tokens": 0, "output_tokens": 0}, status="cancelled")
    assert ledger_rows(live_root)[-1]["cost_usd"] == 0
    # Nothing reported at all: the adapter's zeros are not a measurement; the cost stays unknown.
    unreported = reserve(live_root, response_key="unreported")
    live.settle_live_response(unreported, usage=dict.fromkeys(USAGE_KEYS, 0), raw_usage={}, status="cancelled")
    row = ledger_rows(live_root)[-1]
    assert row["cost_usd"] is None and row["cost_coverage"] == "partial"
    assert "usage_unreported" in row["cost_evidence"]["unpriced"]
    empty = reserve(live_root, response_key="empty")
    live.settle_live_response(empty, usage={}, raw_usage={}, status="failed")
    assert ledger_rows(live_root)[-1]["cost_usd"] is None


def test_direct_text_settlement_keeps_quote_and_uses_its_data_root(live_root, monkeypatch):
    from ouroboros import pricing_catalog
    pricing_catalog.get_catalog_quote("gpt-realtime-2.1", provider="openai", drive_root=live_root)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(live_root / "other-data-root"))
    attempt = accounting.reserve_attempt(accounting.AttemptRequest(model="openai::gpt-realtime-2.1", provider="openai",
        drive_root=live_root, prompt_tokens_estimate=100, max_completion_tokens=10, allow_live_fetch=False))
    accounting.mark_dispatched(attempt)
    accounting.settle_attempt(attempt, {"prompt_tokens": 100, "completion_tokens": 10})
    row, = ledger_rows(live_root)
    assert row["reservation_upper_bound_usd"] == pytest.approx(.0003)  # existing 10% OpenAI reservation margin
    assert row["cost_usd"] == pytest.approx(.00028)
    assert row["cost_evidence"]["tariff_quote"]["catalog_id"] == "gpt-realtime-2.1"
    assert row["price_source"].startswith("litellm@") and row["cost_estimated"] is True


def test_direct_settlement_keeps_the_computing_quote_without_a_second_lookup(live_root, monkeypatch):
    from ouroboros import pricing_catalog
    quote = pricing_catalog.get_catalog_quote("gpt-realtime-2.1", provider="openai", drive_root=live_root)
    attempt = accounting.reserve_attempt(accounting.AttemptRequest(model="openai::gpt-realtime-2.1", provider="openai",
        drive_root=live_root, reservation_usd=1))
    accounting.mark_dispatched(attempt)
    lookup = Mock(side_effect=[quote, AssertionError("second quote lookup")])
    monkeypatch.setattr(pricing_catalog, "get_catalog_quote", lookup)
    accounting.settle_attempt(attempt, {"prompt_tokens": 100, "completion_tokens": 10})
    row, = ledger_rows(live_root)
    assert row["cost_usd"] == pytest.approx(.00028)
    assert row["cost_evidence"]["tariff_quote"] == quote
    assert lookup.call_count == 1


def test_gemini_thinking_tokens_are_billed_output(live_root):
    attempt = reserve(live_root, engine="google", model="gemini-3.8-live")
    raw = {"usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 30, "thoughtsTokenCount": 400,
                             "responseTokensDetails": [{"modality": "TEXT", "tokenCount": 30}]}}
    live.settle_live_response(attempt, usage={"text_input_tokens": 100, "text_output_tokens": 30}, raw_usage=raw,
                              status="completed")
    row, = ledger_rows(live_root)
    assert row["cost_usd"] == pytest.approx((100 * 1 + 30 * 3 + 400 * 3) / 1e6)
    assert row["cost_evidence"]["priced_usage"]["thoughts_output_tokens"] == {"tokens": 400, "usd_per_token": 3e-6}


def test_thinking_tokens_without_an_output_rate_keep_the_cost_unknown(live_root):
    import json
    import requests  # the catalog fixture's mocked fetch
    response = requests.get.return_value
    rows = json.loads(response.content)
    del rows["gemini/gemini-3.8-live"]["output_cost_per_token"]
    response.content = json.dumps(rows).encode()
    attempt = reserve(live_root, engine="google", model="gemini-3.8-live")
    live.settle_live_response(attempt, usage={"audio_output_tokens": 10},
                              raw_usage={"usageMetadata": {"thoughtsTokenCount": 7}}, status="completed")
    row, = ledger_rows(live_root)
    assert row["cost_usd"] is None and "output_cost_per_token" in row["cost_evidence"]["unpriced"]


def test_a_budget_refused_response_keeps_the_providers_charge(live_root):
    scope = accounting.UsageScope(drive_root=live_root, task_id="call-task", root_task_id="root-task",
                                  global_limit_usd=.001, non_task_operation=True)
    with accounting.usage_scope(scope):
        settle(reserve(live_root), audio_input_tokens=100)
        with pytest.raises(accounting.BudgetExceeded) as refusal:
            reserve(live_root, response_key="response-2")
        # The provider never heard of the refusal: a response it already started is still charged.
        live.record_refused_live_response(engine="openai", model="gpt-realtime-2.1", call_id="call-1",
            response_key="response-2", task_id="call-task", rendered_mode="nano", drive_root=live_root,
            reason=str(refusal.value), usage={}, raw_usage={}, status="budget_refused")
        live.record_refused_live_response(engine="openai", model="gpt-realtime-2.1", call_id="call-1",
            response_key="response-2", task_id="call-task", rendered_mode="nano", drive_root=live_root,
            reason=str(refusal.value), usage={"audio_output_tokens": 100}, raw_usage={"output_tokens": 100},
            status="completed")
    rows = [row for row in ledger_rows(live_root) if row.get("source") == "live_response_refused"]
    assert [row["cost_usd"] for row in rows] == [None, pytest.approx(.002)]
    assert rows[0]["cost_coverage"] == "partial" and "usage_unreported" in rows[0]["cost_evidence"]["unpriced"]
    assert rows[0]["cost_evidence"]["budget_refusal"] == str(refusal.value)
    assert rows[1]["cost_evidence"]["status"] == "completed"
    assert accounting.usage_projection(live_root)["settled_usd"] == pytest.approx(.001 + .002)


def test_in_loop_lookups_never_fetch_the_catalog(live_root):
    import requests  # the catalog fixture's mocked fetch
    fetch = requests.get
    attempt = reserve(live_root, warm=False)
    assert fetch.call_count == 0
    assert ledger_rows(live_root)[0]["reservation_upper_bound_usd"] is None  # unknown until warmed, never a wait
    settle(attempt, audio_input_tokens=10)
    assert fetch.call_count == 0
    assert live.warm_quote(engine="openai", model="gpt-realtime-2.1", drive_root=live_root)["catalog_id"] == "gpt-realtime-2.1"
    assert fetch.call_count == 1
    reserve(live_root, warm=False, response_key="after-warm")
    assert ledger_rows(live_root)[-1]["reservation_upper_bound_usd"] is not None and fetch.call_count == 1
