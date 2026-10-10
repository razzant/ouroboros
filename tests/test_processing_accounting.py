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
    usage, _, _ = usage_from_response({"usage": {"service_tier": "standard", "speed": "fast"}})
    assert processing_receipt("anthropic", usage)["observed"] == "fast"
    assert processing_receipt("anthropic", {"service_tier": "standard"}) is None
    receipt = {"requested": "fast", "submitted": "fast", "submittedNative": "fast",
               "observed": "mixed", "observedNative": ["fast", "default"],
               "reason": None, "source": "native_session"}
    assert processing_receipt("claudexor", {"processing": receipt}) == receipt


@pytest.mark.parametrize("attempts,expected", [
    ([{"usageCost": {"cashUsd": 0, "cashKnowledge": "unknown", "valuationUsd": 5}}], (None, False)),
    ([{"usageCost": {"cashUsd": 0, "cashKnowledge": "exact", "valuationUsd": 5}}], (0.0, False)),
    ([{"usageCost": {"cashUsd": 2, "cashKnowledge": "exact", "valuationUsd": 5}}], (2.0, False)),
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
    monkeypatch.setattr(loop_llm_call, "estimate_cost_optional", price)
    monkeypatch.setattr(search, "estimate_cost_optional", price)
    usage = {"cost": None, "provider": "openrouter", "prompt_tokens": 10, "completion_tokens": 2,
             "processing": {"observed": observed, "observedNative": modes}}
    cost, _, _, _ = loop_llm_call._normalize_usage_cost(usage, model="openai/same", use_local=False)
    ctx = SimpleNamespace(pending_events=[], task_metadata={}, task_id="helper")
    search._emit_simple_usage(ctx, provider="openrouter", model="openai/same", usage={**usage, "cost": None})
    assert calls == [expected, expected]
    assert ctx.pending_events[0]["cost"] == cost == (None if expected == "unknown" else 0.1)


def test_host_view_callback_is_invoked_but_never_recorded_as_a_model_argument(tmp_path, monkeypatch):
    from ouroboros import llm_observability as observed

    recorded, views = [], []
    monkeypatch.setattr(observed, "persist_call", lambda *args, **kwargs: recorded.append(kwargs["payload"]) or {})
    callback = lambda messages, schemas: views.append(messages)

    class Client:
        def chat(self, **kwargs):
            kwargs["model_context_observer"](kwargs["messages"], kwargs.get("tools"))
            return {"role": "assistant", "content": "done"}, {}

    observed.chat_observed(Client(), drive_root=tmp_path, messages=[{"role": "user", "content": "exact"}],
                           model="model", model_context_observer=callback)
    assert len(views) == 1 and recorded
    assert all("model_context_observer" not in payload.get("kwargs", {}) for payload in recorded)
