"""Native normalized receipts reach the real host pricer and ledger."""
import copy

import pytest

from ouroboros import live_usage, pricing_catalog, usage_accounting
from tests.test_grok_openai_protocol import providers as providers, deny_network as deny_network, make
from tests.test_live_usage import quote as quote, reserve


def openai_receipt():
    return {"input_token_details": {"text_tokens": 10, "audio_tokens": 20, "image_tokens": 30, "cached_tokens": 0},
            "output_token_details": {"text_tokens": 40, "audio_tokens": 50, "reasoning_tokens": 0}}


@pytest.mark.parametrize("missing", [None, "image_tokens", "audio_tokens"])
def test_openai_supported_axes_reach_pricer_and_ledger(providers, quote, missing):
    root, tariff, _ = quote
    raw = openai_receipt()
    if missing:
        raw["input_token_details" if missing == "image_tokens" else "output_token_details"].pop(missing)
    adapter, _ = make(providers, "openai")
    done, = adapter.convert({"type": "response.done", "response": {"id": "r", "status": "completed", "usage": raw}})
    usage = done.data["usage"]
    cost, evidence = live_usage._price(tariff, usage, done.data["raw_usage"])
    if missing:
        axis = "image_input_tokens" if missing == "image_tokens" else "audio_output_tokens"
        assert axis not in usage and cost is None
        assert "unknown_" + axis in evidence["unpriced"]
    else:
        assert cost == pytest.approx(.00169) and not evidence["unpriced"]
    assert usage["video_input_tokens"] == 0
    attempt = reserve(root, tariff)
    live_usage.settle_live_response(attempt, usage=usage, raw_usage=done.data["raw_usage"], status=done.data["status"])
    row, = usage_accounting.read_usage_records(root)
    assert row["state"] == "settled"
    assert row["cost_usd"] == (pytest.approx(.00169) if missing is None else None)


def test_openai_missing_tariff_and_unreported_billable_axes_stay_unknown(providers, quote):
    _, tariff, _ = quote
    tariff = copy.deepcopy(tariff)
    tariff["entry"].pop("output_cost_per_audio_token")
    normalize = providers["openai"].module._usage
    usage = normalize({"usage": openai_receipt()})
    cost, evidence = live_usage._price(tariff, usage, openai_receipt())
    assert cost is None and "output_cost_per_audio_token" in evidence["unpriced"]
    unknown = normalize({})
    assert unknown["video_input_tokens"] == 0
    assert "text_input_tokens" not in unknown and "audio_output_tokens" not in unknown
    assert live_usage._price(tariff, unknown, {})[0] is None


def test_grok_absent_image_video_are_zero_while_billable_usage_stays_reported(providers, quote):
    root, _, _ = quote
    adapter, _ = make(providers, "grok")
    adapter.connected_seconds = 60.0  # fixed connection watermark for the pricing fixture
    raw = {"input_token_details": {"text_tokens": 10}, "output_token_details": {"audio_tokens": 5}}
    done, = adapter.convert({"type": "response.done", "response": {"id": "r", "status": "completed", "usage": raw}})
    usage = done.data["usage"]
    assert usage["image_input_tokens"] == usage["video_input_tokens"] == 0
    assert usage["text_input_tokens"] == 10 and usage["audio_output_tokens"] == 5
    assert "audio_input_tokens" not in usage and "text_output_tokens" not in usage
    tariff = pricing_catalog.get_catalog_quote("xai/grok-test", provider="grok_realtime", drive_root=root, allow_live_fetch=False)
    cost, evidence = live_usage._price(tariff, usage, done.data["raw_usage"])
    assert cost == pytest.approx(.05) and not evidence["unpriced"]


@pytest.mark.parametrize("text_in,audio_in,cached,text_out,audio_out,reasoning,total_out,expected", [
    (32898, 0, 0, 131, 77, 73, 208, .068384),
    (32978, 15, 0, 63, 120, 22, 183, .069010),
    (33017, 32, 32896, 65, 166, 21, 231, .0109812),
    (10, 20, 0, 40, 50, None, 90, .00169),
])
def test_openai_reasoning_stays_in_raw_text_usage_without_an_extra_charge(
        providers, quote, text_in, audio_in, cached, text_out, audio_out, reasoning, total_out, expected):
    root, tariff, _ = quote
    # First three count combinations are the supplied probe receipts; all tariffs are fixtures.
    raw = {"input_token_details": {"text_tokens": text_in, "audio_tokens": audio_in,
                "image_tokens": 30 if reasoning is None else 0, "cached_tokens": cached,
                "cached_tokens_details": {"text_tokens": cached}},
           "output_tokens": total_out,
           "output_token_details": {"text_tokens": text_out, "audio_tokens": audio_out}}
    if reasoning is not None:
        raw["output_token_details"]["reasoning_tokens"] = reasoning
    original = copy.deepcopy(raw)
    adapter, _ = make(providers, "openai")
    done, = adapter.convert({"type": "response.done", "response": {"id": "r", "status": "completed", "usage": raw}})
    usage = done.data["usage"]
    cost, evidence = live_usage._price(tariff, usage, done.data["raw_usage"])
    assert cost == pytest.approx(expected) and not evidence["unpriced"]
    assert "thoughts_output_tokens" not in evidence["priced_usage"]
    assert usage["thinking_tokens"] == 0  # zero additional billable thoughts, already inside text
    assert usage["text_output_tokens"] + usage["audio_output_tokens"] == total_out
    assert done.data["raw_usage"] == original and raw == original
    attempt = reserve(root, tariff)
    live_usage.settle_live_response(attempt, usage=usage, raw_usage=done.data["raw_usage"], status=done.data["status"])
    row, = usage_accounting.read_usage_records(root)
    assert row["cost_usd"] == pytest.approx(expected)
