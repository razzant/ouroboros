"""The provider usage/cost normalization every paid attempt settles through (``usage_accounting.usage_from_response``).

Moved verbatim out of the retired skill-review usage suite: these two contracts are the ledger's, not a surface's.
"""

from __future__ import annotations

import copy
import json

import pytest

from ouroboros import usage_accounting as ua
from ouroboros._usage_response import BYOK_COST_BASIS, consumed_cost_value

# Real final usage frames of two BYOK replies (OpenRouter, Anthropic-served, the owner's own
# provider key), recorded 2026-10-22; no prompt text. OpenRouter charged nothing; the provider billed.
BYOK_TINY = {"prompt_tokens": 49, "completion_tokens": 50, "total_tokens": 99, "cost": 0, "is_byok": True,
             "cost_details": {"upstream_inference_cost": 0.001196, "upstream_inference_prompt_cost": 0.000196,
                              "upstream_inference_completions_cost": 0.001}}
BYOK_BIG = {"prompt_tokens": 911973, "completion_tokens": 4, "total_tokens": 911977, "cost": 0, "is_byok": True,
            "cost_details": {"upstream_inference_cost": 3.647972, "upstream_inference_prompt_cost": 3.647892,
                             "upstream_inference_completions_cost": 8e-05}}


def _loop_usage(usage):
    """The loop-side projection of the same raw reply (``_normalize_remote_response``)."""
    from ouroboros.llm import LLMClient
    from ouroboros.llm_routing import OPENROUTER_BASE_URL

    target = {"provider": "openrouter", "resolved_model": "anthropic/claude-opus-5.5",
              "usage_model": "anthropic/claude-opus-5.5", "base_url": OPENROUTER_BASE_URL}
    body = {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": copy.deepcopy(usage)}
    return LLMClient()._normalize_remote_response(body, target, skip_cost_fetch=True)[1]


def test_api_token_normalization_preserves_missing_zero_and_body_error_contracts():
    missing, cost, final = ua.usage_from_response({"choices": [{"message": {"content": "ok"}}]})
    assert (missing["prompt_tokens"], missing["completion_tokens"]) == (None, None)
    assert (missing["cached_tokens"], missing["cache_write_tokens"]) == (None, None)
    assert cost is None and final is False

    cache_only, cost, final = ua.usage_from_response({"usage": {"cached_tokens": 7}})
    assert cache_only["prompt_tokens"] is None and cache_only["cached_tokens"] == 7
    assert cost is None and final is False

    generic_mixed, cost, final = ua.usage_from_response({
        "usage": {"input_tokens": 100, "cached_tokens": 40},
    })
    assert generic_mixed["prompt_tokens"] == 100 and generic_mixed["cached_tokens"] == 40
    assert cost is None and final is False

    anthropic, cost, final = ua.usage_from_response({
        "usage": {"input_tokens": 3, "cache_read_input_tokens": 7},
    })
    assert anthropic["prompt_tokens"] == 10 and anthropic["cached_tokens"] == 7
    assert cost is None and final is False

    zeros, cost, final = ua.usage_from_response({
        "usage": {
            "prompt_tokens": 0, "completion_tokens": 0,
            "cached_tokens": 0, "cache_write_tokens": 0,
        },
    })
    assert (
        zeros["prompt_tokens"], zeros["completion_tokens"],
        zeros["cached_tokens"], zeros["cache_write_tokens"],
    ) == (0, 0, 0, 0)
    assert cost is None and final is False

    rejected, cost, final = ua.usage_from_response({
        "error": {"code": 429, "message": "rate limited"}, "usage": None,
    })
    assert (rejected["prompt_tokens"], rejected["completion_tokens"]) == (None, None)
    assert cost is None and final is False

    rejected, cost, final = ua.usage_from_response({
        "error": {"code": 429, "message": "rate limited"},
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    })
    assert (rejected["prompt_tokens"], rejected["completion_tokens"]) == (0, 0)
    assert cost == 0.0 and final is True

    for usage, field in (
        ({"cached_tokens": 7}, "cached_tokens"),
        ({"cache_write_tokens": 7}, "cache_write_tokens"),
        ({"prompt_tokens_details": {"cached_tokens": 7}}, "cached_tokens"),
        ({"cache_creation": {"ephemeral_5m_input_tokens": 7}}, "cache_write_tokens_by_ttl"),
    ):
        rejected_cache, cost, final = ua.usage_from_response({
            "error": {"code": 429}, "usage": usage,
        })
        assert rejected_cache["prompt_tokens"] is None
        assert rejected_cache["completion_tokens"] is None
        assert rejected_cache[field]
        assert cost is None and final is False


@pytest.mark.parametrize("details,expected_cost", [
    ({"completion_tokens_details": {"audio_tokens": 3}}, None),
    ({"output_tokens_details": {"reasoning_tokens": "unreadable"}}, None),
    ({"server_tool_use": {"web_search_requests": False}}, None),
    ({"completion_tokens_details": {"audio_tokens": None, "accepted_prediction_tokens": 0},
      "_reported_token_usage": {"audio_tokens": 2}, "private_extension": {"counter": 7},
      "cache_write_tokens_by_ttl": {"5m": 7}}, 0.0),
])
def test_free_body_error_requires_empty_raw_usage(details, expected_cost):
    body = {"error": {"code": 429}, "usage": {"prompt_tokens": 0, "completion_tokens": 0, **details}}
    normalized, cost, final = ua.usage_from_response(body)
    assert (cost, final) == (expected_cost, expected_cost is not None)
    if "output_tokens_details" in details:
        assert normalized["_reported_token_usage"]["output_tokens_details"] == details["output_tokens_details"]


def test_zero_counters_do_not_price_an_upstream_attempt():
    body = {"error": {"code": 429}, "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
    assert ua.usage_from_response(body, provider="openrouter")[1:] == (0.0, True)
    body["openrouter_metadata"] = {"attempts": [{"status": 504}]}
    assert ua.usage_from_response(body, provider="openrouter")[1:] == (None, False)


def test_numeric_error_code_supplies_missing_timeout_status():
    body = {"error": {"message": "request rejected"},
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
    assert ua.usage_from_response(body, provider="openrouter")[1:] == (0.0, True)
    body["error"]["code"] = 504
    assert ua.usage_from_response(body, provider="openrouter")[1:] == (None, False)


def test_byok_body_error_without_upstream_charge_does_not_infer_zero():
    body = {"error": {"code": 429}, "usage": {
        "is_byok": True, "cost": 0, "prompt_tokens": 0, "completion_tokens": 0}}
    assert ua.usage_from_response(body, provider="openrouter")[1:] == (None, False)
    assert ua.usage_from_response({**body, "total_cost_usd": 0}, provider="openrouter")[1:] == (0.0, True)
    body["usage"]["cost_details"] = {"upstream_inference_cost": 0}
    assert ua.usage_from_response(body, provider="openrouter")[1:] == (0.0, True)


def test_usable_completion_keeps_explicit_zero_with_nonfatal_error_field():
    from ouroboros.llm import LLMClient

    body = {"error": {"code": 429}, "usage": {"prompt_tokens": 4, "completion_tokens": 2, "cost": 0},
            "choices": [{"finish_reason": "stop", "message": {"content": "answer"}}]}
    assert LLMClient._provider_body_error(body) is None
    assert ua.usage_from_response(body, provider="openrouter")[1:] == (0.0, True)
    body["choices"][0]["message"] = {}
    assert LLMClient._provider_body_error(body) == body["error"]
    assert ua.usage_from_response(body, provider="openrouter")[1:] == (None, False)


def test_explicit_error_finish_does_not_make_partial_text_a_usable_zero_bill():
    from ouroboros._usage_response import provider_body_error, refusal_body_conflicts

    body = {"error": {"code": 502, "message": "provider failure"},
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0},
            "choices": [{"finish_reason": "error", "message": {"content": "partial"}}]}
    assert ua.usage_from_response(body, provider="openrouter")[1:] == (None, False)
    assert provider_body_error(body) == body["error"]
    assert refusal_body_conflicts(body)  # Partial output still prevents a replay.


def test_r3_zero_full_bill_cannot_override_nested_byok_charge():
    body = {"total_cost_usd": 0, "usage": {"prompt_tokens": 10}, "error": {
        "code": 502, "usage": {"is_byok": True, "cost": 0.25},
    }}
    assert ua.usage_from_response(body, provider="openrouter")[1:] == (None, False)


@pytest.mark.parametrize("prior", [None, []], ids=["plain", "stream"])
def test_r3_failure_price_reuses_retained_body_without_sdk_reread(prior):
    from ouroboros._usage_response import priced_failure_usage

    class SDKResponse:
        reads = 0

        def json(self):
            self.reads += 1
            return body

    body = {"error": {"code": 502}, "usage": {"cost": 0, "completion_tokens": 1}}
    error = RuntimeError("provider failure")
    error.response = SDKResponse()
    error.stream_prior_usage = prior
    error.stream_usage = {"cost": 0}
    assert priced_failure_usage(error, body, "openrouter") is None
    assert error.response.reads == 0


def test_api_cost_extraction_rejects_untrustworthy_provider_amounts():
    """The AUTHORITATIVE cost boundary: whatever survives `usage_from_response` is
    what the durable attempt ledger settles as money.

    `_number` previously rejected only non-finite values, so a JSON `true` passed
    `float()` and `math.isfinite()` and settled as a FINAL $1.00 — real budget
    admission spent on an amount no provider ever reported — and a negative amount
    settled as a final credit. Both are unknown now, applying the same predicate
    `loop_llm_call._provider_cost_value` applies to the projection lane. Both lanes
    must reject, or an invalid cost still reaches the ledger.
    """
    for bad in (
        True, False, -5, -0.01, float("nan"), float("inf"), float("-inf"),
        "abc", object(), [1.0], {"usd": 1.0},
    ):
        _normalized, cost, final = ua.usage_from_response({"usage": {"cost": bad}})
        assert cost is None, f"{bad!r} must never settle as money"
        assert final is False

    for good, expected in ((0.0, 0.0), (0, 0.0), (1.23, 1.23), ("2.5", 2.5)):
        _normalized, cost, final = ua.usage_from_response({"usage": {"cost": good}})
        assert cost == expected and final is True

    # Rejecting one field does not abandon the candidate chain: the next reported
    # amount is still consulted, exactly as it is when `cost` is simply absent.
    _normalized, cost, final = ua.usage_from_response(
        {"usage": {"cost": True, "total_cost": 2.0}},
    )
    assert cost == 2.0 and final is True


@pytest.mark.parametrize("usage,expected", [(BYOK_TINY, 0.001196), (BYOK_BIG, 3.647972)])
def test_a_byok_reply_is_priced_by_what_it_consumed_in_both_lanes(usage, expected):
    """The incident shape: ``cost=0`` (no OpenRouter credit spent), ``is_byok`` and the
    provider-billed share in ``upstream_inference_cost``. It settled as a final $0.00."""
    normalized, cost, final = ua.usage_from_response({"usage": copy.deepcopy(usage)})
    assert (cost, final) == (expected, True)
    assert normalized["cost"] == expected and normalized["cost_basis"] == BYOK_COST_BASIS
    loop = _loop_usage(usage)
    assert (loop["cost"], loop["cost_final"], loop["cost_basis"]) == (expected, True, BYOK_COST_BASIS)
    assert "cost_invalid" not in normalized and "cost_invalid" not in loop


def _recorded_openrouter_bodies():
    """Every recorded OpenRouter reply body that carries a price (assembled like production)."""
    from ouroboros.llm_stream import consume_stream
    from tests.test_llm_wire_corpus import WIRE, _Wire

    for path in sorted((WIRE / "openrouter").rglob("*")):
        if path.suffix == ".sse":
            body = consume_stream(_Wire(path.read_bytes(), 0), native=False).model_dump()
        elif path.suffix == ".json":
            body = json.loads(path.read_text(encoding="utf-8"))
        else:
            continue
        if isinstance(body.get("usage"), dict) and body["usage"].get("cost") is not None:
            yield path.name, body


def test_ordinary_recorded_replies_keep_their_price_and_are_never_double_counted(monkeypatch):
    """On every recorded ordinary reply OpenRouter repeats the charge in
    ``upstream_inference_cost``: adding it without ``is_byok`` would double each bill."""
    import ouroboros.pricing as pricing

    monkeypatch.setattr(pricing, "estimate_cost_optional", lambda *a, **k: pytest.fail("a priced reply is never estimated"))
    bodies = list(_recorded_openrouter_bodies())
    assert len(bodies) >= 13
    for name, body in bodies:
        usage = body["usage"]
        assert usage["is_byok"] is False and usage["cost_details"]["upstream_inference_cost"] == usage["cost"], name
        normalized, cost, final = ua.usage_from_response(copy.deepcopy(body))
        assert (cost, final) == (usage["cost"], True) and "cost_basis" not in normalized, name
        loop = _loop_usage(usage)
        assert (loop["cost"], loop["cost_final"]) == (usage["cost"], True) and "cost_basis" not in loop, name
    # The shared-capacity fallback of a BYOK request is an ordinary reply too, and only a
    # boolean true is a BYOK fact.
    for ordinary in ({**copy.deepcopy(BYOK_TINY), "is_byok": False, "cost": 0.001196},
                     {**copy.deepcopy(BYOK_TINY), "is_byok": "true", "cost": 0.001196}):
        assert consumed_cost_value(ordinary) == 0.001196
        assert ua.usage_from_response({"usage": ordinary})[1:] == (0.001196, True)


def test_a_byok_fee_adds_and_a_missing_upstream_figure_is_unknown_never_free(monkeypatch):
    import ouroboros.pricing as pricing

    fee = {**copy.deepcopy(BYOK_TINY), "cost": 0.0000598}  # beyond the free allowance: the 5% fee
    assert ua.usage_from_response({"usage": fee})[1] == pytest.approx(0.0012558)
    assert _loop_usage(fee)["cost"] == pytest.approx(0.0012558)
    monkeypatch.setattr(pricing, "estimate_cost_optional", lambda *a, **k: 0.5)
    for broken in (None, -1, True, "abc", "missing"):
        usage = copy.deepcopy(BYOK_TINY)
        if broken == "missing":
            del usage["cost_details"]
        else:
            usage["cost_details"]["upstream_inference_cost"] = broken
        normalized, cost, final = ua.usage_from_response({"usage": usage})
        # Unknown, not invalid: the ledger's tariff estimator may still price it non-final.
        assert (cost, final) == (None, False) and "cost_invalid" not in normalized, broken
        loop = _loop_usage(usage)
        assert (loop["cost"], loop["cost_estimated"], loop["cost_final"]) == (0.5, True, False), broken


@pytest.mark.parametrize("reported,read", [
    pytest.param({"cached_tokens": 300, "prompt_tokens_details": {"cached_tokens": 0}}, 0,
                 id="details-zero-beside-a-top-level-count"),
    pytest.param({"cached_tokens": 0, "prompt_cache_hit_tokens": 300}, 0, id="top-level-zero-beside-the-mirror"),
    pytest.param({"prompt_tokens_details": {"cached_tokens": 0}, "prompt_cache_hit_tokens": 300}, 0,
                 id="details-zero-beside-the-mirror"),
    pytest.param({"cache_read_input_tokens": 300, "prompt_tokens_details": {"cached_tokens": 0}, "cost": 0.031}, 0,
                 id="details-zero-beside-the-native-name-on-a-billed-reply"),
    pytest.param({"cached_tokens": 300, "prompt_tokens_details": {"cached_tokens": 200}}, 200,
                 id="details-count-beside-another-count"),
    pytest.param({"cached_tokens": 0, "prompt_tokens_details": {"cached_tokens": 300}}, 300,
                 id="details-count-beside-a-top-level-zero"),
    pytest.param({"prompt_tokens_details": {"cached_tokens": 0}}, 0, id="details-zero-alone"),
    pytest.param({"cached_tokens": 0}, 0, id="top-level-zero-alone"),
    pytest.param({"prompt_cache_hit_tokens": 0}, 0, id="mirror-zero-alone"),
    pytest.param({"prompt_cache_hit_tokens": 300}, 300, id="mirror-alone"),
    pytest.param({}, None, id="unreported"),
])
def test_the_ledger_and_the_loop_read_one_cache_read_count_when_its_aliases_disagree(reported, read):
    """A reply of 1,000 prompt tokens names its cache reads in up to four places: the details block, the native
    ``cache_read_input_tokens``, the top-level count and DeepSeek's top-level mirror. Both lanes read them in one
    order (the details block, then the first reported top-level name), and a reported zero is a measurement: the
    300 of a name read after it never replaces it in the row, in the loop's counter or in the loop's event. A
    reported $0.031 beside them stays the reply's one final amount in both lanes."""
    from ouroboros.event_usage import event_token_counts

    usage = {"prompt_tokens": 1000, "completion_tokens": 1, **reported}
    ledger, cost, final = ua.usage_from_response({"usage": copy.deepcopy(usage)})
    loop = _loop_usage(usage)
    assert ledger["cached_tokens"] == event_token_counts(loop)["cached_tokens"] == read
    assert (loop.get("cached_tokens") or 0) == (read or 0)  # the billing-shaped counter the loop folds and prices
    if "cost" in reported:
        assert (cost, final) == (loop["cost"], loop["cost_final"]) == (reported["cost"], True)
