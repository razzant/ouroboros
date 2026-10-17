"""Normalize settlement evidence outside the ledger's monetary transaction.

The writer owns state transitions; this projection owns which provider usage and
price facts reach those rows. The estimator is passed in so the historical
``usage_accounting.estimate_cost_optional`` seam remains patchable.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Dict

from ouroboros._usage_response import _reported_token_count, processing_receipt, observed_processing_mode
from ouroboros.usage_ledger import _number


def has_nontext_usage(usage) -> bool:
    """Aggregate prompt totals cannot price reported audio/image usage as text."""
    if isinstance(usage, Mapping):
        if "modality" in usage and str(usage["modality"]).upper() != "TEXT":
            return usage.get("tokenCount", usage.get("token_count")) != 0
        for key, value in usage.items():
            if key in {"audio_input_tokens", "audio_output_tokens", "image_input_tokens", "image_output_tokens",
                       "audio_tokens", "image_tokens", "input_audio_tokens", "output_audio_tokens",
                       "input_image_tokens", "output_image_tokens", "cached_audio_input_tokens",
                       "cached_image_input_tokens"} and value != 0:
                return True
            if has_nontext_usage(value):
                return True
    elif isinstance(usage, (list, tuple)):
        return any(has_nontext_usage(item) for item in usage)
    return False


def settlement_fields(reservation, usage, cost_usd, cost_final, *, estimate_cost_optional) -> Dict[str, Any]:
    """Normalize reported usage before the monetary transaction; no live pricing I/O."""
    normalized = dict(usage or {})
    evidence = copy.deepcopy(normalized.get("cost_evidence") or {})
    if not isinstance(evidence, dict):
        evidence = {"provider_evidence": evidence}
    receipt = processing_receipt(reservation.provider, normalized,
                                 requested=reservation.processing_preference,
                                 submitted_native=reservation.submitted_processing_mode)
    if receipt is not None:
        normalized["processing"] = receipt
    prompt_tokens = _reported_token_count(normalized, "prompt_tokens", "input_tokens")
    completion_tokens = _reported_token_count(normalized, "completion_tokens", "output_tokens")
    cached_tokens = _reported_token_count(normalized, "cached_tokens")
    cache_write_tokens = _reported_token_count(normalized, "cache_write_tokens")
    cost = _number(cost_usd)
    pricing_mode = observed_processing_mode(reservation.provider, normalized)
    has_usage = bool((prompt_tokens or 0) or (completion_tokens or 0))
    if cost is None and str(reservation.provider or "").lower() == "local":
        cost, cost_final = 0.0, True
    elif (cost is None and cost_usd is None and not normalized.get("cost_invalid") and has_usage
          and normalized.get("cost_coverage") != "partial" and not has_nontext_usage(normalized)):
        cost = estimate_cost_optional(
            reservation.model,
            int(prompt_tokens or 0),
            int(completion_tokens or 0),
            cache_usage={"cached_tokens": int(cached_tokens or 0),
                         "cache_write_tokens": int(cache_write_tokens or 0),
                         "cache_write_tokens_by_ttl": normalized.get("cache_write_tokens_by_ttl"),
                         "prompt_cache_ttl": str(normalized.get("prompt_cache_ttl") or "")},
            allow_live_fetch=False,
            provider=reservation.provider,
            **({"drive_root": reservation.drive_root} if reservation.provider in {"openai", "anthropic", "google"} else {}),
            **({"processing_mode": pricing_mode} if pricing_mode else {}),
        )
        cost_final = False
        quote = getattr(cost, "quote", None)
        if quote:
            evidence["tariff_quote"] = copy.deepcopy(quote)
            normalized["price_source"] = f"litellm@{quote['fetched_at']}"
    # The row's counters are a projection of text usage. Usage they cannot carry
    # (audio and image modalities) is preserved beside them, inspectable.
    if has_nontext_usage(normalized):
        evidence.setdefault("usage", copy.deepcopy({key: value for key, value in normalized.items()
                                                   if key != "cost_evidence"}))
    return dict(
        cost_usd=cost,
        cost_final=bool(cost_final and cost is not None),
        cost_estimated=cost is not None and not cost_final,
        cost_evidence=evidence,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
        prompt_cache_ttl=str(normalized.get("prompt_cache_ttl") or ""),
        **{key: copy.deepcopy(normalized[key]) for key in ("effort", "effort_resolution", "processing", "speed", "service_tier", "cost_basis", "cost_coverage", "price_source")
           if key in normalized},
    )
