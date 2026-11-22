"""Normalize settlement evidence outside the ledger's monetary transaction.

The writer owns state transitions; this projection owns which provider usage and
price facts reach those rows. The estimator is passed in so the historical
``usage_accounting.estimate_cost_optional`` seam remains patchable.

A DeepSeek attempt is priced only from the evidence pinned on its own reserved row
(``pricing_catalog.pin_tariff``), never a later catalog, and only for a key billed in USD:
a text call when the whole interval from the row's dispatch to this settlement (a reading
taken after the answer arrived, so never shorter than the call) lies in the entry's
published off-peak windows; a search answered HTTP 200 with results, one request.
Anything else, including incomplete usage, stays unknown; provider cash wins.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from ouroboros._usage_response import (
    _reported_token_count, cache_write_split, cache_write_total, nontext_counts, observed_processing_mode,
    processing_receipt, provider_cost_value, text_token_counts,
)


def has_nontext_usage(usage) -> bool:
    """Aggregate prompt totals cannot price reported audio/image usage as text. A null count is unreported (an SDK
    model fills its absent detail fields with None), not usage; any other nonzero value, or a non-text modality
    without a zero count, is."""
    return any(count != 0 for count in nontext_counts(usage))


_MALFORMED = object()


def _exact(source: Any, key: str) -> Any:
    """``None`` when absent, ``_MALFORMED`` when not a nonnegative integer, else the count."""
    if not isinstance(source, Mapping) or source.get(key) is None:
        return None
    value = source[key]
    return value if type(value) is int and value >= 0 else _MALFORMED


def _billed_tokens(usage: Mapping[str, Any]) -> Tuple[Optional[Tuple[int, int, int]], str]:
    """``(prompt, cache hit, completion)`` when the reply states its whole billed split, else why not. An absent
    hit count is unknown, never zero; DeepSeek states it twice (top level and details), and both must agree. A
    stated total must be prompt plus completion, and stated reasoning lies within completion (priced there once);
    an absent total or reasoning count is not required. Validate the pre-coercion witness when present: exact
    integers only, so a malformed reported count never becomes a known tariff."""
    nontext = has_nontext_usage(usage)
    usage = usage.get("_reported_token_usage", usage)
    counts = {key: _exact(usage, key) for key in ("prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens",
                                                  "prompt_cache_miss_tokens", "cache_write_tokens", "total_tokens")}
    detail_hit = _exact(usage.get("prompt_tokens_details"), "cached_tokens")
    reasoning = _exact(usage.get("completion_tokens_details"), "reasoning_tokens")
    writes = [counts["cache_write_tokens"], _exact(usage, "cache_creation_input_tokens")]
    for detail in ("prompt_tokens_details", "input_tokens_details"):
        writes.extend(_exact(usage.get(detail), key) for key in (
            "cache_write_tokens", "cache_creation_tokens", "cache_creation_input_tokens"))
    writes.extend(_exact(usage.get("cache_creation"), key) for key in (
        "ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"))
    if _MALFORMED in (*counts.values(), detail_hit, reasoning, *writes):
        return None, "usage_malformed"
    prompt, completion, miss = counts["prompt_tokens"], counts["completion_tokens"], counts["prompt_cache_miss_tokens"]
    if prompt is None or completion is None:
        return None, "usage_unreported"
    if counts["total_tokens"] is not None and counts["total_tokens"] != prompt + completion:
        return None, "usage_total_inconsistent"
    if reasoning is not None and reasoning > completion:
        return None, "reasoning_exceeds_completion"
    hits = {value for value in (counts["prompt_cache_hit_tokens"], detail_hit) if value is not None}
    if len(hits) != 1:
        return None, "cache_hit_unreported" if not hits else "cache_hit_reports_disagree"
    hit = hits.pop()
    if hit > prompt or (miss is not None and hit + miss != prompt):
        return None, "cache_split_inconsistent"
    if any(writes) or nontext:
        return None, "usage_unpriced_class"
    return (prompt, hit, completion), ""


def _pinned_attempt(reservation) -> Tuple[Dict[str, Any], str]:
    """The reserved row's own pinned evidence and its dispatch instant: the carrier every settler shares."""
    from ouroboros import usage_store

    try:
        with usage_store.read(reservation.drive_root) as txn:
            record = txn.conn.execute("SELECT ts_dispatched, extra FROM attempts WHERE attempt_id=?",
                                      (str(reservation.attempt_id),)).fetchone()
        evidence = json.loads(record["extra"]).get("cost_evidence") if record is not None else None
        return (copy.deepcopy(evidence) if isinstance(evidence, dict) else {}), str(record["ts_dispatched"] or "")
    except Exception:
        return {}, ""


def _pinned_price(evidence: Mapping[str, Any], usage: Mapping[str, Any],
                  dispatched_at: str) -> Tuple[Optional[float], Dict[str, Any]]:
    """``(cost, tariff_pricing facts)`` of a DeepSeek attempt from its pinned quote; cost None is unknown."""
    from ouroboros import pricing_catalog as catalog

    quote = evidence.get("tariff_quote")
    settled_at = datetime.now(timezone.utc)
    try:
        start = datetime.fromisoformat(dispatched_at)
        sent_on = start.astimezone(timezone.utc).date() if start.tzinfo else None
    except ValueError:
        start = sent_on = None
    if sent_on is None:
        return None, {"interval": [dispatched_at, settled_at.isoformat()], "unpriced": "dispatch_instant_unknown"}
    # The pinned quote is judged on the day it was sent: a later retirement never unprices a call made before it.
    per_query = catalog.per_query_terms(quote, on=sent_on)
    terms = None if per_query is not None else catalog.off_peak_terms(quote, on=sent_on)
    currency = (evidence.get("account_currency") or {}).get("currency")
    if per_query is None and terms is None:
        return None, {"unpriced": "no_applicable_quote"}
    if per_query is None and evidence.get("input") != "text":
        return None, {"unpriced": "media_input_unqualified"}  # aggregate usage could price media as text
    if currency != "USD":
        return None, {"unpriced": "currency_not_usd" if currency else "currency_unknown"}
    if per_query is not None:
        outcome = evidence.get("outcome")  # gateways/search_api.search_usage: the guide's billable answer or why not
        if outcome != "results" or evidence.get("http_status") != 200:
            return None, {"unpriced": f"search_{outcome or 'outcome_unknown'}"}
        return per_query, {"billable_requests": 1}
    facts: Dict[str, Any] = {"interval": [dispatched_at, settled_at.isoformat()]}
    tokens, reason = _billed_tokens(usage)
    if reason or tokens is None:
        return None, {**facts, "unpriced": reason}
    if not catalog.wholly_off_peak(terms["windows"], start, settled_at):
        return None, {**facts, "unpriced": "not_wholly_off_peak"}
    prompt, hit, completion = tokens
    rates = terms["rates"]
    cost = ((prompt - hit) * rates["input_cost_per_token"] + hit * rates["cache_read_input_token_cost"]
            + completion * rates["output_cost_per_token"])
    return cost, {**facts, "period": "off_peak", "rates": dict(rates),
                  "tokens": {"prompt": prompt, "cache_hit": hit, "completion": completion}}


def settled_cost_facts(reply_usage: Mapping[str, Any]) -> Dict[str, Any]:
    """The amount the pinned tariff settled on this reply's own attempt row, so the reply shows exactly that and is
    never priced again. The attempt is this context's own last one; a row that did not settle these very counts on
    this route, or that the pin did not price, is not borrowed."""
    from ouroboros import usage_store
    from ouroboros.usage_accounting import current_physical_attempt_drive_root, last_physical_attempt_capture

    capture, root = last_physical_attempt_capture(), current_physical_attempt_drive_root()
    if capture is None or root is None:
        return {}
    try:
        with usage_store.read(root) as txn:
            row = txn.attempt(capture.attempt_id)
    except Exception:
        return {}
    priced = ((row or {}).get("cost_evidence") or {}).get("tariff_pricing")
    if not row or not isinstance(priced, dict) or "unpriced" in priced or row.get("state") != "settled" \
            or row.get("cost_usd") is None or row.get("cost_final") or any(
            row.get(key) != _reported_token_count(reply_usage, key) for key in ("prompt_tokens", "completion_tokens")) \
            or (row.get("provider"), row.get("model")) != (reply_usage.get("provider"), reply_usage.get("resolved_model")):
        return {}
    return {"cost": float(row["cost_usd"]), "cost_estimated": True,
            **{key: row[key] for key in ("price_source", "cost_basis") if row.get(key)}}


def attempt_money_fact(drive_root: Any, attempt_id: Any) -> Dict[str, Any]:
    """What the ledger settled for one attempt, for the display event that names it (``ledger_attempt_ids``): its
    amount (None: unknown), whether that is the provider's final amount or the ledger's own estimate, whether the
    row found a reported amount unreadable (``cost_invalid``, the settlement's own fact) and the row's counts.
    The reader prices nothing itself. A display read (``usage_store.read(allow_stale=True)``): it never holds its
    caller for the money wait. Empty when the row cannot be read in that wait or is not settled: the event then
    states none of these."""
    from ouroboros import usage_store

    try:
        with usage_store.read(drive_root, allow_stale=True) as txn:
            row = txn.attempt(str(attempt_id))
    except Exception:
        return {}
    if not row or row.get("state") != "settled":
        return {}
    cost = provider_cost_value(row.get("cost_usd"))
    counts = ("prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens")
    return {"cost": cost, "cost_final": cost is not None and bool(row.get("cost_final")),
            "cost_estimated": cost is not None and bool(row.get("cost_estimated")),
            "cost_invalid": cost is None and bool(row.get("cost_invalid")),
            **{key: row.get(key) for key in counts}}


def frame_estimate_ttl(reservation, request, exc) -> Optional[str]:
    """An OpenRouter failure frame may use its own measured send's tier for an estimate. Empty means unbound."""
    if reservation.provider != "openrouter":
        return None
    receipt = getattr(exc, "stream_receipt", None)
    if (request is not None and request.candidate_measurement_kind == "canonical_json_v1" and request.candidate_raw_sha256
            and isinstance(receipt, dict) and receipt.get("attempt_id") == reservation.attempt_id):
        return request.prompt_cache_ttl or ""
    return ""


def settlement_fields(reservation, usage, cost_usd, cost_final, *, estimate_cost_optional,
                      estimate_cache_ttl: Optional[str] = None) -> Dict[str, Any]:
    """Normalize reported usage before the monetary transaction; no live pricing I/O."""
    normalized = dict(usage or {})
    evidence = copy.deepcopy(normalized.get("cost_evidence") or {})
    if not isinstance(evidence, dict):
        evidence = {"provider_evidence": evidence}
    if reservation.provider == "openrouter":
        evidence.pop("cache_write_ttl", None)  # only the measured send below can supply this source
    pinned = str(reservation.provider or "").lower() == "deepseek"
    if pinned:  # the reserved row's pin stays the attempt's evidence; a reply's own price claims are not
        own, dispatched_at = _pinned_attempt(reservation)
        for key in ("tariff_quote", "account_currency", "tariff_pricing"):
            evidence.pop(key, None)
        evidence.update(own)
        for key in ("price_source", "cost_coverage"):
            normalized.pop(key, None)
    receipt = processing_receipt(reservation.provider, normalized,
                                 requested=reservation.processing_preference,
                                 submitted_native=reservation.submitted_processing_mode)
    if receipt is not None:
        normalized["processing"] = receipt
    prompt_tokens, completion_tokens = text_token_counts(normalized)
    cached_tokens = _reported_token_count(normalized, "cached_tokens")
    split = cache_write_split(normalized)
    cache_write_tokens, invalid_write = cache_write_total(normalized, split)
    if invalid_write:
        normalized["cache_write_tokens_invalid"] = True
    if split:
        normalized["cache_write_tokens_by_ttl"] = split
    reported_ttl = str(normalized.get("prompt_cache_ttl") or "")
    pricing_ttl = reported_ttl or estimate_cache_ttl
    # A failure frame needs an observation or the explicit tier on this send to price aggregate writes.
    frame_tier_missing = (estimate_cache_ttl is not None and cache_write_tokens and not split
                          and pricing_ttl not in ("5m", "1h"))
    cost = provider_cost_value(cost_usd)
    if cost_usd is not None and cost is None:
        normalized["cost_invalid"] = True
    pricing_mode = observed_processing_mode(reservation.provider, normalized)
    has_usage = bool((prompt_tokens or 0) or (completion_tokens or 0))
    if cost is None and str(reservation.provider or "").lower() == "local":
        cost, cost_final = 0.0, True
    elif pinned and evidence.get("tariff_quote"):
        if cost is None and cost_usd is None and not normalized.get("cost_invalid"):
            cost, evidence["tariff_pricing"] = _pinned_price(evidence, normalized, dispatched_at)
            cost_final = False
            if cost is not None:
                normalized["price_source"] = f"litellm@{evidence['tariff_quote']['fetched_at']}"
        elif (cost is None and cost_usd is None and normalized.get("cost_invalid")
              and evidence.get("operation") == "chat" and evidence.get("input") == "text"):
            # A rejected charge must not be estimated. The raw counters can still explain why pricing is unknown.
            _, reason = _billed_tokens(normalized)
            if reason == "usage_malformed":
                evidence["tariff_pricing"] = {"unpriced": reason}
    elif (cost is None and cost_usd is None and not normalized.get("cost_invalid") and has_usage
          and not reservation.provider_reported_cost_only and not reservation.unpriced_add_on
          and not normalized.get("cache_write_tokens_invalid")
          and not frame_tier_missing and normalized.get("cost_coverage") != "partial" and not has_nontext_usage(normalized)):
        cost = estimate_cost_optional(
            reservation.model,
            prompt_tokens,  # an unreported count stays None: both base text rates stay required
            completion_tokens,
            cache_usage={"cached_tokens": cached_tokens, "cache_write_tokens": cache_write_tokens,
                         "cache_write_tokens_by_ttl": split, "prompt_cache_ttl": pricing_ttl},
            allow_live_fetch=False,
            provider=reservation.provider,
            **({"processing_mode": pricing_mode} if pricing_mode else {}),
        )
        cost_final = False
        if cost is not None and estimate_cache_ttl and cache_write_tokens and not (reported_ttl or split):
            evidence["cache_write_ttl"] = {"source": "sent_candidate", "ttl": estimate_cache_ttl}
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
        prompt_cache_ttl=reported_ttl,
        # ``response_provider``: the upstream that served the call, as the response named it (a cache is one
        # upstream's); ``cache_write_tokens_by_ttl``: the provider's own split of the writes by lifetime.
        **{key: copy.deepcopy(normalized[key]) for key in (
            "effort", "effort_resolution", "processing", "speed", "service_tier", "cost_basis", "cost_coverage",
            "price_source", "cost_invalid", "response_provider", "cache_write_tokens_by_ttl", "cache_write_tokens_invalid")
           if key in normalized},
    )
