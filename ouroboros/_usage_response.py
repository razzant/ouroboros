"""Provider-response usage normalization and private physical-failure evidence."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional, Tuple

_UNREAD_PAYLOAD = object()


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, dict, list)):
        return value
    for method_name in ("model_dump", "dict", "json"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return method()
            except Exception:
                pass
    return value


def provider_cost_value(value: Any) -> Optional[float]:
    """Parse a provider-reported cost; ``None`` means the value cannot be trusted.

    THE cost-trust predicate, defined once at the AUTHORITATIVE boundary (what
    survives here is what the durable attempt ledger settles as final money) and
    imported by the loop-side projection, so the two lanes cannot fork: ``bool``
    is rejected FIRST (``float(True)`` is a plausible-looking 1.0 that would
    settle as a FINAL $1.00 and eat real budget admission), then anything
    unparseable, non-finite, or negative — ``OverflowError`` included, because
    ``float(10**1000)`` raises rather than returning inf. A reported ``0.0``
    stays a legitimate zero. Rejecting settles the attempt as unknown rather
    than as a fabricated amount (BIBLE P1). Never raises.
    """
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


_number = provider_cost_value  # historical local name at this boundary

# Finish reasons that mean a reply reached its output allowance: OpenAI-family ``length``
# and Anthropic ``max_tokens``. The one vocabulary every finish-reason reader shares.
OUTPUT_LIMIT_FINISH_REASONS = frozenset({"length", "max_tokens"})


def response_finish_reason(usage: Any, msg: Any) -> Tuple[bool, Optional[str]]:
    """The provider's finish fact for one response → ``(present, value)``.

    Read by PRESENCE of the key, in this order: the usage fact ``response_finish_reason``
    (written by the OpenAI-compatible, Claudexor, local and GigaChat lanes), then the
    message's ``finish_reason``, then its ``stop_reason`` (the native Anthropic lane).
    An explicit null stays ``(True, None)``; a lower field never replaces it.
    """
    for source, key in ((usage, "response_finish_reason"), (msg, "finish_reason"), (msg, "stop_reason")):
        if isinstance(source, dict) and key in source:
            return True, source[key]
    return False, None


def reported_reasoning_tokens(usage: Any) -> Optional[int]:
    """Reasoning tokens a provider reported for one reply: top level (the Claudexor
    mapping) or inside its completion/output token details (OpenAI-family shapes);
    ``None`` when none was reported, never a guessed zero."""
    usage = usage if isinstance(usage, dict) else {}
    for source in (usage, usage.get("completion_tokens_details"), usage.get("output_tokens_details")):
        value = source.get("reasoning_tokens") if isinstance(source, dict) else None
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def output_exhaustion_facts(usage: Any, sent_max_tokens: Any) -> Dict[str, Optional[int]]:
    """What one output-exhausted reply's own records prove, for the host fact the next
    round reads: the allowance its physical receipt shows was sent — none when its usage
    says no output cap was applied (Claudexor's receipt carries a reservation, not a
    sent cap) — and the reasoning tokens its provider reported. Absent stays None."""
    usage = usage if isinstance(usage, dict) else {}
    capped = (usage.get("claudexor") or {}).get("output_cap_applied") is not False
    sent = sent_max_tokens if capped and isinstance(sent_max_tokens, int) and not isinstance(sent_max_tokens, bool) else 0
    return {"sent_allowance_tokens": sent if sent > 0 else None, "reasoning_tokens": reported_reasoning_tokens(usage)}


def processing_receipt(provider: str, usage: Dict[str, Any], *, requested: str = "",
                       submitted_native: str = "", reason: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Keep native observations separate from captured intent on the shared receipt."""
    import copy

    if isinstance(usage.get("processing"), dict):
        return copy.deepcopy(usage["processing"])
    if provider in {"openai", "openrouter"}:
        observed_native = usage.get("service_tier")
        modes = {"default": "standard", "priority": "fast", "fast": "fast", "flex": "economy"}
    elif provider == "anthropic":
        observed_native = usage.get("speed")
        modes = {"standard": "standard", "fast": "fast"}
    else:
        observed_native, modes = None, {}
    observed_native = observed_native if isinstance(observed_native, str) and observed_native else None
    if not requested and not submitted_native and not observed_native:
        return None
    submitted, observed = modes.get(submitted_native), modes.get(observed_native, "unknown")
    if reason is None:
        if requested and not submitted_native:
            reason = "processing_not_submitted"
        elif requested and submitted and requested != submitted:
            reason = "submitted_mode_differs"
        elif observed != "unknown" and submitted and submitted != observed:
            reason = "provider_mode_changed"
    return {"requested": requested or None, "submitted": submitted,
            "submittedNative": submitted_native or None, "observed": observed,
            "observedNative": [observed_native] if observed_native else [], "reason": reason,
            "source": "provider_response" if observed_native else "host_request"}


def _reported_token_count(usage: Dict[str, Any], *keys: str) -> Optional[int]:
    """Return the first reported count; absence stays distinct from explicit zero."""
    for key in keys:
        if key in usage and usage.get(key) is not None:
            try:
                return max(0, int(usage[key])) if not isinstance(usage[key], bool) else None
            except (TypeError, ValueError, OverflowError):
                return None  # A malformed counter must not discard an explicit price.
    return None


def observed_processing_mode(provider: str, usage: Dict[str, Any]) -> str:
    """A pricing qualifier from observation; absent explicit-mode proof stays unknown."""
    receipt = usage.get("processing")
    if isinstance(receipt, dict):
        modes = receipt.get("observedNative")
        return (str(modes[0]) if receipt.get("observed") not in {"unknown", "mixed"}
                and isinstance(modes, list) and len(modes) == 1 else "unknown")
    value = usage.get("speed") if provider == "anthropic" else usage.get("service_tier")
    return value if isinstance(value, str) else ""


def _normalized_input_token_usage(raw: Any) -> Optional[Dict[str, Any]]:
    """The complete native input split, or None; never repair partial evidence."""
    keys = ("total_tokens", "cache_read_tokens", "cache_write_tokens")
    if not isinstance(raw, dict) or set(raw) != set(keys):
        return None
    normalized: Dict[str, Any] = {}
    for key in keys:
        value = raw[key]
        if value is None:
            normalized[key] = None
        elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        else:
            normalized[key] = value
    return normalized


def usage_from_response(response: Any, *, payload=_UNREAD_PAYLOAD) -> Tuple[Dict[str, Any], Optional[float], bool]:
    """Extract common usage/cost facts without retaining response text."""
    if payload is _UNREAD_PAYLOAD:
        payload = _plain(response)
    if not isinstance(payload, dict) and callable(getattr(response, "json", None)):
        try:
            payload = response.json()
        except Exception:
            payload = None
    usage: Any = payload.get("usage") if isinstance(payload, dict) else getattr(response, "usage", None)
    usage = _plain(usage)
    if not isinstance(usage, dict):
        usage = {}
    native_cache_read = _reported_token_count(usage, "cache_read_input_tokens")
    native_cache_write = _reported_token_count(usage, "cache_creation_input_tokens")
    cache_read = _reported_token_count(
        usage, "cache_read_input_tokens", "cached_tokens", "precached_prompt_tokens",
    )
    cache_write = _reported_token_count(
        usage, "cache_creation_input_tokens", "cache_write_tokens",
    )
    input_tokens = _reported_token_count(usage, "input_tokens")
    prompt = _reported_token_count(usage, "prompt_tokens")
    if prompt is None and any(
        value is not None for value in (input_tokens, native_cache_read, native_cache_write)
    ):
        # Anthropic native input_tokens excludes cache reads and writes.
        prompt = int(input_tokens or 0) + int(native_cache_read or 0) + int(native_cache_write or 0)
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    if isinstance(details, dict):
        detail_read = _reported_token_count(details, "cached_tokens")
        cache_read = detail_read if detail_read is not None else cache_read
        detail_write = _reported_token_count(
            details, "cache_write_tokens", "cache_creation_tokens", "cache_creation_input_tokens",
        )
        cache_write = detail_write if detail_write is not None else cache_write
    normalized = {
        **usage,
        "prompt_tokens": prompt,
        "completion_tokens": _reported_token_count(usage, "completion_tokens", "output_tokens"),
        "cached_tokens": cache_read,
        "cache_write_tokens": cache_write,
    }
    # Candidate facts and validated engine reports come from their host owners,
    # never arbitrary extensions inside a provider's token-usage block.
    normalized.pop("effort", None)
    normalized.pop("effort_resolution", None)
    if isinstance(payload, dict):
        if "service_tier" in payload:
            normalized["service_tier"] = payload["service_tier"]
        if isinstance(payload.get("processing"), dict):
            normalized["processing"] = dict(payload["processing"])
    creation = usage.get("cache_creation")
    if isinstance(creation, dict):
        split = {
            tier: count
            for tier, key in (("5m", "ephemeral_5m_input_tokens"),
                              ("1h", "ephemeral_1h_input_tokens"))
            if (count := _reported_token_count(creation, key)) is not None and count > 0
        }
        if split:
            normalized["cache_write_tokens_by_ttl"] = split
    candidates = (
        usage.get("cost"), usage.get("total_cost"),
        payload.get("total_cost_usd") if isinstance(payload, dict) else None,
        getattr(response, "total_cost_usd", None),
    )
    cost = next((number for value in candidates if (number := _number(value)) is not None), None)
    normalized.pop("cost_invalid", None)  # Derive from provider prices, never trust their marker.
    if cost is None and any(value is not None for value in candidates):
        normalized["cost_invalid"] = True
    if cost is not None:
        return normalized, cost, True
    completion = normalized["completion_tokens"]
    split = normalized.get("cache_write_tokens_by_ttl")
    cache_usage_reported = bool(cache_read or cache_write or (
        any(split.values()) if isinstance(split, dict) else split))
    if (
        isinstance(payload, dict)
        and isinstance(payload.get("error"), dict)
        and prompt == 0
        and completion == 0
        and all(value is None for value in candidates)
        and not cache_usage_reported
    ):
        normalized.update(prompt_tokens=0, completion_tokens=0, cached_tokens=0, cache_write_tokens=0)
        return normalized, 0.0, True
    return normalized, None, False


def provider_failure_payload(exc: BaseException) -> Any:
    """The received body, when available; never synthesize it from an error string."""
    payload = getattr(exc, "body", None)
    response = getattr(exc, "response", None)
    if payload is None and callable(getattr(response, "json", None)):
        try:
            payload = response.json()
        except Exception:
            try:
                payload = response.text
            except Exception:
                pass
    return _plain(payload)


def _provider_exception_facts(exc: BaseException, *, payload=_UNREAD_PAYLOAD) -> Tuple[Optional[int], str, str, str]:
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError, OverflowError):
        status = None
    if payload is _UNREAD_PAYLOAD:
        payload = provider_failure_payload(exc)
    error = payload.get("error") if isinstance(payload, dict) and isinstance(payload.get("error"), dict) else payload
    code, error_type, message = getattr(exc, "code", None), getattr(exc, "type", None), str(exc or "")
    if isinstance(error, dict):
        code, error_type = error.get("code", code), error.get("type", error_type)
        details = json.dumps(error, ensure_ascii=False, sort_keys=True, default=str)
        message = f"{message}; provider_error={details}" if message else details
    try:
        from ouroboros.observability import redact_projection
        message = str(redact_projection(message).value)
    except Exception:
        message = f"{type(exc).__name__}: provider error details unavailable"
    return status, str(code or ""), str(error_type or type(exc).__name__), message


def physical_failure_evidence(root, attempt_id: str, *, payload, exc=None, response=None) -> Optional[dict]:
    """Retain full failure source in private CAS and expose only structural facts.

    A returned provider body error is a failure even when its monetary evidence
    settles. Retention failure leaves the original call outcome and an explicit gap.
    """
    from ouroboros.observability import write_blob
    from ouroboros.transport_custody import attempt_custody_event_fields

    if not isinstance(payload, dict) and callable(getattr(response, "json", None)):
        try:
            payload = response.json()
        except Exception:
            pass
    error = payload.get("error") if isinstance(payload, dict) else None
    if exc is None and not isinstance(error, dict):
        return None
    status, code, error_type, _ = _provider_exception_facts(exc, payload=payload) if exc is not None else (
        getattr(response, "status_code", None), error.get("code"), error.get("type"), "")
    # Provider strings can contain echoed prompt text or secrets. The complete
    # private source is the diagnostic authority; the row contains no free text.
    facts = {"stage": "raised_exception" if exc is not None else "response_body_error",
             "exception_type": type(exc).__name__ if exc is not None else None,
             "provider_status_code": status if isinstance(status, int) else None,
             "provider_code": int(code) if str(code).isascii() and str(code).isdigit() and len(str(code)) < 10 else None,
             "transport_cause_type": attempt_custody_event_fields(exc).get("transport_cause_type") if exc is not None else None}
    try:
        facts["evidence_ref"] = write_blob(root, {"attempt_id": attempt_id, "body": payload,
            "exception_type": facts["exception_type"], "message": str(exc) if exc is not None else None,
            "provider_code": code, "provider_error_type": error_type,
            "stream_usage": getattr(exc, "stream_usage", None),
            "stream_receipt": getattr(exc, "stream_receipt", None),
            "stream_evidence": getattr(exc, "stream_evidence", None)})
    except Exception as failure:
        facts["retention_gap"] = type(failure).__name__
    return facts
