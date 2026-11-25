"""Provider-response usage normalization and private physical-failure evidence."""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple

_UNREAD_PAYLOAD = object()


def _first_chat_choice(body: dict) -> dict:
    choices = body.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    return first if isinstance(first, dict) else {}


def provider_body_error(body: Any) -> Optional[Dict[str, Any]]:
    """A top-level provider error is nonfatal if the first completion is usable."""
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return None
    first = _first_chat_choice(body)
    message = first.get("message")
    usable = isinstance(message, dict) and (message.get("content") or message.get("tool_calls"))
    return None if usable and reply_usability(first.get("finish_reason"), None, message) != "unusable" else error


def failed_chat_completion(body: Any) -> bool:
    """A Chat error finish is a failure even without a top-level error object."""
    finish = _first_chat_choice(body).get("finish_reason") if isinstance(body, dict) else None
    return reply_usability(finish, None, None) == "unusable"


def failed_chat_completion_shows_work(body: Any) -> bool:
    """A Chat error finish may carry partial work, not permission for another physical send."""
    return failed_chat_completion(body) and refusal_body_conflicts(body)


def _response_failed(body: Any) -> bool:
    if not isinstance(body, dict):
        return False
    if (body.get("object") == "response" or "output" in body) and "status" in body:
        incomplete = body.get("incomplete_details")
        reason = incomplete.get("reason") if isinstance(incomplete, dict) else None
        return reply_usability(reason, body.get("status"), None) == "unusable"
    return provider_body_error(body) is not None or failed_chat_completion(body)


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


# The ``cost_basis`` of a reply priced as the gateway charge plus the provider-billed share.
BYOK_COST_BASIS = "byok_gateway_plus_upstream"


def consumed_cost_value(usage: Any) -> Any:
    """What one reply consumed, whichever wallet paid, read from RAW provider ``usage``.

    ``cost`` is the gateway's own charge. A request served on the caller's own provider
    key (``is_byok is True``) was billed by that provider too, reported as
    ``cost_details.upstream_inference_cost``: the reply consumed both. Every other reply
    returns ``cost`` untouched; there OpenRouter repeats the charge in that field (recorded
    wire: equal), so adding it would double the bill. A BYOK reply without a valid upstream
    figure is unknown (``None``), never free. A normalized dict already carries the sum in
    ``cost`` (``cost_basis``): never price it again.
    """
    charge = usage.get("cost") if isinstance(usage, dict) else None
    if not isinstance(usage, dict) or usage.get("is_byok") is not True or provider_cost_value(charge) is None:
        return charge
    details = usage.get("cost_details")
    upstream = provider_cost_value(details.get("upstream_inference_cost")) if isinstance(details, dict) else None
    return None if upstream is None else provider_cost_value(charge) + upstream


def reported_price(usage: Dict[str, Any], *fallbacks: Any) -> Tuple[Optional[float], bool, str]:
    """``(price, invalid, cost_basis)`` of one reply: the first trustworthy amount among what
    it consumed, its total and the envelope's fallbacks. ``invalid`` = amounts were
    reported but none is trustworthy. The ledger and loop lanes share it so they cannot fork."""
    consumed = consumed_cost_value(usage)
    candidates = (consumed, usage.get("total_cost"), usage.get("total_cost_usd"), *fallbacks)
    cost = next((number for value in candidates if (number := provider_cost_value(value)) is not None), None)
    byok = usage.get("is_byok") is True and cost is not None and provider_cost_value(consumed) is not None
    return cost, cost is None and any(value is not None for value in candidates), BYOK_COST_BASIS if byok else ""


class _RefusalPrice(NamedTuple):
    total: Optional[float]
    byok_parts: tuple[Optional[float], Optional[float]]  # gateway, upstream
    unlabelled_min: float
    disagreement: bool
    malformed: bool

    @property
    def basis(self) -> str:
        return BYOK_COST_BASIS if None not in self.byok_parts else ""


def _byok_subtotal(parts: tuple[Optional[float], Optional[float]]) -> float:
    return float(sum(Decimal(str(part)) for part in parts if part is not None))


def _refusal_price(usage: Any, *fallbacks: Any, byok_known: bool = False) -> _RefusalPrice:
    """Read independently reported totals and the BYOK components that constrain them."""
    usage = usage if isinstance(usage, dict) else {}
    charge = usage.get("cost")
    parts, unlabelled_min, malformed = (None, None), 0.0, False
    if usage.get("is_byok") is True:
        details = usage.get("cost_details")
        upstream = details.get("upstream_inference_cost") if isinstance(details, dict) else None
        parts = provider_cost_value(charge), provider_cost_value(upstream)
        malformed = any(raw is not None and parsed is None for raw, parsed in zip((charge, upstream), parts))
        charge = _byok_subtotal(parts) if None not in parts else None
    elif byok_known:
        # With no matching label this could repeat the gateway fee or the whole bill.
        # It constrains a total but is neither a second wallet nor a final total.
        parsed = provider_cost_value(charge)
        malformed = charge is not None and parsed is None
        unlabelled_min, charge = parsed or 0.0, None
    prices = [provider_cost_value(raw) for raw in (
        charge, usage.get("total_cost"), usage.get("total_cost_usd"), *fallbacks) if raw is not None]
    malformed |= None in prices
    totals = {amount for amount in prices if amount is not None}
    total = next(iter(totals)) if len(totals) == 1 else None
    disagreement = len(totals) > 1 or (total is not None and total < _byok_subtotal(parts))
    return _RefusalPrice(total if not disagreement else None, parts, unlabelled_min, disagreement, malformed)


def normalize_go_price(usage: Dict[str, Any]) -> Tuple[Optional[float], bool]:
    """Only ``usage.cost`` is cash (or explicitly a BYOK estimate) on this exact route.

    https://opencode.ai/docs/console/api/inference/#cost: list price, plan allowance
    and billed-token split are different units, not substitute prices or the
    OpenAI-compatible prompt's token count.
    """
    raw = usage.get("cost")
    cost = provider_cost_value(raw)
    opencode = usage.get("opencode")
    estimated = cost is not None and isinstance(opencode, dict) and opencode.get("estimated") is True
    for host_key in ("cost_basis", "cost_invalid", "price_source"):
        usage.pop(host_key, None)
    final = cost is not None and not estimated
    usage.update(cost=cost, cost_final=final, cost_estimated=estimated)
    if raw is not None and cost is None:
        usage["cost_invalid"] = True
    return cost, final


def update_snapshot(target: dict, update: dict) -> None:
    """Usage counters and metadata are cumulative snapshots, never token deltas."""
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            update_snapshot(target[key], value)
        elif value is not None:
            target[key] = copy.deepcopy(value)


_PRICE_AMOUNTS = ("cost", "total_cost", "total_cost_usd")
_PRICE_REPORT = (*_PRICE_AMOUNTS, "is_byok", "cost_details", "cost_basis")


def merge_usage_reports(previous: Any, latest: Any, *, previous_total: Any = None) -> dict:
    """Overlay cumulative counters, taking the price and its qualifiers from a single report."""
    previous = previous if isinstance(previous, dict) else {}
    if not isinstance(latest, dict):
        return copy.deepcopy(previous)
    if not previous and previous_total is None:
        return copy.deepcopy(latest)
    counters = {key: copy.deepcopy(value) for key, value in previous.items() if key not in _PRICE_REPORT}
    update_snapshot(counters, {key: value for key, value in latest.items() if key not in _PRICE_REPORT})
    source = previous
    if reported_price(latest)[0] is not None:
        source = latest
    elif reported_price(previous, previous_total)[0] is None:
        # Keep the latest unusable amount for diagnosis. A counters-only frame cannot erase it.
        if any(latest.get(key) is not None for key in _PRICE_AMOUNTS) or not any(key in previous for key in _PRICE_REPORT):
            source = latest
    counters.update({key: copy.deepcopy(source[key]) for key in _PRICE_REPORT if key in source})
    return counters


# Finish reasons that mean a reply reached its output allowance: OpenAI-family ``length``, Anthropic
# ``max_tokens`` and Responses ``max_output_tokens``. The one vocabulary every finish-reason reader shares.
OUTPUT_LIMIT_FINISH_REASONS = frozenset({"length", "max_tokens", "max_output_tokens"})
# A provider's own content refusal: OpenAI-family ``content_filter`` and Anthropic ``refusal``.
CONTENT_REFUSAL_FINISH_REASONS = frozenset({"content_filter", "refusal"})
# A reply that ended to call a tool: OpenAI-family ``tool_calls``/``function_call`` and Anthropic ``tool_use``.
TOOL_CALL_FINISH_REASONS = frozenset({"tool_calls", "function_call", "tool_use"})
# A completed reply its provider marked as intermediate progress, not an answer: Responses' last assistant
# ``phase`` ``commentary`` (the adapter's label). Healthy: its owner continues it as an ordinary round.
INTERMEDIATE_FINISH_REASONS = frozenset({"commentary"})


def reply_usability(finish: Any, status: Any, msg: Any) -> str:
    """One reply's usability from its provider's own terminal facts (finish label, terminal ``status``), never
    its words. ``intermediate``: completed commentary with text and no call — continue it as an ordinary round;
    ``unusable``: a provider-stated terminal that is no result — ``incomplete`` for any reason but its output
    limit (``content_filter``, ``max_messages``, ``steered``, none), a completed one ending as the provider's
    content refusal, or a failed/cancelled one (including a Chat ``error`` finish) — neither text, JSON nor call
    counts; ``usable``: everything else,
    which the existing finish rules (output-limit cut, tool-call finish, refusal, empty) still read. A reply
    without a terminal status (every Chat-shaped lane) is judged by its finish alone. The reply's items, reason
    and usage are retained by their owners either way."""
    label, state = str(finish or "").strip().lower(), str(status or "").strip().lower()
    if state and state != "completed" and not (state == "incomplete" and label in OUTPUT_LIMIT_FINISH_REASONS):
        return "unusable"
    if label == "error" or (state and label in CONTENT_REFUSAL_FINISH_REASONS):
        return "unusable"
    text = isinstance(msg, dict) and bool(str(msg.get("content") or "").strip())
    if label in INTERMEDIATE_FINISH_REASONS and text and not (isinstance(msg, dict) and msg.get("tool_calls")):
        return "intermediate"
    return "usable"


def usage_reply_usability(usage: Any, msg: Any) -> str:
    """``reply_usability`` read from one normalized ``(usage, msg)`` pair."""
    status = usage.get("response_status") if isinstance(usage, dict) else None
    return reply_usability(response_finish_reason(usage, msg)[1], status, msg)


def usage_reply_unfinished(usage: Any) -> bool:
    """Whether the provider's own terminal says the reply did not finish (Responses ``incomplete``). Its usable
    output cut is continued only by Main's output-cut recovery; a caller without that recovery, a tool-less
    one-shot or a review verdict, never takes it as a result, however well its text parses."""
    status = usage.get("response_status") if isinstance(usage, dict) else None
    return str(status or "").strip().lower() == "incomplete"


class UnusableReplyError(RuntimeError):
    """A tool-less one-shot call whose reply is no result by its provider's own terminal facts
    (``reply_usability`` intermediate or unusable, or ``unfinished``: ``usage_reply_unfinished``): raised before
    any caller parses its text, carrying the reply and its usage. The attempt is already settled; no other turn
    is started."""

    code = "unusable_reply"

    def __init__(self, usability: str, message: Any, usage: Any):
        self.usability = usability
        self.reply_message = copy.deepcopy(message) if isinstance(message, dict) else {}
        self.usage = copy.deepcopy(usage) if isinstance(usage, dict) else {}
        finish = self.usage.get("response_finish_reason")
        super().__init__(f"unusable_reply: the provider ended this reply as {usability} "
                         f"(status={self.usage.get('response_status') or 'unknown'}, finish_reason={finish}); "
                         "it is not a result")


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


_OUTPUT_TOKEN_DETAIL_FIELDS = {
    "completion_tokens_details": ("reasoning_tokens", "accepted_prediction_tokens", "rejected_prediction_tokens"),
    "output_tokens_details": ("reasoning_tokens",),
}


def reported_reasoning_tokens(usage: Any) -> Optional[int]:
    """Reasoning tokens a provider reported for one reply: top level (the Claudexor
    mapping) or inside its completion/output token details (OpenAI-family shapes);
    ``None`` when none was reported, never a guessed zero."""
    if isinstance(usage, dict):
        for source in (usage, *(usage.get(key) for key in _OUTPUT_TOKEN_DETAIL_FIELDS)):
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

    if provider not in {"openai", "openrouter", "anthropic"} and isinstance(usage.get("processing"), dict):
        return copy.deepcopy(usage["processing"])
    if provider in {"openai", "openrouter"}:
        observed_native = usage.get("service_tier")
        modes = {"default": "standard", "priority": "fast", "fast": "fast", "flex": "economy", "ultrafast": "ultrafast"}
    elif provider == "anthropic":
        observed_native = usage.get("speed")
        modes = {"standard": "standard", "fast": "fast"}
    else:
        observed_native, modes = None, {}
    observed_native = observed_native if isinstance(observed_native, str) and observed_native else None
    reported_tier = provider in {"openai", "openrouter"} and "service_tier" in usage
    if not requested and not submitted_native and not observed_native and not reported_tier:
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
            "source": "provider_response" if observed_native or reported_tier else "host_request"}


def _valid_count(value: Any) -> Optional[int]:
    """A reported count that is a nonnegative integer (an integral number or its digits), else None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return count if count >= 0 and (isinstance(value, str) or value == count) else None


def _reported_token_count(usage: Dict[str, Any], *keys: str) -> Optional[int]:
    """The first reported count when it is a nonnegative integer; a reported zero stays zero. A malformed one
    reads None like an absent one, never a clamped or truncated number, and never costs the reply its explicit
    price: ``malformed_token_counts`` and ``cache_write_total`` say which of the two it was."""
    for key in keys:
        if usage.get(key) is not None:
            return _valid_count(usage[key])
    return None


def has_reported_service_tier(response: Any, body: dict) -> bool:
    """An SDK dump can synthesize a null for an omitted field. Keep only a provider's own field."""
    fields = getattr(response, "model_fields_set", None)
    return "service_tier" in body and (not isinstance(fields, set) or "service_tier" in fields)


def malformed_token_counts(usage: Any) -> bool:
    """Whether a reply reported a text or cache-read count that is not a nonnegative integer. It is unknown on
    the row and no tariff reads it: taken for zero it would free its class of a rate, as only a reported zero
    may. The ledger and the loop share this (written counts: ``cache_write_total``)."""
    if not isinstance(usage, dict):
        return False
    reported = [usage.get(key) for key in (
        "prompt_tokens", "completion_tokens", "input_tokens", "output_tokens",
        "cached_tokens", "cache_read_input_tokens", "precached_prompt_tokens", "prompt_cache_hit_tokens")]
    reported += [details.get("cached_tokens") for key in ("prompt_tokens_details", "input_tokens_details")
                 if isinstance(details := usage.get(key), dict)]
    return any(value is not None and _valid_count(value) is None for value in reported)


def text_token_counts(usage: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """``(prompt, completion)`` as settlement and the loop read one normalized reply. An explicit ``prompt_tokens``
    null keeps the inclusive prompt unknown: native ``input_tokens`` is then only a subtotal."""
    return (_reported_token_count(usage, "prompt_tokens" if "prompt_tokens" in usage else "input_tokens"),
            _reported_token_count(usage, "completion_tokens", "output_tokens"))


def native_prompt_tokens(usage: Dict[str, Any], split: Dict[str, Optional[int]], written: Optional[int],
                         invalid_write: bool) -> Optional[int]:
    """The inclusive prompt of a native Messages reply, whose ``input_tokens`` excludes cache reads and writes:
    fresh input plus reads plus the one write total (``cache_write_total``: any valid alias, or both lifetime
    bins). The ledger and the adapter share it. Unknown without a reported fresh input (reads or writes alone
    are no prompt), when a reported part is not a count or when writes were reported without a total (a
    malformed aggregate, one lifetime bin); an unreported read or write count adds nothing."""
    fresh, read = _valid_count(usage.get("input_tokens")), _valid_count(usage.get("cache_read_input_tokens"))
    if fresh is None or invalid_write or (written is None and split) or (
            read is None and usage.get("cache_read_input_tokens") is not None):
        return None
    return fresh + (read or 0) + (written or 0)


def cache_read_count(usage: Dict[str, Any]) -> Optional[int]:
    """The reply's one cache-read count, read alike by the ledger and the loop: its details block when that reports
    a count, else the first reported top-level name (DeepSeek's ``prompt_cache_hit_tokens`` mirror last). A
    reported zero is a measurement: no later name replaces it. A count that is no count is unknown."""
    read = _reported_token_count(
        usage, "cache_read_input_tokens", "cached_tokens", "precached_prompt_tokens", "prompt_cache_hit_tokens")
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    detail_read = _reported_token_count(details, "cached_tokens") if isinstance(details, dict) else None
    return detail_read if detail_read is not None else read


def cache_write_split(usage: Dict[str, Any]) -> Dict[str, Optional[int]]:
    """The reported lifetime bins. Keep explicit zeros and mark unreadable bins so they cannot look absent."""
    bins = usage.get("cache_creation")
    names = {"ephemeral_5m_input_tokens": "5m", "ephemeral_1h_input_tokens": "1h"}
    if bins is None or bins == {}:
        bins = usage.get("cache_write_tokens_by_ttl")
        names = {"5m": "5m", "1h": "1h"}
    if bins is None:
        return {}
    if not isinstance(bins, dict):
        return {"unknown": None}
    return {names.get(key, "unknown"): count if key in names and type(count) is int and count >= 0 else None
            for key, count in bins.items()}


def cache_write_total(usage: Dict[str, Any], split: Dict[str, Optional[int]]) -> Tuple[Optional[int], bool]:
    """The reply's one write total and whether only malformed aggregates were reported.

    A usable alias (including zero) or both lifetime bins establish the total; otherwise it is unknown.
    """
    malformed = False
    detail_keys = ("cache_write_tokens", "cache_creation_tokens", "cache_creation_input_tokens")
    for source, keys in ((usage.get("prompt_tokens_details"), detail_keys), (usage.get("input_tokens_details"), detail_keys),
                         (usage, ("cache_creation_input_tokens", "cache_write_tokens"))):
        if isinstance(source, dict):
            for key in keys:
                if source.get(key) is None:
                    continue
                count = _valid_count(source[key])
                if count is not None:
                    return count, False
                malformed = True
    short, hourly = split.get("5m"), split.get("1h")
    if len(split) == 2 and short is not None and hourly is not None:
        return short + hourly, False
    return None, malformed


def cache_pricing_usage(usage: Dict[str, Any], written: Optional[int], split: Dict[str, Optional[int]]) -> Dict[str, Any]:
    """Cache counts, their validity and lifetime for the reply estimator, including detail-only reports."""
    from ouroboros.event_usage import event_token_counts

    read = usage.get("cached_tokens")
    if read is None:
        read = event_token_counts(usage)["cached_tokens"]
    return {"cached_tokens": read, "cache_write_tokens": written, "prompt_cache_ttl": usage.get("prompt_cache_ttl"),
            "cache_write_tokens_by_ttl": split or None,
            **({"cache_write_tokens_invalid": True} if usage.get("cache_write_tokens_invalid") else {})}
def observed_processing_mode(provider: str, usage: Dict[str, Any]) -> str:
    """A pricing qualifier from observation; absent explicit-mode proof stays unknown."""
    receipt = usage.get("processing")
    if isinstance(receipt, dict):
        modes = receipt.get("observedNative")
        if receipt.get("observed") in {"unknown", "mixed"} or not isinstance(modes, list) or len(modes) != 1:
            return "unknown"
        value = modes[0]
    else:
        field = "speed" if provider == "anthropic" else "service_tier"
        value = usage.get(field)
        if field in usage and value is None:
            return "unknown"
    if value == "ordinary":  # Host-only price of an unspecified pool, never a served tier.
        return "unknown"
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


_TOKEN_DETAIL_FIELDS = {
    "prompt_tokens_details": ("cached_tokens", "cache_write_tokens", "cache_creation_tokens", "cache_creation_input_tokens"),
    "input_tokens_details": ("cached_tokens", "cache_write_tokens", "cache_creation_tokens", "cache_creation_input_tokens"),
    **_OUTPUT_TOKEN_DETAIL_FIELDS,
    "cache_creation": ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"),
}


def _reported_token_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
    """Only known wire counters, before any alias selection or int/clamp projection."""
    reported = {key: usage[key] for key in (
        "prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "total_tokens",
        "input_tokens", "output_tokens", "cached_tokens", "precached_prompt_tokens", "cache_read_input_tokens",
        "cache_write_tokens", "cache_creation_input_tokens", "reasoning_tokens") if key in usage}
    reported.update({key: {field: usage[key][field] for field in fields if field in usage[key]}
                     for key, fields in _TOKEN_DETAIL_FIELDS.items() if isinstance(usage.get(key), dict)})
    return reported


_MEDIA_DETAIL_FIELDS = frozenset(_TOKEN_DETAIL_FIELDS) | {
    "input_token_details", "output_token_details", "cached_tokens_details",
    "promptTokensDetails", "responseTokensDetails", "response_tokens_details", "cacheTokensDetails", "cache_tokens_details",
}


def nontext_counts(usage: Any, *, wire_only: bool = False) -> Iterator[Any]:
    """Read media units in native usage details, or anywhere in the settlement projection.

    A named non-text modality with an absent count is itself evidence. On the wire an
    extension or a provider-supplied copy of a host field is not a usage detail.
    """
    if isinstance(usage, Mapping):
        if "modality" in usage and str(usage["modality"]).upper() != "TEXT":
            yield usage.get("tokenCount", usage.get("token_count"))
            return
        for key, value in usage.items():
            if key in {"audio_input_tokens", "audio_output_tokens", "image_input_tokens", "image_output_tokens",
                       "audio_tokens", "image_tokens", "video_tokens", "input_audio_tokens", "output_audio_tokens",
                       "input_image_tokens", "output_image_tokens", "cached_audio_input_tokens",
                       "cached_image_input_tokens", "video_input_tokens", "thinking_tokens", "connected_seconds"}:
                if value is not None:
                    yield value
            elif not wire_only or key in _MEDIA_DETAIL_FIELDS:
                yield from nontext_counts(value, wire_only=wire_only)
    elif isinstance(usage, (list, tuple)):
        for item in usage:
            yield from nontext_counts(item, wire_only=wire_only)


def _reported_usage_empty(raw: dict, reported: dict) -> bool:
    """Only absent or explicitly zero work counters can support a pre-generation rejection."""
    counts = [count for key, value in reported.items()
              for count in (value.values() if key in _TOKEN_DETAIL_FIELDS else (value,))]
    if isinstance(tools := raw.get("server_tool_use"), dict):
        counts.append(tools.get("web_search_requests"))
    return (all(count is None or (type(count) is int and count == 0) for count in counts)
            and all(type(count) in (int, float) and count == 0 for count in nontext_counts(raw, wire_only=True)))


def _contradicts_free_rejection(usage: Dict[str, Any]) -> bool:
    """A zero-text rejection that still reports half a price (a BYOK reply's gateway fee without the provider's
    share, or that share alone), non-text usage or server-tool work (a ``server_tool_use`` count that is not zero:
    a search the provider ran) consumed something its text counts do not show: never free."""
    details, tools = usage.get("cost_details"), usage.get("server_tool_use")
    worked = bool(tools) and (not isinstance(tools, dict)
                              or any(count is not None and count != 0 for count in tools.values()))
    # A copy of the host's normalized witness is not evidence of provider work.
    return (usage.get("cost") is not None or worked
            or any(type(count) not in (int, float) or count != 0 for count in nontext_counts(usage, wire_only=True))
            or (isinstance(details, dict) and details.get("upstream_inference_cost") is not None))


def usage_from_response(response: Any, *, payload=_UNREAD_PAYLOAD, provider: str = "",
                        provider_reported_cost_only: bool = False,
                        opencode_go: bool = False) -> Tuple[Dict[str, Any], Optional[float], bool]:
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
    cache_read = cache_read_count(usage)
    split = cache_write_split(usage)
    cache_write, invalid_write = cache_write_total(usage, split)
    native_cache_write = cache_write if usage.get("cache_creation_input_tokens") is not None else None
    input_tokens = _reported_token_count(usage, "input_tokens")
    prompt = _reported_token_count(usage, "prompt_tokens")
    if usage.get("prompt_tokens") is None and any(  # a malformed prompt count is not rebuilt from other fields
        value is not None for value in (input_tokens, native_cache_read, native_cache_write)
    ):
        if "input_tokens_details" in usage or (isinstance(payload, dict) and payload.get("object") == "response"):
            prompt = input_tokens  # Responses already includes cache reads and writes.
        elif (isinstance(payload, dict) and payload.get("type") == "message") or any(
                usage.get(key) is not None for key in ("cache_creation", "cache_read_input_tokens", "cache_creation_input_tokens")):
            prompt = native_prompt_tokens(usage, split, cache_write, invalid_write)  # one total with the adapter
        elif not invalid_write:
            prompt = input_tokens  # No native field: the count already holds its cache reads and writes.
    normalized = {
        **usage,
        "prompt_tokens": prompt,
        "completion_tokens": _reported_token_count(usage, "completion_tokens", "output_tokens"),
        "cached_tokens": cache_read,
        "cache_write_tokens": cache_write,
    }
    normalized.pop("cache_write_tokens_invalid", None)
    if invalid_write:
        normalized["cache_write_tokens_invalid"] = True
    # Tariff validation needs the counts as reported, before normalization reads a malformed one as unknown.
    # Keep only token fields; never accept a provider-supplied witness under this internal key.
    reported = _reported_token_usage(usage)
    normalized["_reported_token_usage"] = copy.deepcopy(reported)
    # Candidate facts and validated engine reports come from their host owners,
    # never arbitrary extensions inside a provider's token-usage block.
    normalized.pop("effort", None)
    normalized.pop("effort_resolution", None)
    normalized.pop("response_provider", None)  # the response's own outer label below, never a nested claim
    normalized.pop("processing", None)  # Native session receipts enter through model_operation_usage.
    normalized.pop("provider_reported_cost_only", None)
    # Messages reports capacity in usage and speed separately. Chat/Responses report tier on the envelope.
    if provider != "anthropic":
        normalized.pop("service_tier", None)
    if isinstance(payload, dict):
        from ouroboros.llm_openai_compatible import _bounded_response_metadata_label

        upstream = _bounded_response_metadata_label(payload.get("provider"))
        if upstream is not None:  # the upstream that served the call; the accounting provider stays the route's
            normalized["response_provider"] = upstream
        if has_reported_service_tier(response, payload):
            normalized["service_tier"] = payload["service_tier"]
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
    if opencode_go:
        cost, final = normalize_go_price(normalized)
        return normalized, cost, final
    cost, invalid, basis = response_reported_price(payload, usage, getattr(response, "total_cost_usd", None))
    normalized.pop("cost_invalid", None)  # Derive from provider prices, never trust their marker.
    if invalid:
        normalized["cost_invalid"] = True
    if basis:  # the same consumed amount the loop lane reports, tagged on the settled row
        normalized.update(cost=cost, cost_basis=basis)
    if cost is not None:
        return normalized, cost, True
    if malformed_token_counts(usage):
        normalized["cost_coverage"] = "partial"  # the count is unknown on the row; settlement prices nothing from it
    completion = normalized["completion_tokens"]
    # The normalized by-TTL alias is not a wire witness for a rejection.
    cache_usage_reported = bool(cache_read or cache_write or (creation is not None and (
        not isinstance(creation, dict) or any(count != 0 for count in creation.values()))))
    if (
        not provider_reported_cost_only and provider_body_error(payload) is not None
        and prompt == 0
        and completion == 0
        and _reported_usage_empty(usage, reported)
        and not invalid  # no amount was reported at all
        and not cache_usage_reported
        and not refusal_body_conflicts(payload)
        and not _contradicts_free_rejection(usage)
    ):
        normalized.update(prompt_tokens=0, completion_tokens=0, cached_tokens=0, cache_write_tokens=0)
        return normalized, 0.0, True
    return normalized, None, False


def usage_from_failure(payload: Any, stream_usage: Any, *, opencode_go: bool = False) -> Tuple[Dict[str, Any], Optional[float], bool]:
    """Settle reported money with this attempt's counters; only stream counters may support a failure estimate."""
    has_frame = isinstance(stream_usage, dict) and bool(stream_usage)
    if not isinstance(payload, dict):
        if not has_frame:
            return usage_from_response(payload, opencode_go=opencode_go)
        payload = {}
    payload = {key: value for key, value in payload.items() if key != "error"}
    combined = merge_usage_reports(payload.get("usage"), stream_usage, previous_total=payload.get("total_cost_usd"))
    if "cost_basis" in combined:
        raw_price, _, basis = reported_price(combined)
        envelope_only = raw_price is None and provider_cost_value(payload.get("total_cost_usd")) is not None
        if envelope_only or (combined["cost_basis"] == BYOK_COST_BASIS and not basis):
            combined.pop("cost_basis")  # An envelope fallback or standalone total does not prove a BYOK sum.
    facts, cost, final = usage_from_response({**payload, "usage": combined}, opencode_go=opencode_go)
    if cost is not None or not has_frame:
        return facts, cost, final
    # Only the stream's counters may support an estimate; the received frame's own qualifiers (its service tier,
    # processing and serving upstream) still name the tariff those counters were billed at.
    frame, _, _ = usage_from_response({**payload, "usage": stream_usage}, opencode_go=opencode_go)
    # A failed stream can report only one side. Keep its counters for the row;
    # a complete text tariff needs both, each as an exact count.
    if (None in (frame["prompt_tokens"], frame["completion_tokens"])
            or any(type(value) is not int or value < 0
                   for key in ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens")
                   if (value := stream_usage.get(key)) is not None)):
        frame["cost_coverage"] = "partial"
    if frame.get("cost_basis") == BYOK_COST_BASIS:
        frame.pop("cost_basis")
    if facts.get("cost_invalid"):
        frame["cost_invalid"] = True
    if "service_tier" not in frame and isinstance(stream_usage.get("service_tier"), str):
        frame["service_tier"] = stream_usage["service_tier"]
    return frame, None, False


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


def failure_settlement_price(error: BaseException, payload: Any, provider: str, *, opencode_go: bool):
    if opencode_go:
        # Go's only designated cash field may be explicitly estimated.
        for evidence in ({"usage": getattr(error, "stream_usage", None)}, payload):
            usage, cost, final = usage_from_response(evidence, opencode_go=True)
            if cost is not None:
                return usage, cost, final
        return None
    if priced := priced_failure_usage(error, payload, provider):
        return (*priced, True)
    return None


_ANSWER_CODE_KEYS = ("code", "type", "provider_code", "provider_error_code", "error_type", "limit_source")
_ANSWER_WALK_LIMIT = 400  # nodes and JSON scans of one answer: a parser bound, never a judgment
_SSE_FRAME = re.compile(r"(?:^|\n)[ \t]*(?:(?:data|event|id|retry)[ \t]*:|:)[^\n]*")
_CITATION_MARKER = re.compile(r"previous (?:response|diagnostic)\s*:", re.I)


def _wire_envelope(prefix: str) -> bool:
    """True when only SSE framing (or whitespace) precedes an embedded JSON document."""
    return not _SSE_FRAME.sub("", prefix).strip()


@dataclass(frozen=True)
class AnswerObject:
    """One object and its place in the received answer; never serialized into usage."""
    value: Dict[str, Any]
    parent: int
    speaker: str
    history: bool
    sentences: Tuple[str, ...]
    codes: Tuple[str, ...]
    message: str = ""
    inline: bool = False  # embedded JSON after non-SSE prose in a string


@dataclass(frozen=True)
class ProviderAnswer:
    """What each party in a provider's answer said (``provider_answer``): the current path outermost first, then
    earlier upstream attempts of the same request (``history``)."""

    status: Optional[int] = None  # the integer ``error.code`` of a top-level error object: an HTTP 200 body's status
    said: Tuple[Tuple[str, str, bool], ...] = ()  # (speaker, sentence, history)
    codes: Tuple[Tuple[str, str, bool], ...] = ()  # (speaker, code or type value, history)

    objects: Tuple[AnswerObject, ...] = field(default=(), repr=False, compare=False)  # structural fields, read once with the sentences
    walk_complete: bool = field(default=True, repr=False, compare=False)  # the bounded walk did not exhaust its budget

    def own_codes(self) -> List[str]:
        return [code for _speaker, code, history in self.codes if not history]

    def innermost(self) -> Tuple[str, str]:
        """``(speaker, sentence)`` of the deepest party on the current path: an upstream's words, not its wrapper's."""
        own = [(speaker, text) for speaker, text, history in self.said if not history]
        return next((row for row in own if row[0] == own[-1][0]), ("", ""))

    def recognized_subtree(self, speaker: str, sentence: str, code: str) -> List[int]:
        """Indices of the quoted current object (the one ``speaker`` said ``sentence`` in, else the one carrying
        ``code``) and its subtree, excluding sibling diagnostics and earlier attempts."""
        root = next((i for i, obj in enumerate(self.objects) if not obj.history and obj.speaker == speaker
                     and (sentence in obj.sentences if sentence else code in obj.codes)), None)
        if root is None:
            return []
        selected = [root]
        for i, obj in enumerate(self.objects):  # a parent is read before its children
            if i > root and not obj.history and obj.parent in selected:
                selected.append(i)
        return selected

    def facts(self, quoted: Tuple[str, str] = ("", ""), **recognition: Any) -> Dict[str, Any]:
        """This answer for the owner line and the mind that continues, sanitized: the quoted ``speaker`` and
        ``sentence`` (the one that carried a recognition, else the innermost party's own; uncut, the arithmetic
        reads it), every sentence as ``[speaker, sentence, history]`` (``said``), the current path's typed values
        as ``[speaker, value]`` (``codes``) and what recognised it (``recognition``); only what it has."""
        from ouroboros.utils import sanitize_tool_result_for_log

        def clean(text: str) -> str:
            return sanitize_tool_result_for_log(" ".join(text.split()))

        speaker, sentence = quoted if quoted[1] else self.innermost()
        facts = {"speaker": speaker, "sentence": clean(sentence), **recognition,
                 "said": [[who, clean(text), history] for who, text, history in self.said],
                 "codes": [[who, code] for who, code, history in self.codes if not history]}
        return {key: value for key, value in facts.items() if value}


def provider_answer(source: Any) -> ProviderAnswer:
    """Read one provider answer structurally; no gateway is named and no phrase matched.

    ``source`` is an exception (its received body, then its explicit causes') or a body. Every error object at
    any depth is read. A sentence is any string with a space, under any key (a ``message`` also as one word); JSON
    inside a string is opened (the string itself, SSE ``data:`` lines, an object with words around it, whose words
    stay a sentence); an explicit citation tail after ``previous response:`` / ``previous diagnostic:`` (#3858) is
    not part of this answer. Codes are ``code``/``type`` and a gateway's typed metadata fields, never ``param``.
    ``provider_name`` names who speaks below it. A list holds earlier attempts (history) only where its own
    document also states a sentence outside lists; otherwise the list is the answer. An engine-typed problem
    (``problem``) keeps its typed attributes only (``typed_values``): its message is the engine's, its body private.
    """
    said: List[Tuple[str, str, bool]] = []
    codes: List[Tuple[str, str, bool]] = []
    objects = []
    left = [_ANSWER_WALK_LIMIT]

    def opened(text: str) -> Tuple[List[Tuple[Any, bool]], str]:
        found, kept, index, decoder = [], [], 0, json.JSONDecoder()
        cited = inline = False
        while left[0] > 0 and (start := text.find("{", index)) >= 0:
            left[0] -= 1
            try:
                value, end = decoder.raw_decode(text, start)
            except ValueError:
                value, end = None, start + 1
            document = isinstance(value, dict) and bool(value)
            segment = text[index:start] if document else text[index:end]
            if not cited and (marker := _CITATION_MARKER.search(segment)):
                kept.append(segment[:marker.start()])
                cited = True
            elif not cited:
                kept.append(segment)
            if document:
                inline = inline or not _wire_envelope(segment)
                if not cited:
                    found.append((value, inline))
            index = end
        tail = text[index:]
        if not cited:
            marker = _CITATION_MARKER.search(tail)
            kept.append(tail[:marker.start()] if marker else tail)
            cited = marker is not None
        raw_words = "".join(kept)
        words = "" if (found and not inline and not cited and _wire_envelope(raw_words)) else " ".join(raw_words.split())
        return found, words

    def states(node: Any) -> bool:
        return isinstance(node, dict) and any(
            states(value) or (isinstance(value, str) and key not in _ANSWER_CODE_KEYS and key != "provider_name"
                              and " " in opened(value)[1])
            for key, value in node.items())

    def walk(node: Any, speaker: str, history: bool, lists_are_history: bool, parent: int = -1,
             inline: bool = False) -> None:
        left[0] -= 1
        if left[0] <= 0:
            return
        if isinstance(node, list):
            for item in node:
                walk(item, speaker, history or lists_are_history, lists_are_history, parent, inline)
            return
        if isinstance(node, str):
            node = {"": node}
        if not isinstance(node, dict):
            return
        index, said_start, code_start = len(objects), len(said), len(codes)
        if isinstance(node.get("provider_name"), str) and node["provider_name"].strip():
            speaker = node["provider_name"].strip()
        inner, message = [], ""
        for key, value in sorted(node.items(), key=lambda item: item[0] != "message"):  # its message leads
            if key in _ANSWER_CODE_KEYS:
                if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
                    codes.append((speaker, str(value).strip(), history))
            elif isinstance(value, str) and key != "provider_name":
                documents, words = opened(value)
                if key == "message":
                    message = words
                if " " in words or (words and key == "message"):  # a ``message`` is a sentence even as one word
                    said.append((speaker, words, history))
                inner.extend((document, None, embedded) for document, embedded in documents)
            elif isinstance(value, (dict, list)):
                inner.append((value, lists_are_history, inline))
        objects.append(AnswerObject(node, parent, speaker, history,
                                    tuple(row[1] for row in said[said_start:]),
                                    tuple(row[1] for row in codes[code_start:]), message, inline))
        for value, rule, embedded in inner:  # an opened document applies the history rule to itself
            walk(value, speaker, history, states(value) if rule is None else rule, index, inline or embedded)

    def read(payload: Any) -> Optional[int]:
        top = payload["error"] if isinstance(payload, dict) and isinstance(payload.get("error"), (dict, str)) else payload
        walk(top, "", False, states(top))
        code = top.get("code") if isinstance(top, dict) else None
        return code if type(code) is int and 100 <= code <= 599 else None

    status = None
    if isinstance(source, BaseException):
        for error in (source, source.__cause__, getattr(source.__cause__, "__cause__", None)):  # explicit causes only
            if error is not None and not isinstance(getattr(error, "problem", None), dict):
                read(provider_failure_payload(error))
    else:
        status = read(source)
    for rows in (said, codes):
        rows.sort(key=lambda row: row[2])  # the current path first, then earlier attempts
    return ProviderAnswer(status, tuple(dict.fromkeys(said)), tuple(dict.fromkeys(codes)), tuple(objects), left[0] > 0)


def refusal_failure_payload(error: BaseException) -> Any:
    """Keep the full received provider error for accounting when the SDK exposes only its inner error."""
    response = getattr(error, "response", None)
    if response is None:
        return provider_failure_payload(error)
    body = _plain(getattr(error, "body", None))
    envelope = _plain(response)
    if envelope is response and not isinstance(envelope, dict) and not callable(getattr(response, "json", None)):
        return body
    if not isinstance(envelope, dict):
        error.refusal_envelope_conflict = True
        if body is None:
            try:
                return response.text  # Private evidence only; this decode failure proves no bill or retry.
            except Exception:
                pass  # Some SDKs have already released or never loaded the response body.
        return body
    if isinstance(body, dict) and "error" not in envelope and "usage" in envelope and "code" not in envelope:
        envelope = {"error": body.get("error", body), **envelope}
    if body and body != envelope and body != envelope.get("error") and not (
            isinstance(body, dict) and all(k in envelope and envelope[k] == v for k, v in body.items())):
        error.refusal_envelope_conflict = True
        if isinstance(body, dict) and _reply_shows_work(body):
            error.refusal_alternate_payload = envelope
            return body
        error.refusal_alternate_payload = body
    error.body = envelope
    return envelope


def _confirmed_zero_bill(payload: Any, cost: Optional[float], final: bool) -> bool:
    if final and cost == 0 and isinstance(payload, dict):
        reports = _billing_reports(((payload, None), (payload.get("error"), None)))
        chosen, _, invalid, confirmed = _reconcile_failure_price(reports, len(reports))
        return chosen is not None and chosen[0] == 0 and confirmed and not invalid
    return False


def response_reported_price(payload: Any, usage: dict, sdk_total: Any = None):
    """The ledger and loop price the original reply before any lossy Chat projection."""
    if not _response_failed(payload):
        total = payload.get("total_cost_usd") if isinstance(payload, dict) else None
        return reported_price(usage, total, sdk_total)
    reports = _billing_reports(((payload, sdk_total), (payload.get("error"), None)))
    chosen, basis, invalid, confirmed = _reconcile_failure_price(reports, len(reports))
    if chosen is None:
        return None, invalid, ""
    cost = chosen[0]
    if cost == 0 and (invalid or (not confirmed and refusal_body_conflicts(payload))):
        return None, True, ""
    return cost, False, basis


def _billing_reports(frames):
    return [(body, sdk_total, _plain(body.get("usage"))) for body, sdk_total in frames if isinstance(body, dict)]


def _reconcile_failure_price(reports, terminal_count):
    """Terminal reports agree; older snapshots can be lower, and wallet bounds never mint totals."""
    has_byok = any(isinstance(raw, dict) and raw.get("is_byok") is True for _, _, raw in reports)
    chosen = None
    basis = ""
    gateway_min = upstream_min = unlabelled_min = 0.0
    malformed = confirmed_zero = False
    for index, (body, sdk_total, raw) in enumerate(reports):
        total = body.get("total_cost_usd")
        price = _refusal_price(raw, total, sdk_total, byok_known=has_byok)
        if price.disagreement:
            return None, "", True, False
        malformed |= price.malformed
        gateway, upstream = price.byok_parts
        gateway_min, upstream_min = max(gateway_min, gateway or 0.0), max(upstream_min, upstream or 0.0)
        unlabelled_min = max(unlabelled_min, price.unlabelled_min)
        if price.total is None:
            continue
        confirmed_zero |= index < terminal_count and price.total == 0 and (
            provider_cost_value(total) == 0 or provider_cost_value(sdk_total) == 0 or bool(price.basis))
        if chosen is None:
            chosen = (price.total, index)
            basis = price.basis
        elif price.total != chosen[0]:
            if index >= terminal_count and price.total < chosen[0]:
                continue  # A later report can include work after this snapshot.
            return None, "", True, False
    if chosen is not None and chosen[0] < max(unlabelled_min, _byok_subtotal((gateway_min, upstream_min))):
        return None, "", True, False
    return chosen, basis, malformed or (has_byok and chosen is None), confirmed_zero


def search_failure_usage(error: BaseException, payload: Any, reported: Any, *, work_seen: bool):
    usage, cost, final = reported or usage_from_response(
        getattr(error, "response", None), payload=payload, provider="openai", provider_reported_cost_only=True)
    if cost is None and isinstance(getattr(error, "stream_usage", None), dict):
        usage, cost, final = usage_from_response({"usage": error.stream_usage}, provider="openai",
                                                provider_reported_cost_only=True)
    if not _confirmed_zero_bill(payload, cost, final) and cost == 0 and (
            work_seen or refusal_conflicts(error, check_stream_framing=False)):
        return usage, None, False
    return usage, cost, final


def priced_failure_usage(error: BaseException, payload: Any, provider: str):
    """Read a failed send's own prices, without deriving a free rejection from zero token counts."""
    import httpx
    from ouroboros.llm_stream import IncompleteProviderStream
    from ouroboros.transport_custody import is_processing_refusal

    stream = {"usage": getattr(error, "stream_usage", None)}
    prior = getattr(error, "stream_prior_usage", None)
    if prior is None and not is_processing_refusal(error):
        for body in (stream, payload):
            usage, cost, final = usage_from_response(body, payload=body, provider=provider, provider_reported_cost_only=True)
            # A later wire close doesn't invalidate a usage bill. An explicit provider failure
            # can contradict a zero in the cumulative stream view, as it can in its own body.
            check_zero = (body is not stream or provider_body_error(payload) is not None) and not _confirmed_zero_bill(
                body, cost, final)
            if final and cost is not None and not (
                    cost == 0 and check_zero and refusal_conflicts(
                        error, check_stream_framing=False, retained_payload=payload)):
                return usage, cost
        return None

    response = getattr(error, "response", None)
    body = payload if isinstance(payload, dict) else {}
    alternate = getattr(error, "refusal_alternate_payload", None)
    frames = [(body, getattr(response, "total_cost_usd", None)), (body.get("error"), None)]
    if isinstance(alternate, dict):
        frames.extend(((alternate, None), (alternate.get("error"), None)))
    reports = _billing_reports(frames)
    choice_terminal_start = len(reports)
    reports.extend(_billing_reports((frame, None) for frame in reversed(getattr(error, "stream_terminal_usage", ()))))
    terminal_count = len(reports)
    # The legacy merged view is not another report on this wire.
    reports.extend(_billing_reports((frame, None) for frame in (reversed(prior) if prior is not None else (stream,))))
    chosen, basis, malformed, confirmed_zero = _reconcile_failure_price(reports, terminal_count)
    if chosen is None:
        return None
    cost, selected = chosen
    confirmed_zero |= choice_terminal_start <= selected < terminal_count
    if cost == 0:
        if malformed and not (confirmed_zero and selected < choice_terminal_start):
            return None  # A malformed snapshot alone does not revoke the terminal envelope's own full zero.
        if selected >= terminal_count:
            if (not (isinstance(error, httpx.TransportError) or type(error) is IncompleteProviderStream)
                    or _response_failed(body) or is_processing_refusal(error)):
                return None
            # A wire interruption does not revoke an explicit bill, but a provisional
            # zero-output report contradicted by observed generation is not that bill.
            raw = reports[selected][2]
            receipt = getattr(error, "stream_receipt", None)
            if (isinstance(receipt, dict) and receipt.get("refusal_prior_work") and isinstance(raw, dict)
                    and any(provider_cost_value(raw.get(key)) == 0 for key in ("completion_tokens", "output_tokens"))):
                return None
        elif not confirmed_zero and refusal_conflicts(error, check_stream_framing=False, retained_payload=payload):
            return None
    body = reports[selected][0]
    usage, _, _ = usage_from_response(body, payload=body, provider=provider, provider_reported_cost_only=True)
    # Reports are newest first; the complete price and each cumulative counter are independent facts.
    latest = {}
    for index, (report, _, raw) in enumerate(reports):
        if isinstance(raw, dict) and raw:
            counts = usage if index == selected else usage_from_response(
                report, payload=report, provider=provider, provider_reported_cost_only=True)[0]
            for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens"):
                if counts[key] is not None:
                    latest.setdefault(key, counts[key])
    if usage.get("cache_write_tokens_invalid"):
        latest.pop("cache_write_tokens", None)  # A malformed reported total is not an absent report.
    usage.update(latest)
    usage.pop("cost_basis", None)  # Only the matching, complete wallet report supplies this basis.
    if basis:
        usage.update(cost=cost, cost_basis=basis)
    return usage, cost


def typed_search_refusal(status, code, error_type) -> bool:
    """A provider rejection that precedes work, subject to this send's other evidence."""
    kinds = (code, error_type)
    if status == 429:
        return any(kind in ("rate_limit_exceeded", "rate_limit_error", "insufficient_quota") for kind in kinds)
    if status in (503, 529):
        return "overloaded_error" in kinds
    return status in (400, 401, 402, 403, 404, 422) and any(kind in (
        "invalid_request_error", "authentication_error", "invalid_api_key", "permission_error",
        "billing_error", "model_not_found", "not_found_error") for kind in kinds)


def _reply_shows_work(body: dict) -> bool:
    routing = body.get("openrouter_metadata")
    if isinstance(routing, dict) and (routing.get("attempts") or routing.get("attempt")):
        return True  # The router reports upstream attempts, with only their HTTP statuses.
    error = body.get("error", body)
    metadata = error.get("metadata") if isinstance(error, dict) else None
    if (body.get("output") or body.get("content") or body.get("status") in ("in_progress", "completed", "incomplete")
            or body.get("stop_reason") in ("end_turn", "stop_sequence", "pause_turn", "max_tokens", "tool_use", "refusal")
            or (isinstance(metadata, dict) and metadata.get("error_type") == "timeout")):
        return True
    choices = body.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict):
                finish = choice.get("finish_reason")
                if isinstance(finish, str) and (finish == "stop" or finish in (
                        OUTPUT_LIMIT_FINISH_REASONS | CONTENT_REFUSAL_FINISH_REASONS | TOOL_CALL_FINISH_REASONS
                        | INTERMEDIATE_FINISH_REASONS)):
                    return True
                message = choice.get("message") or choice.get("delta")
                if isinstance(message, dict) and any(value for key, value in message.items() if key != "role"):
                    return True
    raw = _plain(body.get("usage"))
    price = _refusal_price(raw, body.get("total_cost_usd"))
    incomplete_wallets = isinstance(raw, dict) and raw.get("is_byok") is True and price.total is None
    if price.disagreement or price.malformed or incomplete_wallets:
        return True  # An unreadable, inconsistent or incomplete charge cannot establish a free rejection.
    if price.total or any(price.byok_parts):
        return True
    if not isinstance(raw, dict):
        return False
    return not _reported_usage_empty(raw, _reported_token_usage(raw))


def refusal_body_conflicts(body: Any, status: Optional[int] = None) -> bool:
    if not isinstance(body, dict):
        return False
    if _reply_shows_work(body):
        return True
    error = body.get("error", body)
    if isinstance(error, dict) and error is not body and _reply_shows_work(error):
        return True
    if not isinstance(error, dict):
        return False
    code = error.get("code")
    body_status = error.get("status_code", code if type(code) is int and 100 <= code <= 599 else None)
    if body_status in (408, 504) or (status is not None and body_status is not None and status != body_status):
        return True
    metadata = error.get("metadata")
    if not isinstance(metadata, dict) or not metadata.get("raw"):
        return False
    # Read each quoted JSON/SSE reply, not just the recognizable objects in a diagnostic excerpt.
    raw = metadata["raw"]
    try:
        if isinstance(raw, str) and raw.lstrip().startswith(("{", "[")):
            raw = json.loads(raw)
        elif isinstance(raw, str):
            from ouroboros.llm_stream import _SSEFrames

            # The gateway can quote only the frame data, omitting its final blank line.
            events = _SSEFrames().feed((raw + "\n\n").encode(), final=True)
            raw = []
            for i, (event, data) in enumerate(events):
                if i >= _ANSWER_WALK_LIMIT:
                    return True
                if (event, data) == ("", "[DONE]"):
                    continue
                frame = json.loads(data)
                if not isinstance(frame, dict) or (event not in ("", "message") and event != frame.get("type")):
                    return True
                raw.append(frame)
        replies = raw if isinstance(raw, list) else [raw]
        if not replies or any(not isinstance(reply, dict) for reply in replies):
            return True
        answer = provider_answer({"raw": replies})
    except (ValueError, RecursionError):
        return True
    if not answer.walk_complete:
        return True
    outer_status = status or body_status
    rejected = False
    for obj in answer.objects[1:]:  # skip the synthetic root; its direct children are the quoted frames
        if _reply_shows_work(obj.value):
            return True
        native = obj.value.get("error")
        if obj.parent == 0 and obj.value != {"type": "ping"} and not (
                isinstance(native, dict) and obj.value.get("type") in (None, "error")):
            return True
        if not isinstance(native, dict):
            continue
        code = native.get("code")
        inner_status = native.get("status_code", code if type(code) is int else outer_status)
        if not typed_search_refusal(inner_status, code, native.get("type")):
            return True
        rejected = True
    return not rejected


def refusal_conflicts(source: Any, *, reported_cost: Optional[float] = None, check_stream_framing: bool = True,
                      retained_payload: Any = _UNREAD_PAYLOAD) -> bool:
    """Contrary evidence on the same physical send, including the SDK envelope and explicit causes."""
    if reported_cost is not None and reported_cost > 0:
        return True
    if not isinstance(source, BaseException):
        return refusal_body_conflicts(source)
    import httpx
    from ouroboros.transport_custody import is_pre_dispatch_transport_failure

    seen = set()
    current = source
    status_seen = None
    while isinstance(current, BaseException) and id(current) not in seen:
        seen.add(id(current))
        body = retained_payload if current is source and retained_payload is not _UNREAD_PAYLOAD else refusal_failure_payload(current)
        response = getattr(current, "response", None)
        native_cost = getattr(response, "total_cost_usd", None)
        if native_cost is not None and provider_cost_value(native_cost) != 0:
            return True
        status = getattr(current, "status_code", None)
        wire_status = getattr(response, "status_code", None)
        if getattr(current, "response_body_error", False) and type(wire_status) is int and 200 <= wire_status < 300:
            wire_status = None  # The HTTP success envelope does not override its body's error status.
        prior = getattr(current, "stream_prior_usage", None)
        receipt = getattr(current, "stream_receipt", None)
        receipt = receipt if isinstance(receipt, dict) else {}
        # An explicit Chat error can report zero even when its framing cannot
        # prove no generation. Only billing may inspect that narrower question.
        check_stream = check_stream_framing or prior is None or receipt.get("refusal_prior_work") is not False
        if (getattr(current, "refusal_envelope_conflict", False)
                or (check_stream and getattr(current, "processing_refusal_conflict", False))
                or (status is not None and wire_status is not None and status != wire_status)):
            return True
        capture = getattr(current, "physical_attempt_capture", None)
        state = getattr(capture, "state", None)
        capture_status = getattr(capture, "provider_status_code", None)
        if state == "dispatched" or (state in ("settled", "unresolved") and not typed_search_refusal(
                capture_status, getattr(capture, "provider_code", None), getattr(capture, "provider_error_type", None))):
            return True
        status = status or wire_status or capture_status
        if status is not None:
            if status in (408, 504) or (status_seen is not None and status != status_seen):
                return True
            status_seen = status
        if (receipt.get("model_output_received") or receipt.get("refusal_prior_work") or receipt.get("generation_conflict")
                or (check_stream and getattr(current, "stream_incomplete", False))
                or getattr(current, "receiver_abandoned", False)
                or getattr(current, "explicit_timeout", False) or getattr(current, "code", "") == "model_outcome_unknown"
                or (isinstance(current, (httpx.TransportError, TimeoutError)) and not is_pre_dispatch_transport_failure(current))):
            return True
        if refusal_body_conflicts(body, status) or _reply_shows_work({
            "openrouter_metadata": getattr(current, "openrouter_metadata", None),
        }) or any(_reply_shows_work(frame) for frame in (prior if prior is not None else (
                {"usage": getattr(current, "stream_usage", None)},))):
            return True
        current = current.__cause__
    return False


def processing_refusal_reason(provider, mode, status, body: Any) -> str:
    """The submitted tier and the provider's documented error identify a refusal, independently of cost."""
    native_error = body.get("error", body) if isinstance(body, dict) else None
    if not isinstance(native_error, dict):
        return ""
    if (provider == "anthropic" and mode == "fast" and status == 429
            and native_error.get("type") == "rate_limit_error"):
        return "capacity"
    elif provider in {"openai", "openrouter"} and mode in {"priority", "fast", "flex", "ultrafast"}:
        code = native_error.get("code")
        if mode == "flex" and status in {400, 429, 503} and code in ("resource_unavailable", "unsupported_service_tier"):
            return "capacity"
        elif status == 400:
            tier_param = native_error.get("param") == "service_tier"
            if code == "unsupported_service_tier" or (tier_param and code in ("unsupported_parameter", "unsupported_value")):
                return "unsupported"
    return ""


def typed_values(exc: Any, answer: ProviderAnswer) -> List[str]:
    """Every typed value on a failure's current path, first seen first: the exception's ``code``/``type``, each
    party's codes outermost first, then its capture's. An engine-typed problem speaks through the first alone."""
    capture = getattr(exc, "physical_attempt_capture", None)
    values = (getattr(exc, "code", None), getattr(exc, "type", None), *answer.own_codes(),
              getattr(capture, "provider_code", None), getattr(capture, "provider_error_type", None))
    return list(dict.fromkeys(text for value in values if (text := str(value or "").strip())))


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
    body_error = exc is None or getattr(exc, "response_body_error", False)
    facts = {"stage": "response_body_error" if body_error else "raised_exception",
             "exception_type": type(exc).__name__ if exc is not None else None,
             "provider_status_code": status if isinstance(status, int) else None,
             "provider_code": int(code) if str(code).isascii() and str(code).isdigit() and len(str(code)) < 10 else None,
             "transport_cause_type": attempt_custody_event_fields(exc).get("transport_cause_type") if exc is not None else None}
    sdk_total = getattr(getattr(exc, "response", None), "total_cost_usd", None)
    alternate = getattr(exc, "refusal_alternate_payload", None)
    prior = getattr(exc, "stream_prior_usage", None)
    try:
        facts["evidence_ref"] = write_blob(root, {"attempt_id": attempt_id, "body": payload,
            "exception_type": facts["exception_type"], "message": str(exc) if exc is not None else None,
            "provider_code": code, "provider_error_type": error_type,
            **({"reported_total_cost_usd": sdk_total} if sdk_total is not None else {}),
            **({"alternate_body": alternate} if alternate is not None else {}),
            **({"stream_prior_usage": prior} if prior is not None else {}),
            **({"stream_terminal_usage": exc.stream_terminal_usage} if getattr(exc, "stream_terminal_usage", None) else {}),
            "stream_usage": getattr(exc, "stream_usage", None),
            "stream_receipt": getattr(exc, "stream_receipt", None),
            "stream_evidence": getattr(exc, "stream_evidence", None)})
    except Exception as failure:
        facts["retention_gap"] = type(failure).__name__
    return facts


def model_operation_usage(result: dict, effort: dict | None = None, *, request_kind: str = "chat") -> tuple[dict, float | None, bool]:
    """Normalize explicit model usage; never run the generic body-error/free branch."""
    from ouroboros.effort_evidence import model_effort_usage

    counters = result.get("usage") or {}
    cost_evidence = result.get("cost") or {}
    cash = provider_cost_value(cost_evidence.get("cashUsd"))
    knowledge = cost_evidence.get("knowledge")
    cost = cash if knowledge in {"exact", "estimated"} else None
    if knowledge == "estimated" and cost is None:
        cost = provider_cost_value(cost_evidence.get("estimatedUsd"))
    if request_kind == "image":
        # Image units are not text tokens; preserve evidence without invoking
        # the ledger's text tariff estimator when cash knowledge is unknown.
        return {"cost_basis": "image_generation", "cost_evidence": {
            "usage": copy.deepcopy(counters), "cost": copy.deepcopy(cost_evidence)}}, cost, cost is not None and knowledge == "exact"
    usage = {key: counters.get(native) for key, native in (
        ("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens"),
        ("cached_tokens", "cached_input_tokens"), ("cache_write_tokens", "cache_write_tokens"),
        ("reasoning_tokens", "reasoning_tokens"))}
    usage.update(model_effort_usage(result, effort))
    if isinstance(result.get("processing"), dict):
        usage["processing"] = copy.deepcopy(result["processing"])
    if cost_evidence:
        usage["cost_evidence"] = copy.deepcopy(cost_evidence)
    usage["total_tokens"] = (
        int(usage["prompt_tokens"] or 0) + int(usage["completion_tokens"] or 0)
        if all(usage[key] is not None for key in ("prompt_tokens", "completion_tokens")) else None
    )
    return usage, cost, cost is not None and knowledge == "exact"
