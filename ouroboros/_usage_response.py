"""Provider-response usage normalization and private physical-failure evidence."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
    content refusal, or a failed/cancelled one — neither text, JSON nor call counts; ``usable``: everything else,
    which the existing finish rules (output-limit cut, tool-call finish, refusal, empty) still read. A reply
    without a terminal status (every Chat-shaped lane) is judged by its finish alone. The reply's items, reason
    and usage are retained by their owners either way."""
    label, state = str(finish or "").strip().lower(), str(status or "").strip().lower()
    if state and state != "completed" and not (state == "incomplete" and label in OUTPUT_LIMIT_FINISH_REASONS):
        return "unusable"
    if state and label in CONTENT_REFUSAL_FINISH_REASONS:
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
    """The ledger count and whether only malformed aggregates were reported.

    A usable alias (including zero) or both lifetime bins can establish the total.
    Otherwise the ledger keeps its historical int/clamp projection for reporting alone.
    """
    malformed = False
    projection = None
    detail_keys = ("cache_write_tokens", "cache_creation_tokens", "cache_creation_input_tokens")
    for source, keys in ((usage.get("prompt_tokens_details"), detail_keys), (usage.get("input_tokens_details"), detail_keys),
                         (usage, ("cache_creation_input_tokens", "cache_write_tokens"))):
        if isinstance(source, dict):
            for key in keys:
                value = source.get(key)
                if value is None:
                    continue
                try:
                    count = int(value)
                    if not isinstance(value, bool) and count >= 0 and (isinstance(value, str) or value == count):
                        return count, False
                except (TypeError, ValueError, OverflowError):
                    pass
                malformed = True
                if projection is None:
                    projection = _reported_token_count(source, key)
    short, hourly = split.get("5m"), split.get("1h")
    if len(split) == 2 and short is not None and hourly is not None:
        return short + hourly, False
    return projection, malformed


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
    cache_read = _reported_token_count(  # DeepSeek's top-level prompt_cache_hit_tokens last; its details block wins
        usage, "cache_read_input_tokens", "cached_tokens", "precached_prompt_tokens", "prompt_cache_hit_tokens",
    )
    split = cache_write_split(usage)
    cache_write, invalid_write = cache_write_total(usage, split)
    native_cache_write = cache_write if usage.get("cache_creation_input_tokens") is not None else None
    input_tokens = _reported_token_count(usage, "input_tokens")
    prompt = _reported_token_count(usage, "prompt_tokens")
    if prompt is None and any(
        value is not None for value in (input_tokens, native_cache_read, native_cache_write)
    ):
        if "input_tokens_details" in usage or (isinstance(payload, dict) and payload.get("object") == "response"):
            prompt = input_tokens  # Responses already includes cache reads and writes.
        elif split and (usage.get("cache_creation") is not None or native_cache_read is not None
                        or native_cache_write is not None or isinstance(payload, dict) and payload.get("type") == "message"):
            if cache_write is not None and not invalid_write:
                prompt = (input_tokens or 0) + (native_cache_read or 0) + cache_write
        elif not invalid_write:
            # Native input excludes cache reads and writes. Older replies may name only the aggregate.
            prompt = (input_tokens or 0) + (native_cache_read or 0) + (native_cache_write or 0)
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    if isinstance(details, dict):
        detail_read = _reported_token_count(details, "cached_tokens")
        cache_read = detail_read if detail_read is not None else cache_read
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
    # Tariff validation needs the reported counts, not the ledger's historical int/clamp projection.
    # Keep only token fields; never accept a provider-supplied witness under this internal key.
    reported = {key: usage[key] for key in (
        "prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "total_tokens",
        "input_tokens", "output_tokens", "cached_tokens", "precached_prompt_tokens", "cache_read_input_tokens",
        "cache_write_tokens", "cache_creation_input_tokens") if key in usage}
    nested_keys = {
        "prompt_tokens_details": ("cached_tokens", "cache_write_tokens", "cache_creation_tokens", "cache_creation_input_tokens"),
        "input_tokens_details": ("cached_tokens", "cache_write_tokens", "cache_creation_tokens", "cache_creation_input_tokens"),
        "completion_tokens_details": ("reasoning_tokens",),
        "cache_creation": ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"),
    }
    reported.update({key: {field: usage[key][field] for field in fields if field in usage[key]}
                     for key, fields in nested_keys.items() if isinstance(usage.get(key), dict)})
    normalized["_reported_token_usage"] = copy.deepcopy(reported)
    # Candidate facts and validated engine reports come from their host owners,
    # never arbitrary extensions inside a provider's token-usage block.
    normalized.pop("effort", None)
    normalized.pop("effort_resolution", None)
    normalized.pop("response_provider", None)  # the response's own outer label below, never a nested claim
    if isinstance(payload, dict):
        from ouroboros.llm_openai_compatible import _bounded_response_metadata_label

        upstream = _bounded_response_metadata_label(payload.get("provider"))
        if upstream is not None:  # the upstream that served the call; the accounting provider stays the route's
            normalized["response_provider"] = upstream
        if "service_tier" in payload:
            normalized["service_tier"] = payload["service_tier"]
        if isinstance(payload.get("processing"), dict):
            normalized["processing"] = dict(payload["processing"])
    if split:
        normalized["cache_write_tokens_by_ttl"] = split
    cost, invalid, basis = reported_price(
        usage, payload.get("total_cost_usd") if isinstance(payload, dict) else None,
        getattr(response, "total_cost_usd", None),
    )
    normalized.pop("cost_invalid", None)  # Derive from provider prices, never trust their marker.
    if invalid:
        normalized["cost_invalid"] = True
    if basis:  # the same consumed amount the loop lane reports, tagged on the settled row
        normalized.update(cost=cost, cost_basis=basis)
    if cost is not None:
        return normalized, cost, True
    completion = normalized["completion_tokens"]
    cache_usage_reported = bool(cache_read or cache_write or (
        split and (cache_write is None or any(count != 0 for count in split.values()))))
    if (
        isinstance(payload, dict)
        and isinstance(payload.get("error"), dict)
        and prompt == 0
        and completion == 0
        and all(value is None or (type(value) is int and value == 0)
                for source in ({key: value for key, value in reported.items() if key not in nested_keys},
                               *(reported.get(key, {}) for key in nested_keys)) for value in source.values())
        and not invalid  # no amount was reported at all
        and not cache_usage_reported
    ):
        normalized.update(prompt_tokens=0, completion_tokens=0, cached_tokens=0, cache_write_tokens=0)
        return normalized, 0.0, True
    return normalized, None, False


def usage_from_failure(payload: Any, stream_usage: Any) -> Tuple[Dict[str, Any], Optional[float], bool]:
    """Settle reported money with this attempt's counters; only stream facts may support a failure estimate."""
    has_frame = isinstance(stream_usage, dict) and bool(stream_usage)
    if not isinstance(payload, dict):
        if not has_frame:
            return usage_from_response(payload)
        payload = {}
    payload = {key: value for key, value in payload.items() if key != "error"}
    combined = merge_usage_reports(payload.get("usage"), stream_usage, previous_total=payload.get("total_cost_usd"))
    if "cost_basis" in combined:
        raw_price, _, basis = reported_price(combined)
        envelope_only = raw_price is None and provider_cost_value(payload.get("total_cost_usd")) is not None
        if envelope_only or (combined["cost_basis"] == BYOK_COST_BASIS and not basis):
            combined.pop("cost_basis")  # An envelope fallback or standalone total does not prove a BYOK sum.
    facts, cost, final = usage_from_response({**payload, "usage": combined})
    if cost is not None or not has_frame:
        return facts, cost, final
    frame, _, _ = usage_from_response({"usage": stream_usage})
    # A failed stream can report only one side. Keep its counters for the row;
    # a complete text tariff needs both, before the ledger's legacy int/clamp projection.
    if (None in (frame["prompt_tokens"], frame["completion_tokens"])
            or any(type(value) is not int or value < 0
                   for key in ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens")
                   if (value := stream_usage.get(key)) is not None)):
        frame["cost_coverage"] = "partial"
    if frame.get("cost_basis") == BYOK_COST_BASIS:
        frame.pop("cost_basis")
    if facts.get("cost_invalid"):
        frame["cost_invalid"] = True
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


_ANSWER_CODE_KEYS = ("code", "type", "provider_code", "provider_error_code", "error_type", "limit_source")
_ANSWER_WALK_LIMIT = 400  # nodes and JSON scans of one answer: a parser bound, never a judgment


@dataclass(frozen=True)
class AnswerObject:
    """One object and its place in the received answer; never serialized into usage."""
    value: Dict[str, Any]
    parent: int
    speaker: str
    history: bool
    sentences: Tuple[str, ...]
    codes: Tuple[str, ...]


@dataclass(frozen=True)
class ProviderAnswer:
    """What each party in a provider's answer said (``provider_answer``): the current path outermost first, then
    earlier upstream attempts of the same request (``history``)."""

    status: Optional[int] = None  # the integer ``error.code`` of a top-level error object: an HTTP 200 body's status
    said: Tuple[Tuple[str, str, bool], ...] = ()  # (speaker, sentence, history)
    codes: Tuple[Tuple[str, str, bool], ...] = ()  # (speaker, code or type value, history)

    objects: Tuple[AnswerObject, ...] = field(default=(), repr=False, compare=False)  # structural fields, read once with the sentences

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
    stay a sentence). Codes are ``code``/``type`` and a gateway's typed metadata fields, never ``param``.
    ``provider_name`` names who speaks below it. A list holds earlier attempts (history) only where its own
    document also states a sentence outside lists; otherwise the list is the answer. An engine-typed problem
    (``problem``) keeps its typed attributes only (``typed_values``): its message is the engine's, its body private.
    """
    said: List[Tuple[str, str, bool]] = []
    codes: List[Tuple[str, str, bool]] = []
    objects = []
    left = [_ANSWER_WALK_LIMIT]

    def opened(text: str) -> Tuple[List[Any], str]:
        found, kept, index, decoder = [], [], 0, json.JSONDecoder()
        while left[0] > 0 and (start := text.find("{", index)) >= 0:
            left[0] -= 1
            try:
                value, end = decoder.raw_decode(text, start)
            except ValueError:
                value, end = None, start + 1
            document = isinstance(value, dict) and bool(value)
            kept.append(text[index:start] if document else text[index:end])
            found.extend([value] if document else [])
            index = end
        return found, " ".join(("".join(kept) + text[index:]).split())

    def states(node: Any) -> bool:
        return isinstance(node, dict) and any(
            states(value) or (isinstance(value, str) and key not in _ANSWER_CODE_KEYS and key != "provider_name"
                              and " " in opened(value)[1])
            for key, value in node.items())

    def walk(node: Any, speaker: str, history: bool, lists_are_history: bool, parent: int = -1) -> None:
        left[0] -= 1
        if left[0] <= 0:
            return
        if isinstance(node, list):
            for item in node:
                walk(item, speaker, history or lists_are_history, lists_are_history, parent)
            return
        if isinstance(node, str):
            node = {"": node}
        if not isinstance(node, dict):
            return
        index, said_start, code_start = len(objects), len(said), len(codes)
        if isinstance(node.get("provider_name"), str) and node["provider_name"].strip():
            speaker = node["provider_name"].strip()
        inner = []
        for key, value in sorted(node.items(), key=lambda item: item[0] != "message"):  # its message leads
            if key in _ANSWER_CODE_KEYS:
                if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
                    codes.append((speaker, str(value).strip(), history))
            elif isinstance(value, str) and key != "provider_name":
                documents, words = opened(value)
                if " " in words or (words and key == "message"):  # a ``message`` is a sentence even as one word
                    said.append((speaker, words, history))
                inner.extend((document, None) for document in documents)
            elif isinstance(value, (dict, list)):
                inner.append((value, lists_are_history))
        objects.append(AnswerObject(node, parent, speaker, history,
                                    tuple(row[1] for row in said[said_start:]),
                                    tuple(row[1] for row in codes[code_start:])))
        for value, rule in inner:  # an opened document applies the history rule to itself
            walk(value, speaker, history, states(value) if rule is None else rule, index)

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
    return ProviderAnswer(status, tuple(dict.fromkeys(said)), tuple(dict.fromkeys(codes)), tuple(objects))


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
