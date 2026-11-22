"""Token facts for diagnostic events; no accounting or historical rewriting."""

from __future__ import annotations

from typing import Any


# An adapter whose billing-shaped counters cannot keep the provider's own report
# (a synthesized 0, an incomplete total) states it here from the RAW usage (#1578).
REPORTED_TOKENS_KEY = "reported_tokens"

TOKEN_COUNTER_ALIASES = {
    "prompt_tokens": ("prompt_tokens", "input_tokens"),
    "completion_tokens": ("completion_tokens", "output_tokens"),
    # A detail block (OpenAI chat/Responses, OpenRouter) or DeepSeek's mirror may hold
    # the only report: adapters lift a positive count to the top level, never a 0.
    "cached_tokens": ("cached_tokens", "cache_read_input_tokens", "prompt_tokens_details.cached_tokens",
                      "input_tokens_details.cached_tokens", "prompt_cache_hit_tokens"),
    "cache_write_tokens": ("cache_write_tokens", "cache_creation_input_tokens",
                           "prompt_tokens_details.cache_write_tokens", "prompt_tokens_details.cache_creation_tokens",
                           "prompt_tokens_details.cache_creation_input_tokens", "input_tokens_details.cache_write_tokens",
                           "input_tokens_details.cache_creation_tokens", "input_tokens_details.cache_creation_input_tokens"),
}


def token_count(*candidates: Any) -> int | None:
    """First usable reported count, preserving zero and unknown independently."""
    for value in candidates:
        if value is None or value == "" or isinstance(value, bool):
            continue
        try:
            number = value if isinstance(value, int) else float(value)
            count = int(number)
            if count >= 0 and number == count:
                return count
        except (TypeError, ValueError, OverflowError):
            pass
    return None


def _field(source: dict, key: str) -> tuple[bool, Any]:
    """``(present, value)``; a dotted key reads one level of a detail block."""
    block, _, name = key.rpartition(".")
    holder = source.get(block) if block else source
    return (name in holder, holder.get(name)) if isinstance(holder, dict) else (False, None)


def event_token_counts(usage: Any, fallback: Any = None) -> dict[str, int | None]:
    """Normalize four independent counters without inventing undisclosed zeros.

    An adapter's ``reported_tokens`` fact outranks its billing-shaped counters."""
    reported = usage.get(REPORTED_TOKENS_KEY) if isinstance(usage, dict) else None
    if isinstance(reported, dict):
        return {name: token_count(reported.get(name)) for name in TOKEN_COUNTER_ALIASES}
    sources = [source for source in (usage, fallback) if isinstance(source, dict)]
    counts = {}
    for name, aliases in TOKEN_COUNTER_ALIASES.items():
        counts[name] = None
        for source in sources:
            if name == "cache_write_tokens" and source.get("cache_write_tokens_invalid"):
                break  # An unreadable write total is no reported count.
            fields = [_field(source, key) for key in aliases]
            if any(present for present, _ in fields):
                counts[name] = token_count(*(value for _, value in fields))
                break
    return counts


def cumulative_reported_tokens(total: Any, usage: Any) -> dict[str, int | None]:
    """A running total's reported counters: known only while every summed call reported it.

    A total without this fact starts empty only where a counter is absent: a present one,
    zero included, was summed or restored (a pre-fact checkpoint) or seeded, and a billing
    zero cannot say which, so it stays unknown instead of becoming a reported 0."""
    total = total if isinstance(total, dict) else {}
    prior = total.get(REPORTED_TOKENS_KEY)
    if not isinstance(prior, dict):
        prior = {name: None if name in total else 0 for name in TOKEN_COUNTER_ALIASES}
    current = event_token_counts(usage)
    before = {name: token_count(prior.get(name)) for name in TOKEN_COUNTER_ALIASES}
    return {name: None if None in (before[name], current[name]) else before[name] + current[name]
            for name in TOKEN_COUNTER_ALIASES}


def anthropic_token_counts(raw: Any) -> dict[str, int | None]:
    """Anthropic native usage: ``input_tokens`` EXCLUDES cache reads and writes, so the
    total prompt is known only when all three of its parts were reported."""
    raw = raw if isinstance(raw, dict) else {}
    parts = [token_count(raw.get(key)) for key in ("input_tokens", "cache_read_input_tokens",
                                                     "cache_creation_input_tokens")]
    return {"prompt_tokens": None if None in parts else sum(parts),
            "completion_tokens": token_count(raw.get("output_tokens")),
            "cached_tokens": parts[1], "cache_write_tokens": parts[2]}
