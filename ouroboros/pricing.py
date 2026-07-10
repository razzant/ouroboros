"""
Ouroboros — LLM pricing and cost estimation.

Provides model pricing lookup (static + live OpenRouter sync),
cost estimation from token counts, and usage event emission.
"""

from __future__ import annotations

import os
import queue
import threading
from typing import Any, Dict, Optional, Tuple

import logging

from ouroboros.provider_models import normalize_model_identity, provider_for_model
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

# Historical/backcompat pricing rows keep old logs/settings billable when live
# OpenRouter pricing is unavailable. They are not active runtime defaults.
_LEGACY_GEMINI_31_PRO_PREVIEW = "google/gemini-" + "3.1-pro-preview"
_LEGACY_GEMINI_31_FLASH_LITE = "google/gemini-" + "3.1-flash-lite"
_LEGACY_GEMINI_3_FLASH_PREVIEW = "google/gemini-" + "3-flash-preview"
_LEGACY_GEMINI_25_PRO_PREVIEW = "google/gemini-" + "2.5-pro-preview"
_LEGACY_GEMINI_3_PRO_PREVIEW = "google/gemini-" + "3-pro-preview"

# Static fallback pricing; live OpenRouter pricing is fetched when available.
MODEL_PRICING_STATIC = {
    "anthropic/claude-opus-4.6": (5.0, 0.5, 25.0),
    "anthropic/claude-opus-4-6": (5.0, 0.5, 25.0),
    "anthropic/claude-opus-4.7": (5.0, 0.5, 25.0),
    "anthropic/claude-opus-4-7": (5.0, 0.5, 25.0),
    "anthropic/claude-opus-4.8": (5.0, 0.5, 25.0),
    "anthropic/claude-opus-4-8": (5.0, 0.5, 25.0),
    "anthropic/claude-fable-5": (10.0, 1.0, 50.0),
    "anthropic/claude-opus-4": (15.0, 1.5, 75.0),
    "anthropic/claude-sonnet-4": (3.0, 0.30, 15.0),
    # Sonnet 5 sticker price ($3/$15; intro $2/$10 through 2026-08-31 billed by
    # providers is LOWER, so this static fallback is budget-conservative; live
    # OpenRouter pricing overrides when fetched).
    "anthropic/claude-sonnet-5": (3.0, 0.30, 15.0),
    "anthropic/claude-sonnet-4.6": (3.0, 0.30, 15.0),
    "anthropic/claude-sonnet-4-6": (3.0, 0.30, 15.0),
    "anthropic/claude-sonnet-4.5": (3.0, 0.30, 15.0),
    "openai/o3": (2.0, 0.50, 8.0),
    "openai/o3-pro": (20.0, 20.0, 80.0),
    "openai/o4-mini": (1.10, 0.275, 4.40),
    "openai/gpt-4.1": (2.0, 0.50, 8.0),
    # Mirrors latest available GPT-5 family pricing until live OpenRouter
    # pricing is fetched.
    "openai/gpt-5.5": (5.0, 0.50, 30.0),
    "openai/gpt-5.5-pro": (30.0, 30.0, 180.0),
    # gpt-5.4-mini is the live mini lane (the 5.5 family shipped no mini); 0.15x gpt-5.5.
    "openai/gpt-5.4-mini": (0.75, 0.075, 4.50),
    "openai/gpt-5.2": (1.75, 0.175, 14.0),
    "openai/gpt-5.2-codex": (1.75, 0.175, 14.0),
    "openai/gpt-5.3-codex": (1.75, 0.175, 14.0),
    _LEGACY_GEMINI_25_PRO_PREVIEW: (1.25, 0.125, 10.0),
    _LEGACY_GEMINI_3_PRO_PREVIEW: (2.0, 0.20, 12.0),
    "google/gemini-3.5-flash": (1.50, 0.15, 9.00),
    _LEGACY_GEMINI_31_PRO_PREVIEW: (2.0, 0.20, 12.0),
    _LEGACY_GEMINI_31_FLASH_LITE: (0.25, 0.025, 1.50),
    _LEGACY_GEMINI_3_FLASH_PREVIEW: (0.15, 0.015, 0.60),
    "x-ai/grok-3-mini": (0.30, 0.075, 0.50),
    "qwen/qwen3.5-plus-02-15": (0.40, 0.04, 2.40),
    # Cloud.ru Foundation Models are priced LIVE from cloud.ru's GET /v1/models
    # catalog (per-model metadata.{prompt,generated,cache_*}_tokens_cost, RUB per 1M),
    # synced via llm.fetch_cloudru_pricing() and converted to USD with
    # OUROBOROS_RUB_USD_RATE. The catalog (not a hardcoded row) is the SSOT for every
    # cloud.ru model (GLM, Qwen, DeepSeek, MiniMax, GigaChat-via-cloud, ...). The two
    # rows below are ONLY a last-resort FLOOR for the SHIPPED DEFAULT model
    # (cloudru::zai-org/GLM-4.7, provider_models DIRECT_DEFAULTS) so a transient
    # catalog-fetch failure / offline start never bills the default at $0 (P8 budget
    # integrity); the live catalog overrides them whenever it is reachable.
    "cloudru/zai-org/GLM-4.7": (0.50, 0.50, 2.00),
    "cloudru::zai-org/GLM-4.7": (0.50, 0.50, 2.00),
    # Sber GigaChat tariffs (developers.sber.ru/docs/ru/gigachat/tariffs,
    # effective 2026-02-01, incl. VAT) converted to USD per 1M tokens at the
    # ~90 RUB/USD 2026 rate: GigaChat-2 Lite 65 RUB (~$0.72), Pro 500 RUB
    # (~$5.55), Max 650 RUB (~$7.20). GigaChat bills a single flat per-token
    # rate (no separate input/cached/output), so the tuple repeats it. A
    # GigaChat-only user has no live cost API, so this static table is the only
    # cost source (P8 budget integrity). GigaChat-3 API per-token tariffs are
    # not separately published yet (the 3-series is preview/open-weight), so the
    # GigaChat-3 flagship is approximated at the GigaChat-2 Max tier; refine when
    # Sber publishes official GigaChat-3 API pricing. Longest-prefix match keeps
    # versioned ids (e.g. GigaChat-2-Max:2.0.x) and the bare-Lite fallback correct.
    "gigachat/GigaChat-3-Ultra": (7.20, 7.20, 7.20),
    "gigachat::GigaChat-3-Ultra": (7.20, 7.20, 7.20),
    "gigachat/GigaChat-2-Max": (7.20, 7.20, 7.20),
    "gigachat::GigaChat-2-Max": (7.20, 7.20, 7.20),
    "gigachat/GigaChat-2-Pro": (5.55, 5.55, 5.55),
    "gigachat::GigaChat-2-Pro": (5.55, 5.55, 5.55),
    "gigachat/GigaChat-2": (0.72, 0.72, 0.72),
    "gigachat::GigaChat-2": (0.72, 0.72, 0.72),
    "gigachat/GigaChat": (0.72, 0.72, 0.72),
    "gigachat::GigaChat": (0.72, 0.72, 0.72),
}

import time

_pricing_fetched_at: float = 0.0
_pricing_rate_at_fetch: float = -1.0   # RUB/USD rate baked into the cached cloud.ru rows
_cached_pricing = None
_pricing_ever_fetched: bool = False    # has a live fetch ever populated the cache?
_pricing_fetch_in_progress: bool = False
_pricing_lock = threading.Lock()


def _pricing_ttl_sec() -> float:
    """Live-pricing refetch interval (provider prices/ FX rates drift). Default 6h."""
    try:
        return max(60.0, float(os.environ.get("OUROBOROS_PRICING_TTL_SEC", "") or 21600.0))
    except (TypeError, ValueError):
        return 21600.0


def _current_rub_usd_rate() -> float:
    """RUB->USD rate cloud.ru rows are converted with (mirror of fetch_cloudru_pricing)."""
    try:
        rate = float(os.environ.get("OUROBOROS_RUB_USD_RATE", "") or 95.0)
    except (TypeError, ValueError):
        return 95.0
    return rate if rate > 0 else 95.0


def _fetch_live_rows() -> Tuple[Dict[str, Tuple[float, ...]], bool]:
    """Fetch the LIVE pricing rows (OpenRouter + cloud.ru catalog) only — NOT the static
    table. Returns (live_rows, latch_ok). ``latch_ok`` is True only when every CONFIGURED
    live source succeeded (a transient failure is retried next call, not latched for the
    whole TTL). Returning ONLY live rows lets the caller layer them OVER the existing
    cache, so a partial or total failure keeps the prior good rows for the source that
    did not refresh (never overwriting live pricing with the static-only floor)."""
    import logging as _log
    live: Dict[str, Tuple[float, ...]] = {}
    openrouter_ok = False
    try:
        from ouroboros.llm import fetch_openrouter_pricing
        _or = fetch_openrouter_pricing()
        if _or and len(_or) > 5:
            live.update(_or)
            openrouter_ok = True
        else:
            _log.getLogger(__name__).warning(
                "OpenRouter pricing fetch returned no data; will retry on next call"
            )
    except Exception as e:
        _log.getLogger(__name__).warning("Failed to sync pricing from OpenRouter: %s", e)

    # cloud.ru catalog is the SSOT for cloud.ru models when reachable. If a cloud.ru key
    # IS configured but the fetch fails, do NOT latch. No key => nothing to fetch => ok.
    cloud_key_present = bool((os.environ.get("CLOUDRU_FOUNDATION_MODELS_API_KEY", "") or "").strip())
    cloud_ok = not cloud_key_present
    try:
        from ouroboros.llm import fetch_cloudru_pricing
        _cloud = fetch_cloudru_pricing()
        if _cloud:
            live.update(_cloud)
            cloud_ok = True
    except Exception as e:
        _log.getLogger(__name__).warning("Failed to sync pricing from cloud.ru: %s", e)

    return live, (openrouter_ok and cloud_ok)


def get_pricing(*, allow_live_fetch: bool = True) -> Dict[str, Tuple[float, ...]]:
    """
    Lazy-load pricing. Syncs LIVE from OpenRouter (Anthropic/OpenAI/Google/...) AND
    cloud.ru Foundation Models (per-model catalog, RUB->USD), layered over the static
    table (providers without a pricing API). Refetches after a TTL (or when the RUB/USD
    rate changes) so price/FX drift is picked up.

    Concurrency: the network fetch runs OUTSIDE the cache lock, and while one caller
    refreshes, CONCURRENT callers get the current cached table instead of blocking on the
    ~15-30s round-trip (the lock is only held for the fast read/write). The refreshing
    caller itself blocks on the fetch — the cold population must be synchronous so the
    first cost estimate is real (relied on by callers/tests), and a per-TTL refresh
    blocks at most one caller, once. Live rows are layered OVER the existing cache, so a
    partial/total fetch failure keeps the prior good rows (never drops live -> static).
    """
    global _pricing_fetched_at, _pricing_rate_at_fetch, _cached_pricing
    global _pricing_ever_fetched, _pricing_fetch_in_progress

    rate = _current_rub_usd_rate()
    with _pricing_lock:
        if _cached_pricing is None:
            _cached_pricing = dict(MODEL_PRICING_STATIC)
        now = time.time()
        fresh = (
            _pricing_fetched_at
            and (now - _pricing_fetched_at) < _pricing_ttl_sec()
            and _pricing_rate_at_fetch == rate  # rate change invalidates converted rows
        )
        # Serve the cache without blocking when: live fetch disabled, fresh, or another
        # thread is already refreshing it.
        if not allow_live_fetch or fresh or _pricing_fetch_in_progress:
            return _cached_pricing
        _pricing_fetch_in_progress = True

    try:
        live_rows, latch_ok = _fetch_live_rows()
    except Exception:  # defensive: never let pricing crash the response path
        live_rows, latch_ok = {}, False
    finally:
        with _pricing_lock:
            if live_rows:
                # Layer live rows OVER the existing cache (static floor + prior live),
                # so a source that didn't refresh keeps its last-good rows.
                _cached_pricing = {**_cached_pricing, **live_rows}
                _pricing_ever_fetched = True
            if latch_ok:
                _pricing_fetched_at = time.time()
                _pricing_rate_at_fetch = rate
            _pricing_fetch_in_progress = False
            result = _cached_pricing
    return result


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int,
                  cached_tokens: int = 0, cache_write_tokens: int = 0,
                  prompt_cache_ttl: Optional[str] = None,
                  allow_live_fetch: bool = True) -> float:
    """Estimate cost from token counts using known pricing. Returns 0 if model unknown."""
    raw_model = str(model or "").strip()
    model = normalize_model_identity(raw_model)
    lookup_candidates = list(dict.fromkeys([raw_model, model]))
    model_pricing = get_pricing(allow_live_fetch=allow_live_fetch)
    # Try exact match first
    pricing = next((model_pricing[candidate] for candidate in lookup_candidates if candidate in model_pricing), None)
    if not pricing:
        # Try longest prefix match
        best_match = None
        best_length = 0
        for candidate in lookup_candidates:
            for key, val in model_pricing.items():
                if candidate and candidate.startswith(key):
                    if len(key) > best_length:
                        best_match = val
                        best_length = len(key)
        pricing = best_match
    if not pricing:
        return 0.0
    input_price = float(pricing[0])
    cached_price = float(pricing[1])
    explicit_write_price = float(pricing[2]) if len(pricing) >= 4 else None
    output_price = float(pricing[3] if len(pricing) >= 4 else pricing[2])
    if explicit_write_price is not None:
        write_price = explicit_write_price
    elif str(model or "").strip().startswith(("anthropic/", "anthropic::")):
        write_price = input_price * (2.0 if str(prompt_cache_ttl or "").strip().lower() == "1h" else 1.25)
    else:
        write_price = input_price
    regular_input = max(0, prompt_tokens - cached_tokens - cache_write_tokens)
    cost = (
        regular_input * input_price / 1_000_000
        + cached_tokens * cached_price / 1_000_000
        + cache_write_tokens * write_price / 1_000_000
        + completion_tokens * output_price / 1_000_000
    )
    return round(cost, 6)


def infer_api_key_type(model: str, provider: Optional[str] = None) -> str:
    """Infer which API key is used based on model name."""
    provider_name = str(provider or "").strip().lower()
    if provider_name in {"local", "openrouter", "openai", "anthropic", "openai-compatible", "cloudru", "gigachat"}:
        return provider_name
    raw_model = str(model or "").strip()
    direct_provider = provider_for_model(raw_model)
    # ``openrouter::``-prefixed and un-prefixed ids both bill OpenRouter and
    # fall through to the normalized-identity heuristics below.
    if direct_provider not in ("openrouter",):
        return direct_provider
    normalized = normalize_model_identity(raw_model)
    if normalized.startswith("openai/"):
        return "openrouter"
    if normalized.startswith("openai-compatible/"):
        return "openai-compatible"
    if normalized.startswith("cloudru/"):
        return "cloudru"
    if normalized.startswith("gigachat/"):
        return "gigachat"
    if normalized.startswith(("anthropic/", "google/", "openai/", "x-ai/", "qwen/")):
        return "openrouter"
    if "claude" in normalized.lower():
        return "anthropic"
    return "openrouter"


def infer_provider_from_model(model: str) -> str:
    """Derive the billing provider string from a model identifier.

    Rules (same prefix logic as infer_api_key_type, returns canonical provider name):
      anthropic::*          → "anthropic"
      openai::*             → "openai"
      openai-compatible::*  → "openai-compatible"
      cloudru::*            → "cloudru"
      gigachat::*           → "gigachat"
      anything else         → "openrouter"  (un-prefixed OpenRouter routing)

    Used by review-pipeline emitters to ensure /api/cost-breakdown attribution
    is correct regardless of which provider the model actually routes through.
    """
    provider = provider_for_model(model)
    # Historical billing attribution: local-suffixed ids billed as openrouter.
    return "openrouter" if provider == "local" else provider


def infer_model_category(model: str) -> str:
    """Infer model category by comparing against configured model env vars."""
    model = str(model or "").strip()
    if model.endswith(" (local)"):
        model = model[:-8]
    normalized = normalize_model_identity(model)
    for cat, val in (
        ("main", os.environ.get("OUROBOROS_MODEL", "")),
        ("heavy", os.environ.get("OUROBOROS_MODEL_HEAVY", "")),
        ("light", os.environ.get("OUROBOROS_MODEL_LIGHT", "")),
    ):
        if val and normalized == normalize_model_identity(val):
            return cat
    # Fallbacks is a comma chain -> a model is "fallback" if it is ANY link of the chain
    # (parsed via the shared SSOT, which also honors the legacy singular env), not only
    # when it equals the whole raw comma-string.
    from ouroboros.config import parse_fallback_chain
    for fb in parse_fallback_chain():
        if fb and normalized == normalize_model_identity(fb):
            return "fallback"
    return "other"


def emit_llm_usage_event(
    event_queue: Optional[queue.Queue],
    task_id: str,
    model: str,
    usage: Dict[str, Any],
    cost: float,
    category: str = "task",
    provider: Optional[str] = None,
    source: str = "loop",
    cost_estimated: Optional[bool] = None,
) -> None:
    """
    Emit llm_usage event to the event queue.

    Args:
        event_queue: Queue to emit events to (may be None)
        task_id: Task ID for the event
        model: Model name used for the LLM call
        usage: Usage dict from LLM response
        cost: Calculated cost for this call
        category: Budget category (task, evolution, consciousness, review, summarize, other)
    """
    if not event_queue:
        return
    try:
        resolved_provider = provider or ("local" if str(model or "").endswith(" (local)") else "openrouter")
        event_queue.put_nowait({
            "type": "llm_usage",
            "ts": utc_now_iso(),
            "task_id": task_id,
            "model": model,
            "api_key_type": infer_api_key_type(model, resolved_provider),
            "model_category": infer_model_category(model),
            "provider": resolved_provider,
            "source": source,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "cached_tokens": int(usage.get("cached_tokens") or 0),
            "cache_write_tokens": int(usage.get("cache_write_tokens") or 0),
            "prompt_cache_ttl": str(usage.get("prompt_cache_ttl") or ""),
            "cost": cost,
            "cost_estimated": (
                bool(cost_estimated)
                if cost_estimated is not None
                else bool(usage.get("cost_estimated")) or not bool(usage.get("cost"))
            ),
            "usage": usage,
            "category": category,
        })
    except Exception:
        log.debug("Failed to put llm_usage event to queue", exc_info=True)
