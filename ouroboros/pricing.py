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

from ouroboros.provider_models import normalize_model_identity
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
    "anthropic/claude-opus-4": (15.0, 1.5, 75.0),
    "anthropic/claude-sonnet-4": (3.0, 0.30, 15.0),
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
    # Mirrors the previous static mini lane until live OpenRouter pricing is fetched.
    "openai/gpt-5.5-mini": (0.75, 0.075, 4.50),
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
    # Cloud.ru Foundation Models default (GLM-4.7). Cloud.ru does not expose a
    # generation-cost API and these IDs are absent from OpenRouter pricing, so a
    # static estimate is the only cost source for cloudru-only users (P8 budget
    # integrity). Approximate (per 1M tokens, input/cached/output); refine if
    # Cloud.ru publishes exact rates.
    "cloudru/zai-org/GLM-4.7": (0.50, 0.50, 2.00),
    "cloudru::zai-org/GLM-4.7": (0.50, 0.50, 2.00),
}

_pricing_fetched = False
_cached_pricing = None
_pricing_lock = threading.Lock()


def get_pricing(*, allow_live_fetch: bool = True) -> Dict[str, Tuple[float, ...]]:
    """
    Lazy-load pricing. On first call, attempts to fetch from OpenRouter API.
    Falls back to static pricing if fetch fails.
    Thread-safe via module-level lock.
    """
    global _pricing_fetched, _cached_pricing

    # Single locked path: avoids races between flag/cache updates.
    with _pricing_lock:
        if _cached_pricing is None:
            _cached_pricing = dict(MODEL_PRICING_STATIC)
        if not allow_live_fetch:
            return _cached_pricing
        if _pricing_fetched:
            return _cached_pricing

        try:
            from ouroboros.llm import fetch_openrouter_pricing
            _live = fetch_openrouter_pricing()
            if _live and len(_live) > 5:
                _cached_pricing.update(_live)
            _pricing_fetched = True
        except Exception as e:
            import logging as _log
            _log.getLogger(__name__).warning("Failed to sync pricing from OpenRouter: %s", e)
            # Keep flag false so we retry on next call.
            _pricing_fetched = False

        return _cached_pricing


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
    if raw_model.endswith(" (local)"):
        return "local"
    if raw_model.startswith("openai::"):
        return "openai"
    if raw_model.startswith("anthropic::"):
        return "anthropic"
    if raw_model.startswith("openai-compatible::"):
        return "openai-compatible"
    if raw_model.startswith("cloudru::"):
        return "cloudru"
    if raw_model.startswith("gigachat::"):
        return "gigachat"
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
    raw = str(model or "").strip()
    if raw.endswith(" (local)"):
        raw = raw[:-8]
    if raw.startswith("anthropic::"):
        return "anthropic"
    if raw.startswith("openai::"):
        return "openai"
    if raw.startswith("openai-compatible::"):
        return "openai-compatible"
    if raw.startswith("cloudru::"):
        return "cloudru"
    if raw.startswith("gigachat::"):
        return "gigachat"
    return "openrouter"


def infer_model_category(model: str) -> str:
    """Infer model category by comparing against configured model env vars."""
    model = str(model or "").strip()
    if model.endswith(" (local)"):
        model = model[:-8]
    normalized = normalize_model_identity(model)
    configured = {
        "main": os.environ.get("OUROBOROS_MODEL", ""),
        "code": os.environ.get("OUROBOROS_MODEL_CODE", ""),
        "light": os.environ.get("OUROBOROS_MODEL_LIGHT", ""),
        "fallback": os.environ.get("OUROBOROS_MODEL_FALLBACK", ""),
    }
    for cat, val in configured.items():
        if val and normalized == normalize_model_identity(val):
            return cat
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
