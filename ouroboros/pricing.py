"""
Ouroboros — LLM pricing and cost estimation.

Provides best-effort provider-catalog pricing, nullable cost estimation, and
usage event emission. Missing pricing is data, not a model-admission gate.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import logging

from ouroboros.provider_models import normalize_model_identity, parse_model_route, provider_for_model
from ouroboros.utils import utc_now_iso
from ouroboros.config import runtime_setting

log = logging.getLogger(__name__)


class PricingSchedule(tuple):
    """A tuple-compatible base price with provider prompt-length tiers."""

    def __new__(
        cls,
        base: Tuple[Optional[float], ...],
        tiers: Tuple[Tuple[int, Tuple[Optional[float], ...]], ...] = (),
        *, cache_write_1h: Optional[float] = None, source: Optional[Dict[str, Any]] = None,
    ) -> "PricingSchedule":
        value = super().__new__(cls, base)
        value.tiers = tuple(sorted(tiers, key=lambda item: int(item[0])))
        value.cache_write_1h = cache_write_1h
        value.source = dict(source or {})
        return value

# Prices are intentionally never seeded from hand-maintained model rows. Each
# provider cache contains only data returned by that exact route's catalog.
_cached_pricing: Dict[str | tuple[str, str], Dict[str, Tuple[Optional[float], ...]]] = {}
_pricing_fetched_at: Dict[str | tuple[str, str], float] = {}
_pricing_retry_after: Dict[str | tuple[str, str], float] = {}
_pricing_fetch_in_progress: set[str | tuple[str, str]] = set()
_pricing_lock = threading.Lock()


def _pricing_ttl_sec() -> float:
    """Live-pricing refetch interval (provider prices/ FX rates drift). Default 6h."""
    try:
        return max(60.0, float(runtime_setting("OUROBOROS_PRICING_TTL_SEC", "") or 21600.0))
    except (TypeError, ValueError):
        return 21600.0


def _fetch_live_rows(provider: str, model: str = "") -> Dict[str, Tuple[Optional[float], ...]]:
    if provider == "openrouter":
        if model:
            from ouroboros.llm_pricing import fetch_openrouter_endpoint_pricing
            return fetch_openrouter_endpoint_pricing(model, timeout_sec=5.0)
        from ouroboros.llm import fetch_openrouter_pricing
        return fetch_openrouter_pricing(timeout_sec=5.0)
    if provider == "cloudru":
        from ouroboros.llm import fetch_cloudru_pricing
        return fetch_cloudru_pricing(timeout_sec=5.0)
    return {}


# The routes ``get_pricing`` reads a live catalog for. Cloud.ru prices in roubles: its catalog is known only with a
# positive ``OUROBOROS_RUB_USD_RATE`` (llm_pricing.fetch_cloudru_pricing).
CATALOG_PROVIDERS = frozenset({"openrouter", "cloudru"})


def get_pricing(
    *, provider: str = "openrouter", allow_live_fetch: bool = True, model: str = "",
) -> Dict[str, Tuple[Optional[float], ...]]:
    """Return pricing from the exact provider route's live catalog.

    Direct/OpenAI-compatible/GigaChat routes have no automatic catalog here and
    therefore return an empty mapping. A cold/expired fetch is bounded to five
    seconds; failures expose unknown pricing rather than a fabricated fallback.
    An exact OpenRouter model selects its endpoint-tagged tariff catalog lazily,
    sharing this cache's TTL and in-flight owner. It never fetches all endpoints.
    """
    provider = str(provider or "").strip().lower()
    if provider not in CATALOG_PROVIDERS or (model and provider != "openrouter"):
        return {}
    exact = str(model or "").strip()  # the slug the endpoints API names; the cache keys its identity
    model = normalize_model_identity(exact) if exact else ""
    key = (provider, model) if model else provider
    with _pricing_lock:
        cached = dict(_cached_pricing.get(key, {}))
        fresh = bool(_pricing_fetched_at.get(key)) and (
            time.time() - _pricing_fetched_at[key]
        ) < _pricing_ttl_sec()
        retry_later = time.time() < _pricing_retry_after.get(key, 0.0)
        if not allow_live_fetch or fresh or retry_later or key in _pricing_fetch_in_progress:
            return {} if model and not fresh else cached
        _pricing_fetch_in_progress.add(key)
    try:
        rows = _fetch_live_rows(provider, exact) if model else _fetch_live_rows(provider)
    except Exception as exc:
        log.warning("Failed to fetch %s pricing catalog: %s", provider, exc)
        rows = {}
    with _pricing_lock:
        _pricing_fetch_in_progress.discard(key)
        _cached_pricing[key] = dict(rows)
        if rows:
            _pricing_fetched_at[key] = time.time()
            _pricing_retry_after.pop(key, None)
        else:
            _pricing_fetched_at.pop(key, None)
            # Avoid adding the same five-second outage delay to every dispatch.
            # This is process-local and deliberately short, not a stale tariff cache.
            _pricing_retry_after[key] = time.time() + 30.0
        return dict(rows)


def estimate_cost_optional(model: str, prompt_tokens: int, completion_tokens: int, *,
                           cache_usage: Optional[Dict[str, Any]] = None,
                           allow_live_fetch: bool = True,
                           provider: Optional[str] = None,
                           processing_mode: str = "") -> Optional[float]:
    """Estimate cost from exact provider/model data, preserving unknown as None.

    ``cache_usage`` and everything after it are KEYWORD-ONLY: the 4th slot used
    to be a positional ``cached_tokens: int``, and a stale positional caller
    would otherwise be silently coerced through the isinstance guard to ``{}``,
    dropping cache accounting invisibly. Keyword-only makes such a caller a
    loud ``TypeError`` instead.

    ``cache_usage`` folds the prompt-cache facts into one mapping (the <8-parameter
    contract; keys are the usage-row field names, all optional):

    - ``cached_tokens``: prompt tokens served from cache (read tier).
    - ``cache_write_tokens``: prompt tokens written to cache this call.
    - ``prompt_cache_ttl``: the requested write tier (``"5m"``/``"1h"``).
    - ``cache_write_tokens_by_ttl``: Anthropic's per-tier write split
      (``usage.cache_creation`` → ``{"5m": n, "1h": n}``), harvested when the
      provider reports it: on a ``1h`` request whose payload also produced 5m
      writes (e.g. a server-tool block cached at the default tier beside the 1h
      prefix) only the genuine 1h share bills the extended-tier price. Absent the
      split, every write bills the reported tier — the pre-split behavior, never
      a loosened ratio.

    A named ``processing_mode`` uses an OpenRouter endpoint-pool upper estimate
    with literal tier prices, including Standard fallback for Priority. It is
    not an exact endpoint quote; missing eligible tariffs keep the amount unknown.
    Empty mode retains the legacy model-catalog estimate. Provider-reported cost
    remains the settlement authority.
    """
    raw_model = str(model or "").strip()
    normalized = normalize_model_identity(raw_model)
    route = str(provider or provider_for_model(raw_model) or "openrouter").strip().lower()
    if route == "local":
        return 0.0
    if processing_mode:
        mode = {"standard": "default", "fast": "priority"}.get(processing_mode, processing_mode)
        if route != "openrouter" or mode not in {"default", "priority", "flex"}:
            return None
        endpoints = get_pricing(provider=route, model=raw_model, allow_live_fetch=allow_live_fetch)
        # A bound over all published tier-eligible routes, not a quote from an
        # endpoint we did not select. Retain unhealthy rows: health can change
        # during dispatch without invalidating the tariff we must reserve for.
        # OpenRouter priority may fall back to Standard; Flex uses only Flex
        # when present, otherwise the provider routes normally at Standard.
        tiers = {row.source.get("service_tier") for row in endpoints.values()}
        eligible = ({"priority", "default"} if mode == "priority" else
                    {"flex"} if mode == "flex" and "flex" in tiers else {"default"})
        costs = [_cost_from_pricing(row, prompt_tokens, completion_tokens, cache_usage, exact_cache_tiers=True)
                 for row in endpoints.values() if row.source.get("service_tier") in eligible]
        return max(costs) if costs and all(cost is not None for cost in costs) else None
    model_pricing = get_pricing(provider=route, allow_live_fetch=allow_live_fetch)
    pricing = model_pricing.get(normalized)
    if not pricing:
        return None
    return _cost_from_pricing(pricing, prompt_tokens, completion_tokens, cache_usage)


def _cost_addends(pricing: tuple, prompt_tokens: int, completion_tokens: int, cache_usage: Optional[Dict[str, Any]],
                  *, exact_cache_tiers: bool = False) -> Optional[Tuple[float, ...]]:
    """One complete tariff's five addends in dollars for one call: uncached input, cache reads, default-tier
    cache writes, extended-tier cache writes, output; None when a price it needs is unknown."""
    cache_row = cache_usage if isinstance(cache_usage, dict) else {}
    cached_tokens = int(cache_row.get("cached_tokens") or 0)
    cache_write_tokens = int(cache_row.get("cache_write_tokens") or 0)
    prompt_cache_ttl = cache_row.get("prompt_cache_ttl")
    cache_write_tokens_by_ttl = cache_row.get("cache_write_tokens_by_ttl")
    tiers = getattr(pricing, "tiers", ())
    for min_prompt_tokens, tier_pricing in tiers:
        if max(0, int(prompt_tokens or 0)) >= int(min_prompt_tokens):
            pricing = tier_pricing
    if len(pricing) != 4 or pricing[0] is None or pricing[3] is None:
        return None
    input_price = float(pricing[0])
    cached_price = float(pricing[1]) if pricing[1] is not None else None
    write_price = float(pricing[2]) if pricing[2] is not None else None
    output_price = float(pricing[3])
    extended_write_tokens = 0
    if (write_price is not None or exact_cache_tiers) and str(prompt_cache_ttl or "") == "1h":
        # Split writes before pricing them. Endpoint schedules use the exact
        # 1h tariff; the legacy model-only path keeps its existing 2/1.25 ratio.
        extended_write_tokens = max(0, int(cache_write_tokens or 0))
        if isinstance(cache_write_tokens_by_ttl, dict):
            try:
                reported_1h = int(cache_write_tokens_by_ttl.get("1h") or 0)
            except (TypeError, ValueError):
                reported_1h = extended_write_tokens
            # Clamp into the reported total: a malformed split never bills MORE
            # extended-tier tokens than were written at all.
            extended_write_tokens = max(0, min(reported_1h, extended_write_tokens))
    if cached_tokens and cached_price is None:
        return None
    regular_input = max(0, prompt_tokens - cached_tokens - cache_write_tokens)
    default_write_tokens = max(0, int(cache_write_tokens or 0) - extended_write_tokens)
    extended_write_price = (getattr(pricing, "cache_write_1h", None) if exact_cache_tiers
                            else float(write_price or 0.0) * (2.0 / 1.25))
    if default_write_tokens and write_price is None or extended_write_tokens and extended_write_price is None:
        return None
    return (
        regular_input * input_price / 1_000_000,
        cached_tokens * float(cached_price or 0.0) / 1_000_000,
        default_write_tokens * float(write_price or 0.0) / 1_000_000,
        extended_write_tokens * float(extended_write_price or 0.0) / 1_000_000,
        completion_tokens * output_price / 1_000_000,
    )


def _cost_from_pricing(pricing: tuple, prompt_tokens: int, completion_tokens: int,
                       cache_usage: Optional[Dict[str, Any]], *, exact_cache_tiers: bool = False) -> Optional[float]:
    """Price one complete tariff; endpoint bounds compare whole costs, not rate maxima."""
    addends = _cost_addends(pricing, prompt_tokens, completion_tokens, cache_usage, exact_cache_tiers=exact_cache_tiers)
    if addends is None:
        return None
    cost = addends[0] + addends[1] + addends[2] + addends[3] + addends[4]  # the order the one expression added them in
    return cost if exact_cache_tiers else round(cost, 6)


COST_CLASSES = ("cache_write", "cache_read", "output", "uncached_input")


def display_tariffs(provider: Optional[str]) -> Tuple[Dict[str, tuple], str]:
    """The route's tariffs a price DISPLAY may use, never fetched: what this process holds, else what the install's
    dated catalogue record holds (OpenRouter; the server prices no call itself). Keys are
    ``normalize_model_identity`` names; the second value is the record's date when the record answered."""
    route = str(provider or "openrouter").strip().lower()
    rows = get_pricing(provider=route, allow_live_fetch=False)
    if rows or route != "openrouter":
        return rows, ""
    from ouroboros.llm_pricing import recorded_openrouter_pricing

    return recorded_openrouter_pricing()


def cost_class_weights(pricing: Optional[tuple], usage: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """What ``pricing`` charges one call for each class of its tokens (``COST_CLASSES``), as WEIGHTS for sharing
    that call's RECORDED cost out by class; None when a price or a token count it needs is unknown.

    ``usage`` carries the usage-row fields (``prompt_tokens``, ``completion_tokens``, ``cached_tokens``,
    ``cache_write_tokens``, ``prompt_cache_ttl``, the reported ``cache_write_tokens_by_ttl``). These are the
    addends ``_cost_from_pricing`` sums; only their ratios are meant, the recorded cost stays the amount."""
    counts = [usage.get(name) for name in ("prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens")]
    if not pricing or any(count is None for count in counts):
        return None  # a count the row does not have is unknown, never zero
    prompt, completion, read, writes = (max(0, int(count)) for count in counts)
    lifetimes = write_lifetimes(usage, writes)
    if lifetimes is None:
        return None  # written at a lifetime the row does not name: never assumed to be the cheaper one
    addends = _cost_addends(pricing, prompt, completion, {"cached_tokens": read, "cache_write_tokens": writes})
    if addends is None:
        return None
    rates = pricing
    for min_prompt_tokens, tier_pricing in getattr(pricing, "tiers", ()):
        if prompt >= int(min_prompt_tokens):
            rates = tier_pricing
    write_price = float(rates[2] or 0.0)
    extended = getattr(rates, "cache_write_1h", None)  # a long-prompt tier's own 1-hour price, as the bill takes it
    extended = float(extended) if extended is not None else write_price * (2.0 / 1.25)  # the estimator's own ratio
    uncached, cache_read, _written, _extended, output = addends
    write = (lifetimes[0] * write_price + lifetimes[1] * extended) / 1_000_000
    return {"cache_write": write, "cache_read": cache_read, "output": output, "uncached_input": uncached}


def write_lifetimes(usage: Dict[str, Any], writes: int) -> Optional[Tuple[int, int]]:
    """``(5-minute, 1-hour)`` written tokens: the provider's own split when it accounts for every write, else the
    one lifetime the row names; None when neither says (``default`` or an absent lifetime, a partial split)."""
    if writes == 0:
        return 0, 0
    split = usage.get("cache_write_tokens_by_ttl")
    if (isinstance(split, dict) and split and set(split) <= {"5m", "1h"}
            and all(type(count) is int and count >= 0 for count in split.values()) and sum(split.values()) == writes):
        return split.get("5m", 0), split.get("1h", 0)
    named = str(usage.get("prompt_cache_ttl") or "")
    return (writes, 0) if named == "5m" else (0, writes) if named == "1h" else None


def infer_api_key_type(model: str, provider: Optional[str] = None) -> str:
    """Infer which API key is used: an explicit provider first, else the model name.

    ``claudexor`` is explicit route evidence (#1578): a session's seat or observed
    model text never relabels it as an OpenRouter or Anthropic key, empty model included."""
    provider_name = str(provider or "").strip().lower()
    if provider_name in {"local", "openrouter", "openai", "anthropic", "openai-compatible", "cloudru", "gigachat",
                         "minimax", "deepseek", "zai", "claudexor"}:
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
    # NB: un-prefixed "minimax/..." deliberately falls through to OpenRouter below —
    # unlike cloudru/gigachat, minimax IS a real OpenRouter vendor namespace, and
    # slash-form ids stay router-style by design (direct routing uses minimax::,
    # already resolved by provider_for_model above). Classifying minimax/ as the
    # direct key would make safety.py demand MINIMAX_API_KEY on OpenRouter installs.
    if normalized.startswith(("anthropic/", "google/", "openai/", "x-ai/", "qwen/", "minimax/", "deepseek/")):
        return "openrouter"
    if "claude" in normalized.lower():
        return "anthropic"
    return "openrouter"


def infer_provider_from_model(model: str) -> str:
    """Derive the billing provider string from a model identifier.

    Rules (same prefix logic as infer_api_key_type, returns canonical provider name;
    the registry drives it, so every direct prefix — anthropic::, openai::,
    openai-compatible::, cloudru::, gigachat::, minimax::, deepseek:: — maps to
    its provider):
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
        ("main", runtime_setting("OUROBOROS_MODEL", "")),
        ("heavy", runtime_setting("OUROBOROS_MODEL_HEAVY", "")),
        ("light", runtime_setting("OUROBOROS_MODEL_LIGHT", "")),
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


def normalize_usage_cost(usage: Dict[str, Any], *, model: str, use_local: bool,
                         provider: str = "openrouter") -> tuple[Optional[float], str, str, bool]:
    """Project reported cash, estimating only absent prices; mutate the usage facts together.

    Callers pass normalized usage, including the host's ``cost_invalid`` marker.
    Local execution is the explicit free exception; subscription absence is unknown.
    """
    from ouroboros._usage_response import observed_processing_mode, provider_cost_value

    raw_cost = usage.get("cost")
    cost = provider_cost_value(raw_cost)
    display_model = str(usage.get("resolved_model") or model)
    provider = "local" if use_local else str(usage.get("provider") or provider)
    estimated = bool(usage.get("cost_estimated"))
    if use_local:
        cost, estimated = 0.0, False
        display_model = f"{model} (local)"
        usage.update(cost_invalid=False, cost_final=True)
    elif cost is None and (raw_cost is not None or usage.get("cost_invalid")):
        log.warning("Provider reported an invalid cost (type=%s) for %s; keeping unknown, skipping estimation",
                    type(raw_cost).__name__, display_model)
        usage["cost_invalid"] = True
    elif cost is None and provider != "claudexor":
        processing_mode = observed_processing_mode(provider, usage)
        cost = estimate_cost_optional(
            display_model, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0),
            cache_usage={
                "cached_tokens": int(usage.get("cached_tokens") or 0),
                "cache_write_tokens": int(usage.get("cache_write_tokens") or 0),
                "prompt_cache_ttl": usage.get("prompt_cache_ttl"),
                "cache_write_tokens_by_ttl": usage.get("cache_write_tokens_by_ttl")
                    if isinstance(usage.get("cache_write_tokens_by_ttl"), dict) else None,
            },
            provider=provider, **({"processing_mode": processing_mode} if processing_mode else {}),
        )
        estimated = cost is not None
    usage.update(cost=cost, cost_estimated=estimated)
    if cost is None or estimated:
        usage["cost_final"] = False
    return cost, display_model, provider, estimated


def emit_llm_usage_event(
    event_queue: Optional[queue.Queue],
    task_id: str,
    model: str,
    usage: Dict[str, Any],
    cost: Optional[float],
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
        category: Budget category (task, evolution, consciousness, consciousness_task,
            review, summarize, other)
    """
    if not event_queue:
        return
    try:
        # Task-tree attribution from the bound usage scope (worker-side truth;
        # the supervisor additionally backfills lane/role from RUNNING).
        root_task_id = parent_task_id = ""
        try:
            from ouroboros.usage_accounting import current_usage_scope

            scope = current_usage_scope()
            if scope is not None:
                root_task_id = str(scope.root_task_id or "")
                parent_task_id = str(scope.parent_task_id or "")
        except Exception:
            pass
        from ouroboros.event_usage import event_token_counts

        resolved_provider = provider or str(usage.get("provider") or "") or (
            "local" if str(model or "").endswith(" (local)") else "openrouter")
        event_queue.put_nowait({
            "type": "llm_usage",
            "ts": utc_now_iso(),
            "task_id": task_id,
            "root_task_id": root_task_id,
            "parent_task_id": parent_task_id,
            "model": model,
            "api_key_type": infer_api_key_type(model, resolved_provider),
            "model_category": infer_model_category(model),
            "provider": resolved_provider,
            "source": source,
            **{key: usage[key] for key in ("llm_call_id", "execution_id", "round_id", "round", "session_id")
               if key in usage},
            # #1578: each counter is reported, a measured 0, or null — never an invented zero.
            **event_token_counts(usage),
            "prompt_cache_ttl": str(usage.get("prompt_cache_ttl") or ""),
            "cost": cost,
            "cost_estimated": (
                bool(cost_estimated)
                if cost_estimated is not None
                else bool(usage.get("cost_estimated"))
            ),
            "usage": usage,
            "category": category,
            # Compatibility telemetry only. Monetary authority is the durable
            # physical-attempt ledger; ids allow joining without double charge.
            "accounting_authority": "physical_attempt_ledger",
            "ledger_attempt_ids": list(usage.get("ledger_attempt_ids") or []),
        })
    except Exception:
        log.debug("Failed to put llm_usage event to queue", exc_info=True)


def price_source(provider: Any, settings: Optional[Mapping[str, Any]] = None) -> str:
    """How a route's price is known: ``local`` (no charge), ``subscription`` (the plan's time, not money),
    ``catalog`` (its provider's live catalog, ``get_pricing``) or ``unknown`` (a call adds nothing to known spend)."""
    kind = str(provider or "").strip().lower()
    if kind in ("local", "claudexor"):
        return "local" if kind == "local" else "subscription"
    if kind == "cloudru":
        read = settings.get if settings is not None else (lambda key: runtime_setting(key, ""))
        try:
            rate = float(read("OUROBOROS_RUB_USD_RATE") or 0)
        except (TypeError, ValueError):
            rate = 0.0
        # One catalog for every Cloud.ru connection, read with the built-in one's key as the resolver finds it
        # (llm_pricing.fetch_cloudru_pricing).
        from ouroboros.provider_models import model_has_credentials, provider_has_credentials_in_settings
        keyed = (provider_has_credentials_in_settings("cloudru", dict(settings)) if settings is not None
                 else model_has_credentials("cloudru::"))
        return "catalog" if rate > 0 and keyed else "unknown"
    return "catalog" if kind in CATALOG_PROVIDERS else "unknown"


# The Models roles in their order, with the label Settings gives each.
_SPEND_ROLES = (("main", "Main"), ("light", "Light"), ("safety", "Safety Supervisor"), ("vision", "Vision"),
                ("consciousness", "Consciousness"), ("fallback", "Fallback"))


def _role_routes(settings: Mapping[str, Any], role: str) -> List[Tuple[str, bool]]:
    """The ``(model, local)`` routes a role's calls may take by this document, as the runtime resolves them (a model
    labelled `` (local)`` parses as the local lane). A slot naming its own model takes its own flag
    (route_choice.model_role_slot_override). An empty one runs Main's model: Consciousness on its own flag when set,
    Vision on Main's (slot_lane_label), Light on both (its helpers keep Main's flag: the runtime splits). Safety runs
    its own slot, else a set Light, else Main, on that slot's flag (safety_config)."""
    from ouroboros.model_slots import MODEL_ROLE_SETTINGS

    def flag(key: str, words: Tuple[str, ...] = ("true", "1")) -> Optional[bool]:
        # A stored False is set, as ``runtime_setting`` reads it ('False'); only an absent or blank key is unset.
        raw = settings.get(key)
        if raw is None or not str(raw).strip():
            return None
        return str(raw).strip().lower() in words

    def route(model: Any, local: Optional[bool]) -> Tuple[str, bool]:
        return str(model or "").strip(), bool(local)

    def own(slot: str) -> str:
        return str(settings.get(MODEL_ROLE_SETTINGS[slot]) or "").strip()

    main = route(own("main"), flag("USE_LOCAL_MAIN"))
    if role == "main":
        return [main]
    if role == "fallback":
        local = flag("USE_LOCAL_FALLBACK")
        return [route(entry, local) for entry in own("fallback").split(",") if entry.strip()]
    if role == "safety":
        slot = "safety" if own("safety") else "light" if own("light") else "main"
        return [route(own(slot), flag(f"USE_LOCAL_{slot.upper()}", ("true", "1", "yes", "on")))]
    local = flag(f"USE_LOCAL_{role.upper()}")
    if own(role):
        return [route(own(role), local)]
    if local is None or role == "vision":
        return [main]
    return list(dict.fromkeys([route(own("main"), local)] + ([main] if role == "light" else [])))


def _subagent_rows(value: Any) -> List[Mapping[str, Any]]:
    """The catalog's rows: each enabled one can run even with the catalog switched off, since a review names any
    enabled row (reviewer_slot_config.catalog_review_row)."""
    if isinstance(value, str):
        try:
            value = json.loads(value or "null")
        except ValueError:
            return []
    items = value.get("items") if isinstance(value, Mapping) else value
    return [row for row in items if isinstance(row, Mapping)] if isinstance(items, list) else []


def known_spend_gaps(settings: Mapping[str, Any], subagents: Any = None) -> List[str]:
    """What spends money the limits cannot count, as Dashboard › Costs names it under them: a role whose route has
    no known price (``all model roles`` when none of the configured ones has), the first leg of built-in web
    search, and enabled API helpers (``Agents``; ``subagents`` is the resolved list, else the document's)."""
    def unpriced(model: str, local: bool) -> bool:
        return bool(model) and price_source("local" if local else parse_model_route(model).kind, settings) == "unknown"

    routed = [(label, _role_routes(settings, role)) for role, label in _SPEND_ROLES]
    routed = [(label, routes) for label, routes in routed if any(model for model, _ in routes)]
    roles = [label for label, routes in routed if any(unpriced(model, local) for model, local in routes)]
    gaps = ["all model roles"] if len(roles) > 1 and len(roles) == len(routed) else roles
    try:
        from ouroboros.search_routes import resolve_web_search_route
        legs = resolve_web_search_route(settings=settings).get("legs") or []
    except Exception:
        log.debug("Could not resolve the web search route for the spend gaps", exc_info=True)
        legs = []
    first = str((legs[0] if legs else {}).get("source") or "")
    if first and first != "ddgs" and price_source(first, settings) == "unknown":
        gaps.append("Web search")
    for row in _subagent_rows(subagents if subagents is not None else settings.get("OUROBOROS_SUBAGENTS")):
        route = row.get("route") if isinstance(row.get("route"), Mapping) else {}
        target = str(route.get("target_id") or "")
        if row.get("enabled", True) is not False and route.get("kind") == "api_model" and unpriced(target, False):
            gaps.append("Agents")
            break
    return gaps
