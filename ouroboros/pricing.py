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
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

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

PriceRows = Dict[str, Tuple[Optional[float], ...]]


class PartialEndpointCatalog(NamedTuple):
    """Individually bound endpoint rows, never a price bound for the pool."""

    rows: PriceRows


# Prices are intentionally never seeded from hand-maintained model rows. Each
# provider cache contains only data returned by that exact route's catalog.
_cached_pricing: Dict[str | tuple[str, str], PriceRows | PartialEndpointCatalog] = {}
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


def _fetch_live_rows(provider: str, model: str = "") -> PriceRows | PartialEndpointCatalog:
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
) -> PriceRows:
    """Return pricing from the exact provider route's live catalog.

    Direct/OpenAI-compatible/GigaChat routes have no automatic catalog here and
    therefore return an empty mapping. A cold/expired fetch is bounded to five
    seconds; failures expose unknown pricing rather than a fabricated fallback.
    An OpenRouter route (``model``) selects its endpoint-tagged tariff catalog lazily, sharing this cache's TTL
    and in-flight owner, keyed by its exact public slug (``llm_pricing.openrouter_endpoint_slug``: case kept, a
    ``:nitro``/``:floor`` shortcut shares its model's read). A route with no public slug reads, shares and scans
    nothing: it never falls back to the all-model catalog or another route's warm rows. An incomplete pool
    prices nothing here; the catalog's endpoint-specific view still reports individually bound facts.
    """
    provider = str(provider or "").strip().lower()
    if provider not in CATALOG_PROVIDERS or (model and provider != "openrouter"):
        return {}
    rows = _read_endpoint_catalog(model, allow_live_fetch) if model else _read_catalog(provider, "", allow_live_fetch)
    return {} if isinstance(rows, PartialEndpointCatalog) else dict(rows)


def _read_endpoint_catalog(model: str, allow_live_fetch: bool = True) -> PriceRows | PartialEndpointCatalog:
    from ouroboros.llm_pricing import openrouter_endpoint_slug

    slug = openrouter_endpoint_slug(model)
    return _read_catalog(("openrouter", slug), slug, allow_live_fetch) if slug else {}


def _read_catalog(key: str | tuple[str, str], slug: str,
                  allow_live_fetch: bool) -> PriceRows | PartialEndpointCatalog:
    """One catalog through the shared cache: ``key`` is a provider (its all-model catalog) or
    ``(provider, slug)`` (one model's endpoints, never answered stale)."""
    provider = key[0] if isinstance(key, tuple) else key
    with _pricing_lock:
        cached = _cached_pricing.get(key, {})
        retry_later = time.time() < _pricing_retry_after.get(key, 0.0)
        fresh = retry_later if isinstance(cached, PartialEndpointCatalog) else bool(_pricing_fetched_at.get(key)) and (
            time.time() - _pricing_fetched_at[key]
        ) < _pricing_ttl_sec()
        if not allow_live_fetch or fresh or retry_later or key in _pricing_fetch_in_progress:
            return {} if slug and not fresh else cached
        _pricing_fetch_in_progress.add(key)
    try:
        rows = _fetch_live_rows(provider, slug) if slug else _fetch_live_rows(provider)
    except Exception as exc:
        log.warning("Failed to fetch %s pricing catalog: %s", provider, exc)
        rows = {}
    with _pricing_lock:
        _pricing_fetch_in_progress.discard(key)
        _cached_pricing[key] = rows
        if rows and not isinstance(rows, PartialEndpointCatalog):
            _pricing_fetched_at[key] = time.time()
            _pricing_retry_after.pop(key, None)
        else:
            _pricing_fetched_at.pop(key, None)
            # A missing or incomplete catalogue prices nothing. Avoid its outage delay
            # on every dispatch, but retry on the short cooldown, not the tariff TTL.
            _pricing_retry_after[key] = time.time() + 30.0
        return rows


def declared_token_limits(provider: str, model: str) -> Dict[int, Tuple[str, str]]:
    """The token limits the route's endpoint catalogue declares for ``model``, each with what it limits and where:
    ``max_prompt_tokens`` caps the input, ``context_length`` is a total that holds the reply.

    Read from the endpoint rows ``get_pricing`` keeps beside the tariffs: its cache, else its one free bounded
    fetch; a route without a catalogue, or a failed read, declares nothing. What one refusal's number may equal
    (``context_anchor.cited_limit_refusal``), never a planning limit: one model id spans endpoints with different
    windows, and the observation scope does not name the serving one."""
    limits: Dict[int, Tuple[str, str]] = {}
    route = str(model or "").strip()  # none names no endpoints: never the all-model catalogue
    for tag, row in (get_pricing(provider=provider, model=route) if route else {}).items():
        source = getattr(row, "source", {})
        for key, kind in (("max_prompt_tokens", "input"), ("context_length", "total")):
            if type(source.get(key)) is int and source[key] > 0 and (
                    source[key] not in limits or (kind, limits[source[key]][0]) == ("input", "total")):
                limits[source[key]] = (kind, f"the endpoint catalogue ({tag} {key})")  # an input cap wins: it sheds less
    return limits


# The tier a call asked, from a host tier or the native word its request carried; another word ("auto") prices
# nothing. An explicit Standard (``default``) keeps every other tier out, a sort shortcut's admission included.
# ``ordinary`` is no tier: see ``_ordinary_bound``.
_POOL_MODES = {"standard": "standard", "default": "standard", "fast": "fast", "priority": "fast",
               "economy": "economy", "flex": "economy", "ultrafast": "ultrafast"}
# The endpoint tiers each asked tier may be served and billed by (OpenRouter's service-tier routing): Ultrafast and
# Priority fall back down. Flex with no Flex endpoint routes normally at the default router (``_ROUTED_NORMALLY``),
# where a Priority endpoint of an open-weight model takes part (no catalogue field says which models are open-weight),
# never Ultrafast.
_ROUTED_NORMALLY = ("standard", "fast")
_POOL_TIERS = {"ultrafast": ("ultrafast", "fast", "standard"), "fast": ("fast", "standard"), "standard": ("standard",),
               "economy": _ROUTED_NORMALLY}


def _no_dearer(row: tuple, ceiling: tuple) -> bool:
    """Whether neither ``row``'s input nor its output rate is proven above ``ceiling``'s; an unknown rate proves nothing."""
    for index in (0, 3):
        if row[index] is not None and ceiling[index] is not None and float(row[index]) > float(ceiling[index]):
            return False
    return True


def _rates_at(row: tuple, prompt_tokens: int) -> tuple:
    """The rates ``row`` bills a prompt of ``prompt_tokens`` at: its last input-length tier reached, else its base."""
    rates = row
    for minimum, tier in getattr(row, "tiers", ()):
        if prompt_tokens >= int(minimum):
            rates = tier
    return rates


def processing_pool(endpoints: Mapping[str, tuple], mode: str, *, prompt_tokens: int = 0,
                    pinned: bool = False) -> List[tuple]:
    """The endpoint tariffs (``get_pricing`` rows of one model) a call asked at host tier ``mode`` may be billed by.

    Flex takes its Flex endpoints and any other its rule could admit: no dearer by input and by output than one of
    them, at base rates or at this prompt's input-length band (either reading the provider may apply), an unknown
    rate proving nothing; without a Flex endpoint, what normal routing reaches (``_POOL_TIERS``). An endpoint whose
    tier is unknown (``source["processing"]`` absent) may serve any tier, so it is in every pool; with provider
    routing ``pinned`` (an order or preset may admit any endpoint) the whole list is. The bound overstates rather
    than prices the call below what it may cost."""
    rows = list(endpoints.values())
    if pinned:
        return rows

    def tier(row: tuple) -> Optional[str]:
        return getattr(row, "source", {}).get("processing")

    unknown = [row for row in rows if tier(row) is None]
    prompt = max(0, int(prompt_tokens or 0))
    flex = [row for row in rows if tier(row) == "economy"] if mode == "economy" else []
    if flex:
        return flex + [row for row in rows if tier(row) not in (None, "economy") and any(
            _no_dearer(row, ceiling) or _no_dearer(_rates_at(row, prompt), _rates_at(ceiling, prompt))
            for ceiling in flex)] + unknown
    return [row for row in rows if tier(row) in _POOL_TIERS[mode]] + unknown


def _endpoint_cost(row: tuple, prompt_tokens: int, completion_tokens: int,
                   cache_usage: Optional[Dict[str, Any]]) -> Optional[float]:
    """One endpoint's whole cost for this call, or None when its token rates cannot price it."""
    if getattr(row, "source", {}).get("unpriced"):
        return None
    return _cost_from_pricing(row, prompt_tokens, completion_tokens, cache_usage, exact_cache_tiers=True)


def _ordinary_bound(rows: Mapping[str, tuple], prompt_tokens: int, completion_tokens: int,
                    cache_usage: Optional[Dict[str, Any]]) -> Optional[float]:
    """``processing_mode="ordinary"``: an upper bound for a public OpenRouter call that carries no native tier, which
    a workspace default or preset the host cannot see may serve at ANY tier, Ultra included.

    It prices every endpoint on the selected model's fresh list. Missing tier or billing evidence leaves
    the bound unknown."""
    if not rows or any(not getattr(row, "source", {}).get("processing") for row in rows.values()):
        return None
    costs = [_endpoint_cost(row, prompt_tokens, completion_tokens, cache_usage) for row in rows.values()]
    return max(costs) if all(cost is not None for cost in costs) else None


def _routing_pinned(route: str, *, flex: bool = False) -> bool:
    """Whether OpenRouter provider routing for ``route`` as settings read NOW (``OUROBOROS_OR_PROVIDER`` merged with
    its connection's own, ``llm_routing._resolve_or_provider``) may admit endpoints a tier pool leaves out
    (``llm_routing.routing_admits_any_endpoint``). Only a figure with no candidate reads it: a candidate's own routing
    (``AttemptRequest.provider_routing_pinned``) is what its send carries."""
    from ouroboros.llm_attempt import ConnectionUnavailable
    from ouroboros.llm_routing import _resolve_or_provider, resolve_target, routing_admits_any_endpoint

    try:
        target = resolve_target(route)
    except ConnectionUnavailable:
        return True  # the connection's routing cannot be read: only the whole list bounds its calls
    return routing_admits_any_endpoint(_resolve_or_provider(target.get("provider_routing")), flex=flex)


def wrapup_screen_bound(model: str, native_tier: str, output_limit: int, *,
                        routing_pinned: Optional[bool] = None) -> Optional[float]:
    """A transcript-independent public router bound for an authored last-fit screen.

    None means there is no tariff to consult. Infinity asks for the actual candidate: the previous
    round's input count cannot bound a new prompt, a cache write, or a prompt-dependent tier.
    Only a flat output-only tariff can clear this screen without reading the transcript.
    ``routing_pinned`` is the screened candidate's own routing fact; None reads the settings (``_routing_pinned``).
    """
    from ouroboros.llm_pricing import openrouter_endpoint_slug

    if not openrouter_endpoint_slug(model) or (native_tier and native_tier not in _POOL_MODES):
        return None
    rows = get_pricing(provider="openrouter", model=model)
    if not rows:
        return None
    mode = _POOL_MODES.get(native_tier)
    if not native_tier and any(not getattr(row, "source", {}).get("processing") for row in rows.values()):
        return None  # even the exact ordinary request would have no bound
    pool = (processing_pool(rows, mode, pinned=_routing_pinned(model) if routing_pinned is None else routing_pinned)
            if mode and mode != "economy" else list(rows.values()))  # Flex eligibility depends on input length
    if not pool:
        return None
    cache_writes = normalize_model_identity(model).startswith("anthropic/")
    highest_output = 0.
    for row in pool:
        if getattr(row, "source", {}).get("unpriced"):
            return float("inf") if mode == "economy" else None
        for rates in (row, *(tier for _, tier in getattr(row, "tiers", ()))):
            input_rates = (rates[0], rates[1], rates[2], getattr(rates, "cache_write_1h", None)) if cache_writes else rates[:1]
            if any(rate != 0 for rate in input_rates) or rates[3] is None:
                return float("inf")
            highest_output = max(highest_output, rates[3])
    return highest_output * output_limit / 1_000_000


def endpoint_processing(catalog: PriceRows | PartialEndpointCatalog, only: Optional[list[str]] = None) -> Dict[str, Any]:
    """One model's endpoint catalogue as the tiers it lists: ``{modes, complete, endpoints}``.

    ``modes`` are the host tiers some endpoint's own tier word reads as, in ``CALL_TIERS`` order; ``complete`` says
    every endpoint's word was read (a catalogue fact, never a price or a key's entitlement), so an empty or unread
    catalogue is ``[]`` and false. Each endpoint keeps its ``tag``, its native ``service_tier`` only where its record
    carries the field (null stays null), ``mode`` where read, and ``price`` only where its figure is honest
    (``llm_pricing.openrouter_catalog_price``): a tier without a price is still a tier."""
    from ouroboros.model_slots import CALL_TIERS

    partial = isinstance(catalog, PartialEndpointCatalog)
    endpoints = catalog.rows if partial else catalog
    if only:
        endpoints = {tag: row for tag, row in endpoints.items() if tag in only}
    listed = [{"tag": source["endpoint_tag"], **{key: source[key] for key in ("service_tier", "price") if key in source},
               **({"mode": source["processing"]} if source.get("processing") else {})}
              for source in (getattr(row, "source", {}) for row in endpoints.values()) if source.get("endpoint_tag")]
    seen = {row["mode"] for row in listed if "mode" in row}
    return {"modes": [mode for mode in CALL_TIERS if mode in seen],
            "complete": not (partial or only) and len(listed) == len(endpoints) and bool(listed)
                        and all("mode" in row for row in listed),
            "endpoints": listed}


def route_processing(value: str, read_settings: Any) -> Dict[str, Any]:
    """``GET /api/model-catalog?processing_model=``: the tiers one OpenRouter route's model lists, ``{value, processing}``.

    Only a public slug (``llm_pricing.openrouter_endpoint_slug``) reads anything: then the settings
    (``read_settings()``), and only on a configured OpenRouter connection whose resolved base is OpenRouter's own its
    public endpoint list, through the pricing cache. Any other value, connection or base, an unread or empty list
    is ``[]`` and not complete. With provider restrictions, only explicitly included endpoint tags can
    attest a tier. Aliases, broader provider families and other routing rules stay unverified."""
    from ouroboros.llm_attempt import ConnectionUnavailable
    from ouroboros.llm_pricing import openrouter_endpoint_slug
    from ouroboros.llm_routing import _or_provider_value, processing_transport_modes, resolve_target
    from ouroboros.provider_connections import connection_configured, connection_routes, connection_rows
    from ouroboros.provider_models import CONNECTIONS_KEY

    unknown = {"value": value, "processing": {"modes": [], "complete": False, "endpoints": []}}
    if not openrouter_endpoint_slug(value):
        return unknown
    settings = dict(read_settings())
    rows, connection = connection_rows(settings.get(CONNECTIONS_KEY)), parse_model_route(value).connection
    own = next((known for known in connection_routes(rows) if (known.kind, known.connection) == ("openrouter", connection)),
               None)
    if own is None or not connection_configured(own, settings.get, rows):
        return unknown
    try:
        target = resolve_target(value, settings)
    except ConnectionUnavailable:
        return unknown
    if target.get("provider") != "openrouter" or not processing_transport_modes(target):
        return unknown
    routing = {**_or_provider_value(settings.get("OUROBOROS_OR_PROVIDER")),
               **_or_provider_value(target.get("provider_routing"))}
    if set(routing) - {"only", "allow_fallbacks"} or routing.get("allow_fallbacks", True) is not True:
        return unknown
    only = routing.get("only")
    if "only" in routing and (not isinstance(only, list) or not only or
                              not all(isinstance(tag, str) and tag for tag in only)):
        return unknown
    return {"value": value, "processing": endpoint_processing(_read_endpoint_catalog(value), only)}


def estimate_cost_optional(model: str, prompt_tokens: int, completion_tokens: int, *,
                           cache_usage: Optional[Dict[str, Any]] = None,
                           allow_live_fetch: bool = True,
                           provider: Optional[str] = None,
                           processing_mode: str = "",
                           routing_pinned: Optional[bool] = None) -> Optional[float]:
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

    A named ``processing_mode`` (a host tier or its native word) uses an OpenRouter endpoint-pool upper estimate
    with literal tier prices (``processing_pool``); it is not an exact endpoint quote, and a missing tariff or a
    charge token rates cannot carry keeps the amount unknown. ``ordinary`` (only a public OpenRouter reservation or
    preflight that carries no native tier asks it) bounds every tier on the selected model (``_ordinary_bound``).
    Empty mode keeps the legacy model-catalog estimate: a reference figure, no bound on the endpoint that serves the
    call, and it adds no per-model read. Provider-reported cost remains the settlement authority.
    ``routing_pinned`` is the priced candidate's own routing fact (``llm_routing.routing_admits_any_endpoint``);
    None, for a figure with no candidate, reads the settings (``_routing_pinned``).
    """
    raw_model = str(model or "").strip()
    normalized = normalize_model_identity(raw_model)
    route = str(provider or provider_for_model(raw_model) or "openrouter").strip().lower()
    if route == "local":
        return 0.0
    if processing_mode:
        from ouroboros.llm_pricing import openrouter_endpoint_slug

        mode = _POOL_MODES.get(processing_mode)
        if route != "openrouter" or not openrouter_endpoint_slug(raw_model) or (
                mode is None and processing_mode != "ordinary"):
            return None  # no public slug: nothing to price it by, and no scan of the whole catalogue
        endpoints = get_pricing(provider=route, model=raw_model, allow_live_fetch=allow_live_fetch)
        if mode is None:
            return _ordinary_bound(endpoints, prompt_tokens, completion_tokens, cache_usage)
        pinned = bool(endpoints) and (_routing_pinned(raw_model, flex=mode == "economy")
                                      if routing_pinned is None else routing_pinned)
        pool = processing_pool(endpoints, mode, prompt_tokens=prompt_tokens, pinned=pinned)
        costs = [_endpoint_cost(row, prompt_tokens, completion_tokens, cache_usage) for row in pool]
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
    elif cost is None and provider != "claudexor" and not usage.get("provider_reported_cost_only"):
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


def price_source(provider: Any, settings: Optional[Mapping[str, Any]] = None, model: str = "", *,
                 search: bool = False) -> str:
    """How a route's price is known: ``local`` (no charge), ``subscription`` (the plan's time, not money),
    ``catalog`` (its provider's live catalog, ``get_pricing``), ``unknown`` (a call adds nothing to known spend), or
    only under a condition its published entry for ``model`` (or its ``search``) sets (DeepSeek,
    ``pricing_catalog.price_condition``): ``off_peak_usd`` (a call wholly off-peak, on a key billed in USD) or
    ``usd`` (a search on a key billed in USD)."""
    kind = str(provider or "").strip().lower()
    if kind in ("local", "claudexor"):
        return "local" if kind == "local" else "subscription"
    if kind == "deepseek":
        from ouroboros.pricing_catalog import price_condition

        return price_condition(model, search=search) or "unknown"
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
    """The ``(model, local)`` routes a role's calls may take by this document, as the runtime resolves them: the
    one rule of a role's slot asked over the document (``route_choice.following_route``; a model labelled
    `` (local)`` parses as the local lane). The fallback chain runs its entries on its own box."""
    from ouroboros.route_choice import box_checked, following_route

    read = settings.get
    if role == "fallback":
        local = box_checked(read("USE_LOCAL_FALLBACK"))
        return [(entry.strip(), local) for entry in str(read("OUROBOROS_MODEL_FALLBACKS") or "").split(",") if entry.strip()]
    return [following_route(role, read=read)]


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


def spend_coverage(settings: Mapping[str, Any], subagents: Any = None) -> Tuple[List[str], List[Dict[str, str]]]:
    """What spends money the limits cannot count, as Dashboard › Costs names it under them: ``(not counted,
    partly counted)``. Not counted: a role whose route has no known price, a catalog kind's route resolved to another
    endpoint included (``all model roles`` when none of the
    configured ones has; the Safety Supervisor only while a Safety mode asks it), a leg of built-in web search (a
    later one runs when the earlier fails, as a fallback model does), enabled API helpers (``Agents``;
    ``subagents`` is the resolved list, else the document's). Partly counted, ``{name, condition}``: a group whose
    every route is priced, some only under a ``price_source`` condition (``off_peak_usd`` outranks ``usd``).
    ``off_peak_usd`` prices text input alone, so Vision, whose every call carries an image, is not counted there."""
    from ouroboros.settings_scales import normalize_safety_mode

    def private_base(model: str) -> bool:
        """A route whose resolved endpoint is not its provider's public one is priced only by what that endpoint
        reports (``llm_routing.provider_reported_cost_only``), whatever the public catalog lists; a connection
        that cannot resolve keeps its kind's answer."""
        from ouroboros.llm_routing import provider_reported_cost_only, resolve_target

        try:
            return provider_reported_cost_only(resolve_target(model, dict(settings)))
        except Exception:
            return False

    def source(model: str, local: bool) -> str:
        if not model:
            return "catalog"
        route = parse_model_route(model)
        said = price_source("local" if local else route.kind, settings, "" if local else route.model)
        return "unknown" if said == "catalog" and private_base(model) else said

    def condition(sources: List[str]) -> str:
        return ("unknown" if "unknown" in sources else "off_peak_usd" if "off_peak_usd" in sources
                else "usd" if "usd" in sources else "")

    checks = any(normalize_safety_mode(settings.get(key)) != "off"
                 for key in ("OUROBOROS_SAFETY_MODE", "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE"))
    routed = [(role, label, _role_routes(settings, role)) for role, label in _SPEND_ROLES if role != "safety" or checks]
    routed = [(label, condition([("unknown" if said == "off_peak_usd" and role == "vision" else said)
                                 for said in (source(model, local) for model, local in routes)]))
              for role, label, routes in routed if any(model for model, _ in routes)]
    groups: List[Tuple[str, str]] = []
    for kind in ("unknown", "off_peak_usd"):
        roles = [label for label, said in routed if said == kind]
        groups += [("all model roles", kind)] if len(roles) > 1 and len(roles) == len(routed) else [
            (label, kind) for label in roles]
    try:
        from ouroboros.search_routes import resolve_web_search_route
        legs = resolve_web_search_route(settings=settings).get("legs") or []
    except Exception:
        log.debug("Could not resolve the web search route for the spend gaps", exc_info=True)
        legs = []
    sources = {str(leg.get("source") or "") for leg in legs} - {"", "ddgs"}
    said = condition([price_source(name, settings, search=True) for name in sources])
    if said:
        groups.append(("Web search", said))
    routes = [row.get("route") for row in _subagent_rows(subagents if subagents is not None
                                                        else settings.get("OUROBOROS_SUBAGENTS"))
              if row.get("enabled", True) is not False and isinstance(row.get("route"), Mapping)]
    said = condition([source(str(route.get("target_id") or ""), False) for route in routes
                      if route.get("kind") == "api_model"])
    if said:
        groups.append(("Agents", said))
    return ([name for name, kind in groups if kind == "unknown"],
            [{"name": name, "condition": kind} for name, kind in groups if kind != "unknown"])
