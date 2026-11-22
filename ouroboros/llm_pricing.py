"""Provider price catalogs and settled-cost projection.

Prices are never hand-maintained here: each catalog is read from the provider
that will bill the call, and a missing price stays unknown rather than
inheriting a synthetic coefficient. Explicit late generation-price receipts
belong to ``openrouter_cost`` and the accounting writer, never normalization.
"""


from __future__ import annotations

import logging
import re
from math import isfinite
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from ouroboros.provider_models import normalize_model_identity
from ouroboros.config import runtime_setting
from ouroboros.net_transport import requests_verify_kwargs

if TYPE_CHECKING:
    from ouroboros.pricing import PartialEndpointCatalog, PriceRows

# The moved warnings keep the logger identity they were emitted under.
log = logging.getLogger("ouroboros.llm")


def add_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Accumulate usage from one LLM call into a running total.

    The numeric fold below is the billing shape; events read the separate cumulative
    ``reported_tokens`` fact, unknown per counter once any summed call was (#1578)."""
    from ouroboros.event_usage import REPORTED_TOKENS_KEY, cumulative_reported_tokens
    from ouroboros.request_wire_recovery import merge_request_wire_usage

    total[REPORTED_TOKENS_KEY] = cumulative_reported_tokens(total, usage)  # reads the pre-fold seed
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens"):
        total[k] = int(total.get(k) or 0) + int(usage.get(k) or 0)
    if (cost := usage.get("cost")) is not None:
        total["cost"] = float(total.get("cost") or 0) + float(cost)
    if usage.get("cost_estimated") is True:
        total["cost_estimated"] = True
    if cost is None or usage.get("cost_partial") is True:
        total["cost_partial"] = True
    if total.get("cost_partial") or total.get("cost_estimated") or usage.get("cost_final") is False:
        total["cost_final"] = False
    merge_request_wire_usage(total, usage)


def _openrouter_rate_per_million(value: Any, *, strict: bool = False) -> Optional[float]:
    """Convert a stated per-token rate to dollars per million, keeping a missing or invalid rate unknown."""
    from decimal import Decimal, DecimalException

    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
        if not number.is_finite() or number < 0:
            return None
        result = float(number * 1_000_000)
        return result if isfinite(result) else None
    except (DecimalException, TypeError, ValueError, OverflowError) as exc:
        if strict:
            raise ValueError("Malformed OpenRouter token price") from exc
        return None


_OPENROUTER_TOKEN_RATES = ("prompt", "input_cache_read", "input_cache_write", "completion", "input_cache_write_1h")
_OPENROUTER_MODEL_RATE_CAP = 1000


def _openrouter_base_rates(pricing: dict, *, strict: bool = False) -> Tuple[Optional[float], ...]:
    return tuple(_openrouter_rate_per_million(pricing.get(key), strict=strict and key != "input_cache_write_1h")
                 for key in _OPENROUTER_TOKEN_RATES)


def _openrouter_minimum(rule: dict) -> Optional[int]:
    value = rule.get("min_prompt_tokens")
    return value if type(value) is int and value >= 0 else None


def _openrouter_has_utc_condition(rule: dict) -> bool:
    return any(str(key).startswith("utc_") for key in rule)


def _openrouter_override_schedule(raw: Any, base_rates: tuple, *, source: Optional[dict] = None,
                                  model_rate_cap: Optional[float] = None):
    """Normalize provider-order, per-key overrides to the effective rows at each length boundary.

    A missing key leaves the last applicable value (or base) alone. With an unresolved condition,
    only listed rates that could change become unknown; a later definite price still wins.
    """
    from ouroboros.pricing import PricingSchedule

    raw = [] if raw is None else raw
    if not isinstance(raw, list) or any(not isinstance(rule, dict) for rule in raw):
        return PricingSchedule((None,) * 4, source=source)
    rules = []
    for rule in raw:
        minimum = _openrouter_minimum(rule)
        uncertain = minimum is None or _openrouter_has_utc_condition(rule)
        changes = {}
        for index, key in enumerate(_OPENROUTER_TOKEN_RATES):
            if key not in rule:
                continue
            rate = _openrouter_rate_per_million(rule[key])
            if (model_rate_cap is not None and key in ("prompt", "completion")
                    and rate is not None and rate > model_rate_cap):
                rate = None
            changes[index] = rate
        rules.append((minimum, uncertain, changes))

    def row_at(prompt_tokens: int):
        rates = list(base_rates)
        for minimum, uncertain, changes in rules:
            if minimum is not None and prompt_tokens <= minimum:
                continue
            for index, rate in changes.items():
                rates[index] = None if uncertain and rates[index] != rate else rate
        return PricingSchedule(tuple(rates[:4]), cache_write_1h=rates[4], source=source)

    base = row_at(0)
    thresholds = sorted({minimum for minimum, _, _ in rules if minimum is not None})
    tiers = tuple((minimum, row_at(minimum + 1)) for minimum in thresholds)
    return PricingSchedule(base, tiers, cache_write_1h=base.cache_write_1h, source=source)


def openrouter_price_row(model: Dict[str, Any]) -> Optional[Tuple[Optional[float], ...]]:
    """One model catalog record as four per-1M rates and its separate stated hourly write rate.

    Missing, negative or implausibly large base input/output prices give no model row. An
    unreadable base rate parsed strictly raises; invalid stated override rates are unknown.
    """
    pricing = model.get("pricing")
    if not isinstance(pricing, dict) or pricing.get("prompt") is None or pricing.get("completion") is None:
        return None
    rates = _openrouter_base_rates(pricing, strict=True)
    prompt_price, _, _, completion_price, _ = rates
    if prompt_price is None or completion_price is None:
        return None
    if max(prompt_price, completion_price) > _OPENROUTER_MODEL_RATE_CAP:
        model_id = str(model.get("id") or "").strip()
        log.warning("Skipping %s: prices seem wrong (prompt=%s, completion=%s)", model_id, prompt_price, completion_price)
        return None
    return _openrouter_override_schedule(pricing.get("overrides"), rates, model_rate_cap=_OPENROUTER_MODEL_RATE_CAP)


def _number(value: Any) -> bool:
    try:
        return isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def openrouter_catalog_price(model: Any, cache_ttl: str = "") -> Optional[Dict[str, Any]]:
    """The price a model picker shows for one OpenRouter catalog record, USD per 1M tokens.

    ``{"input", "output"}``, plus ``tiers`` (``[{"min_prompt_tokens", "input",
    "output"}]``) when the price changes with input length: the row accounting
    reads (``openrouter_price_row``). ``cache_write`` is the record's own figure for
    the prompt-cache tier the caller names (``cache_ttl``, the install's resolved
    setting: ``1h`` reads ``input_cache_write_1h``, any other named tier
    ``input_cache_write``), only for a price without length tiers and only where the
    record states that field: never the other tier's figure, never a derived one, none
    when no tier is named. ``cache_write_ttl`` carries that tier with the figure, so a
    reader says it only while the settings still name it. None, so a picker shows no price rather than
    a partial or false one, when the record states no complete token price, adds a
    per-request fee or bills reasoning at another rate, is free by tokens yet charges for
    something else, varies its price by anything but input length (a time window),
    holds a malformed meter or an out-of-range input/output price in any tier,
    or charges less than four decimals per 1M print (it would read as free).
    """
    pricing = model.get("pricing") if isinstance(model, dict) else None
    if not isinstance(pricing, dict):
        return None
    overrides = pricing.get("overrides")
    overrides = [] if overrides is None else overrides
    try:
        # A per-request fee is part of every call's price, which per-token figures would not say.
        if pricing.get("request") is not None and float(pricing["request"]) != 0:
            return None
        if not isinstance(overrides, list) or not all(
                isinstance(tier, dict) and _openrouter_minimum(tier) is not None and float(tier.get("request") or 0) == 0
                and not _openrouter_has_utc_condition(tier) for tier in overrides):
            return None
        stated = (pricing.get("prompt"), pricing.get("completion"), *(
            tier[key] for tier in overrides for key in ("prompt", "completion") if key in tier))
        # Before displaying a complete token price, keep rejecting tiers whose cache rates
        # could not be parsed even when that invalid value never wins a length band.
        for tier in overrides:
            for key in ("input_cache_read", "input_cache_write"):
                _openrouter_rate_per_million(tier.get(key), strict=True)
        reasoning = pricing.get("internal_reasoning")
        if reasoning is not None and float(reasoning) not in (0, float(pricing.get("completion"))):
            return None  # reasoning tokens billed at their own rate: the output figure is not the output price
        figures = [_openrouter_rate_per_million(value) for value in stated]
        row = openrouter_price_row(model)
    except (TypeError, ValueError, ArithmeticError):
        return None
    tiers = getattr(row, "tiers", ())
    if row is None or any(figure is None or figure > _OPENROUTER_MODEL_RATE_CAP
                          or (figure > 0 and round(figure, 4) == 0) for figure in figures):
        return None
    if not (row[0] or row[3] or any(tier[0] or tier[3] for _, tier in tiers)):
        # The picker would say "free". A cache, image or other meter may charge only in a length tier.
        for band in (pricing, *overrides):
            if any(_number(value) and float(value) > 0 for key, value in band.items()
                   if key not in ("prompt", "completion", "overrides", "min_prompt_tokens") and not isinstance(value, bool)):
                return None
    price = {"input": round(row[0], 4), "output": round(row[3], 4), **({"tiers": [
        {"min_prompt_tokens": minimum, "input": round(tier[0], 4), "output": round(tier[3], 4)}
        for minimum, tier in tiers]} if tiers else {})}
    tier = str(cache_ttl or "").strip()
    stated = pricing.get("input_cache_write_1h" if tier == "1h" else "input_cache_write")
    if tier and not tiers and stated is not None and not isinstance(stated, bool) and _number(stated) and float(stated) >= 0:
        write = round(float(stated) * 1_000_000, 4)
        if write or float(stated) == 0:  # a charge below four decimals would read as free
            price["cache_write"], price["cache_write_ttl"] = write, tier
    return price


def openrouter_pricing_rows(models: Any) -> Dict[str, Tuple[Optional[float], ...]]:
    """OpenRouter ``/models`` records as model_id -> per-1M prices (``openrouter_price_row``), under the id and its
    normalized form: one parser for a fetched answer and for the catalogue the install recorded
    (``recorded_openrouter_pricing``)."""
    pricing_dict = {}
    for model in models:
        row = openrouter_price_row(model)
        if row is None:
            continue
        model_id = str(model.get("id") or "").strip()
        pricing_dict[model_id] = row
        normalized_model_id = normalize_model_identity(model_id)
        if normalized_model_id != model_id:
            pricing_dict[normalized_model_id] = row
    return pricing_dict


def recorded_openrouter_pricing() -> Tuple[Dict[str, Tuple[Optional[float], ...]], str]:
    """Tariffs of the install's dated catalogue record (``llm_capability_policy.recorded_openrouter_catalog``) and
    its date: no fetch and no freshness claim, for a price DISPLAY in a process that has priced no call itself."""
    from ouroboros.llm_capability_policy import recorded_openrouter_catalog

    models, observed = recorded_openrouter_catalog()
    try:
        rows = openrouter_pricing_rows([{"id": mid, "pricing": row["pricing"]} for mid, row in models.items()
                                        if isinstance(row.get("pricing"), dict)])
    except (TypeError, ValueError, KeyError):  # an unreadable record prices nothing
        return {}, ""
    return rows, (observed if rows else "")


def fetch_openrouter_pricing(*, timeout_sec: float = 5.0) -> Dict[str, Tuple[Optional[float], ...]]:
    """Fetch OpenRouter pricing as model_id -> per-1M prices.

    Tuples are ``(input, cached_read, cache_write, output)``. Missing cache
    prices remain ``None`` instead of inheriting a synthetic coefficient.
    """
    import logging
    log = logging.getLogger("ouroboros.llm")

    try:
        import requests
    except ImportError:
        log.warning("requests not installed, cannot fetch pricing")
        return {}

    try:
        url = "https://openrouter.ai/api/v1/models"
        resp = requests.get(url, timeout=max(0.1, min(5.0, float(timeout_sec))), **requests_verify_kwargs())
        resp.raise_for_status()

        data = resp.json()
        models = data.get("data", [])

        pricing_dict = openrouter_pricing_rows(models)
        log.info(f"Fetched pricing for {len(pricing_dict)} models from OpenRouter")
        return pricing_dict

    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning(f"Failed to fetch OpenRouter pricing: {e}")
        return {}


def _charge_condition(pricing: Any) -> str:
    """Read the part of an endpoint tariff that its published token rates cannot account for.

    ``unknown`` denotes an unreadable schedule, including a condition on the base row or an unreadable window bound.
    ``unpriced`` denotes an unreadable figure, a missing base text rate, or a charge that the
    token estimator cannot account for. An override may leave individual rates unknown:
    the schedule handles its numeric UTC bounds without deciding which window applies.
    An unpriced row still retains its readable rates for comparison, never estimation.
    """
    overrides = pricing.get("overrides") if isinstance(pricing, dict) else None
    overrides = [] if overrides is None else overrides
    if not isinstance(pricing, dict) or not isinstance(overrides, list) or not all(
            isinstance(tier, dict) for tier in overrides):
        return "unknown"
    stated = [pricing, *overrides]
    if _openrouter_has_utc_condition(pricing) or any(
            isinstance(value, bool) for part in stated for value in part.values()):
        return "unknown"
    for index, part in enumerate((pricing, *({**pricing, **tier} for tier in overrides))):
        given = {key: value for key, value in part.items() if key not in ("overrides", "min_prompt_tokens")
                 and value is not None}
        if not all(_number(value) for key, value in given.items() if str(key).startswith("utc_")):
            return "unknown"  # An unreadable window is an unreadable schedule.
        if not all(_number(value) for key, value in given.items() if not str(key).startswith("utc_")):
            return "unpriced"  # an unreadable or non-finite charge is no free one
        figures = {key: float(value) for key, value in given.items() if not str(key).startswith("utc_")}
        if index == 0 and ("prompt" not in figures or "completion" not in figures):
            return "unpriced"
        token_fields = {"prompt", "completion", "input_cache_read", "input_cache_write", "input_cache_write_1h",
                        "internal_reasoning", "discount"}  # discount is metadata on the published rates
        if any(value != 0 for key, value in figures.items() if key not in token_fields):
            return "unpriced"
        if figures.get("internal_reasoning", 0) not in (0, figures.get("completion")):
            return "unpriced"
        if not (figures.get("prompt") or figures.get("completion")) and any(
                value > 0 for key, value in figures.items() if key not in ("prompt", "completion")):
            return "unpriced"
    return ""


def _endpoint_pricing_schedule(pricing: Any, source: dict):
    """Retain exact endpoint rates and source, even when its tariff is unknown."""
    pricing = pricing if isinstance(pricing, dict) else {}
    return _openrouter_override_schedule(pricing.get("overrides"), _openrouter_base_rates(pricing), source=source)


# The endpoints API's ``service_tier`` words (null, ``flex``, ``priority``, ``ultrafast``: no other is documented) as
# host processing modes (``model_slots.CALL_TIERS``). Null is the endpoint's ordinary tier, Standard; an absent field,
# another word or a non-string is an unknown tier, never Standard. A tag (``…/fast``) is a name, never a tier.
OPENROUTER_ENDPOINT_TIERS = {"flex": "economy", "priority": "fast", "ultrafast": "ultrafast"}

# Routing shortcuts that sort one model's endpoint list and admit a tier into the sort, never narrowing it:
# ``:nitro`` admits Priority endpoints, ``:floor`` Flex ones. Their tariffs are the model's.
_POOL_SHORTCUTS = frozenset({"nitro", "floor"})

# A public OpenRouter model slug as its endpoints API names one: ``author/slug``, nothing a route head could hide.
_PUBLIC_SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._:~-]*")


def openrouter_endpoint_mode(endpoint: Any) -> Optional[str]:
    """The host mode one endpoint record's own ``service_tier`` names, or None when it names none it can be read as."""
    if not isinstance(endpoint, dict) or "service_tier" not in endpoint:
        return None
    native = endpoint["service_tier"]
    return "standard" if native is None else OPENROUTER_ENDPOINT_TIERS.get(native) if isinstance(native, str) else None


def _openrouter_route_model(model: str) -> str:
    """An OpenRouter route's model as written (``openrouter[@name]::`` folded), '' for any other route."""
    from ouroboros.provider_models import parse_model_route

    route = parse_model_route(str(model or "").strip())
    return route.model.strip() if route.valid and route.kind == "openrouter" and not route.local else ""


def openrouter_pool_sort(model: str) -> str:
    """The route's sort shortcut, ``nitro`` or ``floor``, or ''."""
    _base, colon, variant = _openrouter_route_model(model).rpartition(":")
    return variant if colon and variant in _POOL_SHORTCUTS else ""


def openrouter_endpoint_slug(model: str) -> str:
    """The slug OpenRouter's endpoints API lists a route under, or '' when it is no public OpenRouter slug.

    An OpenRouter route's model, a sort shortcut dropped (``openrouter_pool_sort``). Any other head, a ``::`` left
    inside (an unknown kind stays an opaque id: ``parse_model_route``) or a character no public slug has is none,
    so nothing private is sent to the public catalogue, and its case is kept: two spellings are two models."""
    slug = _openrouter_route_model(model)
    if openrouter_pool_sort(model):
        slug = slug.rpartition(":")[0]
    return slug if "::" not in slug and _PUBLIC_SLUG.fullmatch(slug) else ""


def fetch_openrouter_endpoint_pricing(model: str, *, timeout_sec: float = 5.0) -> PriceRows | PartialEndpointCatalog:
    """Read one model's endpoint tariffs. Unbound rows leave only uniquely bound peers as partial evidence.

    Tier words and prices are provider facts from the documented endpoints API: each row's source keeps the
    endpoint's own ``service_tier`` only where the record states it as null or a string, ``processing`` the host
    mode it reads as (``openrouter_endpoint_mode``; absent when unknown), ``unpriced`` where its token rates cannot
    price a call (``_charge_condition``) and only otherwise ``price``, the figure a picker may show
    (``openrouter_catalog_price``, no cache tier). Status is retained as evidence, never used to erase an
    expensive endpoint; a temporary outage cannot turn a tariff into an execution prohibition.
    """
    from urllib.parse import quote
    from ouroboros.pricing import PartialEndpointCatalog
    import requests

    model = openrouter_endpoint_slug(model)
    author, separator, slug = model.partition("/")
    if not separator or not author or not slug:
        return {}
    url = f"https://openrouter.ai/api/v1/models/{quote(author, safe='')}/{quote(slug, safe='')}/endpoints"
    try:
        response = requests.get(url, timeout=max(0.1, min(5.0, float(timeout_sec))), **requests_verify_kwargs())
        response.raise_for_status()
        payload = response.json().get("data") or {}
        endpoints = payload.get("endpoints")
        if payload.get("id") != model or not isinstance(endpoints, list):
            return {}
        bound, seen = {}, set()
        incomplete = False
        for endpoint in endpoints:
            tag = endpoint.get("tag") if isinstance(endpoint, dict) else None
            if not isinstance(tag, str) or not tag.strip() or endpoint.get("model_id") != model:
                incomplete = True
            elif tag in seen:
                incomplete = True
                bound.pop(tag, None)  # No unique endpoint behind this tag, regardless of row order.
            else:
                seen.add(tag)
                bound[tag] = endpoint
        rows = {}
        for tag, endpoint in bound.items():
            mode, pricing = openrouter_endpoint_mode(endpoint), endpoint.get("pricing")
            condition = _charge_condition(pricing)
            price = None if condition else openrouter_catalog_price({"id": tag, "pricing": pricing})
            native = endpoint.get("service_tier")
            source = {"provider": "openrouter", "model": model, "endpoint_tag": tag, "url": url,
                      "status": endpoint.get("status"),
                      **({"service_tier": native} if "service_tier" in endpoint and (
                          native is None or isinstance(native, str)) else {}),
                      **({"processing": mode} if mode else {}), **({"price": price} if price else {}),
                      **({"unpriced": True} if condition else {}),
                      # The endpoint's declared token limits, kept beside its tariff (``pricing.declared_token_limits``).
                      **{key: endpoint[key] for key in ("context_length", "max_prompt_tokens") if endpoint.get(key) is not None}}
            # An unreadable schedule shows no rates at all; an unpriced one still shows what it is dearer than.
            rows[tag] = _endpoint_pricing_schedule(None if condition == "unknown" else pricing, source)
        return PartialEndpointCatalog(rows) if incomplete and rows else rows
    except (requests.RequestException, TypeError, ValueError, AttributeError) as error:
        log.warning("Failed to fetch OpenRouter endpoint pricing for %s: %s", model, error)
        return {}


def fetch_cloudru_pricing(*, timeout_sec: float = 5.0) -> Dict[str, Tuple[Optional[float], ...]]:
    """Fetch cloud.ru Foundation Models pricing as ``cloudru/<id>`` -> per-1M USD.

    cloud.ru's ``GET /v1/models`` returns per-model ``metadata`` with token costs
    (``prompt_tokens_cost``, ``generated_tokens_cost``, ``cache_read_tokens_cost``,
    ``cache_write_tokens_cost``) in RUB per 1M tokens — i.e. the real resale price
    the owner pays. We convert to USD via ``OUROBOROS_RUB_USD_RATE`` so the catalog
    is the SSOT for ALL cloud.ru models (no hardcoded per-model table). Models with
    ``is_billable=false`` is an exact free row; missing billability or an absent
    explicit ``OUROBOROS_RUB_USD_RATE`` stays unknown. Returns {} when the catalog
    cannot be queried. Tuples are ``(input, cached_read, cache_write, output)``."""
    import logging
    log = logging.getLogger("ouroboros.llm")

    from ouroboros.llm_routing import resolve_target

    try:
        target = resolve_target("cloudru::")  # the built-in connection's key and endpoint, as dispatch reads them
    except Exception:
        return {}  # a switched-off built-in connection prices nothing
    api_key, base_url = str(target.get("api_key") or "").strip(), str(target.get("base_url") or "")
    if not api_key:
        return {}
    try:
        import requests
    except ImportError:
        return {}

    try:
        rate = float(runtime_setting("OUROBOROS_RUB_USD_RATE", ""))
    except (TypeError, ValueError):
        return {}
    if rate <= 0:
        return {}

    try:
        resp = requests.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=max(0.1, min(5.0, float(timeout_sec))),
            **requests_verify_kwargs(),
        )
        resp.raise_for_status()
        models = resp.json().get("data", []) or []

        def _rub_per_1m_to_usd(value: Any) -> Optional[float]:
            try:
                num = float(value)
            except (TypeError, ValueError):
                return None
            if num < 0:  # cloud.ru uses -1 for "n/a" (e.g. embedding output)
                return None
            return round(num / rate, 6)

        pricing_dict: Dict[str, Tuple[Optional[float], ...]] = {}
        for model in models:
            model_id = str(model.get("id") or "").strip()
            meta = model.get("metadata") if isinstance(model.get("metadata"), dict) else {}
            if not model_id or not meta or meta.get("is_billable") is None:
                continue
            if meta.get("is_billable") is False:
                pricing_dict[normalize_model_identity(f"cloudru::{model_id}")] = (0.0, 0.0, 0.0, 0.0)
                continue
            prompt_price = _rub_per_1m_to_usd(meta.get("prompt_tokens_cost"))
            output_price = _rub_per_1m_to_usd(meta.get("generated_tokens_cost"))
            if prompt_price is None or output_price is None:
                continue
            cached_price = _rub_per_1m_to_usd(meta.get("cache_read_tokens_cost"))
            cache_write_price = _rub_per_1m_to_usd(meta.get("cache_write_tokens_cost"))
            row = (
                prompt_price,
                cached_price,
                cache_write_price,
                output_price,
            )
            pricing_dict[normalize_model_identity(f"cloudru::{model_id}")] = row

        log.info(f"Fetched pricing for {len(pricing_dict)} models from cloud.ru")
        return pricing_dict
    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning(f"Failed to fetch cloud.ru pricing: {e}")
        return {}
