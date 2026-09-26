"""Provider price catalogs and settled-cost projection.

Prices are never hand-maintained here: each catalog is read from the provider
that will bill the call, and a missing price stays unknown rather than
inheriting a synthetic coefficient. The generation-cost fetch is the same fact
arriving late — the authoritative settlement for a call whose response carried
no cost.
"""


from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

from ouroboros.provider_models import normalize_model_identity
from ouroboros.config import runtime_setting
from ouroboros.net_transport import requests_verify_kwargs


# The moved warnings keep the logger identity they were emitted under.
log = logging.getLogger("ouroboros.llm")


def add_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Accumulate usage from one LLM call into a running total."""
    from ouroboros.request_wire_recovery import merge_request_wire_usage

    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens"):
        total[k] = int(total.get(k) or 0) + int(usage.get(k) or 0)
    if usage.get("cost") is not None:
        total["cost"] = float(total.get("cost") or 0) + float(usage["cost"])
        if usage.get("cost_final") is False or usage.get("cost_estimated"):
            total["cost_final"] = False
    else:
        total["cost_final"] = False
    merge_request_wire_usage(total, usage)


def fetch_openrouter_pricing(*, timeout_sec: float = 5.0) -> Dict[str, Tuple[Optional[float], ...]]:
    """Fetch OpenRouter pricing as model_id -> per-1M prices.

    Tuples are ``(input, cached_read, cache_write, output)``. Missing cache
    prices remain ``None`` instead of inheriting a synthetic coefficient.
    """
    import logging
    from ouroboros.pricing import PricingSchedule
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

        pricing_dict = {}
        for model in models:
            model_id = str(model.get("id") or "").strip()

            pricing = model.get("pricing", {})
            if not pricing or pricing.get("prompt") is None or pricing.get("completion") is None:
                continue

            raw_prompt = float(pricing.get("prompt", 0))
            raw_completion = float(pricing.get("completion", 0))
            raw_cached_str = pricing.get("input_cache_read")
            raw_cached = float(raw_cached_str) if raw_cached_str is not None else None
            raw_cache_write_str = pricing.get("input_cache_write")
            raw_cache_write = float(raw_cache_write_str) if raw_cache_write_str is not None else None
            if raw_prompt < 0 or raw_completion < 0:
                continue
            if raw_cached is not None and raw_cached < 0:
                raw_cached = None
            if raw_cache_write is not None and raw_cache_write < 0:
                raw_cache_write = None

            prompt_price = round(raw_prompt * 1_000_000, 4)
            completion_price = round(raw_completion * 1_000_000, 4)
            cached_price = round(raw_cached * 1_000_000, 4) if raw_cached is not None else None
            cache_write_price = (
                round(raw_cache_write * 1_000_000, 4)
                if raw_cache_write is not None else None
            )

            if prompt_price > 1000 or completion_price > 1000:
                log.warning(f"Skipping {model_id}: prices seem wrong (prompt={prompt_price}, completion={completion_price})")
                continue

            row = (prompt_price, cached_price, cache_write_price, completion_price)

            tiers = []
            raw_overrides = pricing.get("overrides") or []
            if isinstance(raw_overrides, list):
                for override in raw_overrides:
                    if not isinstance(override, dict):
                        continue
                    try:
                        min_prompt_tokens = int(override.get("min_prompt_tokens") or 0)
                        if min_prompt_tokens <= 0:
                            continue
                        tier_raw_prompt = float(override.get("prompt", raw_prompt))
                        tier_raw_completion = float(override.get("completion", raw_completion))
                        tier_prompt = round(tier_raw_prompt * 1_000_000, 4)
                        tier_completion = round(tier_raw_completion * 1_000_000, 4)
                        override_cached = override.get("input_cache_read")
                        tier_cached = (
                            round(float(override_cached) * 1_000_000, 4)
                            if override_cached is not None else None
                        )
                        override_write = override.get("input_cache_write")
                        if override_write is not None:
                            tier_write = round(float(override_write) * 1_000_000, 4)
                        else:
                            tier_write = None
                        if tier_prompt > 1000 or tier_completion > 1000:
                            continue
                        tier_row = (tier_prompt, tier_cached, tier_write, tier_completion)
                        tiers.append((min_prompt_tokens, tier_row))
                    except (TypeError, ValueError):
                        log.warning("Skipping malformed pricing override for %s", model_id)
            if tiers:
                row = PricingSchedule(row, tuple(tiers))
            pricing_dict[model_id] = row
            normalized_model_id = normalize_model_identity(model_id)
            if normalized_model_id != model_id:
                pricing_dict[normalized_model_id] = row

        log.info(f"Fetched pricing for {len(pricing_dict)} models from OpenRouter")
        return pricing_dict

    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning(f"Failed to fetch OpenRouter pricing: {e}")
        return {}


def _endpoint_pricing_schedule(pricing: Any, source: dict):
    """Retain exact endpoint rates and prompt overrides; malformed prices stay unknown."""
    from decimal import Decimal, InvalidOperation
    from math import isfinite
    from ouroboros.pricing import PricingSchedule

    def rate(value):
        if value is None or isinstance(value, bool):
            return None
        try:
            number = Decimal(str(value))
            result = float(number * 1_000_000)
            return result if number >= 0 and isfinite(result) else None
        except (InvalidOperation, TypeError, ValueError, OverflowError):
            return None

    def row(values):
        return PricingSchedule(tuple(rate(values.get(key)) for key in (
            "prompt", "input_cache_read", "input_cache_write", "completion")),
            cache_write_1h=rate(values.get("input_cache_write_1h")), source=source)

    if not isinstance(pricing, dict):
        return row({})
    tiers = []
    try:
        for override in pricing.get("overrides") or []:
            minimum = int(override["min_prompt_tokens"])
            if minimum <= 0:
                return row({})
            # An override replaces the fields it supplies. Missing base cache
            # prices still remain unknown; none are inferred from token input.
            tiers.append((minimum, row({**pricing, **override})))
    except (TypeError, ValueError, KeyError, OverflowError):
        return row({})
    base = row(pricing)
    return PricingSchedule(base, tuple(tiers), cache_write_1h=base.cache_write_1h, source=source)


def fetch_openrouter_endpoint_pricing(model: str, *, timeout_sec: float = 5.0) -> Dict[str, Tuple[Optional[float], ...]]:
    """Read one model's endpoint tariffs, preserving the complete potential route pool.

    Tier tags and prices are provider facts from the documented endpoints API.
    Status is retained as evidence, never used to erase an expensive endpoint;
    a temporary outage cannot turn a tariff into an execution prohibition.
    """
    from urllib.parse import quote
    import requests

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
        rows = {}
        for endpoint in endpoints:
            tag = endpoint.get("tag") if isinstance(endpoint, dict) else None
            if not isinstance(tag, str) or not tag or tag in rows or endpoint.get("model_id") != model:
                return {}  # An unbound/omitted row cannot certify a pool upper estimate.
            suffix = tag.rsplit("/", 1)[-1] if "/" in tag else ""
            tier = "priority" if suffix in {"fast", "priority"} else "flex" if suffix == "flex" else "default"
            source = {"provider": "openrouter", "model": model, "endpoint_tag": tag,
                      "service_tier": tier, "url": url, "status": endpoint.get("status")}
            rows[tag] = _endpoint_pricing_schedule(endpoint.get("pricing"), source)
        return rows
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

    api_key = (runtime_setting("CLOUDRU_FOUNDATION_MODELS_API_KEY", "") or "").strip()
    if not api_key:
        return {}
    try:
        import requests
    except ImportError:
        return {}

    base_url = (
        runtime_setting("CLOUDRU_FOUNDATION_MODELS_BASE_URL", "") or ""
    ).strip() or "https://foundation-models.api.cloud.ru/v1"
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


class _GenerationCostMixin:
    """Late cost settlement for a route that reports it out of band."""

    def _fetch_generation_cost(
        self,
        generation_id: str,
        target: Optional[Dict[str, Any]] = None,
    ) -> Optional[float]:
        """Fetch cost from OpenRouter Generation API when usage lacks it."""
        active_target = target or self._resolve_remote_target("openrouter::")
        if not active_target.get("supports_generation_cost"):
            return None
        try:
            import requests
            base_url = str(active_target.get("base_url") or "").rstrip("/")
            api_key = str(active_target.get("api_key") or "")
            url = f"{base_url}/generation?id={generation_id}"
            resp = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=5, **requests_verify_kwargs())
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
            # Generation cost can lag the chat response; retry once.
            time.sleep(0.5)
            resp = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=5, **requests_verify_kwargs())
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
        except Exception:
            log.debug("Failed to fetch generation cost from OpenRouter", exc_info=True)
            pass
        return None
