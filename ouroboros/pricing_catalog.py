"""Exact direct-route tariffs from the public LiteLLM catalog.

The cache retains the fetched bytes, timestamp and digest under the data root;
an attempt keeps just its exact entry as price evidence. Expired or malformed
data is unknown, including after a failed refresh. No model family inherits a
tariff and no provider borrows another provider's entry.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import pathlib
import threading
import time
from datetime import datetime
from typing import Mapping

from ouroboros.net_transport import requests_verify_kwargs
from ouroboros.utils import atomic_write_json, utc_now_iso

CATALOG_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
CACHE_REL = pathlib.Path("state/pricing/litellm.json")
DIRECT_PROVIDERS = {"openai": "openai", "anthropic": "anthropic", "google": "gemini"}
RATE_FIELDS = {
    "text_input_tokens": "input_cost_per_token",
    "text_output_tokens": "output_cost_per_token",
    "audio_input_tokens": "input_cost_per_audio_token",
    "audio_output_tokens": "output_cost_per_audio_token",
    "image_input_tokens": "input_cost_per_image_token",
    "cached_text_input_tokens": "cache_read_input_token_cost",
    "cached_audio_input_tokens": "cache_read_input_audio_token_cost",
    "cached_image_input_tokens": "cache_read_input_image_token_cost",
}
log = logging.getLogger(__name__)
_lock = threading.Lock()
_cache: dict[pathlib.Path, dict] = {}
_retry_after: dict[pathlib.Path, float] = {}
_fetching: set[pathlib.Path] = set()


class CatalogEstimate(float):
    """Keep the exact tariff beside a float-compatible estimate through settlement.

    Looking the quote up again after computing cost can bind a refreshed tariff
    to an amount computed with the previous one.
    """

    def __new__(cls, amount, quote=None):
        value = super().__new__(cls, amount)
        value.quote = copy.deepcopy(quote)
        return value


def catalog_id(model: str, provider: str) -> str | None:
    """Translate route syntax only; ids and provider identity remain exact."""
    if provider not in DIRECT_PROVIDERS:
        return None
    name = str(model or "").strip()
    if "::" in name:
        owner, name = name.split("::", 1)
        if owner != provider:
            return None
    if "/" in name:
        owner, name = name.split("/", 1)
        if owner not in {provider, DIRECT_PROVIDERS[provider]} or "/" in name:
            return None
    if not name:
        return None
    return f"gemini/{name}" if provider == "google" else name


def _cache_path(drive_root) -> pathlib.Path:
    if drive_root is None:
        from ouroboros.config import DATA_DIR
        drive_root = os.environ.get("OUROBOROS_DATA_DIR") or DATA_DIR
    return pathlib.Path(drive_root).resolve() / CACHE_REL


def _fresh(snapshot: dict) -> bool:
    from ouroboros.pricing import _pricing_ttl_sec
    try:
        fetched = datetime.fromisoformat(snapshot["fetched_at"]).timestamp()
        return 0 <= time.time() - fetched < _pricing_ttl_sec()
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _decode(snapshot: dict) -> dict:
    raw = snapshot["content"].encode("utf-8")
    if snapshot["url"] != CATALOG_URL or hashlib.sha256(raw).hexdigest() != snapshot["sha256"]:
        raise ValueError("catalog cache source or digest mismatch")
    entries = json.loads(raw)
    if not isinstance(entries, dict) or not entries:
        raise ValueError("catalog must contain model entries")
    return {**snapshot, "entries": entries}


def _resident(snapshot: dict) -> dict:
    """The decoded entries stay in memory; the raw text lives only in the cache file."""
    return {key: value for key, value in snapshot.items() if key != "content"}


def _fetch_catalog() -> dict:
    import requests
    response = requests.get(CATALOG_URL, timeout=5.0, **requests_verify_kwargs())
    response.raise_for_status()
    raw = response.content
    return _decode({"url": CATALOG_URL, "fetched_at": utc_now_iso(),
                    "sha256": hashlib.sha256(raw).hexdigest(), "content": raw.decode("utf-8")})


def _snapshot(*, drive_root=None, allow_live_fetch: bool = True) -> dict | None:
    path = _cache_path(drive_root)
    with _lock:
        snapshot = _cache.get(path)
        if snapshot is None and path not in _retry_after:
            try:
                snapshot = _resident(_decode(json.loads(path.read_text(encoding="utf-8"))))
                _cache[path] = snapshot
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                snapshot = None
        if snapshot is not None and _fresh(snapshot):
            return snapshot
        if not allow_live_fetch or path in _fetching or time.time() < _retry_after.get(path, 0):
            return None
        _fetching.add(path)
    try:
        snapshot = _fetch_catalog()
        atomic_write_json(path, {key: value for key, value in snapshot.items() if key != "entries"})
    except Exception:
        log.warning("LiteLLM pricing catalog unavailable", exc_info=True)
        snapshot = None
    with _lock:
        _fetching.discard(path)
        if snapshot is None:
            _cache.pop(path, None)
            _retry_after[path] = time.time() + 30.0
        else:
            snapshot = _cache[path] = _resident(snapshot)
            _retry_after.pop(path, None)
    return snapshot


def get_catalog_quote(model: str, *, provider: str, drive_root=None,
                      allow_live_fetch: bool = True) -> dict | None:
    entry_id = catalog_id(model, provider)
    if entry_id is None:
        return None
    snapshot = _snapshot(drive_root=drive_root, allow_live_fetch=allow_live_fetch)
    entry = snapshot["entries"].get(entry_id) if snapshot else None
    if not isinstance(entry, dict) or entry.get("litellm_provider") != DIRECT_PROVIDERS[provider]:
        return None
    return {"url": CATALOG_URL, "fetched_at": snapshot["fetched_at"], "sha256": snapshot["sha256"],
            "catalog_id": entry_id, "provider": provider, "entry": _priced(entry)}


def _priced(entry: Mapping) -> dict:
    """The entry's tariff fields: a ledger row keeps the price it used, the digest names the rest."""
    return copy.deepcopy({key: value for key, value in entry.items() if "cost" in key or key == "litellm_provider"})


def rate(quote: Mapping | None, field: str, *, input_tokens: int = 0) -> float | None:
    """Use a published prompt-length tier literally, never a price coefficient."""
    entry = (quote or {}).get("entry", {})
    chosen, threshold = field, 0
    for key in entry:
        suffix = key.removeprefix(field + "_above_") if key.startswith(field + "_above_") else ""
        if suffix.endswith("k_tokens") and suffix[:-8].isdigit():
            bound = int(suffix[:-8]) * 1000
            if input_tokens > bound >= threshold:
                chosen, threshold = key, bound
    value = entry.get(chosen)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    except (ValueError, OverflowError):
        return None


def text_schedule(quote: dict):
    from ouroboros.pricing import PricingSchedule
    entry = quote["entry"]
    fields = ("input_cost_per_token", "cache_read_input_token_cost",
              "cache_creation_input_token_cost", "output_cost_per_token")

    def schedule(input_tokens=0):
        values = [rate(quote, key, input_tokens=input_tokens) for key in fields]
        hour = rate(quote, "cache_creation_input_token_cost_above_1hr", input_tokens=input_tokens)
        return PricingSchedule(tuple(None if v is None else v * 1_000_000 for v in values),
                               cache_write_1h=None if hour is None else hour * 1_000_000,
                               source=quote)

    bounds = set()
    for key in entry:
        suffix = key.rsplit("_above_", 1)[-1]
        if suffix.endswith("k_tokens") and suffix[:-8].isdigit():
            bounds.add(int(suffix[:-8]) * 1000 + 1)
    base = schedule()
    return PricingSchedule(base, tuple((bound, schedule(bound)) for bound in sorted(bounds)),
                           cache_write_1h=base.cache_write_1h, source=quote)


def get_text_pricing(*, provider: str, model: str = "", allow_live_fetch: bool = True, drive_root=None) -> dict:
    """Expose direct text tariffs through pricing.py's tuple-compatible interface."""
    if provider not in DIRECT_PROVIDERS:
        return {}
    if model:
        quote = get_catalog_quote(model, provider=provider, allow_live_fetch=allow_live_fetch, drive_root=drive_root)
        quotes = [quote] if quote else []
    else:
        snapshot = _snapshot(allow_live_fetch=allow_live_fetch, drive_root=drive_root)
        quotes = []
        for name, entry in (snapshot or {}).get("entries", {}).items():
            if (isinstance(entry, dict) and entry.get("litellm_provider") == DIRECT_PROVIDERS[provider]
                    and catalog_id(name, provider) == name):
                quotes.append({key: snapshot[key] for key in ("url", "fetched_at", "sha256")} |
                              {"catalog_id": name, "provider": provider, "entry": _priced(entry)})
    rows = {}
    for quote in quotes:
        name = quote["catalog_id"].removeprefix("gemini/") if provider == "google" else quote["catalog_id"]
        rows[f"{provider}/{name}"] = text_schedule(quote)
    return rows


def search_response_cost(model: str, usage: dict, *, calls: int | None, context_size: str, drive_root=None):
    """Price a direct search receipt from one catalog quote, including actual server tool calls.

    Missing tool tariffs or call counts keep the complete amount unknown; the
    ordinary text fallback must not silently erase a paid search component.
    """
    from ouroboros.pricing import _cost_from_pricing

    quote = get_catalog_quote(model, provider="openai", drive_root=drive_root, allow_live_fetch=False)
    tool_rates = (quote or {}).get("entry", {}).get("search_context_cost_per_query") or {}
    per_call = tool_rates.get(f"search_context_size_{context_size}") if isinstance(tool_rates, dict) else None
    tool_known = (calls == 0 or isinstance(per_call, (int, float)) and not isinstance(per_call, bool)
                  and math.isfinite(per_call) and per_call >= 0)
    # A receipt without its token counts prices nothing: an absent count is unknown, never zero.
    counts = [usage.get(key, usage.get(alias)) for key, alias in (("input_tokens", "prompt_tokens"),
                                                                  ("output_tokens", "completion_tokens"))]
    counted = all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in counts)
    text = (_cost_from_pricing(text_schedule(quote), counts[0], counts[1], usage, exact_cache_tiers=True)
            if quote and counted else None)
    standard = (usage.get("processing") or {}).get("observed") in (None, "", "standard", "default")
    known = (text is not None and type(calls) is int and calls >= 0 and tool_known and standard)
    search = (calls * (per_call or 0)) if known else None
    usage.update(cost_basis="litellm_text_and_search", cost_coverage="complete" if known else "partial",
                 cost_evidence={"tariff_quote": quote, "web_search": {
                     "calls": calls, "context_size": context_size, "usd": search}})
    return text + search if known else None
