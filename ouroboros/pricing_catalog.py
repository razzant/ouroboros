"""Exact direct-route tariffs from the public LiteLLM catalog.

The cache retains the fetched bytes, timestamp and digest under the data root;
an attempt keeps just its exact entry as price evidence. Expired or malformed
data is unknown, including after a failed refresh. Every process reads the record
currently published there, decoding it again only when its file identity changes
(another worker's refresh). No model family inherits a tariff and no provider
borrows another provider's entry.

DeepSeek's entries are conditional and explicitly scoped (``DEEPSEEK_ENTRIES``): ``pin_tariff``
pins one physical candidate's exact entry, its input kind and its key's billing currency
(``/user/balance``) before the send; settlement prices that pin alone. Balances are not kept;
an inference from multiple wallets is fresh for each pin. A text call is priced only wholly
inside the published off-peak windows, a search per documented billable answer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import pathlib
import re
import threading
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, NamedTuple

from ouroboros.net_transport import requests_verify_kwargs
from ouroboros.utils import atomic_write_json, utc_now_iso

CATALOG_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
CACHE_REL = pathlib.Path("state/pricing/litellm.json")
DIRECT_PROVIDERS = {"gemini_live": "gemini", "openai_realtime": "openai", "grok_realtime": "xai", "deepseek": "deepseek"}
RATE_FIELDS = {
    "text_input_tokens": "input_cost_per_token",
    "text_output_tokens": "output_cost_per_token",
    "audio_input_tokens": "input_cost_per_audio_token",
    "audio_output_tokens": "output_cost_per_audio_token",
    "image_input_tokens": "input_cost_per_image_token",
    "video_input_tokens": "input_cost_per_image_token",  # Gemini Live prices image and video input at one rate
    "cached_text_input_tokens": "cache_read_input_token_cost",
    "cached_audio_input_tokens": "cache_read_input_audio_token_cost",
    "cached_image_input_tokens": "cache_read_input_image_token_cost",
    "cached_video_input_tokens": "cache_read_input_image_token_cost",
}
log = logging.getLogger(__name__)
_lock = threading.Lock()
_cache: dict[pathlib.Path, dict] = {}  # the decoded record by path, tagged with its file's ``published`` identity
_retry_after: dict[pathlib.Path, float] = {}
_fetching: set[pathlib.Path] = set()
# A DeepSeek key's billing currency, by a digest of the key that never leaves RAM:
# (currency or None, observed_at, monotonic expiry, reason).
_currency: dict[str, tuple[str | None, str, float, str]] = {}
_currency_fetching: set[str] = set()
_TOKEN_RATES = ("input_cost_per_token", "cache_read_input_token_cost", "output_cost_per_token")
# The DeepSeek entries in this delivery's scope, by exact id; their figures stay the catalog's.
DEEPSEEK_ENTRIES = frozenset({"deepseek/deepseek-flash", "deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-pro",
                              "deepseek/search"})
_SPAN = re.compile(r"([01]\d|2[0-3]):([0-5]\d)-([01]\d|2[0-3]):([0-5]\d)")


def catalog_id(model: str, provider: str) -> str | None:
    """Translate route syntax only; ids and provider identity remain exact."""
    if provider not in DIRECT_PROVIDERS:
        return None
    name = str(model or "").strip()
    if provider == "deepseek":  # built-in or named connection, usage form or bare id: never a guessed alias
        from ouroboros.provider_models import parse_model_route

        route = parse_model_route(name) if "::" in name else None
        if route is not None:
            name = route.model.strip() if route.valid and route.kind == "deepseek" else ""
        elif name.startswith("deepseek/"):
            name = name[len("deepseek/"):]
        return f"deepseek/{name}" if name and "/" not in name and not name.endswith(" (local)") else None
    if "::" in name:
        owner, name = name.split("::", 1)
        if owner not in {provider, DIRECT_PROVIDERS[provider], {"gemini_live": "google", "grok_realtime": "grok"}.get(provider, "")}:
            return None
    if "/" in name:
        owner, name = name.split("/", 1)
        if owner not in {provider, DIRECT_PROVIDERS[provider]} or "/" in name:
            return None
    if not name:
        return None
    return f"{DIRECT_PROVIDERS[provider]}/{name}" if provider in {"gemini_live", "grok_realtime"} else name


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


def _published(path: pathlib.Path) -> tuple | None:
    """The record file's identity; an atomic replacement (any process's refresh) changes it. None when absent."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_ino, stat.st_mtime_ns, stat.st_size


def _snapshot(*, drive_root=None, allow_live_fetch: bool = True) -> dict | None:
    """The fresh catalog (``pricing._pricing_ttl_sec``): the record now published, decoded once per file identity (an
    unreadable one is remembered so), else one live fetch when allowed."""
    path = _cache_path(drive_root)
    with _lock:
        published, held = _published(path), _cache.get(path)
        if held is None or held.get("published") != published:
            held = {"published": published}
            if published is not None:
                try:
                    held.update(_resident(_decode(json.loads(path.read_text(encoding="utf-8")))))
                except (OSError, ValueError, TypeError, KeyError, AttributeError):
                    pass
            _cache[path] = held
        snapshot = held if "entries" in held else None
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
            _retry_after[path] = time.time() + 30.0
        else:
            # This call answers from its own fetch; the next read decodes whatever is published by then (another
            # worker may have replaced the file since this write), never these bytes under a later file's identity.
            snapshot = _resident(snapshot)
            _cache.pop(path, None)
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
            "catalog_id": entry_id, "provider": provider, "entry": copy.deepcopy(entry)}


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
    return _number(entry.get(chosen))


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    except (ValueError, OverflowError):
        return None


# ---- DeepSeek: conditional entries, pinned per physical attempt ----------------------------------------------

def _applicable(quote: Mapping | None, mode: str, on: date | None) -> Mapping | None:
    """The quote's entry when it is a supported ``mode`` entry not retired on ``on`` (UTC; a new quote: today,
    a pinned one: the day it was sent); an unreadable retirement date is unknown."""
    entry = (quote or {}).get("entry")
    if (quote or {}).get("catalog_id") not in DEEPSEEK_ENTRIES or not isinstance(entry, Mapping) \
            or entry.get("mode") != mode:
        return None
    stated = entry.get("deprecation_date")
    try:
        retired = stated is not None and datetime.strptime(str(stated), "%Y-%m-%d").date() <= (
            on or datetime.now(timezone.utc).date())
    except ValueError:
        return None
    return None if retired else entry


def _windows(value: Any) -> tuple | None:
    """Published off-peak windows as ``(ISO weekday, start minute, end minute)``; ``24:00`` is written ``00:00``.
    Any other shape (a timezone, a holiday list, an inverted span) is not read: None."""
    spans = []
    for window in value if isinstance(value, list) and value else [None]:
        if not isinstance(window, Mapping) or set(window) != {"hours_utc", "weekdays"}:
            return None
        hours = [window["hours_utc"]] if isinstance(window["hours_utc"], str) else window["hours_utc"]
        days = window["weekdays"]
        if not (isinstance(hours, list) and hours and isinstance(days, list) and days
                and all(type(day) is int and 1 <= day <= 7 for day in days)):
            return None
        for text in hours:
            match = _SPAN.fullmatch(text) if isinstance(text, str) else None
            if match is None:
                return None
            start = int(match[1]) * 60 + int(match[2])
            end = int(match[3]) * 60 + int(match[4]) or 1440
            if start >= end:
                return None
            spans.extend((day, start, end) for day in days)
    return tuple(spans)


def off_peak_terms(quote: Mapping | None, *, on: date | None = None) -> dict | None:
    """A DeepSeek token entry's off-peak rates and windows, or None when this body cannot price it: not an applicable
    chat entry, a rate missing or malformed, the two cache-hit rates disagreeing, prompt-length tiers, or
    ``off_peak_pricing`` other than exactly the three token rates and its windows."""
    entry = _applicable(quote, "chat", on)
    if entry is None or any("_above_" in str(key) for key in entry):
        return None
    peak = {field: _number(entry.get(field)) for field in _TOKEN_RATES}
    hit = entry.get("input_cost_per_token_cache_hit")
    off = entry.get("off_peak_pricing")
    if None in peak.values() or (hit is not None and _number(hit) != peak["cache_read_input_token_cost"]) \
            or not isinstance(off, Mapping) or set(off) != {*_TOKEN_RATES, "windows"}:
        return None
    rates, windows = {field: _number(off.get(field)) for field in _TOKEN_RATES}, _windows(off.get("windows"))
    return None if None in rates.values() or windows is None else {"rates": rates, "windows": windows}


def per_query_terms(quote: Mapping | None, *, on: date | None = None) -> float | None:
    """An applicable DeepSeek search entry's price of one billable request; None for any other entry."""
    entry = _applicable(quote, "search", on)
    if entry is None or any("cost" in str(key) for key in entry if key != "input_cost_per_query"):
        return None
    return _number(entry.get("input_cost_per_query"))


def wholly_off_peak(windows: tuple, start: datetime, end: datetime) -> bool:
    """Whether every instant of ``[start, end]`` (aware, UTC weekdays) lies inside a published window."""
    if start.tzinfo is None or end.tzinfo is None or end < start:
        return False
    at, end = start.astimezone(timezone.utc), end.astimezone(timezone.utc)
    for _ in range(64):
        midnight = at.replace(hour=0, minute=0, second=0, microsecond=0)
        minute = (at - midnight).total_seconds() / 60
        ends = [stop for day, begin, stop in windows if day == at.isoweekday() and begin <= minute < stop]
        if not ends:
            return False
        at = midnight + timedelta(minutes=max(ends))
        if end < at:
            return True
    return False


class _CurrencyReading(NamedTuple):
    currency: str | None
    reason: str = ""
    inferred: bool = False


def _wallet_funded(info: Mapping[str, Any]) -> bool | None:
    """True for a valid positive total, False for three exact zeroes, None for unknown funding."""
    zeros = []
    for field in ("total_balance", "granted_balance", "topped_up_balance"):
        value = info.get(field)
        if not isinstance(value, str) or not value.isascii() or "_" in value:
            return None
        try:
            amount = Decimal(value)
        except InvalidOperation:
            return None
        zero = amount.is_zero()
        if not amount.is_finite() or (amount.is_signed() and not zero):
            return None
        zeros.append(zero)
    if not zeros[0]:
        return True
    return False if all(zeros) else None  # a positive component with zero total does not prove an empty wallet


def _billing_currency(infos: list[dict[str, Any]], available: object) -> _CurrencyReading:
    """Use a single named currency as-is. Across multiple currencies, infer only from one funded wallet
    alongside provably empty alternatives; unreadable amounts are not evidence of funding."""
    currencies = {info["currency"] for info in infos}
    if len(currencies) == 1:
        return _CurrencyReading(currencies.pop())
    funded = None
    for info in infos:
        has_funds = _wallet_funded(info)
        if has_funds is None or (has_funds and funded is not None):
            return _CurrencyReading(None, "currency_not_single")
        if has_funds:
            funded = info["currency"]
    if funded is None:
        return _CurrencyReading(None, "currency_unfunded")
    if available is not True:  # A positive balance alone need not be sufficient for any API call.
        return _CurrencyReading(None, "balance_unavailable" if available is False else "balance_unreadable")
    return _CurrencyReading(funded, inferred=True)


def _fetch_currency(api_key: str) -> _CurrencyReading:
    """One documented ``GET /user/balance`` with the selected key: evidence for a currency estimate, not a debit.
    A redirect is an answer (unknown), never followed with the key."""
    import requests

    from ouroboros.provider_models import DEEPSEEK_BASE_URL

    origin = DEEPSEEK_BASE_URL.rsplit("/v1", 1)[0]
    try:
        response = requests.get(f"{origin}/user/balance", timeout=5.0, allow_redirects=False,
                                headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
                                **requests_verify_kwargs())
        if response.status_code != 200:
            return _CurrencyReading(None, f"http_{response.status_code}")
        body = response.json()
    except Exception as exc:  # transport, TLS, JSON: no currency; neither key nor body reaches a log
        return _CurrencyReading(None, type(exc).__name__)
    infos = body.get("balance_infos") if isinstance(body, dict) else None
    if not isinstance(infos, list) or not 1 <= len(infos) <= 8 or not all(
            isinstance(info, dict) and info.get("currency") in ("USD", "CNY") for info in infos):
        return _CurrencyReading(None, "balance_unreadable")
    return _billing_currency(infos, body.get("is_available"))


def account_currency(api_key: str, *, allow_live_fetch: bool = True) -> dict | None:
    """The key's billing currency fact: ``{currency, source, observed_at[, reason or basis]}``, or None with
    no usable cached or live read. A failure pauses for 30 seconds; a single named currency uses the pricing TTL.
    A currency inferred from wallet balances is only for this pin: balances can change before the next attempt.
    Balances, the key and its digest never reach disk. A failed read leaves the currency unknown;
    time spent reading counts against the caller's deadline."""
    from ouroboros.pricing import _pricing_ttl_sec

    if not api_key:
        return None
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    with _lock:
        held = _currency.get(digest)
        if held is None or time.monotonic() >= held[2]:
            if not allow_live_fetch or digest in _currency_fetching:
                return None
            _currency_fetching.add(digest)
            _currency.pop(digest, None)  # retire expired evidence, also when the next result won't be cached
            held = None
    if held is None:
        try:
            try:
                currency, reason, inferred = _fetch_currency(api_key)
            except Exception as exc:  # a reader fault is an unread currency, paused like any failure
                currency, reason, inferred = None, type(exc).__name__, False
            observed_at = utc_now_iso()
            if not inferred:
                ttl = _pricing_ttl_sec() if currency else 30.0
                held = (currency, observed_at, time.monotonic() + ttl, reason)
        finally:
            with _lock:  # publish reusable results before releasing the in-flight mark
                if held is not None:
                    _currency[digest] = held
                _currency_fetching.discard(digest)
    else:
        currency, observed_at, _, reason = held
        inferred = False
    return {"currency": currency, "source": "deepseek_user_balance", "observed_at": observed_at,
            **({"reason": reason} if reason else {}),
            **({"basis": "single_funded_wallet"} if inferred else {})}


def pin_tariff(target: Mapping[str, Any], *, search: bool = False, text_only: bool = False,
               drive_root=None) -> dict | None:
    """One physical DeepSeek candidate's price evidence, pinned after its key was chosen and before reservation:
    the exact catalog quote this body can apply, the operation, for a call whether its input is text alone (never
    the content), and that key's currency fact. None without I/O for any other route; no currency read when no
    entry applies or the input is not text. Settlement reads this pin, never a later catalog."""
    from ouroboros.provider_models import DEEPSEEK_BASE_URL

    if target.get("provider") != "deepseek" or str(target.get("base_url") or "").rstrip("/") != DEEPSEEK_BASE_URL:
        return None
    try:
        quote = get_catalog_quote("search" if search else str(target.get("resolved_model") or ""),
                                  provider="deepseek", drive_root=drive_root)
        if (per_query_terms(quote) if search else off_peak_terms(quote)) is None:
            return None
        kind = {"operation": "search"} if search else {"operation": "chat", "input": "text" if text_only else "media"}
        priced = search or text_only
        return {"tariff_quote": quote, **kind,
                "account_currency": account_currency(str(target.get("api_key") or "")) if priced else None}
    except Exception:
        log.warning("DeepSeek tariff evidence unavailable", exc_info=True)
        return None


def price_condition(model: str, *, search: bool = False) -> str:
    """What a DeepSeek route's price depends on, for a display: ``off_peak_usd`` (``model``'s token entry), ``usd``
    (``search``), '' (no applicable entry in the install's unexpired catalog record, which it never fetches). Static
    applicability of the published entry for text input (``off_peak_usd`` prices no image), never an account read."""
    quote = get_catalog_quote("search" if search else model, provider="deepseek", allow_live_fetch=False)
    if search:
        return "usd" if per_query_terms(quote) is not None else ""
    return "off_peak_usd" if off_peak_terms(quote) is not None else ""
