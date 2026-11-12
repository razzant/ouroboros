"""The Costs page's period view: what the model calls started in a period cost, by token class and by what
each cache write followed (ARCHITECTURE 06 "Prompt caching").

The physical-attempt ledger is the one cost authority. A period is read by position
(``usage_store.Txn.period_bounds``) in short range reads, so a display never holds a rollback-journal store's
writers behind it, and membership is the reservation time alone. Money by class is an ALLOCATION: each call's
RECORDED cost is shared out by what the route's tariff charges for each class of that call's own tokens, so
only price ratios enter and the parts of a model, with the cost that could not be shared, add up to its
recorded cost. The tariff is one a display may read without a fetch (``pricing.display_tariff``); a model it
does not price keeps its tokens and its whole cost unshared. Nothing is stored here, and no verdict,
threshold or signal is derived.
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

PERIODS = {"24h": 24 * 3600, "7d": 7 * 24 * 3600, "30d": 30 * 24 * 3600}
_CHUNK_ROWS = 20_000  # one short read per chunk of the ledger's tail
_BUSY_TRIES = 8  # a display read gives way to the ledger's writers after its short wait; a chunk asks again this often
_CLASSES = ("cache_write", "cache_read", "output", "uncached_input")


def _tokens_by_class(row: Dict[str, Any]) -> Dict[str, int]:
    prompt, read, write = (max(0, int(row.get(name) or 0)) for name in ("prompt_tokens", "cached_tokens", "cache_write_tokens"))
    return {"cache_write": write, "cache_read": read, "output": max(0, int(row.get("completion_tokens") or 0)),
            "uncached_input": max(0, prompt - read - write)}


def _shares(row: Dict[str, Any], tariff: Optional[tuple]) -> Optional[Dict[str, float]]:
    """Each class's share of one call's recorded cost; None when the tariff or the call's counts cannot say."""
    from ouroboros.pricing import cost_class_weights

    usage = dict(row)
    try:
        split = json.loads(row["write_split"]) if row.get("write_split") else None
    except (TypeError, ValueError):
        split = None
    if isinstance(split, dict):
        usage["cache_write_tokens_by_ttl"] = split
    weights = cost_class_weights(tariff, usage)
    total = sum(weights.values()) if weights else 0.0
    return {name: weight / total for name, weight in weights.items()} if total > 0 else None


def _write_price_in_reads(tariff: Optional[tuple], ttl: str) -> Optional[float]:
    """How many read tokens one written token costs on this tariff at this lifetime (40 on a 1-hour Opus write)."""
    from ouroboros.pricing import cost_class_weights

    unit = cost_class_weights(tariff, {"prompt_tokens": 2_000_000, "completion_tokens": 0, "cached_tokens": 1_000_000,
                                       "cache_write_tokens": 1_000_000, "prompt_cache_ttl": ttl})
    return round(unit["cache_write"] / unit["cache_read"], 1) if unit and unit["cache_read"] > 0 else None


def _read(data_dir: pathlib.Path, take: Any) -> Any:
    """One short display read of the ledger; a writer in the way is waited out a bounded number of times."""
    from ouroboros import usage_store
    from ouroboros.usage_ledger import UsageLockUnavailable

    for attempt in range(_BUSY_TRIES):
        try:
            with usage_store.read(data_dir, allow_stale=True) as txn:
                return take(txn)
        except UsageLockUnavailable:
            if attempt == _BUSY_TRIES - 1:
                raise


def cost_period(data_dir: pathlib.Path, period: str, *, now: Optional[float] = None) -> Dict[str, Any]:
    """The period's models (cost, calls, tokens and money by class) and its cache writes by what they followed."""
    from ouroboros.pricing import display_tariffs
    from ouroboros.provider_models import normalize_model_identity, provider_for_model
    from ouroboros.transcript_prefix import prefix_class

    until = time.time() if now is None else float(now)
    since = until - PERIODS[period]
    since_iso = _dt.datetime.fromtimestamp(since, _dt.timezone.utc).isoformat()
    first, end = _read(data_dir, lambda txn: txn.period_bounds(since))
    catalogs: Dict[str, Tuple[Dict[str, tuple], str]] = {}
    models: Dict[str, Dict[str, Any]] = {}
    causes: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def tariff(model: str, provider: str) -> Optional[tuple]:
        route = str(provider or provider_for_model(model) or "openrouter").strip().lower()
        if route not in catalogs:
            catalogs[route] = display_tariffs(route)  # read once per route of a period
        return catalogs[route][0].get(normalize_model_identity(model)) or None

    def entry(model: str) -> Dict[str, Any]:
        return models.setdefault(model or "unattributed", {
            "calls": 0, "cost": 0.0, "open_rows": 0, "unpriced_rows": 0, "shared_cost": 0.0,
            "tokens": dict.fromkeys(_CLASSES, 0), "by_class": dict.fromkeys(_CLASSES, 0.0), "_written": {}})

    for start in range(first, end, _CHUNK_ROWS):
        stop = min(end, start + _CHUNK_ROWS)
        # Two short reads; the ledger's writers go between them, and the tariffs are read outside both.
        groups = _read(data_dir, lambda txn: txn.period_models(start, stop, since_iso))
        priced = sorted({row["model"] for row in groups if tariff(row["model"], row["provider"])})
        calls = _read(data_dir, lambda txn: txn.period_priced_calls(start, stop, since_iso, priced))
        for row in groups:
            model = entry(row["model"])
            model["calls"] += int(row["calls"] or 0)
            model["cost"] += float(row["usd"] or 0.0)
            model["open_rows"] += int(row["open_rows"] or 0)
            model["unpriced_rows"] += int(row["unpriced_rows"] or 0)
            for name, count in _tokens_by_class(row).items():
                model["tokens"][name] += count
        for row in calls:
            route_tariff = tariff(row["model"], row["provider"])
            shares = _shares(row, route_tariff) if row.get("usd") is not None else None
            model = entry(row["model"])
            if shares is not None:
                model["shared_cost"] += float(row["usd"])
                for name, share in shares.items():
                    model["by_class"][name] += float(row["usd"]) * share
            written = int(row.get("cache_write_tokens") or 0)
            if written <= 0:
                continue
            if route_tariff is not None:  # written tokens by lifetime, for the price ratio of the lifetime most were written at
                held = model["_written"].setdefault(row["prompt_cache_ttl"], [0, route_tariff])
                held[0] += written
            try:
                fact = json.loads(row["prefix"]) if row.get("prefix") else None
            except (TypeError, ValueError):
                fact = None
            name, detail = prefix_class(fact, row["prompt_cache_ttl"])
            cause = causes.setdefault((name, detail), {"class": name, "detail": detail, "calls": 0, "write_tokens": 0,
                                                       "write_cost": 0.0, "unshared_calls": 0})
            cause["calls"] += 1
            cause["write_tokens"] += written
            if shares is None:
                cause["unshared_calls"] += 1
            else:
                cause["write_cost"] += float(row["usd"]) * shares["cache_write"]
    rows: List[Dict[str, Any]] = []
    for name, model in models.items():
        lifetimes = sorted(model.pop("_written").items(), key=lambda item: (-item[1][0], item[0]))
        ratios = [(ttl, _write_price_in_reads(held[1], ttl)) for ttl, held in lifetimes]
        lifetime, ratio = next(((ttl, value) for ttl, value in ratios if value), ("", None))
        model.update(write_price_in_reads=ratio, write_price_lifetime=lifetime)
        shared = model.pop("shared_cost")
        rows.append({"model": name, **model, "cost": round(model["cost"], 6),
                     # What no tariff or token count could share out stays one disclosed amount, never spread.
                     "unshared_cost": round(max(0.0, model["cost"] - shared), 6),
                     "by_class": {key: round(value, 6) for key, value in model["by_class"].items()} if shared > 0 else None})
    rows.sort(key=lambda row: (-row["cost"], -sum(row["tokens"].values()), row["model"]))  # unpriced models by their tokens
    writes = sorted(causes.values(), key=lambda cause: (-cause["write_cost"], -cause["write_tokens"], cause["class"], cause["detail"]))
    for cause in writes:
        cause["write_cost"] = round(cause["write_cost"], 6)
    return {"available": True, "period": period, "since": since_iso,
            "until": _dt.datetime.fromtimestamp(until, _dt.timezone.utc).isoformat(),
            # The date of the recorded catalogue when it, not this process's own tariffs, priced the shares.
            "tariff_observed_at": min((seen for _rows, seen in catalogs.values() if seen), default=""),
            "models": rows, "cache_writes": writes}
