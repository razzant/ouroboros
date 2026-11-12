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
    """A model's summed tokens by class; uncached input only from calls that reported both cache counts."""
    return {"cache_write": max(0, int(row.get("cache_write_tokens") or 0)), "cache_read": max(0, int(row.get("cached_tokens") or 0)),
            "output": max(0, int(row.get("completion_tokens") or 0)), "uncached_input": max(0, int(row.get("uncached_tokens") or 0))}


def _call_tokens(row: Dict[str, Any]) -> Dict[str, int]:
    """One shared call's tokens by class (a shared call reported all four counts): the tokens its shared dollars are of."""
    prompt, read, write = (max(0, int(row.get(key) or 0)) for key in ("prompt_tokens", "cached_tokens", "cache_write_tokens"))
    return {"cache_write": write, "cache_read": read, "output": max(0, int(row.get("completion_tokens") or 0)),
            "uncached_input": max(0, prompt - read - write)}


def _epoch(stamp: Any) -> Optional[float]:
    try:  # a "Z" stamp (an imported row) reads on Python 3.10 too; a stamp without a zone is UTC
        moment = _dt.datetime.fromisoformat(str(stamp).replace("Z", "+00:00")) if stamp else None
    except ValueError:
        return None
    return None if moment is None else (moment if moment.tzinfo else moment.replace(tzinfo=_dt.timezone.utc)).timestamp()


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


def _row_lifetimes(row: Dict[str, Any], written: int) -> Optional[tuple]:
    """``(5-minute, 1-hour)`` tokens one call wrote: the provider's full split, else its named lifetime."""
    from ouroboros.pricing import write_lifetimes

    usage = {"prompt_cache_ttl": row.get("prompt_cache_ttl")}
    try:
        split = json.loads(row["write_split"]) if row.get("write_split") else None
    except (TypeError, ValueError):
        split = None
    if isinstance(split, dict):
        usage["cache_write_tokens_by_ttl"] = split
    return write_lifetimes(usage, written)


def _write_price_in_reads(tariff: Optional[tuple], ttl: str) -> Optional[float]:
    """How many read tokens one written token costs at this lifetime at the model's BASE price, both from the same
    catalogue: a prompt-length tier is a property of one call, not of a model. None for an unnamed lifetime."""
    if not tariff or len(tariff) != 4 or not tariff[1] or tariff[2] is None or ttl not in ("5m", "1h"):
        return None
    write = float(tariff[2])
    if ttl == "1h":
        extended = getattr(tariff, "cache_write_1h", None)
        write = float(extended) if extended is not None else write * (2.0 / 1.25)  # the estimator's own ratio
    return round(write / float(tariff[1]), 1)


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
            "calls": 0, "cost": 0.0, "priced_rows": 0, "open_rows": 0, "unpriced_rows": 0, "shared_cost": 0.0,
            "uncounted_calls": 0, "uncounted_input_tokens": 0, "unshared_calls": 0,
            "tokens": dict.fromkeys(_CLASSES, 0), "tokens_shared": dict.fromkeys(_CLASSES, 0),
            "by_class": dict.fromkeys(_CLASSES, 0.0), "_written": {}})

    for start in range(first, end, _CHUNK_ROWS):
        stop = min(end, start + _CHUNK_ROWS)
        # One short read of the chunk (its models and its calls agree); the tariffs are read after it, outside it.
        groups, calls = _read(data_dir, lambda txn: (txn.period_models(start, stop, since_iso),
                                                     txn.period_settled_calls(start, stop, since_iso)))
        for row in groups:
            model = entry(row["model"])
            model["calls"] += int(row["calls"] or 0)
            model["cost"] += float(row["usd"] or 0.0)
            model["open_rows"] += int(row["open_rows"] or 0)
            model["unpriced_rows"] += int(row["unpriced_rows"] or 0)
            model["priced_rows"] += int(row["priced_rows"] or 0)
            model["uncounted_calls"] += int(row["uncounted_calls"] or 0)
            model["uncounted_input_tokens"] += int(row["uncounted_input_tokens"] or 0)
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
                for name, count in _call_tokens(row).items():  # beside the shared dollars, the tokens they are of
                    model["tokens_shared"][name] += count
            elif row.get("usd") is not None:
                model["unshared_calls"] += 1
            written = int(row.get("cache_write_tokens") or 0)
            if written <= 0:
                continue
            lifetimes = _row_lifetimes(row, written) if route_tariff is not None else None
            for ttl, count in zip(("5m", "1h"), lifetimes or ()):  # by the lifetimes the call was written at
                if count:
                    model["_written"].setdefault(ttl, [0, route_tariff])[0] += count
            try:
                fact = json.loads(row["prefix"]) if row.get("prefix") else None
            except (TypeError, ValueError):
                fact = None
            name, detail = prefix_class(fact, row["prompt_cache_ttl"], _epoch(row.get("ts_dispatched")))
            cause = causes.setdefault((name, detail), {"class": name, "detail": detail, "calls": 0, "write_tokens": 0,
                                                       "write_cost": 0.0, "unshared_calls": 0, "shared_write_tokens": 0})
            cause["calls"] += 1
            cause["write_tokens"] += written
            if shares is None:
                cause["unshared_calls"] += 1
            else:
                cause["write_cost"] += float(row["usd"]) * shares["cache_write"]
                cause["shared_write_tokens"] += written
    rows: List[Dict[str, Any]] = []
    for name, model in models.items():
        if not (model["calls"] or model["open_rows"]):
            continue  # every reservation of the model was released before it was sent: no call started
        lifetimes = sorted(model.pop("_written").items(), key=lambda item: (-item[1][0], item[0]))
        ratios = [(ttl, _write_price_in_reads(held[1], ttl)) for ttl, held in lifetimes]
        lifetime, ratio = next(((ttl, value) for ttl, value in ratios if value), ("", None))
        model.update(write_price_in_reads=ratio, write_price_lifetime=lifetime)
        shared = model.pop("shared_cost")
        if not model["priced_rows"]:
            model["cost"] = None  # no call of the model has a recorded price: unknown, never $0
        rows.append({"model": name, **model, "cost": None if model["cost"] is None else round(model["cost"], 6),
                     # What no tariff or token count could share out stays one disclosed amount, never spread.
                     "unshared_cost": round(max(0.0, (model["cost"] or 0.0) - shared), 6),
                     "by_class": {key: round(value, 6) for key, value in model["by_class"].items()} if shared > 0 else None})
    rows.sort(key=lambda row: (-(row["cost"] or 0.0), -sum(row["tokens"].values()), row["model"]))  # unpriced by their tokens
    writes = sorted(causes.values(), key=lambda cause: (-cause["write_cost"], -cause["write_tokens"], cause["class"], cause["detail"]))
    for cause in writes:
        cause["write_cost"] = round(cause["write_cost"], 6)
    return {"available": True, "period": period, "since": since_iso,
            "until": _dt.datetime.fromtimestamp(until, _dt.timezone.utc).isoformat(),
            # The date of the recorded catalogue when it, not this process's own tariffs, priced the shares.
            "tariff_observed_at": min((seen for _rows, seen in catalogs.values() if seen), default=""),
            "models": rows, "cache_writes": writes}
