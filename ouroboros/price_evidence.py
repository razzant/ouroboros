"""Direct-provider price evidence: prices that arrive as documents, not catalogs.

DeepSeek and GigaChat publish no machine-readable tariff endpoint this body can
poll, so their prices arrive as documents — DeepSeek's pricing page in US dollars
with a peak clock, and a dated Cloud.ru appendix in roubles. This module keeps
those two documents as ONE dated, sourced datasheet and turns them into the
per-million-token prices :mod:`ouroboros.pricing` already speaks.

It is deliberately not the hand-maintained tariff table DEVELOPMENT forbids
("they go stale silently and look authoritative"): every row names the document
and version it was read from, the datasheet carries the day it was observed, and a
row older than ``MAX_EVIDENCE_AGE_DAYS`` resolves to UNKNOWN. Unknown is never
zero — a route with no live evidence keeps pricing as ``None`` and the ledger keeps
``cost_final`` false, exactly as before this module existed.

Three facts only this module owns:

* **DeepSeek's peak clock.** Peak is 01:00-04:00 and 06:00-10:00 UTC, Monday to
  Friday, Chinese public holidays excluded. Every other hour — all of Saturday and
  Sunday included — is off-peak at exactly half the peak rate. The Chinese holiday
  calendar is a fact this module does not hold; with an empty calendar those days
  price at PEAK (an overstatement, disclosed in ``evidence()``) rather than
  inventing a discount.
* **The rouble-to-dollar conversion**, from the CBR daily rate. The rouble is the
  cross for CNY, so one table serves both currencies. A rate older than
  ``MAX_FX_AGE_DAYS`` is unknown, not approximate.
* **The VAT factor** of the Russian appendix. Its tariff column is VAT-exclusive and
  the money that actually leaves the account is the VAT-inclusive one, so every
  rouble amount is multiplied by 1 + VAT (22%). DeepSeek bills in dollars with no
  Russian VAT: its published price already is the full cash cost.

Scope is exactly two providers. Every other route keeps its existing behaviour
(OpenRouter/Cloud.ru catalogs, everything else unknown).
"""

from __future__ import annotations

import logging
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional, Tuple

from ouroboros.provider_models import normalize_model_identity
from ouroboros.utils import atomic_write_json, read_json_dict

log = logging.getLogger(__name__)

__all__ = [
    "PROVIDERS",
    "MAX_EVIDENCE_AGE_DAYS",
    "MAX_FX_AGE_DAYS",
    "DEEPSEEK_PEAK_WINDOWS_UTC",
    "UnitPrices",
    "catalog",
    "cbr_rates",
    "deepseek_tier",
    "prices",
    "to_usd",
]

PROVIDERS = frozenset({"deepseek", "gigachat"})

#: A price read from a document is evidence with a shelf life, not a fact of nature.
MAX_EVIDENCE_AGE_DAYS = 45
#: A daily published FX rate keeps meaning for a few days; past this it is a guess.
MAX_FX_AGE_DAYS = 7
#: How long a fetched CBR table is reused before another attempt.
FX_REFRESH_AFTER_SEC = 12 * 3600
#: The CBR's daily table (windows-1251 XML); the site's HTML page is the human view.
CBR_URL = "https://www.cbr.ru/scripts/XML_daily.asp"

#: Peak windows in UTC, half-open ``[start, end)`` — the provider's own definition.
DEEPSEEK_PEAK_WINDOWS_UTC: Tuple[Tuple[int, int], ...] = ((1, 4), (6, 10))

#: Chinese public holidays, as ``{year: {"MM-DD", ...}}``. Empty on purpose: a
#: wrong guess would invent a discount, so an absent calendar overstates instead.
_CN_HOLIDAYS: Dict[int, frozenset] = {}

#: One datasheet, two documents. Rows are keyed by their canonical route id
#: (lower-cased); ``_LEGACY_ROWS`` maps retired provider spellings onto them, so no
#: invented name can ever become a catalog key.
_DATASHEET: Dict[str, Any] = {
    "schema": 1,
    "observed_at": "2026-11-02",
    "providers": {
        "deepseek": {
            "currency": "USD",
            "vat_exclusive": False,
            "observed_at": "2026-11-02",
            "source": {
                "kind": "web_page",
                "ref": "https://api-docs.deepseek.com/quick_start/pricing/",
                "title": "DeepSeek API Docs — Models & Pricing",
                "observed_at": "2026-11-02",
            },
            "tier_rule": (
                "peak 01:00-04:00 and 06:00-10:00 UTC, Monday-Friday, Chinese public "
                "holidays excluded; every other hour (all weekend included) is off-peak "
                "at half the peak rate"
            ),
            "currency_note": "already the full cash cost; no Russian VAT applies",
            "rows": {
                "deepseek-flash": {
                    "id": "deepseek-flash",
                    "cache_hit": {"off_peak": 0.003, "peak": 0.006},
                    "cache_miss": {"off_peak": 0.15, "peak": 0.30},
                    "output": {"off_peak": 0.60, "peak": 1.20},
                    "provenance": "table row deepseek-flash, footnotes 1-2",
                },
                "deepseek-v4-pro": {
                    "id": "deepseek-v4-pro",
                    "cache_hit": {"off_peak": 0.022, "peak": 0.044},
                    "cache_miss": {"off_peak": 0.66, "peak": 1.32},
                    "output": {"off_peak": 1.98, "peak": 3.96},
                    "provenance": "table row deepseek-v4-pro, footnotes 1-2",
                },
            },
        },
        "gigachat": {
            "currency": "RUB",
            "vat_exclusive": True,
            "vat_rate": 0.22,
            "observed_at": "2026-11-02",
            "source": {
                "kind": "document",
                "ref": "foundation-models.pdf",
                "title": "Приложение №7.EVO.11.2 — Тарифы на Evolution Foundation Models",
                "doc_version": "261029",
                "observed_at": "2026-11-02",
            },
            "tier_rule": (
                "flat: the appendix publishes no night and no weekend row for any "
                "GigaChat model (its night rows cover seven non-GigaChat models only)"
            ),
            "currency_note": "VAT-exclusive column times the VAT factor; converted at the CBR rate",
            "rows": {
                "gigachat-3-ultra": {
                    "id": "GigaChat-3-Ultra",
                    "input": 78.87,
                    "output": 236.56,
                    "provenance": "rows 448-449 (БЯМ GigaChat: GigaChat Ultra)",
                },
                "gigachat-3-pro": {
                    "id": "GigaChat-3-Pro",
                    "input": 59.86,
                    "output": 144.58,
                    "provenance": "rows 450-451 (БЯМ GigaChat: GigaChat 3 Pro)",
                },
                "gigachat-2-max": {
                    "id": "GigaChat-2-Max",
                    "input": 466.67,
                    "output": 466.67,
                    "provenance": "rows 257-258 (БЯМ GigaChat-2-Max)",
                },
                "gigachat3-10b-a1.8b": {
                    "id": "GigaChat3-10B-A1.8B",
                    "input": 10.00,
                    "output": 10.00,
                    "provenance": "rows 259-260 (БЯМ GigaChat3-10B-A1.8B)",
                },
                # The appendix has no row for this model at all: the owner priced it
                # directly as one fifth of GigaChat Ultra (2026-11-02). `proxy: True`
                # travels with every number built from this row, so a reader never
                # mistakes an owner's decision for a published tariff.
                "gigachat-3-lightning": {
                    "id": "GigaChat-3-Lightning",
                    "input": 15.774,
                    "output": 47.312,
                    "proxy": True,
                    "provenance": "owner decision 2026-11-02: one fifth of GigaChat Ultra "
                                  "(rows 448-449); the appendix has no Lightning row",
                },
            },
        },
    },
}

#: Retired provider spellings, mapped onto a datasheet row key. Canonical ids resolve
#: by themselves, so this map holds aliases only.
_LEGACY_ROWS: Dict[str, Dict[str, str]] = {
    "deepseek": {
        # The provider's own footnote 1: the retired names are served by deepseek-flash
        # and billed at the Flash price.
        "deepseek-v4-flash": "deepseek-flash",
        "deepseek-v4-flash-vision-exp": "deepseek-flash",
    },
}

_fx_memo: Dict[str, Any] = {}
_fx_lock = threading.Lock()


@dataclass(frozen=True)
class UnitPrices:
    """One model's prices, in dollars per million tokens, with the evidence behind them."""

    input_per_million: Optional[float]
    cached_input_per_million: Optional[float]
    cache_write_per_million: Optional[float]
    output_per_million: Optional[float]
    provider: str
    model: str
    tier: str
    native_currency: str
    usd_per_native: float
    vat_factor: float
    as_of: str
    source: Dict[str, Any] = field(default_factory=dict)
    provenance: Tuple[str, ...] = ()

    def evidence(self) -> Dict[str, Any]:
        """Provenance projected for logs, the ledger row and the Costs panel."""
        return {
            "kind": "direct_price_evidence",
            "provider": self.provider,
            "model": self.model,
            "tier": self.tier,
            "as_of": self.as_of,
            "native_currency": self.native_currency,
            "usd_per_native": self.usd_per_native,
            "vat_factor": self.vat_factor,
            "source": dict(self.source),
            "provenance": list(self.provenance),
        }


def _native_name(provider: str, model: str) -> str:
    """The provider's own spelling of a route model, e.g. ``gigachat::X`` -> ``X``."""
    text = str(model or "").strip()
    normalized = normalize_model_identity(text)
    prefix = f"{provider}/"
    if normalized.lower().startswith(prefix):
        return normalized[len(prefix):]
    return text.split("::", 1)[-1] if "::" in text else text


def _row_key(provider: str, model: str) -> Optional[str]:
    """The datasheet row this route model bills against, or None when it has none."""
    rows = ((_DATASHEET.get("providers") or {}).get(provider) or {}).get("rows") or {}
    native = _native_name(provider, model).strip().lower().replace("_", "-")
    for row_key, row in rows.items():
        if native in (row_key, str(row.get("id") or "").strip().lower()):
            return row_key
    return (_LEGACY_ROWS.get(provider) or {}).get(native)


def _reference_day(at: Optional[datetime]) -> date:
    moment = at or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).date()


def _age_days(iso_day: str, at: Optional[datetime]) -> Optional[int]:
    try:
        observed = date.fromisoformat(str(iso_day))
    except (TypeError, ValueError):
        return None
    return (_reference_day(at) - observed).days


def deepseek_tier(at: Optional[datetime] = None) -> str:
    """Return ``"peak"`` or ``"off_peak"`` for one instant, by the provider's own rule.

    Weekend days are off-peak in full. A Chinese public holiday is off-peak too, but
    because this module holds no holiday calendar, such a day is priced at peak: the
    error can only overstate the bill, never invent a discount.
    """
    moment = at or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)
    if moment.weekday() >= 5:
        return "off_peak"
    if moment.strftime("%m-%d") in _CN_HOLIDAYS.get(moment.year, frozenset()):
        return "off_peak"
    if any(start <= moment.hour < end for start, end in DEEPSEEK_PEAK_WINDOWS_UTC):
        return "peak"
    return "off_peak"


def _fx_state_path():
    from pathlib import Path

    from ouroboros.config import DATA_DIR

    return Path(DATA_DIR) / "state" / "cbr_rates.json"


def _parse_cbr_xml(raw: bytes) -> Dict[str, Any]:
    """Parse the CBR daily table; an unreadable payload yields no rates at all."""
    text = raw.decode("cp1251", "replace")
    root = ET.fromstring(text)
    rates: Dict[str, float] = {}
    for node in root.findall("Valute"):
        code = (node.findtext("CharCode") or "").strip().upper()
        try:
            nominal = float((node.findtext("Nominal") or "1").replace(",", "."))
            value = float((node.findtext("Value") or "").replace(",", "."))
        except ValueError:
            continue
        if code and nominal:
            rates[code] = round(value / nominal, 6)
    return {"currency_date": str(root.get("Date") or ""), "rates": rates}


def _fetch_cbr(timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    try:
        import httpx

        from ouroboros.net_transport import extra_ca_bundle

        # ``verify`` is a CLIENT parameter in httpx, never a request one. Passing it to
        # ``Client.get`` raised TypeError, the broad except below swallowed it, and so
        # the daily table was NEVER written: every rouble-priced GigaChat row stayed
        # unknown while the site itself answered 200 (measured 2026-11-02 — the bug was
        # in the call, not in the network, the CA bundle or the operator).
        verify = extra_ca_bundle() or True
        with httpx.Client(timeout=httpx.Timeout(timeout), follow_redirects=True,
                          verify=verify) as client:
            response = client.get(CBR_URL)
            response.raise_for_status()
            return _parse_cbr_xml(response.content)
    except Exception as exc:  # transport, TLS, parse — all one outcome: no fresh rate
        log.debug("CBR rate fetch failed: %s", exc)
        return None


def cbr_rates(*, allow_live_fetch: bool = True, force: bool = False) -> Optional[Dict[str, Any]]:
    """The CBR daily table with roubles per unit, memoised, refreshed off the hot path.

    Order: process memo, the on-disk record, then (only when allowed) one bounded
    fetch. An expired record is still returned so callers can disclose its age, but
    ``to_usd`` refuses one older than ``MAX_FX_AGE_DAYS``.
    """
    with _fx_lock:
        memo = _fx_memo.get("table")
        if memo and not force:
            age = (datetime.now(timezone.utc).timestamp() - float(_fx_memo.get("at") or 0.0))
            if age < FX_REFRESH_AFTER_SEC:
                return memo
    path = _fx_state_path()
    stored = read_json_dict(path) or {}
    table: Optional[Dict[str, Any]] = None
    if stored.get("rates"):
        table = {
            "currency_date": str(stored.get("currency_date") or ""),
            "rates": dict(stored.get("rates") or {}),
            "fetched_at": str(stored.get("fetched_at") or ""),
            "source": str(stored.get("source") or CBR_URL),
        }
    stale = True
    if table is not None:
        try:
            fetched = datetime.fromisoformat(str(table.get("fetched_at") or ""))
            stale = (datetime.now(timezone.utc) - fetched).total_seconds() >= FX_REFRESH_AFTER_SEC
        except (TypeError, ValueError):
            stale = True
    if allow_live_fetch and (table is None or stale or force):
        fresh = _fetch_cbr()
        if fresh and fresh.get("rates"):
            table = {
                **fresh,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "source": CBR_URL,
            }
            try:
                atomic_write_json(path, {"schema": 1, **table})
            except Exception as exc:  # a read-only drive must not lose the rate
                log.debug("CBR rate record not persisted: %s", exc)
    if table is None:
        return None
    with _fx_lock:
        _fx_memo["table"] = table
        _fx_memo["at"] = datetime.now(timezone.utc).timestamp()
    return table


def _fx_factor(currency: str, table: Dict[str, Any]) -> Optional[float]:
    """Roubles per one unit of ``currency``, or None when the table cannot carry it."""
    code = str(currency or "USD").upper()
    rates = table.get("rates") or {}
    if code == "RUB":
        return 1.0
    rate = rates.get(code)
    try:
        value = float(rate)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def to_usd(amount: Optional[float], currency: str, table: Dict[str, Any],
           *, at: Optional[datetime] = None) -> Optional[float]:
    """Convert an amount in ``currency`` to dollars through the rouble.

    A USD amount passes through unchanged (its price already is the cash cost). Every
    other currency divides by its rouble rate, which makes the rouble the single cross
    for CNY as well. An expired or unusable table yields None — never a stale number.
    """
    if amount is None:
        return None
    code = str(currency or "USD").upper()
    if code == "USD":
        return round(float(amount), 6)
    if _fx_expired(table, at):
        return None
    rub_per_usd = _fx_factor("USD", table)
    currency_rub = _fx_factor(code, table)
    if not rub_per_usd or currency_rub is None:
        return None
    return round(float(amount) * currency_rub / rub_per_usd, 6)


def _fx_expired(table: Dict[str, Any], at: Optional[datetime]) -> bool:
    """True when the table's own currency date is older than the FX shelf life."""
    currency_day = str(table.get("currency_date") or "")
    age = _age_days(_iso_from_dmy(currency_day), at) if currency_day else None
    return age is None or age > MAX_FX_AGE_DAYS


def _iso_from_dmy(day: str) -> str:
    """The CBR writes ``31.10.2026``; the rest of the body speaks ISO."""
    parts = str(day or "").strip().split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return ""
    return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"


def _converted(value: Optional[float], *, vat_factor: float,
               usd_per_native: float) -> Optional[float]:
    """One native-currency price in dollars: the VAT factor applied, then converted."""
    if value is None:
        return None
    return round(float(value) * float(vat_factor) * float(usd_per_native), 6)


def prices(provider: str, model: str, *, at: Optional[datetime] = None,
           allow_live_fetch: bool = True) -> Optional[UnitPrices]:
    """This model's dollar prices per million tokens, or None when evidence is absent.

    None is the honest answer for: an unknown provider, a model the datasheet does not
    cover, a datasheet older than ``MAX_EVIDENCE_AGE_DAYS``, or a rouble price whose FX
    rate is missing or expired. Callers must keep that None — it becomes ``cost_usd =
    NULL`` and ``cost_final = false``, not a zero.
    """
    provider = str(provider or "").strip().lower()
    sheet = (_DATASHEET.get("providers") or {}).get(provider)
    if not sheet:
        return None
    key = _row_key(provider, model)
    if not key:
        return None
    row = (sheet.get("rows") or {}).get(key)
    if not row:
        return None
    age = _age_days(str(sheet.get("observed_at") or _DATASHEET.get("observed_at") or ""), at)
    if age is None or age > MAX_EVIDENCE_AGE_DAYS:
        return None
    tier = deepseek_tier(at) if provider == "deepseek" else "flat"
    native_currency = str(sheet.get("currency") or "USD").upper()
    vat_factor = (1.0 + float(sheet.get("vat_rate") or 0.0)) if sheet.get("vat_exclusive") else 1.0

    usd_per_native = 1.0
    if native_currency != "USD":
        table = cbr_rates(allow_live_fetch=allow_live_fetch)
        if table is None or _fx_expired(table, at):
            return None
        rub_per_usd = _fx_factor("USD", table)
        currency_rub = _fx_factor(native_currency, table)
        if not rub_per_usd or currency_rub is None:
            return None
        # The rouble is the single cross: rouble amounts divide by the dollar's own
        # rouble rate, yuan amounts multiply by the yuan's first.
        usd_per_native = round(float(currency_rub) / float(rub_per_usd), 8)

    provenance = [str(row.get("provenance") or "").strip()] if row.get("provenance") else []
    if sheet.get("tier_rule"):
        provenance.append(str(sheet["tier_rule"]))
    if row.get("proxy"):
        provenance.append("PROXY row: the source document has no price for this model")

    if provider == "deepseek":
        # Cache writes are the miss path itself, so they are free on top of it; a
        # reported write counter must not blank the estimate.
        base = (
            row["cache_miss"][tier],
            row["cache_hit"][tier],
            0.0,
            row["output"][tier],
        )
        usd = tuple(_converted(value, vat_factor=vat_factor, usd_per_native=usd_per_native)
                    for value in base)
    else:
        # No separate cache-read tariff is published for GigaChat models. Billing a
        # cache hit at the input rate keeps the number defined and overstates rather
        # than discounts a price nobody documented.
        usd = (
            _converted(row["input"], vat_factor=vat_factor, usd_per_native=usd_per_native),
            _converted(row["input"], vat_factor=vat_factor, usd_per_native=usd_per_native),
            None,
            _converted(row["output"], vat_factor=vat_factor, usd_per_native=usd_per_native),
        )
    if usd[0] is None or usd[3] is None:
        return None
    return UnitPrices(
        input_per_million=usd[0],
        cached_input_per_million=usd[1],
        cache_write_per_million=usd[2],
        output_per_million=usd[3],
        provider=provider,
        model=key,
        tier=tier,
        native_currency=native_currency,
        usd_per_native=usd_per_native,
        vat_factor=vat_factor,
        as_of=str(sheet.get("observed_at") or _DATASHEET.get("observed_at") or ""),
        source=dict(sheet.get("source") or {}),
        provenance=tuple(provenance),
    )


def catalog(provider: str, *, at: Optional[datetime] = None,
            allow_live_fetch: bool = True) -> Dict[str, UnitPrices]:
    """Every priced model of one provider, keyed by the identity routes normalize to.

    Keys are exactly what ``normalize_model_identity`` returns for our route spellings
    (``gigachat::X`` -> ``gigachat/X``), which is what ``pricing.estimate_cost_optional``
    looks up. Rows without live evidence are simply absent, never priced as zero.
    """
    provider = str(provider or "").strip().lower()
    sheet = (_DATASHEET.get("providers") or {}).get(provider)
    if not sheet:
        return {}
    out: Dict[str, UnitPrices] = {}
    for row_key, row in (sheet.get("rows") or {}).items():
        native_id = str(row.get("id") or row_key)
        unit = prices(provider, native_id, at=at, allow_live_fetch=allow_live_fetch)
        if unit is None:
            continue
        out[normalize_model_identity(f"{provider}::{native_id}")] = unit
    return out
