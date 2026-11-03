"""The price of one review by one reviewer row, measured from the usage ledger.

A review is not one model call: a reading reviewer makes many calls in one wave, a packet
reviewer one, and every seat's verdict extraction adds a light call. The usage ledger records
each physical attempt with its wave (``review_wave_id``) and its row (``review_slot_id``), so
the price of one review by a row is the median, over that row's recent waves on its current
model, of what the seat cost in one wave — every attempt summed — with how many reviews it
covers and their range. No history is "cost unknown", never a one-call estimate. A session row
spends a seat and time, not money; callers say so before asking for a price.

The table is read once per process and kept for half an hour, so a context capture (the price rides in
``## Review seat activity``, the changing part of every start) does not scan the ledger at every start.
A reader waits on the ledger lock only for the display wait and otherwise reads the last table (or none).
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

REVIEW_USAGE_CATEGORY = "multi_model_review_review"
COST_UNKNOWN = "cost unknown"
# Bounds of the measurement, not of any decision: the newest waves one row's median reads, the
# grouped seat rows one read scans, and how long one process keeps a table it read.
RECENT_REVIEWS = 20
_SCAN_LIMIT = 5000
_TABLE_TTL_SEC = 1800.0


@dataclass(frozen=True)
class ReviewPrice:
    """What one review by a row cost: the median, over ``reviews`` measured waves, with their range."""

    usd: float
    reviews: int
    low: float
    high: float

    def as_cost(self) -> Dict[str, Any]:
        """The ``GET /api/review-pool`` cost of a row with history."""
        return {"usd_per_review": self.usd, "basis": "observed_reviews", "reviews": self.reviews,
                "low": self.low, "high": self.high}


_lock = threading.Lock()
_tables: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}


def _seat_rows(drive_root: Any) -> List[Dict[str, Any]]:
    """This process's last read of the grouped seat costs, refreshed after ``_TABLE_TTL_SEC``."""
    key, now = str(drive_root), time.monotonic()
    with _lock:
        held = _tables.get(key)
    if held is not None and now - held[0] < _TABLE_TTL_SEC:
        return held[1]
    rows = held[1] if held is not None else []
    try:
        from ouroboros import usage_store

        with usage_store.read(drive_root, allow_stale=True) as txn:
            rows = txn.review_seat_costs(REVIEW_USAGE_CATEGORY, limit=_SCAN_LIMIT)
    except Exception:
        log.debug("review prices unavailable from the usage ledger", exc_info=True)
    with _lock:
        _tables[key] = (now, rows)
    return rows


def review_price(drive_root: Any, row_id: str, model: str) -> Optional[ReviewPrice]:
    """One review by ``row_id`` on ``model``: its median cost, or ``None`` without history."""
    row_id, model = str(row_id or ""), str(model or "")
    if not row_id or not model or drive_root is None:
        return None
    costs = [float(seat["usd"]) for seat in _seat_rows(drive_root)
             if seat.get("row_id") == row_id and seat.get("priced") and seat.get("usd") is not None
             and model in str(seat.get("models") or "").split(",")][:RECENT_REVIEWS]
    if not costs:
        return None
    return ReviewPrice(usd=statistics.median(costs), reviews=len(costs), low=min(costs), high=max(costs))


def _usd(value: float) -> str:
    places = Decimal("0.0001") if value < 0.01 else Decimal("0.01")
    return f"${Decimal(value).quantize(places, rounding=ROUND_HALF_UP)}"


def review_price_text(price: Optional[ReviewPrice]) -> str:
    """A row's price in words. Settings → Agents prints the same words (``subagents_settings.js``
    ``reviewCostText``), the dollars rounded as JavaScript's ``toFixed`` does."""
    if price is None:
        return COST_UNKNOWN
    if not price.usd and not price.high:
        return "no API cost per review"
    if price.reviews < 2:
        return f"≈{_usd(price.usd)} per review (1 measured)"
    return f"≈{_usd(price.usd)} per review (median of {price.reviews} measured, {_usd(price.low)}–{_usd(price.high)})"
