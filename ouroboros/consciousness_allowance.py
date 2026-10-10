"""Rolling-24h spend of consciousness: its wake-ups plus the tasks they started.

Owner decision В11/В18 (PLAN 5.5, 5.13 п.11, 5.14 п.1): the daily allowance
gates NEW starts (a wake, a promoted/scheduled root) on the money the whole
consciousness tree spent in the last 24 hours. There is no root index —
the roots are read off the usage store: every attempt row whose transition
timestamp (``ts_last``) lies in the last 48 h and whose category is
``consciousness`` (a wake's own rows) or ``consciousness_task`` (a started
root's rows) names a consciousness root; the window is then every money row
(attempt / subscription_session / external_unmetered — never an imported
``usage_baseline_group`` aggregate or a ``legacy_*`` row) under one of those
roots whose transition timestamp lies in the last 24 h, open rows included,
reduced by the ONE money reducer ``_usage_rows._summary``. Both selections are
addressed (the ``(category, ts_last)`` and ``root_task_id`` indexes), so the
answer never depends on unrelated history. ``settled_usd`` — the KNOWN spend,
confirmed prices and disclosed estimates — is the number that decides (owner
Q4-A, #1487), the same rule as every other money limit: a reservation or the
upper bound of an unresolved call is shown beside it as ``accounted_usd``
exposure, never counted as spent, so one refused oversized request cannot
close the allowance for a day. ``unknown_unmetered`` makes the known spend "at
least". The TIME filter runs on every call, so an exhausted window frees
itself as rows age out without a new write. A store that cannot be read
yields the typed ``allowance_unknown`` outcome — the caller refuses the start
honestly.
"""

from __future__ import annotations

import datetime as _dt
import time
from typing import Any, Dict, Optional

from ouroboros._usage_rows import _summary, row_ts_epoch
from ouroboros.runtime_limits import get_consciousness_daily_usd
from ouroboros.usage_ledger import _drive_root

CONSCIOUSNESS_CATEGORIES = frozenset({"consciousness", "consciousness_task"})
MONEY_ROW_KINDS = frozenset({"attempt", "subscription_session", "external_unmetered"})
WINDOW_SEC = 24 * 3600
ROOT_HORIZON_SEC = 48 * 3600  # a root is discovered by a consciousness row this recent
STATUS_AVAILABLE, STATUS_EXHAUSTED, STATUS_UNKNOWN = "available", "exhausted", "allowance_unknown"
_MONEY_KIND = "COALESCE(NULLIF(kind, ''), 'attempt') IN ('attempt', 'subscription_session', 'external_unmetered')"


def _window_rows(txn: Any, now_ts: float) -> tuple:
    """(roots, money rows of those roots in the 24 h window), both addressed."""
    roots = sorted(record[0] for record in txn.conn.execute(
        "SELECT DISTINCT root_task_id FROM attempts WHERE category IN ('consciousness', 'consciousness_task') "
        f"AND ts_last_epoch >= ? AND root_task_id != '' AND {_MONEY_KIND}", (now_ts - ROOT_HORIZON_SEC,)))
    rows: list = []
    for root in roots:
        rows.extend(txn.attempts(f"root_task_id = ? AND ts_last_epoch >= ? AND {_MONEY_KIND}",
                                 (root, now_ts - WINDOW_SEC)))
    return roots, rows


def allowance_window(
    drive_root: Any = None, *, now: Optional[float] = None, allow_stale: bool = False,
) -> Dict[str, Any]:
    """The allowance verdict for the 24 h ending at ``now`` (epoch seconds; default: the clock).

    ``allow_stale`` is the STATUS view's read (it admits nothing): after the short display
    wait it reports ``allowance_unknown``. Every admission of a wake reads exactly."""
    from ouroboros import usage_store

    now_ts = time.time() if now is None else float(now)
    limit = float(get_consciousness_daily_usd())
    try:
        root = _drive_root(drive_root)
        with usage_store.read(root, allow_stale=allow_stale) as txn:
            roots, rows = _window_rows(txn, now_ts)
            degraded = usage_store.integrity_degraded(root)
    except Exception as exc:  # noqa: BLE001 — every read failure is the one typed outcome
        return {"status": STATUS_UNKNOWN, "error": f"{type(exc).__name__}: {exc}",
                "limit_usd": limit, "settled_usd": None, "accounted_usd": None, "remaining_usd": None,
                "resets_at": ""}
    window = [(ts, row) for ts, row in ((row_ts_epoch(row), row) for row in rows) if ts is not None]
    summary = _summary([row for _ts, row in window])
    known = float(summary["settled_usd"])
    oldest = min((ts for ts, _row in window), default=None)
    resets_at = (
        _dt.datetime.fromtimestamp(oldest + WINDOW_SEC, tz=_dt.timezone.utc).isoformat()
        if oldest is not None else ""
    )
    return {
        "status": STATUS_EXHAUSTED if known >= limit else STATUS_AVAILABLE,
        "limit_usd": limit,
        "settled_usd": known,
        "accounted_usd": float(summary["accounted_usd"]),
        "remaining_usd": round(max(0.0, limit - known), 6),
        "unknown_unmetered": int(summary["unknown_unmetered"]),
        "non_final_rows": int(summary["non_final_rows"]),
        "roots": sorted(roots),
        "window_rows": len(window),
        "resets_at": resets_at,
        "integrity_degraded": degraded,
    }
