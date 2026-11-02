"""Rolling-24h spend of consciousness: its wake-ups plus the tasks they started.

Owner decision В11/В18 (PLAN 5.5, 5.13 п.11, 5.14 п.1): the daily allowance
gates NEW starts (a wake, a promoted/scheduled root) on the money the whole
consciousness tree spent in the last 24 hours. Root discovery retains its
category horizon in the usage store: every attempt row whose transition
timestamp (``ts_last``) lies in the last 48 h and whose category is
``consciousness`` (a wake's own rows) or ``consciousness_task`` (a started
root's rows) names a consciousness root; the window is then every money row
(attempt / subscription_session / external_unmetered — never an imported
``usage_baseline_group`` aggregate or a ``legacy_*`` row) under one of those
roots whose transition timestamp lies in the last 24 h, open rows included,
reduced by the ONE money reducer ``_usage_rows._summary``. Both selections are
addressed ranges (the ``(category, ts_last_epoch)`` and ``(root_task_id,
ts_last_epoch)`` indexes), so the tick that admits a wake reads the same rows
with a day of usage history or a month, and a root still at work is read for its
last day, never its whole life. ``settled_usd`` — the KNOWN spend,
confirmed prices and disclosed estimates — is the number that decides (owner
Q4-A, #1487), the same rule as every other money limit: a reservation or the
upper bound of an unresolved call is shown beside it as ``accounted_usd``
exposure, never counted as spent, so one refused oversized request cannot
close the allowance for a day. ``unknown_unmetered`` makes the known spend "at
least". The TIME filter runs on every call, so an exhausted window frees
itself as rows age out without a new write. A store that cannot be read
yields the typed ``allowance_unknown`` outcome — the caller refuses the start
honestly. ``spend_by_cause`` partitions that same window into own thoughts,
places and started work, with known dollars and unmetered calls; it introduces
no extra allowance. Stamped root causes come from the existing money rows.
"""

from __future__ import annotations

import datetime as _dt
import time
from typing import Any, Dict, Optional

from ouroboros._usage_rows import _summary, row_ts_epoch
from ouroboros.runtime_limits import get_consciousness_daily_usd
from ouroboros.usage_ledger import _drive_root
from ouroboros.usage_origin import cause_label, root_cause

CONSCIOUSNESS_CATEGORIES = frozenset({"consciousness", "consciousness_task"})
MONEY_ROW_KINDS = frozenset({"attempt", "subscription_session", "external_unmetered"})
WINDOW_SEC = 24 * 3600
ROOT_HORIZON_SEC = 48 * 3600  # a root is discovered by a consciousness row this recent
STATUS_AVAILABLE, STATUS_EXHAUSTED, STATUS_UNKNOWN = "available", "exhausted", "allowance_unknown"
_MONEY_KIND = "COALESCE(NULLIF(kind, ''), 'attempt') IN ('attempt', 'subscription_session', 'external_unmetered')"


def _window_rows(txn: Any, now_ts: float) -> tuple:
    """(roots, their 24 h money rows captured per root for ``decode_attempts``, root causes), addressed selections."""
    discovered = list(txn.conn.execute(
        "SELECT DISTINCT root_task_id, category, json_extract(extra, '$.consciousness_cause') AS cause "
        "FROM attempts WHERE category IN ('consciousness', 'consciousness_task') "
        f"AND ts_last_epoch >= ? AND root_task_id != '' AND {_MONEY_KIND}", (now_ts - ROOT_HORIZON_SEC,)))
    defaults = {record[0]: record[2] or ("own" if record[1] == "consciousness" else "started")
                for record in discovered}
    causes = {root: root_cause(txn, root) or fallback for root, fallback in defaults.items()}
    roots = sorted(causes)
    rows = [txn.attempt_rows(f"root_task_id = ? AND ts_last_epoch >= ? AND {_MONEY_KIND}", (root, now_ts - WINDOW_SEC))
            for root in roots]  # captured here, decoded after the read closed (``allowance_window``)
    return roots, rows, causes


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
            roots, captured, root_causes = _window_rows(txn, now_ts)
        degraded = usage_store.integrity_degraded(root)
        rows = [row for part in captured for row in usage_store.decode_attempts(part)]
    except Exception as exc:  # noqa: BLE001 — every read failure is the one typed outcome
        return {"status": STATUS_UNKNOWN, "error": f"{type(exc).__name__}: {exc}",
                "limit_usd": limit, "settled_usd": None, "accounted_usd": None, "remaining_usd": None,
                "resets_at": "", "spend_by_cause": None}
    window = [(ts, row) for ts, row in ((row_ts_epoch(row), row) for row in rows) if ts is not None]
    summary = _summary([row for _ts, row in window])
    by_cause: dict = {}
    for _ts, row in window:
        cause = root_causes[row["root_task_id"]]
        by_cause.setdefault(cause, []).append(row)
    spend_by_cause = []
    for cause, members in sorted(by_cause.items()):
        part = _summary(members)
        spend_by_cause.append({"cause": cause, "label": cause_label(cause),
                              "usd": float(part["settled_usd"]),
                              "unmetered_calls": int(part["unknown_unmetered"]),
                              "accounted_usd": float(part["accounted_usd"]),
                              "non_final_rows": int(part["non_final_rows"])})
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
        "spend_by_cause": spend_by_cause,
        "window_rows": len(window),
        "resets_at": resets_at,
        "integrity_degraded": degraded,
    }
