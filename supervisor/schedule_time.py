"""Schedule time parsing helpers (cron + timezone) for the task queue."""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)


def timezone_for_schedule(record: Dict[str, Any]) -> datetime.tzinfo:
    raw = str(record.get("timezone") or "").strip()
    if raw:
        try:
            return ZoneInfo(raw)
        except Exception:
            log.warning("Invalid schedule timezone %r; falling back to local time", raw)
    # Blank timezone -> DST-aware system local zone (platform-layer SSOT).
    from ouroboros.platform_layer import local_zoneinfo

    return local_zoneinfo()


def parse_schedule_time(value: Any, tz: datetime.tzinfo) -> Optional[datetime.datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tz)
    return parsed.astimezone(tz)


def next_cron_time(expr: str, base: datetime.datetime) -> datetime.datetime:
    from croniter import croniter

    return croniter(str(expr or ""), base).get_next(datetime.datetime)


def once_due(trigger: Dict[str, Any], tz: datetime.tzinfo, now: datetime.datetime) -> tuple[bool, str]:
    """``(due, error)`` for a one-shot ``{"type": "once", "run_at": <ISO>}`` trigger.

    Fires at/after ``run_at`` (a past instant is due immediately); an unparseable or
    missing ``run_at`` is a typed record error, never a silent skip. Pure — the
    caller supplies the clock, so the selection logic is unit-testable."""
    run_at = parse_schedule_time((trigger or {}).get("run_at"), tz)
    if run_at is None:
        return False, "invalid or missing run_at for one-shot schedule"
    return now >= run_at, ""


def record_last_error(record: Dict[str, Any], message: str) -> bool:
    """Set ``record["last_error"]`` and report whether the value actually CHANGED.

    Write-churn guard: a permanently invalid record produces the identical error
    text on every scheduler tick — that is not a new fact and must not rewrite the
    whole table each tick."""
    text = str(message)
    if str(record.get("last_error") or "") == text:
        return False
    record["last_error"] = text
    return True


def prune_consumed_once_records(tasks: list, cutoff_epoch: float, *, work_settled=None) -> tuple[list, int]:
    """``(kept, pruned_count)`` — drop CONSUMED one-shot records (``trigger.type ==
    "once"`` + ``enabled=False`` + ``completed_at``) whose completion is older than
    the unified GC retention cutoff (epoch seconds; ``retention.age_cutoff``). The
    caller's ``work_settled`` predicate keeps a consumed continuation as its
    task's binding/Restore row until that task settles (consumption alone is not
    settlement); an independent one-shot is a durable receipt that ages out. ONLY one-shots are pruned: a
    disabled CRON row is a standing schedule the owner may re-enable, and is kept
    even if it carries a stray ``completed_at``. ENABLED records are never pruned,
    nor one whose occurrence has not settled (#1315); an unparseable ``completed_at``
    is kept, conservatively."""
    kept, pruned = [], 0
    for record in tasks:
        if (isinstance(record, dict) and not record.get("enabled", True)
                and not record.get("delete_requested_at")
                and record.get("completed_at") and not isinstance(record.get("occurrence"), dict)):
            trigger = record.get("trigger") if isinstance(record.get("trigger"), dict) else {}
            if str(trigger.get("type") or "") == "once":
                done = parse_schedule_time(record.get("completed_at"), datetime.timezone.utc)
                if done is not None and done.timestamp() < float(cutoff_epoch):
                    if work_settled is not None and not work_settled(record):
                        kept.append(record)
                        continue
                    pruned += 1
                    continue
        kept.append(record)
    return kept, pruned


def schedule_next_run(record: Dict[str, Any], *, base: Optional[datetime.datetime] = None) -> str:
    trigger = record.get("trigger") if isinstance(record.get("trigger"), dict) else {}
    if str(trigger.get("type") or "cron") != "cron":
        return ""
    expr = str(trigger.get("expr") or record.get("cron") or "").strip()
    if not expr:
        return ""
    tz = timezone_for_schedule(record)
    base_dt = base.astimezone(tz) if base is not None else datetime.datetime.now(tz)
    return next_cron_time(expr, base_dt).isoformat()
