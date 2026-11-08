"""Schedule time facts for the task queue: cron and timezone parsing, due instants, and the
automatic pass's wake (``pass_due`` / ``note_pass``: when a pass has something new to decide)."""

from __future__ import annotations

import datetime
import logging
import os
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


# ---------------------------------------------------------------------------
# The automatic pass's wake (cost law L9): when ``queue_schedules.check_scheduled_tasks`` has
# something new to decide. The supervisor tick used to start it on every tick (31 passes a
# minute on 01.11), each taking Q and the table's file lock and copying the table, though
# nothing it decides on had changed. A completed pass records, per data root, what it decided
# on (``note_pass``) and the tick asks ``pass_due`` before starting the next one:
#
# * a process with no completed pass passes first;
# * a write of the table by any writer in any process wakes it — the file identity the pass
#   recorded under its own table lock, so no write can land unseen between its decision and it;
# * the earliest FUTURE instant the pass left wakes it: a cron or one-shot occurrence, a hold's
#   ``retry_after``, a withdrawn or consumed row's retention cutoff;
# * only while rows wait on facts the table does not revise — an occurrence in flight, an open
#   or unobserved previous run, a follow-up's origin controls, a hold, a due row it could not
#   start, a removal its settlement still blocks — the existing maintenance cadence
#   (``BODY_MAINTENANCE_INTERVAL_SEC``) observes them, never a per-tick loop.
#
# An incomplete pass (contention, a stop, a raise) records nothing, so the next tick retries as
# before. Process memory only: the table and the queue stay the authorities.
# ---------------------------------------------------------------------------

_WAKES: Dict[str, Dict[str, Any]] = {}


def table_identity(path: Any) -> Any:
    """The table file's identity: every committed write replaces it (new inode, size, mtime)."""
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        return None
    except OSError:
        return object()  # unreadable: never equal, so the pass runs and reports it
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns)


def pass_due(root: Any, now: float) -> bool:
    """Whether the automatic pass has anything new to decide for ``root`` at epoch ``now``."""
    from ouroboros.runtime_limits import BODY_MAINTENANCE_INTERVAL_SEC
    from supervisor.queue_schedules import _scheduled_tasks_path

    wake = _WAKES.get(str(root))
    if wake is None or table_identity(_scheduled_tasks_path(root)) != wake["identity"]:
        return True
    if wake["due_at"] is not None and now >= wake["due_at"]:
        return True
    return wake["observe"] and now >= wake["at"] + BODY_MAINTENANCE_INTERVAL_SEC


def note_pass(root: Any, data: Dict[str, Any], identity: Any, cutoff: float, now_utc: datetime.datetime) -> None:
    """Record, from the table a completed pass decided on, when the next one is due.

    Only FUTURE instants wake by time: a row already due that the pass could not start (its
    previous run still open, unknown queue state, a hold, a contended read) waits on other
    facts, which the observation cadence re-reads. ``cutoff`` is the pass's retention cutoff
    (epoch). Conservative by construction: a row whose timing cannot be read is observed."""
    from supervisor.followup_policy import relation_kind
    from supervisor.schedule_view import _is_consumed_once

    utc, instants = datetime.timezone.utc, []
    retention = datetime.timedelta(seconds=max(0.0, now_utc.timestamp() - cutoff))
    observe = bool(data.get("followup_restart"))
    for row in data.get("tasks") or []:
        if not isinstance(row, dict):
            continue
        hold = row.get("hold") if isinstance(row.get("hold"), dict) else {}
        withdrawn = row.get("delete_requested_at") or (row.get("completed_at") if _is_consumed_once(row) else "")
        instants.append(parse_schedule_time(hold.get("retry_after"), utc) if hold else None)
        if withdrawn:
            at = parse_schedule_time(withdrawn, utc)
            instants.append(at + retention if at is not None else now_utc)
        observe = observe or bool(
            isinstance(row.get("occurrence"), dict) or row.get("previous_run_open") or hold
            or row.get("followup_hold") or row.get("followup_wait")
            or (row.get("last_task_id") and (row.get("last_outcome") or {}).get("task_id") != row.get("last_task_id"))
            or (row.get("enabled", True) and not withdrawn and relation_kind(row) != "independent"))
        if withdrawn or not row.get("enabled", True) or isinstance(row.get("occurrence"), dict):
            continue
        trigger = row.get("trigger") if isinstance(row.get("trigger"), dict) else {}
        kind, tz = str(trigger.get("type") or "cron").strip().lower(), timezone_for_schedule(row)
        if kind == "cron" and str(trigger.get("expr") or row.get("cron") or "").strip():
            instants.append(parse_schedule_time(row.get("next_run_at"), tz) or now_utc)
        elif kind == "once" and not row.get("completed_at"):
            instants.append(parse_schedule_time(trigger.get("run_at"), tz))  # unreadable: its error is recorded
    future = [at for at in instants if at is not None and at > now_utc]
    observe = observe or any(at is not None and at <= now_utc for at in instants)
    _WAKES[str(root)] = {"identity": identity, "at": now_utc.timestamp(), "observe": observe,
                         "due_at": min(future).timestamp() if future else None}
