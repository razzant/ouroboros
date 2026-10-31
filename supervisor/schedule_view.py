"""Pure schedule facts shared by Activity, model tools and the context digest."""

from __future__ import annotations

import datetime
import re
from typing import Any

from supervisor.schedule_time import parse_schedule_time, timezone_for_schedule

_LEGACY_NAME = re.compile(r"^(Follow-up|Reminder) of task \S+$")


def authored_name(record: dict[str, Any]) -> str:
    """The schedule's own name; "" for the host's placeholder (``Follow-up of task <id>``), which
    names no work: surfaces read the request's first line instead."""
    name = str(record.get("name") or "")
    return "" if _LEGACY_NAME.fullmatch(name) else name


def _is_consumed_once(record: dict[str, Any]) -> bool:
    """A one-shot that already fired: a durable receipt, not a standing schedule."""
    trigger = record.get("trigger") if isinstance(record.get("trigger"), dict) else {}
    return str(trigger.get("type") or "") == "once" and bool(record.get("completed_at"))


def schedule_lifecycle_status(record: dict[str, Any]) -> str:
    """The one lifecycle word every surface says about a row.

    ``consumed`` is retained history, never an active schedule.
    ``delete_pending`` outranks them: a withdrawal stays visible through retention
    and until any work that needs its row has settled.
    """
    if record.get("delete_requested_at"):
        return "delete_pending"
    if _is_consumed_once(record):
        return "consumed"
    if record.get("followup_hold"):
        return "held"
    if record.get("followup_wait"):
        return "waiting"
    return "active" if record.get("enabled", True) else "disabled"


def decorate(row: dict[str, Any], now: datetime.datetime) -> dict[str, Any]:
    """Add display facts without changing the definition or reading task stores."""
    out = dict(row)
    note = row.get("kind") in {"notify", "note"}
    template = row.get("notification" if note else "task")
    template = template if isinstance(template, dict) else {}
    name = str(row.get("name") or "")
    objective = str(template.get("text") or template.get("description") or row.get("description") or "")
    first_line = objective.splitlines()[0].strip() if objective else ""
    out.update(display_name=first_line[:96] if _LEGACY_NAME.fullmatch(name) and first_line else name,
               kind="note" if note else "task", timezone=str(row.get("timezone") or ""),
               destination={"chat_id": template.get("chat_id"), "project_id": template.get("project_id")})
    last_id = str(row.get("last_task_id") or "")
    cached = row.get("last_outcome") or {}
    observed = bool(last_id and cached.get("task_id") == last_id)
    out.pop("outcome", None)
    if last_id:
        out["outcome"] = dict(cached) if observed else {"state": "pending_observation", "task_id": last_id}
    hold = row.get("hold") or {}
    out["waiting_detail"] = {"reason": str(hold.get("reason") or ""),
                             "detail": str(hold.get("detail") or ""),
                             "followup_wait": str(row.get("followup_wait") or "")}
    out.pop("withdrawal", None)
    if row.get("delete_requested_at"):
        consumed = _is_consumed_once(row)
        work_in_flight = True if isinstance(row.get("occurrence"), dict) or row.get("previous_run_open") else (
            False if observed else None)
        out["withdrawal"] = {"at": row["delete_requested_at"], "by": row.get("delete_requested_by", ""),
                             "reason": row.get("delete_reason", ""), "work_in_flight": work_in_flight,
                             "restorable": not consumed and not row.get("followup_hold")}
    return out


def _entry(row: dict[str, Any], now: datetime.datetime) -> dict[str, Any]:
    view = decorate(row, now)
    trigger = row.get("trigger") or {}
    entry = {"id": row.get("id"), "name": view["display_name"][:96], "status": schedule_lifecycle_status(row),
             "timezone": view["timezone"] or "local", "next_run_at": row.get("next_run_at") or ""}
    once = trigger.get("type") == "once"
    entry["run_at" if once else "cron"] = trigger.get("run_at" if once else "expr", "")
    if view.get("outcome"):
        entry["outcome"] = {key: view["outcome"][key] for key in ("state", "quiet") if key in view["outcome"]}
    waiting = view["waiting_detail"]
    reason = waiting["reason"] or waiting["followup_wait"] or (row.get("followup_hold") or {}).get("reason")
    if reason:
        entry["waiting_reason"] = str(reason)[:160]
    if row.get("followup_hold"):
        entry["hold_persisted"] = row.get("hold_persisted", True)
    if row.get("previous_run_open"):
        entry["previous_run_open"] = row["previous_run_open"]
    if row.get("last_error"):
        entry["last_error"] = str(row["last_error"])[:160]
    if once and not row.get("enabled", True) and not row.get("completed_at"):
        at = parse_schedule_time(trigger.get("run_at"), timezone_for_schedule(row))
        if at and at <= now:
            entry["time_passed_on"] = at.isoformat()
    return entry


def digest(rows: list[dict[str, Any]], now: datetime.datetime, limit: int = 8) -> dict[str, Any]:
    """Keep standing promises ahead of exceptions and one-shots within one bound."""
    groups: dict[str, list[dict[str, Any]]] = {"standing": [], "attention": [], "one_shots": []}
    for row in rows:
        if not isinstance(row, dict):
            continue
        trigger = row.get("trigger") or {}
        if trigger.get("type", "cron") == "cron":
            group = "standing"
        elif any(row.get(key) for key in ("hold", "followup_hold", "followup_wait", "last_error", "previous_run_open")):
            group = "attention"
        elif trigger.get("type") == "once" and (not row.get("completed_at") or row.get("delete_requested_at")):
            group = "one_shots"
        else:
            continue
        groups[group].append(row)
    remaining = max(0, int(limit))
    out: dict[str, Any] = {}
    for group, members in groups.items():
        selected = members[:remaining]
        out[group] = [_entry(row, now) for row in selected]
        remaining -= len(selected)
    out["omitted_count"] = sum(map(len, groups.values())) - sum(len(out[key]) for key in groups)
    out["pointer"] = "manage_schedules(action='list')"
    return out
