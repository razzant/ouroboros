"""Shared validation helpers for queue-backed schedules."""

from __future__ import annotations

import datetime
import hashlib
import re
from zoneinfo import ZoneInfo


_SCHEDULE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
RESERVED_TEMPLATE_FIELDS = frozenset({
    "workspace_root", "workspace_mode", "memory_mode", "drive_root",
    "child_drive_root", "budget_drive_root", "parent_task_id",
    "root_task_id", "delegation_role", "task_constraint",
    "task_id", "session_id", "actor_id", "headless_child_drive_root",
    # Placement keys (RWS v2, Appendix C-3 #10): schedule templates persist
    # project_id ONLY — placement is resolved and sealed at fire time through
    # the same admission path as ordinary task creation. A template smuggling
    # any placement spelling is rejected fail-closed.
    "executor_ref", "workspace_ref", "connection_id", "_sealed_workspace_ref",
})


def schedule_id_error(schedule_id: str) -> str:
    text = str(schedule_id or "").strip()
    if not text:
        return ""
    if not _SCHEDULE_ID_RE.match(text) or ".." in text:
        return "schedule id must be a single URL-safe token"
    return ""


def schedule_slug(*parts: str) -> str:
    raw = "-".join(str(part or "") for part in parts)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-._")
    if not slug or not slug[0].isalnum():
        slug = "schedule-" + slug
    if len(slug) <= 81:
        return slug
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"{slug[:70].rstrip('-._')}-{digest}"


def cron_error(expr: str) -> str:
    text = str(expr or "").strip()
    if len(text.split()) != 5:
        return "cron schedules require a 5-field expression"
    try:
        from croniter import croniter

        croniter(text, datetime.datetime.now(datetime.timezone.utc)).get_next(datetime.datetime)
        return ""
    except Exception as exc:
        return f"invalid cron expression: {type(exc).__name__}: {exc}"


def timezone_error(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        return ""
    try:
        ZoneInfo(text)
        return ""
    except Exception as exc:
        return f"invalid timezone: {type(exc).__name__}: {exc}"
