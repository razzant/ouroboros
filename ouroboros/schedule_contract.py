"""Shared validation helpers for queue-backed schedules."""

from __future__ import annotations

import datetime
import hashlib
import re
from zoneinfo import ZoneInfo


_SCHEDULE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
RESERVED_TEMPLATE_FIELDS = frozenset({
    "lead",  # the room's primary is captured by the host when the occurrence is admitted
    "work_kind", "owner_ingress", "schedule_writer_kind", "wake_cause", "focus", "consciousness_cause", "origin_wake_task_id",
    "workspace_root", "workspace_mode", "memory_mode", "drive_root",
    "child_drive_root", "budget_drive_root", "parent_task_id",
    "root_task_id", "delegation_role", "task_constraint",
    "task_id", "session_id", "actor_id", "headless_child_drive_root",
    # Owner Surface Fact: a scheduled task is machine-fired — no owner message
    # stands behind it, so a template must never smuggle a sending-surface
    # descriptor (incl. a forged received_at) into owner_client. Admission
    # rejects the key loudly; _task_from_schedule filters it from records
    # persisted before this rule.
    "client_surface",
    # The owner door's stamp (``run_origin.owner_ingress``) is minted by owner
    # routing for one accepted turn; a machine-fired template never carries it,
    # nor ``start_task``'s recorded absence of one.
    "origin_message_ref",
    "origin_suppressed", "origin_message_absent",
    "billing_group", "continuation", "followup_relation", "followup_origin",
    "followup_hold", "followup_released", "followup_wait",
    "continuation_of", "origin_task_id", "origin_root_task_id",
    # The occurrence identity is minted by the scheduler for one admission (#1315).
    "schedule_occurrence", "last_outcome", "previous_run_open",
    # The words a tree carries (``owner_words.FIELD``/``SCHEDULE_FIELD``) and a control press
    # (``ui_control``) are minted for a helper or a pressed run, never by a template.
    "governing_owner_words", "governing_owner_words_absent", "governing_schedule_text", "ui_control",
})
# Host facts a template legitimately holds, which only the schedule writers set
# (``schedule_lifecycle``, ``tools.followup``): who supplied each text field and the
# words of the run that scheduled or edited it. The owner's Schedules door ignores a
# caller's copy (a GET's included) and names it in its answer, rather than refusing.
HOST_ATTESTED_TEMPLATE_FIELDS = frozenset({"objective_author", "context_author", "owner_corpus"})


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
