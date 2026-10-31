"""Typed schedule surface responses; the write seam's semantics live in the queue.

One home for what `/api/schedules` says back, because the three responses share
one vocabulary: a row's lifecycle word, and what a write actually achieved.
``supervisor/queue_schedules.py`` owns the behaviour these describe.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from typing import TypedDict
except ImportError:  # pragma: no cover - Python 3.10 compatibility.
    from typing_extensions import TypedDict


class ScheduledTasksResponse(TypedDict):
    """GET /api/schedules. Every row carries its lifecycle word in ``status``:
    ``active``, ``disabled``, ``consumed`` (a one-shot that already fired),
    ``held``/``waiting`` (a follow-up control) or ``delete_pending`` (withdrawn;
    future dispatch stopped, kept until accepted work settles and retention
    expires — ``delete_pending`` is also a boolean).
    ``retained`` marks history rather than a standing schedule; only
    ``restorable`` can come back. An unreadable table
    is an error here, never an empty list — "no schedules" is a claim."""

    schema_version: int
    tasks: List[Dict[str, Any]]


class ScheduleUpsertResponse(TypedDict):
    """POST /api/schedules. ``schedule`` is the STORED row plus an ``audit``
    field (``recorded``/``incomplete``); ``ok`` follows it, so a change whose
    audit outcome record was lost does not read as a clean success."""

    ok: bool
    schedule: Dict[str, Any]


class ScheduleActionResponse(TypedDict, total=False):
    """POST /api/schedules/{id}/action, and DELETE /api/schedules/{id}.

    ``changed`` is the durable fact; ``ok`` means the requested lifecycle state
    was achieved and both audit records landed.
    ``status`` names what happened (``updated``, ``withdrawn`` (future dispatch
    stopped; the row is retained until accepted work settles and
    retention expires), ``consumed_not_rearmed``, ``lock_timeout`` (the table
    lock was not acquired within its bound; nothing changed), ``store_unreadable``, ``audit_unavailable``, ``changed_audit_incomplete``),
    and ``operation_id`` pairs the intent and outcome events. Generic restore
    may cancel a pending deletion; exact hold release keeps the deletion intent.
    ``running_or_queued`` reports a task this schedule already admitted:
    lifecycle actions govern future dispatch and never stop a run in flight.
    It is ``null`` when the answering process cannot see the live queue — an
    unknown in-flight run is a different statement from no run at all.
    """

    ok: bool
    changed: bool
    status: str
    schedule_id: str
    operation_id: str
    running_or_queued: Optional[bool]
    audit: str
    detail: str
    schedule: Optional[Dict[str, Any]]
    allowed: List[str]


class ScheduleDeleteResponse(TypedDict):
    """DELETE /api/schedules/{id}, legacy name — the public shape it always had.

    The endpoint now answers the richer ``ScheduleActionResponse``, of which this
    is the subset every previous caller read, so the legacy name stays exported
    beside the new one: dropping a published contract name is not an additive
    change, whatever the endpoint gained.
    """

    ok: bool
