"""Host-owned per-attempt terminal occurrence, separate from publication time.

An occurrence is observed by whoever performs the live transition: an executor ending
its own attempt (``executor_terminal``), or the supervisor carrying out a requested cancel
(``supervisor_cancel``: when it recorded the cancel, never the moment a process died).
Recovery and legacy records stay unknown. A trusted replica transfers a matching fact;
its receiver never becomes a new clock. No history scan or timestamp inference.
"""
from __future__ import annotations

from datetime import datetime

SUPERVISOR_CANCEL = "supervisor_cancel"
OBSERVED_SOURCES = ("executor_terminal", SUPERVISOR_CANCEL)


def task_attempt_witness(row: dict) -> dict:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return {**{key: row.get(key) for key in ("task_attempt", "_attempt", "started_at")},
            "metadata_attempt": metadata.get("attempt"), "metadata_task_attempt": metadata.get("task_attempt")}


def terminal_time_fact(row: dict) -> dict:
    """Validate a host record's fact; generic ts/updated_at are never evidence."""
    attempt = task_attempt_witness(row)
    unknown = {"v": 1, "occurred_at": None, "source": "unknown", "attempt": attempt}
    fact = row.get("terminal_time")
    if not isinstance(fact, dict) or fact.get("v") != 1 or fact.get("attempt") != attempt:
        return unknown
    if fact.get("source") not in OBSERVED_SOURCES or not isinstance(fact.get("occurred_at"), str):
        return unknown
    try:
        parsed = datetime.fromisoformat(fact["occurred_at"].replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return unknown
    except ValueError:
        return unknown
    return {"v": 1, "occurred_at": fact["occurred_at"], "source": fact["source"], "attempt": attempt}


def preserve_terminal_attempt(canonical: dict, incoming: dict) -> dict:
    """An immutable terminal fact and its validating witness travel together.

    Enrichment cannot turn a settled attempt into another attempt. Explicit
    reopen/reset still owns that transition; unrelated metadata remains writable.
    """
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES

    fields = dict(incoming)
    if canonical.get("status") not in _TRULY_TERMINAL_STATUSES:
        return fields
    for key in ("task_attempt", "_attempt", "started_at"):
        if key in canonical:
            fields[key] = canonical[key]
        else:
            fields.pop(key, None)
    if "metadata" in fields:
        metadata = dict(fields["metadata"]) if isinstance(fields["metadata"], dict) else {}
        original = canonical.get("metadata")
        original = original if isinstance(original, dict) else {}
        for key in ("attempt", "task_attempt"):
            if key in original:
                metadata[key] = original[key]
            else:
                metadata.pop(key, None)
        fields["metadata"] = metadata
    return fields


def terminal_time_patch(existing, incoming, *, status, task_id, observed_at=None, replica=None,
                        source="executor_terminal"):
    """``source`` names the observer of ``observed_at``; the supervisor observes only its own cancel."""
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES

    if status not in _TRULY_TERMINAL_STATUSES:
        return {}
    merged = {**existing, **incoming, "task_id": task_id}
    if existing.get("status") in _TRULY_TERMINAL_STATUSES:
        # A legacy/unknown terminal is just as immutable as a known one.
        fact = terminal_time_fact(existing)
    elif (replica is not None and replica.get("task_id") == task_id
          and task_attempt_witness(replica) == task_attempt_witness(merged)
          and (not existing or task_attempt_witness(replica) == task_attempt_witness(existing))):
        # Compare BEFORE the replica overlay: it cannot replace the receiver's
        # attempt witness and thereby authenticate its own stale occurrence.
        fact = terminal_time_fact(replica)
    elif observed_at and (source == "executor_terminal" or (source == SUPERVISOR_CANCEL and status == "cancelled")):
        fact = {"v": 1, "occurred_at": observed_at, "source": source,
                "attempt": task_attempt_witness(merged)}
    else:
        fact = terminal_time_fact({**merged, "terminal_time": None})
    return {"terminal_time": fact}


def replica_terminal_time(canonical, replica):
    """Read-side overlay follows the same attempt and immutable-fact rule."""
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES

    if canonical.get("status") in _TRULY_TERMINAL_STATUSES:
        return terminal_time_fact(canonical)
    if not canonical or (canonical.get("task_id") == replica.get("task_id")
                         and task_attempt_witness(canonical) == task_attempt_witness(replica)):
        return terminal_time_fact(replica)
    return terminal_time_fact({**canonical, **replica, "terminal_time": None})
