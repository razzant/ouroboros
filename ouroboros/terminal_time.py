"""Host-owned per-attempt terminal occurrence, separate from publication time.

An occurrence is observed by whoever performs the live transition: an executor ending
its own attempt (``executor_terminal``), or the supervisor cancelling work it holds
(``supervisor_cancel``: when it recorded the cancel, never the moment a process died).
Recovery and legacy records stay unknown. A trusted replica transfers a matching fact;
its receiver never becomes a new clock. No history scan or timestamp inference.
"""
from __future__ import annotations

from datetime import datetime
from math import isfinite

SUPERVISOR_CANCEL = "supervisor_cancel"
OBSERVED_SOURCES = ("executor_terminal", SUPERVISOR_CANCEL)


def task_attempt_witness(row: dict) -> dict:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return {**{key: row.get(key) for key in ("task_attempt", "_attempt", "started_at")},
            "metadata_attempt": metadata.get("attempt"), "metadata_task_attempt": metadata.get("task_attempt")}


def stated_task_attempt(row: dict, *, include_zero: bool = True):
    witness = task_attempt_witness(row)
    for key in ("task_attempt", "_attempt", "metadata_task_attempt", "metadata_attempt"):
        number = witness[key]
        if number or (include_zero and type(number) is int and number == 0):
            return number
    return None


def execution_owner_attempt(task_id, owner) -> int | None:
    """Read a host-stamped owner by exact task identity, not by lane or origin alone."""
    if not task_id or not isinstance(owner, dict) or owner.get("task_id") != task_id:
        return None
    value = owner.get("task_attempt")
    return value if type(value) is int and value >= 0 else None


def source_task_attempt(task: dict, *, recovered: bool = False) -> int | None:
    """Attempt of this producer's input, never the result current at publication."""
    owner = task.get("execution_owner" if recovered else "_execution_owner")
    value = task.get("task_attempt") if recovered else None
    if value is None and owner is not None:
        # A first direct turn has physical attempt 0 even when the queue tracks
        # its first dispatch as 1. An unbound owner cannot disambiguate them.
        return execution_owner_attempt(task.get("id") or task.get("task_id"), owner)
    if value is None:
        value = task.get("_attempt")
    if value is None:
        value = task.get("task_attempt")
    return value if type(value) is int and value >= 0 else None


def replica_advances_attempt(canonical: dict, replica: dict) -> bool:
    """Only an ordered attempt in the task's replica can catch up an open mirror."""
    from ouroboros.budget_pause import LIVE_PAUSE_STATES

    current, incoming = stated_task_attempt(canonical), stated_task_attempt(replica)
    pause = canonical.get("budget_pause")
    return (type(current) is int and type(incoming) is int and 0 <= current < incoming
            and bool(canonical.get("task_id")) and replica.get("task_id") == canonical["task_id"]
            and canonical.get("status") in {"requested", "scheduled", "running", "interrupted"}
            and not (isinstance(pause, dict) and pause.get("state") in LIVE_PAUSE_STATES))


def same_attempt_witness(left: dict, right) -> bool:
    return isinstance(right, dict) and left.keys() == right.keys() and all(
        value == right[key] and type(value) is type(right[key]) for key, value in left.items())


def terminal_time_fact(row: dict) -> dict:
    """Validate a host record's fact; generic ts/updated_at are never evidence."""
    attempt = task_attempt_witness(row)
    unknown = {"v": 1, "occurred_at": None, "source": "unknown", "attempt": attempt}
    fact = row.get("terminal_time")
    if not isinstance(fact, dict) or fact.get("v") != 1 or not same_attempt_witness(attempt, fact.get("attempt")):
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
    merged = {**existing, **incoming, "task_id": task_id, "status": status}
    if existing.get("status") in _TRULY_TERMINAL_STATUSES:
        # A legacy/unknown terminal is just as immutable as a known one.
        fact = terminal_time_fact(existing)
    elif replica is not None:
        fact = replica_terminal_time(existing, replica, selected=merged)
    elif observed_at and (source == "executor_terminal" or (source == SUPERVISOR_CANCEL and status == "cancelled")):
        fact = {"v": 1, "occurred_at": observed_at, "source": source,
                "attempt": task_attempt_witness(merged)}
    else:
        fact = terminal_time_fact({**merged, "terminal_time": None})
    return {"terminal_time": fact}


def replica_started_attempt_advances(canonical: dict, replica: dict) -> bool:
    """A lagging mirror may take a newer replica's independently named start."""
    start = replica.get("started_at")
    if start == canonical.get("started_at") or not replica_advances_attempt(canonical, replica):
        return False
    owner = replica.get("execution_owner")
    if owner is not None and execution_owner_attempt(replica.get("task_id"), owner) != stated_task_attempt(replica):
        return False
    if isinstance(start, str):
        try:
            return datetime.fromisoformat(start.replace("Z", "+00:00")).tzinfo is not None
        except ValueError:
            return False
    return type(start) in (int, float) and start > 0 and (type(start) is int or isfinite(start))


def replica_terminal_time(canonical, replica, *, selected):
    """Read-side overlay follows the same attempt and immutable-fact rule."""
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES

    if canonical.get("status") in _TRULY_TERMINAL_STATUSES:
        return terminal_time_fact(canonical)
    witness = task_attempt_witness(replica)
    before = task_attempt_witness(canonical)
    if ((not canonical or canonical.get("task_id") == replica.get("task_id"))
            and selected.get("task_id") == replica.get("task_id") and selected.get("status") == replica.get("status")
            and same_attempt_witness(task_attempt_witness(selected), witness)
            and (not canonical or same_attempt_witness(before, witness)
                 or replica_started_attempt_advances(canonical, replica))):
        # Compare BEFORE the overlay. A replica cannot prove an ambiguous start
        # by writing its own witness into the selected row.
        return terminal_time_fact(replica)
    return terminal_time_fact({**selected, "terminal_time": None})
