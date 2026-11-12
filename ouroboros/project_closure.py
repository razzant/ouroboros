"""Pure reversible project closure transitions and list selection.

Closure is the mind's or owner's recorded decision, separate from routing lifecycle.
The registry owns persistence; this module reads no storage and imports no writer.
"""
from __future__ import annotations


def apply_closed(project: dict, value: dict | None) -> bool:
    """Compare and apply once; repeated closure preserves the first decision, lifecycle unchanged. A mind's close
    also names the delivery of the System notice it posts in the room (``delivery``), whose row the record then holds
    (``memory_sealing.closure_notices``); reopening removes it with the record."""
    if value is None:
        existed = "closed" in project
        project.pop("closed", None)
        return existed
    keys = set(value) if isinstance(value, dict) else set()
    if (not {"at", "by", "note"} <= keys <= {"at", "by", "note", "delivery"}
            or any(not isinstance(value[key], str) for key in keys)
            or not value["at"].strip() or not value["by"].strip() or not value.get("delivery", "-").strip()):
        raise ValueError("closed must be None or {at, by, note[, delivery]} of strings, at/by/delivery nonempty")
    if "closed" in project:
        return False
    project["closed"] = dict(value)
    return True


def select_projects(projects, *, closed: bool | None = None, oldest_first: bool = False,
                    by: str = "last_active_at") -> list[dict]:
    """Filter before limiting; recency (``by``: activity, or the newest message) comes from the caller's
    captured room facts, then the id."""
    if closed is not None and type(closed) is not bool:
        raise ValueError("closed must be a boolean or None")
    if type(oldest_first) is not bool:
        raise ValueError("oldest_first must be a boolean")
    rows = [p for p in projects if closed is None or ("closed" in p) == closed]
    return sorted(rows, key=lambda p: (str(p.get(by) or ""), str(p.get("id") or "")), reverse=not oldest_first)
