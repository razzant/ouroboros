"""Pure reversible project closure transitions and list selection.

Closure is the mind's or owner's recorded decision, separate from routing lifecycle.
The registry owns persistence; this module reads no storage and imports no writer.
"""
from __future__ import annotations


def apply_closed(project: dict, value: dict | None) -> bool:
    """Compare and apply once; repeated closure preserves the first decision, lifecycle unchanged."""
    if value is None:
        existed = "closed" in project
        project.pop("closed", None)
        return existed
    if (not isinstance(value, dict) or set(value) != {"at", "by", "note"}
            or any(not isinstance(value[key], str) for key in ("at", "by", "note"))
            or not value["at"].strip() or not value["by"].strip()):
        raise ValueError("closed must be None or {at, by, note} with string values and nonempty at/by")
    if "closed" in project:
        return False
    project["closed"] = dict(value)
    return True


def select_projects(projects, *, closed: bool | None = None, oldest_first: bool = False) -> list[dict]:
    """Filter before limiting; recency comes from the caller's captured room facts."""
    if closed is not None and type(closed) is not bool:
        raise ValueError("closed must be a boolean or None")
    if type(oldest_first) is not bool:
        raise ValueError("oldest_first must be a boolean")
    rows = [p for p in projects if closed is None or ("closed" in p) == closed]
    return sorted(rows, key=lambda p: (str(p.get("last_active_at") or ""), str(p.get("id") or "")),
                  reverse=not oldest_first)
