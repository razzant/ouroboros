"""Temporary space: the wire shapes of the ordinary pool and its pages, computed from the plan and run
``storage_free`` holds in memory. No disk walk per page. The sizes are observed, never a promise of space
freed; ``eligible_after`` is the earliest the background may consider an object, never a promised deletion."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Literal, Optional

try:  # Python 3.11+
    from typing import NotRequired, TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import NotRequired, TypedDict  # type: ignore[assignment]

VIEWS = ("pool", "excluded", "outcomes", "manual")
UNSAVED_LISTED = 50  # the largest paths a manual entry names; ``unsaved_count`` says how many there are
OUTCOME_ORDER = ("failed", "stuck", "kept", "not_started", "deleting", "removed")
ADVANCED = "advanced_group"  # offered only as a whole saved-work group, with that group's own words


class StorageFreeManual(TypedDict):
    """What the owner's explicit discard of one finished copy would remove without proof it is saved. Not
    proven saved is not proven unique: the same bytes may exist elsewhere, nothing here has looked."""

    not_proven_saved_count: int
    not_proven_saved_bytes: int  # lstat sizes, always whole: a path that cannot be read withholds the offer
    not_proven_saved: List[Dict[str, Any]]  # ``{path, bytes}`` relative to the copy, largest first, at most UNSAVED_LISTED
    omitted_count: int  # paths the list leaves out; ``not_proven_saved_count`` includes them
    keeps_commits: bool  # its branch or pin stays, with any commits it holds


class StorageFreeEntry(TypedDict):
    id: str  # exactly one filesystem object, inside this plan
    group: str
    kind: Literal["work_folder", "working_copy", "environment", "other"]
    name: str
    path: str
    task_id: Optional[str]
    bytes: Optional[int]
    bytes_complete: bool
    eligible_after: Optional[str]
    reason: Optional[str]
    outcome: Optional[str]
    availability: Literal["ordinary", "saved_group", "manual_discard", "none"]  # how the owner can remove it, whatever ``reason`` says
    manual: NotRequired[StorageFreeManual]  # ``view=manual`` only


class StorageFreeKept(TypedDict):
    reason: str  # results_pending | in_use | busy | unreadable_record | capture_unproven | unproven_excluded_inputs
    # | owner_keep | changed | retained_evidence | undeclared_environment | advanced_group | restart | other
    count: int
    manual_count: NotRequired[int]  # ``pool.excluded`` only: how many of ``count`` the owner can discard by hand (``view=manual``)


class StorageFreePool(TypedDict):
    count: int
    bytes: Optional[int]
    bytes_complete: bool
    kinds: Dict[str, int]
    excluded: List[StorageFreeKept]
    manual: Dict[str, Any]  # ``{count, bytes, bytes_complete}`` of ``view=manual``: the owner's explicit discard, never in ``count``


class StorageFreeItemsResponse(TypedDict):
    """``GET /api/storage/free/items``: one page; send ``version`` back with the next ``cursor``."""

    plan: str
    view: Literal["pool", "excluded", "outcomes", "manual"]
    version: str
    total: int
    counts: Dict[str, int]
    next_cursor: Optional[int]
    items: List[StorageFreeEntry]


def entry(item: Any, *, reason: Optional[str] = None, outcome: Optional[str] = None) -> StorageFreeEntry:
    return {
        "id": item.key, "group": item.group, "kind": item.kind, "name": item.name, "path": str(item.path),
        "task_id": item.task or None, "bytes": item.bytes, "bytes_complete": item.bytes is not None and not item.unreadable,
        "eligible_after": item.eligible_after, "reason": reason, "outcome": outcome, "availability": _availability(item),
    }


def _availability(item: Any) -> str:
    if item.ordinary:
        return "ordinary"
    return "manual_discard" if item.manual else "none" if item.reason else "saved_group"


def _ordinary(plan: Dict[str, Any]) -> List[Any]:
    found = [item for group in plan["groups"] for item in group["candidates"] if item.ordinary]
    return sorted(found, key=lambda item: (item.bytes is None, -(item.bytes or 0), item.key))


def _excluded(plan: Dict[str, Any]) -> List[Any]:
    found = [(item.reason or "in_use", item) for group in plan["groups"] for item in group["excluded"]]
    found += [(ADVANCED, item) for group in plan["groups"] for item in group["candidates"] if not item.ordinary]
    return sorted(found, key=lambda pair: (pair[0], pair[1].key))


def _manual(plan: Dict[str, Any]) -> List[Any]:
    found = [item for group in plan["groups"] for item in group["excluded"] if item.manual]
    return sorted(found, key=lambda item: (item.bytes is None, -(item.bytes or 0), item.key))


def ordinary_ids(plan: Dict[str, Any]) -> set:
    return {item.key for item in _ordinary(plan)}


def manual_ids(plan: Dict[str, Any]) -> set:
    return {item.key for item in _manual(plan)}


def _manual_entry(item: Any) -> StorageFreeEntry:
    offer = item.manual  # the plan holds the largest UNSAVED_LISTED paths with the whole count and bytes
    return {**entry(item, reason=item.reason), "manual": {
        "not_proven_saved_count": offer["unsaved_count"], "not_proven_saved_bytes": offer["unsaved_bytes"],
        "not_proven_saved": [{"path": rel, "bytes": size} for rel, size in offer["unsaved"][:UNSAVED_LISTED]],
        "omitted_count": max(0, offer["unsaved_count"] - UNSAVED_LISTED),
        "keeps_commits": bool(offer["keep_branch"])}}


def pool(plan: Dict[str, Any]) -> StorageFreePool:
    items, kinds, excluded = _ordinary(plan), {}, {}
    for item in items:
        kinds[item.kind] = kinds.get(item.kind, 0) + 1
    by_hand: Dict[str, int] = {}
    for reason, item in _excluded(plan):
        excluded[reason] = excluded.get(reason, 0) + 1
        if item.manual:
            by_hand[reason] = by_hand.get(reason, 0) + 1
    return {"count": len(items), **_sized(items), "kinds": kinds,
            "excluded": [{"reason": reason, "count": excluded[reason], **({"manual_count": by_hand[reason]} if reason in by_hand else {})}
                         for reason in sorted(excluded)],
            "manual": {"count": len(_manual(plan)), **_sized(_manual(plan))}}


def _sized(items: List[Any]) -> Dict[str, Any]:
    sized = [item for item in items if item.bytes is not None]
    return {"bytes": sum(item.bytes for item in sized) if sized or not items else None,
            "bytes_complete": len(sized) == len(items) and not any(item.unreadable for item in sized)}


def outcomes(run: Optional[Dict[str, Any]], settled: Any) -> List[StorageFreeEntry]:
    """Every selected object's own result as it stands now, failures first."""
    if not run:
        return []
    removed, deleting, stuck = (set(part) for part in settled)
    rows = []
    for item in run["items"]:
        kept = run["kept_items"].get(item.key)
        outcome = ("failed" if item.key in run["failed"] else "stuck" if item.key in stuck else "kept" if kept
                   else "deleting" if item.key in deleting else "removed" if item.key in removed else "not_started")
        rows.append(entry(item, reason=kept, outcome=outcome))
    return sorted(rows, key=lambda row: (OUTCOME_ORDER.index(row["outcome"]), row["id"]))


def page(plan: Dict[str, Any], view: str, rows: Optional[List[StorageFreeEntry]], cursor: int, limit: int) -> StorageFreeItemsResponse:
    """One page of one view; ``rows`` are the run's outcomes for that view."""
    counts: Dict[str, int] = {}
    if view == "pool":
        listed = [entry(item) for item in _ordinary(plan)]
    elif view == "manual":
        listed = [_manual_entry(item) for item in _manual(plan)]
    elif view == "excluded":
        listed = [entry(item, reason=reason) for reason, item in _excluded(plan)]
        for row in listed:
            counts[row["reason"]] = counts.get(row["reason"], 0) + 1
    else:
        listed = rows or []
        for row in listed:
            counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
    version = plan["id"]
    if view == "outcomes":
        version = hashlib.sha256("\n".join(f"{row['id']}\0{row['outcome']}" for row in listed).encode()).hexdigest()[:16]
    end = cursor + limit
    return {"plan": plan["id"], "view": view, "version": version, "total": len(listed), "counts": counts,  # type: ignore[typeddict-item]
            "next_cursor": end if end < len(listed) else None, "items": listed[cursor:end]}
