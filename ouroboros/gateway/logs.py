"""Read-only runtime log endpoints for headless clients.

``chat`` and ``progress`` are the dialogue store's streams (``state/dialog.sqlite``): the tail is
an addressed read -- the newest rows, or one indexed probe per task identifier -- whose cost is
the page, never another room's history; ``_line`` is the row's durable position among every
stored object of its stream (``dialogue_positions.ordinals_of``), the count the chain tail kept.
While the archive import has not settled those counts the endpoint answers 503 ``migrating``,
never a guessed number. ``events``, ``tools`` and ``supervisor`` stay rotated files, read through
the bounded window-doubling tail (``_line`` is a position within that read window).
"""

from __future__ import annotations

import asyncio
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros import dialogue_store as ds
from ouroboros.dialogue_positions import ordinals_of
from ouroboros.gateway._helpers import (
    coerce_int,
    json_error,
    read_rotated_jsonl_entries,
    request_drive_root,
)
from ouroboros.outcomes import public_task_result
from ouroboros.store import StoreError, StoreMigrating
from ouroboros.task_status import find_child_tasks, load_effective_task_result


_ALLOWED_LOGS = {
    "chat": "chat.jsonl",
    "progress": "progress.jsonl",
    "events": "events.jsonl",
    "tools": "tools.jsonl",
    "supervisor": "supervisor.jsonl",
}
_DIALOGUE = frozenset(ds.STREAMS)  # the logs the dialogue store holds; the rest are rotated files
_TASK_COLUMNS = ("task_id", "parent_task_id", "root_task_id", "subagent_task_id")
# A task filter reads candidate pages until the limit is met; a tree whose probes name mostly rows
# the filter refuses (another root's children) is bounded here, never read whole.
_CANDIDATE_PAGES = 10


def replica_key(row: Any) -> Optional[tuple]:
    """A row the passport writes to the task's drive and the canonical log alike (REPLICA_RULE), by its
    natural key, so a reader of both keeps one; None for every other row and for one missing a key field."""
    from ouroboros.contracts.record_contract import ANCHOR_BY_TYPE

    anchor = ANCHOR_BY_TYPE.get(str(row.get("type") or "")) if isinstance(row, dict) else None
    key = tuple(row.get(name) for name in anchor.natural_key) if anchor is not None and anchor.plane == "both" else ()
    return (anchor.type, *key) if key and all(value not in (None, "") for value in key) else None


async def api_logs_tail(request: Request) -> JSONResponse:
    # Child discovery, archived reads and projection can all touch disk.
    return await asyncio.to_thread(_logs_tail, request)


def _logs_tail(request: Request) -> JSONResponse:
    name = str(request.path_params.get("name") or "").strip().lower()
    filename = _ALLOWED_LOGS.get(name)
    if not filename:
        return json_error(f"unknown log {name!r}", 404, allowed=sorted(_ALLOWED_LOGS))
    limit = max(1, min(coerce_int(request.query_params.get("limit"), 100), 2000))
    task_id = str(request.query_params.get("task_id") or "").strip()
    drive_root = request_drive_root(request)
    roots = [drive_root]
    task_filter_ids = {task_id} if task_id else set()
    if task_id:
        # Root/children DISCOVERY only — a status/cost projection suffices, so skip
        # the artifact materialization + disposition lookup (materialize_artifacts
        # contract, task_status.effective_task_result).
        result = load_effective_task_result(drive_root, task_id, materialize_artifacts=False)
        child = str(result.get("child_drive_root") or result.get("headless_child_drive_root") or "").strip()
        if child:
            roots.append(pathlib.Path(child))
        for child_row in find_child_tasks(
            drive_root, parent_task_id=task_id, root_task_id=task_id, materialize_artifacts=False
        ):
            child_id = str(child_row.get("task_id") or child_row.get("id") or "").strip()
            if child_id:
                task_filter_ids.add(child_id)
            child_root = str(child_row.get("child_drive_root") or child_row.get("headless_child_drive_root") or "").strip()
            if child_root:
                roots.append(pathlib.Path(child_root))
    def _matches_task_filter(entry: Any) -> bool:
        if not isinstance(entry, dict):
            return False
        if not task_id:
            return True
        entry_task = str(entry.get("task_id") or "")
        entry_subagent = str(entry.get("subagent_task_id") or "")
        entry_parent = str(entry.get("parent_task_id") or "")
        entry_root = str(entry.get("root_task_id") or "")
        return not (
            entry_task not in task_filter_ids
            and entry_subagent not in task_filter_ids
            and entry_parent != task_id
            and entry_root != task_id
        )

    rows: List[Dict[str, Any]] = []
    try:
        for root in roots:
            root = pathlib.Path(root)
            if name in _DIALOGUE:
                numbered = _dialogue_tail(root, name, limit, task_filter_ids if task_id else None, _matches_task_filter)
            else:
                # A bounded live byte tail plus newest-first archive backfill until the filtered
                # quota is met; ``_line`` is a position within this read window, not the whole file.
                numbered = list(enumerate(read_rotated_jsonl_entries(
                    root / "logs" / filename, root / "archive", name, limit, _matches_task_filter), 1))
            for line_no, entry in numbered:
                if not isinstance(entry, dict) or not _matches_task_filter(entry):
                    continue
                item = public_task_result(
                    entry,
                    include_outcome_axes=any(key in entry for key in ("status", "outcome_axes", "result_status", "loop_outcome")),
                )
                item.setdefault("_source_root", str(root))
                item.setdefault("_line", line_no)
                rows.append(item)
    except StoreMigrating:
        return json_error("the dialogue is still importing; retry shortly", 503, reason_code="migrating")
    except StoreError as exc:
        return json_error(f"the dialogue store is unavailable: {exc}", 503, reason_code="history_source_unavailable")
    rows.sort(key=lambda item: (str(item.get("ts") or ""), str(item.get("_source_root") or ""), int(item.get("_line") or 0)))
    seen: set = set()
    rows = [row for row in rows if (key := replica_key(row)) is None or not (key in seen or seen.add(key))]
    return JSONResponse({"name": name, "entries": rows[-limit:]})


def _dialogue_tail(root: pathlib.Path, stream: str, limit: int, task_ids: Optional[set],
                   matches: Callable[[Any], bool]) -> List[Tuple[int, Dict[str, Any]]]:
    """The newest ``limit`` rows of one dialogue stream that pass the task filter, oldest first,
    each with its ``_line``: the row's position among every stored object of the stream, read
    from the store's durable ordinals (typed ``StoreMigrating`` while the archive import has not
    settled them). Without a filter the tail is the stream's newest rows; with one it is the
    union of one indexed probe per task identifier, paged newest first until the limit."""
    if task_ids is None:
        rows = ds.newest(root, stream, limit=limit)
    else:
        keys = [(column, task) for task in sorted(task_ids) for column in _TASK_COLUMNS]
        rows, before = [], None
        for _page in range(_CANDIDATE_PAGES):
            page = ds.probe_rows(root, stream, keys, before=before, limit=limit, newest_first=True)
            rows.extend(row for row in page if matches(row.row))
            if len(rows) >= limit or len(page) < limit:
                break
            before = page[-1].seq
        rows = sorted(rows[:limit], key=lambda row: row.seq)
    ordinals = ordinals_of(root, stream, [row.seq for row in rows])
    return [(ordinals[row.seq].ordinal + 1, row.row) for row in rows]


__all__ = ["api_logs_tail"]
