"""Read-only access to recent task summaries for LLM-first context recovery."""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Dict, List

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.outcomes import normalize_outcome_axes
from ouroboros.task_status import effective_task_result


_MAX_TASKS = 20
_PREVIEW_CHARS = 800


def _coerce_limit(value: Any) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError):
        limit = 5
    return max(1, min(_MAX_TASKS, limit))


def _read_json(path: pathlib.Path) -> tuple[Dict[str, Any] | None, str]:
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(data, dict):
        return None, f"expected JSON object, got {type(data).__name__}"
    return data, ""


def _preview(text: Any) -> str:
    value = str(text or "")
    if len(value) <= _PREVIEW_CHARS:
        return value
    return value[:_PREVIEW_CHARS] + f"\n... (truncated preview from {len(value)} chars)"


def _task_record(
    path: pathlib.Path,
    *,
    drive_root: pathlib.Path,
    include_results: bool,
    include_traces: bool,
) -> tuple[Dict[str, Any] | None, Dict[str, str] | None]:
    data, error = _read_json(path)
    if data is None:
        return None, {"path": str(path), "error": error}
    data = effective_task_result(drive_root, data)
    result = str(data.get("result") or "")
    from ouroboros.cost_projection import cost_projection

    _cost = cost_projection(data)
    record: Dict[str, Any] = {
        "task_id": str(data.get("task_id") or path.stem),
        "ts": str(data.get("ts") or ""),
        "status": str(data.get("status") or ""),
        "outcome_axes": normalize_outcome_axes(data),
        "description": str(data.get("description") or ""),
        # SSOT cost projection (C2/ABI-3): honest null (never a fabricated $0),
        # the honest name only, finality unfabricated.
        "accounted_upper_bound_usd": _cost["accounted_upper_bound_usd"],
        "cost_final": _cost["cost_final"],
        "total_rounds": data.get("total_rounds"),
        "result_preview": _preview(result),
    }
    if isinstance(data.get("task_contract"), dict):
        record["task_contract"] = data.get("task_contract")
    if isinstance(data.get("artifact_bundle"), dict):
        record["artifact_bundle"] = data.get("artifact_bundle")
    ledger = data.get("verification_ledger") if isinstance(data.get("verification_ledger"), dict) else {}
    if ledger:
        # An omitted-to-artifact stub carries no entries; its summary is the
        # count authority, and for a full ledger the two always agree.
        ledger_summary = ledger.get("summary") if isinstance(ledger.get("summary"), dict) else {}
        record["verification_ledger"] = {
            "schema_version": ledger.get("schema_version"),
            "summary": ledger_summary,
            "entry_count": ledger_summary.get("entry_count", len(ledger.get("entries") or []) if isinstance(ledger.get("entries"), list) else 0),
        }
    if include_results:
        record["result"] = result
    if include_traces:
        record["trace_summary"] = str(data.get("trace_summary") or "")
    return record, None


def _running_tasks(drive_root: pathlib.Path) -> List[Dict[str, Any]]:
    snapshot, _error = _read_json(drive_root / "state" / "queue_snapshot.json")
    snapshot = snapshot or {}
    running = snapshot.get("running")
    if not isinstance(running, list):
        return []
    rows: List[Dict[str, Any]] = []
    for item in running:
        if not isinstance(item, dict):
            continue
        rows.append({
            "task_id": str(item.get("id") or item.get("task_id") or ""),
            "status": "running",
            "description": str(item.get("text") or item.get("description") or ""),
            "ts": str(item.get("ts") or snapshot.get("ts") or ""),
        })
    return rows


def _task_file_inventory(task_dir: pathlib.Path) -> List[Dict[str, Any]]:
    inventory: List[Dict[str, Any]] = []
    if not task_dir.is_dir():
        return inventory
    for path in task_dir.glob("*.json"):
        try:
            stat = path.stat()
        except OSError:
            continue
        if not path.is_file():
            continue
        inventory.append({
            "name": path.name,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })
    inventory.sort(key=lambda row: (row["mtime_ns"], row["name"]), reverse=True)
    return inventory


def _recent_tasks_snapshot(
    inventory: List[Dict[str, Any]],
    *,
    include_results: bool,
    include_traces: bool,
) -> str:
    payload = {
        "schema_version": 1,
        "query": {
            "include_results": bool(include_results),
            "include_traces": bool(include_traces),
        },
        "files": inventory,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _handle_recent_tasks(
    ctx: ToolContext,
    limit: int = 5,
    offset: int = 0,
    snapshot: str = "",
    include_results: bool = False,
    include_traces: bool = False,
    **_kwargs: Any,
) -> str:
    """Return recent completed task summaries from the canonical task root."""
    from ouroboros.tool_access import canonical_data_root

    drive_root = canonical_data_root(ctx)
    task_dir = drive_root / "task_results"
    task_limit = _coerce_limit(limit)
    try:
        skip = max(0, int(offset or 0))
    except (TypeError, ValueError):
        skip = 0
    requested_snapshot = str(snapshot or "").strip().lower()
    tasks: List[Dict[str, Any]] = []
    unreadable_tasks: List[Dict[str, str]] = []
    inventory: List[Dict[str, Any]] = []
    current_snapshot = ""
    stable = False
    for _attempt in range(2):
        tasks = []
        unreadable_tasks = []
        before = _task_file_inventory(task_dir)
        current_snapshot = _recent_tasks_snapshot(
            before,
            include_results=bool(include_results),
            include_traces=bool(include_traces),
        )
        selected = before[skip:skip + task_limit]
        for item in selected:
            path = task_dir / str(item["name"])
            record, error = _task_record(
                path,
                drive_root=drive_root,
                include_results=bool(include_results),
                include_traces=bool(include_traces),
            )
            if record is not None:
                tasks.append(record)
            elif error is not None:
                unreadable_tasks.append(error)
        inventory = _task_file_inventory(task_dir)
        stable = before == inventory
        if stable:
            break
    total = len(inventory)
    returned = min(task_limit, max(0, total - skip))
    remaining = max(0, total - skip - returned)
    base = {
        "running": _running_tasks(drive_root),
        "tasks": tasks,
        "unreadable_tasks": unreadable_tasks,
        "source": {"reader": "recent_tasks", "root": "canonical_task_results"},
        "total": total,
        "returned": returned,
        "offset": skip,
        "remaining": remaining,
        "snapshot": current_snapshot,
        "next": ({
            "limit": task_limit,
            "offset": skip + returned,
            "snapshot": current_snapshot,
            "include_results": bool(include_results),
            "include_traces": bool(include_traces),
        } if remaining else None),
    }
    if not stable:
        return json.dumps({
            **base,
            "tasks": [],
            "unreadable_tasks": [],
            "error": {
                "code": "RECENT_TASKS_SNAPSHOT_CHANGED_DURING_READ",
                "message": (
                    "Task results changed while the page was captured; no mixed page "
                    "was returned; restart with offset=0 and no snapshot."
                ),
            },
        }, ensure_ascii=False, indent=2)
    if requested_snapshot and requested_snapshot != current_snapshot:
        return json.dumps({
            **base,
            "tasks": [],
            "unreadable_tasks": [],
            "error": {
                "code": "RECENT_TASKS_SNAPSHOT_CHANGED",
                "message": (
                    "Task results changed after the prior page; no mixed page was "
                    "returned; restart with offset=0 and no snapshot."
                ),
            },
        }, ensure_ascii=False, indent=2)
    return json.dumps(base, ensure_ascii=False, indent=2)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("recent_tasks", {
            "name": "recent_tasks",
            "description": (
                "Read recent task results from the canonical task root. Use when prior work, "
                "continuations, retries, or incomplete current context may matter."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Page size for completed tasks (1-20).",
                        "default": 5,
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Number of newer completed task files already consumed.",
                        "default": 0,
                    },
                    "snapshot": {
                        "type": "string",
                        "description": "Stable cursor returned by the preceding page.",
                        "default": "",
                    },
                    "include_results": {
                        "type": "boolean",
                        "description": "Include full result text instead of only result_preview.",
                        "default": False,
                    },
                    "include_traces": {
                        "type": "boolean",
                        "description": "Include each task's trace_summary.",
                        "default": False,
                    },
                },
                "required": [],
            },
        }, _handle_recent_tasks),
    ]
