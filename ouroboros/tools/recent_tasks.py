"""Read-only access to recent task summaries for LLM-first context recovery."""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Dict, List

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.outcomes import normalize_outcome_axes
from ouroboros.task_status import effective_task_result
from ouroboros.dialogue_provenance import (
    PRESENCE_OWN_WORK_SCOPE,
    is_presence_task,
    presence_caller_binding,
    presence_effective_related,
    presence_provenance_from_task,
    presence_related_work,
)


_MAX_TASKS = 20
_PREVIEW_CHARS = 800
_ORIGIN_KEYS = ("provider", "conversation_id", "thread_id", "conversation_key", "source_event_id")


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
    binding: str | None = None,
) -> tuple[Dict[str, Any] | None, Dict[str, str] | None]:
    raw, error = _read_json(path)
    if raw is None:
        return None, {"path": str(path), "error": error}
    data = effective_task_result(drive_root, raw)
    withheld = False
    if binding is not None:
        # A scoped page lists this binding's row; a retry successor it redirects to is judged too.
        withheld = not presence_effective_related(binding, str(raw.get("task_id") or path.stem), data,
                                                  drive_root=drive_root)
        data = raw if withheld else data
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
    if isinstance(data.get("focus"), dict):
        from ouroboros.focus import compact_focus
        focus = compact_focus(data.get("focus"))
        if focus is not None:
            record["focus"] = focus
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
    if data.get("cancel_state"):
        record["cancel_state"] = str(data["cancel_state"])  # requested is not stopped
    origin = presence_provenance_from_task(data)
    if origin:
        # Which of the binding's conversations started it: the source room is a fact
        # for the reader, never a reply address or a public disclosure.
        record["presence_origin"] = {key: origin[key] for key in _ORIGIN_KEYS if origin.get(key)}
    if withheld:
        record["effective_result"] = "withheld: it continues in work not started from this binding"
    return record, None


def _queue_snapshot(drive_root: pathlib.Path) -> tuple[Dict[str, Any], bool]:
    """The persisted queue snapshot, and whether one exists that could not be read.

    Never written means nothing was ever queued; a written snapshot that cannot be
    read, or lists its rows in a shape this reader cannot walk, proves no absence.
    """
    path = drive_root / "state" / "queue_snapshot.json"
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}, False
    except (OSError, UnicodeDecodeError):
        return {}, True
    try:
        data = json.loads(raw)
    except ValueError:
        return {}, True
    if not isinstance(data, dict) or any(not isinstance(data.get(key, []), list) for key in ("running", "pending")):
        return {}, True
    return data, False


def _owner_record(row: Dict[str, Any] | None, queued: Dict[str, Any]) -> Dict[str, Any]:
    """Whose work one task is: a readable result row's binding fact decides, else the queue's own task."""
    if row and (row.get("presence_binding_id") or row.get("presence_authority_recorded")):
        # A readable malformed/empty carrier (or a ceiling whose carrier was lost)
        # outranks a stale queue claim: the empty binding grants no scoped read.
        return {"metadata": {"presence_binding_authority": {"binding_id": row.get("presence_binding_id") or ""}},
                "delegation_role": row.get("delegation_role"), "parent_task_id": row.get("parent_task_id")}
    return queued


def _running_tasks(drive_root: pathlib.Path, binding: str | None = None) -> List[Dict[str, Any]]:
    snapshot, _error = _read_json(drive_root / "state" / "queue_snapshot.json")
    snapshot = snapshot or {}
    running = snapshot.get("running")
    if not isinstance(running, list):
        return []
    facts: Dict[str, Dict[str, Any]] = {}
    if binding is not None:
        from ouroboros.gateway.task_list_scan import raw_result_facts

        try:
            facts, _malformed = raw_result_facts(drive_root / "task_results")
        except OSError:
            pass  # no result row is readable: the queue rows decide, as on the scoped task list
    rows: List[Dict[str, Any]] = []
    for item in running:
        if not isinstance(item, dict):
            continue
        task = item.get("task") if isinstance(item.get("task"), dict) else {}
        task_id = str(item.get("id") or item.get("task_id") or "")
        if binding is not None and not presence_related_work(
                binding, _owner_record(facts.get(f"{task_id}.json"), task)):
            continue  # a scoped page lists no foreign running work, whatever a stale queue row claims
        rows.append({
            "task_id": task_id,
            "status": "running",
            "description": str(item.get("text") or item.get("description")
                               or task.get("description") or task.get("text") or ""),
            "ts": str(item.get("ts") or snapshot.get("ts") or ""),
        })
    return rows


def _presence_scope_inventory(
    drive_root: pathlib.Path, task_dir: pathlib.Path, binding: str, exclude: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """This binding's own work (queue-only active rows first, then result files newest first) and its read gap.

    The memo's scalar binding fact selects files without a second read; a row
    without Presence provenance may be established only by the queue's own task
    metadata (a legacy pending promotion), and a row naming another binding stays out.
    Only a READABLE result row replaces a queue row: an unreadable one leaves the
    queued work listed, and unreadable rows nothing attributes are counted, never dropped.
    An unreadable queue snapshot is a gap too: its queued work cannot be listed, not absent.
    """
    from ouroboros.gateway.task_list_scan import raw_result_facts

    gap: Dict[str, Any] = {}
    try:
        facts, malformed = raw_result_facts(task_dir)
    except OSError:
        facts, malformed = {}, []
        gap["result_root"] = "unreadable"  # queued rows remain; no result row could be read
    snapshot, snapshot_unreadable = _queue_snapshot(drive_root)
    if snapshot_unreadable:
        gap["queue_snapshot"] = "unreadable"  # its queued work cannot be listed; that is not absence
    queued: Dict[str, tuple[str, Dict[str, Any]]] = {}
    for status in ("running", "pending"):
        for item in snapshot.get(status) or []:
            task = item.get("task") if isinstance(item, dict) and isinstance(item.get("task"), dict) else {}
            task_id = str(item.get("id") or task.get("id") or "") if isinstance(item, dict) else ""
            if task_id and task_id not in queued:
                queued[task_id] = (status, task)
    selected: set[str] = set()
    for name, row in facts.items():
        task_id = row.get("task_id") or row.get("id") or name[:-5]
        record = _owner_record(row, queued.get(task_id, ("", {}))[1])
        if task_id != exclude and presence_related_work(binding, record):
            selected.add(name)
    unreadable = set(malformed)
    queue_only = [
        {"queue_task_id": task_id, "status": status,
         "description": str(task.get("description") or task.get("text") or ""),
         **({"result_row": "unreadable"} if f"{task_id}.json" in unreadable else {})}
        for task_id, (status, task) in queued.items()
        if f"{task_id}.json" not in facts and task_id != exclude and presence_related_work(binding, task)
    ]
    unattributed = sum(1 for name in unreadable if name[:-5] not in queued)  # a queue row attributed the rest
    if unattributed:
        gap["unattributed_unreadable_rows"] = unattributed  # any of them may be this binding's work
    return queue_only + [row for row in _task_file_inventory(task_dir) if row["name"] in selected], gap


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
    binding: str | None = None,
) -> str:
    query: Dict[str, Any] = {
        "include_results": bool(include_results),
        "include_traces": bool(include_traces),
    }
    if binding is not None:
        query["presence_binding"] = binding  # another binding or scope never continues this cursor
    payload = {
        "schema_version": 1,
        "query": query,
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
    presence_scope: str = "",
    **_kwargs: Any,
) -> str:
    """Return recent completed task summaries from the canonical task root."""
    from ouroboros.tool_access import canonical_data_root

    binding = None
    if str(presence_scope or "").strip():
        binding = presence_caller_binding(ctx)
        if str(presence_scope).strip() != PRESENCE_OWN_WORK_SCOPE or not binding:
            return json.dumps({"ok": False, "host_code": "TOOL_ARG_ERROR", "error": {
                "code": "PRESENCE_SCOPE_UNAVAILABLE",
                "message": "presence_scope=own_binding needs a Presence task with a binding id.",
            }}, ensure_ascii=False)
    page = recent_tasks_page(
        canonical_data_root(ctx), limit=limit, offset=offset, snapshot=snapshot,
        include_results=bool(include_results), include_traces=bool(include_traces),
        restricted=_restricted_actor(ctx), binding=binding,
        exclude=str(getattr(ctx, "task_id", "") or "") if binding is not None else "",
    )
    if binding is not None:
        page["presence_scope"] = {"scope": PRESENCE_OWN_WORK_SCOPE, "binding_id": binding}
    return json.dumps(page, ensure_ascii=False, indent=2)


def recent_tasks_page(
    drive_root: pathlib.Path,
    *,
    limit: Any = 5,
    offset: Any = 0,
    snapshot: str = "",
    include_results: bool = False,
    include_traces: bool = False,
    restricted: bool = False,
    binding: str | None = None,
    exclude: str = "",
) -> Dict[str, Any]:
    """One stable page; ``binding`` filters to that Presence binding's own work BEFORE paging."""
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
    read_gap: Dict[str, Any] = {}

    def _inventory() -> List[Dict[str, Any]]:
        if binding is None:
            return _task_file_inventory(task_dir)
        rows, gap = _presence_scope_inventory(drive_root, task_dir, binding, exclude)
        read_gap.clear()
        read_gap.update(gap)
        return rows

    for _attempt in range(2):
        tasks = []
        unreadable_tasks = []
        before = _inventory()
        current_snapshot = _recent_tasks_snapshot(
            before,
            include_results=bool(include_results),
            include_traces=bool(include_traces),
            binding=binding,
        )
        selected = before[skip:skip + task_limit]
        for item in selected:
            if item.get("queue_task_id"):
                tasks.append({
                    "task_id": str(item["queue_task_id"]), "status": str(item.get("status") or ""),
                    "description": str(item.get("description") or ""), "source": "queue_snapshot",
                    **({"result_row": item["result_row"]} if item.get("result_row") else {}),
                })
                continue
            path = task_dir / str(item["name"])
            record, error = _task_record(
                path,
                drive_root=drive_root,
                include_results=bool(include_results),
                include_traces=bool(include_traces),
                binding=binding,
            )
            if record is not None:
                if restricted:
                    # A restricted actor gets no cross-focus catalogue (see
                    # _handle_live_roots); a root's authored focus is part of it.
                    record.pop("focus", None)
                tasks.append(record)
            elif error is not None:
                unreadable_tasks.append(error)
        inventory = _inventory()
        stable = before == inventory
        if stable:
            break
    total = len(inventory)
    returned = min(task_limit, max(0, total - skip))
    remaining = max(0, total - skip - returned)
    base = {
        "running": _running_tasks(drive_root, binding),
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
            **({"presence_scope": PRESENCE_OWN_WORK_SCOPE} if binding is not None else {}),
        } if remaining else None),
        **({"read_gap": dict(read_gap)} if read_gap else {}),
    }
    if not stable:
        return {
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
        }
    if requested_snapshot and requested_snapshot != current_snapshot:
        return {
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
        }
    return base


def _restricted_actor(ctx: ToolContext) -> bool:
    """Children and Presence turns, or work acting for a binding, hold no live cross-focus catalogue."""
    metadata = getattr(ctx, "task_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    return bool(str(metadata.get("parent_task_id") or "").strip()
            or str(metadata.get("delegation_role") or "") == "subagent"
            or is_presence_task({"metadata": metadata}) or presence_caller_binding(ctx) is not None)


def _handle_live_roots(ctx: ToolContext, limit: int = 20, offset: int = 0, snapshot: str = "", **_kwargs: Any) -> str:
    """Page the existing host live-root projection without scanning task results."""
    if _restricted_actor(ctx):
        # ``ok: false`` is what the registry's result adapter reads as a typed
        # refusal; a bare ``error`` object would be recorded as a successful call.
        return json.dumps({"ok": False, "host_code": "TOOL_FORBIDDEN",
                           "error": {"code": "TOOL_FORBIDDEN", "message": "restricted actors have no live cross-focus catalogue"}},
                          ensure_ascii=False)
    from ouroboros.peer_roster import live_root_catalogue
    from ouroboros.tool_access import canonical_data_root
    page = live_root_catalogue(canonical_data_root(ctx), limit=limit, offset=offset, snapshot=snapshot)
    if page.get("error"):
        page = {"ok": False, "host_code": str(page["error"].get("code") or "LIVE_ROOTS_ERROR"), **page}
    return json.dumps(page, ensure_ascii=False, indent=2)


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
                    "presence_scope": {
                        "type": "string",
                        "enum": ["own_binding"],
                        "description": (
                            "Presence tasks only: list just the independent work started from this "
                            "Presence binding in any of its conversations, pending and running included."
                        ),
                    },
                },
                "required": [],
            },
        }, _handle_recent_tasks),
        ToolEntry("live_roots", {
            "name": "live_roots",
            "description": "Read the full paginated host-listed live-root catalogue, grouped by project in the same projection used for exact-live messaging.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 20, "description": "Page size (1-100)."},
                    "offset": {"type": "integer", "default": 0},
                    "snapshot": {"type": "string", "default": ""},
                },
                "required": [],
            },
        }, _handle_live_roots),
    ]
